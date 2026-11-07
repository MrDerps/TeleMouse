"""Train the multi-branch causal TCN from scratch (or warm-start encoders) on CPU in ~1-3 min.

    python -m backend.train --sessions all --gestures swipe_left fist --name m1
    python -m backend.train --synth --name synth_tcn
    python -m backend.train --warm-start m1 --name m2      # reuse branch/TCN weights where shapes match
Writes data/models/<name>/{model.pt, model.json, report.png, curve.png}.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from backend.common.config import load_config, repo_path  # noqa: E402
from backend.data.dataset import load_dataset  # noqa: E402
from backend.data.normalize import Normalizer  # noqa: E402
from backend.data.session import resolve_sessions  # noqa: E402
from backend.eval.evaluate import evaluate_model, summary  # noqa: E402
from backend.eval.report import save_report  # noqa: E402
from backend.model.augment import WindowSampler, augment  # noqa: E402
from backend.model.io import model_dir, predict_full, save_model  # noqa: E402
from backend.model.tcn import build_model, n_params  # noqa: E402

log = logging.getLogger("train")


def class_weights(data, n_cls, pow_):
    y = np.concatenate([d.y[d.split == 0] for d in data])
    cnt = np.bincount(y[y >= 0], minlength=n_cls).astype(float)
    w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** pow_, 0.0)
    w /= (w * cnt).sum() / cnt.sum()
    return torch.tensor(w, dtype=torch.float32)


def _batch(sampler, data, norm, cfg, rng):
    t = cfg["train"]
    L = t["win_frames"]
    Zs, Ms, ys = [], [], []
    for _ in range(t["batch"]):
        si, s, Ls = sampler.sample()
        d = data[si]
        X, M, y = augment(d.X[s:s + Ls], d.M[s:s + Ls], d.y[s:s + Ls], L, rng, t["aug"], cfg["emg"]["log_clip"])
        Z = norm(X, M)
        Z += rng.normal(0, t["aug"]["noise"], Z.shape).astype(np.float32)
        for k, sl in enumerate(norm_slices()):
            Z[~M[:, k], sl] = 0.0
        y = y.copy()
        y[: t["warmup_frames"]] = -1
        Zs.append(Z.T)
        Ms.append(M.T)
        ys.append(y)
    return (torch.from_numpy(np.stack(Zs)), torch.from_numpy(np.stack(Ms).astype(np.float32)),
            torch.from_numpy(np.stack(ys)))


def norm_slices():
    from backend.data.features import LAYOUT
    return list(LAYOUT.values())


def val_loss(model, norm, data, w):
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for d in data:
            if not (d.split == 1).any():
                continue
            Z = torch.from_numpy(norm(d.X, d.M).T[None])
            Mt = torch.from_numpy(d.M.T[None].astype(np.float32))
            y = torch.from_numpy(np.where(d.split == 1, d.y, -1)[None])
            lg = model(Z, Mt)
            tot += F.cross_entropy(lg, y, weight=w, ignore_index=-1, reduction="sum").item()
            n += int((y >= 0).sum())
    model.train()
    return tot / max(n, 1)


def train(paths, gestures, name, cfg, warm_start=None, progress=None):
    """Returns (out_dir, meta). progress(frac, message) is called during training (for the live server)."""
    progress = progress or (lambda f, m: None)
    t0 = time.perf_counter()
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["train"]["threads"])
    progress(0.0, "building features")
    data, classes, kinds = load_dataset(paths, gestures, cfg)
    log.info("classes %s, %d session(s), features in %.1fs", classes, len(data), time.perf_counter() - t0)
    Xtr = np.concatenate([d.X[d.split == 0] for d in data])
    Mtr = np.concatenate([d.M[d.split == 0] for d in data])
    norm = Normalizer().fit(Xtr, Mtr)
    model = build_model(len(classes), cfg["model"])
    if warm_start:
        _warm_start(model, warm_start, cfg)
    log.info("model: %d params, receptive field %d frames", n_params(model), model.receptive_field)
    w = class_weights(data, len(classes), cfg["train"]["class_weight_pow"])
    sampler = WindowSampler(data, cfg, rng)
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["steps"], pct_start=0.1)
    hist = {"step": [], "train": [], "val": []}
    best = (np.inf, None, 0)
    run = 0.0
    model.train()
    for step in range(1, tc["steps"] + 1):
        Z, M, y = _batch(sampler, data, norm, cfg, rng)
        loss = F.cross_entropy(model(Z, M), y, weight=w, ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            vl = val_loss(model, norm, data, w)
            hist["step"].append(step)
            hist["train"].append(run)
            hist["val"].append(vl)
            if vl < best[0]:
                best = (vl, {k: v.clone() for k, v in model.state_dict().items()}, step)
            log.info("step %d  train %.4f  val %.4f  (%.0fs)", step, run, vl, time.perf_counter() - t0)
        if step % 25 == 0:
            progress(0.05 + 0.85 * step / tc["steps"], f"step {step}/{tc['steps']} loss {run:.3f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
        log.info("using best checkpoint from step %d (val %.4f)", best[2], best[0])
    model.eval()
    t_train = time.perf_counter() - t0
    progress(0.92, "evaluating")
    predict = lambda X, M: predict_full(model, norm, X, M)  # noqa: E731
    metrics, thr, (T, P, y, events, cm, tr) = evaluate_model(predict, data, classes, kinds, cfg)
    out = model_dir(cfg, name)
    meta = {"name": name, "type": "tcn", "classes": classes, "kinds": kinds,
            "created": datetime.now().isoformat(timespec="seconds"), "sessions": [p.name for p in paths],
            "thresholds": thr, "events": {k: cfg["events"][k] for k in ("ema_alpha", "k_on", "refractory_s")},
            "model_cfg": cfg["model"], "normalizer": norm.to_dict(), "n_params": n_params(model),
            "receptive_field": model.receptive_field, "live_window_frames": tc["live_window_frames"],
            "rest_std": {d.id: d.rest_std.tolist() for d in data}, "feature_cfg": {
                k: cfg[k] for k in ("frame", "emg", "imu", "hand")},
            "train": {"steps": tc["steps"], "best_step": best[2], "seconds": round(t_train, 1), "history": hist},
            "val_metrics": summary(metrics), "val_detail": metrics}
    save_model(out, model, meta)
    save_report(out / "report.png", f"TCN {name}  ({len(paths)} session(s), {n_params(model)} params)",
                classes, cm, metrics, T, P, y, events, thr, t_from=(tr[0]["cue"][0] - 5) if tr else None)
    _curve(out / "curve.png", hist)
    log.info("done in %.1fs -> %s", time.perf_counter() - t0, out)
    progress(1.0, f"trained {name}")
    return out, meta


def _warm_start(model, name, cfg):
    from backend.model.io import load_model
    try:
        old, _, _ = load_model(name, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("warm start from %s failed: %s", name, e)
        return
    sd, own = old.state_dict(), model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not k.startswith("head")}
    model.load_state_dict(keep, strict=False)
    log.info("warm start: %d/%d tensors from %s", len(keep), len(own), name)


def _curve(path, hist):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["step"], hist["train"], "o-", label="train (EMA)")
    ax.plot(hist["step"], hist["val"], "o-", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("weighted CE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=["all"])
    ap.add_argument("--gestures", nargs="*", default=None)
    ap.add_argument("--name", default="tcn")
    ap.add_argument("--synth", action="store_true", help="use backend/_synth instead of data/sessions")
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    if a.steps:
        cfg["train"]["steps"] = a.steps
    root = repo_path(cfg["paths"]["synth"] if a.synth else cfg["paths"]["sessions"])
    paths = resolve_sessions("all" if a.sessions == ["all"] else a.sessions, [root])
    if not paths:
        raise SystemExit(f"no sessions in {root} (record some, or run `python -m backend.tools.synth` + --synth)")
    out, meta = train(paths, a.gestures, a.name, cfg, a.warm_start)
    print(json.dumps(meta["val_metrics"], indent=2))
    for g, m in meta["val_detail"]["per_gesture"].items():
        print(f"  {g:14s} P {m['precision']:.2f} R {m['recall']:.2f} n {m['n']} "
              f"FPnone/min {m['fp_none_per_min']:.2f} lat {m['latency_ms']:.0f} ms  thr {meta['thresholds'][g]['on_thr']}")


if __name__ == "__main__":
    main()
