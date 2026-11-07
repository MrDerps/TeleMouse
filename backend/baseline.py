"""Windowed-feature LDA baseline (sanity floor before the TCN) + event-level evaluation report.

Per 50 Hz frame, over the last 400 ms (causal): mean, std and (last - first) of every standardised feature,
plus the modality-mask means. Shrinkage LDA -> per-frame probabilities -> the same event logic as live.

    python -m backend.baseline                         # real sessions in data/sessions
    python -m backend.baseline --synth                 # synthetic sessions in backend/_synth
    python -m backend.baseline --sessions <id> <id> --gestures swipe_left fist --name base1
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from datetime import datetime

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from backend.common.config import load_config, repo_path
from backend.data.dataset import load_dataset, val_none_blocks, val_trials
from backend.data.features import LAYOUT
from backend.data.session import resolve_sessions
from backend.eval.events import EventDetector, run_detector
from backend.eval.metrics import event_metrics, frame_confusion, tune_thresholds
from backend.eval.report import save_report

log = logging.getLogger("baseline")


class Normalizer:
    """Per-feature standardisation with training stats; masked modalities stay exactly zero."""

    def fit(self, X, M):
        self.mu = np.zeros(X.shape[1], np.float32)
        self.sd = np.ones(X.shape[1], np.float32)
        for k, (name, sl) in enumerate(LAYOUT.items()):
            m = M[:, k]
            if m.sum() > 10:
                self.mu[sl] = X[m, sl].mean(0)
                self.sd[sl] = np.maximum(X[m, sl].std(0), 1e-3)
        return self

    def __call__(self, X, M):
        Z = (X - self.mu) / self.sd
        for k, sl in enumerate(LAYOUT.values()):
            Z[~M[:, k], sl] = 0.0
        return np.clip(Z, -10, 10).astype(np.float32)


def window_features(Z: np.ndarray, M: np.ndarray, W: int) -> np.ndarray:
    """Causal window stats over the last W frames (first frames padded by repetition)."""
    F = len(Z)
    Zp = np.concatenate([np.repeat(Z[:1], W - 1, 0), Z])
    Mp = np.concatenate([np.repeat(M[:1], W - 1, 0), M]).astype(np.float32)
    c1 = np.concatenate([np.zeros((1, Z.shape[1])), np.cumsum(Zp, 0, dtype=np.float64)])
    c2 = np.concatenate([np.zeros((1, Z.shape[1])), np.cumsum(Zp.astype(np.float64) ** 2, 0)])
    cm = np.concatenate([np.zeros((1, 3)), np.cumsum(Mp, 0, dtype=np.float64)])
    hi = np.arange(W, F + W)
    lo = hi - W
    mean = (c1[hi] - c1[lo]) / W
    var = np.maximum((c2[hi] - c2[lo]) / W - mean**2, 0)
    delta = Zp[hi - 1] - Zp[lo]
    mm = (cm[hi] - cm[lo]) / W
    return np.concatenate([mean, np.sqrt(var), delta, mm], 1).astype(np.float32)


class BaselineModel:
    def __init__(self, cfg, classes):
        self.cfg, self.classes = cfg, classes
        self.W = cfg["baseline"]["win_frames"]
        self.norm = Normalizer()
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg["baseline"]["shrinkage"])

    def features(self, X, M):
        return window_features(self.norm(X, M), M, self.W)

    def fit(self, data):
        Xtr = np.concatenate([d.X[d.split == 0] for d in data])
        Mtr = np.concatenate([d.M[d.split == 0] for d in data])
        self.norm.fit(Xtr, Mtr)
        feats, labels = [], []
        sub = self.cfg["baseline"]["none_subsample"]
        rng = np.random.default_rng(self.cfg["seed"])
        for d in data:
            Fw = self.features(d.X, d.M)
            keep = (d.split == 0) & (d.y >= 0)
            keep &= (d.y > 0) | (rng.random(len(d.y)) < 1.0 / sub)
            feats.append(Fw[keep])
            labels.append(d.y[keep])
        self.lda.fit(np.concatenate(feats), np.concatenate(labels))
        return self

    def predict_proba(self, X, M):
        p = self.lda.predict_proba(self.features(X, M))
        out = np.zeros((len(X), len(self.classes)), np.float32)
        out[:, self.lda.classes_] = p
        return out


def evaluate_model(predict, data, classes, kinds, cfg, thresholds=None):
    """Run predict(X,M)->P per session; tune thresholds on val (if not given); event + frame metrics on val."""
    slack = cfg["events"]["match_slack_s"]
    Ts, Ps, ys, tr, nb, masks = [], [], [], [], [], []
    for d in data:
        P = predict(d.X, d.M)
        Ts.append(d.T)
        Ps.append(P)
        ys.append(np.where(d.split == 1, d.y, -1))
        tr += val_trials(d)
        nb += val_none_blocks(d)
        masks.append(d.split == 1)
    # concatenate sessions on one timeline (sessions don't overlap in time: offset to be safe)
    off, T_all = 0.0, []
    tr_all, nb_all = [], []
    for d, T in zip(data, Ts):
        shift = off - T[0]
        T_all.append(T + shift)
        tr_all += [{**t, "cue": (t["cue"][0] + shift, t["cue"][1] + shift),
                    "span": (t["span"][0] + shift, t["span"][1] + shift)} for t in val_trials(d)]
        nb_all += [(a + shift, b + shift) for a, b in val_none_blocks(d)]
        off = T_all[-1][-1] + 10.0
    T = np.concatenate(T_all)
    P = np.concatenate(Ps)
    y = np.concatenate(ys)
    vm = np.concatenate(masks)
    if thresholds is None:
        thresholds = tune_thresholds(T, P, tr_all, nb_all, classes, kinds, cfg)
    det = EventDetector.from_config(classes, kinds, cfg, thresholds)
    events = run_detector(T, P, det)
    idx = np.clip(np.searchsorted(T, [e["t"] for e in events]), 0, len(T) - 1)
    events = [e for e, k in zip(events, idx) if vm[k]]
    m = event_metrics(events, tr_all, nb_all, classes, slack)
    cm = frame_confusion(P, y, len(classes))
    m["frame_acc"] = float(np.trace(cm) / max(cm.sum(), 1))
    m["confusion"] = cm.tolist()
    return m, thresholds, (T, P, y, events, cm, tr_all)


def session_roots(cfg, synth: bool):
    return [repo_path(cfg["paths"]["synth"] if synth else cfg["paths"]["sessions"])]


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=["all"])
    ap.add_argument("--gestures", nargs="*", default=None)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--synth", action="store_true", help="use backend/_synth instead of data/sessions")
    a = ap.parse_args(argv)
    np.random.seed(cfg["seed"])
    t0 = time.perf_counter()
    paths = resolve_sessions("all" if a.sessions == ["all"] else a.sessions, session_roots(cfg, a.synth))
    if not paths:
        raise SystemExit("no sessions found (record some, or run `python -m backend.tools.synth` and pass --synth)")
    log.info("sessions: %s", [p.name for p in paths])
    data, classes, kinds = load_dataset(paths, a.gestures, cfg)
    log.info("classes %s; built features in %.1fs", classes, time.perf_counter() - t0)
    model = BaselineModel(cfg, classes).fit(data)
    log.info("trained in %.1fs", time.perf_counter() - t0)
    metrics, thr, (T, P, y, events, cm, tr) = evaluate_model(model.predict_proba, data, classes, kinds, cfg)
    out = repo_path(cfg["paths"]["models"]) / a.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline.pkl", "wb") as f:
        pickle.dump({"norm": model.norm, "lda": model.lda, "W": model.W}, f)
    summary = {k: metrics[k] for k in ("precision", "recall", "fp_per_min", "latency_ms", "frame_acc")}
    meta = {"name": a.name, "type": "baseline_lda", "classes": classes, "kinds": kinds,
            "created": datetime.now().isoformat(timespec="seconds"), "sessions": [p.name for p in paths],
            "thresholds": thr, "events": {k: cfg["events"][k] for k in ("ema_alpha", "k_on", "refractory_s")},
            "rest_std": {d.id: d.rest_std.tolist() for d in data}, "val_metrics": summary,
            "val_detail": metrics}
    (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    t_from = tr[0]["cue"][0] - 5 if tr else None
    save_report(out / "report.png", f"baseline LDA  {a.name}  ({len(paths)} session(s))", classes, cm, metrics,
                T, P, y, events, thr, t_from=t_from)
    log.info("done in %.1fs -> %s", time.perf_counter() - t0, out)
    print(json.dumps(summary, indent=2))
    for g, m in metrics["per_gesture"].items():
        print(f"  {g:14s} P {m['precision']:.2f} R {m['recall']:.2f} n {m['n']} "
              f"FPnone/min {m['fp_none_per_min']:.2f} lat {m['latency_ms']:.0f} ms  thr {thr[g]['on_thr']}")


if __name__ == "__main__":
    main()
