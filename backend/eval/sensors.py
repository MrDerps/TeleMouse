"""Accuracy for every combination of sensors (a model trained on several can run with any subset of them,
thanks to modality dropout + presence masks). Thresholds are re-tuned per combination."""
from __future__ import annotations

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backend.data.features import MODALITIES, apply_mask, sensor_key  # noqa: E402
from backend.eval.evaluate import evaluate_model, summary  # noqa: E402


def available_sensors(data) -> list[str]:
    """Modalities actually present somewhere in the validation frames."""
    pres = np.zeros(3, bool)
    for d in data:
        v = d.split == 1
        if v.any():
            pres |= d.M[v].any(0)
    return [m for m, p in zip(MODALITIES, pres) if p]


def sensor_combos(sensors) -> list[tuple[str, ...]]:
    s = [m for m in MODALITIES if m in sensors]
    return [c for k in range(len(s), 0, -1) for c in combinations(s, k)]


def evaluate_by_sensors(predict, data, classes, kinds, cfg, sensors) -> dict:
    """predict(X, M) -> P. Returns {key: {**summary, thresholds, per_gesture}} for every non-empty subset of the
    sensors that the model uses AND that exist in the validation data."""
    use = [m for m in sensors if m in available_sensors(data)]
    out = {}
    for combo in sensor_combos(use):
        keep = np.array([m in combo for m in MODALITIES])

        def pred(X, M, keep=keep):
            M2 = M & keep[None, :]
            return predict(apply_mask(X.copy(), M2), M2)

        m, thr, _ = evaluate_model(pred, data, classes, kinds, cfg)
        out[sensor_key(combo)] = {**summary(m), "thresholds": thr,
                                  "per_gesture": {g: {k: v[k] for k in ("precision", "recall", "latency_ms")}
                                                  for g, v in m["per_gesture"].items()}}
    return out


def format_table(by: dict) -> str:
    lines = [f"{'sensors':16s} {'precision':>9s} {'recall':>7s} {'FP/min':>7s} {'latency':>8s} {'frame acc':>9s}"]
    for k, v in by.items():
        fp = "n/a" if not np.isfinite(v["fp_per_min"]) else f"{v['fp_per_min']:.2f}"
        lat = "n/a" if not np.isfinite(v["latency_ms"]) else f"{v['latency_ms']:.0f} ms"
        lines.append(f"{k:16s} {v['precision']:9.2f} {v['recall']:7.2f} {fp:>7s} {lat:>8s} {v['frame_acc']:9.2f}")
    return "\n".join(lines)


def save_sensor_plot(path, by: dict, title: str):
    keys = list(by)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(1, 2, figsize=(max(8, 1.6 * len(keys) + 4), 4))
    ax[0].bar(x - 0.2, [by[k]["precision"] for k in keys], 0.4, label="precision")
    ax[0].bar(x + 0.2, [by[k]["recall"] for k in keys], 0.4, label="recall")
    ax[0].set_xticks(x, keys, rotation=30, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=8)
    ax[0].set_title("event precision / recall (val)")
    fp = [by[k]["fp_per_min"] for k in keys]
    ax[1].bar(x, [f if np.isfinite(f) else 0 for f in fp], color="tab:red")
    for i, f in enumerate(fp):
        if not np.isfinite(f):
            ax[1].text(i, 0.02, "n/a", ha="center", fontsize=8)
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_xticks(x, keys, rotation=30, ha="right")
    ax[1].set_title("false positives / min on free activity (target <= 1)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
