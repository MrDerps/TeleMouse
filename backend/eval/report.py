"""PNG evaluation report: confusion matrix, per-gesture P/R, headline metrics, probability timeline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_report(path, title, classes, cm, metrics, T, P, y, events, thresholds, span_s=60.0, t_from=None):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    # confusion (row-normalised)
    ax = fig.add_subplot(gs[0, 0])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted (frame argmax)")
    ax.set_ylabel("label")
    ax.set_title("frame confusion (val)")
    # per-gesture P/R
    ax = fig.add_subplot(gs[0, 1])
    gest = list(metrics["per_gesture"])
    x = np.arange(len(gest))
    pr = [metrics["per_gesture"][g]["precision"] for g in gest]
    rc = [metrics["per_gesture"][g]["recall"] for g in gest]
    ax.bar(x - 0.2, pr, 0.4, label="precision")
    ax.bar(x + 0.2, rc, 0.4, label="recall")
    ax.set_xticks(x, gest, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("event-level (val)")
    # text
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lines = [title, "",
             f"precision {metrics['precision']:.3f}   recall {metrics['recall']:.3f}",
             f"false positives / min on free activity: {metrics['fp_per_min']:.2f}"
             f"  (over {metrics['none_minutes']:.1f} min)",
             f"median latency (end of motion -> event; hold: start): {metrics['latency_ms']:.0f} ms",
             f"val trials: {metrics['n_trials']}", "",
             "per gesture: P / R / FP-none/min / latency ms / on_thr"]
    for g in gest:
        m = metrics["per_gesture"][g]
        lines.append(f"  {g:14s} {m['precision']:.2f} / {m['recall']:.2f} / {m['fp_none_per_min']:.2f} / "
                     f"{m['latency_ms']:.0f} / {thresholds.get(g, {}).get('on_thr', float('nan')):.2f}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    # timeline
    ax = fig.add_subplot(gs[1, :])
    t0 = T[0] if t_from is None else t_from
    m = (T >= t0) & (T < t0 + span_s)
    for i, c in enumerate(classes[1:], 1):
        line, = ax.plot(T[m] - t0, P[m, i], lw=0.9, label=c)
        lab = (y[m] == i).astype(float)
        ax.fill_between(T[m] - t0, 0, lab * 1.05, color=line.get_color(), alpha=0.15, step="mid")
    for e in events:
        if t0 <= e["t"] < t0 + span_s:
            ax.axvline(e["t"] - t0, color="k", ls="--" if e["kind"] == "end" else "-", lw=0.6)
            ax.text(e["t"] - t0, 1.07, e["id"] + ("" if e["kind"] == "instant" else f" {e['kind']}"),
                    rotation=90, fontsize=6, va="bottom")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel(f"s (from t={t0:.1f})")
    ax.set_ylabel("p")
    ax.legend(fontsize=7, loc="upper right", ncol=len(classes))
    ax.set_title("val timeline: lines = probabilities, shading = localised labels, vertical = events")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path
