"""Run a model over all sessions, tune thresholds on validation, compute event + frame metrics."""
from __future__ import annotations

import numpy as np

from backend.data.dataset import val_none_blocks, val_trials
from backend.eval.events import EventDetector, run_detector
from backend.eval.metrics import event_metrics, frame_confusion, tune_thresholds


def evaluate_model(predict, data, classes, kinds, cfg, thresholds=None):
    """predict(X, M) -> P (F, n_cls). Sessions are laid end to end on one timeline.
    Returns (metrics, thresholds, (T, P, y_val, events, confusion, val_trials))."""
    slack = cfg["events"]["match_slack_s"]
    off = 0.0
    T_all, P_all, y_all, vm_all, tr_all, nb_all = [], [], [], [], [], []
    for d in data:
        P = predict(d.X, d.M)
        shift = off - d.T[0]
        T_all.append(d.T + shift)
        P_all.append(P)
        y_all.append(np.where(d.split == 1, d.y, -1))
        vm_all.append(d.split == 1)
        tr_all += [{**t, "cue": (t["cue"][0] + shift, t["cue"][1] + shift),
                    "span": (t["span"][0] + shift, t["span"][1] + shift)} for t in val_trials(d)]
        nb_all += [(a + shift, b + shift) for a, b in val_none_blocks(d)]
        off = T_all[-1][-1] + 10.0
    T, P, y, vm = (np.concatenate(x) for x in (T_all, P_all, y_all, vm_all))
    if thresholds is None:
        thresholds = tune_thresholds(T, P, tr_all, nb_all, classes, kinds, cfg)
    events = run_detector(T, P, EventDetector.from_config(classes, kinds, cfg, thresholds))
    idx = np.clip(np.searchsorted(T, [e["t"] for e in events]), 0, len(T) - 1)
    events = [e for e, k in zip(events, idx) if vm[k]]
    m = event_metrics(events, tr_all, nb_all, classes, slack)
    cm = frame_confusion(P, y, len(classes))
    m["frame_acc"] = float(np.trace(cm) / max(cm.sum(), 1))
    m["confusion"] = cm.tolist()
    return m, thresholds, (T, P, y, events, cm, tr_all)


def summary(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("precision", "recall", "fp_per_min", "latency_ms", "frame_acc")}
