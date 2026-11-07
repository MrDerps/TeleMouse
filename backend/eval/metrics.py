"""Event-level and frame-level evaluation, plus per-gesture threshold tuning."""
from __future__ import annotations

import numpy as np

from backend.eval.events import EventDetector, run_detector


def onsets(events: list[dict]) -> list[dict]:
    """One detection per occurrence: `instant` events and `start` of holds."""
    return [e for e in events if e["kind"] in ("instant", "start")]


def event_metrics(events: list[dict], trials: list[dict], none_blocks: list[tuple[float, float]],
                  classes: list[str], slack: float) -> dict:
    """trials: validation trials (label, kind, cue, span). none_blocks: free-activity windows in validation.
    A detection of g is a TP if it falls in an unmatched cue window of a g trial (+slack)."""
    gest = [c for c in classes if c != "none"]
    det = onsets(events)
    matched = set()
    tp = {g: 0 for g in gest}
    fp = {g: 0 for g in gest}
    lat = {g: [] for g in gest}
    for e in det:
        hit = None
        for k, tr in enumerate(trials):
            if k in matched or tr["label"] != e["id"]:
                continue
            if tr["cue"][0] <= e["t"] <= tr["cue"][1] + slack:
                hit = k
                break
        if hit is None:
            fp[e["id"]] += 1
        else:
            matched.add(hit)
            tp[e["id"]] += 1
            tr = trials[hit]
            ref = tr["span"][0] if tr["kind"] == "hold" else tr["span"][1]
            lat[e["id"]].append(e["t"] - ref)
    n = {g: sum(t["label"] == g for t in trials) for g in gest}
    none_min = sum(b - a for a, b in none_blocks) / 60.0
    in_none = [e for e in det if any(a <= e["t"] <= b for a, b in none_blocks)]
    fp_none = {g: sum(e["id"] == g for e in in_none) for g in gest}
    per = {}
    for g in gest:
        p = tp[g] / (tp[g] + fp[g]) if tp[g] + fp[g] else (1.0 if n[g] == 0 else 0.0)
        r = tp[g] / n[g] if n[g] else float("nan")
        per[g] = {"precision": p, "recall": r, "tp": tp[g], "fp": fp[g], "n": n[g],
                  "fp_none_per_min": fp_none[g] / none_min if none_min else float("nan"),
                  "latency_ms": float(np.median(lat[g]) * 1000) if lat[g] else float("nan")}
    all_lat = [x for g in gest for x in lat[g]]
    TP, FP, N = sum(tp.values()), sum(fp.values()), sum(n.values())
    return {
        "per_gesture": per,
        "precision": TP / (TP + FP) if TP + FP else float("nan"),
        "recall": TP / N if N else float("nan"),
        "fp_per_min": len(in_none) / none_min if none_min else float("nan"),
        "none_minutes": none_min,
        "latency_ms": float(np.median(all_lat) * 1000) if all_lat else float("nan"),
        "n_trials": N,
    }


def frame_confusion(P: np.ndarray, y: np.ndarray, n_cls: int) -> np.ndarray:
    m = y >= 0
    cm = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(cm, (y[m], P[m].argmax(1)), 1)
    return cm


def tune_thresholds(T, P, trials, none_blocks, classes, kinds, cfg) -> dict[str, dict]:
    """Per gesture, pick on_thr from the grid: best F1 among thresholds whose false positives on free
    activity stay <= target / n_gestures per minute; if none qualifies, the highest threshold."""
    e = cfg["events"]
    gest = [c for c in classes if c != "none"]
    budget = e["target_fp_per_min"] / max(1, len(gest))
    out = {}
    for g in gest:
        best, best_key = None, None
        for on in e["thr_grid"]:
            th = {"on_thr": on, "off_thr": max(0.15, on - 0.2)}
            det = EventDetector.from_config(["none", g], kinds, cfg, {g: th})
            gi = classes.index(g)
            Pg = np.stack([1 - P[:, gi], P[:, gi]], 1)
            m = event_metrics(run_detector(T, Pg, det), [t for t in trials if t["label"] == g], none_blocks,
                              ["none", g], e["match_slack_s"])["per_gesture"][g]
            ok = not (m["fp_none_per_min"] > budget)
            f1 = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"] + 1e-9)
            key = (ok, f1 if ok else on)
            if best_key is None or key > best_key:
                best, best_key = th, key
        out[g] = best
    return out
