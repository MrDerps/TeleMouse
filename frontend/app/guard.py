"""Model quality: per-gesture warnings, and the accuracy guard that compares a newly trained model with the
previous one before switching. All numbers come from the backend's model.json (validation on held-out reps)."""
from __future__ import annotations

import json

from . import data, sensor_picker

RECALL_DROP = 0.25        # a gesture losing this much recall counts as "worse" (one missed rep in a 3-rep check = 0.33)
FP_RISE = 1.5             # false triggers per minute
NEW_MIN_RECALL = 0.6


def load_model(name):
    if not name:
        return None
    try:
        return json.loads((data.MODELS / name / "model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ok(x):
    return isinstance(x, (int, float)) and x == x


def _metrics(m, key):
    bs = (m.get("by_sensors") or {}).get(key) or {}
    pg = bs.get("per_gesture") or (m.get("val_detail") or {}).get("per_gesture") or {}
    fp = bs.get("fp_per_min", (m.get("val_metrics") or {}).get("fp_per_min"))
    return {g: v.get("recall") for g, v in pg.items()}, fp


def compare(new_m, old_m, sensors=None):
    """-> list of human-readable problems (empty = the new model is not worse)."""
    if not new_m or not old_m:
        return []
    key = sensor_picker.key(sensors or sensor_picker.current())
    new_r, new_fp = _metrics(new_m, key)
    old_r, old_fp = _metrics(old_m, key)
    issues = []
    for g, old in old_r.items():
        new = new_r.get(g)
        if _ok(old) and _ok(new) and old - new >= RECALL_DROP:
            issues.append(f"{data.gesture_label(g)}: recognised {old:.0%} → {new:.0%}")
    for g, new in new_r.items():
        if g not in old_r and _ok(new) and new < NEW_MIN_RECALL:
            issues.append(f"{data.gesture_label(g)} (new) is recognised only {new:.0%}")
    if _ok(new_fp) and _ok(old_fp) and new_fp - old_fp >= FP_RISE and new_fp >= 2:
        issues.append(f"false triggers during normal activity: {old_fp:.1f} → {new_fp:.1f} per minute")
    return issues


def gesture_warnings(m) -> dict:
    """gesture id -> list of short warnings for its card."""
    if not m:
        return {}
    out = {}
    vd = m.get("val_detail") or {}
    classes = m.get("classes") or []
    for g, v in (vd.get("per_gesture") or {}).items():
        w = []
        fp = v.get("fp_none_per_min")
        if _ok(fp) and fp >= 1.0:
            w.append(f"Fires during normal activity (~{fp:.0f}/min)")
        rec = v.get("recall")
        if _ok(rec) and rec < NEW_MIN_RECALL:
            w.append("Often missed")
        out[g] = w
    conf = vd.get("confusion") or []
    if len(conf) == len(classes):
        for i, gi in enumerate(classes):
            if gi == "none":
                continue
            row = conf[i]
            tot = sum(row) or 1
            for j, gj in enumerate(classes):
                if j != i and gj != "none" and row[j] / tot >= 0.15:
                    out.setdefault(gi, []).append(f"Gets confused with {data.gesture_label(gj)}")
    return out
