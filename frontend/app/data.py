"""Read / delete recordings and models on disk (data/sessions, data/models). No Qt here."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "data" / "sessions"
MODELS = REPO / "data" / "models"


def list_sessions():
    out = []
    for p in sorted(SESSIONS.glob("*/meta.json"), reverse=True):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = m.get("summary", {})
        labs = dict(s.get("labels") or {})
        if not labs:
            labs = {g["id"]: "?" for g in m.get("gestures_snapshot", {}).get("gestures", [])}
        out.append({"id": m.get("session_id", p.parent.name), "path": p.parent, "seconds": s.get("seconds"),
                    "labels": labs, "emg_valid": m.get("emg_valid", True), "created": m.get("created", ""),
                    "imu": m.get("imu_present"), "hand": m.get("hand_present"),
                    "hand_pct": s.get("hand_seen_pct"), "warnings": s.get("warnings", [])})
    return out


def example_counts(sessions=None):
    counts = {}
    for s in sessions if sessions is not None else list_sessions():
        for gid, n in s["labels"].items():
            if gid != "none" and isinstance(n, int):
                counts[gid] = counts.get(gid, 0) + n
    return counts


def sessions_for(gid, sessions=None):
    return [s for s in (sessions if sessions is not None else list_sessions()) if gid in s["labels"]]


def delete_sessions(ids):
    n = 0
    for sid in ids:
        p = SESSIONS / sid
        if p.is_dir() and p.parent == SESSIONS:
            shutil.rmtree(p)
            n += 1
    return n


def list_models():
    out = []
    for p in sorted(MODELS.glob("*/model.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m.setdefault("name", p.parent.name)
        m["_dir"] = p.parent
        out.append(m)
    return out


def delete_model(name):
    p = MODELS / name
    if p.is_dir() and p.parent == MODELS:
        shutil.rmtree(p)
        return True
    return False


def wipe_all():
    """Delete every recording, model and the gesture list (keeps frontend_config.json)."""
    for d in (SESSIONS, MODELS):
        if d.exists():
            for c in d.iterdir():
                if c.is_dir():
                    shutil.rmtree(c)
    g = REPO / "data" / "gestures.json"
    if g.exists():
        g.unlink()
