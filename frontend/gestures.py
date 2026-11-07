"""data/gestures.json (INTERFACE §6): written by the frontend, read by the backend."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "data" / "gestures.json"
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load() -> dict:
    try:
        d = json.loads(PATH.read_text(encoding="utf-8"))
        d.setdefault("version", 1)
        d.setdefault("gestures", [])
        return d
    except (OSError, ValueError):
        return {"version": 1, "gestures": []}


def save(d: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")


def get(d: dict, gid: str):
    return next((g for g in d["gestures"] if g["id"] == gid), None)


def add(d: dict, gid: str, name: str | None = None, kind: str = "instant", action: dict | None = None) -> dict:
    if not ID_RE.match(gid):
        raise ValueError(f"gesture id must be snake_case (a-z, 0-9, _): {gid!r}")
    if gid == "none":
        raise ValueError("'none' is reserved")
    if kind not in ("instant", "hold"):
        raise ValueError("kind must be 'instant' or 'hold'")
    if get(d, gid):
        raise ValueError(f"gesture {gid!r} already exists")
    g = {"id": gid, "name": name or gid.replace("_", " ").capitalize(), "kind": kind,
         "action": action or {"type": "none"}, "created": dt.datetime.now().isoformat(timespec="seconds")}
    d["gestures"].append(g)
    return g
