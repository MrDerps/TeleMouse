"""Frontend machine settings in data/frontend_config.json (not committed: it's per-machine)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "data" / "frontend_config.json"
DEFAULTS = {
    "cam_latency": 0.08,        # s, subtracted from frame capture time (INTERFACE §2)
    "cam_latency_calibrated": None,   # ISO time of the last sync calibration
    "camera_index": 0,
    "serial_port": None,
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save(**updates) -> dict:
    cfg = load()
    cfg.update(updates)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg
