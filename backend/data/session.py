"""Load recorded sessions (INTERFACE.md section 6)."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Session:
    path: Path
    id: str
    meta: dict
    emg: np.ndarray       # (N,2) uint16
    emg_t: np.ndarray     # (N,) float64
    imu: np.ndarray       # (M,6) float32, NaN rows when missing
    imu_t: np.ndarray     # (M,)
    hand: np.ndarray      # (K,21,3) float32, NaN rows when no hand
    hand_t: np.ndarray    # (K,)
    hand_score: np.ndarray
    labels: list[dict] = field(default_factory=list)  # {t_start, t_end, label, source}

    @property
    def gestures(self) -> list[dict]:
        return self.meta.get("gestures_snapshot", {}).get("gestures", [])

    @property
    def rest_periods(self) -> list[list[float]]:
        return self.meta.get("rest_periods") or []

    def img_size(self, cfg) -> tuple[int, int]:
        return (int(self.meta.get("img_w") or cfg["hand"]["default_img_w"]),
                int(self.meta.get("img_h") or cfg["hand"]["default_img_h"]))


def _load(path: Path, name: str, default_shape, dtype):
    p = path / name
    if p.exists():
        return np.load(p)
    log.warning("%s: missing %s", path.name, name)
    return np.zeros(default_shape, dtype)


def load_session(path: str | Path) -> Session:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    emg = _load(path, "emg.npy", (0, 2), np.uint16)
    emg_t = _load(path, "emg_t.npy", (0,), np.float64).astype(np.float64)
    imu = _load(path, "imu.npy", (0, 6), np.float32).astype(np.float32)
    imu_t = _load(path, "imu_t.npy", (0,), np.float64).astype(np.float64)
    hand = _load(path, "hand.npy", (0, 21, 3), np.float32).astype(np.float32)
    hand_t = _load(path, "hand_t.npy", (0,), np.float64).astype(np.float64)
    hand_score = _load(path, "hand_score.npy", (len(hand_t),), np.float32)
    if len(hand) != len(hand_t):
        raise ValueError(f"{path.name}: hand {len(hand)} rows vs hand_t {len(hand_t)}")
    if len(emg) != len(emg_t) or len(imu) != len(imu_t):
        raise ValueError(f"{path.name}: data/timestamp length mismatch")
    if not meta.get("imu_present", True):
        imu = np.full_like(imu, np.nan)
    labels = []
    lp = path / "labels.csv"
    if lp.exists():
        with open(lp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels.append({"t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
                               "label": r["label"].strip(), "source": r.get("source", "cue")})
    labels.sort(key=lambda r: r["t_start"])
    return Session(path, meta.get("session_id", path.name), meta, emg, emg_t, imu, imu_t,
                   hand, hand_t, hand_score, labels)


def list_sessions(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "meta.json").exists())


def resolve_sessions(spec, roots: list[Path]) -> list[Path]:
    """spec = "all" | list of session ids or paths. Searches each root."""
    found = {p.name: p for r in roots for p in list_sessions(r)}
    if spec in (None, "all", ["all"]):
        return [found[k] for k in sorted(found)]
    out = []
    for s in spec:
        p = Path(s)
        if (p / "meta.json").exists():
            out.append(p)
        elif s in found:
            out.append(found[s])
        else:
            raise FileNotFoundError(f"session {s!r} not found in {[str(r) for r in roots]}")
    return out


def load_truth(path: Path) -> list[dict]:
    """Synthetic sessions only: true motion spans."""
    p = Path(path) / "truth.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [{"t_start": float(r["t_start"]), "t_end": float(r["t_end"]), "label": r["label"]}
                for r in csv.DictReader(f)]
