"""Session -> aligned 50 Hz tensors + localised labels + train/val split."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

from backend.common.config import repo_path
from backend.data.features import MODALITIES, featurize_session
from backend.data.labels import IGNORE, localise
from backend.data.session import Session, load_session

log = logging.getLogger(__name__)


@dataclass
class SessionData:
    id: str
    T: np.ndarray            # (F,)
    X: np.ndarray            # (F,D) float32
    M: np.ndarray            # (F,3) bool
    y: np.ndarray            # (F,) int, -1 ignore
    trials: list[dict]
    none_blocks: list[tuple[float, float]]   # free-activity windows (for FP/min)
    rest_std: np.ndarray
    score: np.ndarray
    split: np.ndarray = field(default=None)  # (F,) 0 train, 1 val, -1 unused


def gesture_catalog(cfg, sessions: list[Session]) -> dict[str, dict]:
    """id -> gesture dict. data/gestures.json wins; session snapshots fill in the rest."""
    cat = {}
    for s in sessions:
        for g in s.gestures:
            cat.setdefault(g["id"], g)
    p = repo_path(cfg["paths"]["gestures"])
    if p.exists():
        try:
            for g in json.loads(p.read_text(encoding="utf-8"))["gestures"]:
                cat[g["id"]] = g
        except Exception as e:  # noqa: BLE001
            log.warning("could not read %s: %s", p, e)
    return cat


def build_session(sess: Session, classes: list[str], kinds: dict[str, str], cfg: dict,
                  sensors=MODALITIES, negatives=()) -> SessionData:
    T, X, M, rest_std = featurize_session(sess, cfg, sensors=sensors)
    y, trials, score = localise(T, X, M, sess.labels, classes, kinds, cfg, sess.rest_periods, negatives)
    blocks = [(r["t_start"], r["t_end"]) for r in sess.labels
              if r["label"] == "none" and r["t_end"] - r["t_start"] >= cfg["events"]["min_none_block_s"]]
    return SessionData(sess.id, T, X, M, y, trials, blocks, rest_std, score)


def load_dataset(paths, gestures: list[str] | None, cfg: dict, sensors=MODALITIES, val_sessions=None,
                 negatives=None):
    """negatives: gesture ids whose cue windows are labelled `none` (motions the model must ignore)."""
    """val_sessions: optional session ids held out whole as validation (default: per-gesture stratified split)."""
    sessions = [load_session(p) for p in paths]
    empty = [s.id for s in sessions if len(s.emg) == 0]
    if empty:                                      # EMG drives the frame clock: nothing to learn from these
        log.warning("skipping %d session(s) with no EMG samples: %s", len(empty), empty)
        sessions = [s for s in sessions if len(s.emg) > 0]
        if not sessions:
            raise ValueError("no usable sessions (none has EMG samples)")
    cat = gesture_catalog(cfg, sessions)
    if not gestures:
        present = {r["label"] for s in sessions for r in s.labels} - {"none"}
        gestures = [g for g in cat if g in present] + sorted(present - set(cat))
    missing = [g for g in gestures if not any(r["label"] == g for s in sessions for r in s.labels)]
    if missing:
        raise ValueError(f"no recorded trials for gestures: {missing}")
    classes = ["none"] + list(gestures)
    kinds = {g: cat.get(g, {}).get("kind", "instant") for g in gestures}
    neg = tuple(g for g in (negatives or ()) if g not in gestures)
    if neg:
        log.info("negatives (labelled none): %s", list(neg))
    data = [build_session(s, classes, kinds, cfg, sensors, neg) for s in sessions]
    assign_split(data, cfg, val_sessions=val_sessions)
    return data, classes, kinds


def assign_split(data: list[SessionData], cfg: dict, seed: int | None = None, val_sessions=None):
    """Train/val split, frame-wise (0 train, 1 val, -1 unused guard band).

    Default ("stratified", works when sessions hold different gestures): in EVERY session, the last
    val_frac of each gesture's cue windows (with their surrounding relax segment) plus the last val_frac of
    each free-activity block are validation. A gesture with fewer than split.min_trials_for_val trials in
    a session keeps them all for training there.
    val_sessions=[ids] (or split.mode "newest_session"): those sessions (or the newest) are validation, whole.
    """
    sc = cfg["split"]
    ids = {d.id for d in data}
    if val_sessions:
        unknown = [v for v in val_sessions if v not in ids]
        if unknown:
            raise ValueError(f"val_sessions not among the training sessions: {unknown}")
        if set(val_sessions) >= ids:
            raise ValueError("val_sessions must leave at least one session for training")
        for d in data:
            d.split = np.full(len(d.T), 1 if d.id in val_sessions else 0, np.int64)
        return
    if sc.get("mode", "stratified") == "newest_session" and len(data) >= 2:
        newest = max(d.id for d in data)
        for d in data:
            d.split = np.full(len(d.T), 1 if d.id == newest else 0, np.int64)
        return
    for d in data:
        _stratified(d, cfg)


def _stratified(d: SessionData, cfg: dict):
    sc = cfg["split"]
    frac = sc["val_frac"]
    split = np.zeros(len(d.T), np.int64)
    cues = sorted({(t["cue"], t["label"]) for t in d.trials})  # unique cue windows, in time order
    by_label: dict[str, list[int]] = {}
    for j, (_, lab) in enumerate(cues):
        by_label.setdefault(lab, []).append(j)
    for lab, js in by_label.items():
        if len(js) < sc.get("min_trials_for_val", 3):
            continue
        n_val = max(1, int(round(frac * len(js))))
        for j in js[-n_val:]:  # the LAST reps: also tests drift over the session (fatigue, pad shift)
            (a, b), _ = cues[j]
            lo = 0.5 * (cues[j - 1][0][1] + a) if j > 0 else a - 1.0
            hi = 0.5 * (b + cues[j + 1][0][0]) if j + 1 < len(cues) else b + 1.0
            split[(d.T >= lo) & (d.T < hi)] = 1
    for a, b in d.none_blocks:
        cut = b - frac * (b - a)
        split[(d.T >= cut) & (d.T < b)] = 1
    # guard band: training frames right after a val segment would see val data through the window history
    guard = int(round(1.0 / cfg["frame"]["hop_s"]))
    v = np.flatnonzero(np.diff(split) == -1) + 1
    for k in v:
        split[k:k + guard][split[k:k + guard] == 0] = -1
    d.split = split


def val_trials(d: SessionData) -> list[dict]:
    return [t for t in d.trials if d.split[t["frames"][0]] == 1]


def val_none_blocks(d: SessionData) -> list[tuple[float, float]]:
    out = []
    for a, b in d.none_blocks:
        m = (d.T >= a) & (d.T < b) & (d.split == 1)
        if m.any():
            out.append((float(d.T[m][0]), float(d.T[m][-1])))
    return out


__all__ = ["SessionData", "load_dataset", "build_session", "assign_split", "val_trials", "val_none_blocks",
           "gesture_catalog", "IGNORE"]
