"""Label localisation: find the actual motion inside each generous cue window.

Activity score per frame = max over available modalities of a robust z-score (vs relax baselines) of:
  EMG: mean log-RMS of the two channels;  IMU: |gyro|;  hand: wrist speed + landmark shape speed.
Inside a gesture cue window, frames with score > max(k_abs, k_rel * window peak) are active; spans closer
than merge_gap are merged. instant -> the span with the largest integrated score is the occurrence; other strong
spans (peak >= burst_rel x window peak: a repeat, or a preparatory phase such as opening the hand before a pinch)
are IGNORED, not taught as `none` (with labels.multi_burst=true every strong span becomes its own occurrence
instead); hold -> from the first active frame (the onset: closing the fist, ...) to the END of the cue window,
because the cue says "hold until the bar ends" and a static held pose has almost no activity of its own (only the
onset and the release move). If a clearly separate later burst appears inside the window (the user released early),
the hold ends where that burst starts. (labels.hold_to_cue_end=false restores first..last active.)
Frames in the span get the gesture label, frames just outside it (edge_ignore_s) are ignored (-1), everything
else is `none` (0). Cue windows of gestures not in the class list are ignored entirely (not called `none`).
If nothing is active, fall back to the window centre (logged).
Unlabelled relax gaps are `none` only where they are quiet: clearly active gap frames (e.g. the user still
repeating the previous gesture after its cue ended) are ignored, so real gestures are never taught as `none`.
Explicitly labelled `none` windows (rest, free activity) stay `none` whatever happens in them.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.data.features import LAYOUT

log = logging.getLogger(__name__)
IGNORE = -1


def _robust_z(x: np.ndarray, base: np.ndarray, floor: float) -> np.ndarray:
    if base.sum() < 10:
        base = np.ones_like(base)
    med = np.median(x[base])
    mad = 1.4826 * np.median(np.abs(x[base] - med))
    return (x - med) / max(mad, floor)


def modality_signals(X: np.ndarray, M: np.ndarray):
    e, i, h = LAYOUT["emg"], LAYOUT["imu"], LAYOUT["hand"]
    emg = X[:, e][:, 0:2].mean(1)
    gyro = X[:, i][:, 11]
    H = X[:, h]
    shape = np.zeros(len(X))
    both = M[1:, 2] & M[:-1, 2]
    shape[1:][both] = np.linalg.norm(H[1:, :60] - H[:-1, :60], axis=1)[both] * 50.0 / 10.0
    speed = np.linalg.norm(H[:, 62:64], axis=1) * 5.0
    hand = np.where(M[:, 2], shape + speed, 0.0)
    return {"emg": emg, "gyro": gyro, "hand": hand}


def activity_score(X, M, baseline: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    lc = cfg["labels"]
    sig = modality_signals(X, M)
    z = {}
    present = {"emg": M[:, 0], "gyro": M[:, 1], "hand": M[:, 2]}
    # hand features at frame T describe the hand at T - hand.lag_s (live look-back). For LABELS we want the true
    # motion time, so shift the hand signal back to line it up with EMG/IMU (offline only; the model input is
    # untouched). Otherwise one motion shows up twice: IMU first, camera lag_s later.
    k_lag = int(round(cfg["hand"].get("lag_s", 0.0) / cfg["frame"]["hop_s"]))
    if k_lag > 0:
        for key, arr in (("hand", sig["hand"]),):
            sh = np.zeros_like(arr)
            sh[:-k_lag] = arr[k_lag:]
            sig[key] = sh
        ph = np.zeros_like(present["hand"])
        ph[:-k_lag] = present["hand"][k_lag:]
        present["hand"] = ph
    for k, x in sig.items():
        b = baseline & present[k]
        zz = _robust_z(x, b, lc["floors"][k]) if b.sum() >= 10 else np.zeros(len(x))
        z[k] = np.where(present[k], zz, 0.0)
    s = np.max(np.stack(list(z.values())), 0)
    n = lc["smooth_frames"]
    if n > 1:
        s = np.convolve(s, np.ones(n) / n, "same")  # offline labelling only, centred is fine
    return s, z


def _spans(active: np.ndarray, merge: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    out = [[idx[0], idx[0]]]
    for k in idx[1:]:
        if k - out[-1][1] <= merge + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [(a, b + 1) for a, b in out]  # half-open


def localise(T, X, M, labels: list[dict], classes: list[str], kinds: dict[str, str], cfg: dict,
             rest_periods=None):
    """Returns y (F,) int (-1 ignore, 0 none, 1.. gestures) and a list of trial dicts."""
    lc = cfg["labels"]
    hop = cfg["frame"]["hop_s"]
    F = len(T)
    cls_idx = {c: i for i, c in enumerate(classes)}
    # baseline = frames not inside any gesture cue window (relax gaps + rest), excluding free-activity blocks
    in_cue = np.zeros(F, bool)
    in_none_block = np.zeros(F, bool)
    for r in labels:
        m = (T >= r["t_start"]) & (T < r["t_end"])
        if r["label"] == "none":
            if r["t_end"] - r["t_start"] >= cfg["events"]["min_none_block_s"] and not _is_rest(r, rest_periods):
                in_none_block |= m
        else:
            in_cue |= m
    baseline = ~in_cue & ~in_none_block
    score, z = activity_score(X, M, baseline, cfg)

    y = np.zeros(F, np.int64)
    in_none_label = np.zeros(F, bool)
    for r in labels:
        if r["label"] == "none":
            in_none_label |= (T >= r["t_start"]) & (T < r["t_end"])
    gap = ~in_cue & ~in_none_label
    if lc.get("ignore_active_gaps", True):
        busy = gap & (score > lc["k_abs"])
        pad = int(round(lc["edge_ignore_s"] / cfg["frame"]["hop_s"]))
        if pad and busy.any():
            busy = np.convolve(busy.astype(float), np.ones(2 * pad + 1), "same") > 0
        y[busy & gap] = IGNORE
    trials = []
    merge = int(round(lc["merge_gap_s"] / hop))
    merge_inst = int(round(lc.get("merge_gap_instant_s", lc["merge_gap_s"]) / hop))
    edge = int(round(lc["edge_ignore_s"] / hop))
    min_len = int(round(lc["min_span_s"] / hop))
    pre = lc.get("pre_cue_s", 0.0)
    rows = sorted(labels, key=lambda r: r["t_start"])
    for j, r in enumerate(rows):
        lab = r["label"]
        if lab == "none":
            continue
        # users often anticipate the cue: also search a little before it, but never into the previous window
        a0 = r["t_start"] - pre
        if j > 0:
            a0 = max(a0, rows[j - 1]["t_end"])
        w = np.flatnonzero((T >= a0) & (T < r["t_end"]))
        if len(w) == 0:
            continue
        if lab not in cls_idx:
            y[w] = IGNORE
            continue
        kind = kinds.get(lab, "instant")
        sw = score[w]
        thr = max(lc["k_abs"], lc["k_rel"] * sw.max())
        spans = _spans(sw > thr, merge if kind == "hold" else merge_inst)
        fallback = not spans
        if fallback:
            cue_w = np.flatnonzero((T[w] >= r["t_start"]))  # centre of the CUE window, not the search window
            c = int(cue_w[len(cue_w) // 2]) if len(cue_w) else len(w) // 2
            half = int(round(lc["fallback_half_s"] / hop)) if kind == "instant" else int(0.3 * len(cue_w))
            found = [(max(0, c - half), min(len(w), c + half))]
            log.info("fallback to window centre: %s at t=%.2f", lab, r["t_start"])
        elif kind == "hold":
            if lc.get("hold_to_cue_end", True):
                end = len(w)                              # w ends at the cue end
                gap = int(round(lc.get("hold_release_gap_s", 0.4) / hop))
                for s in spans[1:]:                       # a separate later burst = an early release
                    if s[0] - spans[0][1] >= gap:
                        end = s[0]
                        break
                found = [(spans[0][0], end)]
            else:
                found = [(spans[0][0], spans[-1][1])]
        else:
            pk = sw.max()
            strong = [s for s in spans if sw[s[0]:s[1]].max() >= lc.get("burst_rel", 0.5) * pk]
            if lc.get("multi_burst", False):
                found = strong
            else:
                found = [max(spans, key=lambda s: sw[s[0]:s[1]].sum())]
                for a, b in strong:
                    if (a, b) != found[0]:
                        y[max(0, w[0] + a - edge):min(F, w[0] + b + edge)] = IGNORE
        fixed = []
        for a, b in found:
            if b - a < min_len:
                c = (a + b) // 2
                a, b = max(0, c - min_len // 2), min(len(w), c + (min_len + 1) // 2)
            fixed.append((w[0] + a, w[0] + b))
        for ga, gb in fixed:  # ignore edges first, so a neighbouring burst's label is not overwritten
            y[max(0, ga - edge):ga] = IGNORE
            y[gb:min(F, gb + edge)] = IGNORE
        for ga, gb in fixed:
            y[ga:gb] = cls_idx[lab]
            trials.append({"label": lab, "kind": kind, "cue": (r["t_start"], r["t_end"]),
                           "span": (float(T[ga]), float(T[gb - 1]) + hop), "frames": (int(ga), int(gb)),
                           "fallback": fallback, "peak": float(sw.max()), "n_in_cue": len(fixed)})
    n_fb = sum(t["fallback"] for t in trials)
    if n_fb:
        log.warning("label localiser: %d/%d trials fell back to window centre", n_fb, len(trials))
    return y, trials, score


def _is_rest(r, rest_periods) -> bool:
    for a, b in rest_periods or []:
        if abs(r["t_start"] - a) < 0.5 and abs(r["t_end"] - b) < 0.5:
            return True
    return False
