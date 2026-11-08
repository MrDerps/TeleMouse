"""Detect swapped EMG channels (ch1 <-> ch2) without retraining.

Idea: IMU and camera don't care how the EMG leads are plugged in, so gestures they help detect can be used
to check the EMG wiring. Each gesture has a typical balance between the two channels, learned from your
recordings: r = log(ch1 activity / ch2 activity), each channel measured relative to its own resting level.
Swapping the leads flips the sign of r. Live, for each confident gesture, compare the observed r with the
gesture's profile both ways round. Only gestures that clearly favour one channel count, and the decision needs
a clear majority over several gestures. If neither order fits, that's "EMG doesn't match training" (pads
moved), not a swap.

Polarity (+/- swapped on one channel) needs nothing: the EMG features (RMS, MAV, WL, ZC, SSC) and this
check are all blind to the sign of the signal.

    python -m frontend.acquire.emg_swap --check     offline test on data/sessions (with swapped copies)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

FS = 1000
ENV_S = 0.05                 # envelope RMS window
BURST_S = 0.30               # the strongest 300 ms inside a gesture window is "the gesture"
MIN_ASYM = 0.40              # |mean log-ratio| a gesture needs to be informative (~1.5x one channel over the other)
MIN_S = 0.15                 # floor on the profile spread
_HP = butter(2, 20, "highpass", fs=FS, output="sos")
_NB, _NA = iirnotch(60, 30, fs=FS)
SOS = np.vstack([_HP, tf2sos(_NB, _NA)])


def envelope(emg: np.ndarray) -> np.ndarray:
    """(N,2) raw counts -> (N,2) causal 20 Hz HP + 60 Hz notch, 50 ms moving RMS."""
    x = emg.astype(np.float64)
    zi = sosfilt_zi(SOS)[:, :, None] * x[0][None, None, :]
    y, _ = sosfilt(SOS, x, axis=0, zi=zi)
    n = int(ENV_S * FS)
    c = np.vstack([np.zeros((1, 2)), np.cumsum(y ** 2, axis=0)])
    env = np.zeros_like(y)
    env[n:] = np.sqrt(np.maximum(c[n + 1:] - c[1:-n], 0) / n)
    env[:n] = env[n] if len(env) > n else 0
    return env


def rest_level(env, t, rest_periods=None):
    """Per-channel resting envelope: median inside the rest calibration, else the quiet 20th percentile."""
    if rest_periods:
        m = np.zeros(len(t), bool)
        for a, b in rest_periods:
            m |= (t >= a + 1.0) & (t <= b)
        if m.sum() > FS:
            return np.median(env[m], axis=0) + 1e-6
    return np.percentile(env, 20, axis=0) + 1e-6


def burst_ratio(env_n, t, a, b):
    """-> (log(ch1/ch2), strength) of the strongest 300 ms inside [a, b]; None if too short."""
    m = (t >= a) & (t <= b)
    if m.sum() < BURST_S * FS:
        return None
    e = env_n[m]
    w = int(BURST_S * FS)
    tot = np.convolve(e.sum(axis=1), np.ones(w) / w, mode="valid")
    i = int(np.argmax(tot))
    seg = e[i:i + w].mean(axis=0)
    return float(np.log(seg[0] / seg[1])), float(tot[i])


def _load(session_dir: Path):
    emg = np.load(session_dir / "emg.npy")
    t = np.load(session_dir / "emg_t.npy")
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    with open(session_dir / "labels.csv") as f:
        labels = [(float(r["t_start"]), float(r["t_end"]), r["label"]) for r in csv.DictReader(f)]
    return emg, t, meta, labels


def session_ratios(session_dir: Path, swap=False):
    """-> {gesture: [log-ratios]} for every cue window in a session (optionally with the channels swapped)."""
    emg, t, meta, labels = _load(session_dir)
    if meta.get("emg_valid") is False or len(emg) < FS * 5:
        return {}
    if swap:
        emg = emg[:, ::-1]
    env = envelope(emg)
    env_n = env / rest_level(env, t, meta.get("rest_periods"))
    out = {}
    for a, b, lab in labels:
        if lab == "none":
            continue
        r = burst_ratio(env_n, t, a - 0.5, b)          # people anticipate the cue
        if r is not None and r[1] > 2.0:              # a real burst: > 2x the resting level
            out.setdefault(lab, []).append(r[0])
    return out


def build_profiles(session_dirs) -> dict:
    """gesture -> {"m": median log-ratio, "s": spread, "n": reps, "asym": informative?}"""
    allr = {}
    for d in session_dirs:
        try:
            ratios = session_ratios(Path(d))
        except (OSError, ValueError, KeyError) as e:      # a missing or broken recording must not stop the check
            print(f"[emg_swap] skipping {d}: {e}")
            continue
        for g, rs in ratios.items():
            allr.setdefault(g, []).extend(rs)
    prof = {}
    for g, rs in allr.items():
        r = np.array(rs)
        m = float(np.median(r))
        s = max(MIN_S, float(1.4826 * np.median(np.abs(r - m))))
        prof[g] = {"m": m, "s": s, "n": len(r), "asym": abs(m) >= MIN_ASYM and abs(m) / s >= 1.5 and len(r) >= 5}
    return prof


class SwapDetector:
    """Feed it (gesture, observed log-ratio) for confident gesture events; it votes and decides once."""

    def __init__(self, profiles, need=4, agree=0.8):
        self.p, self.need, self.agree = profiles, need, agree
        self.reset()

    def reset(self):
        self.votes = []          # "ok" | "swap" | "mismatch"
        self.decision = None     # None (still checking) | "ok" | "swap" | "mismatch"

    @property
    def informative(self):
        return [g for g, v in self.p.items() if v["asym"]]

    def feed(self, gid, r):
        if self.decision is not None:
            return None
        prof = self.p.get(gid)
        if not prof or not prof["asym"]:
            return None
        m, s = prof["m"], prof["s"]
        d_same, d_swap = abs(r - m) / s, abs(-r - m) / s
        if min(d_same, d_swap) > 3.5:
            v = "mismatch"
        elif d_swap - d_same > 1.0:
            v = "ok"
        elif d_same - d_swap > 1.0:
            v = "swap"
        else:
            return None
        self.votes.append(v)
        n = len(self.votes)
        for kind in ("swap", "ok", "mismatch"):
            k = self.votes.count(kind)
            if n >= self.need and k / n >= self.agree:
                self.decision = kind
                return kind
        return None


def _check():
    """Offline: leave-one-session-out profiles; replay each held-out session as-is and swapped."""
    root = Path(__file__).resolve().parents[2] / "data" / "sessions"
    dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    if len(dirs) < 2:
        print("need at least 2 sessions")
        return
    res = {"orig": [], "swapped": []}
    for i, d in enumerate(dirs):
        prof = build_profiles([x for j, x in enumerate(dirs) if j != i])
        print(f"held out {d.name}: profiles " + ", ".join(
            f"{g} m={v['m']:+.2f} s={v['s']:.2f} n={v['n']} {'INFORMATIVE' if v['asym'] else 'ambiguous'}"
            for g, v in prof.items()))
        for kind, swap in (("orig", False), ("swapped", True)):
            det = SwapDetector(prof)
            for g, rs in session_ratios(d, swap=swap).items():
                for r in rs:
                    det.feed(g, r)
            res[kind].append(det.decision)
            print(f"   {kind:8} -> decision {det.decision}  votes {det.votes}")
    caught = sum(x == "swap" for x in res["swapped"])
    false = sum(x == "swap" for x in res["orig"])
    print(f"\nswapped copies caught: {caught}/{len(res['swapped'])}   "
          f"false swaps on originals: {false}/{len(res['orig'])}   "
          f"(undecided originals: {sum(x is None for x in res['orig'])})")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        _check()
    else:
        print(__doc__)
