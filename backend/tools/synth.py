"""Synthetic session generator (INTERFACE.md section 6 layout). For plumbing and tests only.

Each fake gesture leaves a distinct signature in every modality:
  swipe_left / swipe_right : wrist flick (gyro z pulse, lateral accel) + wrist moves across the image,
                             EMG on ch1 (left) or ch2 (right). Slow drift back afterwards.
  fist                     : strong EMG on both channels + fingers curl.
  pinch_hold (hold)        : sustained moderate EMG on ch1 + thumb/index tips meet for the held duration.
The free-activity block contains look-alike distractors (typing bursts, slow moves, grasps, rotations,
hand leaving view) so false-positive counts mean something.

Writes truth.csv (true motion spans) next to the contract files; the loader ignores it.

    python -m backend.tools.synth                       # 1 session into backend/_synth/
    python -m backend.tools.synth --n 2 --reps 20 --free 150
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from backend.common.config import load_config, repo_path

FS_EMG, FS_IMU, FPS = 1000, 200, 30.0
T0 = 1000.0  # perf_counter-like origin

DEFAULT_GESTURES = [
    {"id": "swipe_left", "name": "Swipe left", "kind": "instant", "action": {"type": "scroll", "dx": 0, "dy": -5}},
    {"id": "swipe_right", "name": "Swipe right", "kind": "instant", "action": {"type": "scroll", "dx": 0, "dy": 5}},
    {"id": "fist", "name": "Fist", "kind": "instant", "action": {"type": "click", "button": "right"}},
    {"id": "pinch_hold", "name": "Pinch and hold", "kind": "hold", "action": {"type": "drag", "button": "left"}},
]

# Hand template: right hand, palm to camera, fingers up. Units = palm size (wrist -> middle MCP = 1).
# Image y grows downward, so "up" is negative y.
_FINGERS = {  # MCP x, MCP y, segment lengths, direction angle (deg from vertical)
    "thumb": (-0.35, -0.25, [0.35, 0.32, 0.28], -50),
    "index": (-0.32, -0.98, [0.42, 0.25, 0.22], -6),
    "middle": (0.0, -1.0, [0.46, 0.28, 0.24], 0),
    "ring": (0.27, -0.95, [0.42, 0.26, 0.22], 6),
    "pinky": (0.50, -0.85, [0.33, 0.20, 0.18], 14),
}


def hand_template(curl: float = 0.0, pinch: float = 0.0) -> np.ndarray:
    """21x3 landmarks in palm units, wrist at origin. curl in [0,1] (fist), pinch in [0,1]."""
    # MediaPipe order: 0 wrist; thumb 1-4 (CMC,MCP,IP,TIP); index 5-8; middle 9-12; ring 13-16; pinky 17-20.
    pts = [np.zeros(3)]
    for name, (mx, my, segs, ang) in _FINGERS.items():
        p = np.array([mx, my, 0.0])  # thumb: CMC; fingers: MCP
        a = np.deg2rad(ang)
        d = np.array([np.sin(a), -np.cos(a), 0.0])
        bend = curl * (0.3 if name == "thumb" else 1.0) * np.deg2rad(70)
        pts.append(p.copy())
        for L in segs:  # each joint bends toward the palm: rotate direction into -z
            d = np.array([d[0] * np.cos(bend), d[1] * np.cos(bend), d[2] * np.cos(bend) - np.sin(bend)])
            d /= np.linalg.norm(d)
            p = p + L * d
            pts.append(p.copy())
    arr = np.array(pts)
    assert arr.shape == (21, 3)
    if pinch > 0:  # thumb tip (4) and index tip (8) meet; index DIP (7) follows a bit
        mid = 0.5 * (arr[4] + arr[8])
        arr[4] += pinch * (mid - arr[4])
        arr[8] += pinch * (mid - arr[8])
        arr[3] += 0.5 * pinch * (mid - arr[3])
        arr[7] += 0.5 * pinch * (mid - arr[7])
    return arr


def _hann_env(n: int) -> np.ndarray:
    return np.hanning(n + 2)[1:-1] if n > 0 else np.zeros(0)


def _plateau_env(n: int, ramp: int) -> np.ndarray:
    e = np.ones(n)
    r = min(ramp, n // 2)
    if r > 0:
        w = np.hanning(2 * r + 2)[1:-1]
        e[:r] = w[:r]
        e[-r:] = w[r:]
    return e


class _Builder:
    def __init__(self, dur: float, rng: np.random.Generator):
        self.rng = rng
        self.dur = dur
        self.ne = int(dur * FS_EMG)
        self.ni = int(dur * FS_IMU)
        self.nh = int(dur * FPS)
        self.emg_env = np.zeros((self.ne, 2))
        self.acc = np.zeros((self.ni, 3))
        self.gyro = np.zeros((self.ni, 3))
        self.wrist = np.zeros((self.nh, 2))  # displacement in image units
        self.curl = np.zeros(self.nh)
        self.pinch = np.zeros(self.nh)
        self.visible = np.ones(self.nh, bool)

    # ---- primitives (t in seconds from session start) ----
    def emg_burst(self, t, d, a1, a2, hold=False):
        i0, n = int(t * FS_EMG), max(int(d * FS_EMG), 2)
        i1 = min(i0 + n, self.ne)
        env = (_plateau_env(n, int(0.08 * FS_EMG)) if hold else _hann_env(n))[: i1 - i0]
        self.emg_env[i0:i1, 0] += a1 * env
        self.emg_env[i0:i1, 1] += a2 * env

    def gyro_pulse(self, t, axis, amp, sigma):
        tt = np.arange(self.ni) / FS_IMU
        m = np.abs(tt - t) < 4 * sigma
        self.gyro[m, axis] += amp * np.exp(-0.5 * ((tt[m] - t) / sigma) ** 2)

    def acc_biphasic(self, t, axis, amp, sigma):
        tt = np.arange(self.ni) / FS_IMU
        m = np.abs(tt - t) < 4 * sigma
        x = (tt[m] - t) / sigma
        self.acc[m, axis] += -amp * x * np.exp(0.5 - 0.5 * x**2)  # derivative-of-gaussian, peak |amp|

    def wrist_move(self, t, d, dx, dy, back_d=None):
        """Smooth move by (dx,dy) over d seconds; optionally drift back over back_d seconds."""
        tt = np.arange(self.nh) / FPS
        s = np.clip((tt - t) / d, 0, 1)
        prof = 0.5 - 0.5 * np.cos(np.pi * s)
        if back_d:
            sb = np.clip((tt - t - d) / back_d, 0, 1)
            prof = prof - (0.5 - 0.5 * np.cos(np.pi * sb))
        self.wrist[:, 0] += dx * prof
        self.wrist[:, 1] += dy * prof

    def shape(self, arr, t, d, amp, hold=False):
        i0, n = int(t * FPS), max(int(d * FPS), 2)
        i1 = min(i0 + n, self.nh)
        env = (_plateau_env(n, max(int(0.12 * FPS), 1)) if hold else _hann_env(n))[: i1 - i0]
        arr[i0:i1] = np.maximum(arr[i0:i1], amp * env)

    # ---- gestures ----
    def gesture(self, gid, t, d, a):
        r = self.rng
        if gid in ("swipe_left", "swipe_right"):
            sgn = -1.0 if gid == "swipe_left" else 1.0
            self.emg_burst(t, d, a * (0.8 if sgn < 0 else 0.25), a * (0.25 if sgn < 0 else 0.8))
            c = t + d / 2
            self.gyro_pulse(c, 2, -sgn * 220 * a, d / 5)            # yaw flick about z
            self.acc_biphasic(c, 1, 0.6 * a * (1 if sgn < 0 else -1), d / 5)  # toward thumb (-y) for left
            self.wrist_move(t, d, sgn * 0.22 * a, r.normal(0, 0.01), back_d=r.uniform(1.0, 1.6))
            self.gyro_pulse(c + d / 2 + 0.7, 2, sgn * 40 * a, 0.25)  # slow return, weak
        elif gid == "fist":
            self.emg_burst(t, d, 1.2 * a, 1.0 * a)
            self.gyro_pulse(t + d / 2, 1, 35 * a, d / 4)
            self.acc_biphasic(t + d / 2, 2, 0.08 * a, d / 4)
            self.shape(self.curl, t, d, min(1.0, 0.9 * a))
        elif gid == "pinch_hold":
            self.emg_burst(t, d, 0.55 * a, 0.2 * a, hold=True)
            self.shape(self.pinch, t, d, 1.0, hold=True)
            self.gyro_pulse(t + 0.05, 0, 15 * a, 0.05)
        else:  # unknown gesture id: random but consistent-per-id signature
            h = abs(hash(gid)) % 1000 / 1000
            self.emg_burst(t, d, a * h, a * (1 - h))
            self.gyro_pulse(t + d / 2, int(h * 3) % 3, 150 * a, d / 5)
            self.wrist_move(t, d, 0.0, (h - 0.5) * 0.3, back_d=1.0)

    def distractor(self, t):
        r = self.rng
        kind = r.choice(["typing", "move", "grasp", "rotate", "mouse", "half_swipe"], p=[0.3, 0.2, 0.15, 0.15, 0.15, 0.05])
        if kind == "typing":
            for k in range(r.integers(3, 10)):
                tk = t + k * r.uniform(0.08, 0.25)
                ch = r.integers(0, 2)
                a = r.uniform(0.15, 0.5)
                self.emg_burst(tk, 0.05, a if ch == 0 else 0.3 * a, a if ch == 1 else 0.3 * a)
                self.acc_biphasic(tk + 0.02, 2, r.uniform(0.02, 0.08), 0.015)
            self.shape(self.curl, t, 1.0, r.uniform(0.05, 0.2))
            return 1.5
        if kind == "move":
            d = r.uniform(0.8, 2.0)
            self.wrist_move(t, d, r.normal(0, 0.12), r.normal(0, 0.08))
            self.acc_biphasic(t + d / 2, r.integers(0, 3), r.uniform(0.05, 0.15), d / 4)
            self.gyro_pulse(t + d / 2, r.integers(0, 3), r.normal(0, 30), d / 4)
            return d
        if kind == "grasp":
            d = r.uniform(0.6, 1.5)
            self.emg_burst(t, d, r.uniform(0.3, 0.7), r.uniform(0.3, 0.7), hold=True)
            self.shape(self.curl, t, d, r.uniform(0.3, 0.6), hold=True)
            return d
        if kind == "rotate":
            d = r.uniform(0.3, 0.8)
            self.gyro_pulse(t + d / 2, r.integers(0, 2), r.choice([-1, 1]) * r.uniform(60, 150), d / 4)
            self.emg_burst(t, d, r.uniform(0.1, 0.3), r.uniform(0.1, 0.3))
            return d
        if kind == "mouse":
            for k in range(r.integers(2, 6)):
                tk = t + k * r.uniform(0.2, 0.6)
                self.emg_burst(tk, 0.08, r.uniform(0.2, 0.5), r.uniform(0.05, 0.2))
                self.wrist_move(tk, 0.15, r.normal(0, 0.03), r.normal(0, 0.03))
            return 2.0
        # half_swipe: a weak, slow lateral move (should not trigger)
        d = r.uniform(0.6, 1.0)
        self.wrist_move(t, d, r.choice([-1, 1]) * r.uniform(0.06, 0.12), 0.0, back_d=1.0)
        self.gyro_pulse(t + d / 2, 2, r.choice([-1, 1]) * r.uniform(30, 70), d / 3)
        return d + 1.0


def _bandlimited(n, rng, lo=30, hi=250):
    sos = butter(2, [lo, hi], "bandpass", fs=FS_EMG, output="sos")
    x = sosfilt(sos, rng.standard_normal((n, 2)), axis=0)
    return x / (x.std(0, keepdims=True) + 1e-9)


def generate_session(out_root: Path, gestures: list[dict], reps: int = 20, free_s: float = 150.0, seed: int = 0,
                     tag: str = "synth", created: datetime | None = None, cam_dropout: float = 0.03) -> Path:
    rng = np.random.default_rng(seed)
    created = created or datetime(2026, 11, 7, 12, 0, 0) + timedelta(minutes=seed)
    sid = created.strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    out = Path(out_root) / sid
    out.mkdir(parents=True, exist_ok=True)

    # ---- schedule ----
    labels, truth = [], []
    t = 0.5
    rest = (t, t + 10.0)
    labels.append((rest[0], rest[1], "none"))
    t = rest[1]
    order = [g for g in gestures for _ in range(reps)]
    rng.shuffle(order)
    plan = []
    for g in order:
        t += rng.uniform(1.5, 2.5)
        win = 3.0 if g["kind"] == "hold" else 2.0
        plan.append((g, t, t + win))
        labels.append((t, t + win, g["id"]))
        t += win
    t += 2.0
    free = (t, t + free_s)
    labels.append((free[0], free[1], "none"))
    dur = free[1] + 2.0

    b = _Builder(dur, rng)
    gain = rng.uniform(0.7, 1.3, 2) * 420.0  # per-session electrode gain (ADC counts per unit activation)

    for g, ws, we in plan:
        a = rng.uniform(0.75, 1.25)
        if g["kind"] == "hold":
            d = rng.uniform(1.5, (we - ws) - 0.8)
            on = ws + rng.uniform(0.3, (we - ws) - d - 0.15)
        else:
            d = rng.uniform(0.28, 0.45) if g["id"].startswith("swipe") else rng.uniform(0.3, 0.5)
            on = ws + rng.uniform(0.25, (we - ws) - d - 0.15)
        b.gesture(g["id"], on, d, a)
        truth.append((on, on + d, g["id"]))

    # relax-gap micro activity + free activity distractors
    tt = free[0] + 0.5
    while tt < free[1] - 2.0:
        tt += b.distractor(tt) + rng.uniform(0.3, 2.0)
    for _ in range(2):  # hand leaves view during free activity
        s = rng.uniform(free[0], free[1] - 8)
        b.visible[int(s * FPS): int((s + rng.uniform(3, 8)) * FPS)] = False

    # ---- EMG ----
    te = np.arange(b.ne) / FS_EMG
    act = _bandlimited(b.ne, rng)
    base = rng.standard_normal((b.ne, 2)) * 8.0
    hum = (22 * np.sin(2 * np.pi * 60 * te + rng.uniform(0, 6.28)) + 5 * np.sin(2 * np.pi * 180 * te))[:, None]
    drift = 30 * np.sin(2 * np.pi * 0.05 * te)[:, None]
    emg = 2048 + base + hum + drift + act * b.emg_env * gain[None, :]
    emg = np.clip(np.round(emg), 0, 4095).astype(np.uint16)

    # ---- IMU ----
    ti = np.arange(b.ni) / FS_IMU
    grav = np.array([0.03, -0.02, -1.0])  # -z is up (palm down): accel reads -1 g on z at rest
    acc = grav + b.acc + rng.normal(0, 0.01, (b.ni, 3))
    gyro = b.gyro + rng.normal(0, 1.0, (b.ni, 3)) + np.array([0.5, -0.3, 0.2])
    imu = np.hstack([acc, gyro]).astype(np.float32)

    # ---- hand ----
    th = np.arange(b.nh) / FPS + rng.normal(0, 0.003, b.nh)
    th = np.maximum.accumulate(th)
    wx0, wy0 = 0.5, 0.72
    walk = np.cumsum(rng.normal(0, 0.0015, (b.nh, 2)), 0)
    walk -= np.convolve(walk[:, 0], np.ones(90) / 90, "same")[:, None] * 0  # keep it simple
    aspect = 640 / 480
    scale = 0.16  # palm size in image-height units
    hand = np.full((b.nh, 21, 3), np.nan, np.float32)
    score = np.zeros(b.nh, np.float32)
    drop = rng.random(b.nh) < cam_dropout
    for k in range(b.nh):
        if not b.visible[k] or drop[k]:
            continue
        lm = hand_template(b.curl[k], b.pinch[k])
        w = np.array([wx0, wy0]) + b.wrist[k] + np.clip(walk[k], -0.1, 0.1)
        x = w[0] + lm[:, 0] * scale / aspect
        y = w[1] + lm[:, 1] * scale
        z = lm[:, 2] * scale * 0.5
        pts = np.stack([x, y, z], 1) + rng.normal(0, 0.002, (21, 3))
        hand[k] = pts
        score[k] = rng.uniform(0.85, 0.99)

    # ---- write ----
    np.save(out / "emg.npy", emg)
    np.save(out / "emg_t.npy", T0 + te)
    np.save(out / "imu.npy", imu)
    np.save(out / "imu_t.npy", T0 + ti + 0.0002)
    np.save(out / "hand.npy", hand)
    np.save(out / "hand_t.npy", T0 + th)
    np.save(out / "hand_score.npy", score)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "label", "source"])
        for a, c, lab in labels:
            w.writerow([f"{T0 + a:.4f}", f"{T0 + c:.4f}", lab, "cue"])
    with open(out / "truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "label"])
        for a, c, lab in truth:
            w.writerow([f"{T0 + a:.4f}", f"{T0 + c:.4f}", lab])
    snap = {"version": 1, "gestures": [dict(g, created=created.isoformat()) for g in gestures]}
    meta = {
        "session_id": sid, "created": created.isoformat(), "fs_emg": FS_EMG, "fs_imu": FS_IMU,
        "cam_latency": 0.08, "imu_present": True, "electrode_notes": "synthetic", "imu_notes": "synthetic",
        "gestures_snapshot": snap, "rest_periods": [[T0 + rest[0], T0 + rest[1]]], "protocol_version": "0.2",
        "img_w": 640, "img_h": 480, "synthetic": True,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=cfg["paths"]["synth"])
    ap.add_argument("--n", type=int, default=1, help="number of sessions")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--free", type=float, default=150.0, help="free-activity seconds")
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--gestures", nargs="*", help="subset of default ids, or new ids (kind=instant)")
    a = ap.parse_args(argv)
    gl = DEFAULT_GESTURES
    if a.gestures:
        known = {g["id"]: g for g in DEFAULT_GESTURES}
        gl = [known.get(g, {"id": g, "name": g, "kind": "instant", "action": {}}) for g in a.gestures]
    for i in range(a.n):
        t = time.perf_counter()
        p = generate_session(repo_path(a.out), gl, a.reps, a.free, seed=a.seed + i)
        print(f"wrote {p}  ({time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
