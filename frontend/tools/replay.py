"""Publish sensor data on port 5555 exactly per INTERFACE.md §3.

    python -m frontend.tools.replay --sim                     synthetic emg/imu/hand/status
    python -m frontend.tools.replay --sim --truth truth.csv   ...and log ground-truth gestures
    python -m frontend.tools.replay --sim --no-imu --drop-rate 0.01
    python -m frontend.tools.replay --selftest                run the sim and check it with the monitor

(Replaying a recorded data/sessions/<id>/ arrives with the recorder in M4.)

The sim draws gestures from data/gestures.json if it exists, otherwise a default set.
Every gesture has a consistent signature in all three sensors (EMG burst pattern,
IMU motion, hand landmark motion), so a model can actually learn it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):          # allow `python frontend/tools/replay.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import protocol as P  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EMG_CHUNK = 20                      # samples per emg message (20 ms)
IMU_CHUNK = 4                       # samples per imu message (20 ms)
CAM_FPS = 30.0
CAM_LATENCY = 0.08                  # hand messages arrive this late (like a real webcam)

DEFAULT_GESTURES = [
    {"id": "swipe_left", "kind": "instant"},
    {"id": "swipe_right", "kind": "instant"},
    {"id": "tap", "kind": "instant"},
    {"id": "fist_hold", "kind": "hold"},
]

# Hand-written signatures for the defaults; unknown ids get one derived from a hash of the id.
SIGNATURES = {
    "swipe_left":  dict(emg=(1.0, 0.35), gyro=(0, 0, 160), acc=(-0.6, 0, 0), dx=-0.18, dy=0.0, curl=0.0, pinch=0.0),
    "swipe_right": dict(emg=(0.35, 1.0), gyro=(0, 0, -160), acc=(0.6, 0, 0), dx=0.18, dy=0.0, curl=0.0, pinch=0.0),
    "tap":         dict(emg=(0.9, 0.9), gyro=(220, 0, 0), acc=(0, 0, 1.5), dx=0.0, dy=0.03, curl=0.3, pinch=0.0),
    "fist_hold":   dict(emg=(0.8, 0.7), gyro=(0, 0, 0), acc=(0, 0, 0), dx=0.0, dy=0.0, curl=1.0, pinch=0.0),
    "pinch_hold":  dict(emg=(0.5, 0.25), gyro=(0, 0, 0), acc=(0, 0, 0), dx=0.0, dy=0.0, curl=0.0, pinch=1.0),
}


def signature(gid: str) -> dict:
    if gid in SIGNATURES:
        return SIGNATURES[gid]
    r = np.random.default_rng(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16))
    return dict(emg=tuple(r.uniform(0.2, 1.0, 2)), gyro=tuple(r.uniform(-200, 200, 3)),
                acc=tuple(r.uniform(-0.8, 0.8, 3)), dx=r.uniform(-0.15, 0.15), dy=r.uniform(-0.1, 0.1),
                curl=r.uniform(0, 1), pinch=r.uniform(0, 1) if r.random() < 0.3 else 0.0)


def load_gestures() -> list:
    p = REPO / "data" / "gestures.json"
    try:
        g = json.loads(p.read_text(encoding="utf-8"))["gestures"]
        g = [{"id": x["id"], "kind": x.get("kind", "instant")} for x in g]
        if g:
            return g
    except Exception:
        pass
    return DEFAULT_GESTURES


# ----------------------------------------------------------------------------- hand model
def _template_hand() -> np.ndarray:
    """A flat right hand, palm to camera, fingers up. (21,3) normalised image coords."""
    lm = np.zeros((21, 3), np.float32)
    wrist = np.array([0.5, 0.78])
    lm[0, :2] = wrist
    # thumb: CMC, MCP, IP, TIP heading up-left
    for k, (x, y) in enumerate([(-0.05, -0.04), (-0.09, -0.09), (-0.12, -0.13), (-0.145, -0.17)]):
        lm[1 + k, :2] = wrist + (x, y)
    bases = [(-0.045, -0.20), (-0.012, -0.21), (0.02, -0.20), (0.05, -0.18)]    # index..pinky MCP
    lens = [0.075, 0.085, 0.078, 0.062]
    for f, ((bx, by), L) in enumerate(zip(bases, lens)):
        mcp = wrist + (bx, by)
        lm[5 + 4 * f, :2] = mcp
        for j in range(1, 4):
            lm[5 + 4 * f + j, :2] = mcp + (bx * 0.15 * j, -L * j * 0.6)
    lm[:, 2] = -0.02 * np.linspace(0, 1, 21)
    return lm


TEMPLATE = _template_hand()


def pose_hand(dx, dy, curl, pinch, rng) -> np.ndarray:
    lm = TEMPLATE.copy()
    # finger curl: pull PIP/DIP/TIP of index..pinky toward their MCP and the palm centre
    if curl > 0:
        palm = lm[[0, 5, 9, 13, 17], :2].mean(0)
        for f in range(4):
            m = 5 + 4 * f
            for j, w in zip((1, 2, 3), (0.35, 0.6, 0.8)):
                tgt = lm[m, :2] * 0.5 + palm * 0.5
                lm[m + j, :2] += (tgt - lm[m + j, :2]) * w * curl
                lm[m + j, 2] -= 0.04 * curl * j / 3
        lm[2:5, :2] += (palm - lm[2:5, :2]) * 0.4 * curl
    if pinch > 0:
        mid = (lm[4, :2] + lm[8, :2]) / 2
        lm[4, :2] += (mid - lm[4, :2]) * 0.9 * pinch
        lm[8, :2] += (mid - lm[8, :2]) * 0.9 * pinch
        lm[7, :2] += (mid - lm[7, :2]) * 0.4 * pinch
    lm[:, 0] += dx
    lm[:, 1] += dy
    lm[:, :2] += rng.normal(0, 0.0015, (21, 2))     # landmark jitter
    return lm.astype(np.float32)


# ----------------------------------------------------------------------------- the world
class SimWorld:
    """Ground truth: which gesture is active when, and what each sensor looks like."""

    def __init__(self, gestures, seed=None, mean_gap=(3.0, 8.0)):
        self.rng = np.random.default_rng(seed)
        self.gestures = gestures
        self.mean_gap = mean_gap
        self.events = []                 # (t_start, t_end, id, kind, sig)
        self.next_t = 2.0
        self.drift = np.zeros(2)         # slow hand wander
        self.tilt = np.array([0.0, 0.0, 1.0])
        self.no_hand_until = -1.0
        self.next_dropout = 15.0
        self.log = []

    def schedule_until(self, t):
        while self.next_t < t + 1.0:
            g = self.gestures[self.rng.integers(len(self.gestures))]
            dur = self.rng.uniform(1.0, 2.5) if g["kind"] == "hold" else self.rng.uniform(0.25, 0.45)
            ev = (self.next_t, self.next_t + dur, g["id"], g["kind"], signature(g["id"]))
            self.events.append(ev)
            self.log.append(ev)
            self.next_t = ev[1] + self.rng.uniform(*self.mean_gap)

    def prune(self, t):
        self.events = [e for e in self.events if e[1] > t - 2.0]

    @staticmethod
    def _shape(t, ev):
        """Per-event activation (0..1) for EMG and hand, and a signed pulse for IMU."""
        ts, te, _, kind, _ = ev
        ph = (t - ts) / (te - ts)
        inside = (ph >= 0) & (ph <= 1)
        phc = np.clip(ph, 0, 1)
        if kind == "hold":
            ramp = 0.12 / (te - ts)
            act = np.clip(np.minimum(phc / ramp, (1 - phc) / ramp), 0, 1)
            pulse = np.zeros_like(phc)
        else:
            act = np.sin(np.pi * phc) ** 2
            pulse = np.sin(2 * np.pi * phc)
        return np.where(inside, act, 0.0), np.where(inside, pulse, 0.0)

    def emg(self, idx0, n):
        t = (idx0 + np.arange(n)) / P.FS_EMG
        x = 2048 + self.rng.normal(0, 6, (n, 2)) + 12 * np.sin(2 * np.pi * 60 * t)[:, None]
        for ev in self.events:
            act, _ = self._shape(t, ev)
            if act.any():
                g = np.asarray(ev[4]["emg"]) * 450
                # band-limited-ish burst: white noise smoothed a little
                w = self.rng.normal(0, 1, (n, 2))
                x += w * act[:, None] * g
        return np.clip(np.rint(x), 0, 4095).astype(np.uint16)

    def imu(self, idx0, n):
        t = (idx0 + np.arange(n)) / P.FS_IMU
        out = np.zeros((n, 6), np.float32)
        out[:, :3] = self.tilt + self.rng.normal(0, 0.01, (n, 3))
        out[:, 3:] = self.rng.normal(0, 0.6, (n, 3)) + 0.3 * np.sin(2 * np.pi * 0.2 * t)[:, None]
        for ev in self.events:
            act, pulse = self._shape(t, ev)
            if act.any():
                s = ev[4]
                out[:, 3:] += pulse[:, None] * np.asarray(s["gyro"], np.float32)
                out[:, :3] += pulse[:, None] * np.asarray(s["acc"], np.float32)
                if ev[3] == "hold":   # held gestures: tremor while active
                    out[:, 3:] += act[:, None] * self.rng.normal(0, 4, (n, 3))
        return out

    def hand(self, t):
        """-> (landmarks (21,3) or (0,3), score)."""
        if t >= self.next_dropout:
            self.no_hand_until = t + self.rng.uniform(0.3, 1.5)
            self.next_dropout = t + self.rng.uniform(8, 20)
        if t < self.no_hand_until:
            return np.zeros((0, 3), np.float32), 0.0
        self.drift += self.rng.normal(0, 0.002, 2) - 0.02 * self.drift
        dx, dy, curl, pinch = self.drift[0], self.drift[1], 0.0, 0.0
        for ev in self.events:
            act, _ = self._shape(np.array([t]), ev)
            a = float(act[0])
            if a > 0:
                s = ev[4]
                ph = np.clip((t - ev[0]) / (ev[1] - ev[0]), 0, 1)
                if ev[3] == "instant":
                    dx += s["dx"] * np.sin(np.pi * ph)
                    dy += s["dy"] * np.sin(np.pi * ph)
                curl += s["curl"] * a
                pinch += s["pinch"] * a
        return pose_hand(dx, dy, min(curl, 1), min(pinch, 1), self.rng), float(self.rng.uniform(0.9, 0.99))


# ----------------------------------------------------------------------------- publisher
def run_sim(args, stop: threading.Event | None = None, ready: threading.Event | None = None):
    stop = stop or threading.Event()
    pub = P.make_pub(args.port)
    gestures = load_gestures()
    world = SimWorld(gestures, seed=args.seed)
    rng = np.random.default_rng(None if args.seed is None else args.seed + 1)
    truth_f = None
    if args.truth:
        truth_f = open(args.truth, "w", newline="")
        tw = csv.writer(truth_f)
        tw.writerow(["t_start", "t_end", "label", "source"])
    print(f"[sim] publishing on {P.addr(args.port)}  gestures={[g['id'] for g in gestures]}"
          f"  imu={'off' if args.no_imu else 'on'} hand={'off' if args.no_hand else 'on'}", flush=True)

    T0 = P.now()                       # host time of sim sample 0
    n_emg = n_imu = n_frame = 0
    dropped = 0
    last_emg_row = np.full(2, 2048, np.uint16)
    stat_next = 1.0
    stat_cnt = {"emg": 0, "imu": 0, "cam": 0}
    logged = 0
    if ready:
        ready.set()
    try:
        while not stop.is_set():
            el = P.now() - T0
            world.schedule_until(el)

            while (n_emg + EMG_CHUNK) / P.FS_EMG <= el:
                x = world.emg(n_emg, EMG_CHUNK)
                if args.drop_rate > 0 and rng.random() < args.drop_rate:
                    k = int(rng.integers(1, 10))            # lost samples -> repeat last value
                    a = int(rng.integers(0, EMG_CHUNK - k + 1))
                    x[a:a + k] = x[a - 1] if a > 0 else last_emg_row
                    dropped += k
                last_emg_row = x[-1].copy()
                P.send_array(pub, P.TOPIC_EMG, {"t0": T0 + n_emg / P.FS_EMG, "fs": P.FS_EMG, "index0": n_emg}, x)
                n_emg += EMG_CHUNK
                stat_cnt["emg"] += EMG_CHUNK

            while (n_imu + IMU_CHUNK) / P.FS_IMU <= el:
                y = world.imu(n_imu, IMU_CHUNK)
                if args.no_imu:
                    y[:] = np.nan
                P.send_array(pub, P.TOPIC_IMU, {"t0": T0 + n_imu / P.FS_IMU, "fs": P.FS_IMU, "index0": n_imu}, y)
                n_imu += IMU_CHUNK
                stat_cnt["imu"] += IMU_CHUNK

            while not args.no_hand and n_frame / CAM_FPS + CAM_LATENCY <= el:
                tf = n_frame / CAM_FPS
                lm, score = world.hand(tf)
                hdr = {"t": T0 + tf, "score": score, "handedness": "Right" if len(lm) else None,
                       "img_w": 640, "img_h": 480}
                P.send_array(pub, P.TOPIC_HAND, hdr, lm)
                n_frame += 1
                stat_cnt["cam"] += 1

            if el >= stat_next:
                P.send_json(pub, P.TOPIC_STATUS, {
                    "emg_ok": True, "imu_ok": not args.no_imu, "cam_ok": not args.no_hand,
                    "emg_rate": float(stat_cnt["emg"]), "imu_rate": float(stat_cnt["imu"]),
                    "cam_fps": float(stat_cnt["cam"]), "dropped": dropped,
                    "battery_warning_shown": True, "source": "sim"})
                stat_cnt = {k: 0 for k in stat_cnt}
                stat_next += 1.0
                world.prune(el)

            while logged < len(world.log) and world.log[logged][0] <= el:
                ts, te, gid, kind, _ = world.log[logged]
                if not args.quiet:
                    print(f"[sim] t={T0 + ts:10.3f}  {gid:<14} {kind:<7} {te - ts:.2f}s", flush=True)
                if truth_f:
                    tw.writerow([f"{T0 + ts:.4f}", f"{T0 + te:.4f}", gid, "sim"])
                    truth_f.flush()
                logged += 1

            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(0)
        if truth_f:
            truth_f.close()
        print("[sim] stopped", flush=True)


def selftest(args) -> int:
    from frontend.tools.monitor import Monitor
    args.port = args.port or P.PORT_SENSORS
    stop, ready = threading.Event(), threading.Event()
    args.quiet = True
    th = threading.Thread(target=run_sim, args=(args, stop, ready), daemon=True)
    th.start()
    ready.wait(5)
    mon = Monitor(port=args.port)
    res = mon.run(seconds=6.0, print_every=2.0)
    stop.set()
    th.join(3)
    ok = (abs(res["emg_rate"] - 1000) < 15 and abs(res["imu_rate"] - 200) < 5 and res["cam_fps"] >= 27
          and res["gaps"] == 0 and res["status_msgs"] >= 4)
    print("replay --sim selftest:", "OK" if ok else f"FAIL {res}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="publish synthetic data")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--port", type=int, default=P.PORT_SENSORS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-imu", action="store_true", help="IMU missing: NaN rows, imu_ok=false")
    ap.add_argument("--no-hand", action="store_true", help="no camera messages, cam_ok=false")
    ap.add_argument("--drop-rate", type=float, default=0.0, help="probability per emg chunk of a fake serial drop")
    ap.add_argument("--truth", help="write ground-truth gesture windows to this CSV (labels.csv format)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a)
    if a.sim:
        run_sim(a)
        return 0
    ap.error("session replay arrives in M4; use --sim for now")


if __name__ == "__main__":
    sys.exit(main())
