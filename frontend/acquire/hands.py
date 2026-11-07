"""Which visible hand is which: the GESTURE hand (wears the IMU, goes to the backend as `hand`) and the
POINTER hand (the other one, drives the cursor, published as `pointer_hand`).

Evidence, per tracked hand:
  1. Motion vs IMU (strongest): if a hand moves in the video while the IMU is quiet, it is NOT the IMU
     hand; if its motion rises and falls together with the IMU's, it IS. A still hand while the IMU moves
     is not the IMU hand either.
  2. MediaPipe's left/right label (weaker; it can flip, especially with one hand in view).
Hands are tracked frame to frame by palm position, so evidence accumulates per hand, and a role only
changes when the evidence clearly says so (hysteresis).

    python -m frontend.acquire.hands --selftest
"""
from __future__ import annotations

import collections
import itertools

import numpy as np

PALM = [0, 5, 9, 13, 17]


class Track:
    _ids = itertools.count(1)

    def __init__(self, palm, t, label_vote):
        self.id = next(self._ids)
        self.palm = palm
        self.last_t = t
        self.hist = collections.deque(maxlen=40)       # (t_raw, x_px, y_px)
        self.hvote = label_vote                        # +1 = label says IMU hand, -1 = says other
        self.iscore = 0.0                              # motion-vs-IMU evidence, + = IMU hand
        self.role = None                               # "gesture" | "pointer"
        self.last_eval = 0.0

    @property
    def score(self):
        return 0.7 * self.iscore + 0.3 * self.hvote


class HandAssigner:
    def __init__(self, imu_hand="Right"):
        self.imu_hand = imu_hand
        self.tracks: list[Track] = []

    def _label_vote(self, h):
        if h.get("handedness") is None:
            return 0.0
        v = h.get("score", 0.5)
        return v if h["handedness"] == self.imu_hand else -v

    def assign(self, t_raw, hands, w, h, imu_motion, cam_latency, imu_ok=True):
        """hands: camera output list. imu_motion(t0, t1) -> (times, motion) on the host clock (IMU time).
        Returns (gesture_hand_dict or None, pointer_hand_dict or None)."""
        dets = []
        for hd in hands:
            p = hd["lm"][PALM, :2].mean(axis=0)
            dets.append((hd, p))
        # --- match detections to tracks (greedy nearest palm)
        free = list(self.tracks)
        matched = []
        for hd, p in sorted(dets, key=lambda d: d[1][0]):
            best = min(free, key=lambda tr: np.linalg.norm(tr.palm - p), default=None)
            if best is not None and np.linalg.norm(best.palm - p) < 0.2:
                free.remove(best)
                tr = best
                tr.hvote = 0.9 * tr.hvote + 0.1 * self._label_vote(hd)
            else:
                tr = Track(p, t_raw, self._label_vote(hd))
                self.tracks.append(tr)
            tr.palm, tr.last_t = p, t_raw
            tr.hist.append((t_raw, p[0] * w, p[1] * h))
            matched.append((tr, hd))
        self.tracks = [tr for tr in self.tracks if t_raw - tr.last_t < 0.5]

        # --- motion evidence (every ~0.15 s per track)
        if imu_ok:
            for tr, _ in matched:
                if t_raw - tr.last_eval > 0.15 and len(tr.hist) >= 10:
                    tr.last_eval = t_raw
                    e = self._evidence(tr, imu_motion, cam_latency)
                    if e is not None:
                        tr.iscore = float(np.clip(0.8 * tr.iscore + 0.2 * e, -1, 1))

        # --- roles
        if len(matched) >= 2:
            matched.sort(key=lambda m: -m[0].score)
            g, p = matched[0], matched[1]
            if g[0].role == "pointer" and p[0].role == "gesture" and g[0].score - p[0].score < 0.15:
                g, p = p, g                            # hysteresis: keep the previous assignment
            g[0].role, p[0].role = "gesture", "pointer"
            return g[1], p[1]
        if len(matched) == 1:
            tr, hd = matched[0]
            s = tr.score
            if tr.role is None or (tr.role == "gesture" and s < -0.15) or (tr.role == "pointer" and s > 0.15):
                tr.role = "gesture" if s >= 0 else "pointer"
            return (hd, None) if tr.role == "gesture" else (None, hd)
        return None, None

    @staticmethod
    def _evidence(tr, imu_motion, cam_latency):
        a = np.array(tr.hist)[-18:]                    # ~0.6 s
        t = a[:, 0] - cam_latency                      # camera stamps -> true time
        dt = np.maximum(np.diff(t), 1e-3)
        sp = np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1) / dt     # px/s
        tm = 0.5 * (t[1:] + t[:-1])
        it, im = imu_motion(tm[0] - 0.05, tm[-1] + 0.05)
        if len(it) < 10:
            return None
        imu = np.interp(tm, it, im)
        cam_level, imu_level = float(np.median(sp)), float(np.median(imu))
        if cam_level > 120:                            # this hand is clearly moving
            if imu_level < 12:
                return -1.0                            # ...but the IMU is quiet: not the IMU hand
            if sp.std() > 1e-6 and imu.std() > 1e-6 and np.corrcoef(sp, imu)[0, 1] > 0.4:
                return 1.0
            return 0.5 if imu_level > 30 else 0.0
        if cam_level < 40 and imu_level > 40:
            return -0.7                                # still in view while the IMU hand moves
        return None


def imu_motion_from(hist_rows):
    """hist rows (t, ax, ay, az, gx, gy, gz) -> motion (deg/s-like): |gyro| + 200 * | |a| - 1 g |"""
    a = np.asarray(hist_rows, float)
    if len(a) == 0:
        return np.zeros(0), np.zeros(0)
    acc = np.linalg.norm(a[:, 1:4], axis=1)
    gyr = np.linalg.norm(a[:, 4:7], axis=1)
    return a[:, 0], gyr + 200 * np.abs(acc - 1.0)


# ----------------------------------------------------------------------------- self test
def _selftest():
    rng = np.random.default_rng(0)
    fs_cam, lat, T = 30, 0.12, 12.0
    # true motion: right (IMU) hand on the left side of the image moves during 2-4 s and 8-10 s;
    # left (pointer) hand on the right side moves during 5-7 s and 9.5-11 s.
    def moving(t, spans):
        return any(a <= t < b for a, b in spans)
    R_sp, L_sp = [(2, 4), (8, 10)], [(5, 7), (9.5, 11)]
    imu_t = np.arange(0, T, 1 / 200)
    rows = []
    for t in imu_t:
        g = 80 * (1 + np.sin(9 * t)) if moving(t, R_sp) else 2.0
        rows.append((t, 0, 0, -1, g, 0, 0))
    rows = np.array(rows)

    def imu_motion(t0, t1):
        m = (rows[:, 0] >= t0) & (rows[:, 0] <= t1)
        return imu_motion_from(rows[m])

    def hand(x0, y0, t, spans, label):
        x = x0 + (0.08 * np.sin(6 * t) if moving(t, spans) else 0) + rng.normal(0, 0.001)
        lm = np.zeros((21, 3), np.float32)
        lm[:, 0], lm[:, 1] = x, y0
        return {"lm": lm, "handedness": label, "score": 0.8}

    A = HandAssigner("Right")
    roles = []
    for k in range(int(T * fs_cam)):
        tt = k / fs_cam
        # MediaPipe labels deliberately WRONG half the time to prove the motion evidence wins
        lab_r = "Right" if (k // 45) % 2 == 0 else "Left"
        lab_l = "Left" if (k // 45) % 2 == 0 else "Right"
        hands = []
        if tt < 7.5 or tt >= 8:                         # both hands visible except a short gap
            hands.append(hand(0.3, 0.5, tt, R_sp, lab_r))
        hands.append(hand(0.7, 0.5, tt, L_sp, lab_l))
        g, p = A.assign(tt + lat, hands, 640, 480, imu_motion, lat)
        gx = None if g is None else float(g["lm"][0, 0])
        px = None if p is None else float(p["lm"][0, 0])
        roles.append((tt, gx, px))
    late = [r for r in roles if r[0] > 4.5]
    ok = sum(1 for _, gx, px in late if (gx is None or gx < 0.5) and (px is None or px > 0.5)) / len(late)
    alone = [r for r in roles if 7.6 <= r[0] < 7.95]   # only the left hand in view
    assert all(gx is None and px is not None for _, gx, px in alone), alone[:3]
    assert ok > 0.97, ok
    print(f"hands selftest: OK ({100 * ok:.1f}% correct after 4.5 s despite flipping labels; "
          f"left hand alone -> pointer)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
