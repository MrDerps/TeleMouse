"""Pointer control (not the NN's job): moves the OS cursor from the IMU and the camera.

Modes
  * "right_pinch" (default): the RIGHT hand (the only hand tracked) steers by the centre of its palm, but only
    while you PINCH (thumb tip on index tip). Release the pinch and the cursor freezes. Detected directly
    from the landmarks, so no training is needed and it reacts within a frame.
  * "right_gesture": same, but the clutch is any trained HOLD gesture whose action is "pointer_hold"
    (Move mouse while held): cursor follows the hand from the gesture's start event until its end event.
  * "left" (needs two-hand tracking, `two_hands` in data/frontend_config.json; off for now): your OTHER hand (no IMU) steers. The camera tracks the centre of its palm; an OPEN hand
    moves the cursor, a FIST freezes it (close the fist, reposition, open again = lifting a mouse). No pointer
    hand in view = no movement. The acquisition service decides which visible hand is which and publishes the
    pointer hand on 5555 as `pointer_hand`.
  * "auto": the IMU gyro drives the cursor (fast: ~40 ms), the camera decides WHEN it moves
    (only while your hand is in view, so dropping the hand out of view is the "lift the mouse" clutch) and
    takes over if the IMU is missing.
  * "imu": gyro only.  * "camera": palm position only (~150 ms behind).  * "off".

Like a real mouse: speed (sensitivity), optional acceleration ("enhance pointer precision"), smoothing,
invert X/Y, plus "freeze while clenching": the cursor holds still while the EMG shows a muscle burst, so a
fist click lands where you aimed. Gyro axes are found by a 2-step direction calibration (any mounting).

Runs in its own thread with its own SUB socket on 5555, independent of the UI.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from shared import protocol as P

PALM = [0, 5, 9, 13, 17]
TIPS = [8, 12, 16, 20]
TOPIC_POINTER_HAND = b"pointer_hand"
OPEN_ABOVE, CLOSED_BELOW = 1.55, 1.25      # openness = mean(tip-wrist) / palm length, with hysteresis


PINCH_ON, PINCH_OFF = 0.30, 0.45          # thumb-index distance / palm length, with hysteresis


def pinch_ratio(lm) -> float:
    palm_len = np.linalg.norm(lm[9, :2] - lm[0, :2]) + 1e-6
    return float(np.linalg.norm(lm[4, :2] - lm[8, :2]) / palm_len)


def openness(lm) -> float:
    """~1.9 for an open hand, ~1.0 for a fist (scale- and distance-invariant)."""
    palm_len = np.linalg.norm(lm[9, :2] - lm[0, :2]) + 1e-6
    return float(np.mean(np.linalg.norm(lm[TIPS, :2] - lm[0, :2], axis=1)) / palm_len)

DEFAULTS = {
    "mode": "right_pinch", "speed": 12.0, "accel": True, "smoothing": 0.5, "invert_x": False, "invert_y": False,
    "need_hand": True, "freeze_on_clench": True,
    # gyro axis mapping: cursor x <- sign * gyro[axis_x], cursor y <- sign * gyro[axis_y]
    # default for the documented mount (+x fingertips, -y thumb, -z up), palm down: turn right = yaw, tilt down = pitch
    "axis_x": 2, "sign_x": -1, "axis_y": 1, "sign_y": -1,
    "deadzone_dps": 4.0, "cam_speed": 1.0,
}


class OneEuro:
    """One Euro filter (Casiez et al.) for a 2-D signal."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x

    def reset(self):
        self.x = self.dx = None


class ClenchGate:
    """True while the EMG shows a burst well above its own resting level (fist / pinch starting)."""

    def __init__(self, k=3.0, hold_s=0.15):
        self.k, self.hold_s = k, hold_s
        self.base = None
        self.until = 0.0
        self.n = 0

    def update(self, emg: np.ndarray, now: float) -> bool:
        x = emg.astype(np.float32)
        if np.mean((x <= 8) | (x >= 4087)) > 0.2:      # pad off / rails: no information
            return now < self.until
        r = np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=0)) + 1e-3   # cheap high-pass energy per channel
        self.n += 1
        if self.base is None:
            self.base = r
            return False
        active = bool(np.any(r > self.k * self.base)) and self.n > 50
        if not active:
            self.base = 0.98 * self.base + 0.02 * r     # follow the resting level slowly
        else:
            self.until = now + self.hold_s
        return active or now < self.until


class Pointer:
    def __init__(self, enabled_fn, settings: dict | None = None, log=print):
        """enabled_fn() -> bool: actions enabled and not killed (and pointer toggled on)."""
        from pynput import mouse
        self.mouse = mouse.Controller()
        self.enabled_fn = enabled_fn
        self.log = log
        self.s = dict(DEFAULTS)
        self.s.update(settings or {})
        self.bias = np.zeros(3)
        self._still = 0
        self.hand_seen_t = 0.0
        self.imu_t = 0.0
        self.last_palm = None
        self.frozen = False
        self.f_imu = OneEuro()
        self.f_cam = OneEuro()
        self.gate = ClenchGate()
        self.acc = np.zeros(2)                       # sub-pixel remainder
        self.source = "-"
        self.left_open = False
        self.clutch_gesture = False                 # set by the executor while a "pointer_hold" gesture is held
        self.pinched = False
        self._pinch_n = 0
        self.r_prev = None
        self.f_right = OneEuro()
        self.pinch = 0.0
        self.left_prev = None
        self.f_left = OneEuro()
        self.openness = 0.0
        self._rec = None                             # direction calibration capture
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, name="pointer", daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self._stop.set()
        self.th.join(1)

    def update(self, **kw):
        self.s.update(kw)
        self.f_imu.reset()
        self.f_cam.reset()

    # ------------------------------------------------------------------ direction calibration
    def capture(self, seconds=2.0) -> np.ndarray:
        """Mean bias-corrected gyro vector (deg/s) over the next `seconds` (blocking; call off the UI thread)."""
        self._rec = []
        time.sleep(seconds)
        rec, self._rec = self._rec, None
        if not rec:
            raise RuntimeError("no IMU data")
        g = np.array(rec)
        # use the dominant direction (mean of the samples with the most motion)
        mag = np.linalg.norm(g, axis=1)
        top = g[mag >= np.percentile(mag, 60)]
        return top.mean(axis=0)

    @staticmethod
    def axis_from(v: np.ndarray):
        ax = int(np.argmax(np.abs(v)))
        return ax, (1 if v[ax] > 0 else -1), float(abs(v[ax]) / (np.linalg.norm(v) + 1e-9))

    # ------------------------------------------------------------------ main loop
    def _gain(self, v):
        """v: px/s before gain shaping. Acceleration like Windows' 'enhance pointer precision'."""
        if not self.s["accel"]:
            return v
        sp = np.linalg.norm(v)
        return v * (0.6 + min(sp / 600.0, 1.4))

    def _move(self, d):
        self.acc += d
        step = np.trunc(self.acc)
        self.acc -= step
        if step.any():
            self.mouse.move(int(step[0]), int(step[1]))

    def _run(self):
        sub = P.make_sub(P.PORT_SENSORS, topics=(P.TOPIC_IMU, P.TOPIC_HAND, P.TOPIC_EMG, TOPIC_POINTER_HAND))
        while not self._stop.is_set():
            if not sub.poll(50):
                continue
            try:
                topic, h, a = P.recv(sub)
            except Exception:
                continue
            now = time.perf_counter()
            if topic == P.TOPIC_EMG:
                self.frozen = self.gate.update(a, now) if self.s["freeze_on_clench"] else False
            elif topic == P.TOPIC_IMU:
                self._imu(a, now)
            elif topic == P.TOPIC_HAND:
                self._hand(a, h, now)
            elif topic == TOPIC_POINTER_HAND:
                self._left(a, h, now)

    def _imu(self, a, now):
        if not np.isfinite(a).all():
            return
        self.imu_t = now
        g = a[:, 3:6].astype(float)
        # gyro bias: learn it whenever the hand is still
        for row in g:
            if np.linalg.norm(row - self.bias) < 3.0:
                self._still += 1
                if self._still > 40:
                    self.bias = 0.995 * self.bias + 0.005 * row
            else:
                self._still = 0
        g = g - self.bias
        if self._rec is not None:
            self._rec.extend(g.tolist())
        mode = self.s["mode"]
        if mode not in ("auto", "imu") or not self.enabled_fn():
            if mode != "left":
                self.source = "-"
            return
        if mode == "auto" and self.s["need_hand"] and now - self.hand_seen_t > 0.4:
            self.source = "paused (hand out of view)"
            self.f_imu.reset()
            return
        if self.frozen:
            self.source = "frozen (clench)"
            return
        self.source = "imu"
        dt = len(g) / P.FS_IMU
        w = g.mean(axis=0)
        v = np.array([self.s["sign_x"] * w[self.s["axis_x"]], self.s["sign_y"] * w[self.s["axis_y"]]])
        dz = self.s["deadzone_dps"]
        v = np.sign(v) * np.maximum(np.abs(v) - dz, 0.0)
        mc = 0.5 + 4.5 * (1 - self.s["smoothing"])          # smoothing 0..1 -> min cutoff 5..0.5 Hz
        self.f_imu.min_cutoff = mc
        v = self.f_imu(v, dt)
        v = self._gain(v * self.s["speed"])                 # deg/s * px/deg = px/s
        if self.s["invert_x"]:
            v[0] = -v[0]
        if self.s["invert_y"]:
            v[1] = -v[1]
        self._move(v * dt)

    def _left(self, a, h, now):
        """Pointer-hand frame (~30 Hz): open hand moves the cursor by its palm-centre motion, fist freezes."""
        if self.s["mode"] != "left":
            return
        if len(a) == 0:
            self.left_prev, self.left_open = None, False
            self.f_left.reset()
            self.source = "no pointer hand in view"
            return
        o = openness(a)
        self.openness = o
        if self.left_open and o < CLOSED_BELOW:
            self.left_open = False
        elif not self.left_open and o > OPEN_ABOVE:
            self.left_open = True
            self.left_prev = None                      # don't jump on re-open
            self.f_left.reset()
        palm = a[PALM, :2].mean(axis=0) * np.array([h.get("img_w") or 640, h.get("img_h") or 480])
        prev, self.left_prev = self.left_prev, palm
        if not self.left_open:
            self.source = f"fist: frozen (openness {o:.2f})"
            self.left_prev = None
            return
        if not self.enabled_fn():
            self.source = "-"
            return
        self.source = f"open hand: moving (openness {o:.2f})"
        if prev is None:
            return
        d = palm - prev
        d[0] = -d[0]                                    # webcam faces you: your right is image left
        mc = 0.5 + 4.5 * (1 - self.s["smoothing"])
        self.f_left.min_cutoff = mc
        d = self.f_left(d, 1 / 30)
        if np.linalg.norm(d) < 0.4:                     # landmark jitter
            return
        v = self._gain(d * 30 * self.s["speed"] / 2.5)  # image px/s -> screen px/s
        if self.s["invert_x"]:
            v[0] = -v[0]
        if self.s["invert_y"]:
            v[1] = -v[1]
        self._move(v / 30)

    def _right(self, a, h, now):
        """Right hand (the `hand` topic): the palm centre moves the cursor while the clutch is engaged."""
        mode = self.s["mode"]
        if len(a) == 0:
            self.r_prev = None
            self.f_right.reset()
            self.pinched, self._pinch_n = False, 0
            self.source = "no hand in view"
            return
        if mode == "right_pinch":
            r = pinch_ratio(a)
            self.pinch = r
            want = r < PINCH_ON if not self.pinched else r < PINCH_OFF
            self._pinch_n = self._pinch_n + 1 if want != self.pinched else 0
            if self._pinch_n >= 2:                      # 2 frames in a row: debounce
                self.pinched, self._pinch_n = want, 0
                self.r_prev = None
                self.f_right.reset()
            engaged, why = self.pinched, f"pinch {r:.2f}"
        else:
            engaged, why = self.clutch_gesture, "gesture held" if self.clutch_gesture else "gesture not held"
        palm = a[PALM, :2].mean(axis=0) * np.array([h.get("img_w") or 640, h.get("img_h") or 480])
        prev, self.r_prev = self.r_prev, palm
        if not engaged:
            self.source = f"frozen ({why})"
            self.r_prev = None
            return
        if not self.enabled_fn():
            self.source = "actions off"
            return
        self.source = f"moving ({why})"
        if prev is None:
            return
        d = palm - prev
        d[0] = -d[0]                                    # webcam faces you: your right is image left
        self.f_right.min_cutoff = 0.5 + 4.5 * (1 - self.s["smoothing"])
        d = self.f_right(d, 1 / 30)
        if np.linalg.norm(d) < 0.4:
            return
        v = self._gain(d * 30 * self.s["speed"] / 2.5)
        if self.s["invert_x"]:
            v[0] = -v[0]
        if self.s["invert_y"]:
            v[1] = -v[1]
        self._move(v / 30)

    def set_clutch(self, on: bool):
        self.clutch_gesture = bool(on)
        if not on:
            self.r_prev = None

    def _hand(self, a, h, now):
        if self.s["mode"] in ("right_pinch", "right_gesture"):
            self._right(a, h, now)
            return
        if len(a) == 0:
            self.last_palm = None
            return
        self.hand_seen_t = now
        palm = a[PALM, :2].mean(axis=0) * np.array([h.get("img_w") or 640, h.get("img_h") or 480])
        mode = self.s["mode"]
        imu_fresh = now - self.imu_t < 0.3
        use_cam = mode == "camera" or (mode == "auto" and not imu_fresh)
        prev, self.last_palm = self.last_palm, palm
        if not use_cam or not self.enabled_fn() or prev is None:
            return
        if self.frozen:
            self.source = "frozen (clench)"
            return
        self.source = "camera"
        d = palm - prev
        d[0] = -d[0]                                        # webcam faces you: your right is image left
        mc = 0.5 + 4.5 * (1 - self.s["smoothing"])
        self.f_cam.min_cutoff = mc
        d = self.f_cam(d, 1 / 30)
        if np.linalg.norm(d) < 0.6:                         # pixel jitter deadzone
            return
        v = self._gain(d * 30 * self.s["cam_speed"] * self.s["speed"] / 6.0)
        if self.s["invert_x"]:
            v[0] = -v[0]
        if self.s["invert_y"]:
            v[1] = -v[1]
        self._move(v / 30)
