"""Acquisition service: ESP32 (EMG + IMU) and webcam (hand landmarks) -> port 5555 (INTERFACE §3).

Used headless by `python -m frontend.acquire` and in-process by the desktop app.
Each sensor runs in its own thread; publishing is serialised by a lock (zmq sockets are
not thread-safe). Either sensor may be absent: the stream and status say so.
"""
from __future__ import annotations

import collections
import datetime as _dt
import threading
import time

import numpy as np

from shared import protocol as P

from . import config
from .hands import HandAssigner, imu_motion_from
from .serial_reader import SerialReader

# Frontend-internal topic (INTERFACE §3 note): the non-IMU hand, used only for the cursor. It must not start
# with b"hand", because ZeroMQ subscriptions are prefix matches and the backend subscribes to b"hand".
TOPIC_POINTER_HAND = b"pointer_hand"

SAFETY = ("SAFETY: while electrodes are on a person, the ESP32 must be powered from a battery, "
          "a power bank or an UNPLUGGED laptop. Never from a wall-powered PC.")


class Acquisition:
    def __init__(self, serial_port=None, camera_index=0, pub_port=P.PORT_SENSORS, baud=921600,
                 log=print, battery_warning_shown=False):
        self.cfg = config.load()
        self.serial_port = serial_port
        self.camera_index = camera_index
        self.pub_port = pub_port
        self.baud = baud
        self.log = log
        self.battery_warning_shown = battery_warning_shown
        self.cam_latency = float(self.cfg["cam_latency"])
        self.pub = None
        self.lock = threading.Lock()
        self.reader: SerialReader | None = None
        self.camera = None
        self.camera_error = None
        self._stop = threading.Event()
        self._status_thread = None
        self._win = {"emg": 0, "imu": 0, "cam": 0}
        self.last_status = {}
        # short history for sync calibration: IMU (t, ax, ay, az) and camera (t_raw, lm)
        self._imu_hist = collections.deque(maxlen=200 * 30)   # (t, ax, ay, az, gx, gy, gz)
        self._hand_hist = collections.deque(maxlen=30 * 30)
        self.img_size = (640, 480)
        self.assigner = HandAssigner(imu_hand=self.cfg.get("imu_hand", "Right"))
        self.roles = (None, None)              # (gesture hand, pointer hand) of the latest frame, for previews

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        import zmq
        try:
            self.pub = P.make_pub(self.pub_port)
        except zmq.ZMQError as e:
            raise RuntimeError(f"port {self.pub_port} is already in use ({e}). Is the simulator "
                               f"(frontend.tools.replay) or another acquire still running?") from e
        if self.serial_port:
            self.reader = SerialReader(self.serial_port, self.baud, on_block=self._on_block, on_log=self.log)
            self.reader.start()
            self.log(f"[acquire] serial {self.serial_port}")
        if self.camera_index is not None:
            try:
                from .camera import Camera
                self.camera = Camera(self.camera_index, on_hands=self._on_hands, log=self.log)
                self.camera.start()
                self.log(f"[acquire] camera {self.camera_index}, latency correction {1000 * self.cam_latency:.0f} ms"
                         + ("" if self.cfg.get("cam_latency_calibrated") else " (default; run sync calibration)"))
            except Exception as e:
                self.camera, self.camera_error = None, str(e)
                self.log(f"[acquire] camera unavailable: {e}")
        self._status_thread = threading.Thread(target=self._status_loop, name="status", daemon=True)
        self._status_thread.start()
        self.log(f"[acquire] publishing on {P.addr(self.pub_port)}")

    def stop(self):
        self._stop.set()
        if self._status_thread:
            self._status_thread.join(2)
        if self.reader:
            self.reader.stop()
        if self.camera:
            self.camera.stop()
        if self.pub:
            with self.lock:
                self.pub.close(0)
            self.pub = None
        self.log("[acquire] stopped")

    # ------------------------------------------------------------------ sensor callbacks
    def _on_block(self, t0, i0, emg, imu_t0, k0, imu):
        with self.lock:
            if self.pub is None:
                return
            P.send_array(self.pub, P.TOPIC_EMG, {"t0": t0, "fs": P.FS_EMG, "index0": i0}, emg)
            self._win["emg"] += len(emg)
            if imu is not None:
                P.send_array(self.pub, P.TOPIC_IMU, {"t0": imu_t0, "fs": P.FS_IMU, "index0": k0}, imu)
                self._win["imu"] += int(np.isfinite(imu[:, 0]).sum())
        if imu is not None:
            for j in range(len(imu)):
                self._imu_hist.append((imu_t0 + j / P.FS_IMU, *imu[j, :6]))

    def _imu_motion(self, t0, t1):
        rows = [r for r in list(self._imu_hist)[-400:] if t0 <= r[0] <= t1 and np.isfinite(r[1])]
        return imu_motion_from(rows)

    def _on_hands(self, t_raw, hands, w, h):
        """Camera callback: split the hands into the gesture (IMU) hand -> `hand` and the other -> `pointer_hand`."""
        self.img_size = (w, h)
        imu_ok = bool(self.last_status.get("imu_ok"))
        g, ptr = self.assigner.assign(t_raw, hands, w, h, self._imu_motion, self.cam_latency, imu_ok)
        self.roles = (g, ptr)
        t = t_raw - self.cam_latency
        empty = np.zeros((0, 3), np.float32)
        with self.lock:
            if self.pub is None:
                return
            P.send_array(self.pub, P.TOPIC_HAND,
                         {"t": t, "score": g["score"] if g else 0.0, "handedness": g["handedness"] if g else None,
                          "img_w": w, "img_h": h}, g["lm"] if g else empty)
            P.send_array(self.pub, TOPIC_POINTER_HAND,
                         {"t": t, "score": ptr["score"] if ptr else 0.0, "img_w": w, "img_h": h},
                         ptr["lm"] if ptr else empty)
            self._win["cam"] += 1
        self._hand_hist.append((t_raw, g["lm"] if g else None))

    # ------------------------------------------------------------------ status
    def _status_loop(self):
        with self.lock:                       # don't count what arrived while the camera model loaded
            self._win = {"emg": 0, "imu": 0, "cam": 0}
        nxt = time.perf_counter() + 1.0
        while not self._stop.wait(max(0.0, nxt - time.perf_counter())):
            nxt += 1.0
            now = time.perf_counter()
            rd, cam = self.reader, self.camera
            st = rd.status if rd else None
            emg_ok = bool(rd and now - rd.last_rx < 0.5)
            with self.lock:
                win, self._win = self._win, {"emg": 0, "imu": 0, "cam": 0}
            hdr = {
                "emg_ok": emg_ok,
                "imu_ok": bool(emg_ok and st is not None and st.imu_state == 0 and win["imu"] > 0),
                "cam_ok": bool(cam and now - cam.last_frame_t < 1.0),
                "emg_rate": float(win["emg"]), "imu_rate": float(win["imu"]), "cam_fps": float(win["cam"]),
                "dropped": rd.dropped if rd else 0,
                "battery_warning_shown": self.battery_warning_shown,
                "imu_state": st.imu_state_name if st else None,
                "crc_errors": rd.parser.crc_errors if rd else 0,
                "cam_latency": self.cam_latency,
            }
            self.last_status = hdr
            with self.lock:
                if self.pub is None:
                    return
                P.send_json(self.pub, P.TOPIC_STATUS, hdr)

    # ------------------------------------------------------------------ sync calibration
    def calibrate_sync(self, seconds=8.0, progress=None, save=True) -> dict:
        """Blocks for `seconds` while the user taps the hand on the desk ~5 times, then estimates
        the camera latency from IMU vs hand motion. Applies (and saves) it if the estimate is good."""
        from .sync import estimate_latency
        if not self.reader or not self.camera:
            return {"ok": False, "message": "sync calibration needs both the ESP32 (IMU) and the camera"}
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < seconds:
            if progress:
                progress(time.perf_counter() - t_start, seconds)
            time.sleep(0.1)
        time.sleep(0.1)                       # let the last IMU block arrive
        imu = np.array([r for r in list(self._imu_hist) if r[0] >= t_start - 0.5], np.float64)
        hand = [(t, lm) for t, lm in list(self._hand_hist) if t >= t_start]
        if len(imu) < 100 or len(hand) < 30:
            return {"ok": False, "message": f"too little data (imu {len(imu)}, frames {len(hand)})"}
        ht = np.array([t for t, _ in hand])
        lm = np.stack([x if x is not None else np.full((21, 3), np.nan, np.float32) for _, x in hand])
        good = np.isfinite(imu[:, 1])
        r = estimate_latency(imu[good, 0], imu[good, 1:4], ht, lm, *self.img_size)
        r.pop("curve", None)
        if r["ok"]:
            self.cam_latency = float(r["latency"])
            if save:
                self.cfg = config.save(cam_latency=self.cam_latency,
                                       cam_latency_calibrated=_dt.datetime.now().isoformat(timespec="seconds"))
        return r
