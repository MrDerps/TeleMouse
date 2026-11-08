"""Live EMG channel-swap check. Watches the first confident gestures of a session and flips the acquisition's
channel order if the EMG clearly matches your recordings only when swapped. See frontend/acquire/emg_swap.py."""
from __future__ import annotations

import collections
import threading
import time

import numpy as np
from PyQt6 import QtCore

from shared import protocol as P

from ..acquire import config
from ..acquire.emg_swap import SOS, SwapDetector, build_profiles, burst_ratio
from . import data

CONF_MIN = 0.90
ENV_N = 50                         # 50 ms RMS at 1 kHz


class _StreamEnv:
    """Streaming version of emg_swap.envelope(): same filter, same 50 ms RMS."""

    def __init__(self):
        self.zi = None
        self.tail = np.zeros((ENV_N - 1, 2))

    def __call__(self, x):
        from scipy.signal import sosfilt, sosfilt_zi
        x = x.astype(np.float64)
        if self.zi is None:
            self.zi = sosfilt_zi(SOS)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(SOS, x, axis=0, zi=self.zi)
        full = np.vstack([self.tail, y ** 2])
        c = np.vstack([np.zeros((1, 2)), np.cumsum(full, axis=0)])
        env = np.sqrt(np.maximum(c[ENV_N:] - c[:-ENV_N], 0) / ENV_N)
        self.tail = full[-(ENV_N - 1):]
        return env


class SwapWatch(QtCore.QObject):
    decided = QtCore.pyqtSignal(str)          # "ok" | "swap" | "mismatch"

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.detector = SwapDetector({})
        self.lock = threading.Lock()
        self.env = _StreamEnv()
        self.t = collections.deque(maxlen=12000)          # last ~12 s of envelope (stream time)
        self.e = collections.deque(maxlen=12000)
        self.rest_hist = collections.deque(maxlen=3000)   # 30 s of 10 ms envelope samples for the resting level
        self.pending = []                                 # (due_wallclock, gesture event)
        self.status = "waiting for gestures"
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, name="swap-watch", daemon=True)
        self.rebuild()

    def start(self):
        self.th.start()

    def stop(self):
        self._stop.set()
        self.th.join(1)

    @property
    def enabled(self):
        return bool(config.load().get("emg_swap_auto", True))

    def rebuild(self):
        """Recompute the gesture profiles from the recordings (in the background)."""
        def work():
            from .teach import trainable
            ids, _ = trainable()
            prof = build_profiles([data.SESSIONS / i for i in ids]) if ids else {}
            with self.lock:
                self.detector = SwapDetector(prof)
                self._update_status()
        threading.Thread(target=work, daemon=True).start()

    def reset(self):
        """New session (connect, calibrate, manual change): check again."""
        with self.lock:
            self.detector.reset()
            self.t.clear()
            self.e.clear()
            self.rest_hist.clear()
            self.env = _StreamEnv()
            self._update_status()

    def _update_status(self):
        d = self.detector
        if not d.informative:
            self.status = "no gesture clearly favours one channel yet (teach one with a strong EMG pattern)"
        elif d.decision == "ok":
            self.status = "channels look correct"
        elif d.decision == "swap":
            self.status = "channels were swapped: fixed automatically"
        elif d.decision == "mismatch":
            self.status = "EMG doesn't match your recordings (pads moved?): recalibrate or re-teach"
        else:
            self.status = f"checking: {len(d.votes)} of {d.need}+ gestures so far ({', '.join(d.informative)})"

    def _run(self):
        sub_s = P.make_sub(P.PORT_SENSORS, topics=(P.TOPIC_EMG,))
        sub_e = P.make_sub(P.PORT_EVENTS, topics=(P.TOPIC_GESTURE,))
        poller = __import__("zmq").Poller()
        poller.register(sub_s, 1)
        poller.register(sub_e, 1)
        while not self._stop.is_set():
            ev = dict(poller.poll(50))
            if sub_s in ev:
                topic, h, a = P.recv(sub_s)
                env = self.env(a)
                t = h["t0"] + np.arange(len(a)) / P.FS_EMG
                with self.lock:
                    self.t.extend(t.tolist())
                    self.e.extend(env.tolist())
                    self.rest_hist.extend(env[::10].tolist())
            if sub_e in ev:
                topic, h, _ = P.recv(sub_e)
                if h.get("kind") in ("instant", "start") and float(h.get("conf", 0)) >= CONF_MIN:
                    self.pending.append((time.perf_counter() + 0.45, h))
            now = time.perf_counter()
            due = [p for p in self.pending if p[0] <= now]
            self.pending = [p for p in self.pending if p[0] > now]
            for _, h in due:
                self._judge(h)
        sub_s.close(0)
        sub_e.close(0)

    def _judge(self, h):
        if not self.enabled:
            return
        with self.lock:
            if self.detector.decision is not None or len(self.t) < 2000 or len(self.rest_hist) < 300:
                return
            t = np.fromiter(self.t, float)
            e = np.array(self.e)
            rest = np.percentile(np.array(self.rest_hist), 20, axis=0) + 1e-6
            r = burst_ratio(e / rest, t, h["t"] - 1.2, h["t"] + 0.3)   # covers the onset even if the event is late
            if r is None or r[1] < 2.0:
                return
            out = self.detector.feed(h.get("id"), r[0])
            self._update_status()
        if out:
            self.decided.emit(out)
