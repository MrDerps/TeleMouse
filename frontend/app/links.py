"""Background links between the app and the outside world. The UI thread only reads snapshots.

  * SensorLink   - SUB 5555 (reuses the viewer's ring buffers + display filtering)
  * BackendLink  - SUB 5556 (probs / gesture / status) and a single worker thread for REQ 5557,
                   whose replies come back to the UI thread through a Qt signal.
  * Sources      - what publishes 5555: the in-process Acquisition, the simulator, or nothing
                   (an external publisher).
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

from PyQt6 import QtCore

from shared import protocol as P

from ..tools.viewer import Streams


class SensorLink:
    def __init__(self, port=P.PORT_SENSORS):
        self.s = Streams()
        self.port = port
        self.th = threading.Thread(target=self.s.run, args=(port,), name="sensor-link", daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self.s.stop.set()
        self.th.join(1)

    def snapshot(self):
        s = self.s
        with s.lock:
            return dict(emg=s.emg.a.copy(), env=s.env.a.copy(), imu=s.imu.a.copy(), hand=s.hand,
                        hand_score=s.hand_score, status=dict(s.status or {}), last_msg=s.last_msg,
                        counts=dict(s.counts))


class BackendLink(QtCore.QObject):
    reply = QtCore.pyqtSignal(object, object)        # (callback, reply dict) delivered on the UI thread

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.status = {}
        self.status_t = 0.0
        self.probs = None                            # (t, classes, p)
        self.events = []                             # recent gesture events (UI drains)
        self.on_gesture = None                       # called in the link thread (action executor)
        self._stop = threading.Event()
        self._q = queue.Queue()
        self.reply.connect(lambda cb, r: cb(r) if cb else None)
        self._sub = threading.Thread(target=self._run_sub, name="backend-sub", daemon=True)
        self._req = threading.Thread(target=self._run_req, name="backend-req", daemon=True)

    def start(self):
        self._sub.start()
        self._req.start()

    def stop(self):
        self._stop.set()
        self._q.put(None)
        self._sub.join(1)
        self._req.join(2)

    @property
    def online(self):
        return time.perf_counter() - self.status_t < 3.0

    def control(self, msg: dict, callback=None, timeout_s=2.0):
        """Queue a 5557 request; callback(reply) runs on the UI thread."""
        self._q.put((msg, callback, timeout_s))

    def drain_events(self):
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def _run_sub(self):
        sub = P.make_sub(P.PORT_EVENTS)
        while not self._stop.is_set():
            if not sub.poll(100):
                continue
            try:
                topic, h, _ = P.recv(sub)
            except Exception:
                continue
            if topic == P.TOPIC_GESTURE:
                h = dict(h, rx=time.perf_counter())
                if self.on_gesture:                  # act first (lowest latency), then log
                    try:
                        h["_outcome"] = self.on_gesture(h)
                    except Exception as e:  # an action failure must never kill the link
                        h["_outcome"] = f"error: {e}"
            with self.lock:
                if topic == P.TOPIC_STATUS:
                    self.status, self.status_t = h, time.perf_counter()
                elif topic == P.TOPIC_PROBS:
                    self.probs = (h.get("t"), h.get("classes", []), h.get("p", []))
                elif topic == P.TOPIC_GESTURE:
                    self.events.append(h)
                    self.events = self.events[-200:]
        sub.close(0)

    def _run_req(self):
        cc = P.ControlClient(timeout_s=2.0)
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            msg, cb, tmo = item
            r = cc.request(msg, timeout_s=tmo)
            self.reply.emit(cb, r)
        cc.close()


class Sources:
    """Owns whatever publishes 5555 from inside the app."""

    def __init__(self, log=print):
        self.log = log
        self.mode = None            # None | "hardware" | "sim"
        self.acq = None
        self._sim_stop = None
        self._sim_th = None

    def start_hardware(self, port, camera_index, use_camera=True):
        from ..acquire.service import Acquisition
        self.stop()
        acq = Acquisition(serial_port=port, camera_index=camera_index if use_camera else None,
                          log=self.log, battery_warning_shown=True)
        acq.start()                 # raises RuntimeError if 5555 is busy
        self.acq, self.mode = acq, "hardware"

    def start_sim(self):
        from ..tools.replay import run_sim
        self.stop()
        args = argparse.Namespace(port=P.PORT_SENSORS, seed=None, no_imu=False, no_hand=False,
                                  drop_rate=0.0, truth=None, quiet=True)
        self._sim_stop, ready = threading.Event(), threading.Event()
        self._sim_th = threading.Thread(target=run_sim, args=(args, self._sim_stop, ready), daemon=True)
        self._sim_th.start()
        if not ready.wait(3):
            self._sim_stop = None
            raise RuntimeError(f"simulator could not bind port {P.PORT_SENSORS} (another publisher running?)")
        self.mode = "sim"

    def stop(self):
        if self.acq:
            self.acq.stop()
            self.acq = None
        if self._sim_stop:
            self._sim_stop.set()
            self._sim_th.join(2)
            self._sim_stop = None
        self.mode = None

    @property
    def camera(self):
        return self.acq.camera if self.acq else None
