"""Live backend server (INTERFACE.md sections 4-5).

  SUB 5555 (emg/imu/hand/status) -> LiveEngine -> PUB 5556 probs (50 Hz) / gesture / status (1 Hz + on change)
  REP 5557 control: status, train (background process), load, start, stop, list_models, calibrate_rest

Training runs in a separate process, so a crash there can't take the server down. Nothing raised
while handling a request or a message escapes the main loop.

    python -m backend.serve                     # sessions from data/sessions
    python -m backend.serve --synth             # also allow training on backend/_synth sessions
    python -m backend.serve --load m1 --start   # load a model and start inference immediately
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque

import numpy as np
import torch
import zmq

from backend.common import zmq_io as zio
from backend.common.config import load_config, repo_path
from backend.data.features import MODALITIES, EmgFilter, robust_std, sensor_key
from backend.model.io import list_models, load_model
from backend.serve.engine import LiveEngine

log = logging.getLogger("serve")


def _train_worker(paths, gestures, name, cfg, q, sensors=None, sensor_weights=None, val_sessions=None,
                  warm_start=None):
    """Runs in a child process."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s train-worker: %(message)s")
    try:
        from pathlib import Path

        from backend.train import train
        _, meta = train([Path(p) for p in paths], gestures, name, cfg, warm_start=warm_start,
                        progress=lambda f, m: q.put(("progress", f, m)), sensors=sensors,
                        sensor_weights=sensor_weights, val_sessions=val_sessions)
        q.put(("done", name, meta["val_metrics"]))
    except BaseException as e:  # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))


def calib_std(y: np.ndarray, cfg) -> np.ndarray:
    """Robust rest noise from live EMG: robust std per 250 ms block, mean of the quieter half of the blocks,
    so a stray movement during calibration can't inflate it."""
    floor = cfg["emg"]["rest_std_floor"]
    n = cfg["emg"]["fs"] // 4
    if len(y) < 2 * n:
        return robust_std(y, floor)
    blocks = np.stack([robust_std(y[i:i + n], floor) for i in range(0, len(y) - n + 1, n)])
    k = max(1, len(blocks) // 2)
    return np.sort(blocks, 0)[:k].mean(0)


class EmgMonitor:
    """Always-on EMG filter for calibrate_rest and electrode-off detection (independent of the model)."""

    def __init__(self, cfg, keep_s=12.0):
        self.cfg = cfg
        self.filt = EmgFilter(cfg)
        self.buf = deque(maxlen=int(keep_s * cfg["emg"]["fs"] / 20) + 10)
        self.rail = deque(maxlen=50)  # rail fraction per chunk, ~1 s
        self.last_t = None

    def push(self, h, arr):
        y = self.filt(arr)
        self.buf.append(y)
        self.rail.append(((arr <= 8) | (arr >= 4087)).mean(0))
        self.last_t = float(h["t0"]) + (len(arr) - 1) / h.get("fs", 1000)

    def recent(self, seconds):
        y = np.concatenate(list(self.buf)) if self.buf else np.zeros((0, 2))
        return y[-int(seconds * self.cfg["emg"]["fs"]):]

    def railed_channels(self, frac=0.2):
        if len(self.rail) < 10:
            return []
        r = np.mean(self.rail, 0)
        return [i + 1 for i in range(2) if r[i] > frac]


class Server:
    def __init__(self, cfg, roots, ports=(zio.PORT_SENSORS, zio.PORT_OUTPUTS, zio.PORT_CONTROL)):
        self.cfg, self.roots, self.ports = cfg, roots, ports
        ctx = zmq.Context.instance()
        self.sub = zio.P.make_sub(ports[0], (b"emg", b"imu", b"hand", b"status"), ctx)
        self.pub = zio.P.make_pub(ports[1], ctx)
        self.rep = ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.rep.bind(zio.addr(ports[2]))
        self.stop_flag = False
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        self.poller.register(self.rep, zmq.POLLIN)
        self.monitor = EmgMonitor(cfg)
        self.model_name, self.meta, self.engine = None, None, None
        self._model = None
        self.running = False
        self.rest_std = None          # from calibrate_rest
        self.calib = None             # (t_end_stream, seconds)
        self.train_proc, self.train_q = None, None
        self.progress, self.message, self.error = 0.0, "ready", None
        self.lags = deque(maxlen=250)
        self.hand_lags = deque(maxlen=90)
        self.sensors = None           # sensors requested in `start` (None = the model's own)
        self.sensor_status = {}
        self._last_state = None
        torch.set_num_threads(1)

    # ------------------------------------------------------------------ status
    @property
    def state(self):
        if self.train_proc is not None:
            return "training"
        if self.error:
            return "error"
        return "running" if self.running else "idle"

    def publish_status(self, force=False):
        st = {"state": self.state, "model": self.model_name, "progress": round(self.progress, 3),
              "message": self.message}
        if self.lags:
            st["lag_ms"] = round(float(np.median(self.lags)) * 1000, 1)  # extra: frame age when published
        if self.hand_lags:
            hl = float(np.median(self.hand_lags))
            st["hand_lag_ms"] = round(hl * 1000, 1)  # extra: how late camera frames arrive
            look = self.cfg["hand"]["lag_s"] + self.cfg["frame"]["max_wait_s"]
            if hl > look:
                st["hand_warning"] = (f"camera frames arrive {hl * 1000:.0f} ms late (> {look * 1000:.0f} ms budget): "
                                      "hand data is mostly too old to use; raise hand.lag_s and retrain")
        if self.engine is not None:
            st["sensors"] = self.engine.sensors
            if self.engine.accuracy:
                st["accuracy"] = self.engine.accuracy  # extra: validation accuracy of this model+sensors
        rail = self.monitor.railed_channels()
        if rail:
            st["emg_warning"] = f"EMG channel(s) {rail} look disconnected (rail-to-rail); check electrodes"
        zio.send(self.pub, b"status", st)
        self._last_state = self.state

    def set_message(self, msg, level=logging.INFO):
        self.message = msg
        log.log(level, msg)
        self.publish_status()

    # ------------------------------------------------------------------ control
    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            return {"ok": True, "state": self.state, "model": self.model_name,
                    "classes": self.meta["classes"] if self.meta else []}
        if cmd == "train":
            return self.cmd_train(req)
        if cmd == "load":
            return self.cmd_load(req.get("name"))
        if cmd == "start":
            if self.meta is None:
                return {"ok": False, "error": "no model loaded"}
            sensors = self._sensors_arg(req)
            if isinstance(sensors, dict):
                return sensors
            self.sensors = sensors
            self.engine = LiveEngine(self.cfg, self._model, self.meta, self._norm, self.current_rest_std(), sensors)
            self.running, self.error = True, None
            acc = self.engine.accuracy
            extra = (f" (val recall {acc['recall']:.2f}, precision {acc['precision']:.2f})"
                     if acc.get("recall") is not None else "")
            self.set_message(f"running {self.model_name} with {sensor_key(self.engine.sensors)}{extra}")
            return {"ok": True, "sensors": self.engine.sensors, "accuracy": acc}
        if cmd == "stop":
            self.running = False
            self.set_message("stopped")
            return {"ok": True}
        if cmd == "list_models":
            return {"ok": True, "models": [{k: m.get(k) for k in ("name", "classes", "created", "val_metrics",
                                                                   "sensors", "by_sensors")}
                                           for m in list_models(self.cfg) if m.get("type") == "tcn"]}
        if cmd == "delete_model":
            name = req.get("name")
            if not name or name == self.model_name:
                return {"ok": False, "error": "can't delete the loaded model" if name else "name missing"}
            from backend.common.config import repo_path
            import shutil
            root = repo_path(self.cfg["paths"]["models"])
            p = root / str(name)
            if p.parent != root or not p.is_dir():
                return {"ok": False, "error": f"no model {name!r}"}
            shutil.rmtree(p)
            return {"ok": True}
        if cmd == "calibrate_rest":
            s = float(req.get("seconds", 3))
            if not 0.5 <= s <= 10:
                return {"ok": False, "error": "seconds must be in [0.5, 10]"}
            if self.monitor.last_t is None:
                return {"ok": False, "error": "no EMG stream on 5555"}
            self.calib = (self.monitor.last_t + s, s)
            self.set_message(f"calibrating rest for {s:g} s: relax your hand")
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def cmd_train(self, req):
        if self.train_proc is not None:
            return {"ok": False, "error": "already training"}
        from backend.data.session import resolve_sessions
        spec = req.get("sessions", "all")
        paths = resolve_sessions(spec, self.roots)
        if not paths:
            return {"ok": False, "error": f"no sessions found in {[str(r) for r in self.roots]}"}
        name = req.get("name") or time.strftime("model_%Y%m%d_%H%M%S")
        gestures = req.get("gestures") or None
        sensors = self._sensors_arg(req)
        if isinstance(sensors, dict):
            return sensors
        sw = req.get("sensor_weights")
        if sw is not None:
            ok = isinstance(sw, dict) and all(k in MODALITIES and isinstance(v, (int, float)) and 0 <= v <= 1
                                              for k, v in sw.items())
            if not ok:
                return {"ok": False, "error": f"sensor_weights must map {list(MODALITIES)} to numbers in 0..1"}
        vs = req.get("val_sessions")
        if vs is not None:
            known = {p.name for p in paths}
            if not isinstance(vs, list) or not vs or any(v not in known for v in vs):
                return {"ok": False, "error": "val_sessions must be a non-empty list of the training session ids"}
            if set(vs) >= known:
                return {"ok": False, "error": "val_sessions must leave at least one session for training"}
        ws = req.get("warm_start")
        if ws is not None and ws not in {m.get("name") for m in list_models(self.cfg)}:
            log.info("warm_start %r not found: training from scratch", ws)
            ws = None
        self.train_q = mp.get_context("spawn").Queue()
        self.train_proc = mp.get_context("spawn").Process(
            target=_train_worker, args=([str(p) for p in paths], gestures, name, self.cfg, self.train_q, sensors,
                                        sw, vs, ws),
            daemon=True)
        self.train_proc.start()
        self.progress, self.error = 0.0, None
        self.set_message(f"training {name} on {len(paths)} session(s)" + (f", warm start from {ws}" if ws else ""))
        return {"ok": True}

    @staticmethod
    def _sensors_arg(req):
        """Optional "sensors": ["emg","imu","hand"] subset. Returns list, None, or an error reply dict."""
        s = req.get("sensors")
        if s is None:
            return None
        if not isinstance(s, list) or not s or any(m not in MODALITIES for m in s):
            return {"ok": False, "error": f"sensors must be a non-empty subset of {list(MODALITIES)}"}
        return [m for m in MODALITIES if m in s]

    def cmd_load(self, name):
        if not name:
            return {"ok": False, "error": "missing name"}
        model, meta, norm = load_model(name, self.cfg)
        self._model, self.meta, self._norm, self.model_name = model, meta, norm, meta["name"]
        if self.running:  # hot swap
            self.engine = LiveEngine(self.cfg, model, meta, norm, self.current_rest_std(), self.sensors)
        self.error = None
        self.set_message(f"loaded {self.model_name} ({len(meta['classes']) - 1} gestures)")
        return {"ok": True, "classes": meta["classes"]}

    def current_rest_std(self):
        if self.rest_std is not None:
            return self.rest_std
        rs = list((self.meta or {}).get("rest_std", {}).values())
        if rs:
            log.warning("no live rest calibration yet: using the training sessions' median rest std")
            return np.median(np.array(rs), 0)
        return None

    # ------------------------------------------------------------------ background work
    def poll_training(self):
        if self.train_proc is None:
            return
        try:
            while True:
                msg = self.train_q.get_nowait()
                if msg[0] == "progress":
                    self.progress, self.message = msg[1], msg[2]
                    self.publish_status()
                elif msg[0] == "done":
                    self.train_proc.join(timeout=5)
                    self.train_proc = None
                    vm = msg[2]
                    self.progress = 1.0
                    self.set_message(f"trained {msg[1]}: recall {vm['recall']:.2f}, precision {vm['precision']:.2f}, "
                                     f"FP/min {vm['fp_per_min']:.2f}")
                    if not self.running:
                        try:
                            self.cmd_load(msg[1])
                        except Exception as e:  # noqa: BLE001
                            log.warning("auto-load failed: %s", e)
                    return
                elif msg[0] == "error":
                    log.error("training failed:\n%s", msg[2])
                    self.train_proc.join(timeout=5)
                    self.train_proc = None
                    self.error = msg[1]
                    self.set_message(f"training failed: {msg[1]}", logging.ERROR)
                    return
        except queue.Empty:
            pass
        if self.train_proc is not None and not self.train_proc.is_alive():
            code = self.train_proc.exitcode
            self.train_proc = None
            self.error = f"training process died (exit code {code})"
            self.set_message(self.error, logging.ERROR)

    def poll_calibration(self):
        if self.calib and self.monitor.last_t is not None and self.monitor.last_t >= self.calib[0]:
            s = self.calib[1]
            self.calib = None
            self.rest_std = calib_std(self.monitor.recent(s), self.cfg)
            if self.engine:
                self.engine.set_rest_std(self.rest_std)
            self.set_message(f"rest calibrated: noise std {self.rest_std.round(1).tolist()} counts")

    def on_sensor(self, topic, h, arr):
        if topic == b"emg" and arr is not None and len(arr):
            self.monitor.push(h, arr)
        elif topic == b"status":
            self.sensor_status = h
        elif topic == b"hand" and "t" in h:
            self.hand_lags.append(time.perf_counter() - float(h["t"]))
        if self.running and self.engine is not None:
            self.engine.push(topic, h, arr)

    def run_inference(self):
        frames, events = self.engine.step()
        now = time.perf_counter()
        for t, p in frames:
            zio.send(self.pub, b"probs", {"t": t, "classes": self.engine.classes,
                                          "p": [round(float(x), 4) for x in p]})
            self.lags.append(now - t)
        for e in events:
            zio.send(self.pub, b"gesture", e)
            log.info("gesture %s %s conf %.2f (t=%.3f, age %.0f ms)", e["id"], e["kind"], e["conf"], e["t"],
                     (now - e["t"]) * 1000)

    # ------------------------------------------------------------------ main loop
    def run(self):
        log.info("backend server: SUB %d, PUB %d, REP %d", *self.ports)
        next_status = time.perf_counter()
        while not self.stop_flag:
            try:
                events = dict(self.poller.poll(5))
                if self.sub in events:
                    for _ in range(2000):
                        try:
                            topic, h, arr = zio.recv(self.sub, zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self.on_sensor(topic, h, arr)
                if self.rep in events:
                    try:
                        req = json.loads(self.rep.recv())
                        try:
                            rep = self.handle(req)
                        except Exception as e:  # noqa: BLE001
                            log.exception("request %s failed", req)
                            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    except Exception as e:  # noqa: BLE001 - bad JSON
                        rep = {"ok": False, "error": f"bad request: {e}"}
                    zio.P.reply_json(self.rep, rep)
                if self.running and self.engine is not None:
                    self.run_inference()
                self.poll_training()
                self.poll_calibration()
                now = time.perf_counter()
                if now >= next_status or self.state != self._last_state:
                    next_status = now + 1.0
                    self.publish_status()
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001 - never die
                log.exception("main loop error (continuing)")
                time.sleep(0.05)
        for sock in (self.sub, self.pub, self.rep):
            sock.close(0)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true", help="also train on backend/_synth sessions")
    ap.add_argument("--load", default=None, help="model to load at startup")
    ap.add_argument("--start", action="store_true", help="start inference at startup (needs --load)")
    a = ap.parse_args(argv)
    roots = [repo_path(cfg["paths"]["sessions"])] + ([repo_path(cfg["paths"]["synth"])] if a.synth else [])
    srv = Server(cfg, roots)
    if a.load:
        print(srv.cmd_load(a.load))
        if a.start:
            print(srv.handle({"cmd": "start"}))
    try:
        srv.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
