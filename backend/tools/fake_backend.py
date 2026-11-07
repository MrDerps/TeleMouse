"""Fake backend (INTERFACE.md section 7): lets the frontend build its UI before the real backend exists.

- REP on 5557: answers every section-5 command (train fakes ~8 s of progress, then the model is listed/loadable).
- PUB on 5556: `status` at 1 Hz and on change; while running, `probs` at 50 Hz with occasional fake gestures,
  plus matching `gesture` events (instant -> one event, hold -> start ... end).
- Optional --sub: subscribe to 5555 and stamp outputs with the newest sensor time (otherwise perf_counter).

    python -m backend.tools.fake_backend
    python -m backend.tools.fake_backend --sub --gesture-every 3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import zmq

from backend.common import zmq_io as zio
from backend.common.config import load_config, repo_path


def _gestures_from_file(cfg) -> list[dict]:
    p = repo_path(cfg["paths"]["gestures"])
    try:
        return json.loads(p.read_text(encoding="utf-8"))["gestures"]
    except Exception:
        return [{"id": "swipe_left", "kind": "instant"}, {"id": "pinch_hold", "kind": "hold"}]


class FakeBackend:
    def __init__(self, cfg, sub: bool, gesture_every: float, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(zio.addr(zio.PORT_OUTPUTS))
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(zio.addr(zio.PORT_CONTROL))
        self.sub = None
        if sub:
            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(zio.addr(zio.PORT_SENSORS))
            for t in (b"emg", b"imu", b"hand"):
                self.sub.setsockopt(zmq.SUBSCRIBE, t)
        self.poller = zmq.Poller()
        self.poller.register(self.rep, zmq.POLLIN)
        if self.sub is not None:
            self.poller.register(self.sub, zmq.POLLIN)
        self.state, self.model, self.progress, self.message = "idle", None, 0.0, "fake backend ready"
        self.models: dict[str, dict] = {}
        self.classes: list[str] = ["none"]
        self.kinds: dict[str, str] = {}
        self.train_job = None  # (name, classes, kinds, t_start)
        self.stream_t = None
        self.gesture_every = gesture_every
        self.next_gesture = time.perf_counter() + gesture_every
        self.active = None  # (gid, t_end, kind)
        self.logits = None

    def now(self) -> float:
        return self.stream_t if self.stream_t is not None else time.perf_counter()

    def publish_status(self):
        zio.send(self.pub, b"status", {"state": self.state, "model": self.model,
                                       "progress": round(self.progress, 3), "message": self.message})

    def set_state(self, state, message=""):
        self.state, self.message = state, message
        self.publish_status()

    # ---- control ----
    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            return {"ok": True, "state": self.state, "model": self.model, "classes": self.classes}
        if cmd == "train":
            if self.state == "training":
                return {"ok": False, "error": "already training"}
            all_g = {g["id"]: g for g in _gestures_from_file(self.cfg)}
            ids = req.get("gestures") or list(all_g)
            kinds = {g: all_g.get(g, {}).get("kind", "instant") for g in ids}
            name = req.get("name") or f"model_{datetime.now():%H%M%S}"
            self.train_job = (name, ["none"] + list(ids), kinds, time.perf_counter())
            self.progress = 0.0
            self.set_state("training", f"training {name} (fake)")
            return {"ok": True}
        if cmd == "load":
            m = self.models.get(req.get("name"))
            if m is None:
                return {"ok": False, "error": f"unknown model {req.get('name')!r}"}
            self.model, self.classes, self.kinds = m["name"], m["classes"], m["kinds"]
            self.logits = np.zeros(len(self.classes))
            self.set_state("idle", f"loaded {self.model}")
            return {"ok": True, "classes": self.classes}
        if cmd == "start":
            if self.model is None:
                return {"ok": False, "error": "no model loaded"}
            self.set_state("running", "live inference (fake)")
            return {"ok": True}
        if cmd == "stop":
            self.active = None
            self.set_state("idle", "stopped")
            return {"ok": True}
        if cmd == "list_models":
            return {"ok": True, "models": [{k: m[k] for k in ("name", "classes", "created", "val_metrics")}
                                           for m in self.models.values()]}
        if cmd == "calibrate_rest":
            self.message = f"rest calibrated over {req.get('seconds', 3)} s (fake)"
            self.publish_status()
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    # ---- simulation ----
    def tick_training(self):
        name, classes, kinds, t0 = self.train_job
        self.progress = min(1.0, (time.perf_counter() - t0) / 8.0)
        if self.progress >= 1.0:
            self.models[name] = {"name": name, "classes": classes, "kinds": kinds,
                                 "created": datetime.now().isoformat(timespec="seconds"),
                                 "val_metrics": {"fp_per_min": 0.4, "recall": 0.93, "precision": 0.95,
                                                 "latency_ms": 85}}
            self.train_job = None
            self.model, self.classes, self.kinds = name, classes, kinds
            self.logits = np.zeros(len(classes))
            self.set_state("idle", f"trained {name} (fake)")

    def tick_running(self):
        t = self.now()
        n = len(self.classes)
        target = np.zeros(n)
        target[0] = 4.0
        wall = time.perf_counter()
        if self.active is None and n > 1 and wall >= self.next_gesture:
            gi = int(self.rng.integers(1, n))
            gid = self.classes[gi]
            kind = self.kinds.get(gid, "instant")
            dur = 1.5 if kind == "hold" else 0.3
            self.active = [gid, wall + dur, kind, False]
            self.next_gesture = wall + dur + self.rng.exponential(self.gesture_every)
        if self.active is not None:
            gid, t_end, kind, sent = self.active
            target[self.classes.index(gid)] = 8.0
            if not sent:
                zio.send(self.pub, b"gesture", {"t": t, "id": gid, "conf": 0.9,
                                                "kind": "start" if kind == "hold" else "instant"})
                self.active[3] = True
            if wall >= t_end:
                if kind == "hold":
                    zio.send(self.pub, b"gesture", {"t": t, "id": gid, "conf": 0.9, "kind": "end"})
                self.active = None
        self.logits += 0.3 * (target - self.logits) + self.rng.normal(0, 0.15, n)
        p = np.exp(self.logits - self.logits.max())
        p /= p.sum()
        zio.send(self.pub, b"probs", {"t": t, "classes": self.classes, "p": [round(float(x), 4) for x in p]})

    def run(self):
        print(f"fake backend: PUB {zio.PORT_OUTPUTS}, REP {zio.PORT_CONTROL}" + (", SUB 5555" if self.sub else ""))
        next_frame = next_status = time.perf_counter()
        while True:
            timeout = max(0.0, next_frame - time.perf_counter())
            for sock, _ in self.poller.poll(timeout * 1000):
                if sock is self.rep:
                    try:
                        req = json.loads(self.rep.recv().decode())
                        rep = self.handle(req)
                    except Exception as e:  # never die on a bad request
                        rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    self.rep.send(json.dumps(rep).encode())
                elif sock is self.sub:
                    while True:
                        try:
                            topic, h, _ = zio.recv(self.sub, zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        t = h.get("t", h.get("t0"))
                        if t is not None:
                            self.stream_t = max(self.stream_t or t, t)
            now = time.perf_counter()
            if now >= next_frame:
                next_frame += 0.02
                if next_frame < now:
                    next_frame = now + 0.02
                if self.train_job is not None:
                    self.tick_training()
                elif self.state == "running":
                    self.tick_running()
            if now >= next_status:
                next_status = now + 1.0
                self.publish_status()


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", action="store_true", help="use sensor times from port 5555")
    ap.add_argument("--gesture-every", type=float, default=4.0, help="mean seconds between fake gestures")
    a = ap.parse_args(argv)
    try:
        FakeBackend(cfg, a.sub, a.gesture_every, cfg["seed"]).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
