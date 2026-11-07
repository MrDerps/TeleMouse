"""Tiny smoke-test client for the backend's 5556/5557 ports (fake or real).

    python -m backend.tools.fake_client            # train -> load -> start, print events for ~6 s, stop
"""
from __future__ import annotations

import argparse
import json
import time

import zmq

from backend.common import zmq_io as zio


def request(req_sock, msg: dict, timeout_ms: int = 3000) -> dict:
    req_sock.send(json.dumps(msg).encode())
    if not req_sock.poll(timeout_ms):
        raise TimeoutError(f"no reply to {msg}")
    return json.loads(req_sock.recv().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--gestures", nargs="*", default=["swipe_left", "pinch_hold"])
    a = ap.parse_args(argv)
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(zio.addr(zio.PORT_CONTROL))
    sub = ctx.socket(zmq.SUB)
    sub.connect(zio.addr(zio.PORT_OUTPUTS))
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    print("status:", request(req, {"cmd": "status"}))
    print("train:", request(req, {"cmd": "train", "gestures": a.gestures, "sessions": "all", "name": "smoke"}))
    while request(req, {"cmd": "status"})["state"] == "training":
        time.sleep(0.5)
    print("list_models:", request(req, {"cmd": "list_models"}))
    print("load:", request(req, {"cmd": "load", "name": "smoke"}))
    print("start:", request(req, {"cmd": "start"}))
    n_probs, t_end = 0, time.perf_counter() + a.seconds
    while time.perf_counter() < t_end:
        if sub.poll(100):
            topic, h, _ = zio.recv(sub)
            if topic == b"probs":
                n_probs += 1
            else:
                print(topic.decode(), h)
    print(f"probs messages: {n_probs} in {a.seconds:.0f} s ({n_probs / a.seconds:.1f} Hz)")
    print("stop:", request(req, {"cmd": "stop"}))
    print("bad cmd:", request(req, {"cmd": "nope"}))


if __name__ == "__main__":
    main()
