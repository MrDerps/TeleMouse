"""Wire helpers for docs/INTERFACE.md (v0.2) sections 0-5.

Both the frontend and the backend import this module. Changes follow the same
process as changes to INTERFACE.md (propose in your handoff file first).

Only depends on numpy and pyzmq.

    python -m shared.protocol --selftest
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import zmq

PROTOCOL_VERSION = "0.2"

# ---- §0 ports -----------------------------------------------------------------
HOST = "127.0.0.1"
PORT_SENSORS = 5555   # PUB, frontend binds  -> backend SUB
PORT_EVENTS = 5556    # PUB, backend binds   -> frontend SUB
PORT_CONTROL = 5557   # REP, backend binds   <- frontend REQ


def addr(port: int, host: str = HOST) -> str:
    return f"tcp://{host}:{port}"


# ---- §3 / §4 topics -----------------------------------------------------------
# 5555 (sensor stream)
TOPIC_EMG = b"emg"
TOPIC_IMU = b"imu"
TOPIC_HAND = b"hand"
TOPIC_STATUS = b"status"      # used on 5555 (sensor status) and 5556 (backend status)
# 5556 (backend outputs)
TOPIC_PROBS = b"probs"
TOPIC_GESTURE = b"gesture"

SENSOR_TOPICS = (TOPIC_EMG, TOPIC_IMU, TOPIC_HAND, TOPIC_STATUS)
EVENT_TOPICS = (TOPIC_PROBS, TOPIC_GESTURE, TOPIC_STATUS)

FS_EMG = 1000
FS_IMU = 200
N_LANDMARKS = 21


# ---- §1 encoding --------------------------------------------------------------
def _jsonable(o: Any):
    """json.dumps default= hook: numpy scalars/arrays -> plain Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _dumps(header: dict) -> bytes:
    return json.dumps(header, default=_jsonable, separators=(",", ":")).encode("utf-8")


def send_json(sock: zmq.Socket, topic: bytes, header: dict, flags: int = 0) -> None:
    """Send a JSON-only message: [topic, header]."""
    sock.send_multipart([topic, _dumps(header)], flags=flags)


def send_array(sock: zmq.Socket, topic: bytes, header: dict, arr: np.ndarray, flags: int = 0) -> None:
    """Send [topic, header, payload]. `dtype` and `shape` are filled in from `arr`
    (forced little-endian, C-contiguous). The caller's header dict is not modified."""
    a = np.asarray(arr)
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not np.little_endian):
        a = a.astype(a.dtype.newbyteorder("<"))
    a = np.ascontiguousarray(a)
    h = dict(header)
    h["dtype"] = a.dtype.str          # e.g. "<u2", "<f4", "|u1"
    h["shape"] = list(a.shape)
    payload = a.reshape(-1).view(np.uint8) if a.size else b""   # empty arrays (e.g. no hand) are legal
    sock.send_multipart([topic, _dumps(h), payload], flags=flags, copy=False)


def recv(sock: zmq.Socket, flags: int = 0, copy_array: bool = False):
    """Receive one message -> (topic, header, array_or_None).

    The array is a read-only view on the ZeroMQ frame unless copy_array=True.
    Raises zmq.Again with flags=zmq.NOBLOCK when nothing is waiting."""
    parts = sock.recv_multipart(flags=flags)
    topic = parts[0]
    header = json.loads(parts[1]) if len(parts) > 1 and parts[1] else {}
    arr = None
    if len(parts) > 2:
        arr = np.frombuffer(parts[2], dtype=np.dtype(header["dtype"])).reshape(header["shape"])
        if copy_array:
            arr = arr.copy()
    return topic, header, arr


def decode(parts: list) -> tuple:
    """Same as recv() but for a multipart list you already received."""
    topic = bytes(parts[0])
    header = json.loads(bytes(parts[1])) if len(parts) > 1 else {}
    arr = None
    if len(parts) > 2:
        arr = np.frombuffer(bytes(parts[2]), dtype=np.dtype(header["dtype"])).reshape(header["shape"])
    return topic, header, arr


# ---- socket helpers -----------------------------------------------------------
def make_pub(port: int, ctx: zmq.Context | None = None, hwm: int = 10000) -> zmq.Socket:
    """Bind a PUB socket on 127.0.0.1:port."""
    ctx = ctx or zmq.Context.instance()
    s = ctx.socket(zmq.PUB)
    s.setsockopt(zmq.SNDHWM, hwm)
    s.setsockopt(zmq.LINGER, 0)
    s.bind(addr(port))
    return s


def make_sub(port: int, topics=(b"",), ctx: zmq.Context | None = None, hwm: int = 10000) -> zmq.Socket:
    """Connect a SUB socket to 127.0.0.1:port, subscribed to `topics` (prefix match)."""
    ctx = ctx or zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, hwm)
    s.setsockopt(zmq.LINGER, 0)
    for t in topics:
        s.setsockopt(zmq.SUBSCRIBE, t)
    s.connect(addr(port))
    return s


class ControlClient:
    """REQ client for port 5557 with a timeout.

    A REQ socket that never gets a reply is stuck forever, so after a timeout the
    socket is closed and rebuilt. request() never raises for transport problems:
    it returns {"ok": False, "error": "..."} (with "offline": True on timeout)."""

    def __init__(self, port: int = PORT_CONTROL, timeout_s: float = 1.0, ctx: zmq.Context | None = None):
        self.ctx = ctx or zmq.Context.instance()
        self.port = port
        self.timeout_ms = int(timeout_s * 1000)
        self.sock = None
        self._open()

    def _open(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(addr(self.port))

    def request(self, msg: dict, timeout_s: float | None = None) -> dict:
        tmo = self.timeout_ms if timeout_s is None else int(timeout_s * 1000)
        try:
            self.sock.send(_dumps(msg))
            if self.sock.poll(tmo, zmq.POLLIN):
                return json.loads(self.sock.recv())
            self._open()
            return {"ok": False, "error": "backend offline (timeout)", "offline": True}
        except Exception as e:  # noqa: BLE001 - the caller wants a dict, not a crash
            self._open()
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "offline": True}

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None


def reply_json(sock: zmq.Socket, msg: dict) -> None:
    """For REP servers: send a JSON reply."""
    sock.send(_dumps(msg))


def now() -> float:
    """The shared host clock (INTERFACE §2)."""
    return time.perf_counter()


# ---- self test ----------------------------------------------------------------
def _selftest() -> None:
    ctx = zmq.Context()
    port = 25555
    pub = ctx.socket(zmq.PUB)
    pub.bind(addr(port))
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(addr(port))
    time.sleep(0.3)  # slow-joiner

    emg = (np.arange(40, dtype=np.uint16).reshape(20, 2) + 2000)
    imu = np.full((4, 6), np.nan, np.float32)
    big = np.arange(6, dtype=">f4")  # big-endian input must arrive little-endian
    send_array(pub, TOPIC_EMG, {"t0": np.float64(1.5), "fs": FS_EMG, "index0": np.int64(7)}, emg)
    send_array(pub, TOPIC_IMU, {"t0": 1.5, "fs": FS_IMU, "index0": 0}, imu)
    send_array(pub, b"x", {}, big)
    send_array(pub, TOPIC_HAND, {"t": 2.0, "score": 0.0, "handedness": None}, np.zeros((0, 3), np.float32))
    send_json(pub, TOPIC_STATUS, {"emg_ok": True, "dropped": np.int32(3)})

    assert sub.poll(2000)
    t, h, a = recv(sub)
    assert t == TOPIC_EMG and h["dtype"] == "<u2" and h["shape"] == [20, 2] and h["index0"] == 7
    assert np.array_equal(a, emg)
    t, h, a = recv(sub)
    assert t == TOPIC_IMU and a.dtype == np.float32 and np.isnan(a).all()
    t, h, a = recv(sub)
    assert h["dtype"] == "<f4" and np.array_equal(a, np.arange(6))
    t, h, a = recv(sub)
    assert t == TOPIC_HAND and a.shape == (0, 3) and h["handedness"] is None
    t, h, a = recv(sub)
    assert t == TOPIC_STATUS and a is None and h["dropped"] == 3

    # control client against nothing: must time out cleanly, twice
    cc = ControlClient(port=25557, timeout_s=0.2, ctx=ctx)
    for _ in range(2):
        r = cc.request({"cmd": "status"})
        assert r["ok"] is False and r.get("offline"), r
    # and against a real REP
    rep = ctx.socket(zmq.REP)
    rep.bind(addr(25557))
    import threading

    def serve():
        if rep.poll(2000):
            q = json.loads(rep.recv())
            reply_json(rep, {"ok": True, "echo": q["cmd"]})
    th = threading.Thread(target=serve)
    th.start()
    r = cc.request({"cmd": "status"}, timeout_s=2)
    th.join()
    assert r == {"ok": True, "echo": "status"}, r
    for s in (pub, sub, rep):
        s.close(0)
    cc.close()
    ctx.term()
    print("shared.protocol selftest: OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
