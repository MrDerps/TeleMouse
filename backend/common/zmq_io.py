"""Multipart ZeroMQ helpers per INTERFACE.md section 1.

Stand-in until/unless shared/protocol.py exists (Jerry: it won't be ready, so this is the backend's copy).
Message = [topic, header JSON] or [topic, header JSON, raw payload] with header dtype/shape.
"""
from __future__ import annotations

import json

import numpy as np
import zmq

HOST = "127.0.0.1"
PORT_SENSORS = 5555
PORT_OUTPUTS = 5556
PORT_CONTROL = 5557


def addr(port: int) -> str:
    return f"tcp://{HOST}:{port}"


def send(sock: zmq.Socket, topic: bytes | str, header: dict, array: np.ndarray | None = None, flags: int = 0) -> None:
    if isinstance(topic, str):
        topic = topic.encode()
    header = dict(header)
    if array is not None:
        arr = np.ascontiguousarray(array)
        dt = arr.dtype.newbyteorder("<") if arr.dtype.byteorder == ">" else arr.dtype
        arr = arr.astype(dt, copy=False)
        header["dtype"] = arr.dtype.str
        header["shape"] = list(arr.shape)
        sock.send_multipart([topic, json.dumps(header).encode(), arr.tobytes()], flags=flags)
    else:
        sock.send_multipart([topic, json.dumps(header).encode()], flags=flags)


def decode(parts: list[bytes]) -> tuple[bytes, dict, np.ndarray | None]:
    topic = parts[0]
    header = json.loads(parts[1].decode()) if len(parts) > 1 else {}
    arr = None
    if len(parts) > 2:
        arr = np.frombuffer(parts[2], dtype=np.dtype(header["dtype"])).reshape(header["shape"])
    return topic, header, arr


def recv(sock: zmq.Socket, flags: int = 0) -> tuple[bytes, dict, np.ndarray | None]:
    return decode(sock.recv_multipart(flags=flags))
