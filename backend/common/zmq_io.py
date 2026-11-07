"""Thin wrapper over shared/protocol.py (INTERFACE.md section 1), so backend code has one import point."""
from __future__ import annotations

import numpy as np
import zmq

from shared import protocol as P

HOST = P.HOST
PORT_SENSORS = P.PORT_SENSORS
PORT_OUTPUTS = P.PORT_EVENTS
PORT_CONTROL = P.PORT_CONTROL
addr = P.addr
decode = P.decode


def send(sock: zmq.Socket, topic: bytes | str, header: dict, array: np.ndarray | None = None, flags: int = 0) -> None:
    if isinstance(topic, str):
        topic = topic.encode()
    if array is None:
        P.send_json(sock, topic, header, flags)
    else:
        P.send_array(sock, topic, header, array, flags)


def recv(sock: zmq.Socket, flags: int = 0, copy_array: bool = False):
    return P.recv(sock, flags, copy_array)
