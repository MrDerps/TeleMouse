"""Watch a 5555 sensor stream (or 5556 backend stream) and print rates and problems.

    python -m frontend.tools.monitor                  sensor stream on 5555
    python -m frontend.tools.monitor --events         backend probs/gesture/status on 5556
    python -m frontend.tools.monitor --seconds 10

Checks: per-stream sample rate, index0 continuity (gaps/overlaps), header fields per
INTERFACE §3, hand presence, and how late messages arrive (now - timestamp of newest sample).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import protocol as P  # noqa: E402

REQ = {
    b"emg": {"t0", "fs", "index0", "dtype", "shape"},
    b"imu": {"t0", "fs", "index0", "dtype", "shape"},
    b"hand": {"t", "score", "handedness", "img_w", "img_h", "dtype", "shape"},
    b"status": {"emg_ok", "imu_ok", "cam_ok", "emg_rate", "imu_rate", "cam_fps", "dropped", "battery_warning_shown"},
}


class Monitor:
    def __init__(self, port=P.PORT_SENSORS, events=False):
        self.events = events
        self.sock = P.make_sub(port)

    def run(self, seconds=None, print_every=2.0) -> dict:
        start = P.now()
        win_start = start
        tot = dict(emg=0, imu=0, cam=0, status_msgs=0, gaps=0, bad_hdr=0, nan_imu=0)
        win = dict(emg=0, imu=0, cam=0, hand=0)
        nxt = {b"emg": None, b"imu": None}
        lat = {b"emg": [], b"imu": [], b"hand": []}
        last_status = None
        first = {}
        try:
            while seconds is None or P.now() - start < seconds:
                if not self.sock.poll(100):
                    continue
                topic, h, a = P.recv(self.sock)
                t_now = P.now()
                first.setdefault(topic, t_now)
                if self.events:
                    if topic == b"gesture":
                        print(f"  gesture {h}")
                    elif topic == b"status":
                        print(f"  status  {h}")
                    win["emg"] += topic == b"probs"
                    continue
                if topic in REQ and not REQ[topic] <= set(h):
                    tot["bad_hdr"] += 1
                    print(f"  ! {topic.decode()} header missing {REQ[topic] - set(h)}")
                if topic in (b"emg", b"imu"):
                    n = a.shape[0]
                    if nxt[topic] is not None and h["index0"] != nxt[topic]:
                        tot["gaps"] += 1
                        print(f"  ! {topic.decode()} index gap: expected {nxt[topic]} got {h['index0']}")
                    nxt[topic] = h["index0"] + n
                    lat[topic].append(t_now - (h["t0"] + (n - 1) / h["fs"]))
                    key = "emg" if topic == b"emg" else "imu"
                    win[key] += n
                    tot[key] += n
                    if topic == b"imu" and np.isnan(a).all():
                        tot["nan_imu"] += n
                elif topic == b"hand":
                    win["cam"] += 1
                    tot["cam"] += 1
                    win["hand"] += a.shape[0] > 0
                    lat[topic].append(t_now - h["t"])
                elif topic == b"status":
                    tot["status_msgs"] += 1
                    last_status = h
                if t_now - win_start >= print_every:
                    dt = t_now - win_start
                    ml = {k.decode(): (1000 * np.median(v) if v else float("nan")) for k, v in lat.items()}
                    print(f"emg {win['emg'] / dt:7.1f} Hz | imu {win['imu'] / dt:6.1f} Hz | cam {win['cam'] / dt:5.1f} fps "
                          f"(hand {100 * win['hand'] / max(win['cam'], 1):3.0f}%) | lateness ms emg {ml['emg']:.0f} "
                          f"imu {ml['imu']:.0f} hand {ml['hand']:.0f} | gaps {tot['gaps']}"
                          + (f" | status dropped={last_status.get('dropped')} imu_ok={last_status.get('imu_ok')}"
                             if last_status else ""), flush=True)
                    win = dict(emg=0, imu=0, cam=0, hand=0)
                    lat = {k: [] for k in lat}
                    win_start = t_now
        except KeyboardInterrupt:
            pass
        el = P.now() - start
        # rates measured from first message of each stream, so the subscribe delay doesn't count
        res = dict(tot)
        for k, topic in (("emg_rate", b"emg"), ("imu_rate", b"imu"), ("cam_fps", b"hand")):
            span = P.now() - first.get(topic, P.now())
            res[k] = tot[{"emg_rate": "emg", "imu_rate": "imu", "cam_fps": "cam"}[k]] / span if span > 0.5 else 0.0
        res["seconds"] = el
        res["last_status"] = last_status
        return res

    def close(self):
        self.sock.close(0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", action="store_true", help="watch backend port 5556 instead")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args(argv)
    port = a.port or (P.PORT_EVENTS if a.events else P.PORT_SENSORS)
    print(f"monitoring {P.addr(port)} (Ctrl+C to stop)")
    m = Monitor(port, events=a.events)
    res = m.run(a.seconds)
    m.close()
    print({k: v for k, v in res.items() if k != "last_status"})


if __name__ == "__main__":
    main()
