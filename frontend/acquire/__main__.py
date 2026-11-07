"""Acquisition service: ESP32 (EMG + IMU) -> port 5555, per INTERFACE §3.

    python -m frontend.acquire --port COM7
    python -m frontend.acquire --list

This is the M2 test-run version: serial only. The camera thread and sync calibration
come in M3 (until then status.cam_ok is false and no `hand` messages are sent).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from shared import protocol as P

from .serial_reader import SerialReader

SAFETY = ("SAFETY: while electrodes are on a person, the ESP32 must be powered from a battery, "
          "a power bank or an UNPLUGGED laptop. Never from a wall-powered PC.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM7")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--pub-port", type=int, default=P.PORT_SENSORS)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.port:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            print(f"{p.device:8} {p.description}")
        return 0 if a.list else "give --port COMx"

    print(SAFETY)
    pub = P.make_pub(a.pub_port)
    lock = threading.Lock()                       # zmq sockets are not thread-safe
    win = {"emg": 0, "imu": 0}

    def on_block(t0, i0, emg, imu_t0, k0, imu):
        with lock:
            P.send_array(pub, P.TOPIC_EMG, {"t0": t0, "fs": P.FS_EMG, "index0": i0}, emg)
            win["emg"] += len(emg)
            if imu is not None:
                P.send_array(pub, P.TOPIC_IMU, {"t0": imu_t0, "fs": P.FS_IMU, "index0": k0}, imu)
                win["imu"] += int((~(imu != imu).any(axis=1)).sum())   # rows without NaN

    rd = SerialReader(a.port, a.baud, on_block=on_block)
    rd.start()
    print(f"[acquire] {a.port} -> {P.addr(a.pub_port)}   Ctrl+C to stop")
    nxt = time.perf_counter() + 1.0
    try:
        while True:
            time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt += 1.0
            st = rd.status
            now = time.perf_counter()
            emg_ok = now - rd.last_rx < 0.5
            imu_ok = bool(emg_ok and st is not None and st.imu_state == 0 and win["imu"] > 0)
            hdr = {"emg_ok": emg_ok, "imu_ok": imu_ok, "cam_ok": False,
                   "emg_rate": float(win["emg"]), "imu_rate": float(win["imu"]), "cam_fps": 0.0,
                   "dropped": rd.dropped, "battery_warning_shown": True,
                   "imu_state": st.imu_state_name if st else None,
                   "crc_errors": rd.parser.crc_errors}
            with lock:
                P.send_json(pub, P.TOPIC_STATUS, hdr)
            ppm = (rd.clock.period / rd.clock.nominal - 1) * 1e6
            print(f"[acquire] emg {win['emg']:4d}/s  imu {win['imu']:3d}/s  dropped {rd.dropped}"
                  f"  crc {rd.parser.crc_errors}  imu={hdr['imu_state']}  clock {ppm:+.0f} ppm"
                  + ("" if emg_ok else "  (NO DATA)"), flush=True)
            win = {"emg": 0, "imu": 0}
    except KeyboardInterrupt:
        pass
    finally:
        rd.stop()
        pub.close(0)
        print("[acquire] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
