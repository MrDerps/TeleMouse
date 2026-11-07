"""Acquisition service: ESP32 (EMG + IMU) + webcam (hand landmarks) -> port 5555, per INTERFACE §3.

    python -m frontend.acquire --port COM7                 serial + camera 0
    python -m frontend.acquire --port COM7 --calibrate-sync   tap the hand ~5 times, save camera latency
    python -m frontend.acquire --port COM7 --no-camera
    python -m frontend.acquire --no-serial                 camera only
    python -m frontend.acquire --list                      serial ports
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")        # quieter MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from shared import protocol as P  # noqa: E402

from . import config  # noqa: E402
from .service import SAFETY, Acquisition  # noqa: E402


def main(argv=None):
    cfg = config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=cfg.get("serial_port"), help="ESP32 serial port, e.g. COM7")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--camera", type=int, default=cfg.get("camera_index", 0))
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--pub-port", type=int, default=P.PORT_SENSORS)
    ap.add_argument("--calibrate-sync", action="store_true", help="run the tap calibration after start-up")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            print(f"{p.device:8} {p.description}")
        return 0
    port = None if a.no_serial else a.port
    if not port and not a.no_serial:
        return "give --port COMx (see --list), or --no-serial for camera only"

    if port:
        print(SAFETY)
        config.save(serial_port=port)
    acq = Acquisition(serial_port=port, camera_index=None if a.no_camera else a.camera,
                      pub_port=a.pub_port, baud=a.baud, battery_warning_shown=bool(port))
    try:
        acq.start()
    except RuntimeError as e:
        return f"[acquire] {e}"
    try:
        if a.calibrate_sync:
            time.sleep(2.0)
            print("\n=== SYNC CALIBRATION ===\nKeep your hand in view of the camera, IMU strapped on.\n"
                  "When it says GO, tap the hand sharply flat on the desk about 5 times, ~1 s apart.")
            for k in (3, 2, 1):
                print(f"  {k}...", flush=True)
                time.sleep(1)
            print("  GO! (8 s)", flush=True)
            r = acq.calibrate_sync(8.0)
            print(("  OK: " if r["ok"] else "  FAILED: ") + r["message"]
                  + (f"  -> saved to {config.PATH}" if r["ok"] else "  (keeping the previous value)") + "\n", flush=True)
        while True:
            time.sleep(1.0)
            s = acq.last_status
            if s:
                print(f"[acquire] emg {s['emg_rate']:4.0f}/s  imu {s['imu_rate']:3.0f}/s  cam {s['cam_fps']:2.0f} fps"
                      f"  dropped {s['dropped']}  imu={s['imu_state']}"
                      + ("" if s["emg_ok"] or not port else "  (NO SERIAL DATA)")
                      + ("" if s["cam_ok"] or a.no_camera else "  (NO CAMERA)"), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        acq.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
