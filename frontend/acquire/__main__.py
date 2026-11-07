"""Acquisition service: ESP32 (EMG + IMU) + webcam (hand landmarks) -> port 5555, per INTERFACE §3.

    python -m frontend.acquire --port COM7                 serial + camera 0
    python -m frontend.acquire --port COM7 --calibrate-sync   tap calibration of the camera latency (opens a preview)
    python -m frontend.acquire --port COM7 --preview       show the camera with landmarks
    python -m frontend.acquire --port COM7 --no-camera
    python -m frontend.acquire --no-serial                 camera only
    python -m frontend.acquire --list                      serial ports
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

os.environ.setdefault("GLOG_minloglevel", "2")        # quieter MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from shared import protocol as P  # noqa: E402

from . import config  # noqa: E402
from .service import SAFETY, Acquisition  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class Preview:
    """OpenCV window drawn from the main thread (OpenCV windows must live on one thread)."""

    def __init__(self, acq):
        import cv2
        self.cv2, self.acq = cv2, acq
        self.banner = ""
        self.hits = []                     # recent (t, hand_seen)

    def hand_fraction(self, window=1.0):
        now = time.perf_counter()
        h = [x for t, x in self.hits if now - t < window]
        return sum(h) / len(h) if h else 0.0

    def tick(self):
        cv2, cam = self.cv2, self.acq.camera
        if cam is None or cam.preview is None:
            cv2.waitKey(1)
            return
        frame, lm = cam.preview
        self.hits.append((time.perf_counter(), lm is not None))
        self.hits = self.hits[-90:]
        img = frame.copy()
        h, w = img.shape[:2]
        if lm is not None:
            pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
            for a, b in EDGES:
                cv2.line(img, pts[a], pts[b], (80, 220, 120), 2)
            for p in pts:
                cv2.circle(img, p, 3, (40, 160, 255), -1)
        ok = lm is not None
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, ("HAND OK" if ok else "NO HAND") + f"  {100 * self.hand_fraction():.0f}%",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120) if ok else (60, 60, 255), 2)
        if self.banner:
            cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.putText(img, self.banner, (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("EMG Mouse camera (q to close)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def wait(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.tick()
            time.sleep(0.02)

    def close(self):
        self.cv2.destroyAllWindows()


def run_calibration(acq, pv):
    print("\n=== SYNC CALIBRATION ===\n"
          "IMU strapped on the back of the hand. Hold the hand where the camera sees it clearly\n"
          "(check the preview window: it must say HAND OK). Then, when it says GO, make ~5 sharp taps,\n"
          "~1 s apart, that STOP SUDDENLY: tap the hand onto the desk edge, or slap it onto your other palm,\n"
          "keeping it in view the whole time.", flush=True)
    t0 = time.perf_counter()
    while pv.hand_fraction() < 0.8:
        pv.banner = "Show your hand to the camera"
        pv.wait(0.2)
        if time.perf_counter() - t0 > 30:
            print("  Hand still not visible after 30 s; skipping calibration.", flush=True)
            pv.banner = ""
            return
    for k in (3, 2, 1):
        pv.banner = f"Hand found. Starting in {k}..."
        print(f"  {k}...", flush=True)
        pv.wait(1.0)
    print("  GO! (8 s)", flush=True)
    res = {}
    th = threading.Thread(target=lambda: res.update(acq.calibrate_sync(8.0)), daemon=True)
    th.start()
    t_go = time.perf_counter()
    while th.is_alive():
        pv.banner = f"TAP! sharp stops, ~1 per second   {max(0.0, 8 - (time.perf_counter() - t_go)):.0f}s"
        pv.wait(0.05)
    msg = ("  OK: " if res.get("ok") else "  FAILED: ") + res.get("message", "?")
    msg += f"  -> saved to {config.PATH}" if res.get("ok") else "  (keeping the previous value)"
    print(msg + "\n", flush=True)
    pv.banner = ("OK " if res.get("ok") else "FAILED: ") + res.get("message", "")[:48]


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
    ap.add_argument("--preview", action="store_true", help="show the camera with landmarks")
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
    pv = Preview(acq) if (a.preview or a.calibrate_sync) and acq.camera else None
    try:
        if a.calibrate_sync:
            if pv is None:
                print("Sync calibration needs the camera.")
            else:
                pv.wait(1.5)
                run_calibration(acq, pv)
        nxt = time.perf_counter() + 1.0
        while True:
            if pv:
                pv.wait(max(0.0, nxt - time.perf_counter()))
            else:
                time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt += 1.0
            s = acq.last_status
            if s:
                print(f"[acquire] emg {s['emg_rate']:4.0f}/s  imu {s['imu_rate']:3.0f}/s  cam {s['cam_fps']:2.0f} fps"
                      f"  dropped {s['dropped']}  imu={s['imu_state']}"
                      + ("" if s["emg_ok"] or not port else "  (NO SERIAL DATA)")
                      + ("" if s["cam_ok"] or a.no_camera else "  (NO CAMERA)"), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if pv:
            pv.close()
        acq.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
