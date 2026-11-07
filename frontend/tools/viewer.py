"""Live viewer for the 5555 sensor stream (works with the simulator or real hardware).

    python -m frontend.tools.viewer            (--port N to watch another port)

Shows EMG ch1/ch2 (display-only filtering: 20 Hz high-pass + 60 Hz notch, plus a 25 ms RMS
envelope), IMU accel and gyro, the hand landmarks, and the stream status.
A background thread does all ZeroMQ I/O; the UI thread only draws.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos  # noqa: E402

from shared import protocol as P  # noqa: E402

SECONDS = 5
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class Ring:
    def __init__(self, n, cols, dtype=np.float32, fill=np.nan):
        self.a = np.full((n, cols), fill, dtype)
        self.n = n

    def push(self, x):
        k = len(x)
        if k >= self.n:
            self.a[:] = x[-self.n:]
        else:
            self.a[:-k] = self.a[k:]
            self.a[-k:] = x


class Streams:
    """Filled by the receiver thread, read by the UI under the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.emg = Ring(SECONDS * 1000, 2)
        self.env = Ring(SECONDS * 1000, 2)
        self.imu = Ring(SECONDS * 200, 6)
        self.hand = None
        self.hand_score = 0.0
        self.status = None
        self.counts = {"emg": 0, "imu": 0, "hand": 0}
        self.last_msg = 0.0
        hp = butter(2, 20, "highpass", fs=1000, output="sos")
        b, a = iirnotch(60, 30, fs=1000)
        self.sos = np.vstack([hp, tf2sos(b, a)])
        self.zi = None
        self.env_tail = np.zeros((24, 2))
        self.stop = threading.Event()

    def filt(self, x):
        x = x.astype(np.float64)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        full = np.vstack([self.env_tail, y ** 2])
        c = np.vstack([np.zeros((1, 2)), np.cumsum(full, axis=0)])
        env = np.sqrt(np.maximum(c[25:] - c[:-25], 0) / 25)
        self.env_tail = full[-24:]
        return y, env

    def run(self, port=P.PORT_SENSORS):
        sub = P.make_sub(port)
        while not self.stop.is_set():
            if not sub.poll(100):
                continue
            topic, h, a = P.recv(sub)
            with self.lock:
                self.last_msg = time.perf_counter()
                if topic == P.TOPIC_EMG:
                    y, env = self.filt(a)
                    self.emg.push(y)
                    self.env.push(env)
                    self.counts["emg"] += len(a)
                elif topic == P.TOPIC_IMU:
                    self.imu.push(a)
                    self.counts["imu"] += len(a)
                elif topic == P.TOPIC_HAND:
                    self.hand = a.copy() if len(a) else None
                    self.hand_score = h.get("score", 0)
                    self.counts["hand"] += 1
                elif topic == P.TOPIC_STATUS:
                    self.status = h
        sub.close(0)


def main():
    import pyqtgraph as pg
    from PyQt6 import QtCore, QtWidgets

    S = Streams()
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else P.PORT_SENSORS
    th = threading.Thread(target=S.run, args=(port,), daemon=True)
    th.start()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, background="#15191c", foreground="#c9d1d9")
    win = QtWidgets.QWidget()
    win.setWindowTitle("EMG Mouse - live sensor viewer (port 5555)")
    lay = QtWidgets.QGridLayout(win)
    status = QtWidgets.QLabel("waiting for data on port 5555...")
    status.setStyleSheet("font: 11pt 'Consolas'; padding: 4px;")
    lay.addWidget(status, 0, 0, 1, 2)

    glw = pg.GraphicsLayoutWidget()
    lay.addWidget(glw, 1, 0)
    t_emg = np.arange(-SECONDS * 1000, 0) / 1000
    t_imu = np.arange(-SECONDS * 200, 0) / 200
    cols = ["#4ea1ff", "#ff9f43", "#7bd88f"]
    p1 = glw.addPlot(row=0, col=0, title="EMG ch1 (GPIO32), filtered counts")
    p2 = glw.addPlot(row=1, col=0, title="EMG ch2 (GPIO33), filtered counts")
    p3 = glw.addPlot(row=2, col=0, title="Accel (g)  x / y / z")
    p4 = glw.addPlot(row=3, col=0, title="Gyro (deg/s)  x / y / z")
    for p in (p2, p3, p4):
        p.setXLink(p1)
    c_emg = [p1.plot(pen=pg.mkPen(cols[0], width=1)), p2.plot(pen=pg.mkPen(cols[1], width=1))]
    c_env = [p1.plot(pen=pg.mkPen("w", width=2)), p2.plot(pen=pg.mkPen("w", width=2))]
    c_acc = [p3.plot(pen=pg.mkPen(c, width=1)) for c in cols]
    c_gyr = [p4.plot(pen=pg.mkPen(c, width=1)) for c in cols]

    hw = pg.PlotWidget(title="Hand landmarks (image coords)")
    hw.setFixedWidth(360)
    hw.setXRange(0, 1)
    hw.setYRange(0, 1)
    hw.invertY(True)
    hw.setAspectLocked(True)
    lay.addWidget(hw, 1, 1)
    bones = hw.plot(pen=pg.mkPen("#7bd88f", width=2), connect="pairs")
    dots = hw.plot(pen=None, symbol="o", symbolSize=6, symbolBrush="#ff9f43")

    last = {"t": time.perf_counter(), "c": dict(S.counts)}
    rates = {"emg": 0.0, "imu": 0.0, "hand": 0.0}

    def tick():
        with S.lock:
            e, env, im = S.emg.a.copy(), S.env.a.copy(), S.imu.a.copy()
            hand, score, st, lm = S.hand, S.hand_score, S.status, S.last_msg
            counts = dict(S.counts)
        now = time.perf_counter()
        if now - last["t"] >= 1.0:
            for k in rates:
                rates[k] = (counts[k] - last["c"][k]) / (now - last["t"])
            last["t"], last["c"] = now, counts
        for i in range(2):
            c_emg[i].setData(t_emg, e[:, i])
            c_env[i].setData(t_emg, env[:, i])
        for i in range(3):
            c_acc[i].setData(t_imu, im[:, i])
            c_gyr[i].setData(t_imu, im[:, 3 + i])
        if hand is not None:
            seg = np.array([hand[[a, b], :2] for a, b in HAND_EDGES]).reshape(-1, 2)
            bones.setData(seg[:, 0], seg[:, 1])
            dots.setData(hand[:, 0], hand[:, 1])
        else:
            bones.setData([], [])
            dots.setData([], [])
        if now - lm > 1.0:
            status.setText("NO DATA on port 5555 - start `python -m frontend.acquire --port COMx` "
                           "or `python -m frontend.tools.replay --sim`")
            status.setStyleSheet("font: 11pt 'Consolas'; padding: 4px; color: #ff6b6b;")
        else:
            s = st or {}
            def light(ok):
                return "OK " if ok else "-- "
            status.setText(
                f"EMG {light(s.get('emg_ok'))}{rates['emg']:6.0f} Hz   IMU {light(s.get('imu_ok'))}{rates['imu']:4.0f} Hz"
                f"   CAM {light(s.get('cam_ok'))}{rates['hand']:3.0f} fps (hand {'yes' if hand is not None else 'no '}"
                f" {score:.2f})   dropped {s.get('dropped', '?')}   {s.get('source', '')}")
            status.setStyleSheet("font: 11pt 'Consolas'; padding: 4px; color: #c9d1d9;")

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(33)
    win.resize(1300, 850)
    win.show()
    if "--snapshot" in sys.argv:            # for tests: save a PNG after 4 s and quit
        path = sys.argv[sys.argv.index("--snapshot") + 1]
        QtCore.QTimer.singleShot(4000, lambda: (win.grab().save(path), app.quit()))
    rc = app.exec()
    S.stop.set()
    th.join(1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
