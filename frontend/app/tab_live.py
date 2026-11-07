"""Live tab: sensor source control, EMG/IMU traces, camera, status lights, backend probs and gestures."""
from __future__ import annotations

import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from ..acquire import config
from .widgets import CameraView, ProbBars, light

SAFETY_TEXT = ("Before streaming EMG with electrodes on a person:\n\n"
               "The ESP32 must be powered from a BATTERY, a POWER BANK or an UNPLUGGED laptop.\n"
               "Never from a wall-powered PC.\n\nIs the laptop unplugged from the wall (or are no electrodes attached)?")


class LiveTab(QtWidgets.QWidget):
    calib_done = QtCore.pyqtSignal(dict)

    def __init__(self, app):
        super().__init__()
        self.app = app
        cfg = config.load()
        # --- source controls
        self.src = QtWidgets.QComboBox()
        self.src.addItems(["ESP32 + camera", "ESP32 only", "Simulator", "External publisher"])
        self.port = QtWidgets.QComboBox(editable=True)
        self.port.setMinimumWidth(110)
        self.refresh_ports(cfg.get("serial_port"))
        self.cam = QtWidgets.QSpinBox(minimum=0, maximum=9, value=int(cfg.get("camera_index", 0)))
        self.btn = QtWidgets.QPushButton("Start sensors")
        self.btn.clicked.connect(self.toggle)
        self.sync_btn = QtWidgets.QPushButton("Sync calibration…")
        self.sync_btn.clicked.connect(self.sync_calibration)
        rp = QtWidgets.QPushButton("Refresh")
        rp.clicked.connect(lambda: self.refresh_ports(self.port.currentText()))
        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Source"), self.src, QtWidgets.QLabel("Port"), self.port, rp,
                  QtWidgets.QLabel("Camera"), self.cam, self.btn, self.sync_btn):
            top.addWidget(w)
        top.addStretch(1)
        self.calib_done.connect(self._calib_result)

        # --- plots
        pg.setConfigOptions(antialias=False)
        glw = pg.GraphicsLayoutWidget()
        self.t_emg = np.arange(-5000, 0) / 1000
        self.t_imu = np.arange(-1000, 0) / 200
        cols = ["#4ea1ff", "#ff9f43", "#7bd88f"]
        p1 = glw.addPlot(row=0, col=0, title="EMG ch1 (display filter: 20 Hz HP + 60 Hz notch, white = envelope)")
        p2 = glw.addPlot(row=1, col=0, title="EMG ch2")
        p3 = glw.addPlot(row=2, col=0, title="Accel (g)")
        p4 = glw.addPlot(row=3, col=0, title="Gyro (deg/s)")
        for p in (p2, p3, p4):
            p.setXLink(p1)
        self.c_emg = [p1.plot(pen=pg.mkPen(cols[0])), p2.plot(pen=pg.mkPen(cols[1]))]
        self.c_env = [p1.plot(pen=pg.mkPen("w", width=2)), p2.plot(pen=pg.mkPen("w", width=2))]
        self.c_acc = [p3.plot(pen=pg.mkPen(c)) for c in cols]
        self.c_gyr = [p4.plot(pen=pg.mkPen(c)) for c in cols]

        # --- right column
        self.camview = CameraView()
        self.lights = QtWidgets.QLabel()
        self.lights.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lights.setWordWrap(True)
        self.probs = ProbBars()
        self.log = QtWidgets.QListWidget()
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.camview, 3)
        right.addWidget(self.lights)
        right.addWidget(QtWidgets.QLabel("Backend probabilities"))
        right.addWidget(self.probs, 2)
        right.addWidget(QtWidgets.QLabel("Recent gestures"))
        right.addWidget(self.log, 2)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(420)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(glw, 1)
        body.addWidget(rw)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(body, 1)
        self._rates_t = time.perf_counter()
        self._rates_c = None
        self.rates = {"emg": 0.0, "imu": 0.0, "hand": 0.0}

    # ------------------------------------------------------------------ sources
    def refresh_ports(self, keep=None):
        import serial.tools.list_ports as lp
        self.port.clear()
        ports = [p.device for p in lp.comports() if "Bluetooth" not in p.description]
        self.port.addItems(ports or [])
        if keep:
            i = self.port.findText(keep)
            self.port.setCurrentIndex(i) if i >= 0 else self.port.setEditText(keep)

    def toggle(self):
        srcs = self.app.sources
        if srcs.mode:
            srcs.stop()
            self.btn.setText("Start sensors")
            self.app.log("sensors stopped")
            return
        kind = self.src.currentText()
        try:
            if kind.startswith("ESP32"):
                if QtWidgets.QMessageBox.warning(self, "Safety: battery power", SAFETY_TEXT,
                                                 QtWidgets.QMessageBox.StandardButton.Yes |
                                                 QtWidgets.QMessageBox.StandardButton.Cancel) != \
                        QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                port = self.port.currentText().strip()
                if not port:
                    raise RuntimeError("choose the ESP32 serial port")
                config.save(serial_port=port, camera_index=self.cam.value())
                srcs.start_hardware(port, self.cam.value(), use_camera=(kind == "ESP32 + camera"))
            elif kind == "Simulator":
                srcs.start_sim()
            else:
                self.app.log("listening for an external publisher on 5555")
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not start", str(e))
            return
        self.btn.setText("Stop sensors")
        self.app.log(f"sensors started: {kind}")

    def sync_calibration(self):
        acq = self.app.sources.acq
        if not acq or not acq.camera or not acq.reader:
            QtWidgets.QMessageBox.information(self, "Sync calibration",
                                              "Start 'ESP32 + camera' first (needs the IMU and the camera).")
            return
        if QtWidgets.QMessageBox.information(
                self, "Sync calibration",
                "IMU strapped on, hand in the middle of the camera view (check the preview here).\n\n"
                "After OK you have 8 seconds: make ~5 sharp taps that STOP SUDDENLY "
                "(tap onto the desk edge or slap your other palm), ~1 s apart.",
                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel) != \
                QtWidgets.QMessageBox.StandardButton.Ok:
            return
        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("TAP NOW… (8 s)")
        threading.Thread(target=lambda: self.calib_done.emit(acq.calibrate_sync(8.0)), daemon=True).start()

    def _calib_result(self, r):
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("Sync calibration…")
        (QtWidgets.QMessageBox.information if r.get("ok") else QtWidgets.QMessageBox.warning)(
            self, "Sync calibration", ("Saved: " if r.get("ok") else "Failed: ") + r.get("message", ""))
        self.app.log("sync calibration: " + r.get("message", ""))

    # ------------------------------------------------------------------ refresh (UI timer)
    def refresh(self, snap):
        now = time.perf_counter()
        if self._rates_c is None or now - self._rates_t >= 1.0:
            if self._rates_c is not None:
                dt = now - self._rates_t
                self.rates = {k: (snap["counts"][k] - self._rates_c[k]) / dt for k in self.rates}
            self._rates_t, self._rates_c = now, snap["counts"]
        if not self.isVisible():
            return
        for i in range(2):
            self.c_emg[i].setData(self.t_emg, snap["emg"][:, i])
            self.c_env[i].setData(self.t_emg, snap["env"][:, i])
        for i in range(3):
            self.c_acc[i].setData(self.t_imu, snap["imu"][:, i])
            self.c_gyr[i].setData(self.t_imu, snap["imu"][:, 3 + i])
        cam = self.app.sources.camera
        if cam is not None and cam.preview is not None:
            self.camview.show_frame(*cam.preview)
        else:
            self.camview.show_landmarks(snap["hand"])
        st, fresh = snap["status"], now - snap["last_msg"] < 1.5
        be = self.app.backend
        bs = be.status
        self.lights.setText(
            light(f"EMG {self.rates['emg']:.0f} Hz", fresh and st.get("emg_ok")) + " &nbsp; " +
            light(f"IMU {self.rates['imu']:.0f} Hz", fresh and st.get("imu_ok")) + " &nbsp; " +
            light(f"Camera {self.rates['hand']:.0f} fps", fresh and st.get("cam_ok")) + "<br>" +
            f"dropped {st.get('dropped', 0)} &nbsp; cam latency {1000 * (st.get('cam_latency') or 0):.0f} ms" +
            (f" &nbsp; <b>{st.get('source')}</b>" if st.get("source") else "") + "<br>" +
            light("backend " + (f"{bs.get('state')}  model {bs.get('model')}" if be.online else "offline"),
                  True if be.online else None) +
            (f"<br><span style='color:#f85149'>{bs.get('emg_warning')}</span>" if be.online and bs.get("emg_warning")
             else ""))
        pr = be.probs
        self.probs.set(*(pr[1:] if pr and be.online else ([], [])))

    def add_events(self, events):
        for e in events:
            self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}  {e.get('id')}  {e.get('kind')}  conf {e.get('conf', 0):.2f}")
        while self.log.count() > 100:
            self.log.takeItem(self.log.count() - 1)
