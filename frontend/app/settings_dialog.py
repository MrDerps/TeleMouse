"""Settings: connection, calibration, model sensors, pointer, data. One place for everything technical."""
from __future__ import annotations

import os
import threading

from PyQt6 import QtCore, QtWidgets

from ..acquire import config
from .pointer_panel import PointerPanel
from .sensor_picker import SensorPicker
from .theme import C
from .ui import Panel, button, label

SOURCES = ["ESP32 + camera", "ESP32 only", "Camera only", "Simulator", "External publisher"]


class SettingsDialog(QtWidgets.QDialog):
    calib_done = QtCore.pyqtSignal(dict)
    PAGES = ["Connection", "Calibration", "Model", "Pointer", "Data"]

    def __init__(self, app, page="Connection"):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Settings")
        self.resize(900, 640)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)
        self.nav = QtWidgets.QListWidget()
        self.nav.setFixedWidth(180)
        self.nav.addItems(self.PAGES)
        self.stack = QtWidgets.QStackedWidget()
        for build in (self._connection, self._calibration, self._model, self._pointer, self._data):
            sc = QtWidgets.QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(build())
            self.stack.addWidget(sc)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(self.PAGES.index(page) if page in self.PAGES else 0)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.calib_done.connect(self._calib_result)

    def _page(self, title, sub):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 0, 4, 0)
        v.setSpacing(12)
        v.addWidget(label(title, 20, 600))
        v.addWidget(label(sub, 13, 400, C["text2"], wrap=True))
        return w, v

    # ------------------------------------------------------------------ connection
    def _connection(self):
        w, v = self._page("Connection", "Where the sensor data comes from. The ESP32 carries EMG and the IMU; the "
                                        "webcam tracks your hand.")
        cfg = config.load()
        p = Panel("Source")
        form = QtWidgets.QFormLayout()
        self.src = QtWidgets.QComboBox()
        self.src.addItems(SOURCES)
        self.src.setCurrentText(cfg.get("source") or SOURCES[0])
        self.port = QtWidgets.QComboBox(editable=True)
        self.port.setMinimumWidth(140)
        self._refresh_ports(cfg.get("serial_port"))
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.port, 1)
        prow.addWidget(button("Refresh", None, lambda: self._refresh_ports(self.port.currentText())))
        self.cam = QtWidgets.QSpinBox(minimum=0, maximum=9, value=int(cfg.get("camera_index", 0)))
        self.notes = QtWidgets.QLineEdit(cfg.get("electrode_notes", ""))
        self.notes.setPlaceholderText("e.g. ch1 inner forearm, ch2 outer forearm, ref on elbow")
        form.addRow("Source", self.src)
        form.addRow("ESP32 port", prow)
        form.addRow("Camera", self.cam)
        form.addRow("Electrode notes", self.notes)
        p.add(form)
        row = QtWidgets.QHBoxLayout()
        self.conn_btn = button("Disconnect" if self.app.sources.mode else "Connect", "Primary", self._connect)
        row.addWidget(self.conn_btn)
        row.addStretch(1)
        p.add(row)
        sw = Panel("EMG channels")
        from .ui import Switch
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(label("Swap ch1 and ch2", 14, 500))
        r1.addStretch(1)
        self.swap_sw = Switch(bool(cfg.get("emg_swap", False)))
        self.swap_sw.toggled.connect(self.app.set_emg_swap)
        r1.addWidget(self.swap_sw)
        sw.add(r1)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(label("Detect swapped leads automatically", 14, 500))
        r2.addStretch(1)
        self.auto_sw = Switch(bool(cfg.get("emg_swap_auto", True)))
        self.auto_sw.toggled.connect(lambda on: config.save(emg_swap_auto=bool(on)))
        r2.addWidget(self.auto_sw)
        sw.add(r2)
        self.swap_status = label("", 12, 400, C["text2"], wrap=True)
        sw.add(self.swap_status)
        sw.add(label("Automatic check: after you connect or calibrate, the first few confident gestures are compared "
                     "with your recordings. If the EMG only matches with ch1 and ch2 swapped, they're swapped for you. "
                     "+/− swapped on a channel doesn't matter.", 12, 400, C["label"], wrap=True))
        self._swap_timer = QtCore.QTimer(self, interval=500)
        self._swap_timer.timeout.connect(self._swap_refresh)
        self._swap_timer.start()
        p.add(label("Safety: while electrodes are on a person, the ESP32 must run from a battery, a power bank or an "
                    "unplugged laptop. Never from a wall-powered PC.", 12, 400, C["warn"], wrap=True))
        v.addWidget(p)
        v.addWidget(sw)
        v.addStretch(1)
        for wdg in (self.src, self.port):
            wdg.currentTextChanged.connect(self._save_conn)
        self.cam.valueChanged.connect(self._save_conn)
        self.notes.textChanged.connect(self._save_conn)
        return w

    def _swap_refresh(self):
        cur = bool(config.load().get("emg_swap", False))
        if self.swap_sw.isChecked() != cur:
            self.swap_sw.blockSignals(True)
            self.swap_sw.setChecked(cur)
            self.swap_sw.blockSignals(False)
        self.swap_status.setText("Status: " + self.app.swap_watch.status +
                                 ("" if self.app.sources.acq else "  (needs the sensors running in this app)"))

    def _refresh_ports(self, keep=None):
        import serial.tools.list_ports as lp
        self.port.clear()
        self.port.addItems([p.device for p in lp.comports() if "Bluetooth" not in p.description])
        if keep:
            i = self.port.findText(keep)
            self.port.setCurrentIndex(i) if i >= 0 else self.port.setEditText(keep)

    def _save_conn(self, *_):
        config.save(source=self.src.currentText(), serial_port=self.port.currentText().strip() or None,
                    camera_index=self.cam.value(), electrode_notes=self.notes.text())

    def _connect(self):
        self._save_conn()
        if self.app.sources.mode:
            self.app.disconnect_sensors()
        else:
            self.app.connect_sensors(self)
        self.conn_btn.setText("Disconnect" if self.app.sources.mode else "Connect")

    # ------------------------------------------------------------------ calibration
    def _calibration(self):
        w, v = self._page("Calibration", "Two quick calibrations make the sensors line up and the muscle signal "
                                         "comparable between sessions.")
        p = Panel("Camera timing (sync)")
        lat = config.load().get("cam_latency", 0.08)
        self.sync_lbl = label(f"Current camera delay: {1000 * lat:.0f} ms", 14, 500)
        p.add(self.sync_lbl)
        p.add(label("Needs the ESP32 + camera connected. Hold your hand in view, press Start, then make ~5 sharp taps "
                    "that stop suddenly (onto the desk edge or your other palm), about 1 s apart, for 8 seconds.",
                    13, 400, C["text2"], wrap=True))
        self.sync_btn = button("Start sync calibration", None, self._sync)
        p.add(self.sync_btn)
        v.addWidget(p)
        p2 = Panel("Muscle resting level")
        p2.add(label("Relax your arm and hand completely for 3 seconds. Do this every time the electrode pads go on.",
                     13, 400, C["text2"], wrap=True))
        p2.add(button("Calibrate resting level (3 s)", None, self.app.calibrate_rest))
        v.addWidget(p2)
        v.addStretch(1)
        return w

    def _sync(self):
        acq = self.app.sources.acq
        if not acq or not acq.camera or not acq.reader:
            QtWidgets.QMessageBox.information(self, "Sync calibration", "Connect 'ESP32 + camera' first.")
            return
        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("TAP NOW… (8 s)")
        threading.Thread(target=lambda: self.calib_done.emit(acq.calibrate_sync(8.0)), daemon=True).start()

    def _calib_result(self, r):
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("Start sync calibration")
        if r.get("ok"):
            self.sync_lbl.setText(f"Current camera delay: {1000 * r['latency']:.0f} ms  (just calibrated)")
        (QtWidgets.QMessageBox.information if r.get("ok") else QtWidgets.QMessageBox.warning)(
            self, "Sync calibration", ("Saved. " if r.get("ok") else "Didn't work: ") + r.get("message", ""))

    # ------------------------------------------------------------------ model
    def _model(self):
        w, v = self._page("Model", "Which sensors the gesture model listens to, how much the muscle signal may count, "
                                   "and how sure it must be before acting.")
        p = Panel("Sensors")
        self.picker = SensorPicker(self.app, "Use", with_weight=True)
        p.add(self.picker)
        p.add(label("EMG shifts every time the pads move, so a low influence (≈20%) lets it help without taking over. "
                    "Changes apply to the next training; the sensor switches also apply when you start recognising.",
                    12, 400, C["label"], wrap=True))
        v.addWidget(p)
        p2 = Panel("Confidence")
        row = QtWidgets.QHBoxLayout()
        self.thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=50, maximum=99,
                                     value=int(100 * self.app.conf_threshold))
        self.thr_lbl = label(f"{self.app.conf_threshold:.2f}", 14, 500)
        self.thr.valueChanged.connect(self._thr)
        row.addWidget(label("Only act when at least", 13, 400, C["text2"]))
        row.addWidget(self.thr, 1)
        row.addWidget(self.thr_lbl)
        p2.add(row)
        p2.add(label("Raise it if you get accidental clicks, lower it if gestures are missed.", 12, 400, C["label"]))
        v.addWidget(p2)
        p3 = Panel("Switched-off gestures")
        r = QtWidgets.QHBoxLayout()
        r.addWidget(label("Use them to teach the model what to ignore", 14, 500))
        r.addStretch(1)
        from .ui import Switch
        neg = Switch(bool(config.load().get("off_as_negatives", True)))
        neg.toggled.connect(lambda on: config.save(off_as_negatives=bool(on)))
        r.addWidget(neg)
        p3.add(r)
        p3.add(label("A switched-off gesture's recordings are labelled 'not a gesture', so similar motions stop "
                     "triggering the others. Applies at the next training.", 12, 400, C["label"], wrap=True))
        v.addWidget(p3)
        v.addStretch(1)
        return w

    def _thr(self, val):
        self.app.conf_threshold = val / 100
        self.thr_lbl.setText(f"{val / 100:.2f}")
        config.save(conf_threshold=val / 100)

    # ------------------------------------------------------------------ pointer
    def _pointer(self):
        w, v = self._page("Pointer", "Move the mouse with your right hand: pinch to move, release to freeze. Or bind a "
                                     "hold gesture to 'Move mouse while held'.")
        self.ppanel = PointerPanel(self.app)
        v.addWidget(self.ppanel)
        v.addStretch(1)
        self._ptimer = QtCore.QTimer(self, interval=200)
        self._ptimer.timeout.connect(self.ppanel.refresh)
        self._ptimer.start()
        return w

    # ------------------------------------------------------------------ data
    def _data(self):
        w, v = self._page("Data", "Recordings and models live in the project's data folder.")
        p = Panel("Folder")
        p.add(label(str(config.REPO / "data"), 13, 400, C["text2"]))
        p.add(button("Open data folder", None, lambda: os.startfile(str(config.REPO / "data"))))
        v.addWidget(p)
        p2 = Panel("Start over")
        p2.add(label("Delete every recording, every model and the gesture list. Calibration and settings are kept. "
                     "This can't be undone.", 13, 400, C["text2"], wrap=True))
        p2.add(button("Delete all recordings, models and gestures…", "Danger", self._wipe))
        v.addWidget(p2)
        v.addStretch(1)
        return w

    def _wipe(self):
        txt, ok = QtWidgets.QInputDialog.getText(self, "Start over", "Type DELETE to remove all recordings, models "
                                                                     "and gestures:")
        if ok and txt.strip() == "DELETE":
            self.app.wipe_everything()
            QtWidgets.QMessageBox.information(self, "Start over", "Everything was deleted.")

    def closeEvent(self, e):
        self.app.settings_closed()
        super().closeEvent(e)

    def done(self, r):
        self.app.settings_closed()
        super().done(r)
