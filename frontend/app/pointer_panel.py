"""Pointer settings panel (Run tab): mode, speed, acceleration, smoothing, invert, clutch, direction calibration."""
from __future__ import annotations

import threading

from PyQt6 import QtCore, QtWidgets

from ..acquire import config
from ..actions.pointer import DEFAULTS

MODES = [("Right hand: PINCH to move, release to freeze (no training)", "right_pinch"),
         ("Right hand: move while a gesture bound to 'move mouse while held' is held", "right_gesture"),
         ("Off", "off"), ("Auto: IMU + camera", "auto"), ("IMU only", "imu"), ("Camera only", "camera"),
         ("Left hand (needs two-hand tracking, experimental)", "left")]


def load_settings():
    s = dict(DEFAULTS)
    saved = config.load().get("pointer") or {}
    if saved.get("version", 1) < 4:          # v4: right-hand pinch pointer is the default
        saved["mode"] = "right_pinch"
    s.update(saved)
    return s


class PointerPanel(QtWidgets.QGroupBox):
    calib_step = QtCore.pyqtSignal(object)          # result of one capture step (vector or Exception)

    def __init__(self, app):
        super().__init__("Pointer (cursor movement)")
        self.app = app
        s = load_settings()
        self.mode = QtWidgets.QComboBox()
        for label, key in MODES:
            self.mode.addItem(label, key)
        self.mode.setCurrentIndex([k for _, k in MODES].index(s["mode"]))
        self.speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=1, maximum=60, value=int(s["speed"]))
        self.speed_lbl = QtWidgets.QLabel()
        self.smooth = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=0, maximum=100,
                                        value=int(100 * s["smoothing"]))
        self.accel = QtWidgets.QCheckBox("Acceleration (faster moves go further, like 'enhance pointer precision')")
        self.accel.setChecked(s["accel"])
        self.need_hand = QtWidgets.QCheckBox("Only move while my hand is in view of the camera (drop the hand = lift the mouse)")
        self.need_hand.setChecked(s["need_hand"])
        self.freeze = QtWidgets.QCheckBox("Hold the cursor still while I clench (clicks land where I aimed)")
        self.freeze.setChecked(s["freeze_on_clench"])
        self.inv_x = QtWidgets.QCheckBox("Invert X")
        self.inv_x.setChecked(s["invert_x"])
        self.inv_y = QtWidgets.QCheckBox("Invert Y")
        self.inv_y.setChecked(s["invert_y"])
        self.cal_btn = QtWidgets.QPushButton("Calibrate directions…")
        self.cal_btn.clicked.connect(self.calibrate)
        self.src = QtWidgets.QLabel()
        form = QtWidgets.QFormLayout(self)
        form.addRow("Mode", self.mode)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.speed, 1)
        h.addWidget(self.speed_lbl)
        form.addRow("Speed", h)
        form.addRow("Smoothing", self.smooth)
        form.addRow(self.accel)
        form.addRow(self.need_hand)
        form.addRow(self.freeze)
        h2 = QtWidgets.QHBoxLayout()
        for w in (self.inv_x, self.inv_y, self.cal_btn):
            h2.addWidget(w)
        h2.addStretch(1)
        form.addRow(h2)
        form.addRow("Now", self.src)
        for w in (self.mode,):
            w.currentIndexChanged.connect(self.changed)
        for w in (self.speed, self.smooth):
            w.valueChanged.connect(self.changed)
        for w in (self.accel, self.need_hand, self.freeze, self.inv_x, self.inv_y):
            w.toggled.connect(self.changed)
        self.calib_step.connect(self._calib_next)
        self._axes = {k: s[k] for k in ("axis_x", "sign_x", "axis_y", "sign_y")}
        self.changed()

    def settings(self):
        return {"version": 4, "mode": self.mode.currentData(), "speed": float(self.speed.value()),
                "smoothing": self.smooth.value() / 100, "accel": self.accel.isChecked(),
                "need_hand": self.need_hand.isChecked(), "freeze_on_clench": self.freeze.isChecked(),
                "invert_x": self.inv_x.isChecked(), "invert_y": self.inv_y.isChecked(), **self._axes}

    def changed(self, *_):
        s = self.settings()
        self.speed_lbl.setText(f"{s['speed']:.0f}")
        self.app.pointer.update(**s)
        config.save(pointer=s)

    def refresh(self):
        p, ex = self.app.pointer, self.app.executor
        on = self.app.actions_enabled and not self.app.killed
        why = ("actions off (tick 'Enable actions')" if not on else
               "toggled off by a gesture" if not ex.pointer_on else p.source)
        self.src.setText(why)

    # ------------------------------------------------------------------ direction calibration
    STEPS = [("x", "Turn your hand to the RIGHT (rotate at the wrist, like steering right) and hold it there.\n"
                   "Click OK, then turn right within the next 2 seconds."),
             ("y", "Tilt your hand DOWN (fingers toward the floor) and hold it there.\n"
                   "Click OK, then tilt down within the next 2 seconds.")]

    def calibrate(self):
        if not self.app.sensors.snapshot()["status"].get("imu_ok"):
            QtWidgets.QMessageBox.information(self, "Calibrate directions", "Start the sensors with the IMU first (Live tab).")
            return
        self._step = 0
        self._result = {}
        self._ask()

    def _ask(self):
        which, text = self.STEPS[self._step]
        if QtWidgets.QMessageBox.information(self, f"Calibrate directions ({self._step + 1}/2)", text,
                                             QtWidgets.QMessageBox.StandardButton.Ok |
                                             QtWidgets.QMessageBox.StandardButton.Cancel) != \
                QtWidgets.QMessageBox.StandardButton.Ok:
            return
        self.cal_btn.setEnabled(False)
        self.cal_btn.setText("moving… (2 s)")

        def work():
            try:
                self.calib_step.emit(self.app.pointer.capture(2.0))
            except Exception as e:
                self.calib_step.emit(e)
        threading.Thread(target=work, daemon=True).start()

    def _calib_next(self, v):
        self.cal_btn.setEnabled(True)
        self.cal_btn.setText("Calibrate directions…")
        if isinstance(v, Exception):
            QtWidgets.QMessageBox.warning(self, "Calibrate directions", str(v))
            return
        ax, sign, purity = self.app.pointer.axis_from(v)
        if abs(v[ax]) < 15:
            QtWidgets.QMessageBox.warning(self, "Calibrate directions",
                                          f"Not enough rotation seen ({abs(v[ax]):.0f}°/s). Try a bigger, steadier turn.")
            return
        which = self.STEPS[self._step][0]
        self._result[which] = (ax, sign, purity)
        if self._step == 0:
            self._step = 1
            self._ask()
            return
        (ax_x, sx, px), (ax_y, sy, py) = self._result["x"], self._result["y"]
        if ax_x == ax_y:
            QtWidgets.QMessageBox.warning(self, "Calibrate directions",
                                          "Both moves used the same gyro axis; please try again with cleaner moves.")
            return
        self._axes = {"axis_x": ax_x, "sign_x": sx, "axis_y": ax_y, "sign_y": sy}
        self.inv_x.setChecked(False)
        self.inv_y.setChecked(False)
        self.changed()
        names = "xyz"
        QtWidgets.QMessageBox.information(
            self, "Calibrate directions",
            f"Saved. Right = gyro {'+' if sx > 0 else '-'}{names[ax_x]} ({100 * px:.0f}% clean), "
            f"down = gyro {'+' if sy > 0 else '-'}{names[ax_y]} ({100 * py:.0f}% clean).")
