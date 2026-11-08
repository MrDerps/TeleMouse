"""Which sensors the model uses (INTERFACE v0.3 `sensors` on train/start). Default: IMU + camera, because they
transfer between sessions much better than EMG (which shifts every time the pads move). EMG is still recorded."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ..acquire import config

ALL = ["emg", "imu", "hand"]
LABEL = {"emg": "EMG", "imu": "IMU", "hand": "Camera"}
DEFAULT = ["imu", "hand"]


def current() -> list[str]:
    s = config.load().get("model_sensors") or DEFAULT
    return [x for x in ALL if x in s] or DEFAULT


def key(sensors) -> str:
    return "+".join(x for x in ALL if x in sensors)


class SensorPicker(QtWidgets.QWidget):
    """Three checkboxes kept in sync across tabs through app.sensors_changed()."""
    changed = QtCore.pyqtSignal(list)

    def __init__(self, app, label="Model uses"):
        super().__init__()
        self.app = app
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(label))
        self.boxes = {}
        for s in ALL:
            b = QtWidgets.QCheckBox(LABEL[s])
            b.toggled.connect(self._toggled)
            self.boxes[s] = b
            lay.addWidget(b)
        lay.addStretch(1)
        self.set(current())

    def set(self, sensors):
        for s, b in self.boxes.items():
            b.blockSignals(True)
            b.setChecked(s in sensors)
            b.blockSignals(False)

    def value(self):
        return [s for s in ALL if self.boxes[s].isChecked()]

    def _toggled(self, *_):
        v = self.value()
        if not v:                                   # at least one sensor
            self.set(current())
            return
        config.save(model_sensors=v)
        self.app.sensors_changed(v)
