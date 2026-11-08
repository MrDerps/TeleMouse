"""Which sensors the model uses (INTERFACE `sensors` on train/start) and how much EMG may count
(`sensor_weights`, v0.4). Default: all three sensors with EMG at 20% weight. The backend then trains with
EMG dropped most of the time, so it can help but can't dominate (it shifts whenever the pads move)."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ..acquire import config

ALL = ["emg", "imu", "hand"]
LABEL = {"emg": "EMG", "imu": "IMU", "hand": "Camera"}
DEFAULT = ["emg", "imu", "hand"]
DEFAULT_EMG_WEIGHT = 0.2


def current() -> list[str]:
    cfg = config.load()
    if cfg.get("sensors_version", 1) < 2:          # v2: EMG back on by default (at low weight)
        return DEFAULT
    s = cfg.get("model_sensors") or DEFAULT
    return [x for x in ALL if x in s] or DEFAULT


def emg_weight() -> float:
    return float(config.load().get("emg_weight", DEFAULT_EMG_WEIGHT))


def weights(sensors=None) -> dict:
    """sensor_weights for a train request (only for the sensors in use)."""
    sensors = sensors or current()
    w = {"emg": emg_weight(), "imu": 1.0, "hand": 1.0}
    return {k: v for k, v in w.items() if k in sensors}


def key(sensors) -> str:
    return "+".join(x for x in ALL if x in sensors)


class SensorPicker(QtWidgets.QWidget):
    """Three checkboxes (+ optional EMG-influence slider) kept in sync across tabs via app.sensors_changed()."""

    def __init__(self, app, label="Model uses", with_weight=False):
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
        self.slider = self.wlabel = None
        if with_weight:
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=0, maximum=100,
                                            value=int(100 * emg_weight()))
            self.slider.setFixedWidth(120)
            self.slider.setToolTip("How much the model may rely on EMG (training). Low = IMU + camera do the work.")
            self.wlabel = QtWidgets.QLabel()
            self.slider.valueChanged.connect(self._weight)
            lay.addSpacing(12)
            lay.addWidget(QtWidgets.QLabel("EMG influence"))
            lay.addWidget(self.slider)
            lay.addWidget(self.wlabel)
            self._weight(self.slider.value(), save=False)
        lay.addStretch(1)
        self.set(current())

    def set(self, sensors):
        for s, b in self.boxes.items():
            b.blockSignals(True)
            b.setChecked(s in sensors)
            b.blockSignals(False)
        if self.slider:
            self.slider.setEnabled("emg" in sensors)

    def value(self):
        return [s for s in ALL if self.boxes[s].isChecked()]

    def _weight(self, v, save=True):
        self.wlabel.setText(f"{v}%")
        if save:
            config.save(emg_weight=v / 100)

    def _toggled(self, *_):
        v = self.value()
        if not v:                                   # at least one sensor
            self.set(current())
            return
        config.save(model_sensors=v, sensors_version=2)
        self.app.sensors_changed(v)
