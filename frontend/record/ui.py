"""Full-screen cue window for the recorder (PyQt6). The UI thread never does I/O: the stream
is buffered by StreamBuffer's thread, and the session is written after the run ends."""
from __future__ import annotations

import time

from PyQt6 import QtCore, QtGui, QtWidgets

from .runner import Runner
from .writer import StreamBuffer

COLORS = {"rest": ("#123a5c", "#e8f1ff"), "relax": ("#15191c", "#7d8b96"), "cue": ("#f2a516", "#101010"),
          "free": ("#1d5c34", "#e9ffe9"), "idle": ("#15191c", "#e5ebe6")}
SAFETY = ("SAFETY: electrodes on? The ESP32 must be powered from a battery, a power bank or an UNPLUGGED laptop "
          "- never a wall-powered PC.")


class RecorderWindow(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal(bool)        # True = completed, False = aborted

    def __init__(self, runner: Runner, buf: StreamBuffer, title: str, need=("emg", "imu", "hand")):
        super().__init__()
        self.runner, self.buf, self.need = runner, buf, need
        self.setWindowTitle(f"EMG Mouse recorder - {title}")
        self.big = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.big.setWordWrap(True)
        self.sub = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sub.setWordWrap(True)
        self.step_bar = QtWidgets.QProgressBar(textVisible=False, maximum=1000)
        self.step_bar.setFixedHeight(28)
        self.all_bar = QtWidgets.QProgressBar(maximum=1000)
        self.all_bar.setFixedHeight(22)
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("font: 11pt 'Consolas'; color: #c9d1d9;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 20)
        lay.addWidget(self.status)
        lay.addStretch(1)
        lay.addWidget(self.big)
        lay.addWidget(self.sub)
        lay.addStretch(1)
        lay.addWidget(self.step_bar)
        lay.addWidget(self.all_bar)
        self.started = False
        self.timer = QtCore.QTimer(self, interval=15)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self._set_colors("idle")
        self.resize(1200, 800)

    def _set_colors(self, kind):
        bg, fg = COLORS[kind]
        self.setStyleSheet(f"background: {bg}; color: {fg};")
        self.big.setStyleSheet(f"font: bold 54pt 'Segoe UI'; color: {fg};")
        self.sub.setStyleSheet(f"font: 20pt 'Segoe UI'; color: {fg};")
        self.status.setStyleSheet(f"font: 11pt 'Consolas'; color: {fg};")
        self.step_bar.setStyleSheet("QProgressBar {background: rgba(0,0,0,70); border: 0;} "
                                    "QProgressBar::chunk {background: rgba(255,255,255,200);}")
        self.all_bar.setStyleSheet(f"QProgressBar {{background: rgba(0,0,0,70); border: 0; color: {fg}; "
                                   "font: 10pt 'Segoe UI';} QProgressBar::chunk {background: #4ea1ff;}")

    def _sensor_line(self):
        live = self.buf.live()
        st = live["status"]

        def light(name, key):
            return f"{name} {'OK ' if live.get(key) else '-- '}"
        s = light("EMG", "emg") + light("IMU", "imu") + light("CAM", "hand")
        if st:
            s += f"  dropped {st.get('dropped', 0)}"
            if st.get("imu_ok") is False and live.get("imu"):
                s += "  (IMU values NaN)"
        return s, live

    def tick(self):
        now = time.perf_counter()
        line, live = self._sensor_line()
        if not self.started:
            missing = [k for k in self.need if not live.get(k)]
            r = self.runner
            self.big.setText("Ready" if not missing else "Waiting for sensors")
            self.sub.setText(
                (f"Missing: {', '.join(m.upper() for m in missing)}  (start `python -m frontend.acquire --port COM7`)\n\n"
                 if missing else "") +
                f"{r.n_trials} trials, about {r.total / 60:.1f} min including the rest and free-activity blocks.\n"
                "Do each gesture once when its cue appears. Hold gestures: hold until the bar ends.\n\n"
                f"{SAFETY}\n\nPress SPACE to start, ESC to quit.")
            self.status.setText(line)
            return
        step = self.runner.tick(now)
        if step is None:
            self.timer.stop()
            self.finished.emit(True)
            return
        self._set_colors(step.kind)
        if step.kind == "relax":
            self.big.setText("relax")
            self.sub.setText("")
        else:
            head, _, rest = step.text.partition("\n")
            self.big.setText(head)
            remain = step.seconds - (now - self.runner.t_step)
            self.sub.setText(rest + (f"\n{remain:.0f} s" if step.kind in ("rest", "free") else ""))
        self.step_bar.setValue(int(1000 * self.runner.step_fraction(now)))
        self.all_bar.setValue(int(1000 * self.runner.overall_fraction(now)))
        rem = self.runner.remaining(now)
        self.all_bar.setFormat(f"trial {self.runner.trial()}/{self.runner.n_trials}   "
                               f"{int(rem // 60)}:{int(rem % 60):02d} left")
        self.status.setText(line + "   ESC = stop")

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key.Key_Space and not self.started:
            self.started = True
            self.buf.begin()
            self.runner.start(time.perf_counter())
        elif e.key() == QtCore.Qt.Key.Key_Escape:
            if self.started:
                self.runner.abort(time.perf_counter())
            self.timer.stop()
            self.finished.emit(False)
