"""Record tab: choose gestures and reps, run the full-screen recorder, show the session summary."""
from __future__ import annotations

import re
import threading

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
from ..record.plan import build_plan, total_seconds
from ..record.runner import Runner
from ..record.writer import StreamBuffer, write_session


class RecordTab(QtWidgets.QWidget):
    saved = QtCore.pyqtSignal(object, object)       # (path, summary) or (None, error)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.glist = QtWidgets.QListWidget()
        self.reps = QtWidgets.QSpinBox(minimum=1, maximum=100, value=20)
        self.rest = QtWidgets.QDoubleSpinBox(minimum=3, maximum=60, value=10, suffix=" s")
        self.free = QtWidgets.QDoubleSpinBox(minimum=0, maximum=600, value=150, suffix=" s")
        self.tag = QtWidgets.QLineEdit("session")
        self.enotes = QtWidgets.QLineEdit()
        self.enotes.setPlaceholderText("e.g. ch1 = flexors 3 cm below elbow, ch2 = extensors; ref on elbow bone")
        self.invalid = QtWidgets.QCheckBox("No EMG pads / EMG not usable (dry run)")
        self.est = QtWidgets.QLabel()
        for w in (self.reps, self.rest, self.free):
            w.valueChanged.connect(self._estimate)
        self.glist.itemChanged.connect(self._estimate)
        self.start_btn = QtWidgets.QPushButton("Start recording…")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start)
        form = QtWidgets.QFormLayout()
        form.addRow("Repetitions per gesture", self.reps)
        form.addRow("Rest calibration", self.rest)
        form.addRow("Free activity ('none') block", self.free)
        form.addRow("Tag", self.tag)
        form.addRow("Electrode notes", self.enotes)
        form.addRow("", self.invalid)
        form.addRow("", self.est)
        form.addRow("", self.start_btn)
        tips = QtWidgets.QLabel(
            "Tips: keep the free-activity block in every session (it teaches the model what NOT to react to); "
            "keep the rest calibration ≥ 10 s. After the pads go on, use Run → 'Calibrate rest' before live use.\n"
            "Sessions are saved to data/sessions/. Commit and push them so the backend computer gets them.")
        tips.setWordWrap(True)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Gestures to record"))
        left.addWidget(self.glist)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        right.addLayout(form)
        right.addWidget(tips)
        right.addStretch(1)
        lay.addLayout(right, 2)
        self.saved.connect(self._saved)
        self.win = None

    def reload_gestures(self):
        checked = {self.glist.item(i).data(256) for i in range(self.glist.count())
                   if self.glist.item(i).checkState() == QtCore.Qt.CheckState.Checked}
        self.glist.blockSignals(True)
        self.glist.clear()
        for g in G.load()["gestures"]:
            it = QtWidgets.QListWidgetItem(f"{g['name']}  ({g['id']}, {g['kind']})")
            it.setData(256, g["id"])
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if g["id"] in checked else QtCore.Qt.CheckState.Unchecked)
            self.glist.addItem(it)
        self.glist.blockSignals(False)
        self._estimate()

    def _chosen(self):
        d = G.load()
        ids = [self.glist.item(i).data(256) for i in range(self.glist.count())
               if self.glist.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        return d, [G.get(d, i) for i in ids if G.get(d, i)]

    def _estimate(self, *_):
        _, gl = self._chosen()
        if not gl:
            self.est.setText("Tick at least one gesture (add gestures in the Gestures tab).")
            return
        steps = build_plan(gl, self.reps.value(), self.rest.value(), self.free.value())
        self.est.setText(f"{len(gl) * self.reps.value()} trials, about {total_seconds(steps) / 60:.1f} min")

    def start(self):
        d, gl = self._chosen()
        tag = self.tag.text().strip()
        if not gl:
            QtWidgets.QMessageBox.information(self, "Record", "Tick at least one gesture.")
            return
        if not re.match(r"^[A-Za-z0-9_-]+$", tag):
            QtWidgets.QMessageBox.warning(self, "Record", "Tag: letters, digits, _ and - only.")
            return
        from ..record.ui import RecorderWindow
        self.runner = Runner(build_plan(gl, self.reps.value(), self.rest.value(), self.free.value()))
        self.buf = StreamBuffer()
        self.buf.start()
        self.snapshot = d
        need = ("imu", "hand") if self.invalid.isChecked() else ("emg", "imu", "hand")
        self.win = RecorderWindow(self.runner, self.buf, tag, need=need)
        self.win.finished.connect(lambda ok: self._finished(ok, tag))
        self.win.showMaximized()
        self.start_btn.setEnabled(False)

    def _finished(self, completed, tag):
        self.buf.end()
        self.win.close()
        self.win = None
        r = self.runner
        if not completed and not any(l[2] != "none" for l in r.labels):
            self.buf.stop()
            self.start_btn.setEnabled(True)
            self.app.log("recording cancelled; nothing saved")
            return
        tag = tag + ("" if completed else "_partial")
        args = (self.buf, r.labels, r.rest_periods, tag, self.snapshot, self.enotes.text(),
                "back of the right hand, palm down: +x fingertips, -y thumb, -z up")

        def work():
            try:
                path, summary = write_session(*args, extra_meta={"emg_valid": not self.invalid.isChecked()})
                self.saved.emit(path, summary)
            except Exception as e:
                self.saved.emit(None, str(e))
            finally:
                self.buf.stop()
        threading.Thread(target=work, daemon=True).start()

    def _saved(self, path, summary):
        self.start_btn.setEnabled(True)
        if path is None:
            QtWidgets.QMessageBox.critical(self, "Record", f"Saving failed: {summary}")
            return
        lines = [f"Saved {path.name}", "",
                 f"{summary['seconds']} s   EMG {summary['emg_rate']} Hz   IMU {summary['imu_rate']} Hz   "
                 f"camera {summary['cam_fps']} fps (hand {summary['hand_seen_pct']}%)",
                 f"dropped {summary['dropped']}   labels {summary['labels']}", ""]
        lines += [f"⚠ {w}" for w in summary["warnings"]] or ["All sensors OK."]
        lines += ["", "Commit and push data/sessions/ so the backend computer gets it."]
        (QtWidgets.QMessageBox.warning if summary["warnings"] else QtWidgets.QMessageBox.information)(
            self, "Session saved", "\n".join(lines))
        self.app.log(f"session saved: {path.name}")
        self.app.sessions_changed()
