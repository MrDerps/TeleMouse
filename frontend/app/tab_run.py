"""Run tab: rest calibration, start/stop inference, actions toggle (off by default), confidence threshold."""
from __future__ import annotations

import time

from PyQt6 import QtCore, QtWidgets

from ..actions import describe
from .widgets import ProbBars, light


class RunTab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.state = QtWidgets.QLabel()
        self.state.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.cal_btn = QtWidgets.QPushButton("Calibrate rest (relax 3 s)")
        self.cal_btn.clicked.connect(self.calibrate)
        self.start_btn = QtWidgets.QPushButton("Start inference")
        self.start_btn.clicked.connect(lambda: self.app.backend.control({"cmd": "start"}, self._reply))
        self.stop_btn = QtWidgets.QPushButton("Stop inference")
        self.stop_btn.clicked.connect(lambda: self.app.backend.control({"cmd": "stop"}, self._reply))
        self.enable = QtWidgets.QCheckBox("Enable actions (clicks AND cursor movement)")
        self.enable.setChecked(False)
        self.enable.toggled.connect(self.app.set_actions_enabled)
        self.thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=50, maximum=99, value=80)
        self.thr_lbl = QtWidgets.QLabel()
        self.thr.valueChanged.connect(self._thr)
        self._thr(self.thr.value())
        self.big = QtWidgets.QLabel("—", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.big.setStyleSheet("font: bold 36pt 'Segoe UI'; padding: 12px;")
        self.probs = ProbBars()
        self.log = QtWidgets.QListWidget()
        self.reply = QtWidgets.QLabel()
        kill = QtWidgets.QLabel("Kill switch: F9 disables all actions instantly (works from any window).")
        kill.setStyleSheet("color:#f0b429;")

        ctl = QtWidgets.QHBoxLayout()
        for w in (self.cal_btn, self.start_btn, self.stop_btn):
            ctl.addWidget(w)
        ctl.addStretch(1)
        act = QtWidgets.QHBoxLayout()
        act.addWidget(self.enable)
        act.addSpacing(30)
        act.addWidget(QtWidgets.QLabel("Min confidence"))
        act.addWidget(self.thr, 1)
        act.addWidget(self.thr_lbl)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.state)
        lay.addLayout(ctl)
        lay.addWidget(self.reply)
        lay.addLayout(act)
        lay.addWidget(kill)
        from .pointer_panel import PointerPanel
        self.pointer_panel = PointerPanel(app)
        lay.addWidget(self.pointer_panel)
        h = QtWidgets.QHBoxLayout()
        v = QtWidgets.QVBoxLayout()
        v.addWidget(self.big)
        v.addWidget(self.probs, 1)
        h.addLayout(v, 1)
        lv = QtWidgets.QVBoxLayout()
        lv.addWidget(QtWidgets.QLabel("Events (what was done with each)"))
        lv.addWidget(self.log)
        h.addLayout(lv, 1)
        lay.addLayout(h, 1)
        self._big_t = 0.0

    def _thr(self, v):
        self.thr_lbl.setText(f"{v / 100:.2f}")
        self.app.conf_threshold = v / 100

    def _reply(self, r):
        self.reply.setText("ok" if r.get("ok") else f"<span style='color:#f85149'>{r.get('error')}</span>")

    def calibrate(self):
        QtWidgets.QMessageBox.information(self, "Calibrate rest",
                                          "Relax your arm and hand completely. Click OK, then stay relaxed for 3 seconds.")
        self.app.backend.control({"cmd": "calibrate_rest", "seconds": 3}, self._reply)
        self.cal_btn.setEnabled(False)
        QtCore.QTimer.singleShot(3500, lambda: self.cal_btn.setEnabled(True))

    def refresh(self):
        be = self.app.backend
        st = be.status
        en = self.app.actions_enabled and not self.app.killed
        self.state.setText(
            light("backend " + (f"<b>{st.get('state')}</b>, model <b>{st.get('model')}</b>  {st.get('message') or ''}"
                                if be.online else "offline: run <code>python -m backend.serve</code>"),
                  True if be.online else None) + " &nbsp; " +
            light("actions " + ("ENABLED" if en else ("KILLED (F9)" if self.app.killed else "off")),
                  True if en else (False if self.app.killed else None)))
        if self.enable.isChecked() != self.app.actions_enabled:
            self.enable.blockSignals(True)
            self.enable.setChecked(self.app.actions_enabled)
            self.enable.blockSignals(False)
        pr = be.probs
        self.probs.set(*(pr[1:] if pr and be.online else ([], [])))
        self.pointer_panel.refresh()
        if time.perf_counter() - self._big_t > 1.5:
            self.big.setText("—")

    def add_decisions(self, decisions):
        for e, outcome in decisions:
            if e.get("kind") in ("instant", "start"):
                self.big.setText(f"{e.get('id')}  {e.get('conf', 0):.2f}")
                self._big_t = time.perf_counter()
            self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}  {e.get('id')} {e.get('kind')} "
                                   f"conf {e.get('conf', 0):.2f}  →  {outcome}")
        while self.log.count() > 200:
            self.log.takeItem(self.log.count() - 1)


def decide(app, e, gestures):
    """What to do with a gesture event (M5: decision only; M6 executes it)."""
    g = next((x for x in gestures if x["id"] == e.get("id")), None)
    a = (g or {}).get("action")
    if app.killed:
        return "ignored (kill switch)"
    if not app.actions_enabled:
        return f"ignored (actions off): {describe(a)}"
    if e.get("kind") != "end" and float(e.get("conf", 0)) < app.conf_threshold:
        return f"ignored (conf < {app.conf_threshold:.2f})"
    return describe(a)
