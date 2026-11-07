"""EMG Mouse desktop app (PyQt6 + pyqtgraph).

    python -m frontend.app                 start with no sensors running (pick a source in Live)
    python -m frontend.app --sim           start the simulator immediately

Works with or without the backend (`python -m backend.serve`); shows "backend offline" when absent.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from .. import gestures as G  # noqa: E402
from .links import BackendLink, SensorLink, Sources  # noqa: E402
from .tab_gestures import GesturesTab  # noqa: E402
from .tab_live import LiveTab  # noqa: E402
from .tab_record import RecordTab  # noqa: E402
from .tab_run import RunTab, decide  # noqa: E402
from .tab_train import TrainTab  # noqa: E402

DARK = """
QWidget { background: #161b22; color: #c9d1d9; font: 10pt 'Segoe UI'; }
QPushButton { background: #21262d; border: 1px solid #30363d; padding: 5px 12px; border-radius: 4px; }
QPushButton:hover { background: #30363d; }
QPushButton:disabled { color: #6e7681; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget {
    background: #0d1117; border: 1px solid #30363d; padding: 3px; }
QHeaderView::section { background: #21262d; border: 0; padding: 4px; }
QTabBar::tab { background: #21262d; padding: 8px 18px; margin-right: 2px; }
QTabBar::tab:selected { background: #2f81f7; color: white; }
QProgressBar { background: #0d1117; border: 1px solid #30363d; text-align: center; }
QProgressBar::chunk { background: #2f81f7; }
"""


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Mouse")
        self.actions_enabled = False         # off by default (prompt requirement)
        self.killed = False
        self.conf_threshold = 0.80
        self.gestures = G.load()["gestures"]
        self.sources = Sources(log=self.log)
        self.sensors = SensorLink()
        self.sensors.start()
        self.backend = BackendLink()
        self.backend.start()
        from ..actions.executor import Executor
        self.executor = Executor(self, log=self.log)
        self.backend.on_gesture = self.executor.handle      # runs in the link thread
        self.executor.start_hotkey(self.kill)              # global F9, any window
        from ..actions.pointer import Pointer
        from .pointer_panel import load_settings
        self.pointer = Pointer(lambda: self.actions_enabled and not self.killed and self.executor.pointer_on,
                               load_settings(), log=self.log)
        self.pointer.start()

        self.pending_model = None             # set by "train on everything": announce when it's loaded
        from .teach import TeachFlow
        self.teach = TeachFlow(self)
        self.tabs = QtWidgets.QTabWidget()
        self.live = LiveTab(self)
        self.gtab = GesturesTab(self)
        self.rec = RecordTab(self)
        self.train = TrainTab(self)
        self.run = RunTab(self)
        for w, name in [(self.live, "Live"), (self.gtab, "Gestures"), (self.rec, "Record"), (self.train, "Train"),
                        (self.run, "Run")]:
            self.tabs.addTab(w, name)
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("ready")
        self.gestures_changed()
        QtGui.QShortcut(QtGui.QKeySequence("F9"), self, activated=self.kill)

        self.timer = QtCore.QTimer(self, interval=33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self._last_be_poll = 0.0
        self.resize(1500, 950)

    # ------------------------------------------------------------------ app state
    def log(self, msg):
        print(f"[app] {msg}", flush=True)
        if hasattr(self, "statusBar"):
            QtCore.QMetaObject.invokeMethod(self.statusBar(), "showMessage",
                                            QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(str, str(msg)))

    def gestures_changed(self):
        self.gestures = G.load()["gestures"]
        if hasattr(self, "rec"):
            self.rec.reload_gestures()
            self.train.reload_gestures()

    def sessions_changed(self):
        self.train.reload_sessions()

    def set_actions_enabled(self, on):
        if on:
            self.killed = False
        self.actions_enabled = bool(on)
        if not on:
            self.executor.release_all()
        self.log("actions " + ("ENABLED" if on else "disabled"))

    def kill(self):
        self.killed = True
        self.actions_enabled = False
        self.executor.release_all()
        self.log("KILL SWITCH: all actions disabled")

    # ------------------------------------------------------------------ UI timer (no I/O here)
    def tick(self):
        snap = self.sensors.snapshot()
        self.live.refresh(snap)
        self.train.refresh()
        self.run.refresh()
        if self.pending_model and self.backend.online:
            st = self.backend.status
            if st.get("state") == "error":
                QtWidgets.QMessageBox.warning(self, "Training", f"Training failed: {st.get('message')}")
                self.pending_model = None
            elif st.get("state") != "training" and st.get("model") == self.pending_model:
                name, self.pending_model = self.pending_model, None
                self.train.refresh_models()
                self.statusBar().showMessage(f"model {name} trained and loaded")
                QtWidgets.QMessageBox.information(
                    self, "New model ready", f"'{name}' is trained and loaded ({st.get('message') or ''}).\n\n"
                    "Run tab: Calibrate rest, then Start inference. The Train tab's model list shows its accuracy.")
        events = self.backend.drain_events()
        if events:
            self.live.add_events(events)
            self.run.add_decisions([(e, e.get("_outcome") or decide(self, e, self.gestures)) for e in events])
        now = time.perf_counter()
        if not self.backend.online and self.executor.held:
            self.executor.release_all()            # never leave a button held if the backend vanishes
        if now - self._last_be_poll > 5 and not self.backend.online:
            self._last_be_poll = now           # nudge: a status request also detects a backend that just started
            self.backend.control({"cmd": "status"}, lambda r: None, timeout_s=1.0)

    def closeEvent(self, e):
        self.timer.stop()
        self.kill()
        self.executor.stop()
        self.pointer.stop()
        self.sources.stop()
        self.backend.stop()
        self.sensors.stop()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK)
    w = App()
    if "--sim" in sys.argv:
        w.live.src.setCurrentText("Simulator")
        w.live.toggle()
    w.show()
    if "--snapshot" in sys.argv:              # tests: save one PNG per tab and quit
        out = sys.argv[sys.argv.index("--snapshot") + 1]
        delay = 3000

        def shoot(i=0):
            if i >= w.tabs.count():
                app.quit()
                return
            w.tabs.setCurrentIndex(i)
            QtCore.QTimer.singleShot(1500, lambda: (w.grab().save(f"{out}/app_{w.tabs.tabText(i).lower()}.png"),
                                                    shoot(i + 1)))
        QtCore.QTimer.singleShot(delay, shoot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
