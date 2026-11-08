"""Axion desktop app (PyQt6).

    python -m frontend.app                 normal start
    python -m frontend.app --sim           start with the simulator connected

Works with or without the gesture engine (`python -m backend.serve`); shows "engine offline" when absent.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from .. import gestures as G  # noqa: E402
from ..acquire import config  # noqa: E402
from . import data  # noqa: E402
from .links import BackendLink, SensorLink, Sources  # noqa: E402
from .page_dashboard import DashboardPage  # noqa: E402
from .page_gestures import GesturesPage  # noqa: E402
from .page_overview import OverviewPage  # noqa: E402
from .page_train import TrainPage  # noqa: E402
from .theme import C, QSS  # noqa: E402
from .ui import Dot, Logo, Segmented, button, label  # noqa: E402

SAFETY_TEXT = ("Before streaming EMG with electrodes on a person:\n\n"
               "The ESP32 must be powered from a BATTERY, a POWER BANK or an UNPLUGGED laptop. Never from a "
               "wall-powered PC.\n\nIs the laptop unplugged from the wall (or are no electrodes attached)?")
PAGES = ["overview", "dashboard", "gestures", "train"]


class AxionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Axion")
        cfg = config.load()
        self.actions_enabled = False         # off by default (safety)
        self.killed = False
        self.conf_threshold = float(cfg.get("conf_threshold", 0.80))
        self.gestures = G.load()["gestures"]
        self.model_info = {}
        self.pending_model = None
        self.rates = {"emg": 0.0, "imu": 0.0, "hand": 0.0}
        self.probs_rate = 0.0
        self._rate_t, self._rate_c, self._probs_n, self._probs_last = time.perf_counter(), None, 0, None
        self.mode = "experience"
        self.settings = None

        self.sources = Sources(log=self.log)
        self.sensors = SensorLink()
        self.sensors.start()
        self.backend = BackendLink()
        self.backend.start()
        from ..actions.executor import Executor
        from ..actions.pointer import Pointer
        from .pointer_panel import load_settings
        self.executor = Executor(self, log=self.log)
        self.backend.on_gesture = self.executor.handle      # runs in the link thread, lowest latency
        self.executor.start_hotkey(self.kill)              # global F9
        self.pointer = Pointer(lambda: self.actions_enabled and not self.killed and self.executor.pointer_on,
                               load_settings(), log=self.log)
        self.pointer.start()
        self.executor.pointer = self.pointer
        from .teach import TeachFlow
        self.teach = TeachFlow(self)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._topbar())
        self.stack = QtWidgets.QStackedWidget()
        self.pages = {"overview": OverviewPage(self), "dashboard": DashboardPage(self),
                      "gestures": GesturesPage(self), "train": TrainPage(self)}
        for k in PAGES:
            self.stack.addWidget(self.pages[k])
        v.addWidget(self.stack, 1)
        v.addWidget(self._statusbar())
        self.setCentralWidget(central)
        self.gestures_changed()
        self.sessions_changed()
        self.go("dashboard" if self.gestures else "overview")
        QtGui.QShortcut(QtGui.QKeySequence("F9"), self, activated=self.kill)

        self.timer = QtCore.QTimer(self, interval=33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self._last_poll = 0.0
        self.resize(1440, 900)

    # ------------------------------------------------------------------ chrome
    def _topbar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("TopBar")
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(52)
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 16, 0)
        h.setSpacing(6)
        logo = Logo(26)
        logo.clicked.connect(lambda: self.go("overview"))
        h.addWidget(logo)
        h.addSpacing(26)
        self.tabs = {}
        for key, text in [("dashboard", "Dashboard"), ("gestures", "Gestures"), ("train", "Train")]:
            b = QtWidgets.QPushButton(text)
            b.setObjectName("Tab")
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.go(k))
            h.addWidget(b)
            self.tabs[key] = b
        h.addStretch(1)
        self.live_dot = Dot(C["disabled"])
        self.live_lbl = label("PAUSED", 11, 600, C["label"], tracking=0.08)
        h.addWidget(self.live_dot)
        h.addWidget(self.live_lbl)
        h.addSpacing(16)
        self.seg = Segmented(["Experience", "Data"])
        self.seg.changed.connect(lambda i: self.set_mode("experience" if i == 0 else "data"))
        h.addWidget(self.seg)
        h.addSpacing(8)
        gear = button("⚙", None, lambda: self.open_settings(), "Settings")
        gear.setFixedWidth(40)
        h.addWidget(gear)
        return bar

    def _statusbar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("StatusBar")
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(28)
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)
        self.status_lbl = label("", 12, 400, C["label"])
        self.msg_lbl = label("", 12, 400, C["text2"])
        h.addWidget(self.status_lbl)
        h.addStretch(1)
        h.addWidget(self.msg_lbl)
        return bar

    def go(self, page):
        self.stack.setCurrentWidget(self.pages[page])
        for k, b in self.tabs.items():
            b.setChecked(k == page)
        if page in ("gestures", "train"):
            self.pages[page].reload()

    def set_mode(self, mode):
        self.mode = mode
        for p in self.pages.values():
            p.set_mode(mode)

    def log(self, msg):
        print(f"[axion] {msg}", flush=True)
        if hasattr(self, "msg_lbl"):
            QtCore.QMetaObject.invokeMethod(self.msg_lbl, "setText", QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(str, str(msg)))

    # ------------------------------------------------------------------ data changes
    @property
    def electrode_notes(self):
        return config.load().get("electrode_notes", "")

    def gestures_changed(self):
        self.gestures = G.load()["gestures"]
        if hasattr(self, "pages"):
            self.pages["dashboard"].reload_gestures(self.gestures)
            self.pages["gestures"].reload()

    def sessions_changed(self):
        if hasattr(self, "pages"):
            self.pages["gestures"].reload()
            self.pages["train"].reload()

    def sensors_changed(self, sensors):
        self.pages["dashboard"].sync_sensor_switches(sensors)
        if self.settings is not None and hasattr(self.settings, "picker"):
            self.settings.picker.set(sensors)
        if (self.backend.status or {}).get("state") == "running":
            self.start_inference()                    # restart with the new sensor set
        self.pages["gestures"].reload()

    def refresh_models(self):
        def got(r):
            # model.json on disk is complete (per-gesture accuracy); list_models fills in anything not on disk
            info = {m["name"]: m for m in data.list_models()}
            for m in (r.get("models", []) if r.get("ok") else []):
                info.setdefault(m.get("name"), m)
            self.model_info = info
            self.pages["train"].reload()
            self.pages["gestures"].reload()
        self.backend.control({"cmd": "list_models"}, got)

    # ------------------------------------------------------------------ engine control
    def start_inference(self, callback=None):
        from .sensor_picker import current

        def done(r):
            self.log("recognising" if r.get("ok") else f"start failed: {r.get('error')}")
            if callback:
                callback(r)
        self.backend.control({"cmd": "start", "sensors": current()}, done, timeout_s=5.0)

    def stop_inference(self):
        self.executor.release_all()
        self.backend.control({"cmd": "stop"}, lambda r: self.log("paused"))

    def calibrate_rest(self):
        QtWidgets.QMessageBox.information(self, "Calibrate", "Relax your arm and hand completely. Click OK, then stay "
                                                             "relaxed for 3 seconds.")
        self.backend.control({"cmd": "calibrate_rest", "seconds": 3},
                             lambda r: self.log("calibrating resting level (3 s)…" if r.get("ok") else
                                                f"calibrate failed: {r.get('error')}"))

    def use_model(self, name):
        def loaded(r):
            if r.get("ok"):
                self.log(f"using {name}")
                self.start_inference(lambda _: self.refresh_models())
            else:
                QtWidgets.QMessageBox.warning(self, "Use model", f"Couldn't load {name}: {r.get('error')}")
        self.backend.control({"cmd": "load", "name": name}, loaded, timeout_s=10.0)

    def retrain(self, reason=""):
        from .teach import train_everything
        if not self.backend.online:
            QtWidgets.QMessageBox.information(self, "Retrain", "The gesture engine is offline. Start it with:\n\n"
                                                               "python -m backend.serve")
            return
        ok, info = train_everything(self, on_reply=lambda r: None if r.get("ok") else
                                    self.log(f"train failed: {r.get('error')}"))
        self.log(info)
        if not ok and reason == "manual":
            QtWidgets.QMessageBox.information(self, "Retrain", info)

    def teach_new(self, parent):
        self.teach.start(parent)

    def teach_more(self, parent, gid):
        self.teach.start(parent, gid)

    # ------------------------------------------------------------------ sensors
    def connect_sensors(self, parent=None):
        cfg = config.load()
        kind = cfg.get("source") or "ESP32 + camera"
        try:
            if kind.startswith("ESP32"):
                if QtWidgets.QMessageBox.warning(parent or self, "Safety: battery power", SAFETY_TEXT,
                                                 QtWidgets.QMessageBox.StandardButton.Yes |
                                                 QtWidgets.QMessageBox.StandardButton.Cancel) != \
                        QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                port = cfg.get("serial_port")
                if not port:
                    self.open_settings("Connection")
                    return
                self.sources.start_hardware(port, int(cfg.get("camera_index", 0)), use_camera=(kind == "ESP32 + camera"))
            elif kind == "Camera only":
                self.sources.start_camera(int(cfg.get("camera_index", 0)))
            elif kind == "Simulator":
                self.sources.start_sim()
            else:
                self.log("listening for an external publisher on 5555")
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent or self, "Couldn't connect",
                                           f"{e}\n\nCheck the port and source in Settings → Connection.")
            return
        self.log(f"sensors connected: {kind}")

    def disconnect_sensors(self):
        self.sources.stop()
        self.log("sensors disconnected")

    # ------------------------------------------------------------------ settings / wipe
    def open_settings(self, page="Connection"):
        from .settings_dialog import SettingsDialog
        self.settings = SettingsDialog(self, page)
        self.settings.show()

    def settings_closed(self):
        self.pages["dashboard"].load_pointer()
        from .pointer_panel import load_settings
        self.pointer.update(**load_settings())
        self.settings = None

    def wipe_everything(self):
        self.stop_inference()
        data.wipe_all()
        self.model_info = {}
        self.gestures_changed()
        self.sessions_changed()
        self.log("all recordings, models and gestures deleted")

    # ------------------------------------------------------------------ safety
    def set_actions_enabled(self, on):
        if on:
            self.killed = False
        self.actions_enabled = bool(on)
        if not on:
            self.executor.release_all()
        self.log("controlling the computer" if on else "not controlling the computer")

    def kill(self):
        self.killed = True
        self.actions_enabled = False
        self.executor.release_all()
        self.log("F9: all actions stopped")

    # ------------------------------------------------------------------ UI timer (no I/O here)
    def tick(self):
        now = time.perf_counter()
        snap = self.sensors.snapshot()
        if self._rate_c is None or now - self._rate_t >= 1.0:
            if self._rate_c is not None:
                dt = now - self._rate_t
                self.rates = {k: (snap["counts"][k] - self._rate_c[k]) / dt for k in self.rates}
                self.probs_rate = self._probs_n / dt
            self._rate_t, self._rate_c, self._probs_n = now, snap["counts"], 0
        pr = self.backend.probs
        if pr is not None and pr[0] != self._probs_last:
            self._probs_n += 1
            self._probs_last = pr[0]
        self.stack.currentWidget().refresh(snap)
        be = self.backend
        st = be.status if be.online else {}
        running = st.get("state") == "running"
        self.live_dot.set(C["accent"] if running else (C["warn"] if st.get("state") == "training" else C["disabled"]))
        self.live_lbl.setText("LIVE" if running else ("TRAINING" if st.get("state") == "training" else
                                                      ("PAUSED" if be.online else "OFFLINE")))
        live = snap["status"] if now - snap["last_msg"] < 1.5 else {}
        self.status_lbl.setText(
            f"EMG {self.rates['emg']:.0f} Hz · IMU {self.rates['imu']:.0f} Hz · Camera {self.rates['hand']:.0f} fps"
            f"   ·   model {data.model_label(st.get('model')) if st.get('model') else '–'}   ·   probs {self.probs_rate:.0f} Hz"
            + ("" if live else "   ·   sensors not connected") + ("" if be.online else "   ·   engine offline"))

        if self.pending_model and be.online:
            if st.get("state") == "error":
                QtWidgets.QMessageBox.warning(self, "Training", f"Training failed: {st.get('message')}")
                self.pending_model = None
            elif st.get("state") != "training" and st.get("model") == self.pending_model:
                name, self.pending_model = self.pending_model, None
                self.start_inference(lambda r: self.refresh_models())
                self.log(f"new model {name} is running")
                QtWidgets.QMessageBox.information(
                    self, "New model ready", "Your gestures are trained and being recognised now.\n\n"
                    "Turn on 'Control my computer' on the dashboard to let them act.")
        events = be.drain_events()
        if events:
            self.pages["dashboard"].add_events([(e, e.get("_outcome") or "") for e in events])
        if not be.online and self.executor.held:
            self.executor.release_all()            # never leave a button held if the engine vanishes
        if be.online and not self.model_info and now - self._last_poll > 5:
            self._last_poll = now
            self.refresh_models()
        elif not be.online and now - self._last_poll > 5:
            self._last_poll = now
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
    from .theme import font
    app.setFont(font(14))                      # per-widget fonts (titles, numbers) override this
    app.setStyleSheet(QSS)
    w = AxionApp()
    if "--sim" in sys.argv:
        w.sources.start_sim()
    w.show()
    if "--snapshot" in sys.argv:              # tests: save PNGs of each page in both modes and quit
        out = sys.argv[sys.argv.index("--snapshot") + 1]
        shots = [(p, m) for p in PAGES for m in ("experience", "data")]

        def shoot(i=0):
            if i >= len(shots):
                app.quit()
                return
            p, m = shots[i]
            w.go(p)
            w.seg.group.button(0 if m == "experience" else 1).setChecked(True)
            w.set_mode(m)
            QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(f"{out}/axion_{p}_{m}.png"), shoot(i + 1)))
        QtCore.QTimer.singleShot(3500, shoot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
