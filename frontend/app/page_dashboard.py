"""Dashboard: Experience mode (plain language) and Data mode (signals, metrics, accuracy table, log)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from ..actions import describe
from ..actions.pointer import DEFAULTS as POINTER_DEFAULTS
from . import sensor_picker
from .theme import C, SENSOR_COLOR, font
from .ui import ColorBar, Metric, Panel, Switch, button, divider, label, tag
from .widgets import CameraView, ProbBars

SENSORS = [("emg", "EMG", "Forearm muscle signal", "2 channels · 1000 Hz"),
           ("imu", "IMU", "Motion on the back of the hand", "accel + gyro · 200 Hz"),
           ("hand", "Camera", "Webcam hand tracking", "21 landmarks · 30 fps")]
POINTER_MODES = [("Pinch to move (right hand)", "right_pinch"), ("Move while a gesture is held", "right_gesture"),
                 ("Off", "off")]


def pct(x):
    return f"{x:.0%}" if isinstance(x, (int, float)) and x == x else "–"


class DashboardPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._experience())
        self.stack.addWidget(self._data())
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 16)
        lay.setSpacing(14)
        head = QtWidgets.QHBoxLayout()
        tcol = QtWidgets.QVBoxLayout()
        tcol.setSpacing(2)
        tcol.addWidget(label("Dashboard", 22, 600))
        self.subtitle = label("", 13, 400, C["text2"])
        tcol.addWidget(self.subtitle)
        head.addLayout(tcol)
        head.addStretch(1)
        self.connect_btn = button("Connect sensors", "Primary", self._toggle_sensors)
        head.addWidget(self.connect_btn)
        head.addSpacing(10)
        self.run_btn = button("Start recognising", None, self._toggle_run)
        head.addWidget(self.run_btn)
        head.addSpacing(18)
        head.addWidget(label("Control my computer", 14, 500))
        self.control = Switch(False)
        self.control.toggled.connect(self._control)
        head.addWidget(self.control)
        lay.addLayout(head)
        lay.addWidget(self.stack, 1)
        self._last_event = None
        self._rates_t, self._rates_c = time.perf_counter(), None

    # ================================================================== Experience
    def _experience(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(14)

        # 01 last event
        p1 = Panel("Last event", 1)
        self.ev_name = label("Waiting for a gesture", 36, 600)
        self.ev_action = label("", 14, 400, C["text2"])
        p1.add(self.ev_name)
        p1.add(self.ev_action)
        row = QtWidgets.QHBoxLayout()
        self.ev_conf = self._readout(row, "Confidence")
        self.ev_lat = self._readout(row, "Response")
        self.ev_ago = self._readout(row, "When")
        row.addStretch(1)
        p1.add(row)
        p1.add(divider())
        br = QtWidgets.QHBoxLayout()
        self.pause_btn = button("Pause", None, self._toggle_run)
        br.addWidget(self.pause_btn)
        br.addWidget(button("Calibrate", None, self.app.calibrate_rest,
                            "Relax your hand for 3 seconds so the muscle signal has a fresh baseline"))
        br.addStretch(1)
        self.ev_hint = label("F9 stops everything instantly, from any window.", 12, 400, C["label"])
        br.addWidget(self.ev_hint)
        p1.add(br)
        g.addWidget(p1, 0, 0, 1, 2)

        # 02 sensors
        p2 = Panel("Sensors", 2)
        self.srow = {}
        for key, name, desc, spec in SENSORS:
            r = QtWidgets.QHBoxLayout()
            r.setSpacing(12)
            r.addWidget(ColorBar(SENSOR_COLOR[key], 38))
            c = QtWidgets.QVBoxLayout()
            c.setSpacing(0)
            top = QtWidgets.QHBoxLayout()
            top.addWidget(label(name, 15, 600))
            st = label("offline", 12, 400, C["label"])
            top.addWidget(st)
            top.addStretch(1)
            c.addLayout(top)
            sp = label(f"{desc} · {spec}", 12, 400, C["text2"])
            c.addWidget(sp)
            if key == "emg":
                c.addWidget(label("Battery power only while electrodes are on.", 11, 400, C["warn"]))
            r.addLayout(c, 1)
            sw = Switch(key in sensor_picker.current())
            sw.setToolTip("Let the gesture model use this sensor")
            sw.toggled.connect(lambda on, k=key: self._sensor_toggled(k, on))
            r.addWidget(sw)
            p2.add(r)
            self.srow[key] = (st, sp, sw)
        p2.add(label("Switches choose which sensors the gesture model listens to.", 12, 400, C["label"], wrap=True))
        g.addWidget(p2, 0, 2, 1, 1)

        # 03 accuracy
        p3 = Panel("Accuracy", 3)
        self.acc_big = label("–", 40, 600)
        p3.add(self.acc_big)
        p3.add(label("of gestures recognised", 13, 400, C["text2"]))
        self.acc_fp = label("", 14, 500)
        p3.add(self.acc_fp)
        self.acc_with = label("", 12, 400, C["label"], wrap=True)
        p3.add(self.acc_with)
        p3.lay.addStretch(1)
        g.addWidget(p3, 1, 0)

        # 04 gestures -> actions
        p4 = Panel("Gestures → actions", 4, right=button("Edit", "Ghost", lambda: self.app.go("gestures")))
        self.glist = QtWidgets.QVBoxLayout()
        self.glist.setSpacing(4)
        p4.add(self.glist)
        p4.lay.addStretch(1)
        g.addWidget(p4, 1, 1)
        self._grows = {}

        # 05 pointer
        p5 = Panel("Mouse pointer", 5, right=button("More", "Ghost", lambda: self.app.open_settings("Pointer")))
        self.pmode = QtWidgets.QComboBox()
        for t, k in POINTER_MODES:
            self.pmode.addItem(t, k)
        self.pspeed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=1, maximum=60)
        self.pnow = label("", 12, 400, C["text2"], wrap=True)
        p5.add(self.pmode)
        sr = QtWidgets.QHBoxLayout()
        sr.addWidget(label("Speed", 13, 400, C["text2"]))
        sr.addWidget(self.pspeed, 1)
        p5.add(sr)
        p5.add(self.pnow)
        p5.lay.addStretch(1)
        g.addWidget(p5, 1, 2)
        self.pmode.currentIndexChanged.connect(self._pointer_changed)
        self.pspeed.valueChanged.connect(self._pointer_changed)
        self.load_pointer()

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 1)
        g.setRowStretch(1, 1)
        return w

    def _readout(self, row, title):
        c = QtWidgets.QVBoxLayout()
        c.setSpacing(0)
        c.addWidget(label(title.upper(), 11, 500, C["label"], tracking=0.08))
        v = label("–", 18, 500)
        c.addWidget(v)
        row.addLayout(c)
        row.addSpacing(28)
        return v

    # ================================================================== Data
    def _data(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        strip = QtWidgets.QHBoxLayout()
        strip.setSpacing(14)
        self.m_resp = Metric("Response time", "ms")
        self.m_age = Metric("Hand frame age", "ms")
        self.m_rate = Metric("Model rate", "Hz")
        self.m_cam = Metric("Camera delay", "ms")
        for m in (self.m_resp, self.m_age, self.m_rate, self.m_cam):
            strip.addWidget(m)
        lay.addLayout(strip)
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(14)
        traces = Panel("Live signals", 1)
        pg.setConfigOptions(antialias=False, background=C["panel"], foreground=C["label"])
        glw = pg.GraphicsLayoutWidget()
        self.t_emg = np.arange(-5000, 0) / 1000
        self.t_imu = np.arange(-1000, 0) / 200
        p1 = glw.addPlot(row=0, col=0, title="EMG ch1 / ch2 (display filter)")
        p2 = glw.addPlot(row=1, col=0, title="Accelerometer (g)")
        p3 = glw.addPlot(row=2, col=0, title="Gyroscope (°/s)")
        for p in (p2, p3):
            p.setXLink(p1)
        self.c_emg = [p1.plot(pen=pg.mkPen(C["emg"])), p1.plot(pen=pg.mkPen("#3E7FB0"))]
        cols = [C["imu"], "#C7C8FA", "#6C6DD1"]
        self.c_acc = [p2.plot(pen=pg.mkPen(c)) for c in cols]
        self.c_gyr = [p3.plot(pen=pg.mkPen(c)) for c in cols]
        traces.add(glw, 1)
        mid.addWidget(traces, 3)
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(14)
        cam = Panel("Camera", 2)
        self.camview = CameraView()
        cam.add(self.camview, 1)
        right.addWidget(cam, 3)
        out = Panel("Model output", 3)
        self.bars = ProbBars()
        out.add(self.bars, 1)
        right.addWidget(out, 2)
        mid.addLayout(right, 2)
        lay.addLayout(mid, 3)
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(14)
        acc = Panel("Accuracy by sensors (validation)", 4)
        self.acc_table = QtWidgets.QTableWidget(0, 5)
        self.acc_table.setHorizontalHeaderLabels(["sensors", "recognised", "precision", "false / min", "response"])
        self.acc_table.horizontalHeader().setStretchLastSection(True)
        self.acc_table.verticalHeader().setVisible(False)
        self.acc_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.acc_table.setShowGrid(False)
        acc.add(self.acc_table, 1)
        bottom.addWidget(acc, 1)
        log = Panel("Event log", 5)
        self.log = QtWidgets.QListWidget()
        log.add(self.log, 1)
        bottom.addWidget(log, 1)
        lay.addLayout(bottom, 2)
        self._acc_key = None
        return w

    # ================================================================== actions
    def set_mode(self, mode):
        self.stack.setCurrentIndex(0 if mode == "experience" else 1)

    def _toggle_sensors(self):
        if self.app.sources.mode:
            self.app.disconnect_sensors()
        else:
            self.app.connect_sensors(self)

    def _toggle_run(self):
        st = self.app.backend.status if self.app.backend.online else {}
        if st.get("state") == "running":
            self.app.stop_inference()
        else:
            self.app.start_inference()

    def _control(self, on):
        self.app.set_actions_enabled(on)

    def _sensor_toggled(self, key, on):
        cur = set(sensor_picker.current())
        cur = cur | {key} if on else cur - {key}
        if not cur:
            self.srow[key][2].setChecked(True)
            return
        v = [s for s in sensor_picker.ALL if s in cur]
        from ..acquire import config
        config.save(model_sensors=v, sensors_version=2)
        self.app.sensors_changed(v)

    def sync_sensor_switches(self, sensors):
        for k, (_, _, sw) in self.srow.items():
            sw.blockSignals(True)
            sw.setChecked(k in sensors)
            sw.blockSignals(False)

    def load_pointer(self):
        from .pointer_panel import load_settings
        s = load_settings()
        self.pmode.blockSignals(True)
        self.pspeed.blockSignals(True)
        keys = [k for _, k in POINTER_MODES]
        self.pmode.setCurrentIndex(keys.index(s["mode"]) if s["mode"] in keys else keys.index("off"))
        self.pspeed.setValue(int(s.get("speed", POINTER_DEFAULTS["speed"])))
        self.pmode.blockSignals(False)
        self.pspeed.blockSignals(False)

    def _pointer_changed(self, *_):
        from ..acquire import config
        from .pointer_panel import load_settings
        s = load_settings()
        s.update(mode=self.pmode.currentData(), speed=float(self.pspeed.value()), version=4)
        config.save(pointer=s)
        self.app.pointer.update(**s)

    # ================================================================== refresh (UI timer)
    def refresh(self, snap):
        app, be = self.app, self.app.backend
        st = be.status if be.online else {}
        running = st.get("state") == "running"
        now = time.perf_counter()
        sensors_live = snap["status"] if now - snap["last_msg"] < 1.5 else {}
        rates = app.rates
        connected = bool(app.sources.mode) or bool(sensors_live)
        self.connect_btn.setText("Disconnect" if app.sources.mode else "Connect sensors")
        self.connect_btn.setObjectName("" if app.sources.mode else "Primary")
        self.connect_btn.style().polish(self.connect_btn)
        self.run_btn.setText("Pause recognising" if running else "Start recognising")
        self.run_btn.setEnabled(be.online and bool(st.get("model")))
        self.pause_btn.setText("Pause" if running else "Resume")
        self.pause_btn.setEnabled(be.online and bool(st.get("model")))
        if self.control.isChecked() != app.actions_enabled:
            self.control.blockSignals(True)
            self.control.setChecked(app.actions_enabled)
            self.control.blockSignals(False)
        model = st.get("model")
        parts = ["sensors " + ("connected" if connected else "not connected")]
        from . import data as _data
        parts.append(f"model: {_data.model_label(model)}" if model else
                     ("no model yet: teach a gesture" if be.online else "engine offline"))
        parts.append("recognising" if running else "paused")
        parts.append("controlling the computer" if app.actions_enabled and not app.killed else
                     ("STOPPED with F9" if app.killed else "not controlling the computer"))
        self.subtitle.setText("  ·  ".join(parts))

        if self.stack.currentIndex() == 0:
            self._refresh_experience(sensors_live, rates, st)
        else:
            self._refresh_data(snap, st, rates)

    def _refresh_experience(self, live, rates, st):
        ok = {"emg": live.get("emg_ok"), "imu": live.get("imu_ok"), "hand": live.get("cam_ok")}
        rate_txt = {"emg": f"{rates['emg']:.0f} Hz", "imu": f"{rates['imu']:.0f} Hz", "hand": f"{rates['hand']:.0f} fps"}
        for k, (stl, _, _) in self.srow.items():
            if ok[k]:
                stl.setText(f"● live · {rate_txt[k]}")
                stl.setStyleSheet(f"color: {C['ok']};")
            else:
                stl.setText("offline")
                stl.setStyleSheet(f"color: {C['label']};")
        # accuracy
        m = self.app.model_info.get(st.get("model")) if st.get("model") else None
        sens = sensor_picker.current()
        if m:
            bs = (m.get("by_sensors") or {}).get(sensor_picker.key(sens)) or m.get("val_metrics") or {}
            self.acc_big.setText(pct(bs.get("recall")))
            fp = bs.get("fp_per_min")
            self.acc_fp.setText(f"{fp:.1f} false triggers per minute" if isinstance(fp, (int, float)) and fp == fp else "")
            self.acc_with.setText("with " + " + ".join(sensor_picker.LABEL[s] for s in sens) +
                                  " · measured on held-out reps of your recordings")
        else:
            self.acc_big.setText("–")
            self.acc_fp.setText("")
            self.acc_with.setText("Teach a gesture to see how well it's recognised.")
        # last event + highlight
        ev = self._last_event
        if ev:
            ago = time.perf_counter() - ev["rx"]
            self.ev_ago.setText("just now" if ago < 2 else f"{ago:.0f} s ago" if ago < 120 else f"{ago / 60:.0f} min ago")
        hl = ev["id"] if ev and time.perf_counter() - ev["rx"] < 1.5 else None
        for gid, row in self._grows.items():
            row.setStyleSheet(f"background: {C['tint']}; border-radius: 6px;" if gid == hl else "")
        on = self.app.actions_enabled and not self.app.killed
        self.pnow.setText(f"Now: {self.app.pointer.source}" if on else
                          "Works when 'Control my computer' is on.")

    def _refresh_data(self, snap, st, rates):
        lag = st.get("lag_ms")
        self.m_resp.set(f"{lag:.0f}" if isinstance(lag, (int, float)) else "–")
        hl = st.get("hand_lag_ms")
        self.m_age.set(f"{hl:.0f}" if isinstance(hl, (int, float)) else "–")
        self.m_rate.set(f"{self.app.probs_rate:.0f}")
        cl = snap["status"].get("cam_latency")
        self.m_cam.set(f"{1000 * cl:.0f}" if cl else "–")
        for i in range(2):
            self.c_emg[i].setData(self.t_emg, snap["emg"][:, i] + (0 if i == 0 else 0))
        for i in range(3):
            self.c_acc[i].setData(self.t_imu, snap["imu"][:, i])
            self.c_gyr[i].setData(self.t_imu, snap["imu"][:, 3 + i])
        cam = self.app.sources.camera
        if cam is not None and cam.preview is not None:
            g, _ = self.app.sources.acq.roles
            self.camview.show_frame(cam.preview[0], [(g["lm"] if g else None, (195, 214, 111), "")])
        else:
            self.camview.show_landmarks(snap["hand"])
        pr = self.app.backend.probs
        self.bars.set(*(pr[1:] if pr and self.app.backend.online else ([], [])))
        m = self.app.model_info.get(st.get("model")) if st.get("model") else None
        key = (st.get("model"), sensor_picker.key(sensor_picker.current()))
        if key != self._acc_key:
            self._acc_key = key
            self.acc_table.setRowCount(0)
            for combo, d in ((m or {}).get("by_sensors") or {}).items():
                r = self.acc_table.rowCount()
                self.acc_table.insertRow(r)
                lat = d.get("latency_ms")
                vals = [combo.replace("hand", "camera").replace("+", " + "), pct(d.get("recall")), pct(d.get("precision")),
                        f"{d.get('fp_per_min', 0):.1f}" if d.get("fp_per_min") == d.get("fp_per_min") else "–",
                        f"{lat:.0f} ms" if isinstance(lat, (int, float)) and lat == lat else "–"]
                for c, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if combo == key[1]:
                        it.setBackground(pg.mkColor(C["tint"]))
                    self.acc_table.setItem(r, c, it)
            self.acc_table.resizeColumnsToContents()

    def reload_gestures(self, gestures):
        while self.glist.count():
            it = self.glist.takeAt(0)
            if it.widget():
                it.widget().setParent(None)
        self._grows = {}
        if not gestures:
            self.glist.addWidget(label("No gestures yet.", 14, 400, C["text2"]))
            self.glist.addWidget(button("Teach your first gesture", "Primary", lambda: self.app.teach_new(self)))
            return
        for g in gestures:
            row = QtWidgets.QWidget()
            r = QtWidgets.QHBoxLayout(row)
            r.setContentsMargins(8, 6, 8, 6)
            r.addWidget(label(g.get("name") or g["id"], 15, 500))
            r.addWidget(tag("hold" if g.get("kind") == "hold" else "instant",
                            C["imu"] if g.get("kind") == "hold" else C["text2"]))
            r.addStretch(1)
            r.addWidget(label(describe(g.get("action")), 13, 400, C["text2"]))
            self.glist.addWidget(row)
            self._grows[g["id"]] = row

    def add_events(self, decided):
        for e, outcome in decided:
            if e.get("kind") in ("instant", "start"):
                self._last_event = e
                g = next((x for x in self.app.gestures if x["id"] == e.get("id")), {})
                self.ev_name.setText(g.get("name") or e.get("id"))
                self.ev_action.setText(outcome)
                self.ev_conf.setText(f"{e.get('conf', 0):.0%}")
                lag = (self.app.backend.status or {}).get("lag_ms")
                self.ev_lat.setText(f"{lag:.0f} ms" if isinstance(lag, (int, float)) else "–")
            self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}   {e.get('id')} {e.get('kind')}   "
                                   f"conf {e.get('conf', 0):.2f}   →  {outcome}")
        while self.log.count() > 300:
            self.log.takeItem(self.log.count() - 1)
