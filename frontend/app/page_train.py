"""Train: Experience = steps 01-03 (teach / train with progress / use) + what the current model knows and listens to.
Data = visuals of the network (architecture with live outputs, learning curve, confusion, live trace) + recordings."""
from __future__ import annotations

import datetime as dt

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
from ..actions import describe
from . import data, sensor_picker
from .ai_viz import ConfusionMap, CurveChart, LiveTrace, NetworkView
from .theme import C, SENSOR_COLOR
from .ui import Panel, button, divider, label, tag


def _when(m):
    c = m.get("created")
    try:
        return dt.datetime.fromisoformat(str(c)).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        return str(c or "")[:16]


def _clear(lay):
    while lay.count():
        it = lay.takeAt(0)
        if it.widget():
            it.widget().setParent(None)
        elif it.layout():
            _clear(it.layout())


class TrainPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._shown = None                 # model name the panels were last built for
        self._seen = None                  # last model name seen in the engine status
        self._probs_t = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 16)
        lay.setSpacing(14)
        head = QtWidgets.QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("Train", 22, 600))
        lay.addLayout(head)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._experience())
        self.stack.addWidget(self._data())
        lay.addWidget(self.stack, 1)

    # ================================================================== Experience
    def _experience(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        steps = QtWidgets.QHBoxLayout()
        steps.setSpacing(14)
        s1 = Panel("Teach", 1)
        self.s1_txt = label("", 15, 500, wrap=True)
        s1.add(self.s1_txt)
        s1.add(label("Record one gesture at a time: do it once per cue, then some normal activity.",
                     13, 400, C["text2"], wrap=True))
        s1.lay.addStretch(1)
        s1.add(button("+  Teach a gesture", "Primary", lambda: self.app.teach_new(self)))
        steps.addWidget(s1, 1)
        s2 = Panel("Train", 2)
        row = QtWidgets.QHBoxLayout()
        self.s2_txt = label("", 15, 500, wrap=True)
        row.addWidget(self.s2_txt, 1)
        self.s2_pct = label("", 15, 600, C["accent"])
        row.addWidget(self.s2_pct)
        s2.add(row)
        self.s2_sub = label("", 13, 400, C["text2"], wrap=True)
        s2.add(self.s2_sub)
        self.progress = QtWidgets.QProgressBar(maximum=1000)
        self.progress.setTextVisible(False)
        s2.add(self.progress)
        s2.lay.addStretch(1)
        self.retrain_btn = button("Retrain now", None, lambda: self.app.retrain(reason="manual"))
        s2.add(self.retrain_btn)
        steps.addWidget(s2, 1)
        s3 = Panel("Use", 3)
        self.s3_txt = label("", 15, 500, wrap=True)
        s3.add(self.s3_txt)
        s3.add(label("Start recognising, then turn on 'Control my computer' on the dashboard. F9 stops everything.",
                     13, 400, C["text2"], wrap=True))
        s3.lay.addStretch(1)
        s3.add(button("Go to dashboard", None, lambda: self.app.go("dashboard")))
        steps.addWidget(s3, 1)
        lay.addLayout(steps)

        self.rename_btn = button("Rename", "Ghost", self.rename_model)
        mp = Panel("Your model", 4)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(14)
        self.m_name = label("", 20, 600)
        top.addWidget(self.m_name)
        self.m_state = label("", 12, 500, C["accent"])
        top.addWidget(self.m_state)
        top.addStretch(1)
        top.addWidget(self.rename_btn)
        mp.add(top)
        self.m_when = label("", 12, 400, C["text2"])
        mp.add(self.m_when)
        mp.add(divider())
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(36)
        gcol = QtWidgets.QVBoxLayout()
        gcol.setSpacing(6)
        gcol.addWidget(label("GESTURES IT KNOWS", 11, 500, C["label"], tracking=0.08))
        self.g_rows = QtWidgets.QGridLayout()
        self.g_rows.setHorizontalSpacing(18)
        self.g_rows.setVerticalSpacing(6)
        gcol.addLayout(self.g_rows)
        gcol.addStretch(1)
        body.addLayout(gcol, 3)
        scol = QtWidgets.QVBoxLayout()
        scol.setSpacing(6)
        scol.addWidget(label("LISTENS TO", 11, 500, C["label"], tracking=0.08))
        self.s_rows = QtWidgets.QVBoxLayout()
        self.s_rows.setSpacing(8)
        scol.addLayout(self.s_rows)
        scol.addStretch(1)
        body.addLayout(scol, 2)
        mp.add(body)
        lay.addWidget(mp)
        lay.addStretch(1)
        return w

    def _current_model(self):
        st = self.app.backend.status if self.app.backend.online else {}
        name = st.get("model")
        models = data.list_models()
        for m in models:
            if m["name"] == name:
                return m
        return models[0] if models else None

    def _build_model_panel(self, m):
        _clear(self.g_rows)
        _clear(self.s_rows)
        self.rename_btn.setVisible(bool(m))
        if not m:
            self.m_name.setText("No model yet")
            self.m_state.setText("")
            self.m_when.setText("Teach a gesture and a model is trained automatically.")
        else:
            self.m_name.setText(data.model_label(m["name"], m))
            tr = m.get("train") or {}
            bits = [f"Trained {_when(m)}"]
            if tr.get("seconds"):
                bits.append(f"took {tr['seconds']:.0f} s")
            n = len(m.get("sessions") or [])
            if n:
                bits.append(f"from {n} recording{'s' if n != 1 else ''}")
            self.m_when.setText("  ·  ".join(bits))
        in_model = set(data.model_gestures(m)) if m else set()
        counts = data.example_counts()
        from .teach import negatives
        ignored = set(negatives()[1])
        gl = G.load()["gestures"]
        if not gl:
            self.g_rows.addWidget(label("None yet.", 14, 400, C["text2"]), 0, 0)
        for i, g in enumerate(gl):
            on = g.get("enabled", True)
            self.g_rows.addWidget(label(g.get("name") or g["id"], 15, 500, C["text"] if on else C["disabled"]), i, 0)
            self.g_rows.addWidget(tag("hold" if g.get("kind") == "hold" else "instant",
                                      C["imu"] if g.get("kind") == "hold" else C["text2"]), i, 1)
            self.g_rows.addWidget(label(("→  " + describe(g.get("action"))) if on else "", 13, 400, C["text2"]), i, 2)
            n = counts.get(g["id"], 0)
            if not on:
                st, col = ("taught to ignore" if g["id"] in ignored else "switched off"), C["label"]
            elif g["id"] in in_model:
                st, col = f"{n} examples", C["ok"]
            else:
                st, col = "not trained yet", C["warn"]
            self.g_rows.addWidget(label(st, 12, 500, col), i, 3)
        self.g_rows.setColumnStretch(4, 1)
        if m:
            used = m.get("sensors") or sensor_picker.ALL
            mode = ((m.get("feature_cfg") or {}).get("imu") or {}).get("mode", "raw")
            imu_txt = {"body": "hand-relative motion (any arm direction)",
                       "magnitude": "direction-free motion (any arm direction)"}.get(mode, "motion incl. gravity direction")
            ew = (m.get("sensor_weights") or {}).get("emg")
            what = {"emg": f"forearm muscles" + (f" · influence {ew:.0%}" if isinstance(ew, (int, float)) else ""),
                    "imu": imu_txt, "hand": "webcam hand shape"}
            for s in ("emg", "imu", "hand"):
                if s not in used:
                    continue
                r = QtWidgets.QHBoxLayout()
                r.setSpacing(10)
                r.addWidget(tag(sensor_picker.LABEL[s], SENSOR_COLOR[s]))
                r.addWidget(label(what[s], 13, 400, C["text2"], wrap=True), 1)
                self.s_rows.addLayout(r)
            if "hand" not in used:
                self.s_rows.addWidget(label("No camera needed: works in any direction, no line of sight.",
                                            12, 400, C["label"], wrap=True))
            if self.app.model_stale or sorted(sensor_picker.current()) != sorted(used) or \
                    (("imu" in used) and mode != sensor_picker.imu_mode()):
                self.s_rows.addWidget(label("Your settings changed since this training. Retrain to apply them.",
                                            12, 400, C["warn"], wrap=True))

    # ================================================================== Data
    def _data(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(14)
        pn = Panel("The network", 1)
        pn.add(label("How your sensor data flows through the model. Each output bar fills with how sure it is "
                     "right now.", 12, 400, C["text2"], wrap=True))
        self.net = NetworkView()
        pn.add(self.net, 1)
        g.addWidget(pn, 0, 0, 1, 2)
        pl = Panel("Live thinking", 2)
        pl.add(label("The network's output per gesture. A gesture fires when its line crosses its threshold.",
                     12, 400, C["text2"], wrap=True))
        self.trace = LiveTrace()
        pl.add(self.trace, 1)
        g.addWidget(pl, 0, 2)
        pc = Panel("How it learned", 3)
        self.curve = CurveChart()
        pc.add(self.curve, 1)
        g.addWidget(pc, 1, 0)
        px = Panel("What it confuses", 4)
        px.add(label("Unseen practice reps, frame by frame: rows = what you did.", 12, 400, C["text2"], wrap=True))
        self.conf = ConfusionMap()
        px.add(self.conf, 1)
        g.addWidget(px, 1, 1)
        sp = Panel("Recordings", 5, right=button("Delete selected", "Danger", self.delete_sessions))
        self.sess = QtWidgets.QTableWidget(0, 4)
        self.sess.setHorizontalHeaderLabels(["recorded", "gestures", "s", "warnings"])
        self.sess.horizontalHeader().setStretchLastSection(True)
        self.sess.verticalHeader().setVisible(False)
        self.sess.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sess.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.sess.setShowGrid(False)
        sp.add(self.sess, 1)
        g.addWidget(sp, 1, 2)
        for c in range(3):
            g.setColumnStretch(c, 1)
        g.setRowStretch(0, 5)
        g.setRowStretch(1, 4)
        return w

    # ================================================================== mode / refresh
    def set_mode(self, mode):
        self.stack.setCurrentIndex(0 if mode == "experience" else 1)

    def reload(self):
        sess = data.list_sessions()
        m = self._current_model()
        self._shown = m["name"] if m else None
        self._build_model_panel(m)
        for v in (self.net, self.curve, self.conf, self.trace):
            v.set_model(m)
        self.sess.setRowCount(0)
        for s in sess:
            r = self.sess.rowCount()
            self.sess.insertRow(r)
            gl = ", ".join(f"{data.gesture_label(k)} ×{v}" for k, v in s["labels"].items() if k != "none")
            try:
                when = dt.datetime.strptime(s["id"][:15], "%Y%m%d_%H%M%S").strftime("%b %d, %H:%M")
            except ValueError:
                when = s["id"]
            vals = [when, gl, f"{s['seconds'] or 0:.0f}", "; ".join(s["warnings"]) or "–"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setToolTip(s["id"] if c == 0 else v)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, s["id"])
                self.sess.setItem(r, c, it)
        self.sess.resizeColumnsToContents()
        gl = G.load()["gestures"]
        counts = data.example_counts(sess)
        n = len([g for g in gl if counts.get(g["id"])])
        self.s1_txt.setText(f"{n} gesture{'s' if n != 1 else ''} taught" if n else "No gestures yet")

    def refresh(self, snap):
        be = self.app.backend
        st = be.status if be.online else {}
        sm = st.get("model") if st.get("state") != "training" else None
        if sm and sm != self._seen:                         # a different model was loaded: rebuild the panels
            self._seen = sm
            self.reload()
        pct = ""
        if not be.online:
            self.s2_txt.setText("Engine offline")
            self.s2_sub.setText("Start it with:  python -m backend.serve")
            self.progress.setValue(0)
        elif st.get("state") == "training":
            f = float(st.get("progress") or 0)
            self.s2_txt.setText("Training…")
            pct = f"{f:.0%}"
            self.s2_sub.setText(st.get("message") or "")
            self.progress.setValue(int(1000 * f))
        elif st.get("state") == "error":
            self.s2_txt.setText("Training failed")
            self.s2_sub.setText(st.get("message") or "")
            self.progress.setValue(0)
        elif self.app.model_stale:
            self.s2_txt.setText("Changes not trained yet")
            self.s2_sub.setText("You postponed retraining. Press Retrain now when you're ready.")
            self.progress.setValue(0)
        else:
            self.s2_txt.setText("Up to date" if st.get("model") else "No model yet")
            self.s2_sub.setText("Retrains automatically whenever you add, re-teach or remove a gesture.")
            self.progress.setValue(1000 if st.get("model") else 0)
        self.s2_pct.setText(pct)
        self.retrain_btn.setEnabled(be.online and st.get("state") != "training")
        running = st.get("state") == "running"
        self.s3_txt.setText("Recognising gestures" if running else ("Ready to start" if st.get("model") else "Waiting for a model"))
        self.m_state.setText("● running" if running else ("● training a new version" if st.get("state") == "training"
                                                          else ("● loaded" if st.get("model") else "")))
        if self.stack.currentIndex() == 1:
            pr = be.probs if running else None
            if pr and pr[0] != self._probs_t:
                self._probs_t = pr[0]
                self.trace.push(*pr)
                self.net.set_probs(pr[1], pr[2])
            elif not running and self.net.live:
                self.net.live = {}
                self.net.update()
            self.trace.update()

    # ================================================================== actions
    def rename_model(self):
        m = self._current_model()
        if not m:
            return
        cur = data.model_label(m["name"], m)
        txt, ok = QtWidgets.QInputDialog.getText(self, "Rename model", "Model name:", text=cur)
        if ok:
            data.set_model_label(m["name"], txt.strip() or None)
            self.reload()

    def delete_sessions(self):
        rows = sorted({i.row() for i in self.sess.selectedIndexes()})
        ids = [self.sess.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole) for r in rows]
        if not ids:
            QtWidgets.QMessageBox.information(self, "Delete recordings", "Select recordings in the table first.")
            return
        if QtWidgets.QMessageBox.question(self, "Delete recordings",
                                          f"Delete {len(ids)} recording(s)? They're left out of future training.") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            data.delete_sessions(ids)
            self.app.sessions_changed()
