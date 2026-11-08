"""Train: Step 01/02/03 cards + model cards (Experience), or sessions, training curve and models (Data)."""
from __future__ import annotations

import datetime as dt

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import gestures as G
from . import data, sensor_picker
from .theme import C
from .ui import FlowGrid, Panel, button, label, tag


def _pct(x):
    return f"{x:.0%}" if isinstance(x, (int, float)) and x == x else "–"


def _when(m):
    c = m.get("created")
    try:
        return dt.datetime.fromisoformat(str(c)).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        return str(c or "")[:16]


class TrainPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 16)
        lay.setSpacing(14)
        head = QtWidgets.QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("Train", 22, 600))
        head.addWidget(label("Teach gestures, retrain, and choose which model runs.", 13, 400, C["text2"]))
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
        s1.add(label("Each gesture is recorded on its own (~2 min). Do it once per cue, then a minute of normal activity.",
                     13, 400, C["text2"], wrap=True))
        s1.lay.addStretch(1)
        s1.add(button("+  Teach a gesture", "Primary", lambda: self.app.teach_new(self)))
        steps.addWidget(s1, 1)
        s2 = Panel("Train", 2)
        self.s2_txt = label("", 15, 500, wrap=True)
        s2.add(self.s2_txt)
        self.s2_sub = label("", 13, 400, C["text2"], wrap=True)
        s2.add(self.s2_sub)
        self.progress = QtWidgets.QProgressBar(maximum=1000)
        self.progress.setTextVisible(False)
        s2.add(self.progress)
        s2.lay.addStretch(1)
        self.retrain_btn = button("Retrain on everything", None, lambda: self.app.retrain(reason="manual"))
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
        mp = Panel("Model", 4)
        mp.add(label("Axion keeps one model. It's replaced automatically whenever you add, re-teach or remove a "
                     "gesture. Rename it from its ••• menu.", 13, 400, C["text2"], wrap=True))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.mgrid = FlowGrid(300)
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.mgrid)
        hl.addStretch(1)
        scroll.setWidget(holder)
        mp.add(scroll, 1)
        lay.addWidget(mp, 1)
        return w

    def _model_card(self, m, current):
        f = QtWidgets.QFrame()
        f.setObjectName("Card")
        if current:
            f.setStyleSheet(f"QFrame#Card {{ border: 1px solid {C['accent']}; background: {C['tint']}; }}")
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(4)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label(data.model_label(m["name"], m), 16, 600, wrap=True), 1)
        if current:
            top.addWidget(label("● running" if self.app.backend.status.get("state") == "running" else "● loaded",
                                12, 500, C["accent"]))
        more = button("•••", "Ghost")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Rename…", lambda: self.rename_model(m))
        menu.addAction("Delete…", lambda: self.delete_model(m["name"]))
        more.setMenu(menu)
        top.addWidget(more)
        lay.addLayout(top)
        tags = QtWidgets.QHBoxLayout()
        tags.setSpacing(6)
        for g in data.model_gestures(m):
            tags.addWidget(tag(data.gesture_label(g), C["link"]))
        tags.addStretch(1)
        lay.addLayout(tags)
        lay.addWidget(label(_when(m), 12, 400, C["text2"]))
        vm = (m.get("by_sensors") or {}).get(sensor_picker.key(m.get("sensors") or sensor_picker.ALL)) or m.get("val_metrics") or {}
        lay.addWidget(label(f"recognised {_pct(vm.get('recall'))}  ·  precision {_pct(vm.get('precision'))}", 13, 400))
        sens = " + ".join(sensor_picker.LABEL.get(s, s) for s in (m.get("sensors") or sensor_picker.ALL))
        w = (m.get("sensor_weights") or {}).get("emg")
        lay.addWidget(label(sens + (f"  ·  EMG weight {w:.0%}" if isinstance(w, (int, float)) and "emg" in (m.get("sensors") or ["emg"]) else ""),
                            12, 400, C["label"]))
        btns = QtWidgets.QHBoxLayout()
        running_here = current and self.app.backend.status.get("state") == "running"
        use = button("In use" if running_here else "Use", "Primary" if not current else None,
                     lambda: self.app.use_model(m["name"]))
        use.setEnabled(not running_here)
        btns.addWidget(use)
        btns.addStretch(1)
        lay.addLayout(btns)
        return f

    # ================================================================== Data
    def _data(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(14)
        sp = Panel("Recordings", 1, right=button("Delete selected", "Danger", self.delete_sessions))
        self.sess = QtWidgets.QTableWidget(0, 6)
        self.sess.setHorizontalHeaderLabels(["session", "gestures", "seconds", "hand visible", "EMG", "warnings"])
        self.sess.horizontalHeader().setStretchLastSection(True)
        self.sess.verticalHeader().setVisible(False)
        self.sess.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sess.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.sess.setShowGrid(False)
        sp.add(self.sess, 1)
        left.addWidget(sp, 3)
        mp = Panel("Models", 2)
        self.models = QtWidgets.QTableWidget(0, 7)
        self.models.setHorizontalHeaderLabels(["name", "model id", "gestures", "created", "recognised", "precision",
                                               "false / min"])
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.verticalHeader().setVisible(False)
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.models.setShowGrid(False)
        self.models.itemSelectionChanged.connect(self._show_curve)
        mp.add(self.models, 1)
        mb = QtWidgets.QHBoxLayout()
        mb.addWidget(button("Use selected", None, lambda: self._sel_model() and self.app.use_model(self._sel_model())))
        mb.addWidget(button("Delete selected", "Danger", lambda: self._sel_model() and self.delete_model(self._sel_model())))
        mb.addStretch(1)
        mp.add(mb)
        left.addWidget(mp, 2)
        lay.addLayout(left, 3)
        cp = Panel("Training curve", 3)
        self.curve = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.curve.setMinimumWidth(360)
        self.curve.setText("Select a model")
        self.curve.setStyleSheet(f"color: {C['label']};")
        cp.add(self.curve, 1)
        lay.addWidget(cp, 2)
        return w

    def _sel_model(self):
        r = self.models.currentRow()
        return self.models.item(r, 1).text() if r >= 0 and self.models.item(r, 1) else None

    def _show_curve(self):
        name = self._sel_model()
        if not name:
            return
        for fn in ("curve.png", "report.png"):
            p = data.MODELS / name / fn
            if p.exists():
                pm = QtGui.QPixmap(str(p))
                self.curve.setPixmap(pm.scaled(self.curve.width(), max(300, self.curve.height()),
                                               QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                               QtCore.Qt.TransformationMode.SmoothTransformation))
                return
        self.curve.setText("No curve saved for this model")

    # ================================================================== mode / refresh
    def set_mode(self, mode):
        self.stack.setCurrentIndex(0 if mode == "experience" else 1)

    def reload(self):
        sess = data.list_sessions()
        models = data.list_models()
        current = (self.app.backend.status or {}).get("model") if self.app.backend.online else None
        self.mgrid.set_items([self._model_card(m, m["name"] == current) for m in models] or
                             [label("No models yet. Teach a gesture and one is trained automatically.", 14, 400,
                                    C["text2"], wrap=True)])
        self.sess.setRowCount(0)
        for s in sess:
            r = self.sess.rowCount()
            self.sess.insertRow(r)
            gl = ", ".join(f"{k} ×{v}" for k, v in s["labels"].items() if k != "none")
            vals = [s["id"], gl, f"{s['seconds'] or 0:.0f}", f"{s['hand_pct']:.0f}%" if s["hand_pct"] is not None else "–",
                    "ok" if s["emg_valid"] else "off", "; ".join(s["warnings"])]
            for c, v in enumerate(vals):
                self.sess.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.sess.resizeColumnsToContents()
        self.models.setRowCount(0)
        for m in models:
            r = self.models.rowCount()
            self.models.insertRow(r)
            vm = m.get("val_metrics") or {}
            fp = vm.get("fp_per_min")
            vals = [data.model_label(m["name"], m), m["name"],
                    ", ".join(data.gesture_label(g) for g in data.model_gestures(m)), _when(m), _pct(vm.get("recall")),
                    _pct(vm.get("precision")), f"{fp:.1f}" if isinstance(fp, (int, float)) and fp == fp else "–"]
            for c, v in enumerate(vals):
                self.models.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.models.resizeColumnsToContents()
        self._update_steps(sess)

    def _update_steps(self, sess=None):
        gl = G.load()["gestures"]
        counts = data.example_counts(sess)
        n = len([g for g in gl if counts.get(g["id"])])
        self.s1_txt.setText(f"{n} gesture{'s' if n != 1 else ''} taught" if n else "No gestures yet")

    def refresh(self, snap):
        be = self.app.backend
        st = be.status if be.online else {}
        if not be.online:
            self.s2_txt.setText("Engine offline")
            self.s2_sub.setText("Start it with:  python -m backend.serve")
            self.progress.setValue(0)
        elif st.get("state") == "training":
            self.s2_txt.setText("Training…")
            self.s2_sub.setText(st.get("message") or "")
            self.progress.setValue(int(1000 * float(st.get("progress") or 0)))
        elif st.get("state") == "error":
            self.s2_txt.setText("Training failed")
            self.s2_sub.setText(st.get("message") or "")
            self.progress.setValue(0)
        elif self.app.model_stale:
            self.s2_txt.setText("Changes not trained yet")
            self.s2_sub.setText("You postponed retraining. Press Retrain when you're ready.")
            self.progress.setValue(0)
        else:
            self.s2_txt.setText(f"Model: {data.model_label(st.get('model'))}" if st.get("model") else "No model yet")
            self.s2_sub.setText("Retrains automatically whenever you add, re-teach or remove a gesture.")
            self.progress.setValue(1000 if st.get("model") else 0)
        self.retrain_btn.setEnabled(be.online and st.get("state") != "training")
        running = st.get("state") == "running"
        self.s3_txt.setText("Recognising gestures" if running else ("Ready to start" if st.get("model") else "Waiting for a model"))

    # ================================================================== deletes
    def rename_model(self, m):
        cur = data.model_label(m["name"], m)
        txt, ok = QtWidgets.QInputDialog.getText(self, "Rename model", "Model name:", text=cur)
        if ok:
            data.set_model_label(m["name"], txt.strip() or None)
            self.reload()

    def delete_model(self, name):
        st = self.app.backend.status if self.app.backend.online else {}
        if st.get("model") == name:
            QtWidgets.QMessageBox.information(self, "Delete model", f"'{data.model_label(name)}' is the model in use. Use another model "
                                              "first, or retrain, then delete this one.")
            return
        if QtWidgets.QMessageBox.question(self, "Delete model", f"Delete model '{data.model_label(name)}'? This can't be undone.") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            data.delete_model(name)
            self.app.refresh_models()

    def delete_sessions(self):
        rows = sorted({i.row() for i in self.sess.selectedIndexes()})
        ids = [self.sess.item(r, 0).text() for r in rows]
        if not ids:
            QtWidgets.QMessageBox.information(self, "Delete recordings", "Select recordings in the table first.")
            return
        if QtWidgets.QMessageBox.question(self, "Delete recordings",
                                          f"Delete {len(ids)} recording(s)? They're left out of future training.") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            data.delete_sessions(ids)
            self.app.sessions_changed()
