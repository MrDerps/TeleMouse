"""Train tab: pick sessions + gestures, send `train` on 5557, follow progress on 5556; list and load models."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G

SESSIONS = Path(__file__).resolve().parents[2] / "data" / "sessions"


def list_sessions():
    out = []
    for p in sorted(SESSIONS.glob("*/meta.json"), reverse=True):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = m.get("summary", {})
        labs = s.get("labels") or {}
        if not labs:
            snap = m.get("gestures_snapshot", {}).get("gestures", [])
            labs = {g["id"]: "?" for g in snap}
        out.append({"id": m.get("session_id", p.parent.name), "seconds": s.get("seconds"),
                    "labels": labs, "emg_valid": m.get("emg_valid", True),
                    "imu": m.get("imu_present"), "hand": m.get("hand_present"),
                    "warnings": len(s.get("warnings", []))})
    return out


class TrainTab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.sess = QtWidgets.QTableWidget(0, 6)
        self.sess.setHorizontalHeaderLabels(["session", "seconds", "labels", "EMG", "IMU / hand", "warnings"])
        self.sess.horizontalHeader().setStretchLastSection(True)
        self.sess.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.glist = QtWidgets.QListWidget()
        self.name = QtWidgets.QLineEdit(dt.datetime.now().strftime("m_%m%d_%H%M"))
        self.train_btn = QtWidgets.QPushButton("Train")
        self.train_btn.clicked.connect(self.train)
        self.progress = QtWidgets.QProgressBar(maximum=1000)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        self.models = QtWidgets.QTableWidget(0, 4)
        self.models.setHorizontalHeaderLabels(["model", "classes", "created", "validation"])
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        rb = QtWidgets.QPushButton("Refresh sessions")
        rb.clicked.connect(self.reload_sessions)
        mb = QtWidgets.QPushButton("Refresh models")
        mb.clicked.connect(self.refresh_models)
        lb = QtWidgets.QPushButton("Load selected model")
        lb.clicked.connect(self.load_model)

        left = QtWidgets.QVBoxLayout()
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Sessions (tick the ones to train on)"))
        h.addStretch(1)
        h.addWidget(rb)
        left.addLayout(h)
        left.addWidget(self.sess, 2)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Models"))
        h2.addStretch(1)
        h2.addWidget(mb)
        h2.addWidget(lb)
        left.addLayout(h2)
        left.addWidget(self.models, 1)
        right = QtWidgets.QFormLayout()
        right.addRow(QtWidgets.QLabel("Gestures"))
        right.addRow(self.glist)
        right.addRow("Model name", self.name)
        right.addRow(self.train_btn)
        right.addRow(self.progress)
        right.addRow(self.msg)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(left, 3)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        lay.addWidget(rw, 1)
        self.reload_sessions()

    def reload_sessions(self):
        self.sess.setRowCount(0)
        for s in list_sessions():
            r = self.sess.rowCount()
            self.sess.insertRow(r)
            it = QtWidgets.QTableWidgetItem(s["id"])
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if "legacy" not in s["id"] else QtCore.Qt.CheckState.Unchecked)
            self.sess.setItem(r, 0, it)
            vals = [s["seconds"], ", ".join(f"{k}:{v}" for k, v in s["labels"].items()),
                    "ok" if s["emg_valid"] else "invalid", f"{'✓' if s['imu'] else '–'} / {'✓' if s['hand'] else '–'}",
                    s["warnings"]]
            for c, v in enumerate(vals, 1):
                self.sess.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.sess.resizeColumnsToContents()

    def reload_gestures(self):
        self.glist.clear()
        for g in G.load()["gestures"]:
            it = QtWidgets.QListWidgetItem(g["id"])
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked)
            self.glist.addItem(it)

    def train(self):
        sessions = [self.sess.item(r, 0).text() for r in range(self.sess.rowCount())
                    if self.sess.item(r, 0).checkState() == QtCore.Qt.CheckState.Checked]
        gest = [self.glist.item(i).text() for i in range(self.glist.count())
                if self.glist.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        name = self.name.text().strip()
        if not sessions or not gest or not name:
            QtWidgets.QMessageBox.information(self, "Train", "Choose sessions, gestures and a model name.")
            return
        self.msg.setText("sending train request…")
        self.app.backend.control({"cmd": "train", "gestures": gest, "sessions": sessions, "name": name},
                                 self._train_reply, timeout_s=5.0)

    def _train_reply(self, r):
        self.msg.setText("training started" if r.get("ok") else f"error: {r.get('error')}")

    def refresh_models(self):
        self.app.backend.control({"cmd": "list_models"}, self._models_reply)

    def _models_reply(self, r):
        if not r.get("ok"):
            self.msg.setText(f"list_models: {r.get('error')}")
            return
        self.models.setRowCount(0)
        for m in r.get("models", []):
            row = self.models.rowCount()
            self.models.insertRow(row)
            vm = m.get("val_metrics") or {}
            short = ", ".join(f"{k} {v:.2f}" if isinstance(v, (int, float)) else f"{k} {v}"
                              for k, v in list(vm.items())[:4])
            for c, v in enumerate([m.get("name"), ", ".join(m.get("classes", [])), str(m.get("created", "")), short]):
                self.models.setItem(row, c, QtWidgets.QTableWidgetItem(str(v)))
        self.models.resizeColumnsToContents()

    def load_model(self):
        r = self.models.currentRow()
        if r < 0:
            return
        name = self.models.item(r, 0).text()
        self.app.backend.control({"cmd": "load", "name": name},
                                 lambda rep: self.msg.setText(f"loaded {name}" if rep.get("ok") else f"load: {rep.get('error')}"),
                                 timeout_s=10.0)

    def refresh(self):
        st = self.app.backend.status if self.app.backend.online else {}
        if st.get("state") == "training":
            self.progress.setValue(int(1000 * float(st.get("progress") or 0)))
            self.msg.setText(st.get("message") or "training…")
        elif st:
            if self.progress.value() and st.get("state") != "training":
                self.progress.setValue(1000 if st.get("state") != "error" else 0)
            if st.get("state") == "error":
                self.msg.setText(f"error: {st.get('message')}")
