"""Gestures tab: add / rename / delete gestures and assign actions (data/gestures.json)."""
from __future__ import annotations

from PyQt6 import QtWidgets

from .. import gestures as G
from ..actions import ACTION_TYPES, BUTTONS, describe


class ActionDialog(QtWidgets.QDialog):
    def __init__(self, parent, action, kind):
        super().__init__(parent)
        self.setWindowTitle("Action")
        a = dict(action or {"type": "none"})
        self.type = QtWidgets.QComboBox()
        self.type.addItems(ACTION_TYPES)
        self.type.setCurrentText(a.get("type", "none"))
        self.button = QtWidgets.QComboBox()
        self.button.addItems(BUTTONS)
        self.button.setCurrentText(a.get("button", "left"))
        self.dx = QtWidgets.QSpinBox(minimum=-50, maximum=50, value=int(a.get("dx", 0)))
        self.dy = QtWidgets.QSpinBox(minimum=-50, maximum=50, value=int(a.get("dy", -5)))
        self.cont = QtWidgets.QCheckBox("repeat while held (hold gestures)")
        self.cont.setChecked(bool(a.get("continuous", False)))
        self.combo = QtWidgets.QLineEdit(a.get("combo", "ctrl+z"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["pointer"])
        form = QtWidgets.QFormLayout(self)
        form.addRow("Type", self.type)
        self.rows = {"button": ("Button", self.button), "dx": ("Scroll dx", self.dx), "dy": ("Scroll dy (− = down)", self.dy),
                     "cont": ("", self.cont), "combo": ("Keys (e.g. ctrl+z)", self.combo), "mode": ("Mode", self.mode)}
        for label, w in self.rows.values():
            form.addRow(label, w)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        form.addRow(self.hint)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.kind = kind
        self.type.currentTextChanged.connect(self._update)
        self._update()

    def _update(self):
        t = self.type.currentText()
        show = {"click": ["button"], "double_click": ["button"], "drag": ["button"], "scroll": ["dx", "dy", "cont"],
                "key": ["combo"], "mode_toggle": ["mode"]}.get(t, [])
        for k, (_, w) in self.rows.items():
            w.setVisible(k in show)
            lbl = self.layout().labelForField(w)
            if lbl:
                lbl.setVisible(k in show)
        self.hint.setText("Drag: press on gesture start, release on end; use it with a HOLD gesture."
                          if t == "drag" and self.kind != "hold" else "")

    def action(self):
        t = self.type.currentText()
        if t in ("click", "double_click", "drag"):
            return {"type": t, "button": self.button.currentText()}
        if t == "scroll":
            return {"type": t, "dx": self.dx.value(), "dy": self.dy.value(), "continuous": self.cont.isChecked()}
        if t == "key":
            return {"type": t, "combo": self.combo.text().strip().lower()}
        if t == "mode_toggle":
            return {"type": t, "mode": self.mode.currentText()}
        return {"type": "none"}


class GesturesTab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["id", "name", "kind", "action", "examples", "accuracy (current model)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self.edit_action())
        btns = QtWidgets.QHBoxLayout()
        teach = QtWidgets.QPushButton("★ Teach a new gesture…")
        teach.setStyleSheet("background:#2f81f7; color:white; font-weight:bold;")
        teach.setToolTip("Record only the new gesture (+ a few refresher reps of the others), then retrain on everything")
        teach.clicked.connect(lambda: self.app.teach.start(self))
        btns.addWidget(teach)
        more = QtWidgets.QPushButton("Add examples…")
        more.setToolTip("Record more examples of the selected gesture, then retrain")
        more.clicked.connect(lambda: self._selected() and self.app.teach.start(self, self._selected()["id"]))
        btns.addWidget(more)
        for text, fn in [("Add (no recording)…", self.add), ("Rename…", self.rename), ("Set action…", self.edit_action),
                         ("Delete", self.delete)]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        note = QtWidgets.QLabel("Teach records ONE gesture (no alternating) and retrains on all your recordings. "
                                "Deleted gestures are left out of the next training. A gesture id is permanent: never reuse an id for a different motion (recordings refer to it). "
                                "Actions are only used by this app; the backend ignores them.")
        note.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(btns)
        lay.addWidget(self.table)
        lay.addWidget(note)
        self.reload()

    def _accuracy(self):
        """gesture id -> text, from the backend's validation of the loaded model for the selected sensors."""
        from .sensor_picker import current, key
        be = self.app.backend
        m = self.app.model_info.get((be.status or {}).get("model")) if be.online else None
        if not m:
            return {}
        bs = (m.get("by_sensors") or {}).get(key(current())) or {}

        def ok(x):
            return isinstance(x, (int, float)) and x == x
        out = {}
        for gid, v in (bs.get("per_gesture") or {}).items():
            rec, prec = v.get("recall"), v.get("precision")
            out[gid] = (f"recall {rec:.0%}" if ok(rec) else "recall -") + (f", precision {prec:.0%}" if ok(prec) else "")
        return out

    def reload(self):
        from .tab_train import list_sessions
        self.d = G.load()
        counts = {}
        for sess in list_sessions():
            for gid, n in sess["labels"].items():
                if isinstance(n, int):
                    counts[gid] = counts.get(gid, 0) + n
        acc = self._accuracy()
        self.table.setRowCount(0)
        for g in self.d["gestures"]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [g["id"], g.get("name", ""), g.get("kind", "instant"), describe(g.get("action")),
                    str(counts.get(g["id"], 0)), acc.get(g["id"], "not in the loaded model" if acc else "-")]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.app.gestures_changed()

    def _selected(self):
        r = self.table.currentRow()
        return self.d["gestures"][r] if 0 <= r < len(self.d["gestures"]) else None

    def add(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add gesture")
        gid, name = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        gid.setPlaceholderText("snake_case, e.g. swipe_left")
        kind = QtWidgets.QComboBox()
        kind.addItems(["instant", "hold"])
        form = QtWidgets.QFormLayout(dlg)
        form.addRow("id", gid)
        form.addRow("name", name)
        form.addRow("kind", kind)
        form.addRow(QtWidgets.QLabel("instant = one event per occurrence; hold = start/end, active while held"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            G.add(self.d, gid.text().strip(), name.text().strip() or None, kind.currentText())
            G.save(self.d)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Add gesture", str(e))
        self.reload()

    def rename(self):
        g = self._selected()
        if not g:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename", f"Display name for {g['id']}:", text=g.get("name", ""))
        if ok and name.strip():
            g["name"] = name.strip()
            G.save(self.d)
            self.reload()

    def edit_action(self):
        g = self._selected()
        if not g:
            return
        dlg = ActionDialog(self, g.get("action"), g.get("kind"))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            g["action"] = dlg.action()
            G.save(self.d)
            self.reload()

    def delete(self):
        g = self._selected()
        if not g:
            return
        if QtWidgets.QMessageBox.question(self, "Delete", f"Delete gesture '{g['id']}'? Recordings that use it keep "
                                          "their own copy, and the id must not be reused for a different motion.") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self.d["gestures"].remove(g)
            G.save(self.d)
            self.reload()
