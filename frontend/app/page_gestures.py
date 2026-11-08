"""Gestures: cards (Experience) or a definitions table (Data). Teach, add examples, change action, rename, delete."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
from ..actions import ACTION_TYPES, BUTTONS, describe
from . import data, sensor_picker
from .theme import C
from .ui import FlowGrid, button, divider, label, tag

ACTION_LABELS = {"none": "Nothing", "click": "Click", "double_click": "Double click", "scroll": "Scroll",
                 "drag": "Drag (hold gesture)", "key": "Key combo", "pointer_hold": "Move mouse while held (hold gesture)",
                 "mode_toggle": "Turn pointer on/off"}


class ActionDialog(QtWidgets.QDialog):
    def __init__(self, parent, action, kind, title="What should it do?"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        a = dict(action or {"type": "click", "button": "left"})
        self.kind = kind
        self.type = QtWidgets.QComboBox()
        for t in ACTION_TYPES:
            self.type.addItem(ACTION_LABELS.get(t, t), t)
        self.type.setCurrentIndex(max(0, ACTION_TYPES.index(a.get("type", "none")) if a.get("type") in ACTION_TYPES else 0))
        self.button = QtWidgets.QComboBox()
        self.button.addItems(BUTTONS)
        self.button.setCurrentText(a.get("button", "left"))
        self.dy = QtWidgets.QSpinBox(minimum=-50, maximum=50, value=int(a.get("dy", -5)))
        self.cont = QtWidgets.QCheckBox("keep scrolling while held")
        self.cont.setChecked(bool(a.get("continuous")))
        self.combo = QtWidgets.QLineEdit(a.get("combo", "ctrl+z"))
        form = QtWidgets.QFormLayout(self)
        form.addRow("Action", self.type)
        self.rows = {"button": ("Mouse button", self.button), "dy": ("Scroll amount (− = down)", self.dy),
                     "cont": ("", self.cont), "combo": ("Keys, e.g. ctrl+z", self.combo)}
        for lab, w in self.rows.values():
            form.addRow(lab, w)
        self.hint = label("", 12, 400, C["warn"], wrap=True)
        form.addRow(self.hint)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.type.currentIndexChanged.connect(self._vis)
        self._vis()

    def _vis(self, *_):
        t = self.type.currentData()
        show = {"click": ["button"], "double_click": ["button"], "drag": ["button"], "scroll": ["dy", "cont"],
                "key": ["combo"]}.get(t, [])
        for k, (_, w) in self.rows.items():
            w.setVisible(k in show)
            lb = self.layout().labelForField(w)
            if lb:
                lb.setVisible(k in show)
        self.hint.setText("This action needs a HOLD gesture." if t in ("drag", "pointer_hold") and self.kind != "hold" else "")

    def action(self):
        t = self.type.currentData()
        if t in ("click", "double_click", "drag"):
            return {"type": t, "button": self.button.currentText()}
        if t == "scroll":
            return {"type": t, "dx": 0, "dy": self.dy.value(), "continuous": self.cont.isChecked()}
        if t == "key":
            return {"type": t, "combo": self.combo.text().strip().lower()}
        if t == "mode_toggle":
            return {"type": t, "mode": "pointer"}
        return {"type": t}


class GesturesPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 16)
        lay.setSpacing(14)
        head = QtWidgets.QHBoxLayout()
        tcol = QtWidgets.QVBoxLayout()
        tcol.setSpacing(2)
        tcol.addWidget(label("Gestures", 22, 600))
        tcol.addWidget(label("Each gesture is taught on its own. Adding or removing one retrains the model on everything "
                             "you've recorded.", 13, 400, C["text2"]))
        head.addLayout(tcol)
        head.addStretch(1)
        head.addWidget(button("+  Teach a new gesture", "Primary", lambda: self.app.teach_new(self)))
        lay.addLayout(head)
        self.stack = QtWidgets.QStackedWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid = FlowGrid(320)
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.grid)
        hl.addStretch(1)
        scroll.setWidget(holder)
        self.stack.addWidget(scroll)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["id", "name", "kind", "action", "examples", "recordings",
                                              "recognised", "precision"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setShowGrid(False)
        self.stack.addWidget(self.table)
        lay.addWidget(self.stack, 1)

    def set_mode(self, mode):
        self.stack.setCurrentIndex(0 if mode == "experience" else 1)

    def refresh(self, snap):
        pass

    # ------------------------------------------------------------------ build
    def _accuracy(self):
        be = self.app.backend
        m = self.app.model_info.get((be.status or {}).get("model")) if be.online else None
        if not m:
            return {}
        bs = (m.get("by_sensors") or {}).get(sensor_picker.key(sensor_picker.current())) or {}
        return bs.get("per_gesture") or (m.get("val_detail") or {}).get("per_gesture") or {}

    def reload(self):
        gl = G.load()["gestures"]
        sess = data.list_sessions()
        counts = data.example_counts(sess)
        acc = self._accuracy()
        cards = [self._add_card()] + [self._card(g, counts.get(g["id"], 0), len(data.sessions_for(g["id"], sess)),
                                                  acc.get(g["id"])) for g in gl]
        self.grid.set_items(cards)
        self.table.setRowCount(0)
        for g in gl:
            r = self.table.rowCount()
            self.table.insertRow(r)
            a = acc.get(g["id"]) or {}
            vals = [g["id"], g.get("name", ""), g.get("kind", "instant"), describe(g.get("action")),
                    str(counts.get(g["id"], 0)), str(len(data.sessions_for(g["id"], sess))),
                    _pct(a.get("recall")), _pct(a.get("precision"))]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()

    def _add_card(self):
        f = QtWidgets.QFrame()
        f.setObjectName("AddCard")
        f.setMinimumHeight(190)
        f.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.addStretch(1)
        lay.addWidget(label("+", 34, 300, C["accent"]), 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label("Teach a new gesture", 16, 600), 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label("about 2 minutes: do it ~20 times", 12, 400, C["text2"]), 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addStretch(1)
        f.mousePressEvent = lambda e: self.app.teach_new(self)
        return f

    def _card(self, g, n_ex, n_sess, acc):
        f = QtWidgets.QFrame()
        f.setObjectName("Card")
        f.setMinimumHeight(190)
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label(g.get("name") or g["id"], 18, 600))
        top.addStretch(1)
        top.addWidget(tag("hold" if g.get("kind") == "hold" else "instant",
                          C["imu"] if g.get("kind") == "hold" else C["text2"]))
        lay.addLayout(top)
        ar = QtWidgets.QHBoxLayout()
        ar.addWidget(label(describe(g.get("action")), 14, 400, C["text"]))
        ar.addStretch(1)
        ar.addWidget(button("Change", "Ghost", lambda: self.change_action(g["id"])))
        lay.addLayout(ar)
        lay.addWidget(divider())
        stats = QtWidgets.QHBoxLayout()
        stats.addWidget(self._stat(str(n_ex), "examples"))
        rec = (acc or {}).get("recall")
        col = C["text"] if rec is None or rec != rec else (C["ok"] if rec >= 0.85 else C["warn"] if rec >= 0.6 else C["bad"])
        stats.addWidget(self._stat(_pct(rec), "recognised", col))
        stats.addStretch(1)
        lay.addLayout(stats)
        if n_ex < 15:
            lay.addWidget(label("Add more examples for better accuracy.", 12, 400, C["warn"]))
        lay.addStretch(1)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(button("Add examples", None, lambda: self.app.teach_more(self, g["id"]),
                              "Record more of this gesture, then retrain"))
        btns.addStretch(1)
        more = button("•••", "Ghost")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Rename…", lambda: self.rename(g["id"]))
        menu.addAction("Change action…", lambda: self.change_action(g["id"]))
        menu.addSeparator()
        menu.addAction("Delete…", lambda: self.delete(g["id"]))
        more.setMenu(menu)
        btns.addWidget(more)
        lay.addLayout(btns)
        return f

    @staticmethod
    def _stat(v, t, color=None):
        w = QtWidgets.QWidget()
        c = QtWidgets.QVBoxLayout(w)
        c.setContentsMargins(0, 0, 18, 0)
        c.setSpacing(0)
        c.addWidget(label(v, 20, 600, color))
        c.addWidget(label(t, 12, 400, C["label"]))
        return w

    # ------------------------------------------------------------------ edits
    def change_action(self, gid):
        d = G.load()
        g = G.get(d, gid)
        if not g:
            return
        dlg = ActionDialog(self, g.get("action"), g.get("kind"), f"What should '{g.get('name') or gid}' do?")
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            g["action"] = dlg.action()
            G.save(d)
            self.app.gestures_changed()

    def rename(self, gid):
        d = G.load()
        g = G.get(d, gid)
        if not g:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename", "Name shown in the app:", text=g.get("name", ""))
        if ok and name.strip():
            g["name"] = name.strip()
            G.save(d)
            self.app.gestures_changed()

    def delete(self, gid):
        d = G.load()
        g = G.get(d, gid)
        if not g:
            return
        only = [s for s in data.sessions_for(gid) if set(s["labels"]) - {"none"} == {gid}]
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Delete gesture")
        box.setText(f"Delete '{g.get('name') or gid}'?")
        box.setInformativeText("The model is retrained without it. The id can't be reused for a different motion "
                               "while its recordings exist.")
        cb = QtWidgets.QCheckBox(f"Also delete its {len(only)} recording(s)")
        cb.setChecked(True)
        box.setCheckBox(cb)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel)
        if box.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        d["gestures"].remove(g)
        G.save(d)
        if cb.isChecked():
            data.delete_sessions([s["id"] for s in only])
        self.app.gestures_changed()
        self.app.sessions_changed()
        self.app.retrain(reason=f"'{g.get('name') or gid}' removed")


def _pct(x):
    return f"{x:.0%}" if isinstance(x, (int, float)) and x == x else "–"
