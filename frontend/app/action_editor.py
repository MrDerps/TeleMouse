"""A friendly action picker: one list of plain choices instead of type + button + amount."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .theme import C
from .ui import label

# (label, action template, needs_hold)
CHOICES = [
    ("Left click", {"type": "click", "button": "left"}, False),
    ("Right click", {"type": "click", "button": "right"}, False),
    ("Middle click", {"type": "click", "button": "middle"}, False),
    ("Double click", {"type": "double_click", "button": "left"}, False),
    ("Scroll up", {"type": "scroll", "dx": 0, "dy": 5}, False),
    ("Scroll down", {"type": "scroll", "dx": 0, "dy": -5}, False),
    ("Key combo…", {"type": "key", "combo": "ctrl+z"}, False),
    ("Drag (press while held)", {"type": "drag", "button": "left"}, True),
    ("Move the mouse while held", {"type": "pointer_hold"}, True),
    ("Nothing", {"type": "none"}, False),
]


def choice_index(action: dict | None) -> int:
    a = action or {}
    t = a.get("type")
    for i, (_, tmpl, _) in enumerate(CHOICES):
        if tmpl["type"] != t:
            continue
        if t in ("click", "double_click", "drag") and tmpl.get("button") != a.get("button", "left"):
            continue
        if t == "scroll" and (tmpl["dy"] > 0) != (a.get("dy", -5) > 0):
            continue
        return i
    return 0


class ActionEditor(QtWidgets.QWidget):
    needs_hold = QtCore.pyqtSignal(bool)

    def __init__(self, action=None):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.combo = QtWidgets.QComboBox()
        for text, _, _ in CHOICES:
            self.combo.addItem(text)
        self.combo.setCurrentIndex(choice_index(action))
        self.keys = QtWidgets.QLineEdit((action or {}).get("combo", "ctrl+z"))
        self.keys.setPlaceholderText("keys, e.g. ctrl+z or alt+tab")
        self.hint = label("", 12, 400, C["text2"], wrap=True)
        lay.addWidget(self.combo)
        lay.addWidget(self.keys)
        lay.addWidget(self.hint)
        self.combo.currentIndexChanged.connect(self._changed)
        self._changed()

    def _changed(self, *_):
        _, tmpl, hold = CHOICES[self.combo.currentIndex()]
        self.keys.setVisible(tmpl["type"] == "key")
        self.hint.setText("Active from the moment the gesture starts until it ends, so it's a HOLD gesture." if hold else "")
        self.hint.setVisible(hold)
        self.needs_hold.emit(hold)

    def requires_hold(self) -> bool:
        return CHOICES[self.combo.currentIndex()][2]

    def action(self, kind="instant") -> dict:
        _, tmpl, _ = CHOICES[self.combo.currentIndex()]
        a = dict(tmpl)
        if a["type"] == "key":
            a["combo"] = self.keys.text().strip().lower() or "ctrl+z"
        if a["type"] == "scroll":
            a["continuous"] = kind == "hold"
        return a
