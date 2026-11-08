"""Overview (landing page)."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .theme import C
from .ui import Logo, Panel, SignalPath, button, label


class OverviewPage(QtWidgets.QScrollArea):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        outer = QtWidgets.QHBoxLayout(body)
        outer.addStretch(1)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(18)
        col.setContentsMargins(0, 36, 0, 36)
        wrap = QtWidgets.QWidget()
        wrap.setMaximumWidth(1080)
        wrap.setLayout(col)
        outer.addWidget(wrap, 8)
        outer.addStretch(1)

        col.addWidget(Logo(44), 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        col.addWidget(label("A mouse you wear. Define any hand gesture, give it an action, and control your computer "
                            "without touching it.", 26, 600, wrap=True))
        col.addWidget(label("Axion fuses forearm muscle signals (EMG), a motion sensor on the back of the hand (IMU) and "
                            "the webcam's hand tracking. You show it a gesture a few times, it learns it on this laptop in "
                            "about a minute, and from then on that gesture clicks, scrolls, drags or types for you.",
                            15, 400, C["text2"], wrap=True))
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(button("Open dashboard", "Primary", lambda: app.go("dashboard")))
        btns.addWidget(button("Set up gestures", None, lambda: app.go("gestures")))
        btns.addStretch(1)
        col.addLayout(btns)

        sp = Panel("Signal path", 1)
        sp.add(SignalPath())
        col.addWidget(sp)

        how = Panel("How it works", 2)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(24)
        for n, (t, d) in enumerate([
            ("Define", "Name a gesture (fist, swipe, pinch…) and pick what it does: click, scroll, drag, a key combo, "
                       "or move the mouse while held."),
            ("Teach", "Do it 15 times when the screen says so, then a minute of normal activity so Axion learns "
                      "what NOT to react to. It retrains on everything you have recorded."),
            ("Use", "Turn on 'Control my computer'. Gestures fire their actions; F9 stops everything instantly.")], 1):
            c = QtWidgets.QVBoxLayout()
            c.addWidget(label(f"{n:02d}", 12, 600, C["accent"]))
            c.addWidget(label(t, 17, 600))
            c.addWidget(label(d, 14, 400, C["text2"], wrap=True))
            c.addStretch(1)
            row.addLayout(c, 1)
        how.add(row)
        col.addWidget(how)

        nums = Panel("By the numbers", 3)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(14)
        facts = [("~100 ms", "from end of motion to action"), ("50 Hz", "fused sensor frames"),
                 ("3 sensors", "in any combination"), ("~1–2 min", "to train, on this laptop"),
                 ("117k", "model parameters"), ("0", "cloud calls: everything runs locally")]
        for i, (big, small) in enumerate(facts):
            c = QtWidgets.QVBoxLayout()
            c.addWidget(label(big, 26, 600))
            c.addWidget(label(small, 13, 400, C["text2"]))
            grid.addLayout(c, i // 3, i % 3)
        nums.add(grid)
        col.addWidget(nums)

        team = Panel("Team", 4)
        trow = QtWidgets.QHBoxLayout()
        for name, role in [("Jerry", "Hardware, firmware, sensing and the app"),
                           ("William", "Machine learning: features, model, training and live inference")]:
            c = QtWidgets.QVBoxLayout()
            c.addWidget(label(name, 16, 600))
            c.addWidget(label(role, 13, 400, C["text2"], wrap=True))
            trow.addLayout(c, 1)
        team.add(trow)
        col.addWidget(team)
        col.addStretch(1)

    def set_mode(self, mode):
        pass

    def refresh(self, snap):
        pass
