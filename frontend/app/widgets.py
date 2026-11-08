"""Small shared widgets."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
OK, BAD, OFF = "#3fb950", "#f85149", "#6e7681"


def light(text, state):
    """state: True ok / False bad / None off"""
    c = OK if state else (OFF if state is None else BAD)
    return f"<span style='color:{c}; font-size:16pt'>●</span> {text}"


class ProbBars(pg.PlotWidget):
    """Horizontal bars of the backend's class probabilities."""

    def __init__(self):
        super().__init__()
        self.setXRange(0, 1)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.getPlotItem().getAxis("bottom").setTicks([[(0, "0%"), (0.5, "50%"), (1, "100%")]])
        from .theme import font
        for ax in ("left", "bottom"):
            self.getPlotItem().getAxis(ax).setStyle(tickFont=font(15))
        self.bars = None
        self.classes = []
        self.setMinimumHeight(140)

    def set(self, classes, p):
        if not classes:
            self.clear()
            self.bars, self.classes = None, []
            return
        if list(classes) != self.classes or self.bars is None:
            self.clear()
            self.classes = list(classes)
            from . import data
            from .ai_viz import class_color
            y = np.arange(len(classes))
            gi = [i - 1 if classes[0] == "none" else i for i in range(len(classes))]
            brushes = ["#5C5C66" if c == "none" else class_color(g) for c, g in zip(classes, gi)]
            self.bars = pg.BarGraphItem(x0=0, y=y, height=0.62, width=np.zeros(len(classes)), brushes=brushes, pen=None)
            self.addItem(self.bars)
            names = ["nothing" if c == "none" else data.gesture_label(c) for c in classes]
            self.getPlotItem().getAxis("left").setTicks([[(i, n) for i, n in enumerate(names)]])
            self.setYRange(-0.6, len(classes) - 0.4)
        self.bars.setOpts(width=np.asarray(p, float))


class CameraView(QtWidgets.QLabel):
    """Shows the in-process camera frame with landmarks (or a landmark-only sketch)."""

    def __init__(self):
        super().__init__(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)  # fill the panel, never grow it
        self.setStyleSheet("background:#0E0E10; color:#5C5C66; border-radius: 8px;")
        self.setText("no camera")

    def show_frame(self, frame, items):
        """items: [(landmarks (21,3), (b,g,r), label)]"""
        import cv2
        img = frame.copy()
        h, w = img.shape[:2]
        for lm, color, label in items:
            if lm is None or not len(lm):
                continue
            pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
            for a, b in HAND_EDGES:
                cv2.line(img, pts[a], pts[b], color, 2)
            for p in pts:
                cv2.circle(img, p, 3, color, -1)
            cv2.putText(img, label, (pts[0][0] - 30, min(h - 8, pts[0][1] + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                         QtCore.Qt.TransformationMode.FastTransformation))

    def show_landmarks(self, lm):
        w, h = max(self.width(), 320), max(self.height(), 240)
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtGui.QColor("#0E0E10"))
        if lm is not None:
            qp = QtGui.QPainter(pm)
            qp.setPen(QtGui.QPen(QtGui.QColor("#6FD6C3"), 2))
            pts = [QtCore.QPointF(x * w, y * h) for x, y, _ in lm]
            for a, b in HAND_EDGES:
                qp.drawLine(pts[a], pts[b])
            qp.end()
        self.setPixmap(pm)
