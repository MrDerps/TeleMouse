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
        self.getPlotItem().getAxis("bottom").setTicks([[(0, "0"), (0.5, ".5"), (1, "1")]])
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
            y = np.arange(len(classes))
            self.bars = pg.BarGraphItem(x0=0, y=y, height=0.7, width=np.zeros(len(classes)), brush="#4ea1ff")
            self.addItem(self.bars)
            self.getPlotItem().getAxis("left").setTicks([[(i, c) for i, c in enumerate(classes)]])
            self.setYRange(-0.6, len(classes) - 0.4)
        self.bars.setOpts(width=np.asarray(p, float))


class CameraView(QtWidgets.QLabel):
    """Shows the in-process camera frame with landmarks (or a landmark-only sketch)."""

    def __init__(self):
        super().__init__(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#0d1117; color:#6e7681;")
        self.setText("no camera")

    def show_frame(self, frame, lm):
        import cv2
        img = frame.copy()
        h, w = img.shape[:2]
        if lm is not None:
            pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
            for a, b in HAND_EDGES:
                cv2.line(img, pts[a], pts[b], (80, 220, 120), 2)
            for p in pts:
                cv2.circle(img, p, 3, (40, 160, 255), -1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                         QtCore.Qt.TransformationMode.FastTransformation))

    def show_landmarks(self, lm):
        w, h = max(self.width(), 320), max(self.height(), 240)
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtGui.QColor("#0d1117"))
        if lm is not None:
            qp = QtGui.QPainter(pm)
            qp.setPen(QtGui.QPen(QtGui.QColor("#50dc78"), 2))
            pts = [QtCore.QPointF(x * w, y * h) for x, y, _ in lm]
            for a, b in HAND_EDGES:
                qp.drawLine(pts[a], pts[b])
            qp.end()
        self.setPixmap(pm)
