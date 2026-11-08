"""Axion UI building blocks: logo, panels, switch, segmented control, tags, metric tiles, signal-path diagram."""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import C, font


def label(text, px=14, weight=400, color=None, tracking=0.0, wrap=False, obj=None):
    l = QtWidgets.QLabel(text)
    l.setFont(font(px, weight, tracking))
    if color:
        l.setStyleSheet(f"color: {color};")
    if obj:
        l.setObjectName(obj)
    l.setWordWrap(wrap)
    return l


def divider():
    d = QtWidgets.QFrame()
    d.setObjectName("Divider")
    return d


def button(text, kind=None, slot=None, tip=None):
    b = QtWidgets.QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    return b


class Logo(QtWidgets.QWidget):
    """The 'A' as two strokes whose crossbar is a muscle-signal spike, plus the lowercase wordmark."""
    clicked = QtCore.pyqtSignal()

    def __init__(self, size=28, wordmark=True, parent=None):
        super().__init__(parent)
        self.s, self.wordmark = size, wordmark
        self._font = font(int(size * 0.72), 600, tracking=-0.03, tabular=False)
        w = size + (int(QtGui.QFontMetrics(self._font).horizontalAdvance("axion")) + 10 if wordmark else 0)
        self.setFixedSize(w, size)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, e):
        self.clicked.emit()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        k = self.s / 32.0
        pen = QtGui.QPen(QtGui.QColor(C["text"]), 3 * k, cap=QtCore.Qt.PenCapStyle.RoundCap,
                         join=QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        legs = QtGui.QPainterPath(QtCore.QPointF(6 * k, 27 * k))
        legs.lineTo(16 * k, 5 * k)
        legs.lineTo(26 * k, 27 * k)
        p.drawPath(legs)
        pen2 = QtGui.QPen(QtGui.QColor(C["link"]), 2.2 * k, cap=QtCore.Qt.PenCapStyle.RoundCap,
                          join=QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen2)
        pts = [(9.5, 19), (12.5, 19), (14, 15.5), (16.5, 22), (18, 19), (22.5, 19)]
        spike = QtGui.QPainterPath(QtCore.QPointF(pts[0][0] * k, pts[0][1] * k))
        for x, y in pts[1:]:
            spike.lineTo(x * k, y * k)
        p.drawPath(spike)
        if self.wordmark:
            p.setPen(QtGui.QColor(C["text"]))
            p.setFont(self._font)
            p.drawText(QtCore.QRectF(self.s + 8, 0, self.width() - self.s - 8, self.s),
                       QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, "axion")
        p.end()


class Panel(QtWidgets.QFrame):
    """Rounded panel with a numbered uppercase label: '02 · SENSORS'."""

    def __init__(self, title=None, number=None, right=None):
        super().__init__()
        self.setObjectName("Panel")
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(18, 14, 18, 16)
        self.lay.setSpacing(10)
        if title:
            head = QtWidgets.QHBoxLayout()
            t = f"{number:02d} · {title.upper()}" if number is not None else title.upper()
            head.addWidget(label(t, 11, 500, C["label"], tracking=0.08))
            head.addStretch(1)
            if right is not None:
                head.addWidget(right)
            self.lay.addLayout(head)

    def add(self, w, stretch=0):
        (self.lay.addLayout if isinstance(w, QtWidgets.QLayout) else self.lay.addWidget)(w, stretch)
        return w


class Switch(QtWidgets.QAbstractButton):
    """iOS-style toggle in the accent colour."""

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(38, 22)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        en = self.isEnabled()
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(C["accent"] if on and en else ("#2A3A55" if on else "#2A2A30")))
        p.drawRoundedRect(QtCore.QRectF(0, 0, 38, 22), 11, 11)
        p.setBrush(QtGui.QColor("#FFFFFF" if en else "#8B8B94"))
        p.drawEllipse(QtCore.QRectF(19 if on else 3, 3, 16, 16))
        p.end()


class Segmented(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(int)

    def __init__(self, options, index=0):
        super().__init__()
        self.setObjectName("SegBox")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QtWidgets.QButtonGroup(self)
        for i, o in enumerate(options):
            b = QtWidgets.QPushButton(o)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
        self.group.button(index).setChecked(True)
        self.group.idClicked.connect(self.changed.emit)

    def index(self):
        return self.group.checkedId()


def tag(text, color=None, px=10):
    c = color or C["text2"]
    l = QtWidgets.QLabel(text.upper())
    l.setFont(font(px, 600, tracking=0.08))
    l.setStyleSheet(f"color: {c}; border: 1px solid {c}; border-radius: 4px; padding: 1px 6px;")
    l.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
    return l


class Dot(QtWidgets.QWidget):
    def __init__(self, color=C["disabled"], size=8):
        super().__init__()
        self.color, self.size = color, size
        self.setFixedSize(size, size)

    def set(self, color):
        if color != self.color:
            self.color = color
            self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(self.color))
        p.drawEllipse(0, 0, self.size, self.size)
        p.end()


class ColorBar(QtWidgets.QWidget):
    def __init__(self, color, height=34):
        super().__init__()
        self.color = color
        self.setFixedSize(4, height)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(self.color))
        p.drawRoundedRect(QtCore.QRectF(0, 0, 4, self.height()), 2, 2)
        p.end()


class Metric(QtWidgets.QFrame):
    """Small stat tile: label on top, big number, unit."""

    def __init__(self, title, unit=""):
        super().__init__()
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        lay.addWidget(label(title.upper(), 11, 500, C["label"], tracking=0.08))
        row = QtWidgets.QHBoxLayout()
        self.value = label("–", 26, 600)
        row.addWidget(self.value)
        row.addWidget(label(unit, 13, 400, C["text2"]), 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        lay.addLayout(row)

    def set(self, text):
        self.value.setText(text)


class SignalPath(QtWidgets.QWidget):
    """3 sensors -> fused at 50 Hz -> gesture model -> actions."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bw, bh = min(170, w * 0.2), 40
        col_x = [w * 0.02, w * 0.3, w * 0.55, w * 0.8]
        sensors = [("EMG  ·  muscle", C["emg"]), ("IMU  ·  motion", C["imu"]), ("Camera  ·  hand", C["cam"])]
        ys = [h * 0.18, h * 0.5, h * 0.82]

        def box(x, y, text, color, filled=False):
            r = QtCore.QRectF(x, y - bh / 2, bw, bh)
            p.setPen(QtGui.QPen(QtGui.QColor(color if not filled else C["border"]), 1.2))
            p.setBrush(QtGui.QColor(C["tint"] if filled else C["panel"]))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QtGui.QColor(C["text"]))
            p.setFont(font(13, 500))
            p.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, text)
            return r

        def arrow(a: QtCore.QPointF, b: QtCore.QPointF, color):
            p.setPen(QtGui.QPen(QtGui.QColor(color), 1.4))
            path = QtGui.QPainterPath(a)
            mx = (a.x() + b.x()) / 2
            path.cubicTo(QtCore.QPointF(mx, a.y()), QtCore.QPointF(mx, b.y()), b)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setBrush(QtGui.QColor(color))
            p.drawPolygon(QtGui.QPolygonF([b, QtCore.QPointF(b.x() - 7, b.y() - 4), QtCore.QPointF(b.x() - 7, b.y() + 4)]))

        fused = box(col_x[1], h * 0.5, "fused at 50 Hz", C["accent"], True)
        model = box(col_x[2], h * 0.5, "gesture model", C["accent"], True)
        act = box(col_x[3], h * 0.5, "actions", C["accent"], True)
        for (t, c), y in zip(sensors, ys):
            r = box(col_x[0], y, t, c)
            arrow(QtCore.QPointF(r.right(), y), QtCore.QPointF(fused.left(), fused.center().y()), c)
        arrow(QtCore.QPointF(fused.right(), fused.center().y()), QtCore.QPointF(model.left(), model.center().y()), C["accent"])
        arrow(QtCore.QPointF(model.right(), model.center().y()), QtCore.QPointF(act.left(), act.center().y()), C["accent"])
        p.end()


class FlowGrid(QtWidgets.QWidget):
    """Simple responsive grid of fixed-width cards."""

    def __init__(self, card_w=300, spacing=14):
        super().__init__()
        self.card_w, self.spacing = card_w, spacing
        self.items = []
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(spacing)

    def set_items(self, widgets):
        for w in self.items:
            w.setParent(None)
        self.items = list(widgets)
        self._relayout()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def _relayout(self):
        cols = max(1, (self.width() + self.spacing) // (self.card_w + self.spacing))
        for i in reversed(range(self.grid.count())):
            self.grid.takeAt(i)
        for i, w in enumerate(self.items):
            self.grid.addWidget(w, i // cols, i % cols)
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
