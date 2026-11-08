"""Visuals of the gesture network for the Train page (Data mode), all drawn with QPainter in the Axion theme:
  NetworkView   sensors -> per-sensor branches -> fused temporal layers -> gesture outputs (outputs glow live)
  CurveChart    learning curve from model.json train.history (train / val loss, val event-F1)
  ConfusionMap  held-out frames: true gesture (rows) vs what the network said (columns), row-normalised
  LiveTrace     the network's live output per gesture over the last few seconds
"""
from __future__ import annotations

import time
from collections import deque

from PyQt6 import QtCore, QtGui, QtWidgets

from . import data
from .theme import C, SENSOR_COLOR, font

PALETTE = ["#5B9BFF", "#F59E0B", "#4ADE80", "#F472B6", "#A78BFA", "#22D3EE", "#FB7185", "#FACC15"]
FEATS = {"emg": ("EMG", "muscle", 10), "imu": ("IMU", "motion", 12), "hand": ("Camera", "hand", 65)}
AL = QtCore.Qt.AlignmentFlag


def class_color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _qc(c, alpha=None):
    q = QtGui.QColor(c)
    if alpha is not None:
        q.setAlphaF(max(0.0, min(1.0, alpha)))
    return q


def _empty(p: QtGui.QPainter, r: QtCore.QRect, text: str):
    p.setPen(_qc(C["label"]))
    p.setFont(font(13))
    p.drawText(r, AL.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap, text)


class _Canvas(QtWidgets.QWidget):
    def __init__(self, min_h=200):
        super().__init__()
        self.setMinimumHeight(min_h)
        self.m = None

    def set_model(self, m):
        self.m = m
        self.update()

    def _painter(self):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        return p


# ====================================================================== network
class NetworkView(_Canvas):
    def __init__(self):
        super().__init__(240)
        self.live = {}                     # class -> probability (smoothed)

    def set_probs(self, classes, p):
        for c, v in zip(classes, p):
            self.live[c] = 0.6 * self.live.get(c, 0.0) + 0.4 * float(v)
        self.update()

    def paintEvent(self, e):
        p = self._painter()
        m = self.m
        if not m:
            _empty(p, self.rect(), "No model yet. Teach a gesture and the network appears here.")
            return
        w, h = self.width(), self.height()
        used = [s for s in ("emg", "imu", "hand") if s in (m.get("sensors") or ["emg", "imu", "hand"])]
        mc = m.get("model_cfg") or {}
        classes = m.get("classes") or []
        foot = 22
        hh = h - foot
        cols = [w * 0.01, w * 0.24, w * 0.47, w * 0.78]
        bw = w * 0.19

        def box(x, y, bh, title, sub, color, active=True):
            r = QtCore.QRectF(x, y - bh / 2, bw, bh)
            p.setPen(QtGui.QPen(_qc(color, 1.0 if active else 0.35), 1.3))
            p.setBrush(_qc(C["tint"] if active else C["panel"]))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(_qc(C["text"] if active else C["disabled"]))
            p.setFont(font(13, 600))
            p.drawText(r.adjusted(0, 4, 0, -bh / 2), AL.AlignHCenter | AL.AlignBottom, title)
            p.setPen(_qc(C["text2"] if active else C["disabled"]))
            p.setFont(font(11))
            p.drawText(r.adjusted(0, bh / 2 + 1, 0, 0), AL.AlignHCenter | AL.AlignTop, sub)
            return r

        def link(a, b, color, width=1.3, alpha=0.8):
            p.setPen(QtGui.QPen(_qc(color, alpha), width))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            path = QtGui.QPainterPath(a)
            mx = (a.x() + b.x()) / 2
            path.cubicTo(QtCore.QPointF(mx, a.y()), QtCore.QPointF(mx, b.y()), b)
            p.drawPath(path)

        # column titles
        p.setFont(font(10, 500, 0.08))
        p.setPen(_qc(C["label"]))
        for x, t in zip(cols, ("INPUT FEATURES", "SENSOR BRANCH", "FUSED TIME LAYERS", "OUTPUTS")):
            p.drawText(QtCore.QRectF(x, 0, bw, 14), AL.AlignHCenter, t)
        top = 22
        # fused stack
        dil = mc.get("dilations") or [1, 2, 4, 8, 16]
        fx = cols[2]
        layer_h = min(26, (hh - top - 10) / max(1, len(dil)) - 6)
        stack_h = len(dil) * (layer_h + 6) - 6
        fy0 = top + (hh - top - stack_h) / 2
        fused_rects = []
        for i, d in enumerate(dil):
            r = QtCore.QRectF(fx, fy0 + i * (layer_h + 6), bw, layer_h)
            p.setPen(QtGui.QPen(_qc(C["accent"], 0.5 + 0.1 * i), 1.1))
            p.setBrush(_qc(C["accent"], 0.08 + 0.05 * i))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(_qc(C["text"]))
            p.setFont(font(11, 500))
            p.drawText(r, AL.AlignCenter, f"{mc.get('fuse_ch', 64)} ch  ·  every {d} frame{'s' if d > 1 else ''}")
            fused_rects.append(r)
        f_in = QtCore.QPointF(fx, fy0 + stack_h / 2)
        f_out = QtCore.QPointF(fx + bw, fy0 + stack_h / 2)
        # inputs + branches
        n = 3
        slot = (hh - top) / n
        for i, s in enumerate(("emg", "imu", "hand")):
            y = top + slot * (i + 0.5)
            name, what, nf = FEATS[s]
            act = s in used
            col = SENSOR_COLOR[s]
            sub = f"{nf} features · {what}"
            if s == "imu":
                mode = ((m.get("feature_cfg") or {}).get("imu") or {}).get("mode", "raw")
                sub = {"body": "hand-relative motion", "magnitude": "direction-free motion"}.get(mode, "with gravity")
            r1 = box(cols[0], y, 40, name, sub if act else "not used", col, act)
            r2 = box(cols[1], y, 40, f"{mc.get('branch_ch', 32)} filters", "learns its own patterns" if act else "off",
                     col, act)
            link(QtCore.QPointF(r1.right(), y), QtCore.QPointF(r2.left(), y), col, alpha=0.8 if act else 0.2)
            link(QtCore.QPointF(r2.right(), y), f_in, col, alpha=0.8 if act else 0.15)
        # outputs
        k = max(1, len(classes))
        oslot = (hh - top) / k
        oh = min(30, oslot - 6)
        for i, c in enumerate(classes):
            y = top + oslot * (i + 0.5)
            r = QtCore.QRectF(cols[3], y - oh / 2, w - cols[3] - 4, oh)
            col = C["disabled"] if c == "none" else class_color(i - 1)
            v = self.live.get(c, 0.0)
            link(f_out, QtCore.QPointF(r.left(), y), col, 1.0 + 3 * v, 0.25 + 0.6 * v)
            p.setPen(QtGui.QPen(_qc(col, 0.9), 1.2))
            p.setBrush(_qc(C["panel"]))
            p.drawRoundedRect(r, 6, 6)
            if v > 0.01:                                   # live fill = the network's current output
                fill = QtCore.QRectF(r.left(), r.top(), r.width() * v, r.height())
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(_qc(col, 0.35))
                p.drawRoundedRect(fill, 6, 6)
            p.setPen(_qc(C["text"]))
            p.setFont(font(12, 500))
            p.drawText(r.adjusted(10, 0, -8, 0), AL.AlignVCenter | AL.AlignLeft,
                       "nothing" if c == "none" else data.gesture_label(c))
            if self.live:
                p.setPen(_qc(C["text2"]))
                p.drawText(r.adjusted(10, 0, -8, 0), AL.AlignVCenter | AL.AlignRight, f"{v:.0%}")
        # footer
        rf = m.get("receptive_field") or 0
        p.setPen(_qc(C["label"]))
        p.setFont(font(11))
        txt = f"{(m.get('n_params') or 0):,} learned weights  ·  sees the last {rf * 0.02:.1f} s ({rf} frames at 50 Hz)"
        if self.live:
            txt += "  ·  output bars are live"
        p.drawText(QtCore.QRectF(0, h - foot + 4, w, foot - 4), AL.AlignLeft | AL.AlignVCenter, txt)
        p.end()


# ====================================================================== learning curve
class CurveChart(_Canvas):
    def paintEvent(self, e):
        p = self._painter()
        hist = ((self.m or {}).get("train") or {}).get("history") or {}
        steps = hist.get("step") or []
        if len(steps) < 2:
            _empty(p, self.rect(), "The learning curve appears after a training with a few checkpoints.")
            return
        w, h = self.width(), self.height()
        series = [("train", "mistakes on practice data", C["emg"], None),
                  ("val", "mistakes on unseen reps", C["warn"], None),
                  ("val_f1", "gestures caught (unseen)", C["ok"], None)]
        fm = QtGui.QFontMetrics(font(11))
        x, y = 36, 4
        p.setFont(font(11))
        for _, name, col, _ in series:                       # legend, wrapping onto new lines
            tw = fm.horizontalAdvance(name) + 14
            if x + tw > w - 8 and x > 36:
                x, y = 36, y + 18
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(_qc(col))
            p.drawRoundedRect(QtCore.QRectF(x, y, 10, 10), 2, 2)
            p.setPen(_qc(C["text2"]))
            p.drawText(QtCore.QPointF(x + 14, y + 9), name)
            x += tw + 18
        L, R, T, B = 36, 40, y + 34, 26
        pw, ph = w - L - R, h - T - B
        smax = max(steps)
        lmax = max([x for x in (hist.get("train") or []) + (hist.get("val") or []) if x == x] or [1.0]) * 1.05

        def X(s):
            return L + pw * s / smax

        # grid
        p.setPen(QtGui.QPen(_qc(C["divider"]), 1))
        for i in range(5):
            y = T + ph * i / 4
            p.drawLine(QtCore.QPointF(L, y), QtCore.QPointF(L + pw, y))
        p.setFont(font(10))
        p.setPen(_qc(C["label"]))
        p.drawText(QtCore.QRectF(0, T - 6, L - 6, 12), AL.AlignRight | AL.AlignVCenter, f"{lmax:.1f}")
        p.drawText(QtCore.QRectF(0, T + ph - 6, L - 6, 12), AL.AlignRight | AL.AlignVCenter, "0")
        p.drawText(QtCore.QRectF(L + pw + 6, T - 6, R, 12), AL.AlignLeft | AL.AlignVCenter, "100%")
        p.drawText(QtCore.QRectF(L + pw + 6, T + ph - 6, R, 12), AL.AlignLeft | AL.AlignVCenter, "0%")
        p.drawText(QtCore.QRectF(L, h - B + 6, pw, 14), AL.AlignCenter, f"training steps (0 – {smax})")

        norms = {"train": lambda v: v / lmax, "val": lambda v: v / lmax, "val_f1": lambda v: v}
        for key, _, col, _ in series:
            norm = norms[key]
            ys = hist.get(key) or []
            pts = [QtCore.QPointF(X(s), T + ph * (1 - norm(v))) for s, v in zip(steps, ys) if v == v]
            if len(pts) < 2:
                continue
            path = QtGui.QPainterPath(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.setPen(QtGui.QPen(_qc(col), 2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setBrush(_qc(col))
            for q in pts:
                p.drawEllipse(q, 2.5, 2.5)
        best = ((self.m or {}).get("train") or {}).get("best_step")
        if best:
            x = X(best)
            p.setPen(QtGui.QPen(_qc(C["text2"]), 1, QtCore.Qt.PenStyle.DashLine))
            p.drawLine(QtCore.QPointF(x, T), QtCore.QPointF(x, T + ph))
            p.setFont(font(10))
            p.drawText(QtCore.QRectF(x - 60, T - 18, 120, 12), AL.AlignCenter, "kept this one")
        p.end()


# ====================================================================== confusion
class ConfusionMap(_Canvas):
    def paintEvent(self, e):
        p = self._painter()
        vd = (self.m or {}).get("val_detail") or {}
        cm = vd.get("confusion")
        classes = (self.m or {}).get("classes") or []
        if not cm or len(cm) != len(classes):
            _empty(p, self.rect(), "Shown after the next training.")
            return
        names = ["nothing" if c == "none" else data.gesture_label(c) for c in classes]
        k = len(classes)
        w, h = self.width(), self.height()
        fm = QtGui.QFontMetrics(font(11))
        lw = min(110, max(fm.horizontalAdvance(n) for n in names) + 12)
        top, bottom = 20, 4
        cell = min((w - lw - 8) / k, (h - top - bottom - 18) / k)
        x0, y0 = lw, top + 18
        p.setFont(font(10, 500, 0.06))
        p.setPen(_qc(C["label"]))
        p.drawText(QtCore.QRectF(x0, 0, cell * k, 14), AL.AlignCenter, "NETWORK SAID")
        p.setFont(font(11))
        for j, n in enumerate(names):                                  # column heads
            p.setPen(_qc(C["text2"]))
            p.drawText(QtCore.QRectF(x0 + j * cell, top, cell, 16), AL.AlignCenter,
                       fm.elidedText(n, QtCore.Qt.TextElideMode.ElideRight, int(cell) - 4))
        for i, row in enumerate(cm):
            tot = sum(row) or 1
            p.setPen(_qc(C["text2"]))
            p.drawText(QtCore.QRectF(0, y0 + i * cell, lw - 8, cell), AL.AlignRight | AL.AlignVCenter,
                       fm.elidedText(names[i], QtCore.Qt.TextElideMode.ElideRight, lw - 10))
            for j, v in enumerate(row):
                f = v / tot
                r = QtCore.QRectF(x0 + j * cell + 1.5, y0 + i * cell + 1.5, cell - 3, cell - 3)
                col = C["ok"] if i == j else C["bad"]
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(_qc(col, 0.08 + 0.75 * f) if v else _qc(C["button"]))
                p.drawRoundedRect(r, 5, 5)
                p.setPen(_qc(C["text"] if f > 0.02 else C["disabled"]))
                p.setFont(font(12 if cell > 46 else 10, 600 if i == j else 400))
                p.drawText(r, AL.AlignCenter, f"{f:.0%}" if v else "·")
        p.end()


# ====================================================================== live outputs
class LiveTrace(_Canvas):
    SPAN = 8.0

    def __init__(self):
        super().__init__(160)
        self.buf = deque(maxlen=600)       # (wall time, {class: p})
        self.classes = []
        self._last_t = None
        self.thr = {}

    def set_model(self, m):
        self.thr = {g: (t or {}).get("on_thr") for g, t in ((m or {}).get("thresholds") or {}).items()}
        self.update()

    def push(self, t, classes, p):
        if t == self._last_t:
            return
        self._last_t = t
        self.classes = list(classes)
        self.buf.append((time.perf_counter(), dict(zip(classes, map(float, p)))))
        self.update()

    def paintEvent(self, e):
        p = self._painter()
        now = time.perf_counter()
        if not self.buf or now - self.buf[-1][0] > 2.0:
            _empty(p, self.rect(), "Start recognising on the dashboard to watch the network think, live.")
            return
        w, h = self.width(), self.height()
        L, T, B = 4, 22, 18
        pw, ph = w - L - 4, h - T - B
        p.setPen(QtGui.QPen(_qc(C["divider"]), 1))
        for i in range(3):
            y = T + ph * i / 2
            p.drawLine(QtCore.QPointF(L, y), QtCore.QPointF(L + pw, y))
        gest = [c for c in self.classes if c != "none"]
        for gi, c in enumerate(gest):                  # firing thresholds (dashed)
            th = self.thr.get(c)
            if isinstance(th, (int, float)):
                y = T + ph * (1 - th)
                p.setPen(QtGui.QPen(_qc(class_color(gi), 0.45), 1, QtCore.Qt.PenStyle.DashLine))
                p.drawLine(QtCore.QPointF(L, y), QtCore.QPointF(L + pw, y))
        for gi, c in enumerate(gest):
            pts = [QtCore.QPointF(L + pw * (1 - (now - t) / self.SPAN), T + ph * (1 - d.get(c, 0.0)))
                   for t, d in self.buf if now - t <= self.SPAN]
            if len(pts) < 2:
                continue
            path = QtGui.QPainterPath(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.setPen(QtGui.QPen(_qc(class_color(gi)), 2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        x = L
        p.setFont(font(11))
        for gi, c in enumerate(gest):
            name = data.gesture_label(c)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(_qc(class_color(gi)))
            p.drawRoundedRect(QtCore.QRectF(x, 4, 10, 10), 2, 2)
            p.setPen(_qc(C["text2"]))
            p.drawText(QtCore.QPointF(x + 14, 13), name)
            x += QtGui.QFontMetrics(font(11)).horizontalAdvance(name) + 30
        p.setPen(_qc(C["label"]))
        p.setFont(font(10))
        p.drawText(QtCore.QRectF(L, h - B + 4, pw, 12), AL.AlignLeft, f"{self.SPAN:.0f} s ago")
        p.drawText(QtCore.QRectF(L, h - B + 4, pw, 12), AL.AlignRight, "now")
        p.end()
