"""Teach a gesture (new, or more examples of an existing one).

Flow: record ONLY this gesture (one block, no other gestures), then a short free-activity block. Then it
automatically retrains on EVERY recording of every defined gesture, loads the new model and starts it.
Old gestures are never re-recorded: their earlier sessions are mixed in at training time.
Validation: the backend holds out the last 20% of each gesture's reps in every session (INTERFACE v0.4).
(Optional refresher reps of other gestures exist in the code, `refresh`, but default to 0 by Jerry's choice.)
"""
from __future__ import annotations

import datetime as dt
import threading

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
from ..actions import ACTION_TYPES, BUTTONS
from ..record.plan import build_blocks, total_seconds
from ..record.runner import Runner
from ..record.writer import StreamBuffer, write_session
from . import sensor_picker
from .data import list_sessions

IMU_NOTES = "back of the right hand, palm down: +x fingertips, -y thumb, -z up"


def trainable(sessions=None):
    """-> (session ids, gesture ids): every recorded session containing a currently defined gesture, and the defined
    gestures that have recordings. Deleted gestures drop out automatically."""
    defined = [g["id"] for g in G.load()["gestures"]]
    sess = sessions if sessions is not None else list_sessions()
    ids, have = [], set()
    for s in sess:
        hit = [g for g in s["labels"] if g in defined]
        if hit:
            ids.append(s["id"])
            have.update(hit)
    return ids, [g for g in defined if g in have]


def train_everything(app, name=None, on_reply=None):
    sessions, gestures = trainable()
    if not gestures:
        return False, "No recordings yet for the defined gestures."
    sensors = sensor_picker.current()
    name = name or dt.datetime.now().strftime("auto_%m%d_%H%M%S")
    app.pending_model = name
    app.backend.control({"cmd": "train", "gestures": gestures, "sessions": sessions, "name": name,
                         "sensors": sensors, "sensor_weights": sensor_picker.weights(sensors)}, on_reply, timeout_s=5.0)
    return True, (f"training '{name}' on {len(sessions)} session(s), gestures {', '.join(gestures)}, "
                  f"sensors {sensor_picker.key(sensors)}" + (f" (EMG weight {sensor_picker.emg_weight():.0%})" if "emg" in sensors else ""))


class TeachDialog(QtWidgets.QDialog):
    def __init__(self, parent, existing: dict | None, others: list | None = None):
        super().__init__(parent)
        self.existing = existing
        self.others = others or []                  # gesture dicts already trained (for the refresher blocks)
        self.setWindowTitle(f"More examples of '{existing['id']}'" if existing else "Teach a new gesture")
        self.gid = QtWidgets.QLineEdit(existing["id"] if existing else "")
        self.gid.setPlaceholderText("snake_case, e.g. thumbs_up")
        self.gid.setEnabled(existing is None)
        self.name = QtWidgets.QLineEdit(existing.get("name", "") if existing else "")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["instant", "hold"])
        if existing:
            self.kind.setCurrentText(existing.get("kind", "instant"))
            self.kind.setEnabled(False)
        a = (existing or {}).get("action") or {"type": "click", "button": "left"}
        self.atype = QtWidgets.QComboBox()
        self.atype.addItems(ACTION_TYPES)
        self.atype.setCurrentText(a.get("type", "click"))
        self.button = QtWidgets.QComboBox()
        self.button.addItems(BUTTONS)
        self.button.setCurrentText(a.get("button", "left"))
        self.combo = QtWidgets.QLineEdit(a.get("combo", "ctrl+z"))
        self.dy = QtWidgets.QSpinBox(minimum=-50, maximum=50, value=int(a.get("dy", -5)))
        self.reps = QtWidgets.QSpinBox(minimum=8, maximum=60, value=20)
        self.free = QtWidgets.QDoubleSpinBox(minimum=15, maximum=300, value=45, suffix=" s")
        self.refresh = QtWidgets.QSpinBox(minimum=0, maximum=10, value=0)   # kept, hidden: Jerry prefers one gesture
        self.est = QtWidgets.QLabel()
        form = QtWidgets.QFormLayout(self)
        form.addRow("Gesture id", self.gid)
        form.addRow("Name", self.name)
        form.addRow("Kind", self.kind)
        form.addRow("Action", self.atype)
        form.addRow("  button", self.button)
        form.addRow("  keys", self.combo)
        form.addRow("  scroll dy", self.dy)
        form.addRow("Repetitions", self.reps)
        form.addRow("Free activity (no gestures)", self.free)
        tip = QtWidgets.QLabel("Only this gesture: do it once every time the cue appears, then normal activity. "
                               "Afterwards it retrains on ALL your recordings (old gestures included) and starts "
                               "the new model.")
        tip.setWordWrap(True)
        form.addRow(tip)
        form.addRow(self.est)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Start recording…")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        for w in (self.reps, self.free, self.refresh):
            w.valueChanged.connect(self._est)
        self.atype.currentTextChanged.connect(self._vis)
        self._vis()
        self._est()

    def _vis(self, *_):
        t = self.atype.currentText()
        lay = self.layout()
        for w, show in ((self.button, t in ("click", "double_click", "drag")), (self.combo, t == "key"),
                        (self.dy, t == "scroll")):
            w.setVisible(show)
            lay.labelForField(w).setVisible(show)

    def action(self):
        t = self.atype.currentText()
        if t in ("click", "double_click", "drag"):
            return {"type": t, "button": self.button.currentText()}
        if t == "key":
            return {"type": t, "combo": self.combo.text().strip().lower()}
        if t == "scroll":
            return {"type": t, "dx": 0, "dy": self.dy.value(), "continuous": self.kind.currentText() == "hold"}
        if t == "mode_toggle":
            return {"type": t, "mode": "pointer"}
        if t == "pointer_hold":
            return {"type": t}
        return {"type": "none"}

    def plan(self, g):
        blocks = [(g, self.reps.value())] + [(o, self.refresh.value()) for o in self.others if o["id"] != g["id"]]
        return build_blocks(blocks, rest_s=8.0, free_s=self.free.value())

    def _est(self, *_):
        steps = self.plan({"id": "_x", "name": "x", "kind": self.kind.currentText()})
        self.est.setText(f"about {total_seconds(steps) / 60:.1f} min")

    def _ok(self):
        gid = self.gid.text().strip()
        if self.atype.currentText() in ("drag", "pointer_hold") and self.kind.currentText() != "hold":
            QtWidgets.QMessageBox.warning(self, "Teach", "That action needs a HOLD gesture (kind = hold).")
            return
        if not G.ID_RE.match(gid) or gid == "none":
            QtWidgets.QMessageBox.warning(self, "Teach", "Gesture id must be snake_case (a-z, 0-9, _).")
            return
        if self.existing is None and G.get(G.load(), gid):
            QtWidgets.QMessageBox.warning(self, "Teach", f"'{gid}' already exists: select it and use 'Add examples'.")
            return
        self.accept()


class TeachFlow(QtCore.QObject):
    saved = QtCore.pyqtSignal(object)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.saved.connect(self._saved)
        self.win = None

    def start(self, parent, gid=None):
        if not self.app.sensors.snapshot()["last_msg"] or \
                __import__("time").perf_counter() - self.app.sensors.snapshot()["last_msg"] > 2:
            QtWidgets.QMessageBox.information(parent, "Teach", "Start the sensors first (Live tab).")
            return
        d = G.load()
        existing = G.get(d, gid) if gid else None
        _, trained = trainable()
        others = [G.get(d, t) for t in trained if t != gid and G.get(d, t)]
        dlg = TeachDialog(parent, existing, others)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        gid = dlg.gid.text().strip()
        g = G.get(d, gid)
        if g is None:
            g = G.add(d, gid, dlg.name.text().strip() or None, dlg.kind.currentText(), dlg.action())
        else:
            g["action"] = dlg.action()
            if dlg.name.text().strip():
                g["name"] = dlg.name.text().strip()
        G.save(d)
        self.app.gestures_changed()
        self.gid, self.snapshot = gid, d
        self.runner = Runner(dlg.plan(g))
        self.buf = StreamBuffer()
        self.buf.start()
        from ..record.ui import RecorderWindow
        self.win = RecorderWindow(self.runner, self.buf, f"teach {gid}", need=("imu", "hand"))
        self.win.finished.connect(self._finished)
        self.win.showMaximized()

    def _finished(self, completed):
        self.buf.end()
        self.win.close()
        r = self.runner
        if not any(l[2] != "none" for l in r.labels):
            self.buf.stop()
            self.app.log("teach: cancelled, nothing saved")
            return
        notes = self.app.electrode_notes
        tag = f"teach_{self.gid}" + ("" if completed else "_partial")

        def work():
            out = []
            try:
                out.append(write_session(self.buf, r.labels, r.rest_periods, tag, self.snapshot, notes, IMU_NOTES))
                self.saved.emit(out)
            except Exception as e:
                self.saved.emit(e)
            finally:
                self.buf.stop()
        threading.Thread(target=work, daemon=True).start()

    def _saved(self, out):
        if isinstance(out, Exception):
            QtWidgets.QMessageBox.critical(None, "Teach", f"Saving failed: {out}")
            return
        self.app.sessions_changed()
        warn = [w for _, s in out for w in s["warnings"] if "clipped" not in w or "emg" in sensor_picker.current()]
        self.app.log("teach: saved " + ", ".join(p.name for p, _ in out))
        ok, info = train_everything(self.app)
        msg = ("Recording saved." + ("\n\n" + "\n".join("⚠ " + w for w in warn) if warn else "") +
               f"\n\n{info}.\n\nThe new model loads and starts by itself when training finishes "
               "(about a minute; progress is in the Train tab).")
        QtWidgets.QMessageBox.information(None, "Teach", msg)
