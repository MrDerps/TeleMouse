"""Teach a gesture (new, or more examples of an existing one).

Flow: record ONLY this gesture (one block, no other gestures), then a short free-activity block. Then it
automatically retrains on EVERY recording of every defined gesture, loads the new model and starts it.
Old gestures are never re-recorded: their earlier sessions are mixed in at training time.
Validation: the backend holds out the last 20% of each gesture's reps in every session (INTERFACE v0.4).
(Optional refresher reps of other gestures exist in the code, `refresh`, but default to 0 by Jerry's choice.)
"""
from __future__ import annotations

import datetime as dt
import re
import threading

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
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


REPS, FREE_S = 15, 45.0


def make_id(name: str, taken) -> str:
    """'Thumbs up!' -> 'thumbs_up' (unique among `taken`)."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "gesture"
    if not base[0].isalpha():
        base = "g_" + base
    gid, n = base, 2
    while gid in taken or gid == "none":
        gid, n = f"{base}_{n}", n + 1
    return gid


class TeachDialog(QtWidgets.QDialog):
    """New gesture: name, instant/hold, what it does. Existing gesture: just confirm more examples."""

    def __init__(self, parent, existing: dict | None, others: list | None = None):
        super().__init__(parent)
        from .action_editor import ActionEditor
        from .theme import C
        from .ui import Segmented, button, label
        self.existing = existing
        self.others = others or []
        self.refresh = 0                         # refresher reps of other gestures (off: Jerry prefers one gesture)
        self.setWindowTitle("Add examples" if existing else "Teach a new gesture")
        self.setMinimumWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(14)
        if existing:
            lay.addWidget(label(f"More examples of {existing.get('name') or existing['id']}", 20, 600, wrap=True))
            lay.addWidget(label("You'll do this gesture again whenever the screen says so. Then the model retrains on "
                                "everything you've recorded.", 13, 400, C["text2"], wrap=True))
        else:
            lay.addWidget(label("Teach a new gesture", 20, 600))
            lay.addWidget(label("Give it a name and pick what it does. Then you'll show it a few times.",
                                13, 400, C["text2"], wrap=True))
            lay.addWidget(label("NAME", 11, 500, C["label"], tracking=0.08))
            self.name = QtWidgets.QLineEdit()
            self.name.setPlaceholderText("e.g. Fist, Swipe up, Thumbs up")
            lay.addWidget(self.name)
            lay.addWidget(label("TYPE", 11, 500, C["label"], tracking=0.08))
            self.kind = Segmented(["Instant: do it once", "Hold: keep doing it"])
            lay.addWidget(self.kind)
            lay.addWidget(label("WHAT IT DOES", 11, 500, C["label"], tracking=0.08))
            self.editor = ActionEditor({"type": "click", "button": "left"})
            self.editor.needs_hold.connect(lambda h: h and self.kind.group.button(1).setChecked(True))
            lay.addWidget(self.editor)
        info = QtWidgets.QFrame()
        info.setObjectName("Panel")
        il = QtWidgets.QVBoxLayout(info)
        il.setContentsMargins(14, 10, 14, 10)
        il.addWidget(label(f"{REPS} repetitions  ·  {FREE_S:.0f} s of normal activity  ·  about 2 minutes", 13, 500))
        il.addWidget(label("Do it once each time the cue appears. During the normal-activity part, move naturally but "
                           "don't do the gesture.", 12, 400, C["text2"], wrap=True))
        lay.addWidget(info)
        self.err = label("", 12, 400, C["bad"], wrap=True)
        lay.addWidget(self.err)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Cancel", None, self.reject))
        row.addWidget(button("Start recording", "Primary", self._ok))
        lay.addLayout(row)

    # values -----------------------------------------------------------------
    def kind_value(self):
        if self.existing:
            return self.existing.get("kind", "instant")
        return "hold" if self.kind.index() == 1 else "instant"

    def action(self):
        return self.existing.get("action") if self.existing else self.editor.action(self.kind_value())

    def plan(self, g):
        blocks = [(g, REPS)] + [(o, self.refresh) for o in self.others if o["id"] != g["id"]]
        return build_blocks(blocks, rest_s=8.0, free_s=FREE_S)

    def _ok(self):
        if self.existing:
            self.accept()
            return
        name = self.name.text().strip()
        if not name:
            self.err.setText("Give the gesture a name.")
            return
        d = G.load()
        if any((g.get("name") or "").strip().lower() == name.lower() for g in d["gestures"]):
            self.err.setText(f"You already have a gesture called '{name}'. Use 'Add examples' on it instead.")
            return
        if self.editor.requires_hold() and self.kind_value() != "hold":
            self.err.setText("That action needs the Hold type.")
            return
        self.gid = make_id(name, {g["id"] for g in d["gestures"]})
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
            QtWidgets.QMessageBox.information(parent, "Teach", "Connect the sensors first (Dashboard → Connect sensors).")
            return
        d = G.load()
        existing = G.get(d, gid) if gid else None
        _, trained = trainable()
        others = [G.get(d, t) for t in trained if t != gid and G.get(d, t)]
        dlg = TeachDialog(parent, existing, others)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if existing:
            g, gid = existing, existing["id"]
        else:
            gid = dlg.gid
            g = G.add(d, gid, dlg.name.text().strip(), dlg.kind_value(), dlg.action())
            G.save(d)
        self.app.gestures_changed()
        self.gid, self.snapshot = gid, d
        self.runner = Runner(dlg.plan(g))
        self.buf = StreamBuffer()
        self.buf.start()
        from ..record.ui import RecorderWindow
        self.win = RecorderWindow(self.runner, self.buf, f"teach {g.get('name') or gid}", need=("imu", "hand"))
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
