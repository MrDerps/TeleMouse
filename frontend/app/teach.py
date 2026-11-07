""""Teach a new gesture": record ONLY the new gesture (plus a few refresher reps of the gestures you already
have), then retrain on everything you've ever recorded and load the new model. Old gestures are never
re-recorded in full.

Why refresher reps: they let the model compare the new gesture with the old ones under today's electrode
placement, and they make the held-out check contain every gesture.

Why two session files: the backend validates on the NEWEST session. The last ~third of the run (a few reps of
every gesture + a short free block) is saved as a separate `..._check` session, so the bigger part trains
(including the new gesture) and the check session measures accuracy on everything.
"""
from __future__ import annotations

import datetime as dt
import threading

from PyQt6 import QtCore, QtWidgets

from .. import gestures as G
from ..actions import ACTION_TYPES, BUTTONS
from ..record.plan import build_plan, total_seconds
from ..record.runner import Runner
from ..record.writer import StreamBuffer, write_session
from .tab_train import list_sessions


def trainable(sessions=None):
    """-> (session ids, gesture ids): every recorded session containing a gesture that is currently defined in
    data/gestures.json, and those gestures. Deleted gestures and legacy recordings drop out automatically."""
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
    name = name or dt.datetime.now().strftime("auto_%m%d_%H%M")
    app.pending_model = name
    app.backend.control({"cmd": "train", "gestures": gestures, "sessions": sessions, "name": name},
                        on_reply, timeout_s=5.0)
    return True, f"training '{name}' on {len(sessions)} session(s): {', '.join(gestures)}"


class TeachDialog(QtWidgets.QDialog):
    def __init__(self, parent, existing):
        super().__init__(parent)
        self.setWindowTitle("Teach a new gesture")
        self.gid = QtWidgets.QLineEdit()
        self.gid.setPlaceholderText("snake_case, e.g. thumbs_up")
        self.name = QtWidgets.QLineEdit()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["instant", "hold"])
        self.atype = QtWidgets.QComboBox()
        self.atype.addItems(ACTION_TYPES)
        self.button = QtWidgets.QComboBox()
        self.button.addItems(BUTTONS)
        self.combo = QtWidgets.QLineEdit("ctrl+z")
        self.dy = QtWidgets.QSpinBox(minimum=-50, maximum=50, value=-5)
        self.reps = QtWidgets.QSpinBox(minimum=8, maximum=60, value=20)
        self.refresh = QtWidgets.QSpinBox(minimum=0, maximum=20, value=4)
        self.free = QtWidgets.QDoubleSpinBox(minimum=0, maximum=300, value=60, suffix=" s")
        self.auto = QtWidgets.QCheckBox("Retrain on everything and load the new model when the recording ends")
        self.auto.setChecked(True)
        self.est = QtWidgets.QLabel()
        form = QtWidgets.QFormLayout(self)
        form.addRow("Gesture id", self.gid)
        form.addRow("Name", self.name)
        form.addRow("Kind", self.kind)
        form.addRow("Action", self.atype)
        form.addRow("  button", self.button)
        form.addRow("  keys", self.combo)
        form.addRow("  scroll dy", self.dy)
        form.addRow("Reps of the new gesture", self.reps)
        form.addRow("Refresher reps of each existing gesture", self.refresh)
        form.addRow("Free activity", self.free)
        ex = QtWidgets.QLabel("Existing gestures (refreshed): " + (", ".join(existing) if existing else "none yet"))
        ex.setWordWrap(True)
        form.addRow(ex)
        form.addRow(self.auto)
        form.addRow(self.est)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Start recording…")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.existing = existing
        for w in (self.reps, self.refresh, self.free):
            w.valueChanged.connect(self._est)
        self.atype.currentTextChanged.connect(self._vis)
        self.atype.setCurrentText("click")
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
        return {"type": "none"}

    def plans(self, new_g, existing_g):
        """-> (steps part 1 = training, steps part 2 = check). Check: ~1/4 of the new reps + half the refreshers."""
        n_check = max(3, self.reps.value() // 4)
        r1 = {new_g["id"]: self.reps.value() - n_check}
        r2 = {new_g["id"]: n_check}
        for g in existing_g:
            r1[g["id"]] = self.refresh.value() - self.refresh.value() // 2
            r2[g["id"]] = max(1, self.refresh.value() // 2) if self.refresh.value() else 0
        gl = [new_g] + existing_g
        free = self.free.value()
        return (build_plan(gl, r1, rest_s=10.0, free_s=free * 0.65),
                build_plan(gl, r2, rest_s=6.0, free_s=max(20.0, free * 0.35)))

    def _est(self, *_):
        fake = {"id": "_new", "name": "new", "kind": "instant"}
        ex = [{"id": e, "name": e, "kind": "instant"} for e in self.existing]
        a, b = self.plans(fake, ex)
        self.est.setText(f"about {(total_seconds(a) + total_seconds(b)) / 60:.1f} min of recording")

    def _ok(self):
        gid = self.gid.text().strip()
        if not G.ID_RE.match(gid) or gid == "none":
            QtWidgets.QMessageBox.warning(self, "Teach", "Gesture id must be snake_case (a-z, 0-9, _).")
            return
        self.accept()


class TeachFlow(QtCore.QObject):
    saved = QtCore.pyqtSignal(object)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.saved.connect(self._saved)
        self.win = None

    def start(self, parent):
        _, existing = trainable()
        dlg = TeachDialog(parent, existing)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        gid = dlg.gid.text().strip()
        d = G.load()
        g = G.get(d, gid)
        if g is None:
            g = G.add(d, gid, dlg.name.text().strip() or None, dlg.kind.currentText(), dlg.action())
        else:
            if QtWidgets.QMessageBox.question(parent, "Teach", f"'{gid}' already exists. Record more examples of it?") \
                    != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            g["action"] = dlg.action()
        G.save(d)
        self.app.gestures_changed()
        self.app.gtab.reload()
        existing_g = [G.get(d, e) for e in existing if e != gid and G.get(d, e)]
        p1, p2 = dlg.plans(g, existing_g)
        self.n1 = len(p1)
        self.gid, self.auto, self.snapshot = gid, dlg.auto.isChecked(), d
        self.runner = Runner(p1 + p2)
        self.buf = StreamBuffer()
        self.buf.start()
        from ..record.ui import RecorderWindow
        self.win = RecorderWindow(self.runner, self.buf, f"teach {gid}")
        self.win.finished.connect(self._finished)
        self.win.showMaximized()

    def _finished(self, completed):
        self.buf.end()
        self.win.close()
        r = self.runner
        if not any(l[2] not in ("none",) for l in r.labels):
            self.buf.stop()
            self.app.log("teach: cancelled, nothing saved")
            return
        boundary = r.step_starts[self.n1] if len(r.step_starts) > self.n1 else None
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        notes = self.app.rec.enotes.text()
        imu_notes = "back of the right hand, palm down: +x fingertips, -y thumb, -z up"

        def work():
            out = []
            try:
                if boundary is None:            # stopped before the check part: one session only
                    out.append(write_session(self.buf, r.labels, r.rest_periods, f"teach_{self.gid}_partial",
                                             self.snapshot, notes, imu_notes, stamp=stamp))
                else:
                    l1 = [l for l in r.labels if l[1] <= boundary + 1e-6]
                    l2 = [l for l in r.labels if l[0] >= boundary - 1e-6]
                    rp1 = [p for p in r.rest_periods if p[1] <= boundary + 1e-6]
                    rp2 = [p for p in r.rest_periods if p[0] >= boundary - 1e-6]
                    out.append(write_session(self.buf, l1, rp1, f"teach_{self.gid}", self.snapshot, notes,
                                             imu_notes, t_range=(-1e18, boundary), stamp=stamp))
                    if l2:
                        out.append(write_session(self.buf, l2, rp2, f"teach_{self.gid}_check", self.snapshot, notes,
                                                 imu_notes, t_range=(boundary, 1e18), stamp=stamp))
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
        warn = [w for _, s in out for w in s["warnings"]]
        names = ", ".join(p.name for p, _ in out)
        self.app.log(f"teach: saved {names}")
        msg = f"Saved {names}." + ("\n\n" + "\n".join("⚠ " + w for w in warn) if warn else "")
        if self.auto:
            ok, info = train_everything(self.app)
            msg += f"\n\n{info}.\nThe model loads automatically when training finishes (Train tab shows progress)."
        QtWidgets.QMessageBox.information(None, "Teach a new gesture", msg)
