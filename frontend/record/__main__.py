"""Recorder (INTERFACE §6): rest calibration, cued trials in random order, free-activity block.

Needs a publisher on port 5555: real sensors (`python -m frontend.acquire --port COM7`) or the
simulator (`python -m frontend.tools.replay --sim`).

    python -m frontend.record --gestures swipe_left fist_hold --hold fist_hold --reps 20 --tag jerry
    python -m frontend.record                       all gestures in data/gestures.json, 20 reps
    python -m frontend.record --selftest            short automatic run against the simulator

New gesture ids are added to data/gestures.json (kind "instant" unless listed in --hold).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from shared import protocol as P

from .. import gestures as G
from .plan import build_plan
from .runner import Runner
from .writer import StreamBuffer, write_session


def print_summary(path, summary):
    print(f"\nSaved {path}")
    for k in ("seconds", "emg_rate", "imu_rate", "cam_fps", "hand_seen_pct", "dropped", "emg_index_gaps", "labels"):
        print(f"  {k:15} {summary[k]}")
    for w in summary["warnings"]:
        print(f"  WARNING: {w}")
    if not summary["warnings"]:
        print("  all sensors OK")


def resolve_gestures(ids, hold):
    d = G.load()
    if not ids:
        ids = [g["id"] for g in d["gestures"]]
        if not ids:
            sys.exit("No gestures defined. Pass --gestures id1 id2 ... (snake_case).")
    changed = False
    for gid in ids:
        if not G.get(d, gid):
            G.add(d, gid, kind="hold" if gid in (hold or []) else "instant")
            changed = True
            print(f"added gesture {gid!r} to {G.PATH}")
    if changed:
        G.save(d)
    return d, [G.get(d, gid) for gid in ids]


def run_gui(a):
    from PyQt6 import QtWidgets

    from .ui import RecorderWindow
    snapshot, gl = resolve_gestures(a.gestures, a.hold)
    steps = build_plan(gl, reps=a.reps, rest_s=a.rest, free_s=a.free)
    runner = Runner(steps)
    buf = StreamBuffer(a.port)
    buf.start()
    app = QtWidgets.QApplication(sys.argv)
    need = tuple(k for k in ("emg", "imu", "hand") if k not in (a.allow_missing or []))
    win = RecorderWindow(runner, buf, a.tag, need=need)
    result = {}

    def on_finished(completed):
        buf.end()
        result["completed"] = completed
        app.quit()
    win.finished.connect(on_finished)
    win.showMaximized()
    app.exec()
    buf.stop()
    if not runner.labels or not any(l[2] != "none" for l in runner.labels) and not result.get("completed"):
        print("Nothing recorded; no session written.")
        return 1
    tag = a.tag + ("" if result.get("completed") else "_partial")
    path, summary = write_session(buf, runner.labels, runner.rest_periods, tag, snapshot,
                                  a.electrode_notes, a.imu_notes, extra_meta={"emg_valid": not a.emg_invalid})
    print_summary(path, summary)
    if not result.get("completed"):
        print("  (stopped early: complete windows were kept)")
    print(f"\nTo share it with the backend:  git add {path.relative_to(G.REPO).as_posix()} && git commit && git push")
    return 0


def selftest(a):
    """Automatic short run against an in-process simulator on a private port; checks the written files."""
    import shutil
    import tempfile
    import threading

    import numpy as np

    from frontend.tools.replay import run_sim

    port = 25599
    sim_args = argparse.Namespace(port=port, seed=4, no_imu=False, no_hand=False, drop_rate=0.0,
                                  truth=None, quiet=True)
    stop, ready = threading.Event(), threading.Event()
    th = threading.Thread(target=run_sim, args=(sim_args, stop, ready), daemon=True)
    th.start()
    ready.wait(5)
    buf = StreamBuffer(port)
    buf.start()
    time.sleep(1.0)
    gl = [{"id": "swipe_left", "name": "Swipe left", "kind": "instant"},
          {"id": "fist_hold", "name": "Fist hold", "kind": "hold"}]
    steps = build_plan(gl, reps=2, rest_s=1.5, free_s=2.0, gap=(0.5, 0.6), cue_s=1.0, hold_cue_s=1.5, seed=1)
    r = Runner(steps)
    buf.begin()
    r.start(time.perf_counter())
    while r.tick(time.perf_counter()) is not None:
        time.sleep(0.01)
    buf.end()
    tmp = Path(tempfile.mkdtemp())
    try:
        path, s = write_session(buf, r.labels, r.rest_periods, "selftest", {"version": 1, "gestures": gl},
                                out_root=tmp)
        print_summary(path, s)
        emg, emg_t = np.load(path / "emg.npy"), np.load(path / "emg_t.npy")
        hand = np.load(path / "hand.npy")
        assert emg.dtype == np.uint16 and emg.shape[1] == 2 and len(emg) == len(emg_t)
        assert np.allclose(np.diff(emg_t), 1e-3, atol=1e-6)
        assert hand.shape[1:] == (21, 3)
        assert s["labels"] == {"fist_hold": 2, "none": 2, "swipe_left": 2}, s["labels"]
        lab = [l for l in r.labels]
        assert all(emg_t[0] <= t0 < t1 <= emg_t[-1] + 0.05 for t0, t1, _ in lab), (lab[:2], emg_t[[0, -1]])
        assert abs(s["emg_rate"] - 1000) < 10 and abs(s["imu_rate"] - 200) < 5 and s["cam_fps"] > 25, s
        try:
            from backend.data.session import load_session
            sess = load_session(path)
            print(f"  backend loader: OK ({type(sess).__name__}, emg {sess.emg.shape})")
        except ImportError:
            print("  (backend loader not importable here; skipped)")
        print("record selftest: OK")
        return 0
    finally:
        stop.set()
        buf.stop()
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gestures", nargs="*", help="gesture ids (default: all in data/gestures.json)")
    ap.add_argument("--hold", nargs="*", help="which of the NEW ids are hold gestures")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rest", type=float, default=10.0, help="rest calibration seconds (>= 10 recommended)")
    ap.add_argument("--free", type=float, default=150.0, help="free-activity 'none' block seconds (120-180)")
    ap.add_argument("--tag", default="session")
    ap.add_argument("--electrode-notes", default="")
    ap.add_argument("--imu-notes", default="back of the right hand, palm down: +x fingertips, -y thumb, -z up")
    ap.add_argument("--allow-missing", nargs="*", choices=["emg", "imu", "hand"],
                    help="start even if these sensors aren't streaming (e.g. a dry run without EMG pads)")
    ap.add_argument("--emg-invalid", action="store_true",
                    help="mark meta.emg_valid=false (no pads / bad contact), so training ignores EMG")
    ap.add_argument("--port", type=int, default=P.PORT_SENSORS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9_-]+$", a.tag):
        ap.error("--tag: letters, digits, _ and - only")
    return selftest(a) if a.selftest else run_gui(a)


if __name__ == "__main__":
    sys.exit(main())
