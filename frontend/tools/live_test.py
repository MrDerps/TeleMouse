"""Live accuracy test: cue a gesture N times (plus a free-activity block) while the backend runs live,
then score the `gesture` events it actually sent.

    python -m frontend.tools.live_test --gesture fist --reps 15 --free 60

Needs: sensors on 5555 (app or `python -m frontend.acquire`), `python -m backend.serve` with a model
loaded. It sends `start` if inference isn't running. Turn actions OFF while testing (no clicks are
needed to score it).

Scoring (on the shared host clock):
  * a cue counts as DETECTED if an event of that gesture arrives within [cue start - 0.5 s, cue end + 0.8 s]
    (people anticipate the cue; the end margin allows for latency)
  * every other event is a FALSE TRIGGER: during relax gaps, during the free-activity block, or a wrong gesture
  * extra events inside an already-detected cue window are counted as duplicates (also false triggers)
  * response = event arrival time - cue start (includes your reaction time)
Results are printed and saved to data/live_tests/<time>_<gesture>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from shared import protocol as P

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "live_tests"
PRE, POST = 0.5, 0.8


def score(labels, events, gesture_ids):
    """labels: [(t0, t1, label)] from the Runner; events: backend gesture headers with 'rx'.
    Only instant / start events count (hold `end` is ignored)."""
    ev = [e for e in events if e.get("kind") in ("instant", "start")]
    cues = [(a, b, l) for a, b, l in labels if l in gesture_ids]
    free = [(a, b) for a, b, l in labels if l == "none"]
    used = set()
    detected, responses, dup = [], [], 0
    for a, b, l in cues:
        hits = [i for i, e in enumerate(ev) if a - PRE <= e["rx"] <= b + POST and e.get("id") == l and i not in used]
        if hits:
            detected.append(True)
            responses.append(ev[hits[0]]["rx"] - a)
            used.update(hits[:1])
            dup += len(hits) - 1
            used.update(hits[1:])
        else:
            detected.append(False)
    false_ev = [ev[i] for i in range(len(ev)) if i not in used]
    t_all = (labels[0][0], labels[-1][1]) if labels else (0, 0)
    cue_time = sum(min(b + POST, t_all[1]) - max(a - PRE, t_all[0]) for a, b, _ in cues)
    other_min = max((t_all[1] - t_all[0] - cue_time) / 60, 1e-9)
    free_min = max(sum(b - a for a, b in free) / 60, 1e-9)
    in_free = sum(1 for e in false_ev if any(a <= e["rx"] <= b for a, b in free))
    n = len(cues)
    return {
        "cues": n, "detected": int(sum(detected)), "recall": round(sum(detected) / n, 3) if n else None,
        "false_triggers": len(false_ev) + dup, "duplicates": dup,
        "false_per_min_outside_cues": round((len(false_ev) + dup) / other_min, 2),
        "false_in_free_block": in_free, "false_per_min_free_block": round(in_free / free_min, 2) if free else None,
        "response_s_median": round(float(np.median(responses)), 3) if responses else None,
        "missed_trials": [i + 1 for i, d in enumerate(detected) if not d],
    }


class EventCollector:
    def __init__(self):
        self.events, self.status = [], {}
        self.active = False
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self._stop.set()
        self.th.join(1)

    def _run(self):
        sub = P.make_sub(P.PORT_EVENTS, topics=(P.TOPIC_GESTURE, P.TOPIC_STATUS))
        while not self._stop.is_set():
            if sub.poll(100):
                topic, h, _ = P.recv(sub)
                if topic == P.TOPIC_GESTURE and self.active:
                    self.events.append(dict(h, rx=time.perf_counter()))
                elif topic == P.TOPIC_STATUS:
                    self.status = h
        sub.close(0)


def print_report(r, model, gesture):
    print(f"\n=== live accuracy: {gesture}  (model {model}) ===")
    print(f"  detected       {r['detected']}/{r['cues']}  (recall {r['recall']})")
    print(f"  false triggers {r['false_triggers']}  ({r['false_per_min_outside_cues']}/min outside cues; "
          f"{r['false_in_free_block']} in the free-activity block = {r['false_per_min_free_block']}/min; "
          f"{r['duplicates']} duplicates)")
    print(f"  response       median {r['response_s_median']} s from cue onset (includes your reaction time)")
    if r["missed_trials"]:
        print(f"  missed trials  {r['missed_trials']}")


def main(argv=None):
    from PyQt6 import QtWidgets

    from ..record.plan import build_plan
    from ..record.runner import Runner
    from ..record.ui import RecorderWindow
    from ..record.writer import StreamBuffer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gesture", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--free", type=float, default=60.0)
    a = ap.parse_args(argv)

    cc = P.ControlClient(timeout_s=2.0)
    st = cc.request({"cmd": "status"})
    if not st.get("ok"):
        sys.exit("backend offline: start `python -m backend.serve` and load a model first")
    if not st.get("model"):
        sys.exit("no model loaded on the backend (Train tab, or `load`)")
    missing = [g for g in a.gesture if g not in st.get("classes", [])]
    if missing:
        sys.exit(f"model {st['model']} doesn't know {missing}; its classes are {st.get('classes')}")
    if st.get("state") != "running":
        print("starting inference:", cc.request({"cmd": "start"}, timeout_s=5))

    gl = [{"id": g, "name": g.replace("_", " "), "kind": "instant"} for g in a.gesture]
    runner = Runner(build_plan(gl, reps=a.reps, rest_s=3.0, free_s=a.free))
    col = EventCollector()
    col.start()
    buf = StreamBuffer()            # only used for the sensor-status line and the start check
    buf.start()
    app = QtWidgets.QApplication(sys.argv)
    win = RecorderWindow(runner, buf, f"LIVE TEST ({st['model']})")
    done = {}

    def finished(ok):
        col.active = False
        done["ok"] = ok
        app.quit()
    win.finished.connect(finished)
    orig = win.keyPressEvent

    def key(e):                      # start collecting exactly when the run starts
        orig(e)
        if win.started:
            col.active = True
    win.keyPressEvent = key
    win.showMaximized()
    app.exec()
    time.sleep(1.0)                  # late events for the last cue
    col.stop()
    buf.stop()
    if not runner.labels:
        print("cancelled")
        return 1
    r = score(runner.labels, col.events, set(a.gesture))
    r.update(model=st["model"], gesture=a.gesture, completed=bool(done.get("ok")),
             sensors=col.status.get("sensors"), when=dt.datetime.now().isoformat(timespec="seconds"),
             events=[{k: e.get(k) for k in ("id", "kind", "conf", "t", "rx")} for e in col.events],
             labels=runner.labels)
    print_report(r, st["model"], ", ".join(a.gesture))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(a.gesture)}.json"
    path.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"  saved {path.relative_to(REPO)}")
    return 0


def _selftest():
    labels = [(0, 3, "none"), (5, 7, "fist"), (9, 11, "fist"), (13, 15, "fist"), (16, 76, "none")]
    events = [{"id": "fist", "kind": "instant", "rx": 5.4}, {"id": "fist", "kind": "instant", "rx": 5.9},
              {"id": "fist", "kind": "instant", "rx": 15.6}, {"id": "fist", "kind": "instant", "rx": 40.0},
              {"id": "fist", "kind": "end", "rx": 41.0}]
    r = score(labels, events, {"fist"})
    assert r["detected"] == 2 and r["missed_trials"] == [2] and r["duplicates"] == 1, r
    assert r["false_triggers"] == 2 and r["false_in_free_block"] == 1, r
    print("live_test selftest: OK", {k: r[k] for k in ("recall", "false_triggers", "response_s_median")})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
