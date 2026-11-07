"""Convert legacy emg_click.py recordings (.npz) into INTERFACE §6 sessions in data/sessions/.

    python -m frontend.tools.import_legacy path\\to\\session_YYYYmmdd_HHMMSS.npz [...]

Legacy format: raw (N,2) uint16 at 1000 Hz, events (k,2) int64 = (sample index, code) with
codes 3 = rest calibration, 0 = relax, 1 = LEFT cue (index-finger tap), 2 = RIGHT cue
(middle-finger tap), 9 = end. EMG only: there was no IMU or camera, so imu/hand are NaN.

Labels: one `cue` row per tap cue window (index_tap / middle_tap) and a `none` row for the
rest calibration, the same as the new recorder. Relax gaps are left unlabelled (baseline).
There was no free-activity block in these recordings.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CODE_LABEL = {1: "index_tap", 2: "middle_tap"}
GESTURES = {"version": 1, "gestures": [
    {"id": "index_tap", "name": "Index finger tap (legacy left click)", "kind": "instant",
     "action": {"type": "click", "button": "left"}, "created": "2026-10-25T22:00:00"},
    {"id": "middle_tap", "name": "Middle finger tap (legacy right click)", "kind": "instant",
     "action": {"type": "click", "button": "right"}, "created": "2026-10-25T22:00:00"},
]}


def convert(src: Path, out_root: Path) -> Path:
    d = np.load(src)
    raw = d["raw"].astype(np.uint16)
    ev = d["events"].astype(np.int64)
    fs = int(d["fs"])
    assert fs == 1000 and raw.ndim == 2 and raw.shape[1] == 2, (fs, raw.shape)
    m = re.search(r"(\d{8}_\d{6})", src.name)
    stamp = m.group(1) if m else dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = f"{stamp}_legacytaps"
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)

    n = len(raw)
    emg_t = np.arange(n, dtype=np.float64) / fs
    dur = n / fs
    imu_t = np.arange(0, dur, 1 / 200, dtype=np.float64)
    hand_t = np.arange(0, dur, 1 / 30, dtype=np.float64)
    np.save(out / "emg.npy", raw)
    np.save(out / "emg_t.npy", emg_t)
    np.save(out / "imu.npy", np.full((len(imu_t), 6), np.nan, np.float32))
    np.save(out / "imu_t.npy", imu_t)
    np.save(out / "hand.npy", np.full((len(hand_t), 21, 3), np.nan, np.float32))
    np.save(out / "hand_t.npy", hand_t)
    np.save(out / "hand_score.npy", np.zeros(len(hand_t), np.float32))

    rows, rest = [], []
    for k in range(len(ev) - 1):
        s, c, e = ev[k][0] / fs, int(ev[k][1]), ev[k + 1][0] / fs
        if c == 3:
            rest.append([s, e])
            rows.append((s, e, "none"))
        elif c in CODE_LABEL:
            rows.append((s, e, CODE_LABEL[c]))
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "label", "source"])
        for s, e, lab in rows:
            w.writerow([f"{s:.4f}", f"{e:.4f}", lab, "cue"])

    created = dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat() if m else dt.datetime.now().isoformat()
    meta = {
        "session_id": sid, "created": created, "fs_emg": 1000, "fs_imu": 200,
        "cam_latency": None, "imu_present": False, "hand_present": False,
        "img_w": None, "img_h": None,
        "electrode_notes": "Legacy emg_click.py recording (EMG only). 2 channels on the forearm; exact pad "
                           "placement not recorded. Task: tap the index finger (LEFT cue) or middle finger (RIGHT cue).",
        "imu_notes": "no IMU in this recording (NaN)",
        "gestures_snapshot": GESTURES,
        "rest_periods": rest,
        "protocol_version": "legacy_emg_click_v1",
        "legacy_source": src.name,
        "notes": "Cue windows are 1.6 s; relax gaps 1.5-2.1 s; no free-activity 'none' block. "
                 "Times start at 0 (no host clock).",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(out / "legacy_events.npy", ev)
    n_cues = {lab: sum(r[2] == lab for r in rows) for lab in ("index_tap", "middle_tap", "none")}
    print(f"{src.name} -> {out.relative_to(REPO)}  ({dur:.0f} s, {n_cues})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(REPO / "data" / "sessions"))
    a = ap.parse_args(argv)
    for f in a.files:
        convert(Path(f), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
