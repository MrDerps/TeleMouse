"""Backend test tool: record the live 5555 stream into a §6 session folder.

Used to test end to end against `python -m frontend.tools.replay --sim --truth truth.csv` before the
frontend recorder exists. The truth CSV's gesture windows become generous cue windows (+-pad seconds).
(The real recorder is the frontend's job; this is only for backend testing.)

    python -m backend.tools.record_stream --seconds 360 --truth backend/_out/sim_truth.csv --tag simrec
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq

from backend.common import zmq_io as zio
from backend.common.config import load_config, repo_path


def record(seconds: float):
    sub = zio.P.make_sub(zio.PORT_SENSORS, (b"emg", b"imu", b"hand"))
    emg, emg_t, imu, imu_t, hand, hand_t, score = [], [], [], [], [], [], []
    size = (640, 480)
    t_end = time.perf_counter() + seconds
    next_print = time.perf_counter() + 10
    while time.perf_counter() < t_end:
        if not sub.poll(100):
            continue
        topic, h, a = zio.recv(sub, copy_array=True)
        if topic == b"emg":
            emg.append(a)
            emg_t.append(h["t0"] + np.arange(len(a)) / h["fs"])
        elif topic == b"imu":
            imu.append(a)
            imu_t.append(h["t0"] + np.arange(len(a)) / h["fs"])
        elif topic == b"hand":
            size = (h.get("img_w", 640), h.get("img_h", 480))
            hand.append(a if a.shape == (21, 3) else np.full((21, 3), np.nan, np.float32))
            hand_t.append(h["t"])
            score.append(h.get("score") or 0.0)
        if time.perf_counter() > next_print:
            next_print += 10
            print(f"  recording... {seconds - (t_end - time.perf_counter()):.0f}/{seconds:.0f} s", flush=True)
    sub.close(0)
    return (np.concatenate(emg), np.concatenate(emg_t), np.concatenate(imu), np.concatenate(imu_t),
            np.stack(hand) if hand else np.zeros((0, 21, 3), np.float32), np.array(hand_t), np.array(score), size)


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--truth", required=True, help="truth CSV written by replay --sim --truth")
    ap.add_argument("--pad", type=float, default=0.7)
    ap.add_argument("--tag", default="simrec")
    ap.add_argument("--out", default=cfg["paths"]["synth"])
    a = ap.parse_args(argv)
    emg, emg_t, imu, imu_t, hand, hand_t, score, size = record(a.seconds)
    created = datetime.now()
    sid = created.strftime("%Y%m%d_%H%M%S") + f"_{a.tag}"
    out = repo_path(a.out) / sid
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.truth, newline="") as f:
        for r in csv.DictReader(f):
            ts, te = float(r["t_start"]), float(r["t_end"])
            if ts - a.pad >= emg_t[0] + 2 and te + a.pad <= emg_t[-1]:
                rows.append((ts - a.pad, te + a.pad, r["label"]))
    ids = sorted({r[2] for r in rows})
    kinds = {g: ("hold" if g.endswith("_hold") else "instant") for g in ids}
    for name, arr in [("emg", emg.astype(np.uint16)), ("emg_t", emg_t), ("imu", imu.astype(np.float32)),
                      ("imu_t", imu_t), ("hand", hand.astype(np.float32)), ("hand_t", hand_t),
                      ("hand_score", score.astype(np.float32))]:
        np.save(out / f"{name}.npy", arr)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "label", "source"])
        for ts, te, lab in rows:
            w.writerow([f"{ts:.4f}", f"{te:.4f}", lab, "cue"])
    meta = {"session_id": sid, "created": created.isoformat(timespec="seconds"), "fs_emg": 1000, "fs_imu": 200,
            "cam_latency": 0.08, "imu_present": bool(np.isfinite(imu).any()), "electrode_notes": "sim",
            "imu_notes": "sim", "rest_periods": [], "protocol_version": "0.2", "img_w": size[0], "img_h": size[1],
            "gestures_snapshot": {"version": 1, "gestures": [{"id": g, "name": g, "kind": kinds[g]} for g in ids]},
            "synthetic": True}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {len(emg)} emg, {len(imu)} imu, {len(hand)} hand frames, {len(rows)} cue windows {kinds}")


if __name__ == "__main__":
    main()
