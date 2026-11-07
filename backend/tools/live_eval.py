"""Backend test tool: score live `gesture` events on 5556 against a simulator truth CSV.

Latency is measured in wall-clock time: event ARRIVAL (perf_counter, the shared host clock) minus the true end of
the motion (for hold gestures: the true start).

    python -m backend.tools.live_eval --seconds 180 --truth backend/_out/sim_truth_live.csv
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from backend.common import zmq_io as zio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--slack", type=float, default=0.5, help="event may come this long after the true end")
    a = ap.parse_args(argv)
    sub = zio.P.make_sub(zio.PORT_OUTPUTS, (b"gesture", b"probs", b"status"))
    events, n_probs, lags = [], 0, []
    t_start = time.perf_counter()
    t_end = t_start + a.seconds
    while time.perf_counter() < t_end:
        if sub.poll(100):
            topic, h, _ = zio.recv(sub)
            if topic == b"gesture":
                events.append({**h, "arrival": time.perf_counter()})
            elif topic == b"status":
                if "lag_ms" in h:
                    lags.append(h["lag_ms"])
            else:
                n_probs += 1
    with open(a.truth, newline="") as f:
        truth = [{"ts": float(r["t_start"]), "te": float(r["t_end"]), "id": r["label"]} for r in csv.DictReader(f)]
    truth = [g for g in truth if g["ts"] > t_start + 3 and g["te"] < t_end - 1]
    det = [e for e in events if e["kind"] in ("instant", "start") and t_start + 3 < e["t"] < t_end - 1]
    used = set()
    tp, fp, lat, confused = 0, 0, [], 0
    for e in det:
        hit = None
        for k, g in enumerate(truth):
            if k not in used and g["ts"] - 0.2 <= e["t"] <= g["te"] + a.slack:
                hit = k
                break
        if hit is None:
            fp += 1
        elif truth[hit]["id"] != e["id"]:
            confused += 1
            used.add(hit)
        else:
            used.add(hit)
            tp += 1
            g = truth[hit]
            ref = g["ts"] if e["kind"] == "start" else g["te"]
            lat.append(e["arrival"] - ref)
    minutes = a.seconds / 60
    if lags:
        print(f"server frame age when published (median of 1 Hz reports): {np.median(lags):.0f} ms")
    print(f"probs: {n_probs / a.seconds:.1f} Hz | true gestures: {len(truth)} | detections: {len(det)}")
    print(f"TP {tp}  confused {confused}  FP {fp} ({fp / minutes:.2f}/min)  recall {tp / max(len(truth), 1):.2f}  "
          f"precision {tp / max(len(det), 1):.2f}")
    if lat:
        lat = np.array(lat) * 1000
        print(f"latency (arrival - end of motion; hold: start): median {np.median(lat):.0f} ms, "
              f"p90 {np.percentile(lat, 90):.0f} ms")
    for gid in sorted({g["id"] for g in truth}):
        n = sum(g["id"] == gid for g in truth)
        hit = sum(1 for k in used if truth[k]["id"] == gid)
        print(f"  {gid:14s} true {n:3d}  matched-window {hit:3d}  detections {sum(e['id'] == gid for e in det):3d}")


if __name__ == "__main__":
    main()
