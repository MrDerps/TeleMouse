"""Buffers the 5555 stream in a background thread and writes data/sessions/<id>/ per INTERFACE §6.

Works with any publisher on 5555 (real acquisition, simulator, replay), because it only
subscribes. All times are the publisher's host clock (time.perf_counter(), which on Windows is
system-wide, so the recorder's own cue times share the same clock).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import threading
import time
from pathlib import Path

import numpy as np

from shared import protocol as P

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "data" / "sessions"
PROTOCOL_VERSION = "1.0"


class StreamBuffer:
    def __init__(self, port=P.PORT_SENSORS):
        self.port = port
        self.lock = threading.Lock()
        self.recording = False
        self.emg, self.emg_t0, self.emg_idx = [], [], []
        self.imu, self.imu_t0 = [], []
        self.hand, self.hand_t, self.hand_score = [], [], []
        self.img = (None, None)
        self.status = {}
        self.status_hist = []
        self.last_rx = {"emg": 0.0, "imu": 0.0, "hand": 0.0}
        self.counts = {"emg": 0, "imu": 0, "hand": 0}
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="rec-sub", daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(2)

    def begin(self):
        with self.lock:
            self.emg, self.emg_t0, self.emg_idx = [], [], []
            self.imu, self.imu_t0 = [], []
            self.hand, self.hand_t, self.hand_score = [], [], []
            self.status_hist = []
            self.recording = True

    def end(self):
        with self.lock:
            self.recording = False

    def _run(self):
        sub = P.make_sub(self.port)
        while not self._stop.is_set():
            if not sub.poll(100):
                continue
            topic, h, a = P.recv(sub, copy_array=True)
            now = time.perf_counter()
            with self.lock:
                if topic == P.TOPIC_EMG:
                    self.last_rx["emg"] = now
                    self.counts["emg"] += len(a)
                    if self.recording:
                        self.emg.append(a); self.emg_t0.append(h["t0"]); self.emg_idx.append(h["index0"])
                elif topic == P.TOPIC_IMU:
                    self.last_rx["imu"] = now
                    self.counts["imu"] += len(a)
                    if self.recording:
                        self.imu.append(a); self.imu_t0.append(h["t0"])
                elif topic == P.TOPIC_HAND:
                    self.last_rx["hand"] = now
                    self.counts["hand"] += 1
                    self.img = (h.get("img_w"), h.get("img_h"))
                    if self.recording:
                        self.hand.append(a if len(a) else np.full((21, 3), np.nan, np.float32))
                        self.hand_t.append(h["t"]); self.hand_score.append(h.get("score", 0.0))
                elif topic == P.TOPIC_STATUS:
                    self.status = h
                    if self.recording:
                        self.status_hist.append(h)
        sub.close(0)

    def live(self) -> dict:
        """What's arriving right now (for the pre-flight check)."""
        now = time.perf_counter()
        with self.lock:
            return {k: now - v < 1.0 for k, v in self.last_rx.items()} | {"status": dict(self.status)}


def _expand(chunks, t0s, fs):
    if not chunks:
        return None, None
    x = np.concatenate(chunks)
    t = np.concatenate([t0 + np.arange(len(c)) / fs for c, t0 in zip(chunks, t0s)])
    return x, t


def write_session(buf: StreamBuffer, labels: list, rest_periods: list, tag: str, gestures_snapshot: dict,
                  electrode_notes: str = "", imu_notes: str = "", extra_meta: dict | None = None,
                  out_root: Path = SESSIONS, t_range=None, stamp: str | None = None) -> tuple[Path, dict]:
    """labels: [(t_start, t_end, label)]. Returns (session dir, summary dict).
    t_range=(a, b): only write data with times in [a, b) (to split one run into several sessions).
    stamp: the YYYYmmdd_HHMMSS part of the id (default: now)."""
    a, b = t_range if t_range else (-np.inf, np.inf)

    def pick(chunks, t0s):
        k = [i for i, t0 in enumerate(t0s) if a <= t0 < b]
        return [chunks[i] for i in k], [t0s[i] for i in k]
    with buf.lock:
        ec, et = pick(buf.emg, buf.emg_t0)
        emg_idx = [buf.emg_idx[i] for i, t0 in enumerate(buf.emg_t0) if a <= t0 < b]
        emg, emg_t = _expand(ec, et, P.FS_EMG)
        imu, imu_t = _expand(*pick(buf.imu, buf.imu_t0), P.FS_IMU)
        hk = [i for i, t in enumerate(buf.hand_t) if a <= t < b]
        hand = np.stack([buf.hand[i] for i in hk]).astype(np.float32) if hk else np.zeros((0, 21, 3), np.float32)
        hand_t = np.array([buf.hand_t[i] for i in hk], np.float64)
        hand_score = np.array([buf.hand_score[i] for i in hk], np.float32)
        img_w, img_h = buf.img
        st_hist = list(buf.status_hist)
        lens = [len(c) for c in ec]
    if emg is None:
        emg, emg_t = np.zeros((0, 2), np.uint16), np.zeros(0)
    if imu is None:
        imu, imu_t = np.zeros((0, 6), np.float32), np.zeros(0)

    sid = f"{stamp or dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "emg.npy", emg.astype(np.uint16))
    np.save(out / "emg_t.npy", emg_t.astype(np.float64))
    np.save(out / "imu.npy", imu.astype(np.float32))
    np.save(out / "imu_t.npy", imu_t.astype(np.float64))
    np.save(out / "hand.npy", hand)
    np.save(out / "hand_t.npy", hand_t)
    np.save(out / "hand_score.npy", hand_score)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "label", "source"])
        for s, e, lab in labels:
            w.writerow([f"{s:.4f}", f"{e:.4f}", lab, "cue"])

    imu_valid = int(np.isfinite(imu[:, 0]).sum()) if len(imu) else 0
    hand_seen = int(np.isfinite(hand[:, 0, 0]).sum()) if len(hand) else 0
    last = st_hist[-1] if st_hist else {}
    first = st_hist[0] if st_hist else {}
    span = (emg_t[-1] - emg_t[0]) if len(emg_t) > 1 else 0.0
    idx_gaps = sum(1 for (a, b, c) in zip(emg_idx, emg_idx[1:], lens) if b != a + c)
    summary = {
        "session_id": sid, "seconds": round(float(span), 1),
        "emg_samples": int(len(emg)), "emg_rate": round(len(emg) / span, 1) if span else 0.0,
        "imu_samples": int(len(imu)), "imu_valid": imu_valid,
        "imu_rate": round(imu_valid / span, 1) if span else 0.0,
        "hand_frames": int(len(hand)), "hand_seen": hand_seen,
        "cam_fps": round(len(hand) / span, 1) if span else 0.0,
        "hand_seen_pct": round(100 * hand_seen / max(len(hand), 1), 1),
        "dropped": int(last.get("dropped", 0)) - int(first.get("dropped", 0)),
        "emg_index_gaps": idx_gaps,
        "labels": {lab: sum(1 for x in labels if x[2] == lab) for lab in sorted({x[2] for x in labels})},
    }
    warnings = []
    if summary["emg_rate"] < 990:
        warnings.append(f"EMG rate {summary['emg_rate']} Hz (expected 1000) or EMG missing")
    if summary["imu_rate"] < 190:
        warnings.append(f"IMU rate {summary['imu_rate']} Hz (expected 200) or IMU missing")
    if summary["cam_fps"] < 25:
        warnings.append(f"camera {summary['cam_fps']} fps (expected ~30) or camera missing")
    elif summary["hand_seen_pct"] < 50:
        warnings.append(f"hand visible in only {summary['hand_seen_pct']}% of frames")
    if summary["dropped"] > 0.001 * max(len(emg), 1):
        warnings.append(f"{summary['dropped']} EMG samples dropped")
    if len(emg):
        clip = np.mean((emg < 30) | (emg > 4065), axis=0)
        for ch in range(2):
            if clip[ch] > 0.2:
                warnings.append(f"EMG ch{ch + 1} clipped {100 * clip[ch]:.0f}% of the time: floating input / pads off?")
    summary["warnings"] = warnings

    meta = {
        "session_id": sid, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "fs_emg": P.FS_EMG, "fs_imu": P.FS_IMU,
        "cam_latency": last.get("cam_latency"),
        "imu_present": imu_valid > 0, "hand_present": len(hand) > 0,
        "img_w": img_w, "img_h": img_h,
        "electrode_notes": electrode_notes, "imu_notes": imu_notes,
        "gestures_snapshot": gestures_snapshot,
        "rest_periods": rest_periods,
        "protocol_version": PROTOCOL_VERSION,
        "source": last.get("source", "hardware"),
        "summary": summary,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, summary

