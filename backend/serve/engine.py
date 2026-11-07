"""Live inference: raw 5555 messages -> the SAME StreamFeaturizer as training -> TCN -> probs + events."""
from __future__ import annotations

import logging

import numpy as np
import torch

from backend.common.config import deep_update
from backend.data.features import D, StreamFeaturizer
from backend.eval.events import EventDetector

log = logging.getLogger(__name__)


class LiveEngine:
    def __init__(self, cfg: dict, model, meta: dict, norm, rest_std):
        fc = meta.get("feature_cfg", {})
        if "hand" in fc and "lag_s" not in fc["hand"]:  # models trained before hand lag existed
            fc = deep_update(fc, {"hand": {"lag_s": 0.0}})
        self.cfg = deep_update(cfg, fc)  # features exactly as at training time
        self.model, self.meta, self.norm = model, meta, norm
        self.classes = meta["classes"]
        self.W = int(meta.get("live_window_frames", 96))
        self.sf = StreamFeaturizer(self.cfg, rest_std)
        self.Z = np.zeros((D, self.W), np.float32)
        self.M = np.zeros((3, self.W), np.float32)
        self.det = EventDetector.from_config(self.classes, meta["kinds"], cfg, meta.get("thresholds", {}),
                                             meta.get("events", {}))
        self.n_frames = 0

    def set_rest_std(self, s):
        self.sf.emg.set_rest_std(s)

    def push(self, topic: bytes, h: dict, arr):
        if topic == b"emg" and arr is not None and len(arr):
            self.sf.push_emg(int(h["index0"]), float(h["t0"]), arr)
        elif topic == b"imu" and arr is not None and len(arr):
            self.sf.push_imu(float(h["t0"]), arr)
        elif topic == b"hand":
            self.sf.hand.set_image_size(h.get("img_w"), h.get("img_h"))
            lm = arr if arr is not None and arr.shape == (21, 3) else np.full((21, 3), np.nan, np.float32)
            score = h.get("score") or 0.0
            self.sf.push_hand([float(h["t"])], lm[None], [score if arr is not None and len(arr) else 0.0])

    @torch.no_grad()
    def step(self):
        """Pull every ready frame, run the model once over the last W frames.
        Returns (frames [(t, p)], events [dict])."""
        T, X, M = self.sf.pull()
        n = len(T)
        if n == 0:
            return [], []
        n_keep = min(n, self.W)
        T, X, M = T[-n_keep:], X[-n_keep:], M[-n_keep:]  # (only if we fell far behind)
        Z = self.norm(X, M).T
        self.Z = np.concatenate([self.Z[:, n_keep:], Z], 1)
        self.M = np.concatenate([self.M[:, n_keep:], M.T.astype(np.float32)], 1)
        logits = self.model(torch.from_numpy(self.Z[None]), torch.from_numpy(self.M[None]))
        P = torch.softmax(logits[0, :, -n_keep:], 0).T.numpy()
        frames, events = [], []
        for t, p in zip(T, P):
            frames.append((float(t), p))
            events.extend(self.det.step(float(t), p))
        self.n_frames += n_keep
        return frames, events
