"""Per-frame probabilities -> discrete gesture events. Shared by offline evaluation and the live server.

Per gesture: EMA-smoothed probability s.
  enter when s > on_thr for >= k_on consecutive frames (and the refractory period since the last exit has passed,
        and no other gesture is active);
  exit  when s < off_thr for >= k_off consecutive frames (hysteresis + debounce).
  instant -> one `instant` event at the peak (first frame s drops after entering, or at exit);
  hold    -> `start` on enter, `end` on exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _G:
    idx: int
    kind: str
    on: float
    off: float
    ema: float = 0.0
    count: int = 0
    below: int = 0
    active: bool = False
    emitted: bool = False
    prev: float = 0.0
    peak: float = 0.0
    last_exit: float = -np.inf


@dataclass
class EventDetector:
    classes: list[str]
    kinds: dict[str, str]
    thresholds: dict[str, dict] = field(default_factory=dict)  # gid -> {"on_thr", "off_thr"}
    ema_alpha: float = 0.5
    on_thr: float = 0.6
    off_thr: float = 0.4
    k_on: int = 2
    refractory_s: float = 0.4
    k_off: int = 1

    def __post_init__(self):
        self.g = {}
        for i, c in enumerate(self.classes):
            if c == "none":
                continue
            th = self.thresholds.get(c, {})
            on = th.get("on_thr", self.on_thr)
            off = min(th.get("off_thr", self.off_thr), on)
            self.g[c] = _G(i, self.kinds.get(c, "instant"), on, off)

    @classmethod
    def from_config(cls, classes, kinds, cfg, thresholds=None, overrides=None):
        e = {**cfg["events"], **(overrides or {})}
        return cls(classes, kinds, thresholds or {}, e["ema_alpha"], e["on_thr"], e["off_thr"], e["k_on"],
                   e["refractory_s"], e.get("k_off", 1))

    def reset(self):
        self.__post_init__()

    def step(self, t: float, p) -> list[dict]:
        out = []
        a = self.ema_alpha
        any_active = any(g.active for g in self.g.values())
        for gid, g in self.g.items():
            g.prev = g.ema
            g.ema = a * float(p[g.idx]) + (1 - a) * g.ema
            s = g.ema
            if not g.active:
                g.count = g.count + 1 if s > g.on else 0
                if g.count >= self.k_on and not any_active and t - g.last_exit >= self.refractory_s:
                    g.active, g.emitted, g.peak, g.below = True, False, s, 0
                    any_active = True
                    if g.kind == "hold":
                        out.append({"t": t, "id": gid, "conf": round(s, 4), "kind": "start"})
                        g.emitted = True
                continue
            if g.kind == "instant" and not g.emitted and s < g.prev:
                out.append({"t": t, "id": gid, "conf": round(g.peak, 4), "kind": "instant"})
                g.emitted = True
            g.peak = max(g.peak, s)
            g.below = g.below + 1 if s < g.off else 0
            if g.below >= self.k_off:
                if g.kind == "hold":
                    out.append({"t": t, "id": gid, "conf": round(s, 4), "kind": "end"})
                elif not g.emitted:
                    out.append({"t": t, "id": gid, "conf": round(g.peak, 4), "kind": "instant"})
                g.active, g.count, g.last_exit = False, 0, t
        return out


def run_detector(T: np.ndarray, P: np.ndarray, det: EventDetector) -> list[dict]:
    det.reset()
    ev = []
    for t, p in zip(T, P):
        ev.extend(det.step(float(t), p))
    return ev
