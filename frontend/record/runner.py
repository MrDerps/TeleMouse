"""UI-independent step runner: advances through the plan on the host clock and logs label windows."""
from __future__ import annotations

from .plan import Step, total_seconds


class Runner:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.total = total_seconds(steps)
        self.n_trials = sum(1 for s in steps if s.kind == "cue")
        self.i = -1
        self.t_step = None
        self.t_start = None
        self.labels: list[tuple[float, float, str]] = []
        self.rest_periods: list[list[float]] = []
        self.done = False

    def start(self, now: float):
        self.t_start = now
        self._enter(0, now)

    def _enter(self, i, now):
        if self.i >= 0:
            self._close(now)
        self.i, self.t_step = i, now

    def _close(self, now):
        s = self.steps[self.i]
        if s.label is not None:
            self.labels.append((self.t_step, now, s.label))
        if s.kind == "rest":
            self.rest_periods.append([self.t_step, now])

    def tick(self, now: float) -> Step | None:
        """Call often. Returns the current step (None when finished)."""
        if self.done or self.t_start is None:
            return None
        while now - self.t_step >= self.steps[self.i].seconds:
            if self.i + 1 >= len(self.steps):
                self._close(self.t_step + self.steps[self.i].seconds)
                self.done = True
                return None
            self._enter(self.i + 1, self.t_step + self.steps[self.i].seconds)   # exact boundaries, no drift
        return self.steps[self.i]

    def abort(self, now: float):
        """Stop early. A cue cut short is dropped (its window is incomplete); other windows are kept."""
        if self.done or self.i < 0:
            return
        s = self.steps[self.i]
        if s.kind != "cue":
            self._close(now)
        self.done = True

    # progress helpers
    def step_fraction(self, now):
        s = self.steps[self.i]
        return min(1.0, (now - self.t_step) / s.seconds) if s.seconds else 1.0

    def overall_fraction(self, now):
        return min(1.0, (now - self.t_start) / self.total) if self.t_start is not None else 0.0

    def remaining(self, now):
        return max(0.0, self.total - (now - self.t_start)) if self.t_start is not None else self.total

    def trial(self):
        return max([s.trial for s in self.steps[:self.i + 1] if s.kind == "cue"] or [0])
