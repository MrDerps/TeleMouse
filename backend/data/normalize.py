"""Per-feature standardisation with training-set stats. Masked modalities stay exactly zero."""
from __future__ import annotations

import numpy as np

from backend.data.features import LAYOUT


class Normalizer:
    def __init__(self, mu=None, sd=None):
        self.mu = None if mu is None else np.asarray(mu, np.float32)
        self.sd = None if sd is None else np.asarray(sd, np.float32)

    def fit(self, X, M):
        self.mu = np.zeros(X.shape[1], np.float32)
        self.sd = np.ones(X.shape[1], np.float32)
        for k, sl in enumerate(LAYOUT.values()):
            m = M[:, k]
            if m.sum() > 10:
                self.mu[sl] = X[m, sl].mean(0)
                self.sd[sl] = np.maximum(X[m, sl].std(0), 1e-3)
        return self

    def __call__(self, X, M):
        Z = (X - self.mu) / self.sd
        for k, sl in enumerate(LAYOUT.values()):
            Z[~M[:, k], sl] = 0.0
        return np.clip(Z, -10, 10).astype(np.float32)

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mu"], d["sd"])
