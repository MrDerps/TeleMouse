"""Window sampling + augmentation on RAW features (before standardisation)."""
from __future__ import annotations

import numpy as np

from backend.data.features import LAYOUT

_E = LAYOUT["emg"].start
_I = LAYOUT["imu"].start
_H = LAYOUT["hand"].start


def _rot(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    a = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def augment(X, M, y, L, rng, acfg, log_clip):
    """X (Ls,D) raw, M (Ls,3), y (Ls,) with Ls ~ L*(1+-warp) -> resampled to length L and perturbed."""
    X = X.astype(np.float32, copy=True)
    M = M.copy()
    Ls = len(X)
    if Ls != L:  # time warp by linear resampling
        src = np.linspace(0, Ls - 1, L)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, Ls - 1)
        w = (src - i0)[:, None].astype(np.float32)
        X = X[i0] * (1 - w) + X[i1] * w
        near = np.round(src).astype(int)
        M, y = M[near] & M[i0] & M[i1], y[near]
    # EMG amplitude scaling per channel (log1p features: rms, mav, wl)
    lo, hi = acfg["emg_scale"]
    for c in range(2):
        s = rng.uniform(lo, hi)
        for f in (0, 2, 4):
            k = _E + f + c
            X[:, k] = np.log1p(np.minimum(np.expm1(X[:, k]) * s, log_clip))
    # small 3D rotation of IMU vectors and hand landmarks
    R = _rot(rng, acfg["rot_deg"]).astype(np.float32)
    for k in (0, 3, 6):  # acc, acc_hp, gyro
        X[:, _I + k:_I + k + 3] = X[:, _I + k:_I + k + 3] @ R.T
    lm = X[:, _H:_H + 60].reshape(-1, 20, 3) @ R.T
    X[:, _H:_H + 60] = lm.reshape(-1, 60)
    th = np.deg2rad(rng.uniform(-acfg["rot_deg"], acfg["rot_deg"]))
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    X[:, _H + 62:_H + 64] = X[:, _H + 62:_H + 64] @ R2.T  # wrist velocity in image plane
    # modality dropout (never all three)
    drop = [rng.random() < acfg["drop_emg"], rng.random() < acfg["drop_imu"], rng.random() < acfg["drop_hand"]]
    if all(drop):
        drop[0] = False
    for k, d in enumerate(drop):
        if d:
            M[:, k] = False
    if rng.random() < acfg["hand_gap_p"]:
        g = rng.integers(25, 100)
        s = rng.integers(0, max(1, L - g))
        M[s:s + g, 2] = False
    return X, M, y


class WindowSampler:
    """Samples training windows that lie entirely in training frames (split == 0)."""

    def __init__(self, data, cfg, rng):
        t = cfg["train"]
        self.L, self.warm = t["win_frames"], t["warmup_frames"]
        self.warp = t["aug"]["time_warp"]
        self.Lmax = int(np.ceil(self.L * (1 + self.warp))) + 1
        self.gfrac = t["gesture_window_frac"]
        self.rng = rng
        self.data = data
        self.ok_start = []
        self.spans = []
        for si, d in enumerate(data):
            tr = (d.split == 0).astype(np.int64)
            cs = np.concatenate([[0], np.cumsum(tr)])
            n = len(tr) - self.Lmax
            ok = np.zeros(max(n, 0), bool)
            if n > 0:
                ok = (cs[self.Lmax:self.Lmax + n] - cs[:n]) == self.Lmax
            self.ok_start.append(np.flatnonzero(ok))
            for t_ in d.trials:
                a, b = t_["frames"]
                if d.split[a] == 0:
                    self.spans.append((si, a, b))
        self.sess_w = np.array([len(s) for s in self.ok_start], float)
        if self.sess_w.sum() == 0:
            raise ValueError("no training windows: sessions too short or all frames in validation")
        self.sess_w /= self.sess_w.sum()

    def sample(self):
        Ls = int(round(self.L * self.rng.uniform(1 - self.warp, 1 + self.warp)))
        if self.spans and self.rng.random() < self.gfrac:
            for _ in range(10):
                si, a, b = self.spans[self.rng.integers(len(self.spans))]
                end_pos = self.rng.integers(self.warm + 10, Ls)  # span end lands after the warm-up
                s = b - end_pos
                if s >= 0 and s + self.Lmax <= len(self.data[si].T):
                    ok = self.ok_start[si]
                    k = np.searchsorted(ok, s)
                    if k < len(ok) and ok[k] == s:
                        return si, s, Ls
        si = self.rng.choice(len(self.data), p=self.sess_w)
        s = int(self.rng.choice(self.ok_start[si]))
        return si, s, Ls
