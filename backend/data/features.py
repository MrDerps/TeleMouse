"""Causal per-modality features on a common 50 Hz grid. ONE code path for training and live inference.

Everything is a stateful stream object fed chunk by chunk. Offline featurisation just feeds a whole
session through the same objects (`featurize_session`), so live == training by construction.

Grid: EMG frame j covers global EMG samples [20j, 20j+20); its time T_j is the time of its last sample.
IMU and hand features for frame T only use data stamped <= T (causal).

Feature layout per frame (D = 87), plus a (3,) presence mask [emg, imu, hand]:
  emg  [0:10)   log RMS, log MAV, log WL, ZC, SSC  x 2 channels  (rest-normalised units)
  imu  [10:22)  depends on imu.mode (stored per model in feature_cfg, so old models keep theirs):
                raw (models before 2026-11-08): acc(3, g, gravity kept), acc_hp(3), gyro(3, deg/s), |acc|, |acc_hp|, |gyro|
                body (default): |jerk|, |angular accel|, 0, acc_hp(3), gyro(3), 0, |acc_hp|, |gyro|.
                     No gravity direction, so it works whichever way the arm points; motion stays relative to the hand.
                magnitude: like body but acc_hp(3) and gyro(3) zeroed: only direction-free magnitudes.
  hand [22:87)  60 wrist-relative landmarks / palm size, wrist xy (image), wrist velocity xy (1/s),
                palm size (x5). Hand features for frame T use landmarks stamped <= T - hand.lag_s.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

log = logging.getLogger(__name__)

N_EMG, N_IMU, N_HAND = 10, 12, 65
LAYOUT = {"emg": slice(0, N_EMG), "imu": slice(N_EMG, N_EMG + N_IMU),
          "hand": slice(N_EMG + N_IMU, N_EMG + N_IMU + N_HAND)}
D = N_EMG + N_IMU + N_HAND
MODALITIES = ("emg", "imu", "hand")
EMG_NAMES = [f"{f}{c}" for f in ("rms", "mav", "wl", "zc", "ssc") for c in (1, 2)]
IMU_NAMES = ["ax", "ay", "az", "ahx", "ahy", "ahz", "gx", "gy", "gz", "|a|", "|ah|", "|g|"]
HAND_NAMES = [f"lm{i}{c}" for i in range(1, 21) for c in "xyz"] + ["wx", "wy", "vx", "vy", "palm"]
FEATURE_NAMES = EMG_NAMES + IMU_NAMES + HAND_NAMES
EPS_T = 1e-4


# ----------------------------------------------------------------------------- EMG
class EmgFilter:
    """Causal 20 Hz high-pass + 60 Hz notch (as in legacy/emg_click.py)."""

    def __init__(self, cfg: dict):
        e = cfg["emg"]
        hp = butter(e["hp_order"], e["hp_hz"], "highpass", fs=e["fs"], output="sos")
        b, a = iirnotch(e["notch_hz"], e["notch_q"], fs=e["fs"])
        self.sos = np.vstack([hp, tf2sos(b, a)])
        self.zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if len(x) == 0:
            return x.reshape(0, 2)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def robust_std(y: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(1.4826 * np.median(np.abs(y - np.median(y, 0)), 0), floor)


def estimate_rest_std(emg: np.ndarray, emg_t: np.ndarray, rest_periods, cfg: dict) -> np.ndarray:
    """Per-channel robust noise std (ADC counts) of the filtered EMG during rest.
    Falls back to the quietest 3 s if there is no usable rest period."""
    e = cfg["emg"]
    y = EmgFilter(cfg)(emg.astype(np.float64))
    m = np.zeros(len(y), bool)
    for a, b in rest_periods or []:
        m |= (emg_t >= a + e["rest_skip_s"]) & (emg_t <= b)
    if m.sum() >= e["fs"]:
        return robust_std(y[m], e["rest_std_floor"])
    win = 3 * e["fs"]
    skip = e["fs"]  # filter transient
    if len(y) < win + skip:
        log.warning("EMG too short for rest estimate; using whole signal")
        return robust_std(y, e["rest_std_floor"])
    a = np.abs(y[skip:]).sum(1)
    cs = np.concatenate([[0], np.cumsum(a)])
    starts = np.arange(0, len(a) - win, e["fs"] // 2)
    best = starts[np.argmin(cs[starts + win] - cs[starts])] + skip
    log.warning("no rest period in meta; using quietest 3 s at sample %d", best)
    return robust_std(y[best:best + win], e["rest_std_floor"])


class EmgStream:
    """Raw counts -> 20 ms frames of 10 Hudgins features (rest-normalised, log-compressed)."""

    def __init__(self, cfg: dict, rest_std=None, keep_recent_s: float = 10.0):
        e = cfg["emg"]
        self.cfg = cfg
        self.fs, self.L = e["fs"], e["frame_len"]
        self.thr, self.clip = e["zc_thresh"], e["log_clip"]
        self.filt = EmgFilter(cfg)
        if rest_std is None:
            log.warning("EmgStream without rest_std: using 1 count (call set_rest_std / calibrate_rest)")
            rest_std = [1.0, 1.0]
        self.rest_std = np.asarray(rest_std, np.float64)
        self.rail_lo, self.rail_hi = e.get("rail_lo", 8), e.get("rail_hi", 4087)
        self.rail_frac = e.get("rail_frac", 0.2)
        self.pending = np.zeros((0, 3))  # filtered ch1, ch2, railed flag
        self.next_idx = None    # global index of pending[0]
        self.tail = np.zeros((2, 2))
        self.recent = deque(maxlen=max(1, int(keep_recent_s * self.fs / 20)))  # filtered chunks for calibration

    def set_rest_std(self, s):
        self.rest_std = np.asarray(s, np.float64)

    def recent_filtered(self, seconds: float) -> np.ndarray:
        if not self.recent:
            return np.zeros((0, 2))
        y = np.concatenate(list(self.recent))
        return y[-int(seconds * self.fs):]

    def push(self, index0: int, t0: float, x: np.ndarray):
        """-> (T (f,), features (f,10), valid (f,)). A frame is invalid when either channel sits at an ADC rail
        for more than rail_frac of it (electrode off / floating input): EMG is then masked like a missing sensor."""
        x = np.asarray(x)
        y = self.filt(x)
        if len(y):
            self.recent.append(y)
        railed = ((x <= self.rail_lo) | (x >= self.rail_hi)).any(1).astype(np.float64) if len(x) else np.zeros(0)
        y = np.concatenate([y, railed[:, None]], 1) if len(y) else np.zeros((0, 3))
        if self.next_idx is not None and index0 != self.next_idx + len(self.pending):
            log.warning("EMG index jump %s -> %s; re-aligning frames", self.next_idx + len(self.pending), index0)
            self.next_idx = None
            self.pending = np.zeros((0, 3))
        if self.next_idx is None:
            skip = (-index0) % self.L
            y = y[skip:]
            self.next_idx = index0 + skip
        buf = np.concatenate([self.pending, y]) if len(self.pending) else y
        nf = len(buf) // self.L
        start_idx = self.next_idx
        if nf == 0:
            self.pending = buf
            return np.zeros(0), np.zeros((0, N_EMG), np.float32), np.zeros(0, bool)
        used = buf[: nf * self.L]
        valid = used[:, 2].reshape(nf, self.L).mean(1) <= self.rail_frac
        used = used[:, :2]
        self.pending = buf[nf * self.L:]
        self.next_idx = start_idx + nf * self.L
        last_idx = start_idx + self.L * np.arange(1, nf + 1) - 1
        T = t0 + (last_idx - index0) / self.fs
        full = np.concatenate([self.tail, used]) / self.rest_std
        self.tail = used[-2:].copy()
        return T, self._features(full, nf), valid

    def _features(self, full: np.ndarray, nf: int) -> np.ndarray:
        L, thr = self.L, self.thr
        idx = self.L * np.arange(nf)[:, None] + np.arange(L + 2)[None, :]
        E = full[idx]                       # (nf, L+2, 2): 2 previous samples + this frame
        Y = E[:, 2:]
        rms = np.sqrt(np.mean(Y**2, 1))
        mav = np.mean(np.abs(Y), 1)
        wl = np.sum(np.abs(np.diff(E[:, 1:], axis=1)), 1) / L
        a, b = E[:, 1:L + 1], E[:, 2:L + 2]
        zc = np.sum((a * b < 0) & (np.abs(a - b) >= thr), 1) / L
        c, p, n = E[:, 1:L + 1], E[:, 0:L], E[:, 2:L + 2]
        ssc = np.sum((c - p) * (c - n) >= thr**2, 1) / L
        lg = lambda v: np.log1p(np.clip(v, 0, self.clip))  # noqa: E731
        return np.concatenate([lg(rms), lg(mav), lg(wl), zc, ssc], 1).astype(np.float32)


# ----------------------------------------------------------------------------- IMU
class ImuStream:
    """200 Hz [acc g, gyro deg/s] -> causal anti-alias + gravity-removed copy; averaged per 20 ms frame."""

    def __init__(self, cfg: dict):
        i = cfg["imu"]
        self.fs, self.hop = i["fs"], cfg["frame"]["hop_s"]
        self.lp = butter(2, i["aa_hz"], "lowpass", fs=self.fs, output="sos")
        self.hp = butter(1, i["acc_hp_hz"], "highpass", fs=self.fs, output="sos")
        self.zl = self.zh = None
        self.mode = i.get("mode", "raw")
        if self.mode not in ("raw", "body", "magnitude"):
            raise ValueError(f"imu.mode must be raw, body or magnitude, not {self.mode!r}")
        self.prev = None                       # last (acc_hp, gyro) sample, for the derivative features
        self.t = np.zeros(0)
        self.f = np.zeros((0, N_IMU))
        self.v = np.zeros(0, bool)
        self.last_t = None

    def push(self, t0: float, x: np.ndarray, t: np.ndarray | None = None):
        x = np.asarray(x, np.float64).reshape(-1, 6)
        if len(x) == 0:
            return
        if t is None:
            t = t0 + np.arange(len(x)) / self.fs
        valid = np.isfinite(x).all(1)
        out = np.zeros((len(x), N_IMU))
        i = 0
        while i < len(x):  # filter contiguous valid runs; reset state across gaps
            j = i
            while j < len(x) and valid[j] == valid[i]:
                j += 1
            if valid[i]:
                seg = x[i:j]
                if self.zl is None:
                    self.zl = sosfilt_zi(self.lp)[:, :, None] * seg[0][None, None, :]
                lp, self.zl = sosfilt(self.lp, seg, axis=0, zi=self.zl)
                if self.zh is None:
                    self.zh = sosfilt_zi(self.hp)[:, :, None] * lp[0, :3][None, None, :]
                ah, self.zh = sosfilt(self.hp, lp[:, :3], axis=0, zi=self.zh)
                g = lp[:, 3:]
                mags = [np.linalg.norm(lp[:, :3], axis=1, keepdims=True),
                        np.linalg.norm(ah, axis=1, keepdims=True),
                        np.linalg.norm(g, axis=1, keepdims=True)]
                if self.mode == "raw":
                    out[i:j] = np.concatenate([lp[:, :3], ah, g] + mags, 1)
                else:
                    first = self.prev if self.prev is not None else (ah[0], g[0])
                    d_ah = np.diff(np.vstack([first[0], ah]), axis=0) * self.fs      # jerk (g/s)
                    d_g = np.diff(np.vstack([first[1], g]), axis=0) * self.fs / 100.0  # angular accel (100 deg/s^2)
                    self.prev = (ah[-1].copy(), g[-1].copy())
                    zero3 = np.zeros_like(ah)
                    body = [np.linalg.norm(d_ah, axis=1, keepdims=True), np.linalg.norm(d_g, axis=1, keepdims=True),
                            np.zeros((len(ah), 1))]
                    vec = [ah, g] if self.mode == "body" else [zero3, zero3]
                    mags[0] = np.zeros_like(mags[0])     # |acc| mixes gravity with motion: depends on arm direction
                    out[i:j] = np.concatenate(body + vec + mags, 1)
            else:
                self.zl = self.zh = None
                self.prev = None
            i = j
        self.t = np.concatenate([self.t, t])
        self.f = np.concatenate([self.f, out])
        self.v = np.concatenate([self.v, valid])
        self.last_t = float(t[-1])

    def frames(self, T: np.ndarray):
        F = len(T)
        if len(self.t) == 0 or F == 0:
            return np.zeros((F, N_IMU), np.float32), np.zeros(F, bool)
        lo = np.searchsorted(self.t, T - self.hop + EPS_T, "right")
        hi = np.searchsorted(self.t, T + EPS_T, "right")
        cs = np.concatenate([np.zeros((1, N_IMU)), np.cumsum(self.f * self.v[:, None], 0)])
        cv = np.concatenate([[0], np.cumsum(self.v)])
        cn = cv[hi] - cv[lo]
        tot = hi - lo
        ok = (cn > 0) & (cn * 2 >= tot)
        out = np.zeros((F, N_IMU))
        out[ok] = (cs[hi] - cs[lo])[ok] / cn[ok, None]
        return out.astype(np.float32), ok

    def trim(self, t_min: float):
        k = np.searchsorted(self.t, t_min)
        if k > 0:
            self.t, self.f, self.v = self.t[k:], self.f[k:], self.v[k:]


# ----------------------------------------------------------------------------- hand
class HandStream:
    """MediaPipe landmarks -> wrist-relative, palm-normalised features; hold <= hold_s, else masked."""

    def __init__(self, cfg: dict, img_w: int, img_h: int):
        h = cfg["hand"]
        self.hold, self.vclip = h["hold_s"], h["vel_clip"]
        self.lag = h.get("lag_s", 0.0)
        self.aspect = img_w / img_h
        self.t = np.zeros(0)
        self.f = np.zeros((0, N_HAND))
        self.prev = None  # (t, wrist_xy) of last valid frame
        self.last_t = None

    def push(self, t, lm, score=None):
        """t: (K,) times; lm: (K,21,3) with NaN rows for no hand (or a single frame (21,3)/(0,3))."""
        t = np.atleast_1d(np.asarray(t, np.float64))
        lm = np.asarray(lm, np.float64)
        if lm.ndim == 2:
            lm = np.full((1, 21, 3), np.nan) if lm.shape[0] != 21 else lm[None]
        if len(t) == 0:
            return
        self.last_t = float(t[-1])
        valid = np.isfinite(lm).all((1, 2))
        if score is not None:
            valid &= np.atleast_1d(np.asarray(score)) > 0
        if not valid.any():
            return
        tv, P = t[valid], lm[valid].copy()
        wrist_xy = P[:, 0, :2].copy()
        P[:, :, 0] *= self.aspect  # x in image-height units so distances are isotropic
        palm = np.maximum(np.linalg.norm(P[:, 9, :2] - P[:, 0, :2], axis=1), 1e-3)
        rel = (P[:, 1:] - P[:, :1]) / palm[:, None, None]
        # velocity vs previous valid frame (including state from earlier pushes)
        pt = np.concatenate([[self.prev[0]] if self.prev else [-np.inf], tv[:-1]])
        pw = np.concatenate([self.prev[1][None] if self.prev else np.zeros((1, 2)), wrist_xy[:-1]])
        dt = tv - pt
        ok = (dt > 0) & (dt <= self.hold)
        vel = np.zeros((len(tv), 2))
        vel[ok] = (wrist_xy[ok] - pw[ok]) / dt[ok, None]
        vel = np.clip(vel, -self.vclip, self.vclip)
        f = np.concatenate([rel.reshape(len(tv), -1), wrist_xy, vel, 5 * palm[:, None]], 1)
        self.t = np.concatenate([self.t, tv])
        self.f = np.concatenate([self.f, f])
        self.prev = (float(tv[-1]), wrist_xy[-1].copy())

    def set_image_size(self, img_w, img_h):
        if img_w and img_h:
            self.aspect = img_w / img_h

    def frames(self, T: np.ndarray):
        F = len(T)
        out = np.zeros((F, N_HAND), np.float32)
        if len(self.t) == 0 or F == 0:
            return out, np.zeros(F, bool)
        T = T - self.lag
        k = np.searchsorted(self.t, T + EPS_T, "right") - 1
        ok = (k >= 0) & (T - self.t[np.maximum(k, 0)] <= self.hold + EPS_T)
        out[ok] = self.f[k[ok]]
        return out, ok

    def trim(self, t_min: float):
        k = np.searchsorted(self.t, t_min)
        if k > 1:  # always keep at least the last frame before t_min
            self.t, self.f = self.t[k - 1:], self.f[k - 1:]


# ----------------------------------------------------------------------------- fusion
class StreamFeaturizer:
    """Push raw chunks of any size, pull aligned 50 Hz frames (T, X (F,D), M (F,3))."""

    def __init__(self, cfg: dict, rest_std=None, img_w: int | None = None, img_h: int | None = None,
                 sensors=MODALITIES, emg_valid: bool = True):
        self.cfg = cfg
        self.enabled = np.array([m in sensors for m in MODALITIES])
        self.enabled[0] &= bool(emg_valid)
        self.emg = EmgStream(cfg, rest_std)
        self.imu = ImuStream(cfg)
        self.hand = HandStream(cfg, img_w or cfg["hand"]["default_img_w"], img_h or cfg["hand"]["default_img_h"])
        self.max_wait = cfg["frame"]["max_wait_s"]
        self.pT: list[np.ndarray] = []
        self.pE: list[np.ndarray] = []
        self.pV: list[np.ndarray] = []
        self.last_T = None

    def push_emg(self, index0: int, t0: float, x):
        T, f, v = self.emg.push(index0, t0, x)
        if len(T):
            self.pT.append(T)
            self.pE.append(f)
            self.pV.append(v)
            self.last_T = float(T[-1])

    def push_imu(self, t0: float, x, t=None):
        self.imu.push(t0, x, t)

    def push_hand(self, t, lm, score=None):
        self.hand.push(t, lm, score)

    def _alive(self, last_t):
        return last_t is not None and self.last_T is not None and self.last_T - last_t < 1.0

    def pull(self, force: bool = False):
        if not self.pT:
            return np.zeros(0), np.zeros((0, D), np.float32), np.zeros((0, 3), bool)
        T = np.concatenate(self.pT)
        E = np.concatenate(self.pE)
        V = np.concatenate(self.pV)
        if force:
            n = len(T)
        else:
            ready = T <= self.last_T - self.max_wait
            lim = np.inf
            if self.enabled[1] and self._alive(self.imu.last_t):
                lim = min(lim, self.imu.last_t)
            if self.enabled[2] and self._alive(self.hand.last_t):
                lim = min(lim, self.hand.last_t + self.hand.lag)
            ready |= T <= lim + EPS_T
            n = int(np.argmin(ready)) if not ready.all() else len(T)
        Tr, Er, Vr = T[:n], E[:n], V[:n]
        self.pT, self.pE, self.pV = ([T[n:]], [E[n:]], [V[n:]]) if n < len(T) else ([], [], [])
        if n == 0:
            return np.zeros(0), np.zeros((0, D), np.float32), np.zeros((0, 3), bool)
        Fi, mi = self.imu.frames(Tr)
        Fh, mh = self.hand.frames(Tr)
        X = np.concatenate([Er, Fi, Fh], 1).astype(np.float32)
        M = np.stack([Vr, mi, mh], 1) & self.enabled[None, :]
        X = apply_mask(X, M)
        self.imu.trim(Tr[-1] - 1.0)
        self.hand.trim(Tr[-1] - 1.0)
        return Tr, X, M


def apply_mask(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Zero the features of every masked (missing / disabled) modality."""
    for k, sl in enumerate(LAYOUT.values()):
        X[~M[:, k], sl] = 0.0
    return X


def sensor_key(sensors) -> str:
    return "+".join(m for m in MODALITIES if m in sensors)


def featurize_session(sess, cfg: dict, rest_std=None, chunk_s: float | None = None, sensors=MODALITIES):
    """Run a whole session through StreamFeaturizer. chunk_s=None pushes everything at once;
    otherwise pushes interleaved chunks in time order (exactly like live)."""
    if rest_std is None:
        rest_std = estimate_rest_std(sess.emg, sess.emg_t, sess.rest_periods, cfg)
    w, h = sess.img_size(cfg)
    emg_valid = sess.meta.get("emg_valid", True) is not False
    if not emg_valid:
        log.info("%s: meta.emg_valid=false -> EMG masked for the whole session", sess.id)
    sf = StreamFeaturizer(cfg, rest_std, w, h, sensors, emg_valid)
    if len(sess.emg) == 0:
        raise ValueError(f"{sess.id}: no EMG")
    if chunk_s is None:
        sf.push_emg(0, sess.emg_t[0], sess.emg)
        sf.push_imu(None, sess.imu, sess.imu_t)
        sf.push_hand(sess.hand_t, sess.hand, sess.hand_score)
        T, X, M = sf.pull(force=True)
    else:
        outs = []
        ne = max(1, int(round(chunk_s * cfg["emg"]["fs"])))
        ni = max(1, int(round(chunk_s * cfg["imu"]["fs"])))
        ie = ii = ih = 0
        while ie < len(sess.emg):
            e1 = min(ie + ne, len(sess.emg))
            sf.push_emg(ie, sess.emg_t[ie], sess.emg[ie:e1])
            t_now = sess.emg_t[e1 - 1]
            while ii < len(sess.imu) and sess.imu_t[ii] <= t_now:
                i1 = min(ii + ni, len(sess.imu))
                sf.push_imu(None, sess.imu[ii:i1], sess.imu_t[ii:i1])
                ii = i1
            while ih < len(sess.hand) and sess.hand_t[ih] <= t_now:
                sf.push_hand(sess.hand_t[ih:ih + 1], sess.hand[ih:ih + 1], sess.hand_score[ih:ih + 1])
                ih += 1
            outs.append(sf.pull())
            ie = e1
        if ii < len(sess.imu):
            sf.push_imu(None, sess.imu[ii:], sess.imu_t[ii:])
        if ih < len(sess.hand):
            sf.push_hand(sess.hand_t[ih:], sess.hand[ih:], sess.hand_score[ih:])
        outs.append(sf.pull(force=True))
        T = np.concatenate([o[0] for o in outs])
        X = np.concatenate([o[1] for o in outs])
        M = np.concatenate([o[2] for o in outs])
    return T, X, M, np.asarray(rest_std)
