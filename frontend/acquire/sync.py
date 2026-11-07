"""Camera latency calibration from sharp hand taps (INTERFACE §2).

The user taps the hand on the desk ~5 times. Each impact is a sharp spike in IMU
acceleration (accurate host time, ~1 ms) and a sudden stop of the hand in the video
(camera time = arrival stamp, late by the unknown latency). Cross-correlating the two
"impact strength" signals gives the latency:  t_true = t_frame_arrival - latency.

    python -m frontend.acquire.sync --selftest
"""
from __future__ import annotations

import numpy as np

FS = 200.0                  # common grid
LAG_MIN, LAG_MAX = 0.0, 0.35
PALM = [0, 5, 9, 13, 17]    # wrist + finger MCPs: rigid, robust to finger motion


def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _hp(x, n):
    """subtract a centred moving average (removes gravity / slow drift)"""
    return x - np.stack([_smooth(x[:, i], n) for i in range(x.shape[1])], 1)


EDGE = 0.4                  # s trimmed from each end (filter edge effects)


def imu_impact(imu_t, acc):
    """-> grid times, impact strength (smoothed |high-passed accel|), edges trimmed"""
    g = np.arange(imu_t[0], imu_t[-1], 1 / FS)
    a = np.stack([np.interp(g, imu_t, acc[:, i]) for i in range(3)], 1)
    a = np.nan_to_num(a, nan=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0)
    s = _smooth(np.linalg.norm(_hp(a, int(0.4 * FS)), axis=1), int(0.04 * FS))
    keep = (g > g[0] + EDGE) & (g < g[-1] - EDGE)
    return g[keep], s[keep]


def cam_impact(hand_t, lm, img_w=640, img_h=480):
    """lm (K,21,3), NaN rows for no-hand -> (times, impact strength, hand_ok mask).

    Impact = a sudden stop in any direction: strength = speed before - speed after (>= 0).
    Unlike plain acceleration this ignores the start of a movement (speed going up), which is
    gentle in the IMU, so it works for desk taps, slapping the other palm, etc."""
    ok = np.isfinite(lm[:, PALM, 0]).any(axis=1) if len(lm) else np.zeros(0, bool)
    if ok.sum() < 10:
        return hand_t, np.zeros(len(hand_t)), ok
    p = np.array([np.nanmean(f[PALM, :2], axis=0) if o else [np.nan, np.nan] for f, o in zip(lm, ok)])
    xy = np.stack([np.interp(hand_t, hand_t[ok], p[ok, i]) for i in range(2)], 1) * [img_w, img_h]
    v = np.diff(xy, axis=0) / np.maximum(np.diff(hand_t), 1e-3)[:, None]    # between frames k and k+1
    sp = np.linalg.norm(v, axis=1)
    s = np.zeros(len(xy))
    s[1:-1] = np.maximum(0.0, sp[:-1] - sp[1:])
    s[~ok] = 0.0                                                            # no evidence where the hand was lost
    return hand_t, s, ok


def estimate_latency(imu_t, acc, hand_t, lm, img_w=640, img_h=480):
    """-> dict(latency, corr, taps, hand_fraction, ok, message)"""
    imu_t, acc, hand_t, lm = map(np.asarray, (imu_t, acc, hand_t, lm))
    out = dict(latency=None, corr=0.0, taps=0, hand_fraction=0.0, ok=False, message="")
    if len(imu_t) < FS * 3 or len(hand_t) < 60:
        out["message"] = "not enough data (need IMU and camera for at least 3 s)"
        return out
    gi, si = imu_impact(imu_t, acc)
    tc, sc, ok = cam_impact(hand_t, lm, img_w, img_h)
    out["hand_fraction"] = float(ok.mean())
    if ok.mean() < 0.4:
        out["message"] = f"hand visible in only {100 * ok.mean():.0f}% of frames; keep it in view"
        return out
    # taps = clear IMU peaks (biggest first), at least 300 ms apart
    thr = max(np.median(si) + 6 * (np.median(np.abs(si - np.median(si))) + 1e-6), 0.3 * si.max())
    cand = [i for i in range(1, len(si) - 1) if si[i] > thr and si[i] >= si[i - 1] and si[i] >= si[i + 1]]
    peaks = []
    for i in sorted(cand, key=lambda i: -si[i]):
        if all(abs(gi[i] - gi[j]) > 0.3 for j in peaks):
            peaks.append(i)
    out["taps"] = len(peaks)
    if len(peaks) < 3:
        out["message"] = f"only {len(peaks)} clear taps found; tap the hand sharply on the desk ~5 times"
        return out
    zi = (si - si.mean()) / (si.std() + 1e-9)
    best = (-2.0, None)
    lags = np.arange(LAG_MIN, LAG_MAX + 1e-9, 0.002)
    corrs = []
    for lag in lags:
        tq = gi + lag                                  # camera stamp of the same moment
        m = (tq >= tc[0]) & (tq <= tc[-1])
        if m.sum() < FS * 2:
            corrs.append(np.nan)
            continue
        c = np.interp(tq[m], tc, sc)
        r = float(np.corrcoef(zi[m], c)[0, 1]) if c.std() > 0 else 0.0
        corrs.append(r)
        if r > best[0]:
            best = (r, lag)
    out["corr"], out["latency"] = best
    out["curve"] = (lags, np.array(corrs))
    if best[0] < 0.3:
        out["message"] = f"weak match (correlation {best[0]:.2f}); try again with sharper taps, hand fully in view"
        return out
    if best[1] in (lags[0], lags[-1]):
        out["message"] = f"best lag {best[1] * 1000:.0f} ms is at the edge of the search range; suspicious"
        return out
    out["ok"] = True
    out["message"] = f"camera latency {best[1] * 1000:.0f} ms (correlation {best[0]:.2f}, {len(peaks)} taps)"
    return out


# ----------------------------------------------------------------------------- self test
def _synthetic(latency=0.095, seed=0, n_taps=5, secs=8.0):
    rng = np.random.default_rng(seed)
    taps = np.sort(rng.uniform(1.0, secs - 1.0, n_taps))
    while n_taps > 1 and np.min(np.diff(taps)) < 0.6:
        taps = np.sort(rng.uniform(1.0, secs - 1.0, n_taps))
    imu_t = np.arange(0, secs, 1 / 200)
    acc = np.tile([0.0, 0.0, -1.0], (len(imu_t), 1)) + rng.normal(0, 0.02, (len(imu_t), 3))
    # true hand height: lifts and drops back to the desk during the 250 ms before each impact
    def height(t):
        h = np.zeros_like(t)
        for tp in taps:
            u = (t - (tp - 0.25)) / 0.25
            m = (u >= 0) & (u < 1)
            h[m] += np.sin(np.pi * u[m]) * 0.03
        return h
    for tp in taps:
        acc[:, 2] += 2.5 * np.exp(-0.5 * ((imu_t - tp) / 0.006) ** 2)          # impact
        acc[:, 2] -= 0.3 * np.exp(-0.5 * ((imu_t - tp + 0.22) / 0.03) ** 2)    # gentle lift
    cam_true = np.arange(0, secs, 1 / 30) + rng.uniform(0, 1 / 30)
    hand_t = cam_true + latency + rng.normal(0, 0.003, len(cam_true))      # arrival stamps
    lm = np.full((len(cam_true), 21, 3), 0.5, np.float32)
    lm[:, :, 1] = 0.6 - height(cam_true)[:, None] * 5 + rng.normal(0, 0.0008, (len(cam_true), 21))
    lm[rng.random(len(cam_true)) < 0.05] = np.nan
    return imu_t, acc, hand_t, lm


def _selftest():
    errs = []
    for seed, lat in [(0, 0.095), (1, 0.06), (2, 0.14), (3, 0.08)]:
        r = estimate_latency(*_synthetic(lat, seed))
        assert r["ok"], r["message"]
        errs.append(abs(r["latency"] - lat))
        print(f"  true {lat * 1000:.0f} ms -> {r['message']}")
    assert max(errs) < 0.02, errs
    r = estimate_latency(*_synthetic(0.09, 5, n_taps=1))
    assert not r["ok"]
    print("sync selftest: OK (max error %.1f ms)" % (1000 * max(errs)))


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
