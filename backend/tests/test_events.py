import numpy as np

from backend.eval.events import EventDetector, run_detector
from backend.eval.metrics import event_metrics

CL = ["none", "tap", "hold"]
KI = {"tap": "instant", "hold": "hold"}


def _probs(n, bumps):
    """bumps: list of (class_idx, start, end, height)."""
    P = np.zeros((n, 3))
    P[:, 0] = 1.0
    for c, a, b, h in bumps:
        P[a:b, c] = h
        P[a:b, 0] = 1 - h
    return P


def _det(**kw):
    base = dict(ema_alpha=1.0, on_thr=0.6, off_thr=0.4, k_on=2, refractory_s=0.4)
    base.update(kw)
    return EventDetector(CL, KI, {}, **base)


def test_instant_fires_once_at_peak():
    T = np.arange(100) * 0.02
    P = _probs(100, [(1, 10, 20, 0.9)])
    P[15, 1], P[15, 0] = 0.99, 0.01  # peak at frame 15
    ev = run_detector(T, P, _det())
    assert [e["kind"] for e in ev] == ["instant"]
    assert ev[0]["id"] == "tap" and abs(ev[0]["t"] - T[16]) < 1e-9  # first frame after the peak


def test_k_on_rejects_single_frame_blip():
    T = np.arange(50) * 0.02
    ev = run_detector(T, _probs(50, [(1, 10, 11, 0.95)]), _det())
    assert ev == []


def test_hysteresis_and_hold_start_end():
    T = np.arange(200) * 0.02
    P = _probs(200, [(2, 20, 120, 0.8)])
    P[60:70, 2], P[60:70, 0] = 0.5, 0.5  # dips between off and on: must NOT end the hold
    ev = run_detector(T, P, _det())
    assert [e["kind"] for e in ev] == ["start", "end"]
    assert abs(ev[0]["t"] - T[21]) < 1e-9 and abs(ev[1]["t"] - T[120]) < 1e-9


def test_refractory_blocks_quick_retrigger():
    T = np.arange(100) * 0.02
    P = _probs(100, [(1, 10, 15, 0.9), (1, 20, 25, 0.9), (1, 60, 65, 0.9)])
    ev = run_detector(T, P, _det())
    assert len(ev) == 2  # second bump is 0.1 s after the first exit (< 0.4 s refractory)


def test_ema_smooths_noise():
    rng = np.random.default_rng(0)
    T = np.arange(500) * 0.02
    P = np.zeros((500, 3))
    P[:, 1] = np.clip(rng.normal(0.3, 0.25, 500), 0, 1)
    P[:, 0] = 1 - P[:, 1]
    raw = run_detector(T, P, _det(ema_alpha=1.0, k_on=1))
    smooth = run_detector(T, P, _det(ema_alpha=0.2, k_on=1))
    assert len(smooth) < len(raw)


def test_event_metrics_matching():
    trials = [{"label": "tap", "kind": "instant", "cue": (1.0, 3.0), "span": (1.5, 1.8)},
              {"label": "tap", "kind": "instant", "cue": (5.0, 7.0), "span": (5.5, 5.8)},
              {"label": "hold", "kind": "hold", "cue": (9.0, 12.0), "span": (9.5, 11.5)}]
    events = [{"t": 1.85, "id": "tap", "kind": "instant"},
              {"t": 2.0, "id": "tap", "kind": "instant"},      # duplicate in same window -> FP
              {"t": 9.6, "id": "hold", "kind": "start"},
              {"t": 11.6, "id": "hold", "kind": "end"},         # ends are not counted as detections
              {"t": 20.0, "id": "tap", "kind": "instant"}]      # in free activity -> FP
    m = event_metrics(events, trials, [(15.0, 75.0)], CL, slack=0.3)
    g = m["per_gesture"]
    assert g["tap"]["tp"] == 1 and g["tap"]["fp"] == 2 and g["tap"]["n"] == 2
    assert g["hold"]["tp"] == 1 and g["hold"]["recall"] == 1.0
    assert abs(m["fp_per_min"] - 1.0) < 1e-9
    assert abs(g["tap"]["latency_ms"] - 50) < 1e-6 and abs(g["hold"]["latency_ms"] - 100) < 1e-6


def test_k_off_debounces_hold_flicker():
    T = np.arange(200) * 0.02
    P = _probs(200, [(2, 20, 120, 0.8)])
    P[60:62, 2], P[60:62, 0] = 0.1, 0.9  # 2-frame dropout inside a hold
    # without debounce the dropout splits one hold into two (the flicker seen live)
    assert [e["kind"] for e in run_detector(T, P, _det(k_off=1))] == ["start", "end", "start", "end"]
    assert [e["kind"] for e in run_detector(T, P, _det(k_off=3))] == ["start", "end"]
    ev = run_detector(T, P, _det(k_off=3))
    assert abs(ev[1]["t"] - T[122]) < 1e-9  # ends at the real end (+ debounce)


def test_calib_std_ignores_movement(cfg):
    from backend.serve.server import calib_std
    rng = np.random.default_rng(0)
    y = rng.normal(0, 6, (2000, 2))
    y[500:1100] += rng.normal(0, 300, (600, 2))  # a 0.6 s burst inside a 2 s calibration
    s = calib_std(y, cfg)
    assert (s > 4).all() and (s < 9).all()
