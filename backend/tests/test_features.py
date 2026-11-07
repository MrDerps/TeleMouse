import copy

import numpy as np
import pytest

from backend.data.features import D, LAYOUT, EmgStream, HandStream, featurize_session


def test_shapes_and_grid(sess, cfg):
    T, X, M, rest = featurize_session(sess, cfg)
    assert X.shape == (len(T), D) and M.shape == (len(T), 3)
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert np.allclose(np.diff(T), cfg["frame"]["hop_s"], atol=1e-9)
    assert len(T) == len(sess.emg) // cfg["emg"]["frame_len"]
    assert M[:, 0].all() and M[:, 1].mean() > 0.99 and 0.5 < M[:, 2].mean() < 1.0
    assert rest.shape == (2,) and (rest > 0).all()
    # masked modalities are exactly zero
    assert (X[~M[:, 2], LAYOUT["hand"]] == 0).all()


@pytest.mark.parametrize("chunk", [0.02, 0.037])
def test_chunked_equals_whole(sess, cfg, chunk):
    T, X, M, _ = featurize_session(sess, cfg)
    T2, X2, M2, _ = featurize_session(sess, cfg, chunk_s=chunk)
    assert np.array_equal(M, M2)
    assert np.allclose(T, T2, atol=1e-9)
    assert np.abs(X - X2).max() < 1e-4


def _perturb_after(sess, t_c, rng):
    s = copy.copy(sess)
    s.emg = sess.emg.copy()
    s.emg[sess.emg_t >= t_c] = rng.integers(0, 4096, (int((sess.emg_t >= t_c).sum()), 2))
    s.imu = sess.imu.copy()
    s.imu[sess.imu_t >= t_c] = rng.normal(0, 5, (int((sess.imu_t >= t_c).sum()), 6))
    s.hand = sess.hand.copy()
    s.hand[sess.hand_t >= t_c] = rng.random((int((sess.hand_t >= t_c).sum()), 21, 3))
    return s


def test_causal_no_future_leakage(sess, cfg):
    rng = np.random.default_rng(0)
    T, X, M, rest = featurize_session(sess, cfg)
    for t_c in (sess.emg_t[0] + 30.0, sess.emg_t[0] + 47.3):
        s2 = _perturb_after(sess, t_c, rng)
        T2, X2, M2, _ = featurize_session(s2, cfg, rest_std=rest)  # same rest stats
        past = T < t_c - 1e-6
        assert np.array_equal(X[past], X2[past]) and np.array_equal(M[past], M2[past])
        assert not np.array_equal(X[~past][:50], X2[~past][:50])  # and the future did change


def test_emg_rest_normalised(sess, cfg):
    T, X, M, rest = featurize_session(sess, cfg)
    m = np.zeros(len(T), bool)
    for a, b in sess.rest_periods:
        m |= (T > a + 2) & (T < b)
    rms = np.expm1(X[m, 0:2])  # rest-normalised RMS ~ 1 during rest
    assert 0.7 < np.median(rms) < 1.5


def test_emg_stream_realigns_on_index_jump(cfg):
    s = EmgStream(cfg, [10, 10])
    x = np.full((40, 2), 2048, np.uint16)
    T, f = s.push(3, 0.0, x)          # first 17 samples skipped to align to multiple of 20
    assert len(T) == 1
    T, f = s.push(1000, 1.0, x)       # gap: realign, no crash
    assert len(T) == 2 and np.isfinite(f).all()


def test_hand_hold_and_mask(cfg):
    h = HandStream(cfg, 640, 480)
    h.lag = 0.0
    lm = np.random.default_rng(0).random((21, 3))
    h.push([1.00], lm[None])
    h.push([1.033], np.full((1, 21, 3), np.nan))
    f, ok = h.frames(np.array([1.0, 1.05, 1.0 + cfg["hand"]["hold_s"], 1.2, 0.9]))
    assert ok.tolist() == [True, True, True, False, False]
    assert (f[~ok] == 0).all()
