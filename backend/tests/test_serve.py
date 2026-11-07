import threading
import time

import numpy as np
import pytest

from backend.data.dataset import load_dataset
from backend.model.io import load_model, predict_full
from backend.serve.engine import LiveEngine

GESTURES = ["swipe_left", "pinch_hold"]


@pytest.fixture(scope="module")
def trained(sess, cfg, tmp_path_factory):
    from backend.train import train
    root = tmp_path_factory.mktemp("models")
    c = {**cfg, "paths": {**cfg["paths"], "models": str(root)},
         "train": {**cfg["train"], "steps": 30, "eval_every": 30, "batch": 8}}
    out, _ = train([sess.path], GESTURES, "live_test", c)
    return c, out


def test_live_engine_equals_offline(sess, trained):
    """Feeding raw 20 ms messages through LiveEngine gives the same probabilities as offline training-time
    featurisation + full-sequence inference (after the first live window has filled)."""
    cfg, out = trained
    model, meta, norm = load_model(out, cfg)
    data, _, _ = load_dataset([sess.path], GESTURES, cfg)
    d = data[0]
    P_off = predict_full(model, norm, d.X, d.M)
    eng = LiveEngine(cfg, model, meta, norm, d.rest_std)
    got_t, got_p = [], []
    ie = ii = ih = 0
    t_stop = sess.emg_t[0] + 40.0
    while sess.emg_t[ie] < t_stop:
        eng.push(b"emg", {"t0": sess.emg_t[ie], "index0": ie, "fs": 1000}, sess.emg[ie:ie + 20])
        ie += 20
        t_now = sess.emg_t[ie - 1]
        while sess.imu_t[ii] <= t_now:
            eng.push(b"imu", {"t0": sess.imu_t[ii], "index0": ii, "fs": 200}, sess.imu[ii:ii + 4])
            ii += 4
        while sess.hand_t[ih] <= t_now - 0.1:  # camera frames arrive ~100 ms late
            lm = sess.hand[ih]
            ok = np.isfinite(lm).all()
            eng.push(b"hand", {"t": sess.hand_t[ih], "score": float(sess.hand_score[ih]) if ok else 0.0,
                               "img_w": 640, "img_h": 480}, lm if ok else np.zeros((0, 3), np.float32))
            ih += 1
        frames, _ = eng.step()
        for t, p in frames:
            got_t.append(t)
            got_p.append(p)
    got_t, got_p = np.array(got_t), np.array(got_p)
    k = np.searchsorted(d.T, got_t - 0.01)  # nearest frame (times agree to float rounding)
    assert np.allclose(d.T[k], got_t, atol=1e-6)
    W = meta["live_window_frames"]
    assert np.abs(got_p[W:] - P_off[k[W:]]).max() < 1e-3
    lag = (sess.emg_t[ie - 1] - got_t[-1])
    assert lag < 0.06  # frames are not held back waiting for the late camera


def test_control_server(trained):
    from shared.protocol import ControlClient
    from backend.serve.server import Server
    cfg, out = trained
    ports = (25655, 25656, 25657)
    srv = Server(cfg, [], ports)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        cc = ControlClient(port=ports[2], timeout_s=5)
        assert cc.request({"cmd": "status"})["state"] == "idle"
        r = cc.request({"cmd": "nope"})
        assert r["ok"] is False and "unknown" in r["error"]
        assert cc.request({"cmd": "start"})["ok"] is False          # no model yet
        assert cc.request({"cmd": "load", "name": "does_not_exist"})["ok"] is False
        r = cc.request({"cmd": "load", "name": str(out)})
        assert r["ok"] and r["classes"] == ["none"] + GESTURES
        assert cc.request({"cmd": "start", "sensors": ["imu", "bogus"]})["ok"] is False
        r = cc.request({"cmd": "start", "sensors": ["imu", "emg"]})
        assert r["ok"] and r["sensors"] == ["emg", "imu"] and "recall" in r["accuracy"]
        assert cc.request({"cmd": "start"})["ok"] and cc.request({"cmd": "status"})["state"] == "running"
        assert cc.request({"cmd": "calibrate_rest", "seconds": 3})["ok"] is False  # no EMG stream
        assert cc.request({"cmd": "train", "sessions": ["nope"]})["ok"] is False
        assert cc.request({"cmd": "stop"})["ok"]
        assert cc.request({"cmd": "list_models"})["ok"]
        cc.close()
    finally:
        srv.stop_flag = True
        th.join(3)
