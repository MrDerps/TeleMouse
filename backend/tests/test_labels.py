import numpy as np

from backend.data.dataset import build_session
from backend.data.labels import IGNORE
from backend.data.session import load_truth

CLASSES = ["none", "swipe_left", "swipe_right", "fist", "pinch_hold"]
KINDS = {"swipe_left": "instant", "swipe_right": "instant", "fist": "instant", "pinch_hold": "hold"}


def test_localiser_matches_truth(sess, cfg):
    d = build_session(sess, CLASSES, KINDS, cfg)
    truth = load_truth(sess.path)
    trials = sorted(d.trials, key=lambda t: t["cue"][0])
    assert len(trials) == len(truth) == 16
    assert not any(t["fallback"] for t in trials)
    for t, g in zip(trials, truth):
        assert t["label"] == g["label"]
        # localised span overlaps the true motion well
        inter = min(t["span"][1], g["t_end"]) - max(t["span"][0], g["t_start"])
        union = max(t["span"][1], g["t_end"]) - min(t["span"][0], g["t_start"])
        assert inter / union > 0.6, (t, g)


def test_label_array_structure(sess, cfg):
    d = build_session(sess, CLASSES, KINDS, cfg)
    for t in d.trials:
        a, b = t["frames"]
        assert (d.y[a:b] == CLASSES.index(t["label"])).all()
        assert d.y[a - 1] == IGNORE and d.y[b] == IGNORE
    assert (d.y == 0).mean() > 0.7
    assert len(d.none_blocks) == 1  # free activity (rest calibration is < 20 s)


def test_excluded_gesture_is_ignored_not_none(sess, cfg):
    d = build_session(sess, ["none", "swipe_left"], {"swipe_left": "instant"}, cfg)
    for r in sess.labels:
        if r["label"] in ("fist", "pinch_hold"):
            m = (d.T >= r["t_start"]) & (d.T < r["t_end"])
            assert (d.y[m] == IGNORE).all()


def test_fallback_when_no_activity(sess, cfg):
    s = sess
    labels = [{"t_start": s.emg_t[0] + 2.0, "t_end": s.emg_t[0] + 4.0, "label": "fist", "source": "cue"}]
    s2 = type(s)(**{**s.__dict__, "labels": labels})  # a cue window inside the rest period
    d = build_session(s2, ["none", "fist"], {"fist": "instant"}, cfg)
    assert d.trials[0]["fallback"]
    span = d.trials[0]["span"]
    assert abs(0.5 * (span[0] + span[1]) - (s.emg_t[0] + 3.0)) < 0.1
