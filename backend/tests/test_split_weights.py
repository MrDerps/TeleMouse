"""M7: per-gesture stratified split across sessions, sensor weights."""
import numpy as np
import pytest

from backend.data.dataset import load_dataset, val_trials
from backend.model.augment import drop_probs, resolve_weights
from backend.tools.synth import DEFAULT_GESTURES, generate_session


@pytest.fixture(scope="module")
def teach_sessions(tmp_path_factory):
    """Teach-flow shape: each session holds ONE gesture (+ free activity); the newest holds only the last one."""
    root = tmp_path_factory.mktemp("teach")
    g = {x["id"]: x for x in DEFAULT_GESTURES}
    a = generate_session(root, [g["swipe_left"]], reps=10, free_s=25.0, seed=1, tag="teach_swipe_left")
    b = generate_session(root, [g["pinch_hold"]], reps=10, free_s=25.0, seed=2, tag="teach_pinch_hold")
    return [a, b]


def test_stratified_split_validates_every_gesture(teach_sessions, cfg):
    data, classes, _ = load_dataset(teach_sessions, None, cfg)
    for d in data:
        assert (d.split == 0).any() and (d.split == 1).any(), d.id  # every session trains AND validates
    val = {t["label"] for d in data for t in val_trials(d)}
    tr = {t["label"] for d in data for t in d.trials if d.split[t["frames"][0]] == 0}
    assert val == tr == set(classes) - {"none"}
    for d in data:  # held-out reps are the LAST ones of the session
        cues = sorted(t["cue"][0] for t in d.trials)
        vcues = sorted(t["cue"][0] for t in val_trials(d))
        assert vcues == cues[-len(vcues):]


def test_val_sessions_and_newest_mode(teach_sessions, cfg):
    data, _, _ = load_dataset(teach_sessions, None, cfg, val_sessions=[teach_sessions[0].name])
    assert (data[0].split == 1).all() and (data[1].split == 0).all()
    with pytest.raises(ValueError):
        load_dataset(teach_sessions, None, cfg, val_sessions=["nope"])
    c = {**cfg, "split": {**cfg["split"], "mode": "newest_session"}}
    data, _, _ = load_dataset(teach_sessions, None, c)
    newest = max(range(2), key=lambda i: data[i].id)
    assert (data[newest].split == 1).all() and (data[1 - newest].split == 0).all()


def test_sensor_weights_to_dropout(cfg):
    w = resolve_weights(None, cfg)
    assert w == {"emg": 0.2, "imu": 1.0, "hand": 1.0}
    acfg = {**cfg["train"]["aug"], "weights": w}
    p = drop_probs(acfg)
    assert p[1] == acfg["drop_imu"] and p[2] == acfg["drop_hand"]
    assert 0.55 < p[0] < 0.7  # low EMG weight -> EMG hidden most of the time
    full = drop_probs({**acfg, "weights": resolve_weights({"emg": 1.0}, cfg)})
    assert np.isclose(full[0], acfg["drop_emg"])
    with pytest.raises(ValueError):
        resolve_weights({"eeg": 1.0}, cfg)


def test_train_with_weights_and_zero_weight(teach_sessions, cfg, tmp_path):
    from backend.train import train
    c = {**cfg, "paths": {**cfg["paths"], "models": str(tmp_path)},
         "train": {**cfg["train"], "steps": 20, "eval_every": 20, "batch": 4}}
    _, meta = train(teach_sessions, None, "w", c, sensor_weights={"emg": 0.0, "hand": 0.5})
    assert meta["sensors"] == ["imu", "hand"] and meta["sensor_weights"] == {"imu": 1.0, "hand": 0.5}
    assert set(meta["val_detail"]["per_gesture"]) == {"swipe_left", "pinch_hold"}
