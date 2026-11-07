import numpy as np
import torch

from backend.data.dataset import load_dataset
from backend.data.features import D
from backend.model.augment import WindowSampler, augment
from backend.model.io import load_model, predict_full
from backend.model.tcn import build_model, n_params


def test_size_and_receptive_field(cfg):
    m = build_model(5, cfg["model"])
    assert 50_000 <= n_params(m) <= 150_000
    rf_s = m.receptive_field * cfg["frame"]["hop_s"]
    assert 1.2 <= rf_s <= 1.5
    assert cfg["train"]["live_window_frames"] >= m.receptive_field


def test_tcn_is_causal(cfg):
    torch.manual_seed(0)
    m = build_model(4, cfg["model"]).eval()
    Z = torch.randn(1, D, 300)
    M = torch.ones(1, 3, 300)
    Z2 = Z.clone()
    Z2[:, :, 200:] = torch.randn(1, D, 100) * 5
    with torch.no_grad():
        a, b = m(Z, M), m(Z2, M)
    assert torch.allclose(a[:, :, :200], b[:, :, :200], atol=1e-5)
    assert not torch.allclose(a[:, :, 200:], b[:, :, 200:])


def test_live_window_equals_full_sequence(cfg):
    """Live re-runs the model on the last W frames; its last output must equal the full-sequence output."""
    torch.manual_seed(0)
    m = build_model(4, cfg["model"]).eval()
    W = cfg["train"]["live_window_frames"]
    Z = torch.randn(1, D, 400)
    M = (torch.rand(1, 3, 400) > 0.2).float()
    with torch.no_grad():
        full = m(Z, M)
        for t in (150, 260, 399):
            win = m(Z[:, :, t + 1 - W:t + 1], M[:, :, t + 1 - W:t + 1])
            assert torch.allclose(win[:, :, -1], full[:, :, t], atol=1e-5)


def test_masked_branch_ignores_its_features(cfg):
    torch.manual_seed(0)
    m = build_model(4, cfg["model"]).eval()
    Z = torch.randn(1, D, 100)
    M = torch.ones(1, 3, 100)
    M[:, 2] = 0  # hand missing
    Z2 = Z.clone()
    Z2[:, 22:] = torch.randn(1, D - 22, 100)
    with torch.no_grad():
        assert torch.allclose(m(Z, M), m(Z2, M), atol=1e-6)


def test_augment_shapes(sess, cfg):
    data, classes, kinds = load_dataset([sess.path], None, cfg)
    rng = np.random.default_rng(0)
    s = WindowSampler(data, cfg, rng)
    L = cfg["train"]["win_frames"]
    for _ in range(20):
        si, st, Ls = s.sample()
        d = data[si]
        assert (d.split[st:st + Ls] == 0).all()
        X, M, y = augment(d.X[st:st + Ls], d.M[st:st + Ls], d.y[st:st + Ls], L, rng, cfg["train"]["aug"],
                          cfg["emg"]["log_clip"])
        assert X.shape == (L, D) and M.shape == (L, 3) and y.shape == (L,)
        assert np.isfinite(X).all() and M.any(1).all()


def test_smoke_train_save_load(sess, cfg, tmp_path):
    from backend.train import train
    c = {**cfg, "paths": {**cfg["paths"], "models": str(tmp_path)},
         "train": {**cfg["train"], "steps": 20, "eval_every": 10, "batch": 8}}
    out, meta = train([sess.path], ["swipe_left", "pinch_hold"], "smoke", c)
    assert (out / "model.pt").exists() and (out / "report.png").exists()
    assert meta["classes"] == ["none", "swipe_left", "pinch_hold"]
    model, meta2, norm = load_model(out, c)
    data, _, _ = load_dataset([sess.path], ["swipe_left", "pinch_hold"], c)
    P = predict_full(model, norm, data[0].X, data[0].M)
    assert P.shape == (len(data[0].T), 3) and np.allclose(P.sum(1), 1, atol=1e-5)
