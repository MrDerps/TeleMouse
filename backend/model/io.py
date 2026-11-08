"""Save / load trained TCN models (data/models/<name>/model.pt + model.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from backend.common.config import repo_path
from backend.data.normalize import Normalizer
from backend.model.tcn import build_model


def model_dir(cfg, name: str) -> Path:
    return repo_path(cfg["paths"]["models"]) / name


def save_model(out: Path, model, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(path_or_name, cfg):
    p = Path(path_or_name)
    if not (p / "model.json").exists():
        p = model_dir(cfg, str(path_or_name))
    meta = json.loads((p / "model.json").read_text(encoding="utf-8"))
    if meta.get("type") != "tcn":
        raise ValueError(f"{p.name} is not a TCN model (type={meta.get('type')})")
    model = build_model(len(meta["classes"]), meta["model_cfg"])
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, meta, Normalizer.from_dict(meta["normalizer"])


def list_models(cfg) -> list[dict]:
    root = repo_path(cfg["paths"]["models"])
    out = []
    if root.exists():
        for p in sorted(root.iterdir()):
            try:
                m = json.loads((p / "model.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            by = {k: {x: v.get(x) for x in ("precision", "recall", "fp_per_min", "latency_ms", "per_gesture")}
                  for k, v in m.get("by_sensors", {}).items()}
            out.append({"name": m.get("name", p.name), "classes": m.get("classes", []),
                        "created": m.get("created"), "val_metrics": m.get("val_metrics", {}),
                        "type": m.get("type"), "sensors": m.get("sensors", ["emg", "imu", "hand"]),
                        "by_sensors": by})
    return out


@torch.no_grad()
def predict_full(model, norm, X, M) -> np.ndarray:
    """Causal full-sequence probabilities (F, n_cls)."""
    Z = torch.from_numpy(norm(X, M).T[None])
    Mt = torch.from_numpy(M.T[None].astype(np.float32))
    return torch.softmax(model(Z, Mt), 1)[0].T.numpy()
