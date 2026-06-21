"""Artifact persistence — save/load the trained model bundle.

A model bundle stored on disk consists of:
    save_dir/
        model.pt          # torch state_dict + net class name
        scaler_min.npy    # MinMaxScaler.min_  (or RobustScaler.center_)
        scaler_max.npy    # MinMaxScaler.max_  (or RobustScaler.scale_)
        meta.json         # threshold, feature_names, config, history

RobustScaler bundles use scaler_center.npy / scaler_scale.npy and are
auto-detected on load for forward compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_torch_predictor(save_dir: str, predictor: Any) -> None:
    """Save the wrapped nn.Module's state_dict and class name."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "net_class": type(predictor.net).__name__,
            "name": predictor.name,
        },
        os.path.join(save_dir, "model.pt"),
    )


def save_scaler(save_dir: str, scaler: Any) -> None:
    """Save a fitted scaler to ``.npy`` files.

    Supports both :class:`~data.RobustScaler` (``center_`` / ``scale_``)
    and the legacy :class:`~data.MinMaxScaler` (``min_`` / ``max_``).

    Args:
        save_dir: Directory to write into (created if missing).
        scaler: A fitted ``RobustScaler`` or ``MinMaxScaler``.
    """
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(scaler, "center_"):
        np.save(os.path.join(save_dir, "scaler_center.npy"), scaler.center_)
        np.save(os.path.join(save_dir, "scaler_scale.npy"), scaler.scale_)
    else:
        # Legacy MinMaxScaler path.
        np.save(os.path.join(save_dir, "scaler_min.npy"), scaler.min_)
        np.save(os.path.join(save_dir, "scaler_max.npy"), scaler.max_)


def save_meta(
    save_dir: str,
    threshold: float,
    history: dict,
    feature_names: list[str],
    config: dict,
    prefilter: dict | None = None,
) -> None:
    """Write the bundle's ``meta.json`` (threshold, features, config, history).

    Args:
        save_dir: Directory to write into (created if missing).
        threshold: Calibrated anomaly-score threshold.
        history: Training history dict.
        feature_names: Ordered feature names the model expects.
        config: Serialised application configuration.
        prefilter: Optional serialised fitted prefilter (its calibrated error
            thresholds). ``None`` omits the cascade so inference runs the GRU on
            every window — keeps bundles without a prefilter fully valid.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "threshold": float(threshold),
        "feature_names": feature_names,
        "config": config,
        "history": history,
        "prefilter": prefilter,
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_meta(save_dir: str) -> dict:
    """Load and return the bundle's ``meta.json`` as a dict.

    Args:
        save_dir: Directory containing ``meta.json``.

    Returns:
        The parsed metadata dict.
    """
    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_scaler(save_dir: str):
    """Reconstruct a fitted scaler from saved ``.npy`` files.

    Detects the scaler type automatically:

    * ``scaler_center.npy`` present → :class:`~data.RobustScaler`
    * ``scaler_min.npy`` present → legacy :class:`~data.MinMaxScaler`

    Args:
        save_dir: Directory containing the scaler arrays.

    Returns:
        A fitted ``RobustScaler`` or ``MinMaxScaler`` instance.

    Raises:
        FileNotFoundError: If neither set of scaler files is found.
    """
    from .data import MinMaxScaler, RobustScaler  # local to avoid cycle

    center_path = os.path.join(save_dir, "scaler_center.npy")
    if os.path.exists(center_path):
        scaler = RobustScaler()
        scaler.center_ = np.load(center_path)
        scaler.scale_ = np.load(os.path.join(save_dir, "scaler_scale.npy"))
        return scaler

    # Backward compatibility: load legacy MinMaxScaler bundle.
    min_path = os.path.join(save_dir, "scaler_min.npy")
    if os.path.exists(min_path):
        scaler = MinMaxScaler()
        scaler.min_ = np.load(min_path)
        scaler.max_ = np.load(os.path.join(save_dir, "scaler_max.npy"))
        return scaler

    raise FileNotFoundError(
        f"No scaler files found in {save_dir!r}. "
        "Expected scaler_center.npy (RobustScaler) or scaler_min.npy (legacy)."
    )


def load_network(save_dir: str, n_features: int, model_cfg: dict) -> torch.nn.Module:
    """Reconstruct the trained nn.Module (GRUNet/LSTMNet) and load weights."""
    from .model import GRUNet, LSTMNet  # local

    checkpoint = torch.load(
        os.path.join(save_dir, "model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    net_class = checkpoint["net_class"]
    cls = {"GRUNet": GRUNet, "LSTMNet": LSTMNet}.get(net_class)
    if cls is None:
        raise ValueError(f"Unknown net_class in checkpoint: {net_class!r}")
    net = cls(
        n_features=n_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


@dataclass
class ModelBundle:
    """All artifacts needed to run inference from a saved model directory."""

    net: torch.nn.Module
    scaler: Any
    threshold: float
    feature_names: list[str]
    window_size: int
    n_features: int
    config: dict
    prefilter: Any = None


def load_bundle(save_dir: str, device: str = "cpu") -> ModelBundle:
    """One-shot loader: returns a fully reconstructed model + scaler + threshold."""
    meta = load_meta(save_dir)
    feature_names = meta["feature_names"]
    cfg = meta["config"]
    n_features = len(feature_names)
    net = load_network(save_dir, n_features=n_features, model_cfg=cfg["model"])
    net.to(device)
    scaler = load_scaler(save_dir)

    prefilter = None
    pf_state = meta.get("prefilter")
    if pf_state:
        from .prefilter import prefilter_from_state  # local to avoid cycle

        prefilter = prefilter_from_state(pf_state)

    return ModelBundle(
        net=net,
        scaler=scaler,
        threshold=float(meta["threshold"]),
        feature_names=feature_names,
        window_size=int(cfg["data"]["window_size"]),
        n_features=n_features,
        config=cfg,
        prefilter=prefilter,
    )
