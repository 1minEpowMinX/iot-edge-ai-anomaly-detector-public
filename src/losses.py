"""Loss function factory."""

from __future__ import annotations

import torch.nn as nn

from .config import LossName


def build_loss(name: LossName | str, huber_delta: float = 1.0) -> nn.Module:
    """Build a regression loss module by name.

    Args:
        name: Loss identifier — ``"mse"``, ``"mae"`` or ``"huber"``
            (case-insensitive).
        huber_delta: Transition point between the quadratic and linear
            regions; used only by the Huber loss.

    Returns:
        A ready-to-use ``nn.Module`` loss.

    Raises:
        ValueError: If ``name`` is not a recognised loss.
    """
    n = name.lower()
    if n == "mse":
        return nn.MSELoss()
    if n == "mae":
        return nn.L1Loss()
    if n == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unknown loss: {name!r}. Use 'mse' | 'mae' | 'huber'.")
