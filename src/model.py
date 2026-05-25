"""Pure ``nn.Module`` networks."""

from __future__ import annotations

import torch
import torch.nn as nn

_VALID_ACTIVATIONS = ("linear", "gelu", "relu", "tanh", "silu")
_LINEAR_ALIASES = ("linear", "identity", "none")


def _build_activation(name: str) -> nn.Module:
    """Return the ``nn.Module`` for a named non-linear activation.

    Args:
        name: One of ``"gelu"``, ``"relu"``, ``"tanh"`` or ``"silu"``
            (case-insensitive).

    Returns:
        The corresponding activation module.

    Raises:
        ValueError: If ``name`` is not a recognised activation.
    """
    n = name.lower()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU()
    if n == "tanh":
        return nn.Tanh()
    if n == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name!r}. Use one of {_VALID_ACTIVATIONS}.")


def _build_head(
    hidden_size: int,
    n_features: int,
    dropout: float,
    head_activation: str,
) -> nn.Module:
    """Regression head.

    "linear" / "identity" / "none" → single ``Linear(h, n_features)`` projection
    (recommended default for next-step regression — the GRU already provides
    plenty of non-linearity; a 2-layer MLP without activation collapses to a
    single linear map, and with activation tends not to improve F1 in practice).

    Any other name → 2-layer MLP with the named activation between the linears.
    """
    if head_activation.lower() in _LINEAR_ALIASES:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features),
        )
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        _build_activation(head_activation),
        nn.Linear(hidden_size, n_features),
    )


class GRUNet(nn.Module):
    """Multi-layer GRU + LayerNorm + linear readout for next-step regression."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_activation: str = "linear",
    ) -> None:
        """Build a multi-layer GRU with LayerNorm and a regression head.

        Args:
            n_features: Number of input/output metrics.
            hidden_size: GRU hidden-state size.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout between GRU layers and inside the head.
            head_activation: Head type — ``"linear"`` for a single projection
                or an activation name for a 2-layer MLP head.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head_activation = head_activation
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = _build_head(hidden_size, n_features, dropout, head_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next step from an input window.

        Args:
            x: Input batch, shape ``(batch, window, n_features)``.

        Returns:
            Next-step prediction, shape ``(batch, n_features)``.
        """
        out, _ = self.gru(x)
        return self.head(self.norm(out[:, -1, :]))


class LSTMNet(nn.Module):
    """Multi-layer LSTM mirror of GRUNet, used as a comparison baseline."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_activation: str = "linear",
    ) -> None:
        """Build a multi-layer LSTM with LayerNorm and a regression head.

        Args:
            n_features: Number of input/output metrics.
            hidden_size: LSTM hidden-state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers and inside the head.
            head_activation: Head type — ``"linear"`` for a single projection
                or an activation name for a 2-layer MLP head.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head_activation = head_activation
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = _build_head(hidden_size, n_features, dropout, head_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next step from an input window.

        Args:
            x: Input batch, shape ``(batch, window, n_features)``.

        Returns:
            Next-step prediction, shape ``(batch, n_features)``.
        """
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :]))
