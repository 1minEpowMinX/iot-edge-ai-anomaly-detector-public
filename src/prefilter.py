"""Cascade prefilter: cheap first stage that routes obvious cases past the GRU.

Two implementations:
  - MovingAveragePreFilter: simple mean over the window. Lags spikes.
  - EMAPreFilter: exponential moving average, weights recent samples higher.
    Reacts faster to spikes-at-window-end and is the recommended default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig

# Routing codes
NORMAL = 0
UNCERTAIN = 1
ANOMALY = 2


@dataclass
class RoutingStats:
    """Counts of how a prefilter routed a batch of windows."""

    n_total: int
    n_normal: int
    n_uncertain: int
    n_anomaly: int

    @property
    def fast_path_ratio(self) -> float:
        """Fraction of windows resolved without invoking the main detector."""
        if self.n_total == 0:
            return 0.0
        return (self.n_normal + self.n_anomaly) / self.n_total


class BasePreFilter(ABC):
    """Cheap first stage that classifies windows as normal / uncertain / anomaly."""

    name: str = "base_prefilter"

    @abstractmethod
    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "BasePreFilter":
        """Calibrate the prefilter on validation data and return ``self``."""
        ...

    @abstractmethod
    def classify(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Route each window to NORMAL, UNCERTAIN or ANOMALY."""
        ...


class _PercentilePreFilter(BasePreFilter):
    """Shared implementation: thresholds calibrated by error percentiles on val."""

    name = "percentile_prefilter"

    def __init__(
        self,
        low_percentile: float = 30.0,
        high_percentile: float = 99.7,
    ) -> None:
        """Initialise the percentile-based prefilter.

        Args:
            low_percentile: Error percentile below which a window is NORMAL.
            high_percentile: Error percentile above which a window is ANOMALY.
        """
        if not 0.0 <= low_percentile < high_percentile <= 100.0:
            raise ValueError("Expect 0 <= low_percentile < high_percentile <= 100")
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.low_threshold: float | None = None
        self.high_threshold: float | None = None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Return the cheap per-window prediction (implemented by subclasses)."""
        raise NotImplementedError

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "_PercentilePreFilter":
        """Calibrate the low/high error thresholds on validation data.

        Args:
            X_val: Validation input windows.
            y_val: Validation next-step targets.

        Returns:
            ``self``, with the error thresholds populated.
        """
        pred = self._predict(X_val)
        errors = np.abs(pred - y_val).mean(axis=1)
        self.low_threshold = float(np.percentile(errors, self.low_percentile))
        self.high_threshold = float(np.percentile(errors, self.high_percentile))
        return self

    def classify(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Route each window to NORMAL, UNCERTAIN or ANOMALY.

        Args:
            X: Input windows, shape ``(batch, window, n_features)``.
            y: Next-step targets, shape ``(batch, n_features)``.

        Returns:
            Routing codes per window (NORMAL / UNCERTAIN / ANOMALY).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if self.low_threshold is None or self.high_threshold is None:
            raise RuntimeError("Call fit() before classify()")
        pred = self._predict(X)
        errors = np.abs(pred - y).mean(axis=1)
        result = np.full(len(errors), UNCERTAIN, dtype=np.int64)
        result[errors < self.low_threshold] = NORMAL
        result[errors > self.high_threshold] = ANOMALY
        return result


class MovingAveragePreFilter(_PercentilePreFilter):
    """prediction = arithmetic mean over the window. Kept for ablations."""

    name = "MA-prefilter"

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the next step as the arithmetic mean over the window."""
        return X.mean(axis=1).astype(np.float32)


class EMAPreFilter(_PercentilePreFilter):
    """prediction = exponential moving average over the window.

    EMA weights recent timesteps higher than older ones — better for catching
    short spikes that drift the average too slowly with plain MA.

    s_0 = X_0
    s_t = alpha * X_t + (1 - alpha) * s_{t-1}
    prediction = s_{T-1}  (last EMA state in the window)
    """

    name = "EMA-prefilter"

    def __init__(
        self,
        alpha: float = 0.3,
        low_percentile: float = 30.0,
        high_percentile: float = 99.7,
    ) -> None:
        """Initialise the EMA prefilter.

        Args:
            alpha: EMA smoothing factor in (0, 1]; higher weights recent steps.
            low_percentile: Error percentile below which a window is NORMAL.
            high_percentile: Error percentile above which a window is ANOMALY.
        """
        super().__init__(low_percentile=low_percentile, high_percentile=high_percentile)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the next step as the exponential moving average of the window."""
        alpha = self.alpha
        ema = X[:, 0, :].astype(np.float32, copy=True)
        for t in range(1, X.shape[1]):
            ema = alpha * X[:, t, :] + (1.0 - alpha) * ema
        return ema


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_prefilter(cfg: DetectorConfig) -> BasePreFilter | None:
    """Construct a prefilter from the config or return None when disabled."""
    if not cfg.use_prefilter or cfg.prefilter_type == "none":
        return None
    if cfg.prefilter_type == "ema":
        return EMAPreFilter(
            alpha=cfg.prefilter_ema_alpha,
            low_percentile=cfg.prefilter_low_percentile,
            high_percentile=cfg.prefilter_high_percentile,
        )
    if cfg.prefilter_type == "ma":
        return MovingAveragePreFilter(
            low_percentile=cfg.prefilter_low_percentile,
            high_percentile=cfg.prefilter_high_percentile,
        )
    raise ValueError(f"Unknown prefilter_type: {cfg.prefilter_type!r}")


def compute_routing_stats(routes: np.ndarray) -> RoutingStats:
    """Tally routing codes into a :class:`RoutingStats` summary.

    Args:
        routes: Array of per-window routing codes.

    Returns:
        The aggregated counts.
    """
    return RoutingStats(
        n_total=int(len(routes)),
        n_normal=int((routes == NORMAL).sum()),
        n_uncertain=int((routes == UNCERTAIN).sum()),
        n_anomaly=int((routes == ANOMALY).sum()),
    )
