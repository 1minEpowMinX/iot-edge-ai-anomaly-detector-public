"""Centralised configuration for the anomaly detection system."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

LossName = Literal["mse", "mae", "huber"]
ThresholdMode = Literal["percentile", "auto_f1"]
PrefilterType = Literal["ema", "ma", "none"]
ScoreMode = Literal["mean", "zscore_mean", "zscore_max"]

_VALID_LOSSES = ("mse", "mae", "huber")
_VALID_THRESHOLD_MODES = ("percentile", "auto_f1")
_VALID_PREFILTERS = ("ema", "ma", "none")
_VALID_SCORE_MODES = ("mean", "zscore_mean", "zscore_max")


@dataclass
class DataConfig:
    """Dataset sizing, the train/val split and the no-leak seed protocol."""

    # --- core data sizes ---
    n_train_steps: int = 5000
    n_test_steps: int = 1500  # used during evolutionary search (GA fitness)
    n_calibration_steps: int = 800
    n_holdout_steps: int = 2000  # NEVER seen during search — for final reporting only
    val_split: float = 0.2
    window_size: int = 80

    # --- seeds: train / dev-test / calibration ---
    # NOTE on the no-leak protocol:
    #   - train/val (no anomalies)  trains weights + tunes early stop / lr / prefilter
    #   - calibration (with anomalies)  tunes the detector threshold (auto_f1)
    #   - test_seed=123  is the "dev test" used by evolve.py to score genomes —
    #     the GA optimises the architecture for this distribution, so it cannot
    #     also be used as the unbiased "production" metric
    #   - holdout_*  is the final, never-seen-during-search set — only main.py
    #     and evolve.py Phase 2 use it for honest reporting
    train_seed: int = 42
    test_seed: int = 123
    calibration_seed: int = 77
    holdout_test_seed: int = 999
    anomaly_seed: int = 7
    calibration_anomaly_seed: int = 17
    holdout_anomaly_seed: int = 31

    def __post_init__(self) -> None:
        """Validate field values and the no-leak seed protocol."""
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if self.n_train_steps <= self.window_size:
            raise ValueError("n_train_steps must be greater than window_size")
        if self.n_calibration_steps <= self.window_size:
            raise ValueError("n_calibration_steps must be greater than window_size")
        if self.n_holdout_steps <= self.window_size:
            raise ValueError("n_holdout_steps must be greater than window_size")
        # cross-seed sanity
        used_seeds = {
            "train_seed": self.train_seed,
            "test_seed": self.test_seed,
            "calibration_seed": self.calibration_seed,
            "holdout_test_seed": self.holdout_test_seed,
        }
        if len(set(used_seeds.values())) != len(used_seeds):
            raise ValueError(
                f"data seeds must all differ to prevent leakage, got {used_seeds}"
            )


@dataclass
class ModelConfig:
    """GRU/LSTM architecture hyperparameters."""

    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate architecture hyperparameters."""
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class TrainConfig:
    """Optimiser, loss, scheduler and early-stopping settings for training."""

    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    loss: LossName = "huber"
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    early_stopping: bool = True
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-5
    early_stopping_reset_on_lr_drop: bool = True
    early_stopping_require_progress_for_reset: bool = True
    restore_best_weights: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate training settings and warn on conflicting patience values."""
        if self.loss not in _VALID_LOSSES:
            raise ValueError(f"loss must be one of {_VALID_LOSSES}, got {self.loss!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0, got {self.grad_clip}")
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1, "
                f"got {self.early_stopping_patience}"
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be >= 0, "
                f"got {self.early_stopping_min_delta}"
            )
        if (
            self.early_stopping
            and self.early_stopping_patience <= self.scheduler_patience
        ):
            import warnings

            warnings.warn(
                "early_stopping_patience <= scheduler_patience: "
                "training may stop before ReduceLROnPlateau gets a chance to help.",
                stacklevel=2,
            )


@dataclass
class DetectorConfig:
    """Anomaly-score formula, threshold calibration and prefilter settings."""

    percentile: float = 99.0
    threshold_mode: ThresholdMode = "auto_f1"
    # Anomaly score formula:
    #   "mean":         mean(|pred - actual|)                 — production default
    #   "zscore_mean":  mean(|pred - actual| / val_feature_std)
    #   "zscore_max":   max(|pred - actual| / val_feature_std)
    #
    # Experimentally verified: on this task (MinMax-scaled inputs + 12 features)
    # "mean" gives the best F1 (~0.90 vs 0.85 for zscore_mean). z-score boosts
    # recall (+2 pp, especially on cpu_stress) but amplifies normal-channel
    # noise and tanks precision (−9 pp). Kept as an option for recall-critical
    # configurations where missing an attack is far costlier than a false alarm.
    score_mode: ScoreMode = "mean"
    sweep_low: float = 90.0
    sweep_high: float = 99.9
    sweep_step: float = 0.25
    use_prefilter: bool = True
    prefilter_type: PrefilterType = "ema"
    prefilter_ema_alpha: float = 0.3
    prefilter_low_percentile: float = 30.0
    prefilter_high_percentile: float = 99.7
    # When False (default): asymmetric cascade — the prefilter only fast-paths
    # confidently-NORMAL windows. Anything suspicious goes to the main detector.
    # Preserves recall at the cost of slightly fewer compute savings.
    prefilter_fast_anomaly: bool = False

    def __post_init__(self) -> None:
        """Validate detector, threshold-sweep and prefilter settings."""
        if not 0.0 < self.percentile < 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {self.percentile}")
        if self.threshold_mode not in _VALID_THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {_VALID_THRESHOLD_MODES}, "
                f"got {self.threshold_mode!r}"
            )
        if self.score_mode not in _VALID_SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {_VALID_SCORE_MODES}, "
                f"got {self.score_mode!r}"
            )
        if self.prefilter_type not in _VALID_PREFILTERS:
            raise ValueError(
                f"prefilter_type must be one of {_VALID_PREFILTERS}, "
                f"got {self.prefilter_type!r}"
            )
        if not 0.0 < self.prefilter_ema_alpha <= 1.0:
            raise ValueError(
                f"prefilter_ema_alpha must be in (0, 1], "
                f"got {self.prefilter_ema_alpha}"
            )
        if not (
            0.0
            <= self.prefilter_low_percentile
            < self.prefilter_high_percentile
            <= 100.0
        ):
            raise ValueError(
                "prefilter percentiles must satisfy 0 <= low < high <= 100"
            )
        if not 80.0 <= self.sweep_low < self.sweep_high <= 99.95:
            raise ValueError(
                "sweep range must satisfy 80 <= sweep_low < sweep_high <= 99.95"
            )


@dataclass
class PlotStyle:
    """Shared styling for all generated dashboards and plots."""

    dpi: int = 120
    save_format: str = "png"
    figsize_scale: float = 1.0
    title_fontsize: int = 13
    cmap_heat: str = "magma"
    cmap_cm: str = "Blues"
    bbox_inches: str = "tight"
    palette: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    )

    def __post_init__(self) -> None:
        """Validate DPI and the output image format."""
        if self.dpi < 50:
            raise ValueError(f"dpi must be >= 50, got {self.dpi}")
        if self.save_format not in {"png", "svg", "pdf", "jpg"}:
            raise ValueError(f"unsupported save_format: {self.save_format!r}")


@dataclass
class AppConfig:
    """Top-level configuration aggregating every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    plot: PlotStyle = field(default_factory=PlotStyle)
    device: str = "auto"
    seed: int = 42
    artifacts_dir: str = "artifacts"

    def to_dict(self) -> dict:
        """Return the full configuration as a nested plain dict."""
        return asdict(self)


def resolve_device(name: str = "auto") -> str:
    """Resolve a device string, picking CUDA when available for ``"auto"``.

    Args:
        name: ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Returns:
        The resolved device string usable by torch.
    """
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed the NumPy and torch (incl. CUDA) RNGs for reproducible runs.

    Args:
        seed: Random seed to apply globally.
    """
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
