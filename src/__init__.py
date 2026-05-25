"""Production API for the IoT anomaly detection system.

Experimental code (GA, ablation, sweeps) lives in ``src.experiments`` —
import from there explicitly when you need it.
"""

__version__ = "0.1.0"

from .config import (
    AppConfig,
    DataConfig,
    DetectorConfig,
    ModelConfig,
    PlotStyle,
    TrainConfig,
    resolve_device,
    set_seed,
)
from .data import (
    FEATURE_NAMES,
    N_FEATURES,
    SCENARIO_CODE,
    SCENARIO_NAMES,
    SUBSET_5,
    DataModule,
    MinMaxScaler,
    RobustScaler,
    TestContext,
    generate_synthetic_metrics,
    inject_anomalies,
    make_windows,
    subset_features,
)
from .detector import (
    AnomalyDetector,
    per_scenario_recall,
    prf1,
    roc_pr_curves,
)
from .losses import build_loss
from .model import GRUNet, LSTMNet
from .pipeline import Pipeline, RunResult
from .predictors import BasePredictor, MovingAveragePredictor, TorchPredictor
from .prefilter import (
    ANOMALY,
    NORMAL,
    UNCERTAIN,
    BasePreFilter,
    EMAPreFilter,
    MovingAveragePreFilter,
    RoutingStats,
    build_prefilter,
)
from .reporter import Reporter

__all__ = [
    "__version__",
    "ANOMALY",
    "AnomalyDetector",
    "AppConfig",
    "BasePredictor",
    "BasePreFilter",
    "DataConfig",
    "DataModule",
    "DetectorConfig",
    "EMAPreFilter",
    "FEATURE_NAMES",
    "GRUNet",
    "LSTMNet",
    "MinMaxScaler",
    "RobustScaler",
    "ModelConfig",
    "MovingAveragePredictor",
    "MovingAveragePreFilter",
    "N_FEATURES",
    "NORMAL",
    "Pipeline",
    "PlotStyle",
    "Reporter",
    "RoutingStats",
    "RunResult",
    "SCENARIO_CODE",
    "SCENARIO_NAMES",
    "SUBSET_5",
    "TestContext",
    "TorchPredictor",
    "TrainConfig",
    "UNCERTAIN",
    "build_loss",
    "build_prefilter",
    "generate_synthetic_metrics",
    "inject_anomalies",
    "make_windows",
    "per_scenario_recall",
    "prf1",
    "resolve_device",
    "roc_pr_curves",
    "set_seed",
    "subset_features",
]
