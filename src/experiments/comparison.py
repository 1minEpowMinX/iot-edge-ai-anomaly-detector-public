"""Comparison runner: GRU vs LSTM vs MovingAverage on the same dev test data."""

from __future__ import annotations

from ..config import AppConfig, resolve_device, set_seed
from ..data import (
    DataModule,
    FEATURE_NAMES,
    N_FEATURES,
    generate_synthetic_metrics,
    inject_anomalies,
)
from ..model import GRUNet, LSTMNet
from ..pipeline import Pipeline
from ..predictors import MovingAveragePredictor, TorchPredictor
from ..reporter import Reporter


def _build_config(output_dir: str) -> AppConfig:
    """Build the comparison config (no prefilter, fixed-budget training).

    Args:
        output_dir: Directory for artefacts.

    Returns:
        The configured :class:`AppConfig`.
    """
    cfg = AppConfig(artifacts_dir=output_dir)
    # No prefilter — comparison should test models in isolation.
    cfg.detector.use_prefilter = False
    # Fixed-budget training for fair A/B by time.
    cfg.train.epochs = 30
    cfg.train.early_stopping = False
    # Use percentile mode — auto_f1 would optimise threshold per-model.
    cfg.detector.threshold_mode = "percentile"
    return cfg


def _build_predictors(cfg: AppConfig, device: str) -> list:
    """Build the GRU, LSTM and MovingAverage predictors to compare.

    Args:
        cfg: Comparison configuration.
        device: Torch device string.

    Returns:
        The list of predictors to benchmark.
    """
    common = dict(config=cfg.train, device=device)
    return [
        TorchPredictor(
            net=GRUNet(
                n_features=N_FEATURES,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout,
            ),
            name="GRU",
            **common,
        ),
        TorchPredictor(
            net=LSTMNet(
                n_features=N_FEATURES,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout,
            ),
            name="LSTM",
            **common,
        ),
        MovingAveragePredictor(name="MovingAvg"),
    ]


def run_comparison(output_dir: str = "artifacts/comparison") -> None:
    """Benchmark GRU vs LSTM vs MovingAverage and save the comparison report.

    Args:
        output_dir: Directory for artefacts.
    """
    cfg = _build_config(output_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    print(f"[compare] пристрій: {device}")

    datamodule = DataModule(
        window_size=cfg.data.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_train_steps,
            seed=cfg.data.train_seed,
        )
    )
    test_data, test_labels, test_scenarios = inject_anomalies(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_test_steps,
            seed=cfg.data.test_seed,
        ),
        seed=cfg.data.anomaly_seed,
    )
    test_ctx = datamodule.prepare_test(test_data, test_labels, test_scenarios)

    results = []
    for predictor in _build_predictors(cfg, device):
        set_seed(cfg.seed)
        print(f"\n[{predictor.name}] навчання…")
        pipeline = Pipeline(
            datamodule=datamodule,
            predictor=predictor,
            detector_cfg=cfg.detector,
            prefilter=None,
        )
        results.append(pipeline.run(test_ctx))

    reporter = Reporter(
        output_dir=cfg.artifacts_dir,
        feature_names=FEATURE_NAMES,
        window_size=cfg.data.window_size,
        style=cfg.plot,
    )
    reporter.print_comparison(results)
    reporter.save_comparison_dashboard(results)
    print(f"[compare] збережено у {cfg.artifacts_dir}/model_comparison.png")
