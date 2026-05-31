"""Configurable training runner.

Used by ``python main.py train [--epochs ... --lr ... --hidden ... ...]``.
Defaults to the evolutionary-search winner; CLI flags override individual genes.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _ui
from ..config import AppConfig, resolve_device, set_seed
from ..data import (
    DataModule,
    FEATURE_NAMES,
    N_FEATURES,
    generate_synthetic_metrics,
    inject_anomalies,
    load_csv_metrics,
)
from ..model import GRUNet
from ..pipeline import Pipeline
from ..predictors import TorchPredictor
from ..prefilter import build_prefilter
from ..reporter import Reporter

WINNER_GENOME = {
    "window_size": 40,
    "hidden_size": 8,
    "num_layers": 3,
    "dropout": 0.4,
    "lr": 3e-3,
}


def build_train_config(
    overrides: dict[str, Any] | None = None,
    output_dir: str = "artifacts",
    quick: bool = False,
) -> AppConfig:
    """Build the training config from evolved defaults plus CLI overrides.

    Args:
        overrides: Optional mapping of config fields to override values.
        output_dir: Directory for artefacts.
        quick: If True, shrink the dataset and epoch budget for a fast run.

    Returns:
        The configured :class:`AppConfig`.
    """
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.data.window_size = WINNER_GENOME["window_size"]
    cfg.model.hidden_size = WINNER_GENOME["hidden_size"]
    cfg.model.num_layers = WINNER_GENOME["num_layers"]
    cfg.model.dropout = WINNER_GENOME["dropout"]
    cfg.train.lr = WINNER_GENOME["lr"]
    cfg.train.epochs = 300

    if quick:
        cfg.data.n_train_steps = 1500
        cfg.data.n_holdout_steps = 800
        cfg.data.n_calibration_steps = 400
        cfg.train.epochs = 80
        cfg.train.verbose = False

    overrides = overrides or {}
    if "window_size" in overrides:
        cfg.data.window_size = int(overrides["window_size"])
    if "hidden_size" in overrides:
        cfg.model.hidden_size = int(overrides["hidden_size"])
    if "num_layers" in overrides:
        cfg.model.num_layers = int(overrides["num_layers"])
    if "dropout" in overrides:
        cfg.model.dropout = float(overrides["dropout"])
    if "lr" in overrides:
        cfg.train.lr = float(overrides["lr"])
    if "epochs" in overrides:
        cfg.train.epochs = int(overrides["epochs"])
    if "batch_size" in overrides:
        cfg.train.batch_size = int(overrides["batch_size"])
    if "seed" in overrides:
        cfg.seed = int(overrides["seed"])
    if "device" in overrides:
        cfg.device = str(overrides["device"])
    return cfg


def run_train(
    overrides: dict[str, Any] | None = None,
    output_dir: str = "artifacts",
    quick: bool = False,
    csv_path: str | None = None,
) -> None:
    """Train a GRU with the given hyperparameters and save the report.

    Args:
        overrides: Optional mapping of config fields to override values.
        output_dir: Directory for artefacts.
        quick: If True, run a fast reduced-budget training.
        csv_path: Optional path to a real-metrics CSV from ``collect``.
            When provided the model is trained on real host data instead
            of synthetic data, which prevents distribution-shift false
            positives when running ``live`` afterwards.
    """
    cfg = build_train_config(overrides=overrides, output_dir=output_dir, quick=quick)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    overrides = overrides or {}
    overridden_note = (
        f"перевизначення: {', '.join(overrides.keys())}"
        if overrides
        else "evolved-дефолти"
    )
    _ui.banner(
        "ВИЯВЛЕННЯ АНОМАЛІЙ IoT — НАВЧАННЯ",
        f"device={device}  режим={'швидко' if quick else 'повний'}  {overridden_note}",
    )
    _ui.kv_table(
        "конфігурація",
        [
            ("window_size", cfg.data.window_size),
            ("hidden_size", cfg.model.hidden_size),
            ("num_layers", cfg.model.num_layers),
            ("dropout", cfg.model.dropout),
            ("lr", f"{cfg.train.lr:.0e}"),
            ("макс. епох", cfg.train.epochs),
            ("batch_size", cfg.train.batch_size),
            ("seed", cfg.seed),
        ],
    )

    if csv_path:
        _ui.step(1, 3, f"завантаження реальних даних: {csv_path}")
        raw = load_csv_metrics(csv_path)
        n = len(raw)
        n_train = int(n * 0.70)
        n_holdout = int(n * 0.15)
        train_data = raw[:n_train]
        holdout_raw = raw[n_train : n_train + n_holdout]
        calib_raw = raw[n_train + n_holdout :]
        holdout_data, holdout_labels, holdout_scenarios = inject_anomalies(
            holdout_raw,
            seed=cfg.data.holdout_anomaly_seed,
            n_windows=max(1, len(holdout_raw) // 30),
        )
        calib_data, calib_labels, calib_scenarios = inject_anomalies(
            calib_raw,
            seed=cfg.data.calibration_anomaly_seed,
            n_windows=max(1, len(calib_raw) // 30),
        )
    else:
        _ui.step(1, 3, "підготовка даних")
        train_data = generate_synthetic_metrics(
            n_steps=cfg.data.n_train_steps,
            seed=cfg.data.train_seed,
        )
        holdout_data, holdout_labels, holdout_scenarios = inject_anomalies(
            generate_synthetic_metrics(
                n_steps=cfg.data.n_holdout_steps,
                seed=cfg.data.holdout_test_seed,
            ),
            seed=cfg.data.holdout_anomaly_seed,
        )
        calib_data, calib_labels, calib_scenarios = inject_anomalies(
            generate_synthetic_metrics(
                n_steps=cfg.data.n_calibration_steps,
                seed=cfg.data.calibration_seed,
            ),
            seed=cfg.data.calibration_anomaly_seed,
        )
    datamodule = DataModule(
        window_size=cfg.data.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(train_data)
    test_ctx = datamodule.prepare_test(holdout_data, holdout_labels, holdout_scenarios)
    calib_ctx = datamodule.prepare_test(calib_data, calib_labels, calib_scenarios)
    _ui.info(
        f"train={datamodule.X_train.shape}  "
        f"holdout={test_ctx.X.shape}  calib={calib_ctx.X.shape}"
    )

    _ui.step(2, 3, "навчання + калібрування порогу")
    net = GRUNet(
        n_features=N_FEATURES,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    predictor = TorchPredictor(
        net=net,
        name="GRU",
        config=cfg.train,
        device=device,
    )
    prefilter = build_prefilter(cfg.detector)
    pipeline = Pipeline(
        datamodule=datamodule,
        predictor=predictor,
        detector_cfg=cfg.detector,
        prefilter=prefilter,
    )
    t0 = time.perf_counter()
    result = pipeline.run(test_ctx, calibration_ctx=calib_ctx)
    _ui.success(f"навчання завершено за {time.perf_counter() - t0:.1f}s")

    _ui.step(3, 3, "збереження артефактів")
    reporter = Reporter(
        output_dir=cfg.artifacts_dir,
        feature_names=FEATURE_NAMES,
        window_size=cfg.data.window_size,
        style=cfg.plot,
    )
    reporter.save_production_report(
        result=result,
        test_ctx=test_ctx,
        datamodule=datamodule,
        config={**cfg.to_dict(), "device": device},
    )

    _ui.rule("результати на HOLDOUT")
    _ui.kv_table(
        "метрики",
        [
            ("F1", f"{result.f1:.4f}"),
            ("точність", f"{result.precision:.4f}"),
            ("повнота", f"{result.recall:.4f}"),
            ("параметри", f"{result.n_params:,}"),
            ("інференс", f"{result.infer_ms:.3f} ms"),
            ("поріг", f"{result.threshold:.5f}"),
        ],
    )
    _ui.success(f"артефакти збережено у {cfg.artifacts_dir}/")
