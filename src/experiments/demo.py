"""Production demonstration: train the evolved GRU and report on HOLDOUT."""

from __future__ import annotations

import time

from .. import _ui
from ..config import AppConfig, resolve_device, set_seed
from ..data import (
    DataModule,
    FEATURE_NAMES,
    N_FEATURES,
    generate_synthetic_metrics,
    inject_anomalies,
)
from ..model import GRUNet
from ..pipeline import Pipeline
from ..predictors import TorchPredictor
from ..prefilter import build_prefilter
from ..reporter import Reporter

# Evolutionary-search winner (see artifacts/evolution/best_genome.json).
WINNER_GENOME = {
    "window_size": 40,
    "hidden_size": 8,
    "num_layers": 3,
    "dropout": 0.4,
    "lr": 3e-3,
}


def build_demo_config(quick: bool = False, output_dir: str = "artifacts") -> AppConfig:
    """Build the demo config from the evolutionary-search winner genome.

    Args:
        quick: If True, shrink the dataset and epoch budget for a fast run.
        output_dir: Directory for artefacts.

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
    return cfg


def run_demo(quick: bool = False, output_dir: str = "artifacts") -> None:
    """Train the evolved GRU on holdout data and save the production report.

    Args:
        quick: If True, run a fast reduced-budget showcase.
        output_dir: Directory for artefacts.
    """
    cfg = build_demo_config(quick=quick, output_dir=output_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    _ui.banner(
        "ВИЯВЛЕННЯ АНОМАЛІЙ IoT — ДЕМОНСТРАЦІЯ",
        f"device={device}  режим={'швидко' if quick else 'повний'}  вивід={output_dir}/",
    )
    _ui.kv_table(
        "архітектура",
        [
            ("window_size", cfg.data.window_size),
            ("hidden_size", cfg.model.hidden_size),
            ("num_layers", cfg.model.num_layers),
            ("dropout", cfg.model.dropout),
            ("lr", f"{cfg.train.lr:.0e}"),
            ("макс. епох", cfg.train.epochs),
        ],
    )

    _ui.step(1, 3, "генерація синтетичних даних (train + holdout + calibration)")
    datamodule = DataModule(
        window_size=cfg.data.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_train_steps,
            seed=cfg.data.train_seed,
        )
    )
    holdout_data, holdout_labels, holdout_scenarios = inject_anomalies(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_holdout_steps,
            seed=cfg.data.holdout_test_seed,
        ),
        seed=cfg.data.holdout_anomaly_seed,
    )
    test_ctx = datamodule.prepare_test(holdout_data, holdout_labels, holdout_scenarios)
    calib_data, calib_labels, calib_scenarios = inject_anomalies(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_calibration_steps,
            seed=cfg.data.calibration_seed,
        ),
        seed=cfg.data.calibration_anomaly_seed,
    )
    calib_ctx = datamodule.prepare_test(calib_data, calib_labels, calib_scenarios)
    _ui.info(
        f"train={datamodule.X_train.shape}  "
        f"holdout={test_ctx.X.shape}  calib={calib_ctx.X.shape}"
    )
    _ui.info(
        f"anomalies in holdout: {int(test_ctx.labels_windowed.sum())} "
        f"/ {len(test_ctx.labels_windowed)} windows"
    )

    _ui.step(2, 3, "навчання GRU + калібрування порогу на dev-вибірці")
    net = GRUNet(
        n_features=N_FEATURES,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    predictor = TorchPredictor(
        net=net,
        name="GRU_evolved",
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
    elapsed = time.perf_counter() - t0
    _ui.success(f"навчання завершено за {elapsed:.1f}s")

    _ui.step(3, 3, "збереження артефактів + фінальні метрики")
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

    # Фінальна таблиця метрик
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
    if result.per_scenario:
        rows = [
            [n, f"{d['recall']:.3f}", f"{d['tp']}/{d['n_windows']}"]
            for n, d in sorted(
                result.per_scenario.items(), key=lambda kv: -kv[1]["recall"]
            )
            if d["n_windows"] > 0
        ]
        _ui.comparison_table(
            "recall за сценарієм",
            ["сценарій", "recall", "TP/всього"],
            rows,
        )
    if result.routing is not None:
        r = result.routing
        _ui.kv_table(
            "каскадне маршрутизування",
            [
                ("швидко: норма", r.n_normal),
                ("uncertain -> GRU", r.n_uncertain),
                ("швидко: аномалія", r.n_anomaly),
                ("зекономлено обч.", f"{r.fast_path_ratio * 100:.1f}%"),
            ],
        )

    _ui.success(f"артефакти збережено у {cfg.artifacts_dir}/")
