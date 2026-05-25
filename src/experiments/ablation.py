"""Ablation runner: minimal 5-metric subset (thesis spec) vs full 12-metric set."""

from __future__ import annotations

from ..config import AppConfig, resolve_device, set_seed
from ..data import (
    DataModule,
    FEATURE_NAMES,
    SUBSET_5,
    generate_synthetic_metrics,
    inject_anomalies,
    subset_features,
)
from ..model import GRUNet
from ..pipeline import Pipeline
from ..predictors import TorchPredictor
from ..prefilter import build_prefilter
from ..reporter import Reporter


def _build_config(output_dir: str) -> AppConfig:
    """Build the ablation config with a shorter training budget.

    Args:
        output_dir: Directory for artefacts.

    Returns:
        The configured :class:`AppConfig`.
    """
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.train.epochs = 60
    return cfg


def _run_subset(
    cfg: AppConfig,
    device: str,
    kept: list[str],
    train_full,
    test_full,
    test_lab,
    test_scen,
    calib_full,
    calib_lab,
    calib_scen,
) -> dict:
    """Train and evaluate one feature subset.

    Args:
        cfg: Ablation configuration.
        device: Torch device string.
        kept: Feature names to keep for this run.
        train_full: Full-feature training metrics.
        test_full: Full-feature test metrics.
        test_lab: Per-step test labels.
        test_scen: Per-step test scenario codes.
        calib_full: Full-feature calibration metrics.
        calib_lab: Per-step calibration labels.
        calib_scen: Per-step calibration scenario codes.

    Returns:
        A summary dict of metrics for this subset.
    """
    label = f"{len(kept)} ознак"
    train_sub, _ = subset_features(train_full, FEATURE_NAMES, kept)
    test_sub, _ = subset_features(test_full, FEATURE_NAMES, kept)
    calib_sub, _ = subset_features(calib_full, FEATURE_NAMES, kept)

    datamodule = DataModule(
        window_size=cfg.data.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(train_sub)
    test_ctx = datamodule.prepare_test(test_sub, test_lab, test_scen)
    calib_ctx = datamodule.prepare_test(calib_sub, calib_lab, calib_scen)

    set_seed(cfg.seed)
    predictor = TorchPredictor(
        net=GRUNet(
            n_features=len(kept),
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ),
        name=f"GRU-{len(kept)}f",
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

    print(f"\n[ablate] {label} ({kept})")
    result = pipeline.run(test_ctx, calibration_ctx=calib_ctx)
    row = result.to_summary_dict()
    row["n_features"] = len(kept)
    row["per_scenario"] = {
        n: d["recall"] for n, d in result.per_scenario.items() if d["n_windows"] > 0
    }
    return row


def run_ablation(output_dir: str = "artifacts/ablation") -> None:
    """Run the 5-metric vs 12-metric feature ablation and save the report.

    Args:
        output_dir: Directory for artefacts.
    """
    cfg = _build_config(output_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    print(f"[ablate] пристрій: {device}")

    train_full = generate_synthetic_metrics(
        n_steps=cfg.data.n_train_steps,
        seed=cfg.data.train_seed,
    )
    test_full, test_lab, test_scen = inject_anomalies(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_test_steps,
            seed=cfg.data.test_seed,
        ),
        seed=cfg.data.anomaly_seed,
    )
    calib_full, calib_lab, calib_scen = inject_anomalies(
        generate_synthetic_metrics(
            n_steps=cfg.data.n_calibration_steps,
            seed=cfg.data.calibration_seed,
        ),
        seed=cfg.data.calibration_anomaly_seed,
    )

    rows = []
    for kept in [SUBSET_5, FEATURE_NAMES]:
        row = _run_subset(
            cfg,
            device,
            list(kept),
            train_full,
            test_full,
            test_lab,
            test_scen,
            calib_full,
            calib_lab,
            calib_scen,
        )
        rows.append(row)

    rows[0]["name"] = "5 ознак"
    rows[1]["name"] = "12 ознак"

    reporter = Reporter(
        output_dir=cfg.artifacts_dir,
        feature_names=FEATURE_NAMES,
        window_size=cfg.data.window_size,
        style=cfg.plot,
    )
    reporter.save_ablation(rows)

    print("\n" + "=" * 78)
    print("ABLATION: 5 vs 12 ОЗНАК")
    print("=" * 78)
    print(f"{'набір':<12}{'точність':>11}{'повнота':>10}{'F1':>8}{'параметри':>10}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['name']:<12}{r['precision']:>11.4f}{r['recall']:>10.4f}"
            f"{r['f1']:>8.4f}{r['params']:>10,}"
        )
    print("=" * 78)
    print(f"[ablate] збережено у {cfg.artifacts_dir}/ablation.png")
