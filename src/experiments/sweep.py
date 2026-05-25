"""Hyperparameter sweeps: window_size and hidden_size."""

from __future__ import annotations

from typing import Callable

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


def _build_base_config(output_dir: str) -> AppConfig:
    """Build the base sweep config with a shorter training budget.

    Args:
        output_dir: Directory for artefacts.

    Returns:
        The configured :class:`AppConfig`.
    """
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.train.epochs = 60
    return cfg


def _run_one(cfg: AppConfig, label: str, device: str) -> dict:
    """Train and evaluate one configuration point of a sweep.

    Args:
        cfg: Configuration for this sweep point.
        label: Human-readable label (e.g. ``"window_size=30"``).
        device: Torch device string.

    Returns:
        A summary dict of metrics for this sweep point.
    """
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

    datamodule = DataModule(
        window_size=cfg.data.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(train_full)
    test_ctx = datamodule.prepare_test(test_full, test_lab, test_scen)
    calib_ctx = datamodule.prepare_test(calib_full, calib_lab, calib_scen)

    set_seed(cfg.seed)
    predictor = TorchPredictor(
        net=GRUNet(
            n_features=N_FEATURES,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ),
        name=label,
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
    print(f"\n[sweep] {label}")
    result = pipeline.run(test_ctx, calibration_ctx=calib_ctx)
    return result.to_summary_dict()


def _run_named_sweep(
    sweep_name: str,
    values: list,
    apply_value: Callable,
    device: str,
    reporter: Reporter,
    output_dir: str,
) -> list[dict]:
    """Sweep one named axis over its values and save the plot.

    Args:
        sweep_name: Name of the swept axis.
        values: Values to sweep over.
        apply_value: Callback applying a value to a config in place.
        device: Torch device string.
        reporter: Reporter used to save the sweep plot.
        output_dir: Directory for artefacts.

    Returns:
        One summary dict per swept value.
    """
    rows = []
    for v in values:
        cfg = _build_base_config(output_dir)
        apply_value(cfg, v)
        label = f"{sweep_name}={v}"
        row = _run_one(cfg, label, device)
        row[sweep_name] = v
        rows.append(row)

    reporter.save_sweep(sweep_name, values, rows)

    print("\n" + "=" * 78)
    print(f"SWEEP: {sweep_name}")
    print("=" * 78)
    print(
        f"{sweep_name:<14}{'precision':>11}{'recall':>10}{'F1':>8}"
        f"{'params':>10}{'train,s':>10}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{str(r[sweep_name]):<14}{r['precision']:>11.4f}"
            f"{r['recall']:>10.4f}{r['f1']:>8.4f}"
            f"{r['params']:>10,}{r['train_s']:>10.2f}"
        )
    print("=" * 78)
    return rows


def run_sweep(which: str = "all", output_dir: str = "artifacts/sweeps") -> None:
    """Run window_size, hidden_size, or both sweeps.

    Args:
        which: "window_size", "hidden_size", or "all".
        output_dir: where to save the sweep dashboards.
    """
    base = _build_base_config(output_dir)
    device = resolve_device(base.device)
    reporter = Reporter(
        output_dir=base.artifacts_dir,
        feature_names=FEATURE_NAMES,
        window_size=base.data.window_size,
        style=base.plot,
    )
    print(f"[sweep] пристрій: {device}")

    if which in ("all", "window_size"):
        _run_named_sweep(
            "window_size",
            values=[20, 30, 50, 80],
            apply_value=lambda cfg, v: setattr(cfg.data, "window_size", v),
            device=device,
            reporter=reporter,
            output_dir=output_dir,
        )
    if which in ("all", "hidden_size"):
        _run_named_sweep(
            "hidden_size",
            values=[16, 32, 64, 128],
            apply_value=lambda cfg, v: setattr(cfg.model, "hidden_size", v),
            device=device,
            reporter=reporter,
            output_dir=output_dir,
        )
    print(f"[sweep] збережено у {output_dir}/")
