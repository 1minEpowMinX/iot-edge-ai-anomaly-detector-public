"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DetectorConfig
from .data import DataModule, SCENARIO_NAMES, TestContext
from .detector import AnomalyDetector, per_scenario_recall, prf1
from .predictors import BasePredictor
from .prefilter import (
    ANOMALY,
    NORMAL,
    UNCERTAIN,
    BasePreFilter,
    RoutingStats,
    compute_routing_stats,
)


@dataclass
class RunResult:
    """Everything produced by a single end-to-end pipeline run."""

    name: str
    predictor: BasePredictor
    detector: AnomalyDetector
    history: dict[str, list[float]]
    metrics: dict[str, Any]
    y_pred_test: np.ndarray
    test_errors: np.ndarray
    val_errors: np.ndarray
    per_feature_errors: np.ndarray
    predictions: np.ndarray
    main_predictions: np.ndarray
    train_time_s: float
    infer_ms: float
    routing: RoutingStats | None = None
    per_scenario: dict[str, dict] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        """Detection F1-score."""
        return float(self.metrics["f1"])

    @property
    def precision(self) -> float:
        """Detection precision."""
        return float(self.metrics["precision"])

    @property
    def recall(self) -> float:
        """Detection recall."""
        return float(self.metrics["recall"])

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the predictor."""
        return int(self.predictor.n_params)

    @property
    def threshold(self) -> float:
        """Calibrated anomaly-score threshold (0.0 if unset)."""
        return float(self.detector.threshold or 0.0)

    def to_summary_dict(self) -> dict:
        """Return a compact dict of headline metrics for tables and plots."""
        return {
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "params": self.n_params,
            "train_s": self.train_time_s,
            "infer_ms": self.infer_ms,
        }


class Pipeline:
    """Runs one predictor end-to-end."""

    def __init__(
        self,
        datamodule: DataModule,
        predictor: BasePredictor,
        detector_cfg: DetectorConfig,
        prefilter: BasePreFilter | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            datamodule: Fitted data module supplying train/val windows.
            predictor: Next-step predictor to train and evaluate.
            detector_cfg: Detector configuration (score mode, threshold, ...).
            prefilter: Optional cascade prefilter; ``None`` disables routing.
        """
        self.datamodule = datamodule
        self.predictor = predictor
        self.detector_cfg = detector_cfg
        self.detector = AnomalyDetector(
            predictor,
            percentile=detector_cfg.percentile,
            score_mode=detector_cfg.score_mode,
        )
        self.prefilter = prefilter

    def run(
        self,
        test_ctx: TestContext,
        calibration_ctx: TestContext | None = None,
    ) -> RunResult:
        """Train, calibrate, detect and evaluate end to end.

        Args:
            test_ctx: Labelled test data for the final evaluation.
            calibration_ctx: Labelled calibration data — required when the
                detector uses ``auto_f1`` threshold calibration.

        Returns:
            A :class:`RunResult` with metrics, errors, predictions and routing.

        Raises:
            RuntimeError: If the data module is unfitted or labels are missing.
        """
        if self.datamodule.X_train is None:
            raise RuntimeError("DataModule must be fitted before Pipeline.run()")
        if test_ctx.labels_windowed is None:
            raise RuntimeError("TestContext must include labels for evaluation")

        # 1) train predictor
        history = self.predictor.fit(
            self.datamodule.X_train,
            self.datamodule.y_train,
            self.datamodule.X_val,
            self.datamodule.y_val,
        )

        # 2) calibrate detector threshold
        cfg = self.detector_cfg
        if cfg.threshold_mode == "auto_f1" and calibration_ctx is not None:
            if calibration_ctx.labels_windowed is None:
                raise RuntimeError(
                    "auto_f1 needs labels in the calibration TestContext"
                )
            self.detector.fit_threshold(
                self.datamodule.X_val,
                self.datamodule.y_val,
                mode="auto_f1",
                calibration=(
                    calibration_ctx.X,
                    calibration_ctx.y,
                    calibration_ctx.labels_windowed,
                ),
                sweep_low=cfg.sweep_low,
                sweep_high=cfg.sweep_high,
                sweep_step=cfg.sweep_step,
            )
        else:
            self.detector.fit_threshold(
                self.datamodule.X_val,
                self.datamodule.y_val,
                mode="percentile",
            )

        if self.prefilter is not None:
            self.prefilter.fit(self.datamodule.X_val, self.datamodule.y_val)

        # 3) errors on validation (uses the same score formula as detection)
        val_pred = self.predictor.predict(self.datamodule.X_val)
        _, val_errors = self.detector.score_from_prediction(
            val_pred,
            self.datamodule.y_val,
        )

        # 4) errors on test (single predict + reduce via detector's score formula)
        y_pred = self.predictor.predict(test_ctx.X)
        per_feat, test_errors = self.detector.score_from_prediction(
            y_pred,
            test_ctx.y,
        )
        threshold = self.detector.threshold or 0.0
        main_preds = (test_errors > threshold).astype(np.int64)

        # 5) integrate prefilter routing
        routing = None
        if self.prefilter is not None:
            routes = self.prefilter.classify(test_ctx.X, test_ctx.y)
            if not self.detector_cfg.prefilter_fast_anomaly:
                # Asymmetric cascade: prefilter's ANOMALY verdict is *not* trusted
                # to bypass the main detector. Remap to UNCERTAIN so the GRU gets
                # the final say. This preserves recall — a normal window with a
                # noisy MA pattern can no longer be flagged without GRU confirmation.
                routes = routes.copy()
                routes[routes == ANOMALY] = UNCERTAIN
            preds = main_preds.copy()
            preds[routes == NORMAL] = 0
            preds[routes == ANOMALY] = 1
            routing = compute_routing_stats(routes)
        else:
            preds = main_preds

        metrics = prf1(test_ctx.labels_windowed, preds)
        infer_ms = self.predictor.measure_inference_ms(test_ctx.X)

        per_scenario = {}
        if test_ctx.scenarios_windowed is not None:
            per_scenario = per_scenario_recall(
                test_ctx.scenarios_windowed,
                preds,
                SCENARIO_NAMES,
            )

        return RunResult(
            name=self.predictor.name,
            predictor=self.predictor,
            detector=self.detector,
            history=history,
            metrics=metrics,
            y_pred_test=y_pred,
            test_errors=test_errors,
            val_errors=val_errors,
            per_feature_errors=per_feat,
            predictions=preds,
            main_predictions=main_preds,
            train_time_s=self.predictor.train_time_s,
            infer_ms=infer_ms,
            routing=routing,
            per_scenario=per_scenario,
        )
