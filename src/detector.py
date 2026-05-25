"""Anomaly detection on top of a Predictor.

All evaluation metrics in this module — precision/recall/F1, the confusion
matrix and the ROC/PR curves — are computed with scikit-learn rather than
hand-rolled, so they match the conventions used across the ML ecosystem.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .predictors import BasePredictor


class AnomalyDetector:
    """Threshold-based detector with configurable multivariate score formula.

    Score modes:
      - "mean":        score = mean(|pred - actual|)
      - "zscore_mean": score = mean(|pred - actual| / val_feature_std)
      - "zscore_max":  score = max(|pred - actual|  / val_feature_std)

    The z-score modes normalise per-feature error by that feature's typical
    prediction noise (measured on the validation set). This is the standard
    multivariate AD approach: a small absolute error on a stable feature
    (swap, tcp_conn) becomes a strong signal, while a moderate error on a
    naturally-noisy feature (cpu) is correctly down-weighted.

    ``fit_threshold()`` auto-calls ``fit_normalizer()`` when needed, so the
    typical use is just ``detector.fit_threshold(X_val, y_val, ...)``.
    """

    _EPS = 1e-8

    def __init__(
        self,
        predictor: BasePredictor,
        percentile: float = 99.0,
        score_mode: str = "zscore_mean",
    ) -> None:
        """Initialise the detector.

        Args:
            predictor: Trained next-step predictor that supplies forecasts.
            percentile: Default percentile for ``"percentile"`` threshold mode.
            score_mode: Anomaly-score formula — ``"mean"``, ``"zscore_mean"``
                or ``"zscore_max"``.
        """
        self.predictor = predictor
        self.percentile = percentile
        self.score_mode = score_mode
        self.threshold: float | None = None
        self.threshold_mode: str | None = None
        self.feature_std: np.ndarray | None = None

    # ------------------------------------------------------------------ scoring
    def fit_normalizer(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "AnomalyDetector":
        """Compute per-feature std of prediction errors on (normal) validation data."""
        if self.score_mode == "mean":
            return self  # no normalisation needed
        pred = self.predictor.predict(X_val)
        per_feat = np.abs(pred - y_val)
        self.feature_std = per_feat.std(axis=0) + self._EPS
        return self

    def _reduce(self, per_feat_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw |err| to (per_feature_score, scalar_score) per mode."""
        if self.score_mode == "mean":
            return per_feat_raw, per_feat_raw.mean(axis=1)
        if self.feature_std is None:
            raise RuntimeError(
                f"score_mode={self.score_mode!r} requires fit_normalizer() first"
            )
        per_feat = per_feat_raw / self.feature_std
        if self.score_mode == "zscore_max":
            return per_feat, per_feat.max(axis=1)
        return per_feat, per_feat.mean(axis=1)  # zscore_mean

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the scalar anomaly score for each window in ``X``.

        Args:
            X: Input windows, shape ``(batch, window, n_features)``.
            y: Ground-truth next steps, shape ``(batch, n_features)``.

        Returns:
            One anomaly score per window, shape ``(batch,)``.
        """
        pred = self.predictor.predict(X)
        _, s = self._reduce(np.abs(pred - y))
        return s

    def per_feature_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the per-feature error contributions for each window.

        Args:
            X: Input windows, shape ``(batch, window, n_features)``.
            y: Ground-truth next steps, shape ``(batch, n_features)``.

        Returns:
            Per-feature scores, shape ``(batch, n_features)`` — useful for
            attributing which metric drove a detection.
        """
        pred = self.predictor.predict(X)
        pf, _ = self._reduce(np.abs(pred - y))
        return pf

    def score_from_prediction(
        self,
        y_pred: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (per_feature_score, scalar_score). Use when y_pred is already computed."""
        return self._reduce(np.abs(y_pred - y))

    # ---------------------------------------------------------------- threshold
    def fit_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        mode: str = "percentile",
        calibration: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        sweep_low: float = 90.0,
        sweep_high: float = 99.9,
        sweep_step: float = 0.25,
    ) -> "AnomalyDetector":
        """Calibrate the detection threshold and return ``self``.

        Args:
            X_val: Validation input windows (normal data).
            y_val: Validation next-step targets.
            mode: ``"percentile"`` (fixed percentile of validation errors) or
                ``"auto_f1"`` (sweep percentiles for the best F1 on a labelled
                calibration set).
            calibration: ``(X, y, labels)`` calibration tuple — required when
                ``mode="auto_f1"``.
            sweep_low: Lowest percentile tried in the ``auto_f1`` sweep.
            sweep_high: Highest percentile tried in the ``auto_f1`` sweep.
            sweep_step: Percentile step size for the ``auto_f1`` sweep.

        Raises:
            ValueError: If ``mode`` is unknown, or ``auto_f1`` is requested
                without a calibration tuple.
        """
        # Auto-fit the normaliser before computing any score
        if self.score_mode != "mean" and self.feature_std is None:
            self.fit_normalizer(X_val, y_val)

        val_errors = self.score(X_val, y_val)
        self.threshold_mode = mode

        if mode == "percentile":
            self.threshold = float(np.percentile(val_errors, self.percentile))
        elif mode == "auto_f1":
            if calibration is None:
                raise ValueError(
                    "threshold_mode='auto_f1' requires a calibration tuple"
                )
            self.threshold = self._fit_auto_f1(
                val_errors,
                calibration,
                sweep_low,
                sweep_high,
                sweep_step,
            )
        else:
            raise ValueError(f"Unknown threshold mode: {mode!r}")
        return self

    def _fit_auto_f1(
        self,
        val_errors: np.ndarray,
        calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
        sweep_low: float,
        sweep_high: float,
        sweep_step: float,
    ) -> float:
        """Sweep validation-error percentiles for the best F1 on calibration data.

        Args:
            val_errors: Anomaly scores on the validation set; thresholds are
                taken as percentiles of these values.
            calibration: ``(X, y, labels)`` calibration set containing
                anomalies, used to score each candidate threshold.
            sweep_low: Lowest percentile tried.
            sweep_high: Highest percentile tried.
            sweep_step: Percentile step size.

        Returns:
            The threshold (a validation-error value) that maximised F1.
        """
        X_cal, y_cal, labels_cal = calibration
        cal_errors = self.score(X_cal, y_cal)
        percentiles = np.arange(sweep_low, sweep_high + 1e-9, sweep_step)

        best_f1 = -1.0
        best_p = sweep_high
        best_thr = float(np.percentile(val_errors, sweep_high))

        for p in percentiles:
            thr = float(np.percentile(val_errors, p))
            preds = (cal_errors > thr).astype(np.int64)
            f1 = f1_score(labels_cal, preds, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_p = float(p)
                best_thr = thr

        self.percentile = best_p
        return best_thr

    # --------------------------------------------------------------- inference
    def detect(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Classify each window as normal (0) or anomaly (1).

        Args:
            X: Input windows, shape ``(batch, window, n_features)``.
            y: Ground-truth next steps, shape ``(batch, n_features)``.

        Returns:
            Binary predictions, shape ``(batch,)``.

        Raises:
            RuntimeError: If called before ``fit_threshold()``.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit_threshold() before detect()")
        return (self.score(X, y) > self.threshold).astype(np.int64)

    @staticmethod
    def attribute(
        per_feature_errors: np.ndarray,
        feature_names: list[str],
        top_k: int = 1,
    ) -> list[list[str]]:
        """Name the top-K features that contributed most to each window's error.

        Args:
            per_feature_errors: Per-feature error scores, shape
                ``(batch, n_features)``.
            feature_names: Feature names indexed like the error columns.
            top_k: How many top features to return per window.

        Returns:
            For each window, a list of the ``top_k`` highest-error feature
            names, ordered most-to-least responsible.
        """
        top_indices = np.argsort(-per_feature_errors, axis=1)[:, :top_k]
        return [[feature_names[i] for i in row] for row in top_indices]


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision/Recall/F1 and the confusion matrix, computed via scikit-learn.

    Binary precision/recall/F1 come from
    :func:`sklearn.metrics.precision_recall_fscore_support` (``average=
    "binary"``, ``pos_label=1``); the TP/FP/FN/TN counts come from
    :func:`sklearn.metrics.confusion_matrix` pinned to ``labels=[0, 1]``,
    so the result stays a well-defined 2x2 matrix even when one class is
    missing from ``y_true``/``y_pred``.

    Args:
        y_true: Ground-truth binary labels (0 = normal, 1 = anomaly).
        y_pred: Predicted binary labels.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, the integer counts
        ``tp``/``fp``/``fn``/``tn`` and the raw 2x2 ``confusion_matrix``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "confusion_matrix": cm,
    }


def per_scenario_recall(
    scenarios_windowed: np.ndarray,
    predictions: np.ndarray,
    scenario_names: tuple[str, ...],
) -> dict[str, dict]:
    """Per-scenario detection rate (recall on the windows of that type)."""
    out: dict[str, dict] = {}
    for code, name in enumerate(scenario_names):
        if code == 0:
            continue
        mask = scenarios_windowed == code
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n_windows": 0, "tp": 0, "fn": 0, "recall": 0.0}
            continue
        tp = int(((mask) & (predictions == 1)).sum())
        fn = n - tp
        out[name] = {
            "n_windows": n,
            "tp": tp,
            "fn": fn,
            "recall": tp / n,
        }
    return out


def roc_pr_curves(errors: np.ndarray, labels: np.ndarray) -> dict:
    """ROC and Precision-Recall curves computed with scikit-learn.

    The ROC curve comes from :func:`sklearn.metrics.roc_curve` and the PR
    curve from :func:`sklearn.metrics.precision_recall_curve`; both are
    evaluated at every distinct score threshold, i.e. an exact sweep rather
    than a fixed-size grid. ``auc_roc`` is the ROC area
    (:func:`sklearn.metrics.roc_auc_score`); ``auc_pr`` is the average
    precision (:func:`sklearn.metrics.average_precision_score`) — the
    standard PR summary, which avoids the optimistic linear interpolation
    of a trapezoidal PR-AUC.

    Args:
        errors: Continuous anomaly score, one value per window.
        labels: Binary ground truth (1 = anomaly).

    Returns:
        Dict with the ROC arrays ``fpr``/``tpr``/``roc_thresholds``, the PR
        arrays ``precision``/``recall``/``pr_thresholds``, the per-point PR
        ``f1`` and the scalar areas ``auc_roc``/``auc_pr``. ROC and PR
        arrays have different lengths — each is its own threshold sweep.
    """
    errors = np.asarray(errors, dtype=float)
    labels = np.asarray(labels).astype(int)

    # roc_curve / precision_recall_curve / the AUC scorers all need both
    # classes present; fall back to a degenerate curve otherwise.
    if labels.min() == labels.max():
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "roc_thresholds": np.array([1.0, 0.0]),
            "precision": np.array([1.0, 1.0]),
            "recall": np.array([1.0, 0.0]),
            "pr_thresholds": np.array([0.0]),
            "f1": np.array([0.0, 0.0]),
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
        }

    fpr, tpr, roc_thresholds = roc_curve(labels, errors)
    precision, recall, pr_thresholds = precision_recall_curve(labels, errors)

    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(precision),
        where=denom > 0,
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "f1": f1,
        "auc_roc": float(roc_auc_score(labels, errors)),
        "auc_pr": float(average_precision_score(labels, errors)),
    }
