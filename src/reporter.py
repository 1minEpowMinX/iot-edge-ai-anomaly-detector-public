"""Reporter: console output + dashboards + artifacts."""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable

from .config import PlotStyle
from .data import DataModule, TestContext
from .artifacts import save_meta, save_scaler, save_torch_predictor
from .detector import AnomalyDetector, roc_pr_curves
from .pipeline import RunResult
from .predictors import TorchPredictor
from .visualize import (
    plot_ablation_comparison,
    plot_detection_dashboard,
    plot_error_distribution,
    plot_evolution_history,
    plot_model_comparison,
    plot_per_scenario,
    plot_predictions_dashboard,
    plot_roc_pr_curves,
    plot_routing,
    plot_sweep_results,
    plot_threshold_sweep,
)


class Reporter:
    """Console output + on-disk dashboards + artifacts."""

    def __init__(
        self,
        output_dir: str,
        feature_names: list[str],
        window_size: int,
        style: PlotStyle | None = None,
    ) -> None:
        """Initialise the reporter.

        Args:
            output_dir: Directory where dashboards and artefacts are written.
            feature_names: Ordered feature names.
            window_size: Sliding-window length.
            style: Optional plot styling.
        """
        self.output_dir = output_dir
        self.feature_names = feature_names
        self.window_size = window_size
        self.style = style or PlotStyle()
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------ console ---------------------------------
    def print_comparison(self, results: Iterable[RunResult]) -> None:
        """Print a console table comparing several run results."""
        results = list(results)
        print("\n" + "=" * 78)
        print("РЕЗУЛЬТАТИ МОДЕЛЕЙ")
        print("=" * 78)
        print(
            f"{'модель':<12}{'precision':>11}{'recall':>10}{'F1':>8}"
            f"{'params':>10}{'train,s':>10}{'infer,ms':>11}"
        )
        print("-" * 78)
        for r in results:
            print(
                f"{r.name:<12}{r.precision:>11.4f}{r.recall:>10.4f}"
                f"{r.f1:>8.4f}{r.n_params:>10,}"
                f"{r.train_time_s:>10.2f}{r.infer_ms:>11.4f}"
            )
        print("=" * 78)

    def print_attribution(self, result: RunResult) -> None:
        """Print which feature most often triggered each detection."""
        triggered_idx = (result.predictions == 1).nonzero()[0]
        if len(triggered_idx) == 0:
            return
        attribution = AnomalyDetector.attribute(
            result.per_feature_errors[triggered_idx],
            self.feature_names,
            top_k=1,
        )
        cnt = Counter(a[0] for a in attribution)
        print(f"\n[{result.name}] виявлення за причинами (top-1 ознака):")
        for name, c in cnt.most_common():
            print(f"  {name:>16}: {c}")

    def print_routing(self, result: RunResult) -> None:
        """Print the cascade prefilter's routing breakdown."""
        if result.routing is None:
            return
        r = result.routing
        print(
            f"\n[{result.name}] каскадне маршрутизування "
            f"({r.fast_path_ratio * 100:.1f}% швидким шляхом):"
        )
        print(f"  normal (фільтр):  {r.n_normal}")
        print(f"  uncertain -> GRU:  {r.n_uncertain}")
        print(f"  anomaly (фільтр): {r.n_anomaly}")

    def print_per_scenario(self, result: RunResult) -> None:
        """Print detection recall broken down by anomaly scenario."""
        if not result.per_scenario:
            return
        print(f"\n[{result.name}] recall по типу аномалії:")
        for name, d in sorted(
            result.per_scenario.items(), key=lambda kv: -kv[1]["recall"]
        ):
            if d["n_windows"] == 0:
                continue
            print(f"  {name:>16}: {d['recall']:.3f} " f"({d['tp']}/{d['n_windows']})")

    def print_threshold(self, result: RunResult) -> None:
        """Print the calibrated threshold and the mode used to find it."""
        mode = result.detector.threshold_mode or "?"
        print(
            f"\n[{result.name}] поріг = {result.threshold:.5f}  "
            f"(mode={mode}, percentile={result.detector.percentile:.2f})"
        )

    # -------------------------------- plots ---------------------------------
    def save_predictions_dashboard(
        self,
        result: RunResult,
        test_ctx: TestContext,
        filename: str = "predictions_dashboard.png",
    ) -> str:
        """Save the training-curves and per-feature predictions dashboard.

        Args:
            result: The pipeline run result to visualise.
            test_ctx: Test context supplying the ground-truth series.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.
        """
        path = os.path.join(self.output_dir, filename)
        plot_predictions_dashboard(
            history=result.history,
            y_true=test_ctx.y,
            y_pred=result.y_pred_test,
            feature_names=self.feature_names,
            save_path=path,
            style=self.style,
        )
        return path

    def save_detection_dashboard(
        self,
        result: RunResult,
        test_ctx: TestContext,
        filename: str = "detection_dashboard.png",
    ) -> str:
        """Save the multi-panel anomaly-detection dashboard.

        Args:
            result: The pipeline run result to visualise.
            test_ctx: Test context supplying raw series and labels.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.

        Raises:
            ValueError: If the test context lacks ground-truth labels.
        """
        if test_ctx.labels_full is None or test_ctx.labels_windowed is None:
            raise ValueError("detection dashboard needs ground-truth labels")
        path = os.path.join(self.output_dir, filename)
        plot_detection_dashboard(
            raw_test=test_ctx.raw_scaled,
            labels_full=test_ctx.labels_full,
            errors=result.test_errors,
            threshold=result.threshold,
            per_feature_errors=result.per_feature_errors,
            labels_windowed=test_ctx.labels_windowed,
            predictions=result.predictions,
            metrics=result.metrics,
            feature_names=self.feature_names,
            window_size=self.window_size,
            save_path=path,
            style=self.style,
        )
        return path

    def save_comparison_dashboard(
        self,
        results: Iterable[RunResult],
        filename: str = "model_comparison.png",
    ) -> str:
        """Save the model-comparison dashboard.

        Args:
            results: One run result per compared model.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.
        """
        path = os.path.join(self.output_dir, filename)
        plot_model_comparison(
            comparison=[r.to_summary_dict() for r in results],
            save_path=path,
            style=self.style,
        )
        return path

    def save_error_distribution(
        self,
        result: RunResult,
        test_ctx: TestContext,
        filename: str = "error_distribution.png",
    ) -> str:
        """Save the anomaly-score distribution histogram.

        Args:
            result: The pipeline run result to visualise.
            test_ctx: Test context supplying ground-truth labels.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.

        Raises:
            ValueError: If the test context lacks ground-truth labels.
        """
        if test_ctx.labels_windowed is None:
            raise ValueError("error distribution needs ground-truth labels")
        path = os.path.join(self.output_dir, filename)
        plot_error_distribution(
            val_errors=result.val_errors,
            test_errors=result.test_errors,
            test_labels=test_ctx.labels_windowed,
            threshold=result.threshold,
            save_path=path,
            style=self.style,
        )
        return path

    def save_threshold_sweep(
        self,
        result: RunResult,
        test_ctx: TestContext,
        filename: str = "threshold_sweep.png",
    ) -> str:
        """Save the precision/recall/F1-versus-threshold sweep plot.

        Args:
            result: The pipeline run result to visualise.
            test_ctx: Test context supplying ground-truth labels.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.

        Raises:
            ValueError: If the test context lacks ground-truth labels.
        """
        if test_ctx.labels_windowed is None:
            raise ValueError("threshold sweep needs ground-truth labels")
        path = os.path.join(self.output_dir, filename)
        plot_threshold_sweep(
            test_errors=result.test_errors,
            test_labels=test_ctx.labels_windowed,
            save_path=path,
            style=self.style,
        )
        return path

    def save_roc_pr(
        self,
        result: RunResult,
        test_ctx: TestContext,
        filename: str = "roc_pr_curves.png",
    ) -> str:
        """Save the ROC and Precision-Recall curve plot.

        Args:
            result: The pipeline run result to visualise.
            test_ctx: Test context supplying ground-truth labels.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.

        Raises:
            ValueError: If the test context lacks ground-truth labels.
        """
        if test_ctx.labels_windowed is None:
            raise ValueError("ROC/PR needs ground-truth labels")
        path = os.path.join(self.output_dir, filename)
        # use MAIN detector predictions errors (without prefilter) so the
        # curves reflect the model itself, not the cascade
        curves = roc_pr_curves(result.test_errors, test_ctx.labels_windowed)
        plot_roc_pr_curves(curves, save_path=path, style=self.style)
        return path

    def save_per_scenario(
        self,
        result: RunResult,
        filename: str = "per_scenario.png",
    ) -> str | None:
        """Save the per-scenario recall bar chart.

        Args:
            result: The pipeline run result to visualise.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image, or ``None`` if there is no
            per-scenario data.
        """
        if not result.per_scenario:
            return None
        path = os.path.join(self.output_dir, filename)
        plot_per_scenario(result.per_scenario, save_path=path, style=self.style)
        return path

    def save_routing(
        self,
        result: RunResult,
        filename: str = "routing.png",
    ) -> str | None:
        """Save the cascade-routing summary plot.

        Args:
            result: The pipeline run result to visualise.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image, or ``None`` if no prefilter
            was used.
        """
        if result.routing is None:
            return None
        path = os.path.join(self.output_dir, filename)
        plot_routing(result.routing, save_path=path, style=self.style)
        return path

    def save_ablation(
        self,
        rows: list[dict],
        filename: str = "ablation.png",
    ) -> str:
        """Save the feature-ablation comparison plot.

        Args:
            rows: One summary dict per feature subset.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.
        """
        path = os.path.join(self.output_dir, filename)
        plot_ablation_comparison(rows, save_path=path, style=self.style)
        return path

    def save_evolution_history(
        self,
        history: list,
        filename: str = "evolution_history.png",
    ) -> str:
        """Save the genetic-algorithm convergence dashboard.

        Args:
            history: Per-generation ``GenerationStats`` records.
            filename: Output file name within the reporter's directory.

        Returns:
            The full path of the written image.
        """
        path = os.path.join(self.output_dir, filename)
        plot_evolution_history(history, save_path=path, style=self.style)
        return path

    def save_sweep(
        self,
        sweep_name: str,
        x_values: list,
        rows: list[dict],
        filename: str | None = None,
    ) -> str:
        """Save a hyperparameter-sweep plot.

        Args:
            sweep_name: Name of the swept axis (e.g. ``"window_size"``).
            x_values: Swept values, one per row.
            rows: One summary dict per swept value.
            filename: Output file name; defaults to ``sweep_<name>.png``.

        Returns:
            The full path of the written image.
        """
        filename = filename or f"sweep_{sweep_name}.png"
        path = os.path.join(self.output_dir, filename)
        plot_sweep_results(
            sweep_name,
            x_values,
            rows,
            save_path=path,
            style=self.style,
        )
        return path

    # ---------------------------- artifacts ---------------------------------
    def save_artifacts(
        self,
        result: RunResult,
        datamodule: DataModule,
        config: dict,
    ) -> None:
        """Persist the trained model bundle (weights, scaler, metadata).

        Args:
            result: The pipeline run result holding the trained predictor.
            datamodule: Data module supplying the fitted scaler.
            config: Serialised application configuration.

        Raises:
            TypeError: If the predictor is not a :class:`TorchPredictor`.
        """
        if not isinstance(result.predictor, TorchPredictor):
            raise TypeError(
                "save_artifacts() expects a TorchPredictor "
                f"(got {type(result.predictor).__name__})"
            )
        save_torch_predictor(self.output_dir, result.predictor)
        save_scaler(self.output_dir, datamodule.scaler)
        save_meta(
            self.output_dir,
            threshold=result.threshold,
            history=result.history,
            feature_names=self.feature_names,
            config=config,
        )

    # ---------------------------- convenience -------------------------------
    def save_production_report(
        self,
        result: RunResult,
        test_ctx: TestContext,
        datamodule: DataModule,
        config: dict,
    ) -> None:
        """Save the full production report: every dashboard plus artefacts.

        Args:
            result: The pipeline run result to report on.
            test_ctx: Test context supplying series and labels.
            datamodule: Data module supplying the fitted scaler.
            config: Serialised application configuration.
        """
        self.save_predictions_dashboard(result, test_ctx)
        self.save_detection_dashboard(result, test_ctx)
        self.save_error_distribution(result, test_ctx)
        self.save_threshold_sweep(result, test_ctx)
        self.save_roc_pr(result, test_ctx)
        self.save_per_scenario(result)
        self.save_routing(result)
        self.save_artifacts(result, datamodule, config)
