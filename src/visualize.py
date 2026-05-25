"""Dashboards and standalone analysis plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve

from .config import PlotStyle
from .prefilter import RoutingStats


# ---------------------------------------------------------------------------
# Dashboard #1: predictions
# ---------------------------------------------------------------------------
def plot_predictions_dashboard(
    history: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    save_path: str,
    show_last: int = 200,
    n_cols: int | None = None,
    style: PlotStyle | None = None,
) -> None:
    """Training curves + per-feature actual vs predicted."""
    style = style or PlotStyle()
    n_features = len(feature_names)
    # 3 columns for small feature sets, 4 for richer ones — keeps the figure
    # readable rather than absurdly tall.
    if n_cols is None:
        n_cols = 3 if n_features <= 6 else 4
    n_rows = 1 + int(np.ceil(n_features / n_cols))

    fig = plt.figure(
        figsize=(15 * style.figsize_scale, 3.2 * n_rows * style.figsize_scale),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(n_rows, n_cols)

    # Training curves on the top row
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(
        history["train"], label="train loss", linewidth=2, color=style.palette[0]
    )
    ax_loss.plot(history["val"], label="val loss", linewidth=2, color=style.palette[1])
    ax_loss.set_title("Криві навчання")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Optional LR overlay on a secondary axis
    if "lr" in history and len(history["lr"]) > 0:
        ax_lr = ax_loss.twinx()
        ax_lr.plot(
            history["lr"], color="gray", linestyle=":", linewidth=1.2, label="lr"
        )
        ax_lr.set_ylabel("learning rate", color="gray")
        ax_lr.tick_params(axis="y", labelcolor="gray")
        ax_lr.set_yscale("log")

    # Per-feature: prediction vs actual on the last show_last steps
    n_show = min(show_last, len(y_true))
    t = np.arange(n_show)
    for k, name in enumerate(feature_names):
        row = 1 + k // n_cols
        col = k % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(
            t,
            y_true[-n_show:, k],
            label="actual",
            linewidth=1.4,
            alpha=0.85,
            color=style.palette[0],
        )
        ax.plot(
            t,
            y_pred[-n_show:, k],
            label="predicted",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            color=style.palette[1],
        )
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Прогнозування часових рядів: навчання та якість", fontsize=style.title_fontsize
    )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dashboard #2: detection
# ---------------------------------------------------------------------------
def plot_detection_dashboard(
    raw_test: np.ndarray,
    labels_full: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    per_feature_errors: np.ndarray,
    labels_windowed: np.ndarray,
    predictions: np.ndarray,
    metrics: dict,
    feature_names: list[str],
    window_size: int,
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Render and save the multi-panel anomaly-detection dashboard.

    Args:
        raw_test: Scaled test metrics, shape ``(n_steps, n_features)``.
        labels_full: Per-step ground-truth anomaly labels.
        errors: Per-window anomaly score.
        threshold: Calibrated detection threshold.
        per_feature_errors: Per-feature error contributions per window.
        labels_windowed: Per-window ground-truth labels.
        predictions: Per-window binary predictions.
        metrics: Metrics dict from :func:`prf1` (incl. the confusion matrix).
        feature_names: Ordered feature names.
        window_size: Sliding-window length.
        save_path: Output image path.
        style: Optional plot styling.
    """
    style = style or PlotStyle()
    n_features = len(feature_names)
    fig = plt.figure(
        figsize=(15 * style.figsize_scale, 11 * style.figsize_scale),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 1.0, 1.4, 1.4])

    # (1) Raw normalized metrics with ground-truth anomaly shading
    ax_raw = fig.add_subplot(gs[0, :])
    t_full = np.arange(len(raw_test))
    rng = np.ptp(raw_test, axis=0)
    norm = (raw_test - raw_test.min(axis=0)) / np.where(rng == 0, 1, rng)
    for k, name in enumerate(feature_names):
        ax_raw.plot(
            t_full,
            norm[:, k],
            label=name,
            linewidth=1.0,
            alpha=0.85,
            color=style.palette[k % len(style.palette)],
        )
    _shade_windows(
        ax_raw, labels_full, color="red", alpha=0.15, label="ground-truth аномалії"
    )
    ax_raw.set_title("Тестовий ряд (метрики нормалізовані) з істинними аномаліями")
    ax_raw.set_xlabel("крок часу")
    ax_raw.legend(loc="upper right", fontsize=8, ncol=3)
    ax_raw.grid(True, alpha=0.3)

    # (2) Anomaly score with threshold and detections
    ax_score = fig.add_subplot(gs[1, :])
    t_win = np.arange(len(errors))
    ax_score.plot(
        t_win, errors, color=style.palette[0], linewidth=1.2, label="anomaly score"
    )
    ax_score.axhline(
        threshold, color="red", linestyle="--", label=f"поріг = {threshold:.3f}"
    )
    _shade_windows(ax_score, predictions, color="orange", alpha=0.25, label="детекції")
    ax_score.set_title("Anomaly score = середнє |pred − actual| по вікнах")
    ax_score.set_xlabel("номер вікна")
    ax_score.legend(loc="upper right", fontsize=9)
    ax_score.grid(True, alpha=0.3)

    # (3) Per-feature error heatmap
    ax_heat = fig.add_subplot(gs[2, :])
    im = ax_heat.imshow(
        per_feature_errors.T,
        aspect="auto",
        cmap=style.cmap_heat,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(n_features))
    ax_heat.set_yticklabels(feature_names)
    ax_heat.set_xlabel("номер вікна")
    ax_heat.set_title("Per-feature помилка прогнозу (де яка метрика 'спрацювала')")
    fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
    anom_idx = np.where(labels_windowed == 1)[0]
    if len(anom_idx) > 0:
        for idx in anom_idx:
            ax_heat.add_patch(
                Rectangle(
                    (idx - 0.5, -0.7),
                    1,
                    0.4,
                    facecolor="cyan",
                    edgecolor="none",
                    clip_on=False,
                )
            )

    # (4) Confusion matrix (computed by sklearn inside prf1(), carried in `metrics`)
    ax_cm = fig.add_subplot(gs[3, 0])
    cm = np.asarray(metrics["confusion_matrix"])
    ax_cm.imshow(cm, cmap=style.cmap_cm)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["pred=normal", "pred=anomaly"])
    ax_cm.set_yticklabels(["true=normal", "true=anomaly"])
    ax_cm.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    # (5) Summary text
    ax_txt = fig.add_subplot(gs[3, 1])
    ax_txt.axis("off")
    text = (
        f"Precision:  {metrics['precision']:.3f}\n"
        f"Recall:     {metrics['recall']:.3f}\n"
        f"F1 score:   {metrics['f1']:.3f}\n"
        f"\n"
        f"TP={metrics['tp']}   FP={metrics['fp']}\n"
        f"FN={metrics['fn']}   TN={metrics['tn']}\n"
        f"\n"
        f"window_size = {window_size}\n"
        f"threshold   = {threshold:.4f}"
    )
    ax_txt.text(
        0.05,
        0.95,
        text,
        transform=ax_txt.transAxes,
        fontsize=12,
        family="monospace",
        va="top",
        bbox=dict(facecolor="#f0f0f0", edgecolor="gray", boxstyle="round,pad=0.6"),
    )

    fig.suptitle(
        "Виявлення аномалій — підсумковий дашборд", fontsize=style.title_fontsize
    )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dashboard #3: model comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(
    comparison: list[dict],
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Render and save the GRU/LSTM/MovingAverage comparison dashboard.

    Args:
        comparison: One summary dict per model, each from
            ``RunResult.to_summary_dict()``.
        save_path: Output image path.
        style: Optional plot styling.
    """
    style = style or PlotStyle()
    names = [c["name"] for c in comparison]
    x = np.arange(len(names))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13 * style.figsize_scale, 8 * style.figsize_scale),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    width = 0.25
    ax.bar(
        x - width,
        [c["precision"] for c in comparison],
        width,
        label="precision",
        color=style.palette[0],
    )
    ax.bar(
        x,
        [c["recall"] for c in comparison],
        width,
        label="recall",
        color=style.palette[1],
    )
    ax.bar(
        x + width, [c["f1"] for c in comparison], width, label="F1", color="forestgreen"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Якість детекції аномалій")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    bars = ax.bar(x, [c["params"] for c in comparison], color=style.palette[0])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Кількість параметрів (менше — легша edge-модель)")
    ax.grid(True, alpha=0.3, axis="y")
    for b, c in zip(bars, comparison):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c['params']:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1, 0]
    bars = ax.bar(x, [c["train_s"] for c in comparison], color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Час навчання, с")
    ax.grid(True, alpha=0.3, axis="y")
    for b, c in zip(bars, comparison):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c['train_s']:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1, 1]
    bars = ax.bar(x, [c["infer_ms"] for c in comparison], color="purple")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Час інференсу одного вікна, мс")
    ax.grid(True, alpha=0.3, axis="y")
    for b, c in zip(bars, comparison):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c['infer_ms']:.3f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Порівняння моделей", fontsize=style.title_fontsize)
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #1: error distribution
# ---------------------------------------------------------------------------
def plot_error_distribution(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    save_path: str,
    bins: int = 50,
    style: PlotStyle | None = None,
) -> None:
    """Histogram of errors: val(normal) vs test-normal vs test-anomaly."""
    style = style or PlotStyle()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13 * style.figsize_scale, 4.5 * style.figsize_scale),
        constrained_layout=True,
    )

    test_normal = test_errors[test_labels == 0]
    test_anom = test_errors[test_labels == 1]

    for ax, log_scale in zip(axes, (False, True)):
        ax.hist(
            val_errors,
            bins=bins,
            alpha=0.5,
            label="val (normal)",
            color=style.palette[0],
        )
        ax.hist(
            test_normal,
            bins=bins,
            alpha=0.5,
            label="test normal",
            color=style.palette[2],
        )
        if len(test_anom) > 0:
            ax.hist(
                test_anom,
                bins=bins,
                alpha=0.7,
                label="test anomaly",
                color=style.palette[3],
            )
        ax.axvline(
            threshold, color="red", linestyle="--", label=f"поріг={threshold:.3f}"
        )
        ax.set_xlabel("anomaly score")
        ax.set_ylabel("частота")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("Розподіл помилок (log-шкала)")
        else:
            ax.set_title("Розподіл помилок")

    fig.suptitle(
        "Розподіл anomaly score: нормальні vs аномальні вікна",
        fontsize=style.title_fontsize,
    )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #2: threshold sweep (precision/recall/F1 vs percentile)
# ---------------------------------------------------------------------------
def plot_threshold_sweep(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Sweep the anomaly-score threshold and plot precision/recall/F1.

    The sweep is the exact precision/recall curve from
    :func:`sklearn.metrics.precision_recall_curve` — one point per distinct
    score value — so the plot shows how each metric reacts to the threshold
    and where F1 peaks.
    """
    style = style or PlotStyle()

    precision, recall, thresholds = precision_recall_curve(
        test_labels,
        test_errors,
    )
    # precision_recall_curve appends a final (recall=0, precision=1) point
    # that has no threshold — drop it so all three arrays align.
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(precision),
        where=denom > 0,
    )

    best_idx = int(np.argmax(f1))
    fig, ax = plt.subplots(
        figsize=(10 * style.figsize_scale, 5 * style.figsize_scale),
        constrained_layout=True,
    )
    ax.plot(
        thresholds, precision, label="precision", color=style.palette[0], linewidth=2
    )
    ax.plot(thresholds, recall, label="recall", color=style.palette[1], linewidth=2)
    ax.plot(thresholds, f1, label="F1", color="forestgreen", linewidth=2)
    ax.axvline(
        thresholds[best_idx],
        color="red",
        linestyle="--",
        label=f"best F1 = {f1[best_idx]:.3f} " f"@ поріг {thresholds[best_idx]:.3f}",
    )
    ax.set_xlabel("поріг anomaly score")
    ax.set_ylabel("метрика")
    ax.set_ylim(0, 1.05)
    ax.set_title("Чутливість метрик до вибору порога", fontsize=style.title_fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #3: prefilter routing
# ---------------------------------------------------------------------------
def plot_routing(
    routing: RoutingStats,
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Bar + text summary of how the cascade prefilter routed test windows."""
    style = style or PlotStyle()
    labels = ["fast: normal", "uncertain → GRU", "fast: anomaly"]
    counts = [routing.n_normal, routing.n_uncertain, routing.n_anomaly]
    colors = [style.palette[2], style.palette[0], style.palette[3]]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11 * style.figsize_scale, 4 * style.figsize_scale),
        constrained_layout=True,
    )
    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors)
    ax.set_title("Розподіл вікон по гілках каскаду")
    ax.set_ylabel("кількість вікон")
    ax.grid(True, alpha=0.3, axis="y")
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            str(c),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[1]
    ax.axis("off")
    fast_ratio = routing.fast_path_ratio
    text = (
        f"Всього вікон:       {routing.n_total}\n"
        f"Швидкий шлях (norm): {routing.n_normal}\n"
        f"GRU (uncertain):     {routing.n_uncertain}\n"
        f"Швидкий шлях (anom): {routing.n_anomaly}\n"
        f"\n"
        f"Економія викликів GRU: {fast_ratio * 100:.1f}%"
    )
    ax.text(
        0.05,
        0.9,
        text,
        transform=ax.transAxes,
        fontsize=12,
        family="monospace",
        va="top",
        bbox=dict(facecolor="#f0f0f0", edgecolor="gray", boxstyle="round,pad=0.6"),
    )
    fig.suptitle(
        "Каскадний prefilter: розподіл навантаження", fontsize=style.title_fontsize
    )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #4: ROC + PR curves
# ---------------------------------------------------------------------------
def plot_roc_pr_curves(
    curves: dict,
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Two-panel ROC and PR curves with AUC annotation."""
    style = style or PlotStyle()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12 * style.figsize_scale, 4.5 * style.figsize_scale),
        constrained_layout=True,
    )

    ax = axes[0]
    order = np.argsort(curves["fpr"])
    ax.plot(
        curves["fpr"][order], curves["tpr"][order], color=style.palette[0], linewidth=2
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6)
    ax.fill_between(
        curves["fpr"][order], 0, curves["tpr"][order], color=style.palette[0], alpha=0.1
    )
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ROC curve  (AUC = {curves['auc_roc']:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    order = np.argsort(curves["recall"])
    ax.plot(
        curves["recall"][order],
        curves["precision"][order],
        color=style.palette[2],
        linewidth=2,
    )
    ax.fill_between(
        curves["recall"][order],
        0,
        curves["precision"][order],
        color=style.palette[2],
        alpha=0.1,
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"PR curve  (AP = {curves['auc_pr']:.3f})")
    ax.grid(True, alpha=0.3)

    fig.suptitle("ROC та PR криві", fontsize=style.title_fontsize)
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #5: per-scenario recall
# ---------------------------------------------------------------------------
def plot_per_scenario(
    per_scenario: dict[str, dict],
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Bar chart: detection rate (recall) per anomaly scenario."""
    style = style or PlotStyle()
    items = [(name, d) for name, d in per_scenario.items() if d["n_windows"] > 0]
    items.sort(key=lambda kv: kv[1]["recall"])
    if not items:
        return
    names = [k for k, _ in items]
    recalls = [d["recall"] for _, d in items]
    n_windows = [d["n_windows"] for _, d in items]

    fig, ax = plt.subplots(
        figsize=(10 * style.figsize_scale, 5 * style.figsize_scale),
        constrained_layout=True,
    )
    bars = ax.barh(names, recalls, color=style.palette[0])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall (частка виявлених вікон цього типу)")
    ax.set_title("Виявлення за типом аномалії", fontsize=style.title_fontsize)
    ax.grid(True, alpha=0.3, axis="x")
    for b, r, n in zip(bars, recalls, n_windows):
        ax.text(
            min(r + 0.02, 1.0),
            b.get_y() + b.get_height() / 2,
            f"{r:.2f}  ({n} вікон)",
            va="center",
            fontsize=9,
        )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #6: ablation comparison (e.g. 5 vs 12 features)
# ---------------------------------------------------------------------------
def plot_ablation_comparison(
    rows: list[dict],
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """rows: [{"name": str, "precision":.., "recall":.., "f1":.., "n_features":int, ...}]"""
    style = style or PlotStyle()
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    width = 0.22

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13 * style.figsize_scale, 4.5 * style.figsize_scale),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.bar(
        x - width,
        [r["precision"] for r in rows],
        width,
        label="precision",
        color=style.palette[0],
    )
    ax.bar(
        x, [r["recall"] for r in rows], width, label="recall", color=style.palette[1]
    )
    ax.bar(x + width, [r["f1"] for r in rows], width, label="F1", color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Якість детекції")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, [r.get("n_features", 0) for r in rows], color=style.palette[2])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Кількість ознак")
    ax.grid(True, alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax.text(
            i,
            r.get("n_features", 0),
            str(r.get("n_features", "")),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Ablation: вплив набору ознак", fontsize=style.title_fontsize)
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #7: hyperparameter sweep
# ---------------------------------------------------------------------------
def plot_sweep_results(
    sweep_name: str,
    x_values: list,
    rows: list[dict],
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """Line plot of metric vs swept variable. ``rows`` has the same length as x_values."""
    style = style or PlotStyle()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13 * style.figsize_scale, 4.5 * style.figsize_scale),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(
        x_values,
        [r["precision"] for r in rows],
        "o-",
        label="precision",
        color=style.palette[0],
        linewidth=2,
    )
    ax.plot(
        x_values,
        [r["recall"] for r in rows],
        "o-",
        label="recall",
        color=style.palette[1],
        linewidth=2,
    )
    ax.plot(
        x_values,
        [r["f1"] for r in rows],
        "o-",
        label="F1",
        color="forestgreen",
        linewidth=2.5,
    )
    ax.set_xlabel(sweep_name)
    ax.set_ylabel("метрика")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Якість vs {sweep_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    train_s = [r.get("train_s", 0.0) for r in rows]
    params = [r.get("params", 0) for r in rows]
    ax.plot(x_values, train_s, "o-", color="darkorange", label="train, s", linewidth=2)
    ax.set_xlabel(sweep_name)
    ax.set_ylabel("train time, s", color="darkorange")
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        x_values,
        params,
        "s--",
        color=style.palette[0],
        label="params",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.set_ylabel("параметрів", color=style.palette[0])
    ax2.tick_params(axis="y", labelcolor=style.palette[0])

    ax.set_title(f"Витрати vs {sweep_name}")

    fig.suptitle(f"Sweep по {sweep_name}", fontsize=style.title_fontsize)
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis #8: evolutionary search history
# ---------------------------------------------------------------------------
def plot_evolution_history(
    history: list,
    save_path: str,
    style: PlotStyle | None = None,
) -> None:
    """4-panel dashboard for GA convergence.

    Top-left: best/mean/worst fitness per generation
    Top-right: best F1 per generation
    Bottom-left: population diversity (unique genomes)
    Bottom-right: best individual's params + inference cost
    """
    style = style or PlotStyle()
    gens = [s.generation for s in history]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13 * style.figsize_scale, 8 * style.figsize_scale),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(
        gens,
        [s.best_fitness for s in history],
        "o-",
        label="best",
        color="forestgreen",
        linewidth=2.5,
    )
    ax.plot(
        gens,
        [s.mean_fitness for s in history],
        "s-",
        label="mean",
        color=style.palette[0],
        linewidth=2,
    )
    ax.plot(
        gens,
        [s.worst_fitness for s in history],
        "^-",
        label="worst",
        color=style.palette[3],
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("покоління")
    ax.set_ylabel("fitness")
    ax.set_title("Збіжність фітнесу")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        gens,
        [s.best_result.f1 for s in history],
        "o-",
        color="forestgreen",
        linewidth=2.5,
    )
    ax.set_xlabel("покоління")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 кращої особини")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    bars = ax.bar(gens, [s.diversity for s in history], color=style.palette[1])
    ax.set_xlabel("покоління")
    ax.set_ylabel("унікальних геномів")
    ax.set_title("Різноманітність популяції")
    ax.grid(True, alpha=0.3, axis="y")
    for b, s in zip(bars, history):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            str(s.diversity),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1, 1]
    params = [s.best_result.n_params for s in history]
    infer = [s.best_result.infer_ms for s in history]
    ax.plot(gens, params, "o-", color=style.palette[0], linewidth=2, label="params")
    ax.set_xlabel("покоління")
    ax.set_ylabel("params", color=style.palette[0])
    ax.tick_params(axis="y", labelcolor=style.palette[0])
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(gens, infer, "s--", color="darkorange", linewidth=2, label="infer, ms")
    ax2.set_ylabel("infer, ms", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("Витрати кращої особини")

    fig.suptitle(
        "Еволюційний пошук — динаміка популяції", fontsize=style.title_fontsize
    )
    plt.savefig(save_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------
def _shade_windows(
    ax,
    labels: np.ndarray,
    color: str,
    alpha: float,
    label: str,
) -> None:
    """Shade contiguous regions where labels == 1."""
    if labels.sum() == 0:
        return
    in_window = False
    start = 0
    first = True
    for i, v in enumerate(labels):
        if v == 1 and not in_window:
            in_window = True
            start = i
        elif v == 0 and in_window:
            in_window = False
            ax.axvspan(
                start, i, color=color, alpha=alpha, label=label if first else None
            )
            first = False
    if in_window:
        ax.axvspan(
            start, len(labels), color=color, alpha=alpha, label=label if first else None
        )
