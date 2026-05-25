"""Inference: score a CSV with a saved model bundle.

Usage:
    python -m src infer --model artifacts/ --data real.csv [-o detections.csv]
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from .. import _ui
from ..artifacts import load_bundle
from ..data import make_windows


def _read_csv(path: str, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Read a metrics CSV, validating it has the required feature columns.

    Args:
        path: Path to the input CSV.
        feature_names: Feature columns the model requires, in order.

    Returns:
        A tuple ``(data, times)`` — the metrics array and the optional ``t``
        column values (empty list if absent).

    Raises:
        ValueError: If the CSV is empty or missing required columns.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    missing = [f for f in feature_names if f not in rows[0]]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Got columns: {list(rows[0].keys())}"
        )
    data = np.asarray(
        [[float(r[f]) for f in feature_names] for r in rows],
        dtype=np.float32,
    )
    times = [r["t"] for r in rows] if "t" in rows[0] else []
    return data, times


def run_infer(
    model_dir: str,
    data_path: str,
    output_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Score a CSV with a saved model bundle and report the detections.

    Args:
        model_dir: Directory holding the saved model bundle.
        data_path: Path to the input metrics CSV.
        output_path: Optional CSV path for per-window detections.
        device: Torch device string.

    Raises:
        FileNotFoundError: If the model directory or data file is missing.
        ValueError: If the CSV has too few rows for the model's window size.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    _ui.banner(
        "ІНФЕРЕНС",
        f"model={model_dir}  data={data_path}  device={device}",
    )

    with _ui.spinner("завантаження моделі"):
        bundle = load_bundle(model_dir, device=device)
    _ui.kv_table(
        "модель",
        [
            ("window_size", bundle.window_size),
            ("n_features", bundle.n_features),
            ("поріг", f"{bundle.threshold:.5f}"),
        ],
    )

    with _ui.spinner(f"читання {data_path}"):
        raw, times = _read_csv(data_path, bundle.feature_names)
    _ui.info(f"{len(raw)} рядків × {raw.shape[1]} ознак")

    if len(raw) <= bundle.window_size:
        raise ValueError(
            f"data has {len(raw)} rows but window_size={bundle.window_size} requires more"
        )

    scaled = bundle.scaler.transform(raw)
    X, y = make_windows(scaled, bundle.window_size)

    _ui.step(1, 1, f"оцінювання {len(X)} вікон")
    bundle.net.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        y_pred = bundle.net(x_t).cpu().numpy()
    errors = np.abs(y_pred - y).mean(axis=1)
    flags = (errors > bundle.threshold).astype(np.int64)

    n_anom = int(flags.sum())
    n_total = len(flags)

    _ui.rule("виявлення")
    _ui.kv_table(
        "підсумок",
        [
            ("оцінено вікон", n_total),
            ("аномалій", n_anom),
            ("частка аномалій", f"{n_anom / max(n_total, 1) * 100:.1f}%"),
            ("макс. оцінка", f"{float(errors.max()):.4f}"),
            ("сер. оцінка", f"{float(errors.mean()):.4f}"),
            ("поріг", f"{bundle.threshold:.4f}"),
        ],
    )
    if n_anom > 0:
        idx_anom = np.where(flags == 1)[0]
        shown = ", ".join(str(i) for i in idx_anom[:10])
        if len(idx_anom) > 10:
            shown += f", … (+{len(idx_anom) - 10} ще)"
        _ui.info(f"індекси аномальних вікон: {shown}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["window_idx", "anomaly_score", "is_anomaly"]
            if times:
                header.insert(1, "t_end")
            w.writerow(header)
            for i, (err, flag) in enumerate(zip(errors, flags)):
                row = [i, float(err), int(flag)]
                if times:
                    row.insert(1, times[i + bundle.window_size])
                w.writerow(row)
        _ui.success(f"виявлення по вікнах збережено у {output_path}")
