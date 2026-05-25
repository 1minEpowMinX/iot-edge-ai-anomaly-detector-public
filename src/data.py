"""Synthetic data generation, scaler, sliding windows and the DataModule facade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    # CPU
    "cpu",  # overall CPU usage %
    "cpu_iowait",  # CPU time waiting for disk I/O %
    "load_avg_1m",  # 1-minute system load average
    # Memory
    "ram",  # RAM usage %
    "swap",  # swap usage %
    # Disk
    "disk_read_bps",  # disk read bytes/sec
    "disk_write_bps",  # disk write bytes/sec
    # Network
    "net_tx",  # bytes/sec sent
    "net_rx",  # bytes/sec received
    "net_packets_tx",  # packets/sec sent
    # Connections / processes
    "tcp_conn",  # ESTABLISHED TCP connection count
    "proc_count",  # running process count
]
N_FEATURES = len(FEATURE_NAMES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Subset matching the original thesis spec — used by ablate.py
SUBSET_5 = ["cpu", "ram", "net_tx", "net_rx", "tcp_conn"]

# Scenario codes for per-scenario evaluation
SCENARIO_NAMES = (
    "normal",  # 0
    "cpu_stress",  # 1
    "network_spike",  # 2
    "connection_flood",  # 3
    "io_storm",  # 4
    "memory_pressure",  # 5
    "fork_storm",  # 6
)
SCENARIO_CODE = {name: i for i, name in enumerate(SCENARIO_NAMES)}
_ATTACK_SCENARIOS = tuple(SCENARIO_NAMES[1:])


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------
def generate_synthetic_metrics(n_steps: int, seed: int = 42) -> np.ndarray:
    """Return ``(n_steps, N_FEATURES)`` of plausible normal IoT metrics."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    cpu = 40 + 20 * np.sin(2 * np.pi * t / 200) + rng.normal(0, 3, n_steps)
    cpu = np.clip(cpu, 0, 100)

    cpu_iowait = np.abs(rng.normal(1.5, 1.2, n_steps))
    cpu_iowait += 4 * (rng.random(n_steps) > 0.97).astype(np.float32)
    cpu_iowait = np.clip(cpu_iowait, 0, 100)

    load_avg_1m = cpu / 50.0 + rng.normal(0, 0.15, n_steps)
    load_avg_1m = np.clip(load_avg_1m, 0, 16)

    ram = 55 + 5 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 1.5, n_steps)
    ram = np.clip(ram, 0, 100)

    swap = np.abs(rng.normal(4, 1.5, n_steps))
    swap = np.clip(swap, 0, 100)

    disk_read_bps = np.abs(rng.normal(40_000, 15_000, n_steps))
    disk_read_bps += 400_000 * (np.sin(2 * np.pi * t / 130) > 0.92).astype(np.float32)

    disk_write_bps = np.abs(rng.normal(20_000, 8_000, n_steps))
    disk_write_bps += 200_000 * (np.sin(2 * np.pi * t / 110 + 0.5) > 0.91).astype(
        np.float32
    )

    net_tx = np.abs(rng.normal(1500, 400, n_steps))
    net_tx += 500 * (np.sin(2 * np.pi * t / 150) > 0.7).astype(np.float32)

    net_rx = np.abs(rng.normal(2000, 500, n_steps))
    net_rx += 700 * (np.sin(2 * np.pi * t / 170 + 1) > 0.6).astype(np.float32)

    net_packets_tx = np.round(net_tx / 800 + rng.normal(0, 0.5, n_steps))
    net_packets_tx = np.clip(net_packets_tx, 0, None)

    tcp_conn = np.round(
        15 + 3 * np.sin(2 * np.pi * t / 300) + rng.normal(0, 1, n_steps)
    )
    tcp_conn = np.clip(tcp_conn, 0, None)

    proc_count = np.round(
        110 + 8 * np.sin(2 * np.pi * t / 400) + rng.normal(0, 2, n_steps)
    )
    proc_count = np.clip(proc_count, 0, None)

    return np.stack(
        [
            cpu,
            cpu_iowait,
            load_avg_1m,
            ram,
            swap,
            disk_read_bps,
            disk_write_bps,
            net_tx,
            net_rx,
            net_packets_tx,
            tcp_conn,
            proc_count,
        ],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------
def inject_anomalies(
    data: np.ndarray,
    seed: int = 7,
    n_windows: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert correlated multi-metric anomaly windows into a metrics array.

    Args:
        data: Normal metrics array, shape ``(n_steps, n_features)``.
        seed: RNG seed controlling anomaly placement and intensity.
        n_windows: Number of anomaly windows to inject.

    Returns:
        A tuple ``(data, labels, scenarios)`` where ``data`` is the modified
        copy, ``labels`` are per-step binary anomaly flags and ``scenarios``
        are per-step scenario codes (0 = normal, 1-6 = scenario index).
    """
    rng = np.random.default_rng(seed)
    data = data.copy()
    n = len(data)
    labels = np.zeros(n, dtype=np.int64)
    scenarios = np.zeros(n, dtype=np.int64)

    # Adapt window placement to dataset size so small splits don't crash.
    min_start = min(50, max(5, n // 5))
    max_start = n - 30
    if max_start <= min_start:
        return data, labels, scenarios
    n_windows = min(n_windows, max(1, (max_start - min_start) // 15))

    for _ in range(n_windows):
        start = int(rng.integers(min_start, max_start))
        length = int(rng.integers(10, min(22, n - start - 2)))
        kind = str(rng.choice(_ATTACK_SCENARIOS))
        end = start + length
        labels[start:end] = 1
        scenarios[start:end] = SCENARIO_CODE[kind]
        _apply_scenario(data, start, end, length, kind, rng)

    data[:, _IDX["cpu"]] = np.clip(data[:, _IDX["cpu"]], 0, 100)
    data[:, _IDX["cpu_iowait"]] = np.clip(data[:, _IDX["cpu_iowait"]], 0, 100)
    data[:, _IDX["ram"]] = np.clip(data[:, _IDX["ram"]], 0, 100)
    data[:, _IDX["swap"]] = np.clip(data[:, _IDX["swap"]], 0, 100)
    data[:, _IDX["load_avg_1m"]] = np.clip(data[:, _IDX["load_avg_1m"]], 0, 16)
    data[:, _IDX["tcp_conn"]] = np.clip(data[:, _IDX["tcp_conn"]], 0, None)
    data[:, _IDX["proc_count"]] = np.clip(data[:, _IDX["proc_count"]], 0, None)
    return data, labels, scenarios


def _apply_scenario(
    data: np.ndarray,
    start: int,
    end: int,
    length: int,
    kind: str,
    rng: np.random.Generator,
) -> None:
    """Overwrite one window of ``data`` in place with a scenario's signature.

    Args:
        data: Metrics array being mutated in place.
        start: Inclusive start index of the anomaly window.
        end: Exclusive end index of the anomaly window.
        length: Window length (``end - start``).
        kind: Scenario name (e.g. ``"cpu_stress"``, ``"io_storm"``).
        rng: Random generator for the scenario's noise.
    """
    if kind == "cpu_stress":
        data[start:end, _IDX["cpu"]] = 95 + rng.normal(0, 2, length)
        data[start:end, _IDX["load_avg_1m"]] = 4 + rng.normal(0, 0.5, length)
        data[start:end, _IDX["proc_count"]] += rng.normal(15, 3, length)
    elif kind == "network_spike":
        mul = rng.uniform(5, 10)
        data[start:end, _IDX["net_tx"]] *= mul
        data[start:end, _IDX["net_rx"]] *= mul
        data[start:end, _IDX["net_packets_tx"]] *= mul * 0.8
    elif kind == "connection_flood":
        data[start:end, _IDX["tcp_conn"]] = 80 + rng.normal(0, 5, length)
        data[start:end, _IDX["net_packets_tx"]] *= rng.uniform(3, 6)
        data[start:end, _IDX["cpu"]] += rng.uniform(15, 25)
    elif kind == "io_storm":
        data[start:end, _IDX["disk_read_bps"]] *= rng.uniform(10, 20)
        data[start:end, _IDX["disk_write_bps"]] *= rng.uniform(8, 15)
        data[start:end, _IDX["cpu_iowait"]] = 25 + rng.normal(0, 4, length)
    elif kind == "memory_pressure":
        data[start:end, _IDX["ram"]] = 90 + rng.normal(0, 3, length)
        data[start:end, _IDX["swap"]] = 45 + rng.normal(0, 6, length)
    elif kind == "fork_storm":
        data[start:end, _IDX["proc_count"]] = 350 + rng.normal(0, 20, length)
        data[start:end, _IDX["cpu"]] += rng.uniform(25, 40)
        data[start:end, _IDX["load_avg_1m"]] = 6 + rng.normal(0, 1, length)


# ---------------------------------------------------------------------------
# Feature subsetting (for ablations)
# ---------------------------------------------------------------------------
def subset_features(
    data: np.ndarray,
    feature_names: list[str],
    kept_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return a subset of columns in the order of ``kept_names``."""
    indices = [feature_names.index(n) for n in kept_names]
    return data[:, indices], list(kept_names)


# ---------------------------------------------------------------------------
# Scaler + windows
# ---------------------------------------------------------------------------
class MinMaxScaler:
    """Per-feature [0, 1] scaling. Kept for loading legacy bundles."""

    def __init__(self) -> None:
        """Initialise an unfitted scaler."""
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Learn per-feature min/max from ``data`` and return ``self``."""
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale ``data`` to [0, 1] per feature using the fitted min/max."""
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform()")
        denom = self.max_ - self.min_
        denom = np.where(denom == 0, 1.0, denom)
        return (data - self.min_) / denom

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the scaler on ``data`` and return the scaled array."""
        return self.fit(data).transform(data)


class RobustScaler:
    """ "Per-feature scaling by ``(X − median) / IQR`` (Q3 − Q1). Kept for loading legacy bundles."""

    def __init__(self) -> None:
        """Initialise an unfitted scaler."""
        self.center_: np.ndarray | None = None  # per-feature median
        self.scale_: np.ndarray | None = None  # per-feature IQR (Q3 − Q1)

    def fit(self, data: np.ndarray) -> "RobustScaler":
        """Compute per-feature median and IQR from ``data`` and return ``self``.

        Args:
            data: Reference array, shape ``(n_steps, n_features)``.  Must
                contain only normal (anomaly-free) samples.
        """
        self.center_ = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        iqr = q75 - q25
        # Guard against constant features (IQR == 0).
        self.scale_ = np.where(iqr == 0, 1.0, iqr)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply ``(X − median) / IQR`` per feature.

        Args:
            data: Array to scale, shape ``(n_steps, n_features)`` or
                ``(n_features,)``.

        Returns:
            The scaled array (same shape as input, unbounded).

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler must be fitted before transform()")
        return (data - self.center_) / self.scale_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the scaler on ``data`` and return the scaled array."""
        return self.fit(data).transform(data)


def load_csv_metrics(path: str) -> np.ndarray:
    """Load a metrics CSV produced by ``collect`` into a ``(n_steps, N_FEATURES)`` array.

    Args:
        path: Path to a CSV file that has all :data:`FEATURE_NAMES` columns
            (extra columns such as ``t`` are ignored).

    Returns:
        Float32 array of shape ``(n_steps, N_FEATURES)``.

    Raises:
        ValueError: If the file is empty, missing required columns, or has
            fewer than 100 rows.
    """
    import csv as _csv

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"пустий CSV: {path}")
    missing = [c for c in FEATURE_NAMES if c not in rows[0]]
    if missing:
        raise ValueError(
            f"CSV не містить колонок: {missing}\n"
            f"Наявні колонки: {list(rows[0].keys())}"
        )
    data = np.array(
        [[float(r[c]) for c in FEATURE_NAMES] for r in rows],
        dtype=np.float32,
    )
    if len(data) < 100:
        raise ValueError(
            f"CSV занадто короткий ({len(data)} рядків). "
            f"Мінімум 100, рекомендовано 600+: "
            f"python -m src collect --duration 600"
        )
    return data


def make_windows(
    data: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window inputs and next-step targets from a time series.

    Args:
        data: Time series, shape ``(n_steps, n_features)``.
        window_size: Number of timesteps per input window.

    Returns:
        A tuple ``(X, y)`` where ``X`` has shape
        ``(n_windows, window_size, n_features)`` and ``y`` has shape
        ``(n_windows, n_features)`` — the step following each window.
    """
    X_list, y_list = [], []
    for i in range(len(data) - window_size):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size])
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# DataModule facade
# ---------------------------------------------------------------------------
@dataclass
class TestContext:
    """Test-time (or calibration-time) data prepared by the DataModule."""

    raw_scaled: np.ndarray
    labels_full: np.ndarray | None
    X: np.ndarray
    y: np.ndarray
    labels_windowed: np.ndarray | None
    scenarios_full: np.ndarray | None = None
    scenarios_windowed: np.ndarray | None = None


class DataModule:
    """Owns the scaler and produces train/val/test sliding windows."""

    def __init__(self, window_size: int, val_split: float = 0.2) -> None:
        """Initialise the data module.

        Args:
            window_size: Sliding-window length for inputs.
            val_split: Fraction of normal data held out for validation.
        """
        self.window_size = window_size
        self.val_split = val_split
        self.scaler = MinMaxScaler()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None

    def fit_normal(self, normal_data: np.ndarray) -> "DataModule":
        """Fit the scaler on normal data and build train/val windows.

        Args:
            normal_data: Anomaly-free metrics, shape ``(n_steps, n_features)``.

        Returns:
            ``self``, with ``X_train``/``y_train``/``X_val``/``y_val`` populated.
        """
        n = len(normal_data)
        split = int(n * (1.0 - self.val_split))
        train_raw = normal_data[:split]
        val_raw = normal_data[split:]

        self.scaler.fit(train_raw)
        train_s = self.scaler.transform(train_raw)
        val_s = self.scaler.transform(val_raw)

        self.X_train, self.y_train = make_windows(train_s, self.window_size)
        self.X_val, self.y_val = make_windows(val_s, self.window_size)
        return self

    def prepare_test(
        self,
        test_data: np.ndarray,
        labels: np.ndarray | None = None,
        scenarios: np.ndarray | None = None,
    ) -> TestContext:
        """Scale and window test (or calibration) data into a TestContext.

        Args:
            test_data: Metrics array, shape ``(n_steps, n_features)``.
            labels: Optional per-step binary anomaly labels.
            scenarios: Optional per-step scenario codes.

        Returns:
            A :class:`TestContext` with scaled, windowed inputs and aligned
            (window-trimmed) labels and scenarios.

        Raises:
            RuntimeError: If the scaler has not been fitted via ``fit_normal``.
        """
        # Works for both RobustScaler (center_) and MinMaxScaler (min_).
        is_fitted = (
            getattr(self.scaler, "center_", None) is not None
            or getattr(self.scaler, "min_", None) is not None
        )
        if not is_fitted:
            raise RuntimeError("Call fit_normal() before prepare_test()")
        scaled = self.scaler.transform(test_data)
        X, y = make_windows(scaled, self.window_size)
        labels_w = None if labels is None else labels[self.window_size :]
        scenarios_w = None if scenarios is None else scenarios[self.window_size :]
        return TestContext(scaled, labels, X, y, labels_w, scenarios, scenarios_w)
