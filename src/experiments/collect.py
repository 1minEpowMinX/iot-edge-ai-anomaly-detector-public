"""Real metrics collector — psutil → CSV compatible with the trained model."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

# Column order must match ``data.FEATURE_NAMES``.
FIELDS = [
    "t",
    "cpu",
    "cpu_iowait",
    "load_avg_1m",
    "ram",
    "swap",
    "disk_read_bps",
    "disk_write_bps",
    "net_tx",
    "net_rx",
    "net_packets_tx",
    "tcp_conn",
    "proc_count",
]


def _safe_loadavg(psutil_mod: Any) -> float:
    """Return the 1-minute load average, or 0.0 where unsupported.

    Args:
        psutil_mod: The imported ``psutil`` module.

    Returns:
        The 1-minute load average (0.0 on platforms without it).
    """
    try:
        return float(psutil_mod.getloadavg()[0])
    except (AttributeError, OSError, NotImplementedError):
        return 0.0


def _safe_tcp_conn(psutil_mod: Any) -> int:
    """Count established TCP connections, falling back when access is denied.

    Args:
        psutil_mod: The imported ``psutil`` module.

    Returns:
        Number of established TCP connections (0 if it cannot be determined).
    """
    try:
        return sum(
            1
            for c in psutil_mod.net_connections(kind="tcp")
            if c.status == psutil_mod.CONN_ESTABLISHED
        )
    except (psutil_mod.AccessDenied, PermissionError):
        try:
            return len(psutil_mod.Process().net_connections())
        except Exception:
            return 0


def run_collect(
    duration: float = 60.0,
    interval: float = 1.0,
    output: str = "real_metrics.csv",
) -> None:
    """Stream live host metrics to a CSV compatible with the trained model.

    Args:
        duration: Total collection time in seconds.
        interval: Seconds between samples.
        output: Output CSV path.
    """
    try:
        import psutil
    except ImportError:
        raise SystemExit("Спочатку встановіть psutil:  pip install psutil")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[collect] запис у {out_path}, тривалість {duration} с, інтервал {interval} с"
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)

        prev_net = psutil.net_io_counters()
        prev_disk = psutil.disk_io_counters()
        psutil.cpu_percent(interval=None)  # warmup

        start = time.time()
        n_samples = 0
        while time.time() - start < duration:
            time.sleep(interval)

            cpu = psutil.cpu_percent(interval=None)
            times = psutil.cpu_times_percent(interval=None)
            cpu_iowait = float(getattr(times, "iowait", 0.0) or 0.0)
            load_avg_1m = _safe_loadavg(psutil)

            ram = psutil.virtual_memory().percent
            try:
                swap = psutil.swap_memory().percent
            except Exception:
                swap = 0.0

            cur_disk = psutil.disk_io_counters()
            if cur_disk is not None and prev_disk is not None:
                disk_read_bps = (cur_disk.read_bytes - prev_disk.read_bytes) / interval
                disk_write_bps = (
                    cur_disk.write_bytes - prev_disk.write_bytes
                ) / interval
            else:
                disk_read_bps = 0.0
                disk_write_bps = 0.0
            prev_disk = cur_disk

            cur_net = psutil.net_io_counters()
            net_tx = (cur_net.bytes_sent - prev_net.bytes_sent) / interval
            net_rx = (cur_net.bytes_recv - prev_net.bytes_recv) / interval
            net_packets_tx = (cur_net.packets_sent - prev_net.packets_sent) / interval
            prev_net = cur_net

            tcp_conn = _safe_tcp_conn(psutil)
            proc_count = len(psutil.pids())

            t_s = time.time() - start
            w.writerow(
                [
                    f"{t_s:.3f}",
                    f"{cpu:.2f}",
                    f"{cpu_iowait:.2f}",
                    f"{load_avg_1m:.3f}",
                    f"{ram:.2f}",
                    f"{swap:.2f}",
                    f"{disk_read_bps:.2f}",
                    f"{disk_write_bps:.2f}",
                    f"{net_tx:.2f}",
                    f"{net_rx:.2f}",
                    f"{net_packets_tx:.2f}",
                    tcp_conn,
                    proc_count,
                ]
            )
            n_samples += 1
            if n_samples % 10 == 0:
                print(
                    f"[collect] t={t_s:6.1f}s  cpu={cpu:5.1f}%  ram={ram:5.1f}%  "
                    f"disk_r={disk_read_bps:8.0f}  net_tx={net_tx:8.0f}  "
                    f"proc={proc_count}  tcp={tcp_conn}"
                )

    print(f"[collect] готово, зібрано {n_samples} вимірювань -> {out_path}")
