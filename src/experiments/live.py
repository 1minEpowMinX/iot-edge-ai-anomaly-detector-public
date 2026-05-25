"""Real-time monitor — psutil + saved model, scoring each new sample.

This module demonstrates the production INFERENCE PIPELINE: streaming metric
collection → rolling window → model forward pass → threshold decision.

CAVEAT: the shipped model was trained on synthetic data. Running ``live`` on a
machine whose metric distribution differs substantially from that synthetic
profile (e.g., a desktop OS) will produce many false positives — this is
distribution shift, not a bug. For real IoT deployment retrain the model on
target-device data first; see the diploma's "future work" section.

Usage:
    python -m src live --duration 60
"""

from __future__ import annotations

import collections
import time
from typing import Any

import numpy as np
import torch

from .. import _ui
from ..artifacts import load_bundle


def _sample_metrics(
    psutil_mod: Any,
    prev_net: Any,
    prev_disk: Any,
    interval: float,
) -> tuple[dict, Any, Any]:
    """Collect one sample for all 12 model features."""
    cpu = psutil_mod.cpu_percent(interval=None)
    times = psutil_mod.cpu_times_percent(interval=None)
    cpu_iowait = float(getattr(times, "iowait", 0.0) or 0.0)
    try:
        load_avg_1m = float(psutil_mod.getloadavg()[0])
    except (AttributeError, OSError, NotImplementedError):
        load_avg_1m = 0.0

    ram = psutil_mod.virtual_memory().percent
    try:
        swap = psutil_mod.swap_memory().percent
    except Exception:
        swap = 0.0

    cur_disk = psutil_mod.disk_io_counters()
    if cur_disk is not None and prev_disk is not None:
        disk_read_bps = (cur_disk.read_bytes - prev_disk.read_bytes) / interval
        disk_write_bps = (cur_disk.write_bytes - prev_disk.write_bytes) / interval
    else:
        disk_read_bps = 0.0
        disk_write_bps = 0.0

    cur_net = psutil_mod.net_io_counters()
    net_tx = (cur_net.bytes_sent - prev_net.bytes_sent) / interval
    net_rx = (cur_net.bytes_recv - prev_net.bytes_recv) / interval
    net_packets_tx = (cur_net.packets_sent - prev_net.packets_sent) / interval

    try:
        tcp_conn = sum(
            1
            for c in psutil_mod.net_connections(kind="tcp")
            if c.status == psutil_mod.CONN_ESTABLISHED
        )
    except (psutil_mod.AccessDenied, PermissionError):
        try:
            tcp_conn = len(psutil_mod.Process().net_connections())
        except Exception:
            tcp_conn = 0

    proc_count = len(psutil_mod.pids())

    return (
        {
            "cpu": cpu,
            "cpu_iowait": cpu_iowait,
            "load_avg_1m": load_avg_1m,
            "ram": ram,
            "swap": swap,
            "disk_read_bps": disk_read_bps,
            "disk_write_bps": disk_write_bps,
            "net_tx": net_tx,
            "net_rx": net_rx,
            "net_packets_tx": net_packets_tx,
            "tcp_conn": tcp_conn,
            "proc_count": proc_count,
        },
        cur_net,
        cur_disk,
    )


def _fmt_feature(name: str, value: float) -> str:
    """Pretty-print a metric value with sensible units."""
    if name in ("cpu", "cpu_iowait", "ram", "swap"):
        return f"{value:.1f} %"
    if name == "load_avg_1m":
        return f"{value:.2f}"
    if name in ("disk_read_bps", "disk_write_bps", "net_tx", "net_rx"):
        return f"{value:.0f} B/s" if value < 1e4 else f"{value / 1024:.1f} KB/s"
    if name == "net_packets_tx":
        return f"{value:.1f} pkt/s"
    if name in ("tcp_conn", "proc_count"):
        return f"{int(value)}"
    return f"{value:.3f}"


def run_live(
    model_dir: str = "artifacts",
    duration: float = 60.0,
    interval: float = 1.0,
    device: str = "cpu",
    color: bool = True,
) -> None:
    """Monitor the host in real time, scoring each sample with a saved model.

    Args:
        model_dir: Directory holding the saved model bundle.
        duration: Total monitoring time in seconds.
        interval: Seconds between samples.
        device: Torch device string.
        color: Whether to colour the verdict output.
    """
    try:
        import psutil
    except ImportError:
        raise SystemExit("Спочатку встановіть psutil:  pip install psutil")

    _ui.banner(
        "МОНІТОРИНГ У РЕАЛЬНОМУ ЧАСІ",
        f"тривалість={duration:g}s  інтервал={interval:g}s  модель={model_dir}",
    )

    with _ui.spinner("завантаження збереженої моделі"):
        bundle = load_bundle(model_dir, device=device)

    warmup_samples = bundle.window_size + 1
    warmup_seconds = warmup_samples * interval
    expected_scores = max(0, int((duration - warmup_seconds) / interval))

    _ui.kv_table(
        "модель",
        [
            ("window_size", bundle.window_size),
            ("n_features", bundle.n_features),
            ("поріг", f"{bundle.threshold:.5f}"),
            ("device", device),
            ("семплів прогріву", f"{warmup_samples} (~{warmup_seconds:.0f}s)"),
            ("очікувано оцінок", expected_scores),
        ],
    )

    if expected_scores < 1:
        _ui.warn(
            f"тривалість ({duration:g}s) менша за прогрів ({warmup_seconds:.0f}s) — "
            "оцінок не буде. Спробуйте:"
        )
        _ui.info(f"  --duration {warmup_seconds + 30:.0f}")
        _ui.info(f"  --interval 0.5 --duration {warmup_samples * 0.5 + 15:.0f}")

    psutil.cpu_percent(interval=None)
    prev_net = psutil.net_io_counters()
    prev_disk = psutil.disk_io_counters()

    # One-shot diagnostic — show all 12 features so the user sees the full
    # input vector that flows into the model.
    time.sleep(interval)
    diag_sample, prev_net, prev_disk = _sample_metrics(
        psutil,
        prev_net,
        prev_disk,
        interval,
    )
    rows = [
        (name, _fmt_feature(name, diag_sample[name])) for name in bundle.feature_names
    ]
    _ui.kv_table("перший семпл — усі 12 ознак на вході моделі", rows)
    _ui.info(
        "примітка: на Windows `cpu_iowait` та (старий psutil) `load_avg_1m` "
        "повертають 0.0 — psutil не може їх отримати."
    )
    _ui.info(
        "примітка: модель навчена на синтетичних даних. Реальні машини мають "
        "іншу дистрибуцію -> очікуйте хибних спрацювань на idle-десктопах. "
        "Для реального IoT-деплою перенавчіть модель на цільових даних."
    )

    buffer: collections.deque = collections.deque(maxlen=bundle.window_size + 1)
    bundle.net.eval()

    columns = [
        "t,s",
        "cpu%",
        "ram%",
        "iowait%",
        "net_tx",
        "tcp",
        "proc",
        "score",
        "вердикт",
    ]
    scroller = _ui.LiveScroller(columns=columns, max_rows=20)

    n_anomalies = 0
    n_normal = 0
    start = time.time()
    with scroller:
        try:
            while time.time() - start < duration:
                time.sleep(interval)
                sample, prev_net, prev_disk = _sample_metrics(
                    psutil,
                    prev_net,
                    prev_disk,
                    interval,
                )
                vec = np.array(
                    [sample[name] for name in bundle.feature_names],
                    dtype=np.float32,
                )
                buffer.append(vec)
                t_s = time.time() - start

                if len(buffer) < bundle.window_size + 1:
                    scroller.add_row(
                        [
                            f"{t_s:5.1f}",
                            f"{sample['cpu']:.1f}",
                            f"{sample['ram']:.1f}",
                            f"{sample['cpu_iowait']:.1f}",
                            f"{int(sample['net_tx'])}",
                            f"{int(sample['tcp_conn'])}",
                            f"{int(sample['proc_count'])}",
                            "—",
                            f"warmup {len(buffer)}/{bundle.window_size + 1}",
                        ],
                        style="dim",
                    )
                    continue

                window = np.stack(list(buffer))
                x_raw = window[:-1]
                y_raw = window[-1]
                x_scaled = bundle.scaler.transform(x_raw)
                y_scaled = bundle.scaler.transform(y_raw.reshape(1, -1))[0]
                x_t = torch.from_numpy(x_scaled).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = bundle.net(x_t).cpu().numpy()[0]
                score = float(np.abs(pred - y_scaled).mean())
                is_anomaly = score > bundle.threshold
                n_anomalies += int(is_anomaly)
                n_normal += int(not is_anomaly)

                if color:
                    verdict = "[!] АНОМАЛІЯ" if is_anomaly else "OK нормально"
                    style = "bold red" if is_anomaly else "green"
                else:
                    verdict = "[!] АНОМАЛІЯ" if is_anomaly else "норм"
                    style = None

                scroller.add_row(
                    [
                        f"{t_s:5.1f}",
                        f"{sample['cpu']:.1f}",
                        f"{sample['ram']:.1f}",
                        f"{sample['cpu_iowait']:.1f}",
                        f"{int(sample['net_tx'])}",
                        f"{int(sample['tcp_conn'])}",
                        f"{int(sample['proc_count'])}",
                        f"{score:.4f}",
                        verdict,
                    ],
                    style=style,
                )
        except KeyboardInterrupt:
            _ui.warn("зупинено користувачем")

    total = n_anomalies + n_normal
    _ui.rule()
    _ui.kv_table(
        "підсумок",
        [
            ("оцінено семплів", total),
            ("аномалій", n_anomalies),
            ("частка аномалій", f"{n_anomalies / max(total, 1) * 100:.1f}%"),
        ],
    )
