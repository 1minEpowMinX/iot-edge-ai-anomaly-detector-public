**English** · [Українська](docs/README.uk.md)

# IoT Edge AI Anomaly Detector

> Lightweight GRU-based anomaly detector for IoT host metrics — **1,516 parameters**, **0.31 ms inference**, **F1 = 0.945** on a held-out test set. Designed for edge deployment without cloud dependency.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

The architecture was discovered by a three-phase evolutionary search with a strict no-leak evaluation protocol — dev-test and holdout-test sets stay independent throughout, so the reported number reflects real generalization rather than selection bias.

## Why

- **Tiny.** 1,516 trainable parameters; the model bundle is <30 KB.
- **Fast.** 0.31 ms per window on CPU; no GPU needed.
- **Accurate.** F1 = 0.945 / Precision = 0.918 / Recall = 0.973 / ROC-AUC = 0.995.
- **Honest.** Holdout is sealed off from the search loop — bias measured at 0.016 F1.
- **Self-contained.** Synthetic data generator, evolutionary search, calibration, dashboards — all in one package.
- **Production-style CLI.** 9 subcommands, `rich`-formatted output, full model bundle save/load.

## Installation

```bash
git clone https://github.com/1minEpowMinX/iot-edge-ai-anomaly-detector-public
cd iot-edge-ai-anomaly-detector-public
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, scikit-learn, NumPy, pandas, matplotlib, psutil, rich.

## Quick start

```bash
python -m src demo            # 30s showcase — trains the evolved winner, F1 = 0.945
python -m src demo --quick    # ~5s reduced-budget run
python -m src --help          # all commands
```

All artifacts (dashboards, model bundle, metadata) are written to `artifacts/`.

## CLI

| Command   | Purpose |
|-----------|---------|
| `demo`    | Train the evolved-search winner on holdout data. Main showcase. |
| `train`   | Train with custom hyperparameters via CLI flags. |
| `infer`   | Score a CSV with a saved model bundle. |
| `live`    | Real-time host monitoring via psutil — inference-pipeline demo. |
| `collect` | Stream live host metrics to a CSV. |
| `search`  | Three-phase evolutionary search (GA → shortlist → holdout retrain). |
| `compare` | Benchmark GRU vs LSTM vs MovingAverage on identical data. |
| `ablate`  | Compare 5-metric subset (minimal spec) vs full 12-metric set. |
| `sweep`   | Sweep `window_size` and/or `hidden_size`. |

Global flags: `--version`, `-v/--verbose`, `-q/--quiet`.

## Examples

```bash
python -m src demo
python -m src train --epochs 100 --lr 1e-3 --hidden 16 --window 40
python -m src compare                            # GRU vs LSTM vs MA
python -m src ablate                             # 5 vs 12 metrics
python -m src sweep --axis window_size           # vary input window length
python -m src search --quick                     # fast evolutionary search
python -m src collect --duration 60 -o my.csv    # collect real host metrics
python -m src infer --model artifacts/ --data my.csv
```

## How it works

```mermaid
flowchart TB
    M["psutil metrics<br>(12 channels)"] --> S["MinMax scaler"]
    S --> W["Sliding window<br>(W = 40)"]
    W --> P{"EMA<br>prefilter"}
    P -- confidently NORMAL<br>(~25-30 % of windows) --> FAST["Fast path"]
    P -- uncertain /<br>suspicious --> G["GRU forward<br>(1,516 params)"]
    G --> R["forecast − actual"]
    R --> SC["anomaly score<br>s_t = MAE per window"]
    SC --> T{"s_t &gt; τ ?"}
    T -- no --> N(["NORMAL"])
    T -- yes --> A(["ANOMALY"])
    FAST --> N
     P:::decision
     FAST:::fast
     G:::gru
     R:::gru
     SC:::gru
     T:::decision
     A:::anomalyEnd
     N:::normalEnd
    classDef fast fill:#dff5e1,stroke:#3c9,color:#000
    classDef gru fill:#fff0d4,stroke:#c83,color:#000
    classDef decision fill:#e6e9ff,stroke:#55a,color:#000
    classDef anomalyEnd fill:#ffd9d9,stroke:#a44,color:#000,font-weight:bold
    classDef normalEnd fill:#dff5e1,stroke:#3c9,color:#000
```

The GRU is trained to forecast the next time step. Anomalies appear as large residuals between forecast and observation (MAE-based score). A lightweight EMA prefilter routes obvious-normal windows past the GRU to save compute — only `UNCERTAIN`/`ANOMALY` candidates ever reach the model. The decision threshold τ is auto-calibrated on a separate calibration set to maximize F1 (Auto-F1).

## Deploying to a real IoT device

The shipped model is trained on synthetic data, so it **will** produce false positives on real hardware due to distribution shift. For production use:

1. Collect target-device metrics: `python -m src collect --duration 3600 -o real.csv`
2. Retrain on this data: `python -m src train --epochs 200`
3. Ship the resulting `artifacts/model.pt` + `scaler_*.npy` + `meta.json` bundle.

The bundle is portable — load it with `src.artifacts.load_bundle()` on the target device.

## Results

### Holdout metrics (seed = 999, never seen during search)

| Metric             | Value          |
|--------------------|---------------:|
| **F1**             | **0.9449**     |
| Precision          | 0.9184         |
| Recall             | 0.9730         |
| ROC-AUC            | 0.995          |
| Parameters         | 1,516          |
| Inference latency  | 0.31 ms/window |
| Bundle size        | ~28 KB         |

**Winner genome:** `window_size=40, hidden_size=8, num_layers=3, dropout=0.4, lr=3e-3`.

### Model comparison (`python -m src compare`)

| Model          | Precision | Recall  | F1        | Params | Inference |
|----------------|----------:|--------:|----------:|-------:|----------:|
| **GRU**        | 0.918     | 0.973   | **0.945** | 1,516  | 0.31 ms   |
| LSTM           | 0.741     | 1.000   | 0.851     | 6,348  | 0.29 ms   |
| Moving Average | 0.442     | 0.984   | 0.610     | 0      | 0.004 ms  |

### Selection-bias check

- Dev-test F1 (during search): **0.961**
- Holdout F1 (sealed off): **0.945**
- **Gap: 0.016 F1** — within expected range for a properly-protected no-leak protocol.

## Design rationale

| Decision | Rationale |
|---|---|
| **GRU** over LSTM | Fewer parameters at equal F1; verified empirically by `compare`. |
| **Linear readout** | Without activation, a 2-layer MLP head collapses to a single Linear; with activation, no F1 improvement observed. |
| **Huber loss** (training) | Robust to occasional outliers in nominally-normal training data. |
| **MAE** (anomaly score) | Linear response amplifies the contrast between normal noise and anomalous spikes — opposite of what Huber does. |
| **AdamW + ReduceLROnPlateau** | Decoupled weight decay + adaptive learning rate. |
| **Early stopping with patience-reset on lr-drop** | Avoids killing models that can still improve at a lower lr. |
| **Asymmetric EMA prefilter** | Fast-paths only confidently-normal windows; suspicious ones always reach the GRU. Preserves recall while saving ~35 % of forward passes. |
| **Auto-F1 threshold calibration** | Threshold chosen on a labelled calibration set, not arbitrarily at the 95th percentile. |
| **No-leak protocol** | Holdout test (seed=999) is never used during search. Dev-test (seed=123) is for fitness only. |
| **12 metrics** instead of 5 | Disk/swap/process channels are essential for I/O storms, memory leaks, fork bombs; ablation confirms +0.20 F1. |
| **Evolutionary search** | Finds smaller, better architectures than hand-tuning (1.5K vs 12K params, +0.04 F1). |

## Project layout

```
src/                       Production package
├── __init__.py            Public API + __version__
├── __main__.py            CLI entrypoint (python -m src)
├── cli.py                 argparse + 9 subcommands
├── _ui.py                 rich-based UI with plain-text fallback
├── config.py              AppConfig + all sub-configs
├── data.py                Synthetic generator + DataModule + scaler + windows
├── model.py               GRUNet / LSTMNet
├── predictors.py          BasePredictor + Torch/MovingAverage predictors
├── losses.py              Huber / MSE / MAE factory
├── prefilter.py           EMA / MA cascade prefilter
├── detector.py            AnomalyDetector + prf1 + roc_pr_curves (scikit-learn)
├── pipeline.py            Pipeline orchestration + RunResult
├── reporter.py            Reporter (console + dashboards)
├── visualize.py           matplotlib plotters
├── artifacts.py           Model bundle save/load
└── experiments/           Lab + production runners
    ├── evolution.py       Genome / SearchSpace / GA / shortlist / retrain
    ├── search.py          Three-phase orchestration
    ├── demo.py            Showcase runner
    ├── train.py           Configurable training runner
    ├── infer.py           CSV inference
    ├── live.py            Real-time psutil monitor
    ├── collect.py         psutil sampler
    ├── comparison.py      Model comparison
    ├── ablation.py        Feature-set ablation
    └── sweep.py           Hyperparameter sweeps
```

## Background

Built originally as an engineering thesis on edge AI / time-series anomaly detection. The methodological focus is on:

- Treating anomaly detection as **next-step forecasting with thresholded residuals** rather than reconstruction or classification.
- An end-to-end **no-leak research protocol** that separates hyperparameter selection from final evaluation.
- Demonstrating that **evolutionary search can find compact, edge-deployable architectures** that outperform hand-tuned baselines.

If any of these are useful in your own work — feel free to fork, cite, or open an issue.

## License

[MIT](LICENSE) © 2026 1minEpowMinX.
