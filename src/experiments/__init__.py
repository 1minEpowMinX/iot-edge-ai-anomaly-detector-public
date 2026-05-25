"""Laboratory + production runners.

Importable runners (each exposes ``run_*()``):
  - ``train.run_train``        — production training with CLI-configurable HP
  - ``demo.run_demo``          — fixed-config showcase (alias of train with defaults)
  - ``infer.run_infer``        — load saved model, score a CSV
  - ``live.run_live``          — psutil + model, real-time anomaly detection
  - ``collect.run_collect``    — collect real metrics into a CSV
  - ``search.run_search``      — three-phase GA + shortlist + holdout
  - ``comparison.run_comparison`` — GRU vs LSTM vs MovingAverage
  - ``ablation.run_ablation``  — 5 vs 12 feature ablation
  - ``sweep.run_sweep``        — hyperparameter sweeps

Core search primitives (Genome, GeneticAlgorithm, ...) live in ``evolution``.
"""

from .evolution import (
    FitnessFunction,
    FitnessResult,
    GenerationStats,
    GeneticAlgorithm,
    Genome,
    GenomeEvaluator,
    RetrainArtifacts,
    SearchSpace,
    pick_winner,
    retrain_genome_full,
    shortlist_top_k,
)

__all__ = [
    "FitnessFunction",
    "FitnessResult",
    "GenerationStats",
    "GeneticAlgorithm",
    "Genome",
    "GenomeEvaluator",
    "RetrainArtifacts",
    "SearchSpace",
    "pick_winner",
    "retrain_genome_full",
    "shortlist_top_k",
]
