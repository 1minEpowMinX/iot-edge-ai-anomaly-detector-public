"""Three-phase evolutionary search runner.

Phase 1   GA on a short per-genome budget (dev test data)
Phase 1.5 Shortlist re-evaluation of top-K candidates (longer budget, same dev)
Phase 2   Holdout retraining of the winner (full budget, INDEPENDENT test data)

The dev/holdout split ensures the reported number is free of selection bias.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from ..config import AppConfig, resolve_device, set_seed
from ..data import (
    FEATURE_NAMES,
    generate_synthetic_metrics,
    inject_anomalies,
)
from ..reporter import Reporter
from .evolution import (
    FitnessFunction,
    GeneticAlgorithm,
    GenomeEvaluator,
    SearchSpace,
    pick_winner,
    retrain_genome_full,
    shortlist_top_k,
)


def _build_ga_config(output_dir: str) -> AppConfig:
    """Phase-1 budget: short per-genome training, only relative ranking matters."""
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.train.epochs = 600
    cfg.train.early_stopping = True
    cfg.train.early_stopping_patience = 15
    cfg.train.verbose = False
    cfg.data.n_train_steps = 2500
    cfg.data.n_test_steps = 1200
    cfg.data.n_calibration_steps = 600
    return cfg


def _build_shortlist_config(output_dir: str) -> AppConfig:
    """Phase-1.5 budget: double GA training, same dev-test data."""
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.train.epochs = 1200
    cfg.train.early_stopping = True
    cfg.train.early_stopping_patience = 20
    cfg.train.verbose = False
    cfg.data.n_train_steps = 2500
    cfg.data.n_test_steps = 1200
    cfg.data.n_calibration_steps = 600
    return cfg


def _build_production_config(output_dir: str) -> AppConfig:
    """Phase-2 budget: full training, holdout reporting."""
    cfg = AppConfig(artifacts_dir=output_dir)
    cfg.train.epochs = 800
    cfg.train.early_stopping = True
    cfg.train.early_stopping_patience = 20
    return cfg


def run_search(
    output_dir: str = "artifacts/evolution",
    population_size: int = 16,
    n_generations: int = 12,
    top_k_shortlist: int = 5,
    mutation_rate: float = 0.25,
    quick: bool = False,
) -> None:
    """Run the full three-phase search."""
    ga_cfg = _build_ga_config(output_dir)
    if quick:
        ga_cfg.train.epochs = 200
        n_generations = max(3, n_generations // 2)
        population_size = max(8, population_size // 2)

    device = resolve_device(ga_cfg.device)
    set_seed(ga_cfg.seed)
    os.makedirs(ga_cfg.artifacts_dir, exist_ok=True)

    _banner(ga_cfg, device, population_size, n_generations)

    # --- dev-test data (shared by Phase 1 + Phase 1.5) ---
    ga_train = generate_synthetic_metrics(
        ga_cfg.data.n_train_steps,
        seed=ga_cfg.data.train_seed,
    )
    ga_test, ga_test_lab, ga_test_scen = inject_anomalies(
        generate_synthetic_metrics(
            ga_cfg.data.n_test_steps,
            seed=ga_cfg.data.test_seed,
        ),
        seed=ga_cfg.data.anomaly_seed,
    )
    ga_calib, ga_calib_lab, ga_calib_scen = inject_anomalies(
        generate_synthetic_metrics(
            ga_cfg.data.n_calibration_steps,
            seed=ga_cfg.data.calibration_seed,
        ),
        seed=ga_cfg.data.calibration_anomaly_seed,
    )

    space = SearchSpace()
    fitness_fn = FitnessFunction(
        params_budget=100_000,
        infer_budget_ms=0.5,
        lambda_params=0.05,
        lambda_infer=0.05,
    )

    ga_evaluator = GenomeEvaluator(
        base_config=ga_cfg,
        device=device,
        train_data=ga_train,
        test_data=ga_test,
        test_labels=ga_test_lab,
        test_scenarios=ga_test_scen,
        calib_data=ga_calib,
        calib_labels=ga_calib_lab,
        calib_scenarios=ga_calib_scen,
        fitness_fn=fitness_fn,
        seed=ga_cfg.seed,
    )
    ga = GeneticAlgorithm(
        space=space,
        evaluator=ga_evaluator,
        population_size=population_size,
        n_generations=n_generations,
        elitism=2,
        tournament_size=3,
        mutation_rate=mutation_rate,
        crossover_rate=0.7,
        seed=ga_cfg.seed,
    )

    print("\n" + "=" * 78)
    print("ФАЗА 1 — ГЕНЕТИЧНИЙ АЛГОРИТМ (короткий бюджет, dev-test)")
    print("=" * 78)
    final = ga.run()
    ga_winner_g = final["best_genome"]
    ga_winner_r = final["best_result"]
    print(
        f"\n[GA] лідер фази 1: {ga_winner_g.short_repr()}  "
        f"fit={ga_winner_r.fitness:+.4f}  F1={ga_winner_r.f1:.4f}  "
        f"P={ga_winner_r.n_params:,}"
    )

    # --- Phase 1.5 — shortlist ---
    print("\n" + "=" * 78)
    print("ФАЗА 1.5 — SHORTLIST (розширений бюджет, та сама dev-test)")
    print("=" * 78)
    sl_cfg = _build_shortlist_config(output_dir)
    if quick:
        sl_cfg.train.epochs = 400
    sl_evaluator = GenomeEvaluator(
        base_config=sl_cfg,
        device=device,
        train_data=ga_train,
        test_data=ga_test,
        test_labels=ga_test_lab,
        test_scenarios=ga_test_scen,
        calib_data=ga_calib,
        calib_labels=ga_calib_lab,
        calib_scenarios=ga_calib_scen,
        fitness_fn=fitness_fn,
        seed=ga_cfg.seed,
    )
    refined = shortlist_top_k(
        candidates=final["all_evaluated"],
        evaluator=sl_evaluator,
        top_k=top_k_shortlist,
    )
    winner_genome, winner_result = pick_winner(refined)
    print(
        f"\n[shortlist] переможець: {winner_genome.short_repr()}  "
        f"F1={winner_result.f1:.4f}"
    )

    # --- Phase 2 — holdout retrain ---
    print("\n" + "=" * 78)
    print("ФАЗА 2 — HOLDOUT РЕТРЕНІНГ (чесна оцінка на невиданих даних)")
    print("=" * 78)
    prod_cfg = _build_production_config(output_dir)
    if quick:
        prod_cfg.train.epochs = 300

    prod_train = generate_synthetic_metrics(
        prod_cfg.data.n_train_steps,
        seed=prod_cfg.data.train_seed,
    )
    holdout_test, holdout_lab, holdout_scen = inject_anomalies(
        generate_synthetic_metrics(
            prod_cfg.data.n_holdout_steps,
            seed=prod_cfg.data.holdout_test_seed,
        ),
        seed=prod_cfg.data.holdout_anomaly_seed,
    )
    prod_calib, prod_calib_lab, prod_calib_scen = inject_anomalies(
        generate_synthetic_metrics(
            prod_cfg.data.n_calibration_steps,
            seed=prod_cfg.data.calibration_seed,
        ),
        seed=prod_cfg.data.calibration_anomaly_seed,
    )

    retrain = retrain_genome_full(
        genome=winner_genome,
        config=prod_cfg,
        device=device,
        train_data=prod_train,
        test_data=holdout_test,
        test_labels=holdout_lab,
        test_scenarios=holdout_scen,
        calib_data=prod_calib,
        calib_labels=prod_calib_lab,
        calib_scenarios=prod_calib_scen,
    )
    result = retrain.result

    print("\n" + "=" * 78)
    print("HOLDOUT МЕТРИКИ (НЕБАЧЕНІ ДАНІ)")
    print("=" * 78)
    print(f"  F1         = {result.f1:.4f}   (shortlist dev: {winner_result.f1:.4f})")
    print(f"  точність   = {result.precision:.4f}")
    print(f"  повнота    = {result.recall:.4f}")
    print(f"  параметри  = {result.n_params:,}")
    print(f"  інференс   = {result.infer_ms:.4f} ms")

    reporter = Reporter(
        output_dir=prod_cfg.artifacts_dir,
        feature_names=FEATURE_NAMES,
        window_size=winner_genome.window_size,
        style=prod_cfg.plot,
    )
    reporter.save_evolution_history(final["history"])
    reporter.save_production_report(
        result=result,
        test_ctx=retrain.test_ctx,
        datamodule=retrain.datamodule,
        config={
            **prod_cfg.to_dict(),
            "device": device,
            "winner_genome": (
                asdict(winner_genome)
                if hasattr(winner_genome, "__dataclass_fields__")
                else _genome_dict(winner_genome)
            ),
        },
    )
    _save_best_genome(
        prod_cfg.artifacts_dir,
        winner_genome,
        ga_winner_r,
        winner_result,
        result,
    )
    print(f"\n[search] збережено у {prod_cfg.artifacts_dir}/")


def _banner(cfg: AppConfig, device: str, pop: int, gens: int) -> None:
    """Print the search banner and the data random-seed protocol.

    Args:
        cfg: The Phase-1 GA configuration.
        device: Torch device string.
        pop: Population size.
        gens: Number of generations.
    """
    print(f"[search] пристрій: {device}")
    print(f"[search] artifacts -> {cfg.artifacts_dir}/")
    print(f"[search] population={pop}  generations={gens}")
    print()
    print("seed'и випадкових даних:")
    print(f"  dev test     seed={cfg.data.test_seed}")
    print(f"  holdout test seed={cfg.data.holdout_test_seed}")
    print(f"  calibration  seed={cfg.data.calibration_seed}")


def _genome_dict(genome) -> dict:
    """Return a plain-dict view of a genome's five genes.

    Args:
        genome: The genome to serialise.

    Returns:
        A dict with the five gene values.
    """
    return {
        "window_size": genome.window_size,
        "hidden_size": genome.hidden_size,
        "num_layers": genome.num_layers,
        "dropout": genome.dropout,
        "lr": genome.lr,
    }


def _save_best_genome(
    save_dir: str,
    genome,
    ga_result,
    shortlist_result,
    holdout_result,
) -> None:
    """Write ``best_genome.json`` with metrics from all three search phases.

    Args:
        save_dir: Directory to write into.
        genome: The winning genome.
        ga_result: Phase-1 (GA budget) fitness result.
        shortlist_result: Phase-1.5 (shortlist budget) fitness result.
        holdout_result: Phase-2 holdout :class:`RunResult`.
    """
    payload = {
        "genome": _genome_dict(genome),
        "ga_budget_metrics": {
            "fitness": ga_result.fitness,
            "f1": ga_result.f1,
            "precision": ga_result.precision,
            "recall": ga_result.recall,
            "n_params": ga_result.n_params,
            "infer_ms": ga_result.infer_ms,
        },
        "shortlist_metrics": {
            "fitness": shortlist_result.fitness,
            "f1": shortlist_result.f1,
            "precision": shortlist_result.precision,
            "recall": shortlist_result.recall,
        },
        "holdout_metrics": {
            "f1": holdout_result.f1,
            "precision": holdout_result.precision,
            "recall": holdout_result.recall,
            "n_params": holdout_result.n_params,
            "infer_ms": holdout_result.infer_ms,
        },
    }
    with open(os.path.join(save_dir, "best_genome.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
