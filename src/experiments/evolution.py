"""Evolutionary architecture/hyperparameter search.

Encoding (one genome per individual):
  - window_size  (input sequence length)
  - hidden_size  (GRU hidden state size)
  - num_layers   (stacked GRU layers)
  - dropout      (regularisation)
  - lr           (initial learning rate)

Head activation is fixed (linear projection), see ``Genome`` docstring.

Fitness combines F1 detection quality with soft penalties on parameter count
and inference latency, reflecting edge-deployment constraints.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig, set_seed
from ..data import DataModule, N_FEATURES, TestContext
from ..model import GRUNet
from ..pipeline import Pipeline, RunResult
from ..predictors import TorchPredictor
from ..prefilter import build_prefilter


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Genome:
    """One candidate architecture + the hyperparameters required by the thesis.

    Head activation is deliberately not searched. For our regression task the
    GRU already supplies non-linearity through its gates, and an MLP head
    showed no F1 advantage over a single linear projection.
    """

    window_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float

    def as_tuple(self) -> tuple:
        """Return the five genes as a tuple — used as a cache key."""
        return (
            self.window_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.lr,
        )

    def short_repr(self) -> str:
        """Return a compact single-line representation of the genome."""
        return (
            f"ws={self.window_size:>2} h={self.hidden_size:>3} "
            f"L={self.num_layers} drop={self.dropout:.1f} lr={self.lr:.0e}"
        )


# ---------------------------------------------------------------------------
# Search space + operators
# ---------------------------------------------------------------------------
@dataclass
class SearchSpace:
    """Discrete choices for each gene — enables clean crossover and mutation."""

    window_sizes: tuple[int, ...] = (20, 30, 40, 60, 80, 100)
    hidden_sizes: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 96, 128)
    num_layers: tuple[int, ...] = (1, 2, 3)
    dropouts: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    learning_rates: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """Sample a random genome uniformly from the search space.

        Args:
            rng: Random generator.

        Returns:
            A randomly drawn :class:`Genome`.
        """
        return Genome(
            window_size=int(rng.choice(self.window_sizes)),
            hidden_size=int(rng.choice(self.hidden_sizes)),
            num_layers=int(rng.choice(self.num_layers)),
            dropout=float(rng.choice(self.dropouts)),
            lr=float(rng.choice(self.learning_rates)),
        )

    def mutate(
        self,
        genome: Genome,
        rng: np.random.Generator,
        rate: float = 0.15,
    ) -> Genome:
        """Per-gene mutation: each gene resampled independently with prob `rate`."""

        def pick(values, current):
            """Resample one gene from ``values`` with probability ``rate``."""
            return type(current)(rng.choice(values)) if rng.random() < rate else current

        return Genome(
            window_size=pick(self.window_sizes, genome.window_size),
            hidden_size=pick(self.hidden_sizes, genome.hidden_size),
            num_layers=pick(self.num_layers, genome.num_layers),
            dropout=pick(self.dropouts, genome.dropout),
            lr=pick(self.learning_rates, genome.lr),
        )

    def crossover(
        self,
        p1: Genome,
        p2: Genome,
        rng: np.random.Generator,
    ) -> Genome:
        """Uniform crossover: each gene drawn from either parent with 50% prob."""

        def pick(a, b):
            """Return ``a`` or ``b`` with equal probability."""
            return a if rng.random() < 0.5 else b

        return Genome(
            window_size=pick(p1.window_size, p2.window_size),
            hidden_size=pick(p1.hidden_size, p2.hidden_size),
            num_layers=pick(p1.num_layers, p2.num_layers),
            dropout=pick(p1.dropout, p2.dropout),
            lr=pick(p1.lr, p2.lr),
        )


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
@dataclass
class FitnessResult:
    """Metrics and cost statistics from evaluating one genome."""

    fitness: float
    f1: float
    precision: float
    recall: float
    n_params: int
    infer_ms: float
    train_time_s: float
    epochs_trained: int


@dataclass
class FitnessFunction:
    """Multi-objective scalarisation with soft penalties for edge constraints."""

    params_budget: int = 100_000
    infer_budget_ms: float = 0.5
    lambda_params: float = 0.05
    lambda_infer: float = 0.05

    def __call__(self, f1: float, n_params: int, infer_ms: float) -> float:
        """Scalarise F1 with soft penalties for edge-constraint violations.

        Args:
            f1: Detection F1-score of the candidate.
            n_params: Trainable parameter count.
            infer_ms: Single-window inference latency in milliseconds.

        Returns:
            The penalised fitness score.
        """
        params_excess = max(0.0, (n_params - self.params_budget) / self.params_budget)
        infer_excess = max(
            0.0, (infer_ms - self.infer_budget_ms) / self.infer_budget_ms
        )
        return (
            f1 - self.lambda_params * params_excess - self.lambda_infer * infer_excess
        )


# ---------------------------------------------------------------------------
# Evaluator (with per-tuple cache)
# ---------------------------------------------------------------------------
class GenomeEvaluator:
    """Trains and evaluates one genome. Cache hits skip re-training."""

    def __init__(
        self,
        base_config: AppConfig,
        device: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        test_scenarios: np.ndarray,
        calib_data: np.ndarray,
        calib_labels: np.ndarray,
        calib_scenarios: np.ndarray,
        fitness_fn: FitnessFunction,
        seed: int = 42,
    ) -> None:
        """Initialise the evaluator.

        Args:
            base_config: Base configuration cloned per genome.
            device: Torch device string.
            train_data: Normal training metrics.
            test_data: Test metrics for fitness scoring.
            test_labels: Per-step test labels.
            test_scenarios: Per-step test scenario codes.
            calib_data: Calibration metrics.
            calib_labels: Per-step calibration labels.
            calib_scenarios: Per-step calibration scenario codes.
            fitness_fn: Function scalarising metrics into a fitness value.
            seed: Random seed for reproducible training.
        """
        self.base = base_config
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_scenarios = test_scenarios
        self.calib_data = calib_data
        self.calib_labels = calib_labels
        self.calib_scenarios = calib_scenarios
        self.fitness_fn = fitness_fn
        self.seed = seed
        self._cache: dict[tuple, FitnessResult] = {}

    @property
    def cache_size(self) -> int:
        """Number of distinct genomes currently cached."""
        return len(self._cache)

    def evaluate(self, genome: Genome) -> tuple[FitnessResult, bool]:
        """Return (result, cache_hit)."""
        key = genome.as_tuple()
        if key in self._cache:
            return self._cache[key], True
        result = self._train_and_eval(genome)
        self._cache[key] = result
        return result, False

    def _train_and_eval(self, genome: Genome) -> FitnessResult:
        """Train a genome from scratch and return its fitness result.

        Args:
            genome: The genome to train and evaluate.

        Returns:
            The :class:`FitnessResult` for the genome.
        """
        cfg = self._genome_to_config(genome)
        set_seed(self.seed)

        datamodule = DataModule(
            window_size=genome.window_size,
            val_split=cfg.data.val_split,
        ).fit_normal(self.train_data)
        test_ctx = datamodule.prepare_test(
            self.test_data,
            self.test_labels,
            self.test_scenarios,
        )
        calib_ctx = datamodule.prepare_test(
            self.calib_data,
            self.calib_labels,
            self.calib_scenarios,
        )

        net = GRUNet(
            n_features=N_FEATURES,
            hidden_size=genome.hidden_size,
            num_layers=genome.num_layers,
            dropout=genome.dropout,
        )
        predictor = TorchPredictor(
            net=net,
            name="g",
            config=cfg.train,
            device=self.device,
        )
        prefilter = build_prefilter(cfg.detector)
        pipeline = Pipeline(
            datamodule=datamodule,
            predictor=predictor,
            detector_cfg=cfg.detector,
            prefilter=prefilter,
        )

        result = pipeline.run(test_ctx, calibration_ctx=calib_ctx)
        fitness = self.fitness_fn(
            f1=result.f1,
            n_params=result.n_params,
            infer_ms=result.infer_ms,
        )
        return FitnessResult(
            fitness=fitness,
            f1=result.f1,
            precision=result.precision,
            recall=result.recall,
            n_params=result.n_params,
            infer_ms=result.infer_ms,
            train_time_s=result.train_time_s,
            epochs_trained=len(result.history["train"]),
        )

    def _genome_to_config(self, genome: Genome) -> AppConfig:
        """Return a deep copy of the base config with the genome's genes applied.

        Args:
            genome: The genome whose genes to apply.

        Returns:
            A configured :class:`AppConfig` copy.
        """
        cfg = deepcopy(self.base)
        cfg.data.window_size = genome.window_size
        cfg.model.hidden_size = genome.hidden_size
        cfg.model.num_layers = genome.num_layers
        cfg.model.dropout = genome.dropout
        cfg.train.lr = genome.lr
        return cfg


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
@dataclass
class GenerationStats:
    """Aggregate statistics for one GA generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: int
    best_genome: Genome
    best_result: FitnessResult
    elapsed_s: float
    cache_hits: int = 0
    new_evals: int = 0


class GeneticAlgorithm:
    """Tournament selection + uniform crossover + per-gene mutation + elitism."""

    def __init__(
        self,
        space: SearchSpace,
        evaluator: GenomeEvaluator,
        population_size: int = 16,
        n_generations: int = 6,
        elitism: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        seed: int = 42,
    ) -> None:
        """Initialise the genetic algorithm.

        Args:
            space: Search space to sample genomes from.
            evaluator: Evaluator scoring each genome.
            population_size: Number of genomes per generation.
            n_generations: Number of generations to run.
            elitism: Top genomes carried over unchanged each generation.
            tournament_size: Number of competitors per selection tournament.
            mutation_rate: Per-gene mutation probability.
            crossover_rate: Probability of crossover (vs. cloning a parent).
            seed: Random seed.

        Raises:
            ValueError: If ``elitism`` or ``tournament_size`` is inconsistent
                with ``population_size``.
        """
        if elitism >= population_size:
            raise ValueError("elitism must be < population_size")
        if tournament_size > population_size:
            raise ValueError("tournament_size must be <= population_size")
        self.space = space
        self.evaluator = evaluator
        self.population_size = population_size
        self.n_generations = n_generations
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)

    def run(self) -> dict:
        """Run the full genetic search.

        Returns:
            A dict with ``best_genome``, ``best_result``, ``history`` and
            ``all_evaluated`` (every ``(genome, result)`` pair seen).
        """
        population = [
            self.space.random_genome(self.rng) for _ in range(self.population_size)
        ]
        history: list[GenerationStats] = []
        all_evaluated: list[tuple[Genome, FitnessResult]] = []

        for gen in range(1, self.n_generations + 1):
            t0 = time.perf_counter()
            print(f"\n{'=' * 78}")
            print(f"ПОКОЛІННЯ {gen}/{self.n_generations}")
            print(f"{'=' * 78}")

            results: list[FitnessResult] = []
            new_evals = 0
            cache_hits = 0
            for i, genome in enumerate(population, start=1):
                result, hit = self.evaluator.evaluate(genome)
                tag = "[cache]" if hit else "[train]"
                if hit:
                    cache_hits += 1
                else:
                    new_evals += 1
                print(
                    f"  [{i:>2}/{self.population_size}] {tag} {genome.short_repr()} "
                    f"-> fit={result.fitness:+.4f}  F1={result.f1:.3f}  "
                    f"P={result.n_params:>6,}  infer={result.infer_ms:.3f}ms  "
                    f"ep={result.epochs_trained}"
                )
                results.append(result)
                all_evaluated.append((genome, result))

            fitnesses = [r.fitness for r in results]
            best_idx = int(np.argmax(fitnesses))
            diversity = len({g.as_tuple() for g in population})
            stats = GenerationStats(
                generation=gen,
                best_fitness=fitnesses[best_idx],
                mean_fitness=float(np.mean(fitnesses)),
                worst_fitness=float(np.min(fitnesses)),
                diversity=diversity,
                best_genome=population[best_idx],
                best_result=results[best_idx],
                elapsed_s=time.perf_counter() - t0,
                cache_hits=cache_hits,
                new_evals=new_evals,
            )
            history.append(stats)
            print(
                f"  -> best={stats.best_fitness:+.4f}  "
                f"mean={stats.mean_fitness:+.4f}  "
                f"worst={stats.worst_fitness:+.4f}  "
                f"diversity={diversity}/{self.population_size}  "
                f"new={new_evals}  cached={cache_hits}  "
                f"elapsed={stats.elapsed_s:.1f}s"
            )

            if gen == self.n_generations:
                break
            population = self._next_generation(population, fitnesses)

        best_genome, best_result = max(all_evaluated, key=lambda gr: gr[1].fitness)
        return {
            "best_genome": best_genome,
            "best_result": best_result,
            "history": history,
            "all_evaluated": all_evaluated,
        }

    def _next_generation(
        self,
        population: list[Genome],
        fitnesses: list[float],
    ) -> list[Genome]:
        """Build the next generation via elitism, selection, crossover, mutation.

        Args:
            population: Current generation's genomes.
            fitnesses: Fitness value for each genome in ``population``.

        Returns:
            The next generation's genomes.
        """
        new_pop: list[Genome] = []
        elite_idx = list(np.argsort(fitnesses)[::-1][: self.elitism])
        for i in elite_idx:
            new_pop.append(population[i])

        while len(new_pop) < self.population_size:
            p1 = self._tournament(population, fitnesses)
            p2 = self._tournament(population, fitnesses)
            child = (
                self.space.crossover(p1, p2, self.rng)
                if self.rng.random() < self.crossover_rate
                else p1
            )
            child = self.space.mutate(child, self.rng, rate=self.mutation_rate)
            new_pop.append(child)
        return new_pop

    def _tournament(
        self,
        population: list[Genome],
        fitnesses: list[float],
    ) -> Genome:
        """Select the fittest genome from a random tournament.

        Args:
            population: Genomes to choose from.
            fitnesses: Fitness value for each genome in ``population``.

        Returns:
            The tournament winner.
        """
        indices = self.rng.choice(
            len(population),
            self.tournament_size,
            replace=False,
        )
        best = max(indices, key=lambda i: fitnesses[i])
        return population[best]


# ---------------------------------------------------------------------------
# Shortlist phase — multi-fidelity refinement of top GA candidates
# ---------------------------------------------------------------------------
def shortlist_top_k(
    candidates: list[tuple[Genome, FitnessResult]],
    evaluator: "GenomeEvaluator",
    top_k: int = 5,
    verbose: bool = True,
) -> list[tuple[Genome, FitnessResult]]:
    """Re-evaluate the top-K unique candidates at the evaluator's budget.

    IMPORTANT: ``evaluator`` MUST be a fresh ``GenomeEvaluator`` instance with
    an empty cache — otherwise it returns cached short-budget results.
    """
    seen: set[tuple] = set()
    ordered = sorted(candidates, key=lambda gr: -gr[1].fitness)
    top: list[Genome] = []
    for genome, _ in ordered:
        key = genome.as_tuple()
        if key in seen:
            continue
        seen.add(key)
        top.append(genome)
        if len(top) >= top_k:
            break

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"ШОРТЛІСТ — пере-оцінка топ-{len(top)} на розширеному бюджеті")
        print(f"{'=' * 78}")

    refined: list[tuple[Genome, FitnessResult]] = []
    for i, genome in enumerate(top, start=1):
        result, _ = evaluator.evaluate(genome)
        if verbose:
            print(
                f"  [{i:>2}/{len(top)}] {genome.short_repr()} -> "
                f"fit={result.fitness:+.4f}  F1={result.f1:.3f}  "
                f"P={result.n_params:>6,}  infer={result.infer_ms:.3f}ms  "
                f"ep={result.epochs_trained}"
            )
        refined.append((genome, result))

    if verbose:
        bg, br = max(refined, key=lambda gr: gr[1].fitness)
        print(f"\n  -> шортліст-чемпіон: {bg.short_repr()}")
        print(
            f"     fit={br.fitness:+.4f}  F1={br.f1:.4f}  "
            f"P={br.n_params:,}  ep={br.epochs_trained}"
        )
    return refined


def pick_winner(
    refined: list[tuple[Genome, FitnessResult]],
) -> tuple[Genome, FitnessResult]:
    """Return the highest-fitness pair from a shortlist result."""
    return max(refined, key=lambda gr: gr[1].fitness)


# ---------------------------------------------------------------------------
# Production re-training of the winning genome
# ---------------------------------------------------------------------------
@dataclass
class RetrainArtifacts:
    """Full production-budget evaluation of one genome."""

    result: RunResult
    datamodule: DataModule
    test_ctx: TestContext


def retrain_genome_full(
    genome: Genome,
    config: AppConfig,
    device: str,
    train_data,
    test_data,
    test_labels,
    test_scenarios,
    calib_data,
    calib_labels,
    calib_scenarios,
) -> RetrainArtifacts:
    """Re-train a genome with the full production budget supplied via ``config``."""
    cfg = deepcopy(config)
    cfg.data.window_size = genome.window_size
    cfg.model.hidden_size = genome.hidden_size
    cfg.model.num_layers = genome.num_layers
    cfg.model.dropout = genome.dropout
    cfg.train.lr = genome.lr

    set_seed(cfg.seed)
    datamodule = DataModule(
        window_size=genome.window_size,
        val_split=cfg.data.val_split,
    ).fit_normal(train_data)
    test_ctx = datamodule.prepare_test(test_data, test_labels, test_scenarios)
    calib_ctx = datamodule.prepare_test(calib_data, calib_labels, calib_scenarios)

    net = GRUNet(
        n_features=N_FEATURES,
        hidden_size=genome.hidden_size,
        num_layers=genome.num_layers,
        dropout=genome.dropout,
    )
    predictor = TorchPredictor(
        net=net,
        name="GRU_evolved",
        config=cfg.train,
        device=device,
    )
    prefilter = build_prefilter(cfg.detector)
    pipeline = Pipeline(
        datamodule=datamodule,
        predictor=predictor,
        detector_cfg=cfg.detector,
        prefilter=prefilter,
    )
    result = pipeline.run(test_ctx, calibration_ctx=calib_ctx)
    return RetrainArtifacts(result=result, datamodule=datamodule, test_ctx=test_ctx)
