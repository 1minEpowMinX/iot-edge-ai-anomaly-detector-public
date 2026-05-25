"""Unified command-line interface.

Entry: ``python -m src <command> [options]``

Production commands:
    demo            Train evolved GRU on holdout + dashboards (showcase)
    train           Train with configurable hyperparameters
    infer           Score a CSV with a saved model
    live            Real-time anomaly monitoring of the host (psutil)
    collect         Stream real metrics to CSV

Research commands:
    search          Three-phase evolutionary architecture search
    compare         GRU vs LSTM vs MovingAverage
    ablate          5 vs 12 feature subset comparison
    sweep           Hyperparameter sweep
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from . import __version__
from . import _ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dedent(s: str) -> str:
    """Dedent and strip a multi-line help string."""
    return textwrap.dedent(s).strip()


def _add_train_options(p: argparse.ArgumentParser) -> None:
    """Add the hyperparameter-override flags to a subparser.

    Args:
        p: The argument parser to extend.
    """
    g = p.add_argument_group("гіперпараметри (перевизначення evolved-дефолтів)")
    g.add_argument("--epochs", type=int, metavar="N", help="максимальна кількість епох")
    g.add_argument("--batch-size", type=int, metavar="N", help="розмір міні-батчу")
    g.add_argument("--lr", type=float, metavar="X", help="початкова швидкість навчання")
    g.add_argument(
        "--hidden", type=int, metavar="N", help="розмір прихованого шару GRU"
    )
    g.add_argument("--window", type=int, metavar="N", help="довжина вхідного вікна")
    g.add_argument("--layers", type=int, metavar="N", help="кількість шарів GRU")
    g.add_argument(
        "--dropout", type=float, metavar="X", help="частка відсіву (dropout)"
    )
    g.add_argument(
        "--seed", type=int, metavar="N", help="зерно генератора випадкових чисел"
    )
    g.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        metavar="DEV",
        help="пристрій обчислень",
    )


def _build_overrides(args: argparse.Namespace) -> dict:
    """Collect explicitly-set hyperparameter flags into an overrides dict.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Mapping of config field names to caller-supplied values; flags left
        at their default (``None``) are omitted.
    """
    keys = (
        "epochs",
        "batch_size",
        "lr",
        "hidden",
        "window",
        "layers",
        "dropout",
        "seed",
        "device",
    )
    name_map = {
        "hidden": "hidden_size",
        "window": "window_size",
        "layers": "num_layers",
    }
    out = {}
    for k in keys:
        v = getattr(args, k, None)
        if v is not None:
            out[name_map.get(k, k)] = v
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    """Add the shared ``--output`` option to a subparser.

    Args:
        p: The argument parser to extend.
    """
    g = p.add_argument_group("вивід")
    g.add_argument(
        "--output",
        "-o",
        default="artifacts",
        metavar="DIR",
        help="директорія артефактів (default: %(default)s)",
    )


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build and return the full CLI argument parser with all subcommands.

    Returns:
        The configured top-level ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=_dedent("""
            IoT виявлення аномалій — CLI для виробництва та досліджень.

            Автономна система виявлення аномалій у метриках IoT-хостів
            за допомогою легкої GRU з каскадним префільтром. Архітектуру
            знайдено еволюційним пошуком; фінальні метрики звітуються
            на тестовій вибірці, невиданій під час пошуку.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src demo                       # 30с демонстрація
              python -m src train --epochs 100 --lr 1e-3
              python -m src live --duration 60         # моніторинг хосту в реальному часі
              python -m src search --quick             # швидкий еволюційний пошук

            Запустіть `<команда> --help` для параметрів конкретної команди.
        """),
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="збільшити детальність виводу (можна повторювати: -vv)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="вимкнути несуттєвий вивід"
    )

    subs = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="<command>",
        title="commands",
    )

    # ----- demo -----
    p_demo = subs.add_parser(
        "demo",
        help="навчити evolved-модель + зберегти дашборди (showcase)",
        description="Production showcase: навчає переможця еволюційного пошуку "
        "на holdout-даних та зберігає всі дашборди.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src demo
              python -m src demo --quick
              python -m src demo -o ./run1
        """),
    )
    _add_common(p_demo)
    p_demo.add_argument(
        "--quick",
        action="store_true",
        help="швидкий режим (менший датасет, менше епох)",
    )

    # ----- train -----
    p_train = subs.add_parser(
        "train",
        help="навчання з власними гіперпараметрами",
        description="Навчити GRU з заданими гіперпараметрами. Без перевизначень "
        "використовуються evolved-дефолти.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src train
              python -m src train --epochs 100 --lr 1e-3
              python -m src train --hidden 32 --layers 2 --window 50 --dropout 0.2
              python -m src train --device cpu --seed 0
              python -m src train --train-csv vm_normal.csv   # навчання на реальних даних VM
        """),
    )
    _add_common(p_train)
    p_train.add_argument(
        "--quick",
        action="store_true",
        help="швидкий режим (менший датасет, менше епох)",
    )
    p_train.add_argument(
        "--train-csv",
        metavar="CSV",
        dest="train_csv",
        default=None,
        help="CSV реальних метрик для навчання (з `collect`); "
        "замінює синтетичні дані",
    )
    _add_train_options(p_train)

    # ----- infer -----
    p_infer = subs.add_parser(
        "infer",
        help="оцінити CSV зі збереженою моделлю",
        description="Завантажити збережений bundle моделі та запустити інференс "
        "на CSV, що відповідає навченим іменам ознак.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src infer --model artifacts/ --data real.csv
              python -m src infer --model artifacts/ --data real.csv -o det.csv

            Очікувані стовпці CSV:
              t, cpu, cpu_iowait, load_avg_1m, ram, swap,
              disk_read_bps, disk_write_bps, net_tx, net_rx,
              net_packets_tx, tcp_conn, proc_count

            Генерувати сумісний CSV: `python -m src collect`.
        """),
    )
    p_infer.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="директорія моделі (model.pt + scaler_*.npy + meta.json)",
    )
    p_infer.add_argument(
        "--data", required=True, metavar="CSV", help="вхідний CSV-файл"
    )
    p_infer.add_argument(
        "--output", "-o", metavar="CSV", help="опціональний CSV для виявлень по вікнах"
    )
    p_infer.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="cpu", metavar="DEV"
    )

    # ----- live -----
    p_live = subs.add_parser(
        "live",
        help="моніторинг у реальному часі (psutil + збережена модель)",
        description="Моніторинг хосту в реальному часі за допомогою psutil, "
        "оцінюючи кожен новий семпл навченою моделлю.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src live --duration 60
              python -m src live --model ./my_model --interval 0.5
              python -m src live --no-color    # вимкнути ANSI-кольори

            Порада: навантажте процесор в іншому терміналі для спрацювання алертів:
              python -c "while True: pass"
        """),
    )
    p_live.add_argument(
        "--model",
        default="artifacts",
        metavar="DIR",
        help="директорія моделі (default: %(default)s)",
    )
    p_live.add_argument(
        "--duration",
        type=float,
        default=60.0,
        metavar="S",
        help="тривалість моніторингу в секундах (default: %(default)s)",
    )
    p_live.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="S",
        help="інтервал між семплами в секундах (default: %(default)s)",
    )
    p_live.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="cpu", metavar="DEV"
    )
    p_live.add_argument(
        "--no-color", action="store_true", help="вимкнути кольоровий вивід"
    )

    # ----- collect -----
    p_col = subs.add_parser(
        "collect",
        help="записати метрики psutil -> CSV",
        description="Зібрати метрики хосту через psutil і записати CSV, "
        "сумісний з `infer`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src collect --duration 60
              python -m src collect --duration 300 --interval 0.5 -o long.csv
        """),
    )
    p_col.add_argument(
        "--duration",
        type=float,
        default=60.0,
        metavar="S",
        help="тривалість збору в секундах (default: %(default)s)",
    )
    p_col.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="S",
        help="інтервал між семплами в секундах (default: %(default)s)",
    )
    p_col.add_argument(
        "--output",
        "-o",
        default="real_metrics.csv",
        metavar="CSV",
        help="шлях виводу (default: %(default)s)",
    )

    # ----- search -----
    p_search = subs.add_parser(
        "search",
        help="триетапний еволюційний пошук архітектури",
        description="Запустити ГА -> shortlist -> holdout-ретренінг для пошуку "
        "найкращої архітектури з обмеженнями edge-деплою.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src search --quick
              python -m src search --population 16 --generations 12
              python -m src search --top-k 5 --mutation-rate 0.25
        """),
    )
    _add_common(p_search)
    p_search.set_defaults(output="artifacts/evolution")
    p_search.add_argument(
        "--quick",
        action="store_true",
        help="скорочений бюджет для швидкої демонстрації",
    )
    p_search.add_argument(
        "--population",
        type=int,
        default=16,
        metavar="N",
        help="розмір популяції (default: %(default)s)",
    )
    p_search.add_argument(
        "--generations",
        type=int,
        default=12,
        metavar="N",
        help="кількість поколінь (default: %(default)s)",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="top-K кандидатів для shortlist (default: %(default)s)",
    )
    p_search.add_argument(
        "--mutation-rate",
        type=float,
        default=0.25,
        metavar="X",
        help="частота мутації на ген (default: %(default)s)",
    )

    # ----- compare -----
    p_cmp = subs.add_parser(
        "compare",
        help="GRU vs LSTM vs MovingAverage",
        description="Порівняти три класи предикторів на ідентичних даних.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Приклад:\n  python -m src compare",
    )
    _add_common(p_cmp)
    p_cmp.set_defaults(output="artifacts/comparison")

    # ----- ablate -----
    p_abl = subs.add_parser(
        "ablate",
        help="порівняння підмножин: 5 vs 12 ознак",
        description="Навчити дві моделі — одну на 5 мінімальних метриках "
        "(специфікація диплому), іншу на повних 12 — та порівняти.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Приклад:\n  python -m src ablate",
    )
    _add_common(p_abl)
    p_abl.set_defaults(output="artifacts/ablation")

    # ----- sweep -----
    p_swp = subs.add_parser(
        "sweep",
        help="перебір гіперпараметрів",
        description="Перебрати один або кілька гіперпараметрів та побудувати графік тренду.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_dedent("""
            Приклади:
              python -m src sweep
              python -m src sweep --axis window_size
        """),
    )
    _add_common(p_swp)
    p_swp.set_defaults(output="artifacts/sweeps")
    p_swp.add_argument(
        "--axis",
        choices=["all", "window_size", "hidden_size"],
        default="all",
        help="яку вісь перебирати (default: %(default)s)",
    )

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch the command and return a process exit code.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code (0 on success, non-zero on error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Verbosity
    if args.quiet:
        _ui.set_verbosity(0)
    elif args.verbose:
        _ui.set_verbosity(1 + args.verbose)
    _ui.install_traceback()

    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        _ui.warn("перервано користувачем")
        return 130
    except SystemExit:
        raise
    except FileNotFoundError as e:
        _ui.error(str(e))
        return 2
    except ValueError as e:
        _ui.error(str(e))
        return 2
    except Exception as e:
        _ui.error(f"{type(e).__name__}: {e}")
        if _ui.is_verbose():
            raise
        _ui.info("(запустіть з -v для повного traceback)")
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    """Route parsed arguments to the matching command runner.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code from the command.
    """
    cmd = args.command
    if cmd == "demo":
        from .experiments.demo import run_demo

        run_demo(quick=args.quick, output_dir=args.output)
    elif cmd == "train":
        from .experiments.train import run_train

        run_train(
            overrides=_build_overrides(args),
            output_dir=args.output,
            quick=args.quick,
            csv_path=getattr(args, "train_csv", None),
        )
    elif cmd == "infer":
        from .experiments.infer import run_infer

        run_infer(
            model_dir=args.model,
            data_path=args.data,
            output_path=args.output,
            device=args.device,
        )
    elif cmd == "live":
        from .experiments.live import run_live

        run_live(
            model_dir=args.model,
            duration=args.duration,
            interval=args.interval,
            device=args.device,
            color=not args.no_color,
        )
    elif cmd == "collect":
        from .experiments.collect import run_collect

        run_collect(
            duration=args.duration,
            interval=args.interval,
            output=args.output,
        )
    elif cmd == "search":
        from .experiments.search import run_search

        run_search(
            output_dir=args.output,
            population_size=args.population,
            n_generations=args.generations,
            top_k_shortlist=args.top_k,
            mutation_rate=args.mutation_rate,
            quick=args.quick,
        )
    elif cmd == "compare":
        from .experiments.comparison import run_comparison

        run_comparison(output_dir=args.output)
    elif cmd == "ablate":
        from .experiments.ablation import run_ablation

        run_ablation(output_dir=args.output)
    elif cmd == "sweep":
        from .experiments.sweep import run_sweep

        run_sweep(which=args.axis, output_dir=args.output)
    else:
        _ui.error(f"невідома команда: {cmd!r}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
