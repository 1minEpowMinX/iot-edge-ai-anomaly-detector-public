"""UI primitives for the CLI — rich-backed with a plain-text fallback.

If ``rich`` is installed everything renders with colours, tables, panels and
live displays. If not, helpers degrade to clean plain-text output so nothing
breaks. Use these helpers (not bare ``print``) from runners to keep the CLI
visual style consistent.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterable

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
    _console: Console | None = Console(highlight=False)
except ImportError:
    _HAS_RICH = False
    _console = None


# Verbosity: 0 = quiet, 1 = normal, 2 = verbose
_VERBOSITY = 1


def set_verbosity(level: int) -> None:
    """Set the global verbosity level (0 = quiet, 1 = normal, 2 = verbose).

    Args:
        level: Desired level; values outside [0, 2] are clamped.
    """
    global _VERBOSITY
    _VERBOSITY = max(0, min(2, int(level)))


def is_verbose() -> bool:
    """Return True when verbosity is at the verbose (debug) level."""
    return _VERBOSITY >= 2


def is_quiet() -> bool:
    """Return True when non-essential output is suppressed."""
    return _VERBOSITY <= 0


def has_rich() -> bool:
    """Return True when the optional ``rich`` library is available."""
    return _HAS_RICH


def install_traceback() -> None:
    """Pretty tracebacks with rich (only if available)."""
    if _HAS_RICH:
        try:
            from rich.traceback import install

            install(show_locals=False, suppress=[])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def banner(title: str, subtitle: str = "") -> None:
    """Print a titled panel (rich) or a ruled header (plain text)."""
    if is_quiet():
        return
    if _HAS_RICH:
        text = Text(title, style="bold cyan")
        if subtitle:
            text.append(f"\n{subtitle}", style="dim")
        _console.print(Panel(text, expand=False, border_style="cyan", padding=(0, 2)))
    else:
        print()
        print("=" * 78)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 78)


def rule(title: str = "") -> None:
    """Print a horizontal rule, optionally labelled with ``title``."""
    if is_quiet():
        return
    if _HAS_RICH:
        _console.print(Rule(title, style="dim cyan") if title else Rule(style="dim"))
    else:
        print("-" * 78 + (f"  {title}" if title else ""))


def step(idx: int, total: int, message: str) -> None:
    """Print a numbered progress step like ``[2/3] message``.

    Args:
        idx: 1-based index of the current step.
        total: Total number of steps.
        message: Description of the step.
    """
    if is_quiet():
        return
    if _HAS_RICH:
        _console.print(f"[bold cyan]\\[{idx}/{total}][/bold cyan] {message}")
    else:
        print(f"[{idx}/{total}] {message}")


def info(message: str) -> None:
    """Print a secondary (dimmed) informational line."""
    if is_quiet():
        return
    if _HAS_RICH:
        _console.print(f"  [dim]{message}[/dim]")
    else:
        print(f"  {message}")


def debug(message: str) -> None:
    """Print a line only when running at verbose level."""
    if not is_verbose():
        return
    if _HAS_RICH:
        _console.print(f"  [dim italic]{message}[/dim italic]")
    else:
        print(f"  {message}")


def success(message: str) -> None:
    """Print a success line marked with a check mark."""
    if is_quiet():
        return
    if _HAS_RICH:
        _console.print(f"[bold green]OK[/bold green] {message}")
    else:
        print(f"OK  {message}")


def warn(message: str) -> None:
    """Print a warning line marked with an exclamation mark."""
    if _HAS_RICH:
        _console.print(f"[bold yellow]![/bold yellow] {message}")
    else:
        print(f"WARN  {message}")


def error(message: str) -> None:
    """Print an error line (sent to stderr in plain-text mode)."""
    if _HAS_RICH:
        _console.print(f"[bold red]!![/bold red] {message}", style="red")
    else:
        print(f"ERROR  {message}", file=sys.stderr)


def kv_table(title: str, rows: Iterable[tuple[str, Any]]) -> None:
    """Two-column key/value summary table."""
    if is_quiet():
        return
    rows = list(rows)
    if _HAS_RICH:
        t = Table(title=title, show_header=False, box=None, padding=(0, 1))
        t.add_column("k", style="dim")
        t.add_column("v", justify="right", style="bold")
        for k, v in rows:
            t.add_row(str(k), str(v))
        _console.print(t)
    else:
        print()
        print(f"  {title}")
        print("  " + "-" * 40)
        for k, v in rows:
            print(f"    {k:>22}  {v}")


def comparison_table(
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    """N-column comparison table — for compare/sweep/ablation results."""
    if is_quiet():
        return
    if _HAS_RICH:
        t = Table(title=title, show_header=True, header_style="bold cyan")
        for i, col in enumerate(columns):
            t.add_column(col, justify="right" if i > 0 else "left")
        for row in rows:
            t.add_row(*(str(c) for c in row))
        _console.print(t)
    else:
        print()
        print(f"  {title}")
        widths = [
            max(len(str(c)) for c in [col] + [row[i] for row in rows])
            for i, col in enumerate(columns)
        ]
        sep = "  ".join("-" * w for w in widths)
        header = "  ".join(f"{col:>{w}}" for col, w in zip(columns, widths))
        print(header)
        print(sep)
        for row in rows:
            print("  ".join(f"{str(c):>{w}}" for c, w in zip(row, widths)))


# ---------------------------------------------------------------------------
# Live display (used by `live` command)
# ---------------------------------------------------------------------------
class LiveScroller:
    """Append-style live table for streaming rows (e.g. real-time samples).

    With rich: re-renders a table on each ``add_row``, showing last ``max_rows``.
    Without rich: prints rows as plain colored text.
    """

    def __init__(self, columns: list[str], max_rows: int = 20) -> None:
        """Initialise the scroller.

        Args:
            columns: Column headers for the streaming table.
            max_rows: Maximum number of recent rows kept on screen.
        """
        self.columns = columns
        self.max_rows = max_rows
        self._rows: list[list[str]] = []
        self._live: Live | None = None
        self._row_styles: list[str | None] = []
        self._footer_lines: list[tuple[str, str | None]] | None = None

    def _render(self):
        """Build the current snapshot: the table, plus a live footer if set."""
        t = Table(show_header=True, header_style="bold cyan", expand=False)
        for col in self.columns:
            t.add_column(col, justify="right")
        for row, style in zip(self._rows, self._row_styles):
            t.add_row(*row, style=style)
        if self._footer_lines:
            body = [t, *(Text(text, style=style or "") for text, style in self._footer_lines)]
            return Group(*body)
        return t

    def set_footer(self, lines: list[tuple[str, str | None]] | None) -> None:
        """Set (or clear) a live panel rendered below the table, in place.

        Unlike :meth:`note` (which scrolls a one-off block into history), the
        footer is overwritten on every update — use it for a continuously
        refreshing readout such as a per-channel residual breakdown.

        Args:
            lines: ``(text, style)`` pairs, or ``None`` to clear the footer.
                Rich-only; a no-op on the plain-text fallback.
        """
        self._footer_lines = lines
        if _HAS_RICH and self._live is not None:
            self._live.update(self._render())

    def __enter__(self):
        """Enter the context: start the live display or print a plain header."""
        if _HAS_RICH:
            self._live = Live(self._render(), refresh_per_second=8, console=_console)
            self._live.__enter__()
        else:
            print("  ".join(f"{c:>10}" for c in self.columns))
            print("-" * (12 * len(self.columns)))
        return self

    def __exit__(self, *exc):
        """Exit the context and tear down the live display."""
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def add_row(self, cells: list[str], style: str | None = None) -> None:
        """Append one row to the live table.

        Args:
            cells: Cell values for the row, one per column.
            style: Optional rich style hint (e.g. ``"bold red"``).
        """
        if _HAS_RICH and self._live is not None:
            self._rows.append([str(c) for c in cells])
            self._row_styles.append(style)
            if len(self._rows) > self.max_rows:
                self._rows = self._rows[-self.max_rows :]
                self._row_styles = self._row_styles[-self.max_rows :]
            self._live.update(self._render())
        else:
            # plain fallback — print one row at a time, with ANSI colors if style hints
            line = "  ".join(f"{str(c):>10}" for c in cells)
            if style and "red" in style:
                print(f"\033[91m{line}\033[0m")
            elif style and "green" in style:
                print(f"\033[92m{line}\033[0m")
            else:
                print(line)

    def note(self, lines: list[tuple[str, str | None]]) -> None:
        """Print a detail block above the live table (scrollback), not in it.

        Useful for one-off annotations (e.g. an anomaly breakdown) that should
        persist while the streaming table keeps updating below.

        Args:
            lines: ``(text, style)`` pairs; ``style`` is a rich style hint
                (e.g. ``"bold red"``, ``"yellow"``, ``"dim"``) or ``None``.
        """
        if _HAS_RICH and self._live is not None:
            for text, style in lines:
                self._live.console.print(Text(text, style=style or ""))
        else:
            for text, style in lines:
                if style and "red" in style:
                    print(f"\033[91m{text}\033[0m")
                elif style and ("yellow" in style or "amber" in style):
                    print(f"\033[93m{text}\033[0m")
                elif style and "dim" in style:
                    print(f"\033[90m{text}\033[0m")
                else:
                    print(text)


# ---------------------------------------------------------------------------
# Spinner (for long-running steps without precise progress)
# ---------------------------------------------------------------------------
@contextmanager
def spinner(message: str):
    """Context manager showing a spinner (rich) during a long-running step.

    Args:
        message: Text displayed next to the spinner.
    """
    if _HAS_RICH and not is_quiet():
        with _console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield
    else:
        if not is_quiet():
            print(f"  {message}…")
        yield
