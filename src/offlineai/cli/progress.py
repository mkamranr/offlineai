"""Rich implementation of the progress protocol (section 46).

Shows one aggregate bar for the whole build plus a row per artifact currently
in flight. With eight workers and a hundred shards, keeping every completed
row on screen would bury the ones still running, so a finished row is removed.

All state is guarded by a lock: downloads run in a thread pool, so every
method here is called from several threads at once.
"""

from __future__ import annotations

import sys
import threading
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from offlineai.progress import NullReporter, ProgressReporter

__all__ = ["RichReporter", "select_reporter"]


class RichReporter:
    """Live progress bars on an interactive terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(file=sys.stderr)
        self._progress = Progress(
            SpinnerColumn(style="dim"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(compact=True),
            console=self._console,
            # Progress goes to stderr; stdout belongs to --json payloads, and
            # a transient display keeps the scrollback clean afterwards.
            transient=True,
        )
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskID] = {}
        self._overall: TaskID | None = None
        self.overall_completed = 0
        self._started = False

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> RichReporter:
        self._progress.start()
        self._started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._progress.stop()
        self._started = False
        with self._lock:
            self._tasks.clear()
            self._overall = None

    # -- reporting -------------------------------------------------------

    def set_overall(self, description: str, total: int | None) -> None:
        with self._lock:
            if self._overall is not None:
                self._progress.update(self._overall, description=description, total=total)
                return
            self._overall = self._progress.add_task(description, total=total)

    def start(self, key: str, description: str, total: int | None = None) -> None:
        with self._lock:
            if key in self._tasks:
                return
            self._tasks[key] = self._progress.add_task(description, total=total)

    def advance(self, key: str, amount: int) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is not None:
                self._progress.advance(task, amount)
            self._advance_overall(amount)

    def update(self, key: str, completed: int, total: int | None = None) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            # The aggregate advances by the delta, because `completed` is
            # absolute for this artifact but cumulative across the build.
            previous = int(self._progress.tasks[task].completed)
            self._progress.update(task, completed=completed, total=total)
            self._advance_overall(max(0, completed - previous))

    def finish(self, key: str, *, detail: str | None = None, failed: bool = False) -> None:
        with self._lock:
            task = self._tasks.pop(key, None)
            if task is None:
                return
            if failed and detail:
                self._console.print(f"  {detail}", style="red", highlight=False)
            self._progress.remove_task(task)

    def active_keys(self) -> list[str]:
        with self._lock:
            return list(self._tasks)

    # -- internals -------------------------------------------------------

    def _advance_overall(self, amount: int) -> None:
        """Caller holds the lock."""
        if amount <= 0:
            return
        self.overall_completed += amount
        if self._overall is not None:
            self._progress.advance(self._overall, amount)


def select_reporter(*, json: bool, quiet: bool, is_terminal: bool) -> ProgressReporter:
    """Choose a reporter for this invocation.

    Anything that suppresses human output suppresses progress too. A machine
    consuming ``--json``, a log file collecting piped output, or an operator
    who asked for quiet must not receive escape sequences and redraws.
    """
    if json or quiet or not is_terminal:
        return NullReporter()
    return RichReporter()
