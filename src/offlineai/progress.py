"""Progress reporting interface (section 46).

Lives outside ``cli/`` on purpose: the builder and the artifact sources report
progress, and neither should import from the command-line layer. The Rich
implementation is in :mod:`offlineai.cli.progress`; everything below the CLI
sees only this protocol.

:class:`NullReporter` is the default, and it accepts every call silently. That
means no code path has to ask whether reporting is enabled - it just reports,
and the reporter decides whether anyone is listening.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, runtime_checkable

__all__ = ["NullReporter", "ProgressCallback", "ProgressReporter", "download_callback"]

#: What :func:`~offlineai.artifacts.download.download_resumable` emits:
#: ``(bytes_downloaded, total_or_None)``.
ProgressCallback = Callable[[int, "int | None"], None]


@runtime_checkable
class ProgressReporter(Protocol):
    """Receives progress from anywhere in the build.

    Every method must tolerate a key it has never seen. Sources run in a
    thread pool and can be cancelled mid-flight, so a stray ``advance`` after
    a ``finish`` is normal - and must never be the thing that fails a build.
    """

    def set_overall(self, description: str, total: int | None) -> None:
        """Declare the aggregate, so one bar can show the whole build."""
        ...

    def start(self, key: str, description: str, total: int | None = None) -> None: ...

    def advance(self, key: str, amount: int) -> None: ...

    def update(self, key: str, completed: int, total: int | None = None) -> None: ...

    def finish(self, key: str, *, detail: str | None = None, failed: bool = False) -> None: ...

    def __enter__(self) -> ProgressReporter: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class NullReporter:
    """Reports nothing. The default, and what ``--json`` and ``--quiet`` get."""

    def set_overall(self, description: str, total: int | None) -> None:
        return

    def start(self, key: str, description: str, total: int | None = None) -> None:
        return

    def advance(self, key: str, amount: int) -> None:
        return

    def update(self, key: str, completed: int, total: int | None = None) -> None:
        return

    def finish(self, key: str, *, detail: str | None = None, failed: bool = False) -> None:
        return

    def __enter__(self) -> NullReporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return


def download_callback(reporter: ProgressReporter, key: str) -> ProgressCallback:
    """Adapt a reporter to the callback shape downloads emit.

    ``download_resumable`` reports ``(downloaded, total)`` and knows nothing
    about keys; the reporter needs a key and knows nothing about downloads.
    One adapter here keeps both of them ignorant of the other.
    """

    def report(downloaded: int, total: int | None) -> None:
        reporter.update(key, downloaded, total)

    return report
