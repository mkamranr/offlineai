"""Subprocess helpers.

Every external command goes through here so three things hold everywhere:
argument lists are never passed through a shell, timeouts are always set, and
command lines are redacted before they reach the log (section 48).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from offlineai.logging import get_logger, redact

__all__ = ["CommandResult", "have", "run"]

logger = get_logger("utils.proc")

#: Generous by default: `docker load` on a 12 GB image tar is not quick.
DEFAULT_TIMEOUT = 3600.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """stdout, or stderr when the command wrote its message there."""
        return self.stdout.strip() or self.stderr.strip()

    def first_line(self) -> str:
        return self.stdout.strip().splitlines()[0] if self.stdout.strip() else ""


def have(executable: str) -> bool:
    return shutil.which(executable) is not None


def run(
    args: list[str] | tuple[str, ...],
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> CommandResult:
    """Run a command, capturing output. Never raises on a non-zero exit.

    Callers decide what a failure means - for an installer step it is usually a
    structured :class:`~offlineai.errors.InstallationError` with the command's
    own stderr attached, which is far more useful than a traceback.
    """
    argv = [str(a) for a in args]
    logger.debug("exec: %s", redact(" ".join(argv)))

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, never shell=True
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=stdin,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(tuple(argv), 127, "", f"{argv[0]}: command not found")
    except subprocess.TimeoutExpired:
        return CommandResult(tuple(argv), 124, "", f"{argv[0]}: timed out after {timeout} seconds")

    return CommandResult(
        tuple(argv), completed.returncode, completed.stdout or "", completed.stderr or ""
    )
