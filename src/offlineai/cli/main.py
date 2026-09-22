"""The ``offlineai`` command-line interface.

Section 65 asks for a professional CLI; sections 47 and 66 ask for machine
output and predictable exit codes. The structure here keeps all three cheap:

* every command builds a typed result and hands it to the renderer, so
  ``--json`` is never an afterthought;
* every deliberate failure is an :class:`OfflineAIError`, which carries its own
  exit code, so no command has to remember one;
* anything else escaping is treated as an internal error and labelled as such,
  rather than being dressed up as user error.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ParamSpec

import typer

from offlineai import __version__
from offlineai.cli.output import Output, OutputFormat
from offlineai.config.settings import Settings, load_settings
from offlineai.errors import OfflineAIError
from offlineai.exitcodes import ExitCode
from offlineai.logging import LogFormat, configure_logging

__all__ = ["app", "register", "run"]

app = typer.Typer(
    name="offlineai",
    help="Package complete AI workloads for deployment into air-gapped environments.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class Context:
    """Resolved global state, attached to the Typer context."""

    def __init__(self, settings: Settings, output: Output) -> None:
        self.settings = settings
        self.output = output


P = ParamSpec("P")


def handles_errors(fn: Callable[P, None]) -> Callable[P, None]:
    """Convert an OfflineAIError into its documented exit code.

    This lives on the command rather than in :func:`run` so that the mapping is
    part of the application itself: an embedder invoking ``app`` directly - the
    test runner, for one - gets the same exit codes as the console script, and
    the contract in section 66 is actually exercised by the test suite.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except OfflineAIError as exc:
            _render_error(exc, args)
            raise typer.Exit(int(exc.exit_code)) from None

    return wrapper


def _render_error(exc: OfflineAIError, args: tuple[object, ...]) -> None:
    """Print the error using the invocation's own output settings if we have them."""
    for arg in args:
        context = getattr(arg, "obj", None)
        if isinstance(context, Context):
            context.output.error(exc)
            return
    Output().error(exc)


def register(app: typer.Typer, name: str, fn: Callable[..., None]) -> None:
    """Register a command with error handling applied."""
    app.command(name)(handles_errors(fn))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"offlineai {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Override where bundles and the registry live."),
    ] = None,
    cache_dir: Annotated[
        Path | None, typer.Option("--cache-dir", help="Override the artifact cache location.")
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="debug, info, warning or error.", metavar="LEVEL"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON on stdout.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output.")] = False,
    strict_offline: Annotated[
        bool | None,
        typer.Option(
            "--strict-offline/--no-strict-offline",
            help="Forbid all network access. Recommended on air-gapped targets.",
        ),
    ] = None,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Build online. Transfer once. Run offline."""
    settings = load_settings(
        data_dir=data_dir,
        cache_dir=cache_dir,
        log_level=log_level,
        strict_offline=strict_offline,
    )
    # Logs go to stderr so stdout stays parseable; JSON mode switches them to
    # structured records for the same reason.
    configure_logging(
        level=settings.log_level, fmt=LogFormat.JSON if json_output else LogFormat.TEXT
    )
    ctx.obj = Context(settings=settings, output=Output(OutputFormat(json=json_output, quiet=quiet)))


def run() -> None:
    """Console-script entry point.

    Maps every deliberate failure to its documented exit code (section 66).
    Anything that is not an OfflineAIError is a bug, and is reported as an
    internal error rather than dressed up as user error.
    """
    from click import ClickException, UsageError

    try:
        # standalone_mode=False stops Click from calling sys.exit itself, so we
        # control the mapping. The catch is that in this mode Click *returns*
        # the code for a clean Exit rather than raising it, so the return value
        # has to be honoured - otherwise a command that raised typer.Exit(3)
        # would still exit 0.
        result = app(standalone_mode=False)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from None
    except UsageError as exc:
        exc.show()
        raise SystemExit(ExitCode.GENERAL_ERROR) from None
    except ClickException as exc:
        exc.show()
        raise SystemExit(ExitCode.GENERAL_ERROR) from None
    except OfflineAIError as exc:
        Output().error(exc)
        raise SystemExit(int(exc.exit_code)) from None
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(ExitCode.GENERAL_ERROR) from None
    except Exception as exc:  # noqa: BLE001 - last resort, reported honestly
        print(f"INTERNAL ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "This is a bug in OfflineAI. Re-run with --log-level debug for a traceback.",
            file=sys.stderr,
        )
        if _debug_enabled():
            raise
        raise SystemExit(ExitCode.GENERAL_ERROR) from None
    raise SystemExit(result if isinstance(result, int) else ExitCode.SUCCESS)


def _debug_enabled() -> bool:
    import os

    return os.environ.get("OFFLINEAI_LOG_LEVEL", "").lower() == "debug"


# Commands are registered by importing their modules. The imports sit at the
# bottom because each command module imports Context and register from here;
# doing it at the top would be a cycle.
def _register_commands() -> None:
    from offlineai.cli.commands import (
        build,
        import_bundle,
        init,
        inspect,
        lifecycle,
        registry_cmds,
        verify,
    )

    for module in (
        init,
        build,
        verify,
        inspect,
        import_bundle,
        registry_cmds,
        lifecycle,
    ):
        module.register(app)


_register_commands()
