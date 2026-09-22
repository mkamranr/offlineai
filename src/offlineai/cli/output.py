"""Rendering for command results (sections 46 and 47).

Commands return typed models; this module turns them into text or JSON. The
alternative - scattering ``if json_output:`` through twenty-five command
modules - reliably produces JSON that drifts from the human output and breaks
the automation that section 47 exists to serve.

Everything human goes to stdout; logs go to stderr. A caller doing
``offlineai status --json | jq`` must never receive a log line.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from offlineai.bundler.results import (
    BuildResult,
    CheckStatus,
    InspectResult,
    VerifyResult,
)
from offlineai.errors import OfflineAIError
from offlineai.utils.sizes import format_bytes

__all__ = ["Output", "OutputFormat"]

_STATUS_STYLE = {
    CheckStatus.OK: "green",
    CheckStatus.FAILED: "bold red",
    CheckStatus.WARNING: "yellow",
    CheckStatus.SKIPPED: "dim",
}


@dataclass(slots=True)
class OutputFormat:
    json: bool = False
    quiet: bool = False


class Output:
    """Writes command results in the requested form."""

    def __init__(self, fmt: OutputFormat | None = None) -> None:
        self.fmt = fmt or OutputFormat()
        # force_terminal=None lets rich detect a TTY; colour is dropped when
        # piped, which keeps --json and shell redirection clean.
        self.console = Console(file=sys.stdout, soft_wrap=True)
        self.err = Console(file=sys.stderr, soft_wrap=True)

    # -- primitives ------------------------------------------------------

    def line(self, text: str = "", *, style: str | None = None) -> None:
        if self.fmt.quiet or self.fmt.json:
            return
        self.console.print(text, style=style, highlight=False)

    def warn(self, text: str) -> None:
        if self.fmt.json:
            return
        self.err.print(f"WARNING: {text}", style="yellow", highlight=False)

    def error(self, err: OfflineAIError) -> None:
        if self.fmt.json:
            self.console.print_json(json.dumps(err.as_dict()))
            return
        self.err.print(err.render(), style="red", highlight=False)

    def emit(self, model: BaseModel) -> None:
        """Emit a result model as JSON. No-op in human mode."""
        if self.fmt.json:
            self.console.print_json(json.dumps(model.model_dump(mode="json", by_alias=True)))

    def emit_raw(self, payload: dict[str, Any]) -> None:
        if self.fmt.json:
            self.console.print_json(json.dumps(payload))

    # -- result renderers ------------------------------------------------

    def verify_result(self, result: VerifyResult) -> None:
        if self.fmt.json:
            self.emit(result)
            return
        self.line("OfflineAI Bundle Verification")
        self.line()
        self.line(f"Package: {result.package}")
        self.line(f"Version: {result.version}")
        self.line()
        width = max((len(c.name) for c in result.checks), default=0) + 2
        for check in result.checks:
            label = f"{check.name}:".ljust(width)
            suffix = f"  ({check.detail})" if check.detail else ""
            self.line(
                f"{label}{check.status.value}{suffix}",
                style=_STATUS_STYLE[check.status],
            )
        self.line()
        verdict = "VERIFIED" if result.verified else "FAILED"
        self.line(f"Result: {verdict}", style="bold green" if result.verified else "bold red")

    def inspect_result(self, result: InspectResult) -> None:
        if self.fmt.json:
            self.emit(result)
            return
        self.line(f"Package:\n  {result.package}")
        self.line(f"\nVersion:\n  {result.version}")
        self.line(f"\nFormat version:\n  {result.format_version}")
        if result.platforms:
            self.line(f"\nArchitecture:\n  {', '.join(result.platforms)}")

        self.line("\nArtifacts:")
        if result.artifact_sizes:
            width = max(len(t.value) for t in result.artifact_sizes) + 2
            for artifact_type, size in sorted(result.artifact_sizes.items()):
                count = result.artifact_counts.get(artifact_type, 0)
                label = f"{artifact_type.value}:".ljust(width)
                self.line(f"  {label}{format_bytes(size):>10}  ({count} file(s))")
        else:
            self.line("  (none)")

        self.line(f"\nTotal:\n  {format_bytes(result.total_size)}")
        self.line(f"\nBundle file:\n  {format_bytes(result.file_size)}")

        if result.gpu_vendor:
            self.line(f"\nGPU:\n  {result.gpu_vendor}")
            if result.gpu_minimum_vram_gb:
                self.line(f"  Minimum VRAM: {result.gpu_minimum_vram_gb} GB")
        if result.docker_minimum_version:
            self.line(f"\nRuntime:\n  Docker >= {result.docker_minimum_version}")
        if result.required_secrets:
            self.line("\nRequired secrets (supplied externally):")
            for name in result.required_secrets:
                self.line(f"  {name}: REQUIRED")

        self.line(f"\nSignature:\n  {'PRESENT' if result.signature_present else 'ABSENT'}")
        self.line(f"\nSBOM:\n  {'PRESENT' if result.sbom_present else 'ABSENT'}")
        if result.compression.value != "none":
            self.line(f"\nCompression:\n  {result.compression.value}")

    def build_step(
        self, index: int, total: int, name: str, status: CheckStatus, detail: str | None
    ) -> None:
        if self.fmt.quiet or self.fmt.json:
            return
        label = f"[{index}/{total}] {name}".ljust(44)
        suffix = f"  {detail}" if detail else ""
        self.console.print(
            f"{label}{status.value}{suffix}",
            style=_STATUS_STYLE[status],
            highlight=False,
        )

    def build_result(self, result: BuildResult) -> None:
        if self.fmt.json:
            self.emit(result)
            return
        self.line()
        self.line("Bundle created:")
        self.line()
        self.line(f"  {result.bundle_path}")
        self.line()
        self.line(f"Size:   {format_bytes(result.size)}")
        self.line(f"SHA256: {result.sha256}")
        if result.warnings:
            self.line()
            for warning in result.warnings:
                self.warn(warning)
        self.line()
        self.line("Build completed successfully.", style="bold green")
