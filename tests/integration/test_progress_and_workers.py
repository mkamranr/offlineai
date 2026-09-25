"""Sections 45 and 46, end to end through the CLI.

The property worth protecting is that concurrency never changes the bundle -
section 34 wants reproducibility, and a manifest whose artifact order depended
on which download finished first would not be reproducible at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from offlineai.bundler.archive import BundleReader
from offlineai.bundler.builder import BundleBuilder
from offlineai.cli.main import app
from offlineai.config.settings import load_settings
from offlineai.exitcodes import ExitCode

runner = CliRunner()

#: Matches an ANSI escape sequence, which is what a progress bar emits.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@pytest.fixture
def package(tmp_path: Path) -> Path:
    source = tmp_path / "pkg"
    weights = source / "weights"
    weights.mkdir(parents=True)
    for i in range(1, 13):
        (weights / f"model-{i:05d}-of-00012.safetensors").write_bytes(bytes([i]) * 4096)
    (weights / "config.json").write_text('{"model_type": "demo"}')
    (source / "offlineai.yaml").write_text(
        """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: workers-demo
  version: 1.0.0
models:
  - name: demo
    source:
      type: local
      path: ./weights
"""
    )
    return source


def build_with(package: Path, tmp_path: Path, workers: int, name: str) -> Path:
    settings = load_settings(home=tmp_path / f"home-{name}")
    settings.ensure_directories()
    result = BundleBuilder(settings, workers=workers).build(package, output=tmp_path / name)
    return Path(result.bundle_path)


class TestConcurrencyDoesNotChangeTheBundle:
    """Section 34: a bundle must be reproducible."""

    @pytest.mark.parametrize("workers", [1, 2, 4, 16])
    def test_the_manifest_is_identical_at_any_worker_count(
        self, package: Path, tmp_path: Path, workers: int
    ) -> None:
        baseline = BundleReader.peek_manifest(build_with(package, tmp_path, 1, "baseline"))
        other = BundleReader.peek_manifest(build_with(package, tmp_path, workers, f"w{workers}"))
        assert [(a.path, a.sha256, a.size) for a in baseline.artifacts] == [
            (a.path, a.sha256, a.size) for a in other.artifacts
        ]

    def test_every_shard_is_present_whatever_the_worker_count(
        self, package: Path, tmp_path: Path
    ) -> None:
        manifest = BundleReader.peek_manifest(build_with(package, tmp_path, 8, "many"))
        names = {a.path.rsplit("/", 1)[-1] for a in manifest.artifacts}
        for i in range(1, 13):
            assert f"model-{i:05d}-of-00012.safetensors" in names


class TestWorkerFlag:
    def _env(self, tmp_path: Path) -> dict[str, str]:
        return {"OFFLINEAI_HOME": str(tmp_path / "cli-home")}

    def test_the_flag_is_accepted(self, package: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["build", str(package), "-o", str(tmp_path / "out"), "--workers", "8"],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.SUCCESS, result.output

    def test_the_short_form_works(self, package: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["build", str(package), "-o", str(tmp_path / "out"), "-j", "4"],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.SUCCESS, result.output

    @pytest.mark.parametrize("workers", ["0", "-1", "999"])
    def test_an_out_of_range_count_is_rejected(
        self, package: Path, tmp_path: Path, workers: str
    ) -> None:
        """Section 45 asks for a conservative default and sane bounds; an
        unbounded worker count would saturate storage."""
        result = runner.invoke(
            app,
            ["build", str(package), "-o", str(tmp_path / "out"), "--workers", workers],
            env=self._env(tmp_path),
        )
        assert result.exit_code != ExitCode.SUCCESS

    def test_the_configured_value_is_used_when_no_flag_is_given(
        self, package: Path, tmp_path: Path
    ) -> None:
        home = tmp_path / "configured"
        home.mkdir()
        (home / "config.yaml").write_text("downloads:\n  workers: 7\n")
        settings = load_settings(home=home)
        assert BundleBuilder(settings).workers == 7

    def test_the_flag_overrides_the_configured_value(self, package: Path, tmp_path: Path) -> None:
        home = tmp_path / "configured"
        home.mkdir()
        (home / "config.yaml").write_text("downloads:\n  workers: 7\n")
        settings = load_settings(home=home)
        assert BundleBuilder(settings, workers=2).workers == 2

    def test_the_default_is_conservative(self, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path / "default")
        assert BundleBuilder(settings).workers == 4


class TestProgressStaysOutOfMachineOutput:
    """Escape sequences in a JSON payload would break every consumer."""

    def _env(self, tmp_path: Path) -> dict[str, str]:
        return {"OFFLINEAI_HOME": str(tmp_path / "cli-home")}

    def test_json_build_output_parses(self, package: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["--json", "build", str(package), "-o", str(tmp_path / "out"), "-j", "8"],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.SUCCESS, result.output
        payload = json.loads(result.stdout)
        assert payload["package"] == "workers-demo"

    def test_json_verify_output_has_no_escape_sequences(
        self, package: Path, tmp_path: Path
    ) -> None:
        bundle = build_with(package, tmp_path, 4, "forverify")
        result = runner.invoke(app, ["--json", "verify", str(bundle)], env=self._env(tmp_path))
        assert result.exit_code == ExitCode.SUCCESS
        assert json.loads(result.stdout)["verified"] is True

    def test_quiet_build_emits_nothing_on_stdout(self, package: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["--quiet", "build", str(package), "-o", str(tmp_path / "out")],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert ANSI.sub("", result.stdout).strip() == ""
