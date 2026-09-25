"""Fixtures for command-layer tests.

These drive the CLI in-process through CliRunner rather than a subprocess, so
coverage is recorded for the command modules. The process boundary is covered
separately in ``test_cli_exit_codes.py``, which has to be a subprocess because
it exercises the ``run()`` entry point that CliRunner bypasses.

The container runtime is replaced with the in-memory fake everywhere, so these
tests are deterministic on a machine with no Docker.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from offlineai.bundler.builder import BundleBuilder
from offlineai.cli.main import app
from offlineai.config.settings import load_settings
from offlineai.registry.registry import Registry
from offlineai.runtime.base import RuntimeAvailability
from offlineai.runtime.fake import FakeRuntime

PACKAGE_YAML = """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: cli-demo
  version: 1.0.0
  description: A package for exercising the command layer.
  license: Apache-2.0
runtime:
  type: docker
architecture:
  - amd64
containers:
  - name: app
    image: python:3.12-slim
    platform: linux/amd64
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
environment:
  HF_HUB_OFFLINE: "1"
hardware:
  gpu:
    required: false
    vendor: nvidia
    minimum_vram_gb: 8
secrets:
  external:
    - CLI_DEMO_TOKEN
services:
  - name: app
    container: app
    ports:
      - "8000:8000"
install:
  healthcheck:
    command: ["true"]
    retries: 2
    interval_seconds: 1
"""


class Result:
    """A CLI invocation, with helpers for the assertions these tests make."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.exit_code: int = raw.exit_code
        self.stdout: str = raw.stdout

    @property
    def stderr(self) -> str:
        return self.raw.stderr

    @property
    def output(self) -> str:
        """Both streams.

        Errors are written to stderr on purpose, so stdout stays parseable for
        `--json`. Assertions about *what the operator sees* therefore have to
        look at both; assertions about machine output use `.json()`, which
        reads stdout alone and would fail loudly if anything leaked into it.
        """
        return self.stdout + self.raw.stderr

    def json(self) -> Any:
        return json.loads(self.stdout)

    def assert_ok(self) -> Result:
        assert self.exit_code == 0, (
            f"exited {self.exit_code}\n{self.stdout}\n{self.raw.exception!r}"
        )
        return self


@pytest.fixture
def run_cli() -> Any:
    """Invoke the CLI in-process. Environment comes from the isolated home."""
    runner = CliRunner()

    def invoke(*args: str) -> Result:
        return Result(runner.invoke(app, list(args)))

    return invoke


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    """Replace Docker everywhere a command reaches for it.

    Patched per-module rather than globally because each command module
    imports the class directly, and a test that silently fell through to real
    Docker would be both slow and machine-dependent.
    """
    runtime = FakeRuntime(version="27.3.1", gpu_support=False)

    class Stub:
        def __new__(cls, *args: object, **kwargs: object) -> FakeRuntime:
            return runtime

    for module in (
        "offlineai.cli.commands.lifecycle",
        "offlineai.cli.commands.diagnostics",
        "offlineai.bundler.builder",
    ):
        monkeypatch.setattr(f"{module}.DockerRuntime", Stub, raising=False)
    return runtime


@pytest.fixture
def gpu_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    """A runtime that reports GPU support, for the checks that need one."""
    runtime = FakeRuntime(version="27.3.1", gpu_support=True)

    class Stub:
        def __new__(cls, *args: object, **kwargs: object) -> FakeRuntime:
            return runtime

    for module in ("offlineai.cli.commands.diagnostics", "offlineai.cli.commands.lifecycle"):
        monkeypatch.setattr(f"{module}.DockerRuntime", Stub, raising=False)
    return runtime


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    source = tmp_path / "src" / "cli-demo"
    weights = source / "weights"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text('{"model_type": "demo"}')
    (weights / "model.safetensors").write_bytes(b"weights" * 512)
    (source / "offlineai.yaml").write_text(PACKAGE_YAML)
    (source / "README.md").write_text("# cli-demo\n")
    return source


@pytest.fixture
def bundle(package_dir: Path, tmp_path: Path, fake_runtime: FakeRuntime) -> Path:
    settings = load_settings()
    settings.ensure_directories()
    result = BundleBuilder(settings, runtime=fake_runtime).build(
        package_dir, output=tmp_path / "dist"
    )
    return Path(result.bundle_path)


@pytest.fixture
def imported(bundle: Path, fake_runtime: FakeRuntime) -> Path:
    """Import the bundle, then wipe the runtime to model the air gap.

    The same fake serves as builder and target, so without this the build's
    legitimate `pull` and the image it produced would still be present, and an
    assertion that install never pulls would pass for the wrong reason. After
    this line the runtime holds nothing: anything that works afterwards had to
    come out of the bundle.
    """
    Registry(load_settings()).import_bundle(bundle)
    fake_runtime.images.clear()
    fake_runtime.calls.clear()
    return bundle


@pytest.fixture
def installed(imported: Path, run_cli: Any, fake_runtime: FakeRuntime) -> Path:
    run_cli("install", "cli-demo").assert_ok()
    return imported


@pytest.fixture
def signing_key(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    from offlineai.security.signing import generate_keypair

    yield generate_keypair(tmp_path / "keys" / "signing-key.pem")


__all__ = ["RuntimeAvailability"]
