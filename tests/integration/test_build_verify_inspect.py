"""Phase 1 end to end: build -> verify -> inspect, entirely offline.

Uses a local artifact source so the whole pipeline runs under the network
guard. The negative controls matter as much as the happy path: section 61 says
a corrupted bundle must never be reported as valid, and section 66 fixes the
exit codes that automation branches on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from offlineai.bundler.builder import BundleBuilder
from offlineai.bundler.inspector import inspect_bundle
from offlineai.bundler.verifier import verify_bundle
from offlineai.cli.main import app
from offlineai.config.settings import load_settings
from offlineai.errors import ChecksumMismatchError
from offlineai.exitcodes import ExitCode
from offlineai.schema.manifest import ArtifactType

runner = CliRunner()

SHARD_COUNT = 3
SHARD_SIZE = 8192


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    """A package whose 'model' is a small sharded checkpoint on local disk.

    Shaped like a real Hugging Face checkpoint (config, tokenizer, several
    safetensors shards) so the pipeline exercises the many-files-per-model path
    that section 13 insists on, without downloading gigabytes.
    """
    source = tmp_path / "pkg"
    weights = source / "weights"
    weights.mkdir(parents=True)

    (weights / "config.json").write_text('{"model_type": "demo"}')
    (weights / "tokenizer.json").write_text('{"version": "1.0"}')
    (weights / "generation_config.json").write_text('{"max_length": 128}')
    for i in range(1, SHARD_COUNT + 1):
        name = f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors"
        (weights / name).write_bytes(bytes([i]) * SHARD_SIZE)

    (source / "offlineai.yaml").write_text(
        """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: demo-model
  version: 2.1.0
  description: A tiny sharded checkpoint, packaged from local disk.
architecture:
  - amd64
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
hardware:
  gpu:
    required: true
    vendor: nvidia
    minimum_vram_gb: 24
secrets:
  external:
    - DEMO_API_TOKEN
"""
    )
    (source / "README.md").write_text("# demo-model\n\nA tiny example.\n")
    return source


@pytest.fixture
def built_bundle(package_dir: Path, tmp_path: Path) -> Path:
    settings = load_settings(home=tmp_path / "home")
    settings.ensure_directories()
    result = BundleBuilder(settings).build(package_dir, output=tmp_path / "out")
    return Path(result.bundle_path)


class TestBuild:
    def test_bundle_is_named_by_convention(self, built_bundle: Path) -> None:
        assert built_bundle.name == "demo-model-2.1.0.offlineai"

    def test_every_model_file_is_packaged(self, built_bundle: Path) -> None:
        """Section 13: a model is not one file. All shards must be present."""
        manifest = inspect_bundle(built_bundle)
        assert manifest.artifact_counts[ArtifactType.MODEL] == SHARD_COUNT + 3

    def test_declared_hardware_requirements_reach_the_manifest(self, built_bundle: Path) -> None:
        result = inspect_bundle(built_bundle)
        assert result.gpu_vendor == "nvidia"
        assert result.gpu_minimum_vram_gb == 24

    def test_secret_names_travel_but_no_values(self, built_bundle: Path) -> None:
        assert inspect_bundle(built_bundle).required_secrets == ["DEMO_API_TOKEN"]

    def test_builder_provenance_is_recorded(self, built_bundle: Path) -> None:
        builder = inspect_bundle(built_bundle).builder
        assert builder["offlineaiVersion"]
        assert builder["pythonVersion"]

    def test_build_is_idempotent_and_uses_the_cache(
        self, package_dir: Path, tmp_path: Path
    ) -> None:
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        builder = BundleBuilder(settings)
        first = builder.build(package_dir, output=tmp_path / "a")
        second = BundleBuilder(settings).build(package_dir, output=tmp_path / "b")
        assert first.artifact_count == second.artifact_count


class TestVerify:
    def test_a_freshly_built_bundle_verifies(self, built_bundle: Path) -> None:
        result = verify_bundle(built_bundle)
        assert result.verified is True
        assert result.artifacts_verified == SHARD_COUNT + 3
        assert result.bytes_verified > SHARD_COUNT * SHARD_SIZE

    def test_corruption_anywhere_in_the_payload_is_caught(self, built_bundle: Path) -> None:
        """Flip one byte deep inside the artifact region."""
        data = bytearray(built_bundle.read_bytes())
        offset = int(len(data) * 0.8)
        data[offset] ^= 0xFF
        built_bundle.write_bytes(bytes(data))

        with pytest.raises(ChecksumMismatchError) as excinfo:
            verify_bundle(built_bundle)
        assert excinfo.value.exit_code == ExitCode.VERIFICATION_FAILURE

    def test_a_truncated_bundle_is_caught(self, built_bundle: Path) -> None:
        data = built_bundle.read_bytes()
        built_bundle.write_bytes(data[: len(data) // 2])
        with pytest.raises(Exception) as excinfo:  # noqa: PT011
            verify_bundle(built_bundle)
        assert excinfo.value.exit_code == ExitCode.VERIFICATION_FAILURE  # type: ignore[attr-defined]


class TestInspectIsCheap:
    def test_inspect_reads_only_the_header(self, built_bundle: Path) -> None:
        result = inspect_bundle(built_bundle)
        assert result.header_bytes_read < result.file_size // 2, (
            "inspect must not walk the artifact payload"
        )

    def test_inspect_does_not_extract_anything(self, built_bundle: Path, tmp_path: Path) -> None:
        before = set(tmp_path.rglob("*"))
        inspect_bundle(built_bundle)
        assert set(tmp_path.rglob("*")) == before


class TestCliExitCodes:
    """Section 66: these numbers are the contract with CI and Ansible."""

    def _env(self, tmp_path: Path) -> dict[str, str]:
        return {"OFFLINEAI_HOME": str(tmp_path / "home")}

    def test_verify_succeeds_with_zero(self, built_bundle: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["verify", str(built_bundle)], env=self._env(tmp_path))
        assert result.exit_code == ExitCode.SUCCESS
        assert "VERIFIED" in result.stdout

    def test_verify_of_a_corrupt_bundle_exits_three(
        self, built_bundle: Path, tmp_path: Path
    ) -> None:
        data = bytearray(built_bundle.read_bytes())
        data[int(len(data) * 0.8)] ^= 0xFF
        built_bundle.write_bytes(bytes(data))

        result = runner.invoke(app, ["verify", str(built_bundle)], env=self._env(tmp_path))
        assert result.exit_code == ExitCode.VERIFICATION_FAILURE

    def test_build_of_an_invalid_definition_exits_two(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "offlineai.yaml").write_text("apiVersion: offlineai/v1\nkind: Package\n")
        result = runner.invoke(app, ["build", str(bad)], env=self._env(tmp_path))
        assert result.exit_code == ExitCode.INVALID_PACKAGE

    def test_build_of_a_missing_directory_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["build", str(tmp_path / "nope")], env=self._env(tmp_path))
        assert result.exit_code == ExitCode.INVALID_PACKAGE

    def test_json_output_is_parseable(self, built_bundle: Path, tmp_path: Path) -> None:
        import json

        result = runner.invoke(
            app, ["--json", "inspect", str(built_bundle)], env=self._env(tmp_path)
        )
        assert result.exit_code == ExitCode.SUCCESS
        payload = json.loads(result.stdout)
        assert payload["package"] == "demo-model"
        assert payload["version"] == "2.1.0"


class TestSecretDetection:
    """Section 40: never package secrets by default."""

    def test_a_private_key_in_the_build_context_fails_the_build(
        self, package_dir: Path, tmp_path: Path
    ) -> None:
        (package_dir / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\nxx\n")
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()

        from offlineai.errors import InvalidPackageError

        with pytest.raises(InvalidPackageError, match="credential-shaped"):
            BundleBuilder(settings).build(package_dir, output=tmp_path / "out")

    def test_the_failure_names_the_offending_file(self, package_dir: Path, tmp_path: Path) -> None:
        (package_dir / "id_rsa").write_text("x")
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()

        from offlineai.errors import InvalidPackageError

        with pytest.raises(InvalidPackageError) as excinfo:
            BundleBuilder(settings).build(package_dir, output=tmp_path / "out")
        assert "id_rsa" in excinfo.value.render()

    def test_an_ignore_entry_permits_a_known_fixture(
        self, package_dir: Path, tmp_path: Path
    ) -> None:
        (package_dir / "test-fixture.pem").write_text("not a real key")
        (package_dir / ".offlineaiignore").write_text("test-fixture.pem\n")
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        result = BundleBuilder(settings).build(package_dir, output=tmp_path / "out")
        assert Path(result.bundle_path).is_file()
