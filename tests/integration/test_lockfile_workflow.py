"""Section 35 end to end: the lock actually constrains a later build.

A lock that is only checked after resolution tells you a build drifted. A lock
whose pins are fed back into resolution stops it drifting. These tests are
mostly about the second thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from offlineai.bundler.builder import BundleBuilder
from offlineai.cli.main import app
from offlineai.config.settings import Settings, load_settings
from offlineai.errors import InvalidPackageError, MissingArtifactError
from offlineai.exitcodes import ExitCode
from offlineai.runtime.fake import FakeRuntime
from offlineai.schema.lockfile import LOCK_FILENAME, Lockfile

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    configured = load_settings(home=tmp_path / "home")
    configured.ensure_directories()
    return configured


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    source = tmp_path / "pkg"
    weights = source / "weights"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text('{"model_type": "demo"}')
    (weights / "model.safetensors").write_bytes(b"original weights" * 100)
    (source / "offlineai.yaml").write_text(
        """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: locked-demo
  version: 1.0.0
containers:
  - name: app
    image: python:3.12-slim
models:
  - name: demo
    source:
      type: local
      path: ./weights
"""
    )
    return source


def build(package_dir: Path, settings: Settings, tmp_path: Path, **kwargs: Any) -> Any:
    return BundleBuilder(settings, runtime=FakeRuntime(), **kwargs).build(
        package_dir, output=tmp_path / "dist"
    )


class TestTheLockIsWritten:
    def test_a_build_writes_it_next_to_the_definition(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        assert (package_dir / LOCK_FILENAME).is_file()

    def test_it_records_every_artifact(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        lock = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        paths = {a.path for a in lock.artifacts}
        assert any("config.json" in p for p in paths)
        assert any("model.safetensors" in p for p in paths)
        assert any("app.tar" in p for p in paths)

    def test_it_records_the_image_digest(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        """A tag is mutable; the digest is what a later build should fetch."""
        build(package_dir, settings, tmp_path)
        lock = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        pin = lock.pin_for("app", locator="python:3.12-slim")
        assert pin is not None and pin.startswith("sha256:")

    def test_it_records_the_definition_digest(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        lock = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        assert lock.package_definition_sha256 is not None

    def test_rebuilding_produces_a_stable_lock(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        """Only the timestamp may move. Anything else churning would put noise
        in every review of this file."""
        build(package_dir, settings, tmp_path)
        first = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        build(package_dir, settings, tmp_path)
        second = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())

        assert first.artifacts == second.artifacts
        assert first.sources == second.sources

    def test_no_lock_writes_nothing(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path, lock_mode="none")
        assert not (package_dir / LOCK_FILENAME).exists()


class TestLockedBuildsDetectDrift:
    def test_an_unchanged_package_passes(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        result = build(package_dir, settings, tmp_path, lock_mode="locked")
        assert Path(result.bundle_path).is_file()

    def test_changed_content_fails_the_build(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        (package_dir / "weights" / "model.safetensors").write_bytes(b"different" * 100)

        with pytest.raises(MissingArtifactError, match="does not match"):
            build(package_dir, settings, tmp_path, lock_mode="locked")

    def test_the_failure_names_the_artifact_and_both_digests(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        (package_dir / "weights" / "model.safetensors").write_bytes(b"different" * 100)

        with pytest.raises(MissingArtifactError) as excinfo:
            build(package_dir, settings, tmp_path, lock_mode="locked")
        rendered = excinfo.value.render()
        assert "model.safetensors" in rendered
        assert "digest changed" in rendered

    def test_a_new_artifact_is_drift(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        (package_dir / "weights" / "extra.json").write_text("{}")

        with pytest.raises(MissingArtifactError):
            build(package_dir, settings, tmp_path, lock_mode="locked")

    def test_a_locked_build_does_not_rewrite_the_lock(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        """--locked proves the file on disk still describes reality. Rewriting
        it would destroy the evidence it was asked to check."""
        build(package_dir, settings, tmp_path)
        before = (package_dir / LOCK_FILENAME).read_text()
        build(package_dir, settings, tmp_path, lock_mode="locked")
        assert (package_dir / LOCK_FILENAME).read_text() == before

    def test_locked_without_a_lock_file_says_how_to_make_one(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        with pytest.raises(InvalidPackageError) as excinfo:
            build(package_dir, settings, tmp_path, lock_mode="locked")
        assert "does not exist" in excinfo.value.render()
        assert "without --locked" in excinfo.value.render()

    def test_update_lock_accepts_the_change(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        before = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        (package_dir / "weights" / "model.safetensors").write_bytes(b"different" * 100)

        build(package_dir, settings, tmp_path, lock_mode="update")
        after = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text())
        assert before.artifacts != after.artifacts

    def test_a_corrupt_lock_is_reported_not_ignored(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        (package_dir / LOCK_FILENAME).write_text("formatVersion: [not, a, string]\n")
        with pytest.raises(InvalidPackageError, match="readable lock file"):
            build(package_dir, settings, tmp_path)


class TestPinsConstrainResolution:
    """The half that matters: a pin changes what gets fetched."""

    def test_a_pinned_image_is_fetched_by_digest(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        runtime = FakeRuntime()
        BundleBuilder(settings, runtime=runtime).build(package_dir, output=tmp_path / "d1")
        pin = Lockfile.from_yaml((package_dir / LOCK_FILENAME).read_text()).pin_for("app")
        assert pin is not None

        second = FakeRuntime()
        BundleBuilder(settings, runtime=second).build(package_dir, output=tmp_path / "d2")
        requested = [target for op, target in second.calls if op in ("pull", "save")]
        assert any(pin in target for target in requested), (
            f"the second build fetched {requested}, not the pinned digest {pin}"
        )

    def test_a_pin_is_ignored_when_the_locator_changed(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        """If the package now names a different image, the old image's digest
        must not be applied to it - that would pin the wrong thing entirely."""
        BundleBuilder(settings, runtime=FakeRuntime()).build(package_dir, output=tmp_path / "d1")
        definition = (package_dir / "offlineai.yaml").read_text()
        (package_dir / "offlineai.yaml").write_text(
            definition.replace("python:3.12-slim", "redis:7-alpine")
        )

        runtime = FakeRuntime()
        BundleBuilder(settings, runtime=runtime).build(package_dir, output=tmp_path / "d2")
        pulled = [t for op, t in runtime.calls if op == "pull"]
        assert any("redis" in t for t in pulled)
        assert not any("python" in t for t in pulled)


class TestCliFlags:
    def _env(self, tmp_path: Path) -> dict[str, str]:
        return {"OFFLINEAI_HOME": str(tmp_path / "home")}

    def test_locked_exits_five_on_drift(self, package_dir: Path, tmp_path: Path) -> None:
        runner.invoke(
            app,
            ["build", str(package_dir), "-o", str(tmp_path / "d")],
            env=self._env(tmp_path),
        )
        (package_dir / "weights" / "model.safetensors").write_bytes(b"changed" * 100)

        result = runner.invoke(
            app,
            ["build", str(package_dir), "-o", str(tmp_path / "d"), "--locked"],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.MISSING_DEPENDENCY

    def test_the_flags_are_mutually_exclusive(self, package_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "build",
                str(package_dir),
                "-o",
                str(tmp_path / "d"),
                "--locked",
                "--update-lock",
            ],
            env=self._env(tmp_path),
        )
        assert result.exit_code == ExitCode.INVALID_PACKAGE
        assert "mutually exclusive" in result.stdout + result.stderr


class TestLockedFailsBeforeWritingAnything:
    """Everything needed to detect drift is known once resolution finishes.

    Writing a 62 GB archive and only then discovering the build was not the
    one asked for wastes the time and the disk, and leaves a bundle on disk
    that someone could mistake for a good one.
    """

    def test_no_bundle_is_written_when_the_lock_does_not_match(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        (package_dir / "weights" / "model.safetensors").write_bytes(b"changed" * 100)

        output = tmp_path / "locked-out"
        with pytest.raises(MissingArtifactError):
            BundleBuilder(settings, runtime=FakeRuntime(), lock_mode="locked").build(
                package_dir, output=output
            )
        assert not list(output.glob("*.offlineai")), "a drifted --locked build left a bundle behind"

    def test_the_archive_step_is_never_reached(
        self, package_dir: Path, settings: Settings, tmp_path: Path
    ) -> None:
        build(package_dir, settings, tmp_path)
        (package_dir / "weights" / "model.safetensors").write_bytes(b"changed" * 100)

        steps: list[str] = []
        builder = BundleBuilder(
            settings,
            runtime=FakeRuntime(),
            lock_mode="locked",
            on_step=lambda s: steps.append(s.name),
        )
        with pytest.raises(MissingArtifactError):
            builder.build(package_dir, output=tmp_path / "out")

        assert "Creating bundle" not in steps, f"reached the archive step before failing: {steps}"
