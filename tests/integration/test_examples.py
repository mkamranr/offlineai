"""The shipped examples must be valid, and the small ones must actually build.

A broken example is worse than no example: it is the first thing someone
copies. The heavy ones (whisper, qwen-vllm, rag-stack) cannot be built in CI -
they are tens of gigabytes - so they are validated through --dry-run, which
checks the definition and the files it references without fetching anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offlineai.bundler.builder import BundleBuilder
from offlineai.bundler.verifier import verify_bundle
from offlineai.config.settings import load_settings
from offlineai.resolver.package import load_package
from offlineai.runtime.fake import FakeRuntime

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

ALL_EXAMPLES = ["hello-ai", "simple-python", "whisper", "qwen-vllm", "rag-stack"]
#: Small enough to build with no network and no real images.
BUILDABLE = ["simple-python"]


class TestAllExamplesAreValid:
    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_the_definition_parses_and_validates(self, example: str) -> None:
        package, path, digest = load_package(EXAMPLES / example)
        assert package.metadata.name == example
        assert path.name == "offlineai.yaml"
        assert len(digest) == 64

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_dry_run_accepts_it(self, example: str, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        result = BundleBuilder(settings).build(EXAMPLES / example, dry_run=True)
        assert result.package == example

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_every_referenced_file_exists(self, example: str) -> None:
        """A dockerfile or requirements path that does not exist would fail on
        the builder, but only after a long download."""
        package, path, _ = load_package(EXAMPLES / example)
        base = path.parent
        for container in package.containers:
            if container.dockerfile:
                assert (base / container.dockerfile).is_file(), (
                    f"{example}: {container.dockerfile} is referenced but missing"
                )
        for requirements in package.python.requirements if package.python else []:
            assert (base / requirements).is_file(), (
                f"{example}: {requirements} is referenced but missing"
            )
        for model in package.models:
            if model.source.type == "local":
                assert (base / model.source.path).exists(), (
                    f"{example}: local model path {model.source.path} is missing"
                )

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_it_has_a_readme(self, example: str) -> None:
        assert (EXAMPLES / example / "README.md").is_file()


class TestExamplesFollowTheirOwnAdvice:
    """The examples are the reference for how to write a package. If they get
    the offline discipline wrong, so will everyone copying them."""

    @pytest.mark.parametrize("example", ["whisper", "qwen-vllm", "rag-stack"])
    def test_model_packages_disable_runtime_model_fetching(self, example: str) -> None:
        """Without these, a server that cannot reach huggingface.co hangs
        rather than failing, which is much harder to diagnose."""
        package, _, _ = load_package(EXAMPLES / example)
        assert package.environment.get("HF_HUB_OFFLINE") == "1", (
            f"{example} packages a model but does not set HF_HUB_OFFLINE"
        )
        assert package.environment.get("TRANSFORMERS_OFFLINE") == "1"

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_no_secret_values_are_present(self, example: str) -> None:
        """Section 40: names only, never values."""
        package, _, _ = load_package(EXAMPLES / example)
        for name in package.secrets.external:
            assert isinstance(name, str)
            assert "=" not in name and ":" not in name

    @pytest.mark.parametrize("example", ["qwen-vllm", "rag-stack"])
    def test_gpu_workloads_state_a_vram_minimum(self, example: str) -> None:
        """A GPU requirement with no number cannot be checked, so `check` would
        have nothing useful to say."""
        package, _, _ = load_package(EXAMPLES / example)
        assert package.hardware.gpu is not None
        assert package.hardware.gpu.minimum_vram_gb, (
            f"{example} requires a GPU but states no VRAM minimum"
        )

    @pytest.mark.parametrize("example", ALL_EXAMPLES)
    def test_every_service_maps_to_a_declared_container(self, example: str) -> None:
        package, _, _ = load_package(EXAMPLES / example)
        declared = {c.name for c in package.containers}
        for service in package.services:
            assert service.container in declared


class TestSmallExamplesActuallyBuild:
    @pytest.mark.parametrize("example", BUILDABLE)
    def test_builds_and_verifies(self, example: str, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        # lock_mode="none": the builder writes offlineai.lock beside the
        # definition, and these build the real examples in place. Without this
        # the suite rewrites a tracked file on every run.
        result = BundleBuilder(settings, runtime=FakeRuntime(), lock_mode="none").build(
            EXAMPLES / example, output=tmp_path / "dist"
        )
        assert Path(result.bundle_path).is_file()
        assert verify_bundle(result.bundle_path).verified

    @pytest.mark.parametrize("example", BUILDABLE)
    def test_the_build_reports_no_silent_skips(self, example: str, tmp_path: Path) -> None:
        """A step reporting SKIPPED means something the package declared was
        not packaged, which is the failure mode section 74 forbids."""
        from offlineai.bundler.results import CheckStatus

        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        result = BundleBuilder(settings, runtime=FakeRuntime(), lock_mode="none").build(
            EXAMPLES / example, output=tmp_path / "dist"
        )
        skipped = [s.name for s in result.steps if s.status is CheckStatus.SKIPPED]
        assert skipped == [], f"{example} skipped: {skipped}"


class TestTheSuiteDoesNotMutateTheRepository:
    """A test that writes into the source tree is a bug.

    The builder writes `offlineai.lock` beside the definition it built, and
    these tests build the real examples in place — so an example's lock was
    being rewritten on every run, and one was committed by accident through a
    `git add -A`. Build artifacts do not belong in `examples/`.
    """

    def test_no_example_lock_files_are_tracked(self) -> None:
        import subprocess

        tracked = subprocess.run(  # noqa: S603
            ["git", "ls-files", "examples/*/offlineai.lock"],
            capture_output=True,
            text=True,
            cwd=EXAMPLES.parent,
            check=False,
        ).stdout.strip()
        assert not tracked, (
            "lock files under examples/ are build artifacts and churn on every "
            f"test run; these are tracked:\n{tracked}"
        )

    @pytest.mark.parametrize("example", BUILDABLE)
    def test_building_an_example_leaves_no_lock_behind(self, example: str, tmp_path: Path) -> None:
        from offlineai.schema.lockfile import LOCK_FILENAME

        lock = EXAMPLES / example / LOCK_FILENAME
        existed = lock.exists()
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        BundleBuilder(settings, runtime=FakeRuntime(), lock_mode="none").build(
            EXAMPLES / example, output=tmp_path / "dist"
        )
        assert lock.exists() == existed, (
            f"building {example} wrote {LOCK_FILENAME} into the source tree"
        )
