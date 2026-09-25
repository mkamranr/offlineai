"""A build must not succeed while omitting something the package declared.

Section 74: *a successful build must mean the resulting bundle contains
everything required for the declared offline installation.*

`system.packages` is parsed by the schema but section 19 is not implemented.
Until it is, declaring it produced a bundle that verified, reported success,
and silently lacked the packages - which the operator would discover on the
air-gapped side, where they cannot fix it. Warning was not enough; a bundle
that claims completeness has to be complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offlineai.bundler.builder import BundleBuilder
from offlineai.config.settings import Settings, load_settings
from offlineai.errors import MissingArtifactError
from offlineai.exitcodes import ExitCode


def package_with(body: str, tmp_path: Path) -> Path:
    source = tmp_path / "pkg"
    source.mkdir(parents=True, exist_ok=True)
    (source / "offlineai.yaml").write_text(
        "apiVersion: offlineai/v1\nkind: Package\n"
        "metadata:\n  name: demo\n  version: 1.0.0\n" + body
    )
    return source


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    configured = load_settings(home=tmp_path / "home")
    configured.ensure_directories()
    return configured


class TestSystemPackagesAreRefused:
    def test_declaring_them_fails_the_build(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with("system:\n  packages:\n    - curl\n", tmp_path)
        with pytest.raises(MissingArtifactError, match="system package"):
            BundleBuilder(settings).build(source, output=tmp_path / "out")

    def test_no_bundle_is_left_behind(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with("system:\n  packages:\n    - curl\n", tmp_path)
        with pytest.raises(MissingArtifactError):
            BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert not list((tmp_path / "out").glob("*.offlineai"))

    def test_it_exits_five(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with("system:\n  packages:\n    - curl\n", tmp_path)
        with pytest.raises(MissingArtifactError) as excinfo:
            BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert excinfo.value.exit_code == ExitCode.MISSING_DEPENDENCY

    def test_the_error_names_the_packages(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with("system:\n  packages:\n    - curl\n    - ca-certificates\n", tmp_path)
        with pytest.raises(MissingArtifactError) as excinfo:
            BundleBuilder(settings).build(source, output=tmp_path / "out")
        rendered = excinfo.value.render()
        assert "curl" in rendered and "ca-certificates" in rendered

    def test_the_error_says_what_to_do_instead(self, settings: Settings, tmp_path: Path) -> None:
        """For a containerised workload the right home for an OS dependency is
        the image, and the message should say so rather than just refusing."""
        source = package_with("system:\n  packages:\n    - curl\n", tmp_path)
        with pytest.raises(MissingArtifactError) as excinfo:
            BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert "Dockerfile" in excinfo.value.render()

    def test_dry_run_reports_it_too(self, settings: Settings, tmp_path: Path) -> None:
        """--dry-run answers "would this build?", so it must not say yes."""
        source = package_with("system:\n  packages:\n    - curl\n", tmp_path)
        with pytest.raises(MissingArtifactError):
            BundleBuilder(settings).build(source, dry_run=True)

    @pytest.mark.parametrize("tier", ["optional_packages", "recommended_packages"])
    def test_optional_tiers_are_permitted(
        self, settings: Settings, tmp_path: Path, tier: str
    ) -> None:
        """Section 19 distinguishes required from optional and recommended.
        Only the required tier breaks the completeness promise."""
        source = package_with(f"system:\n  {tier}:\n    - vim\n", tmp_path)
        result = BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert Path(result.bundle_path).is_file()

    def test_an_empty_system_block_is_fine(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with("system:\n  packages: []\n", tmp_path)
        result = BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert Path(result.bundle_path).is_file()

    def test_a_package_without_a_system_block_is_unaffected(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_with("", tmp_path)
        result = BundleBuilder(settings).build(source, output=tmp_path / "out")
        assert Path(result.bundle_path).is_file()


class TestTheSchemaStillAcceptsTheField:
    """Rejecting it in the schema would be a breaking format change, and
    section 19 is a real requirement that will be implemented. The field stays
    valid; what changes is that a build cannot quietly ignore it."""

    def test_the_field_still_parses(self) -> None:
        from offlineai.schema.package import Package

        package = Package.model_validate(
            {
                "apiVersion": "offlineai/v1",
                "kind": "Package",
                "metadata": {"name": "demo", "version": "1.0.0"},
                "system": {"packages": ["curl"]},
            }
        )
        assert package.system is not None
        assert package.system.packages == ["curl"]


class TestInstallAgreesWithBuildAboutSystemPackages:
    """The build and the install must not tell different stories.

    They did: a package declaring optional packages got
    "3 optional/recommended package(s) are not packaged" from the build and
    "none declared" from the install, because the installer only looked at the
    required tier - which the builder now refuses outright. Its message also
    still said "arrives in a later phase" where the builder says "not
    implemented in this release; put it in the Dockerfile".
    """

    def _install(self, source: Path, settings: Settings, tmp_path: Path) -> list:
        from offlineai.bundler.builder import BundleBuilder
        from offlineai.installer.installer import Installer
        from offlineai.registry.registry import Registry
        from offlineai.runtime.fake import FakeRuntime

        built = BundleBuilder(settings, runtime=FakeRuntime()).build(
            source, output=tmp_path / "dist"
        )
        registry = Registry(settings)
        registry.import_bundle(built.bundle_path)
        result = Installer(settings, registry, FakeRuntime()).install(
            source.name if source.name != "pkg" else "demo",
            start=False,
            health_sleep=0,
        )
        return [c for c in result.checks if c.name == "System packages"]

    def test_optional_packages_are_reported_not_called_none(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_with("system:\n  optional_packages:\n    - vim\n    - htop\n", tmp_path)
        checks = self._install(source, settings, tmp_path)
        assert checks, "the installer said nothing about system packages"
        detail = checks[0].detail or ""
        assert "none declared" not in detail, (
            f"install reported {detail!r} for a package that declares two optional "
            "packages; the build reported them"
        )
        assert "2" in detail

    def test_recommended_packages_are_counted_too(self, settings: Settings, tmp_path: Path) -> None:
        source = package_with(
            "system:\n  optional_packages:\n    - vim\n  recommended_packages:\n    - curl\n",
            tmp_path,
        )
        detail = (self._install(source, settings, tmp_path)[0].detail) or ""
        assert "3" in detail or "2" in detail

    def test_a_package_with_no_system_block_still_says_none_declared(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_with("", tmp_path)
        assert "none declared" in (self._install(source, settings, tmp_path)[0].detail or "")

    def test_the_advisory_tiers_do_not_fail_the_install(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        from offlineai.bundler.builder import BundleBuilder
        from offlineai.installer.installer import Installer
        from offlineai.installer.transaction import InstallState
        from offlineai.registry.registry import Registry
        from offlineai.runtime.fake import FakeRuntime

        source = package_with("system:\n  optional_packages:\n    - vim\n", tmp_path)
        built = BundleBuilder(settings, runtime=FakeRuntime()).build(
            source, output=tmp_path / "dist"
        )
        registry = Registry(settings)
        registry.import_bundle(built.bundle_path)
        result = Installer(settings, registry, FakeRuntime()).install(
            "demo", start=False, health_sleep=0
        )
        assert result.state is InstallState.COMPLETED

    def test_the_message_matches_what_the_builder_says(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """No "later phase" anywhere: the builder tells operators to put the
        dependency in the image, and the installer must not contradict it."""
        source = package_with("system:\n  optional_packages:\n    - vim\n", tmp_path)
        detail = (self._install(source, settings, tmp_path)[0].detail or "").lower()
        assert "later phase" not in detail
