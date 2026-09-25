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
