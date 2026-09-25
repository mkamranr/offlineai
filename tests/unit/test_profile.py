"""Section 5.6: target environment profiles.

`offlineai check` validates a bundle against the machine it runs on. In this
tool's workflow that is the builder - the wrong machine, since the target is
air-gapped and elsewhere. A profile describes the target so the question can
be asked from anywhere.

The rule that makes this safe rather than merely convenient: a field the
profile omits must be SKIPPED, never PASS. Inventing a default would let a
profile silently approve a bundle for hardware it was never checked against,
which is worse than having no profile at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from offlineai.bundler.results import CheckStatus
from offlineai.hardware.compat import check_compatibility
from offlineai.hardware.detector import HardwareReport
from offlineai.schema.manifest import FORMAT_VERSION, Manifest
from offlineai.schema.profile import Profile, load_profile

#: Copied verbatim from section 5.6 of the requirements.
SPEC_EXAMPLE = """\
name: h100-server

os:
  family: ubuntu
  version: "24.04"

architecture: amd64

gpu:
  vendor: nvidia
  minimum_driver: "550"
  memory_gb: 80

runtime:
  docker: ">=27"
"""


def manifest(**requirements: object) -> Manifest:
    return Manifest.model_validate(
        {
            "formatVersion": FORMAT_VERSION,
            "package": {"name": "demo", "version": "1.0.0"},
            "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
            "platforms": ["linux/amd64"],
            "requirements": requirements,
        }
    )


def status_of(report: object, name: str) -> CheckStatus | None:
    return next((c.status for c in report.checks if c.name == name), None)  # type: ignore[attr-defined]


class TestTheSpecExampleParses:
    def test_it_loads(self) -> None:
        profile = Profile.from_yaml(SPEC_EXAMPLE)
        assert profile.name == "h100-server"

    def test_every_field_is_read(self) -> None:
        profile = Profile.from_yaml(SPEC_EXAMPLE)
        assert profile.architecture == "amd64"
        assert profile.os is not None and profile.os.family == "ubuntu"
        assert profile.gpu is not None
        assert profile.gpu.memory_gb == 80
        assert profile.gpu.minimum_driver == "550"
        assert profile.runtime is not None and profile.runtime.docker == ">=27"

    def test_it_round_trips(self) -> None:
        profile = Profile.from_yaml(SPEC_EXAMPLE)
        assert Profile.from_yaml(profile.to_yaml()) == profile


class TestConversionToAHardwareReport:
    """A profile only has to produce a HardwareReport; the whole comparison
    engine in compat.py is then reused unchanged."""

    def test_it_produces_a_report(self) -> None:
        report = Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report()
        assert isinstance(report, HardwareReport)
        assert report.architecture == "amd64"

    def test_a_distribution_is_normalised_to_its_platform(self) -> None:
        """A profile says `family: ubuntu`; the detector says `linux`. Without
        normalising, every Linux target would be reported as an unsupported
        platform. The distribution is kept in the version string rather than
        discarded."""
        report = Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report()
        assert report.os_name == "linux"
        assert report.is_linux
        assert "ubuntu" in report.os_version
        assert "24.04" in report.os_version

    def test_the_report_says_it_describes_a_profile(self) -> None:
        """So "could not be determined on this host" cannot appear when no
        host was examined."""
        report = Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report()
        assert report.describes == "profile 'h100-server'"

    @pytest.mark.parametrize(
        "family", ["ubuntu", "debian", "rhel", "rocky", "fedora", "amazonlinux"]
    )
    def test_known_linux_families_are_recognised(self, family: str) -> None:
        report = Profile.from_yaml(
            f"name: d\narchitecture: amd64\nos:\n  family: {family}\n"
        ).to_hardware_report()
        assert report.is_linux, f"{family} should be recognised as Linux"

    def test_a_non_linux_family_is_left_alone(self) -> None:
        report = Profile.from_yaml(
            "name: d\narchitecture: arm64\nos:\n  family: darwin\n"
        ).to_hardware_report()
        assert report.os_name == "darwin"
        assert not report.is_linux

    def test_gpu_count_defaults_to_one_described_device(self) -> None:
        report = Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report()
        assert len(report.gpus) == 1
        assert report.gpus[0].advertised_gb == 80
        assert report.nvidia_driver == "550"

    def test_several_gpus_are_modelled(self) -> None:
        # Built as a separate document rather than appended to the spec
        # example, where a trailing key would land under `runtime:`.
        profile = Profile.from_yaml(
            "name: quad\narchitecture: amd64\ngpu:\n  vendor: nvidia\n  memory_gb: 80\n  count: 4\n"
        )
        assert len(profile.to_hardware_report().gpus) == 4

    def test_the_runtime_constraint_becomes_an_availability(self) -> None:
        availability = Profile.from_yaml(SPEC_EXAMPLE).to_runtime_availability()
        assert availability.available is True
        assert availability.version == "27"

    def test_a_gpu_profile_reports_gpu_support(self) -> None:
        assert Profile.from_yaml(SPEC_EXAMPLE).to_runtime_availability().gpu_support


class TestOmittedFieldsAreSkippedNeverPassed:
    """The rule that makes profiles safe."""

    MINIMAL = "name: bare-metal\narchitecture: amd64\n"

    def test_unstated_ram_does_not_satisfy_a_ram_requirement(self) -> None:
        profile = Profile.from_yaml(self.MINIMAL)
        report = check_compatibility(manifest(minimumRamGB=512), profile.to_hardware_report())
        assert status_of(report, "RAM") is CheckStatus.SKIPPED, (
            "a profile silent about RAM must not approve a 512 GB requirement"
        )

    def test_the_skip_says_why(self) -> None:
        profile = Profile.from_yaml(self.MINIMAL)
        report = check_compatibility(manifest(minimumRamGB=512), profile.to_hardware_report())
        entry = next(c for c in report.checks if c.name == "RAM")
        assert entry.detail and "profile 'bare-metal'" in entry.detail
        assert "this host" not in entry.detail, "no host was examined; saying so would mislead"

    def test_memory_is_none_when_unstated(self) -> None:
        assert Profile.from_yaml(self.MINIMAL).to_hardware_report().memory is None

    def test_disk_is_always_none(self) -> None:
        """A profile describes a class of machine, not its current free space."""
        assert Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report().disk is None

    def test_an_unstated_gpu_fails_a_gpu_requirement(self) -> None:
        """Not skipped: a profile that does not describe a GPU is describing a
        machine without one, and a bundle needing one will not run there."""
        profile = Profile.from_yaml(self.MINIMAL)
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48}),
            profile.to_hardware_report(),
        )
        assert status_of(report, "GPU") is CheckStatus.FAILED

    def test_an_unstated_runtime_is_not_claimed_available(self) -> None:
        availability = Profile.from_yaml(self.MINIMAL).to_runtime_availability()
        assert availability.version is None


class TestTheComparisonEngineIsReused:
    """No changes to compat.py: the existing rules apply unchanged."""

    def test_an_h100_profile_satisfies_a_48gb_requirement(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48, "minimumDriver": "550"}),
            Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report(),
        )
        assert report.compatible

    def test_it_fails_a_requirement_the_profile_cannot_meet(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 640}),
            Profile.from_yaml(SPEC_EXAMPLE).to_hardware_report(),
        )
        assert not report.compatible
        assert status_of(report, "GPU VRAM") is CheckStatus.FAILED

    def test_vram_is_still_compared_per_device(self) -> None:
        """Four 24 GB cards do not satisfy a 48 GB requirement, exactly as for
        a live host."""
        profile = Profile.from_yaml(
            "name: small\narchitecture: amd64\n"
            "gpu:\n  vendor: nvidia\n  memory_gb: 24\n  count: 4\n"
        )
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48}),
            profile.to_hardware_report(),
        )
        assert status_of(report, "GPU VRAM") is CheckStatus.FAILED

    def test_an_old_driver_in_the_profile_fails(self) -> None:
        profile = Profile.from_yaml(
            "name: old\narchitecture: amd64\n"
            "gpu:\n  vendor: nvidia\n  memory_gb: 80\n  minimum_driver: '470'\n"
        )
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumDriver": "550"}),
            profile.to_hardware_report(),
        )
        assert status_of(report, "GPU driver") is CheckStatus.FAILED

    def test_a_mismatched_architecture_fails(self) -> None:
        profile = Profile.from_yaml("name: arm\narchitecture: arm64\n")
        report = check_compatibility(manifest(), profile.to_hardware_report())
        assert status_of(report, "CPU architecture") is CheckStatus.FAILED

    def test_a_non_linux_profile_warns_like_a_non_linux_host(self) -> None:
        profile = Profile.from_yaml("name: mac\narchitecture: arm64\nos:\n  family: darwin\n")
        report = check_compatibility(manifest(), profile.to_hardware_report())
        assert status_of(report, "Operating system") is CheckStatus.WARNING

    def test_an_older_declared_runtime_warns_rather_than_failing(self) -> None:
        profile = Profile.from_yaml(
            "name: old-docker\narchitecture: amd64\nruntime:\n  docker: '24.0.6'\n"
        )
        report = check_compatibility(
            manifest(docker={"minimumVersion": "27"}),
            profile.to_hardware_report(),
            runtime=profile.to_runtime_availability(),
        )
        assert status_of(report, "Container runtime") is CheckStatus.WARNING


class TestValidation:
    def test_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Profile.from_yaml("name: x\narchitecure: amd64\n")

    def test_an_unknown_nested_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Profile.from_yaml("name: x\ngpu:\n  vendr: nvidia\n")

    @pytest.mark.parametrize("architecture", ["x86", "i386", "sparc"])
    def test_an_unsupported_architecture_is_rejected(self, architecture: str) -> None:
        with pytest.raises(ValidationError):
            Profile.from_yaml(f"name: x\narchitecture: {architecture}\n")

    def test_negative_vram_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Profile.from_yaml("name: x\ngpu:\n  memory_gb: -1\n")

    def test_a_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Profile.from_yaml("architecture: amd64\n")


class TestLoading:
    def test_it_reads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "h100.yaml"
        path.write_text(SPEC_EXAMPLE)
        assert load_profile(path).name == "h100-server"

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        from offlineai.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="does not exist"):
            load_profile(tmp_path / "nope.yaml")

    def test_malformed_yaml_is_reported(self, tmp_path: Path) -> None:
        from offlineai.errors import ConfigurationError

        path = tmp_path / "bad.yaml"
        path.write_text("name: [unclosed\n")
        with pytest.raises(ConfigurationError):
            load_profile(path)

    def test_an_invalid_profile_names_the_problem(self, tmp_path: Path) -> None:
        from offlineai.errors import ConfigurationError

        path = tmp_path / "bad.yaml"
        path.write_text("name: x\narchitecture: sparc\n")
        with pytest.raises(ConfigurationError) as excinfo:
            load_profile(path)
        assert "architecture" in excinfo.value.render()


class TestCapturingTheCurrentHost:
    """doctor --save-profile turns a profile from a hand-written guess into
    measured ground truth."""

    def test_a_report_becomes_a_profile(self) -> None:
        from offlineai.hardware.detector import HardwareDetector

        report = HardwareDetector().detect()
        profile = Profile.from_hardware_report(report, name="this-host")
        assert profile.name == "this-host"
        assert profile.architecture == report.architecture

    def test_the_captured_profile_round_trips(self) -> None:
        from offlineai.hardware.detector import HardwareDetector

        report = HardwareDetector().detect()
        profile = Profile.from_hardware_report(report, name="this-host")
        assert Profile.from_yaml(profile.to_yaml()) == profile

    def test_checking_against_a_captured_profile_agrees_with_the_live_host(self) -> None:
        """The loop that makes the feature trustworthy: capture on the target,
        validate on the builder, get the same answer."""
        from offlineai.hardware.detector import HardwareDetector

        live = HardwareDetector().detect()
        captured = Profile.from_hardware_report(live, name="captured").to_hardware_report()

        spec = manifest(minimumRamGB=1, minimumCpuCores=1)
        assert (
            check_compatibility(spec, live).compatible
            == check_compatibility(spec, captured).compatible
        )
