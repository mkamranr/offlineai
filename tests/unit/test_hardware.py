"""Sections 20, 21 and 22: hardware detection and compatibility.

The rule that shapes most of these tests is from section 21: a requirement
that could not be evaluated must be SKIPPED, never PASS. "We could not look"
and "it is fine" are different answers and have to stay different.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from offlineai.bundler.results import CheckStatus
from offlineai.hardware.compat import (
    COMPATIBILITY_CAVEAT,
    check_compatibility,
    estimate_storage,
)
from offlineai.hardware.detector import (
    CpuInfo,
    DiskInfo,
    GpuDevice,
    HardwareDetector,
    HardwareReport,
    MemoryInfo,
    parse_driver_version,
)
from offlineai.runtime.base import RuntimeAvailability
from offlineai.schema.manifest import FORMAT_VERSION, Manifest
from offlineai.utils.proc import CommandResult

# Real nvidia-smi output, 2x H100 with the flags the detector passes.
NVIDIA_SMI_2XH100 = (
    "0, NVIDIA H100 80GB HBM3, 81559, 550.54.15, 9.0, GPU-abc123\n"
    "1, NVIDIA H100 80GB HBM3, 81559, 550.54.15, 9.0, GPU-def456\n"
)
NVIDIA_SMI_OLD_CARD = "0, Tesla T4, 15360, 470.82.01, 7.5, GPU-xyz\n"
# Some drivers cannot report compute capability.
NVIDIA_SMI_PARTIAL = "0, NVIDIA A100-SXM4-40GB, 40960, 525.85.12, [N/A], [N/A]\n"


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


def host(
    *,
    os_name: str = "linux",
    architecture: str = "amd64",
    cores: int = 64,
    ram_gb: float = 512,
    gpus: list[GpuDevice] | None = None,
    gpu_detail: str | None = None,
    driver: str | None = None,
) -> HardwareReport:
    return HardwareReport(
        os_name=os_name,
        os_version="6.8.0",
        architecture=architecture,
        cpu=CpuInfo(architecture=architecture, cores=cores),
        memory=MemoryInfo(total_bytes=int(ram_gb * 1000**3)),
        disk=DiskInfo(path="/var/lib/offlineai", total_bytes=4 * 1000**4, free_bytes=3 * 1000**4),
        gpus=gpus or [],
        gpu_detail=gpu_detail,
        nvidia_driver=driver,
    )


def h100(index: int = 0) -> GpuDevice:
    return GpuDevice(
        index=index,
        name="NVIDIA H100 80GB HBM3",
        memory_total_mb=81559,
        driver_version="550.54.15",
        compute_capability="9.0",
    )


def status_of(report: object, name: str) -> CheckStatus | None:
    return next((c.status for c in report.checks if c.name == name), None)  # type: ignore[attr-defined]


class TestNvidiaSmiParsing:
    def _detect(self, monkeypatch: pytest.MonkeyPatch, output: str, *, present: bool = True):
        monkeypatch.setattr("offlineai.hardware.detector.have", lambda _: present)
        monkeypatch.setattr(
            "offlineai.hardware.detector.run",
            lambda args, **_: CommandResult(tuple(args), 0, output, ""),
        )
        return HardwareDetector().detect_gpus()

    def test_parses_multiple_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus, detail, driver = self._detect(monkeypatch, NVIDIA_SMI_2XH100)
        assert len(gpus) == 2
        assert detail is None
        assert driver == "550.54.15"
        assert gpus[0].name == "NVIDIA H100 80GB HBM3"
        assert gpus[1].index == 1

    def test_reports_vram_as_the_card_is_sold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """nvidia-smi reports 81559 MiB for an "80GB" H100. Binary units give
        79.6, which reads correctly against the spec sheet; decimal units would
        print 85.5 for the same card."""
        gpus, _, _ = self._detect(monkeypatch, NVIDIA_SMI_2XH100)
        assert 79.0 < gpus[0].memory_gb < 80.0
        assert gpus[0].advertised_gb == 80

    def test_an_eighty_gb_card_satisfies_an_eighty_gb_requirement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap: 81559 MiB is 79.6 GiB usable, so a naive comparison
        rejects the very card the requirement names."""
        gpus, _, _ = self._detect(monkeypatch, NVIDIA_SMI_2XH100)
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 80}),
            host(gpus=gpus, driver="550.54.15"),
        )
        assert status_of(report, "GPU VRAM") is CheckStatus.OK

    def test_parses_compute_capability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus, _, _ = self._detect(monkeypatch, NVIDIA_SMI_2XH100)
        assert gpus[0].compute_capability == "9.0"

    def test_unavailable_fields_become_none_not_placeholder_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gpus, _, _ = self._detect(monkeypatch, NVIDIA_SMI_PARTIAL)
        assert gpus[0].compute_capability is None
        assert gpus[0].uuid is None
        assert gpus[0].driver_version == "525.85.12"

    def test_missing_nvidia_smi_explains_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus, detail, _ = self._detect(monkeypatch, "", present=False)
        assert gpus == []
        assert detail is not None and "nvidia-smi" in detail

    def test_a_failing_nvidia_smi_explains_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("offlineai.hardware.detector.have", lambda _: True)
        monkeypatch.setattr(
            "offlineai.hardware.detector.run",
            lambda args, **_: CommandResult(tuple(args), 9, "", "driver/library mismatch"),
        )
        gpus, detail, _ = HardwareDetector().detect_gpus()
        assert gpus == []
        assert detail is not None and "mismatch" in detail


class TestDriverVersionComparison:
    @pytest.mark.parametrize(
        ("a", "b"),
        [("550.54.15", "550"), ("550.54.15", "525.85.12"), ("12.4", "11.8")],
    )
    def test_newer_compares_greater(self, a: str, b: str) -> None:
        assert parse_driver_version(a) > parse_driver_version(b)

    def test_equal_versions_compare_equal(self) -> None:
        assert parse_driver_version("550.54.15") == parse_driver_version("550.54.15")


class TestCompatibility:
    def test_ram_is_reported_in_the_units_it_is_sold_in(self) -> None:
        """A 16 GB machine must read as 16, not 17.2."""
        sixteen_gib = MemoryInfo(total_bytes=16 * 1024**3)
        assert sixteen_gib.total_gb == 16.0

    def test_disk_uses_decimal_units_because_drives_are_sold_that_way(self) -> None:
        """Deliberately a different convention from RAM: a 500 GB SSD really is
        500 x 10^9 bytes."""
        drive = DiskInfo(path="/", total_bytes=500 * 1000**3, free_bytes=500 * 1000**3)
        assert drive.free_gb == 500.0

    def test_a_satisfied_host_is_compatible(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48, "count": 1}),
            host(gpus=[h100()], driver="550.54.15"),
        )
        assert report.compatible
        assert report.verdict == "COMPATIBLE"

    def test_insufficient_vram_fails(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 80}),
            host(
                gpus=[GpuDevice(0, "NVIDIA L4", 23034, "550.54.15")],
                driver="550.54.15",
            ),
        )
        assert not report.compatible
        assert status_of(report, "GPU VRAM") is CheckStatus.FAILED

    def test_vram_is_compared_per_device_not_summed(self) -> None:
        """Two 24 GB cards do not satisfy a 48 GB requirement unless the
        workload shards, which is an application decision (section 30)."""
        small = [GpuDevice(i, "NVIDIA L4", 23034, "550.54.15") for i in range(2)]
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48}),
            host(gpus=small, driver="550.54.15"),
        )
        assert status_of(report, "GPU VRAM") is CheckStatus.FAILED

    def test_gpu_count_is_checked(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "count": 4}),
            host(gpus=[h100(0), h100(1)], driver="550.54.15"),
        )
        assert status_of(report, "GPU count") is CheckStatus.FAILED

    def test_an_old_driver_fails(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumDriver": "550"}),
            host(gpus=[h100()], driver="470.82.01"),
        )
        assert status_of(report, "GPU driver") is CheckStatus.FAILED

    def test_a_missing_gpu_when_required_fails_with_the_reason(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumMemoryGB": 48}),
            host(gpus=[], gpu_detail="nvidia-smi is not installed"),
        )
        failure = next(c for c in report.checks if c.name == "GPU")
        assert failure.status is CheckStatus.FAILED
        assert failure.detail is not None and "nvidia-smi" in failure.detail

    def test_insufficient_ram_fails(self) -> None:
        report = check_compatibility(manifest(minimumRamGB=256), host(ram_gb=32))
        assert status_of(report, "RAM") is CheckStatus.FAILED

    def test_wrong_architecture_fails(self) -> None:
        report = check_compatibility(manifest(), host(architecture="arm64"))
        assert status_of(report, "CPU architecture") is CheckStatus.FAILED


class TestUnevaluatedIsNeverPass:
    """Section 21, the rule that matters most here."""

    def test_no_gpu_requirement_is_skipped_not_passed(self) -> None:
        report = check_compatibility(manifest(), host())
        assert status_of(report, "GPU") is CheckStatus.SKIPPED

    def test_undeterminable_ram_is_skipped_not_passed(self) -> None:
        report = HardwareReport(
            os_name="linux",
            os_version="6.8.0",
            architecture="amd64",
            cpu=CpuInfo("amd64", 8),
            memory=None,
            disk=None,
        )
        result = check_compatibility(manifest(minimumRamGB=16), report)
        assert status_of(result, "RAM") is CheckStatus.SKIPPED

    def test_undeterminable_driver_is_skipped_not_passed(self) -> None:
        report = check_compatibility(
            manifest(gpu={"vendor": "nvidia", "minimumDriver": "550"}),
            host(gpus=[GpuDevice(0, "NVIDIA H100", 81559)], driver=None),
        )
        assert status_of(report, "GPU driver") is CheckStatus.SKIPPED

    def test_a_skipped_check_does_not_make_the_host_incompatible(self) -> None:
        report = check_compatibility(manifest(), host())
        assert report.compatible


class TestNoGuaranteeIsClaimed:
    def test_the_caveat_is_explicit(self) -> None:
        """Section 21 forbids claiming a workload will work merely because the
        minimums are met."""
        assert "not guaranteed" in COMPATIBILITY_CAVEAT.lower()


class TestContainerRuntimeVersion:
    def test_an_older_daemon_warns_rather_than_failing(self) -> None:
        """The bundle's minimum is what it was tested against. An older daemon
        usually still works, and refusing outright would be wrong."""
        report = check_compatibility(
            manifest(docker={"minimumVersion": "27"}),
            host(),
            runtime=RuntimeAvailability(available=True, version="24.0.6"),
        )
        assert status_of(report, "Container runtime") is CheckStatus.WARNING
        assert report.compatible
        assert report.warnings

    def test_a_newer_daemon_passes(self) -> None:
        report = check_compatibility(
            manifest(docker={"minimumVersion": "27"}),
            host(),
            runtime=RuntimeAvailability(available=True, version="27.3.1"),
        )
        assert status_of(report, "Container runtime") is CheckStatus.OK

    def test_an_absent_runtime_fails(self) -> None:
        report = check_compatibility(
            manifest(docker={"minimumVersion": "27"}),
            host(),
            runtime=RuntimeAvailability(available=False, detail="daemon not reachable"),
        )
        assert status_of(report, "Container runtime") is CheckStatus.FAILED
        assert not report.compatible


class TestStorageEstimate:
    def _manifest_with_artifacts(self) -> Manifest:
        return Manifest.model_validate(
            {
                "formatVersion": FORMAT_VERSION,
                "package": {"name": "demo", "version": "1.0.0"},
                "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
                "platforms": ["linux/amd64"],
                "artifacts": [
                    {
                        "id": "m",
                        "type": "model",
                        "path": "artifacts/models/m",
                        "size": 48 * 1000**3,
                        "sha256": "a" * 64,
                    },
                    {
                        "id": "c",
                        "type": "oci-image",
                        "path": "artifacts/containers/c.tar",
                        "size": 12 * 1000**3,
                        "sha256": "b" * 64,
                    },
                ],
            }
        )

    def test_no_extraction_space_is_required(self) -> None:
        """Import streams into content-addressed storage, so the
        specification's extraction term is genuinely zero here."""
        estimate = estimate_storage(
            self._manifest_with_artifacts(),
            bundle_bytes=60 * 1000**3,
            available_bytes=500 * 1000**3,
        )
        assert estimate.artifact_storage_bytes == 60 * 1000**3
        assert estimate.recommended_bytes < 2 * estimate.artifact_storage_bytes

    def test_hardlinked_models_cost_nothing_extra(self) -> None:
        estimate = estimate_storage(
            self._manifest_with_artifacts(),
            bundle_bytes=60 * 1000**3,
            available_bytes=500 * 1000**3,
            models_are_hardlinked=True,
        )
        assert estimate.model_storage_bytes == 0

    def test_copied_models_cost_the_model_again(self) -> None:
        estimate = estimate_storage(
            self._manifest_with_artifacts(),
            bundle_bytes=60 * 1000**3,
            available_bytes=500 * 1000**3,
            models_are_hardlinked=False,
        )
        assert estimate.model_storage_bytes == 48 * 1000**3

    def test_container_storage_allows_for_layer_expansion(self) -> None:
        estimate = estimate_storage(
            self._manifest_with_artifacts(),
            bundle_bytes=60 * 1000**3,
            available_bytes=500 * 1000**3,
        )
        assert estimate.container_storage_bytes > 12 * 1000**3

    def test_insufficient_space_is_reported_with_the_shortfall(self) -> None:
        estimate = estimate_storage(
            self._manifest_with_artifacts(),
            bundle_bytes=60 * 1000**3,
            available_bytes=10 * 1000**3,
        )
        assert not estimate.sufficient
        assert estimate.shortfall_bytes > 0


class TestDetectorOnThisHost:
    """Smoke tests against the real machine: detection must never raise."""

    def test_detect_returns_a_report(self) -> None:
        report = HardwareDetector().detect()
        assert report.os_name
        assert report.architecture in ("amd64", "arm64")
        assert report.cpu.cores >= 1

    def test_detection_never_raises_when_things_are_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("offlineai.hardware.detector.have", lambda _: False)
        report = HardwareDetector().detect()
        assert report.gpus == []
        assert report.gpu_detail is not None
