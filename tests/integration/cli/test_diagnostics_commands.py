"""`check` and `doctor`.

The rule these exist to protect is section 21: a requirement that could not be
evaluated must be SKIPPED with a reason, never PASS. "We could not look" and
"it is fine" are different answers, and an operator deciding whether to deploy
needs to be able to tell them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offlineai.exitcodes import ExitCode
from offlineai.runtime.fake import FakeRuntime


def status_of(payload: dict[str, Any], name: str) -> str | None:
    return next((c["status"] for c in payload["checks"] if c["name"] == name), None)


class TestCheck:
    def test_a_satisfiable_bundle_is_compatible(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        payload = run_cli("--json", "check", str(bundle)).assert_ok().json()
        assert payload["compatible"] is True
        assert payload["verdict"] == "COMPATIBLE"

    def test_it_says_success_is_not_guaranteed(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        """Section 21 forbids claiming a workload will work merely because the
        minimums are met."""
        result = run_cli("check", str(bundle)).assert_ok()
        assert "not guaranteed" in result.output

    def test_it_reports_a_storage_estimate(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        payload = run_cli("--json", "check", str(bundle)).assert_ok().json()
        storage = payload["storage"]
        assert storage["recommended_free"] > 0
        assert storage["sufficient"] is True

    def test_the_estimate_has_no_extraction_term(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        """Import streams into content-addressed storage, so there is nothing
        to extract and saying otherwise would overstate the requirement."""
        result = run_cli("check", str(bundle)).assert_ok()
        assert "no separate extraction space" in result.output

    def test_it_accepts_an_imported_package_name(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("check", "cli-demo").assert_ok()

    def test_an_unsatisfiable_gpu_requirement_exits_four(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        source = tmp_path / "huge"
        source.mkdir()
        (source / "offlineai.yaml").write_text(
            "apiVersion: offlineai/v1\nkind: Package\n"
            "metadata:\n  name: huge\n  version: 1.0.0\n"
            "hardware:\n  gpu:\n    required: true\n    vendor: nvidia\n"
            "    minimum_vram_gb: 640\n  minimum_ram_gb: 8192\n"
        )
        run_cli("build", str(source), "-o", str(tmp_path / "out")).assert_ok()
        result = run_cli("check", str(tmp_path / "out" / "huge-1.0.0.offlineai"))
        assert result.exit_code == ExitCode.HARDWARE_INCOMPATIBLE
        assert "INCOMPATIBLE" in result.output

    def test_it_reads_only_the_header(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        """Nothing is extracted and nothing is installed."""
        before = set(bundle.parent.rglob("*"))
        run_cli("check", str(bundle)).assert_ok()
        assert set(bundle.parent.rglob("*")) == before


class TestDoctor:
    def test_it_reports_on_the_host(self, run_cli: Any, fake_runtime: FakeRuntime) -> None:
        payload = run_cli("--json", "doctor").assert_ok().json()
        names = {c["name"] for c in payload["checks"]}
        assert {"OS", "Architecture", "Memory", "Disk", "Permissions"} <= names

    def test_a_working_runtime_passes(self, run_cli: Any, fake_runtime: FakeRuntime) -> None:
        payload = run_cli("--json", "doctor").assert_ok().json()
        assert status_of(payload, "Container runtime") == "OK"

    def test_an_absent_gpu_is_skipped_not_passed(
        self, run_cli: Any, fake_runtime: FakeRuntime
    ) -> None:
        """The rule this whole module exists to protect."""
        payload = run_cli("--json", "doctor").assert_ok().json()
        assert status_of(payload, "GPU") in ("SKIPPED", "OK")
        gpu = next(c for c in payload["checks"] if c["name"] == "GPU")
        if gpu["status"] == "SKIPPED":
            assert gpu["detail"], "a skipped check must say why"

    def test_a_runtime_without_gpu_support_is_skipped_with_a_reason(
        self, run_cli: Any, fake_runtime: FakeRuntime
    ) -> None:
        payload = run_cli("--json", "doctor").assert_ok().json()
        entry = next(c for c in payload["checks"] if c["name"] == "GPU runtime")
        assert entry["status"] == "SKIPPED"
        assert "no GPU runtime" in entry["detail"]

    def test_a_gpu_capable_runtime_passes(self, run_cli: Any, gpu_runtime: FakeRuntime) -> None:
        payload = run_cli("--json", "doctor").assert_ok().json()
        assert status_of(payload, "GPU runtime") == "OK"

    def test_an_unusable_runtime_makes_the_host_not_ready(
        self, run_cli: Any, monkeypatch: Any
    ) -> None:
        broken = FakeRuntime(available=False)

        class Stub:
            def __new__(cls, *a: object, **k: object) -> FakeRuntime:
                return broken

        monkeypatch.setattr("offlineai.cli.commands.diagnostics.DockerRuntime", Stub)
        result = run_cli("doctor")
        assert result.exit_code == ExitCode.HARDWARE_INCOMPATIBLE
        assert "NOT READY" in result.output

    def test_it_never_claims_more_than_it_checked(
        self, run_cli: Any, fake_runtime: FakeRuntime
    ) -> None:
        payload = run_cli("--json", "doctor").assert_ok().json()
        for check in payload["checks"]:
            if check["status"] == "SKIPPED":
                assert check["detail"], f"{check['name']} was skipped without a reason"


class TestCheckAgainstAProfile:
    """Section 5.6. `check` normally validates the machine it runs on, which
    on a builder is the wrong machine - the target is air-gapped and
    elsewhere. A profile lets the question be asked from anywhere."""

    H100 = """\
name: h100-server
os:
  family: ubuntu
  version: "24.04"
architecture: amd64
gpu:
  vendor: nvidia
  minimum_driver: "550"
  memory_gb: 80
  count: 8
runtime:
  docker: ">=27"
memory_gb: 1024
"""

    @pytest.fixture
    def gpu_bundle(self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime) -> Path:
        source = tmp_path / "gpu-pkg"
        source.mkdir()
        (source / "offlineai.yaml").write_text(
            "apiVersion: offlineai/v1\nkind: Package\n"
            "metadata:\n  name: gpu-workload\n  version: 1.0.0\n"
            "hardware:\n  gpu:\n    required: true\n    vendor: nvidia\n"
            "    minimum_vram_gb: 80\n    minimum_driver: '550'\n"
            "  minimum_ram_gb: 512\n"
        )
        run_cli("build", str(source), "-o", str(tmp_path / "gpu-dist")).assert_ok()
        return tmp_path / "gpu-dist" / "gpu-workload-1.0.0.offlineai"

    @pytest.fixture
    def profile(self, tmp_path: Path) -> Path:
        path = tmp_path / "h100-server.yaml"
        path.write_text(self.H100)
        return path

    def test_a_bundle_this_host_cannot_run_passes_against_the_target(
        self, run_cli: Any, gpu_bundle: Path, profile: Path, fake_runtime: FakeRuntime
    ) -> None:
        """The entire point of the feature."""
        assert run_cli("check", str(gpu_bundle)).exit_code == ExitCode.HARDWARE_INCOMPATIBLE
        payload = (
            run_cli("--json", "check", str(gpu_bundle), "--profile", str(profile))
            .assert_ok()
            .json()
        )
        assert payload["compatible"] is True

    def test_the_output_says_it_is_a_profile_and_which(
        self, run_cli: Any, gpu_bundle: Path, profile: Path, fake_runtime: FakeRuntime
    ) -> None:
        """A profile result mistaken for a host result is the failure mode
        that would actually hurt someone."""
        result = run_cli("check", str(gpu_bundle), "--profile", str(profile)).assert_ok()
        assert "h100-server" in result.output
        assert "NOT this host" in result.output

    def test_json_records_what_was_evaluated(
        self, run_cli: Any, gpu_bundle: Path, profile: Path, fake_runtime: FakeRuntime
    ) -> None:
        payload = (
            run_cli("--json", "check", str(gpu_bundle), "--profile", str(profile))
            .assert_ok()
            .json()
        )
        assert payload["profile"] == "h100-server"
        assert payload["evaluated_against"] == "profile"

    def test_without_a_profile_it_says_this_host(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        payload = run_cli("--json", "check", str(bundle)).assert_ok().json()
        assert payload["profile"] is None
        assert payload["evaluated_against"] == "this host"

    def test_a_linux_distribution_is_not_called_unsupported(
        self, run_cli: Any, gpu_bundle: Path, profile: Path, fake_runtime: FakeRuntime
    ) -> None:
        """A profile says `family: ubuntu`; the detector says `linux`. Without
        normalising, every Linux target would be reported as an unsupported
        platform - wrong, and the opposite of reassuring."""
        result = run_cli("check", str(gpu_bundle), "--profile", str(profile)).assert_ok()
        assert "ubuntu is not a supported target platform" not in result.output

    def test_no_free_space_is_invented(
        self, run_cli: Any, gpu_bundle: Path, profile: Path, fake_runtime: FakeRuntime
    ) -> None:
        """A profile describes a class of machine, not how much space it has
        free today. Reporting a shortfall against zero would be fabricated."""
        result = run_cli("check", str(gpu_bundle), "--profile", str(profile)).assert_ok()
        assert "Insufficient disk space" not in result.output
        assert "does not state free space" in result.output

    def test_a_profile_that_cannot_meet_the_requirement_exits_four(
        self, run_cli: Any, gpu_bundle: Path, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        small = tmp_path / "small.yaml"
        small.write_text(
            "name: workstation\narchitecture: amd64\nos:\n  family: ubuntu\n"
            "gpu:\n  vendor: nvidia\n  memory_gb: 24\n  minimum_driver: '550'\n"
        )
        result = run_cli("check", str(gpu_bundle), "--profile", str(small))
        assert result.exit_code == ExitCode.HARDWARE_INCOMPATIBLE
        assert "INCOMPATIBLE" in result.output

    def test_an_omitted_field_is_skipped_not_passed(
        self, run_cli: Any, gpu_bundle: Path, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        """The rule that makes profiles safe rather than merely convenient."""
        silent = tmp_path / "silent.yaml"
        silent.write_text(
            "name: unspecified\narchitecture: amd64\nos:\n  family: ubuntu\n"
            "gpu:\n  vendor: nvidia\n  memory_gb: 80\n  minimum_driver: '550'\n"
        )
        payload = (
            run_cli("--json", "check", str(gpu_bundle), "--profile", str(silent)).assert_ok().json()
        )
        assert status_of(payload, "RAM") == "SKIPPED", (
            "a profile silent about RAM must not approve a 512 GB requirement"
        )

    def test_a_missing_profile_file_is_reported(
        self, run_cli: Any, bundle: Path, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        result = run_cli("check", str(bundle), "--profile", str(tmp_path / "nope.yaml"))
        assert result.exit_code == ExitCode.INVALID_PACKAGE
        assert "doctor --save-profile" in result.output

    def test_the_shipped_example_profile_works(
        self, run_cli: Any, gpu_bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        example = Path(__file__).resolve().parents[3] / "examples" / "profiles" / "h100-server.yaml"
        assert example.is_file(), "the §5.6 example profile should ship with the project"
        run_cli("check", str(gpu_bundle), "--profile", str(example)).assert_ok()


class TestDoctorSavesAProfile:
    """Turns a profile from a hand-written guess into measured ground truth:
    run on the target, carry the file back, check on the builder."""

    def test_it_writes_a_loadable_profile(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        from offlineai.schema.profile import load_profile

        out = tmp_path / "host.yaml"
        run_cli("doctor", "--save-profile", str(out)).assert_ok()
        assert load_profile(out).name

    def test_the_name_can_be_set(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        from offlineai.schema.profile import load_profile

        out = tmp_path / "host.yaml"
        run_cli("doctor", "--save-profile", str(out), "--profile-name", "rack-07").assert_ok()
        assert load_profile(out).name == "rack-07"

    def test_it_tells_you_what_to_do_with_the_file(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        out = tmp_path / "host.yaml"
        result = run_cli("doctor", "--save-profile", str(out)).assert_ok()
        assert "check <bundle> --profile" in result.output

    def test_json_reports_the_path(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        out = tmp_path / "host.yaml"
        payload = run_cli("--json", "doctor", "--save-profile", str(out)).assert_ok().json()
        assert payload["saved_profile"] == str(out)

    def test_checking_against_the_captured_profile_agrees_with_the_live_check(
        self, run_cli: Any, bundle: Path, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        """The loop that makes the feature trustworthy."""
        out = tmp_path / "host.yaml"
        run_cli("doctor", "--save-profile", str(out)).assert_ok()

        live = run_cli("--json", "check", str(bundle)).assert_ok().json()
        captured = run_cli("--json", "check", str(bundle), "--profile", str(out)).assert_ok().json()
        assert live["compatible"] == captured["compatible"]

    def test_doctor_without_the_flag_writes_nothing(
        self, run_cli: Any, tmp_path: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("doctor").assert_ok()
        assert not list(tmp_path.glob("*.yaml"))
