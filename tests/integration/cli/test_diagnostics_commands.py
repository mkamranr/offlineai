"""`check` and `doctor`.

The rule these exist to protect is section 21: a requirement that could not be
evaluated must be SKIPPED with a reason, never PASS. "We could not look" and
"it is fine" are different answers, and an operator deciding whether to deploy
needs to be able to tell them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
