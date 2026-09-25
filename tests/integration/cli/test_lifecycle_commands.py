"""`install`, `start`, `stop`, `restart`, `status`, `logs`, `rollback`, `uninstall`.

The module with the least coverage and the most surface: eight commands, each
wiring flags through to an engine. Two of the bugs this project hit lived
here - `status` reporting the declared port instead of the bound one, and the
`--config` override path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offlineai.exitcodes import ExitCode
from offlineai.runtime.fake import FakeRuntime
from offlineai.runtime.manager import container_name


class TestInstall:
    def test_installs_and_starts(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        result = run_cli("install", "cli-demo").assert_ok()
        assert "COMPLETED" in result.output
        assert container_name("cli-demo", "app") in fake_runtime.containers

    def test_reports_the_endpoint(self, run_cli: Any, imported: Path) -> None:
        payload = run_cli("--json", "install", "cli-demo").assert_ok().json()
        assert payload["endpoints"] == ["http://localhost:8000"]
        assert payload["state"] == "COMPLETED"

    def test_no_start_installs_without_running_anything(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("install", "cli-demo", "--no-start").assert_ok()
        assert fake_runtime.containers == {}

    def test_the_image_comes_from_the_bundle_not_a_pull(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("install", "cli-demo").assert_ok()
        assert "pull" not in {op for op, _ in fake_runtime.calls}
        assert "load" in {op for op, _ in fake_runtime.calls}

    def test_gpus_flag_is_passed_through(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("install", "cli-demo", "--gpus", "0,1").assert_ok()
        spec = fake_runtime.containers[container_name("cli-demo", "app")].spec
        assert spec.gpu_device_ids == ["0", "1"]

    def test_a_config_override_remaps_the_port(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime, tmp_path: Path
    ) -> None:
        config = tmp_path / "site.yaml"
        config.write_text('services:\n  app:\n    ports: ["8099:8000"]\n')
        payload = (
            run_cli("--json", "install", "cli-demo", "--config", str(config)).assert_ok().json()
        )
        spec = fake_runtime.containers[container_name("cli-demo", "app")].spec
        assert spec.ports == ["8099:8000"]
        assert payload["endpoints"] == ["http://localhost:8099"], (
            "the reported endpoint must be the bound port, not the declared one"
        )

    def test_a_config_environment_override_reaches_the_container(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime, tmp_path: Path
    ) -> None:
        config = tmp_path / "site.yaml"
        config.write_text("environment:\n  DEPLOYMENT_SITE: site-b\n")
        run_cli("install", "cli-demo", "--config", str(config)).assert_ok()
        spec = fake_runtime.containers[container_name("cli-demo", "app")].spec
        assert spec.environment["DEPLOYMENT_SITE"] == "site-b"

    def test_a_missing_config_file_is_reported(
        self, run_cli: Any, imported: Path, tmp_path: Path
    ) -> None:
        result = run_cli("install", "cli-demo", "--config", str(tmp_path / "nope.yaml"))
        assert result.exit_code == ExitCode.INVALID_PACKAGE
        assert "does not exist" in result.output

    def test_installing_a_bundle_path_imports_it_first(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        result = run_cli("install", str(bundle)).assert_ok()
        assert "Importing" in result.output
        assert "COMPLETED" in result.output

    def test_an_unknown_package_fails(self, run_cli: Any) -> None:
        assert run_cli("install", "ghost").exit_code != ExitCode.SUCCESS

    def test_dev_mode_is_labelled_on_an_unsupported_platform(
        self, run_cli: Any, imported: Path
    ) -> None:
        """On Linux this reports False; on macOS it must warn rather than
        quietly pretend the platform is supported."""
        import platform

        payload = run_cli("--json", "install", "cli-demo").assert_ok().json()
        assert payload["dev_mode"] is (platform.system().lower() != "linux")

    def test_reinstalling_is_not_a_name_collision(self, run_cli: Any, imported: Path) -> None:
        run_cli("install", "cli-demo").assert_ok()
        run_cli("install", "cli-demo").assert_ok()


class TestStartStopRestart:
    def test_stop_then_status_reports_stopped(self, run_cli: Any, installed: Path) -> None:
        run_cli("stop", "cli-demo").assert_ok()
        assert run_cli("--json", "status", "cli-demo").assert_ok().json()["status"] == ("STOPPED")

    def test_start_brings_it_back(self, run_cli: Any, installed: Path) -> None:
        run_cli("stop", "cli-demo", "--remove").assert_ok()
        run_cli("start", "cli-demo").assert_ok()
        assert run_cli("--json", "status", "cli-demo").assert_ok().json()["status"] == ("RUNNING")

    def test_start_uses_the_image_the_bundle_carries(
        self, run_cli: Any, installed: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("stop", "cli-demo", "--remove").assert_ok()
        run_cli("start", "cli-demo").assert_ok()
        spec = fake_runtime.containers[container_name("cli-demo", "app")].spec
        assert spec.image == "python:3.12-slim"

    def test_restart(self, run_cli: Any, installed: Path) -> None:
        result = run_cli("restart", "cli-demo").assert_ok()
        assert "Restarted" in result.output
        assert run_cli("--json", "status", "cli-demo").assert_ok().json()["status"] == ("RUNNING")

    def test_stop_reports_what_it_stopped(self, run_cli: Any, installed: Path) -> None:
        payload = run_cli("--json", "stop", "cli-demo").assert_ok().json()
        assert payload["stopped"] == [container_name("cli-demo", "app")]

    def test_stop_with_remove_deletes_the_container(
        self, run_cli: Any, installed: Path, fake_runtime: FakeRuntime
    ) -> None:
        run_cli("stop", "cli-demo", "--remove").assert_ok()
        assert fake_runtime.containers == {}


class TestStatus:
    def test_a_running_package(self, run_cli: Any, installed: Path) -> None:
        result = run_cli("status", "cli-demo").assert_ok()
        assert "RUNNING" in result.output
        assert "http://localhost:8000" in result.output

    def test_json_shape(self, run_cli: Any, installed: Path) -> None:
        payload = run_cli("--json", "status", "cli-demo").assert_ok().json()
        assert payload["package"] == "cli-demo"
        assert payload["services"] == {"app": "RUNNING"}

    def test_status_with_no_argument_covers_every_package(
        self, run_cli: Any, installed: Path
    ) -> None:
        payload = run_cli("--json", "status").assert_ok().json()
        assert [p["package"] for p in payload["packages"]] == ["cli-demo"]

    def test_status_of_an_empty_registry(self, run_cli: Any) -> None:
        payload = run_cli("--json", "status").assert_ok().json()
        assert payload["packages"] == []


class TestLogs:
    def test_shows_container_output(self, run_cli: Any, installed: Path) -> None:
        assert "cli-demo" in run_cli("logs", "cli-demo").assert_ok().output

    def test_json_carries_the_text(self, run_cli: Any, installed: Path) -> None:
        payload = run_cli("--json", "logs", "cli-demo").assert_ok().json()
        assert "starting" in payload["logs"]

    def test_tail_is_accepted(self, run_cli: Any, installed: Path) -> None:
        run_cli("logs", "cli-demo", "--tail", "1").assert_ok()

    def test_an_unknown_service_names_the_real_ones(self, run_cli: Any, installed: Path) -> None:
        result = run_cli("logs", "cli-demo", "--service", "ghost")
        assert result.exit_code == ExitCode.RUNTIME_FAILURE
        assert "app" in result.output


class TestRollback:
    def _installation_id(self, run_cli: Any) -> str:
        return run_cli("--json", "install", "cli-demo").assert_ok().json()["installation_id"]

    def test_undoes_an_installation(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        install_id = self._installation_id(run_cli)
        result = run_cli("rollback", install_id).assert_ok()
        assert "Rolled back" in result.output
        assert fake_runtime.containers == {}

    def test_images_are_kept_by_default(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        install_id = self._installation_id(run_cli)
        payload = run_cli("--json", "rollback", install_id).assert_ok().json()
        assert fake_runtime.images, "a 12 GB image should not be discarded by default"
        assert any("kept" in s for s in payload["skipped"])

    def test_remove_images_discards_them(
        self, run_cli: Any, imported: Path, fake_runtime: FakeRuntime
    ) -> None:
        install_id = self._installation_id(run_cli)
        run_cli("rollback", install_id, "--remove-images").assert_ok()
        assert fake_runtime.images == {}

    def test_an_unknown_installation_id_fails(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("rollback", "install-19700101-001")
        assert result.exit_code != ExitCode.SUCCESS
        assert "no installation" in result.output.lower()


class TestUninstall:
    def test_stops_and_removes_what_install_created(
        self, run_cli: Any, installed: Path, fake_runtime: FakeRuntime
    ) -> None:
        result = run_cli("uninstall", "cli-demo", "--yes").assert_ok()
        assert "Uninstalled" in result.output
        assert fake_runtime.containers == {}

    def test_the_package_stays_in_the_registry(self, run_cli: Any, installed: Path) -> None:
        run_cli("uninstall", "cli-demo", "--yes").assert_ok()
        assert run_cli("--json", "list").assert_ok().json()["packages"], (
            "uninstall removes the installation, not the package"
        )

    def test_it_points_at_remove_for_going_further(self, run_cli: Any, installed: Path) -> None:
        result = run_cli("uninstall", "cli-demo", "--yes").assert_ok()
        assert "offlineai remove cli-demo" in result.output

    def test_keep_data_leaves_installed_files(self, run_cli: Any, installed: Path) -> None:
        from offlineai.config.settings import load_settings

        installed_dir = load_settings().data_dir / "installed" / "cli-demo"
        assert installed_dir.is_dir()
        run_cli("uninstall", "cli-demo", "--yes", "--keep-data").assert_ok()
        assert installed_dir.is_dir(), "--keep-data must not delete installed files"

    def test_without_keep_data_the_files_go(self, run_cli: Any, installed: Path) -> None:
        from offlineai.config.settings import load_settings

        installed_dir = load_settings().data_dir / "installed" / "cli-demo"
        run_cli("uninstall", "cli-demo", "--yes").assert_ok()
        assert not installed_dir.exists()

    @pytest.mark.parametrize("answer", ["n", ""])
    def test_declining_the_prompt_changes_nothing(
        self, installed: Path, fake_runtime: FakeRuntime, answer: str
    ) -> None:
        from typer.testing import CliRunner

        from offlineai.cli.main import app

        result = CliRunner().invoke(app, ["uninstall", "cli-demo"], input=f"{answer}\n")
        assert result.exit_code == ExitCode.SUCCESS
        assert "Cancelled" in result.stdout
        assert fake_runtime.containers, "declining must leave the services running"
