"""`init`, `import`, `list`, `search`, `info`, `remove`.

The engines under these are well covered; what is not is the command layer -
flag wiring, output rendering, and the exit code that reaches a caller. That
is where the bugs this project actually hit were living.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offlineai.config.settings import load_settings
from offlineai.exitcodes import ExitCode


class TestInit:
    def test_creates_the_documented_tree(self, run_cli: Any) -> None:
        run_cli("init").assert_ok()
        settings = load_settings()
        for directory in (
            settings.registry_dir,
            settings.cache_dir,
            settings.bundles_dir,
            settings.logs_dir,
        ):
            assert directory.is_dir(), f"{directory} was not created"

    def test_writes_a_starter_config(self, run_cli: Any) -> None:
        run_cli("init").assert_ok()
        config = load_settings().config_file
        assert config.is_file()
        assert "container_engine" in config.read_text()

    def test_does_not_clobber_an_existing_config(self, run_cli: Any) -> None:
        run_cli("init").assert_ok()
        config = load_settings().config_file
        config.write_text("log_level: debug\n")
        result = run_cli("init").assert_ok()
        assert config.read_text() == "log_level: debug\n"
        assert "left unchanged" in result.output

    def test_force_replaces_it(self, run_cli: Any) -> None:
        run_cli("init").assert_ok()
        load_settings().config_file.write_text("log_level: debug\n")
        run_cli("init", "--force").assert_ok()
        assert "container_engine" in load_settings().config_file.read_text()

    def test_json_reports_the_paths(self, run_cli: Any) -> None:
        payload = run_cli("--json", "init").assert_ok().json()
        assert payload["config_created"] is True
        assert Path(payload["registry_dir"]).is_dir()


class TestImport:
    def test_verifies_then_imports(self, run_cli: Any, bundle: Path) -> None:
        result = run_cli("import", str(bundle)).assert_ok()
        assert "Verification: OK" in result.output
        assert "Imported cli-demo 1.0.0" in result.output

    def test_reports_the_artifact_count(self, run_cli: Any, bundle: Path) -> None:
        payload = run_cli("--json", "import", str(bundle)).assert_ok().json()
        assert payload["artifacts_imported"] == 3
        assert payload["already_present"] is False

    def test_a_second_import_is_a_no_op(self, run_cli: Any, bundle: Path) -> None:
        run_cli("import", str(bundle)).assert_ok()
        payload = run_cli("--json", "import", str(bundle)).assert_ok().json()
        assert payload["already_present"] is True
        assert payload["artifacts_imported"] == 0

    def test_force_reimports(self, run_cli: Any, bundle: Path) -> None:
        run_cli("import", str(bundle)).assert_ok()
        payload = run_cli("--json", "import", str(bundle), "--force").assert_ok().json()
        assert payload["already_present"] is False

    def test_skip_verify_still_imports(self, run_cli: Any, bundle: Path) -> None:
        result = run_cli("import", str(bundle), "--skip-verify").assert_ok()
        assert "Verification" not in result.output
        assert "Imported" in result.output

    def test_a_corrupt_bundle_is_refused_before_anything_is_stored(
        self, run_cli: Any, bundle: Path
    ) -> None:
        data = bytearray(bundle.read_bytes())
        data[int(len(data) * 0.6)] ^= 0xFF
        bundle.write_bytes(bytes(data))

        assert run_cli("import", str(bundle)).exit_code == ExitCode.VERIFICATION_FAILURE
        assert run_cli("--json", "list").assert_ok().json()["packages"] == []

    def test_a_missing_bundle_exits_three(self, run_cli: Any, tmp_path: Path) -> None:
        result = run_cli("import", str(tmp_path / "nope.offlineai"))
        assert result.exit_code == ExitCode.VERIFICATION_FAILURE


class TestList:
    def test_an_empty_registry_says_so_and_suggests_import(self, run_cli: Any) -> None:
        result = run_cli("list").assert_ok()
        assert "No packages imported" in result.output
        assert "offlineai import" in result.output

    def test_an_empty_registry_is_an_empty_json_list(self, run_cli: Any) -> None:
        assert run_cli("--json", "list").assert_ok().json()["packages"] == []

    def test_lists_an_imported_package(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("list").assert_ok()
        assert "cli-demo" in result.output
        assert "1.0.0" in result.output

    def test_json_carries_the_fields_automation_needs(self, run_cli: Any, imported: Path) -> None:
        entry = run_cli("--json", "list").assert_ok().json()["packages"][0]
        assert entry["name"] == "cli-demo"
        assert entry["version"] == "1.0.0"
        assert entry["platforms"] == ["linux/amd64"]
        assert entry["size"] > 0
        assert entry["signed"] is False


class TestSearch:
    def test_finds_a_match(self, run_cli: Any, imported: Path) -> None:
        assert "cli-demo" in run_cli("search", "demo").assert_ok().output

    def test_reports_no_match_without_failing(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("search", "nothing-like-this").assert_ok()
        assert "No packages matching" in result.output

    def test_matching_is_case_insensitive(self, run_cli: Any, imported: Path) -> None:
        assert "cli-demo" in run_cli("search", "DEMO").assert_ok().output


class TestInfo:
    def test_shows_the_package_detail(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("info", "cli-demo").assert_ok()
        assert "cli-demo" in result.output
        assert "linux/amd64" in result.output

    def test_lists_required_secrets_as_names_only(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("info", "cli-demo").assert_ok()
        assert "CLI_DEMO_TOKEN: REQUIRED" in result.output

    def test_json_reports_the_bundle_digest(self, run_cli: Any, imported: Path) -> None:
        payload = run_cli("--json", "info", "cli-demo").assert_ok().json()
        assert len(payload["bundle_sha256"]) == 64
        assert payload["required_secrets"] == ["CLI_DEMO_TOKEN"]

    def test_an_unknown_package_says_what_is_available(self, run_cli: Any) -> None:
        result = run_cli("info", "not-here")
        assert result.exit_code == ExitCode.INSTALLATION_FAILURE
        assert "offlineai import" in result.output


class TestRemove:
    def test_removes_with_confirmation_bypassed(self, run_cli: Any, imported: Path) -> None:
        run_cli("remove", "cli-demo", "--yes").assert_ok()
        assert run_cli("--json", "list").assert_ok().json()["packages"] == []

    def test_reports_what_was_freed(self, run_cli: Any, imported: Path) -> None:
        payload = run_cli("--json", "remove", "cli-demo", "--yes").assert_ok().json()
        assert payload["removed"] == "cli-demo:1.0.0"
        assert payload["artifacts_freed"] == 3
        assert payload["bytes_freed"] > 0

    def test_removing_an_unknown_package_fails(self, run_cli: Any) -> None:
        assert run_cli("remove", "ghost", "--yes").exit_code != ExitCode.SUCCESS

    @pytest.mark.parametrize("answer", ["n", "no", ""])
    def test_declining_the_prompt_keeps_the_package(self, imported: Path, answer: str) -> None:
        from typer.testing import CliRunner

        from offlineai.cli.main import app

        result = CliRunner().invoke(app, ["remove", "cli-demo"], input=f"{answer}\n")
        assert result.exit_code == ExitCode.SUCCESS
        assert "Cancelled" in result.stdout
        from offlineai.registry.registry import Registry

        assert Registry(load_settings()).get("cli-demo") is not None
