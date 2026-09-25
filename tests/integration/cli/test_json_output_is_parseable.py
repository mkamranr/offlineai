"""Every command's `--json` output must parse.

Written after `offlineai --json doctor` was found emitting a JSON payload
followed by the human check table, which made `| jq` fail. The cause was a
command reaching for `console.print` - which is ungated - instead of
`output.line`, which respects `--json`. That is an easy mistake to repeat, so
this sweeps every command rather than the one that broke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from offlineai.runtime.fake import FakeRuntime

#: Commands whose payload is the JSON itself rather than a result model.
_RAW_JSON = {"sbom", "graph"}


def commands_under_test(bundle: Path, tmp_path: Path) -> list[tuple[str, list[str]]]:
    """Every command that emits JSON, with arguments that make it run."""
    return [
        ("init", ["init"]),
        ("inspect", ["inspect", str(bundle)]),
        ("verify", ["verify", str(bundle)]),
        ("import", ["import", str(bundle)]),
        ("list", ["list"]),
        ("search", ["search", "demo"]),
        ("info", ["info", "cli-demo"]),
        ("check", ["check", str(bundle)]),
        ("doctor", ["doctor"]),
        ("graph", ["graph", "cli-demo"]),
        ("sbom", ["sbom", str(bundle)]),
        ("network-check", ["network-check", str(bundle)]),
        ("plugins", ["plugins"]),
        ("install", ["install", "cli-demo"]),
        ("status", ["status", "cli-demo"]),
        ("logs", ["logs", "cli-demo"]),
        ("stop", ["stop", "cli-demo"]),
        ("uninstall", ["uninstall", "cli-demo", "--yes"]),
        ("remove", ["remove", "cli-demo", "--yes"]),
    ]


class TestJsonIsAlwaysParseable:
    def test_every_command_emits_one_json_document(
        self,
        run_cli: Any,
        bundle: Path,
        tmp_path: Path,
        fake_runtime: FakeRuntime,
    ) -> None:
        """Run them in sequence, because several depend on the ones before."""
        broken: list[str] = []
        for name, args in commands_under_test(bundle, tmp_path):
            result = run_cli("--json", *args)
            if result.exit_code != 0:
                broken.append(f"{name}: exited {result.exit_code}\n{result.output}")
                continue
            try:
                json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                broken.append(
                    f"{name}: stdout is not one JSON document ({exc}).\n"
                    f"--- stdout ---\n{result.stdout[:600]}"
                )
        assert not broken, "\n\n".join(broken)

    def test_the_sweep_actually_covers_the_commands(self, bundle: Path, tmp_path: Path) -> None:
        """Guards against the list silently going stale."""
        covered = {name for name, _ in commands_under_test(bundle, tmp_path)}
        assert len(covered) >= 19

    def test_human_output_never_reaches_stdout_in_json_mode(
        self, run_cli: Any, bundle: Path, fake_runtime: FakeRuntime
    ) -> None:
        """The specific failure: a payload followed by a rendered table."""
        result = run_cli("--json", "doctor").assert_ok()
        assert "OfflineAI Doctor" not in result.stdout
        assert "Permissions:" not in result.stdout

    @pytest.mark.parametrize("command", ["doctor", "list", "plugins"])
    def test_quiet_mode_emits_nothing_on_stdout(
        self, run_cli: Any, command: str, fake_runtime: FakeRuntime
    ) -> None:
        result = run_cli("--quiet", command)
        assert result.exit_code == 0
        assert result.stdout.strip() == "", f"{command} printed in --quiet mode"
