"""`graph`, `diff`, `plugins`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offlineai.runtime.fake import FakeRuntime


class TestGraph:
    def test_ascii_is_the_default(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("graph", "cli-demo").assert_ok()
        assert "cli-demo 1.0.0" in result.stdout
        assert "└──" in result.stdout or "├──" in result.stdout

    def test_it_groups_by_artifact_kind(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("graph", "cli-demo").assert_ok()
        assert "containers" in result.stdout
        assert "models" in result.stdout

    def test_a_sharded_model_is_one_node(self, run_cli: Any, imported: Path) -> None:
        """Two weight files, one model row - a tree with a line per shard is
        noise rather than information."""
        payload = json.loads(run_cli("graph", "cli-demo", "--format", "json").assert_ok().stdout)
        models = next(c for c in payload["children"] if c["name"] == "models")
        assert len(models["children"]) == 1
        assert "2 file(s)" in models["children"][0]["detail"]

    def test_dot_output_is_a_digraph(self, run_cli: Any, imported: Path) -> None:
        result = run_cli("graph", "cli-demo", "--format", "dot").assert_ok()
        assert result.stdout.strip().startswith("digraph offlineai {")
        assert "->" in result.stdout

    def test_it_accepts_a_bundle_path(self, run_cli: Any, bundle: Path) -> None:
        assert "cli-demo" in run_cli("graph", str(bundle)).assert_ok().stdout

    def test_json_flag_wins_over_format(self, run_cli: Any, imported: Path) -> None:
        payload = json.loads(run_cli("--json", "graph", "cli-demo").assert_ok().stdout)
        assert payload["name"] == "cli-demo 1.0.0"


class TestDiff:
    def _second_version(
        self,
        run_cli: Any,
        package_dir: Path,
        tmp_path: Path,
        changes: dict[str, Any] | None = None,
    ) -> Path:
        """Build a v2 of the fixture package.

        Edits the parsed document rather than appending text: appending lands
        the new key under whichever block happens to be last, which is how
        this first produced an invalid definition.
        """
        import shutil

        import yaml

        source = tmp_path / "v2"
        shutil.copytree(package_dir, source)
        definition = yaml.safe_load((source / "offlineai.yaml").read_text())
        definition["metadata"]["version"] = "2.0.0"
        for key, value in (changes or {}).items():
            if isinstance(value, dict) and isinstance(definition.get(key), dict):
                definition[key].update(value)
            else:
                definition[key] = value
        (source / "offlineai.yaml").write_text(yaml.safe_dump(definition, sort_keys=False))

        run_cli("build", str(source), "-o", str(tmp_path / "d2")).assert_ok()
        return tmp_path / "d2" / "cli-demo-2.0.0.offlineai"

    def test_identical_bundles_report_no_differences(self, run_cli: Any, bundle: Path) -> None:
        result = run_cli("diff", str(bundle), str(bundle)).assert_ok()
        assert "No differences" in result.output

    def test_requirement_changes_are_reported_separately(
        self,
        run_cli: Any,
        package_dir: Path,
        tmp_path: Path,
        bundle: Path,
        fake_runtime: FakeRuntime,
    ) -> None:
        """A bundle that quietly starts demanding more is deployment-blocking
        and would not show up as an artifact difference."""
        v2 = self._second_version(
            run_cli, package_dir, tmp_path, {"hardware": {"minimum_ram_gb": 512}}
        )
        payload = run_cli("--json", "diff", str(bundle), str(v2)).assert_ok().json()
        assert any("512" in c for c in payload["requirements_changed"])

    def test_it_reports_a_size_delta(
        self,
        run_cli: Any,
        bundle: Path,
        package_dir: Path,
        tmp_path: Path,
        fake_runtime: FakeRuntime,
    ) -> None:
        v2 = self._second_version(run_cli, package_dir, tmp_path)
        payload = run_cli("--json", "diff", str(bundle), str(v2)).assert_ok().json()
        assert "size_before" in payload
        assert "size_after" in payload

    def test_it_reads_only_the_manifests(self, run_cli: Any, bundle: Path, tmp_path: Path) -> None:
        before = set(tmp_path.rglob("*"))
        run_cli("diff", str(bundle), str(bundle)).assert_ok()
        assert set(tmp_path.rglob("*")) == before, "diff must not extract anything"


class TestPlugins:
    def test_lists_the_builtin_sources(self, run_cli: Any) -> None:
        result = run_cli("plugins").assert_ok()
        for source in ("local", "http", "huggingface", "oci"):
            assert source in result.output

    def test_json_names_the_entry_point_group(self, run_cli: Any) -> None:
        payload = run_cli("--json", "plugins").assert_ok().json()
        assert payload["group"] == "offlineai.sources"
        assert "huggingface" in payload["sources"]

    def test_no_plugin_failed_to_load(self, run_cli: Any) -> None:
        assert run_cli("--json", "plugins").assert_ok().json()["failures"] == {}
