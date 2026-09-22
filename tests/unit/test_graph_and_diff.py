"""Sections 49, 58 and 59: graph, diff and plugin discovery."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from offlineai.bundler.differ import diff_manifests
from offlineai.bundler.graph import build_graph, to_ascii, to_dot, to_json
from offlineai.plugins.loader import ENTRY_POINT_GROUP, load_sources
from offlineai.schema.manifest import FORMAT_VERSION, Manifest


def artifact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "a",
        "type": "misc",
        "path": "artifacts/misc/a",
        "size": 100,
        "sha256": "a" * 64,
    }
    return {**base, **overrides}


def manifest(version: str = "1.0.0", **overrides: Any) -> Manifest:
    base: dict[str, Any] = {
        "formatVersion": FORMAT_VERSION,
        "package": {"name": "demo", "version": version},
        "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
        "platforms": ["linux/amd64"],
        "artifacts": [],
    }
    return Manifest.model_validate({**base, **overrides})


SHARDED_MODEL = [
    artifact(
        id=f"model-m-{i}",
        type="model",
        path=f"artifacts/models/m/model-{i:05d}-of-00003.safetensors",
        size=1000,
        sha256=f"{i:064d}",
        source=f"hf://org/model@main/model-{i:05d}-of-00003.safetensors",
        metadata={"model": "m", "repo": "org/model", "revision": "main"},
    )
    for i in range(1, 4)
]


class TestGraph:
    def test_sharded_models_collapse_to_one_node(self) -> None:
        """Eight identical-looking safetensors lines is noise, not information."""
        node = build_graph(manifest(artifacts=SHARDED_MODEL))
        models = next(c for c in node.children if c.name == "models")
        assert len(models.children) == 1
        assert models.children[0].name == "m"
        assert "3 file(s)" in (models.children[0].detail or "")

    def test_collapsed_size_is_the_sum(self) -> None:
        node = build_graph(manifest(artifacts=SHARDED_MODEL))
        models = next(c for c in node.children if c.name == "models")
        assert models.children[0].total_size == 3000

    def test_provenance_names_the_repository_not_a_file(self) -> None:
        node = build_graph(manifest(artifacts=SHARDED_MODEL))
        detail = next(c for c in node.children if c.name == "models").children[0].detail
        assert detail is not None
        assert "hf://org/model@main" in detail
        assert ".safetensors" not in detail

    def test_images_show_their_digest(self) -> None:
        digest = "sha256:" + "b" * 64
        node = build_graph(
            manifest(
                artifacts=[
                    artifact(
                        id="image-vllm",
                        type="oci-image",
                        path="artifacts/containers/vllm.tar",
                        digest=digest,
                        metadata={"image": "vllm/vllm-openai", "tag": "latest"},
                    )
                ]
            )
        )
        container = next(c for c in node.children if c.name == "containers").children[0]
        assert container.name == "vllm/vllm-openai:latest"
        assert container.detail is not None and digest[:19] in container.detail

    def test_ascii_output_is_a_tree(self) -> None:
        text = to_ascii(build_graph(manifest(artifacts=SHARDED_MODEL)))
        assert text.splitlines()[0] == "demo 1.0.0"
        assert "└──" in text or "├──" in text

    def test_json_output_round_trips(self) -> None:
        payload = to_json(build_graph(manifest(artifacts=SHARDED_MODEL)))
        assert json.loads(json.dumps(payload))["name"] == "demo 1.0.0"

    def test_dot_output_is_a_digraph(self) -> None:
        text = to_dot(build_graph(manifest(artifacts=SHARDED_MODEL)))
        assert text.startswith("digraph offlineai {")
        assert text.rstrip().endswith("}")
        assert "->" in text

    def test_an_empty_package_still_produces_a_root(self) -> None:
        node = build_graph(manifest())
        assert node.name == "demo 1.0.0"
        assert node.children == []


class TestDiff:
    def test_identical_manifests_report_no_differences(self) -> None:
        result = diff_manifests(
            manifest(artifacts=SHARDED_MODEL), manifest(artifacts=SHARDED_MODEL)
        )
        assert result.identical

    def test_added_and_removed_artifacts(self) -> None:
        before = manifest(artifacts=[artifact(id="a", path="artifacts/misc/a")])
        after = manifest(artifacts=[artifact(id="b", path="artifacts/misc/b")])
        result = diff_manifests(before, after)
        assert [a.path for a in result.added] == ["artifacts/misc/b"]
        assert [a.path for a in result.removed] == ["artifacts/misc/a"]

    def test_changed_content_is_detected(self) -> None:
        before = manifest(artifacts=[artifact(sha256="a" * 64)])
        after = manifest(artifacts=[artifact(sha256="b" * 64)])
        result = diff_manifests(before, after)
        assert len(result.changed) == 1
        assert "content" in result.changed[0].what_changed

    def test_a_changed_image_digest_is_reported(self) -> None:
        common = {
            "id": "image",
            "type": "oci-image",
            "path": "artifacts/containers/v.tar",
        }
        before = manifest(artifacts=[artifact(**common, digest="sha256:" + "a" * 64)])
        after = manifest(artifacts=[artifact(**common, digest="sha256:" + "b" * 64)])
        result = diff_manifests(before, after)
        assert "origin digest" in result.changed[0].what_changed

    def test_python_version_changes_are_spelled_out(self) -> None:
        common = {"id": "w", "type": "python-wheel", "path": "artifacts/python/wheels/t.whl"}
        before = manifest(
            artifacts=[artifact(**common, sha256="a" * 64, metadata={"version": "4.44.0"})]
        )
        after = manifest(
            artifacts=[artifact(**common, sha256="b" * 64, metadata={"version": "5.0.0"})]
        )
        result = diff_manifests(before, after)
        assert any("4.44.0 -> 5.0.0" in c for c in result.changed[0].what_changed)

    def test_requirement_changes_are_reported_even_with_identical_artifacts(self) -> None:
        """A bundle that quietly starts demanding 80 GB of VRAM instead of 48 is
        a deployment-blocking change that no artifact diff would show."""
        before = manifest(requirements={"gpu": {"vendor": "nvidia", "minimumMemoryGB": 48}})
        after = manifest(requirements={"gpu": {"vendor": "nvidia", "minimumMemoryGB": 80}})
        result = diff_manifests(before, after)
        assert not result.added and not result.removed and not result.changed
        assert not result.identical
        assert any("48 -> 80" in c for c in result.requirements_changed)

    def test_an_added_gpu_requirement_is_reported(self) -> None:
        result = diff_manifests(manifest(), manifest(requirements={"gpu": {"vendor": "nvidia"}}))
        assert any("GPU requirement: added" in c for c in result.requirements_changed)

    def test_new_required_secrets_are_reported(self) -> None:
        result = diff_manifests(manifest(), manifest(requiredSecrets=["NEW_TOKEN"]))
        assert any("NEW_TOKEN" in c for c in result.requirements_changed)

    def test_size_delta(self) -> None:
        before = manifest(artifacts=[artifact(size=1000)])
        after = manifest(artifacts=[artifact(size=2500, sha256="b" * 64)])
        result = diff_manifests(before, after)
        assert result.size_delta == 1500

    def test_artifacts_are_matched_by_path_not_id(self) -> None:
        """Ids derive from source references and can churn without a real
        change; the path is the stable identity."""
        before = manifest(artifacts=[artifact(id="old-id", path="artifacts/misc/x")])
        after = manifest(artifacts=[artifact(id="new-id", path="artifacts/misc/x")])
        result = diff_manifests(before, after)
        assert result.identical, "an id change alone is not a difference"


class TestPluginDiscovery:
    def test_the_builtin_sources_are_discovered(self) -> None:
        loaded = load_sources()
        for expected in ("local", "http", "huggingface", "oci"):
            assert expected in loaded.sources, f"{expected} source was not discovered"

    def test_the_entry_point_group_is_the_documented_one(self) -> None:
        assert ENTRY_POINT_GROUP == "offlineai.sources"

    def test_a_broken_plugin_is_reported_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One broken third-party source must not stop an operator installing
        an unrelated package on a machine where they cannot uninstall it."""

        class Exploding:
            name = "exploding"

            def load(self) -> Any:
                raise ImportError("no module named 'nonexistent'")

        class Fine:
            name = "fine"

            def load(self) -> Any:
                return object

        monkeypatch.setattr(
            "offlineai.plugins.loader.entry_points",
            lambda **_: [Exploding(), Fine()],
        )
        loaded = load_sources()
        assert "fine" in loaded.sources
        assert "exploding" in loaded.failures
        assert "nonexistent" in loaded.failures["exploding"]

    def test_a_broken_environment_is_survivable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(**kwargs: object) -> Any:
            raise RuntimeError("metadata is unreadable")

        monkeypatch.setattr("offlineai.plugins.loader.entry_points", explode)
        assert load_sources().sources == {}
