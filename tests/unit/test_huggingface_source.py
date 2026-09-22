"""Section 13: a model is not one file.

Packaging only the weights yields a bundle that installs cleanly and then
fails at first inference on a machine where nothing can be downloaded to fix
it. These tests pin the enumeration rules that prevent that.

All offline: the Hugging Face API is a stub and downloads go through an httpx
MockTransport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from offlineai.artifacts.base import SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.sources.huggingface import HuggingFaceSource, RepoFile
from offlineai.errors import SourceError
from offlineai.schema.manifest import ArtifactType
from offlineai.utils.hashing import sha256_bytes


@dataclass
class FakeLfs:
    sha256: str | None


@dataclass
class FakeSibling:
    rfilename: str
    size: int | None = None
    lfs: FakeLfs | None = None


class FakeRepoInfo:
    def __init__(self, siblings: list[FakeSibling], license_id: str | None = None) -> None:
        self.siblings = siblings
        self.cardData = {"license": license_id} if license_id else {}


class FakeApi:
    """Stands in for HfApi. Records calls so tests can assert on them."""

    def __init__(self, siblings: list[FakeSibling], license_id: str | None = None) -> None:
        self._siblings = siblings
        self._license = license_id
        self.calls: list[tuple[str, str]] = []

    def repo_info(self, repo_id: str, revision: str = "main", **kwargs: object) -> FakeRepoInfo:
        self.calls.append((repo_id, revision))
        return FakeRepoInfo(self._siblings, self._license)


def sharded_repo(shards: int = 3) -> list[FakeSibling]:
    """A realistic checkpoint: config, tokenizer, index and N weight shards."""
    files = [
        FakeSibling("config.json", 512),
        FakeSibling("generation_config.json", 128),
        FakeSibling("tokenizer.json", 4096),
        FakeSibling("tokenizer_config.json", 256),
        FakeSibling("special_tokens_map.json", 128),
        FakeSibling("model.safetensors.index.json", 1024),
        FakeSibling("README.md", 2048),
        FakeSibling(".gitattributes", 64),
        # A duplicate weight format the default excludes should drop.
        FakeSibling("pytorch_model-00001-of-00003.bin", 10_000_000),
    ]
    files += [
        FakeSibling(
            f"model-{i:05d}-of-{shards:05d}.safetensors",
            10_000_000,
            FakeLfs("a" * 64),
        )
        for i in range(1, shards + 1)
    ]
    return files


def ref(**options: object) -> SourceRef:
    return SourceRef(
        kind="huggingface",
        locator="Qwen/Qwen3-30B",
        name="qwen3",
        artifact_type=ArtifactType.MODEL,
        options=options,
    )


class TestEnumeration:
    def test_every_shard_is_included(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(8)))
        paths = {r.metadata["file"] for r in source.expand(ref())}
        for i in range(1, 9):
            assert f"model-{i:05d}-of-00008.safetensors" in paths

    def test_auxiliary_files_are_included(self) -> None:
        """Weights alone are not a usable model."""
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        paths = {r.metadata["file"] for r in source.expand(ref())}
        for required in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
        ):
            assert required in paths, f"{required} must be packaged"

    def test_duplicate_weight_formats_are_excluded_by_default(self) -> None:
        """A repo shipping both .safetensors and .bin would otherwise double
        the bundle for no benefit."""
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        paths = {r.metadata["file"] for r in source.expand(ref())}
        assert not any(p.endswith(".bin") for p in paths)

    def test_repository_furniture_is_excluded(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        paths = {r.metadata["file"] for r in source.expand(ref())}
        assert "README.md" not in paths
        assert ".gitattributes" not in paths

    def test_explicit_include_narrows_the_selection(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        paths = {r.metadata["file"] for r in source.expand(ref(include=["config.json"]))}
        assert paths == {"config.json"}

    def test_explicit_exclude_replaces_the_defaults(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        paths = {r.metadata["file"] for r in source.expand(ref(exclude=["*.bin"]))}
        assert "README.md" in paths, "an explicit exclude list replaces the defaults"
        assert not any(p.endswith(".bin") for p in paths)

    def test_revision_is_honoured(self) -> None:
        api = FakeApi(sharded_repo())
        HuggingFaceSource(api=api).expand(ref(revision="refs/pr/3"))
        assert api.calls[-1] == ("Qwen/Qwen3-30B", "refs/pr/3")

    def test_bundle_paths_are_namespaced_by_model_name(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(1)))
        requests = source.expand(ref())
        assert all(r.bundle_path.startswith("artifacts/models/qwen3/") for r in requests)

    def test_lfs_digests_are_carried_for_pre_verification(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(1)))
        weights = next(
            r for r in source.expand(ref()) if r.metadata["file"].endswith(".safetensors")
        )
        assert weights.expected_sha256 == "a" * 64
        assert weights.expected_size == 10_000_000


class TestIncompleteShardDetection:
    """A missing shard makes a bundle useless, and the failure would otherwise
    appear on the air-gapped side at first load."""

    def test_a_missing_shard_fails_the_build(self) -> None:
        files = [f for f in sharded_repo(4) if f.rfilename != "model-00003-of-00004.safetensors"]
        source = HuggingFaceSource(api=FakeApi(files))
        with pytest.raises(SourceError, match="incomplete"):
            source.expand(ref())

    def test_the_error_names_the_missing_shard(self) -> None:
        files = [f for f in sharded_repo(4) if f.rfilename != "model-00002-of-00004.safetensors"]
        source = HuggingFaceSource(api=FakeApi(files))
        with pytest.raises(SourceError) as excinfo:
            source.expand(ref())
        assert "00002-of-00004" in excinfo.value.render()

    def test_a_complete_checkpoint_passes(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(8)))
        assert len(source.expand(ref())) > 8

    def test_an_include_filter_that_drops_shards_is_caught(self) -> None:
        """Selecting only some shards is a mistake, not a feature."""
        source = HuggingFaceSource(api=FakeApi(sharded_repo(4)))
        with pytest.raises(SourceError, match="incomplete"):
            source.expand(ref(include=["model-00001-of-00004.safetensors"]))


class TestErrors:
    def test_an_empty_repository_is_reported(self) -> None:
        source = HuggingFaceSource(api=FakeApi([]))
        with pytest.raises(SourceError, match="no files"):
            source.expand(ref())

    def test_patterns_matching_nothing_are_reported(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo()))
        with pytest.raises(SourceError, match="matched"):
            source.expand(ref(include=["nothing-like-this"]))

    def test_an_api_failure_suggests_the_token(self) -> None:
        class Failing:
            def repo_info(self, **kwargs: object) -> object:
                raise RuntimeError("401 Unauthorized")

        source = HuggingFaceSource(api=Failing())
        with pytest.raises(SourceError) as excinfo:
            source.expand(ref())
        assert "HF_TOKEN" in excinfo.value.render()


class TestFetching:
    """Fixtures here are self-consistent: the sizes and digests the fake API
    reports match the bytes the mock transport serves. Otherwise the size check
    fires first and masks whatever the test meant to exercise."""

    CONFIG = b'{"model_type": "demo"}'
    WEIGHTS = b"safetensors payload" * 100

    def _source(self, handler: object) -> tuple[HuggingFaceSource, list[FakeSibling]]:
        siblings = [
            FakeSibling("config.json", len(self.CONFIG)),
            FakeSibling(
                "model-00001-of-00001.safetensors",
                len(self.WEIGHTS),
                FakeLfs(sha256_bytes(self.WEIGHTS)),
            ),
        ]
        client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
        return HuggingFaceSource(api=FakeApi(siblings), client=client), siblings

    def test_downloads_and_caches_a_file(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "Qwen/Qwen3-30B/resolve/main/config.json" in str(request.url)
            return httpx.Response(200, content=self.CONFIG)

        source, _ = self._source(handler)
        cache = ArtifactCache(tmp_path / "cache")
        request = next(r for r in source.expand(ref()) if r.metadata["file"] == "config.json")
        resolved = source.fetch(request, cache)

        assert resolved.sha256 == sha256_bytes(self.CONFIG)
        assert resolved.local_path.read_bytes() == self.CONFIG
        assert resolved.source == "hf://Qwen/Qwen3-30B@main/config.json"

    def test_a_declared_digest_is_verified_on_arrival(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=self.WEIGHTS)

        source, _ = self._source(handler)
        request = next(
            r for r in source.expand(ref()) if r.metadata["file"].endswith(".safetensors")
        )
        resolved = source.fetch(request, ArtifactCache(tmp_path / "cache"))
        assert resolved.sha256 == sha256_bytes(self.WEIGHTS)

    def test_a_token_is_sent_when_configured(self, tmp_path: Path) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, content=self.CONFIG)

        siblings = [FakeSibling("config.json", len(self.CONFIG))]
        client = httpx.Client(transport=httpx.MockTransport(handler))
        source = HuggingFaceSource(api=FakeApi(siblings), client=client, token="hf_secrettoken123")
        request = source.expand(ref())[0]
        source.fetch(request, ArtifactCache(tmp_path / "cache"))
        assert seen["authorization"] == "Bearer hf_secrettoken123"

    def test_content_not_matching_the_declared_digest_is_refused(self, tmp_path: Path) -> None:
        """Correct length, wrong bytes: the digest check is what must catch it."""
        corrupted = b"X" * len(self.WEIGHTS)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=corrupted)

        source, _ = self._source(handler)
        request = next(
            r for r in source.expand(ref()) if r.metadata["file"].endswith(".safetensors")
        )
        with pytest.raises(SourceError, match="digest"):
            source.fetch(request, ArtifactCache(tmp_path / "cache"))

    def test_a_truncated_response_is_refused(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=self.WEIGHTS[:50])

        source, _ = self._source(handler)
        request = next(
            r for r in source.expand(ref()) if r.metadata["file"].endswith(".safetensors")
        )
        with pytest.raises(SourceError, match="size"):
            source.fetch(request, ArtifactCache(tmp_path / "cache"))

    def test_a_failed_download_leaves_nothing_in_the_cache(self, tmp_path: Path) -> None:
        corrupted = b"X" * len(self.WEIGHTS)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=corrupted)

        source, _ = self._source(handler)
        cache = ArtifactCache(tmp_path / "cache")
        request = next(
            r for r in source.expand(ref()) if r.metadata["file"].endswith(".safetensors")
        )
        with pytest.raises(SourceError):
            source.fetch(request, cache)
        assert cache.lookup_ref("huggingface", request.cache_key) is None
        assert not cache.has(sha256_bytes(corrupted))

    def test_a_second_fetch_hits_the_cache(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=self.CONFIG)

        source, _ = self._source(handler)
        cache = ArtifactCache(tmp_path / "cache")
        request = next(r for r in source.expand(ref()) if r.metadata["file"] == "config.json")
        source.fetch(request, cache)
        assert cache.lookup_ref("huggingface", request.cache_key) is not None
        assert calls["n"] == 1


class TestLicenseMetadata:
    def test_license_is_read_when_present(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(1), license_id="apache-2.0"))
        assert source.license_for("Qwen/Qwen3-30B") == "apache-2.0"

    def test_a_missing_license_is_not_an_error(self) -> None:
        """Section 37: license metadata is informational, never a gate."""
        source = HuggingFaceSource(api=FakeApi(sharded_repo(1)))
        assert source.license_for("Qwen/Qwen3-30B") is None


class TestFileListing:
    def test_sizes_and_digests_are_surfaced(self) -> None:
        source = HuggingFaceSource(api=FakeApi(sharded_repo(1)))
        files = source.list_files("Qwen/Qwen3-30B", "main")
        weights = next(f for f in files if f.path.endswith(".safetensors"))
        assert isinstance(weights, RepoFile)
        assert weights.size == 10_000_000
        assert weights.sha256 == "a" * 64
