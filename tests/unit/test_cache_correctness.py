"""Two cache bugs the lock-file work uncovered.

Both were silent, both produced wrong bundles, and neither had a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offlineai.artifacts.base import SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.sources.local import LocalSource
from offlineai.schema.manifest import ArtifactType


@pytest.fixture
def cache(tmp_path: Path) -> ArtifactCache:
    store = ArtifactCache(tmp_path / "cache")
    store.ensure()
    return store


class TestEditingALocalFileIsNoticed:
    """A local file's cache key is its path, and a path does not change when
    its content does. Keying the reference index by it meant a developer could
    edit a model, rebuild, and silently ship the previous weights.
    """

    def _ref(self) -> SourceRef:
        return SourceRef(
            kind="local",
            locator="./weights",
            name="demo",
            artifact_type=ArtifactType.MODEL,
        )

    def test_a_rebuild_picks_up_changed_content(self, cache: ArtifactCache, tmp_path: Path) -> None:
        weights = tmp_path / "weights"
        weights.mkdir()
        model = weights / "model.safetensors"
        model.write_bytes(b"ORIGINAL weights")

        source = LocalSource(tmp_path)
        first = source.fetch(source.expand(self._ref())[0], cache)

        model.write_bytes(b"MODIFIED weights")
        second = source.fetch(source.expand(self._ref())[0], cache)

        assert second.sha256 != first.sha256, (
            "the rebuild returned the original digest; the edit was ignored"
        )
        assert second.local_path.read_bytes() == b"MODIFIED weights"

    def test_local_files_are_not_added_to_the_reference_index(
        self, cache: ArtifactCache, tmp_path: Path
    ) -> None:
        """The index is what made the staleness possible. store_file already
        deduplicates by content, so the index adds nothing here."""
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "model.bin").write_bytes(b"content")

        source = LocalSource(tmp_path)
        request = source.expand(self._ref())[0]
        source.fetch(request, cache)

        assert cache.lookup_ref("local", request.cache_key) is None

    def test_identical_content_is_still_stored_once(
        self, cache: ArtifactCache, tmp_path: Path
    ) -> None:
        """Dropping the index must not cost deduplication."""
        for name in ("a", "b"):
            directory = tmp_path / name
            directory.mkdir()
            (directory / "model.bin").write_bytes(b"identical bytes")

        stored = [
            LocalSource(tmp_path).fetch(
                LocalSource(tmp_path).expand(
                    SourceRef(
                        kind="local",
                        locator=f"./{name}",
                        name=name,
                        artifact_type=ArtifactType.MODEL,
                    )
                )[0],
                cache,
            )
            for name in ("a", "b")
        ]
        assert stored[0].local_path == stored[1].local_path


class TestCacheHitsKeepProvenance:
    """A cache hit used to return content with no source and no origin digest,
    so section 16's digest recording and section 34's provenance quietly
    stopped working on the *second* build - which is the common case.
    """

    def test_a_reference_round_trips_its_provenance(self, cache: ArtifactCache) -> None:
        payload = b"an artifact"
        entry = cache.store_file(_write(cache.root.parent / "in.bin", payload))
        cache.put_ref(
            "oci",
            "vllm/vllm-openai:v0.6.3",
            entry.sha256,
            source="vllm/vllm-openai:v0.6.3",
            digest="sha256:" + "a" * 64,
            license_id="Apache-2.0",
        )

        hit = cache.lookup_ref("oci", "vllm/vllm-openai:v0.6.3")
        assert hit is not None
        assert hit.source == "vllm/vllm-openai:v0.6.3"
        assert hit.digest == "sha256:" + "a" * 64
        assert hit.license == "Apache-2.0"

    def test_a_reference_without_provenance_is_still_usable(self, cache: ArtifactCache) -> None:
        entry = cache.store_file(_write(cache.root.parent / "in.bin", b"x"))
        cache.put_ref("http", "https://example/x", entry.sha256)
        hit = cache.lookup_ref("http", "https://example/x")
        assert hit is not None
        assert hit.sha256 == entry.sha256
        assert hit.source is None

    def test_a_reference_written_by_an_older_version_still_reads(
        self, cache: ArtifactCache
    ) -> None:
        """Earlier releases wrote a bare digest. An upgrade must not invalidate
        a populated cache - on a metered or air-gapped builder, re-downloading
        everything is a real cost."""
        entry = cache.store_file(_write(cache.root.parent / "in.bin", b"legacy"))
        ref = cache._ref_path("http", "https://example/legacy")
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(entry.sha256)

        hit = cache.lookup_ref("http", "https://example/legacy")
        assert hit is not None
        assert hit.sha256 == entry.sha256

    def test_a_dangling_reference_is_cleaned_up(self, cache: ArtifactCache) -> None:
        cache.put_ref("http", "https://example/gone", "f" * 64)
        assert cache.lookup_ref("http", "https://example/gone") is None
        assert not cache._ref_path("http", "https://example/gone").is_file()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
