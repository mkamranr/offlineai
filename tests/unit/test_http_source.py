"""The plain HTTP artifact source.

Declared in the schema and wired into the builder, but until now exercised
only indirectly through the shared download code. Its own expansion logic -
URL validation, filename derivation, digest pass-through - had no tests.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from offlineai.artifacts.base import SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.sources.http import HttpSource
from offlineai.errors import SourceError
from offlineai.schema.manifest import ArtifactType
from offlineai.utils.hashing import sha256_bytes

PAYLOAD = b"a vendor-supplied artifact" * 200


def ref(locator: str, **options: object) -> SourceRef:
    return SourceRef(
        kind="http",
        locator=locator,
        name="vendor-model",
        artifact_type=ArtifactType.MODEL,
        options=options,
    )


def client_serving(payload: bytes = PAYLOAD, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestExpansion:
    def test_a_url_becomes_one_request(self) -> None:
        requests = HttpSource().expand(ref("https://vendor.internal/model.safetensors"))
        assert len(requests) == 1
        assert requests[0].locator == "https://vendor.internal/model.safetensors"

    def test_the_filename_comes_from_the_path(self) -> None:
        request = HttpSource().expand(ref("https://vendor.internal/a/b/weights.bin"))[0]
        assert request.bundle_path.endswith("/weights.bin")

    def test_models_are_namespaced_by_name(self) -> None:
        request = HttpSource().expand(ref("https://v.internal/w.bin"))[0]
        assert request.bundle_path == "artifacts/models/vendor-model/w.bin"

    def test_non_model_artifacts_are_not_namespaced(self) -> None:
        reference = SourceRef(
            kind="http",
            locator="https://v.internal/tool.tar",
            name="tool",
            artifact_type=ArtifactType.MISC,
        )
        request = HttpSource().expand(reference)[0]
        assert request.bundle_path == "artifacts/misc/tool.tar"

    def test_a_percent_encoded_name_is_decoded(self) -> None:
        request = HttpSource().expand(ref("https://v.internal/my%20model.bin"))[0]
        assert request.bundle_path.endswith("/my model.bin")

    def test_a_url_with_no_filename_still_produces_one(self) -> None:
        request = HttpSource().expand(ref("https://vendor.internal/"))[0]
        assert request.bundle_path.endswith("/download")

    def test_a_declared_digest_is_carried_for_verification(self) -> None:
        digest = "a" * 64
        request = HttpSource().expand(ref("https://v.internal/w.bin", sha256=digest))[0]
        assert request.expected_sha256 == digest

    @pytest.mark.parametrize(
        "locator",
        ["ftp://vendor.internal/x", "file:///etc/passwd", "vendor.internal/x", "s3://b/k"],
    )
    def test_non_http_schemes_are_refused(self, locator: str) -> None:
        with pytest.raises(SourceError, match="http"):
            HttpSource().expand(ref(locator))

    def test_the_refusal_shows_the_expected_shape(self) -> None:
        with pytest.raises(SourceError) as excinfo:
            HttpSource().expand(ref("vendor.internal/x"))
        assert "https://" in excinfo.value.render()


class TestFetching:
    def test_downloads_and_caches(self, tmp_path: Path) -> None:
        source = HttpSource(client=client_serving())
        request = source.expand(ref("https://v.internal/w.bin"))[0]
        resolved = source.fetch(request, ArtifactCache(tmp_path / "cache"))

        assert resolved.sha256 == sha256_bytes(PAYLOAD)
        assert resolved.local_path.read_bytes() == PAYLOAD
        assert resolved.source == "https://v.internal/w.bin"

    def test_a_declared_digest_is_enforced(self, tmp_path: Path) -> None:
        source = HttpSource(client=client_serving())
        request = source.expand(ref("https://v.internal/w.bin", sha256="0" * 64))[0]
        with pytest.raises(SourceError, match="digest"):
            source.fetch(request, ArtifactCache(tmp_path / "cache"))

    def test_a_matching_digest_passes(self, tmp_path: Path) -> None:
        source = HttpSource(client=client_serving())
        request = source.expand(ref("https://v.internal/w.bin", sha256=sha256_bytes(PAYLOAD)))[0]
        assert source.fetch(request, ArtifactCache(tmp_path / "cache")).size == len(PAYLOAD)

    def test_the_reference_index_is_populated(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path / "cache")
        source = HttpSource(client=client_serving())
        request = source.expand(ref("https://v.internal/w.bin"))[0]
        source.fetch(request, cache)
        assert cache.lookup_ref("http", request.cache_key) is not None

    def test_an_http_error_is_reported_with_advice(self, tmp_path: Path) -> None:
        source = HttpSource(client=client_serving(status=404))
        request = source.expand(ref("https://v.internal/w.bin"))[0]
        with pytest.raises(SourceError) as excinfo:
            source.fetch(request, ArtifactCache(tmp_path / "cache"))
        assert "does not exist" in excinfo.value.render()

    def test_progress_is_reported_when_a_reporter_is_given(self, tmp_path: Path) -> None:
        seen: list[tuple[int, int | None]] = []

        from offlineai.progress import NullReporter

        class Recorder(NullReporter):
            def update(self, key: str, completed: int, total: int | None = None) -> None:
                seen.append((completed, total))

        source = HttpSource(client=client_serving())
        request = source.expand(ref("https://v.internal/w.bin"))[0]
        source.fetch(request, ArtifactCache(tmp_path / "cache"), progress=Recorder())
        assert seen, "a byte-capable source must report progress"
        assert seen[-1][0] == len(PAYLOAD)


class TestBuilderIntegration:
    def test_the_builder_can_resolve_an_http_source(self, tmp_path: Path) -> None:
        """Wired into the source registry, not just importable."""
        from offlineai.bundler.builder import BundleBuilder
        from offlineai.config.settings import load_settings

        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        builder = BundleBuilder(settings)
        assert builder._source_for("http", tmp_path).kind == "http"
