"""The .offlineai container format (section 9).

The property that matters most here is member ordering. Metadata is written
first and artifacts last, so `inspect` and `verify-signature` read a few KB off
the front of a bundle instead of scanning it. Section 60 requires inspect to
work without installing anything; on a 62 GB bundle that is only affordable if
the read really does stop early - so that is asserted directly, by counting
bytes pulled off the underlying file.
"""

from __future__ import annotations

import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from offlineai import layout
from offlineai.bundler.archive import BundleReader, BundleWriter
from offlineai.errors import MissingArtifactError, UnsafeArchiveError
from offlineai.schema.manifest import (
    FORMAT_VERSION,
    ArtifactEntry,
    ArtifactType,
    Compression,
    Manifest,
)
from offlineai.utils.hashing import sha256_bytes

MODEL_BYTES = b"pretend safetensors payload " * 4000
IMAGE_BYTES = b"pretend docker image layer " * 4000


@pytest.fixture
def artifact_files(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "src" / "model.safetensors"
    image = tmp_path / "src" / "vllm.tar"
    model.parent.mkdir(parents=True)
    model.write_bytes(MODEL_BYTES)
    image.write_bytes(IMAGE_BYTES)
    return {"model": model, "image": image}


def build_manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "formatVersion": FORMAT_VERSION,
            "package": {"name": "demo", "version": "1.0.0"},
            "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
            "platforms": ["linux/amd64"],
            "artifacts": [
                {
                    "id": "model-demo",
                    "type": "model",
                    "path": "artifacts/models/demo/model.safetensors",
                    "size": len(MODEL_BYTES),
                    "sha256": sha256_bytes(MODEL_BYTES),
                },
                {
                    "id": "image-vllm",
                    "type": "oci-image",
                    "path": "artifacts/containers/vllm.tar",
                    "size": len(IMAGE_BYTES),
                    "sha256": sha256_bytes(IMAGE_BYTES),
                },
            ],
        }
    )


def write_bundle(
    path: Path,
    artifact_files: dict[str, Path],
    *,
    compression: Compression = Compression.NONE,
) -> Manifest:
    manifest = build_manifest()
    with BundleWriter(path, compression=compression) as writer:
        writer.write_header(manifest=manifest, package_yaml=b"kind: Package\n")
        writer.add_artifact("artifacts/models/demo/model.safetensors", artifact_files["model"])
        writer.add_artifact("artifacts/containers/vllm.tar", artifact_files["image"])
    return manifest


class TestOrdering:
    def test_metadata_precedes_artifacts(self, tmp_path: Path, artifact_files: dict) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files)

        names = BundleReader.member_names(bundle)
        first_artifact = next(i for i, n in enumerate(names) if n.startswith("artifacts/"))
        assert names[0] == layout.MANIFEST, "manifest must be the very first member"
        assert all(not n.startswith("artifacts/") for n in names[:first_artifact])
        for header in (layout.PACKAGE, layout.CHECKSUMS):
            assert names.index(header) < first_artifact

    def test_header_cannot_be_written_after_an_artifact(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        # abort() rather than close(): this bundle is deliberately incomplete,
        # and close() would fail it on the completeness check instead.
        manifest = build_manifest()
        writer = BundleWriter(tmp_path / "b.offlineai")
        try:
            writer.write_header(manifest=manifest, package_yaml=b"kind: Package\n")
            writer.add_artifact("artifacts/models/demo/model.safetensors", artifact_files["model"])
            with pytest.raises(RuntimeError, match="header"):
                writer.write_header(manifest=manifest, package_yaml=b"x")
        finally:
            writer.abort()

    def test_artifact_cannot_be_written_before_the_header(self, tmp_path: Path) -> None:
        writer = BundleWriter(tmp_path / "b.offlineai")
        try:
            with pytest.raises(RuntimeError, match="write_header"):
                writer.add_artifact("artifacts/x", Path("/dev/null"))
        finally:
            writer.abort()

    def test_abort_leaves_no_bundle_behind(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b.offlineai"
        writer = BundleWriter(bundle)
        writer.write_header(manifest=build_manifest(), package_yaml=b"kind: Package\n")
        writer.abort()
        assert not bundle.exists()


class TestHeaderOnlyRead:
    """The reason the ordering exists."""

    def test_reading_the_header_does_not_read_the_artifacts(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files)
        total = bundle.stat().st_size
        assert total > 200_000, "fixture must be big enough for the claim to mean something"

        header, consumed = BundleReader.read_header_counting_bytes(bundle)

        assert header.manifest.package.name == "demo"
        assert consumed < total // 4, (
            f"header read consumed {consumed} of {total} bytes; the whole point of "
            "putting metadata first is that this stays small"
        )

    def test_header_exposes_manifest_and_package(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files)
        with BundleReader.open(bundle) as reader:
            header = reader.read_header()
        assert header.manifest.total_size == len(MODEL_BYTES) + len(IMAGE_BYTES)
        assert header.package_yaml == b"kind: Package\n"

    def test_checksums_file_lists_every_artifact(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files)
        with BundleReader.open(bundle) as reader:
            checksums = reader.read_header().checksums
        assert f"{sha256_bytes(MODEL_BYTES)}  artifacts/models/demo/model.safetensors" in checksums
        assert f"{sha256_bytes(IMAGE_BYTES)}  artifacts/containers/vllm.tar" in checksums


class TestArtifactStreaming:
    def test_artifacts_stream_back_with_their_manifest_entry(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files)

        recovered: dict[str, bytes] = {}
        with BundleReader.open(bundle) as reader:
            reader.read_header()
            for entry, stream in reader.iter_artifacts():
                assert isinstance(entry, ArtifactEntry)
                recovered[entry.path] = stream.read()

        assert recovered["artifacts/models/demo/model.safetensors"] == MODEL_BYTES
        assert recovered["artifacts/containers/vllm.tar"] == IMAGE_BYTES

    def test_artifact_not_described_by_the_manifest_is_rejected(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        """A bundle carrying an undeclared payload is not a bundle we trust."""
        bundle = tmp_path / "demo.offlineai"
        manifest = build_manifest()
        with BundleWriter(bundle) as writer:
            writer.write_header(manifest=manifest, package_yaml=b"kind: Package\n")
            writer.add_artifact("artifacts/models/demo/model.safetensors", artifact_files["model"])
            writer.add_artifact("artifacts/containers/vllm.tar", artifact_files["image"])
            writer._tar.add(artifact_files["model"], arcname="artifacts/stowaway.bin")

        with BundleReader.open(bundle) as reader:
            reader.read_header()
            with pytest.raises(UnsafeArchiveError, match="stowaway"):
                list(reader.iter_artifacts())


class TestCompletenessInvariant:
    """Section 74, enforced structurally: a build that succeeds must have
    written every artifact the manifest declares."""

    def test_closing_with_a_declared_artifact_missing_fails(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        writer_ctx = BundleWriter(bundle)
        with pytest.raises(MissingArtifactError, match="image-vllm"), writer_ctx as writer:
            writer.write_header(manifest=build_manifest(), package_yaml=b"kind: Package\n")
            writer.add_artifact("artifacts/models/demo/model.safetensors", artifact_files["model"])
            # image-vllm is declared but never added.

    def test_incomplete_bundle_file_is_removed(self, tmp_path: Path, artifact_files: dict) -> None:
        bundle = tmp_path / "demo.offlineai"
        with pytest.raises(MissingArtifactError), BundleWriter(bundle) as writer:
            writer.write_header(manifest=build_manifest(), package_yaml=b"kind: Package\n")
        assert not bundle.exists(), "a failed build must not leave a bundle behind"

    def test_content_not_matching_the_manifest_hash_is_rejected_at_write_time(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        tampered = tmp_path / "tampered.safetensors"
        tampered.write_bytes(b"different content entirely")
        from offlineai.errors import ChecksumMismatchError

        with pytest.raises(ChecksumMismatchError), BundleWriter(tmp_path / "b.offlineai") as w:
            w.write_header(manifest=build_manifest(), package_yaml=b"kind: Package\n")
            w.add_artifact("artifacts/models/demo/model.safetensors", tampered)


class TestCompression:
    def test_gzip_round_trips(self, tmp_path: Path, artifact_files: dict) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files, compression=Compression.GZIP)
        with BundleReader.open(bundle) as reader:
            header = reader.read_header()
            assert header.manifest.compression is Compression.GZIP
            payloads = {e.path: s.read() for e, s in reader.iter_artifacts()}
        assert payloads["artifacts/models/demo/model.safetensors"] == MODEL_BYTES

    def test_uncompressed_is_the_default(self, tmp_path: Path, artifact_files: dict) -> None:
        manifest = write_bundle(tmp_path / "demo.offlineai", artifact_files)
        assert manifest.compression is Compression.NONE

    def test_compression_is_detected_without_being_told(
        self, tmp_path: Path, artifact_files: dict
    ) -> None:
        bundle = tmp_path / "demo.offlineai"
        write_bundle(bundle, artifact_files, compression=Compression.GZIP)
        assert BundleReader.member_names(bundle)[0] == layout.MANIFEST


class TestLayoutHelpers:
    def test_header_members_are_recognised(self) -> None:
        assert layout.is_header_member(layout.MANIFEST)
        assert layout.is_header_member("docs/README.md")
        assert not layout.is_header_member("artifacts/models/x")

    def test_artifact_paths_are_built_under_the_right_subtree(self) -> None:
        assert layout.artifact_path(ArtifactType.MODEL, "qwen3", "config.json") == (
            "artifacts/models/qwen3/config.json"
        )
        assert layout.artifact_path(ArtifactType.OCI_IMAGE, "vllm.tar") == (
            "artifacts/containers/vllm.tar"
        )
        assert layout.artifact_path(ArtifactType.PYTHON_WHEEL, "fastapi.whl") == (
            "artifacts/python/wheels/fastapi.whl"
        )


class TestEmptyAndHeaderOnlyBundles:
    """A bundle with no artifacts is legitimate (a package that only declares
    containers, before those are packaged). The reader must not walk off the
    end of the stream, which cannot seek backwards to recover."""

    def _empty_manifest(self) -> Manifest:
        return Manifest.model_validate(
            {
                "formatVersion": FORMAT_VERSION,
                "package": {"name": "empty", "version": "1.0.0"},
                "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
                "platforms": ["linux/amd64"],
                "artifacts": [],
            }
        )

    def test_bundle_with_no_artifacts_round_trips(self, tmp_path: Path) -> None:
        bundle = tmp_path / "empty.offlineai"
        with BundleWriter(bundle) as writer:
            writer.write_header(manifest=self._empty_manifest(), package_yaml=b"kind: Package\n")

        with BundleReader.open(bundle) as reader:
            header = reader.read_header()
            assert header.manifest.package.name == "empty"
            assert list(reader.iter_artifacts()) == []

    def test_iter_artifacts_can_be_called_twice_without_seeking_back(self, tmp_path: Path) -> None:
        bundle = tmp_path / "empty.offlineai"
        with BundleWriter(bundle) as writer:
            writer.write_header(manifest=self._empty_manifest(), package_yaml=b"kind: Package\n")
        with BundleReader.open(bundle) as reader:
            reader.read_header()
            assert list(reader.iter_artifacts()) == []
            assert list(reader.iter_artifacts()) == []


class TestLargeMemberSupport:
    """A single safetensors shard routinely exceeds 8 GiB.

    That is the old ustar limit, and a bundle writer using it would fail with
    "overflow in number field" on exactly the models this tool exists for. The
    code uses PAX, which handles it, and PAX is also Python's default - but an
    invariant this important should not rest on a default staying put.

    Asserted on the encoded header rather than by writing the bytes: the point
    is the size field, and materialising 10 GiB to prove it would be absurd.
    """

    #: Comfortably past the ustar ceiling of 8 GiB.
    HUGE = 10 * 1024**3

    def test_the_writer_uses_a_format_that_permits_huge_members(self, tmp_path: Path) -> None:
        writer = BundleWriter(tmp_path / "b.offlineai")
        try:
            assert writer._tar.format == tarfile.PAX_FORMAT
        finally:
            writer.abort()

    def test_a_ten_gibibyte_member_encodes(self) -> None:
        info = tarfile.TarInfo("model-00001-of-00008.safetensors")
        info.size = self.HUGE
        assert info.tobuf(tarfile.PAX_FORMAT)

    def test_the_old_format_would_have_refused_it(self) -> None:
        """Establishes that the previous assertion is meaningful."""
        info = tarfile.TarInfo("model-00001-of-00008.safetensors")
        info.size = self.HUGE
        with pytest.raises(ValueError, match="overflow"):
            info.tobuf(tarfile.USTAR_FORMAT)

    def test_a_huge_size_round_trips_through_the_manifest(self) -> None:
        """The manifest must also carry a size past 2^32 without truncating."""
        manifest = Manifest.model_validate(
            {
                "formatVersion": FORMAT_VERSION,
                "package": {"name": "big", "version": "1.0.0"},
                "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
                "platforms": ["linux/amd64"],
                "artifacts": [
                    {
                        "id": "shard",
                        "type": "model",
                        "path": "artifacts/models/m/shard.safetensors",
                        "size": self.HUGE,
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
        restored = Manifest.from_yaml(manifest.to_yaml())
        assert restored.artifacts[0].size == self.HUGE
        assert restored.total_size == self.HUGE
