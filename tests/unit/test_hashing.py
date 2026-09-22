"""SHA-256 helpers. Everything in a bundle is addressed by these digests."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from offlineai.errors import ChecksumMismatchError
from offlineai.utils.hashing import (
    CHUNK_SIZE,
    HashingReader,
    copy_and_hash,
    sha256_bytes,
    sha256_file,
    sha256_stream,
    verify_file,
)


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, str]:
    payload = b"the quick brown fox" * 5000
    path = tmp_path / "sample.bin"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


class TestDigests:
    def test_sha256_bytes(self) -> None:
        assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()

    def test_sha256_file(self, sample: tuple[Path, str]) -> None:
        path, expected = sample
        assert sha256_file(path) == expected

    def test_sha256_stream(self, sample: tuple[Path, str]) -> None:
        path, expected = sample
        with path.open("rb") as handle:
            assert sha256_stream(handle) == expected

    def test_empty_file_hashes_to_the_empty_digest(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.touch()
        assert sha256_file(empty) == hashlib.sha256(b"").hexdigest()

    def test_spans_many_chunks(self, tmp_path: Path) -> None:
        payload = bytes(range(256)) * (CHUNK_SIZE // 64)
        path = tmp_path / "big.bin"
        path.write_bytes(payload)
        assert len(payload) > CHUNK_SIZE * 2, "fixture must exercise the read loop"
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestCopyAndHash:
    """Import streams artifacts into content-addressed storage. Hashing has to
    happen during the copy; a second pass would double the I/O on a 62 GB
    bundle and require the data to still be available."""

    def test_copies_bytes_and_returns_digest_and_size(self, tmp_path: Path) -> None:
        payload = b"model weights" * 1000
        dest = tmp_path / "out.bin"
        digest, size = copy_and_hash(io.BytesIO(payload), dest)
        assert dest.read_bytes() == payload
        assert digest == hashlib.sha256(payload).hexdigest()
        assert size == len(payload)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c.bin"
        copy_and_hash(io.BytesIO(b"x"), dest)
        assert dest.exists()

    def test_enforces_expected_size(self, tmp_path: Path) -> None:
        with pytest.raises(ChecksumMismatchError):
            copy_and_hash(io.BytesIO(b"short"), tmp_path / "o.bin", expected_size=999)

    def test_enforces_expected_digest(self, tmp_path: Path) -> None:
        with pytest.raises(ChecksumMismatchError) as excinfo:
            copy_and_hash(io.BytesIO(b"payload"), tmp_path / "o.bin", expected_sha256="0" * 64)
        assert excinfo.value.expected == "0" * 64
        assert excinfo.value.actual == sha256_bytes(b"payload")

    def test_removes_the_partial_file_on_mismatch(self, tmp_path: Path) -> None:
        dest = tmp_path / "o.bin"
        with pytest.raises(ChecksumMismatchError):
            copy_and_hash(io.BytesIO(b"payload"), dest, expected_sha256="0" * 64)
        assert not dest.exists(), "a file that failed verification must not be left behind"

    def test_accepts_a_matching_digest(self, tmp_path: Path) -> None:
        payload = b"payload"
        digest, size = copy_and_hash(
            io.BytesIO(payload), tmp_path / "o.bin", expected_sha256=sha256_bytes(payload)
        )
        assert (digest, size) == (sha256_bytes(payload), len(payload))


class TestHashingReader:
    """Used when the data must flow somewhere we do not control (a tar writer),
    so we cannot drive the copy loop ourselves."""

    def test_digest_matches_after_full_read(self) -> None:
        payload = b"stream me" * 500
        reader = HashingReader(io.BytesIO(payload))
        assert reader.read() == payload
        assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()
        assert reader.bytes_read == len(payload)

    def test_digest_matches_across_partial_reads(self) -> None:
        payload = b"abcdefghij" * 100
        reader = HashingReader(io.BytesIO(payload))
        chunks = []
        while chunk := reader.read(7):
            chunks.append(chunk)
        assert b"".join(chunks) == payload
        assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()


class TestVerifyFile:
    def test_passes_on_match(self, sample: tuple[Path, str]) -> None:
        path, expected = sample
        verify_file(path, expected)

    def test_raises_with_the_path_on_mismatch(self, sample: tuple[Path, str]) -> None:
        path, _ = sample
        with pytest.raises(ChecksumMismatchError) as excinfo:
            verify_file(path, "0" * 64, label="artifacts/models/x.safetensors")
        assert excinfo.value.path == "artifacts/models/x.safetensors"

    def test_digest_comparison_is_case_insensitive(self, sample: tuple[Path, str]) -> None:
        path, expected = sample
        verify_file(path, expected.upper())
