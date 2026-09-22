"""SHA-256 helpers.

Every artifact in a bundle is addressed by a SHA-256 digest (section 11), so
these functions sit on the hot path for build, verify and import.

They are all streaming. A bundle can be hundreds of gigabytes - larger than
RAM and, on a constrained target, larger than the free disk - so nothing here
ever holds a whole artifact in memory or requires a second pass over the data.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import IO, Any

from offlineai.errors import ChecksumMismatchError

__all__ = [
    "CHUNK_SIZE",
    "HashingReader",
    "copy_and_hash",
    "sha256_bytes",
    "sha256_file",
    "sha256_stream",
    "verify_file",
]

#: 1 MiB. Large enough to keep syscall overhead irrelevant on multi-GB files,
#: small enough to stay friendly to memory on a loaded server.
CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream: IO[bytes], *, chunk_size: int = CHUNK_SIZE) -> str:
    """Digest a binary stream from its current position to EOF."""
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = CHUNK_SIZE) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle, chunk_size=chunk_size)


def verify_file(path: Path | str, expected_sha256: str, *, label: str | None = None) -> None:
    """Raise :class:`ChecksumMismatchError` unless the file matches.

    Never returns a boolean: a caller that forgets to check a bool would
    silently accept a corrupt artifact, which section 61 forbids outright.
    """
    path = Path(path)
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ChecksumMismatchError(
            path=label or str(path),
            expected=expected_sha256.lower(),
            actual=actual,
        )


def copy_and_hash(
    source: IO[bytes],
    destination: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    label: str | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[str, int]:
    """Copy a stream to ``destination``, hashing as the bytes go past.

    Returns ``(hexdigest, bytes_written)``.

    This single-pass shape is what lets ``import`` write bundle members
    straight into content-addressed storage without first extracting them:
    re-reading the file to hash it would double the I/O, and the source stream
    (a tar member) is not seekable anyway.

    If verification fails the partial file is removed. Leaving a
    half-written artifact behind invites a later run from mistaking it for
    a complete one.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)

        actual = digest.hexdigest()
        name = label or str(destination)
        if expected_size is not None and written != expected_size:
            raise ChecksumMismatchError(
                path=name,
                expected=f"{expected_size} bytes",
                actual=f"{written} bytes",
            )
        if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
            raise ChecksumMismatchError(path=name, expected=expected_sha256.lower(), actual=actual)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    return actual, written


class HashingReader:
    """A read-only stream wrapper that digests bytes as they are consumed.

    Used when the copy loop belongs to somebody else - ``tarfile.addfile``
    pulls from the stream at its own pace, so we cannot drive it ourselves, but
    we still need the digest of exactly what was written.
    """

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        if chunk:
            self._digest.update(chunk)
            self.bytes_read += len(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> HashingReader:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def atomic_replace(source: Path, destination: Path) -> None:
    """Move ``source`` onto ``destination`` atomically within a filesystem.

    Content-addressed storage relies on this: a reader either sees no file at
    a digest path or sees the complete, verified one - never a partial write.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def move_across_filesystems(source: Path, destination: Path) -> None:
    """Fall back to a copy when ``os.replace`` cannot cross a device boundary."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(destination)
    except OSError:
        shutil.move(str(source), str(destination))
