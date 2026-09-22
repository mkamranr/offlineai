"""Content-addressed artifact store (section 24).

Layout mirrors the cache::

    <data_dir>/artifacts/sha256/ab/abcdef...

Storing by digest means an artifact shared between packages - the vLLM image
used by both ``qwen-vllm`` and ``rag-stack`` - occupies disk once. Deletion is
refcounted in the database, so removing one package cannot pull content out
from under another.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from offlineai.utils.hashing import copy_and_hash

__all__ = ["ArtifactStore", "StoredArtifact"]


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    sha256: str
    path: Path
    size: int


class ArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.content_dir = self.root / "sha256"
        self.tmp_dir = self.root / "tmp"

    def ensure(self) -> None:
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        digest = sha256.lower()
        return self.content_dir / digest[:2] / digest

    def has(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def add_stream(
        self,
        stream: IO[bytes],
        *,
        expected_sha256: str,
        expected_size: int | None = None,
        label: str | None = None,
    ) -> StoredArtifact:
        """Write a stream straight into the store, verifying as it lands.

        This is the path ``import`` uses. Because the digest is known from the
        manifest up front, a corrupt artifact is caught during the copy rather
        than after it - so a bad bundle never consumes the full write.
        """
        self.ensure()
        destination = self.path_for(expected_sha256)
        if destination.is_file():
            # Already present, byte-identical by construction. Drain the stream
            # so the caller's sequential reader stays in step.
            size = 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
            return StoredArtifact(expected_sha256.lower(), destination, destination.stat().st_size)

        handle, tmp_name = tempfile.mkstemp(dir=self.tmp_dir, prefix="in-")
        os.close(handle)
        tmp_path = Path(tmp_name)
        try:
            digest, size = copy_and_hash(
                stream,
                tmp_path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                label=label,
            )
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(destination)
        return StoredArtifact(digest, destination, size)

    def open(self, sha256: str) -> IO[bytes]:
        return self.path_for(sha256).open("rb")

    def delete(self, sha256: str) -> bool:
        """Remove content. Callers must have checked the refcount first."""
        path = self.path_for(sha256)
        if not path.is_file():
            return False
        path.unlink()
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        return True

    def iter_digests(self) -> Iterator[str]:
        if not self.content_dir.is_dir():
            return
        for path in self.content_dir.rglob("*"):
            if path.is_file():
                yield path.name

    def total_size(self) -> int:
        if not self.content_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.content_dir.rglob("*") if p.is_file())
