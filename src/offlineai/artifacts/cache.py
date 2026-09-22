"""Global content-addressed artifact cache (sections 43 and 44).

Two jobs:

* **Deduplicate.** If ``rag-stack`` and ``qwen-vllm`` both use the vLLM image,
  it is downloaded once and stored once. Content addressing makes that
  automatic rather than something the builder has to reason about.
* **Resume.** Downloads land in ``tmp/`` and are only promoted to their digest
  path once complete and verified, so an interrupted build never leaves a
  truncated file that a later run mistakes for a good one (section 43).

Layout::

    cache/
      sha256/ab/abcdef...        content, named by its own digest
      refs/<source>/<key>        locator -> digest, for hits without re-fetching
      tmp/                       in-flight downloads
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from offlineai.logging import get_logger
from offlineai.utils.hashing import copy_and_hash, sha256_file

__all__ = ["ArtifactCache", "CacheEntry"]

logger = get_logger("artifacts.cache")


@dataclass(frozen=True, slots=True)
class CacheEntry:
    sha256: str
    path: Path
    size: int


class ArtifactCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.content_dir = self.root / "sha256"
        self.refs_dir = self.root / "refs"
        self.tmp_dir = self.root / "tmp"

    def ensure(self) -> None:
        for directory in (self.content_dir, self.refs_dir, self.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- content addressing ----------------------------------------------

    def path_for(self, sha256: str) -> Path:
        """Two-level fan-out keeps directory sizes sane at millions of blobs."""
        digest = sha256.lower()
        return self.content_dir / digest[:2] / digest

    def has(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def get(self, sha256: str) -> CacheEntry | None:
        path = self.path_for(sha256)
        if not path.is_file():
            return None
        return CacheEntry(sha256=sha256.lower(), path=path, size=path.stat().st_size)

    def store_stream(
        self,
        stream: IO[bytes],
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
        label: str | None = None,
    ) -> CacheEntry:
        """Write a stream into the cache, addressed by its own digest."""
        self.ensure()
        handle, tmp_name = tempfile.mkstemp(dir=self.tmp_dir, prefix="dl-")
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

        return self._promote(tmp_path, digest, size)

    def store_file(self, path: Path | str, *, move: bool = False) -> CacheEntry:
        """Add an existing local file to the cache."""
        self.ensure()
        path = Path(path)
        digest = sha256_file(path)
        size = path.stat().st_size
        destination = self.path_for(digest)
        if destination.is_file():
            if move:
                path.unlink(missing_ok=True)
            return CacheEntry(sha256=digest, path=destination, size=size)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if move:
            return self._promote(path, digest, size)
        # Copy via a temp file so a reader never observes a partial blob at the
        # digest path.
        handle, tmp_name = tempfile.mkstemp(dir=self.tmp_dir, prefix="cp-")
        os.close(handle)
        tmp_path = Path(tmp_name)
        with path.open("rb") as source:
            copy_and_hash(source, tmp_path, expected_sha256=digest)
        return self._promote(tmp_path, digest, size)

    def _promote(self, tmp_path: Path, digest: str, size: int) -> CacheEntry:
        """Atomically move a verified temp file to its digest path."""
        destination = self.path_for(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            # Another build already stored identical content. Content addressing
            # means the existing copy is byte-identical, so keep it.
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(destination)
        return CacheEntry(sha256=digest, path=destination, size=size)

    @contextmanager
    def partial(self, key: str) -> Iterator[Path]:
        """A stable temp path for a resumable download of ``key``.

        The path is derived from the key, so an interrupted build reuses the
        same file and can resume from its current length rather than starting
        over (section 43).
        """
        self.ensure()
        name = hashlib.sha256(key.encode()).hexdigest()[:32]
        yield self.tmp_dir / f"partial-{name}"

    # -- reference index -------------------------------------------------

    def _ref_path(self, source_kind: str, key: str) -> Path:
        safe = hashlib.sha256(key.encode()).hexdigest()
        return self.refs_dir / source_kind / safe[:2] / safe

    def lookup_ref(self, source_kind: str, key: str) -> CacheEntry | None:
        """Resolve a source locator to cached content, if we have it."""
        ref = self._ref_path(source_kind, key)
        if not ref.is_file():
            return None
        digest = ref.read_text().strip()
        entry = self.get(digest)
        if entry is None:
            # Content was pruned; the dangling reference is useless.
            ref.unlink(missing_ok=True)
            return None
        logger.debug("cache hit %s:%s -> %s", source_kind, key, digest[:12])
        return entry

    def put_ref(self, source_kind: str, key: str, sha256: str) -> None:
        ref = self._ref_path(source_kind, key)
        ref.parent.mkdir(parents=True, exist_ok=True)
        tmp = ref.with_suffix(".tmp")
        tmp.write_text(sha256.lower())
        tmp.replace(ref)

    # -- maintenance -----------------------------------------------------

    def total_size(self) -> int:
        if not self.content_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.content_dir.rglob("*") if p.is_file())

    def clear_partials(self) -> int:
        """Remove in-flight downloads. Returns bytes reclaimed."""
        if not self.tmp_dir.is_dir():
            return 0
        reclaimed = 0
        for path in self.tmp_dir.iterdir():
            if path.is_file():
                reclaimed += path.stat().st_size
                path.unlink(missing_ok=True)
        return reclaimed
