"""Filesystem helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["atomic_write_bytes", "atomic_write_text", "directory_size", "free_space"]


def free_space(path: Path | str) -> int:
    """Bytes available on the filesystem holding ``path``.

    Walks up to the nearest existing ancestor, so this answers usefully for a
    directory that has not been created yet - which is the normal case when
    deciding whether an import will fit.
    """
    current = Path(path).absolute()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return shutil.disk_usage(current).free


def directory_size(path: Path | str) -> int:
    root = Path(path)
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    """Write via a temp file and rename, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
