"""Secret detection (section 40).

Never package secrets by default. This scans the build context for files that
look like credentials and refuses rather than warning, because the failure mode
- a private key inside a bundle that has already been carried across an air gap
on removable media - is not recoverable by deleting the file afterwards.

Detection is by filename, not content. Content scanning produces false
positives on model weights and is expensive on a 62 GB tree; the filenames
below are the ones that actually cause incidents.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

__all__ = ["IGNORE_FILENAME", "SECRET_PATTERNS", "scan_for_secrets"]

IGNORE_FILENAME = ".offlineaiignore"

#: Matched against each file's name, case-insensitively.
SECRET_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "service-account*.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".htpasswd",
    "*.ppk",
)

#: Never descend into these.
_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
)


def scan_for_secrets(root: Path | str) -> list[Path]:
    """Return credential-shaped files under ``root``.

    Paths listed in ``.offlineaiignore`` (glob patterns, one per line) are
    excluded, which is the escape hatch for a project that legitimately ships
    something matching a pattern - a test fixture key, for instance.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    allowed = _read_ignore(root)
    findings: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in allowed):
            continue
        name = path.name.lower()
        if any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_PATTERNS):
            findings.append(path)

    return findings


def _read_ignore(root: Path) -> list[str]:
    ignore_file = root / IGNORE_FILENAME
    if not ignore_file.is_file():
        return []
    return [
        line.strip()
        for line in ignore_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
