"""SQLite schema for the local registry (section 24).

SQLite because the tool must be self-contained: an air-gapped host should not
need a database server installed before it can install anything else. Section
24 rules out MongoDB or an external database for exactly this reason.

Schema changes go through :data:`MIGRATIONS`. A registry is long-lived - it
holds the record of what is installed on a production machine - so it is
migrated forward, never recreated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from offlineai.errors import RegistryError

__all__ = ["SCHEMA_VERSION", "connect", "open_registry_db"]

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS packages (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    bundle_sha256   TEXT NOT NULL,
    manifest_yaml   TEXT NOT NULL,
    package_yaml    TEXT NOT NULL DEFAULT '',
    description     TEXT,
    platforms       TEXT NOT NULL DEFAULT '',
    total_size      INTEGER NOT NULL DEFAULT 0,
    imported_at     TEXT NOT NULL,
    signed          INTEGER NOT NULL DEFAULT 0,
    UNIQUE (name, version)
);

-- Content-addressed blobs. refcount lets `remove` free storage without
-- deleting an artifact another package still relies on (section 44 dedupe).
CREATE TABLE IF NOT EXISTS artifacts (
    sha256      TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    size        INTEGER NOT NULL,
    refcount    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS package_artifacts (
    package_id      INTEGER NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
    artifact_id     TEXT NOT NULL,
    bundle_path     TEXT NOT NULL,
    PRIMARY KEY (package_id, bundle_path)
);

CREATE TABLE IF NOT EXISTS installations (
    id           TEXT PRIMARY KEY,
    package_id   INTEGER NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    state        TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    config_json  TEXT NOT NULL DEFAULT '{}',
    error        TEXT
);

-- Each completed step records how to undo itself, which is what makes
-- `offlineai rollback` possible (section 27).
CREATE TABLE IF NOT EXISTS installation_steps (
    installation_id TEXT NOT NULL REFERENCES installations(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    name            TEXT NOT NULL,
    state           TEXT NOT NULL,
    inverse_json    TEXT NOT NULL DEFAULT '{}',
    detail          TEXT,
    PRIMARY KEY (installation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_packages_name ON packages(name);
CREATE INDEX IF NOT EXISTS idx_installations_package ON installations(package_id);
"""

MIGRATIONS: dict[int, str] = {1: _SCHEMA_V1}


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with the pragmas this registry relies on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    except sqlite3.Error as exc:
        raise RegistryError(
            f"cannot open the registry database at {path}",
            details={"Detail": str(exc)},
            action="Check the path is writable and the filesystem is not full.",
        ) from exc

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # WAL survives an interrupted write far better than the rollback journal,
    # which matters when a 62 GB import is what interrupted it.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def migrate(connection: sqlite3.Connection) -> int:
    """Apply outstanding migrations. Returns the resulting schema version."""
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RegistryError(
            f"the registry was written by a newer OfflineAI (schema {current}, "
            f"this build understands {SCHEMA_VERSION})",
            action="Upgrade OfflineAI on this machine.",
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        connection.executescript(MIGRATIONS[version])
        connection.execute(f"PRAGMA user_version = {version}")
    return SCHEMA_VERSION


@contextmanager
def open_registry_db(path: Path | str) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        migrate(connection)
        yield connection
    finally:
        connection.close()


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a unit of work so a failure leaves no partial registry state."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")
