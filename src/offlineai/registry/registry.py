"""The local package registry (sections 23, 24 and 25).

Owns the database and the artifact store together, because the two must stay
consistent: a row claiming an artifact that is not on disk, or a blob nothing
references, are both corruption.

The import path is the interesting one. It streams a bundle straight into
content-addressed storage rather than extracting it first. A naive extract
needs twice the bundle size free before it can even begin, which on a 62 GB
bundle is frequently the difference between working and not.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from offlineai.bundler.archive import BundleReader
from offlineai.config.settings import Settings
from offlineai.errors import InsufficientDiskError, RegistryError
from offlineai.logging import get_logger
from offlineai.registry.db import open_registry_db, transaction
from offlineai.registry.store import ArtifactStore
from offlineai.schema.manifest import Manifest
from offlineai.utils.fs import free_space
from offlineai.utils.hashing import sha256_file
from offlineai.utils.sizes import format_bytes

__all__ = ["ImportResult", "PackageRecord", "Registry"]

logger = get_logger("registry")

ProgressHook = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class PackageRecord:
    id: int
    name: str
    version: str
    bundle_sha256: str
    description: str | None
    platforms: list[str]
    total_size: int
    imported_at: str
    signed: bool
    manifest_yaml: str = ""
    package_yaml: str = ""

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    def manifest(self) -> Manifest:
        return Manifest.from_yaml(self.manifest_yaml)


@dataclass(slots=True)
class ImportResult:
    package: str
    version: str
    artifacts_imported: int
    bytes_imported: int
    deduplicated: int = 0
    already_present: bool = False
    warnings: list[str] = field(default_factory=list)


class Registry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = settings.registry_dir / "index.db"
        self.store = ArtifactStore(settings.data_dir / "artifacts")

    def initialise(self) -> None:
        self.settings.ensure_directories()
        self.store.ensure()
        with open_registry_db(self.db_path):
            pass

    # -- import ----------------------------------------------------------

    def import_bundle(
        self,
        bundle_path: Path | str,
        *,
        progress: ProgressHook | None = None,
        force: bool = False,
    ) -> ImportResult:
        """Import a bundle, streaming its artifacts into the store.

        Follows the ten steps of section 25: open, validate, read the manifest,
        check the format version, verify checksums, check the signature, check
        disk, store, register, report.
        """
        bundle_path = Path(bundle_path)
        self.initialise()

        header_manifest = BundleReader.peek_manifest(bundle_path)
        required = header_manifest.total_size

        # Section 25 step 7, before writing anything: refusing up front beats
        # filling the volume and failing at 98%.
        available = free_space(self.store.root)
        if required > available:
            raise InsufficientDiskError(
                "not enough free space to import this bundle",
                details={
                    "Required": format_bytes(required),
                    "Available": format_bytes(available),
                    "Location": str(self.store.root),
                },
                action="Free space, or use --data-dir to import onto another volume.",
            )

        with open_registry_db(self.db_path) as connection:
            existing = self._find(
                connection, header_manifest.package.name, header_manifest.package.version
            )
            if existing is not None and not force:
                return ImportResult(
                    package=existing.name,
                    version=existing.version,
                    artifacts_imported=0,
                    bytes_imported=0,
                    already_present=True,
                )

            imported = 0
            deduplicated = 0
            total_bytes = 0
            rows: list[tuple[str, str, int, str, str]] = []

            with BundleReader.open(bundle_path) as reader:
                header = reader.read_header()
                manifest = header.manifest
                total = len(manifest.artifacts)

                for entry, stream in reader.iter_artifacts():
                    was_present = self.store.has(entry.sha256)
                    stored = self.store.add_stream(
                        stream,
                        expected_sha256=entry.sha256,
                        expected_size=entry.size,
                        label=entry.path,
                    )
                    if was_present:
                        deduplicated += 1
                    imported += 1
                    total_bytes += stored.size
                    rows.append(
                        (stored.sha256, entry.type.value, stored.size, entry.id, entry.path)
                    )
                    if progress is not None:
                        progress(entry.path, imported, total)

            bundle_digest = sha256_file(bundle_path)

            with transaction(connection):
                if existing is not None:
                    self._delete_package_rows(connection, existing.id)
                cursor = connection.execute(
                    """
                    INSERT INTO packages
                        (name, version, bundle_sha256, manifest_yaml, package_yaml,
                         description, platforms, total_size, imported_at, signed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name, version) DO UPDATE SET
                        bundle_sha256 = excluded.bundle_sha256,
                        manifest_yaml = excluded.manifest_yaml,
                        package_yaml  = excluded.package_yaml,
                        total_size    = excluded.total_size,
                        imported_at   = excluded.imported_at,
                        signed        = excluded.signed
                    RETURNING id
                    """,
                    (
                        manifest.package.name,
                        manifest.package.version,
                        bundle_digest,
                        manifest.to_yaml(),
                        header.package_yaml.decode("utf-8", "replace"),
                        None,
                        ",".join(manifest.platforms),
                        manifest.total_size,
                        datetime.now(UTC).isoformat(),
                        int(header.signature is not None),
                    ),
                )
                package_id = int(cursor.fetchone()[0])

                now = datetime.now(UTC).isoformat()
                for digest, artifact_type, size, artifact_id, bundle_member in rows:
                    connection.execute(
                        """
                        INSERT INTO artifacts (sha256, type, size, refcount, created_at)
                        VALUES (?, ?, ?, 0, ?)
                        ON CONFLICT(sha256) DO NOTHING
                        """,
                        (digest, artifact_type, size, now),
                    )
                    connection.execute(
                        """
                        INSERT INTO package_artifacts
                            (package_id, artifact_sha256, artifact_id, bundle_path)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(package_id, bundle_path) DO NOTHING
                        """,
                        (package_id, digest, artifact_id, bundle_member),
                    )
                self._recount(connection)

        return ImportResult(
            package=manifest.package.name,
            version=manifest.package.version,
            artifacts_imported=imported,
            bytes_imported=total_bytes,
            deduplicated=deduplicated,
        )

    # -- queries ---------------------------------------------------------

    def list_packages(self) -> list[PackageRecord]:
        with open_registry_db(self.db_path) as connection:
            rows = connection.execute("SELECT * FROM packages ORDER BY name, version").fetchall()
        return [_to_record(row) for row in rows]

    def search(self, term: str) -> list[PackageRecord]:
        pattern = f"%{term.lower()}%"
        with open_registry_db(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM packages
                WHERE lower(name) LIKE ? OR lower(COALESCE(description, '')) LIKE ?
                ORDER BY name, version
                """,
                (pattern, pattern),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def get(self, name: str, version: str | None = None) -> PackageRecord | None:
        with open_registry_db(self.db_path) as connection:
            return self._find(connection, name, version)

    def require(self, name: str, version: str | None = None) -> PackageRecord:
        record = self.get(name, version)
        if record is None:
            available = ", ".join(sorted({p.name for p in self.list_packages()})) or "none"
            raise RegistryError(
                f"package {name!r} is not in the local registry",
                details={"Imported packages": available},
                action="Import it first:\n  offlineai import <bundle>.offlineai",
            )
        return record

    def artifacts_for(self, package_id: int) -> list[tuple[str, str, str]]:
        """``(sha256, artifact_id, bundle_path)`` for one package."""
        with open_registry_db(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT artifact_sha256, artifact_id, bundle_path
                FROM package_artifacts WHERE package_id = ? ORDER BY bundle_path
                """,
                (package_id,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def artifact_path(self, sha256: str) -> Path:
        return self.store.path_for(sha256)

    # -- removal ---------------------------------------------------------

    def remove(self, name: str, version: str | None = None) -> tuple[int, int]:
        """Remove a package. Returns ``(artifacts freed, bytes freed)``.

        Only content no other package references is deleted - which is what the
        refcount is for. Deduplication would be a liability without it.
        """
        with open_registry_db(self.db_path) as connection:
            record = self._find(connection, name, version)
            if record is None:
                raise RegistryError(f"package {name!r} is not in the local registry")

            with transaction(connection):
                self._delete_package_rows(connection, record.id)
                connection.execute("DELETE FROM packages WHERE id = ?", (record.id,))
                self._recount(connection)
                orphans = connection.execute(
                    "SELECT sha256, size FROM artifacts WHERE refcount = 0"
                ).fetchall()
                connection.execute("DELETE FROM artifacts WHERE refcount = 0")

        freed_bytes = 0
        freed = 0
        for digest, size in orphans:
            if self.store.delete(digest):
                freed += 1
                freed_bytes += int(size)
        return freed, freed_bytes

    # -- internals -------------------------------------------------------

    def _find(
        self, connection: sqlite3.Connection, name: str, version: str | None
    ) -> PackageRecord | None:
        if version:
            row = connection.execute(
                "SELECT * FROM packages WHERE name = ? AND version = ?", (name, version)
            ).fetchone()
        else:
            # Newest import wins when a name has several versions; `info` shows
            # them all so the choice is visible.
            row = connection.execute(
                "SELECT * FROM packages WHERE name = ? ORDER BY imported_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        return _to_record(row) if row else None

    @staticmethod
    def _delete_package_rows(connection: sqlite3.Connection, package_id: int) -> None:
        connection.execute("DELETE FROM package_artifacts WHERE package_id = ?", (package_id,))

    @staticmethod
    def _recount(connection: sqlite3.Connection) -> None:
        """Recompute refcounts from the join table.

        Derived rather than incremented: an incremental counter drifts the
        first time a transaction is interrupted, and a wrong refcount here
        deletes data that is still in use.
        """
        connection.execute(
            """
            UPDATE artifacts SET refcount = (
                SELECT COUNT(*) FROM package_artifacts
                WHERE package_artifacts.artifact_sha256 = artifacts.sha256
            )
            """
        )


def _to_record(row: sqlite3.Row) -> PackageRecord:
    return PackageRecord(
        id=int(row["id"]),
        name=row["name"],
        version=row["version"],
        bundle_sha256=row["bundle_sha256"],
        description=row["description"],
        platforms=[p for p in (row["platforms"] or "").split(",") if p],
        total_size=int(row["total_size"]),
        imported_at=row["imported_at"],
        signed=bool(row["signed"]),
        manifest_yaml=row["manifest_yaml"],
        package_yaml=row["package_yaml"],
    )
