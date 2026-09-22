"""Streaming reader and writer for the ``.offlineai`` archive.

Everything here is single-pass and constant-memory. A bundle can exceed both
RAM and the target's free disk, so nothing is ever buffered whole and nothing
is extracted just to be inspected.

The writer enforces two invariants that the rest of the system depends on:

1. **Ordering.** Header members are written before any artifact, and the
   writer refuses to interleave them. See :mod:`offlineai.layout`.
2. **Completeness.** On close, every artifact the manifest declares must
   actually have been written. This is section 74 - "a successful build must
   mean the resulting bundle contains everything required" - turned into a
   structural guarantee rather than a hope.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, BinaryIO

from offlineai import layout
from offlineai.errors import MissingArtifactError, UnsafeArchiveError, VerificationError
from offlineai.schema.manifest import ArtifactEntry, Compression, Manifest
from offlineai.security.extraction import validate_member_name
from offlineai.utils.hashing import CHUNK_SIZE

__all__ = ["BundleHeader", "BundleReader", "BundleWriter"]

#: Compression schemes this build can write. zstd is reserved in the format
#: enum for forward compatibility but needs a library Python does not ship
#: before 3.14, so it is refused with a clear message rather than half-supported.
_SUPPORTED_WRITE_COMPRESSION = frozenset({Compression.NONE, Compression.GZIP})


@dataclass(slots=True)
class BundleHeader:
    """The metadata block at the front of a bundle."""

    manifest: Manifest
    package_yaml: bytes = b""
    checksums: str = ""
    signature: bytes | None = None
    public_key: bytes | None = None
    sbom: bytes | None = None
    extra: dict[str, bytes] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _open_write(fileobj: BinaryIO, mode: str) -> tarfile.TarFile:
    """Open a streaming tar for writing.

    Wrapped in a function so the literal-mode overload resolution stays in one
    place, and so the returned handle's lifetime is obviously managed by the
    caller rather than a context manager.
    """
    if mode == "w|gz":
        return tarfile.open(  # noqa: SIM115
            fileobj=fileobj, mode="w|gz", format=tarfile.PAX_FORMAT
        )
    return tarfile.open(fileobj=fileobj, mode="w|", format=tarfile.PAX_FORMAT)  # noqa: SIM115


class BundleWriter:
    """Create a bundle, header first, artifacts second."""

    def __init__(self, path: Path | str, *, compression: Compression = Compression.NONE) -> None:
        if compression not in _SUPPORTED_WRITE_COMPRESSION:
            raise VerificationError(
                f"compression {compression.value!r} is not supported by this build",
                action="Use 'none' (the default) or 'gzip'. Model weights and "
                "container layers are already compressed, so 'none' is almost "
                "always the right choice.",
            )
        self.path = Path(path)
        self.compression = compression
        self._manifest: Manifest | None = None
        self._written: set[str] = set()
        self._header_done = False
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Not a context manager: the handle's lifetime spans many calls and is
        # closed by close() or abort().
        self._fileobj: BinaryIO = self.path.open("wb")  # noqa: SIM115
        self._tar = _open_write(self._fileobj, "w|gz" if compression is Compression.GZIP else "w|")

    # -- header ----------------------------------------------------------

    def write_header(
        self,
        *,
        manifest: Manifest,
        package_yaml: bytes,
        signature: bytes | None = None,
        public_key: bytes | None = None,
        sbom: bytes | None = None,
        licenses: bytes | None = None,
        sources: bytes | None = None,
        hardware: bytes | None = None,
        scripts: dict[str, bytes] | None = None,
        docs: dict[str, bytes] | None = None,
    ) -> None:
        """Write the metadata block. Must be called once, before any artifact."""
        if self._header_done:
            raise RuntimeError("the bundle header has already been written")
        if self._written:
            raise RuntimeError("cannot write the header after an artifact")

        # The written manifest must describe the bundle truthfully, so the
        # actual archive compression is stamped onto it here rather than being
        # left for the caller to keep in sync. The caller's object is not
        # mutated.
        manifest = manifest.model_copy(update={"compression": self.compression})
        self._manifest = manifest
        # checksums.sha256 is derived from the manifest rather than accumulated,
        # so the two can never disagree.
        checksums = "".join(f"{a.sha256}  {a.path}\n" for a in manifest.artifacts)

        self._add_bytes(layout.MANIFEST, manifest.to_yaml().encode("utf-8"))
        self._add_bytes(layout.PACKAGE, package_yaml)
        self._add_bytes(layout.CHECKSUMS, checksums.encode("utf-8"))
        for name, payload in (
            (layout.SIGNATURE, signature),
            (layout.PUBKEY, public_key),
            (layout.SBOM, sbom),
            (layout.LICENSES, licenses),
            (layout.SOURCES, sources),
            (layout.HARDWARE, hardware),
        ):
            if payload is not None:
                self._add_bytes(name, payload)

        for prefix, entries in ((layout.SCRIPTS_DIR, scripts), (layout.DOCS_DIR, docs)):
            for name, payload in (entries or {}).items():
                self._add_bytes(f"{prefix}/{validate_member_name(name)}", payload)

        self._header_done = True

    # -- artifacts -------------------------------------------------------

    def add_artifact(self, bundle_path: str, source: Path | str) -> None:
        """Stream one artifact in, verifying it against the manifest.

        Checking here means a build cannot produce a bundle whose payload
        disagrees with its own manifest - the failure surfaces on the builder,
        where it can still be fixed.
        """
        if not self._header_done or self._manifest is None:
            raise RuntimeError("write_header() must be called before add_artifact()")

        entry = self._manifest.artifact_by_path(bundle_path)
        if entry is None:
            raise MissingArtifactError(
                f"artifact {bundle_path!r} is not declared in the manifest",
                action="Every file in a bundle must be described by the manifest. "
                "This is a bug in the builder.",
            )

        source = Path(source)
        # Verifies content and size; raises ChecksumMismatchError on any drift.
        from offlineai.utils.hashing import verify_file

        verify_file(source, entry.sha256, label=bundle_path)
        actual_size = source.stat().st_size
        if actual_size != entry.size:
            raise VerificationError(
                f"artifact {bundle_path!r} is {actual_size} bytes but the manifest "
                f"declares {entry.size}",
            )

        info = tarfile.TarInfo(bundle_path)
        info.size = entry.size
        info.mode = 0o644
        info.mtime = 0  # deterministic: section 34 asks that we avoid timestamps
        with source.open("rb") as handle:
            self._tar.addfile(info, handle)
        self._written.add(bundle_path)

    def _add_bytes(self, name: str, payload: bytes) -> None:
        import io

        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mode = 0o644
        info.mtime = 0
        self._tar.addfile(info, io.BytesIO(payload))

    # -- lifecycle -------------------------------------------------------

    def abort(self) -> None:
        """Discard the bundle without the completeness check.

        For a build that has already failed: the partial file is removed so no
        later step can mistake it for a finished bundle.
        """
        if self._closed:
            return
        self._closed = True
        self._tar.close()
        self._fileobj.close()
        self.path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._verify_complete()
        finally:
            self._tar.close()
            self._fileobj.close()

    def _verify_complete(self) -> None:
        if self._manifest is None:
            self.path.unlink(missing_ok=True)
            raise MissingArtifactError(
                "the bundle header was never written",
                action="This is a bug in the builder.",
            )
        declared = {a.path: a.id for a in self._manifest.artifacts}
        missing = sorted(set(declared) - self._written)
        if missing:
            ids = ", ".join(declared[path] for path in missing)
            self._tar.close()
            self._fileobj.close()
            self.path.unlink(missing_ok=True)
            raise MissingArtifactError(
                f"the manifest declares {len(missing)} artifact(s) that were never written: {ids}",
                details={"Missing": "\n".join(missing)},
                action="A bundle must contain everything it declares. Rebuild, and "
                "if this persists report it as a builder bug.",
            )

    def __enter__(self) -> BundleWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # Failed build: close the handles and leave nothing usable behind.
            self._closed = True
            self._tar.close()
            self._fileobj.close()
            self.path.unlink(missing_ok=True)
            return
        self.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class _CountingReader:
    """Wraps a file object to record how many bytes are actually consumed.

    Used to prove - in tests, and when debugging a slow inspect - that a header
    read really does stop before the artifact payload.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def close(self) -> None:
        self._stream.close()


class BundleReader:
    """Sequential reader. Header first, then artifacts, in archive order."""

    def __init__(self, tar: tarfile.TarFile, *, counter: _CountingReader | None = None) -> None:
        self._tar = tar
        self._counter = counter
        self._header: BundleHeader | None = None
        self._pending: tarfile.TarInfo | None = None

    @classmethod
    @contextmanager
    def open(cls, path: Path | str) -> Iterator[BundleReader]:
        """Open a bundle for streaming reads, detecting compression."""
        handle = Path(path).open("rb")  # noqa: SIM115 - closed in the finally below
        counter = _CountingReader(handle)
        try:
            # "r|*" is the transparent-compression *stream* mode: it never
            # seeks, so it works on a pipe and never reads ahead into the
            # artifact payload.
            # _CountingReader supplies read(), which is all stream mode needs,
            # but it is not one of the typeshed _Fileobj shapes.
            tar = tarfile.open(  # type: ignore[call-overload]  # noqa: SIM115
                fileobj=counter, mode="r|*"
            )
        except tarfile.TarError as exc:
            handle.close()
            raise VerificationError(
                f"{Path(path).name} is not a readable OfflineAI bundle",
                details={"Detail": str(exc)},
                action="The file may be truncated or corrupt. Re-transfer it.",
            ) from exc
        try:
            yield cls(tar, counter=counter)
        finally:
            tar.close()
            handle.close()

    # -- header ----------------------------------------------------------

    def read_header(self) -> BundleHeader:
        """Read metadata members, stopping at the first artifact.

        The archive position is left on that first artifact member, so a
        subsequent :meth:`iter_artifacts` continues without rewinding.
        """
        if self._header is not None:
            return self._header

        blobs: dict[str, bytes] = {}
        while (member := self._tar.next()) is not None:
            if not layout.is_header_member(member.name):
                self._pending = member
                break
            if not member.isfile():
                continue
            validate_member_name(member.name)
            handle = self._tar.extractfile(member)
            if handle is not None:
                blobs[member.name] = handle.read()

        if layout.MANIFEST not in blobs:
            raise VerificationError(
                "the bundle contains no manifest.yaml",
                action="This is not an OfflineAI bundle, or it is corrupt.",
            )

        try:
            manifest = Manifest.from_yaml(blobs[layout.MANIFEST])
        except Exception as exc:
            raise VerificationError(
                "the bundle manifest is malformed or has an unsupported format version",
                details={"Detail": str(exc)},
                action="This bundle may have been produced by a newer OfflineAI. "
                "Upgrade, or rebuild the bundle with this version.",
            ) from exc

        known = set(layout.HEADER_ORDER)
        self._header = BundleHeader(
            manifest=manifest,
            package_yaml=blobs.get(layout.PACKAGE, b""),
            checksums=blobs.get(layout.CHECKSUMS, b"").decode("utf-8", "replace"),
            signature=blobs.get(layout.SIGNATURE),
            public_key=blobs.get(layout.PUBKEY),
            sbom=blobs.get(layout.SBOM),
            extra={k: v for k, v in blobs.items() if k not in known},
        )
        return self._header

    # -- artifacts -------------------------------------------------------

    def iter_artifacts(self) -> Iterator[tuple[ArtifactEntry, IO[bytes]]]:
        """Yield ``(manifest entry, stream)`` for each artifact, in order.

        Each stream is valid only until the next iteration: this is a
        sequential tar reader, not a random-access one. Consume it or copy it
        before advancing.
        """
        header = self.read_header()
        manifest = header.manifest

        member = self._pending
        self._pending = None
        if member is None:
            member = self._tar.next()

        while member is not None:
            if member.isdir():
                continue
            name = validate_member_name(member.name)
            entry = manifest.artifact_by_path(name)
            if entry is None:
                raise UnsafeArchiveError(
                    f"the bundle contains an artifact the manifest does not declare: {name}",
                    details={"Member": name},
                    action="An undeclared payload means the bundle does not match "
                    "its own manifest. Do not install it.",
                )
            handle = self._tar.extractfile(member)
            if handle is not None:
                yield entry, handle
            member = self._tar.next()

    # -- conveniences ----------------------------------------------------

    @staticmethod
    def member_names(path: Path | str) -> list[str]:
        """All member names, in archive order. Reads the whole directory."""
        with BundleReader.open(path) as reader:
            names: list[str] = []
            while (member := reader._tar.next()) is not None:
                names.append(member.name)
            return names

    @staticmethod
    def peek_manifest(path: Path | str) -> Manifest:
        """Read just the manifest. Does not touch the artifact payload."""
        with BundleReader.open(path) as reader:
            return reader.read_header().manifest

    @staticmethod
    def read_header_counting_bytes(path: Path | str) -> tuple[BundleHeader, int]:
        """Read the header and report how many bytes were consumed."""
        with BundleReader.open(path) as reader:
            header = reader.read_header()
            consumed = reader._counter.bytes_read if reader._counter else -1
            return header, consumed


_ = CHUNK_SIZE  # re-exported indirectly by callers that stream artifacts
