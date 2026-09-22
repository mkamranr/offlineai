"""Safe archive extraction.

A bundle crosses the security boundary by definition: it was built elsewhere,
carried on removable media, and handed to a machine that is isolated precisely
because its operators do not trust what reaches it. Extraction therefore treats
the archive as hostile input (sections 38 and 39).

The policy is deliberately stricter than :func:`tarfile.data_filter`:

* every member name is validated and normalised before anything is written;
* validation runs over the whole archive **first**, so a hostile member at the
  end cannot leave the earlier members on disk;
* symlinks and hard links are refused unless explicitly opted into, and even
  then must resolve inside the destination;
* device nodes and FIFOs are always refused;
* archive permission bits are masked, so a bundle cannot smuggle in a setuid or
  world-writable file.
"""

from __future__ import annotations

import shutil
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path as FsPath
from pathlib import PurePosixPath

from offlineai.errors import UnsafeArchiveError

__all__ = ["ExtractionLimits", "safe_extract", "safe_members", "validate_member_name"]

#: Permission bits an extracted file may keep. Drops setuid, setgid, sticky and
#: all "other" write access.
_MODE_MASK = 0o755


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Resource ceilings applied while extracting.

    Defaults are generous on size because legitimate bundles genuinely are
    enormous, and strict on structure because nothing legitimate needs a
    device node.
    """

    #: Guards against inode exhaustion from an archive of millions of entries.
    max_members: int = 1_000_000

    #: Total uncompressed bytes. ``None`` means unlimited; callers that know the
    #: expected size (from the manifest) should pass it to bound a zip bomb.
    max_total_bytes: int | None = None

    #: Largest single member. ``None`` means unlimited.
    max_member_bytes: int | None = None

    #: Symlinks are refused by default. Bundle payloads are model weights,
    #: image tarballs and wheels - regular files. Permitting links buys nothing
    #: and costs a large attack surface.
    allow_symlinks: bool = False


def validate_member_name(name: str) -> str:
    """Return ``name`` as a normalised relative POSIX path, or raise.

    Refuses absolute paths, parent traversal, empty names, NUL bytes and
    Windows-style drive or UNC paths (defence in depth - we are Linux-first,
    but a name is not trustworthy just because the platform ignores it).
    """
    if not name or not name.strip():
        raise UnsafeArchiveError(
            "Archive contains a member with an empty name.",
            action="The bundle is malformed. Rebuild it on the builder machine.",
        )
    if "\x00" in name:
        raise UnsafeArchiveError(
            "Archive member name contains a NUL byte.",
            details={"Member": name.replace("\x00", "\\x00")},
        )
    if name.startswith(("/", "\\")):
        raise UnsafeArchiveError(
            "Archive member uses an absolute path.",
            details={"Member": name},
            action="Absolute paths would write outside the destination. Refusing.",
        )
    # Reject drive letters such as C:\evil before normalisation hides them.
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        raise UnsafeArchiveError(
            "Archive member uses a drive-qualified path.", details={"Member": name}
        )
    if "\\" in name:
        raise UnsafeArchiveError(
            "Archive member name contains a backslash.", details={"Member": name}
        )

    parts: list[str] = []
    for part in PurePosixPath(name).parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise UnsafeArchiveError(
                "Archive member escapes the destination directory.",
                details={"Member": name},
                action="This is a path traversal attempt. The bundle must not be trusted.",
            )
        parts.append(part)

    if not parts:
        raise UnsafeArchiveError("Archive member resolves to no path.", details={"Member": name})
    return "/".join(parts)


def _validate_link(member: tarfile.TarInfo, limits: ExtractionLimits) -> None:
    kind = "hard link" if member.islnk() else "symlink"
    if not limits.allow_symlinks:
        raise UnsafeArchiveError(
            f"Archive contains a {kind}, which is not permitted.",
            details={"Member": member.name, "Target": member.linkname},
            action="Rebuild the bundle with regular files, or extract with "
            "symlinks explicitly enabled if the source is trusted.",
        )
    if member.islnk():
        # Hard links are never allowed: they can alias a file outside the
        # destination that a later member then overwrites.
        raise UnsafeArchiveError(
            "Archive contains a hard link, which is never permitted.",
            details={"Member": member.name, "Target": member.linkname},
        )

    target = member.linkname
    if target.startswith("/"):
        raise UnsafeArchiveError(
            "Archive contains a symlink to an absolute path.",
            details={"Member": member.name, "Target": target},
        )
    # Resolve the link target relative to the link's own directory and make
    # sure it stays inside the archive root.
    base = PurePosixPath(validate_member_name(member.name)).parent
    depth = len(base.parts)
    for part in PurePosixPath(target).parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                raise UnsafeArchiveError(
                    "Archive contains a symlink that escapes the destination.",
                    details={"Member": member.name, "Target": target},
                )
        elif part not in ("", "."):
            depth += 1


def safe_members(
    archive: tarfile.TarFile, *, limits: ExtractionLimits | None = None
) -> Iterator[tarfile.TarInfo]:
    """Yield validated members, rewriting each name to a safe relative path.

    Raises :class:`UnsafeArchiveError` on the first hostile member.
    """
    limits = limits or ExtractionLimits()
    total = 0

    for count, member in enumerate(archive, start=1):
        if count > limits.max_members:
            raise UnsafeArchiveError(
                f"Archive exceeds the maximum member count of {limits.max_members}.",
                action="This looks like an archive bomb. Refusing to extract.",
            )

        safe_name = validate_member_name(member.name)

        if member.isdev() or member.isfifo():
            raise UnsafeArchiveError(
                "Archive contains a device or FIFO member.",
                details={"Member": member.name},
                action="Bundles contain regular files only. Refusing to extract.",
            )
        if member.issym() or member.islnk():
            _validate_link(member, limits)
        elif not (member.isfile() or member.isdir()):
            raise UnsafeArchiveError(
                "Archive contains an unsupported member type.",
                details={"Member": member.name, "Type": repr(member.type)},
            )

        if member.isfile():
            if limits.max_member_bytes is not None and member.size > limits.max_member_bytes:
                raise UnsafeArchiveError(
                    "Archive member exceeds the maximum permitted size.",
                    details={
                        "Member": member.name,
                        "Declared size": str(member.size),
                        "Limit": str(limits.max_member_bytes),
                    },
                )
            total += member.size
            if limits.max_total_bytes is not None and total > limits.max_total_bytes:
                raise UnsafeArchiveError(
                    "Archive exceeds the maximum permitted total size.",
                    details={"Limit": str(limits.max_total_bytes)},
                    action="This looks like an archive bomb. Refusing to extract.",
                )

        member.name = safe_name
        # Drop ownership and dangerous permission bits.
        member.mode &= _MODE_MASK
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        yield member


def _already_validated(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:  # noqa: ARG001
    """Extraction filter: a no-op, because :func:`safe_members` already applied
    a stricter policy than any stock ``tarfile`` filter would."""
    return member


def safe_extract(
    archive: tarfile.TarFile,
    destination: FsPath | str,
    *,
    limits: ExtractionLimits | None = None,
) -> None:
    """Extract ``archive`` into ``destination``, refusing anything hostile.

    Validation is a separate pass over the member list before any byte is
    written, so a malicious member at the end of the archive cannot leave the
    benign members behind on disk. This costs one extra walk of the tar
    directory, which is cheap next to the safety it buys.
    """
    limits = limits or ExtractionLimits()
    destination = FsPath(destination)

    members = list(safe_members(archive, limits=limits))

    created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()

    try:
        for member in members:
            target = (destination / member.name).resolve()
            # Belt and braces: even after name validation, confirm the resolved
            # path is inside the root. This catches escapes through a symlink
            # that already existed in the destination.
            if not target.is_relative_to(resolved_root):
                raise UnsafeArchiveError(
                    "Archive member resolves outside the destination.",
                    details={"Member": member.name, "Resolved": str(target)},
                )
        # filter=_already_validated: every member has been through safe_members,
        # which is stricter than tarfile's own "data" filter. Passing an explicit
        # filter is also required to avoid Python 3.14's default-change warning.
        archive.extractall(  # noqa: S202 - members validated by safe_members above
            destination, members=members, filter=_already_validated
        )
    except BaseException:
        # Do not leave a half-extracted tree that a later step might mistake
        # for a complete one.
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
