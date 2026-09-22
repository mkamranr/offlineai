"""Canonical bundle layout (section 9).

Member order inside the archive is part of the format, not an implementation
detail. Metadata is written first and ``artifacts/`` last, which is what allows
``inspect``, ``verify-signature`` and a format-version check to read a few
kilobytes off the front of a 62 GB bundle instead of scanning it. Section 60
requires ``inspect`` to work without installing anything; on bundles this size
that is only affordable if the read genuinely stops early.

Anything that changes these constants changes the on-disk format and needs a
``FORMAT_VERSION`` bump.
"""

from __future__ import annotations

from offlineai.schema.manifest import ArtifactType

__all__ = [
    "ARTIFACTS_DIR",
    "ARTIFACT_SUBDIR",
    "CHECKSUMS",
    "DOCS_DIR",
    "HARDWARE",
    "HEADER_ORDER",
    "LICENSES",
    "MANIFEST",
    "PACKAGE",
    "PUBKEY",
    "SBOM",
    "SCRIPTS_DIR",
    "SIGNATURE",
    "SOURCES",
    "artifact_path",
    "is_header_member",
]

#: Conventional bundle file extension.
EXTENSION = ".offlineai"

# -- header members, in write order -----------------------------------------

MANIFEST = "manifest.yaml"
PACKAGE = "package.yaml"
CHECKSUMS = "checksums.sha256"
SIGNATURE = "signature/manifest.sig"
PUBKEY = "signature/pubkey.pub"
SBOM = "sbom/sbom.json"
LICENSES = "metadata/licenses.json"
SOURCES = "metadata/sources.json"
HARDWARE = "metadata/hardware.json"

SCRIPTS_DIR = "scripts"
DOCS_DIR = "docs"
ARTIFACTS_DIR = "artifacts"

#: The manifest must come first: a reader needs the format version before it
#: can safely interpret anything else.
HEADER_ORDER: tuple[str, ...] = (
    MANIFEST,
    PACKAGE,
    CHECKSUMS,
    SIGNATURE,
    PUBKEY,
    SBOM,
    LICENSES,
    SOURCES,
    HARDWARE,
)

#: Where each artifact type lives beneath ``artifacts/``.
ARTIFACT_SUBDIR: dict[ArtifactType, str] = {
    ArtifactType.MODEL: "models",
    ArtifactType.OCI_IMAGE: "containers",
    ArtifactType.PYTHON_WHEEL: "python/wheels",
    ArtifactType.PYTHON_SDIST: "python/sdists",
    ArtifactType.SYSTEM_PACKAGE: "system/packages",
    ArtifactType.CONFIG: "config",
    ArtifactType.SCRIPT: "scripts",
    ArtifactType.MISC: "misc",
}


def is_header_member(name: str) -> bool:
    """Whether ``name`` belongs to the metadata block that precedes artifacts."""
    return not name.startswith(f"{ARTIFACTS_DIR}/")


def artifact_path(artifact_type: ArtifactType, *parts: str) -> str:
    """Build a bundle-relative path for an artifact of the given type."""
    if not parts:
        raise ValueError("artifact_path requires at least one path component")
    subdir = ARTIFACT_SUBDIR[artifact_type]
    return "/".join((ARTIFACTS_DIR, subdir, *parts))
