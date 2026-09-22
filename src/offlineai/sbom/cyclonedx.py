"""CycloneDX SBOM generation (sections 36 and 37).

CycloneDX rather than SPDX because its component model maps cleanly onto what
a bundle actually contains - container images, Python wheels, model weights,
OS packages - and because the JSON schema is small enough to emit correctly
without pulling in a dependency the air-gapped target would then need.

Every component carries the SHA-256 the manifest already records, so the SBOM
and the bundle cannot drift apart: both are derived from the same source.

Section 37 governs the licence fields: they are recorded where known, marked
informational, and accompanied by no legal claim whatsoever.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from offlineai import __version__
from offlineai.schema.manifest import ArtifactType, Manifest

__all__ = ["LICENSE_DISCLAIMER", "SPEC_VERSION", "generate_sbom", "license_report"]

SPEC_VERSION = "1.5"

#: Section 37: display this, make no legal claim.
LICENSE_DISCLAIMER = (
    "License metadata is informational. Users are responsible for reviewing applicable licenses."
)

#: CycloneDX component types, per artifact kind.
_COMPONENT_TYPE = {
    ArtifactType.OCI_IMAGE: "container",
    ArtifactType.PYTHON_WHEEL: "library",
    ArtifactType.PYTHON_SDIST: "library",
    ArtifactType.SYSTEM_PACKAGE: "library",
    ArtifactType.MODEL: "machine-learning-model",
    ArtifactType.CONFIG: "file",
    ArtifactType.SCRIPT: "file",
    ArtifactType.MISC: "file",
}


def generate_sbom(manifest: Manifest) -> dict[str, Any]:
    """Build a CycloneDX document describing everything in the bundle."""
    components: list[dict[str, Any]] = []
    seen: set[str] = set()

    for artifact in manifest.artifacts:
        component = _component(artifact)
        # Model shards are many files of one logical component; collapsing them
        # keeps the SBOM about what was shipped rather than how it was split.
        key = f"{component['type']}:{component['name']}:{component.get('version', '')}"
        if artifact.type is ArtifactType.MODEL:
            if key in seen:
                continue
            seen.add(key)
        components.append(component)

    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "tools": [{"vendor": "OfflineAI", "name": "offlineai", "version": __version__}],
            "component": {
                "type": "application",
                "bom-ref": f"pkg:offlineai/{manifest.package.name}@{manifest.package.version}",
                "name": manifest.package.name,
                "version": manifest.package.version,
            },
            "properties": [
                {"name": "offlineai:disclaimer", "value": LICENSE_DISCLAIMER},
                {"name": "offlineai:formatVersion", "value": manifest.format_version},
            ],
        },
        "components": components,
    }


def _component(artifact: Any) -> dict[str, Any]:
    name, version = _identity(artifact)
    component: dict[str, Any] = {
        "type": _COMPONENT_TYPE.get(artifact.type, "file"),
        "bom-ref": artifact.id,
        "name": name,
        "hashes": [{"alg": "SHA-256", "content": artifact.sha256}],
        "properties": [
            {"name": "offlineai:artifactType", "value": artifact.type.value},
            {"name": "offlineai:bundlePath", "value": artifact.path},
            {"name": "offlineai:size", "value": str(artifact.size)},
        ],
    }
    if version:
        component["version"] = version
    if artifact.license:
        # "expression" rather than "id": we record what the source stated and
        # do not assert that it is a valid SPDX identifier.
        component["licenses"] = [{"expression": artifact.license}]
    if artifact.source:
        component["externalReferences"] = [{"type": "distribution", "url": artifact.source}]
    if artifact.digest:
        component["properties"].append({"name": "offlineai:originDigest", "value": artifact.digest})
    return component


def _identity(artifact: Any) -> tuple[str, str | None]:
    metadata = artifact.metadata or {}
    if artifact.type is ArtifactType.PYTHON_WHEEL:
        return str(metadata.get("package") or artifact.id), (
            str(metadata["version"]) if metadata.get("version") else None
        )
    if artifact.type is ArtifactType.OCI_IMAGE:
        return str(metadata.get("image") or artifact.id), (
            str(metadata["tag"]) if metadata.get("tag") else None
        )
    if artifact.type is ArtifactType.MODEL:
        return str(metadata.get("repo") or metadata.get("model") or artifact.id), (
            str(metadata["revision"]) if metadata.get("revision") else None
        )
    return artifact.id, None


def to_json(manifest: Manifest) -> bytes:
    return json.dumps(generate_sbom(manifest), indent=2, sort_keys=True).encode("utf-8")


def license_report(manifest: Manifest) -> dict[str, Any]:
    """Licence metadata per component (section 37).

    Components whose licence is unknown are listed as ``null`` rather than
    omitted: an operator reviewing obligations needs to see what could not be
    determined, not just what could.
    """
    entries: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for artifact in manifest.artifacts:
        name, _ = _identity(artifact)
        if name in seen:
            continue
        seen.add(name)
        entries.append(
            {
                "component": name,
                "type": artifact.type.value,
                "license": artifact.license,
                "source": artifact.source,
            }
        )
    return {
        "disclaimer": LICENSE_DISCLAIMER,
        "licenses": sorted(entries, key=lambda e: str(e["component"])),
    }
