"""Bundle inspection (section 60).

Reads only the header. On a 62 GB bundle this touches roughly ten kilobytes,
which is what makes "inspect must work without installing anything" a usable
requirement rather than an aspiration.
"""

from __future__ import annotations

from pathlib import Path

from offlineai import layout
from offlineai.bundler.archive import BundleReader
from offlineai.bundler.results import InspectResult
from offlineai.schema.manifest import ArtifactType

__all__ = ["inspect_bundle"]


def inspect_bundle(path: Path | str) -> InspectResult:
    path = Path(path)
    with BundleReader.open(path) as reader:
        header = reader.read_header()
        consumed = reader._counter.bytes_read if reader._counter else 0

    manifest = header.manifest
    counts: dict[ArtifactType, int] = {}
    for artifact in manifest.artifacts:
        counts[artifact.type] = counts.get(artifact.type, 0) + 1

    gpu = manifest.requirements.gpu
    docker = manifest.requirements.docker

    return InspectResult(
        package=manifest.package.name,
        version=manifest.package.version,
        format_version=manifest.format_version,
        platforms=manifest.platforms,
        compression=manifest.compression,
        created_at=manifest.created_at.isoformat(),
        artifact_counts=counts,
        artifact_sizes=manifest.size_by_type(),
        total_size=manifest.total_size,
        file_size=path.stat().st_size,
        gpu_vendor=gpu.vendor if gpu else None,
        gpu_minimum_vram_gb=gpu.minimum_memory_gb if gpu else None,
        docker_minimum_version=docker.minimum_version if docker else None,
        python_version=manifest.requirements.python_version,
        signature_present=header.signature is not None,
        sbom_present=header.sbom is not None or layout.SBOM in header.extra,
        required_secrets=manifest.required_secrets,
        builder=(
            manifest.builder.model_dump(by_alias=True, mode="json") if manifest.builder else {}
        ),
        header_bytes_read=consumed,
    )
