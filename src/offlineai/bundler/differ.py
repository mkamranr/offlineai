"""Bundle comparison (section 59).

Useful for controlled deployments: before promoting v2 into a secure
environment, an operator needs to know exactly what changed. "The vLLM image
digest moved and transformers went 4.x to 5.0" is a reviewable statement; "it
is a new 81 GB file" is not.

Compares manifests only, so it reads a few kilobytes from the front of each
bundle rather than both payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from offlineai.schema.manifest import ArtifactEntry, ArtifactType, Manifest

__all__ = ["BundleDiff", "ChangedArtifact", "diff_manifests"]


@dataclass(frozen=True, slots=True)
class ChangedArtifact:
    path: str
    artifact_type: ArtifactType
    before: ArtifactEntry
    after: ArtifactEntry

    @property
    def size_delta(self) -> int:
        return self.after.size - self.before.size

    @property
    def what_changed(self) -> list[str]:
        changes: list[str] = []
        if self.before.sha256 != self.after.sha256:
            changes.append("content")
        if self.before.digest != self.after.digest:
            changes.append("origin digest")
        if self.before.size != self.after.size:
            changes.append("size")
        version_before = self.before.metadata.get("version")
        version_after = self.after.metadata.get("version")
        if version_before != version_after:
            changes.append(f"version {version_before} -> {version_after}")
        return changes or ["metadata"]


@dataclass(slots=True)
class BundleDiff:
    left: str
    right: str
    added: list[ArtifactEntry] = field(default_factory=list)
    removed: list[ArtifactEntry] = field(default_factory=list)
    changed: list[ChangedArtifact] = field(default_factory=list)
    size_before: int = 0
    size_after: int = 0
    requirements_changed: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not (self.added or self.removed or self.changed or self.requirements_changed)

    @property
    def size_delta(self) -> int:
        return self.size_after - self.size_before


def diff_manifests(before: Manifest, after: Manifest) -> BundleDiff:
    """Compare two manifests.

    Artifacts are matched by bundle path. Path is the stable identity here:
    ids are derived from source references and can churn for reasons that are
    not a real change.
    """
    left = {a.path: a for a in before.artifacts}
    right = {a.path: a for a in after.artifacts}

    diff = BundleDiff(
        left=f"{before.package.name} {before.package.version}",
        right=f"{after.package.name} {after.package.version}",
        size_before=before.total_size,
        size_after=after.total_size,
    )

    diff.added = [right[p] for p in sorted(set(right) - set(left))]
    diff.removed = [left[p] for p in sorted(set(left) - set(right))]
    diff.changed = [
        ChangedArtifact(
            path=path,
            artifact_type=right[path].type,
            before=left[path],
            after=right[path],
        )
        for path in sorted(set(left) & set(right))
        if left[path].sha256 != right[path].sha256 or left[path].digest != right[path].digest
    ]
    diff.requirements_changed = _requirement_changes(before, after)
    return diff


def _requirement_changes(before: Manifest, after: Manifest) -> list[str]:
    """Report requirement changes in plain language.

    A bundle that quietly starts demanding 80 GB of VRAM instead of 48 is a
    deployment-blocking change, and it will not show up as an artifact diff.
    """
    changes: list[str] = []
    a, b = before.requirements, after.requirements

    def compare(label: str, old: object, new: object) -> None:
        if old != new:
            changes.append(f"{label}: {old or 'none'} -> {new or 'none'}")

    compare("minimum RAM (GB)", a.minimum_ram_gb, b.minimum_ram_gb)
    compare("minimum disk (GB)", a.minimum_disk_gb, b.minimum_disk_gb)
    compare("minimum CPU cores", a.minimum_cpu_cores, b.minimum_cpu_cores)
    compare("Python version", a.python_version, b.python_version)
    compare(
        "Docker minimum version",
        a.docker.minimum_version if a.docker else None,
        b.docker.minimum_version if b.docker else None,
    )

    old_gpu, new_gpu = a.gpu, b.gpu
    if (old_gpu is None) != (new_gpu is None):
        changes.append("GPU requirement: " + ("added" if new_gpu is not None else "removed"))
    elif old_gpu is not None and new_gpu is not None:
        compare("GPU vendor", old_gpu.vendor, new_gpu.vendor)
        compare("GPU minimum VRAM (GB)", old_gpu.minimum_memory_gb, new_gpu.minimum_memory_gb)
        compare("GPU minimum driver", old_gpu.minimum_driver, new_gpu.minimum_driver)
        compare("GPU count", old_gpu.count, new_gpu.count)

    if set(before.platforms) != set(after.platforms):
        changes.append(
            f"platforms: {', '.join(sorted(before.platforms))} -> "
            f"{', '.join(sorted(after.platforms))}"
        )
    if set(before.required_secrets) != set(after.required_secrets):
        changes.append(
            f"required secrets: {', '.join(sorted(before.required_secrets)) or 'none'} -> "
            f"{', '.join(sorted(after.required_secrets)) or 'none'}"
        )
    return changes
