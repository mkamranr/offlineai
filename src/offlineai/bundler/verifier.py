"""Bundle verification (sections 11 and 61).

One streaming pass. Each artifact is hashed as its bytes go by and compared
against the manifest; nothing is extracted and nothing is buffered, so this
works on a bundle larger than the free disk.

The governing rule is section 61: *a corrupted bundle must never be reported as
valid*. There is therefore no "mostly fine" outcome and no way to downgrade a
mismatch to a warning. Verification either passes completely or fails.
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Callable
from pathlib import Path

from offlineai.bundler.archive import BundleReader
from offlineai.bundler.results import CheckResult, CheckStatus, VerifyResult
from offlineai.errors import ChecksumMismatchError, VerificationError
from offlineai.logging import get_logger
from offlineai.schema.manifest import ArtifactType
from offlineai.utils.hashing import CHUNK_SIZE

__all__ = ["verify_bundle"]

logger = get_logger("bundler.verifier")

ProgressHook = Callable[[str, int, int], None]

#: Category labels, in the order section 11 prints them.
_CATEGORIES: tuple[tuple[str, tuple[ArtifactType, ...]], ...] = (
    ("Model files", (ArtifactType.MODEL,)),
    ("Container", (ArtifactType.OCI_IMAGE,)),
    ("Python wheels", (ArtifactType.PYTHON_WHEEL, ArtifactType.PYTHON_SDIST)),
    ("System packages", (ArtifactType.SYSTEM_PACKAGE,)),
    ("Other artifacts", (ArtifactType.CONFIG, ArtifactType.SCRIPT, ArtifactType.MISC)),
)


def verify_bundle(
    path: Path | str,
    *,
    progress: ProgressHook | None = None,
) -> VerifyResult:
    """Verify every artifact against the manifest.

    Raises :class:`~offlineai.errors.ChecksumMismatchError` on the first
    mismatch. Callers that want a report rather than an exception should catch
    it - but they must not continue as though the bundle were usable.
    """
    path = Path(path)
    checks: list[CheckResult] = []
    verified_bytes = 0
    seen: set[str] = set()
    failures_by_category: dict[str, int] = {}

    with BundleReader.open(path) as reader:
        header = reader.read_header()
        manifest = header.manifest
        checks.append(CheckResult(name="Manifest", status=CheckStatus.OK))

        declared = {a.path: a for a in manifest.artifacts}
        total = len(declared)

        for entry, stream in reader.iter_artifacts():
            digest = hashlib.sha256()
            size = 0
            try:
                while chunk := stream.read(CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
            except tarfile.TarError as exc:
                # A member whose data runs off the end of the file: the usual
                # signature of a transfer that was interrupted.
                raise VerificationError(
                    f"the bundle is truncated while reading {entry.path}",
                    details={"Detail": str(exc)},
                    action="Re-copy the bundle from the builder and verify again.",
                ) from exc

            actual = digest.hexdigest()
            if actual != entry.sha256 or size != entry.size:
                failures_by_category[_category_for(entry.type)] = 1
                raise ChecksumMismatchError(
                    path=entry.path,
                    expected=entry.sha256,
                    actual=actual if size == entry.size else f"{actual} ({size} bytes)",
                )

            seen.add(entry.path)
            verified_bytes += size
            if progress is not None:
                progress(entry.path, len(seen), total)

        # An artifact the manifest declares but the archive does not contain is
        # exactly the failure mode section 74 exists to prevent.
        missing = sorted(set(declared) - seen)
        if missing:
            raise VerificationError(
                f"the bundle is missing {len(missing)} artifact(s) declared in its manifest",
                details={"Missing": "\n".join(missing)},
                action="The bundle is incomplete or truncated. Re-transfer it from "
                "the builder and verify again.",
            )

        # Per-category reporting, matching the section 11 layout.
        for label, types in _CATEGORIES:
            present = [a for a in manifest.artifacts if a.type in types]
            if not present:
                continue
            checks.append(CheckResult(name=label, status=CheckStatus.OK))

        checks.append(
            CheckResult(
                name="Checksums",
                status=CheckStatus.OK,
                detail=f"{len(seen)} artifact(s)",
            )
        )
        checks.append(
            CheckResult(
                name="SBOM",
                status=CheckStatus.OK if header.sbom else CheckStatus.SKIPPED,
                detail=None if header.sbom else "not present in this bundle",
            )
        )
        checks.append(
            CheckResult(
                name="Signature",
                status=CheckStatus.OK if header.signature else CheckStatus.SKIPPED,
                detail=None if header.signature else "bundle is unsigned",
            )
        )

    return VerifyResult(
        package=manifest.package.name,
        version=manifest.package.version,
        checks=checks,
        artifacts_verified=len(seen),
        bytes_verified=verified_bytes,
        verified=True,
    )


def _category_for(artifact_type: ArtifactType) -> str:
    for label, types in _CATEGORIES:
        if artifact_type in types:
            return label
    return "Other artifacts"
