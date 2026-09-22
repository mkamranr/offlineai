"""Python dependency source (sections 17 and 18).

The hard part is not downloading wheels; it is that **the builder is rarely the
target**. A build on macOS that resolves wheels for the machine it is running
on produces a bundle full of ``macosx_11_0_x86_64`` artifacts that cannot
install on the Linux host they were meant for - and the failure surfaces on the
air-gapped side, where it cannot be fixed.

So resolution is always explicit about the target: platform, Python version and
ABI are passed to pip rather than inherited from the process. Two consequences
follow, and both are deliberate:

* ``--only-binary=:all:`` is mandatory. pip cannot cross-compile an sdist, and
  a source distribution that needs a compiler at install time is exactly what
  an air-gapped host cannot provide.
* A package with no matching wheel fails the build, loudly, naming the package.
  Section 74: a successful build must mean the bundle is complete.
"""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from offlineai import layout
from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact, SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.errors import MissingArtifactError, SourceError
from offlineai.logging import get_logger
from offlineai.schema.manifest import ArtifactType
from offlineai.utils.proc import run

__all__ = ["PLATFORM_TAGS", "PythonSource", "WheelInfo", "wheel_tags"]

logger = get_logger("artifacts.pypi")

#: Manylinux tags to try, newest first. pip matches the first the wheel
#: declares, and asking for several covers projects that publish only an older
#: tag alongside those that publish the newest.
PLATFORM_TAGS: dict[str, tuple[str, ...]] = {
    "linux/amd64": (
        "manylinux_2_28_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
        "linux_x86_64",
    ),
    "linux/arm64": (
        "manylinux_2_28_aarch64",
        "manylinux_2_17_aarch64",
        "manylinux2014_aarch64",
        "linux_aarch64",
    ),
}

_WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


@dataclass(frozen=True, slots=True)
class WheelInfo:
    filename: str
    name: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str

    @property
    def pure_python(self) -> bool:
        return self.platform_tag == "any"


def wheel_tags(filename: str) -> WheelInfo | None:
    """Parse a wheel filename into its component tags (PEP 427)."""
    match = _WHEEL_RE.match(filename)
    if not match:
        return None
    return WheelInfo(
        filename=filename,
        name=match.group("name").replace("_", "-"),
        version=match.group("version"),
        python_tag=match.group("python"),
        abi_tag=match.group("abi"),
        platform_tag=match.group("platform"),
    )


class PythonSource:
    kind: ClassVar[str] = "pypi"

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        index_url: str | None = None,
        extra_index_urls: list[str] | None = None,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.index_url = index_url
        self.extra_index_urls = extra_index_urls or []
        self._staging: Path | None = None

    # -- expansion -------------------------------------------------------

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        """Resolve a requirements file into a complete wheel closure.

        Resolution and download are the same operation for pip, so this does
        the work and hands ``fetch`` already-materialised files.
        """
        requirements = Path(ref.locator)
        if not requirements.is_file():
            raise SourceError(
                f"requirements file {requirements} does not exist",
                action="Check the path in the 'python.requirements' section of your "
                "package definition.",
            )

        if _is_effectively_empty(requirements):
            logger.info("%s declares no requirements", requirements.name)
            return []

        target_platform = str(ref.options.get("platform") or "linux/amd64")
        python_version = str(ref.options.get("python_version") or _current_python())

        staging = Path(tempfile.mkdtemp(prefix="offlineai-wheels-"))
        self._staging = staging
        self._download(requirements, staging, target_platform, python_version)

        wheels = sorted(staging.glob("*.whl"))
        sdists = sorted(p for p in staging.iterdir() if p.suffix in (".gz", ".zip") and p.is_file())
        if sdists:
            # Should be unreachable with --only-binary=:all:, but if pip ever
            # hands one back, say so rather than shipping something the target
            # would need a compiler to install.
            raise SourceError(
                "the resolver produced source distributions, which cannot be "
                "installed on an air-gapped host without a build toolchain",
                details={"Source distributions": ", ".join(p.name for p in sdists)},
                action="Pin versions that publish wheels for the target platform.",
            )
        if not wheels:
            raise SourceError(
                f"no wheels were resolved from {requirements.name}",
                action="Check that the requirements file lists installable packages.",
            )

        requests: list[ArtifactRequest] = []
        for wheel in wheels:
            info = wheel_tags(wheel.name)
            metadata: dict[str, object] = {
                "filename": wheel.name,
                "target_platform": target_platform,
                "python_version": python_version,
            }
            if info is not None:
                metadata.update(
                    {
                        "package": info.name,
                        "version": info.version,
                        "python_tag": info.python_tag,
                        "abi_tag": info.abi_tag,
                        "platform_tag": info.platform_tag,
                    }
                )
            requests.append(
                ArtifactRequest(
                    id=f"wheel-{info.name if info else wheel.stem}",
                    artifact_type=ArtifactType.PYTHON_WHEEL,
                    bundle_path=layout.artifact_path(ArtifactType.PYTHON_WHEEL, wheel.name),
                    locator=str(wheel),
                    source_kind=self.kind,
                    expected_size=wheel.stat().st_size,
                    metadata=metadata,
                )
            )

        logger.info("resolved %d wheel(s) for %s", len(requests), target_platform)
        return requests

    def fetch(self, request: ArtifactRequest, cache: ArtifactCache) -> ResolvedArtifact:
        path = Path(request.locator)
        if not path.is_file():
            raise SourceError(
                f"resolved wheel {path.name} is no longer present",
                action="Re-run the build.",
            )
        entry = cache.store_file(path)
        cache.put_ref(self.kind, request.cache_key, entry.sha256)
        return ResolvedArtifact(
            request=request,
            local_path=entry.path,
            sha256=entry.sha256,
            size=entry.size,
            source=f"pypi://{request.metadata.get('package')}=={request.metadata.get('version')}",
        )

    def cleanup(self) -> None:
        if self._staging is not None and self._staging.is_dir():
            import shutil

            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None

    # -- pip -------------------------------------------------------------

    def _download(
        self, requirements: Path, destination: Path, target_platform: str, python_version: str
    ) -> None:
        tags = PLATFORM_TAGS.get(target_platform)
        if tags is None:
            raise SourceError(
                f"no wheel platform tags are known for {target_platform!r}",
                details={"Supported": ", ".join(sorted(PLATFORM_TAGS))},
            )

        major_minor = ".".join(python_version.split(".")[:2])
        abi = "cp" + major_minor.replace(".", "")

        args = [
            self.python_executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(destination),
            "--requirement",
            str(requirements),
            # Mandatory with --platform: pip cannot build an sdist for a
            # platform it is not running on, and a target that needs a
            # compiler at install time defeats the purpose.
            "--only-binary=:all:",
            "--python-version",
            major_minor,
            "--implementation",
            "cp",
            "--abi",
            abi,
            # Keeps pip's upgrade notice out of our error output.
            "--disable-pip-version-check",
        ]
        for tag in tags:
            args += ["--platform", tag]
        if self.index_url:
            args += ["--index-url", self.index_url]
        for extra in self.extra_index_urls:
            args += ["--extra-index-url", extra]

        logger.info("resolving wheels for %s, Python %s", target_platform, major_minor)
        result = run(args, timeout=1800)
        if not result.ok:
            lowered = result.output.lower()
            # A dependency that simply cannot be obtained for the target is
            # section 66's "missing dependency" (exit 5), not a generic error.
            # CI branches on the difference.
            unobtainable = (
                "none of the wheels" in lowered
                or "no matching distribution" in lowered
                or "could not find a version" in lowered
            )
            error = MissingArtifactError if unobtainable else SourceError
            raise error(
                "Python dependency resolution failed",
                details={
                    "Target": f"{target_platform}, Python {major_minor}, {abi}",
                    "Output": _trim(result.output),
                },
                action=_pip_advice(result.output, major_minor, target_platform),
            )


def _pip_advice(output: str, python_version: str, target_platform: str) -> str:
    lowered = output.lower()
    if "no module named pip" in lowered:
        return (
            "The builder's Python has no pip, which is needed to resolve the "
            "dependency closure. Install it:\n"
            "  python -m ensurepip --upgrade\n"
            "pip is only required on the builder, never on the air-gapped target."
        )
    if "none of the wheels" in lowered or "no matching distribution" in lowered:
        package = _guess_package(output)
        subject = f"{package} does not" if package else "A required package does not"
        return (
            f"{subject} publish a wheel for {target_platform} on Python "
            f"{python_version}.\n"
            "An air-gapped host cannot build from source, so the bundle would be "
            "incomplete.\n"
            "Either pin a version that ships a matching wheel, or change the "
            "declared python.version / python.platform to match one that exists."
        )
    if "could not find a version" in lowered:
        return (
            "A requirement could not be resolved. Check the version specifiers in "
            "your requirements file."
        )
    if "connection" in lowered or "network" in lowered or "timed out" in lowered:
        return "The builder could not reach the package index. Check connectivity."
    return (
        "Review the resolver output above. Wheels must exist for the declared "
        "target platform and Python version."
    )


def _guess_package(output: str) -> str | None:
    match = re.search(
        r"(?:none of the wheels|no matching distribution found)[^\n]*?for ([A-Za-z0-9._-]+)",
        output,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _trim(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def _is_effectively_empty(requirements: Path) -> bool:
    for line in requirements.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def _current_python() -> str:
    return ".".join(str(p) for p in sys.version_info[:2])
