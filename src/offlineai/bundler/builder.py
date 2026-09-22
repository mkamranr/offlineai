"""Bundle construction (sections 41 and 42).

The pipeline is deliberately ordered so that everything which can fail cheaply
fails before anything expensive happens: the definition is validated and
secrets are scanned before a single byte is downloaded.

Artifacts are resolved into the content-addressed cache first, and only then
streamed into the archive. That ordering is what lets the manifest - which must
be written *first*, so that inspect stays cheap - already know every artifact's
digest. It also means an interrupted build resumes from the cache rather than
starting over (section 43).
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from offlineai import __version__, layout
from offlineai.artifacts.base import (
    ArtifactRequest,
    ArtifactSource,
    ResolvedArtifact,
    SourceRef,
)
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.sources.local import LocalSource
from offlineai.bundler.archive import BundleWriter
from offlineai.bundler.results import BuildResult, BuildStep, CheckStatus
from offlineai.bundler.verifier import verify_bundle
from offlineai.config.settings import Settings
from offlineai.errors import InvalidPackageError, SourceError
from offlineai.logging import get_logger
from offlineai.resolver.package import load_package
from offlineai.schema.manifest import (
    ArtifactType,
    BuilderInfo,
    Compression,
    DockerRequirement,
    GpuRequirementSummary,
    Manifest,
    ManifestPackage,
    Requirements,
)
from offlineai.schema.package import ModelSource, Package
from offlineai.security.secrets import scan_for_secrets
from offlineai.utils.hashing import sha256_file

__all__ = ["BundleBuilder"]

logger = get_logger("bundler.builder")

StepHook = Callable[[BuildStep], None]

_STEPS = (
    "Validating package definition",
    "Scanning for secrets",
    "Resolving models",
    "Resolving container images",
    "Resolving Python dependencies",
    "Resolving OS packages",
    "Calculating checksums",
    "Generating manifest",
    "Creating bundle",
    "Verifying bundle",
)


class BundleBuilder:
    """Builds a ``.offlineai`` bundle from a package definition."""

    def __init__(
        self,
        settings: Settings,
        *,
        on_step: StepHook | None = None,
        sources: dict[str, ArtifactSource] | None = None,
    ) -> None:
        self.settings = settings
        self.cache = ArtifactCache(settings.cache_dir)
        self._on_step = on_step
        self._sources: dict[str, ArtifactSource] = sources or {}
        self._steps: list[BuildStep] = []
        self._warnings: list[str] = []

    def build(
        self,
        target: Path | str,
        *,
        output: Path | str | None = None,
        compression: Compression = Compression.NONE,
    ) -> BuildResult:
        self.cache.ensure()
        self._steps = []
        self._warnings = []

        # 1 - validate
        with self._step(1):
            package, definition_path, definition_hash = load_package(target)
            base_dir = definition_path.parent

        # 2 - secrets (section 40): refuse before anything is downloaded
        with self._step(2) as step:
            findings = scan_for_secrets(base_dir) if self.settings.security.detect_secrets else []
            if findings:
                raise InvalidPackageError(
                    f"{len(findings)} credential-shaped file(s) would be included in the bundle.",
                    details={"Files": "\n".join(str(f) for f in findings)},
                    action="Remove them, add them to .offlineaiignore, or declare the "
                    "names under 'secrets.external' so the bundle carries the name "
                    "rather than the value. To override, set security.detect_secrets: "
                    "false in config.yaml.",
                )
            step.detail = "none found"

        resolved: list[ResolvedArtifact] = []

        # 3 - models
        with self._step(3) as step:
            model_artifacts = self._resolve_models(package, base_dir)
            resolved.extend(model_artifacts)
            step.detail = _describe(model_artifacts, "model file")

        # 4-6 are implemented in later phases; report honestly rather than
        # printing OK for work that did not happen.
        with self._step(4) as step:
            if package.containers:
                step.status = CheckStatus.SKIPPED
                step.detail = "container packaging arrives in the Docker phase"
                self._warnings.append(
                    f"{len(package.containers)} container image(s) declared but not yet "
                    "packaged by this build"
                )
            else:
                step.detail = "none declared"

        with self._step(5) as step:
            if package.python and package.python.requirements:
                step.status = CheckStatus.SKIPPED
                step.detail = "Python dependency resolution arrives in a later phase"
                self._warnings.append("Python requirements declared but not yet packaged")
            else:
                step.detail = "none declared"

        with self._step(6) as step:
            if package.system and package.system.packages:
                step.status = CheckStatus.SKIPPED
                step.detail = "OS package resolution arrives in a later phase"
                self._warnings.append("System packages declared but not yet packaged")
            else:
                step.detail = "none declared"

        # 7 - checksums are already known: every artifact was hashed on its way
        # into the cache, so this step only reports.
        with self._step(7) as step:
            step.detail = f"{len(resolved)} artifact(s)"

        # 8 - manifest
        with self._step(8):
            manifest = self._build_manifest(
                package, resolved, definition_hash, compression=compression
            )

        # 9 - archive
        bundle_path = self._output_path(package, output)
        with self._step(9) as step:
            with BundleWriter(bundle_path, compression=compression) as writer:
                writer.write_header(
                    manifest=manifest,
                    package_yaml=definition_path.read_bytes(),
                    docs=_collect_docs(base_dir),
                )
                for artifact in resolved:
                    writer.add_artifact(artifact.request.bundle_path, artifact.local_path)
            step.detail = f"{bundle_path.name}"

        # 10 - verify what we just wrote (section 61)
        with self._step(10) as step:
            report = verify_bundle(bundle_path)
            step.detail = f"{report.artifacts_verified} artifact(s)"

        return BuildResult(
            package=package.metadata.name,
            version=package.metadata.version,
            bundle_path=str(bundle_path),
            size=bundle_path.stat().st_size,
            sha256=sha256_file(bundle_path),
            artifact_count=len(resolved),
            steps=self._steps,
            warnings=self._warnings,
        )

    # -- pipeline stages -------------------------------------------------

    def _resolve_models(self, package: Package, base_dir: Path) -> list[ResolvedArtifact]:
        out: list[ResolvedArtifact] = []
        for model in package.models:
            source_kind = model.source.type
            source = self._source_for(source_kind, base_dir)
            locator = _locator_for(model.source)
            ref = SourceRef(
                kind=source_kind,
                locator=locator,
                name=model.name,
                artifact_type=ArtifactType.MODEL,
            )
            out.extend(self._fetch(source, request) for request in source.expand(ref))
        return out

    def _fetch(self, source: ArtifactSource, request: ArtifactRequest) -> ResolvedArtifact:
        cached = self.cache.lookup_ref(request.source_kind, request.cache_key)
        if cached is not None and request.expected_sha256 in (None, cached.sha256):
            logger.debug("cache hit for %s", request.bundle_path)
            return ResolvedArtifact(
                request=request,
                local_path=cached.path,
                sha256=cached.sha256,
                size=cached.size,
                cached=True,
            )
        return source.fetch(request, self.cache)

    def _source_for(self, kind: str, base_dir: Path) -> ArtifactSource:
        registered = self._sources.get(kind)
        if registered is not None:
            return registered
        if kind == "local":
            return LocalSource(base_dir)
        raise SourceError(
            f"no artifact source is registered for {kind!r}",
            details={"Available": ", ".join(sorted({"local", *self._sources}))},
            action="This source type is not available in this build of OfflineAI.",
        )

    def _build_manifest(
        self,
        package: Package,
        resolved: list[ResolvedArtifact],
        definition_hash: str,
        *,
        compression: Compression,
    ) -> Manifest:
        gpu = package.hardware.gpu
        requirements = Requirements(
            docker=DockerRequirement() if package.runtime.type == "docker" else None,
            gpu=(
                GpuRequirementSummary(
                    vendor=gpu.vendor,
                    minimumDriver=gpu.minimum_driver,
                    minimumMemoryGB=gpu.minimum_vram_gb,
                    count=gpu.count,
                )
                if gpu and gpu.required
                else None
            ),
            minimumRamGB=package.hardware.minimum_ram_gb,
            minimumDiskGB=package.hardware.minimum_disk_gb,
            minimumCpuCores=package.hardware.minimum_cpu_cores,
            pythonVersion=package.python.version if package.python else None,
            pythonPlatform=package.python.platform if package.python else None,
            pythonArchitecture=package.python.architecture if package.python else None,
        )

        return Manifest(
            formatVersion="1",
            package=ManifestPackage(name=package.metadata.name, version=package.metadata.version),
            createdAt=datetime.now(UTC),
            platforms=[f"linux/{arch}" for arch in package.architecture],
            compression=compression,
            artifacts=[a.to_manifest_entry() for a in resolved],
            requirements=requirements,
            packageDefinitionSha256=definition_hash,
            builder=BuilderInfo(
                offlineaiVersion=__version__,
                os=platform.system().lower(),
                architecture=platform.machine(),
                pythonVersion=".".join(str(p) for p in sys.version_info[:3]),
            ),
            requiredSecrets=list(package.secrets.external),
        )

    def _output_path(self, package: Package, output: Path | str | None) -> Path:
        """Resolve where the bundle is written.

        A path ending in .offlineai names the file; anything else names a
        directory, which is created if absent. Deciding by suffix rather than
        by whether the path already exists means the same command does the same
        thing on a first and second run.
        """
        if output is None:
            return Path.cwd() / package.bundle_filename()
        path = Path(output)
        if path.suffix == layout.EXTENSION:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        path.mkdir(parents=True, exist_ok=True)
        return path / package.bundle_filename()

    # -- step reporting --------------------------------------------------

    def _step(self, index: int) -> _StepContext:
        return _StepContext(self, index)


class _StepContext:
    """Records one pipeline step and reports it as it completes."""

    def __init__(self, builder: BundleBuilder, index: int) -> None:
        self._builder = builder
        self._step = BuildStep(index=index, total=len(_STEPS), name=_STEPS[index - 1])

    def __enter__(self) -> BuildStep:
        return self._step

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        if exc_type is not None:
            self._step.status = CheckStatus.FAILED
        self._builder._steps.append(self._step)
        if self._builder._on_step is not None:
            self._builder._on_step(self._step)


def _describe(artifacts: list[ResolvedArtifact], noun: str) -> str:
    if not artifacts:
        return "none declared"
    cached = sum(1 for a in artifacts if a.cached)
    suffix = f", {cached} from cache" if cached else ""
    return f"{len(artifacts)} {noun}(s){suffix}"


def _locator_for(source: ModelSource) -> str:
    for attribute in ("repo", "url", "path"):
        value = getattr(source, attribute, None)
        if value is not None:
            return str(value)
    raise SourceError(f"cannot determine a locator for source {source!r}")


def _collect_docs(base_dir: Path) -> dict[str, bytes]:
    """Carry the package's own README into the bundle, if it has one.

    A bundle should be self-describing on a machine with no way to look
    anything up (section 9).
    """
    docs: dict[str, bytes] = {}
    for name in ("README.md", "README.txt", "README"):
        candidate = base_dir / name
        if candidate.is_file():
            docs[name] = candidate.read_bytes()
            break
    _ = layout  # layout constants are used by the writer
    return docs
