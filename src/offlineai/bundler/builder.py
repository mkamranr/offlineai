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

import base64
import json
import platform
import sys
from collections.abc import Callable
from dataclasses import replace
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
from offlineai.artifacts.sources.http import HttpSource
from offlineai.artifacts.sources.huggingface import HuggingFaceSource
from offlineai.artifacts.sources.local import LocalSource
from offlineai.artifacts.sources.oci import OciSource
from offlineai.artifacts.sources.pypi import PythonSource
from offlineai.bundler.archive import BundleWriter
from offlineai.bundler.results import BuildResult, BuildStep, CheckStatus
from offlineai.bundler.verifier import verify_bundle
from offlineai.config.settings import Settings
from offlineai.errors import InvalidPackageError, SourceError
from offlineai.logging import get_logger
from offlineai.resolver.package import load_package
from offlineai.runtime.base import ContainerRuntime
from offlineai.runtime.docker import DockerRuntime
from offlineai.sbom.cyclonedx import license_report
from offlineai.sbom.cyclonedx import to_json as sbom_json
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
from offlineai.security.signing import load_private_key, sign_manifest
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
        runtime: ContainerRuntime | None = None,
    ) -> None:
        self.settings = settings
        self._container_runtime = runtime
        self.cache = ArtifactCache(settings.cache_dir)
        self._on_step = on_step
        self._sources: dict[str, ArtifactSource] = sources or {}
        self._steps: list[BuildStep] = []
        self._python_target: tuple[str, str] | None = None
        self._warnings: list[str] = []

    def build(
        self,
        target: Path | str,
        *,
        output: Path | str | None = None,
        compression: Compression = Compression.NONE,
        sign_key: Path | str | None = None,
        signer: str | None = None,
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

        # 4 - container images
        with self._step(4) as step:
            container_artifacts = self._resolve_containers(package, base_dir)
            resolved.extend(container_artifacts)
            step.detail = _describe(container_artifacts, "image")

        # 5 - Python dependency closure
        with self._step(5) as step:
            wheel_artifacts = self._resolve_python(package, base_dir)
            resolved.extend(wheel_artifacts)
            step.detail = _describe(wheel_artifacts, "wheel")

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

        # 9 - archive. The SBOM and signature are written into the header
        # here, which is free; signing afterwards would mean rewriting the
        # whole archive to insert 64 bytes.
        bundle_path = self._output_path(package, output)
        sbom_bytes = sbom_json(manifest)
        licenses_bytes = json.dumps(license_report(manifest), indent=2, sort_keys=True).encode(
            "utf-8"
        )

        signature_bytes: bytes | None = None
        public_key_bytes: bytes | None = None
        if sign_key is not None:
            key = load_private_key(sign_key)
            envelope = sign_manifest(manifest, key, signer=signer)
            signature_bytes = envelope.to_bytes()
            public_key_bytes = base64.b64decode(envelope.public_key)

        with self._step(9) as step:
            with BundleWriter(bundle_path, compression=compression) as writer:
                writer.write_header(
                    manifest=manifest,
                    package_yaml=definition_path.read_bytes(),
                    sbom=sbom_bytes,
                    licenses=licenses_bytes,
                    signature=signature_bytes,
                    public_key=public_key_bytes,
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
                options=_source_options(model.source),
            )
            license_id = _license_for(source, model.source)
            for request in source.expand(ref):
                resolved = self._fetch(source, request)
                out.append(
                    resolved if license_id is None else replace(resolved, license=license_id)
                )
        return out

    def _resolve_containers(self, package: Package, base_dir: Path) -> list[ResolvedArtifact]:
        """Build or pull each declared image, then save it into the bundle.

        A container with a `dockerfile:` is built here rather than pulled: that
        is how an application image that only exists in this repository gets
        into the bundle at all.
        """
        if not package.containers:
            return []

        runtime = self._runtime()
        source = self._sources.get("oci") or OciSource(runtime)

        out: list[ResolvedArtifact] = []
        for container in package.containers:
            reference = str(container.reference)

            if container.dockerfile:
                dockerfile = base_dir / container.dockerfile
                if not dockerfile.is_file():
                    raise InvalidPackageError(
                        f"container {container.name!r} declares dockerfile "
                        f"{container.dockerfile!r}, which does not exist",
                        details={"Looked for": str(dockerfile)},
                    )
                # Tag with the package identity so the built image is
                # distinguishable from the base image it derives from, and so
                # two packages building from python:3.12-slim do not collide.
                reference = (
                    f"offlineai/{package.metadata.name}-{container.name}:{package.metadata.version}"
                )
                context_dir = base_dir / (container.context or ".")
                logger.info("building image %s", reference)
                runtime.build(
                    dockerfile=dockerfile,
                    context=context_dir,
                    tag=reference,
                    platform=container.platform,
                )

            ref = SourceRef(
                kind="oci",
                locator=reference,
                name=container.name,
                artifact_type=ArtifactType.OCI_IMAGE,
                options={"platform": container.platform},
            )
            for request in source.expand(ref):
                out.append(self._fetch(source, request))

        return out

    def _resolve_python(self, package: Package, base_dir: Path) -> list[ResolvedArtifact]:
        """Resolve the transitive wheel closure for the declared target.

        Explicitly for the *target*, not for the builder: a macOS builder that
        resolved for itself would produce a bundle of macosx wheels that cannot
        install on the Linux host they were meant for.
        """
        if not (package.python and package.python.requirements):
            return []

        target_platform = f"{package.python.platform}/{package.python.architecture}"
        python_version = package.python.version or _current_python_version()

        source = self._sources.get("pypi") or PythonSource()
        out: list[ResolvedArtifact] = []
        try:
            for requirements in package.python.requirements:
                ref = SourceRef(
                    kind="pypi",
                    locator=str(base_dir / requirements),
                    name=Path(requirements).stem,
                    artifact_type=ArtifactType.PYTHON_WHEEL,
                    options={
                        "platform": target_platform,
                        "python_version": python_version,
                    },
                )
                out.extend(self._fetch(source, request) for request in source.expand(ref))
        finally:
            cleanup = getattr(source, "cleanup", None)
            if cleanup is not None:
                cleanup()

        if out:
            self._python_target = (python_version, target_platform)
        return out

    def _runtime(self) -> ContainerRuntime:
        if self._container_runtime is None:
            self._container_runtime = DockerRuntime()
        return self._container_runtime

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
        if kind == "huggingface":
            return HuggingFaceSource()
        if kind == "http":
            return HttpSource()
        raise SourceError(
            f"no artifact source is registered for {kind!r}",
            details={
                "Available": ", ".join(
                    sorted({"local", "huggingface", "http", "oci", *self._sources})
                )
            },
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
            docker=(
                DockerRequirement(minimumVersion=None) if package.runtime.type == "docker" else None
            ),
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
            pythonVersion=(
                self._python_target[0]
                if self._python_target
                else (package.python.version if package.python else None)
            ),
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


def _current_python_version() -> str:
    return ".".join(str(p) for p in sys.version_info[:2])


def _source_options(source: ModelSource) -> dict[str, object]:
    """Per-source options carried from the package definition into expansion."""
    options: dict[str, object] = {}
    for attribute in ("revision", "include", "exclude", "sha256"):
        value = getattr(source, attribute, None)
        if value:
            options[attribute] = value
    return options


def _license_for(source: ArtifactSource, model_source: ModelSource) -> str | None:
    """Best-effort license id (section 37): informational, never a gate."""
    reader = getattr(source, "license_for", None)
    repo = getattr(model_source, "repo", None)
    if reader is None or not repo:
        return None
    try:
        value = reader(repo, getattr(model_source, "revision", None) or "main")
    except Exception:  # noqa: BLE001 - metadata must never fail a build
        return None
    return str(value) if value else None


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
