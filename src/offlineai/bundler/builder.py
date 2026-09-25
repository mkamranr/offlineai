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
from typing import Literal, cast

from offlineai import __version__, layout
from offlineai.artifacts.base import (
    ArtifactRequest,
    ArtifactSource,
    ResolvedArtifact,
    SourceRef,
)
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.fetcher import fetch_all
from offlineai.artifacts.sources.http import HttpSource
from offlineai.artifacts.sources.huggingface import HuggingFaceSource
from offlineai.artifacts.sources.local import LocalSource
from offlineai.artifacts.sources.oci import OciSource
from offlineai.artifacts.sources.pypi import PythonSource
from offlineai.bundler.archive import BundleWriter
from offlineai.bundler.results import BuildResult, BuildStep, CheckStatus
from offlineai.bundler.verifier import verify_bundle
from offlineai.config.settings import Settings
from offlineai.errors import InvalidPackageError, MissingArtifactError, SourceError
from offlineai.logging import get_logger
from offlineai.progress import NullReporter, ProgressReporter
from offlineai.resolver.package import load_package
from offlineai.runtime.base import ContainerRuntime
from offlineai.runtime.docker import DockerRuntime
from offlineai.sbom.cyclonedx import license_report
from offlineai.sbom.cyclonedx import to_json as sbom_json
from offlineai.schema.lockfile import (
    LOCK_FILENAME,
    LockedArtifact,
    LockedPackage,
    LockedSource,
    Lockfile,
)
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
from offlineai.schema.package import ImageReference, ModelSource, Package
from offlineai.security.secrets import scan_for_secrets
from offlineai.security.signing import load_private_key, sign_manifest
from offlineai.utils.fs import atomic_write_text
from offlineai.utils.hashing import sha256_file

__all__ = ["BundleBuilder"]

logger = get_logger("bundler.builder")

StepHook = Callable[[BuildStep], None]

#: How a build treats offlineai.lock.
#:
#:   refresh  apply existing pins, then write the lock back (the default,
#:            so repeated builds are stable without anyone asking)
#:   locked   apply pins and fail on any drift; never writes
#:   update   ignore existing pins, re-resolve, write the result
#:   none     neither read nor write
LockMode = Literal["refresh", "locked", "update", "none"]

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
        reporter: ProgressReporter | None = None,
        workers: int | None = None,
        lock_mode: LockMode = "refresh",
    ) -> None:
        self.settings = settings
        self._container_runtime = runtime
        self.reporter: ProgressReporter = reporter or NullReporter()
        # A CLI flag wins over the configured value, which wins over the
        # conservative default section 45 asks for.
        self.workers = workers or settings.downloads.workers
        self.lock_mode: LockMode = lock_mode
        self._lock: Lockfile | None = None
        self._locked_sources: list[LockedSource] = []
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
        dry_run: bool = False,
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

        _reject_unsupported(package)

        # Read the existing lock before anything resolves, so its pins can
        # constrain resolution rather than merely be checked afterwards. That
        # distinction is the whole point: checking after the fact tells you a
        # build drifted; pinning stops it drifting.
        self._lock = self._load_lock(definition_path)

        if dry_run:
            # Validate everything that is cheap to check and stop before the
            # first byte is fetched. This is what makes a 62 GB definition
            # reviewable on a laptop.
            return self._dry_run_result(package, base_dir)

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
            # Required OS packages are refused before this point, so reaching
            # here means there are none to resolve.
            optional = len(package.system.optional_packages) if package.system else 0
            recommended = len(package.system.recommended_packages) if package.system else 0
            if optional or recommended:
                step.status = CheckStatus.SKIPPED
                step.detail = (
                    f"{optional + recommended} optional/recommended package(s) are not packaged"
                )
                self._warnings.append(
                    f"{optional + recommended} optional or recommended system "
                    "package(s) were not included. They are advisory, so the "
                    "bundle is still complete with respect to what it requires."
                )
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

        # Fail a drifted --locked build here, before anything large is
        # written.
        self._verify_lock(package, definition_hash, resolved)

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

        self._write_lock(package, definition_path, definition_hash, resolved)

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

    def _dry_run_result(self, package: Package, base_dir: Path) -> BuildResult:
        """Report what a real build would fetch, without fetching it."""
        planned: list[str] = []
        for model in package.models:
            planned.append(f"model {model.name} from {model.source.type}")
        for container in package.containers:
            how = "build" if container.dockerfile else "pull"
            planned.append(f"image {container.reference} ({how})")
        for requirements in package.python.requirements if package.python else []:
            path = base_dir / requirements
            planned.append(
                f"python wheels from {requirements}"
                + ("" if path.is_file() else "  [MISSING FILE]")
            )
        for name in package.system.packages if package.system else []:
            planned.append(f"system package {name}")

        missing = [p for p in planned if "[MISSING FILE]" in p]
        for index, description in enumerate(planned, start=3):
            step = BuildStep(
                index=min(index, len(_STEPS)),
                total=len(_STEPS),
                name="Would resolve",
                status=CheckStatus.SKIPPED,
                detail=description,
            )
            self._steps.append(step)
            if self._on_step is not None:
                self._on_step(step)

        if missing:
            raise InvalidPackageError(
                "the package definition references files that do not exist",
                details={"Missing": "\n".join(missing)},
            )

        return BuildResult(
            package=package.metadata.name,
            version=package.metadata.version,
            bundle_path="(dry run - nothing written)",
            size=0,
            sha256="",
            artifact_count=len(planned),
            steps=self._steps,
            warnings=["Dry run: the definition is valid. Nothing was fetched or written."],
        )

    def _resolve_models(self, package: Package, base_dir: Path) -> list[ResolvedArtifact]:
        """Expand every model declaration, then fetch the files concurrently.

        Expansion is cheap and serial; fetching is network-bound and is where
        the time goes. Gathering all the requests first means one pool covers
        every model rather than one pool per declaration, which matters when a
        package has a small config repo alongside a hundred-shard checkpoint.
        """
        jobs: list[tuple[ArtifactSource, ArtifactRequest]] = []
        licenses: dict[str, str] = {}

        for model in package.models:
            source_kind = model.source.type
            source = self._source_for(source_kind, base_dir)
            locator = _locator_for(model.source)
            options = _source_options(model.source)
            # A pinned revision replaces whatever the definition asked for.
            # `main` moves; the commit it resolved to last time does not.
            pin = self._pin_for(model.name, locator)
            if pin and source_kind == "huggingface":
                options["revision"] = pin

            ref = SourceRef(
                kind=source_kind,
                locator=locator,
                name=model.name,
                artifact_type=ArtifactType.MODEL,
                options=options,
            )
            license_id = _license_for(source, model.source)
            expanded = source.expand(ref)
            self._record_source(
                model.name,
                source_kind,
                locator,
                pin=(
                    str(expanded[0].metadata.get("revision_resolved"))
                    if expanded and expanded[0].metadata.get("revision_resolved")
                    else None
                ),
            )
            for request in expanded:
                jobs.append((source, request))
                if license_id is not None:
                    licenses[request.id] = license_id

        if not jobs:
            return []

        self._report_total(jobs, "Downloading models")
        resolved = fetch_all(jobs, self.cache, workers=self.workers, reporter=self.reporter)
        return [
            r if r.request.id not in licenses else replace(r, license=licenses[r.request.id])
            for r in resolved
        ]

    def _report_total(
        self, jobs: list[tuple[ArtifactSource, ArtifactRequest]], description: str
    ) -> None:
        """Give the aggregate bar a total, when the sources declared sizes."""
        known = [j[1].expected_size for j in jobs if j[1].expected_size]
        self.reporter.set_overall(description, sum(known) if known else None)

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

        # Deliberately serial. `docker save` writes gigabytes through the
        # daemon, which serialises much of it anyway, and two concurrent saves
        # mostly produce disk contention. The parallelism win is in models -
        # many files, network-bound - not here, where there are usually four
        # images at most.
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

            # A tag is mutable. If the lock recorded a digest for this
            # container, fetch that exact image instead.
            declared = reference
            pin = self._pin_for(container.name, declared)
            if pin and not container.dockerfile:
                reference = f"{ImageReference.parse(reference).repository}@{pin}"

            ref = SourceRef(
                kind="oci",
                locator=reference,
                name=container.name,
                artifact_type=ArtifactType.OCI_IMAGE,
                options={"platform": container.platform},
            )
            for request in source.expand(ref):
                resolved = self._fetch(source, request)
                out.append(resolved)
                self._record_source(container.name, "oci", declared, pin=resolved.digest)

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

    # -- lock file (section 35) ------------------------------------------

    def _load_lock(self, definition_path: Path) -> Lockfile | None:
        """Read offlineai.lock, unless this build is meant to ignore it."""
        if self.lock_mode in ("none", "update"):
            return None
        path = definition_path.parent / LOCK_FILENAME
        if not path.is_file():
            if self.lock_mode == "locked":
                raise InvalidPackageError(
                    f"--locked was requested but {LOCK_FILENAME} does not exist",
                    details={"Looked for": str(path)},
                    action="Build once without --locked to generate it, then commit "
                    "it alongside the package definition.",
                )
            return None
        try:
            return Lockfile.from_yaml(path.read_text())
        except Exception as exc:
            raise InvalidPackageError(
                f"{path} is not a readable lock file",
                details={"Detail": str(exc)},
                action="Delete it and rebuild to regenerate, or fix it by hand.",
            ) from exc

    def _pin_for(self, name: str, locator: str) -> str | None:
        if self._lock is None:
            return None
        return self._lock.pin_for(name, locator=locator)

    def _record_source(self, name: str, kind: str, locator: str, *, pin: str | None) -> None:
        self._locked_sources.append(LockedSource(name=name, kind=kind, locator=locator, pin=pin))

    def _current_lock(
        self,
        package: Package,
        definition_hash: str,
        resolved: list[ResolvedArtifact],
    ) -> Lockfile:
        return Lockfile(
            formatVersion="1",
            package=LockedPackage(name=package.metadata.name, version=package.metadata.version),
            generatedAt=datetime.now(UTC),
            packageDefinitionSha256=definition_hash,
            sources=self._locked_sources,
            artifacts=[
                LockedArtifact(
                    path=a.request.bundle_path,
                    type=a.request.artifact_type.value,
                    sha256=a.sha256,
                    size=a.size,
                    version=(
                        str(a.request.metadata["version"])
                        if a.request.metadata.get("version")
                        else None
                    ),
                    digest=a.digest,
                )
                for a in resolved
            ],
        )

    def _verify_lock(
        self,
        package: Package,
        definition_hash: str,
        resolved: list[ResolvedArtifact],
    ) -> None:
        """Fail a --locked build that has drifted.

        Called before the archive is written, not after. Everything needed to
        detect drift is known once resolution finishes, and writing 62 GB
        before discovering the build was not the one asked for wastes both the
        time and the disk.
        """
        if self.lock_mode != "locked" or self._lock is None:
            return

        current = self._current_lock(package, definition_hash, resolved)
        drift = self._lock.drift_against(current.artifacts)
        if drift:
            raise MissingArtifactError(
                f"this build does not match {LOCK_FILENAME}",
                details={"Drift": "\n".join(drift)},
                action="Something the package depends on has changed since the "
                "lock was written. Review the differences above. If they are "
                "intended, rebuild with --update-lock and commit the new lock.",
            )

    def _write_lock(
        self,
        package: Package,
        definition_path: Path,
        definition_hash: str,
        resolved: list[ResolvedArtifact],
    ) -> None:
        """Record what this build resolved to.

        A --locked build never writes: its job is to prove the file on disk
        still describes reality, and rewriting it would destroy the evidence
        it was asked to check.
        """
        if self.lock_mode in ("none", "locked"):
            return

        current = self._current_lock(package, definition_hash, resolved)
        path = definition_path.parent / LOCK_FILENAME
        try:
            atomic_write_text(path, current.to_yaml())
        except OSError as exc:
            # A read-only source tree is a legitimate way to build. Losing the
            # lock is worth a warning, not a failed build.
            self._warnings.append(f"could not write {LOCK_FILENAME}: {exc}")

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
                source=cached.source,
                digest=cached.digest,
                license=cached.license,
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
            return cast("ArtifactSource", HuggingFaceSource())
        if kind == "http":
            return cast("ArtifactSource", HttpSource())
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


def _reject_unsupported(package: Package) -> None:
    """Fail on anything declared that this build cannot actually deliver.

    Section 74 is the whole promise: a successful build must mean the bundle
    contains everything required for the declared offline installation. OS
    package resolution (section 19) is not implemented, so a package that
    requires one would otherwise produce a bundle that verifies, reports
    success and silently lacks it - discovered on the air-gapped side, where
    it cannot be fixed. Warning was not enough.

    Optional and recommended packages are advisory by definition, so they warn
    rather than fail.
    """
    required = package.system.packages if package.system else []
    if not required:
        return

    raise MissingArtifactError(
        f"this build cannot package the {len(required)} required system "
        "package(s) this package declares",
        details={
            "Declared": ", ".join(required),
            "Not supported": "OS package resolution (specification section 19) "
            "is not implemented in this release.",
        },
        action="For a containerised workload the right home for an OS dependency "
        "is the image: add it to the container's Dockerfile with apt-get, and the "
        "builder will package the built image.\n"
        "If the dependency is genuinely advisory, move it to "
        "'system.optional_packages' or 'system.recommended_packages'.",
    )


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
