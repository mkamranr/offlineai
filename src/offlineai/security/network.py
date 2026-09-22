"""External dependency audit (section 33).

Not a packet sniffer. It reads the package definition and the manifest and
reports every external reference it finds, classified as **build time** or
**runtime**.

That distinction is the whole point. A package that fetched a model from
huggingface.co while being built is fine - that is what the builder is for.
The same URL appearing in a service's runtime configuration is a defect: it
means the workload will reach out on the air-gapped host, where it cannot.

Findings are reported, never guessed at. Anything that cannot be classified
confidently is marked ``UNKNOWN`` and shown, because a URL an audit quietly
dropped is worse than one it could not categorise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlparse

from offlineai.schema.manifest import Manifest
from offlineai.schema.package import Package

__all__ = ["Finding", "NetworkAudit", "Phase", "audit_package"]

_URL_RE = re.compile(r"\b(?:https?|ftp|s3|gs)://[^\s\"'`<>\\)\]}]+", re.IGNORECASE)

#: Hosts that only ever matter while building. Seeing one at runtime is the
#: finding this audit exists to surface.
_BUILD_TIME_HOSTS = frozenset(
    {
        "huggingface.co",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "pypi.org",
        "files.pythonhosted.org",
        "registry-1.docker.io",
        "index.docker.io",
        "auth.docker.io",
        "ghcr.io",
        "quay.io",
        "archive.ubuntu.com",
        "security.ubuntu.com",
        "deb.debian.org",
        "developer.download.nvidia.com",
        "github.com",
        "raw.githubusercontent.com",
        "modelscope.cn",
    }
)

#: Environment variables whose value being a URL is a runtime dependency.
_RUNTIME_URL_KEYS = re.compile(r"(_URL|_ENDPOINT|_HOST|_URI|_SERVER|_API|_BASE)$", re.IGNORECASE)

#: Loopback and private ranges: a URL pointing at one of these is internal to
#: the deployment and not an external dependency.
_LOCAL_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|\[::1\]|"
    r"10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)$",
    re.IGNORECASE,
)


class Phase(StrEnum):
    BUILD = "BUILD TIME"
    RUNTIME = "RUNTIME"
    LOCAL = "LOCAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Finding:
    url: str
    phase: Phase
    location: str
    detail: str | None = None

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc or self.url


@dataclass(slots=True)
class NetworkAudit:
    package: str
    version: str
    findings: list[Finding] = field(default_factory=list)

    def of_phase(self, phase: Phase) -> list[Finding]:
        return [f for f in self.findings if f.phase is phase]

    @property
    def runtime_dependencies(self) -> list[Finding]:
        """Findings that would make the workload reach out once installed.

        These are the ones that matter on an air-gapped host.
        """
        return self.of_phase(Phase.RUNTIME) + self.of_phase(Phase.UNKNOWN)

    @property
    def clean(self) -> bool:
        return not self.runtime_dependencies

    @property
    def summary(self) -> str:
        if self.clean:
            return "No runtime network dependency detected."
        count = len(self.runtime_dependencies)
        return f"{count} possible runtime network dependency(ies) detected."


def audit_package(package: Package, manifest: Manifest | None = None) -> NetworkAudit:
    """Inspect a package definition, and optionally its manifest."""
    audit = NetworkAudit(package=package.metadata.name, version=package.metadata.version)
    # Service and container names resolve on the deployment network. Knowing
    # them turns http://vllm:8000 from a suspicious URL into a definitively
    # internal one.
    internal = {s.name for s in package.services} | {c.name for c in package.containers}

    for model in package.models:
        locator = getattr(model.source, "repo", None) or getattr(model.source, "url", None)
        if locator:
            audit.findings.append(
                Finding(
                    url=str(locator) if "://" in str(locator) else f"hf://{locator}",
                    phase=Phase.BUILD,
                    location=f"models.{model.name}.source",
                    detail="resolved on the builder and packaged into the bundle",
                )
            )

    for container in package.containers:
        audit.findings.append(
            Finding(
                url=str(container.reference),
                phase=Phase.BUILD,
                location=f"containers.{container.name}.image",
                detail="pulled on the builder and saved into the bundle",
            )
        )

    if package.python and package.python.requirements:
        audit.findings.append(
            Finding(
                url="https://pypi.org/simple/",
                phase=Phase.BUILD,
                location="python.requirements",
                detail="wheels are resolved on the builder and packaged",
            )
        )

    # Environment values are where a genuine runtime dependency usually hides.
    for scope, environment in [("environment", package.environment)] + [
        (f"services.{s.name}.environment", s.environment) for s in package.services
    ]:
        for key, value in environment.items():
            for url in _URL_RE.findall(str(value)):
                audit.findings.append(
                    Finding(
                        url=url,
                        phase=_classify(url, key, internal),
                        location=f"{scope}.{key}",
                        detail=_explain(url, key, internal),
                    )
                )

    for service in package.services:
        for token in service.command:
            for url in _URL_RE.findall(token):
                audit.findings.append(
                    Finding(
                        url=url,
                        phase=_classify(url, None, internal),
                        location=f"services.{service.name}.command",
                        detail="referenced in the service command line",
                    )
                )

    if manifest is not None:
        for artifact in manifest.artifacts:
            if artifact.source and "://" in artifact.source:
                audit.findings.append(
                    Finding(
                        url=artifact.source,
                        phase=Phase.BUILD,
                        location=f"manifest.artifacts.{artifact.id}",
                        detail="already present in the bundle; recorded for provenance",
                    )
                )

    return audit


def _is_internal(host: str, internal: set[str]) -> bool:
    """Whether a host is inside the deployment rather than out on the internet.

    Three cases, all genuinely local:
      * loopback and private ranges;
      * a name the package itself declares as a service or container;
      * any single-label hostname - `http://vllm:8000` is a container on the
        deployment network, and a name with no dot cannot be public DNS.

    That last rule matters more than it looks. Without it every correctly
    written internal service URL is reported as a runtime dependency, and an
    audit that cries wolf on correct configuration is one operators learn to
    ignore.
    """
    if _LOCAL_HOSTS.match(host):
        return True
    if host in internal:
        return True
    return "." not in host and host != ""


def _classify(url: str, key: str | None, internal: set[str] | None = None) -> Phase:
    host = urlparse(url).netloc.split(":")[0]
    if _is_internal(host, internal or set()):
        return Phase.LOCAL
    if host in _BUILD_TIME_HOSTS:
        # A build-time host appearing in runtime configuration is exactly the
        # defect this audit exists to find, so it is reported as RUNTIME.
        return Phase.RUNTIME if key and _RUNTIME_URL_KEYS.search(key) else Phase.BUILD
    if key and _RUNTIME_URL_KEYS.search(key):
        return Phase.RUNTIME
    return Phase.UNKNOWN


def _explain(url: str, key: str, internal: set[str] | None = None) -> str:
    host = urlparse(url).netloc.split(":")[0]
    if _is_internal(host, internal or set()):
        if host in (internal or set()):
            return f"resolves to the {host!r} service in this package"
        return "points inside the deployment; not an external dependency"
    if _RUNTIME_URL_KEYS.search(key):
        return (
            "the variable name suggests the workload contacts this at run time, "
            "which will fail on an air-gapped host"
        )
    return "could not be classified confidently; review it"
