"""Target environment profiles (section 5.6).

``offlineai check`` validates a bundle against the machine it is running on.
In this tool's workflow that is the *builder* - the wrong machine, because the
target is air-gapped and somewhere else. So without profiles an engineer can
ask "will this run here?", which nobody cares about, but not "will this run on
the fleet in the secure facility?", which is the only question worth answering
before committing to a 62 GB transfer.

A profile describes the target, so the question can be asked from anywhere.

The module is deliberately thin. A profile's whole job is to produce a
:class:`~offlineai.hardware.detector.HardwareReport`; every rule about what
satisfies what already lives in :mod:`offlineai.hardware.compat` and is reused
unchanged, so a profile check and a live check can never disagree about the
rules.

**A field the profile omits produces ``None``, never a default.** That is what
makes the feature safe rather than merely convenient: inventing a value would
let a profile silently approve a bundle for hardware it was never checked
against, which is worse than having no profile at all. ``check_compatibility``
already renders a ``None`` as ``SKIPPED`` with a reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from offlineai.errors import ConfigurationError
from offlineai.hardware.detector import (
    CpuInfo,
    GpuDevice,
    HardwareReport,
    MemoryInfo,
)
from offlineai.runtime.base import RuntimeAvailability

__all__ = ["GpuProfile", "OsProfile", "Profile", "RuntimeProfile", "load_profile"]

Architecture = Literal["amd64", "arm64"]

#: Distribution names that mean "this is a Linux host". A profile naturally
#: says `family: ubuntu`; the detector says `linux`. Without this mapping a
#: perfectly good Ubuntu target would be reported as an unsupported platform.
_LINUX_FAMILIES = frozenset(
    {
        "linux",
        "ubuntu",
        "debian",
        "rhel",
        "redhat",
        "centos",
        "rocky",
        "almalinux",
        "fedora",
        "suse",
        "opensuse",
        "amazonlinux",
        "amzn",
        "oracle",
        "photon",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class OsProfile(_Strict):
    family: str = "linux"
    version: str | None = None


class GpuProfile(_Strict):
    vendor: Literal["nvidia", "amd", "intel"] = "nvidia"
    #: Per device, as it is sold - an "80GB" H100 is 80 here.
    memory_gb: int | None = Field(default=None, ge=0)
    minimum_driver: str | None = None
    count: int = Field(default=1, ge=1)
    model: str | None = None


class RuntimeProfile(_Strict):
    #: A constraint as written in the specification's example, ">=27", or a
    #: plain version. Only the number is used.
    docker: str | None = None


class Profile(_Strict):
    """A description of a machine this bundle might be deployed to."""

    name: str
    architecture: Architecture = "amd64"
    os: OsProfile | None = None
    gpu: GpuProfile | None = None
    runtime: RuntimeProfile | None = None
    memory_gb: int | None = Field(default=None, ge=0)
    cpu_cores: int | None = Field(default=None, ge=1)
    description: str | None = None

    # -- conversion ------------------------------------------------------

    def to_hardware_report(self) -> HardwareReport:
        """Describe this profile the way the detector describes a real host.

        Anything the profile leaves out stays ``None``, so the comparison
        reports it as not evaluated rather than as satisfied.
        """
        gpus: list[GpuDevice] = []
        if self.gpu is not None and self.gpu.memory_gb is not None:
            gpus = [
                GpuDevice(
                    index=index,
                    name=self.gpu.model or f"{self.gpu.vendor} device",
                    # GpuDevice stores MiB, as nvidia-smi reports it.
                    memory_total_mb=self.gpu.memory_gb * 1024,
                    driver_version=self.gpu.minimum_driver,
                )
                for index in range(self.gpu.count)
            ]

        # A profile names a distribution - "ubuntu" - where the detector
        # names a platform - "linux". Normalise, or every Linux profile would
        # be reported as an unsupported target platform, which is both wrong
        # and the opposite of reassuring.
        family = (self.os.family if self.os else "linux").lower()
        platform_name = "linux" if family in _LINUX_FAMILIES else family
        version = " ".join(
            part
            for part in (
                family if platform_name == "linux" and family != "linux" else "",
                (self.os.version if self.os else None) or "",
            )
            if part
        )

        return HardwareReport(
            os_name=platform_name,
            os_version=version,
            architecture=self.architecture,
            cpu=CpuInfo(architecture=self.architecture, cores=self.cpu_cores or 1),
            memory=(
                MemoryInfo(total_bytes=self.memory_gb * 1024**3)
                if self.memory_gb is not None
                else None
            ),
            # Never from a profile: a profile describes a class of machine, not
            # how much space it happens to have free today.
            disk=None,
            gpus=gpus,
            gpu_detail=(None if gpus else f"profile {self.name!r} describes no GPU"),
            nvidia_driver=self.gpu.minimum_driver if self.gpu else None,
            describes=f"profile {self.name!r}",
        )

    def to_runtime_availability(self) -> RuntimeAvailability:
        """The container runtime this profile claims the target has."""
        version = _version_from(self.runtime.docker) if self.runtime else None
        return RuntimeAvailability(
            available=True,
            version=version,
            detail=f"declared by profile {self.name!r}",
            gpu_support=bool(self.gpu and self.gpu.memory_gb is not None),
        )

    @classmethod
    def from_hardware_report(cls, report: HardwareReport, *, name: str) -> Profile:
        """Capture a real host as a profile.

        This is what ``doctor --save-profile`` writes. An operator on the
        air-gapped target runs it and carries the result back, which turns a
        profile from a hand-written approximation into measured ground truth.
        """
        gpu = None
        if report.gpus:
            first = report.gpus[0]
            gpu = GpuProfile(
                vendor="nvidia",
                memory_gb=first.advertised_gb,
                minimum_driver=report.nvidia_driver,
                count=len(report.gpus),
                model=first.name,
            )
        architecture: Architecture = "arm64" if report.architecture == "arm64" else "amd64"
        return cls(
            name=name,
            architecture=architecture,
            os=OsProfile(family=report.os_name, version=report.os_version or None),
            gpu=gpu,
            memory_gb=int(report.memory.total_gb) if report.memory else None,
            cpu_cores=report.cpu.cores or None,
            description="Captured from a live host by 'offlineai doctor --save-profile'.",
        )

    # -- serialisation ---------------------------------------------------

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            default_flow_style=False,
        )

    @classmethod
    def from_yaml(cls, text: str | bytes) -> Profile:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("a profile must be a YAML mapping")
        return cls.model_validate(data)


def load_profile(path: Path | str) -> Profile:
    """Read a profile, reporting problems the way the rest of the tool does."""
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(
            f"{path} does not exist",
            action="Check the path given to --profile. Capture one from a real "
            "machine with:\n  offlineai doctor --save-profile <file>",
        )
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"{path} is not valid YAML.", details={"Detail": str(exc)}
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping.")

    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        problems = "\n".join(
            f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        )
        raise ConfigurationError(
            f"{path} is not a valid profile.", details={"Problems": problems}
        ) from exc


def _version_from(constraint: str | None) -> str | None:
    """Pull the version out of ``">=27"`` or ``"27.3.1"``.

    Only the number matters: the comparison in ``compat.py`` treats a declared
    version as what the target has, and applies the bundle's own minimum to it.
    """
    if not constraint:
        return None
    stripped = constraint.lstrip("><=~^ ").strip()
    return stripped or None
