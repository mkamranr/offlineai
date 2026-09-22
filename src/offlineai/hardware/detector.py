"""Hardware and environment detection (sections 20, 21, 22 and 31).

Everything here is read-only and best-effort. Detection failing is not an
error - it produces a ``None`` and, downstream, a ``SKIPPED`` check with a
reason. Section 21 is explicit that an unevaluated requirement must never be
reported as satisfied, so "we could not tell" and "it is fine" stay distinct
all the way to the output.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from offlineai.logging import get_logger
from offlineai.utils.proc import have, run

__all__ = [
    "CpuInfo",
    "DiskInfo",
    "GpuDevice",
    "HardwareDetector",
    "HardwareReport",
    "MemoryInfo",
]

logger = get_logger("hardware")

_ARCH_ALIASES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


@dataclass(frozen=True, slots=True)
class CpuInfo:
    architecture: str
    cores: int
    model: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    total_bytes: int
    available_bytes: int | None = None

    @property
    def total_gb(self) -> float:
        """Capacity in binary GB, which is how RAM is sold and specified.

        A "16 GB" module is 16 GiB. Reporting the decimal 17.2 for it would be
        arithmetically defensible and operationally confusing.
        """
        return self.total_bytes / 1024**3


@dataclass(frozen=True, slots=True)
class DiskInfo:
    path: str
    total_bytes: int
    free_bytes: int

    @property
    def free_gb(self) -> float:
        """Free space in decimal GB, which is how drives are sold.

        Deliberately a different convention from RAM and VRAM above: a "500 GB"
        SSD really is 500 x 10^9 bytes, while a "16 GB" DIMM is 16 x 2^30. Using
        one convention for both would misreport one of them.
        """
        return self.free_bytes / 1000**3


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    memory_total_mb: int
    driver_version: str | None = None
    compute_capability: str | None = None
    uuid: str | None = None

    @property
    def memory_gb(self) -> float:
        """VRAM in binary GB, matching how the card is sold.

        nvidia-smi reports MiB - 81559 for an "80GB" H100, some of it reserved.
        Dividing by 1024 gives 79.6, which reads correctly against the spec
        sheet; converting to decimal GB would print 85.5 for the same card.
        """
        return self.memory_total_mb / 1024

    @property
    def advertised_gb(self) -> int:
        """The capacity an operator would name, for comparison against a
        requirement written as `minimum_vram_gb: 80`.

        Rounded, because a genuine 80 GB card reports 79.6 GiB usable once the
        reserved region is excluded, and requiring 80 must not reject it.
        """
        return round(self.memory_gb)


@dataclass(slots=True)
class HardwareReport:
    os_name: str
    os_version: str
    architecture: str
    cpu: CpuInfo
    memory: MemoryInfo | None
    disk: DiskInfo | None
    gpus: list[GpuDevice] = field(default_factory=list)
    #: Why GPU detection produced nothing, when it did.
    gpu_detail: str | None = None
    nvidia_driver: str | None = None

    @property
    def is_linux(self) -> bool:
        return self.os_name == "linux"

    @property
    def total_vram_gb(self) -> float:
        return sum(g.memory_gb for g in self.gpus)

    @property
    def largest_gpu_gb(self) -> float:
        return max((g.memory_gb for g in self.gpus), default=0.0)


class HardwareDetector:
    """Reads the host environment. Never modifies anything."""

    def __init__(self, *, disk_path: Path | str | None = None) -> None:
        self.disk_path = Path(disk_path) if disk_path else Path.cwd()

    def detect(self) -> HardwareReport:
        gpus, gpu_detail, driver = self.detect_gpus()
        return HardwareReport(
            os_name=platform.system().lower(),
            os_version=platform.release(),
            architecture=self.architecture(),
            cpu=self.detect_cpu(),
            memory=self.detect_memory(),
            disk=self.detect_disk(self.disk_path),
            gpus=gpus,
            gpu_detail=gpu_detail,
            nvidia_driver=driver,
        )

    # -- cpu -------------------------------------------------------------

    @staticmethod
    def architecture() -> str:
        machine = platform.machine().lower()
        return _ARCH_ALIASES.get(machine, machine)

    def detect_cpu(self) -> CpuInfo:
        # sched_getaffinity respects cpuset limits, which matters inside a
        # container; os.cpu_count does not.
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
        return CpuInfo(
            architecture=self.architecture(),
            cores=cores or os.cpu_count() or 1,
            model=self._cpu_model(),
        )

    @staticmethod
    def _cpu_model() -> str | None:
        system = platform.system().lower()
        if system == "linux":
            try:
                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
            except OSError:
                return None
        elif system == "darwin":
            result = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=10)
            if result.ok:
                return result.first_line() or None
        return None

    # -- memory ----------------------------------------------------------

    @staticmethod
    def detect_memory() -> MemoryInfo | None:
        system = platform.system().lower()
        if system == "linux":
            try:
                values: dict[str, int] = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, _, rest = line.partition(":")
                    number = rest.strip().split()
                    if number and number[0].isdigit():
                        values[key] = int(number[0]) * 1024  # kB -> bytes
                total = values.get("MemTotal")
                if total:
                    return MemoryInfo(total_bytes=total, available_bytes=values.get("MemAvailable"))
            except OSError:
                return None
        elif system == "darwin":
            result = run(["sysctl", "-n", "hw.memsize"], timeout=10)
            if result.ok and result.first_line().isdigit():
                return MemoryInfo(total_bytes=int(result.first_line()))
        return None

    # -- disk ------------------------------------------------------------

    @staticmethod
    def detect_disk(path: Path | str) -> DiskInfo | None:
        current = Path(path).absolute()
        while not current.exists():
            parent = current.parent
            if parent == current:
                return None
            current = parent
        try:
            usage = shutil.disk_usage(current)
        except OSError:
            return None
        return DiskInfo(path=str(current), total_bytes=usage.total, free_bytes=usage.free)

    # -- gpu -------------------------------------------------------------

    def detect_gpus(self) -> tuple[list[GpuDevice], str | None, str | None]:
        """Enumerate NVIDIA GPUs via nvidia-smi.

        Returns ``(devices, detail, driver_version)``. ``detail`` explains an
        empty list, which is the difference between "no GPU" and "we could not
        look" - and section 21 requires that difference to survive to the
        output.
        """
        if not have("nvidia-smi"):
            return [], "nvidia-smi is not installed", None

        result = run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap,uuid",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        )
        if not result.ok:
            return [], f"nvidia-smi failed: {result.output.strip()[:200]}", None

        devices: list[GpuDevice] = []
        driver: str | None = None
        for line in result.stdout.strip().splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                index = int(fields[0])
                memory = int(float(fields[2]))
            except ValueError:
                continue
            driver = _optional(fields, 3) or driver
            devices.append(
                GpuDevice(
                    index=index,
                    name=fields[1],
                    memory_total_mb=memory,
                    driver_version=_optional(fields, 3),
                    compute_capability=_optional(fields, 4),
                    uuid=_optional(fields, 5),
                )
            )

        if not devices:
            return [], "nvidia-smi reported no devices", driver
        return devices, None, driver


def _optional(fields: list[str], index: int) -> str | None:
    if index >= len(fields):
        return None
    value = fields[index].strip()
    # nvidia-smi writes these for fields it cannot report.
    if not value or value in ("[N/A]", "[Not Supported]", "N/A"):
        return None
    return value


def parse_driver_version(text: str) -> tuple[int, ...]:
    """Parse ``"550.54.15"`` into a comparable tuple."""
    return tuple(int(p) for p in re.findall(r"\d+", text)) or (0,)
