"""Local configuration overrides (section 29).

Lets an operator adapt a bundle to their machine - different ports, specific
GPUs, tuned environment - *without modifying the bundle*. That separation is
the point: the bundle stays exactly as it was built, verified and signed, so
its checksums and signature remain meaningful after the operator has adjusted
how it runs.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from offlineai.errors import ConfigurationError
from offlineai.schema.package import PORT_RE

__all__ = ["GpuOverride", "RuntimeOverride", "ServiceOverride", "Overrides", "load_overrides"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GpuOverride(_Strict):
    #: Which physical devices to expose (section 30). An empty list means all.
    device_ids: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1)


class RuntimeOverride(_Strict):
    gpu: GpuOverride | None = None


class ServiceOverride(_Strict):
    ports: list[str] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    command: list[str] | None = None

    @field_validator("ports")
    @classmethod
    def _check_ports(cls, values: list[str] | None) -> list[str] | None:
        for value in values or []:
            if not PORT_RE.match(value):
                raise ValueError(f"{value!r} is not a valid port mapping. Use 'HOST:CONTAINER'.")
        return values


class Overrides(_Strict):
    runtime: RuntimeOverride | None = None
    services: dict[str, ServiceOverride] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("environment", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> object:
        if isinstance(value, dict):
            return {k: str(v) if v is not None else "" for k, v in value.items()}
        return value

    @property
    def gpu_device_ids(self) -> list[str] | None:
        if self.runtime is None or self.runtime.gpu is None:
            return None
        return self.runtime.gpu.device_ids or []

    def ports_for(self, service: str) -> list[str] | None:
        override = self.services.get(service)
        return override.ports if override else None

    def environment_for(self, service: str) -> dict[str, str]:
        override = self.services.get(service)
        return {**self.environment, **(override.environment if override else {})}

    def validate_against(self, service_names: set[str]) -> list[str]:
        """Report overrides that name a service the package does not have.

        Returned as warnings rather than raised: a typo here is worth flagging
        loudly, but it should not block an install that is otherwise fine.
        """
        unknown = sorted(set(self.services) - service_names)
        if not unknown:
            return []
        known = ", ".join(sorted(service_names)) or "none"
        return [
            f"override file configures service(s) {', '.join(unknown)}, which this "
            f"package does not declare. Declared services: {known}."
        ]


def load_overrides(path: Path | str) -> Overrides:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(
            f"{path} does not exist",
            action="Check the path given to --config.",
        )
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"{path} is not valid YAML.", details={"Detail": str(exc)}
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping.")
    try:
        return Overrides.model_validate(data)
    except ValidationError as exc:
        problems = "\n".join(
            f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        )
        raise ConfigurationError(
            f"{path} contains invalid overrides.",
            details={"Problems": problems},
        ) from exc
