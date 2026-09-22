"""Layered configuration (sections 67 and 68).

Precedence, lowest to highest::

    built-in defaults  <  ~/.offlineai/config.yaml  <  OFFLINEAI_*  <  CLI flags

Getting that order wrong is not a cosmetic bug - it is how a 62 GB registry
ends up on the wrong volume - so each boundary is covered by a test.

Paths are resolved to absolute form at load time. A relative ``data_dir`` that
moves with the process working directory would be a nasty surprise for a
long-running install.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from offlineai.errors import ConfigurationError

__all__ = [
    "CONFIG_FILENAME",
    "DownloadSettings",
    "OfflineSettings",
    "RuntimeSettings",
    "SecuritySettings",
    "Settings",
    "load_settings",
]

CONFIG_FILENAME = "config.yaml"
DEFAULT_HOME = Path("~/.offlineai")

_TRUTHY = {"1", "true", "yes", "on"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeSettings(_Strict):
    container_engine: Literal["docker"] = "docker"


class SecuritySettings(_Strict):
    #: Section 12: signing must be possible, never mandatory.
    require_signature: bool = False
    #: Refuse to build a bundle containing credential-shaped files (section 40).
    detect_secrets: bool = True


class OfflineSettings(_Strict):
    #: Section 32. When on, nothing may touch the network during install.
    strict: bool = False


class DownloadSettings(_Strict):
    #: Section 45 asks for a conservative default so a build does not
    #: overwhelm storage or the network.
    workers: int = Field(default=4, ge=1, le=64)
    resume: bool = True


class Settings(_Strict):
    """Fully resolved configuration for one invocation."""

    home: Path
    data_dir: Path
    cache_dir: Path
    registry_dir: Path
    bundles_dir: Path
    logs_dir: Path
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    offline: OfflineSettings = Field(default_factory=OfflineSettings)
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)

    @property
    def config_file(self) -> Path:
        return self.home / CONFIG_FILENAME

    def ensure_directories(self) -> None:
        """Create the documented tree (section 7.1). Idempotent."""
        for directory in (
            self.home,
            self.data_dir,
            self.registry_dir,
            self.cache_dir,
            self.bundles_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=True, default_flow_style=False
        )

    @classmethod
    def model_validate_yaml(cls, text: str) -> Settings:
        return cls.model_validate(yaml.safe_load(text))


class _FileConfig(_Strict):
    """The subset of settings a config.yaml may set.

    Deliberately narrower than :class:`Settings`: derived directories such as
    ``registry_dir`` follow from ``data_dir`` unless explicitly overridden, and
    ``home`` is not settable from inside the file it is read from.
    """

    data_dir: Path | None = None
    cache_dir: Path | None = None
    registry_dir: Path | None = None
    bundles_dir: Path | None = None
    logs_dir: Path | None = None
    log_level: Literal["debug", "info", "warning", "error"] | None = None
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    offline: OfflineSettings = Field(default_factory=OfflineSettings)
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def _read_config_file(path: Path) -> _FileConfig:
    if not path.is_file():
        return _FileConfig()
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"{path} is not valid YAML.",
            details={"Detail": str(exc)},
            action="Fix the syntax, or delete the file to fall back to defaults.",
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{path} must contain a YAML mapping.",
            details={"Found": type(data).__name__},
        )
    try:
        return _FileConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            f"{path} contains invalid settings.",
            details={"Detail": _summarise(exc)},
            action="Check the key names and value types against the documented "
            "configuration format.",
        ) from exc


def _summarise(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"{location}: {error['msg']}")
    return "\n".join(lines)


def load_settings(
    *,
    home: Path | str | None = None,
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    registry_dir: Path | str | None = None,
    log_level: str | None = None,
    strict_offline: bool | None = None,
) -> Settings:
    """Resolve settings across all four layers.

    Keyword arguments are the CLI layer and win over everything else; ``None``
    means "the flag was not given", which is why they are not simply defaulted.
    """
    resolved_home = (
        Path(home).expanduser()
        if home is not None
        else _env_path("OFFLINEAI_HOME") or DEFAULT_HOME.expanduser()
    ).resolve()

    file_config = _read_config_file(resolved_home / CONFIG_FILENAME)

    def pick_path(
        cli: Path | str | None, env_var: str, from_file: Path | None, fallback: Path
    ) -> Path:
        for candidate in (cli, _env_path(env_var), from_file):
            if candidate is not None:
                return Path(candidate).expanduser().resolve()
        return fallback

    resolved_data = pick_path(data_dir, "OFFLINEAI_DATA_DIR", file_config.data_dir, resolved_home)
    resolved_cache = pick_path(
        cache_dir, "OFFLINEAI_CACHE_DIR", file_config.cache_dir, resolved_data / "cache"
    )
    resolved_registry = pick_path(
        registry_dir,
        "OFFLINEAI_REGISTRY_DIR",
        file_config.registry_dir,
        resolved_data / "registry",
    )

    env_level = os.environ.get("OFFLINEAI_LOG_LEVEL")
    level = log_level or env_level or file_config.log_level or "info"

    env_offline = _env_bool("OFFLINEAI_OFFLINE")
    strict = (
        strict_offline
        if strict_offline is not None
        else env_offline
        if env_offline is not None
        else file_config.offline.strict
    )

    try:
        return Settings(
            home=resolved_home,
            data_dir=resolved_data,
            cache_dir=resolved_cache,
            registry_dir=resolved_registry,
            bundles_dir=(file_config.bundles_dir or resolved_data / "bundles")
            .expanduser()
            .resolve(),
            logs_dir=(file_config.logs_dir or resolved_data / "logs").expanduser().resolve(),
            log_level=level,  # type: ignore[arg-type]
            runtime=file_config.runtime,
            security=file_config.security,
            offline=OfflineSettings(strict=strict),
            downloads=file_config.downloads,
        )
    except ValidationError as exc:
        raise ConfigurationError(
            "The resolved configuration is invalid.",
            details={"Detail": _summarise(exc)},
        ) from exc


def default_config_yaml() -> str:
    """The commented starter file written by ``offlineai init``."""
    return """\
# OfflineAI configuration.
#
# Every value here can be overridden by an OFFLINEAI_* environment variable,
# and both can be overridden by a command-line flag.

# Where imported bundles, the registry and logs live.
# data_dir: /var/lib/offlineai

# Where build-time artifact downloads are cached and reused across bundles.
# cache_dir: /var/cache/offlineai

# debug | info | warning | error
log_level: info

runtime:
  container_engine: docker

security:
  # Refuse to install a bundle that is not signed.
  require_signature: false
  # Refuse to build a bundle containing credential-shaped files.
  detect_secrets: true

offline:
  # When true, installation may not touch the network under any circumstances.
  # Recommended on air-gapped targets.
  strict: false

downloads:
  # Concurrent artifact downloads during a build. Kept conservative so a build
  # does not saturate storage or the network.
  workers: 4
  resume: true
"""


_ = Any  # re-exported for callers building settings dynamically
