"""Loading and validating ``offlineai.yaml``.

Separated from :mod:`offlineai.schema.package`, which is pure data. This module
owns file discovery and the translation of pydantic's validation errors into
the actionable form section 64 asks for.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from offlineai.errors import InvalidPackageError
from offlineai.schema.package import Package
from offlineai.utils.hashing import sha256_bytes

__all__ = ["DEFINITION_FILENAME", "PackageResolver", "load_package"]

DEFINITION_FILENAME = "offlineai.yaml"


class PackageResolver:
    """Loads a package definition from a directory or an explicit file."""

    def load(self, target: Path | str) -> tuple[Package, Path, str]:
        """Return ``(package, definition_path, definition_sha256)``.

        The digest is recorded in the manifest so a bundle can always be traced
        back to the exact definition that produced it (section 34).
        """
        path = _locate(Path(target))
        raw = path.read_bytes()

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise InvalidPackageError(
                f"{path} is not valid YAML.",
                details={"Detail": str(exc)},
                action="Fix the syntax and try again.",
            ) from exc

        if not isinstance(data, dict):
            raise InvalidPackageError(
                f"{path} must contain a YAML mapping.",
                details={"Found": type(data).__name__},
            )

        try:
            package = Package.model_validate(data)
        except ValidationError as exc:
            raise InvalidPackageError(
                f"{path} is not a valid package definition.",
                details={"Problems": _summarise(exc)},
                action="Correct the fields listed above. Run 'offlineai build --help' "
                "for the expected structure.",
            ) from exc

        return package, path, sha256_bytes(raw)


def load_package(target: Path | str) -> tuple[Package, Path, str]:
    return PackageResolver().load(target)


def _locate(target: Path) -> Path:
    if target.is_dir():
        candidate = target / DEFINITION_FILENAME
        if not candidate.is_file():
            raise InvalidPackageError(
                f"no {DEFINITION_FILENAME} found in {target}",
                action=f"Create a {DEFINITION_FILENAME}, or point at one directly:\n"
                f"  offlineai build path/to/{DEFINITION_FILENAME}",
            )
        return candidate
    if not target.is_file():
        raise InvalidPackageError(
            f"{target} does not exist",
            action="Check the path. 'offlineai build .' builds the current directory.",
        )
    return target


def _summarise(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        message = error["msg"]
        # Pydantic prefixes custom validator messages; the prefix adds nothing
        # for someone reading a YAML error.
        message = message.removeprefix("Value error, ")
        lines.append(f"  {location}: {message}")
    return "\n".join(lines)
