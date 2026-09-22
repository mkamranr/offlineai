"""Structured exception hierarchy.

Two rules drive the design:

* Section 64 - errors must be *actionable*. ``Error 500`` is useless; a reason,
  the relevant values and a concrete next step are not. Every error therefore
  carries an optional ordered ``details`` mapping and an optional ``action``.
* Section 66 - each error class pins exactly one process exit code, so callers
  can branch on ``$?`` without parsing text.

Errors render to human text via :meth:`OfflineAIError.render` and to machine
output via :meth:`OfflineAIError.as_dict`, which keeps ``--json`` honest: the
CLI never has to scrape its own prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from offlineai.exitcodes import ExitCode

__all__ = [
    "ChecksumMismatchError",
    "ConfigurationError",
    "HardwareIncompatibleError",
    "InstallationError",
    "InsufficientDiskError",
    "InvalidPackageError",
    "MissingArtifactError",
    "OfflineAIError",
    "RegistryError",
    "RuntimeFailureError",
    "SignatureError",
    "SourceError",
    "StrictOfflineViolationError",
    "UnsafeArchiveError",
    "VerificationError",
]


class OfflineAIError(Exception):
    """Base class for every failure OfflineAI reports deliberately.

    Anything escaping as a bare ``Exception`` is a bug; the CLI treats those as
    an internal error and says so, rather than pretending it is user-facing.
    """

    exit_code: ExitCode = ExitCode.GENERAL_ERROR

    #: Leading line, e.g. ``Installation failed.``
    headline: str = "Operation failed."

    #: Optional trailing line, e.g. ``Bundle verification FAILED.``
    footer: str | None = None

    def __init__(
        self,
        reason: str,
        *,
        details: Mapping[str, str] | None = None,
        action: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        # dict preserves insertion order, which the rendered output relies on.
        self.details: dict[str, str] = dict(details) if details else {}
        self.action = action

    def __str__(self) -> str:
        # Bare reason only: log lines and `raise ... from` chains stay readable.
        return self.reason

    def render(self) -> str:
        """Format as the labelled block layout the specification shows."""
        blocks: list[str] = [self.headline, f"Reason:\n{self.reason}"]
        blocks.extend(f"{label}:\n{value}" for label, value in self.details.items())
        if self.action:
            blocks.append(f"Action:\n{self.action}")
        if self.footer:
            blocks.append(self.footer)
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable payload for ``--json``."""
        payload: dict[str, Any] = {
            "error": type(self).__name__,
            "exit_code": int(self.exit_code),
            "reason": self.reason,
        }
        if self.details:
            payload["details"] = dict(self.details)
        if self.action:
            payload["action"] = self.action
        return payload


# --------------------------------------------------------------------------
# Exit code 2 - invalid package
# --------------------------------------------------------------------------


class InvalidPackageError(OfflineAIError):
    """The package definition is malformed, unsupported or fails validation."""

    exit_code = ExitCode.INVALID_PACKAGE
    headline = "Invalid package definition."


class ConfigurationError(InvalidPackageError):
    """A config file or override is malformed."""

    headline = "Invalid configuration."


# --------------------------------------------------------------------------
# Exit code 3 - verification failure
# --------------------------------------------------------------------------


class VerificationError(OfflineAIError):
    """A bundle failed integrity verification."""

    exit_code = ExitCode.VERIFICATION_FAILURE
    headline = "Bundle verification failed."
    footer = "Bundle verification FAILED."


class ChecksumMismatchError(VerificationError):
    """An artifact's content does not match the hash recorded in the manifest.

    Never downgrade this to a warning. Section 61: a corrupted bundle must never
    be reported as valid.
    """

    headline = "ERROR: checksum mismatch"

    def __init__(self, *, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Artifact content does not match the manifest hash: {path}",
            details={"Artifact": path, "Expected": expected, "Actual": actual},
            action="The bundle is corrupt or has been tampered with. Re-transfer it "
            "from the builder and verify again.",
        )


class UnsafeArchiveError(VerificationError):
    """An archive member would escape the destination or is otherwise hostile.

    Covers path traversal, absolute paths, symlink and hardlink escapes, device
    nodes and archive bombs (section 39).
    """

    headline = "ERROR: unsafe archive member"
    footer = "Extraction refused."


# --------------------------------------------------------------------------
# Exit code 4 - hardware incompatibility
# --------------------------------------------------------------------------


class HardwareIncompatibleError(OfflineAIError):
    """The host does not satisfy the bundle's declared hardware requirements."""

    exit_code = ExitCode.HARDWARE_INCOMPATIBLE
    headline = "Hardware requirements not satisfied."


# --------------------------------------------------------------------------
# Exit code 5 - missing dependency
# --------------------------------------------------------------------------


class MissingArtifactError(OfflineAIError):
    """Something the package declares is absent from the bundle.

    This is the error that enforces the project's central promise (section 74):
    a bundle must never silently fall back to the network.
    """

    exit_code = ExitCode.MISSING_DEPENDENCY
    headline = "Required artifact is not present in the bundle."


class StrictOfflineViolationError(MissingArtifactError):
    """Something tried to reach the network while strict-offline mode was on."""

    headline = "ERROR: network access attempted in strict-offline mode."


# --------------------------------------------------------------------------
# Exit code 6 - installation failure
# --------------------------------------------------------------------------


class InstallationError(OfflineAIError):
    """An installation step failed."""

    exit_code = ExitCode.INSTALLATION_FAILURE
    headline = "Installation failed."


class RegistryError(OfflineAIError):
    """The local registry could not be read or updated."""

    exit_code = ExitCode.INSTALLATION_FAILURE
    headline = "Local registry operation failed."


# --------------------------------------------------------------------------
# Exit code 7 - signature failure
# --------------------------------------------------------------------------


class SignatureError(OfflineAIError):
    """A signature is missing, malformed, or does not verify."""

    exit_code = ExitCode.SIGNATURE_FAILURE
    headline = "Signature verification failed."


# --------------------------------------------------------------------------
# Exit code 8 - insufficient disk
# --------------------------------------------------------------------------


class InsufficientDiskError(OfflineAIError):
    """Not enough free space to complete the operation safely."""

    exit_code = ExitCode.INSUFFICIENT_DISK
    headline = "Insufficient disk space."


# --------------------------------------------------------------------------
# Exit code 9 - runtime failure
# --------------------------------------------------------------------------


class RuntimeFailureError(OfflineAIError):
    """The container runtime or a managed service failed.

    Named with a ``Failure`` infix to avoid shadowing the builtin
    ``RuntimeError``.
    """

    exit_code = ExitCode.RUNTIME_FAILURE
    headline = "Runtime operation failed."


class SourceError(OfflineAIError):
    """An artifact source could not resolve or fetch a reference.

    Build-time only. On the target this surfaces as
    :class:`MissingArtifactError` instead, because a target that needs a source
    is a bundle that was built wrong.
    """

    exit_code = ExitCode.GENERAL_ERROR
    headline = "Artifact resolution failed."
