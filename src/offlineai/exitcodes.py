"""Process exit codes.

Defined by the specification (section 66) and treated as public API: automation,
CI pipelines and Ansible playbooks branch on these numbers, so a code's meaning
must never be repurposed once released.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Exit status returned by the ``offlineai`` executable."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    INVALID_PACKAGE = 2
    VERIFICATION_FAILURE = 3
    HARDWARE_INCOMPATIBLE = 4
    MISSING_DEPENDENCY = 5
    INSTALLATION_FAILURE = 6
    SIGNATURE_FAILURE = 7
    INSUFFICIENT_DISK = 8
    RUNTIME_FAILURE = 9

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS: dict[ExitCode, str] = {
    ExitCode.SUCCESS: "success",
    ExitCode.GENERAL_ERROR: "general error",
    ExitCode.INVALID_PACKAGE: "invalid package",
    ExitCode.VERIFICATION_FAILURE: "verification failure",
    ExitCode.HARDWARE_INCOMPATIBLE: "hardware incompatibility",
    ExitCode.MISSING_DEPENDENCY: "missing dependency",
    ExitCode.INSTALLATION_FAILURE: "installation failure",
    ExitCode.SIGNATURE_FAILURE: "signature failure",
    ExitCode.INSUFFICIENT_DISK: "insufficient disk",
    ExitCode.RUNTIME_FAILURE: "runtime failure",
}
