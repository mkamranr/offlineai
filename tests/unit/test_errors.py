"""Errors must be actionable (spec section 64) and carry stable exit codes (section 66)."""

import pytest

from offlineai.errors import (
    ChecksumMismatchError,
    HardwareIncompatibleError,
    InstallationError,
    InsufficientDiskError,
    InvalidPackageError,
    MissingArtifactError,
    OfflineAIError,
    SignatureError,
    StrictOfflineViolationError,
    UnsafeArchiveError,
    VerificationError,
)
from offlineai.exitcodes import ExitCode


class TestExitCodeMapping:
    """Each error class pins one exit code. These numbers are public API."""

    @pytest.mark.parametrize(
        ("error_cls", "expected"),
        [
            (OfflineAIError, ExitCode.GENERAL_ERROR),
            (InvalidPackageError, ExitCode.INVALID_PACKAGE),
            (VerificationError, ExitCode.VERIFICATION_FAILURE),
            (ChecksumMismatchError, ExitCode.VERIFICATION_FAILURE),
            (UnsafeArchiveError, ExitCode.VERIFICATION_FAILURE),
            (HardwareIncompatibleError, ExitCode.HARDWARE_INCOMPATIBLE),
            (MissingArtifactError, ExitCode.MISSING_DEPENDENCY),
            (StrictOfflineViolationError, ExitCode.MISSING_DEPENDENCY),
            (InstallationError, ExitCode.INSTALLATION_FAILURE),
            (SignatureError, ExitCode.SIGNATURE_FAILURE),
            (InsufficientDiskError, ExitCode.INSUFFICIENT_DISK),
        ],
    )
    def test_exit_code(self, error_cls: type[OfflineAIError], expected: ExitCode) -> None:
        # Asserted on the class, not an instance: some errors (ChecksumMismatchError)
        # take a specialised constructor, and the code is a class-level contract.
        assert error_cls.exit_code == expected

    def test_every_error_is_catchable_as_the_base(self) -> None:
        with pytest.raises(OfflineAIError):
            raise SignatureError("bad signature")


class TestRendering:
    """Section 64: 'Error 500' is bad; a reason plus an action is good."""

    def test_reason_and_action_are_labelled_blocks(self) -> None:
        err = InstallationError(
            "Docker image vllm was not found in the bundle.",
            details={"Expected": "vllm/vllm-openai@sha256:abc123"},
            action="Rebuild the bundle and ensure the container image is included.",
        )
        rendered = err.render()
        assert "Installation failed." in rendered
        assert "Reason:\nDocker image vllm was not found in the bundle." in rendered
        assert "Expected:\nvllm/vllm-openai@sha256:abc123" in rendered
        assert "Action:\nRebuild the bundle and ensure the container image is included." in rendered

    def test_detail_order_is_preserved(self) -> None:
        err = ChecksumMismatchError(
            path="artifacts/models/qwen3/model.safetensors",
            expected="abc123",
            actual="def456",
        )
        rendered = err.render()
        artifact_at = rendered.index("Artifact:")
        expected_at = rendered.index("Expected:")
        actual_at = rendered.index("Actual:")
        assert artifact_at < expected_at < actual_at
        assert "Bundle verification FAILED." in rendered

    def test_optional_blocks_are_omitted_entirely(self) -> None:
        rendered = OfflineAIError("something went wrong").render()
        assert "Action:" not in rendered
        assert "Expected:" not in rendered

    def test_str_is_the_bare_reason_for_log_lines(self) -> None:
        err = InvalidPackageError("metadata.name is required")
        assert str(err) == "metadata.name is required"


class TestStructuredPayload:
    """--json must surface errors as data, not scraped text (spec section 47)."""

    def test_as_dict_round_trips_the_fields(self) -> None:
        err = HardwareIncompatibleError(
            "GPU VRAM is insufficient.",
            details={"Required": "48 GB", "Available": "24 GB"},
            action="Use a host with more VRAM.",
        )
        payload = err.as_dict()
        assert payload == {
            "error": "HardwareIncompatibleError",
            "exit_code": 4,
            "reason": "GPU VRAM is insufficient.",
            "details": {"Required": "48 GB", "Available": "24 GB"},
            "action": "Use a host with more VRAM.",
        }

    def test_as_dict_omits_empty_optional_fields(self) -> None:
        payload = OfflineAIError("plain failure").as_dict()
        assert "action" not in payload
        assert "details" not in payload


class TestChecksumMismatchErrorShape:
    def test_exposes_fields_for_programmatic_use(self) -> None:
        err = ChecksumMismatchError(path="a/b.bin", expected="aaa", actual="bbb")
        assert err.path == "a/b.bin"
        assert err.expected == "aaa"
        assert err.actual == "bbb"
