"""Sections 29 and 32: local overrides, and strict offline enforcement."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from offlineai.errors import ConfigurationError, StrictOfflineViolationError
from offlineai.exitcodes import ExitCode
from offlineai.schema.overrides import Overrides, load_overrides
from offlineai.security.offline import (
    OFFLINE_ENVIRONMENT,
    offline_environment,
    strict_offline_guard,
)

SAMPLE = """\
runtime:
  gpu:
    device_ids:
      - "0"
      - "1"

services:
  vllm:
    ports:
      - "8001:8000"
    environment:
      VLLM_GPU_MEMORY_UTILIZATION: "0.92"

environment:
  VLLM_MAX_MODEL_LEN: "32768"
"""


class TestOverrideParsing:
    @pytest.fixture
    def overrides(self, tmp_path: Path) -> Overrides:
        path = tmp_path / "server-config.yaml"
        path.write_text(SAMPLE)
        return load_overrides(path)

    def test_gpu_device_ids(self, overrides: Overrides) -> None:
        assert overrides.gpu_device_ids == ["0", "1"]

    def test_service_port_remap(self, overrides: Overrides) -> None:
        assert overrides.ports_for("vllm") == ["8001:8000"]

    def test_unconfigured_service_has_no_port_override(self, overrides: Overrides) -> None:
        assert overrides.ports_for("redis") is None

    def test_global_and_service_environment_merge(self, overrides: Overrides) -> None:
        merged = overrides.environment_for("vllm")
        assert merged["VLLM_MAX_MODEL_LEN"] == "32768"
        assert merged["VLLM_GPU_MEMORY_UTILIZATION"] == "0.92"

    def test_service_environment_wins_over_global(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(
            "environment:\n  X: global\nservices:\n  api:\n    environment:\n      X: specific\n"
        )
        assert load_overrides(path).environment_for("api")["X"] == "specific"

    def test_numeric_values_are_stringified(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("environment:\n  WORKERS: 8\n")
        assert load_overrides(path).environment["WORKERS"] == "8"


class TestOverrideValidation:
    def test_an_unknown_service_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        """A typo is worth flagging loudly, but should not block an install
        that is otherwise fine."""
        path = tmp_path / "c.yaml"
        path.write_text('services:\n  vlmm:\n    ports: ["8001:8000"]\n')
        warnings = load_overrides(path).validate_against({"vllm"})
        assert len(warnings) == 1
        assert "vlmm" in warnings[0]
        assert "vllm" in warnings[0], "the warning should name what is available"

    def test_a_known_service_produces_no_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text('services:\n  vllm:\n    ports: ["8001:8000"]\n')
        assert load_overrides(path).validate_against({"vllm"}) == []

    def test_a_malformed_port_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text('services:\n  vllm:\n    ports: ["not-a-port"]\n')
        with pytest.raises(ConfigurationError) as excinfo:
            load_overrides(path)
        # The specific problem lives in the detail blocks, which render()
        # includes and str() deliberately does not.
        assert "port mapping" in excinfo.value.render()

    def test_an_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("runtme:\n  gpu: {}\n")
        with pytest.raises(ConfigurationError):
            load_overrides(path)

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="does not exist"):
            load_overrides(tmp_path / "nope.yaml")


class TestStrictOfflineGuard:
    """Section 32: in strict mode the installer may use only the bundle, the
    local filesystem, the local runtime and the local registry."""

    def test_network_sockets_are_refused(self) -> None:
        with strict_offline_guard(), pytest.raises(StrictOfflineViolationError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_dns_is_refused(self) -> None:
        with strict_offline_guard(), pytest.raises(StrictOfflineViolationError):
            socket.getaddrinfo("pypi.org", 443)

    def test_the_violation_exits_five(self) -> None:
        """A strict-offline install that needs the network means the bundle was
        built incomplete, which is what the operator must act on."""
        with strict_offline_guard():
            try:
                socket.getaddrinfo("huggingface.co", 443)
            except StrictOfflineViolationError as exc:
                assert exc.exit_code == ExitCode.MISSING_DEPENDENCY
                assert "rebuild the bundle" in exc.render().lower()
            else:
                pytest.fail("the guard did not fire")

    def test_local_ipc_still_works(self) -> None:
        """The Docker socket is AF_UNIX. Blocking it would break the very
        install the guard is protecting."""
        with strict_offline_guard():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.close()

    def test_the_guard_is_removed_on_exit(self) -> None:
        # Compared by identity, not by name: the test suite installs its own
        # network guard whose inner function is also called guarded_socket, so
        # a name check would collide and pass for the wrong reason.
        before = socket.socket
        with strict_offline_guard():
            assert socket.socket is not before
        assert socket.socket is before

    def test_the_guard_is_removed_even_after_a_failure(self) -> None:
        before = socket.socket
        with pytest.raises(RuntimeError), strict_offline_guard():
            raise RuntimeError("boom")
        assert socket.socket is before, "a failed install must not leave sockets patched"

    def test_disabled_guard_is_a_no_op(self) -> None:
        before = socket.socket
        with strict_offline_guard(enabled=False):
            assert socket.socket is before


class TestSubprocessEnvironment:
    """Patching sockets in this process does nothing to a subprocess, and pip
    and docker are subprocesses."""

    @pytest.mark.parametrize(
        "variable",
        ["PIP_NO_INDEX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NO_PROXY"],
    )
    def test_offline_variables_are_set(self, variable: str) -> None:
        assert variable in offline_environment({})

    def test_pip_is_pointed_away_from_any_index(self) -> None:
        assert offline_environment({})["PIP_NO_INDEX"] == "1"

    def test_proxies_are_cleared_so_they_cannot_restore_egress(self) -> None:
        env = offline_environment({"HTTPS_PROXY": "http://corp-proxy:3128"})
        assert env["HTTPS_PROXY"] == ""
        assert env["NO_PROXY"] == "*"

    def test_the_base_environment_is_preserved(self) -> None:
        env = offline_environment({"PATH": "/usr/bin", "CUSTOM": "value"})
        assert env["PATH"] == "/usr/bin"
        assert env["CUSTOM"] == "value"

    def test_the_process_environment_is_set_inside_the_guard(self) -> None:
        with strict_offline_guard():
            for key in OFFLINE_ENVIRONMENT:
                assert os.environ.get(key) == OFFLINE_ENVIRONMENT[key]

    def test_the_process_environment_is_restored_afterwards(self) -> None:
        original = os.environ.get("PIP_NO_INDEX")
        with strict_offline_guard():
            pass
        assert os.environ.get("PIP_NO_INDEX") == original
