"""Exit codes at the real process boundary (section 66).

These run the installed console script in a subprocess rather than through
CliRunner. That is deliberate: CliRunner invokes Click with
standalone_mode=True, but the console script uses standalone_mode=False so it
can own the error mapping - and in that mode Click *returns* an exit code
instead of raising it. A bug in exactly that seam produced exit 0 for a
corrupt bundle while still printing the failure, which CliRunner could not
have caught.

Automation branches on these numbers, so they are checked where automation
sees them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from offlineai.exitcodes import ExitCode

OFFLINEAI = Path(sys.executable).parent / "offlineai"


def run_cli(*args: str, home: Path) -> subprocess.CompletedProcess[str]:
    # No allow_network marker: the guard patches sockets in *this* process,
    # and the child needs no network at all. Marking these would skip them
    # under OFFLINEAI_TEST_OFFLINE=1, which is exactly the run that matters.
    env = {
        **os.environ,
        "OFFLINEAI_HOME": str(home),
        # The child must not inherit a strict-offline setting from the parent
        # test environment; these cases are about exit codes, not offline mode.
        "OFFLINEAI_OFFLINE": "0",
    }
    return subprocess.run(  # noqa: S603
        [str(OFFLINEAI), *args], capture_output=True, text=True, env=env, timeout=120
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


@pytest.fixture
def package(tmp_path: Path) -> Path:
    source = tmp_path / "pkg"
    (source / "weights").mkdir(parents=True)
    (source / "weights" / "model.safetensors").write_bytes(b"weights" * 2000)
    (source / "offlineai.yaml").write_text(
        """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: exitcode-demo
  version: 1.0.0
models:
  - name: demo
    source:
      type: local
      path: ./weights
"""
    )
    return source


@pytest.fixture
def bundle(package: Path, home: Path, tmp_path: Path) -> Path:
    result = run_cli("build", str(package), "-o", str(tmp_path / "dist"), home=home)
    assert result.returncode == ExitCode.SUCCESS, result.stderr
    return tmp_path / "dist" / "exitcode-demo-1.0.0.offlineai"


@pytest.mark.skipif(not OFFLINEAI.exists(), reason="console script not installed")
class TestExitCodes:
    def test_success_is_zero(self, bundle: Path, home: Path) -> None:
        assert run_cli("verify", str(bundle), home=home).returncode == ExitCode.SUCCESS

    def test_invalid_package_is_two(self, tmp_path: Path, home: Path) -> None:
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "offlineai.yaml").write_text("apiVersion: offlineai/v1\nkind: Package\n")
        assert run_cli("build", str(bad), home=home).returncode == ExitCode.INVALID_PACKAGE

    def test_corrupt_payload_is_three(self, bundle: Path, home: Path) -> None:
        """The regression that motivated this file: the failure was printed but
        the process still exited 0."""
        data = bytearray(bundle.read_bytes())
        data[int(len(data) * 0.85)] ^= 0xFF
        bundle.write_bytes(bytes(data))

        result = run_cli("verify", str(bundle), home=home)
        assert result.returncode == ExitCode.VERIFICATION_FAILURE
        assert "FAILED" in result.stdout + result.stderr

    def test_corrupt_header_is_three(self, bundle: Path, home: Path) -> None:
        data = bytearray(bundle.read_bytes())
        data[200] ^= 0xFF
        bundle.write_bytes(bytes(data))
        assert run_cli("verify", str(bundle), home=home).returncode == ExitCode.VERIFICATION_FAILURE

    def test_truncated_bundle_is_three(self, bundle: Path, home: Path) -> None:
        data = bundle.read_bytes()
        bundle.write_bytes(data[: len(data) // 2])
        assert run_cli("verify", str(bundle), home=home).returncode == ExitCode.VERIFICATION_FAILURE

    def test_missing_bundle_is_three(self, tmp_path: Path, home: Path) -> None:
        result = run_cli("verify", str(tmp_path / "nope.offlineai"), home=home)
        assert result.returncode == ExitCode.VERIFICATION_FAILURE
        assert "does not exist" in result.stdout + result.stderr

    def test_help_is_zero(self, home: Path) -> None:
        assert run_cli("--help", home=home).returncode == ExitCode.SUCCESS

    def test_version_is_zero(self, home: Path) -> None:
        result = run_cli("--version", home=home)
        assert result.returncode == ExitCode.SUCCESS
        assert "offlineai" in result.stdout


@pytest.mark.skipif(not OFFLINEAI.exists(), reason="console script not installed")
class TestStreamSeparation:
    """stdout must stay machine-readable: `offlineai --json ... | jq` has to
    work, so nothing human or log-shaped may land there."""

    def test_json_output_is_pure_json(self, bundle: Path, home: Path) -> None:
        import json

        result = run_cli("--json", "inspect", str(bundle), home=home)
        assert result.returncode == ExitCode.SUCCESS
        payload = json.loads(result.stdout)
        assert payload["package"] == "exitcode-demo"

    def test_errors_do_not_pollute_stdout_in_human_mode(self, tmp_path: Path, home: Path) -> None:
        result = run_cli("verify", str(tmp_path / "nope.offlineai"), home=home)
        assert result.stdout.strip() == "", "error text belongs on stderr"


def test_console_script_is_installed() -> None:
    assert shutil.which(str(OFFLINEAI)) or OFFLINEAI.exists(), (
        "the offlineai console script must be installed for these tests"
    )
