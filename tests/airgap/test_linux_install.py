"""The install path, executed on Linux.

Every other test in this repository has run on the developer's machine, which
for this project has been macOS - so the installer has always taken its
*degraded dev-mode* branch. The supported target is Linux (section 77.12), and
until now not one line of that branch had ever executed: OS detection, the
dpkg path, hard links on a real Linux filesystem, `/var/lib` handling.

This runs the real test suite inside a Linux container. It does not need a
GPU, and does not pretend to cover one; what it establishes is that the code
which only runs on Linux, runs.

Opt-in: `pytest -m linux_e2e`. Requires Docker.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.linux_e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "python:3.12-slim"

#: Suites that must pass on Linux. The airgap suite is excluded because it
#: launches its own containers, which would need Docker inside Docker.
SUITES = "tests/unit tests/security tests/integration"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(  # noqa: S603
            ["docker", "info"], capture_output=True, timeout=60, check=False
        ).returncode
        == 0
    )


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker is not available"
)


def run_in_linux(script: str, *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Run a script inside a Linux container with the project mounted.

    The source is mounted read-only and the virtualenv is built on a tmpfs, so
    the host's macOS .venv cannot leak in and a Linux build cannot write back
    over the working tree.
    """
    return subprocess.run(  # noqa: S603
        [
            "docker", "run", "--rm",
            "-v", f"{PROJECT_ROOT}:/src:ro",
            "--tmpfs", "/work:exec,size=2g",
            "-w", "/work",
            "-e", "OFFLINEAI_TEST_OFFLINE=1",
            "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
            IMAGE, "bash", "-ec", script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


#: Copy the source out of the read-only mount, then install and run.
SETUP = """
cp -r /src/src /src/tests /src/pyproject.toml /src/README.md /src/LICENSE /work/
cp -r /src/examples /work/
pip install --quiet -e '.[dev,builder]' 2>&1 | tail -2
"""


@requires_docker
class TestTheContainerIsLinux:
    """Establish the premise before relying on it."""

    def test_the_platform_really_is_linux(self) -> None:
        result = run_in_linux(
            'python -c "import platform; print(platform.system().lower())"', timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "linux"


@requires_docker
class TestTheSuitePassesOnLinux:
    def test_unit_security_and_integration_suites(self) -> None:
        """The whole point: these have only ever run on macOS."""
        result = run_in_linux(SETUP + f"\npython -m pytest {SUITES} -q --no-header")
        assert result.returncode == 0, (
            f"the suite fails on Linux:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
        )


@requires_docker
class TestTheLinuxBranchesExecute:
    """Specific behaviour that is unreachable on macOS."""

    def test_install_is_not_in_dev_mode(self) -> None:
        """On macOS every install reports dev_mode=True and skips the OS
        package and GPU steps. On the supported platform it must not."""
        script = SETUP + """
mkdir -p pkg/weights
echo '{"model_type":"demo"}' > pkg/weights/config.json
head -c 4096 /dev/zero > pkg/weights/model.safetensors
cat > pkg/offlineai.yaml <<'YAML'
apiVersion: offlineai/v1
kind: Package
metadata:
  name: linux-demo
  version: 1.0.0
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
YAML
python - <<'PY'
import json
from offlineai.bundler.builder import BundleBuilder
from offlineai.config.settings import load_settings
from offlineai.installer.installer import Installer
from offlineai.registry.registry import Registry
from offlineai.runtime.fake import FakeRuntime

settings = load_settings(home="/work/home")
settings.ensure_directories()
built = BundleBuilder(settings).build("pkg", output="/work/dist")

registry = Registry(settings)
registry.import_bundle(built.bundle_path)
result = Installer(settings, registry, FakeRuntime()).install(
    "linux-demo", health_sleep=0
)
print(json.dumps({
    "dev_mode": result.dev_mode,
    "state": result.state.value,
    "checks": {c.name: c.status.value for c in result.checks},
}))
PY
"""
        result = run_in_linux(script)
        assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"

        import json

        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["dev_mode"] is False, "the Linux branch was not taken"
        assert payload["state"] == "COMPLETED"
        assert payload["checks"]["Operating system"] == "OK", (
            "Linux must PASS the OS check, not WARNING as macOS does"
        )

    def test_models_are_hard_linked_on_a_real_linux_filesystem(self) -> None:
        """Materialising a 48 GB checkpoint copies it unless hard links work.
        On macOS this falls back to a copy often enough that the link path is
        effectively untested."""
        script = SETUP + """
mkdir -p pkg/weights
head -c 65536 /dev/urandom > pkg/weights/model.safetensors
cat > pkg/offlineai.yaml <<'YAML'
apiVersion: offlineai/v1
kind: Package
metadata:
  name: linkdemo
  version: 1.0.0
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
YAML
python - <<'PY'
import os
from pathlib import Path
from offlineai.bundler.builder import BundleBuilder
from offlineai.config.settings import load_settings
from offlineai.installer.installer import Installer
from offlineai.registry.registry import Registry
from offlineai.runtime.fake import FakeRuntime

settings = load_settings(home="/work/home")
settings.ensure_directories()
built = BundleBuilder(settings).build("pkg", output="/work/dist")
registry = Registry(settings)
registry.import_bundle(built.bundle_path)
Installer(settings, registry, FakeRuntime()).install("linkdemo", health_sleep=0)

materialised = next(
    (settings.data_dir / "installed" / "linkdemo" / "models").rglob("*.safetensors")
)
print("LINKS", os.stat(materialised).st_nlink)
PY
"""
        result = run_in_linux(script)
        assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
        links = int(
            next(l for l in result.stdout.splitlines() if l.startswith("LINKS")).split()[1]
        )
        assert links >= 2, (
            f"the model file has {links} link(s); it was copied rather than hard "
            "linked, which doubles the disk cost of every checkpoint"
        )

    def test_the_cli_works_on_linux(self) -> None:
        script = SETUP + """
offlineai --version
offlineai init
offlineai doctor --help > /dev/null
offlineai --json doctor | python -c "import json,sys; d=json.load(sys.stdin); print('OS_CHECK', [c['detail'] for c in d['checks'] if c['name']=='OS'][0])"
"""
        result = run_in_linux(script)
        assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
        assert "linux" in result.stdout
        assert "not a supported target platform" not in result.stdout, (
            "Linux must not be reported as unsupported"
        )
