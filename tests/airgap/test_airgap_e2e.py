"""End-to-end proof, inside a genuinely network-isolated container.

Every other test in this suite blocks the network in-process. That catches our
own code reaching out, but it cannot prove anything about subprocesses, and it
is a guard we wrote ourselves. This test removes the network at the kernel
level with `docker run --network none` and does the real work inside.

It proves two claims the product rests on:

1. OfflineAI itself installs and runs with no network at all - demonstrated by
   installing it from a wheelhouse using exactly the `pip install --no-index
   --find-links` mechanism it prescribes for packaged workloads.
2. A bundle can be verified, imported and inspected on a host that has no
   route to anywhere.

Opt-in: `pytest -m airgap_e2e`. Requires Docker and takes a few minutes,
because the wheelhouse has to be assembled on the connected side first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.airgap_e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_IMAGE = "python:3.12-slim"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(  # noqa: S603
            ["docker", "info"], capture_output=True, timeout=60, check=False
        ).returncode
        == 0
    )


def image_present(reference: str) -> bool:
    return (
        subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", reference],
            capture_output=True,
            timeout=60,
            check=False,
        ).returncode
        == 0
    )


requires_docker = pytest.mark.skipif(not docker_available(), reason="Docker is not available")


@pytest.fixture(scope="module")
def wheelhouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build OfflineAI and its dependencies as Linux wheels.

    This is the builder-side step, and it is the only part that touches the
    network. Everything after it runs with none.
    """
    directory = tmp_path_factory.mktemp("wheelhouse")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            str(PROJECT_ROOT),
            "--dest",
            str(directory),
            "--only-binary=:all:",
            "--platform",
            "manylinux_2_17_x86_64",
            "--platform",
            "manylinux2014_x86_64",
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            "--abi",
            "cp312",
            "--disable-pip-version-check",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"could not assemble a Linux wheelhouse: {result.stderr[-800:]}")

    # `pip download` of a local source tree yields an sdist for the project
    # itself; build the wheel so the offline install needs no build step.
    if not list(directory.glob("offlineai-*.whl")):
        built = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(PROJECT_ROOT),
                "--no-deps",
                "--wheel-dir",
                str(directory),
                "--disable-pip-version-check",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if built.returncode != 0:
            pytest.skip(f"could not build the offlineai wheel: {built.stderr[-800:]}")

    for leftover in directory.glob("*.tar.gz"):
        leftover.unlink()
    return directory


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a bundle on the connected side, then move it to a 'USB' directory."""
    from offlineai.bundler.builder import BundleBuilder
    from offlineai.config.settings import load_settings

    workspace = tmp_path_factory.mktemp("airgap-build")
    source = workspace / "pkg"
    (source / "weights").mkdir(parents=True)
    (source / "weights" / "config.json").write_text('{"model_type": "demo"}')
    (source / "weights" / "model.safetensors").write_bytes(b"weights" * 5000)
    (source / "offlineai.yaml").write_text(
        """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: airgap-demo
  version: 1.0.0
  description: Built online, verified and imported with no network at all.
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
"""
    )

    settings = load_settings(home=workspace / "builder")
    settings.ensure_directories()
    result = BundleBuilder(settings).build(source, output=workspace / "usb")
    return Path(result.bundle_path)


def run_isolated(
    *,
    script: str,
    mounts: dict[Path, str],
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    """Run a shell script inside a container with no network whatsoever."""
    args = ["docker", "run", "--rm", "--network", "none"]
    for host_path, container_path in mounts.items():
        args += ["-v", f"{host_path}:{container_path}:ro"]
    # The registry must be writable, so give it a tmpfs inside the container.
    args += ["-e", "OFFLINEAI_HOME=/work/home", "--tmpfs", "/work"]
    args += [BASE_IMAGE, "sh", "-ec", script]
    return subprocess.run(  # noqa: S603
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


@requires_docker
class TestTheNetworkIsReallyGone:
    """Establish the premise before relying on it."""

    def test_the_base_image_is_available_locally(self) -> None:
        if not image_present(BASE_IMAGE):
            pulled = subprocess.run(  # noqa: S603
                ["docker", "pull", BASE_IMAGE], capture_output=True, timeout=600, check=False
            )
            if pulled.returncode != 0:
                pytest.skip(f"{BASE_IMAGE} is not available locally and cannot be pulled")
        assert image_present(BASE_IMAGE)

    def test_a_network_none_container_cannot_resolve_or_connect(self) -> None:
        probe = (
            "import socket, sys\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 443), timeout=5)\n"
            "except OSError as exc:\n"
            "    print('unreachable:', type(exc).__name__)\n"
            "    sys.exit(0)\n"
            "print('REACHED THE NETWORK')\n"
            "sys.exit(1)\n"
        )
        result = run_isolated(
            script=f"python - <<'PROBE'\n{probe}PROBE",
            mounts={},
            timeout=180,
        )
        assert result.returncode == 0, (
            "the isolated container reached the network, so nothing below proves "
            f"anything.\n{result.stdout}\n{result.stderr}"
        )
        assert "unreachable" in result.stdout


@requires_docker
class TestOfflineAIInstallsWithNoNetwork:
    """Claim 1, demonstrated with the product's own prescribed mechanism."""

    def test_pip_install_no_index_works_from_the_wheelhouse(self, wheelhouse: Path) -> None:
        result = run_isolated(
            script=(
                "pip install --no-index --find-links /wheels offlineai "
                "--quiet --disable-pip-version-check\n"
                "offlineai --version"
            ),
            mounts={wheelhouse: "/wheels"},
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "offlineai" in result.stdout

    def test_pip_cannot_fall_back_to_an_index(self, wheelhouse: Path) -> None:
        """Asking for something the wheelhouse does not contain must fail
        locally rather than hang trying to reach an index."""
        result = run_isolated(
            script=(
                "pip install --no-index --find-links /wheels "
                "definitely-not-a-real-package-xyz 2>&1 | tail -3; "
                "exit 0"
            ),
            mounts={wheelhouse: "/wheels"},
            timeout=300,
        )
        combined = result.stdout + result.stderr
        assert "No matching distribution" in combined or "Could not find" in combined


@requires_docker
class TestBundleLifecycleWithNoNetwork:
    """Claim 2: verify, import and inspect on a host with no route anywhere."""

    def _script(self, bundle_name: str, commands: str) -> str:
        return (
            "pip install --no-index --find-links /wheels offlineai "
            "--quiet --disable-pip-version-check\n"
            "mkdir -p /work/home\n"
            f"BUNDLE=/bundle/{bundle_name}\n" + commands
        )

    def test_verify(self, wheelhouse: Path, bundle: Path) -> None:
        result = run_isolated(
            script=self._script(bundle.name, 'offlineai verify "$BUNDLE"'),
            mounts={wheelhouse: "/wheels", bundle.parent: "/bundle"},
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "VERIFIED" in result.stdout

    def test_inspect(self, wheelhouse: Path, bundle: Path) -> None:
        result = run_isolated(
            script=self._script(bundle.name, 'offlineai --json inspect "$BUNDLE"'),
            mounts={wheelhouse: "/wheels", bundle.parent: "/bundle"},
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        payload = json.loads(result.stdout)
        assert payload["package"] == "airgap-demo"

    def test_import_then_list(self, wheelhouse: Path, bundle: Path) -> None:
        result = run_isolated(
            script=self._script(
                bundle.name,
                'offlineai import "$BUNDLE" > /dev/null\nofflineai --json list',
            ),
            mounts={wheelhouse: "/wheels", bundle.parent: "/bundle"},
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        listing = json.loads(result.stdout)
        assert [p["name"] for p in listing["packages"]] == ["airgap-demo"]
        assert listing["packages"][0]["version"] == "1.0.0"

    def test_a_corrupt_bundle_is_rejected_offline_too(
        self, wheelhouse: Path, bundle: Path, tmp_path: Path
    ) -> None:
        """Integrity checking must not depend on anything remote."""
        corrupt_dir = tmp_path / "corrupt"
        corrupt_dir.mkdir()
        corrupted = corrupt_dir / bundle.name
        data = bytearray(bundle.read_bytes())
        data[int(len(data) * 0.5)] ^= 0xFF
        corrupted.write_bytes(bytes(data))

        result = run_isolated(
            script=self._script(bundle.name, 'offlineai verify "$BUNDLE" || echo "EXIT=$?"'),
            mounts={wheelhouse: "/wheels", corrupt_dir: "/bundle"},
        )
        assert "EXIT=3" in result.stdout, f"{result.stdout}\n{result.stderr}"

    def test_doctor_runs_offline(self, wheelhouse: Path, bundle: Path) -> None:
        result = run_isolated(
            script=self._script(bundle.name, "offlineai doctor || true"),
            mounts={wheelhouse: "/wheels", bundle.parent: "/bundle"},
        )
        assert "OfflineAI Doctor" in result.stdout
