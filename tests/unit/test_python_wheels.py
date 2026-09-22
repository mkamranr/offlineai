"""Sections 17 and 18: Python dependency packaging.

The builder is rarely the target. A macOS build that resolves wheels for
itself produces macosx artifacts that cannot install on the Linux host they
were meant for, and the failure lands on the air-gapped side where it cannot
be fixed. These tests pin the cross-platform contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offlineai.artifacts.base import SourceRef
from offlineai.artifacts.sources.pypi import PLATFORM_TAGS, PythonSource, wheel_tags
from offlineai.errors import MissingArtifactError, SourceError
from offlineai.exitcodes import ExitCode
from offlineai.schema.manifest import ArtifactType
from offlineai.utils.proc import CommandResult


class TestWheelFilenameParsing:
    @pytest.mark.parametrize(
        ("filename", "name", "version", "platform_tag"),
        [
            (
                "numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
                "numpy",
                "2.1.0",
                "manylinux_2_17_x86_64",
            ),
            ("fastapi-0.115.0-py3-none-any.whl", "fastapi", "0.115.0", "any"),
            (
                "torch-2.4.0-cp312-cp312-manylinux1_x86_64.whl",
                "torch",
                "2.4.0",
                "manylinux1_x86_64",
            ),
            (
                "charset_normalizer-3.3.2-cp312-cp312-macosx_11_0_arm64.whl",
                "charset-normalizer",
                "3.3.2",
                "macosx_11_0_arm64",
            ),
        ],
    )
    def test_parses_tags(self, filename: str, name: str, version: str, platform_tag: str) -> None:
        info = wheel_tags(filename)
        assert info is not None
        assert (info.name, info.version, info.platform_tag) == (name, version, platform_tag)

    def test_pure_python_wheels_are_identified(self) -> None:
        info = wheel_tags("fastapi-0.115.0-py3-none-any.whl")
        assert info is not None and info.pure_python

    def test_platform_specific_wheels_are_not_pure(self) -> None:
        info = wheel_tags("numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.whl")
        assert info is not None and not info.pure_python

    def test_a_non_wheel_returns_none(self) -> None:
        assert wheel_tags("numpy-2.1.0.tar.gz") is None


class TestResolutionIsExplicitlyCrossPlatform:
    """The core requirement: never inherit the builder's own platform."""

    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            calls.append(list(args))
            # Produce a plausible wheel so expansion continues.
            dest = Path(args[args.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "fastapi-0.115.0-py3-none-any.whl").write_bytes(b"PK\x03\x04wheel")
            return CommandResult(tuple(args), 0, "", "")

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        return calls

    def _ref(self, requirements: Path, **options: object) -> SourceRef:
        return SourceRef(
            kind="pypi",
            locator=str(requirements),
            name="requirements",
            artifact_type=ArtifactType.PYTHON_WHEEL,
            options=options,
        )

    @pytest.fixture
    def requirements(self, tmp_path: Path) -> Path:
        path = tmp_path / "requirements.txt"
        path.write_text("fastapi>=0.115\n")
        return path

    def test_only_binary_is_mandatory(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pip cannot cross-compile an sdist, and an air-gapped host has no
        build toolchain."""
        calls = self._capture(monkeypatch)
        PythonSource().expand(self._ref(requirements, platform="linux/amd64"))
        assert "--only-binary=:all:" in calls[0]

    def test_the_target_platform_is_passed_not_the_builders(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)
        PythonSource().expand(self._ref(requirements, platform="linux/amd64"))
        passed = {calls[0][i + 1] for i, a in enumerate(calls[0]) if a == "--platform"}
        assert passed == set(PLATFORM_TAGS["linux/amd64"])
        assert not any("macosx" in p for p in passed), (
            "the builder's own platform must never leak into resolution"
        )

    def test_arm64_targets_get_aarch64_tags(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)
        PythonSource().expand(self._ref(requirements, platform="linux/arm64"))
        passed = {calls[0][i + 1] for i, a in enumerate(calls[0]) if a == "--platform"}
        assert all("aarch64" in p for p in passed)

    def test_the_declared_python_version_and_abi_are_passed(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)
        PythonSource().expand(
            self._ref(requirements, platform="linux/amd64", python_version="3.11")
        )
        args = calls[0]
        assert args[args.index("--python-version") + 1] == "3.11"
        assert args[args.index("--abi") + 1] == "cp311"
        assert args[args.index("--implementation") + 1] == "cp"

    def test_an_unknown_target_platform_is_rejected(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._capture(monkeypatch)
        with pytest.raises(SourceError, match="platform tags"):
            PythonSource().expand(self._ref(requirements, platform="solaris/sparc"))

    def test_platform_tags_are_recorded_for_the_manifest(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 18 wants cp312 / manylinux / linux_x86_64 recorded."""
        self._capture(monkeypatch)
        requests = PythonSource().expand(
            self._ref(requirements, platform="linux/amd64", python_version="3.12")
        )
        metadata = requests[0].metadata
        assert metadata["package"] == "fastapi"
        assert metadata["python_tag"] == "py3"
        assert metadata["platform_tag"] == "any"
        assert metadata["target_platform"] == "linux/amd64"
        assert metadata["python_version"] == "3.12"


class TestFailureModes:
    @pytest.fixture
    def requirements(self, tmp_path: Path) -> Path:
        path = tmp_path / "requirements.txt"
        path.write_text("some-package==1.0\n")
        return path

    def _ref(self, requirements: Path) -> SourceRef:
        return SourceRef(
            kind="pypi",
            locator=str(requirements),
            name="requirements",
            artifact_type=ArtifactType.PYTHON_WHEEL,
            options={"platform": "linux/amd64", "python_version": "3.12"},
        )

    def test_a_package_with_no_matching_wheel_fails_the_build(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 74: a successful build must mean the bundle is complete."""

        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(
                tuple(args),
                1,
                "",
                "ERROR: Could not find a version that satisfies the requirement\n"
                "None of the wheels are compatible for some-package",
            )

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        with pytest.raises(MissingArtifactError) as excinfo:
            PythonSource().expand(self._ref(requirements))
        rendered = excinfo.value.render()
        assert "wheel" in rendered.lower()
        assert "linux/amd64" in rendered
        assert "cannot build from source" in rendered
        # Section 66: automation branches on "missing dependency", not a
        # generic failure.
        assert excinfo.value.exit_code == ExitCode.MISSING_DEPENDENCY

    def test_the_error_names_the_offending_package(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(
                tuple(args),
                1,
                "",
                "ERROR: No matching distribution found for tensorflow-gpu",
            )

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        with pytest.raises(MissingArtifactError) as excinfo:
            PythonSource().expand(self._ref(requirements))
        assert "tensorflow-gpu" in excinfo.value.render()

    def test_a_source_distribution_is_refused(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An sdist would need a compiler on a host that has none."""

        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            dest = Path(args[args.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "legacy-1.0.tar.gz").write_bytes(b"sdist")
            return CommandResult(tuple(args), 0, "", "")

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        with pytest.raises(SourceError, match="source distributions"):
            PythonSource().expand(self._ref(requirements))

    def test_a_missing_requirements_file_is_reported(self, tmp_path: Path) -> None:
        ref = SourceRef(
            kind="pypi",
            locator=str(tmp_path / "nope.txt"),
            name="requirements",
            artifact_type=ArtifactType.PYTHON_WHEEL,
        )
        with pytest.raises(SourceError, match="does not exist"):
            PythonSource().expand(ref)

    def test_a_network_failure_says_so(
        self, requirements: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(tuple(args), 1, "", "Connection timed out")

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        with pytest.raises(SourceError) as excinfo:
            PythonSource().expand(self._ref(requirements))
        assert "connectivity" in excinfo.value.render()


class TestEmptyRequirements:
    def test_a_comment_only_file_resolves_to_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """examples/hello-ai ships exactly this. It must not invoke pip at all."""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("# nothing here\n\n#  not even this\n")

        called = {"n": 0}

        def fake_run(args: list[str], **kwargs: object) -> CommandResult:
            called["n"] += 1
            return CommandResult(tuple(args), 0, "", "")

        monkeypatch.setattr("offlineai.artifacts.sources.pypi.run", fake_run)
        ref = SourceRef(
            kind="pypi",
            locator=str(requirements),
            name="requirements",
            artifact_type=ArtifactType.PYTHON_WHEEL,
        )
        assert PythonSource().expand(ref) == []
        assert called["n"] == 0, "an empty requirements file must not invoke the resolver"
