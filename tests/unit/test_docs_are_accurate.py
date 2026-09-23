"""The documentation must describe the tool that exists.

Docs drift silently: a command gets renamed, a flag changes, and the README
keeps confidently telling people to run something that no longer works. These
checks are cheap and catch exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from offlineai.cli.main import app
from offlineai.exitcodes import ExitCode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = sorted(
    [
        *PROJECT_ROOT.glob("*.md"),
        *(PROJECT_ROOT / "docs").glob("*.md"),
        *(PROJECT_ROOT / "examples").glob("*/README.md"),
    ]
)

#: The requirements document describes an aspiration, not this release.
EXCLUDED = {"OfflineAI_Project_Requirements.md"}

#: Flags documented in prose that are not literal CLI options.
_NOT_A_COMMAND = {
    "init",  # is a command, kept for clarity
}


def registered_commands() -> set[str]:
    import typer

    command = typer.main.get_command(app)
    return set(getattr(command, "commands", {}))


@pytest.fixture(scope="module")
def commands() -> set[str]:
    return registered_commands()


def documented_invocations() -> list[tuple[Path, str]]:
    """Every `offlineai <word>` occurrence in the docs."""
    # (?<![.\w]) keeps this from matching inside a filename such as
    # "bundle.offlineai manifest.yaml", where \b would happily match after the
    # dot and report "manifest" as a command.
    pattern = re.compile(r"(?<![.\w])offlineai\s+([a-z][a-z-]*)")
    found: list[tuple[Path, str]] = []
    for path in DOC_FILES:
        if path.name in EXCLUDED:
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            # Skip prose mentions of the project name and pip lines.
            if stripped.startswith(("pip ", "#", ">")) or "pip install" in stripped:
                continue
            for match in pattern.finditer(line):
                found.append((path, match.group(1)))
    return found


class TestDocumentedCommandsExist:
    def test_every_documented_command_is_registered(self, commands: set[str]) -> None:
        offenders = []
        for path, word in documented_invocations():
            # `offlineai --flag` and `offlineai <path>` are not commands.
            if word.startswith("-") or word in _NOT_A_COMMAND and word in commands:
                continue
            if word not in commands:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: 'offlineai {word}'")
        assert not offenders, "documented commands that do not exist:\n" + "\n".join(
            sorted(set(offenders))
        )

    def test_the_documented_command_set_is_not_trivially_small(self) -> None:
        """Guards against the regex silently matching nothing."""
        words = {w for _, w in documented_invocations()}
        assert len(words) > 10, f"only found {words}; the extractor is probably broken"


class TestExitCodeTableIsCorrect:
    def test_troubleshooting_lists_every_code_correctly(self) -> None:
        text = (PROJECT_ROOT / "docs" / "troubleshooting.md").read_text()
        for code in ExitCode:
            row = re.search(rf"^\|\s*{int(code)}\s*\|\s*([^|]+?)\s*\|", text, re.M)
            assert row, f"exit code {int(code)} is not documented"
            assert row.group(1).strip().lower() == code.description, (
                f"exit code {int(code)} is documented as {row.group(1)!r} "
                f"but means {code.description!r}"
            )


class TestRequiredDocumentsExist:
    @pytest.mark.parametrize(
        "name",
        [
            "README.md",
            "QUICKSTART.md",
            "ARCHITECTURE.md",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "LICENSE",
        ],
    )
    def test_top_level_document(self, name: str) -> None:
        path = PROJECT_ROOT / name
        assert path.is_file(), f"{name} is required by the specification"
        assert path.stat().st_size > 200, f"{name} is a stub"

    @pytest.mark.parametrize(
        "name",
        [
            "package-format.md",
            "bundle-format.md",
            "air-gapped-deployment.md",
            "model-packaging.md",
            "docker-packaging.md",
            "troubleshooting.md",
        ],
    )
    def test_docs_directory_document(self, name: str) -> None:
        path = PROJECT_ROOT / "docs" / name
        assert path.is_file(), f"docs/{name} is required by the specification"
        assert path.stat().st_size > 200, f"docs/{name} is a stub"


class TestInternalLinksResolve:
    def test_relative_markdown_links_point_at_real_files(self) -> None:
        pattern = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")
        broken: list[str] = []
        for path in DOC_FILES:
            if path.name in EXCLUDED:
                continue
            for target in pattern.findall(path.read_text()):
                resolved = (path.parent / target.split("#")[0]).resolve()
                if not resolved.exists():
                    broken.append(f"{path.relative_to(PROJECT_ROOT)} -> {target}")
        assert not broken, "broken relative links:\n" + "\n".join(broken)


class TestSecurityDocumentIsHonest:
    def test_it_states_its_limitations(self) -> None:
        """A security document that only lists strengths is not useful."""
        text = (PROJECT_ROOT / "SECURITY.md").read_text().lower()
        assert "limitation" in text

    def test_it_distinguishes_verification_from_provenance(self) -> None:
        text = (PROJECT_ROOT / "SECURITY.md").read_text()
        assert "does not prove where the bundle came from" in text
