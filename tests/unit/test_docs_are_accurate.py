"""The documentation must describe the tool that exists.

Docs drift silently: a command gets renamed, a flag changes, and the README
keeps confidently telling people to run something that no longer works. These
checks are cheap and catch exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from offlineai.cli.main import app
from offlineai.exitcodes import ExitCode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "offlineai"
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


class TestEveryFeatureIsDocumented:
    """The converse of TestDocumentedCommandsExist, and the reason it exists.

    That class asserts every *documented* command is real. It says nothing
    about whether a real command is documented — which is how three features
    (parallel downloads, the lock file, target profiles) each shipped
    documented in exactly one place, leaving the README silent about a file
    users are told to commit.

    An undocumented feature may as well not exist.
    """

    def _all_docs(self) -> str:
        return "\n".join(path.read_text() for path in DOC_FILES if path.name not in EXCLUDED)

    def test_every_registered_command_is_documented(self, commands: set[str]) -> None:
        text = self._all_docs()
        undocumented = sorted(c for c in commands if c not in text)
        assert not undocumented, "commands that exist but appear in no document: " + ", ".join(
            undocumented
        )

    #: Flags that change what the tool *does*, as opposed to how it prints.
    #: A user who does not know these exist is missing capability, not polish.
    BEHAVIOURAL_FLAGS = [
        "--profile",
        "--save-profile",
        "--locked",
        "--update-lock",
        "--workers",
        "--strict-offline",
        "--config",
        "--sign-key",
        "--dry-run",
        "--gpus",
    ]

    @pytest.mark.parametrize("flag", BEHAVIOURAL_FLAGS)
    def test_behavioural_flags_are_documented(self, flag: str) -> None:
        assert flag in self._all_docs(), (
            f"{flag} changes what the tool does and appears in no document"
        )

    def test_the_lock_file_is_mentioned_where_users_will_look(self) -> None:
        """Users are told to commit it, so the README and QUICKSTART have to
        say it exists."""
        for name in ("README.md", "QUICKSTART.md"):
            text = (PROJECT_ROOT / name).read_text()
            assert "offlineai.lock" in text, f"{name} does not mention offlineai.lock"


class TestTheArchitectureMapIsCurrent:
    """A stale architecture document is worse than none: it actively misleads
    someone trying to find their way around.

    This is the check that would have caught all three documentation gaps at
    once, because every one of them added or changed a subsystem.
    """

    def _module_map(self) -> str:
        text = (PROJECT_ROOT / "ARCHITECTURE.md").read_text()
        start = text.index("## Module map")
        return text[start:]

    def test_every_package_has_a_row(self) -> None:
        packages = sorted(
            p.name
            for p in SOURCE_ROOT.iterdir()
            if p.is_dir() and not p.name.startswith(("_", "."))
        )
        mapped = self._module_map()
        missing = [p for p in packages if f"`{p}/`" not in mapped]
        assert not missing, (
            "packages under src/offlineai with no row in the module map: " + ", ".join(missing)
        )

    def test_top_level_modules_that_carry_a_subsystem_are_named(self) -> None:
        """`progress.py` and `layout.py` are single files but each defines a
        contract the rest of the codebase depends on."""
        mapped = self._module_map()
        for module in ("progress.py", "layout.py"):
            assert module in mapped, f"{module} is not described in the module map"

    def test_the_schema_row_lists_what_schema_actually_holds(self) -> None:
        """The row read "offlineai.yaml, the manifest, overrides" for two
        features after the lock file and profiles were added to that package."""
        row = next(line for line in self._module_map().splitlines() if "`schema/`" in line)
        for expected in ("lock", "profile"):
            assert expected in row.lower(), (
                f"the schema/ row does not mention {expected}: {row.strip()}"
            )


class TestTheReadmeExampleMatchesTheRealOne:
    """The README walks through `examples/hello-ai`. If the example changes and
    the README does not, the first thing a new user does — copy the README and
    run it — produces something different from what the page promised.

    Written after the README's YAML was first drafted with a `curl` healthcheck
    when the real example uses a Python one-liner, precisely because
    python:3.12-slim has no curl. The README version would not have worked.
    """

    def _readme_yaml(self) -> dict[str, Any]:
        import re

        import yaml

        text = (PROJECT_ROOT / "README.md").read_text()
        match = re.search(r"```yaml\n(apiVersion: offlineai/v1.*?)```", text, re.S)
        assert match, "the README no longer shows an offlineai.yaml"
        return yaml.safe_load(match.group(1))

    def _real_yaml(self) -> dict[str, Any]:
        import yaml

        return yaml.safe_load(
            (PROJECT_ROOT / "examples" / "hello-ai" / "offlineai.yaml").read_text()
        )

    def test_the_shown_yaml_is_a_valid_package(self) -> None:
        """A README example that the schema would reject is worse than none."""
        from offlineai.schema.package import Package

        package = Package.model_validate(self._readme_yaml())
        assert package.metadata.name == "hello-ai"

    def test_everything_it_shows_agrees_with_the_real_example(self) -> None:
        """A subset check, not equality: the README abridges deliberately —
        omitting description, platform, interval_seconds — and should be free
        to. What it does show has to be true."""
        differences = _subset_differences(self._readme_yaml(), self._real_yaml(), "")
        assert not differences, (
            "the README example has drifted from examples/hello-ai:\n  " + "\n  ".join(differences)
        )

    def test_the_healthcheck_uses_a_command_the_image_has(self) -> None:
        """python:3.12-slim ships no curl. A healthcheck the container cannot
        run makes `install` fail after the workload has already started, which
        is a confusing way to learn this."""
        command = self._readme_yaml()["install"]["healthcheck"]["command"]
        assert command[0] != "curl", (
            "the shown healthcheck calls curl, which the declared base image does not contain"
        )

    def test_the_walkthrough_names_the_bundle_it_produces(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text()
        real = self._real_yaml()["metadata"]
        expected = f"{real['name']}-{real['version']}.offlineai"
        assert expected in text, f"the README does not mention {expected}"


def _subset_differences(shown: Any, real: Any, path: str) -> list[str]:
    """Every leaf present in ``shown`` must equal its counterpart in ``real``.

    Keys absent from ``shown`` are fine — that is abridgement, not drift.
    """
    if isinstance(shown, dict):
        if not isinstance(real, dict):
            return [f"{path or '(root)'}: shown as a mapping, real is {type(real).__name__}"]
        out: list[str] = []
        for key, value in shown.items():
            if key not in real:
                out.append(f"{path}.{key}: in the README, absent from the example")
            else:
                out.extend(_subset_differences(value, real[key], f"{path}.{key}"))
        return out

    if isinstance(shown, list):
        if not isinstance(real, list) or len(shown) > len(real):
            return [f"{path}: list differs in shape"]
        out = []
        for index, item in enumerate(shown):
            out.extend(_subset_differences(item, real[index], f"{path}[{index}]"))
        return out

    return [] if shown == real else [f"{path}: README shows {shown!r}, example has {real!r}"]
