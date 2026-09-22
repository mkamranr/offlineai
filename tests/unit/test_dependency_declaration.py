"""Guards the core/builder dependency split.

This exists because of a real bug. ``cli/main.py`` imports ``click`` directly
but never declared it: it used to arrive through typer, typer 0.27 dropped the
dependency, and click then reached the project only through ``huggingface_hub``
- a *builder-only* extra. A core install, which is exactly what goes on the
air-gapped target, therefore had no click and crashed on every invocation.

The failure would have appeared on the isolated machine, where it cannot be
fixed. These tests are static and fast, so the same shape of mistake cannot
come back through some other transitive path.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "offlineai"

#: Import name -> distribution name, where they differ.
_DISTRIBUTION_NAMES = {
    "yaml": "pyyaml",
    "huggingface_hub": "huggingface-hub",
}

#: Modules a core install may import lazily because they live behind an
#: ImportError that tells the operator to install the builder extra.
BUILDER_ONLY = {"huggingface_hub", "httpx"}


def load_pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def requirement_names(entries: list[str]) -> set[str]:
    names = set()
    for entry in entries:
        name = entry.split(";")[0].strip()
        for separator in (">=", "==", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


@pytest.fixture(scope="module")
def core_dependencies() -> set[str]:
    return requirement_names(load_pyproject()["project"]["dependencies"])


@pytest.fixture(scope="module")
def builder_dependencies() -> set[str]:
    extras = load_pyproject()["project"]["optional-dependencies"]
    return requirement_names(extras["builder"])


def top_level_imports(path: Path) -> set[tuple[str, bool]]:
    """Return ``(module, is_lazy)`` for every third-party import in a file.

    ``is_lazy`` means the import happens inside a function, which is how the
    builder-only sources are allowed to reference their dependencies.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[tuple[str, bool]] = set()

    function_bodies: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                function_bodies.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # Relative imports and __future__ are not dependencies.
            if node.level or not node.module:
                continue
            modules = [node.module.split(".")[0]]
        else:
            continue
        lazy = id(node) in function_bodies
        for module in modules:
            found.add((module, lazy))
    return found


def is_stdlib(module: str) -> bool:
    return module in sys.stdlib_module_names


class TestEveryImportIsDeclared:
    """If we import it, we declare it. No relying on a transitive path."""

    def test_no_undeclared_eager_imports(
        self, core_dependencies: set[str], builder_dependencies: set[str]
    ) -> None:
        declared = core_dependencies | builder_dependencies
        offenders: list[str] = []

        for path in sorted(SOURCE_ROOT.rglob("*.py")):
            for module, lazy in top_level_imports(path):
                if module in ("offlineai", "__future__") or is_stdlib(module):
                    continue
                distribution = _DISTRIBUTION_NAMES.get(module, module).lower()
                if distribution not in declared:
                    where = "lazily" if lazy else "at module level"
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)} imports {module!r} {where}, "
                        "which no dependency declares"
                    )

        assert not offenders, "\n".join(sorted(set(offenders)))

    def test_click_is_declared_because_we_import_it(self, core_dependencies: set[str]) -> None:
        """The specific regression. Typer no longer provides click."""
        assert "click" in core_dependencies


class TestCoreInstallIsSelfSufficient:
    """The air-gapped target installs core only. Nothing it reaches at import
    time may come from a builder extra."""

    def test_builder_only_modules_are_imported_lazily(self, builder_dependencies: set[str]) -> None:
        offenders: list[str] = []
        for path in sorted(SOURCE_ROOT.rglob("*.py")):
            for module, lazy in top_level_imports(path):
                if module not in BUILDER_ONLY or lazy:
                    continue
                # A TYPE_CHECKING-guarded import costs nothing at runtime.
                if _under_type_checking(path, module):
                    continue
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)} imports builder-only "
                    f"{module!r} at module level; a core install would fail to "
                    "import it"
                )
        assert not offenders, "\n".join(offenders)

    def test_the_builder_extra_is_not_in_the_core_set(
        self, core_dependencies: set[str], builder_dependencies: set[str]
    ) -> None:
        """Keeping the target's install minimal is the point of the split."""
        overlap = core_dependencies & builder_dependencies
        assert not overlap, f"builder dependencies leaked into core: {overlap}"

    def test_cryptography_is_core_because_the_target_must_verify(
        self, core_dependencies: set[str]
    ) -> None:
        """The target never signs, but it must be able to verify a signature,
        so this one genuinely belongs in the core set."""
        assert "cryptography" in core_dependencies


def _under_type_checking(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not guarded:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.ImportFrom)
                and child.module
                and child.module.split(".")[0] == module
            ):
                return True
            if isinstance(child, ast.Import) and any(
                a.name.split(".")[0] == module for a in child.names
            ):
                return True
    return False
