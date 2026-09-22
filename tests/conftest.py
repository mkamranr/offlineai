"""Shared pytest configuration.

The suite is air-gapped by default: every test runs with outbound network
access blocked in-process. Tests that genuinely need the network declare
``@pytest.mark.allow_network``.

Setting ``OFFLINEAI_TEST_OFFLINE=1`` (the spelling section 63 specifies) makes
the run strict: the opt-out marker stops granting access and those tests are
skipped instead, proving the whole suite passes with no network whatsoever.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.netguard import install_guard

OFFLINE_ENV_VAR = "OFFLINEAI_TEST_OFFLINE"


def strict_offline() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR, "").strip() == "1"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "allow_network: test needs real outbound access; skipped under OFFLINEAI_TEST_OFFLINE=1",
    )


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block outbound network access for the duration of each test."""
    if request.node.get_closest_marker("allow_network"):
        if strict_offline():
            pytest.skip(f"needs network; {OFFLINE_ENV_VAR}=1 forbids it")
        return
    install_guard(monkeypatch)


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point every OFFLINEAI_* location at a temp directory.

    Without this a test run would read - or worse, write - the developer's real
    ``~/.offlineai`` registry.
    """
    home = tmp_path / "offlineai-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OFFLINEAI_HOME", str(home))
    monkeypatch.setenv("OFFLINEAI_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("OFFLINEAI_CACHE_DIR", str(home / "cache"))
    monkeypatch.setenv("OFFLINEAI_REGISTRY_DIR", str(home / "data" / "registry"))
    monkeypatch.delenv("OFFLINEAI_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OFFLINEAI_OFFLINE", raising=False)
    yield home
