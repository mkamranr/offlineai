"""Sections 67 and 68: layered configuration.

Precedence is defaults < config.yaml < OFFLINEAI_* environment < CLI flags.
Getting this order wrong is the kind of bug that silently writes a 62 GB
registry to the wrong volume, so each layer boundary is tested directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offlineai.config.settings import Settings, load_settings
from offlineai.errors import ConfigurationError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OFFLINEAI_HOME",
        "OFFLINEAI_DATA_DIR",
        "OFFLINEAI_CACHE_DIR",
        "OFFLINEAI_REGISTRY_DIR",
        "OFFLINEAI_LOG_LEVEL",
        "OFFLINEAI_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)


class TestDefaults:
    def test_home_defaults_under_the_user_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A real directory, not a fictional one: macOS resolves /home through a
        # synthetic firmlink, which would make the assertion platform-specific.
        monkeypatch.setenv("HOME", str(tmp_path))
        assert load_settings().home == (tmp_path / ".offlineai").resolve()

    def test_derived_directories_sit_under_home(self, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path)
        assert settings.data_dir == tmp_path
        assert settings.registry_dir == tmp_path / "registry"
        assert settings.cache_dir == tmp_path / "cache"
        assert settings.bundles_dir == tmp_path / "bundles"
        assert settings.logs_dir == tmp_path / "logs"

    def test_conservative_download_concurrency(self, tmp_path: Path) -> None:
        """Section 45: the default must not overwhelm storage or network."""
        assert load_settings(home=tmp_path).downloads.workers == 4

    def test_signatures_are_not_required_by_default(self, tmp_path: Path) -> None:
        """Section 12: external signing infrastructure must not be mandatory."""
        assert load_settings(home=tmp_path).security.require_signature is False


class TestPrecedence:
    def test_config_file_overrides_defaults(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("downloads:\n  workers: 16\n")
        assert load_settings(home=home).downloads.workers == 16

    def test_environment_overrides_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text(f"cache_dir: {tmp_path / 'from-file'}\n")
        monkeypatch.setenv("OFFLINEAI_CACHE_DIR", str(tmp_path / "from-env"))
        assert load_settings(home=home).cache_dir == tmp_path / "from-env"

    def test_cli_overrides_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OFFLINEAI_DATA_DIR", str(tmp_path / "from-env"))
        settings = load_settings(data_dir=tmp_path / "from-cli")
        assert settings.data_dir == tmp_path / "from-cli"

    def test_cli_overrides_config_file(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("offline:\n  strict: false\n")
        assert load_settings(home=home, strict_offline=True).offline.strict is True

    def test_full_chain_resolves_to_the_cli_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("log_level: warning\n")
        monkeypatch.setenv("OFFLINEAI_LOG_LEVEL", "error")
        assert load_settings(home=home, log_level="debug").log_level == "debug"

    def test_environment_wins_when_no_cli_flag_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("log_level: warning\n")
        monkeypatch.setenv("OFFLINEAI_LOG_LEVEL", "error")
        assert load_settings(home=home).log_level == "error"


class TestOfflineFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_offline_env_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("OFFLINEAI_OFFLINE", value)
        assert load_settings(home=tmp_path).offline.strict is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_offline_env_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("OFFLINEAI_OFFLINE", value)
        assert load_settings(home=tmp_path).offline.strict is False


class TestValidation:
    def test_malformed_config_file_is_reported_with_its_path(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("downloads: [not, a, mapping]\n")
        with pytest.raises(ConfigurationError) as excinfo:
            load_settings(home=home)
        assert "config.yaml" in excinfo.value.render()

    def test_unparseable_yaml_is_reported(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("key: [unclosed\n")
        with pytest.raises(ConfigurationError):
            load_settings(home=home)

    def test_unknown_config_key_is_rejected(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("data_dri: /typo\n")
        with pytest.raises(ConfigurationError) as excinfo:
            load_settings(home=home)
        # The offending key is named in the detail blocks, which render()
        # includes and str() deliberately does not.
        assert "data_dri" in excinfo.value.render()

    def test_zero_workers_is_rejected(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("downloads:\n  workers: 0\n")
        with pytest.raises(ConfigurationError):
            load_settings(home=home)

    def test_missing_config_file_is_fine(self, tmp_path: Path) -> None:
        assert load_settings(home=tmp_path / "does-not-exist").downloads.workers == 4


class TestDirectoryCreation:
    def test_ensure_directories_creates_the_documented_tree(self, tmp_path: Path) -> None:
        """Section 7.1."""
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        for directory in (
            settings.registry_dir,
            settings.cache_dir,
            settings.bundles_dir,
            settings.logs_dir,
        ):
            assert directory.is_dir()

    def test_ensure_directories_is_idempotent(self, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path / "home")
        settings.ensure_directories()
        settings.ensure_directories()

    def test_paths_are_absolute(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        settings = load_settings(data_dir=Path("relative-dir"))
        assert settings.data_dir.is_absolute()


class TestSerialisation:
    def test_round_trips_to_yaml(self, tmp_path: Path) -> None:
        settings = load_settings(home=tmp_path)
        restored = Settings.model_validate_yaml(settings.to_yaml())
        assert restored.downloads.workers == settings.downloads.workers
