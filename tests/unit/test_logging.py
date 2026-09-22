"""Section 48: structured logging that never logs secrets."""

from __future__ import annotations

import json
import logging

import pytest

from offlineai.logging import LogFormat, configure_logging, get_logger, redact


class TestRedaction:
    """Redaction is applied to every record, so a careless log call in some
    future module still cannot leak a credential."""

    @pytest.mark.parametrize(
        "secret",
        [
            "hf_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "sk-abcdefghijklmnopqrstuvwxyz0123456789",
        ],
    )
    def test_known_token_shapes_are_removed(self, secret: str) -> None:
        assert secret not in redact(f"using token {secret} for download")

    def test_url_userinfo_is_removed(self) -> None:
        out = redact("fetching https://alice:hunter2@registry.internal/v2/img")
        assert "hunter2" not in out
        assert "registry.internal" in out, "host must survive; only the credential goes"

    @pytest.mark.parametrize("key", ["token", "password", "secret", "api_key", "authorization"])
    def test_key_value_pairs_are_removed(self, key: str) -> None:
        out = redact(f"{key}=swordfish next")
        assert "swordfish" not in out
        assert key in out, "the key is useful context; only the value is dropped"

    def test_private_key_blocks_are_removed(self) -> None:
        pem = "-----BEGIN PRIVATE KEY-----\nMIIBVgIBADANBg\n-----END PRIVATE KEY-----"
        out = redact(f"loaded {pem}")
        assert "MIIBVgIBADANBg" not in out

    def test_ordinary_text_is_untouched(self) -> None:
        msg = "Loading container image vllm/vllm-openai@sha256:abc123 (12.1 GB)"
        assert redact(msg) == msg

    def test_sha256_digests_are_not_mistaken_for_secrets(self) -> None:
        digest = "sha256:" + "a" * 64
        assert digest in redact(f"digest {digest}")


class TestRedactionInPipeline:
    """Asserted against real emitted output, not caplog.

    The offlineai logger sets propagate=False, so caplog's root handler never
    sees these records - a caplog-based assertion here would pass even with
    redaction entirely removed. Each test carries a positive control so it
    cannot pass on empty output either.
    """

    def test_message_is_redacted(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="debug")
        get_logger("offlineai.test").info("auth with hf_abcdefghijklmnopqrstuvwxyz0123456789")
        err = capsys.readouterr().err
        assert "auth with" in err, "positive control: the record must actually be emitted"
        assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in err
        assert "[REDACTED]" in err

    def test_interpolation_arguments_are_redacted(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="debug")
        get_logger("offlineai.test").info("token is %s", "hf_abcdefghijklmnopqrstuvwxyz0123456789")
        err = capsys.readouterr().err
        assert "token is" in err, "positive control: the record must actually be emitted"
        assert "hf_abcdefghijk" not in err
        assert "[REDACTED]" in err

    def test_exception_tracebacks_are_redacted(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="debug", fmt=LogFormat.JSON)
        try:
            raise ValueError("failed for token=hunter2")
        except ValueError:
            get_logger("offlineai.test").exception("download failed")
        err = capsys.readouterr().err
        assert "download failed" in err, "positive control"
        assert "hunter2" not in err


class TestJsonFormat:
    def test_emits_one_json_object_per_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="info", fmt=LogFormat.JSON)
        get_logger("offlineai.test").warning("disk is nearly full")
        line = capsys.readouterr().err.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["level"] == "warning"
        assert record["message"] == "disk is nearly full"
        assert record["logger"] == "offlineai.test"
        assert "timestamp" in record

    def test_extra_fields_are_included(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="info", fmt=LogFormat.JSON)
        get_logger("offlineai.test").info("artifact stored", extra={"size_bytes": 4096})
        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["size_bytes"] == 4096


class TestLevels:
    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    def test_documented_levels_are_accepted(self, level: str) -> None:
        configure_logging(level=level)
        assert logging.getLogger("offlineai").level == getattr(logging, level.upper())

    def test_unknown_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="trace"):
            configure_logging(level="trace")

    def test_logs_go_to_stderr_so_stdout_stays_machine_readable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="info")
        get_logger("offlineai.test").info("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.err
        assert captured.out == ""
