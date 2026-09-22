"""Structured logging with mandatory secret redaction.

Section 48 requires structured logging across four levels and states plainly:
do not log secrets. Redaction is implemented as a logging *filter* rather than
a call-site discipline, so a careless ``logger.debug(url)`` in some future
module still cannot leak a credential.

Everything goes to stderr. Stdout belongs to ``--json`` payloads, and mixing
the two would break every machine consumer.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = ["LogFormat", "configure_logging", "get_logger", "redact"]

ROOT_LOGGER_NAME = "offlineai"

LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_PLACEHOLDER = "[REDACTED]"

# Ordered most-specific first. Each pattern keeps enough surrounding context to
# stay useful for debugging (the host, the key name) while dropping the value.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # PEM private key blocks, including the body.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        _PLACEHOLDER,
    ),
    # Credentials embedded in a URL: https://user:pass@host -> https://[REDACTED]@host
    (
        re.compile(r"(?P<scheme>\b[a-zA-Z][\w+.-]*://)[^/\s:@]+:[^/\s@]+@"),
        r"\g<scheme>" + _PLACEHOLDER + "@",
    ),
    # Provider token shapes.
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), _PLACEHOLDER),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), _PLACEHOLDER),
    (re.compile(r"\bsk-[A-Za-z0-9._-]{20,}"), _PLACEHOLDER),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _PLACEHOLDER),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), _PLACEHOLDER),
    # Bearer headers.
    (
        re.compile(r"\b(?P<kind>Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
        r"\g<kind> " + _PLACEHOLDER,
    ),
    # key=value / key: value for sensitive key names. The key survives.
    (
        re.compile(
            r"(?P<key>\b(?:password|passwd|secret|api[_-]?key|access[_-]?key|"
            r"auth[_-]?token|authorization|token|credential)s?\b)"
            r"(?P<sep>\s*[:=]\s*)"
            r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+)",
            re.IGNORECASE,
        ),
        r"\g<key>\g<sep>" + _PLACEHOLDER,
    ),
)


def redact(text: str) -> str:
    """Remove credential-shaped substrings, preserving surrounding context."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class LogFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class _RedactingFilter(logging.Filter):
    """Scrub the message and its interpolation arguments before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact_value(v) for k, v in record.args.items()}
            else:
                record.args = tuple(_redact_value(a) for a in record.args)
        return True


def _redact_value(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


# Attributes the stdlib puts on every record; anything else came from `extra=`.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shipping and CI parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="%(levelname)-7s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = record.levelname.lower()
        return super().format(record)


def configure_logging(
    level: str = "info",
    fmt: LogFormat | str = LogFormat.TEXT,
    *,
    stream: Any = None,
) -> None:
    """Install the OfflineAI log handler.

    Idempotent: calling it again replaces the previous handler rather than
    stacking duplicates, which matters because the CLI configures logging once
    global flags are parsed.
    """
    normalized = level.strip().lower()
    if normalized not in LEVELS:
        raise ValueError(f"Unknown log level {level!r}. Valid levels: {', '.join(LEVELS)}.")
    fmt = LogFormat(fmt)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt is LogFormat.JSON else _TextFormatter())
    handler.addFilter(_RedactingFilter())

    logger.addHandler(handler)
    logger.setLevel(LEVELS[normalized])
    # Our handler is terminal; without this, records also reach the root logger
    # and would print twice under pytest or an embedding application.
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``offlineai`` namespace."""
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
