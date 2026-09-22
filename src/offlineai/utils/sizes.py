"""Human-readable sizes and durations."""

from __future__ import annotations

import re

__all__ = ["format_bytes", "format_duration", "parse_bytes"]

_DECIMAL_UNITS = ("B", "kB", "MB", "GB", "TB", "PB")
_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")

_SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtTpP]?i?[bB]?)\s*$")
_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "ki": 1024,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gi": 1024**3,
    "gib": 1024**3,
    "t": 1000**4,
    "tb": 1000**4,
    "ti": 1024**4,
    "tib": 1024**4,
    "p": 1000**5,
    "pb": 1000**5,
    "pi": 1024**5,
    "pib": 1024**5,
}


def format_bytes(count: int | float, *, binary: bool = False, precision: int = 1) -> str:
    """Format a byte count.

    Decimal by default, because that is what the specification's own output
    examples use ("Size: 82.4 GB") and what storage vendors label.
    """
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    step = 1024.0 if binary else 1000.0
    value = float(count)
    if value < step:
        return f"{int(value)} B"
    for unit in units[1:]:
        value /= step
        if value < step:
            return f"{value:.{precision}f} {unit}"
    return f"{value:.{precision}f} {units[-1]}"


def parse_bytes(text: str) -> int:
    """Parse ``"48 GB"`` or ``"48GiB"`` into a byte count."""
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"{text!r} is not a recognised size")
    unit = match.group("unit").lower().rstrip("b") if match.group("unit") else ""
    key = match.group("unit").lower()
    multiplier = _MULTIPLIERS.get(key) or _MULTIPLIERS.get(unit)
    if multiplier is None:
        raise ValueError(f"{text!r} has an unrecognised unit")
    return int(float(match.group("value")) * multiplier)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} sec"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
