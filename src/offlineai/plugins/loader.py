"""Plugin discovery (section 49).

Artifact sources are found through the ``offlineai.sources`` entry-point group,
so a third party can add ModelScope, S3, NGC or an APT mirror by shipping a
package - no change to the CLI, which section 49 explicitly asks for.

A plugin that fails to load is reported and skipped, never fatal. One broken
third-party source must not stop an operator installing an unrelated package
on a machine where they cannot pip-uninstall anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from offlineai.logging import get_logger

__all__ = ["ENTRY_POINT_GROUP", "LoadedSources", "load_sources"]

logger = get_logger("plugins")

ENTRY_POINT_GROUP = "offlineai.sources"


@dataclass(slots=True)
class LoadedSources:
    sources: dict[str, Any] = field(default_factory=dict)
    #: ``name -> reason``, for plugins that could not be loaded.
    failures: dict[str, str] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.sources)


def load_sources(*, include_failures: bool = True) -> LoadedSources:
    """Discover artifact source plugins."""
    result = LoadedSources()

    try:
        points = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 - a broken environment must not be fatal
        logger.warning("could not enumerate plugins: %s", exc)
        return result

    for point in points:
        try:
            result.sources[point.name] = _instantiate(point)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break the rest
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("plugin %r failed to load: %s", point.name, reason)
            if include_failures:
                result.failures[point.name] = reason

    return result


def _instantiate(point: EntryPoint) -> Any:
    """Load an entry point, accepting either a class or a factory.

    Classes needing constructor arguments (the OCI source needs a runtime) are
    returned uninstantiated; the builder supplies what they need. Anything
    constructible with no arguments is instantiated here so it is ready to use.
    """
    target = point.load()
    if isinstance(target, type):
        try:
            return target()
        except TypeError:
            return target
    return target
