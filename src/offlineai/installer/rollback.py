"""Rollback (section 27).

Replays each completed step's recorded inverse actions in reverse order.

Two rules the implementation holds to:

* **Never delete user-owned data.** The action set is closed and covers only
  things OfflineAI created: containers it started, networks it made, images it
  loaded, paths it wrote. A declared volume containing an operator's data is
  never touched.
* **Keep going.** A rollback that stops at the first error leaves the system in
  a worse state than one that does its best and reports what it could not
  undo. Failures are collected and reported, not raised.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from offlineai.config.settings import Settings
from offlineai.errors import RegistryError
from offlineai.installer.transaction import (
    InstallState,
    InstallTransaction,
    Inverse,
    InverseAction,
)
from offlineai.logging import get_logger
from offlineai.registry.db import open_registry_db
from offlineai.registry.registry import Registry
from offlineai.runtime.base import ContainerRuntime

__all__ = ["RollbackResult", "rollback_installation"]

logger = get_logger("installer.rollback")


@dataclass(slots=True)
class RollbackResult:
    installation_id: str
    package: str
    undone: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.failed


def rollback_installation(
    settings: Settings,
    registry: Registry,
    runtime: ContainerRuntime,
    installation_id: str,
    *,
    keep_images: bool = True,
) -> RollbackResult:
    """Undo an installation.

    ``keep_images`` defaults to true: loading a 12 GB image is expensive and
    keeping it is harmless, so images stay unless the caller asks otherwise.
    """
    with open_registry_db(registry.db_path) as connection:
        row = connection.execute(
            """
            SELECT i.id, i.package_id, i.state, p.name
            FROM installations i JOIN packages p ON p.id = i.package_id
            WHERE i.id = ?
            """,
            (installation_id,),
        ).fetchone()

    if row is None:
        raise RegistryError(
            f"no installation with id {installation_id!r}",
            action="List installations with:\n  offlineai list --installations",
        )

    txn = InstallTransaction(registry.db_path, installation_id, int(row["package_id"]))
    result = RollbackResult(installation_id=installation_id, package=row["name"])

    # Reverse order: the last thing done is the first thing undone.
    for step in reversed(txn.steps()):
        for inverse in reversed(step.inverses):
            if inverse.action is InverseAction.REMOVE_IMAGE and keep_images:
                result.skipped.append(f"{inverse.action.value} {inverse.target} (kept)")
                continue
            try:
                _apply(inverse, runtime, settings)
            except Exception as exc:  # noqa: BLE001 - report, never abort a rollback
                logger.warning(
                    "rollback action failed: %s %s: %s", inverse.action.value, inverse.target, exc
                )
                result.failed.append(f"{inverse.action.value} {inverse.target}: {exc}")
            else:
                result.undone.append(f"{inverse.action.value} {inverse.target}")

    txn.set_state(InstallState.ROLLED_BACK)
    return result


def _apply(inverse: Inverse, runtime: ContainerRuntime, settings: Settings) -> None:
    match inverse.action:
        case InverseAction.REMOVE_CONTAINER:
            runtime.stop_container(inverse.target)
            runtime.remove_container(inverse.target)
        case InverseAction.REMOVE_NETWORK:
            runtime.remove_network(inverse.target)
        case InverseAction.REMOVE_IMAGE:
            runtime.remove_image(inverse.target)
        case InverseAction.REMOVE_PATH:
            _remove_path(Path(inverse.target), settings)


def _remove_path(path: Path, settings: Settings) -> None:
    """Delete a path OfflineAI created, refusing anything outside its own tree.

    The containment check is the safety rail: a corrupted or tampered journal
    must not be able to turn rollback into a way to delete arbitrary files.
    """
    resolved = path.resolve()
    root = (settings.data_dir / "installed").resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"refusing to remove {resolved}: outside the OfflineAI install tree")
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=False)
    elif resolved.exists():
        resolved.unlink()
