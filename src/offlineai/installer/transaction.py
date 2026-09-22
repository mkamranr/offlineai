"""Installation transactions (section 27).

Every step that changes the system records how to undo itself, in the database,
as it completes. Rollback then replays those inverses in reverse order.

Recording the inverse rather than inferring it at rollback time matters: after
a failure the system is in an unknown state, and "what did we actually do?" is
not a question we want to answer by guessing. The journal is written as we go,
so it survives the process dying mid-install.

One rule from section 27 is absolute: rollback never deletes user-owned data.
Inverses only ever undo things OfflineAI itself created.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from offlineai.logging import get_logger
from offlineai.registry.db import open_registry_db

__all__ = ["InstallState", "InstallTransaction", "Inverse", "InverseAction", "next_install_id"]

logger = get_logger("installer.transaction")


class InstallState(StrEnum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    INSTALLING = "INSTALLING"
    STARTING = "STARTING"
    HEALTH_CHECK = "HEALTH_CHECK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class InverseAction(StrEnum):
    """What undoing a step means.

    Deliberately a closed set. Rollback must never be able to run an arbitrary
    command recorded earlier - that would turn a failed install into an
    execution primitive.
    """

    REMOVE_CONTAINER = "remove_container"
    REMOVE_NETWORK = "remove_network"
    REMOVE_IMAGE = "remove_image"
    REMOVE_PATH = "remove_path"


@dataclass(frozen=True, slots=True)
class Inverse:
    action: InverseAction
    target: str


@dataclass(slots=True)
class StepRecord:
    seq: int
    name: str
    state: str
    inverses: list[Inverse]
    detail: str | None = None


def next_install_id(db_path: Path, *, now: datetime | None = None) -> str:
    """Allocate ``install-YYYYMMDD-NNN`` (section 27)."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d")
    prefix = f"install-{stamp}-"
    with open_registry_db(db_path) as connection:
        rows = connection.execute(
            "SELECT id FROM installations WHERE id LIKE ?", (f"{prefix}%",)
        ).fetchall()
    highest = 0
    for row in rows:
        suffix = str(row[0]).rsplit("-", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{prefix}{highest + 1:03d}"


class InstallTransaction:
    """Records an installation's progress and how to undo it."""

    def __init__(self, db_path: Path, install_id: str, package_id: int) -> None:
        self.db_path = db_path
        self.id = install_id
        self.package_id = package_id
        self._seq = 0
        self.state = InstallState.PENDING

    def begin(self, config: dict[str, object] | None = None) -> None:
        with open_registry_db(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO installations (id, package_id, state, started_at, config_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.id,
                    self.package_id,
                    InstallState.PENDING.value,
                    datetime.now(UTC).isoformat(),
                    json.dumps(config or {}, default=str),
                ),
            )

    def set_state(self, state: InstallState, *, error: str | None = None) -> None:
        self.state = state
        finished = (
            datetime.now(UTC).isoformat()
            if state in (InstallState.COMPLETED, InstallState.FAILED, InstallState.ROLLED_BACK)
            else None
        )
        with open_registry_db(self.db_path) as connection:
            connection.execute(
                "UPDATE installations SET state = ?, finished_at = ?, error = ? WHERE id = ?",
                (state.value, finished, error, self.id),
            )

    def record(
        self, name: str, *, inverses: list[Inverse] | None = None, detail: str | None = None
    ) -> None:
        """Journal a completed step and its undo actions."""
        self._seq += 1
        payload = json.dumps([asdict(i) for i in (inverses or [])], default=str)
        with open_registry_db(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO installation_steps
                    (installation_id, seq, name, state, inverse_json, detail)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.id, self._seq, name, "COMPLETED", payload, detail),
            )

    def steps(self) -> list[StepRecord]:
        with open_registry_db(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT seq, name, state, inverse_json, detail
                FROM installation_steps WHERE installation_id = ? ORDER BY seq
                """,
                (self.id,),
            ).fetchall()
        records = []
        for row in rows:
            raw = json.loads(row["inverse_json"] or "[]")
            records.append(
                StepRecord(
                    seq=int(row["seq"]),
                    name=row["name"],
                    state=row["state"],
                    inverses=[
                        Inverse(action=InverseAction(i["action"]), target=i["target"]) for i in raw
                    ],
                    detail=row["detail"],
                )
            )
        return records

    @contextmanager
    def step(self, name: str) -> Iterator[list[Inverse]]:
        """Run a step, journaling whatever it registers as its undo actions.

        Inverses are appended by the step body as each change is made, so a
        failure halfway through still leaves an accurate record of what needs
        undoing.
        """
        inverses: list[Inverse] = []
        try:
            yield inverses
        except BaseException:
            # Journal what did happen before propagating, so rollback can act
            # on partial progress.
            if inverses:
                self.record(name, inverses=inverses, detail="failed partway")
            raise
        else:
            self.record(name, inverses=inverses)
