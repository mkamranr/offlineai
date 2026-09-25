"""``offlineai.lock`` (section 35).

The manifest describes a bundle that *was* built. The lock pins what a *future*
build should resolve to, which is a different job and needs a different file.

Without one, ``revision: main`` drifts, a re-tagged ``vllm:v0.6.3`` silently
becomes a different image, and a dependency resolves to whatever is newest.
For an environment that approves artifacts before they are allowed in, that is
the difference between "we rebuilt it" and "we rebuilt the thing that was
approved".

Two sections, because they answer two questions:

* ``sources`` records what each declaration *resolved to* - a Hugging Face
  commit, an image digest. On a locked build these are fed back in, so
  resolution is constrained rather than merely checked afterwards.
* ``artifacts`` records the digest of every file that came out. These are
  compared after resolution, which catches anything the pins could not.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LOCK_FILENAME",
    "LOCK_FORMAT_VERSION",
    "LockedArtifact",
    "LockedPackage",
    "LockedSource",
    "Lockfile",
]

LOCK_FILENAME = "offlineai.lock"
LOCK_FORMAT_VERSION = "1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _short(digest: str | None) -> str:
    if not digest:
        return "(none)"
    body = digest.removeprefix("sha256:")
    return f"sha256:{body[:8]}…"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class LockedPackage(_Strict):
    name: str
    version: str


class LockedSource(_Strict):
    """What one declaration resolved to, last time.

    ``pin`` is deliberately an opaque string: its meaning belongs to the
    source. Hugging Face reads it as a commit, OCI as an image digest. A
    source with nothing to pin - a local directory - simply leaves it unset,
    and that is not an error.
    """

    name: str
    kind: str
    locator: str
    pin: str | None = None
    #: Anything else worth recording for a human reading the diff.
    resolved: dict[str, str] = Field(default_factory=dict)


class LockedArtifact(_Strict):
    path: str
    type: str
    sha256: str
    size: int = Field(ge=0)
    #: Populated for Python wheels, where the version is the thing reviewers
    #: actually look at.
    version: str | None = None
    #: The artifact's identity at its origin, when it has one distinct from
    #: its bytes. For a container image this is the registry digest.
    digest: str | None = None

    @field_validator("sha256")
    @classmethod
    def _check_digest(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(f"{value!r} is not a SHA-256 digest (64 lowercase hex characters)")
        return value

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        # A lock file is read from the working tree and could have been edited
        # by anything, so its paths get the same scrutiny as a manifest's.
        if not value or value.startswith("/") or "\\" in value:
            raise ValueError(f"{value!r} is not a valid bundle-relative path")
        if any(part == ".." for part in PurePosixPath(value).parts):
            raise ValueError(f"{value!r} escapes the bundle root")
        return value


class Lockfile(_Strict):
    format_version: Literal["1"] = Field(alias="formatVersion")
    package: LockedPackage
    generated_at: datetime = Field(alias="generatedAt")
    #: Digest of the offlineai.yaml this lock was generated from, so an
    #: obviously stale lock can be spotted.
    package_definition_sha256: str | None = Field(default=None, alias="packageDefinitionSha256")
    sources: list[LockedSource] = Field(default_factory=list)
    artifacts: list[LockedArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique(self) -> Lockfile:
        for label, values in (
            ("source name", [s.name for s in self.sources]),
            ("artifact path", [a.path for a in self.artifacts]),
        ):
            seen: set[str] = set()
            for value in values:
                if value in seen:
                    raise ValueError(f"duplicate {label}: {value!r}")
                seen.add(value)
        return self

    # -- pins ------------------------------------------------------------

    def pin_for(self, name: str, *, locator: str | None = None) -> str | None:
        """The recorded pin for a declaration, if it still applies.

        When ``locator`` is given it must match what was locked. A package
        that now points at a different repository must not have the old
        repository's commit applied to it - that would pin the wrong thing
        entirely, which is worse than not pinning at all.
        """
        source = next((s for s in self.sources if s.name == name), None)
        if source is None:
            return None
        if locator is not None and source.locator != locator:
            return None
        return source.pin

    def source_for(self, name: str) -> LockedSource | None:
        return next((s for s in self.sources if s.name == name), None)

    # -- drift -----------------------------------------------------------

    def drift_against(self, resolved: list[LockedArtifact]) -> list[str]:
        """Compare a fresh resolution against what was locked.

        Returns human-readable descriptions, empty when they agree. Size is
        not reported separately: different bytes always mean a different
        digest, so mentioning both would be noise.
        """
        locked = {a.path: a for a in self.artifacts}
        current = {a.path: a for a in resolved}
        drift: list[str] = []

        for path in sorted(set(locked) - set(current)):
            drift.append(f"{path}: in the lock but no longer resolved")
        for path in sorted(set(current) - set(locked)):
            drift.append(f"{path}: resolved but not in the lock")
        for path in sorted(set(locked) & set(current)):
            before, after = locked[path], current[path]

            # A container image is compared on its registry digest, not on the
            # bytes of the tar we produced from it. `docker save` is not
            # byte-reproducible - saving the same image by tag and by digest
            # gives different archives, and even two saves by tag can differ -
            # so comparing tar bytes would report drift on an image that never
            # changed. The registry digest is the identity that means
            # something, and section 16 already says to rely on it.
            if before.digest or after.digest:
                if before.digest != after.digest:
                    drift.append(
                        f"{path}: image digest changed, "
                        f"{_short(before.digest)} -> {_short(after.digest)}"
                    )
                continue

            if before.sha256 != after.sha256:
                drift.append(f"{path}: digest changed, {before.sha256[:8]}… -> {after.sha256[:8]}…")
            elif before.version != after.version and after.version:
                drift.append(f"{path}: version changed, {before.version} -> {after.version}")
        return drift

    # -- serialisation ---------------------------------------------------

    def to_yaml(self) -> str:
        """Deterministic YAML.

        Sorted and stable: a lock that reordered itself on every build would
        put noise in every diff, and this file exists to be reviewed.
        """
        payload = self.model_dump(by_alias=True, mode="json", exclude_none=True)
        # Drop empty optional maps: this file is meant to be read in a diff,
        # and `resolved: {}` on every entry is pure noise.
        for source in payload.get("sources", []):
            if not source.get("resolved"):
                source.pop("resolved", None)
        payload["sources"] = sorted(payload.get("sources", []), key=lambda s: s["name"])
        payload["artifacts"] = sorted(payload.get("artifacts", []), key=lambda a: a["path"])
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, text: str | bytes) -> Lockfile:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("lock file must be a YAML mapping")
        return cls.model_validate(data)
