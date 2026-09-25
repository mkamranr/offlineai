"""Section 35: offlineai.lock.

The manifest describes a bundle that *was* built. The lock pins what a *future*
build resolves to. Without it `revision: main` drifts and a re-tagged
`vllm:v0.6.3` silently changes what ships - which for an environment that
approves artifacts before deployment is the difference between "we rebuilt it"
and "we rebuilt the thing that was approved".
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from offlineai.schema.lockfile import (
    LOCK_FORMAT_VERSION,
    LockedArtifact,
    LockedSource,
    Lockfile,
)


def lockfile(**overrides: object) -> Lockfile:
    base: dict[str, object] = {
        "formatVersion": LOCK_FORMAT_VERSION,
        "package": {"name": "qwen3-30b", "version": "1.0.0"},
        "generatedAt": datetime(2026, 9, 22, tzinfo=UTC),
        "sources": [
            {
                "name": "model",
                "kind": "huggingface",
                "locator": "Qwen/Qwen3-30B",
                "pin": "5a7c1b4e9f2d8c3a1b0e7f6d5c4b3a2918273645",
            },
            {
                "name": "vllm",
                "kind": "oci",
                "locator": "vllm/vllm-openai:v0.6.3",
                "pin": "sha256:" + "a" * 64,
            },
        ],
        "artifacts": [
            {
                "path": "artifacts/models/model/config.json",
                "type": "model",
                "sha256": "b" * 64,
                "size": 807,
            }
        ],
    }
    return Lockfile.model_validate({**base, **overrides})


class TestFormat:
    def test_the_current_version_validates(self) -> None:
        assert lockfile().format_version == LOCK_FORMAT_VERSION

    def test_an_unknown_version_is_refused(self) -> None:
        """A lock written by a newer OfflineAI must fail loudly rather than be
        half-understood."""
        with pytest.raises(ValidationError, match="formatVersion"):
            lockfile(formatVersion="99")

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            lockfile(artifcats=[])

    def test_it_round_trips_through_yaml(self) -> None:
        original = lockfile()
        assert Lockfile.from_yaml(original.to_yaml()) == original

    def test_the_yaml_is_stable_across_writes(self) -> None:
        """A lock that reordered itself on every build would produce noise in
        every diff and code review."""
        document = lockfile()
        assert document.to_yaml() == document.to_yaml()


class TestPins:
    """A pin is what actually constrains the next resolution."""

    def test_a_source_pin_is_retrievable_by_name(self) -> None:
        assert lockfile().pin_for("model") == "5a7c1b4e9f2d8c3a1b0e7f6d5c4b3a2918273645"

    def test_an_unpinned_name_returns_none(self) -> None:
        assert lockfile().pin_for("not-declared") is None

    def test_a_pin_is_only_used_for_the_matching_locator(self) -> None:
        """If the package now points at a different repository, the old pin
        must not be applied to it."""
        document = lockfile()
        assert document.pin_for("model", locator="Qwen/Qwen3-30B") is not None
        assert document.pin_for("model", locator="meta-llama/Llama-3") is None

    def test_a_digest_pin_survives_round_trip(self) -> None:
        restored = Lockfile.from_yaml(lockfile().to_yaml())
        assert restored.pin_for("vllm") == "sha256:" + "a" * 64


class TestDriftDetection:
    def test_identical_artifacts_report_no_drift(self) -> None:
        document = lockfile()
        assert document.drift_against(document.artifacts) == []

    def test_a_changed_digest_is_drift(self) -> None:
        document = lockfile()
        changed = [
            LockedArtifact(
                path="artifacts/models/model/config.json",
                type="model",
                sha256="c" * 64,
                size=807,
            )
        ]
        drift = document.drift_against(changed)
        assert len(drift) == 1
        assert "config.json" in drift[0]

    def test_a_new_artifact_is_drift(self) -> None:
        document = lockfile()
        drift = document.drift_against(
            [
                *document.artifacts,
                LockedArtifact(
                    path="artifacts/models/model/extra.bin",
                    type="model",
                    sha256="d" * 64,
                    size=10,
                ),
            ]
        )
        assert any("extra.bin" in d for d in drift)

    def test_a_missing_artifact_is_drift(self) -> None:
        assert lockfile().drift_against([]) != []

    def test_the_message_says_what_changed(self) -> None:
        document = lockfile()
        drift = document.drift_against(
            [
                LockedArtifact(
                    path="artifacts/models/model/config.json",
                    type="model",
                    sha256="c" * 64,
                    size=807,
                )
            ]
        )
        assert "b" * 8 in drift[0] and "c" * 8 in drift[0], (
            "the drift message should show both digests"
        )

    def test_size_alone_is_not_reported_separately(self) -> None:
        """A different size always means a different digest, so reporting both
        would be noise."""
        document = lockfile()
        drift = document.drift_against(
            [
                LockedArtifact(
                    path="artifacts/models/model/config.json",
                    type="model",
                    sha256="b" * 64,
                    size=999,
                )
            ]
        )
        assert drift == []


class TestValidation:
    @pytest.mark.parametrize("digest", ["short", "g" * 64, ""])
    def test_a_malformed_artifact_digest_is_refused(self, digest: str) -> None:
        with pytest.raises(ValidationError):
            LockedArtifact(path="artifacts/x", type="model", sha256=digest, size=1)

    def test_an_artifact_path_may_not_escape_the_bundle(self) -> None:
        with pytest.raises(ValidationError):
            LockedArtifact(path="../../etc/passwd", type="model", sha256="a" * 64, size=1)

    def test_duplicate_source_names_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            lockfile(
                sources=[
                    {"name": "m", "kind": "huggingface", "locator": "a/b"},
                    {"name": "m", "kind": "huggingface", "locator": "c/d"},
                ]
            )

    def test_a_source_without_a_pin_is_allowed(self) -> None:
        """Not every source can be pinned - a local directory has nothing to
        pin to - and that must not make the lock invalid."""
        document = lockfile(sources=[{"name": "weights", "kind": "local", "locator": "./weights"}])
        assert document.pin_for("weights") is None


class TestSourceEntry:
    def test_it_records_what_was_resolved(self) -> None:
        source = LockedSource(
            name="model", kind="huggingface", locator="Qwen/Qwen3-30B", pin="abc123"
        )
        assert source.name == "model"
        assert source.pin == "abc123"
