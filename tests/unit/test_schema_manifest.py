"""Section 10: the manifest is the authoritative description of a bundle.

It is also the root of trust. Every artifact hash lives here, so signing the
manifest transitively covers the whole payload - which is what makes signature
verification a header-only, instant operation on a 62 GB file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from offlineai.schema.manifest import (
    FORMAT_VERSION,
    ArtifactEntry,
    ArtifactType,
    BuilderInfo,
    Manifest,
)


def entry(**overrides: object) -> ArtifactEntry:
    base: dict[str, object] = {
        "id": "model-qwen3",
        "type": "model",
        "path": "artifacts/models/qwen3/model.safetensors",
        "size": 123456789,
        "sha256": "a" * 64,
    }
    return ArtifactEntry.model_validate({**base, **overrides})


def manifest(**overrides: object) -> Manifest:
    base: dict[str, object] = {
        "formatVersion": FORMAT_VERSION,
        "package": {"name": "qwen3-30b", "version": "1.0.0"},
        "createdAt": datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
        "platforms": ["linux/amd64"],
        "artifacts": [entry().model_dump(by_alias=True)],
    }
    return Manifest.model_validate({**base, **overrides})


class TestFormatVersioning:
    def test_current_version_validates(self) -> None:
        assert manifest().format_version == FORMAT_VERSION

    def test_unknown_format_version_is_rejected(self) -> None:
        """A future bundle must fail loudly rather than be half-understood."""
        with pytest.raises(ValidationError, match="formatVersion"):
            manifest(formatVersion="99")


class TestArtifacts:
    @pytest.mark.parametrize(
        "artifact_type",
        ["model", "oci-image", "python-wheel", "system-package", "config", "script", "misc"],
    )
    def test_documented_artifact_types(self, artifact_type: str) -> None:
        assert entry(type=artifact_type).type == ArtifactType(artifact_type)

    def test_unknown_artifact_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            entry(type="sorcery")

    @pytest.mark.parametrize("digest", ["short", "g" * 64, "A" * 64, ""])
    def test_malformed_sha256_is_rejected(self, digest: str) -> None:
        with pytest.raises(ValidationError):
            entry(sha256=digest)

    def test_negative_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            entry(size=-1)

    def test_artifact_path_must_stay_inside_the_bundle(self) -> None:
        """The manifest itself is untrusted input on the target side."""
        with pytest.raises(ValidationError):
            entry(path="../../etc/passwd")

    def test_absolute_artifact_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            entry(path="/etc/passwd")

    def test_duplicate_artifact_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            manifest(
                artifacts=[
                    entry(id="dup").model_dump(by_alias=True),
                    entry(id="dup", path="artifacts/other").model_dump(by_alias=True),
                ]
            )

    def test_duplicate_artifact_paths_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            manifest(
                artifacts=[
                    entry(id="a").model_dump(by_alias=True),
                    entry(id="b").model_dump(by_alias=True),
                ]
            )


class TestLookups:
    def test_total_size_sums_artifacts(self) -> None:
        doc = manifest(
            artifacts=[
                entry(id="a", path="artifacts/a", size=100).model_dump(by_alias=True),
                entry(id="b", path="artifacts/b", size=250).model_dump(by_alias=True),
            ]
        )
        assert doc.total_size == 350

    def test_size_breakdown_by_type(self) -> None:
        doc = manifest(
            artifacts=[
                entry(id="m", type="model", path="artifacts/m", size=1000).model_dump(
                    by_alias=True
                ),
                entry(id="c", type="oci-image", path="artifacts/c", size=400).model_dump(
                    by_alias=True
                ),
            ]
        )
        breakdown = doc.size_by_type()
        assert breakdown[ArtifactType.MODEL] == 1000
        assert breakdown[ArtifactType.OCI_IMAGE] == 400

    def test_artifact_lookup_by_path(self) -> None:
        doc = manifest()
        found = doc.artifact_by_path("artifacts/models/qwen3/model.safetensors")
        assert found is not None
        assert found.id == "model-qwen3"

    def test_artifact_lookup_misses_return_none(self) -> None:
        assert manifest().artifact_by_path("artifacts/nope") is None


class TestCanonicalForm:
    """The signature covers a canonical serialisation, not the YAML bytes, so
    reformatting the file cannot invalidate a signature and reordering keys
    cannot forge one."""

    def test_canonical_bytes_are_stable_across_key_order(self) -> None:
        a = manifest()
        b = Manifest.model_validate(
            json.loads(json.dumps(a.model_dump(by_alias=True, mode="json")))
        )
        assert a.canonical_bytes() == b.canonical_bytes()

    def test_canonical_bytes_are_sorted_and_compact(self) -> None:
        raw = manifest().canonical_bytes()
        assert b", " not in raw, "canonical form must not contain formatting whitespace"
        decoded = json.loads(raw)
        assert list(decoded) == sorted(decoded)

    def test_changing_any_artifact_hash_changes_the_canonical_form(self) -> None:
        before = manifest().canonical_bytes()
        after = manifest(
            artifacts=[entry(sha256="b" * 64).model_dump(by_alias=True)]
        ).canonical_bytes()
        assert before != after

    def test_canonical_form_is_deterministic_across_calls(self) -> None:
        doc = manifest()
        assert doc.canonical_bytes() == doc.canonical_bytes()


class TestReproducibility:
    """Section 34: record enough to explain how a bundle came to be."""

    def test_builder_provenance_round_trips(self) -> None:
        doc = manifest(
            builder=BuilderInfo(
                offlineai_version="0.1.0",
                os="linux",
                architecture="amd64",
                python_version="3.12.14",
            ).model_dump(by_alias=True)
        )
        assert doc.builder is not None
        assert doc.builder.offlineai_version == "0.1.0"

    def test_package_definition_hash_is_recorded(self) -> None:
        doc = manifest(packageDefinitionSha256="c" * 64)
        assert doc.package_definition_sha256 == "c" * 64


class TestYamlRoundTrip:
    def test_round_trips_through_yaml(self) -> None:
        original = manifest()
        restored = Manifest.from_yaml(original.to_yaml())
        assert restored.canonical_bytes() == original.canonical_bytes()

    def test_created_at_is_serialised_as_utc_iso8601(self) -> None:
        # Section 10 shows the Z-suffixed form: createdAt: "2026-09-22T12:00:00Z"
        assert "2026-09-22T12:00:00Z" in manifest().to_yaml()
