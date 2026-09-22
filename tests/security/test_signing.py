"""Section 12: signing, and what it is actually good for.

The interesting case is not a corrupted byte - checksums already catch that.
It is an attacker who rewrites an artifact *and* the manifest so the bundle is
internally consistent. Checksums cannot detect that; only a signature over the
manifest can, which is why the manifest is the root of trust.
"""

from __future__ import annotations

import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from offlineai import layout
from offlineai.bundler.archive import BundleReader, BundleWriter
from offlineai.errors import ChecksumMismatchError, SignatureError
from offlineai.exitcodes import ExitCode
from offlineai.schema.manifest import FORMAT_VERSION, Manifest
from offlineai.security.signing import (
    SignatureEnvelope,
    fingerprint,
    generate_keypair,
    load_private_key,
    load_public_key,
    read_detached,
    sign_manifest,
    verify_manifest,
    write_detached,
)
from offlineai.utils.hashing import sha256_bytes

GENUINE = b"the model weights everyone expects" * 500
MALICIOUS = b"weights that do something else entirely" * 440


def manifest_for(payload: bytes) -> Manifest:
    return Manifest.model_validate(
        {
            "formatVersion": FORMAT_VERSION,
            "package": {"name": "signed-demo", "version": "1.0.0"},
            "createdAt": datetime(2026, 9, 22, tzinfo=UTC),
            "platforms": ["linux/amd64"],
            "artifacts": [
                {
                    "id": "model",
                    "type": "model",
                    "path": "artifacts/models/m/model.safetensors",
                    "size": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            ],
        }
    )


@pytest.fixture
def keys(tmp_path: Path) -> tuple[Path, Path]:
    return generate_keypair(tmp_path / "signing-key.pem")


@pytest.fixture
def signed_bundle(tmp_path: Path, keys: tuple[Path, Path]) -> Path:
    private, _ = keys
    payload = tmp_path / "model.safetensors"
    payload.write_bytes(GENUINE)

    manifest = manifest_for(GENUINE)
    envelope = sign_manifest(manifest, load_private_key(private), signer="release@example")

    import base64

    bundle = tmp_path / "signed-demo-1.0.0.offlineai"
    with BundleWriter(bundle) as writer:
        writer.write_header(
            manifest=manifest,
            package_yaml=b"kind: Package\n",
            signature=envelope.to_bytes(),
            public_key=base64.b64decode(envelope.public_key),
        )
        writer.add_artifact("artifacts/models/m/model.safetensors", payload)
    return bundle


class TestHappyPath:
    def test_a_signed_bundle_verifies(self, signed_bundle: Path, keys: tuple[Path, Path]) -> None:
        _, public = keys
        with BundleReader.open(signed_bundle) as reader:
            header = reader.read_header()
        assert header.signature is not None
        envelope = SignatureEnvelope.from_bytes(header.signature)
        verify_manifest(header.manifest, envelope, trusted_key=load_public_key(public))

    def test_verification_reads_only_the_header(self, signed_bundle: Path) -> None:
        """The reason the manifest is what gets signed: this stays instant on a
        62 GB bundle."""
        total = signed_bundle.stat().st_size
        _, consumed = BundleReader.read_header_counting_bytes(signed_bundle)
        assert consumed < total // 2

    def test_the_signer_identity_is_carried(self, signed_bundle: Path) -> None:
        with BundleReader.open(signed_bundle) as reader:
            header = reader.read_header()
        assert header.signature is not None
        assert SignatureEnvelope.from_bytes(header.signature).signer == "release@example"


class TestTamperDetection:
    def test_a_flipped_artifact_byte_is_caught_by_checksums(self, signed_bundle: Path) -> None:
        from offlineai.bundler.verifier import verify_bundle

        data = bytearray(signed_bundle.read_bytes())
        data[int(len(data) * 0.7)] ^= 0xFF
        signed_bundle.write_bytes(bytes(data))
        with pytest.raises(ChecksumMismatchError):
            verify_bundle(signed_bundle)

    def test_a_consistently_rewritten_bundle_is_caught_only_by_the_signature(
        self, tmp_path: Path, keys: tuple[Path, Path], signed_bundle: Path
    ) -> None:
        """The attack that matters.

        An attacker replaces the artifact AND rewrites the manifest so its
        recorded hash matches the new content. The bundle is now internally
        consistent, so checksum verification passes. Only the signature - which
        covers the manifest the attacker had to change - detects it.
        """
        _, public = keys

        # Rebuild the bundle around malicious content, with a matching manifest,
        # but keep the original signature (the attacker cannot produce a new one).
        with BundleReader.open(signed_bundle) as reader:
            original_signature = reader.read_header().signature
        assert original_signature is not None

        malicious_payload = tmp_path / "malicious.safetensors"
        malicious_payload.write_bytes(MALICIOUS)
        forged = tmp_path / "forged.offlineai"
        with BundleWriter(forged) as writer:
            writer.write_header(
                manifest=manifest_for(MALICIOUS),
                package_yaml=b"kind: Package\n",
                signature=original_signature,
            )
            writer.add_artifact("artifacts/models/m/model.safetensors", malicious_payload)

        # Checksums pass: the forged bundle agrees with itself.
        from offlineai.bundler.verifier import verify_bundle

        assert verify_bundle(forged).verified is True

        # The signature does not.
        with BundleReader.open(forged) as reader:
            header = reader.read_header()
        envelope = SignatureEnvelope.from_bytes(header.signature or b"")
        with pytest.raises(SignatureError, match="does not match"):
            verify_manifest(header.manifest, envelope, trusted_key=load_public_key(public))

    def test_swapping_in_an_attackers_signature_is_caught_by_the_trusted_key(
        self, tmp_path: Path, signed_bundle: Path, keys: tuple[Path, Path]
    ) -> None:
        """An attacker can always re-sign with their own key. Supplying the
        expected public key is what makes that detectable."""
        _, genuine_public = keys
        attacker_private, _ = generate_keypair(tmp_path / "attacker.pem")

        with BundleReader.open(signed_bundle) as reader:
            manifest = reader.read_header().manifest
        forged_envelope = sign_manifest(manifest, load_private_key(attacker_private))

        # Without a trusted key this passes - the bundle is self-consistent.
        verify_manifest(manifest, forged_envelope)

        # With one, it does not.
        with pytest.raises(SignatureError, match="different key"):
            verify_manifest(manifest, forged_envelope, trusted_key=load_public_key(genuine_public))


class TestEnvelopeHandling:
    def test_a_corrupt_envelope_is_rejected(self) -> None:
        with pytest.raises(SignatureError, match="not readable"):
            SignatureEnvelope.from_bytes(b"{not json")

    def test_an_unknown_format_version_is_rejected(self) -> None:
        with pytest.raises(SignatureError, match="format"):
            SignatureEnvelope.from_bytes(b'{"format": "something-else/v9"}')

    def test_a_truncated_envelope_names_the_missing_field(self) -> None:
        with pytest.raises(SignatureError, match="algorithm"):
            SignatureEnvelope.from_bytes(b'{"format": "offlineai-sig/v1"}')

    def test_signature_errors_exit_seven(self) -> None:
        assert SignatureError("x").exit_code == ExitCode.SIGNATURE_FAILURE


class TestDetachedSignatures:
    def test_round_trips(self, tmp_path: Path, keys: tuple[Path, Path]) -> None:
        private, _ = keys
        bundle = tmp_path / "b.offlineai"
        bundle.write_bytes(b"pretend bundle")
        envelope = sign_manifest(manifest_for(GENUINE), load_private_key(private))

        path = write_detached(bundle, envelope)
        assert path.name.endswith(".offlineai.sig")
        assert read_detached(bundle) == envelope

    def test_absent_detached_signature_is_none_not_an_error(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b.offlineai"
        bundle.write_bytes(b"x")
        assert read_detached(bundle) is None

    def test_a_detached_signature_verifies_the_same_way(
        self, tmp_path: Path, keys: tuple[Path, Path]
    ) -> None:
        private, public = keys
        manifest = manifest_for(GENUINE)
        bundle = tmp_path / "b.offlineai"
        bundle.write_bytes(b"x")
        write_detached(bundle, sign_manifest(manifest, load_private_key(private)))

        envelope = read_detached(bundle)
        assert envelope is not None
        verify_manifest(manifest, envelope, trusted_key=load_public_key(public))


class TestKeyManagement:
    def test_the_private_key_is_not_world_readable(self, tmp_path: Path) -> None:
        private, _ = generate_keypair(tmp_path / "k.pem")
        mode = private.stat().st_mode & 0o777
        assert mode == 0o600, f"private key mode is {oct(mode)}"

    def test_refuses_to_overwrite_an_existing_key(self, tmp_path: Path) -> None:
        path = tmp_path / "k.pem"
        generate_keypair(path)
        with pytest.raises(SignatureError, match="already exists"):
            generate_keypair(path)

    def test_a_non_ed25519_key_is_rejected(self, tmp_path: Path) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path = tmp_path / "rsa.pem"
        path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        with pytest.raises(SignatureError, match="Ed25519"):
            load_private_key(path)

    def test_a_missing_key_says_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(SignatureError) as excinfo:
            load_private_key(tmp_path / "nope.pem")
        assert "offlineai keygen" in excinfo.value.render()

    def test_fingerprints_are_stable_and_short(self, tmp_path: Path) -> None:
        _, public = generate_keypair(tmp_path / "k.pem")
        key = load_public_key(public)
        assert fingerprint(key) == fingerprint(key)
        assert len(fingerprint(key)) == 19  # four 4-char groups plus separators

    def test_different_keys_have_different_fingerprints(self, tmp_path: Path) -> None:
        _, a = generate_keypair(tmp_path / "a.pem")
        _, b = generate_keypair(tmp_path / "b.pem")
        assert fingerprint(load_public_key(a)) != fingerprint(load_public_key(b))


class TestSignatureStability:
    def test_reformatting_the_manifest_does_not_break_the_signature(
        self, tmp_path: Path, keys: tuple[Path, Path]
    ) -> None:
        """Canonical JSON is signed, not the stored YAML bytes."""
        private, public = keys
        manifest = manifest_for(GENUINE)
        envelope = sign_manifest(manifest, load_private_key(private))

        # Round-trip through YAML, which may reorder or re-wrap.
        reparsed = Manifest.from_yaml(manifest.to_yaml())
        verify_manifest(reparsed, envelope, trusted_key=load_public_key(public))

    def test_changing_any_artifact_hash_breaks_the_signature(
        self, tmp_path: Path, keys: tuple[Path, Path]
    ) -> None:
        private, public = keys
        envelope = sign_manifest(manifest_for(GENUINE), load_private_key(private))
        with pytest.raises(SignatureError):
            verify_manifest(manifest_for(MALICIOUS), envelope, trusted_key=load_public_key(public))


def _members(path: Path) -> list[str]:
    with tarfile.open(path, "r") as archive:
        return archive.getnames()


class TestBundleLayout:
    def test_the_signature_precedes_the_artifacts(self, signed_bundle: Path) -> None:
        names = _members(signed_bundle)
        first_artifact = next(i for i, n in enumerate(names) if n.startswith("artifacts/"))
        assert names.index(layout.SIGNATURE) < first_artifact
