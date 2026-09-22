"""Bundle signing with Ed25519 (section 12).

**What is signed.** The canonical serialisation of ``manifest.yaml`` - nothing
else. The manifest records a SHA-256 for every artifact, so a signature over it
transitively covers the whole payload. Two consequences follow, and both are
the reason for this design: verification is a header-only operation that stays
instant on a 62 GB bundle, and reformatting the manifest cannot invalidate a
signature while key reordering cannot forge one.

**Where the signature lives.** Two placements, and deliberately not a third:

* ``build --sign-key`` writes it into the bundle header as it is created. Free.
* ``sign`` writes a detached ``<bundle>.offlineai.sig`` next to the file. Free.

What is *not* supported is rewriting an existing bundle to insert a signature
member. Copying 62 GB to add 64 bytes is not a reasonable operation, and
offering it would invite someone to do it.

Section 12 also says external signing infrastructure must not be mandatory, so
keys are plain local files and nothing here contacts a service. The format is
compatible with future Sigstore support without requiring it.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from offlineai.errors import SignatureError
from offlineai.logging import get_logger
from offlineai.schema.manifest import Manifest
from offlineai.utils.fs import atomic_write_bytes, atomic_write_text

__all__ = [
    "DETACHED_SUFFIX",
    "SIGNATURE_FORMAT",
    "SignatureEnvelope",
    "generate_keypair",
    "load_private_key",
    "load_public_key",
    "sign_manifest",
    "verify_manifest",
]

logger = get_logger("security.signing")

#: Versioned so the envelope can evolve - a different algorithm, or a Sigstore
#: bundle - without a verifier misreading an old signature as a new one.
SIGNATURE_FORMAT = "offlineai-sig/v1"

DETACHED_SUFFIX = ".sig"


@dataclass(frozen=True, slots=True)
class SignatureEnvelope:
    """A signature plus what a verifier needs to check it."""

    format: str
    algorithm: str
    signature: str
    public_key: str
    manifest_sha256: str
    signed_at: str
    signer: str | None = None

    def to_bytes(self) -> bytes:
        payload = {
            "format": self.format,
            "algorithm": self.algorithm,
            "signature": self.signature,
            "publicKey": self.public_key,
            "manifestSha256": self.manifest_sha256,
            "signedAt": self.signed_at,
        }
        if self.signer:
            payload["signer"] = self.signer
        return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"

    @classmethod
    def from_bytes(cls, raw: bytes) -> SignatureEnvelope:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SignatureError(
                "the signature is not readable",
                details={"Detail": str(exc)},
                action="The bundle's signature is corrupt. Re-transfer it.",
            ) from exc

        fmt = data.get("format")
        if fmt != SIGNATURE_FORMAT:
            raise SignatureError(
                f"unsupported signature format {fmt!r}",
                details={"Expected": SIGNATURE_FORMAT},
                action="This signature was produced by a different version of "
                "OfflineAI. Upgrade, or re-sign the bundle.",
            )
        try:
            return cls(
                format=fmt,
                algorithm=data["algorithm"],
                signature=data["signature"],
                public_key=data["publicKey"],
                manifest_sha256=data["manifestSha256"],
                signed_at=data["signedAt"],
                signer=data.get("signer"),
            )
        except KeyError as exc:
            raise SignatureError(f"the signature is missing the {exc.args[0]!r} field") from exc


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def generate_keypair(
    private_path: Path | str, *, public_path: Path | str | None = None
) -> tuple[Path, Path]:
    """Create an Ed25519 keypair. Returns ``(private, public)`` paths.

    The private key is written unencrypted but with 0600 permissions, because
    an air-gapped signing workflow usually has no way to prompt for a
    passphrase. Protecting the file is the operator's job, and the mode makes
    the expectation explicit.
    """
    private_path = Path(private_path)
    public_path = Path(public_path) if public_path else private_path.with_suffix(".pub")

    if private_path.exists():
        raise SignatureError(
            f"{private_path} already exists",
            action="Refusing to overwrite an existing key. Choose another path, or "
            "remove the old key deliberately.",
        )

    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    atomic_write_bytes(private_path, private_bytes)
    private_path.chmod(0o600)
    atomic_write_bytes(public_path, public_bytes)
    return private_path, public_path


def load_private_key(path: Path | str) -> Ed25519PrivateKey:
    path = Path(path)
    if not path.is_file():
        raise SignatureError(
            f"{path} does not exist",
            action="Generate one with:\n  offlineai keygen --out signing-key.pem",
        )
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except Exception as exc:
        raise SignatureError(
            f"{path} is not a readable unencrypted PEM private key",
            details={"Detail": type(exc).__name__},
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError(
            f"{path} is not an Ed25519 key",
            details={"Found": type(key).__name__},
            action="OfflineAI signs with Ed25519. Generate one with 'offlineai keygen'.",
        )
    return key


def load_public_key(source: Path | str | bytes) -> Ed25519PublicKey:
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    try:
        key = serialization.load_pem_public_key(raw)
    except Exception as exc:
        raise SignatureError(
            "the public key is not a readable PEM key",
            details={"Detail": type(exc).__name__},
        ) from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SignatureError("the public key is not an Ed25519 key")
    return key


# ---------------------------------------------------------------------------
# Sign and verify
# ---------------------------------------------------------------------------


def sign_manifest(
    manifest: Manifest, key: Ed25519PrivateKey, *, signer: str | None = None
) -> SignatureEnvelope:
    """Sign a manifest's canonical form."""
    from offlineai.utils.hashing import sha256_bytes

    payload = manifest.canonical_bytes()
    signature = key.sign(payload)
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return SignatureEnvelope(
        format=SIGNATURE_FORMAT,
        algorithm="ed25519",
        signature=base64.b64encode(signature).decode("ascii"),
        public_key=base64.b64encode(public_bytes).decode("ascii"),
        manifest_sha256=sha256_bytes(payload),
        signed_at=datetime.now(UTC).isoformat(),
        signer=signer,
    )


def verify_manifest(
    manifest: Manifest,
    envelope: SignatureEnvelope,
    *,
    trusted_key: Ed25519PublicKey | None = None,
) -> Ed25519PublicKey:
    """Verify a signature over ``manifest``. Returns the verifying key.

    ``trusted_key`` is what makes this meaningful. Without it, verification
    only proves the bundle is internally consistent - it was signed by
    *whoever* holds the embedded key, which an attacker who rewrote the bundle
    also satisfies. Passing a key the operator obtained separately is what
    turns this into proof of origin, and the CLI says so when none is given.
    """
    embedded = load_public_key(base64.b64decode(envelope.public_key))
    key = trusted_key or embedded

    if trusted_key is not None:
        expected = trusted_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        actual = embedded.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if expected != actual:
            raise SignatureError(
                "the bundle was signed by a different key than the one supplied",
                action="This bundle did not come from the expected signer. Do not "
                "install it without establishing where it came from.",
            )

    payload = manifest.canonical_bytes()
    try:
        key.verify(base64.b64decode(envelope.signature), payload)
    except InvalidSignature as exc:
        raise SignatureError(
            "the signature does not match the bundle manifest",
            details={"Signed at": envelope.signed_at},
            action="The manifest has been altered since it was signed, or the "
            "signature belongs to a different bundle. Do not install it.",
        ) from exc
    except Exception as exc:
        raise SignatureError(
            "the signature could not be verified",
            details={"Detail": type(exc).__name__},
        ) from exc

    return key


def write_detached(bundle: Path | str, envelope: SignatureEnvelope) -> Path:
    """Write ``<bundle>.sig`` beside the bundle."""
    path = Path(str(bundle) + DETACHED_SUFFIX)
    atomic_write_text(path, envelope.to_bytes().decode("utf-8"))
    return path


def read_detached(bundle: Path | str) -> SignatureEnvelope | None:
    path = Path(str(bundle) + DETACHED_SUFFIX)
    if not path.is_file():
        return None
    return SignatureEnvelope.from_bytes(path.read_bytes())


def fingerprint(key: Ed25519PublicKey) -> str:
    """Short, comparable identifier for a public key.

    Operators compare these by eye when confirming a bundle came from the
    expected signer, so it is short enough to read aloud.
    """
    from offlineai.utils.hashing import sha256_bytes

    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    digest = sha256_bytes(raw)
    return ":".join(digest[i : i + 4] for i in range(0, 16, 4))
