"""``keygen``, ``sign``, ``verify-signature``, ``sbom`` and ``network-check``.

Sections 12, 33, 36 and 37.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from offlineai.bundler.archive import BundleReader
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.errors import SignatureError
from offlineai.sbom.cyclonedx import LICENSE_DISCLAIMER, generate_sbom, license_report
from offlineai.schema.package import Package
from offlineai.security.network import Phase, audit_package
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


def register(app: typer.Typer) -> None:
    _register(app, "keygen", keygen)
    _register(app, "sign", sign)
    _register(app, "verify-signature", verify_signature)
    _register(app, "sbom", sbom)
    _register(app, "network-check", network_check)


def keygen(
    ctx: typer.Context,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the private key.")
    ] = Path("offlineai-signing-key.pem"),
) -> None:
    """Generate an Ed25519 signing keypair.

    The private key is written with 0600 permissions and no passphrase, because
    an air-gapped signing workflow usually cannot prompt for one. Protecting
    that file is your responsibility.
    """
    context: Context = ctx.obj
    private, public = generate_keypair(out)
    key_id = fingerprint(load_public_key(public))

    context.output.emit_raw(
        {"private_key": str(private), "public_key": str(public), "fingerprint": key_id}
    )
    context.output.line(f"Private key: {private}  (mode 0600)")
    context.output.line(f"Public key:  {public}")
    context.output.line(f"Fingerprint: {key_id}")
    context.output.line()
    context.output.line("Distribute the PUBLIC key to anyone who must verify your bundles.")
    context.output.line("Never copy the private key onto an air-gapped target.")


def sign(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Bundle to sign.")],
    key: Annotated[Path, typer.Option("--key", "-k", help="Ed25519 private key.")],
    signer: Annotated[
        str | None, typer.Option("--signer", help="Identity recorded in the signature.")
    ] = None,
) -> None:
    """Sign a bundle, writing a detached <bundle>.sig beside it.

    Detached rather than embedded because inserting a member into an existing
    archive means rewriting it, and copying 62 GB to add 64 bytes is not a
    reasonable operation. To embed a signature instead, sign at build time:

      offlineai build . --sign-key signing-key.pem
    """
    context: Context = ctx.obj
    manifest = BundleReader.peek_manifest(bundle)
    envelope = sign_manifest(manifest, load_private_key(key), signer=signer)
    path = write_detached(bundle, envelope)

    context.output.emit_raw(
        {
            "bundle": str(bundle),
            "signature": str(path),
            "fingerprint": fingerprint(load_public_key(base64.b64decode(envelope.public_key))),
            "signed_at": envelope.signed_at,
        }
    )
    context.output.line(f"Signed {bundle.name}")
    context.output.line(f"Signature: {path}")
    context.output.line()
    context.output.line("Transfer the .sig file alongside the bundle.")


def verify_signature(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Bundle to check.")],
    key: Annotated[
        Path | None,
        typer.Option("--key", "-k", help="Trusted public key to require."),
    ] = None,
) -> None:
    """Verify a bundle's signature.

    Reads only the bundle header, so this is instant at any size.

    Without --key this proves only that the bundle is internally consistent:
    it was signed by whoever holds the embedded key, which an attacker who
    rewrote the bundle also satisfies. Supply the signer's public key,
    obtained separately, to make it proof of origin.
    """
    context: Context = ctx.obj
    output = context.output

    with BundleReader.open(bundle) as reader:
        header = reader.read_header()
    envelope_bytes = header.signature or (
        detached.to_bytes() if (detached := read_detached(bundle)) else None
    )

    if envelope_bytes is None:
        raise SignatureError(
            f"{bundle.name} is not signed",
            action="Sign it with:\n  offlineai sign <bundle> --key signing-key.pem\n"
            "Or build it signed:\n  offlineai build . --sign-key signing-key.pem",
        )

    envelope = SignatureEnvelope.from_bytes(envelope_bytes)
    trusted = load_public_key(key) if key else None
    verified_key = verify_manifest(header.manifest, envelope, trusted_key=trusted)
    key_id = fingerprint(verified_key)

    output.emit_raw(
        {
            "bundle": str(bundle),
            "package": header.manifest.package.name,
            "version": header.manifest.package.version,
            "verified": True,
            "fingerprint": key_id,
            "signed_at": envelope.signed_at,
            "signer": envelope.signer,
            "trusted_key_supplied": key is not None,
            "embedded": header.signature is not None,
        }
    )

    output.line("OfflineAI Signature Verification")
    output.line()
    output.line(f"Package:     {header.manifest.package.name} {header.manifest.package.version}")
    output.line(f"Algorithm:   {envelope.algorithm}")
    output.line(f"Fingerprint: {key_id}")
    output.line(f"Signed at:   {envelope.signed_at}")
    if envelope.signer:
        output.line(f"Signer:      {envelope.signer}")
    output.line(f"Location:    {'embedded in bundle' if header.signature else 'detached .sig'}")
    output.line()
    output.line("Result: SIGNATURE VALID", style="bold green")
    if key is None:
        output.line()
        output.line(
            "No trusted key was supplied, so this proves the bundle is internally\n"
            "consistent, not where it came from. Re-run with --key <signer.pub>\n"
            "using a key you obtained separately to establish origin.",
            style="dim",
        )


def sbom(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Bundle to describe.")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write to a file instead of stdout.")
    ] = None,
    licenses: Annotated[
        bool, typer.Option("--licenses", help="Show the licence summary instead.")
    ] = False,
) -> None:
    """Emit a CycloneDX SBOM for a bundle.

    Reads only the header. If the bundle already carries an SBOM it is
    returned verbatim; otherwise one is generated from the manifest.
    """
    context: Context = ctx.obj
    with BundleReader.open(bundle) as reader:
        header = reader.read_header()

    if licenses:
        report = license_report(header.manifest)
        context.output.emit_raw(report)
        if not context.output.fmt.json:
            context.output.line("Component Licenses")
            context.output.line()
            width = max((len(str(e["component"])) for e in report["licenses"]), default=10) + 2
            for entry in report["licenses"]:
                value = entry["license"] or "(not recorded)"
                context.output.line(f"  {str(entry['component']).ljust(width)}{value}")
            context.output.line()
            context.output.line(LICENSE_DISCLAIMER, style="dim")
        return

    payload = json.loads(header.sbom) if header.sbom else generate_sbom(header.manifest)
    text = json.dumps(payload, indent=2, sort_keys=True)

    if out is not None:
        out.write_text(text + "\n")
        context.output.emit_raw(
            {"written": str(out), "components": len(payload.get("components", []))}
        )
        context.output.line(f"Wrote {out} ({len(payload.get('components', []))} components)")
        return

    # The SBOM is the payload here, so it goes to stdout in both modes.
    context.output.console.print_json(text)


def network_check(
    ctx: typer.Context,
    target: Annotated[
        Path,
        typer.Argument(help="A package directory, an offlineai.yaml, or a bundle."),
    ] = Path(),
) -> None:
    """Audit a package for external dependencies (section 33).

    Distinguishes build-time references, which are fine and expected, from
    runtime ones, which will fail on an air-gapped host.

    Exits nonzero if a possible runtime network dependency is found.
    """
    context: Context = ctx.obj
    output = context.output

    manifest = None
    if target.is_file() and target.suffix == ".offlineai":
        with BundleReader.open(target) as reader:
            header = reader.read_header()
        manifest = header.manifest
        package = Package.model_validate(yaml.safe_load(header.package_yaml))
    else:
        from offlineai.resolver.package import load_package

        package, _, _ = load_package(target)

    audit = audit_package(package, manifest)

    output.emit_raw(
        {
            "package": audit.package,
            "version": audit.version,
            "clean": audit.clean,
            "summary": audit.summary,
            "findings": [
                {
                    "url": f.url,
                    "phase": f.phase.value,
                    "location": f.location,
                    "detail": f.detail,
                }
                for f in audit.findings
            ],
        }
    )

    output.line("External Dependency Audit")
    output.line()
    output.line(f"Package: {audit.package} {audit.version}")

    for phase in (Phase.BUILD, Phase.RUNTIME, Phase.UNKNOWN, Phase.LOCAL):
        findings = audit.of_phase(phase)
        if not findings:
            continue
        output.line()
        output.line(f"{phase.value}:")
        for finding in findings:
            output.line(f"  {finding.url}")
            output.line(f"    at {finding.location}", style="dim")
            if finding.detail:
                output.line(f"    {finding.detail}", style="dim")

    output.line()
    if audit.clean:
        output.line("These references are BUILD-TIME dependencies.", style="dim")
        output.line(audit.summary, style="bold green")
    else:
        output.line(audit.summary, style="bold red")
        output.line()
        output.line(
            "A runtime dependency will fail on an air-gapped host. Either remove it,\n"
            "or point it at something inside the deployment."
        )
        raise typer.Exit(5)
