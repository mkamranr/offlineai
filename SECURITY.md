# Security

OfflineAI moves software across a trust boundary. A bundle is built somewhere
else, carried on removable media, and handed to a machine that is isolated
precisely because its operators do not trust what reaches it. Everything below
follows from treating a bundle as hostile input until proven otherwise.

## Reporting a vulnerability

Please report security issues privately rather than through a public issue.
Include the version, a description, and a reproduction if you have one.

## Threat model

| Threat | Mitigation |
|---|---|
| Tampered bundle | SHA-256 per artifact, recorded in the manifest; `verify` streams and checks every one |
| Manifest rewritten to match tampered content | Ed25519 signature over the manifest's canonical form |
| Bundle replaced wholesale | Signature plus a trusted public key obtained out of band |
| Path traversal in the archive | Member names validated and normalised before anything is written |
| Symlink or hardlink escape | Links refused by default; hard links always refused |
| Archive bomb | Member count, per-member and total size ceilings |
| Malicious permission bits | Archive modes masked; setuid, setgid and world-write dropped |
| Secrets packaged by accident | Build-time scan refuses credential-shaped files |
| Silent network access during install | `--strict-offline` enforced in-process and in the subprocess environment |
| Image substitution | Immutable digests recorded at build time and compared on load |
| Rollback abused to delete files | Closed inverse action set; path removal confined to the install tree |

## What verification does and does not prove

**`offlineai verify`** proves the bundle's content matches its own manifest.
It does not prove where the bundle came from. A tampered bundle whose manifest
was rewritten to match is internally consistent and passes.

**`offlineai verify-signature` without `--key`** proves the bundle was signed
by whoever holds the embedded key. An attacker who rewrote the bundle can
always re-sign with their own key, so this also does not prove origin. The CLI
says so.

**`offlineai verify-signature --key signer.pub`**, with a key obtained
separately from the bundle, proves origin. This is the only form that does.

The distinction is deliberate and is covered by tests: one constructs a
consistently-rewritten bundle, confirms `verify` reports VERIFIED, and confirms
the signature rejects it.

## Archive extraction

Extraction is stricter than `tarfile`'s own `data` filter:

- every member name is validated and normalised before any write;
- validation runs over the whole archive **first**, so a hostile member at the
  end cannot leave the earlier members on disk;
- symlinks are refused unless explicitly enabled, and hard links always;
- device nodes and FIFOs are always refused;
- permission bits are masked to `0755`, dropping setuid, setgid and
  world-write;
- after name validation, the resolved destination is re-checked to be inside
  the root, which catches escapes through a symlink that already existed.

`tests/security/test_safe_extraction.py` builds each attack archive inline, so
the threat being defended against is readable next to the defence.

## Secrets

Section 40 of the specification: never package secrets by default.

The build scans its context for credential-shaped filenames — `.env`, `*.pem`,
`*.key`, `id_rsa`, `credentials.json`, `.netrc` and others — and **refuses**
rather than warning. A private key inside a bundle that has already crossed an
air gap on removable media cannot be recovered by deleting the file afterwards.

Declare externally-supplied secrets by name:

```yaml
secrets:
  external:
    - HUGGINGFACE_TOKEN
    - DATABASE_PASSWORD
```

The bundle carries the name so the installer can tell the operator it is
required. It never carries a value; the schema rejects an entry that tries.

To allow a file that legitimately matches a pattern — a test fixture key —
list it in `.offlineaiignore`.

Tokens are read from the environment only, never from a config file, so a
credential cannot be written into a bundle by a misconfiguration.

## Strict offline

`--strict-offline` enforces in two layers, because one is not enough:

- **in-process**: `socket.socket` and DNS resolution raise, catching our own
  code reaching out;
- **in the environment**: children get `PIP_NO_INDEX`, `HF_HUB_OFFLINE`,
  `TRANSFORMERS_OFFLINE` and cleared proxy variables, and containers run with
  `--pull never`. Patching sockets in this process does nothing to a
  subprocess, and pip and docker are subprocesses.

Proxy variables are cleared specifically so they cannot quietly restore the
egress that was just removed.

A violation exits `5` — *missing dependency* — not a generic error, because a
strict-offline install that needs the network means the bundle was built
incomplete, and that is what the operator must act on.

This does not make egress impossible; a determined workload can open its own
socket. It makes an accidental one impossible to miss.

## Signing keys

`offlineai keygen` writes an unencrypted private key with mode `0600`. An
air-gapped signing workflow usually has no way to prompt for a passphrase;
protecting the file is the operator's responsibility, and the mode makes the
expectation explicit. Generating over an existing key is refused.

Never copy a private key onto an air-gapped target. The target verifies; it
never signs.

## Known limitations

Stated plainly, because a security document that only lists strengths is not
useful.

- **Trailing archive padding is not covered by verification.** Those bytes
  carry no content, so no artifact and no manifest field changes. Compare the
  SHA-256 that `build` prints if you need whole-file integrity.
- **Signing is not mandatory.** Section 12 requires that external signing
  infrastructure not be a prerequisite, so `security.require_signature` is off
  by default. Turn it on for production.
- **We do not scan model weights or container images for malicious content.**
  A signature proves who shipped a bundle, not that its contents are safe. Run
  your own scanners on the connected side.
- **Secret detection is by filename, not content.** Content scanning produces
  false positives on model weights and is expensive on a 62 GB tree.
- **The installer does not sandbox what it starts.** Containers run with the
  privileges the package declares. Review a package definition before
  installing it, exactly as you would a compose file.
