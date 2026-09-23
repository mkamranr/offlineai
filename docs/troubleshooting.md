# Troubleshooting

## Exit codes

Automation branches on these, so they do not change.

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | general error |
| 2 | invalid package |
| 3 | verification failure |
| 4 | hardware incompatibility |
| 5 | missing dependency |
| 6 | installation failure |
| 7 | signature failure |
| 8 | insufficient disk |
| 9 | runtime failure |

## Build

### `credential-shaped file(s) would be included in the bundle`

The build found something that looks like a secret — `.env`, `*.pem`, `id_rsa`.
It refuses rather than warning, because a private key inside a bundle that has
already crossed an air gap cannot be recalled.

Remove the file, or if it is legitimate (a test fixture key), list it in
`.offlineaiignore`.

### `<package> does not publish a wheel for linux/amd64 on Python 3.12`

An air-gapped host cannot build from source, so a package with no matching
wheel would make the bundle incomplete. Either pin a version that ships one,
or change `python.version` / `python.platform` to a combination that exists.

Exits 5 — missing dependency — not a generic error.

### `The builder's Python has no pip`

```bash
python -m ensurepip --upgrade
```

Only the builder needs pip. The target does not.

### `a container runtime is required to package container images`

Start Docker on the builder, or remove the `containers:` section.

### `the sharded checkpoint in '<repo>' is incomplete`

A weight shard is missing. Usually an over-narrow `include` pattern; sometimes
the repository itself. The error names which shard.

### The build seems to restart a download from zero

It should resume. Check that `~/.offlineai/cache/tmp` is writable and was not
cleared between runs. A partial file that failed its digest is deleted
deliberately, so that case does restart — and should.

## Verify and import

### `Result: FAILED` with a checksum mismatch

The bundle is corrupt or was tampered with. Re-transfer it. Do not install it.

Compare the SHA-256 against what `build` printed; if they differ, the transfer
is at fault.

### `the bundle archive is truncated or corrupt`

The copy did not finish, or ran out of space. Check the file size against the
source.

### `not enough free space to import this bundle`

Import needs room for the artifacts. Free space, or:

```bash
offlineai --data-dir /mnt/big import bundle.offlineai
```

### `verify` passes but I changed a byte

If you changed a byte in the tar's trailing padding, that is expected — those
bytes carry no content. Compare the file's SHA-256 for whole-file integrity.

## Install

### A service starts and immediately dies

Most often the `dockerfile` case: check what image is actually running.

```bash
docker inspect offlineai-<package>-<service> --format '{{.Config.Image}} {{.Config.Cmd}}'
```

If it shows the **base** image and not `offlineai/<package>-<service>:<version>`,
the bundle was built before the image was, or without a container runtime
available.

```bash
offlineai logs <package>
```

### `the workload did not become healthy`

The container started but the health check never passed.

```bash
offlineai logs <package>
offlineai rollback install-20260922-001
```

For large models, raise `install.healthcheck.retries`. Loading 30B parameters
from disk takes longer than a default timeout allows.

### `image <ref> is not present locally`

The bundle did not contain the image. Containers run `--pull never` by design,
so this is reported rather than silently fetched. Rebuild on a machine with a
container runtime.

### `this host does not satisfy the bundle's requirements`

```bash
offlineai check <bundle>
```

gives the full report. Exits 4.

### `installation attempted a DNS lookup ... strict-offline mode is enabled`

Something needed the network during a strict-offline install, which means the
bundle was built incomplete. The message names what was attempted. Rebuild with
the missing artifact included.

Exits 5.

### `container <name> already exists`

Should not happen — install removes its own leftovers first. If it does:

```bash
docker rm -f offlineai-<package>-<service>
```

## Runtime

### `status` says DEGRADED

Some services are running and some are not.

```bash
offlineai status <package> --json | jq '.services'
offlineai logs <package> --service <name>
```

`STOPPED` means none are running; `DEGRADED` genuinely means partial.

### The endpoint in `status` is not what I expected

`status` reports the ports the containers are **actually bound to**, which
differ from the declared ones if you used `--config`. The reported endpoint is
the correct one to use.

### GPU not visible in the container

```bash
offlineai doctor          # reports whether the runtime exposes GPUs
nvidia-smi                # reports whether the host sees them
```

Install the NVIDIA container toolkit on the target, then:

```bash
offlineai install <package> --gpus 0,1
```

## Signatures

### `the bundle was signed by a different key than the one supplied`

The bundle did not come from the expected signer. Do not install it without
establishing where it came from.

### `the signature does not match the bundle manifest`

The manifest was altered after signing. Do not install it.

### `<bundle> is not signed`

```bash
offlineai sign <bundle> --key signing-key.pem       # detached
offlineai build . --sign-key signing-key.pem        # embedded, at build time
```

## Diagnostics

```bash
offlineai doctor                       # is this host ready at all
offlineai check <bundle>               # is it ready for this bundle
offlineai inspect <bundle>             # what is in the bundle
offlineai graph <package>              # what it is made of and where from
offlineai network-check <package>      # will it reach out at run time
offlineai --log-level debug <command>  # verbose, secrets redacted
offlineai --json <command>             # machine-readable, for a bug report
```

`--log-level debug` is safe to share: log output is redacted, so tokens,
passwords and private keys do not appear.
