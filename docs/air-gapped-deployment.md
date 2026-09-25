# Air-gapped deployment

The operational guide: what to do, in what order, and what to check.

## The two environments

| | Builder | Target |
|---|---|---|
| Network | Yes | None |
| Install | `pip install 'offlineai[builder]'` | `pip install offlineai` (core only) |
| Does | resolve, build, sign | verify, import, install, run |

The target never depends on the builder. If something is missing at install
time, the bundle was built wrong, and that is a builder-side problem.

## Procedure

### 1. On the builder — prepare

```bash
offlineai init
offlineai network-check ./my-package     # before building, not after
```

`network-check` reads the definition and reports external references split
into build-time and runtime. A runtime one will fail on the target, and
learning that here costs seconds instead of a transfer cycle.

### 1b. Check against the target, not against yourself

`offlineai check` normally validates the machine it runs on. On a builder that
is the wrong machine — the target is air-gapped and elsewhere.

```bash
offlineai check my-package-1.0.0.offlineai --profile h100-server.yaml
```

```
Evaluated against profile: h100-server   (NOT this host)

CPU architecture:  OK  amd64
Operating system:  OK  linux ubuntu 24.04
RAM:               OK  requires 512 GB, profile 'h100-server' has 1024.0 GB
GPU VRAM:          OK  requires 80 GB per device, largest device has 80 GB

Result: COMPATIBLE
```

Anything the profile does not state is reported `SKIPPED`, never satisfied —
a profile silent about RAM must not approve a bundle that needs 512 GB.

**Prefer a captured profile over a written one.** On the target:

```bash
offlineai doctor --save-profile h100-server.yaml --profile-name rack-07
```

Carry that small file back across the gap. The builder is then checking
against measured ground truth rather than somebody's recollection of the
hardware. See `examples/profiles/h100-server.yaml`.

A profile check does not verify that the target matches its own description.
It narrows the question from "will this run?" to "will this run on a machine
that looks like this?", which is as far as anything can go without being there.

### 2. Build and sign

```bash
offlineai build ./my-package \
  --sign-key signing-key.pem \
  --signer "platform-team@example"
```

Record the SHA-256 that `build` prints. It is how you confirm the transfer.

### 3. Transfer

Move the `.offlineai` file by whatever your policy permits. If you signed
detached rather than at build time, carry the `.sig` alongside.

Carry the **public** key by a different route than the bundle. A signature and
a key that travelled together prove nothing about origin — an attacker who
could replace one could replace both.

### 4. On the target — check before trusting

```bash
sha256sum my-package-1.0.0.offlineai       # compare with the builder's output

offlineai verify-signature my-package-1.0.0.offlineai --key signer.pub
offlineai inspect my-package-1.0.0.offlineai
offlineai check   my-package-1.0.0.offlineai
```

All three read only the bundle header and are instant at any size. `check`
reports whether this host qualifies and what installing will cost in disk.

### 5. Verify in full

```bash
offlineai verify my-package-1.0.0.offlineai
```

Streams the whole archive and hashes every artifact. Slower — it is bounded by
disk read speed — and it is the step that proves the payload arrived intact.

### 6. Import and install

```bash
offlineai import  my-package-1.0.0.offlineai
offlineai install my-package --strict-offline
```

Use `--strict-offline` on a target. Always. It makes the guarantee structural:
sockets are blocked in-process, pip and Hugging Face are pointed at nothing,
proxies are cleared, and containers run `--pull never`.

### 7. Confirm

```bash
offlineai status my-package
offlineai logs   my-package
```

`status` reports the ports the containers are **actually** bound to, which is
what you should curl — not necessarily what the package declared, if you used
an override.

## Adapting to the site

Never edit a bundle. It would invalidate its checksums and its signature, and
those are the only reasons to trust it.

```yaml
# site-a.yaml
runtime:
  gpu:
    device_ids: ["0", "1"]
services:
  vllm:
    ports: ["8001:8000"]
environment:
  VLLM_MAX_MODEL_LEN: "32768"
```

```bash
offlineai install my-package --config site-a.yaml
```

The bundle stays byte-identical and still verifies afterwards.

## Disk

Bundles are large; plan before transferring, not after.

```bash
offlineai check my-package-1.0.0.offlineai
```

```
Installation Storage Estimate

  Bundle                    62.0 GB
  Artifact storage          62.0 GB
  Container storage         15.7 GB
  Model storage                 0 B
  Temporary                  620 MB
  --------------------------------
  Recommended free          78.3 GB
  Available                180.0 GB
```

Note there is no extraction line. Import streams into content-addressed
storage rather than extracting, and models are hard linked into place, so two
of the terms a naive estimate would include are genuinely zero here.

If the target's system volume is small, put the data directory elsewhere:

```bash
offlineai --data-dir /mnt/big import my-package-1.0.0.offlineai
```

Or set it permanently in `~/.offlineai/config.yaml`.

## Multiple versions

The registry holds several versions of a package at once. Shared artifacts are
stored once, and removing one version never deletes content another still
references — that is what the refcount is for.

```bash
offlineai list
offlineai install my-package --version 1.1.0
offlineai remove  my-package --version 1.0.0
```

## When an install fails

```bash
offlineai logs my-package
offlineai rollback install-20260922-001
```

Rollback replays each step's recorded inverse in reverse order. Images are kept
by default, because loading a 12 GB image is expensive and keeping it is
harmless; pass `--remove-images` if you want them gone.

Data you own is never deleted. Host directories declared as volumes are not
touched.

## Comparing releases

Before promoting a new version:

```bash
offlineai diff my-package-1.0.0.offlineai my-package-1.1.0.offlineai
```

Reads only the manifests, so it is instant. It reports requirement changes
separately from artifact changes — a bundle that quietly starts demanding
80 GB of VRAM instead of 48, or a new required secret, is deployment-blocking
and would not appear as an artifact difference at all.

## Checklist

- [ ] `network-check` clean on the builder
- [ ] `check --profile` passes against the target's captured profile
- [ ] Built with `--sign-key`
- [ ] SHA-256 recorded and compared after transfer
- [ ] Public key carried by a different route
- [ ] `verify-signature --key` passes on the target
- [ ] `verify` passes on the target
- [ ] `check` reports COMPATIBLE and enough disk
- [ ] Installed with `--strict-offline`
- [ ] `status` shows RUNNING and the endpoint responds
