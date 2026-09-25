# Package format

`offlineai.yaml` describes a workload: what it is made of, what it needs, and
how it runs.

The schema is **strict**. Unknown keys are errors, everywhere. That looks
pedantic and is not: a typo like `containters:` that validates silently
produces a bundle with no image in it, and the operator discovers that on the
air-gapped side, where they cannot simply rebuild.

## Minimal

```yaml
apiVersion: offlineai/v1
kind: Package
metadata:
  name: my-app
  version: 1.0.0
```

## Complete

```yaml
apiVersion: offlineai/v1       # exact match; the format is versioned
kind: Package

metadata:
  name: qwen-vllm              # lowercase, digits, hyphens; becomes a filename
  version: 1.0.0
  description: Qwen served by vLLM
  license: Apache-2.0

runtime:
  type: docker

architecture: [amd64]          # amd64 | arm64
os: ["ubuntu:24.04"]           # informational

models:
  - name: model
    source:
      type: huggingface        # huggingface | http | local
      repo: Qwen/Qwen3-30B
      revision: main
      exclude: ["*.bin"]       # see model-packaging.md
    destination: /models/model # where it is mounted in the container

containers:
  - name: vllm
    image: vllm/vllm-openai:v0.6.3
    platform: linux/amd64
    digest: sha256:...         # optional but recommended
    dockerfile: Dockerfile     # build instead of pull
    context: .

python:
  requirements: [requirements.txt]
  version: "3.12"              # the TARGET's interpreter, not the builder's
  platform: linux
  architecture: amd64

system:
  packages: [curl, ca-certificates]   # NOT YET SUPPORTED - see below
  recommended_packages: []
  optional_packages: []

environment:
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"

volumes:
  - host: ./data
    container: /data
    read_only: false

services:
  - name: vllm
    container: vllm            # must name a declared container
    command: [--model, /models/model]
    ports: ["8000:8000"]       # HOST:CONTAINER
    environment: {}
    depends_on: []             # start order

hardware:
  gpu:
    required: true
    vendor: nvidia
    minimum_vram_gb: 48        # per device, not summed
    minimum_driver: "550"
    count: 1
  minimum_ram_gb: 64
  minimum_disk_gb: 200
  minimum_cpu_cores: 8

secrets:
  external: [HUGGINGFACE_TOKEN]   # names only; never values

install:
  healthcheck:
    command: [curl, --fail, "http://localhost:8000/health"]
    retries: 60
    interval_seconds: 10
```

## Field notes

### `metadata.name`

Becomes part of a filename (`<name>-<version>.offlineai`) and a registry key,
so it must be lowercase alphanumeric with hyphens, starting and ending
alphanumeric. A name containing a path separator would write outside the
output directory, and is rejected.

### `containers[].dockerfile`

With a `dockerfile`, the declared `image` is the **base**. The builder builds
a new image tagged `offlineai/<package>-<container>:<version>` and packages
*that*. Install starts the built image, not the base.

This catches people out: without it, install starts `python:3.12-slim`, whose
entrypoint is a REPL, and the container exits immediately.

### `python.version`

The interpreter on the **target**, which is usually the one inside the
container image — not the Python running the build. Wheels are resolved
explicitly for it. A macOS builder resolving for itself would produce a bundle
of macOS wheels that cannot install on the Linux host they were meant for.

### `hardware.gpu.minimum_vram_gb`

Compared **per device**, never summed. Two 24 GB cards do not satisfy a 48 GB
requirement unless the workload shards, and whether it does is an application
decision OfflineAI must not assume.

Values are binary GB, matching how cards are sold. An "80GB" H100 reports
81559 MiB, of which some is reserved; the comparison rounds to the advertised
capacity so an 80 GB card satisfies an 80 GB requirement.

### `system.packages`

**Not implemented in this release, and declaring it fails the build.**

Section 19 is a real requirement and the field stays in the schema, but OS
package resolution does not exist yet. A build that quietly omitted them would
produce a bundle that verifies, reports success and is missing something it
declared — discovered on the air-gapped side, where it cannot be fixed. That
is precisely what section 74 forbids, so the build refuses instead:

```
Reason:
this build cannot package the 2 required system package(s) this package declares

Declared:
curl, ca-certificates
```

For a containerised workload the right home for an OS dependency is the image
anyway:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
```

The builder packages the built image, so the dependency travels with it.

`optional_packages` and `recommended_packages` are advisory by definition, so
they warn rather than fail.

### `secrets.external`

A list of **environment variable names**. The bundle carries the name so the
installer can report `HUGGINGFACE_TOKEN: REQUIRED`. It never carries a value,
and the schema rejects an entry that tries to supply one.

### `services[].depends_on`

Determines start order. Cycles are detected and reported; services then start
in declaration order rather than deadlocking.

## offlineai.lock

A successful build writes `offlineai.lock` next to `offlineai.yaml`. **Commit
it.** The manifest describes a bundle that *was* built; the lock pins what a
*future* build resolves to.

```yaml
formatVersion: "1"
package: {name: qwen-vllm, version: 1.0.0}
sources:
  - name: model
    kind: huggingface
    locator: Qwen/Qwen3-30B
    pin: 5a7c1b4e9f2d8c3a1b0e7f6d5c4b3a2918273645   # the commit `main` resolved to
  - name: vllm
    kind: oci
    locator: vllm/vllm-openai:v0.6.3
    pin: sha256:abc123...                          # the image digest
artifacts:
  - path: artifacts/models/model/config.json
    type: model
    sha256: ...
```

`sources` is the half that matters. Those pins are fed **back into** the next
resolution, so `revision: main` fetches the recorded commit rather than
wherever the branch has moved to, and a re-tagged `vllm:v0.6.3` is pulled by
digest rather than by tag. A lock that were only checked afterwards would tell
you a build drifted; one whose pins constrain resolution stops it drifting.

| | |
|---|---|
| *(default)* | apply the pins, then refresh the lock |
| `--locked` | apply the pins and fail on any drift. Never writes. Use in CI. |
| `--update-lock` | ignore the pins, re-resolve, write the result |
| `--no-lock` | neither read nor write |

A `--locked` build that has drifted fails **before** the archive is written,
and names what changed:

```
Drift:
artifacts/models/demo/config.json: digest changed, 7ea87318… -> 1be08cac…
```

Container images are compared on their registry digest rather than the bytes
of the tar. `docker save` is not byte-reproducible — saving the same image by
tag and by digest gives different archives — so comparing tar bytes would
report drift on an image that never changed.

## Target profiles

A profile is the other YAML you write: a description of a machine you might
deploy to (specification section 5.6).

```yaml
# h100-server.yaml
name: h100-server

os:
  family: ubuntu          # any Linux distribution; mapped to the platform
  version: "24.04"

architecture: amd64       # amd64 | arm64

gpu:
  vendor: nvidia
  model: NVIDIA H100 80GB HBM3
  memory_gb: 80           # per device, as the card is sold
  minimum_driver: "550"
  count: 8

runtime:
  docker: ">=27"

memory_gb: 1024
cpu_cores: 112
```

```bash
offlineai check qwen-vllm-1.0.0.offlineai --profile h100-server.yaml
```

Without `--profile`, `check` validates the machine it is running on. On a
builder that is the wrong machine — the target is air-gapped and elsewhere — so
the only answer you can get is about hardware nobody is deploying to.

### Anything omitted is SKIPPED, never satisfied

A profile that says nothing about RAM does not approve a bundle needing 512 GB:

```
RAM:  SKIPPED  not stated by profile 'h100-server'
```

This is what makes the feature safe rather than merely convenient. Inventing a
default would let a profile quietly approve a bundle for hardware it was never
checked against — worse than having no profile at all.

The one exception is the GPU: a profile that describes no GPU is describing a
machine without one, so a bundle requiring one **fails** rather than skipping.

### Prefer a captured profile to a written one

```bash
# on the air-gapped target
offlineai doctor --save-profile h100-server.yaml --profile-name rack-07
```

Carry that file back across the gap. The builder is then checking against
measured hardware rather than somebody's recollection of it. A captured profile
gives the same verdict as checking the live host, which the test suite asserts.

Note what a profile check does and does not establish: it narrows the question
from "will this run?" to "will this run on a machine that looks like this?".
Nothing here verifies the target matches its own description.

See `examples/profiles/h100-server.yaml`.

## Validating without building

```bash
offlineai build . --dry-run
```

Validates the definition, checks that every referenced file exists, and reports
what a real build would fetch — without fetching anything. This is how a 62 GB
package stays reviewable on a laptop.
