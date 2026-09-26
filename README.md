<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo.svg" alt="OfflineAI - build online, transfer once, run offline" width="340">
</picture>

**Package complete AI workloads for deployment into air-gapped environments.**

<!--
  The CI badge uses GitHub's own endpoint, which requires authentication while
  this repository is private: it renders for anyone with access, and for
  everyone the moment the repository is made public. The other badges are
  static and always render.
-->

[![CI](https://github.com/mkamranr/offlineai/actions/workflows/ci.yml/badge.svg)](https://github.com/mkamranr/offlineai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-1B4965.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-1B4965.svg)](pyproject.toml)
![Target](https://img.shields.io/badge/target-Linux-1B4965.svg)

</div>

Modern AI deployments are hard to reproduce without a network. A single inference stack can
need model weights, OCI images, a transitive Python wheel closure, OS packages, CUDA
runtime dependencies, tokenizers, configuration and startup scripts. Moving that into an
isolated environment usually means ferrying pieces by hand and hoping nothing reaches for
the internet at install time.

OfflineAI turns that into one artifact and one command.

## How it works

On an internet-connected builder:

```bash
offlineai build .
```

produces a single self-describing bundle:

```
my-ai-app-1.0.0.offlineai
```

containing model weights, Docker images, Python wheels, configuration,
checksums, an SBOM, license metadata, startup scripts and documentation.

It also writes **`offlineai.lock`** beside your package definition. Commit it:
it records the commit each model revision resolved to and the digest behind
each image tag, so a rebuild six months from now produces the same bundle
rather than whatever those tags point at by then.

```bash
offlineai build . --locked     # fail if anything has drifted. Use this in CI.
```

Transfer it by USB or any approved one-way channel. Then, on a machine with **no network
access at all**:

```bash
offlineai verify  my-ai-app-1.0.0.offlineai
offlineai import  my-ai-app-1.0.0.offlineai
offlineai install my-ai-app
offlineai status  my-ai-app
```

And before you carry 62 GB anywhere, ask whether the target can even run it —
from wherever you happen to be:

```bash
offlineai check my-ai-app-1.0.0.offlineai --profile h100-server.yaml
```

`check` on its own validates the machine you are standing on, which on a
builder is the wrong machine. A target profile describes the destination
instead. Capture one from the real hardware with
`offlineai doctor --save-profile`.

## Install

Python 3.11 or newer.

On the **connected builder**, where bundles are produced:

```bash
git clone https://github.com/mkamranr/offlineai.git
cd offlineai
pip install -e ".[builder]"
```

On the **air-gapped target** there is no index to install from, so build the
wheels on the builder and carry them across alongside the bundle:

```bash
pip wheel -w dist/ .                                   # on the builder
pip install --no-index --find-links dist/ offlineai    # on the target
```

`pip wheel` builds for the platform it runs on, and several of these
dependencies ship compiled wheels, so build them on a machine whose OS and
Python version match the target.

Note the asymmetry: the target install takes **no extras**. `huggingface-hub`
and `httpx` exist to resolve artifacts, so they live in the `[builder]` extra --
nothing required to verify, import, install or run a bundle reaches for the
network. CI enforces that split by installing the core package on its own and
failing if `huggingface_hub` turns out to be importable.

Linux is the supported target. macOS runs in a degraded dev mode: Docker works,
and the `dpkg` and NVIDIA checks report SKIPPED with a reason rather than
pretending to pass.

## Example

Everything below is real output from `examples/hello-ai`, captured on Linux.
It is the smallest complete package — a containerised HTTP service, no model,
no GPU — and it builds in about a minute.

### What you write

`offlineai.yaml`, beside your Dockerfile:

```yaml
apiVersion: offlineai/v1
kind: Package

metadata:
  name: hello-ai
  version: 1.0.0

containers:
  - name: app
    image: python:3.12-slim     # the BASE image...
    dockerfile: Dockerfile      # ...because this builds a new one, and that is
                                #    what gets packaged. Not pulled on the target.

services:
  - name: app
    container: app
    ports: ["8000:8000"]

install:
  healthcheck:                  # what `install` waits on before reporting success
    command:                    # runs INSIDE the container - python:3.12-slim
      - python                  # has no curl, so use what the image has
      - -c
      - "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
    retries: 12
```

### On the connected builder

```console
$ offlineai build examples/hello-ai

[1/10] Validating package definition        OK
[2/10] Scanning for secrets                 OK  none found
[3/10] Resolving models                     OK  none declared
[4/10] Resolving container images           OK  1 image(s)
[5/10] Resolving Python dependencies        OK  none declared
[6/10] Resolving OS packages                OK  none declared
[7/10] Calculating checksums                OK  1 artifact(s)
[8/10] Generating manifest                  OK
[9/10] Creating bundle                      OK  hello-ai-1.0.0.offlineai
[10/10] Verifying bundle                    OK  1 artifact(s)

Bundle created:

  hello-ai-1.0.0.offlineai

Size:   131.6 MB
SHA256: 32e9074ba12bc20429fc62f683cff84e19fbcfb0106e79c9f31af1e0ffb0ed09
```

Copy that one file to the isolated machine, and compare the SHA-256 after the
transfer.

Your digest will differ from the one above: this package builds its image from
a Dockerfile, and `docker build` is not byte-reproducible. That is also why
OfflineAI compares images on their registry digest rather than on the bytes of
the saved tar — see [`offlineai.lock`](docs/package-format.md#offlineailock),
which pins the digest so a rebuild resolves to the same image.

### On the air-gapped target

To make the point honestly, delete the image first — so what starts there can
only have come out of the bundle:

```console
$ docker rmi -f offlineai/hello-ai-app:1.0.0
Untagged: offlineai/hello-ai-app:1.0.0

$ offlineai verify hello-ai-1.0.0.offlineai
Manifest:  OK
Container: OK
Checksums: OK  (1 artifact(s))
SBOM:      OK
Signature: SKIPPED  (bundle is unsigned)

Result: VERIFIED

$ offlineai import hello-ai-1.0.0.offlineai
Imported hello-ai 1.0.0

  Artifacts:   1
  Stored:      131.5 MB

$ offlineai install hello-ai --strict-offline
CPU architecture:      OK  amd64
Operating system:      OK  linux 6.4.16-linuxkit
Disk space:            OK  needs 369.7 MB, 2.8 GB available
Container runtime:     OK  24.0.6
GPU:                   SKIPPED  not required
Container images:      OK  1 image(s) loaded
Models:                SKIPPED  none declared
System packages:       SKIPPED  none declared
Python dependencies:   SKIPPED  none declared
Runtime configuration: OK
Services:              OK  1 started

Installation: install-20260926-001
State:        COMPLETED

Endpoints:
  http://localhost:8000

Requirements satisfied. Runtime success is not guaranteed.
```

`--strict-offline` is not decoration: sockets are blocked in-process, pip and
Hugging Face are pointed at nothing, proxies are cleared, and containers run
`--pull never`. Nothing can quietly fetch a missing piece.

And it serves:

```console
$ offlineai status hello-ai
Status: RUNNING

Services:
  app  RUNNING

Endpoint:
  http://localhost:8000

$ curl localhost:8000/health
{"status": "healthy"}
```

Note what every `SKIPPED` above says: *why*. A check that could not be
evaluated is never reported as passed.

### At scale

[`examples/qwen-vllm`](examples/qwen-vllm) is the same shape with a model and a
GPU requirement added — Qwen3-30B on vLLM, around 62 GB. The commands are
identical; only the numbers change, and `inspect` still reads about ten
kilobytes off the front of the file regardless.

```bash
offlineai build examples/qwen-vllm --sign-key signing-key.pem
offlineai check qwen-vllm-1.0.0.offlineai --profile h100-server.yaml
offlineai install qwen-vllm --gpus 0,1 --strict-offline
```

## The guarantee

> A successful build means the bundle contains everything required for the declared
> offline installation.

A bundle never silently depends on PyPI, Hugging Face, Docker Hub, APT repositories,
GitHub, DNS or any external API during installation. If something is required and absent,
installation fails clearly and says what is missing. Run with `--strict-offline` to make
that structural rather than a promise.

## Who it is for

Defense, government, healthcare, banking, industrial control and research environments —
anywhere outbound network access is restricted and deployments must be auditable,
verifiable and reproducible.

## What a bundle contains

```
manifest.yaml            metadata: read first, always small
checksums.sha256
signature/manifest.sig
sbom/sbom.json
artifacts/               payload: everything large, always last
├── models/
├── containers/
├── python/wheels/
└── system/packages/
```

Metadata comes first so `inspect`, `check` and `verify-signature` read a few
kilobytes from the front of the file rather than scanning it — a constant cost,
whether the bundle is 10 MB or 62 GB.

## Commands

```
build     verify    inspect   sign      verify-signature   keygen
import    install   uninstall rollback  remove
start     stop      restart   status    logs
list      search    info      graph     diff       sbom
check     doctor    network-check       plugins    init
```

Every one supports `--json`. Exit codes are documented and stable, so CI can
branch on them.

## Documentation

| | |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Ten minutes, start to finish |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How it is built and why |
| [SECURITY.md](SECURITY.md) | Threat model, and what verification does *not* prove |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and house rules |
| [docs/package-format.md](docs/package-format.md) | Writing `offlineai.yaml` |
| [docs/bundle-format.md](docs/bundle-format.md) | What is inside a `.offlineai` file |
| [docs/air-gapped-deployment.md](docs/air-gapped-deployment.md) | Deploying for real |
| [docs/model-packaging.md](docs/model-packaging.md) | Models, shards, resumable downloads |
| [docs/docker-packaging.md](docs/docker-packaging.md) | Images, digests, GPUs |
| [docs/troubleshooting.md](docs/troubleshooting.md) | When something goes wrong |

## Examples

| | |
|---|---|
| [`simple-python`](examples/simple-python) | Local files, no container. The smallest useful package. |
| [`hello-ai`](examples/hello-ai) | A containerised HTTP service. Builds in under a minute — [walked through above](#example). |
| [`whisper`](examples/whisper) | Speech-to-text, ~8 GB |
| [`qwen-vllm`](examples/qwen-vllm) | Qwen3-30B on vLLM with an OpenAI-compatible API, ~62 GB |
| [`rag-stack`](examples/rag-stack) | vLLM + Qdrant + Redis + app, three models, ~14 GB |

## Status

Under active development, and specific about what that means -- the same
PASS / FAIL / SKIPPED honesty the tool applies to your machine.

**Verified end to end.** The full `build` -> `verify` -> `import` -> `install` ->
`run` flow, against real Docker, on Linux and inside a `--network none`
container. 910 tests, including an air-gap suite that fails if anything opens a
socket.

**Not yet demonstrated.** No real model has been *installed*, though several
have been packaged. Python wheels have never been mounted into a running
container. No GPU path has met real hardware -- only captured `nvidia-smi`
output and target profiles. The largest bundle built so far is 131.6 MB,
against a design meant for tens of gigabytes.

Ready to evaluate, and to carry small bundles across a real air gap. Not yet
proven for a production AI workload.

## License

Apache-2.0. See [LICENSE](LICENSE).

License metadata recorded in bundles is informational. Users are responsible for reviewing
the licenses applicable to the models, images and packages they distribute.
