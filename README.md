# OfflineAI

**OfflineAI packages complete AI workloads for deployment into air-gapped environments.**

```
Build online.
Transfer once.
Run offline.
```

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
| [`hello-ai`](examples/hello-ai) | A containerised HTTP service. Builds in under a minute. |
| [`whisper`](examples/whisper) | Speech-to-text, ~8 GB |
| [`qwen-vllm`](examples/qwen-vllm) | Qwen3-30B on vLLM with an OpenAI-compatible API, ~62 GB |
| [`rag-stack`](examples/rag-stack) | vLLM + Qdrant + Redis + app, three models, ~14 GB |

## Status

Under active development.

## License

Apache-2.0. See [LICENSE](LICENSE).

License metadata recorded in bundles is informational. Users are responsible for reviewing
the licenses applicable to the models, images and packages they distribute.
