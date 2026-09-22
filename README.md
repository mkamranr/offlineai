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

containing model weights, Docker images, Python wheels, OS packages, configuration,
checksums, an SBOM, license metadata, startup scripts and documentation.

Transfer it by USB or any approved one-way channel. Then, on a machine with **no network
access at all**:

```bash
offlineai verify  my-ai-app-1.0.0.offlineai
offlineai import  my-ai-app-1.0.0.offlineai
offlineai install my-ai-app
offlineai status  my-ai-app
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

## Status

Under active development. See [`docs/`](docs/) for the package format, bundle format and
air-gapped deployment guide, and [ARCHITECTURE.md](ARCHITECTURE.md) for internals.

## License

Apache-2.0. See [LICENSE](LICENSE).

License metadata recorded in bundles is informational. Users are responsible for reviewing
the licenses applicable to the models, images and packages they distribute.
