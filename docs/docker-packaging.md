# Container packaging

## Pull or build

Two shapes, and the difference matters more than it looks.

**Pull** an image that exists in a registry:

```yaml
containers:
  - name: vllm
    image: vllm/vllm-openai:v0.6.3
    platform: linux/amd64
```

**Build** an image that only exists in your repository:

```yaml
containers:
  - name: app
    image: python:3.12-slim    # the BASE
    dockerfile: Dockerfile
    context: .
```

With a `dockerfile`, `image` is the base to build *from*. The builder produces
a new image tagged `offlineai/<package>-<container>:<version>` and packages
that. Install starts the built image.

This catches people out. Without that distinction, install starts
`python:3.12-slim` — whose entrypoint is a Python REPL — and the container
exits immediately and restart-loops. If a service starts and dies with no
output, this is the first thing to check:

```bash
docker inspect offlineai-<package>-<service> --format '{{.Config.Image}} {{.Config.Cmd}}'
```

## Digests, not tags

A tag is mutable. `vllm/vllm-openai:latest` means something different next
month, and on an air-gapped host you cannot check.

The builder resolves the immutable digest and records it in the manifest. On
install, the loaded image's digest is compared against it and a mismatch is
reported.

Pin it explicitly if you want the build itself to be reproducible:

```yaml
containers:
  - name: vllm
    image: vllm/vllm-openai
    digest: sha256:abc123...
```

```bash
offlineai inspect bundle.offlineai --json | jq '.artifact_counts'
offlineai graph my-package          # shows each image's recorded digest
```

## Platform

```yaml
    platform: linux/amd64
```

Set it explicitly. A builder on Apple silicon that omits this will package
arm64 images for an amd64 target, and the failure appears on the isolated
machine.

## How images travel

Build time: `docker pull` (or `docker build`), then `docker save` into the
cache, addressed by the image's digest so a rebuild with an unchanged image is
a cache hit rather than another multi-gigabyte save.

Install time: `docker load` from the bundle. Never a pull — containers are
started with `--pull never`, so a missing image is reported as a bundle problem
instead of being papered over by a silent fetch.

## GPUs

```yaml
hardware:
  gpu:
    required: true
    vendor: nvidia
    minimum_vram_gb: 48
    minimum_driver: "550"
    count: 2
```

```bash
offlineai install my-package --gpus 0,1
```

That exposes the devices. It does **not** configure tensor parallelism or any
other sharding — how a workload divides itself across devices is an application
decision, and OfflineAI deliberately does not guess. Set it per site:

```yaml
# site-config.yaml
services:
  vllm:
    command: [--model, /models/model, --tensor-parallel-size, "2"]
```

The target needs the NVIDIA container toolkit installed. `offlineai doctor`
reports whether the runtime exposes GPUs at all.

## Services and start order

```yaml
services:
  - name: app
    container: app
    depends_on: [redis, qdrant, vllm]
    ports: ["8080:8080"]
```

Containers are named `offlineai-<package>-<service>` and labelled, so they are
findable with plain `docker ps` and so OfflineAI only ever touches its own.

```bash
docker ps --filter label=ai.offlineai.package=my-package
```

## Health checks

```yaml
install:
  healthcheck:
    command: [curl, --fail, "http://localhost:8000/health"]
    retries: 60
    interval_seconds: 10
```

Runs inside the container after start. Be generous with `retries` for anything
that loads large weights — 30B parameters take a while to read off disk, and a
health check that gives up first turns a working install into a failed one.

A container that exits before becoming healthy is detected immediately rather
than waited out.

## Python dependencies inside containers

Wheels from the bundle are mounted read-only at `/opt/offlineai/wheels`, with
`PIP_NO_INDEX=1` and `PIP_FIND_LINKS` already set. Any `pip install` inside the
workload resolves from the bundle and fails loudly otherwise.

```dockerfile
RUN pip install --no-index --find-links /wheels -r requirements.txt
```

## Podman and others

Not yet implemented, but the seam is there: `ContainerRuntime` is a protocol
and `runtime.container_engine` is a setting. Adding Podman means writing an
implementation, not editing the CLI.
