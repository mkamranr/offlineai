# Bundle format

A `.offlineai` bundle is a tar archive with a defined member order. The order
is part of the format, not an implementation detail.

## Layout

```
manifest.yaml            ─┐
package.yaml              │
checksums.sha256          │  header: read first, always small
signature/manifest.sig    │
signature/pubkey.pub      │
sbom/sbom.json            │
metadata/licenses.json    │
docs/README.md           ─┘
artifacts/               ─┐
├── models/               │
├── containers/           │  payload: everything large, always last
├── python/wheels/        │
├── system/packages/      │
└── misc/                ─┘
```

## Why the order matters

Metadata first means `inspect`, `check`, `diff` and `verify-signature` read a
few kilobytes from the front of the file and stop. Measured on a 40 MB bundle:
**10.5 KB read, 0.026% of the file** — and that cost is constant, not
proportional, so a 62 GB bundle inspects for the same ten kilobytes.

Specification section 60 requires `inspect` to work without installing
anything. At AI-workload sizes that is only affordable if the read genuinely
stops early, so the test suite asserts it by counting bytes pulled off the
underlying file rather than trusting the design.

## Compression

Uncompressed by default. Safetensors and container layers are already
compressed; gzipping a 62 GB bundle costs hours of CPU for close to no saving.

```bash
offlineai build . --compress gzip    # for small bundles, if you want it
```

The manifest records which was used, so a bundle is self-describing.

## manifest.yaml

The authoritative description, and the root of trust.

```yaml
formatVersion: "1"
package:
  name: qwen-vllm
  version: 1.0.0
createdAt: "2026-09-22T12:00:00Z"
platforms: [linux/amd64]
compression: none

artifacts:
  - id: model-qwen3-config.json
    type: model
    path: artifacts/models/model/config.json
    size: 807
    sha256: 9ad49cd2ff58...
    source: hf://Qwen/Qwen3-30B@main/config.json
    license: apache-2.0

  - id: image-vllm
    type: oci-image
    path: artifacts/containers/vllm.tar
    size: 12884901888
    sha256: 4f2e...
    digest: sha256:abc123...        # immutable digest at the origin

requirements:
  docker: {minimumVersion: "27"}
  gpu: {vendor: nvidia, minimumDriver: "550", minimumMemoryGB: 48, count: 1}
  minimumRamGB: 64
  pythonVersion: "3.12"

packageDefinitionSha256: c3f1...    # traces the bundle to its definition
builder:
  offlineaiVersion: 0.1.0
  os: linux
  architecture: x86_64
  pythonVersion: 3.12.14
requiredSecrets: [HUGGINGFACE_TOKEN]
```

Because the manifest records a hash for every artifact, a signature over the
manifest covers the whole payload transitively. That is what keeps signature
verification header-only.

## Guarantees the writer enforces

- **Ordering.** The header cannot be written after an artifact, or vice versa.
- **Completeness.** On close, every artifact the manifest declares must have
  been written, and its content must hash to what the manifest says.
- **Cleanliness.** A failed or aborted build removes the partial file, so no
  later step can mistake it for a finished bundle.

Together these mean a build cannot succeed and produce a bundle that is missing
something it promised — specification section 74 as an invariant rather than an
aspiration.

## Reading one by hand

It is an ordinary tar. Nothing here requires OfflineAI to inspect:

```bash
tar -tvf bundle.offlineai | head -20
tar -xOf bundle.offlineai manifest.yaml
tar -xOf bundle.offlineai sbom/sbom.json | jq .
```

That is deliberate. On an isolated host, being able to look inside with the
tools already present matters.

## Versioning

`formatVersion` is checked exactly. A bundle written by a newer OfflineAI is
refused with a clear message rather than half-understood.
