# OfflineAI — Air-Gapped AI Package & Deployment Manager

## 1. Project Overview

Build an open-source tool called **OfflineAI** that makes it easy to package, transfer, verify, install, and run complete AI/ML workloads in **air-gapped or internet-restricted environments**.

The core problem:

Modern AI deployments are difficult to reproduce offline because a single application may require:

- Large model weights
- Docker/OCI images
- Python packages
- OS packages
- CUDA/runtime dependencies
- Configuration files
- Tokenizers
- Model-specific assets
- Startup scripts
- Plugins/extensions
- Metadata and licenses
- Checksums
- Hardware/runtime requirements

An administrator may have an internet-connected "builder" machine and a completely isolated target server.

OfflineAI should provide a workflow such as:

```text
INTERNET-CONNECTED MACHINE
        |
        | offlineai bundle qwen3-30b
        v
+-------------------------+
| OfflineAI Bundle         |
|-------------------------|
| Model weights            |
| Docker images            |
| Python wheels            |
| OS packages             |
| Config                  |
| Runtime metadata        |
| Checksums               |
| SBOM                    |
| License metadata        |
| Startup scripts         |
+-------------------------+
        |
        | USB / approved transfer
        v
AIR-GAPPED SERVER
        |
        | offlineai import
        v
+-------------------------+
| Local Offline Registry  |
+-------------------------+
        |
        | offlineai install
        v
Running AI workload
```

The project must be designed as an **infrastructure/developer tool**, not as a hosted SaaS product.

The primary design goal is:

> Package an AI application and everything required to reproduce it offline, transfer it safely, verify it, install it, and operate it without internet access.

---

# 2. Goals

## 2.1 Primary goals

OfflineAI must:

1. Work without internet access on the target environment.
2. Work with Docker/OCI containers.
3. Package models and model assets.
4. Package Python dependencies.
5. Package OS dependencies where practical.
6. Package configuration.
7. Generate deterministic manifests.
8. Generate cryptographic checksums.
9. Verify bundle integrity before installation.
10. Support resumable/interrupted transfers where possible.
11. Provide a local package registry/cache.
12. Install workloads from local bundles.
13. Provide hardware compatibility information.
14. Detect missing GPU/runtime requirements.
15. Support multiple AI frameworks.
16. Be usable from a CLI.
17. Have a clean plugin/extensibility architecture.
18. Be usable by organizations with no outbound network access.
19. Avoid requiring a subscription.
20. Be fully self-hostable and open-source.

---

# 3. Non-Goals for MVP

Do NOT attempt to build all of these initially:

- Kubernetes orchestration
- Cloud provisioning
- Multi-cloud deployment
- Full VM image creation
- Automatic NVIDIA driver installation
- Automatic BIOS configuration
- Full enterprise IAM
- SaaS control plane
- AI model training orchestration
- Distributed training
- Automatic security certification/compliance
- Automatic downloading from arbitrary private websites without credentials
- Replacement for Docker
- Replacement for apt/yum
- Replacement for Git

These can be future extensions.

---

# 4. Primary Users

## 4.1 AI/ML engineer

Example:

```bash
offlineai bundle qwen3-30b
```

They receive a portable bundle that can be moved to an isolated server.

## 4.2 Infrastructure administrator

Example:

```bash
offlineai verify qwen3-30b.offlineai
offlineai install qwen3-30b.offlineai
offlineai status
```

## 4.3 Enterprise/air-gapped administrator

Needs:

- auditability
- checksums
- reproducibility
- package manifests
- offline operation
- controlled transfer
- clear dependencies

## 4.4 Open-source developer

Should be able to define a workload:

```yaml
name: my-ai-app
version: 1.0.0

containers:
  - image: myorg/my-ai-app:1.0

models:
  - hf://some/model

python:
  requirements: requirements.txt

services:
  - name: api
    port: 8000
```

Then build an offline bundle.

---

# 5. Core Concepts

OfflineAI should use these concepts.

## 5.1 Package

A logical AI workload definition.

Example:

```text
qwen3-30b
```

A package contains:

- metadata
- model
- containers
- Python dependencies
- system dependencies
- configuration
- startup instructions
- hardware requirements

## 5.2 Bundle

A portable archive containing all required artifacts.

Example:

```text
qwen3-30b-1.0.0.offlineai
```

A bundle must be self-describing.

## 5.3 Artifact

Any binary/resource required by the package:

- Docker image
- model weight
- wheel
- tarball
- OS package
- configuration
- script
- tokenizer
- plugin

## 5.4 Manifest

Machine-readable description of the bundle.

## 5.5 Local Registry

A local catalog/cache of imported bundles and artifacts.

## 5.6 Profile

A target environment definition.

Example:

```yaml
name: h100-server

os:
  family: ubuntu
  version: "24.04"

architecture: amd64

gpu:
  vendor: nvidia
  minimum_driver: "550"
  memory_gb: 80

runtime:
  docker: ">=27"
```

---

# 6. High-Level Architecture

Recommended architecture:

```text
                    OfflineAI CLI
                         |
             +-----------+-----------+
             |                       |
        Package Engine         Environment Engine
             |                       |
       +-----+------+          +-----+------+
       |            |          |            |
   Resolver     Bundler     Hardware     Runtime
       |            |          detection   detection
       |
 +-----+-----------------------------------------+
 |              Artifact Manager                 |
 +----------------------+------------------------+
                        |
          +-------------+-------------+
          |             |             |
       Models        OCI Images    Packages
          |             |             |
          +-------------+-------------+
                        |
                  Bundle Builder
                        |
                 Manifest + Hashes
                        |
                 .offlineai bundle
                        |
              ======================
                Air-Gapped Server
              ======================
                        |
                 OfflineAI CLI
                        |
                Local Registry
                        |
               Install / Verify
                        |
                Docker / Runtime
```

---

# 7. CLI Requirements

The CLI is the primary interface.

Executable:

```bash
offlineai
```

Use a consistent command structure.

---

## 7.1 Initialization

```bash
offlineai init
```

Creates:

```text
~/.offlineai/
    config.yaml
    registry/
    cache/
    bundles/
    logs/
```

Allow custom data directory:

```bash
offlineai --data-dir /opt/offlineai ...
```

---

# 8. Package Definition

OfflineAI needs a package specification.

Suggested filename:

```text
offlineai.yaml
```

Example:

```yaml
apiVersion: offlineai/v1

kind: Package

metadata:
  name: qwen3-30b
  version: 1.0.0
  description: Qwen AI inference workload

runtime:
  type: docker

architecture:
  - amd64

os:
  - ubuntu:24.04

models:
  - name: qwen3
    source:
      type: huggingface
      repo: Qwen/Qwen3-30B
    destination: /models/qwen3

containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    platform: linux/amd64

python:
  requirements:
    - requirements.txt

system:
  packages:
    - curl
    - ca-certificates

environment:
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"

volumes:
  - host: ./data
    container: /data

services:
  - name: vllm
    container: vllm
    command:
      - python
      - -m
      - vllm.entrypoints.openai.api_server
    ports:
      - "8000:8000"

hardware:
  gpu:
    required: true
    vendor: nvidia
    minimum_vram_gb: 48

install:
  healthcheck:
    command:
      - curl
      - http://localhost:8000/health
```

The exact schema can evolve, but it must be versioned.

---

# 9. Bundle Format

Define a custom bundle format:

```text
.offlineai
```

Internally it can initially be a compressed tar archive.

Example:

```text
qwen3-30b-1.0.0.offlineai
```

Suggested structure:

```text
bundle/
├── manifest.yaml
├── package.yaml
├── checksums.sha256
├── sbom/
│   └── sbom.json
├── metadata/
│   ├── licenses.json
│   ├── sources.json
│   └── hardware.json
├── artifacts/
│   ├── models/
│   │   └── qwen3/
│   ├── containers/
│   │   └── vllm.tar
│   ├── python/
│   │   └── wheels/
│   ├── system/
│   │   └── packages/
│   └── misc/
├── scripts/
│   ├── install.sh
│   ├── start.sh
│   ├── stop.sh
│   └── healthcheck.sh
└── docs/
    └── README.md
```

The bundle must be extractable without internet access.

---

# 10. Manifest

Example:

```yaml
formatVersion: "1"

package:
  name: qwen3-30b
  version: 1.0.0

createdAt: "2026-09-22T12:00:00Z"

platforms:
  - linux/amd64

artifacts:

  - id: model-qwen3
    type: model
    path: artifacts/models/qwen3
    size: 123456789
    sha256: ...

  - id: image-vllm
    type: oci-image
    path: artifacts/containers/vllm.tar
    size: 987654321
    sha256: ...

  - id: wheel-fastapi
    type: python-wheel
    path: artifacts/python/wheels/fastapi.whl
    size: 123456
    sha256: ...

requirements:

  docker:
    minimumVersion: "27"

  gpu:
    vendor: nvidia
    minimumDriver: "550"
    minimumMemoryGB: 48
```

The manifest is the authoritative description of the bundle.

---

# 11. Checksums

Every artifact must have a SHA-256 checksum.

Command:

```bash
offlineai verify bundle.offlineai
```

Expected output:

```text
OfflineAI Bundle Verification

Package: qwen3-30b
Version: 1.0.0

Manifest:       OK
Model files:    OK
Container:      OK
Python wheels:  OK
Checksums:      OK
SBOM:           OK

Result: VERIFIED
```

Failure:

```text
ERROR: checksum mismatch

Artifact:
artifacts/models/qwen3/model.safetensors

Expected:
abc123...

Actual:
def456...

Bundle verification FAILED.
```

---

# 12. Optional Digital Signatures

Design the architecture so digital signatures can be added.

MVP can support:

```bash
offlineai sign bundle.offlineai
offlineai verify-signature bundle.offlineai
```

Use modern cryptography.

Possible future implementation:

- Ed25519
- Sigstore/Cosign-compatible signatures

Do not make external signing infrastructure mandatory.

---

# 13. Model Support

The first implementation should support Hugging Face-style models.

Example:

```yaml
models:
  - name: qwen3
    source:
      type: huggingface
      repo: Qwen/Qwen3-30B
```

OfflineAI should resolve:

- config.json
- tokenizer files
- model files
- generation configuration
- special token files
- safetensors
- auxiliary files

Do NOT assume that a model consists of one file.

For sharded models:

```text
model-00001-of-00008.safetensors
...
model-00008-of-00008.safetensors
```

all files must be included.

---

# 14. Model Source Abstraction

Implement a source interface.

Example conceptual interface:

```python
class ArtifactSource:
    def resolve(self, reference):
        ...

    def download(self, reference, destination):
        ...

    def metadata(self, reference):
        ...
```

Initial source:

```text
HuggingFaceSource
```

Future sources:

```text
S3Source
HTTPSource
LocalSource
GitSource
ModelScopeSource
OCIRegistrySource
```

The target system must not depend directly on one provider.

---

# 15. Docker/OCI Support

OfflineAI must support Docker/OCI images.

Example:

```bash
offlineai bundle myapp
```

It should identify:

```text
vllm/vllm-openai:latest
redis:7
nginx:alpine
```

and save them into the bundle.

Preferred approach:

```bash
docker pull ...
docker save ...
```

or OCI-native tooling where practical.

On the offline machine:

```bash
offlineai install bundle.offlineai
```

should load images into the local container runtime.

Support Docker first.

Design for future support of:

- Podman
- containerd
- OCI registries

---

# 16. Container Image Digests

Never rely only on:

```text
image:tag
```

Record immutable digests whenever available.

Example:

```yaml
image: vllm/vllm-openai
tag: latest
digest: sha256:abc123...
```

During verification/install, warn if the expected digest does not match.

---

# 17. Python Dependencies

Support Python requirements.

Example:

```text
requirements.txt
```

OfflineAI should resolve and package:

```text
*.whl
*.tar.gz
```

including transitive dependencies.

Example:

```bash
offlineai resolve-python requirements.txt
```

Then:

```bash
offlineai bundle .
```

must include all required wheels.

Offline installation:

```bash
pip install --no-index \
  --find-links ./artifacts/python/wheels \
  -r requirements.txt
```

The application must never attempt to access PyPI during offline installation.

---

# 18. Python Compatibility

Record:

```yaml
python:
  version: "3.12"
  platform: linux
  architecture: amd64
```

If wheels are platform-specific, record:

```text
cp312
manylinux
linux_x86_64
```

The installer must detect incompatibilities.

Example:

```text
ERROR

Bundle requires:
Python 3.12
Linux amd64

Detected:
Python 3.11
Linux amd64

Installation cannot continue.
```

---

# 19. OS Packages

Support Linux packages where practical.

Initial target:

```text
Ubuntu/Debian
```

Examples:

```text
apt .deb packages
```

The builder machine should download required packages and dependencies.

Bundle:

```text
artifacts/system/packages/*.deb
```

Offline installer:

```bash
dpkg -i ...
apt install ...
```

Do not assume internet repositories exist.

Important:

The installer must clearly distinguish:

```text
Required package
Optional package
Recommended package
```

---

# 20. GPU Support

GPU detection is a major feature.

Initial GPU support:

```text
NVIDIA
CUDA
```

Detect:

```bash
nvidia-smi
```

Collect:

- GPU model
- VRAM
- driver version
- CUDA compatibility
- GPU count
- compute capability where available

Example:

```text
GPU Environment

GPU 0:
  NVIDIA H100 80GB
  Driver: 580.xx
  VRAM: 80 GB

GPU 1:
  NVIDIA H100 80GB
  Driver: 580.xx
  VRAM: 80 GB

Detected: 2 GPUs
```

---

# 21. Hardware Compatibility

Before installation:

```bash
offlineai check bundle.offlineai
```

Output:

```text
Hardware Compatibility

CPU architecture:      PASS
RAM:                    PASS
Disk space:             PASS
Docker:                 PASS
NVIDIA driver:          PASS
GPU VRAM:               PASS

Result: COMPATIBLE
```

Failure example:

```text
GPU VRAM

Required: 48 GB
Available: 24 GB

Result: FAIL
```

The tool must not automatically claim that a workload will work merely because the minimum hardware requirement is met.

Use wording such as:

```text
Requirements satisfied.
Runtime success is not guaranteed.
```

---

# 22. Disk Space Planning

Before installation calculate:

```text
Bundle size
+
Extraction overhead
+
Docker image storage
+
Model storage
+
Temporary working space
```

Example:

```text
Installation Storage Estimate

Bundle:               72 GB
Extraction:           72 GB
Container storage:    14 GB
Model storage:        48 GB
Temporary:             8 GB
--------------------------------
Recommended free:    214 GB

Available:            180 GB

WARNING:
Insufficient recommended disk space.
```

---

# 23. Local Registry

After importing a bundle:

```bash
offlineai import qwen3-30b.offlineai
```

store metadata locally.

Commands:

```bash
offlineai list
offlineai info qwen3-30b
offlineai remove qwen3-30b
offlineai search qwen
```

Example:

```text
NAME          VERSION   PLATFORM     SIZE
qwen3-30b     1.0.0     linux/amd64  82 GB
whisper       2.0.0     linux/amd64   8 GB
rag-stack     1.1.0     linux/amd64  14 GB
```

---

# 24. Registry Storage

Suggested structure:

```text
/var/lib/offlineai/

registry/
    index.db

bundles/
    sha256/
        ab/
            abcdef....

artifacts/
    models/
    containers/
    python/
    system/

logs/
```

Use SQLite for metadata.

Do not use MongoDB or another external database for the MVP.

OfflineAI should be self-contained.

---

# 25. Bundle Import

Command:

```bash
offlineai import ./qwen3-30b.offlineai
```

Steps:

1. Open bundle.
2. Validate structure.
3. Read manifest.
4. Verify format version.
5. Verify checksums.
6. Verify optional signature.
7. Check available disk.
8. Store bundle/artifacts.
9. Register package metadata.
10. Report success.

---

# 26. Installation

Command:

```bash
offlineai install qwen3-30b
```

or:

```bash
offlineai install ./qwen3-30b.offlineai
```

Installation pipeline:

```text
Validate
   ↓
Verify checksum
   ↓
Check signature
   ↓
Check OS
   ↓
Check CPU architecture
   ↓
Check disk
   ↓
Check Docker
   ↓
Check GPU
   ↓
Load container images
   ↓
Install system packages
   ↓
Install Python dependencies
   ↓
Install models
   ↓
Generate runtime configuration
   ↓
Start services
   ↓
Health check
```

If a step fails, provide a clear error and recovery instructions.

---

# 27. Installation Transactions

Installation should be transactional where practical.

Create:

```text
installation ID
```

Example:

```text
install-20260922-001
```

Track:

```text
PENDING
VALIDATING
INSTALLING
STARTING
HEALTH_CHECK
COMPLETED
FAILED
ROLLED_BACK
```

If installation fails:

```bash
offlineai rollback install-20260922-001
```

MVP rollback can focus on:

- stopping created containers
- removing created networks
- removing generated files
- restoring registry state

Do not automatically delete user-owned data.

---

# 28. Runtime Management

Commands:

```bash
offlineai start qwen3-30b
offlineai stop qwen3-30b
offlineai restart qwen3-30b
offlineai status qwen3-30b
offlineai logs qwen3-30b
```

Example:

```text
Package: qwen3-30b

Status: RUNNING

Services:
  vllm       RUNNING
  redis      RUNNING

Health:
  API        HEALTHY

Endpoint:
  http://localhost:8000
```

---

# 29. Configuration Overrides

Allow local environment configuration without modifying the original bundle.

Example:

```text
offlineai-config.yaml
```

```yaml
runtime:
  gpu:
    device_ids:
      - "0"
      - "1"

services:
  vllm:
    ports:
      - "8001:8000"

environment:
  VLLM_MAX_MODEL_LEN: "32768"
```

Command:

```bash
offlineai install qwen3-30b \
  --config ./server-config.yaml
```

---

# 30. Multi-GPU

Support configuration such as:

```yaml
gpu:
  count: 2
  device_ids:
    - "0"
    - "1"
```

The system should expose GPU configuration to containers.

For Docker:

```text
NVIDIA_VISIBLE_DEVICES
```

or equivalent Docker GPU configuration.

Do not hard-code assumptions about tensor parallelism. Package definitions should specify application-level configuration separately.

---

# 31. Environment Validation

Command:

```bash
offlineai doctor
```

Example:

```text
OfflineAI Doctor

OS                  PASS
Architecture        PASS
Docker              PASS
Docker Compose      PASS
NVIDIA runtime      PASS
GPU                  PASS
Disk                 PASS
Filesystem           PASS
Permissions          PASS

Warnings:
- Docker version newer than tested version
- GPU driver not explicitly validated

Overall: READY
```

---

# 32. Air-Gap Enforcement

Provide:

```bash
offlineai install --strict-offline bundle.offlineai
```

In strict mode:

- no HTTP
- no HTTPS
- no DNS dependency
- no package repository access
- no model repository access

The installer must use only:

- bundle contents
- local filesystem
- local Docker runtime
- local package manager cache
- local registry

Optionally provide an installation network guard.

The application itself should never silently download missing dependencies.

If something is missing:

```text
ERROR:
Required artifact is not present in the bundle.

Artifact:
transformers>=5.x

Network access is disabled because strict-offline mode is enabled.

Rebuild the bundle with the required dependency.
```

---

# 33. Network Audit

Provide an optional command:

```bash
offlineai network-check
```

The goal is not to act as a general packet sniffer.

Instead, inspect package definitions and runtime configuration for:

- external URLs
- remote model endpoints
- package repositories
- container pull policies
- HTTP/HTTPS configuration

Report:

```text
External Dependency Audit

Found:
https://huggingface.co/...
https://pypi.org/...

These URLs are BUILD-TIME dependencies.

No runtime network dependency detected.
```

This distinction is important:

```text
BUILD TIME
vs
RUNTIME
```

---

# 34. Reproducibility

A bundle should be reproducible as much as practical.

Record:

```text
Package definition hash
Artifact hashes
Container digests
Python package versions
OS package versions
Builder OS
Builder architecture
Builder OfflineAI version
Creation timestamp
```

Avoid timestamps inside artifact archives where possible if deterministic packaging is supported.

---

# 35. Lock File

Introduce:

```text
offlineai.lock
```

Example:

```yaml
formatVersion: "1"

package:
  name: qwen3-30b
  version: 1.0.0

artifacts:
  - type: model
    name: qwen3
    digest: sha256:...

  - type: oci
    name: vllm
    digest: sha256:...

  - type: python
    name: transformers
    version: 5.0.0
    sha256: ...
```

The lock file ensures future builds use the exact same artifacts.

---

# 36. SBOM

Generate a Software Bill of Materials.

Preferred initial format:

```text
SPDX JSON
```

or:

```text
CycloneDX JSON
```

Include:

- application packages
- Python packages
- OS packages
- container metadata
- versions
- hashes where possible

Command:

```bash
offlineai sbom bundle.offlineai
```

---

# 37. License Metadata

Record license information where available.

Example:

```yaml
licenses:
  - component: transformers
    license: Apache-2.0

  - component: qwen3
    license: ...
```

Do not make legal claims.

Display:

```text
License metadata is informational.
Users are responsible for reviewing applicable licenses.
```

---

# 38. Security Model

Security must be considered from the beginning.

Threats:

1. Tampered bundle
2. Malicious model
3. Malicious container
4. Compromised dependency
5. Bundle replacement
6. Path traversal
7. Malicious archive
8. Privilege escalation
9. Untrusted package definition
10. Unexpected network access

Mitigations:

- SHA-256 hashes
- optional digital signatures
- safe archive extraction
- path traversal protection
- no automatic privileged operations unless explicitly requested
- artifact provenance
- immutable digest recording
- explicit installation confirmation for privileged actions

---

# 39. Safe Archive Extraction

Never blindly extract archive paths.

Reject paths such as:

```text
../../etc/passwd
```

or:

```text
/etc/passwd
```

or symlink attacks.

Use a safe extraction utility.

Add tests specifically for malicious archives.

---

# 40. Secrets

Never package secrets by default.

Reject or warn about files such as:

```text
.env
*.pem
*.key
credentials.json
id_rsa
```

unless explicitly allowed.

Support:

```yaml
secrets:
  external:
    - HUGGINGFACE_TOKEN
    - DATABASE_PASSWORD
```

The bundle contains only the name, not the value.

Example:

```text
HUGGINGFACE_TOKEN: REQUIRED
DATABASE_PASSWORD: REQUIRED
```

---

# 41. Build Process

Command:

```bash
offlineai build .
```

or:

```bash
offlineai bundle ./offlineai.yaml
```

Pipeline:

```text
Read package definition
        ↓
Validate schema
        ↓
Resolve dependencies
        ↓
Resolve model artifacts
        ↓
Resolve container artifacts
        ↓
Resolve Python dependencies
        ↓
Resolve system packages
        ↓
Collect metadata
        ↓
Calculate hashes
        ↓
Generate SBOM
        ↓
Generate manifest
        ↓
Create archive
        ↓
Verify resulting bundle
```

---

# 42. Build Output

Example:

```text
Building package: qwen3-30b:1.0.0

[1/9] Validating package definition       OK
[2/9] Resolving model                     OK
[3/9] Pulling container image             OK
[4/9] Resolving Python dependencies       OK
[5/9] Resolving OS packages               OK
[6/9] Generating SBOM                     OK
[7/9] Calculating checksums               OK
[8/9] Creating bundle                     OK
[9/9] Verifying bundle                    OK

Bundle created:

qwen3-30b-1.0.0.offlineai

Size: 82.4 GB
SHA256: ...

Build completed successfully.
```

---

# 43. Resume Support

Large AI models may be hundreds of GB.

The builder should support resumable artifact downloads.

Example:

```bash
offlineai build . --resume
```

If interrupted:

```text
Model download:
48% complete
```

Resume rather than restarting.

Artifacts should be downloaded into a cache.

---

# 44. Global Artifact Cache

Use:

```text
~/.offlineai/cache
```

or configurable location.

If multiple bundles require the same artifact:

```text
Qwen model
vLLM image
PyTorch wheel
```

download/build it once.

Example:

```text
Cache hit:
vllm/vllm-openai@sha256:abc...

Skipping download.
```

---

# 45. Parallel Downloads

Support configurable concurrency:

```bash
offlineai build . --workers 8
```

But avoid overwhelming storage/network.

Default should be conservative.

---

# 46. Progress UI

For large artifacts show:

```text
Qwen model

████████████████████░░░░ 82%

41.2 GB / 50.0 GB
Speed: 420 MB/s
ETA: 21 sec
```

For non-interactive environments support:

```bash
--quiet
--json
```

---

# 47. Machine-Readable Output

Every important command should support:

```bash
--json
```

Example:

```bash
offlineai status --json
```

Output:

```json
{
  "package": "qwen3-30b",
  "version": "1.0.0",
  "status": "running",
  "services": {
    "vllm": "healthy"
  }
}
```

This allows integration with:

- CI/CD
- monitoring
- automation
- Ansible
- shell scripts

---

# 48. Logging

Use structured logging.

Support:

```bash
offlineai --log-level debug ...
```

Log levels:

```text
debug
info
warning
error
```

Do not log secrets.

---

# 49. Plugin Architecture

Design source/resolver interfaces so additional artifact types can be added.

Potential future plugins:

```text
HuggingFace
ModelScope
S3
Azure Blob
OCI registry
GitLab registry
NVIDIA NGC
PyPI mirror
APT mirror
RPM/YUM
```

Do not hard-code every provider into the CLI.

---

# 50. Initial Technology Direction

The implementation should prioritize simplicity and portability.

Suggested implementation:

```text
Language:
Python 3.11+

CLI:
Typer or Click

Validation:
Pydantic

Database:
SQLite

Archive:
tar + compression

Hash:
SHA-256

SBOM:
CycloneDX or SPDX

Container:
Docker CLI / OCI-compatible abstractions

HTTP:
httpx

YAML:
PyYAML

Testing:
pytest
```

These are recommendations, not absolute requirements.

The implementation should use mature, well-maintained libraries.

---

# 51. Project Structure

Suggested:

```text
offlineai/
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── docs/
│   ├── architecture.md
│   ├── package-format.md
│   ├── bundle-format.md
│   ├── security.md
│   └── development.md
│
├── src/
│   └── offlineai/
│       ├── cli/
│       ├── config/
│       ├── models/
│       ├── resolver/
│       ├── artifacts/
│       ├── bundler/
│       ├── registry/
│       ├── installer/
│       ├── runtime/
│       ├── hardware/
│       ├── security/
│       ├── sbom/
│       ├── plugins/
│       └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── fixtures/
│
└── examples/
    ├── qwen-vllm/
    ├── whisper/
    ├── rag-stack/
    └── simple-python/
```

---

# 52. API Design

The internal API should be modular.

Example:

```python
PackageResolver
ArtifactResolver
BundleBuilder
BundleVerifier
Registry
Installer
RuntimeManager
HardwareDetector
EnvironmentChecker
SBOMGenerator
SignatureManager
```

Example:

```python
resolver = PackageResolver()
package = resolver.load("offlineai.yaml")

resolved = resolver.resolve(package)

builder = BundleBuilder()

bundle = builder.build(
    package=package,
    artifacts=resolved
)

verifier = BundleVerifier()

result = verifier.verify(bundle)
```

Avoid tightly coupling the CLI to implementation details.

---

# 53. REST API — Future/Optional

Do not make REST API mandatory for MVP.

Design internal services so a future API can expose:

```text
GET /packages
GET /packages/{name}
POST /bundles/import
POST /packages/{name}/install
GET /installations
GET /system
GET /hardware
```

This could later support a web UI.

---

# 54. Web UI — Future

Do not build a complex web UI initially.

Future dashboard:

```text
OfflineAI

Packages
--------------------------------
Qwen3       1.0.0    Installed
Whisper     2.0.0    Installed
RAG Stack   1.2.0    Available

System
--------------------------------
GPU: H100 80GB x4
RAM: 512 GB
Disk: 3.2 TB free
Docker: Ready

Running
--------------------------------
Qwen3
Whisper
```

CLI-first architecture is mandatory.

---

# 55. Example: Simple Python Application

Package:

```yaml
apiVersion: offlineai/v1

kind: Package

metadata:
  name: hello-ai
  version: 1.0.0

runtime:
  type: docker

containers:
  - name: app
    image: python:3.12-slim
    dockerfile: Dockerfile

python:
  requirements:
    - requirements.txt

services:
  - name: app
    container: app
    ports:
      - "8000:8000"
```

Build:

```bash
offlineai build .
```

Transfer:

```text
hello-ai-1.0.0.offlineai
```

Install offline:

```bash
offlineai import hello-ai-1.0.0.offlineai
offlineai install hello-ai
```

---

# 56. Example: vLLM Workload

The project should provide an example for a local LLM server.

Example package:

```yaml
apiVersion: offlineai/v1

kind: Package

metadata:
  name: qwen-vllm
  version: 1.0.0

runtime:
  type: docker

models:
  - name: model
    source:
      type: huggingface
      repo: Qwen/Qwen3-30B
    destination: /models/model

containers:
  - name: vllm
    image: vllm/vllm-openai
    digest: sha256:...

hardware:
  gpu:
    required: true
    vendor: nvidia
    minimum_vram_gb: 48

environment:
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"

services:
  - name: vllm
    container: vllm
    ports:
      - "8000:8000"
```

Runtime must mount:

```text
/models/model
```

and prevent model downloading at runtime.

---

# 57. Example: RAG Stack

Provide a more complex example:

```text
RAG application
+
vLLM
+
Qdrant
+
Redis
```

Bundle contains:

```text
vLLM image
Qdrant image
Redis image
Application image
LLM model
Embedding model
Reranker model
Python dependencies
```

One command:

```bash
offlineai install rag-stack
```

should make the complete stack available offline.

This example demonstrates why OfflineAI is more than a Docker image exporter.

---

# 58. Dependency Graph

Generate dependency graph:

```text
rag-stack
│
├── application
│   ├── Python dependencies
│   └── container
│
├── vLLM
│   ├── container
│   └── Qwen model
│
├── Qdrant
│   └── container
│
└── Redis
    └── container
```

Command:

```bash
offlineai graph rag-stack
```

Optional output formats:

```text
ASCII
JSON
DOT
```

---

# 59. Package Diff

Support future command:

```bash
offlineai diff bundle-v1.offlineai bundle-v2.offlineai
```

Output:

```text
Package Diff

Added:
  model file X
  transformers 5.0.0

Removed:
  transformers 4.x

Changed:
  vLLM image digest
  configuration

Size:
  72 GB → 81 GB
```

This is useful for controlled enterprise deployments.

---

# 60. Bundle Inspection

Command:

```bash
offlineai inspect qwen3-30b.offlineai
```

Output:

```text
Package:
  qwen3-30b

Version:
  1.0.0

Architecture:
  linux/amd64

Artifacts:
  Model:       48.3 GB
  Containers:  12.1 GB
  Python:       1.2 GB
  System:       0.4 GB

Total:
  62.0 GB

GPU:
  NVIDIA
  Minimum VRAM: 48 GB

Runtime:
  Docker >= 27

Signature:
  PRESENT

SBOM:
  PRESENT
```

This command must work without installing anything.

---

# 61. Bundle Integrity Test

After building:

```bash
offlineai verify bundle.offlineai
```

The builder itself should automatically perform verification.

A corrupted bundle must never be reported as valid.

---

# 62. Testing Requirements

Testing is a major part of the project.

## Unit tests

Test:

- YAML parsing
- schema validation
- manifest generation
- hash calculation
- archive creation
- archive extraction
- dependency resolution
- hardware parsing
- version comparison
- disk calculation

## Security tests

Test:

- path traversal
- malicious symlink
- malformed archive
- checksum mismatch
- corrupted manifest
- invalid signature
- unexpected executable files
- secret detection

## Integration tests

Test:

```text
Build → Transfer → Import → Verify → Install → Start → Healthcheck
```

Use small fake artifacts in CI.

Do not download multi-GB models in normal CI.

---

# 63. Offline CI Test

The project itself must have an offline test mode.

Example:

```bash
OFFLINEAI_TEST_OFFLINE=1 pytest
```

Tests must ensure:

- no external HTTP calls
- no external DNS dependency
- no package download
- local fixtures are sufficient

---

# 64. Failure Handling

Errors must be actionable.

Bad:

```text
Error 500
```

Good:

```text
Installation failed.

Reason:
Docker image vllm was not found in the bundle.

Expected:
vllm/vllm-openai@sha256:abc...

Action:
Rebuild the bundle and ensure the container image is included.
```

---

# 65. User Experience Principles

OfflineAI should feel like a professional CLI tool.

Commands should be:

```bash
offlineai build
offlineai verify
offlineai import
offlineai install
offlineai start
offlineai stop
offlineai status
offlineai logs
offlineai doctor
offlineai list
offlineai inspect
```

Avoid excessively complicated commands.

Provide:

```bash
offlineai --help
offlineai build --help
```

with useful examples.

---

# 66. Exit Codes

Define predictable exit codes.

Example:

```text
0   success
1   general error
2   invalid package
3   verification failure
4   hardware incompatibility
5   missing dependency
6   installation failure
7   signature failure
8   insufficient disk
9   runtime failure
```

Document them.

---

# 67. Environment Variables

Support:

```text
OFFLINEAI_HOME
OFFLINEAI_DATA_DIR
OFFLINEAI_CACHE_DIR
OFFLINEAI_REGISTRY_DIR
OFFLINEAI_LOG_LEVEL
OFFLINEAI_OFFLINE
```

CLI flags should override environment variables.

---

# 68. Configuration

Example:

```yaml
data_dir: /var/lib/offlineai
cache_dir: /var/cache/offlineai

runtime:
  container_engine: docker

security:
  require_signature: false

offline:
  strict: true

downloads:
  workers: 4
```

---

# 69. Documentation Requirements

The repository must include:

```text
README.md
QUICKSTART.md
ARCHITECTURE.md
SECURITY.md
CONTRIBUTING.md
LICENSE
```

Also include:

```text
docs/
  package-format.md
  bundle-format.md
  air-gapped-deployment.md
  model-packaging.md
  docker-packaging.md
  troubleshooting.md
```

---

# 70. README Positioning

The README should clearly communicate:

> OfflineAI packages complete AI workloads for deployment into air-gapped environments.

Example:

```text
Build online.
Transfer once.
Run offline.
```

Example workflow:

```bash
offlineai build .
```

Transfer:

```text
my-ai-app-1.0.0.offlineai
```

On air-gapped server:

```bash
offlineai verify my-ai-app-1.0.0.offlineai
offlineai import my-ai-app-1.0.0.offlineai
offlineai install my-ai-app
```

---

# 71. Open-Source Licensing

Choose a permissive open-source license unless there is a strong reason otherwise.

Candidate:

```text
Apache-2.0
```

The project should not include dependencies whose licenses are incompatible with the selected license.

Document third-party licenses.

---

# 72. Implementation Phases

## Phase 1 — Foundation

Implement:

- project structure
- CLI
- config
- package schema
- manifest
- archive format
- SHA-256
- build
- verify
- inspect

Example:

```bash
offlineai build .
offlineai verify bundle.offlineai
offlineai inspect bundle.offlineai
```

---

## Phase 2 — Docker

Implement:

- Docker image discovery
- Docker image pull
- Docker save
- Docker load
- image digest
- runtime start/stop/status

---

## Phase 3 — Models

Implement:

- Hugging Face source
- model download
- model cache
- resumable downloads
- model metadata
- model checksums

---

## Phase 4 — Python

Implement:

- requirements resolution
- wheel collection
- offline pip installation
- Python compatibility checks

---

## Phase 5 — Hardware

Implement:

- CPU detection
- RAM
- disk
- NVIDIA GPU
- NVIDIA driver
- VRAM
- Docker GPU runtime

---

## Phase 6 — Installation Engine

Implement:

- dependency checking
- transaction tracking
- installation
- rollback
- health checks
- logs

---

## Phase 7 — Registry

Implement:

- SQLite registry
- import
- list
- search
- remove
- package versions

---

## Phase 8 — Security

Implement:

- signature support
- SBOM
- secret detection
- secure extraction
- provenance
- strict offline mode

---

## Phase 9 — Advanced Features

Implement:

- dependency graph
- package diff
- JSON output
- plugin architecture
- OCI-native support
- Podman support
- web API

---

# 73. MVP Definition

The MVP is complete when this exact scenario works.

## Online machine

Create:

```text
examples/hello-ai/
```

Run:

```bash
offlineai build examples/hello-ai
```

Produce:

```text
hello-ai-1.0.0.offlineai
```

Move the file to a machine with:

```text
NO INTERNET
```

Run:

```bash
offlineai verify hello-ai-1.0.0.offlineai
```

Then:

```bash
offlineai import hello-ai-1.0.0.offlineai
```

Then:

```bash
offlineai install hello-ai
```

Then:

```bash
offlineai status hello-ai
```

Then:

```bash
offlineai logs hello-ai
```

The application must run successfully without network access.

---

# 74. Critical Design Rule

The most important principle:

> A successful build must mean that the resulting bundle contains everything required for the declared offline installation.

Do not create a bundle that silently depends on:

```text
PyPI
Hugging Face
Docker Hub
APT repositories
GitHub
DNS
external APIs
```

during installation.

If something is required and absent, installation must fail clearly.

---

# 75. Important Distinction: Build vs Runtime

OfflineAI has two environments.

## Build environment

Usually internet-connected:

```text
Internet
   ↓
OfflineAI Builder
   ↓
Download/resolve dependencies
   ↓
Create bundle
```

## Target environment

Air-gapped:

```text
USB / secure transfer
       ↓
OfflineAI
       ↓
Verify
       ↓
Install
       ↓
Run
```

The target environment must not depend on the build environment.

---

# 76. Future Vision

The long-term goal is to make OfflineAI a standard distribution mechanism for self-hosted AI.

A developer could publish:

```text
my-rag-system-1.4.0.offlineai
```

A company could download it once on an approved connected machine, scan/verify it, transfer it into the secure environment, and install it without manually reconstructing the dependency stack.

Eventually:

```text
Open-source AI application
        ↓
OfflineAI package
        ↓
Artifact registry
        ↓
Secure transfer
        ↓
Air-gapped environment
        ↓
One-command installation
```

Potential ecosystem:

```text
OfflineAI Registry
        |
        +-- LLM packages
        +-- Vision packages
        +-- Speech packages
        +-- RAG packages
        +-- Agent packages
        +-- Embedding models
        +-- Rerankers
        +-- AI infrastructure
```

---

# 77. Code Generation Instructions

You are implementing a real open-source project, not a proof-of-concept script.

Follow these principles:

1. Use clean modular architecture.
2. Use type hints.
3. Use structured exceptions.
4. Validate all user input.
5. Write tests alongside implementation.
6. Do not hard-code absolute paths.
7. Do not hard-code one model/provider.
8. Do not assume internet connectivity.
9. Never silently download dependencies during offline installation.
10. Never silently ignore verification failures.
11. Never log credentials or secrets.
12. Make the implementation Linux-first for the initial runtime.
13. Keep provider-specific functionality behind interfaces.
14. Keep Docker-specific code behind a runtime abstraction.
15. Keep artifact sources behind source interfaces.
16. Make the bundle format versioned.
17. Maintain backward compatibility where practical.
18. Provide useful error messages.
19. Document non-obvious implementation decisions.
20. Prefer simple, maintainable code over premature abstraction.

---

# 78. Definition of Done

The first production-quality release should have:

- [ ] CLI
- [ ] Package specification
- [ ] Package validation
- [ ] Dependency resolution
- [ ] Bundle creation
- [ ] Bundle inspection
- [ ] SHA-256 verification
- [ ] Docker image packaging
- [ ] Docker image loading
- [ ] Hugging Face model packaging
- [ ] Python wheel packaging
- [ ] SQLite local registry
- [ ] Offline installation
- [ ] Hardware detection
- [ ] NVIDIA GPU detection
- [ ] Disk-space validation
- [ ] Runtime management
- [ ] Health checks
- [ ] Transaction tracking
- [ ] Basic rollback
- [ ] Strict offline mode
- [ ] SBOM
- [ ] Secret detection
- [ ] Secure archive extraction
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Example packages
- [ ] JSON CLI output
- [ ] CI pipeline

---

# 79. Suggested Initial Commands

The final CLI should approximately support:

```bash
offlineai init

offlineai build .
offlineai verify bundle.offlineai
offlineai inspect bundle.offlineai

offlineai import bundle.offlineai

offlineai list
offlineai search qwen
offlineai info qwen3-30b

offlineai check qwen3-30b
offlineai doctor

offlineai install qwen3-30b
offlineai uninstall qwen3-30b

offlineai start qwen3-30b
offlineai stop qwen3-30b
offlineai restart qwen3-30b

offlineai status qwen3-30b
offlineai logs qwen3-30b

offlineai graph qwen3-30b

offlineai sbom bundle.offlineai

offlineai sign bundle.offlineai
offlineai verify-signature bundle.offlineai
```

---

# 80. Final Product Principle

OfflineAI should make this:

```text
"How do I move this entire AI application
into an air-gapped server?"
```

a straightforward engineering problem.

Instead of manually transferring:

```text
Docker images
+
model files
+
Python wheels
+
OS packages
+
configuration
+
scripts
+
documentation
+
checksums
+
licenses
```

the developer should be able to produce:

```text
my-ai-application-1.0.0.offlineai
```

and the administrator should be able to:

```bash
offlineai verify my-ai-application-1.0.0.offlineai
offlineai install my-ai-application
```

with confidence that the complete declared runtime is available locally.

The project should prioritize **reliability, reproducibility, security, transparency, and offline-first operation** over feature count.
