# Architecture

OfflineAI has two environments and one artifact between them.

```
     CONNECTED BUILDER                     AIR-GAPPED TARGET
     ─────────────────                     ─────────────────
     offlineai build                       offlineai verify
            │                                     │
       resolve artifacts                    offlineai import
            │                                     │
       content-addressed cache            content-addressed store
            │                                     │
       manifest + hashes                   offlineai install
            │                                     │
       .offlineai bundle  ───── transfer ──▶  container runtime
```

The target never depends on the builder. That is the property everything else
serves.

## Layers

```
                        CLI  (Typer)
             thin; returns typed results, renders them
                             │
        ┌────────────────────┼────────────────────┐
   Package Engine                          Environment Engine
   resolver │ bundler                      hardware │ runtime
        │                                       detection
        ▼
   Artifact Manager  ──  ArtifactSource plugins
   cache │ hashing        local │ http │ huggingface │ oci │ pypi
        │
        ▼
   Bundle Builder → manifest + hashes → .offlineai
   ══════════════════════ air gap ══════════════════════
   Registry (SQLite + CAS) → Installer → Runtime (Docker)
```

Two rules keep the layers honest:

- **The CLI never touches an implementation detail.** Commands build a typed
  result model and hand it to one renderer, which produces human text or JSON.
  The alternative — `if json_output:` scattered through twenty-five modules —
  reliably produces JSON that drifts from the human output.
- **Anything provider-specific lives behind an interface.** Artifact sources
  behind `ArtifactSource`, container engines behind `ContainerRuntime`. Adding
  Podman or ModelScope means writing an implementation, not editing the CLI.

## Decisions worth knowing

### Member ordering is part of the bundle format

Metadata is written first and `artifacts/` last. That is what lets `inspect`,
`check` and `verify-signature` read a few kilobytes from the front of a 62 GB
file instead of scanning it. Measured on a 40 MB bundle: 10.5 KB read, 0.026%
— and the cost is constant, not proportional.

Section 60 of the specification requires `inspect` to work without installing
anything. At these sizes that is only affordable if the read genuinely stops
early, so the test suite asserts it by counting bytes pulled off the file.

### Import streams; it never extracts

A naive `tar -xf` needs twice the bundle size free before it can start. Import
hashes each member while writing it straight into
`artifacts/sha256/ab/abcdef…`, so it needs room for the artifacts and nothing
more. Content addressing also gives deduplication for free: two packages
sharing a vLLM image store it once.

### The manifest is the root of trust

`manifest.yaml` records a SHA-256 for every artifact. A signature over the
manifest therefore covers the whole payload transitively, which is what keeps
`verify-signature` header-only and instant.

Signing covers the *canonical JSON serialisation* of the manifest, not the
stored YAML bytes. Reformatting cannot invalidate a signature; reordering keys
cannot forge one.

### Completeness is structural, not aspirational

`BundleWriter` refuses to close if any artifact the manifest declares was not
written, and verifies each artifact's content against the manifest as it goes.
A build cannot succeed and produce a bundle missing something it promised.

That is specification section 74 — *a successful build must mean the bundle
contains everything required* — turned into an invariant rather than a hope.

### Rollback records inverses as it goes

Each installation step journals how to undo itself, in SQLite, at the moment
it completes. Rollback replays those in reverse.

Recording rather than inferring matters: after a failure the system is in an
unknown state, and "what did we actually do?" is not a question to answer by
guessing. The journal also survives the process being killed.

The inverse action set is closed — remove container, remove network, remove
image, remove path — and path removal is confined to the install tree. A
tampered journal cannot turn a failed install into a way to delete arbitrary
files.

### Unevaluated is never "pass"

Every compatibility check is `PASS`, `FAIL` or `SKIPPED(reason)`. A GPU check
on a host with no `nvidia-smi` is `SKIPPED`, never `PASS`. And a passing report
says *"Requirements satisfied. Runtime success is not guaranteed."* — nothing
stronger, because nothing stronger is true.

## Module map

| Module | Responsibility |
|---|---|
| `schema/` | Pydantic models for `offlineai.yaml`, the manifest, overrides |
| `resolver/` | Loading and validating a package definition |
| `artifacts/` | The source interface, the content-addressed cache, resumable downloads |
| `bundler/` | Archive read/write, build, verify, inspect, diff, graph |
| `registry/` | SQLite index and refcounted content-addressed storage |
| `installer/` | The install pipeline, transactions, rollback |
| `runtime/` | Container engine abstraction, Docker, an in-memory fake |
| `hardware/` | Detection and requirement matching |
| `security/` | Safe extraction, signing, secret detection, offline enforcement |
| `sbom/` | CycloneDX generation and licence reporting |
| `cli/` | Commands and the single output renderer |

## Testing

The suite is air-gapped by default: a pytest fixture patches sockets so any
outbound call fails a test. `OFFLINEAI_TEST_OFFLINE=1` additionally skips the
few tests that opt out, proving the whole suite passes with no network.

`tests/airgap/` goes further and runs inside `docker run --network none`,
removing the network at the kernel level. It installs OfflineAI there from a
wheelhouse — using the same `pip install --no-index --find-links` mechanism
the tool prescribes for packaged workloads — and verifies and imports a bundle.

`FakeRuntime` ships in the package rather than the test tree: it is the
reference for what `ContainerRuntime` requires, and it lets the whole install
pipeline run in CI with no daemon. Its `run_container` refuses an image that
was never loaded, mirroring `--pull never`, so a test cannot pass by
accidentally reaching out.
