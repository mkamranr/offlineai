# Model packaging

## A model is not one file

This is the single most important thing on this page. A checkpoint is weights
**plus** a config, a tokenizer, a generation config, special-token maps and
often a shard index. Packaging only the weights produces a bundle that installs
cleanly and then fails at first inference on a machine where nothing can be
downloaded to fix it.

OfflineAI therefore enumerates the whole repository and takes everything except
an explicit exclude list, with the files a runtime cannot start without pinned
as always-included regardless of patterns.

```bash
offlineai graph my-package
```

```
my-package 1.0.0
└── models  (487.8 kB)
    └── tiny  (487.8 kB)  [7 file(s), from hf://hf-internal-testing/tiny-random-gpt2@main]
```

Seven files, for a model people would describe as one.

## Declaring a model

```yaml
models:
  - name: model
    source:
      type: huggingface
      repo: Qwen/Qwen3-30B
      revision: main
    destination: /models/model
```

`destination` is where it is mounted inside the container, **read-only** — a
workload should not be able to modify the weights it was shipped.

## Pin the revision

```yaml
    revision: 5a7c1b4e9f2d8c3a1b0e7f6d5c4b3a2918273645
```

`main` is mutable. In six months, "the model we tested" has to still mean
something.

## Excludes

Repositories frequently ship the same tensors in several formats. Taking all
of them can double a 48 GB bundle for no benefit.

```yaml
    exclude:
      - "*.bin"          # PyTorch pickle format, duplicate of safetensors
      - "*.msgpack"      # Flax
      - "*.h5"           # TensorFlow
```

Those three are excluded by default, along with `README.md` and `.gitattributes`.
Supplying your own `exclude` replaces the defaults entirely.

Files a runtime cannot start without — `config.json`, `tokenizer.json`,
`*.safetensors`, the shard index and friends — are always included, whatever
the patterns say.

## Sharded checkpoints

```
model-00001-of-00008.safetensors
...
model-00008-of-00008.safetensors
```

The filename announces how many shards there should be, so OfflineAI checks
and **fails the build** if any are missing:

```
Reason:
the sharded checkpoint in 'Qwen/Qwen3-30B' is incomplete

Expected shards:
8

Missing:
00003-of-00008
```

This also fires when an over-narrow `include` pattern is what dropped them.
Discovering a missing shard on the builder costs a rebuild; discovering it on
the air-gapped host costs a transfer cycle.

## Gated and private repositories

```bash
export HF_TOKEN=hf_...
offlineai build .
```

Read from the environment only, never from a config file, so a credential
cannot be written into a bundle. The token is used on the builder and never
travels.

## Resumable downloads

A 200 GB checkpoint on a link that drops is the normal case, not the edge case.
Downloads land in a stable partial file and resume with a Range request:

```bash
offlineai build .          # interrupted at 48%
offlineai build .          # resumes at 48%
```

Three behaviours worth knowing:

- a server that ignores `Range` and sends the whole body causes a clean
  restart, rather than appending to what was already there and silently
  corrupting the file;
- a file that fails its declared digest is **deleted**, not kept, so the next
  attempt cannot resume from known-bad bytes forever;
- a file interrupted by a network error is kept, because that is what makes
  the next attempt a resume.

## Concurrency

Downloads run in parallel. The default is 4, which section 45 asks to keep
conservative so a build does not saturate storage or the network:

```bash
offlineai build . --workers 8      # or -j 8
```

Or permanently:

```yaml
# ~/.offlineai/config.yaml
downloads:
  workers: 8
```

Concurrency never changes the bundle. Results are reassembled in declaration
order whatever order they finish in, so the manifest — and therefore the
bundle's hash — is identical at any worker count. There is a test that builds
the same package at 1, 2, 4 and 16 workers and compares manifests.

Container images are deliberately fetched serially. `docker save` writes
gigabytes through the daemon, which serialises much of it anyway, and two
concurrent saves mostly produce disk contention. The win is in models, where
there are many files and the bottleneck is the network.

## Caching across bundles

Artifacts are cached by content hash in `~/.offlineai/cache`, so a model used
by several packages is downloaded once.

```
[3/10] Resolving models    OK  7 model file(s), 7 from cache
```

```bash
offlineai --cache-dir /mnt/big/cache build .   # if the cache needs to live elsewhere
```

## Runtime must not fetch

Package the model, then tell the runtime not to look for it:

```yaml
environment:
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
```

Without these, a library that cannot find something locally will try to fetch
it. On a host with no route out, that usually means a long hang rather than a
clear failure — much harder to diagnose than an error.

`offlineai install --strict-offline` sets them anyway, but declaring them in
the package documents the intent and protects anyone who runs the containers
by hand.

## Local models

Weights produced in-house need no hub:

```yaml
models:
  - name: scoring-model
    source:
      type: local
      path: ./model
    destination: /models/scoring
```

The whole directory is packaged, each file hashed individually. See
`examples/simple-python`.
