# Quickstart

Ten minutes, start to finish. You need Docker on the builder; the target needs
Docker and nothing else.

## Install

On the **builder** (the machine with internet access):

```bash
pip install 'offlineai[builder]'
```

On the **target** (the isolated machine), install the core package only — it
has no network-facing dependencies:

```bash
pip install offlineai
```

If the target cannot reach PyPI either, which is the usual case, bring a
wheelhouse across with everything else:

```bash
# builder
pip download offlineai --dest wheelhouse \
  --only-binary=:all: --platform manylinux_2_17_x86_64 \
  --python-version 3.12 --implementation cp --abi cp312

# target
pip install --no-index --find-links wheelhouse offlineai
```

## 1. Set up

```bash
offlineai init
offlineai doctor
```

`doctor` reports on the host. Anything it cannot evaluate is marked `SKIPPED`
with a reason rather than passed — "we could not look" and "it is fine" are
different answers.

## 2. Build

```bash
offlineai build examples/hello-ai
```

```
[1/10] Validating package definition        OK
[2/10] Scanning for secrets                 OK  none found
...
[10/10] Verifying bundle                    OK  1 artifact(s)

Bundle created:
  hello-ai-1.0.0.offlineai

Size:   131.6 MB
SHA256: 23185389ff17ff38cc330bf8f0f84b3b470b82cffa280205932cf1bd40628334
```

Note the SHA-256. Compare it after the transfer.

## 3. Transfer

Copy the `.offlineai` file by whatever means your environment permits. It is
one file, and it is self-describing.

## 4. Inspect before trusting

```bash
offlineai inspect hello-ai-1.0.0.offlineai
```

This reads about ten kilobytes from the front of the file regardless of its
size, so it is instant even on a 62 GB bundle. Nothing is extracted.

```bash
offlineai check hello-ai-1.0.0.offlineai
```

Tells you whether this host satisfies the bundle's requirements, and what
installing it will cost in disk.

## 5. Verify

```bash
offlineai verify hello-ai-1.0.0.offlineai
```

```
Manifest:  OK
Container: OK
Checksums: OK  (1 artifact(s))

Result: VERIFIED
```

This streams the whole archive and hashes every artifact. It exits `3` if
anything does not match.

## 6. Import and install

```bash
offlineai import  hello-ai-1.0.0.offlineai
offlineai install hello-ai --strict-offline
```

`--strict-offline` makes it structural: nothing may touch the network during
installation, and an attempt fails loudly rather than silently succeeding.

## 7. Run

```bash
offlineai status hello-ai
offlineai logs   hello-ai
curl http://localhost:8000/health
```

```
Package: hello-ai

Status: RUNNING

Services:
  app  RUNNING

Endpoint:
  http://localhost:8000
```

## Adapting to your host

You should not edit a bundle to change a port — that would invalidate its
checksums and signature. Use an override file instead:

```yaml
# site-config.yaml
services:
  app:
    ports:
      - "8099:8000"
environment:
  DEPLOYMENT_SITE: "site-b"
```

```bash
offlineai install hello-ai --config site-config.yaml
```

The bundle is untouched and still verifies.

## Undoing things

```bash
offlineai stop      hello-ai            # stop the services
offlineai uninstall hello-ai            # stop and remove what install created
offlineai rollback  install-20260922-001  # undo one installation exactly
offlineai remove    hello-ai            # drop it from the registry entirely
```

`rollback` replays each installation step's recorded inverse in reverse order.
It never deletes data you own.

## Next

- [docs/package-format.md](docs/package-format.md) — writing `offlineai.yaml`
- [docs/air-gapped-deployment.md](docs/air-gapped-deployment.md) — deploying for real
- [docs/troubleshooting.md](docs/troubleshooting.md) — when something goes wrong
