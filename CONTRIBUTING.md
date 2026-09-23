# Contributing

## Setting up

```bash
uv venv --python 3.12
uv pip install -e ".[dev,builder]"
```

## The checks

```bash
ruff check src tests
ruff format --check src tests
mypy
OFFLINEAI_TEST_OFFLINE=1 pytest
```

All four must pass. `mypy` runs in strict mode.

The end-to-end air-gap tests need Docker and are opt-in:

```bash
pytest -m airgap_e2e
```

## The rule that matters most

**The test suite is air-gapped.** A fixture patches sockets so any outbound
call fails a test. If you see:

```
NetworkAccessDuringTestError: Blocked network access during tests
```

that is not a flaky test. It means code on a path that must work offline tried
to reach the network. Fix the code, not the test.

If a test genuinely needs the network — it is exercising a builder-side source
— mark it `@pytest.mark.allow_network`. Under `OFFLINEAI_TEST_OFFLINE=1` such
tests are skipped rather than run, which is how CI proves the whole suite
passes with nothing available.

## Dependencies

If you import it, declare it. Do not rely on something arriving transitively.

The core dependency set is what installs on the **air-gapped target**, so keep
it minimal and free of anything network-facing. Builder-only libraries go in
the `builder` extra and must be imported lazily, inside the function that needs
them, behind an `ImportError` that tells the operator to install the extra.

`tests/unit/test_dependency_declaration.py` enforces both rules statically. It
exists because this went wrong once: `click` was imported directly but never
declared, arrived transitively through a *builder-only* extra, and a core
install crashed on every invocation — which would have surfaced on the
isolated machine, where it cannot be fixed.

## Adding an artifact source

Implement `expand` and `fetch` from `offlineai.artifacts.base.ArtifactSource`,
then register an entry point:

```toml
[project.entry-points."offlineai.sources"]
modelscope = "your_package:ModelScopeSource"
```

`expand` turns one declaration into the concrete files it implies — one model
reference is frequently dozens of files — and `fetch` obtains one of them,
using and populating the cache.

Sources are **build-time only**. Nothing on the target ever calls one; a target
that needs a source is a bundle that was built wrong.

## Style

- Type hints everywhere; `mypy --strict` must pass.
- Errors subclass `OfflineAIError` and carry an exit code, a reason, and an
  action the operator can take. `Error 500` is not acceptable; specification
  section 64 has the shape.
- Comments explain *why*, not *what*. If a decision looks odd, say what the
  obvious alternative was and why it was rejected.
- Tests state the property being protected, not the mechanics. A test named
  `test_an_eighty_gb_card_satisfies_an_eighty_gb_requirement` is more useful
  than `test_vram_check`.

## Before opening a pull request

Run the full flow at least once by hand. Several real bugs in this project's
history were invisible to the test suite and obvious within thirty seconds of
running the tool:

```bash
offlineai build examples/hello-ai
offlineai verify hello-ai-1.0.0.offlineai
offlineai import hello-ai-1.0.0.offlineai
offlineai install hello-ai
curl localhost:8000/health
offlineai uninstall hello-ai --yes
```
