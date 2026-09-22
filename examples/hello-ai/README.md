# hello-ai

The minimal end-to-end OfflineAI example: a containerised HTTP service with no
model and no GPU requirement.

## On the connected builder

```bash
offlineai build examples/hello-ai
```

Produces `hello-ai-1.0.0.offlineai`.

## On the air-gapped target

```bash
offlineai verify  hello-ai-1.0.0.offlineai
offlineai import  hello-ai-1.0.0.offlineai
offlineai install hello-ai --strict-offline
offlineai status  hello-ai
```

Then:

```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

No step above requires network access.
