# rag-stack

The example that shows why OfflineAI is more than a Docker image exporter.

One bundle contains four container images, three models with different jobs
(generation, embedding, reranking), and a resolved Python dependency closure.
Reconstructing that set by hand on an isolated host — in the right versions,
with the right wheels for the right interpreter — is the problem this tool
exists to remove.

```bash
offlineai install rag-stack
```

Roughly 14 GB built, plus model weights.

## What the bundle carries

```bash
offlineai graph rag-stack
```

```
rag-stack 1.1.0
├── containers
│   ├── vllm/vllm-openai:v0.6.3
│   ├── qdrant/qdrant:v1.12.1
│   ├── redis:7-alpine
│   └── offlineai/rag-stack-app:1.1.0
├── models
│   ├── llm          (Qwen3-8B)
│   ├── embeddings   (BAAI/bge-m3)
│   └── reranker     (BAAI/bge-reranker-v2-m3)
└── python dependencies
    └── fastapi, uvicorn, httpx, qdrant-client, redis, …
```

## Why every address is internal

`LLM_BASE_URL`, `QDRANT_URL` and `REDIS_URL` all point at service names inside
the deployment network. Run the audit and you will see them classified as
`LOCAL`, not as runtime dependencies:

```bash
offlineai network-check examples/rag-stack
```

That distinction is the point of the command. A URL pointing outward here
would mean the stack fails on the isolated host, and it is much cheaper to
learn that on the builder.

## Secrets

`RAG_API_KEY` is declared under `secrets.external`. The bundle carries the
**name** so the installer can tell you it is required; it never carries the
value.

```
RAG_API_KEY: REQUIRED
```
