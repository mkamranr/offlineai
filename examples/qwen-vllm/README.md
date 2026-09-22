# qwen-vllm

A production LLM server for an air-gapped environment: Qwen3-30B served by
vLLM with an OpenAI-compatible API. Roughly 62 GB built.

## Build (connected)

```bash
offlineai build examples/qwen-vllm --sign-key signing-key.pem --signer "ops@example"
```

Expect this to take a while and to need ~130 GB free: the weights are
downloaded once into the cache, then streamed into the bundle.

## Deploy (air-gapped)

```bash
offlineai verify-signature qwen-vllm-1.0.0.offlineai --key signing-key.pub
offlineai check   qwen-vllm-1.0.0.offlineai       # reads only the header
offlineai import  qwen-vllm-1.0.0.offlineai
offlineai install qwen-vllm --gpus 0,1 --strict-offline
offlineai status  qwen-vllm
```

## Multi-GPU

`--gpus 0,1` exposes the devices. It does **not** configure tensor
parallelism — that is an application setting, and OfflineAI deliberately does
not guess at it. Set it per site:

```yaml
# site-config.yaml
services:
  vllm:
    command:
      - --model
      - /models/model
      - --tensor-parallel-size
      - "2"
      - --max-model-len
      - "32768"
```

```bash
offlineai install qwen-vllm --gpus 0,1 --config site-config.yaml
```

## Why the excludes matter

The repository ships both `.safetensors` and `.bin` copies of the same
tensors. Taking both would add tens of gigabytes to something already being
carried on removable media. `offlineai graph qwen-vllm` shows exactly what was
packaged.
