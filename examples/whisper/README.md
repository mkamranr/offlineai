# whisper

Speech-to-text, packaged for an isolated environment. Around 8 GB built.

The GPU is declared `required: false`: whisper.cpp runs on CPU, slower. That
distinction matters on a target where you cannot simply add a card — OfflineAI
will warn rather than refuse.

```bash
# Connected builder
offlineai build examples/whisper --sign-key signing-key.pem

# Air-gapped target
offlineai verify-signature whisper-2.0.0.offlineai --key signing-key.pub
offlineai check   whisper-2.0.0.offlineai
offlineai import  whisper-2.0.0.offlineai
offlineai install whisper --strict-offline
```

Drop audio into `./audio` — it is mounted at `/audio` in the container.
