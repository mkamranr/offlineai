"""Hugging Face model source (sections 13 and 14).

The governing constraint is section 13: **a model is not one file.** A single
checkpoint is a set of sharded safetensors plus a config, a tokenizer, a
generation config and assorted special-token files. Packaging only the weights
produces a bundle that installs cleanly and then fails at first inference on a
machine where nothing can be downloaded to fix it.

So expansion enumerates the repository and takes everything, minus an explicit
exclude list. The default exclusions drop duplicate weight formats - a repo
commonly ships both ``.safetensors`` and ``.bin`` of the same tensors, and
carrying both can double a 48 GB bundle for no benefit.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from offlineai import layout
from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact, SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.download import download_resumable
from offlineai.errors import SourceError
from offlineai.logging import get_logger
from offlineai.progress import ProgressReporter, download_callback
from offlineai.schema.manifest import ArtifactType

if TYPE_CHECKING:
    import httpx

__all__ = ["DEFAULT_EXCLUDES", "HuggingFaceSource", "RepoFile"]

logger = get_logger("artifacts.huggingface")

#: Skipped unless a package explicitly includes them.
#:
#: PyTorch .bin and TensorFlow/Flax weights are usually duplicates of the
#: safetensors already being taken. Everything else here is repository
#: furniture that no runtime reads.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    "*.msgpack",
    "*.h5",
    "*.ot",
    "pytorch_model*.bin",
    "tf_model*.h5",
    "flax_model*.msgpack",
    ".gitattributes",
    "README.md",
    "*.md",
    ".git*",
)

#: Always taken, even if an exclude pattern would otherwise drop them. Missing
#: any of these is what turns a packaged model into an unusable one.
ALWAYS_INCLUDE: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "*.model",
    "*.safetensors",
    "*.safetensors.index.json",
)


@dataclass(frozen=True, slots=True)
class RepoFile:
    path: str
    size: int | None = None
    sha256: str | None = None


class HuggingFaceSource:
    kind: ClassVar[str] = "huggingface"

    def __init__(
        self,
        *,
        api: Any = None,
        client: httpx.Client | None = None,
        token: str | None = None,
        endpoint: str = "https://huggingface.co",
    ) -> None:
        self._api = api
        self._client = client
        self.endpoint = endpoint.rstrip("/")
        # Read from the environment rather than any config file: a token is a
        # credential and must never end up written into a bundle or a settings
        # file (section 40).
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # -- expansion -------------------------------------------------------

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        repo = ref.locator
        revision = str(ref.options.get("revision") or "main")
        include = _as_patterns(ref.options.get("include"))
        exclude = _as_patterns(ref.options.get("exclude")) or list(DEFAULT_EXCLUDES)

        files = self.list_files(repo, revision)
        if not files:
            raise SourceError(
                f"the Hugging Face repository {repo!r} appears to contain no files",
                action="Check the repository id and revision.",
            )

        selected = [f for f in files if _wanted(f.path, include, exclude)]
        if not selected:
            raise SourceError(
                f"no files in {repo!r} matched the include/exclude patterns",
                details={
                    "Include": ", ".join(include) or "(everything)",
                    "Exclude": ", ".join(exclude) or "(nothing)",
                },
                action="Relax the patterns in your package definition.",
            )

        _warn_if_shards_incomplete(selected, repo)

        return [
            ArtifactRequest(
                id=f"model-{ref.name}-{file.path}".replace("/", "-"),
                artifact_type=ArtifactType.MODEL,
                bundle_path=layout.artifact_path(ArtifactType.MODEL, ref.name, file.path),
                locator=f"{repo}@{revision}/{file.path}",
                source_kind=self.kind,
                expected_sha256=file.sha256,
                expected_size=file.size,
                metadata={
                    "repo": repo,
                    "revision": revision,
                    "file": file.path,
                    "model": ref.name,
                },
            )
            for file in selected
        ]

    def list_files(self, repo: str, revision: str) -> list[RepoFile]:
        """Enumerate a repository, with sizes and digests where available."""
        api = self._hf_api()
        try:
            info = api.repo_info(repo_id=repo, revision=revision, files_metadata=True)
        except Exception as exc:
            raise SourceError(
                f"cannot read the Hugging Face repository {repo!r}",
                details={"Revision": revision, "Detail": f"{type(exc).__name__}: {exc}"},
                action="Check the repository id, the revision, and - if it is gated "
                "or private - that HF_TOKEN is set on the builder.",
            ) from exc

        out: list[RepoFile] = []
        for sibling in getattr(info, "siblings", None) or []:
            path = getattr(sibling, "rfilename", None)
            if not path:
                continue
            lfs = getattr(sibling, "lfs", None)
            # Large files carry a sha256 in their LFS pointer; small ones only
            # have a git blob id, which is not a content hash of the file.
            digest = None
            if lfs is not None:
                digest = getattr(lfs, "sha256", None) or (
                    lfs.get("sha256") if isinstance(lfs, dict) else None
                )
            out.append(
                RepoFile(
                    path=path,
                    size=getattr(sibling, "size", None),
                    sha256=digest,
                )
            )
        return out

    # -- fetching --------------------------------------------------------

    def fetch(
        self,
        request: ArtifactRequest,
        cache: ArtifactCache,
        *,
        progress: ProgressReporter | None = None,
    ) -> ResolvedArtifact:
        repo = str(request.metadata["repo"])
        revision = str(request.metadata["revision"])
        filename = str(request.metadata["file"])
        url = f"{self.endpoint}/{repo}/resolve/{revision}/{filename}"

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        cache.ensure()
        with cache.partial(f"{self.kind}:{request.locator}") as staging:
            result = download_resumable(
                url,
                staging,
                client=self._http(),
                expected_sha256=request.expected_sha256,
                expected_size=request.expected_size,
                headers=headers,
                progress=(download_callback(progress, request.id) if progress else None),
            )
            entry = cache.store_file(result.path, move=True)

        cache.put_ref(self.kind, request.cache_key, entry.sha256)
        return ResolvedArtifact(
            request=request,
            local_path=entry.path,
            sha256=entry.sha256,
            size=entry.size,
            source=f"hf://{repo}@{revision}/{filename}",
        )

    def license_for(self, repo: str, revision: str = "main") -> str | None:
        """Best-effort license id, recorded as informational metadata only.

        Section 37 is explicit that OfflineAI makes no legal claims; this is
        provenance, not advice.
        """
        try:
            info = self._hf_api().repo_info(repo_id=repo, revision=revision)
        except Exception:  # noqa: BLE001 - metadata is optional by design
            return None
        card = getattr(info, "cardData", None) or {}
        value = card.get("license") if isinstance(card, dict) else None
        return str(value) if value else None

    # -- helpers ---------------------------------------------------------

    def _hf_api(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise SourceError(
                    "packaging Hugging Face models requires the builder extra",
                    action="Install it on the builder machine:\n"
                    "  pip install 'offlineai[builder]'\n"
                    "It is not needed on the air-gapped target.",
                ) from exc
            self._api = HfApi(endpoint=self.endpoint, token=self.token)
        return self._api

    def _http(self) -> httpx.Client:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise SourceError(
                    "packaging Hugging Face models requires the builder extra",
                    action="Install it on the builder machine:\n  pip install 'offlineai[builder]'",
                ) from exc
            self._client = httpx.Client(timeout=httpx.Timeout(30.0, read=300.0))
        return self._client


def _as_patterns(value: object) -> list[str]:
    """Accept a single pattern or a list of them, as YAML allows either."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _wanted(path: str, include: list[str], exclude: list[str]) -> bool:
    name = path.rsplit("/", 1)[-1]
    if include:
        return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p) for p in include)
    if any(fnmatch.fnmatch(name, p) for p in ALWAYS_INCLUDE):
        return True
    return not any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p) for p in exclude)


def _warn_if_shards_incomplete(files: list[RepoFile], repo: str) -> None:
    """Check that a sharded checkpoint is complete.

    ``model-00003-of-00008.safetensors`` announces how many shards there should
    be. A bundle missing one is useless, and the failure would otherwise
    surface on the air-gapped side at first load.
    """
    import re

    pattern = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")
    groups: dict[int, set[int]] = {}
    for file in files:
        match = pattern.search(file.path)
        if match:
            total = int(match.group(2))
            groups.setdefault(total, set()).add(int(match.group(1)))

    for total, present in groups.items():
        missing = sorted(set(range(1, total + 1)) - present)
        if missing:
            raise SourceError(
                f"the sharded checkpoint in {repo!r} is incomplete",
                details={
                    "Expected shards": str(total),
                    "Missing": ", ".join(f"{m:05d}-of-{total:05d}" for m in missing),
                },
                action="A partial checkpoint cannot be loaded offline. Check the "
                "include/exclude patterns in your package definition, or the "
                "repository itself.",
            )
