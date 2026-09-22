"""Resumable HTTP downloads (sections 43 and 46).

A single model checkpoint can be hundreds of gigabytes. On a link that drops,
restarting from zero is not merely slow - it can mean the build never finishes
at all. So downloads land in a stable partial file and resume with a Range
request from wherever they stopped.

Two safety properties matter more than speed:

* a partial file is never promoted to the cache, so an interrupted build cannot
  leave a truncated artifact that a later run mistakes for a good one;
* when the origin states a digest, it is checked before the bytes are accepted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from offlineai.errors import SourceError, StrictOfflineViolationError
from offlineai.logging import get_logger
from offlineai.utils.hashing import sha256_file
from offlineai.utils.sizes import format_bytes

if TYPE_CHECKING:
    import httpx

__all__ = ["DownloadResult", "ProgressCallback", "download_resumable"]

logger = get_logger("artifacts.download")

#: ``(downloaded_bytes, total_bytes_or_None)``
ProgressCallback = Callable[[int, int | None], None]

_CHUNK = 1024 * 1024


class ClientFactory(Protocol):
    def __call__(self) -> httpx.Client: ...


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    size: int
    sha256: str
    resumed_from: int = 0
    #: Origin-supplied validator, recorded for provenance.
    etag: str | None = None


def download_resumable(
    url: str,
    partial: Path,
    *,
    client: httpx.Client,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    headers: dict[str, str] | None = None,
    progress: ProgressCallback | None = None,
    offline: bool = False,
) -> DownloadResult:
    """Download ``url`` into ``partial``, resuming an earlier attempt if present.

    Returns once the complete file is on disk and verified. The caller is
    responsible for moving it into content-addressed storage.
    """
    if offline:
        # Belt and braces. Strict-offline is enforced at the pipeline level
        # too, but a source that reaches this far has a bug, and failing here
        # is much better than a silent egress.
        raise StrictOfflineViolationError(
            f"a download was attempted while strict-offline mode is enabled: {url}",
            action="Rebuild the bundle on a connected machine with this artifact "
            "included, then transfer it again.",
        )

    partial.parent.mkdir(parents=True, exist_ok=True)
    already = partial.stat().st_size if partial.is_file() else 0

    request_headers = dict(headers or {})
    if already:
        request_headers["Range"] = f"bytes={already}-"
        logger.info("resuming %s from %s", url, format_bytes(already))

    mode = "ab" if already else "wb"
    resumed_from = already

    try:
        with client.stream("GET", url, headers=request_headers, follow_redirects=True) as response:
            if already and response.status_code == 200:
                # The server ignored the Range header, so it is sending the
                # whole file. Start over rather than appending to what we have,
                # which would corrupt the result.
                logger.info("server ignored Range; restarting download of %s", url)
                mode = "wb"
                already = 0
                resumed_from = 0
            elif already and response.status_code == 416:
                # Already have the whole thing.
                response.close()
                return _finalise(partial, expected_sha256, expected_size, resumed_from, None)
            elif response.status_code not in (200, 206):
                raise SourceError(
                    f"download failed with HTTP {response.status_code}",
                    details={"URL": url},
                    action=_http_advice(response.status_code),
                )

            total = _total_size(response, already)
            etag = response.headers.get("etag")

            downloaded = already
            with partial.open(mode) as handle:
                for chunk in response.iter_bytes(_CHUNK):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)

    except SourceError:
        raise
    except Exception as exc:
        # The partial file is deliberately left in place: it is what makes the
        # next attempt a resume rather than a restart.
        raise SourceError(
            f"download failed: {url}",
            details={"Detail": f"{type(exc).__name__}: {exc}"},
            action="Check connectivity and re-run the build; the partial download "
            "will resume where it stopped.",
        ) from exc

    return _finalise(partial, expected_sha256, expected_size, resumed_from, etag)


def _finalise(
    partial: Path,
    expected_sha256: str | None,
    expected_size: int | None,
    resumed_from: int,
    etag: str | None,
) -> DownloadResult:
    size = partial.stat().st_size

    if expected_size is not None and size != expected_size:
        partial.unlink(missing_ok=True)
        raise SourceError(
            "downloaded artifact has the wrong size",
            details={"Expected": str(expected_size), "Actual": str(size)},
            action="The download was truncated or the origin changed. Re-run the build.",
        )

    digest = sha256_file(partial)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        # Delete: a file that fails its digest must not be resumable, or the
        # next attempt would append to known-bad bytes forever.
        partial.unlink(missing_ok=True)
        raise SourceError(
            "downloaded artifact does not match the digest the origin stated",
            details={"Expected": expected_sha256, "Actual": digest},
            action="The download was corrupted in transit, or the artifact changed "
            "at the source. Re-run the build.",
        )

    return DownloadResult(
        path=partial, size=size, sha256=digest, resumed_from=resumed_from, etag=etag
    )


def _total_size(response: Any, already: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        tail = content_range.rsplit("/", 1)[-1]
        if tail.isdigit():
            return int(tail)
    length = response.headers.get("content-length")
    if length and length.isdigit():
        return int(length) + already
    return None


def _http_advice(status: int) -> str:
    if status in (401, 403):
        return (
            "Access was denied. If this repository is gated or private, set the "
            "appropriate token in the environment on the builder machine."
        )
    if status == 404:
        return "The artifact does not exist at that location. Check the reference."
    if status == 429:
        return "The origin is rate limiting. Reduce --workers and try again."
    if 500 <= status < 600:
        return "The origin reported a server error. Try again shortly."
    return "Check the reference and the builder's network access."
