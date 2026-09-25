"""Concurrent artifact fetching (section 45).

Downloads are network-bound, and a checkpoint is routinely a hundred files, so
fetching them one at a time wastes most of the available bandwidth. A thread
pool is the right shape here: the work is I/O, and the GIL is released for the
duration of every socket read and file write.

Three properties matter more than the speed-up:

* **Order is preserved.** The manifest is built from this list, and section 34
  wants bundles that are reproducible, so results come back in request order
  whatever order they complete in.
* **A failure stops the build.** Remaining work is cancelled rather than run to
  completion - there is no point spending twenty minutes downloading artifacts
  for a bundle that is already going to fail.
* **Duplicate requests are fetched once.** Two declarations can name the same
  artifact, and because a resumable download uses a partial file keyed on the
  locator, fetching both at once would have them writing over each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace

from offlineai.artifacts.base import ArtifactRequest, ArtifactSource, ResolvedArtifact
from offlineai.artifacts.cache import ArtifactCache
from offlineai.logging import get_logger
from offlineai.progress import NullReporter, ProgressReporter
from offlineai.utils.sizes import format_bytes

__all__ = ["FetchJob", "fetch_all"]

logger = get_logger("artifacts.fetcher")

#: ``(source, request)``
FetchJob = tuple[ArtifactSource, ArtifactRequest]


def fetch_all(
    jobs: Sequence[FetchJob],
    cache: ArtifactCache,
    *,
    workers: int = 4,
    reporter: ProgressReporter | None = None,
) -> list[ResolvedArtifact]:
    """Fetch every job, concurrently, returning results in request order."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if not jobs:
        return []

    reporter = reporter or NullReporter()

    # Collapse duplicates before scheduling. The first request for a locator
    # does the work; the rest reuse its result with their own request attached,
    # because the bundle path comes from the request rather than the artifact.
    leaders: dict[str, int] = {}
    followers: list[tuple[int, int]] = []
    for index, (_, request) in enumerate(jobs):
        key = request.cache_key
        if key in leaders:
            followers.append((index, leaders[key]))
        else:
            leaders[key] = index

    results: list[ResolvedArtifact | None] = [None] * len(jobs)
    effective = min(workers, len(leaders))

    with ThreadPoolExecutor(max_workers=effective, thread_name_prefix="offlineai-fetch") as pool:
        futures: dict[Future[ResolvedArtifact], int] = {
            pool.submit(_fetch_one, jobs[index], cache, reporter): index
            for index in leaders.values()
        }
        try:
            for future in _as_completed_cancelling(futures, pool):
                results[futures[future]] = future.result()
        except BaseException:
            # Stop the rest promptly; a build that has already failed should
            # not keep pulling gigabytes it will discard.
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    for follower_index, leader_index in followers:
        leader = results[leader_index]
        if leader is None:  # pragma: no cover - the leader would have raised
            continue
        results[follower_index] = replace(leader, request=jobs[follower_index][1])

    return [r for r in results if r is not None]


def _as_completed_cancelling(
    futures: dict[Future[ResolvedArtifact], int], pool: ThreadPoolExecutor
) -> list[Future[ResolvedArtifact]]:
    """Yield futures as they finish, raising on the first failure.

    ``concurrent.futures.as_completed`` would keep the remaining work running
    while the exception unwinds. Waiting on FIRST_EXCEPTION lets us cancel
    instead.
    """
    pending = set(futures)
    completed: list[Future[ResolvedArtifact]] = []
    while pending:
        done, pending = wait(pending, return_when="FIRST_EXCEPTION")
        for future in done:
            if future.exception() is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                raise future.exception()  # type: ignore[misc]
            completed.append(future)
    return completed


def _fetch_one(job: FetchJob, cache: ArtifactCache, reporter: ProgressReporter) -> ResolvedArtifact:
    source, request = job
    key = request.id

    cached = cache.lookup_ref(request.source_kind, request.cache_key)
    if cached is not None and request.expected_sha256 in (None, cached.sha256):
        logger.debug("cache hit for %s", request.bundle_path)
        # Still reported, so a build that is mostly cache hits does not look
        # stalled while it verifies a hundred files.
        reporter.start(key, _describe(request), cached.size)
        reporter.update(key, cached.size, cached.size)
        reporter.finish(key, detail="from cache")
        return ResolvedArtifact(
            request=request,
            local_path=cached.path,
            sha256=cached.sha256,
            size=cached.size,
            cached=True,
        )

    reporter.start(key, _describe(request), request.expected_size)
    try:
        resolved = source.fetch(request, cache, progress=reporter)
    except BaseException as exc:
        reporter.finish(key, detail=f"{_describe(request)}: {exc}", failed=True)
        raise
    reporter.finish(key)
    return resolved


def _describe(request: ArtifactRequest) -> str:
    """A short label for the progress row."""
    name = str(request.metadata.get("file") or request.bundle_path.rsplit("/", 1)[-1])
    if request.expected_size:
        return f"{name} ({format_bytes(request.expected_size)})"
    return name
