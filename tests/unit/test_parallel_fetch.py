"""Section 45: configurable download concurrency.

Two properties matter more than the speed-up. Results must stay in request
order whatever order they complete in, because the manifest is built from that
list and section 34 wants reproducible bundles. And a failure must stop the
build promptly rather than letting the remaining workers run to completion.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.fetcher import fetch_all
from offlineai.errors import SourceError
from offlineai.progress import NullReporter
from offlineai.schema.manifest import ArtifactType


def request(index: int, *, locator: str | None = None) -> ArtifactRequest:
    return ArtifactRequest(
        id=f"artifact-{index}",
        artifact_type=ArtifactType.MODEL,
        bundle_path=f"artifacts/models/m/file-{index:03d}.bin",
        locator=locator or f"remote://file-{index:03d}",
        source_kind="fake",
        metadata={"index": index},
    )


class RecordingSource:
    """A source that records call order and can be made slow or failing."""

    kind = "fake"

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_on: set[str] | None = None,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.barrier = barrier
        self.calls: list[str] = []
        self.concurrent = 0
        self.peak_concurrency = 0
        self._lock = threading.Lock()

    def expand(self, ref: object) -> list[ArtifactRequest]:
        return []

    def fetch(
        self, req: ArtifactRequest, cache: ArtifactCache, *, progress: object = None
    ) -> ResolvedArtifact:
        with self._lock:
            self.calls.append(req.id)
            self.concurrent += 1
            self.peak_concurrency = max(self.peak_concurrency, self.concurrent)
        try:
            if self.barrier is not None:
                # Only releases if enough workers are genuinely in flight.
                self.barrier.wait(timeout=5)
            if self.delay:
                time.sleep(self.delay)
            if req.id in self.fail_on:
                raise SourceError(f"deliberate failure for {req.id}")
            return ResolvedArtifact(
                request=req,
                local_path=Path("/dev/null"),
                sha256=f"{hash(req.id) & 0xFFFFFFFF:064x}",
                size=len(req.id),
            )
        finally:
            with self._lock:
                self.concurrent -= 1


@pytest.fixture
def cache(tmp_path: Path) -> ArtifactCache:
    store = ArtifactCache(tmp_path / "cache")
    store.ensure()
    return store


class TestConcurrency:
    def test_requests_really_do_run_in_parallel(self, cache: ArtifactCache) -> None:
        """A barrier that only releases when four workers are in flight. If
        this runs serially it times out."""
        barrier = threading.Barrier(4)
        source = RecordingSource(barrier=barrier)
        jobs = [(source, request(i)) for i in range(4)]

        fetch_all(jobs, cache, workers=4, reporter=NullReporter())
        assert source.peak_concurrency == 4

    def test_one_worker_is_serial(self, cache: ArtifactCache) -> None:
        source = RecordingSource(delay=0.01)
        jobs = [(source, request(i)) for i in range(4)]
        fetch_all(jobs, cache, workers=1, reporter=NullReporter())
        assert source.peak_concurrency == 1

    def test_concurrency_is_capped_at_the_worker_count(self, cache: ArtifactCache) -> None:
        source = RecordingSource(delay=0.02)
        jobs = [(source, request(i)) for i in range(12)]
        fetch_all(jobs, cache, workers=3, reporter=NullReporter())
        assert source.peak_concurrency <= 3

    def test_parallel_is_faster_than_serial(self, cache: ArtifactCache) -> None:
        jobs_serial = [(RecordingSource(delay=0.05), request(i)) for i in range(8)]
        source = jobs_serial[0][0]
        jobs = [(source, request(i)) for i in range(8)]

        start = time.monotonic()
        fetch_all(jobs, cache, workers=1, reporter=NullReporter())
        serial = time.monotonic() - start

        start = time.monotonic()
        fetch_all(jobs, cache, workers=8, reporter=NullReporter())
        parallel = time.monotonic() - start

        assert parallel < serial / 2, f"serial {serial:.3f}s vs parallel {parallel:.3f}s"


class TestOrderIsPreserved:
    """The manifest is built from this list, and section 34 wants bundles that
    are reproducible."""

    def test_results_follow_request_order_not_completion_order(self, cache: ArtifactCache) -> None:
        # Later requests finish first, so completion order is reversed.
        class Reversing(RecordingSource):
            def fetch(
                self, req: ArtifactRequest, cache: ArtifactCache, *, progress: object = None
            ) -> ResolvedArtifact:
                time.sleep(0.05 * (5 - int(req.metadata["index"])))  # type: ignore[arg-type]
                return super().fetch(req, cache, progress=progress)

        source = Reversing()
        jobs = [(source, request(i)) for i in range(5)]
        results = fetch_all(jobs, cache, workers=5, reporter=NullReporter())

        assert [r.request.id for r in results] == [f"artifact-{i}" for i in range(5)]
        assert source.calls != [f"artifact-{i}" for i in range(5)], (
            "fixture is wrong: completion order should differ from request order"
        )

    @pytest.mark.parametrize("workers", [1, 2, 4, 8])
    def test_the_result_is_identical_at_any_worker_count(
        self, cache: ArtifactCache, workers: int
    ) -> None:
        source = RecordingSource()
        jobs = [(source, request(i)) for i in range(8)]
        results = fetch_all(jobs, cache, workers=workers, reporter=NullReporter())
        assert [(r.request.id, r.sha256) for r in results] == [
            (f"artifact-{i}", results[i].sha256) for i in range(8)
        ]


class TestFailureHandling:
    def test_a_failure_propagates(self, cache: ArtifactCache) -> None:
        source = RecordingSource(fail_on={"artifact-3"})
        jobs = [(source, request(i)) for i in range(8)]
        with pytest.raises(SourceError, match="artifact-3"):
            fetch_all(jobs, cache, workers=4, reporter=NullReporter())

    def test_remaining_work_is_cancelled(self, cache: ArtifactCache) -> None:
        """A build that has already failed should not keep downloading
        gigabytes it is going to throw away."""
        source = RecordingSource(delay=0.05, fail_on={"artifact-0"})
        jobs = [(source, request(i)) for i in range(40)]
        with pytest.raises(SourceError):
            fetch_all(jobs, cache, workers=2, reporter=NullReporter())
        assert len(source.calls) < 40, f"all {len(source.calls)} jobs ran despite an early failure"

    def test_no_threads_are_left_running(self, cache: ArtifactCache) -> None:
        before = threading.active_count()
        source = RecordingSource(fail_on={"artifact-1"})
        jobs = [(source, request(i)) for i in range(6)]
        with pytest.raises(SourceError):
            fetch_all(jobs, cache, workers=3, reporter=NullReporter())
        time.sleep(0.2)
        assert threading.active_count() <= before + 1


class TestCacheInteraction:
    def test_a_cache_hit_skips_the_source_entirely(self, cache: ArtifactCache) -> None:
        payload = b"already have this"
        entry = cache.store_file(_write(cache.root.parent / "src.bin", payload))
        req = request(0)
        cache.put_ref(req.source_kind, req.cache_key, entry.sha256)

        source = RecordingSource()
        results = fetch_all([(source, req)], cache, workers=2, reporter=NullReporter())

        assert source.calls == [], "a cached artifact must not be fetched again"
        assert results[0].cached is True
        assert results[0].sha256 == entry.sha256

    def test_duplicate_requests_are_fetched_once(self, cache: ArtifactCache) -> None:
        """Two declarations pointing at the same artifact share a partial-file
        path, so fetching both at once would have them writing over each
        other."""
        source = RecordingSource(delay=0.02)
        shared = "remote://identical"
        jobs = [
            (source, request(0, locator=shared)),
            (source, request(1, locator=shared)),
            (source, request(2, locator=shared)),
        ]
        results = fetch_all(jobs, cache, workers=3, reporter=NullReporter())

        assert len(source.calls) == 1, f"fetched {len(source.calls)} times, expected 1"
        assert len(results) == 3, "every request still gets a result"
        assert {r.sha256 for r in results} == {results[0].sha256}

    def test_deduplicated_results_keep_their_own_request(self, cache: ArtifactCache) -> None:
        """Each result must carry the request that asked for it, because the
        bundle path comes from there."""
        source = RecordingSource()
        shared = "remote://identical"
        jobs = [(source, request(i, locator=shared)) for i in range(3)]
        results = fetch_all(jobs, cache, workers=3, reporter=NullReporter())
        assert [r.request.bundle_path for r in results] == [
            f"artifacts/models/m/file-{i:03d}.bin" for i in range(3)
        ]


class TestProgressIsReported:
    def test_each_artifact_is_started_and_finished(self, cache: ArtifactCache) -> None:
        events: list[tuple[str, str]] = []
        lock = threading.Lock()

        class Recording(NullReporter):
            def start(self, key: str, description: str, total: int | None = None) -> None:
                with lock:
                    events.append(("start", key))

            def finish(self, key: str, *, detail: str | None = None, failed: bool = False) -> None:
                with lock:
                    events.append(("finish", key))

        source = RecordingSource()
        jobs = [(source, request(i)) for i in range(4)]
        fetch_all(jobs, cache, workers=2, reporter=Recording())

        assert len([e for e in events if e[0] == "start"]) == 4
        assert len([e for e in events if e[0] == "finish"]) == 4

    def test_a_failed_artifact_is_finished_as_failed(self, cache: ArtifactCache) -> None:
        failures: list[str] = []

        class Recording(NullReporter):
            def finish(self, key: str, *, detail: str | None = None, failed: bool = False) -> None:
                if failed:
                    failures.append(key)

        source = RecordingSource(fail_on={"artifact-0"})
        with pytest.raises(SourceError):
            fetch_all([(source, request(0))], cache, workers=1, reporter=Recording())
        assert failures == ["artifact-0"]


class TestWorkerCountValidation:
    @pytest.mark.parametrize("workers", [0, -1])
    def test_a_nonsense_worker_count_is_rejected(self, cache: ArtifactCache, workers: int) -> None:
        with pytest.raises(ValueError, match="workers"):
            fetch_all([], cache, workers=workers, reporter=NullReporter())

    def test_an_empty_job_list_is_fine(self, cache: ArtifactCache) -> None:
        assert fetch_all([], cache, workers=4, reporter=NullReporter()) == []


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
