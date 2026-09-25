"""Section 46: progress reporting for large artifacts.

A 62 GB build that prints nothing for two hours is indistinguishable from a
hung one, and an operator will kill it. But progress must also stay completely
out of the way when the output is being consumed by something other than a
human, which is what most of these tests are about.
"""

from __future__ import annotations

import io
import threading

import pytest

from offlineai.progress import NullReporter, ProgressReporter, download_callback


class TestNullReporter:
    """The default. Must accept every call and do nothing, so no code path has
    to check whether reporting is enabled."""

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(NullReporter(), ProgressReporter)

    def test_every_call_is_a_no_op(self) -> None:
        reporter = NullReporter()
        with reporter:
            reporter.set_overall("building", 1000)
            reporter.start("a", "artifact a", 100)
            reporter.advance("a", 50)
            reporter.update("a", 75, 100)
            reporter.finish("a", detail="done")

    def test_calls_for_unknown_keys_do_not_raise(self) -> None:
        """A source that reports progress for something it never started must
        not take the build down."""
        reporter = NullReporter()
        reporter.advance("never-started", 10)
        reporter.finish("never-started")

    def test_is_reusable_as_a_context_manager(self) -> None:
        reporter = NullReporter()
        with reporter:
            pass
        with reporter:
            pass


class TestDownloadCallbackBridge:
    """download_resumable reports (downloaded, total); the reporter wants
    (key, completed, total). One adapter, so neither has to know the other."""

    def test_forwards_to_update(self) -> None:
        seen: list[tuple[str, int, int | None]] = []

        class Recorder(NullReporter):
            def update(self, key: str, completed: int, total: int | None = None) -> None:
                seen.append((key, completed, total))

        callback = download_callback(Recorder(), "shard-1")
        callback(1024, 4096)
        callback(4096, 4096)
        assert seen == [("shard-1", 1024, 4096), ("shard-1", 4096, 4096)]

    def test_an_unknown_total_is_passed_through(self) -> None:
        seen: list[int | None] = []

        class Recorder(NullReporter):
            def update(self, key: str, completed: int, total: int | None = None) -> None:
                seen.append(total)

        download_callback(Recorder(), "x")(100, None)
        assert seen == [None]


class TestRichReporter:
    def _reporter(self) -> tuple[object, io.StringIO]:
        from rich.console import Console

        from offlineai.cli.progress import RichReporter

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100)
        return RichReporter(console=console), buffer

    def test_renders_the_artifact_name(self) -> None:
        reporter, buffer = self._reporter()
        with reporter:
            reporter.start("m", "qwen3 model", 1000)  # type: ignore[attr-defined]
            reporter.update("m", 820, 1000)  # type: ignore[attr-defined]
        assert "qwen3 model" in buffer.getvalue()

    def test_shows_a_percentage(self) -> None:
        reporter, buffer = self._reporter()
        with reporter:
            reporter.start("m", "model", 1000)  # type: ignore[attr-defined]
            reporter.update("m", 820, 1000)  # type: ignore[attr-defined]
        assert "82%" in buffer.getvalue()

    def test_an_overall_bar_aggregates_across_artifacts(self) -> None:
        reporter, buffer = self._reporter()
        with reporter:
            reporter.set_overall("Downloading", 2000)  # type: ignore[attr-defined]
            reporter.start("a", "a", 1000)  # type: ignore[attr-defined]
            reporter.start("b", "b", 1000)  # type: ignore[attr-defined]
            reporter.advance("a", 1000)  # type: ignore[attr-defined]
            reporter.advance("b", 500)  # type: ignore[attr-defined]
            assert reporter.overall_completed == 1500  # type: ignore[attr-defined]

    def test_a_finished_artifact_stops_being_displayed(self) -> None:
        """With eight workers and a hundred shards, leaving every completed
        row on screen buries the ones still running."""
        reporter, _ = self._reporter()
        with reporter:
            reporter.start("a", "a", 100)  # type: ignore[attr-defined]
            assert reporter.active_keys() == ["a"]  # type: ignore[attr-defined]
            reporter.finish("a")  # type: ignore[attr-defined]
            assert reporter.active_keys() == []  # type: ignore[attr-defined]

    def test_unknown_keys_are_ignored(self) -> None:
        reporter, _ = self._reporter()
        with reporter:
            reporter.advance("ghost", 10)  # type: ignore[attr-defined]
            reporter.finish("ghost")  # type: ignore[attr-defined]

    def test_concurrent_updates_from_many_threads(self) -> None:
        """Downloads run in a pool, so every reporter method is called from
        several threads at once."""
        reporter, _ = self._reporter()
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                key = f"w{index}"
                reporter.start(key, f"artifact {index}", 100)  # type: ignore[attr-defined]
                for _ in range(20):
                    reporter.advance(key, 5)  # type: ignore[attr-defined]
                reporter.finish(key)  # type: ignore[attr-defined]
            except BaseException as exc:  # noqa: BLE001 - recorded and re-raised
                errors.append(exc)

        with reporter:
            reporter.set_overall("Downloading", 8 * 100)  # type: ignore[attr-defined]
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors, errors
            assert reporter.overall_completed == 800  # type: ignore[attr-defined]


class TestReporterSelection:
    """Progress must never reach stdout when something is parsing it."""

    def _select(self, **kwargs: object) -> object:
        from offlineai.cli.progress import select_reporter

        return select_reporter(**kwargs)  # type: ignore[arg-type]

    def test_json_mode_gets_no_progress(self) -> None:
        assert isinstance(self._select(json=True, quiet=False, is_terminal=True), NullReporter)

    def test_quiet_mode_gets_no_progress(self) -> None:
        assert isinstance(self._select(json=False, quiet=True, is_terminal=True), NullReporter)

    def test_a_non_terminal_gets_no_progress(self) -> None:
        """Piping to a file or a log collector must not fill it with escape
        sequences."""
        assert isinstance(self._select(json=False, quiet=False, is_terminal=False), NullReporter)

    def test_an_interactive_terminal_gets_bars(self) -> None:
        reporter = self._select(json=False, quiet=False, is_terminal=True)
        assert not isinstance(reporter, NullReporter)

    @pytest.mark.parametrize(
        ("json", "quiet", "is_terminal"),
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_any_reason_to_suppress_wins(self, json: bool, quiet: bool, is_terminal: bool) -> None:
        assert isinstance(
            self._select(json=json, quiet=quiet, is_terminal=is_terminal), NullReporter
        )


class TestReporterReuse:
    """`import` runs verify and then import, entering the same reporter twice."""

    def _reporter(self) -> object:
        import io

        from rich.console import Console

        from offlineai.cli.progress import RichReporter

        return RichReporter(console=Console(file=io.StringIO(), force_terminal=True))

    def test_can_be_entered_more_than_once(self) -> None:
        reporter = self._reporter()
        with reporter:
            reporter.set_overall("Verifying", 100)  # type: ignore[attr-defined]
            reporter.advance("k", 100)  # type: ignore[attr-defined]
        with reporter:
            reporter.set_overall("Importing", 200)  # type: ignore[attr-defined]
            reporter.advance("k", 50)  # type: ignore[attr-defined]

    def test_state_does_not_leak_between_uses(self) -> None:
        reporter = self._reporter()
        with reporter:
            reporter.start("a", "a", 10)  # type: ignore[attr-defined]
        # Leaving the context clears tasks; a stale row from the previous
        # phase would otherwise sit there forever.
        assert reporter.active_keys() == []  # type: ignore[attr-defined]
        with reporter:
            assert reporter.active_keys() == []  # type: ignore[attr-defined]
