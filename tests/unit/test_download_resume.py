"""Section 43: resumable downloads.

A checkpoint can be hundreds of gigabytes. On a link that drops, restarting
from zero is not merely slow - it can mean the build never completes. These
tests drive the resume path directly with a mock transport that honours Range.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from offlineai.artifacts.download import download_resumable
from offlineai.errors import SourceError, StrictOfflineViolationError
from offlineai.utils.hashing import sha256_bytes

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


def ranged_server(payload: bytes = PAYLOAD, *, honour_range: bool = True):
    """A transport that serves ``payload``, optionally honouring Range."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("range")
        seen.append(range_header)
        if range_header and honour_range:
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            if start >= len(payload):
                return httpx.Response(416)
            chunk = payload[start:]
            return httpx.Response(
                206,
                content=chunk,
                headers={
                    "content-range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                    "content-length": str(len(chunk)),
                },
            )
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


class TestFreshDownload:
    def test_downloads_completely(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        result = download_resumable("https://x/f.bin", tmp_path / "p", client=client)
        assert result.size == len(PAYLOAD)
        assert result.sha256 == sha256_bytes(PAYLOAD)
        assert result.resumed_from == 0

    def test_reports_progress(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        seen: list[tuple[int, int | None]] = []
        download_resumable(
            "https://x/f.bin",
            tmp_path / "p",
            client=client,
            progress=lambda d, t: seen.append((d, t)),
        )
        assert seen, "progress must be reported for large artifacts"
        assert seen[-1][0] == len(PAYLOAD)
        assert seen[-1][1] == len(PAYLOAD)


class TestResume:
    def test_resumes_from_a_partial_file(self, tmp_path: Path) -> None:
        partial = tmp_path / "p"
        partial.write_bytes(PAYLOAD[:40_000])

        client, seen = ranged_server()
        result = download_resumable("https://x/f.bin", partial, client=client)

        assert seen == ["bytes=40000-"], "a Range request must be issued"
        assert result.resumed_from == 40_000
        assert result.sha256 == sha256_bytes(PAYLOAD), "the joined file must be correct"

    def test_resume_produces_identical_bytes_to_a_fresh_download(self, tmp_path: Path) -> None:
        fresh = tmp_path / "fresh"
        client, _ = ranged_server()
        download_resumable("https://x/f.bin", fresh, client=client)

        resumed = tmp_path / "resumed"
        resumed.write_bytes(PAYLOAD[:12_345])
        client2, _ = ranged_server()
        download_resumable("https://x/f.bin", resumed, client=client2)

        assert resumed.read_bytes() == fresh.read_bytes()

    def test_a_complete_partial_file_is_accepted(self, tmp_path: Path) -> None:
        """The server answers 416 because there is nothing left to send."""
        partial = tmp_path / "p"
        partial.write_bytes(PAYLOAD)
        client, _ = ranged_server()
        result = download_resumable("https://x/f.bin", partial, client=client)
        assert result.sha256 == sha256_bytes(PAYLOAD)

    def test_a_server_ignoring_range_restarts_cleanly(self, tmp_path: Path) -> None:
        """If the server sends 200 with the whole body, appending would corrupt
        the file. It must start over instead."""
        partial = tmp_path / "p"
        partial.write_bytes(PAYLOAD[:40_000])

        client, _ = ranged_server(honour_range=False)
        result = download_resumable("https://x/f.bin", partial, client=client)

        assert result.size == len(PAYLOAD), "appending would have produced 142,400 bytes"
        assert result.sha256 == sha256_bytes(PAYLOAD)


class TestVerification:
    def test_digest_mismatch_is_refused(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        with pytest.raises(SourceError, match="digest"):
            download_resumable(
                "https://x/f.bin", tmp_path / "p", client=client, expected_sha256="0" * 64
            )

    def test_a_file_failing_its_digest_is_deleted(self, tmp_path: Path) -> None:
        """Otherwise the next attempt would resume from known-bad bytes and
        fail identically, forever."""
        partial = tmp_path / "p"
        client, _ = ranged_server()
        with pytest.raises(SourceError):
            download_resumable("https://x/f.bin", partial, client=client, expected_sha256="0" * 64)
        assert not partial.exists()

    def test_size_mismatch_is_refused(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        with pytest.raises(SourceError, match="size"):
            download_resumable(
                "https://x/f.bin", tmp_path / "p", client=client, expected_size=999_999
            )

    def test_a_matching_digest_passes(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        result = download_resumable(
            "https://x/f.bin",
            tmp_path / "p",
            client=client,
            expected_sha256=sha256_bytes(PAYLOAD),
            expected_size=len(PAYLOAD),
        )
        assert result.size == len(PAYLOAD)


class TestFailureHandling:
    @pytest.mark.parametrize(
        ("status", "phrase"),
        [(401, "token"), (403, "token"), (404, "does not exist"), (429, "rate limiting")],
    )
    def test_http_errors_carry_advice(self, tmp_path: Path, status: int, phrase: str) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(SourceError) as excinfo:
            download_resumable("https://x/f.bin", tmp_path / "p", client=client)
        assert phrase in excinfo.value.render()

    def test_a_partial_file_survives_a_network_error(self, tmp_path: Path) -> None:
        """This is what makes the next attempt a resume rather than a restart."""
        partial = tmp_path / "p"
        partial.write_bytes(PAYLOAD[:1000])

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("link dropped")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(SourceError):
            download_resumable("https://x/f.bin", partial, client=client)
        assert partial.exists() and partial.stat().st_size == 1000


class TestStrictOffline:
    """Section 32: nothing may reach the network in strict-offline mode."""

    def test_a_download_is_refused_outright(self, tmp_path: Path) -> None:
        client, seen = ranged_server()
        with pytest.raises(StrictOfflineViolationError):
            download_resumable("https://x/f.bin", tmp_path / "p", client=client, offline=True)
        assert seen == [], "no request may be issued at all"

    def test_the_error_says_what_to_do(self, tmp_path: Path) -> None:
        client, _ = ranged_server()
        with pytest.raises(StrictOfflineViolationError) as excinfo:
            download_resumable("https://x/f.bin", tmp_path / "p", client=client, offline=True)
        assert "Rebuild the bundle" in excinfo.value.render()
        assert excinfo.value.exit_code == 5
