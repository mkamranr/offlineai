"""The guard that makes section 63 real.

An accidental outbound call anywhere in the codebase must fail a test rather
than pass quietly. That property is the entire point of the product, so the
guard itself is tested.
"""

import socket

import pytest

from tests.netguard import NetworkAccessDuringTestError


class TestBlocksRealNetwork:
    def test_inet_socket_construction_is_refused(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_inet6_socket_construction_is_refused(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError):
            socket.socket(socket.AF_INET6, socket.SOCK_STREAM)

    def test_dns_resolution_is_refused(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError):
            socket.getaddrinfo("huggingface.co", 443)

    def test_gethostbyname_is_refused(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError):
            socket.gethostbyname("pypi.org")

    def test_create_connection_is_refused(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError):
            socket.create_connection(("pypi.org", 443), timeout=0.1)

    def test_the_error_names_the_destination(self) -> None:
        with pytest.raises(NetworkAccessDuringTestError, match="huggingface.co"):
            socket.getaddrinfo("huggingface.co", 443)


class TestAllowsLocalIpc:
    """AF_UNIX is local IPC, not network. Blocking it would break unrelated
    machinery (multiprocessing, the Docker socket) without improving the
    guarantee we actually care about."""

    def test_unix_socket_is_permitted(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.close()


class TestHttpClientsAreCaught:
    """The guard has to stop the libraries we actually use, not just raw sockets."""

    def test_httpx_cannot_reach_out(self) -> None:
        httpx = pytest.importorskip("httpx")
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - httpx wraps ours
            httpx.get("https://pypi.org/simple/", timeout=1.0)
        assert _caused_by_guard(excinfo.value)

    def test_urllib_cannot_reach_out(self) -> None:
        import urllib.error
        import urllib.request

        with pytest.raises((NetworkAccessDuringTestError, urllib.error.URLError)) as excinfo:
            urllib.request.urlopen("https://pypi.org/simple/", timeout=1.0)  # noqa: S310
        assert _caused_by_guard(excinfo.value)


def _caused_by_guard(exc: BaseException) -> bool:
    """Walk the exception chain looking for our guard."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, NetworkAccessDuringTestError):
            return True
        seen.add(id(exc))
        nxt = exc.__cause__ or exc.__context__
        if nxt is None and hasattr(exc, "args") and exc.args:
            inner = exc.args[0]
            nxt = inner if isinstance(inner, BaseException) else None
        exc = nxt  # type: ignore[assignment]
    return False


@pytest.mark.allow_network
class TestOptOut:
    """Tests that genuinely need the network declare it. Under
    OFFLINEAI_TEST_OFFLINE=1 they are skipped instead of run."""

    def test_marker_lifts_the_guard(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.close()
