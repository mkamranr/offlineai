"""In-process network guard used by the test suite.

Lives outside ``conftest.py`` so tests can import the exception type without
relying on pytest plugin import order.
"""

from __future__ import annotations

import socket
from typing import Any

__all__ = ["NetworkAccessDuringTestError", "install_guard", "real_socket"]

#: Captured before patching so the guard can be lifted cleanly.
real_socket = socket.socket
_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname
_real_create_connection = socket.create_connection

#: Address families that are local IPC rather than network. Blocking these
#: would break multiprocessing and the Docker socket without strengthening the
#: guarantee we care about, which is "no packets leave this machine".
_LOCAL_FAMILIES = {
    getattr(socket, name) for name in ("AF_UNIX", "AF_UNSPEC") if hasattr(socket, name)
}


class NetworkAccessDuringTestError(RuntimeError):
    """Raised when test code attempts to reach the network.

    Seeing this is not a flaky test - it means production code tried to make an
    outbound call on a path that must work air-gapped.
    """

    def __init__(self, what: str) -> None:
        super().__init__(
            f"Blocked network access during tests: {what}\n"
            "OfflineAI code paths must work without a network. If this call is "
            "legitimately build-time only, mark the test @pytest.mark.allow_network."
        )


def install_guard(monkeypatch: Any) -> None:
    """Patch the socket module so outbound access raises."""

    def guarded_socket(
        family: int = socket.AF_INET,
        type_: int = socket.SOCK_STREAM,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if family not in _LOCAL_FAMILIES:
            raise NetworkAccessDuringTestError(f"socket(family={family})")
        return real_socket(family, type_, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessDuringTestError(f"DNS lookup for {host!r}:{port!r}")

    def guarded_gethostbyname(host: Any) -> Any:
        raise NetworkAccessDuringTestError(f"DNS lookup for {host!r}")

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessDuringTestError(f"connection to {address!r}")

    monkeypatch.setattr(socket, "socket", guarded_socket)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
