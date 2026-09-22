"""Strict offline enforcement (section 32).

In strict mode the installer may use only: the bundle, the local filesystem,
the local container runtime and the local registry. Nothing else.

Enforcement is in two layers, because one is not enough:

* **In-process.** ``socket.socket`` and DNS resolution are patched to raise, so
  any Python code path that tries to reach out fails loudly and names itself.
  This catches our own bugs.
* **In the environment.** Child processes get ``PIP_NO_INDEX``,
  ``HF_HUB_OFFLINE``, ``TRANSFORMERS_OFFLINE`` and friends, and container runs
  get ``--pull never``. Patching sockets in this process does nothing to a
  subprocess, and pip and docker are subprocesses.

The point is not to make egress impossible - a determined workload can always
open its own socket - but to make an accidental one impossible to miss. The
application must never *silently* download a missing dependency.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from offlineai.errors import StrictOfflineViolationError
from offlineai.logging import get_logger

__all__ = [
    "OFFLINE_ENVIRONMENT",
    "offline_environment",
    "strict_offline_guard",
]

logger = get_logger("security.offline")

#: Environment handed to every child process in strict mode. Each entry turns
#: a tool that would otherwise reach out into one that fails locally.
OFFLINE_ENVIRONMENT: dict[str, str] = {
    # pip: never consult an index, never check for its own updates.
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_RETRIES": "0",
    "PIP_TIMEOUT": "1",
    # Hugging Face: use only what is already local.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # Do not let a proxy quietly provide the egress we just removed.
    "no_proxy": "*",
    "NO_PROXY": "*",
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
}

#: Address families that are local IPC rather than network. The Docker socket
#: is one of these, and blocking it would break the install we are protecting.
_LOCAL_FAMILIES = {
    getattr(socket, name) for name in ("AF_UNIX", "AF_UNSPEC") if hasattr(socket, name)
}


def offline_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with offline enforcement applied."""
    return {**(base if base is not None else dict(os.environ)), **OFFLINE_ENVIRONMENT}


@contextmanager
def strict_offline_guard(*, enabled: bool = True) -> Iterator[None]:
    """Block outbound network access from this process.

    Any attempt raises :class:`StrictOfflineViolationError`, which exits 5 -
    "missing dependency" - because a strict-offline install that needs the
    network means the bundle was built incomplete, and that is what the
    operator has to act on.
    """
    if not enabled:
        yield
        return

    real_socket = socket.socket
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_create_connection = socket.create_connection

    def guarded_socket(
        family: int = socket.AF_INET,
        type_: int = socket.SOCK_STREAM,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if family not in _LOCAL_FAMILIES:
            raise _violation(f"a network socket (family {family})")
        return real_socket(family, type_, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        raise _violation(f"a DNS lookup for {host!r}")

    def guarded_gethostbyname(host: Any) -> Any:
        raise _violation(f"a DNS lookup for {host!r}")

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        raise _violation(f"a connection to {address!r}")

    socket.socket = guarded_socket  # type: ignore[assignment,misc]
    socket.getaddrinfo = guarded_getaddrinfo
    socket.gethostbyname = guarded_gethostbyname
    socket.create_connection = guarded_create_connection

    previous = {key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}
    os.environ.update(OFFLINE_ENVIRONMENT)
    logger.debug("strict-offline mode engaged")

    try:
        yield
    finally:
        socket.socket = real_socket  # type: ignore[misc]
        socket.getaddrinfo = real_getaddrinfo
        socket.gethostbyname = real_gethostbyname
        socket.create_connection = real_create_connection
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _violation(what: str) -> StrictOfflineViolationError:
    return StrictOfflineViolationError(
        f"installation attempted {what} while strict-offline mode is enabled",
        action="Nothing may be fetched during a strict-offline install. If an "
        "artifact is genuinely missing, rebuild the bundle on a connected "
        "machine with it included and transfer it again.",
    )
