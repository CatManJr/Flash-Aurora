"""Loopback ZMQ address helpers for local scheduler serving."""

from __future__ import annotations

import uuid
from pathlib import Path

import zmq


def ipc_pair(socket_dir: Path | str, *, prefix: str = "scheduler") -> tuple[str, str]:
    """Return command/event ``ipc://`` paths under ``socket_dir``."""
    directory = Path(socket_dir)
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    command = directory / f"{prefix}-{token}-commands.ipc"
    event = directory / f"{prefix}-{token}-events.ipc"
    return f"ipc://{command}", f"ipc://{event}"


def tcp_ephemeral_placeholders(*, host: str = "127.0.0.1") -> tuple[str, str]:
    """Return ``tcp://host:0`` placeholders for bind + ``LAST_ENDPOINT`` resolve."""
    return f"tcp://{host}:0", f"tcp://{host}:0"


def resolve_bound_endpoint(socket: zmq.Socket) -> str:
    """Return the concrete endpoint after ``bind`` (resolves port ``0``)."""
    return socket.getsockopt_string(zmq.LAST_ENDPOINT)
