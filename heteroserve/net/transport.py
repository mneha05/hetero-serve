"""Length-prefixed message framing over real asyncio TCP sockets.

A frame is:

    [4B meta_len][4B blob_len][meta: utf-8 JSON][blob: raw bytes]

Control messages carry only `meta`. KV migrations carry a multi-megabyte `blob`,
which is what actually pushes against the link budget.

Sends go through `ShapedLink.pace()` first, so the bandwidth/latency model is
enforced on the wire rather than accounted for after the fact.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from .shaper import ShapedLink

_HEADER = struct.Struct("!II")
MAX_FRAME = 512 * 1024 * 1024


class ConnectionClosed(Exception):
    pass


def encode_array(a: np.ndarray) -> tuple[dict, bytes]:
    """Describe a numpy array well enough to rebuild it on the far side."""
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape)}, a.tobytes()


def decode_array(desc: dict, blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.dtype(desc["dtype"])).reshape(desc["shape"])


class Channel:
    """One bidirectional framed connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        link: ShapedLink | None = None,
        name: str = "",
    ):
        self.reader = reader
        self.writer = writer
        self.link = link
        self.name = name
        self._send_lock = asyncio.Lock()
        self.closed = False

    # -- io -----------------------------------------------------------------

    async def send(self, meta: dict[str, Any], blob: bytes = b"") -> float:
        """Frame + shape + write. Returns the delay the link imposed."""
        payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        header = _HEADER.pack(len(payload), len(blob))
        total = len(header) + len(payload) + len(blob)

        delay = 0.0
        if self.link is not None:
            delay = await self.link.pace(total)

        async with self._send_lock:
            if self.closed:
                raise ConnectionClosed(self.name)
            self.writer.write(header)
            self.writer.write(payload)
            if blob:
                self.writer.write(blob)
            await self.writer.drain()
        return delay

    async def recv(self) -> tuple[dict[str, Any], bytes]:
        try:
            head = await self.reader.readexactly(_HEADER.size)
        except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
            raise ConnectionClosed(self.name) from exc

        meta_len, blob_len = _HEADER.unpack(head)
        if meta_len > MAX_FRAME or blob_len > MAX_FRAME:
            raise ConnectionClosed(f"{self.name}: frame too large")

        try:
            meta_raw = await self.reader.readexactly(meta_len)
            blob = await self.reader.readexactly(blob_len) if blob_len else b""
        except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
            raise ConnectionClosed(self.name) from exc

        return json.loads(meta_raw.decode("utf-8")), blob

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionResetError, OSError):
            pass

    # -- helpers ------------------------------------------------------------

    async def serve(self, handler: Callable[[dict, bytes], Awaitable[None]]) -> None:
        """Read frames until the peer goes away, dispatching each to `handler`."""
        try:
            while True:
                meta, blob = await self.recv()
                await handler(meta, blob)
        except ConnectionClosed:
            pass
        finally:
            await self.close()


async def connect(
    host: str, port: int, link: ShapedLink | None = None, name: str = "", retries: int = 40
) -> Channel:
    """Dial a peer, retrying briefly so boot order doesn't matter."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            return Channel(reader, writer, link=link, name=name)
        except (ConnectionRefusedError, OSError) as exc:
            last = exc
            await asyncio.sleep(0.05)
    raise ConnectionError(f"could not reach {name or ''} {host}:{port}: {last}")
