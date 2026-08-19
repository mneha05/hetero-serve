"""Egress shaping: a token bucket for bandwidth plus a delay for propagation.

Every worker owns one `ShapedLink` for its outgoing traffic, so concurrent
transfers *contend for the same bucket* — which is the whole point. Without
contention, migrating three prefixes at once would look as cheap as migrating
one, and the scheduler would learn the wrong lesson.

The shaping is applied to genuine TCP writes, not simulated: when the benchmark
reports 3.2 s spent moving KV blocks, the process really did sit there for 3.2 s.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from ..config import LinkConfig


class TokenBucket:
    """Classic token bucket. `rate` is bytes/sec; capacity is a small burst allowance."""

    def __init__(self, rate: float, burst_seconds: float = 0.02):
        self.rate = rate
        self.capacity = float("inf") if rate == float("inf") else max(rate * burst_seconds, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self.total_bytes = 0
        self.total_wait = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        if self.rate != float("inf"):
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def consume(self, nbytes: int) -> float:
        """Block until `nbytes` may be sent. Returns seconds spent waiting."""
        if self.rate == float("inf"):
            self.total_bytes += nbytes
            return 0.0

        t0 = time.monotonic()
        remaining = float(nbytes)
        # Serialise so concurrent transfers queue behind each other on the link.
        async with self._lock:
            while remaining > 0:
                self._refill()
                take = min(self._tokens, remaining)
                self._tokens -= take
                remaining -= take
                if remaining > 0:
                    await asyncio.sleep(min(remaining / self.rate, 0.05))
        waited = time.monotonic() - t0
        self.total_bytes += nbytes
        self.total_wait += waited
        return waited


@dataclass
class LinkStats:
    bytes_sent: int = 0
    messages: int = 0
    shaping_delay_s: float = 0.0
    propagation_delay_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "bytes_sent": self.bytes_sent,
            "messages": self.messages,
            "shaping_delay_s": round(self.shaping_delay_s, 4),
            "propagation_delay_s": round(self.propagation_delay_s, 4),
        }


@dataclass
class ShapedLink:
    """Bandwidth + latency budget for one node's egress."""

    cfg: LinkConfig
    stats: LinkStats = field(default_factory=LinkStats)
    _bucket: TokenBucket | None = None
    _rng: random.Random = field(default_factory=lambda: random.Random(1234))

    def __post_init__(self) -> None:
        self._bucket = TokenBucket(self.cfg.bytes_per_sec)

    def reconfigure(self, cfg: LinkConfig) -> None:
        """Retune mid-flight — the sweep harness uses this between runs."""
        self.cfg = cfg
        self._bucket = TokenBucket(cfg.bytes_per_sec)

    async def pace(self, nbytes: int) -> float:
        """Apply the full link cost for a message of `nbytes`. Returns total delay."""
        prop = self.cfg.latency_ms / 1000.0
        if self.cfg.jitter_ms:
            prop += self._rng.uniform(-self.cfg.jitter_ms, self.cfg.jitter_ms) / 1000.0
            prop = max(0.0, prop)

        shaping = await self._bucket.consume(nbytes)
        if prop:
            await asyncio.sleep(prop)

        self.stats.bytes_sent += nbytes
        self.stats.messages += 1
        self.stats.shaping_delay_s += shaping
        self.stats.propagation_delay_s += prop
        return shaping + prop

    def estimate(self, nbytes: int) -> float:
        """What the scheduler *predicts* a transfer will cost, before doing it."""
        return self.cfg.transfer_seconds(nbytes)
