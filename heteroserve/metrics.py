"""Serving metrics: the numbers a scheduler actually gets judged on.

TTFT  time-to-first-token, dominated by queueing + prefill (and by how much
      prefill a cache hit let us skip)
TPOT  time-per-output-token once streaming has started, dominated by batch size
      and by how much KV gathering the paging costs
E2E   what the caller experiences
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


@dataclass
class RequestRecord:
    req_id: str
    worker_id: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    cached_prefix_tokens: int = 0
    prefill_tokens_computed: int = 0
    preemptions: int = 0
    migrated: bool = False
    migration_bytes: int = 0
    migration_s: float = 0.0
    placement_reason: str = ""
    queued_prefill_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    t_submit: float = 0.0
    t_admit: float = 0.0
    t_first_token: float = 0.0
    t_done: float = 0.0

    @property
    def ttft(self) -> float:
        return self.t_first_token - self.t_submit if self.t_first_token else float("nan")

    @property
    def e2e(self) -> float:
        return self.t_done - self.t_submit if self.t_done else float("nan")

    @property
    def tpot(self) -> float:
        n = self.generated_tokens - 1
        if n <= 0 or not self.t_first_token or not self.t_done:
            return float("nan")
        return (self.t_done - self.t_first_token) / n

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_prefix_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def as_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "worker_id": self.worker_id,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "prefill_tokens_computed": self.prefill_tokens_computed,
            "preemptions": self.preemptions,
            "migrated": self.migrated,
            "migration_bytes": self.migration_bytes,
            "migration_s": round(self.migration_s, 4),
            "placement_reason": self.placement_reason,
            "ttft": round(self.ttft, 4),
            "tpot": round(self.tpot, 5),
            "e2e": round(self.e2e, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


@dataclass
class RunSummary:
    label: str = ""
    policy: str = ""
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    n_requests: int = 0
    wall_s: float = 0.0
    throughput_tok_s: float = 0.0
    request_throughput: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    e2e_p50: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0
    cache_hit_rate: float = 0.0
    prefill_tokens_saved: int = 0
    migrations: int = 0
    migration_bytes: int = 0
    migration_s: float = 0.0
    preemptions: int = 0
    repeats: int = 1
    throughput_std: float = 0.0
    per_worker: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("per_worker", None)
        return d


def summarise(
    records: list[RequestRecord],
    wall_s: float,
    label: str = "",
    policy: str = "",
    bandwidth_mbps: float = 0.0,
    latency_ms: float = 0.0,
    repeat_throughputs: list[float] | None = None,
) -> RunSummary:
    done = [r for r in records if r.t_done]
    ttfts = [r.ttft for r in done if not math.isnan(r.ttft)]
    tpots = [r.tpot for r in done if not math.isnan(r.tpot)]
    e2es = [r.e2e for r in done if not math.isnan(r.e2e)]

    gen = sum(r.generated_tokens for r in done)
    prompt = sum(r.prompt_tokens for r in done)
    cached = sum(r.cached_prefix_tokens for r in done)

    per_worker: dict[str, dict] = {}
    for r in done:
        w = per_worker.setdefault(
            r.worker_id, {"requests": 0, "generated": 0, "prompt": 0, "cached": 0}
        )
        w["requests"] += 1
        w["generated"] += r.generated_tokens
        w["prompt"] += r.prompt_tokens
        w["cached"] += r.cached_prefix_tokens

    return RunSummary(
        label=label,
        policy=policy,
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        n_requests=len(done),
        wall_s=round(wall_s, 4),
        throughput_tok_s=round(gen / wall_s, 2) if wall_s else 0.0,
        request_throughput=round(len(done) / wall_s, 3) if wall_s else 0.0,
        ttft_p50=round(percentile(ttfts, 0.50), 4),
        ttft_p95=round(percentile(ttfts, 0.95), 4),
        ttft_p99=round(percentile(ttfts, 0.99), 4),
        tpot_p50=round(percentile(tpots, 0.50), 5),
        e2e_p50=round(percentile(e2es, 0.50), 4),
        e2e_p95=round(percentile(e2es, 0.95), 4),
        e2e_p99=round(percentile(e2es, 0.99), 4),
        cache_hit_rate=round(cached / prompt, 4) if prompt else 0.0,
        prefill_tokens_saved=cached,
        migrations=sum(1 for r in done if r.migrated),
        migration_bytes=sum(r.migration_bytes for r in done),
        migration_s=round(sum(r.migration_s for r in done), 4),
        preemptions=sum(r.preemptions for r in done),
        repeats=len(repeat_throughputs) if repeat_throughputs else 1,
        throughput_std=(
            round(statistics.pstdev(repeat_throughputs), 2)
            if repeat_throughputs and len(repeat_throughputs) > 1
            else 0.0
        ),
        per_worker=per_worker,
    )


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")
