"""Configuration objects for the whole system.

Everything the scheduler reasons about — KV byte costs, link budgets, device
capacity — is derived from these, so the cost model and the actual runtime can
never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt2"
    n_layer: int = 12
    n_head: int = 12
    # Grouped-query attention: fewer KV heads than query heads. None means MHA
    # (one KV head per query head), which is what GPT-2 does. Every current
    # model -- Llama 3, Mistral, Qwen -- uses GQA, and it shrinks the KV cache
    # by exactly n_head / n_kv_head, which moves the migrate-vs-recompute
    # crossover just as much as a faster link would.
    n_kv_head: int | None = None
    n_embd: int = 768
    vocab_size: int = 50257
    n_ctx: int = 1024
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_heads(self) -> int:
        """KV heads actually stored. Equals n_head under MHA."""
        return self.n_kv_head or self.n_head

    @property
    def kv_group(self) -> int:
        """Query heads sharing each KV head."""
        return self.n_head // self.kv_heads

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def kv_bytes_per_token(self, dtype: np.dtype) -> int:
        """Bytes of KV cache one token occupies across every layer."""
        return int(2 * self.n_layer * self.kv_dim * np.dtype(dtype).itemsize)

    @classmethod
    def from_hf_config(cls, path: Path) -> "ModelConfig":
        cfg = json.loads(Path(path).read_text())
        return cls(
            name=cfg.get("model_type", "gpt2"),
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            vocab_size=cfg["vocab_size"],
            n_ctx=cfg["n_ctx"],
            layer_norm_eps=cfg.get("layer_norm_epsilon", 1e-5),
        )

    @classmethod
    def tiny(cls, n_kv_head: int | None = None) -> "ModelConfig":
        """Small synthetic config for fast sweeps and unit tests."""
        return cls(
            name="tiny" if n_kv_head is None else f"tiny-gqa{n_kv_head}",
            n_layer=4,
            n_head=4,
            n_kv_head=n_kv_head,
            n_embd=256,
            vocab_size=50257,
            n_ctx=1024,
        )

    @classmethod
    def llama3_8b_shape(cls) -> "ModelConfig":
        """Llama-3-8B's attention geometry: 32 query heads, 8 KV heads.

        Not the weights -- the *shape*, so the KV-cache economics are the real
        ones. GQA cuts KV per token 4x here, which is the whole point.
        """
        return cls(name="llama3-8b-shape", n_layer=32, n_head=32,
                   n_kv_head=8, n_embd=4096, vocab_size=128256, n_ctx=8192)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVConfig:
    block_size: int = 16
    num_blocks: int = 512
    dtype: str = "float16"

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def block_bytes(self, model: ModelConfig) -> int:
        return model.kv_bytes_per_token(self.np_dtype) * self.block_size

    def pool_bytes(self, model: ModelConfig) -> int:
        return self.block_bytes(model) * self.num_blocks

    @property
    def capacity_tokens(self) -> int:
        return self.block_size * self.num_blocks


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkConfig:
    """A shaped point-to-point link between two workers.

    `bandwidth_mbps` is megabits/sec (0 or None => unlimited). `latency_ms` is
    one-way propagation delay. `jitter_ms` is uniform +/- noise on top of it.
    """

    bandwidth_mbps: float = 1000.0
    latency_ms: float = 0.5
    jitter_ms: float = 0.0
    loss: float = 0.0

    @property
    def bytes_per_sec(self) -> float:
        if not self.bandwidth_mbps or self.bandwidth_mbps <= 0:
            return float("inf")
        return self.bandwidth_mbps * 1e6 / 8.0

    def transfer_seconds(self, nbytes: int) -> float:
        """Model-side estimate the scheduler uses when deciding migrate vs recompute."""
        return self.latency_ms / 1000.0 + nbytes / self.bytes_per_sec


# ---------------------------------------------------------------------------
# Workers / cluster
# ---------------------------------------------------------------------------


@dataclass
class WorkerConfig:
    worker_id: str
    device: str = "CPU"          # CPU | GPU | NPU  (OpenVINO device name)
    engine: str = "auto"          # auto | numpy | openvino
    kv: KVConfig = field(default_factory=KVConfig)
    max_batch: int = 8
    max_prefill_tokens: int = 512
    host: str = "127.0.0.1"
    port: int = 0                 # 0 => OS-assigned, discovered at boot


ROUTING_POLICIES = (
    "round_robin",       # ignores cache entirely — the naive baseline
    "least_loaded",      # queue-depth only, still cache-blind
    "prefix_affinity",   # always follow the cache, never migrate, never balance
    "cache_aware",       # full cost model: hit length vs load vs migration budget
)


@dataclass
class ClusterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    workers: list[WorkerConfig] = field(default_factory=list)
    link: LinkConfig = field(default_factory=LinkConfig)
    policy: str = "cache_aware"
    # Scheduler knobs
    max_running: int = 16
    watermark: float = 0.90        # KV utilisation at which we stop admitting
    enable_migration: bool = True
    seed: int = 0

    def worker(self, worker_id: str) -> WorkerConfig:
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        raise KeyError(worker_id)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def default_cluster(
    devices: list[str] | None = None,
    model: ModelConfig | None = None,
    num_blocks: int = 512,
    block_size: int = 16,
) -> ClusterConfig:
    """One worker per physical device, sized identically so comparisons are fair."""
    devices = devices or ["CPU"]
    model = model or ModelConfig()
    kv = KVConfig(block_size=block_size, num_blocks=num_blocks)
    workers = [
        WorkerConfig(worker_id=f"w{i}-{dev.lower()}", device=dev, kv=kv)
        for i, dev in enumerate(devices)
    ]
    return ClusterConfig(model=model, workers=workers)
