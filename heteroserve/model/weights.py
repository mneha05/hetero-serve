"""Load real GPT-2 weights out of a HuggingFace safetensors file into numpy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ModelConfig
from .fetch import load_safetensors


@dataclass
class LayerWeights:
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    attn_w: np.ndarray      # [E, E + 2*kv_dim]  (== [E, 3E] under MHA)
    attn_b: np.ndarray      # [E + 2*kv_dim]
    proj_w: np.ndarray      # [E, E]
    proj_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray        # [E, 4E]
    fc_b: np.ndarray
    fcp_w: np.ndarray       # [4E, E]
    fcp_b: np.ndarray


@dataclass
class GPT2Weights:
    cfg: ModelConfig
    wte: np.ndarray         # [V, E]
    wpe: np.ndarray         # [n_ctx, E]
    layers: list[LayerWeights]
    lnf_g: np.ndarray
    lnf_b: np.ndarray
    synthetic: bool = False

    @property
    def lm_head(self) -> np.ndarray:
        # GPT-2 ties the output projection to the input embedding.
        return self.wte

    def nbytes(self) -> int:
        total = self.wte.nbytes + self.wpe.nbytes + self.lnf_g.nbytes + self.lnf_b.nbytes
        for l in self.layers:
            total += sum(
                getattr(l, f).nbytes for f in l.__dataclass_fields__  # type: ignore[attr-defined]
            )
        return total


def _strip(name: str) -> str:
    for p in ("transformer.", "model."):
        if name.startswith(p):
            return name[len(p) :]
    return name


def load_gpt2(weights_dir: Path, dtype=np.float32) -> tuple[GPT2Weights, ModelConfig]:
    weights_dir = Path(weights_dir)
    cfg = ModelConfig.from_hf_config(weights_dir / "config.json")
    raw = load_safetensors(weights_dir / "model.safetensors")
    t = {_strip(k): v for k, v in raw.items()}

    def get(name: str) -> np.ndarray:
        return np.ascontiguousarray(t[name], dtype=dtype)

    layers = []
    for i in range(cfg.n_layer):
        p = f"h.{i}."
        layers.append(
            LayerWeights(
                ln1_g=get(p + "ln_1.weight"),
                ln1_b=get(p + "ln_1.bias"),
                attn_w=get(p + "attn.c_attn.weight"),
                attn_b=get(p + "attn.c_attn.bias"),
                proj_w=get(p + "attn.c_proj.weight"),
                proj_b=get(p + "attn.c_proj.bias"),
                ln2_g=get(p + "ln_2.weight"),
                ln2_b=get(p + "ln_2.bias"),
                fc_w=get(p + "mlp.c_fc.weight"),
                fc_b=get(p + "mlp.c_fc.bias"),
                fcp_w=get(p + "mlp.c_proj.weight"),
                fcp_b=get(p + "mlp.c_proj.bias"),
            )
        )

    return (
        GPT2Weights(
            cfg=cfg,
            wte=get("wte.weight"),
            wpe=get("wpe.weight"),
            layers=layers,
            lnf_g=get("ln_f.weight"),
            lnf_b=get("ln_f.bias"),
        ),
        cfg,
    )


def synthetic_gpt2(cfg: ModelConfig, seed: int = 0, dtype=np.float32) -> GPT2Weights:
    """Seeded random weights — used by the `tiny` model for fast sweeps/tests.

    Output text is meaningless, but every FLOP, every cache byte and every
    scheduling decision is identical to the real thing.
    """
    rng = np.random.default_rng(seed)
    E, L = cfg.n_embd, cfg.n_layer
    # Under GQA the K and V projections are narrower than Q: n_kv_head * head_dim
    # rather than n_embd. That asymmetry is the whole saving.
    KVD = cfg.kv_dim
    QKV = E + 2 * KVD
    s = 0.02

    def n(*shape):
        return (rng.standard_normal(shape) * s).astype(dtype)

    def z(*shape):
        return np.zeros(shape, dtype=dtype)

    layers = [
        LayerWeights(
            ln1_g=np.ones(E, dtype=dtype), ln1_b=z(E),
            attn_w=n(E, QKV), attn_b=z(QKV),
            proj_w=n(E, E), proj_b=z(E),
            ln2_g=np.ones(E, dtype=dtype), ln2_b=z(E),
            fc_w=n(E, 4 * E), fc_b=z(4 * E),
            fcp_w=n(4 * E, E), fcp_b=z(E),
        )
        for _ in range(L)
    ]
    return GPT2Weights(
        cfg=cfg,
        wte=n(cfg.vocab_size, E),
        wpe=n(cfg.n_ctx, E),
        layers=layers,
        lnf_g=np.ones(E, dtype=dtype),
        lnf_b=z(E),
        synthetic=True,
    )
