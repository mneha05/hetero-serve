"""Workload generation.

A KV-cache-aware scheduler only earns its keep when requests *share prefixes*,
which is the normal case in production serving: a fixed system prompt, a RAG
document reused across follow-ups, a multi-turn conversation replayed each turn.
So the generator builds a pool of shared contexts and draws from it with a Zipf
distribution — a few very hot prefixes, a long tail of cold ones — then appends a
unique suffix per request.

Arrivals are Poisson, so queueing is real rather than a synchronised burst.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORPUS = [
    "You are a careful assistant. Answer using only the provided context, cite "
    "the section you used, and say plainly when the context does not contain the "
    "answer. Never invent numbers, names, or dates.",
    "The following is an excerpt from a technical manual describing the operation "
    "of a wireless modem subsystem, including its power states, the handover "
    "procedure between base stations, and the retransmission policy used when the "
    "acknowledgement window expires.",
    "Summarise the attached quarterly report for a non-technical reader. Focus on "
    "revenue, margin, and guidance. Ignore the boilerplate legal disclaimer at the "
    "end of the document and any forward looking statements.",
    "Below is a transcript of a customer support conversation. Identify the root "
    "cause of the customer's problem, the steps already attempted, and what the "
    "agent should do next to resolve the outstanding issue.",
    "Translate the following passage into plain English suitable for a general "
    "audience, preserving every technical claim exactly as stated but removing "
    "jargon wherever a common word will do.",
    "You are reviewing a pull request. Point out correctness bugs first, then "
    "performance problems, then style. Quote the specific line you are describing "
    "and explain the failure it would cause in production.",
]

TAILS = [
    "What is the main risk described above?",
    "List the three most important points.",
    "Explain the second paragraph in one sentence.",
    "Who is responsible for the next step?",
    "What happens if the timeout expires early?",
    "Give a short answer and then a longer one.",
    "Which section should I read first and why?",
    "Rewrite the conclusion more concisely.",
]


@dataclass
class WorkloadSpec:
    n_requests: int = 48
    n_prefixes: int = 6
    prefix_tokens: int = 256
    suffix_tokens: int = 24
    max_new_tokens: int = 24
    arrival_rate: float = 8.0       # requests/sec (Poisson)
    zipf_s: float = 1.1             # prefix popularity skew
    seed: int = 0


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_offset: float
    prefix_id: int
    text: str = ""


def _zipf_weights(n: int, s: float) -> np.ndarray:
    w = 1.0 / np.power(np.arange(1, n + 1), s)
    return w / w.sum()


def build_prefixes(spec: WorkloadSpec, tokenizer=None, vocab_size: int = 50257) -> list[list[int]]:
    """`n_prefixes` distinct contexts, each exactly `prefix_tokens` long."""
    rng = np.random.default_rng(spec.seed)
    prefixes: list[list[int]] = []
    for i in range(spec.n_prefixes):
        if tokenizer is not None:
            # Repeat real text until it is long enough, then trim to length.
            base = CORPUS[i % len(CORPUS)] + f" (context #{i}) "
            ids: list[int] = []
            while len(ids) < spec.prefix_tokens:
                ids.extend(tokenizer.encode(base))
            prefixes.append(ids[: spec.prefix_tokens])
        else:
            ids = rng.integers(1, vocab_size - 1, size=spec.prefix_tokens).tolist()
            prefixes.append([int(t) for t in ids])
    return prefixes


def generate(spec: WorkloadSpec, tokenizer=None, vocab_size: int = 50257) -> list[GenRequest]:
    rng = np.random.default_rng(spec.seed + 1)
    prefixes = build_prefixes(spec, tokenizer, vocab_size)
    weights = _zipf_weights(spec.n_prefixes, spec.zipf_s)

    # Poisson arrivals -> exponential inter-arrival gaps.
    gaps = rng.exponential(1.0 / spec.arrival_rate, size=spec.n_requests)
    offsets = np.cumsum(gaps)

    out: list[GenRequest] = []
    for i in range(spec.n_requests):
        pid = int(rng.choice(spec.n_prefixes, p=weights))
        if tokenizer is not None:
            tail_text = f" {TAILS[i % len(TAILS)]} (request {i}) "
            tail = tokenizer.encode(tail_text)
            while len(tail) < spec.suffix_tokens:
                tail = tail + tokenizer.encode(f"more detail {i} ")
            tail = tail[: spec.suffix_tokens]
        else:
            tail = [
                int(t)
                for t in rng.integers(1, vocab_size - 1, size=spec.suffix_tokens).tolist()
            ]

        out.append(
            GenRequest(
                prompt_ids=prefixes[pid] + tail,
                max_new_tokens=spec.max_new_tokens,
                arrival_offset=float(offsets[i]),
                prefix_id=pid,
            )
        )
    return out


def describe(spec: WorkloadSpec, reqs: list[GenRequest]) -> dict:
    counts = np.bincount([r.prefix_id for r in reqs], minlength=spec.n_prefixes)
    total_prompt = sum(len(r.prompt_ids) for r in reqs)
    # Tokens a perfect cache would never recompute: every repeat of a prefix.
    ideal_saved = int(sum(max(0, c - 1) * spec.prefix_tokens for c in counts))
    return {
        "n_requests": len(reqs),
        "n_prefixes": spec.n_prefixes,
        "prefix_tokens": spec.prefix_tokens,
        "prompt_tokens_total": total_prompt,
        "prefix_hit_ceiling": ideal_saved,
        "popularity": counts.tolist(),
        "span_s": round(reqs[-1].arrival_offset, 2) if reqs else 0.0,
    }
