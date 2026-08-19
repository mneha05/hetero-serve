"""GPT-2 byte-level BPE, implemented from scratch.

Only needs vocab.json + merges.txt, both of which we pull from HuggingFace. No
`transformers`, no `tokenizers`. Uses the `regex` package for the exact GPT-2
pre-tokenisation pattern when available and falls back to a stdlib-`re`
approximation otherwise (the fallback matches on ASCII text, which is all the
benchmark workloads use).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

try:  # exact GPT-2 pattern needs \p{L} / \p{N}
    import regex as _re

    _PATTERN = _re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    _EXACT_PRETOKENISER = True
except ImportError:  # pragma: no cover - exercised only without `regex`
    import re as _re

    _PATTERN = _re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""",
        _re.UNICODE,
    )
    _EXACT_PRETOKENISER = False


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Reversible byte <-> unicode map that dodges control/whitespace codepoints."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GPT2Tokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path):
        self.encoder: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}

        merge_lines = Path(merges_path).read_text(encoding="utf-8").split("\n")
        # first line is a "#version:" comment
        merges = [tuple(l.split()) for l in merge_lines[1:] if l.strip()]
        self.bpe_ranks: dict[tuple[str, str], int] = {m: i for i, m in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache: dict[str, str] = {}

        self.eos_token_id = self.encoder.get("<|endoftext|>", 50256)
        self.vocab_size = len(self.encoder)

    # -- core BPE -----------------------------------------------------------

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            self._cache[token] = token
            return token

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        out = " ".join(word)
        self._cache[token] = out
        return out

    # -- public API ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _PATTERN.findall(text):
            token = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[bp] for bp in self._bpe(token).split(" "))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder.get(int(i), "") for i in ids)
        raw = bytearray(self.byte_decoder[c] for c in text if c in self.byte_decoder)
        return raw.decode("utf-8", errors="replace")

    @classmethod
    def from_dir(cls, d: Path) -> "GPT2Tokenizer":
        d = Path(d)
        return cls(d / "vocab.json", d / "merges.txt")


def exact_pretokeniser_available() -> bool:
    return _EXACT_PRETOKENISER
