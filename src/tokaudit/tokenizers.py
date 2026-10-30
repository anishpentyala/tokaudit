"""Token counting backends.

Two backends are provided:

- ``TiktokenBackend``: exact token counts using OpenAI's ``tiktoken``
  (optional dependency, ``pip install tokaudit[exact]``).
- ``EstimatorBackend``: dependency-free heuristic. It chunks text into
  fixed-size character blocks, averaging one "token" per ``CHUNK_CHARS``
  characters, except that structural JSON-ish blocks count double (real
  BPE tokenizers are much denser on JSON). Real BPE tokenizers vary by
  content type, so treat estimator counts as approximate; measured error
  against tiktoken o200k_base is ~2% on structural JSON, ~10% on dense
  code, and ~27% on prose (overestimate, the conservative direction for
  savings analysis). ``tests/test_tokenizers.py`` measures the error
  against tiktoken on a mixed corpus and asserts it stays within
  documented per-content tolerances.

Both backends expose ``encode(text) -> list[int]`` so prefix analysis can
compare token sequences, and ``count(text) -> int``.
"""

from __future__ import annotations

from typing import Callable, List

CHUNK_CHARS = 4
"""Average characters per token assumed by the estimator backend."""

_STRUCT_CHARS = frozenset('{}[]",:')
_STRUCT_WEIGHT = 2
"""A 4-char block with 2+ structural characters (JSON-ish) counts as 2
pseudo-tokens. Real BPE tokenizers average ~2 characters per token on
structural JSON versus ~4+ on prose, and API request/response logs are
JSON-dominated, so this keeps the estimator close on its main input. The
decision is per block, so prefix stability is preserved: a shared text
prefix still yields a shared pseudo-token prefix."""


class TiktokenBackend:
    """Exact counting via tiktoken. Requires the optional dependency."""

    name = "exact"

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        import tiktoken  # noqa: F401 - raises ImportError with a clear message if missing

        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class EstimatorBackend:
    """Dependency-free estimate. Prefix-stable: a shared text prefix always
    yields a shared pseudo-token prefix (up to chunk-boundary rounding)."""

    name = "estimate"

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        tokens: List[int] = []
        for i in range(0, len(text), CHUNK_CHARS):
            block = text[i : i + CHUNK_CHARS]
            weight = (
                _STRUCT_WEIGHT
                if sum(c in _STRUCT_CHARS for c in block) >= 2
                else 1
            )
            tokens.extend([hash(block)] * weight)
        return tokens

    def count(self, text: str) -> int:
        return len(self.encode(text))


def get_backend(name: str = "auto") -> object:
    """Resolve a tokenizer backend.

    ``auto`` prefers tiktoken when installed and falls back to the
    estimator. ``exact`` requires tiktoken. ``estimate`` forces the
    heuristic backend.
    """
    if name not in ("auto", "exact", "estimate"):
        raise ValueError(f"unknown tokenizer backend: {name}")
    if name in ("auto", "exact"):
        try:
            return TiktokenBackend()
        except ImportError:
            if name == "exact":
                raise SystemExit(
                    "tokaudit[exact] (tiktoken) is not installed; "
                    "use --tokenizer estimate or install tiktoken"
                )
    return EstimatorBackend()
