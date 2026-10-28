"""Token counting backends.

Two backends are provided:

- ``TiktokenBackend``: exact token counts using OpenAI's ``tiktoken``
  (optional dependency, ``pip install tokaudit[exact]``).
- ``EstimatorBackend``: dependency-free heuristic. It chunks text into
  fixed-size character blocks, so it averages one "token" per
  ``CHUNK_CHARS`` characters. Real BPE tokenizers vary by content type
  (prose is lighter per character, dense code is heavier), so treat
  estimator counts as approximate. ``tests/test_tokenizers.py`` measures
  the error against tiktoken on a mixed corpus and asserts it stays
  within a documented tolerance.

Both backends expose ``encode(text) -> list[int]`` so prefix analysis can
compare token sequences, and ``count(text) -> int``.
"""

from __future__ import annotations

from typing import Callable, List

CHUNK_CHARS = 4
"""Average characters per token assumed by the estimator backend."""


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
        return [
            hash(text[i : i + CHUNK_CHARS])
            for i in range(0, len(text), CHUNK_CHARS)
        ]

    def count(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + CHUNK_CHARS - 1) // CHUNK_CHARS


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
