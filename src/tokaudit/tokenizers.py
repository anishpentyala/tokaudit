"""Token counting backends.

Two backends are provided:

- ``TiktokenBackend``: exact token counts using OpenAI's ``tiktoken``
  (optional dependency, ``pip install tokaudit[exact]``).
- ``EstimatorBackend``: dependency-free heuristic. It scores each
  whitespace-delimited word in thirds of a pseudo-token: natural-language
  words (85%+ ASCII letters) score 4/3 (the standard ~0.75 words/token
  English heuristic), while symbol-heavy words score per character -
  ~2/3 token per structural character (``{}[]",:``) and ~1 token per 4
  remaining characters - because real BPE tokenizers merge structural
  JSON punctuation densely. Word weights are additive, so the estimate
  is prefix-stable up to rounding at the boundary word. Measured mean
  absolute error against tiktoken o200k_base on the deterministic
  synthetic corpus in ``tokaudit.benchmark`` (60 samples per category):
  prose ~18%, code ~15%, JSON ~19% - see the README table. Highly
  repetitive text compresses better under BPE than any length heuristic
  can see, so error grows on repetitive input (overestimate).
  ``tests/test_tokenizers.py`` measures the error against tiktoken on
  fixed fixtures and asserts it stays within documented per-content
  tolerances.

Both backends expose ``encode(text) -> list[int]`` so prefix analysis can
compare token sequences, and ``count(text) -> int``.
"""

from __future__ import annotations

from typing import Callable, List

_STRUCT_CHARS = frozenset('{}[]",:')
_NATURAL_ALPHA_RATIO = 0.85
_UNITS_PER_TOKEN = 3
_NATURAL_WORD_UNITS = 4
"""Word weights in thirds of a pseudo-token. A natural-language word
scores 4/3 (OpenAI's rule of thumb of ~0.75 words per token for English).
A symbol-heavy word scores per character: ~2/3 token per structural
character and ~1 token per 4 remaining characters, because real BPE
tokenizers merge structural JSON punctuation densely while ordinary
characters average ~4 per token. The decision is per word and weights
are additive, so a shared text prefix yields a shared pseudo-token
prefix up to rounding at the boundary word."""


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

    @staticmethod
    def _word_units(word: str) -> int:
        letters = sum(c.isalpha() for c in word)
        if letters / len(word) >= _NATURAL_ALPHA_RATIO:
            return _NATURAL_WORD_UNITS
        structural = sum(c in _STRUCT_CHARS for c in word)
        # Structural characters cost ~2/3 token each (BPE merges common
        # pairs like `":`); the remaining characters average 4 per token.
        return 2 * structural + _UNITS_PER_TOKEN * max(
            1, -(-(len(word) - structural) // 4)
        )

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        tokens: List[int] = []
        units = 0
        for word in text.split():
            units += self._word_units(word)
            while len(tokens) < units // _UNITS_PER_TOKEN:
                tokens.append(hash(word))
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
