"""Token counting backends.

Two backends are provided:

- ``TiktokenBackend``: exact token counts using OpenAI's ``tiktoken``
  (optional dependency, ``pip install tokaudit[exact]``).
- ``EstimatorBackend``: dependency-free heuristic. It scores each
  whitespace-delimited word in thirds of a pseudo-token: short
  natural-language words (85%+ letters, <=20 chars) score 4/3 (the
  standard ~0.75 words/token English heuristic); symbol-heavy words
  score per character (~2/3 token per structural character
  (``{}[]",:``), ~1 token per 4 others); dense-script characters (CJK,
  kana, Hangul, Thai, Khmer, Myanmar) score ~2/3 token each, emoji 1
  each, and long alphanumeric ID/hash runs ~1 token per 2 characters.
  Exact prefix semantics: when the cut lands at a completed word (next
  char is whitespace) or just after whitespace, ``encode(prefix)`` is
  guaranteed to be an exact prefix of ``encode(full)`` (per-word
  additive weights; pseudo-ids are word hashes, so exact up to hash
  collisions). On ANY cut, the shared pseudo-token prefix extends at
  least to the last completed word before the cut. The converse does not
  hold: a mid-word cut can still match exactly when the boundary word's
  weight crosses no additional third-token boundaries differently
  (example: ``"ab {"`` vs ``"ab { cdef"`` match exactly), so callers
  must not rely on mid-word divergence as a signal. Property tests cover
  the guaranteed directions over thousands of random cuts. Measured mean absolute error against tiktoken
  o200k_base on the deterministic synthetic corpus in
  ``tokaudit.benchmark`` (60 samples per category): prose ~18%, code
  ~16%, JSON ~19%; held-out measurements on independently sourced texts
  (and the known weak spots: Vietnamese ~47%, Russian/Arabic ~23%, Thai
  ~52% over, repetitive prose ~27% over) are tabulated in the README.
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
_NATURAL_MAX_LEN = 20
_UNITS_PER_TOKEN = 3
_NATURAL_WORD_UNITS = 4
"""Word weights in thirds of a pseudo-token. A short natural-language
word (85%+ letters, at most 20 characters) scores 4/3 (OpenAI's rule of
thumb of ~0.75 words per token for English); the length cap stops
unspaced-script runs from claiming the English heuristic. A symbol-heavy
word scores per character: ~2/3 token per structural character and ~1
token per 4 remaining characters, because real BPE tokenizers merge
structural JSON punctuation densely while ordinary characters average
~4 per token. Words containing dense-script characters (CJK ideographs,
kana, Hangul, Thai, Khmer, Myanmar) score ~2/3 token per such character
(~1.5 characters per token, the measured o200k range for these scripts
is ~1.2-2.4); emoji score 1 pseudo-token each. Long alphanumeric runs
with digits (IDs, hashes; 16+ characters, no structural characters)
score ~1 token per 2 characters, since BPE merges little in them. All
decisions are per word and weights are additive, so a shared text prefix
yields a shared pseudo-token prefix exactly when the prefix ends at a
completed word (otherwise the shared prefix ends at the last completed
word before the cut); pseudo-ids are word hashes, so matching is exact
up to hash collisions."""


def _dense_char_units(char: str) -> int:
    """Pseudo-token thirds for one dense-script or emoji char, else 0."""
    o = ord(char)
    if (
        0x4E00 <= o <= 0x9FFF      # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF   # CJK Extension A
        or 0xF900 <= o <= 0xFAFF   # CJK Compatibility Ideographs
        or 0x3040 <= o <= 0x30FF   # Hiragana + Katakana
        or 0xAC00 <= o <= 0xD7AF   # Hangul Syllables
        or 0x1100 <= o <= 0x11FF   # Hangul Jamo
        or 0x3130 <= o <= 0x318F   # Hangul Compatibility Jamo
        or 0x0E00 <= o <= 0x0E7F   # Thai
        or 0x1780 <= o <= 0x17FF   # Khmer
        or 0x1000 <= o <= 0x109F   # Myanmar
    ):
        return 2
    if (
        0x1F000 <= o <= 0x1FAFF    # emoji and pictographs
        or 0x2600 <= o <= 0x27BF   # misc symbols + dingbats
        or 0x2190 <= o <= 0x21FF   # arrows
        or 0xFE00 <= o <= 0xFE0F   # variation selectors
        or o == 0x200D             # zero-width joiner
    ):
        return 3
    return 0


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
        dense = [_dense_char_units(c) for c in word]
        structural = sum(c in _STRUCT_CHARS for c in word)
        if any(dense):
            rest = sum(dense)
            other = len(word) - structural - sum(1 for d in dense if d)
            return 2 * structural + rest + -(-3 * other // 4)
        letters = sum(c.isalpha() for c in word)
        if letters / len(word) >= _NATURAL_ALPHA_RATIO and len(word) <= _NATURAL_MAX_LEN:
            return _NATURAL_WORD_UNITS
        nonstructural = len(word) - structural
        if (
            structural == 0
            and len(word) >= 16
            and any(c.isdigit() for c in word)
            and any(c.isalpha() for c in word)
        ):
            return _UNITS_PER_TOKEN * max(1, -(-len(word) // 2))
        # Structural characters cost ~2/3 token each (BPE merges common
        # pairs like `":`); the remaining characters average 4 per token.
        return 2 * structural + _UNITS_PER_TOKEN * max(1, -(-nonstructural // 4))

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
