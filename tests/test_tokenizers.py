import pytest

from tokaudit.tokenizers import CHUNK_CHARS, EstimatorBackend, get_backend

PROSE = (
    "The quick brown fox jumps over the lazy dog while the morning sun "
    "climbs over the hills and the birds begin their usual chorus. "
) * 8
CODE = '''
def rolling_average(values, window):
    """Compute the rolling mean over a list of floats."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i:i + window]) / window)
    return out
''' * 6
JSONISH = ('{"name": "tokaudit", "version": "0.1.0", "private": true, '
           '"scripts": {"test": "pytest", "build": "python -m build"}, '
           '"keywords": ["llm", "tokens", "caching"]}') * 6


def test_estimator_chunk_math():
    est = EstimatorBackend()
    assert est.count("") == 0
    assert est.count("abcd") == 1
    assert est.count("abcde") == 2
    toks = est.encode("abcdefghij")
    assert len(toks) == est.count("abcdefghij")


def test_estimator_prefix_stable():
    est = EstimatorBackend()
    full = est.encode("hello world, this is a shared prefix and then more")
    prefix = est.encode("hello world, this is a shared prefix")
    # Shared text yields a shared pseudo-token prefix up to chunk rounding.
    n = len(prefix)
    assert full[: n - 1] == prefix[: n - 1]


@pytest.mark.parametrize(
    "text,label,tol",
    [(PROSE, "prose", 0.30), (CODE, "code", 0.12), (JSONISH, "json", 0.10)],
)
def test_estimator_within_tolerance_of_tiktoken(text, label, tol):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("o200k_base")
    exact = len(enc.encode(text))
    est = EstimatorBackend().count(text)
    err = abs(est - exact) / exact
    assert err <= tol, f"{label}: estimator {est} vs exact {exact} (err {err:.2%}, tol {tol:.0%})"


def test_structural_blocks_count_double():
    """JSON-ish 4-char blocks (2+ structural chars) count as 2 pseudo-tokens;
    the decision is per block, preserving prefix stability."""
    est = EstimatorBackend()
    # '{"a"' is one 4-char block with 3 structural chars -> 2 tokens
    assert est.count('{"a"') == 2
    # plain prose block stays 1
    assert est.count("abcd") == 1
    # prefix stability with mixed content
    full = est.encode('pre {"a": 1} post')
    prefix = est.encode('pre {"a": 1}')
    n = len(prefix)
    assert full[: n - 1] == prefix[: n - 1]


def test_get_backend_fallback():
    assert get_backend("estimate").name == "estimate"
    auto = get_backend("auto")
    assert auto.name in ("exact", "estimate")
    with pytest.raises(ValueError):
        get_backend("nope")
