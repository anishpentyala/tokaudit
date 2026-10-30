import pytest

from tokaudit.tokenizers import EstimatorBackend, get_backend

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


def test_estimator_word_math():
    est = EstimatorBackend()
    assert est.count("") == 0
    # one natural-language word scores 4/3 pseudo-tokens, floored to 1
    assert est.count("abcd") == 1
    # three natural words score 3 * 4/3 = 4 pseudo-tokens
    assert est.count("abcd abcd abcd") == 4
    # a 10-char symbol-heavy word scores ceil(10/4) = 3 pseudo-tokens
    assert est.count("ab12-+/*^~") == 3
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


def test_structural_characters_cost_more():
    """Structural characters cost ~2/3 token each; the decision is per
    word, preserving prefix stability."""
    est = EstimatorBackend()
    # '{"a"': 3 structural chars (3 * 2/3) + 1 letter (1/4-token floor)
    # -> 2 + 1 = 3 pseudo-tokens
    assert est.count('{"a"') == 3
    # plain prose word stays ~1
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
