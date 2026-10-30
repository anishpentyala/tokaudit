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


def test_dense_script_characters_score_per_character():
    est = EstimatorBackend()
    # 10 unspaced CJK characters: ~1.5 chars/token -> 20 units -> 6 tokens.
    # The English word heuristic must NOT apply (would be 1 token).
    assert est.count("これはテストですよね") == 6
    # emoji score ~1 token each
    assert est.count("🌍🚀😀") == 3


def test_natural_word_length_cap():
    est = EstimatorBackend()
    # a 25-char alphabetic word is not an ordinary English word
    assert est.count("antidisestablishmentarian") == 7  # ceil(25/4) tokens
    assert est.count("short") == 1


def test_long_id_runs_score_dense():
    est = EstimatorBackend()
    # 40-char hex sha: letters+digits, no structural chars -> 2 chars/token
    assert est.count("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0") == 20
    # same length but no digits: ordinary per-character rule
    assert est.count("pneumonoultramicroscopicsilicovolcano") == 10


def test_prefix_stability_completed_words_exact():
    """Exact semantics: a prefix cut at a completed word or just after
    whitespace yields an exact pseudo-token prefix; a mid-word cut shares
    only up to the last completed word before the cut."""
    est = EstimatorBackend()
    full = "alpha beta gamma delta epsilon"
    # cut at completed word (next char is whitespace)
    prefix = "alpha beta"
    assert est.encode(full)[: len(est.encode(prefix))] == est.encode(prefix)
    # cut after whitespace
    prefix = "alpha beta "
    assert est.encode(full)[: len(est.encode(prefix))] == est.encode(prefix)
    # mid-word cut: shared prefix ends at last completed word ("alpha ")
    prefix = "alpha bet"
    shared = est.encode("alpha ")
    got = est.encode(full)[: len(est.encode(prefix))]
    assert got[: len(shared)] == shared
    assert got != est.encode(prefix)


def test_prefix_stability_dense_scripts():
    est = EstimatorBackend()
    full = "結果 these are mixed 結果テスト words"
    prefix = "結果 these are mixed"
    assert est.encode(full)[: len(est.encode(prefix))] == est.encode(prefix)
