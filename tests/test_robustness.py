"""Hostile-input robustness: fuzz sweeps and crash-class regressions.

Fail-closed contract: malformed logs must raise a clean SystemExit with
file:line context, never an uncaught traceback (TypeError,
UnicodeDecodeError, RecursionError, OSError, OverflowError). The
estimator must never crash, must be deterministic within a process, and
must keep its documented monotonicity guarantee at word boundaries.
"""

import json
import random
import string

import pytest

from tokaudit.logs import load_log
from tokaudit.tokenizers import EstimatorBackend


def _write(tmp_path, text=None, raw_bytes=None):
    p = tmp_path / "log.jsonl"
    if raw_bytes is not None:
        p.write_bytes(raw_bytes)
    else:
        p.write_text(text, encoding="utf-8")
    return str(p)


# --- crash-class regressions (each was an uncaught traceback) ---

def test_messages_wrong_type_fails_closed(tmp_path):
    p = _write(tmp_path, json.dumps({"model": "gpt-5", "messages": 5}) + "\n")
    with pytest.raises(SystemExit, match="messages must be a list"):
        load_log(p)


def test_actual_usage_none_value_fails_closed(tmp_path):
    p = _write(
        tmp_path,
        json.dumps({"prompt": "x", "actual_usage": {"prompt_tokens": None}}) + "\n",
    )
    with pytest.raises(SystemExit, match="actual_usage"):
        load_log(p)


def test_actual_usage_list_value_fails_closed(tmp_path):
    p = _write(
        tmp_path,
        json.dumps({"prompt": "x", "actual_usage": {"prompt_tokens": [3]}}) + "\n",
    )
    with pytest.raises(SystemExit, match="actual_usage"):
        load_log(p)


def test_invalid_utf8_file_fails_closed(tmp_path):
    p = _write(tmp_path, raw_bytes=b'{"prompt": "caf\xe9"}\n')
    with pytest.raises(SystemExit, match="not valid UTF-8"):
        load_log(p)


def test_deeply_nested_json_fails_closed(tmp_path):
    p = _write(tmp_path, '{"prompt":"x","extra":' + "[" * 10000 + "]" * 10000 + "}")
    with pytest.raises(SystemExit, match="nested too deeply"):
        load_log(p)


def test_missing_log_file_fails_closed(tmp_path):
    with pytest.raises(SystemExit, match="cannot read log"):
        load_log(str(tmp_path / "does-not-exist.jsonl"))


def test_helicone_nonfinite_usage_skipped_not_crash(tmp_path):
    # json.loads accepts NaN/Infinity; malformed reported-usage values are
    # skipped (no reported usage), matching the LiteLLM/Langfuse adapters.
    p = _write(
        tmp_path,
        '{"request_id":"r1","request_body":{"prompt":"x"},'
        '"prompt_tokens": Infinity, "prompt_cache_read_tokens": NaN}\n',
    )
    (req,) = load_log(p)
    assert req.actual_usage is None


def test_helicone_bool_usage_skipped(tmp_path):
    p = _write(
        tmp_path,
        '{"request_id":"r1","request_body":{"prompt":"x"},'
        '"prompt_tokens": true}\n',
    )
    (req,) = load_log(p)
    assert req.actual_usage is None


def test_non_finite_timestamp_treated_as_missing(tmp_path):
    p = _write(tmp_path, '{"model":"gpt-5","prompt":"x","ts": NaN}\n')
    (req,) = load_log(p)
    assert req.ts is None
    p = _write(tmp_path, '{"model":"gpt-5","prompt":"x","ts": -Infinity}\n')
    (req,) = load_log(p)
    assert req.ts is None


# --- parser fuzz sweep: only SystemExit or success, never a traceback ---

object_repr = "\ud800 surrogate content"

_FIELD_MUTATIONS = [
    lambda r: r.update({"messages": 5}),
    lambda r: r.update({"messages": {"role": "user"}}),
    lambda r: r.update({"messages": [None, {"role": "user", "content": "x"}]}),
    lambda r: r.update({"messages": [{"role": "user", "content": object_repr}]}),
    lambda r: r.update({"system": [1, 2, 3]}),
    lambda r: r.update({"actual_usage": {"prompt_tokens": None}}),
    lambda r: r.update({"actual_usage": {"prompt_tokens": [1]}}),
    lambda r: r.update({"actual_usage": {"prompt_tokens": {"x": 1}}}),
    lambda r: r.update({"ts": [2020]}),
    lambda r: r.update({"session_id": {"s": 1}}),
    lambda r: r.update({"prompt_cache_key": ["k"]}),
    lambda r: r.update({"id": {"nested": True}}),
    lambda r: r.update({"model": ["gpt-5"]}),
    lambda r: r.update({"provider": 42}),
    lambda r: r.update({"messages": [{"role": "user", "content": ["not-a-dict"]}]}),
    lambda r: r.update({"extra": "\ud800\udfff lone surrogates"}),
]


def test_record_mutation_sweep_never_tracebacks(tmp_path):
    base = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    for i, mutate in enumerate(_FIELD_MUTATIONS):
        rec = json.loads(json.dumps(base))
        mutate(rec)
        # records containing lone surrogates cannot round-trip through
        # strict UTF-8; write with surrogatepass where needed
        p = tmp_path / "m.jsonl"
        with open(p, "w", encoding="utf-8", errors="surrogatepass") as fh:
            fh.write(json.dumps(rec) + "\n")
        try:
            load_log(str(p))
        except SystemExit:
            pass  # expected: clean rejection
        except Exception as exc:
            raise AssertionError(
                f"mutation {i} escaped fail-closed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def test_random_structure_sweep_never_tracebacks(tmp_path):
    rng = random.Random(20261030)
    atoms = [None, True, 0, -1, 3.5, "", "x", [], {}, "\ud800", "你", "🙂"]
    keys = ["model", "messages", "prompt", "text", "system", "id", "ts",
            "provider", "session_id", "cache_key", "actual_usage", "role",
            "content", "type", "request_id", "request_body", "startTime"]

    def rnd(depth):
        if depth <= 0 or rng.random() < 0.4:
            return rng.choice(atoms)
        if rng.random() < 0.5:
            return [rnd(depth - 1) for _ in range(rng.randint(0, 4))]
        return {rng.choice(keys): rnd(depth - 1)
                for _ in range(rng.randint(0, 4))}

    for i in range(500):
        rec = rnd(3)
        p = tmp_path / "r.jsonl"
        with open(p, "w", encoding="utf-8", errors="surrogatepass") as fh:
            fh.write(json.dumps(rec) + "\n")
        try:
            load_log(str(p))
        except SystemExit:
            pass
        except Exception as exc:
            raise AssertionError(
                f"random record {i} escaped fail-closed: "
                f"{type(exc).__name__}: {exc}\nrecord: {rec!r:.200}"
            ) from exc


# --- estimator robustness ---

_UNICODE_POOL = (
    string.ascii_letters + string.digits + string.punctuation + " \t\n"
    + "你好吗日本語한국어ภาษไทยខ្មែរឡခြေခံ" + "🙂🚀👨‍👩‍👧‍👦❤️"
    + "\x00\x01\x1f\x7f" + "\ud800\udbff\udfff" + "\u200d\ufe0f"
    + "éüñø" + "\u2028\u2029"
)


def test_estimator_unicode_soup_never_crashes():
    rng = random.Random(20261030)
    be = EstimatorBackend()
    for _ in range(2000):
        text = "".join(rng.choice(_UNICODE_POOL)
                       for _ in range(rng.randint(0, 120)))
        n1 = be.count(text)
        n2 = be.count(text)
        assert n1 == n2 >= 0  # deterministic, non-negative


def test_estimator_huge_inputs_complete():
    be = EstimatorBackend()
    big_word = "a" * 1_000_000
    assert be.count(big_word) > 0
    many_words = "word " * 200_000
    assert be.count(many_words) > 0


def test_estimator_count_monotone_at_word_boundaries():
    """The documented guarantee direction: cutting at a completed word
    (or just after whitespace) can only extend the token sequence, so
    counts are monotone there. Sweeps random mixed-category texts."""
    rng = random.Random(20261030)
    be = EstimatorBackend()
    fragments = [
        "hello", "world", "{}", '{"key":', '"value"}', "x" * 25,
        "aaaaaaaaaaaaa123", "你好吗", "🙂🙂", "id-9f8a7b6c5d", "def",
        "f(x):", "return", "42", "lorem", "ipsum",
    ]
    for _ in range(2000):
        words = [rng.choice(fragments) for _ in range(rng.randint(1, 12))]
        text = " ".join(words)
        full = be.count(text)
        for cut in range(1, len(words) + 1):
            prefix = " ".join(words[:cut])
            assert be.count(prefix) <= full


def test_estimator_mid_word_cut_anomaly_documented():
    """Known, documented non-monotonicity OUTSIDE the guarantee: a cut
    that lands mid-word can produce MORE pseudo-tokens than the full
    text, because the cut word can score under the long-ID-run regime
    (no structural characters) while the completed word re-scores under
    the per-character regime. Pinned here so the boundary of the
    prefix-stability guarantee is explicit; callers must cut only at
    completed words (see the tokenizers module docstring)."""
    be = EstimatorBackend()
    prefix = "x aaaaaaaaaaaaa123"   # 16-char alnum run, no struct chars
    full = prefix + "{"
    assert be.count(prefix) > be.count(full)  # documented anomaly
    # ...while the completed-word cut of the same text is monotone:
    assert be.count("x aaaaaaaaaaaaa123{") <= be.count("x aaaaaaaaaaaaa123{ y")
