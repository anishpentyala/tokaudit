"""Tests for the synthetic estimator benchmark."""

import json

import pytest

from tokaudit.benchmark import (
    CATEGORIES,
    BenchmarkRow,
    main,
    run_benchmark,
    run_file_benchmark,
    synthetic_corpus,
    to_markdown,
)


def test_corpus_is_deterministic():
    a = synthetic_corpus(samples_per_category=5, seed=1)
    b = synthetic_corpus(samples_per_category=5, seed=1)
    assert a == b


def test_corpus_seed_changes_output():
    a = synthetic_corpus(samples_per_category=5, seed=1)
    b = synthetic_corpus(samples_per_category=5, seed=2)
    assert a != b


def test_corpus_shape_and_nonempty():
    corpus = synthetic_corpus(samples_per_category=7)
    assert set(corpus) == set(CATEGORIES)
    for samples in corpus.values():
        assert len(samples) == 7
        assert all(isinstance(s, str) and s.strip() for s in samples)


def test_corpus_rejects_zero_samples():
    with pytest.raises(ValueError):
        synthetic_corpus(samples_per_category=0)


def test_run_benchmark_fails_closed_on_missing_category():
    corpus = synthetic_corpus(samples_per_category=3)
    del corpus["json"]
    with pytest.raises(ValueError):
        run_benchmark(corpus)


def test_run_benchmark_fails_closed_on_empty_sample():
    corpus = synthetic_corpus(samples_per_category=3)
    corpus["code"][1] = ""
    with pytest.raises(ValueError):
        run_benchmark(corpus)


def test_run_benchmark_rows_are_finite_and_bounded():
    rows = run_benchmark(synthetic_corpus(samples_per_category=10))
    assert len(rows) == len(CATEGORIES)
    for row in rows:
        assert isinstance(row, BenchmarkRow)
        assert row.samples == 10
        assert row.exact_mean > 0
        for stat in (row.mean_abs_pct, row.median_abs_pct, row.p90_abs_pct, row.max_abs_pct):
            assert 0.0 <= stat < 100.0
        assert row.p90_abs_pct <= row.max_abs_pct
        assert row.median_abs_pct <= row.max_abs_pct


def test_estimator_median_error_within_generous_bound():
    rows = run_benchmark(synthetic_corpus(samples_per_category=10))
    for row in rows:
        # Generous bound; the README table carries the measured values.
        assert row.median_abs_pct < 40.0, row


def test_markdown_table_renders_all_categories():
    rows = run_benchmark(synthetic_corpus(samples_per_category=3))
    table = to_markdown(rows)
    assert "| Category |" in table
    for category in CATEGORIES:
        assert f"| {category} |" in table


def test_cli_json_output(capsys):
    rc = main(["--json", "--samples", "3", "--seed", "7"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["category"] for row in payload} == set(CATEGORIES)
    assert all(row["samples"] == 3 for row in payload)


def test_cli_markdown_output(capsys):
    rc = main(["--samples", "3"])
    assert rc == 0
    assert "| Category |" in capsys.readouterr().out


def test_file_benchmark_roundtrip(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("The quick brown fox jumps over the lazy dog. " * 200)
    row = run_file_benchmark(str(f), [200, 800])
    assert row.category == "sample.txt"
    assert row.samples > 0
    assert 0.0 <= row.median_abs_pct < 100.0
    assert row.exact_mean > 0


def test_file_benchmark_fails_closed(tmp_path):
    with pytest.raises(ValueError):
        run_file_benchmark(str(tmp_path / "missing.txt"), [200])
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n  ")
    with pytest.raises(ValueError):
        run_file_benchmark(str(empty), [200])
    short = tmp_path / "short.txt"
    short.write_text("tiny")
    with pytest.raises(ValueError):
        run_file_benchmark(str(short), [500])


def test_file_benchmark_rejects_bad_chunk_size(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("words " * 100)
    with pytest.raises(ValueError):
        run_file_benchmark(str(f), [0])


def test_cli_file_mode(tmp_path, capsys):
    f = tmp_path / "log_excerpt.txt"
    f.write_text('{"role": "user", "content": "hello"}' * 50)
    rc = main(["--file", str(f), "--chunk-sizes", "100", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["category"] == "log_excerpt.txt"


def test_corpus_deterministic_across_processes():
    """Regression: the corpus must be identical across processes and
    Python versions. It previously seeded random.Random with a tuple,
    whose hash is randomized per process (and raises TypeError on
    3.11+); the corpus is now seeded with a string."""
    import hashlib
    import os
    import subprocess
    import sys

    snippet = (
        "import hashlib, json\n"
        "from tokaudit.benchmark import synthetic_corpus\n"
        "c = synthetic_corpus(5, 20261029)\n"
        "print(hashlib.sha256(json.dumps(c, sort_keys=True).encode())"
        ".hexdigest())\n"
    )
    digests = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, env=env, check=True,
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]
