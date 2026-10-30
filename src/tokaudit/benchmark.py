"""Deterministic synthetic benchmark for the estimator backend.

Measures EstimatorBackend against tiktoken o200k_base on a seeded,
fully synthetic corpus (prose, code, JSON). The corpus is generated
from templates; it contains no real user or provider data. Results
describe estimator behaviour on THIS synthetic corpus only - they are
not accuracy claims about real workloads.

Run: python -m tokaudit.benchmark [--samples N] [--seed S] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .tokenizers import EstimatorBackend, TiktokenBackend

CATEGORIES = ("prose", "code", "json")

_SUBJECTS = ["the cache", "a retry loop", "the scheduler", "our metrics pipeline",
             "the queue depth", "this dashboard", "the batch window", "a cold start"]
_VERBS = ["reduces", "increases", "obscures", "reveals", "delays", "smooths", "amplifies"]
_OBJECTS = ["tail latency", "token usage", "memory pressure", "error rates",
            "cache hits", "queue drain time", "allocator churn"]
_ADVERBS = ["markedly", "slightly", "consistently", "occasionally", "sharply", "gradually"]

_CODE_TEMPLATES = [
    "def {name}({args}):\n    total = 0\n    for item in {iterable}:\n        total += item * {factor}\n    return total\n",
    "function {name}({args}) {{\n  let acc = 0;\n  for (const x of {iterable}) {{\n    acc += x - {factor};\n  }}\n  return acc;\n}}\n",
    "func {name}({args} []int) int {{\n    acc := 0\n    for _, v := range {iterable} {{\n        acc += v + {factor}\n    }}\n    return acc\n}}\n",
    "class {Name}:\n    def __init__(self, limit={factor}):\n        self.limit = limit\n        self.items = []\n\n    def push(self, value):\n        if len(self.items) >= self.limit:\n            raise OverflowError('limit exceeded')\n        self.items.append(value)\n",
]

_JSON_KEY_BANK = ["id", "name", "status", "latency_ms", "retries", "payload", "meta",
                  "created_at", "tokens_in", "tokens_out", "model", "region", "tags"]


def _prose_sample(rng: random.Random) -> str:
    n = rng.randint(3, 8)
    sentences = []
    for _ in range(n):
        s = f"{rng.choice(_SUBJECTS).capitalize()} {rng.choice(_ADVERBS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"
        if rng.random() < 0.4:
            s += f" when {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"
        sentences.append(s + ".")
    return " ".join(sentences)


def _code_sample(rng: random.Random) -> str:
    tmpl = rng.choice(_CODE_TEMPLATES)
    name = rng.choice(["accumulate", "summarize", "fold_values", "tally", "aggregate"])
    return tmpl.format(
        name=name + str(rng.randint(0, 99)),
        Name=name.capitalize() + str(rng.randint(0, 99)),
        args=", ".join(f"v{i}" for i in range(rng.randint(1, 3))),
        iterable=rng.choice(["values", "batch", "rows", "chunks"]),
        factor=rng.randint(2, 17),
    )


def _json_value(rng: random.Random, depth: int):
    if depth >= 4:
        return rng.choice([None, True, rng.randint(0, 9999), "leaf"])
    r = rng.random()
    if r < 0.35:
        return {rng.choice(_JSON_KEY_BANK) + str(rng.randint(0, 9)): _json_value(rng, depth + 1)
                for _ in range(rng.randint(1, 5))}
    if r < 0.6:
        return [_json_value(rng, depth + 1) for _ in range(rng.randint(1, 6))]
    return rng.choice([None, True, False, rng.randint(0, 99999), rng.uniform(0, 100),
                       "tok-" + str(rng.randint(100, 999))])


def _json_sample(rng: random.Random) -> str:
    obj = {rng.choice(_JSON_KEY_BANK) + str(i): _json_value(rng, 1)
           for i in range(rng.randint(2, 6))}
    return json.dumps(obj, indent=rng.choice([None, 2, 2, 4]))


def synthetic_corpus(samples_per_category: int = 30, seed: int = 20261029) -> Dict[str, List[str]]:
    """Deterministic synthetic corpus. Raises ValueError on bad inputs."""
    if samples_per_category < 1:
        raise ValueError("samples_per_category must be >= 1")
    corpus: Dict[str, List[str]] = {}
    for category in CATEGORIES:
        rng = random.Random((seed, category))
        if category == "prose":
            gen = _prose_sample
        elif category == "code":
            gen = _code_sample
        else:
            gen = _json_sample
        samples = [gen(rng) for _ in range(samples_per_category)]
        if any(not s for s in samples):
            raise ValueError(f"generator produced an empty {category} sample")
        corpus[category] = samples
    return corpus


@dataclass
class BenchmarkRow:
    category: str
    samples: int
    exact_mean: float
    mean_abs_pct: float
    median_abs_pct: float
    p90_abs_pct: float
    max_abs_pct: float


def _pct_rank(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def run_benchmark(corpus: Dict[str, List[str]],
                  estimator: Optional[EstimatorBackend] = None,
                  exact: Optional[TiktokenBackend] = None) -> List[BenchmarkRow]:
    """Measure estimator error per category. Fails closed on empty inputs."""
    estimator = estimator or EstimatorBackend()
    exact = exact or TiktokenBackend()
    rows: List[BenchmarkRow] = []
    for category in CATEGORIES:
        samples = corpus.get(category)
        if not samples:
            raise ValueError(f"benchmark corpus is missing category: {category}")
        errors: List[float] = []
        exact_counts: List[int] = []
        for text in samples:
            if not text:
                raise ValueError(f"empty sample in category: {category}")
            exact_n = exact.count(text)
            if exact_n <= 0:
                raise ValueError(f"exact backend returned no tokens for a {category} sample")
            est_n = estimator.count(text)
            errors.append(abs(est_n - exact_n) / exact_n * 100.0)
            exact_counts.append(exact_n)
        rows.append(BenchmarkRow(
            category=category,
            samples=len(samples),
            exact_mean=statistics.fmean(exact_counts),
            mean_abs_pct=statistics.fmean(errors),
            median_abs_pct=statistics.median(errors),
            p90_abs_pct=_pct_rank(errors, 90),
            max_abs_pct=max(errors),
        ))
    return rows


def to_markdown(rows: List[BenchmarkRow]) -> str:
    lines = [
        "| Category | Samples | Mean exact tokens | Mean abs err | Median abs err | p90 abs err | Max abs err |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.category} | {r.samples} | {r.exact_mean:.0f} | "
            f"{r.mean_abs_pct:.1f}% | {r.median_abs_pct:.1f}% | "
            f"{r.p90_abs_pct:.1f}% | {r.max_abs_pct:.1f}% |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the estimator on a deterministic synthetic corpus.")
    parser.add_argument("--samples", type=int, default=30, help="samples per category")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args(argv)
    rows = run_benchmark(synthetic_corpus(samples_per_category=args.samples, seed=args.seed))
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
