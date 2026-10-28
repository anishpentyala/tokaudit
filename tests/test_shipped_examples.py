"""Shipped fixtures and examples must keep analyzing cleanly.

These guard against silent drift between the files we ship to users
(examples/, tests/fixtures/) and the adapters the CLI actually runs.
"""

import json
import pathlib

import pytest

from tokaudit.cli import main

ROOT = pathlib.Path(__file__).resolve().parent.parent

SHIPPED = [
    "examples/quickstart.jsonl",
    "examples/helicone_usage.jsonl",
    "tests/fixtures/sample_log.jsonl",
    "tests/fixtures/langfuse_observations_v2.jsonl",
    "tests/fixtures/litellm_spendlogs.jsonl",
]


@pytest.mark.parametrize("rel", SHIPPED)
def test_shipped_file_analyzes(rel, capsys):
    path = ROOT / rel
    assert path.exists(), f"shipped file missing: {rel}"
    rc = main(["analyze", str(path), "--tokenizer", "estimate",
               "--format", "json"])
    assert rc == 0, rel
    out = capsys.readouterr().out
    data = json.loads(out)
    groups = data["groups"] if "groups" in data else [data]
    assert any(g["summary"]["requests"] > 0 for g in groups), rel
