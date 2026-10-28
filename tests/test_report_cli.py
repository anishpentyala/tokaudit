import json
import os
import subprocess
import sys

from tokaudit.cli import main
from tokaudit.report import build_report, to_markdown

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_log.jsonl")


def test_cli_analyze_markdown(tmp_path, capsys):
    out = tmp_path / "report.md"
    rc = main(["analyze", FIXTURE, "--tokenizer", "estimate", "--output", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "# tokaudit report" in text
    assert "## Assumptions and sources" in text
    assert "Simulated savings" in text


def test_cli_analyze_json(capsys):
    rc = main(["analyze", FIXTURE, "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "summary" in data
    assert data["summary"]["requests"] == 8


def test_cli_rules(capsys):
    rc = main(["rules"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "as of" in out and "openai" in out and "anthropic" in out


def test_report_savings_positive_on_fixture():
    from tokaudit.dedup import find_duplicates
    from tokaudit.logs import load_log
    from tokaudit.prefix import simulate
    from tokaudit.rules import load_rules
    from tokaudit.tokenizers import get_backend

    reqs = [r for r in load_log(FIXTURE) if r.provider == "openai"]
    ruleset = load_rules()
    backend = get_backend("estimate")
    rules = ruleset.for_provider("openai")
    rules.families["gpt6_verified_01x"].min_prefix_tokens = 100  # fixture prompts are below the real 1024 minimum
    sim = simulate(reqs, backend.encode, rules)
    dups = find_duplicates(reqs, backend.encode, min_tokens=128, shingle=40)
    report = build_report(reqs, sim, dups, ruleset, rules, backend.name)
    assert report["summary"]["simulated_savings_pct"] > 0
    assert report["duplicate_blocks"], "fixture system prompt should be flagged as repeated"
