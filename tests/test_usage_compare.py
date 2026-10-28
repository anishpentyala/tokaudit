"""Provider-reported usage vs simulation comparison (Helicone rows)."""

import json
import os

from tokaudit.cli import main


def _helicone_log(tmp_path):
    rows = []
    base = 1761724800.0
    for i in range(3):
        rows.append({
            "request_id": f"h-{i}", "response_id": f"r-{i}",
            "request_created_at": "2026-10-29T08:00:00Z",
            "request_model": "gpt-6-astra", "provider": "OPENAI",
            "response_status": 200,
            "request_body": {"model": "gpt-6-astra", "messages": [
                {"role": "system", "content": "policy " * 700},
                {"role": "user", "content": f"question {i}"}]},
            "response_body": {},
            "prompt_tokens": 2900,
            "prompt_cache_read_tokens": 2400 if i else 0,
            "prompt_cache_write_tokens": 2400 if i == 0 else 0,
        })
    p = tmp_path / "helicone.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_provider_reported_section_json(tmp_path, capsys):
    rc = main(["analyze", _helicone_log(tmp_path), "--tokenizer", "estimate",
               "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    pr = data["provider_reported"]
    assert pr["requests_with_usage"] == 3
    assert pr["actual_cache_read_tokens"] == 4800
    assert pr["actual_cache_write_tokens"] == 2400
    assert pr["simulated_cache_read_tokens"] > 0
    assert "not expected" in pr["note"]


def test_no_usage_no_section(capsys):
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_log.jsonl")
    rc = main(["analyze", fixture, "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    groups = data["groups"] if "groups" in data else [data]
    for g in groups:
        assert g["provider_reported"] is None


def test_markdown_section(tmp_path):
    out = tmp_path / "r.md"
    rc = main(["analyze", _helicone_log(tmp_path), "--tokenizer", "estimate",
               "--output", str(out)])
    assert rc == 0
    assert "Provider-reported vs simulated" in out.read_text()
