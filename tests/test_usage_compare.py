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
                {"role": "system", "content": "policy " * 2000},
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


def _langfuse_obs(i, usage):
    return {
        "id": f"lf-{i}", "traceId": "t", "projectId": "p", "type": "GENERATION",
        "startTime": "2026-10-29T08:00:00Z", "model": "claude-sonnet-4-6",
        "input": {"messages": [
            {"role": "system", "content": "policy " * 2000},
            {"role": "user", "content": f"question {i}"}]},
        "usageDetails": usage,
    }


def test_langfuse_usage_details_extracted(tmp_path, capsys):
    rows = [
        _langfuse_obs(0, {"input": 2900, "output": 50,
                          "cache_creation_input_tokens": 2400}),
        _langfuse_obs(1, {"input": 2900, "output": 50,
                          "cache_read_input_tokens": 2400,
                          "input_cache_read": 2400}),  # alias: no double count
    ]
    p = tmp_path / "lf.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    rc = main(["analyze", str(p), "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    pr = json.loads(capsys.readouterr().out)["provider_reported"]
    assert pr["requests_with_usage"] == 2
    assert pr["actual_cache_read_tokens"] == 2400
    assert pr["actual_cache_write_tokens"] == 2400


def test_langfuse_no_usage_details_means_no_section(tmp_path, capsys):
    p = tmp_path / "lf.jsonl"
    p.write_text(json.dumps(_langfuse_obs(0, None)))
    rc = main(["analyze", str(p), "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["provider_reported"] is None


def test_examples_helicone_file_parses(capsys):
    import pathlib
    ex = pathlib.Path(__file__).resolve().parent.parent / "examples" / "helicone_usage.jsonl"
    rc = main(["analyze", str(ex), "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    pr = json.loads(capsys.readouterr().out)["provider_reported"]
    assert pr["requests_with_usage"] == 5
    assert pr["actual_cache_read_tokens"] == 4476


def test_litellm_prompt_tokens_compared_cache_not_mapped(tmp_path, capsys):
    rows = [{
        "request_id": f"sp-{i}", "call_type": "completion",
        "startTime": "2026-10-29T08:00:00Z", "model": "gpt-6-astra",
        "custom_llm_provider": "openai", "session_id": "s-1",
        "messages": [{"role": "system", "content": "policy " * 2000},
                     {"role": "user", "content": f"question {i}"}],
        "prompt_tokens": 2900, "completion_tokens": 40,
        "cache_hit": "True" if i else "",
        "spend": 0.001, "total_tokens": 2940,
    } for i in range(3)]
    p = tmp_path / "spend.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    rc = main(["analyze", str(p), "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    pr = json.loads(capsys.readouterr().out)["provider_reported"]
    assert pr["requests_with_usage"] == 3
    assert pr["actual_prompt_tokens"] == 8700
    # cache_hit must not leak into cache comparison
    assert pr["actual_cache_read_tokens"] == 0
    assert pr["actual_cache_write_tokens"] == 0
