"""Integrity-check tests. Every fixture record is SYNTHETIC, constructed to
reproduce the documented failure-class signature in the linked upstream
issue; none of it is observed provider data."""

import json
import subprocess
import sys

import pytest

from tokaudit.integrity import check_record, run_integrity


def _litellm(rid, **kw):
    rec = {"request_id": rid, "startTime": "2026-10-30T00:00:00Z",
           "model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    rec.update(kw)
    return rec


def _names(findings):
    return sorted(f.invariant for f in findings)


def test_zero_spend_with_usage_fires():
    # litellm#48937 signature: streamed call, usage present, spend 0
    f = check_record(_litellm("r1", spend=0, prompt_tokens=10,
                              completion_tokens=5, total_tokens=15), 1)
    assert _names(f) == ["zero_spend_with_usage"]


def test_zero_spend_not_flagged_for_errored_call():
    # a $0 record on a failed call is correct behavior (litellm#4604)
    f = check_record(_litellm("r1", spend=0, status="failure",
                              prompt_tokens=10, completion_tokens=0,
                              total_tokens=10), 1)
    assert "zero_spend_with_usage" not in _names(f)


def test_zero_spend_not_flagged_without_usage():
    # nothing to bill -> nothing to flag (fail closed on absence)
    f = check_record(_litellm("r1", spend=0), 1)
    assert f == []


def test_spend_cost_breakdown_disagreement():
    # litellm#40100 signature
    f = check_record(_litellm("r2", spend=0.0,
                              cost_breakdown={"total_cost": 0.0042}), 2)
    assert "spend_cost_breakdown_disagreement" in _names(f)


def test_spend_cost_breakdown_agreement_clean():
    f = check_record(_litellm("r2", spend=0.0042,
                              cost_breakdown={"total_cost": 0.0042}), 2)
    assert f == []


def test_usage_sum_mismatch():
    # langfuse#14987 signature, applied to the litellm usage triple
    f = check_record(_litellm("r3", spend=0.001, prompt_tokens=10,
                              completion_tokens=5, total_tokens=99), 3)
    assert _names(f) == ["usage_sum_mismatch"]


def test_usage_sum_clean():
    f = check_record(_litellm("r3", spend=0.001, prompt_tokens=10,
                              completion_tokens=5, total_tokens=15), 3)
    assert f == []


def test_upstream_cost_field_in_usage():
    # litellm#35829 signature: provider-reported cost inside usage
    f = check_record(_litellm("r4", spend=0.001, prompt_tokens=10,
                              completion_tokens=5, total_tokens=15,
                              usage={"cost": 0.001}), 4)
    assert _names(f) == ["upstream_cost_field_in_usage"]


def test_non_numeric_accounting_field():
    f = check_record(_litellm("r5", spend="0.01", prompt_tokens=10,
                              completion_tokens=5, total_tokens=15), 5)
    assert "non_numeric_accounting_field" in _names(f)


def test_non_finite_tokens_flagged_not_crashing():
    # json.loads accepts Infinity; it must surface as a finding, never as
    # a crash or a silent pass (same discipline as the PR #6 robustness work)
    line = ('{"request_id": "r6", "startTime": "t", "messages": [], '
            '"spend": 0.001, "prompt_tokens": Infinity, "completion_tokens": 5, '
            '"total_tokens": 15}')
    import io, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(line + "\n")
        path = fh.name
    try:
        findings = run_integrity(path)
    finally:
        os.unlink(path)
    names = _names(findings)
    assert "non_numeric_accounting_field" in names


def test_duplicate_spend_record(tmp_path):
    # litellm#25351 signature: one request_id billed twice
    recs = [
        _litellm("dup", spend=0.002, prompt_tokens=10, completion_tokens=5,
                 total_tokens=15),
        _litellm("dup", spend=0.002, prompt_tokens=10, completion_tokens=5,
                 total_tokens=15),
    ]
    p = tmp_path / "dup.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    names = _names(run_integrity(str(p)))
    assert names == ["duplicate_spend_record"]


def test_duplicate_zero_spend_not_flagged(tmp_path):
    # a $0 first occurrence is not a bill; only positively-billed repeats count
    recs = [_litellm("dup", spend=0, prompt_tokens=1, completion_tokens=1,
                     total_tokens=2),
            _litellm("dup", spend=0.002, prompt_tokens=1, completion_tokens=1,
                     total_tokens=2)]
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    names = _names(run_integrity(str(p)))
    assert "duplicate_spend_record" not in names
    assert "zero_spend_with_usage" in names  # the $0 row is its own finding


def test_langfuse_usage_sum_mismatch():
    rec = {"id": "o1", "type": "GENERATION", "model": "gpt-5",
           "usageDetails": {"input": 10, "output": 5, "total": 99}}
    f = check_record(rec, 1)
    assert _names(f) == ["usage_sum_mismatch"]


def test_helicone_row_clean(tmp_path):
    rec = {"request_id": "h1", "request_body": {"model": "gpt-5"},
           "prompt_tokens": 10, "completion_tokens": 5}
    assert check_record(rec, 1) == []


def test_fail_closed_on_garbage(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n")
    with pytest.raises(SystemExit):
        run_integrity(str(p))
    with pytest.raises(SystemExit):
        run_integrity(str(tmp_path / "missing.jsonl"))


def test_cli_exit_codes(tmp_path, capsys):
    from tokaudit.cli import main
    clean = _litellm("c1", spend=0.001, prompt_tokens=10,
                     completion_tokens=5, total_tokens=15)
    dirty = _litellm("d1", spend=0, prompt_tokens=10, completion_tokens=5,
                     total_tokens=15)
    pc = tmp_path / "clean.jsonl"
    pc.write_text(json.dumps(clean) + "\n")
    pd = tmp_path / "dirty.jsonl"
    pd.write_text(json.dumps(dirty) + "\n")
    assert main(["integrity", str(pc)]) == 0
    assert "0 finding" in capsys.readouterr().out
    assert main(["integrity", str(pd)]) == 0  # informational by default
    assert main(["integrity", str(pd), "--fail-on-findings"]) == 1
    out = capsys.readouterr().out
    assert "zero_spend_with_usage" in out and "48937" in out
