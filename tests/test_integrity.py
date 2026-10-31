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


# --- Synthetic NEGATIVE cases: documented legitimate patterns must stay
# quiet, so the checks remain signals rather than false accusations.

def test_usage_sum_not_flagged_when_cache_fields_explain_gap():
    # Documented alternative accounting (e.g. a mapper that counts
    # cache_read/cache_creation OUTSIDE prompt+completion, unlike LiteLLM's
    # verified fold-in): the explicit cache fields exactly explain the gap,
    # so no finding.
    f = check_record(_litellm("r1", spend=0.01, prompt_tokens=100,
                              completion_tokens=50, total_tokens=180,
                              cache_read_input_tokens=20,
                              cache_creation_input_tokens=10), 1)
    assert "usage_sum_mismatch" not in _names(f)


def test_usage_sum_still_fires_when_cache_fields_do_not_explain_gap():
    # Gap larger than the documented cache components: real mismatch.
    f = check_record(_litellm("r1", spend=0.01, prompt_tokens=100,
                              completion_tokens=50, total_tokens=200,
                              cache_read_input_tokens=20), 1)
    assert "usage_sum_mismatch" in _names(f)


def test_usage_sum_finding_states_schema_assumption():
    f = check_record(_litellm("r1", spend=0.01, prompt_tokens=100,
                              completion_tokens=50, total_tokens=200), 1)
    (finding,) = [x for x in f if x.invariant == "usage_sum_mismatch"]
    assert "OpenAI/LiteLLM usage shape" in finding.detail
    assert "verify the schema" in finding.detail


def test_upstream_cost_field_not_flagged_for_openrouter():
    # OpenRouter's usage accounting API returns a usage cost field as a
    # documented contract; flagging it would be a false accusation.
    f = check_record(_litellm("r1", spend=0.01,
                              custom_llm_provider="openrouter",
                              prompt_tokens=10, completion_tokens=5,
                              total_tokens=15,
                              usage={"prompt_tokens": 10,
                                     "completion_tokens": 5,
                                     "cost": 0.0001}), 1)
    assert "upstream_cost_field_in_usage" not in _names(f)


def test_upstream_cost_field_fires_for_other_providers_with_caveat():
    f = check_record(_litellm("r1", spend=0.01,
                              custom_llm_provider="azure_ai",
                              prompt_tokens=10, completion_tokens=5,
                              total_tokens=15,
                              usage={"prompt_tokens": 10,
                                     "completion_tokens": 5,
                                     "cost": 0.0001}), 1)
    (finding,) = [x for x in f
                  if x.invariant == "upstream_cost_field_in_usage"]
    assert "legitimate contract" in finding.detail
    assert "OpenRouter" in finding.detail


def test_spend_breakdown_within_float_tolerance_not_flagged():
    # Rounding noise below the relative tolerance is not a disagreement.
    f = check_record(_litellm("r1", spend=0.1000000001,
                              cost_breakdown={"total_cost": 0.1}), 1)
    assert "spend_cost_breakdown_disagreement" not in _names(f)


def test_spend_breakdown_finding_names_legitimate_explanations():
    f = check_record(_litellm("r1", spend=0.0,
                              cost_breakdown={"total_cost": 0.05}), 1)
    (finding,) = [x for x in f
                  if x.invariant == "spend_cost_breakdown_disagreement"]
    assert "discounts or a unit mismatch" in finding.detail
    assert "not proof of misbilling" in finding.detail


def test_duplicate_spend_finding_names_retry_explanation(tmp_path):
    rows = [_litellm("dup", spend=0.02, prompt_tokens=1,
                     completion_tokens=1, total_tokens=2),
            _litellm("dup", spend=0.03, prompt_tokens=1,
                     completion_tokens=1, total_tokens=2)]
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    findings = run_integrity(str(p))
    (finding,) = [x for x in findings
                  if x.invariant == "duplicate_spend_record"]
    assert "retried or streamed re-emissions" in finding.detail


def test_zero_spend_finding_names_free_and_cached_explanations():
    f = check_record(_litellm("r1", spend=0, prompt_tokens=10,
                              completion_tokens=5, total_tokens=15), 1)
    (finding,) = [x for x in f if x.invariant == "zero_spend_with_usage"]
    assert "zero-priced models" in finding.detail
    assert "fully cached" in finding.detail or "cached calls" in finding.detail



def test_i2_cost_breakdown_nested_in_metadata():
    """Real SpendLogs rows nest cost_breakdown inside metadata JSON."""
    f = check_record(_litellm("r1", spend=0.0,
                     metadata={"cost_breakdown": {"total_cost": 0.0123}}), 1)
    assert "spend_cost_breakdown_disagreement" in _names(f)
    assert "0.0123" in f[0].detail


def test_i2_nested_breakdown_agrees_quiet():
    f = check_record(_litellm("r1", spend=0.0123,
                     metadata={"cost_breakdown": {"total_cost": 0.0123}}), 1)
    assert f == []


def test_i6_usage_object_nested_in_metadata():
    """SpendLogs rows nest the usage dict at metadata.usage_object."""
    f = check_record(_litellm("r1", spend=0.0, custom_llm_provider="anthropic",
                     metadata={"usage_object": {"cost": 0.004}}), 1)
    assert "upstream_cost_field_in_usage" in _names(f)


def test_i6_openrouter_nested_suppressed():
    f = check_record(_litellm("r1", spend=0.0, custom_llm_provider="openrouter",
                     metadata={"usage_object": {"cost": 0.004}}), 1)
    assert f == []


def test_i8_zero_cost_diagnostic_flagged():
    """The gateway's own zero-cost diagnosis is provider-recorded evidence."""
    f = check_record(_litellm("r1", spend=0.0,
                     zero_cost_diagnostic={"reason": "pricing_not_applied"}), 1)
    assert "zero_cost_diagnostic_flagged" in _names(f)
    f8 = [x for x in f if x.invariant == "zero_cost_diagnostic_flagged"][0]
    assert "pricing_not_applied" in f8.detail
    assert f8.anchor.endswith("48937")


def test_i8_absent_quiet():
    assert check_record(_litellm("r1", spend=0.01), 1) == []


def test_i8_zero_cost_diagnostic_non_dict_ignored():
    assert check_record(_litellm("r1", spend=0.01,
                        zero_cost_diagnostic="weird"), 1) == []


def test_examples_integrity_spendlogs_demo():
    """The shipped example must keep exercising every check family."""
    import pathlib
    ex = pathlib.Path(__file__).parent.parent / "examples" / "integrity_spendlogs.jsonl"
    names = set(_names(run_integrity(str(ex))))
    assert {"zero_spend_with_usage", "spend_cost_breakdown_disagreement",
            "usage_sum_mismatch", "duplicate_spend_record",
            "upstream_cost_field_in_usage", "non_numeric_accounting_field",
            "zero_cost_diagnostic_flagged"} <= names
