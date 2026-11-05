"""One-token tolerance must use represented numbers, not rounded substitutes."""
import json
import pytest
from tokaudit.integrity import check_record, build_integrity_report


def row(shape, prompt, completion, total):
    if shape == 'langfuse':
        return {'id':'r','usageDetails':{'input':prompt,'output':completion,'total':total}}
    key = 'request_id' if shape == 'spendlogs' else 'id'
    cost = 'spend' if shape == 'spendlogs' else 'response_cost'
    return {key:'r','startTime':'t',cost:1,'prompt_tokens':prompt,
            'completion_tokens':completion,'total_tokens':total}


@pytest.mark.parametrize('shape',['spendlogs','standard','langfuse'])
@pytest.mark.parametrize('gap',[-3,-2,-1,0,1,2,3])
def test_large_integer_usage_gap_has_exact_one_unit_tolerance(shape, gap):
    n = 2**54
    findings = check_record(row(shape,n,0,n+gap),1)
    assert any(f.invariant=='usage_sum_mismatch' for f in findings) is (abs(gap)>1)


@pytest.mark.parametrize('shape',['spendlogs','standard','langfuse'])
def test_mixed_integer_float_addition_does_not_round_before_comparison(shape):
    n = 2**54
    assert any(f.invariant=='usage_sum_mismatch' for f in check_record(row(shape,n,3.0,n),1))


def test_finite_float_sum_has_no_infinity_in_diagnostic():
    finding = next(f for f in check_record(row('standard',1e308,1e308,1e308),1)
                   if f.invariant=='usage_sum_mismatch')
    assert 'inf' not in finding.detail.lower()


@pytest.mark.parametrize('cache_gap', [0,1,2])
def test_cache_explanation_uses_same_exact_tolerance(cache_gap):
    n = 2**54
    record = row('spendlogs',0,0,n+cache_gap)
    record['cache_read_input_tokens'] = n
    findings = check_record(record,1)
    assert any(f.invariant=='usage_sum_mismatch' for f in findings) is (cache_gap>1)


def test_matching_large_integer_diagnostic_does_not_hide_difference():
    n = 2**54
    finding = next(f for f in check_record(row('standard',n,0,n+2),1)
                   if f.invariant=='usage_sum_mismatch')
    assert str(n) in finding.detail and str(n+2) in finding.detail


@pytest.mark.parametrize('fmt',['json','text'])
def test_cli_and_report_cannot_call_rounded_usage_clean(tmp_path,capsys,fmt):
    from tokaudit.cli import main
    path = tmp_path/'log.jsonl'
    path.write_text(json.dumps(row('standard',2**54,0,2**54+2)))
    assert build_integrity_report(str(path))['summary']['finding_count'] == 1
    assert main(['integrity',str(path),'--format',fmt,'--fail-on-findings']) == 1
    assert 'usage_sum_mismatch' in capsys.readouterr().out


@pytest.mark.parametrize('field',['cache_read_input_tokens','cache_creation_input_tokens'])
def test_cache_explanation_does_not_hide_infinite_float_accumulation(field):
    record = row('standard',1e308,1e308,1e308)
    record[field] = 1e308
    assert any(f.invariant=='usage_sum_mismatch' for f in check_record(record,1))


def test_langfuse_cache_gap_preserves_large_exact_buckets():
    n = 2**54
    record = row('langfuse',0,0,n+2)
    record['usageDetails']['input_cache_read'] = n
    assert any(f.invariant=='usage_sum_mismatch' for f in check_record(record,1))


def test_large_zero_spend_usage_finding_remains_available():
    record = row('standard',1e308,1e308,0)
    record['response_cost'] = 0
    findings = check_record(record,1)
    finding = next(f for f in findings if f.invariant=='zero_spend_with_usage')
    assert 'inf' not in finding.detail.lower()
