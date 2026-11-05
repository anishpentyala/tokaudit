"""Balanced totals cannot make negative/fractional token counts valid."""
import json

import pytest

from tokaudit.cli import main
from tokaudit.integrity import check_record


def row(shape, value):
    if shape == 'spendlogs':
        return {'request_id':'a','startTime':'2026-11-05','spend':1,
                'prompt_tokens':value,'completion_tokens':2,'total_tokens':value+2}
    if shape == 'standard':
        return {'id':'a','startTime':'2026-11-05','response_cost':1,
                'prompt_tokens':value,'completion_tokens':2,'total_tokens':value+2}
    if shape == 'helicone':
        return {'request_id':'a','request_body':{},'prompt_tokens':value,'completion_tokens':2}
    return {'id':'a','usageDetails':{'input':value,'output':2,'total':value+2}}


@pytest.mark.parametrize('shape',['spendlogs','standard','helicone'])
@pytest.mark.parametrize('value',[-1,-.5,.5])
def test_invalid_domain_cannot_pass_balanced_usage_sum(shape,value):
    findings=check_record(row(shape,value),7)
    found=[f for f in findings if f.invariant=='invalid_token_count']
    assert found and all(f.line==7 and f.rid=='a' for f in found)
    assert all('nonnegative integer' in f.detail for f in found)
    assert not any(f.invariant=='usage_sum_mismatch' for f in findings)


@pytest.mark.parametrize('shape',['spendlogs','standard','helicone','langfuse'])
@pytest.mark.parametrize('value',[0,1,1.,123456789012345678901234567890])
def test_valid_integral_counts_remain_quiet(shape,value):
    assert not any(f.invariant=='invalid_token_count' for f in check_record(row(shape,value),1))


def test_negative_cost_is_not_mislabeled_as_negative_token_count():
    record=row('spendlogs',1);record['spend']=-.5
    assert not any(f.invariant=='invalid_token_count' for f in check_record(record,1))


@pytest.mark.parametrize('value',[True,'1',float('nan'),float('inf')])
def test_nonnumeric_values_keep_existing_single_diagnostic(value):
    record=row('spendlogs',1);record['prompt_tokens']=value
    findings=check_record(record,1)
    assert any(f.invariant=='non_numeric_accounting_field' for f in findings)
    assert not any(f.invariant=='invalid_token_count' for f in findings)


def test_cli_balanced_negative_count_is_not_clean(tmp_path,capsys):
    path=tmp_path/'log.jsonl';path.write_text(json.dumps(row('spendlogs',-1)))
    assert main(['integrity',str(path),'--format','json','--fail-on-findings'])==1
    result=json.loads(capsys.readouterr().out)
    assert result['summary']['invariant_counts']['invalid_token_count']==1
    assert result['findings'][0]['line']==1


@pytest.mark.parametrize('field',['prompt_tokens','completion_tokens','total_tokens'])
@pytest.mark.parametrize('value',[-1,.5])
def test_every_top_level_token_field_has_domain_validation(field,value):
    record=row('spendlogs',1);record[field]=value
    assert any(f.invariant=='invalid_token_count' and field in f.detail for f in check_record(record,1))


@pytest.mark.parametrize('field',['input','output','total'])
def test_flat_langfuse_usage_does_not_assume_integral_token_units(field):
    record=row('langfuse',1);record['usageDetails'][field]=.5
    assert not any(f.invariant=='invalid_token_count' for f in check_record(record,1))
