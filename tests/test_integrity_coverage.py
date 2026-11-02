"""A clean integrity verdict requires actual accounting input coverage."""
import json
import pytest
from tokaudit.cli import main
from tokaudit.integrity import run_integrity

@pytest.mark.parametrize('row',[
    {},{'hello':'world'},
    {'data':[{'spend':0,'prompt_tokens':10}],'total':1,'page':1},
    {'model':'gpt-6-astra','messages':[{'role':'user','content':'secret'}]},
    {'request_id':'x','startTime':'t','messages':[]},
    {'request_id':'x','request_body':{'model':'gpt-6-astra'}},
    {'id':'x','type':'GENERATION','usageDetails':{}},
    {'id':'x','type':'GENERATION','usageDetails':'broken'},
    {'spend':0,'prompt_tokens':10},
])
def test_unchecked_input_does_not_look_clean(tmp_path,capsys,row):
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    out=tmp_path/'report.txt';out.write_text('preserve')
    with pytest.raises(SystemExit,match='accounting'):
        main(['integrity',str(p),'--fail-on-findings','--output',str(out)])
    assert '0 finding' not in capsys.readouterr().out
    assert out.read_text()=='preserve'

@pytest.mark.parametrize('text',['','\n \n'])
def test_no_records_is_not_clean_coverage(tmp_path,text):
    p=tmp_path/'log.jsonl';p.write_text(text)
    with pytest.raises(SystemExit,match='accounting'):run_integrity(str(p))

@pytest.mark.parametrize('row',[
    {'request_id':'r','startTime':'t','spend':.01,'prompt_tokens':10},
    {'id':'r','startTime':'t','response_cost':.01,'prompt_tokens':10},
    {'request_id':'r','request_body':{},'prompt_tokens':10},
    {'id':'r','type':'GENERATION','usageDetails':{'input':10,'output':5,'total':15}},
    {'id':'r','type':'GENERATION','usage_details':{'input':10}},
])
def test_supported_partial_accounting_stays_supported(tmp_path,row):
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    assert run_integrity(str(p))==[]


def test_mixed_supported_and_unchecked_rows_fail_with_line_context(tmp_path):
    p=tmp_path/'log.jsonl';p.write_text(json.dumps({'request_id':'r','startTime':'t','spend':.01})+'\n'+json.dumps({'hello':'world'})+'\n')
    with pytest.raises(SystemExit,match=r'log.jsonl:2:.*accounting'):run_integrity(str(p))

@pytest.mark.parametrize('value',[True,'10',float('nan'),float('inf'),10**400])
@pytest.mark.parametrize('shape',['slp','helicone','langfuse'])
def test_malformed_present_accounting_is_a_finding(tmp_path,value,shape):
    if shape=='slp':row={'id':'r','startTime':'t','response_cost':value};field='response_cost'
    elif shape=='helicone':row={'request_id':'r','request_body':{},'prompt_tokens':value};field='prompt_tokens'
    else:row={'id':'r','usageDetails':{'input':value}};field='usageDetails.input'
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    findings=run_integrity(str(p))
    assert any(f.invariant=='non_numeric_accounting_field' and field in f.detail for f in findings)

@pytest.mark.parametrize('extra',[
    {'metadata':{}},{'cost_breakdown':{}},{'usage':{}},
    {'zero_cost_diagnostic':{}},{'metadata':{'usage_object':{}}},
])
def test_empty_accounting_containers_are_not_coverage(tmp_path,extra):
    row={'request_id':'r','startTime':'t'};row.update(extra)
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    with pytest.raises(SystemExit,match='accounting'):run_integrity(str(p))

@pytest.mark.parametrize('extra',[
    {'spend':.01,'cost_breakdown':{'total_cost':.01}},
    {'usage':{'cost':.01}},
    {'zero_cost_diagnostic':{'reason':'missing_price'}},
    {'metadata':{'usage_object':{'cost':.01}}},
])
def test_actual_accounting_diagnostics_are_coverage(tmp_path,extra):
    row={'request_id':'r','startTime':'t'};row.update(extra)
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    assert isinstance(run_integrity(str(p)),list)

@pytest.mark.parametrize('extra',[
    {'cost_breakdown':{'total_cost':.01}},
    {'zero_cost_diagnostic':{'reason':10}},
    {'metadata':{'cost_breakdown':{'total_cost':.01}}},
])
def test_uncheckable_diagnostic_only_is_not_coverage(tmp_path,extra):
    row={'request_id':'r','startTime':'t'};row.update(extra)
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    with pytest.raises(SystemExit,match='accounting'):run_integrity(str(p))
