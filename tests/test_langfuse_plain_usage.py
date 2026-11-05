"""Plain-text Langfuse input must retain the same usage as structured input."""
import json
import pytest
from tokaudit.adapters import adapt
from tokaudit.logs import parse_record
from tokaudit.cli import main

@pytest.mark.parametrize('input_value',['plain policy text','not json {','"plain policy text"'])
@pytest.mark.parametrize('details_key',['usageDetails','usage_details'])
def test_plain_input_retains_usage(input_value,details_key):
    row={'id':'lf-plain','traceId':'trace','type':'GENERATION','model':'gpt-6-astra',
         'startTime':'2026-10-29T08:00:00Z','input':input_value,
         details_key:{'input':123,'input_cache_read':0,'cache_creation_input_tokens':20}}
    normalized=adapt(row)
    assert normalized['actual_usage']=={'prompt_tokens':143,'cache_read_tokens':0,'cache_write_tokens':20}
    request=parse_record(normalized)
    assert request.actual_usage==normalized['actual_usage']
    expected='not json {' if input_value=='not json {' else 'plain policy text'
    assert request.messages[0].text==expected

@pytest.mark.parametrize('fmt',['json','md'])
def test_plain_input_cli_usage_coverage(tmp_path,capsys,fmt):
    row={'startTime':'2026-10-29T08:00:00Z','id':'plain','type':'GENERATION','traceId':'t','model':'gpt-6-astra','input':'policy '*2000,
         'usageDetails':{'input':3000,'input_cache_read':0}}
    p=tmp_path/'log.jsonl';p.write_text(json.dumps(row)+'\n')
    assert main(['analyze',str(p),'--tokenizer','estimate','--format',fmt])==0
    out=capsys.readouterr().out
    if fmt=='json':
        usage=json.loads(out)['provider_reported']
        assert usage['coverage']=={'prompt_tokens':1,'cache_read_tokens':1,'cache_write_tokens':0}
        assert usage['actual_prompt_tokens']==3000
        assert usage['actual_cache_read_tokens']==0
        assert usage['actual_cache_write_tokens'] is None
    else:
        assert 'coverage 1 request(s)' in out

@pytest.mark.parametrize('details',[None,{}, {'input':True,'input_cache_read':-1}, {'input':'123'}])
def test_plain_input_missing_or_invalid_optional_usage_stays_absent(details):
    row={'startTime':'2026-10-29T08:00:00Z','id':'plain','traceId':'t','input':'plain text','usageDetails':details}
    assert adapt(row)['prompt']=='plain text'
    assert 'actual_usage' not in adapt(row)
