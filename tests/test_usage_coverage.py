"""Compare only explicit usage fields on matched requests, never missing-as-zero."""
import pytest
from tokaudit.logs import Request
from tokaudit.prefix import RequestCost,SimulationResult
from tokaudit.report import _provider_reported


def cost(rid,read=100,written=20):return RequestCost(rid,200,read,written,80,200.,100.)
def req(rid,usage):return Request(rid,'model','openai',[],actual_usage=usage)
def sim(*rows):return SimulationResult('openai',list(rows),400.,200.,0)

@pytest.mark.parametrize('field,actual,simulated',[('cache_read_tokens','actual_cache_read_tokens','simulated_cache_read_tokens'),('cache_write_tokens','actual_cache_write_tokens','simulated_cache_write_tokens')])
def test_absent_cache_usage_is_unknown_not_zero(field,actual,simulated):
    report=_provider_reported([req('a',{'prompt_tokens':200})],sim(cost('a')))
    assert report[actual] is None
    assert report[simulated] is None
    assert report['coverage'][field]==0

@pytest.mark.parametrize('value',[0,7])
@pytest.mark.parametrize('field,actual,simulated,expected',[('cache_read_tokens','actual_cache_read_tokens','simulated_cache_read_tokens',100),('cache_write_tokens','actual_cache_write_tokens','simulated_cache_write_tokens',20)])
def test_reported_zero_is_distinct_from_absence(value,field,actual,simulated,expected):
    report=_provider_reported([req('a',{field:value}),req('b',{'prompt_tokens':200})],sim(cost('a'),cost('b',900,800)))
    assert report[actual]==value
    assert report[simulated]==expected
    assert report['coverage'][field]==1

@pytest.mark.parametrize('field,actual,simulated,expected',[('cache_read_tokens','actual_cache_read_tokens','simulated_cache_read_tokens',100),('cache_write_tokens','actual_cache_write_tokens','simulated_cache_write_tokens',20),('prompt_tokens','actual_prompt_tokens','simulated_prompt_tokens',200)])
def test_unsimulated_request_is_not_added_to_comparison(field,actual,simulated,expected):
    report=_provider_reported([req('a',{field:9}),req('unverified',{field:999})],sim(cost('a')))
    assert report[actual]==9
    assert report[simulated]==expected
    assert report['coverage'][field]==1
    assert report['requests_with_usage']==2
    assert report['requests_compared']==1

@pytest.mark.parametrize('side',['request','simulation'])
def test_ambiguous_duplicate_ids_do_not_pair_usage(side):
    requests=[req('a',{'cache_read_tokens':7})]
    rows=[cost('a')]
    if side=='request':requests.append(req('a',{'cache_read_tokens':99}))
    else:rows.append(cost('a',999))
    report=_provider_reported(requests,sim(*rows))
    assert report['requests_compared']==0
    assert report['actual_cache_read_tokens'] is None
    assert report['ambiguous_request_ids']==['a']


def test_empty_usage_stays_absent():assert _provider_reported([req('a',{})],sim(cost('a'))) is None

@pytest.mark.parametrize('fmt',['json','md'])
def test_cli_reports_field_specific_coverage(tmp_path,capsys,fmt):
    import json
    from tokaudit.cli import main
    rows=[{'id':'a','model':'gpt-6-astra','prompt':'policy '*2000,'actual_usage':{'prompt_tokens':2500}},
          {'id':'b','model':'gpt-6-astra','prompt':'policy '*2000,'actual_usage':{'cache_read_tokens':0}},
          {'id':'c','model':'gpt-6-astra','prompt':'policy '*2000}]
    path=tmp_path/'logs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert main(['analyze',str(path),'--tokenizer','estimate','--format',fmt])==0
    output=capsys.readouterr().out
    if fmt=='json':
        usage=json.loads(output)['provider_reported']
        assert usage['coverage']=={'prompt_tokens':1,'cache_read_tokens':1,'cache_write_tokens':0}
        assert usage['actual_cache_write_tokens'] is None
        assert usage['actual_cache_read_tokens']==0
    else:
        assert 'unknown (no comparable reported field)' in output
        assert 'coverage 1 request(s)' in output


def test_known_langfuse_model_retains_reported_usage(tmp_path,capsys):
    import json
    from tokaudit.cli import main
    rows=[{'id':f'lf-{i}','traceId':'trace','type':'GENERATION','startTime':'2026-10-29T08:00:00Z',
           'model':'opus-4-6','input':{'messages':[{'role':'user','content':'policy '*4000}]},
           'usageDetails':usage} for i,usage in enumerate([
               {'input':5000,'cache_creation_input_tokens':4000},
               {'input':5000,'cache_read_input_tokens':4000,'input_cache_read':4000}])]
    path=tmp_path/'logs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert main(['analyze',str(path),'--tokenizer','estimate','--format','json','--provider','anthropic'])==0
    usage=json.loads(capsys.readouterr().out)['provider_reported']
    assert usage['actual_cache_read_tokens']==4000
    assert usage['actual_cache_write_tokens']==4000
    assert usage['coverage']=={'prompt_tokens':2,'cache_read_tokens':1,'cache_write_tokens':1}
