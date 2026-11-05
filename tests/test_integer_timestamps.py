"""JSON integer epoch seconds must survive ingestion before TTL arithmetic."""
import json
import math

import pytest

from tokaudit.logs import load_log, parse_record
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules


@pytest.mark.parametrize('epoch',[2**53+1,2**54+2,-2**53-1,-2**54-2,10**100+1])
@pytest.mark.parametrize('field',['ts','timestamp','created'])
def test_direct_integer_timestamp_is_preserved_exactly(epoch,field):
    req=parse_record({'prompt':'x',field:epoch})
    assert type(req.ts) is int and req.ts == epoch


def row(kind,epoch,rid):
    if kind=='plain':return {'id':rid,'model':'m','prompt':'same','ts':epoch}
    if kind=='litellm':return {'request_id':rid,'model':'m','startTime':epoch,
                               'messages':[{'role':'user','content':'same'}]}
    if kind=='langfuse':return {'id':rid,'model':'m','type':'GENERATION','startTime':epoch,
                                'input':'same'}
    return {'request_id':rid,'request_model':'m','request_created_at':epoch,
            'request_body':{'messages':[{'role':'user','content':'same'}]}}


@pytest.mark.parametrize('kind',['plain','litellm','langfuse','helicone'])
@pytest.mark.parametrize('epoch',[2**54,-2**54])
@pytest.mark.parametrize('gap,ttl,hit',[(2,1,False),(3,3,True)])
def test_loaded_integer_seconds_do_not_invent_hit_or_miss(tmp_path,kind,epoch,gap,ttl,hit):
    path=tmp_path/'rows.jsonl'
    path.write_text('\n'.join(json.dumps(row(kind,t,rid))
                              for t,rid in [(epoch,'a'),(epoch+gap,'b')]))
    reqs=load_log(str(path))
    rules=ProviderRules('custom',1,ttl,1.25,.1,False,True)
    result=simulate(reqs,lambda s:list(s.encode()),rules)
    assert (result.per_request[1].read>0) is hit
    assert reqs[1].ts-reqs[0].ts == gap


@pytest.mark.parametrize('value',[True,False,10**400,-10**400,float('nan'),float('inf'),-float('inf'),[],{}])
def test_unsupported_timestamps_stay_missing_without_exception(value):
    assert parse_record({'prompt':'x','ts':value}).ts is None


@pytest.mark.parametrize('epoch',[0,1.,-1.5,1e100,None,'2026-11-05T10:00:00Z'])
def test_ordinary_supported_timestamp_behavior(epoch):
    result=parse_record({'prompt':'x','ts':epoch}).ts
    if epoch is None:assert result is None
    else:assert result is not None and math.isfinite(result)


def test_dense_scenario_can_use_preserved_integer_epoch_seconds():
    from tokaudit.whatif import run_scenarios
    rules=ProviderRules('custom',1,1,1.25,.1,False,True)
    reqs=[parse_record({'prompt':'same','ts':2**54+i,'id':str(i)}) for i in [0,3]]
    actual=run_scenarios(['dense'],reqs,lambda s:list(s.encode()),rules)[0]
    normalized=[parse_record({'prompt':'same','ts':i,'id':str(i)}) for i in [0,3]]
    expected=run_scenarios(['dense'],normalized,lambda s:list(s.encode()),rules)[0]
    assert actual['result']==expected['result']


@pytest.mark.parametrize('mode',['json','md'])
def test_cli_preserves_integer_ttl_boundary(tmp_path,capsys,mode):
    from tokaudit.cli import main
    # Shipped anthropic policy has a 300s write TTL; large epochs round the
    # 301s gap down to 300s, which previously invented a hit.
    path=tmp_path/'rows.jsonl'
    path.write_text('\n'.join(json.dumps({'id':str(i),'model':'opus-4-6',
                                          'prompt':'policy '*4000,'ts':2**54+i}) for i in [0,301]))
    assert main(['analyze',str(path),'--provider','anthropic','--tokenizer','estimate',
                 '--format',mode])==0
    output=capsys.readouterr()
    assert not output.err
    if mode=='json':
        data=json.loads(output.out)
        assert data['cache_simulation'][1]['cache_read'] == 0
    else:
        normalized=tmp_path/'normalized.jsonl'
        normalized.write_text('\n'.join(json.dumps({'id':str(i),'model':'opus-4-6',
                              'prompt':'policy '*4000,'ts':i}) for i in [0,301]))
        assert main(['analyze',str(normalized),'--provider','anthropic','--tokenizer','estimate',
                     '--format',mode])==0
        expected=capsys.readouterr().out
        cost=lambda text:next(line for line in text.splitlines() if line.startswith('- With caching:'))
        assert cost(output.out)==cost(expected)
    reqs=load_log(str(path))
    assert reqs[1].ts-reqs[0].ts == 301
