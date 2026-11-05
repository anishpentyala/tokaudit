"""Finite input timestamps cannot create non-finite compressed time."""
from dataclasses import replace
import math
import pytest
from tokaudit.logs import Request,Message
from tokaudit.rules import ProviderRules
from tokaudit.whatif import run_scenarios

RULES=ProviderRules('openai',1,1,1.25,.1,False,True)
ENC=lambda text:list(text.encode())


def requests(times):
    return [Request(str(i),'gpt-test','openai',[Message('user','shared '*40)],ts=t)
            for i,t in enumerate(times)]


@pytest.mark.parametrize('times',[
    [-1e308,1e308],[-1.7e308,0,1.7e308],[-1e308,0,1e308],
    [-1e308,-5e307,0,5e307,1e308],
])
def test_extreme_span_matches_equivalent_unit_interval(times):
    reqs=requests(times)
    normalized=[replace(r,ts=i/(len(reqs)-1)) for i,r in enumerate(reqs)]
    actual=run_scenarios(['dense'],reqs,ENC,RULES)[0]
    expected=run_scenarios(['dense'],normalized,ENC,RULES)[0]
    assert actual['result']==expected['result']
    assert all(math.isfinite(v) for v in actual['result'].values())
    assert [r.ts for r in reqs]==times


@pytest.mark.parametrize('times',[
    [-1e308,1e308],[1e308,1.0000000000000002e308],[1e20,1e20+16384],
    [-1e308,0,1e308],
])
def test_compressed_times_remain_finite_and_preserve_endpoints(times,monkeypatch):
    import tokaudit.whatif as whatif
    seen=[]
    original=whatif.simulate
    def capture(reqs,*args,**kwargs):
        seen.extend(r.ts for r in reqs)
        return original(reqs,*args,**kwargs)
    monkeypatch.setattr(whatif,'simulate',capture)
    run_scenarios(['dense'],requests(times),ENC,RULES)
    assert all(math.isfinite(t) for t in seen)
    assert seen[0]==0
    assert seen[-1]==1
    assert seen==sorted(seen)


@pytest.mark.parametrize('times',[[0.,7200.],[-7200.,0.],[1e9,1e9+7200.]])
def test_shift_invariance_and_ordinary_range(times):
    actual=run_scenarios(['dense'],requests(times),ENC,RULES)[0]
    expected=run_scenarios(['dense'],requests([0.,1.]),ENC,RULES)[0]
    assert actual['result']==expected['result']


@pytest.mark.parametrize('fmt',['json','md'])
def test_cli_extreme_finite_timestamps_matches_unit_interval(tmp_path,capsys,fmt):
    import json
    from tokaudit.cli import main
    def analyze(times):
        rows=[{'id':str(i),'model':'opus-4-6','ts':t,'prompt':'policy '*4000} for i,t in enumerate(times)]
        path=tmp_path/'logs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
        assert main(['analyze',str(path),'--tokenizer','estimate','--provider','anthropic',
                     '--what-if','dense','--format',fmt])==0
        return capsys.readouterr().out
    actual=analyze([-1e308,1e308])
    assert 'nan' not in actual.lower()
    if fmt=='json':
        scenario=json.loads(actual)['what_if'][0]
        expected=json.loads(analyze([0.,300.]))['what_if'][0]
        assert scenario['result']==expected['result']
    else:
        assert 'rebased to zero' in actual
