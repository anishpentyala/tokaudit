"""Pricing must follow the simulated request, not a lossy ID->model map."""
import json
import pytest
from tokaudit.cli import main

@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('same_id',[False,True])
@pytest.mark.parametrize('missing_rate',[False,True])
def test_pricing_uses_each_simulated_model(tmp_path,capsys,reverse,same_id,missing_rate):
    rows=[{'id':'dup','model':'gpt-6-astra','prompt':'hi '*100,'ts':2},
          {'id':'dup' if same_id else 'other','model':'gpt-6.1-sol','prompt':'hi '*100,'ts':1}]
    if reverse:rows.reverse()
    log=tmp_path/'logs.jsonl';log.write_text('\n'.join(json.dumps(r) for r in rows))
    rates={'gpt-6-astra':{'input':1.,'cached_input':1.,'cache_write':1.}}
    if not missing_rate:rates['gpt-6.1-sol']={'input':100.,'cached_input':100.,'cache_write':100.}
    sheet=tmp_path/'prices.json';sheet.write_text(json.dumps({'models':rates}))
    assert main(['analyze',str(log),'--tokenizer','estimate','--format','json','--prices',str(sheet)])==0
    report=json.loads(capsys.readouterr().out);p=report['pricing']
    # Equal-length prompts stay below cache minimum. No cache price or policy
    # assumption is needed to witness incorrect model attribution.
    n=report['cache_simulation'][0]['total_input']
    expected=round(n*(1 if missing_rate else 101)/1e6,4)
    assert p['priced_requests']==(1 if missing_rate else 2)
    assert p['baseline_usd']==expected
    assert p['simulated_usd']==expected
    assert len(p['unpriced_requests'])==(1 if missing_rate else 0)

@pytest.mark.parametrize('ambiguous',[False,True])
def test_legacy_cost_without_model_only_uses_unique_request_id(ambiguous):
    from dataclasses import replace
    from tokaudit.logs import parse_record
    from tokaudit.rules import load_rules
    from tokaudit.prefix import simulate
    from tokaudit.report import build_report
    from tokaudit.tokenizers import get_backend
    requests=[parse_record({'id':'dup','model':'gpt-6-astra','prompt':'hi '*100})]
    if ambiguous:requests.append(parse_record({'id':'dup','model':'gpt-6.1-sol','prompt':'hi '*100}))
    ruleset=load_rules();rules=ruleset.for_provider('openai');b=get_backend('estimate')
    sim=simulate(requests,b.encode,rules)
    if 'model' in sim.per_request[0].__dataclass_fields__:
        sim.per_request=[replace(c,model='') for c in sim.per_request]
    rates={m:{'input':r,'cached_input':r,'cache_write':r} for m,r in [('gpt-6-astra',1.),('gpt-6.1-sol',100.)]}
    p=build_report(requests,sim,[],ruleset,rules,b.name,price_sheet=rates)['pricing']
    assert p['priced_requests']==(0 if ambiguous else 1)
    assert len(p['unpriced_requests'])==(2 if ambiguous else 0)
    if ambiguous:assert p['baseline_usd']==0


def test_markdown_price_total_uses_simulated_models(tmp_path,capsys):
    # Exercise rendering, not just the internal dictionary.
    rows=[{'id':'same','model':model,'prompt':'hi '*100} for model in ('gpt-6-astra','gpt-6.1-sol')]
    log=tmp_path/'logs.jsonl';log.write_text('\n'.join(json.dumps(r) for r in rows))
    sheet=tmp_path/'prices.json';sheet.write_text(json.dumps({'models':{
        'gpt-6-astra':{'input':1000.,'cached_input':1000.,'cache_write':1000.},
        'gpt-6.1-sol':{'input':100000.,'cached_input':100000.,'cache_write':100000.}}}))
    assert main(['analyze',str(log),'--tokenizer','estimate','--format','json','--prices',str(sheet)])==0
    p=json.loads(capsys.readouterr().out)['pricing']
    assert main(['analyze',str(log),'--tokenizer','estimate','--format','md','--prices',str(sheet)])==0
    text=capsys.readouterr().out
    assert f"${p['baseline_usd']:.4f}" in text
