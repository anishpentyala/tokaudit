"""Aggregate agreement must not conceal request-level diagnostic differences."""
import json
import pytest
from tokaudit.logs import Request
from tokaudit.prefix import RequestCost,SimulationResult
from tokaudit.report import _provider_reported,to_markdown


def request(rid,usage,model='model'):
    return Request(rid,model,'openai',[],actual_usage=usage)


def cost(rid,read=100,written=20,total=200):
    return RequestCost(rid,total,read,written,total-read-written,float(total),100.)


def sim(*costs):return SimulationResult('openai',list(costs),400.,200.,0)


def test_equal_aggregate_hides_opposite_errors():
    data=_provider_reported([request('a',{'cache_read_tokens':0}),request('b',{'cache_read_tokens':200})],sim(cost('a'),cost('b')))
    assert data['actual_cache_read_tokens']==data['simulated_cache_read_tokens']==200
    assert [r['fields']['cache_read_tokens']['simulated_minus_reported'] for r in data['per_request']]==[100,-100]


@pytest.mark.parametrize('field,attribute,expected',[
 ('cache_read_tokens','read',100),('cache_write_tokens','written',20),('prompt_tokens','total_input',200)])
@pytest.mark.parametrize('actual',[0,7])
def test_present_fields_only(field,attribute,expected,actual):
    data=_provider_reported([request('a',{field:actual})],sim(cost('a')))
    row=data['per_request'][0]
    assert row['model']=='model'
    assert row['fields']=={field:{'reported':actual,'simulated':expected,'simulated_minus_reported':expected-actual}}


@pytest.mark.parametrize('kind',['request-duplicate','simulation-duplicate','unmatched','unknown-field'])
def test_only_unique_matches_with_supported_fields(kind):
    requests=[request('a',{'cache_read_tokens':10})];costs=[cost('a')]
    if kind=='request-duplicate':requests.append(request('a',{'cache_read_tokens':50}))
    if kind=='simulation-duplicate':costs.append(cost('a'))
    if kind=='unmatched':costs=[]
    if kind=='unknown-field':requests=[request('a',{'unknown_count':10})]
    data=_provider_reported(requests,sim(*costs))
    assert data['per_request']==[]


def test_original_input_order_preserved():
    data=_provider_reported([request('b',{'prompt_tokens':300}),request('a',{'prompt_tokens':200})],sim(cost('a'),cost('b')))
    assert [r['id'] for r in data['per_request']]==['b','a']


@pytest.mark.parametrize('fmt',['json','md'])
def test_cli_exposes_per_request_not_only_aggregate(tmp_path,capsys,fmt):
    from tokaudit.cli import main
    rows=[{'id':name,'model':'gpt-6-astra','prompt':'policy '*2000,'actual_usage':{'cache_read_tokens':actual}} for name,actual in [('a',0),('b',100)]]
    path=tmp_path/'log.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    assert main(['analyze',str(path),'--tokenizer','estimate','--format',fmt])==0
    output=capsys.readouterr().out
    if fmt=='json':assert len(json.loads(output)['provider_reported']['per_request'])==2
    else:
        assert 'Per-request usage comparison' in output
        assert '<code>a</code>' in output and '<code>b</code>' in output
        assert 'not billing findings' in output


def test_markdown_identifiers_are_literal_cells(tmp_path,capsys):
    from tokaudit.cli import main
    rid='a|b\n<user>`'
    path=tmp_path/'log.jsonl';path.write_text(json.dumps({'id':rid,'model':'gpt-6-astra','prompt':'policy '*2000,'actual_usage':{'prompt_tokens':7}}))
    assert main(['analyze',str(path),'--tokenizer','estimate'])==0
    output=capsys.readouterr().out
    assert '<code>a&#124;b\\n&lt;user&gt;&#96;</code>' in output


def test_markdown_bounded_json_complete(tmp_path,capsys):
    from tokaudit.cli import main
    rows=[{'id':f'r{i:03}','model':'gpt-6-astra','prompt':'policy '*2000,'actual_usage':{'prompt_tokens':7}} for i in range(25)]
    path=tmp_path/'log.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    assert main(['analyze',str(path),'--tokenizer','estimate'])==0
    text=capsys.readouterr().out
    assert 'First 20 of 25 comparable requests shown' in text
    assert text.index('First 20 of 25') < text.index('| Request |')
    table = text[text.index('| Request |'):text.index('## Recommendations')].strip().splitlines()
    assert len(table)==22
    assert all(line.startswith('|') and line.endswith('|') for line in table)
    assert '<code>r019</code>' in text and '<code>r020</code>' not in text
    assert main(['analyze',str(path),'--tokenizer','estimate','--format','json'])==0
    assert len(json.loads(capsys.readouterr().out)['provider_reported']['per_request'])==25
