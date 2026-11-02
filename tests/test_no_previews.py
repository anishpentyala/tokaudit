"""Opt-in reports omit raw content without changing duplicate accounting."""
import json
import pytest
from tokaudit.cli import main

SECRET='PRIVATE-MEMO-ZETA'
@pytest.mark.parametrize('fmt',['json','md'])
@pytest.mark.parametrize('config',[False,True])
def test_content_previews_can_be_omitted_without_losing_analysis(tmp_path,capsys,fmt,config):
    rows=[{'id':str(i),'model':'gpt-6-astra','prompt':(SECRET+' policy ')*300} for i in range(2)]
    log=tmp_path/'log.jsonl';log.write_text('\n'.join(json.dumps(r) for r in rows))
    normal=['analyze',str(log),'--tokenizer','estimate','--format',fmt]
    assert main(normal)==0
    original=capsys.readouterr().out
    assert SECRET in original
    flags=['--omit-content-previews']
    if config:
        cfg=tmp_path/'cfg.toml';cfg.write_text('omit_content_previews = true\n');flags=['--config',str(cfg)]
    assert main(normal+flags)==0
    output=capsys.readouterr().out
    assert SECRET not in output
    if fmt=='json':
        a,b=json.loads(original),json.loads(output)
        assert a['summary']==b['summary']
        assert len(a['duplicate_blocks'])==len(b['duplicate_blocks'])>0
        for before,after in zip(a['duplicate_blocks'],b['duplicate_blocks']):
            assert 'preview' not in after
            assert {k:v for k,v in before.items() if k!='preview'}==after
        assert b['assumptions']['content_previews_omitted'] is True
    else:
        assert 'content previews omitted' in output.lower()
        assert 'repeated tokens' in output


def test_default_report_contract_stays_unchanged(tmp_path,capsys):
    log=tmp_path/'log.jsonl';log.write_text('\n'.join(json.dumps({'id':str(i),'model':'gpt-6-astra','prompt':(SECRET+' policy ')*300}) for i in range(2)))
    assert main(['analyze',str(log),'--tokenizer','estimate','--format','json'])==0
    report=json.loads(capsys.readouterr().out)
    assert SECRET in report['duplicate_blocks'][0]['preview']
    assert 'content_previews_omitted' not in report['assumptions']

@pytest.mark.parametrize('value',['"true"','1','[]'])
def test_config_requires_real_preview_boolean(tmp_path,value):
    cfg=tmp_path/'cfg.toml';cfg.write_text('omit_content_previews = '+value+'\n')
    with pytest.raises(SystemExit,match='boolean'):
        main(['analyze','missing','--config',str(cfg)])

@pytest.mark.parametrize('fmt',['json','md'])
def test_mixed_provider_scenarios_and_file_output_omit_previews(tmp_path,capsys,fmt):
    rows=[{'id':f'{provider}-{i}','model':model,'provider':provider,'prompt':(SECRET+' policy ')*300}
          for provider,model in [('openai','gpt-6-astra'),('anthropic','opus-4-6')] for i in range(2)]
    log=tmp_path/'log.jsonl';log.write_text('\n'.join(json.dumps(r) for r in rows))
    output=tmp_path/('report.'+fmt)
    assert main(['analyze',str(log),'--tokenizer','estimate','--format',fmt,
                 '--omit-content-previews','--what-if','dedup','--output',str(output)])==0
    report=output.read_text()
    assert SECRET not in report
    assert SECRET not in capsys.readouterr().out
    if fmt=='json':
        groups=json.loads(report)['groups']
        assert len(groups)==2
        assert all(g['assumptions']['content_previews_omitted'] for g in groups)
        assert all(g['duplicate_blocks'] for g in groups)
        assert all(g['what_if'] for g in groups)
        assert all('preview' not in d for g in groups for d in g['duplicate_blocks'])


def test_omission_does_not_claim_to_anonymize_ids(tmp_path,capsys):
    log=tmp_path/'log.jsonl';log.write_text('\n'.join(json.dumps({'id':SECRET+str(i),'model':'gpt-6-astra','prompt':'policy '*300}) for i in range(2)))
    assert main(['analyze',str(log),'--tokenizer','estimate','--format','json','--omit-content-previews'])==0
    report=json.loads(capsys.readouterr().out)
    assert SECRET in json.dumps(report['duplicate_blocks'][0]['requests'])
