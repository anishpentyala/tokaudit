"""TOML defaults must not silently bypass CLI value/type contracts."""
import pytest
from tokaudit.config import load_config
from tokaudit.cli import main

@pytest.mark.parametrize('entry',[
    'isolate_sessions = "false"','isolate_sessions = 0','isolate_sessions = []',
    'dup_min_tokens = true','dup_min_tokens = 1.5','dup_min_tokens = "128"',
    'dup_min_tokens = nan','dup_min_tokens = inf',
    'format = "jsno"','provider = "opneai"','tokenizer = "estmate"',
    'what_if = ["desne"]',
])
def test_config_rejects_invalid_values_before_any_report(tmp_path,capsys,entry):
    cfg=tmp_path/'config.toml';cfg.write_text(entry+'\n')
    out=tmp_path/'report.json';out.write_text('preserve')
    # Config is validated before opening the (deliberately missing) log.
    with pytest.raises(SystemExit,match='config file'):
        main(['analyze',str(tmp_path/'missing.jsonl'),'--config',str(cfg),'--output',str(out)])
    assert out.read_text()=='preserve'
    assert not capsys.readouterr().out

@pytest.mark.parametrize('entry,expected',[
    ('isolate_sessions = false',{'isolate_sessions':False}),
    ('isolate_sessions = true',{'isolate_sessions':True}),
    ('dup_min_tokens = 128',{'dup_min_tokens':128}),
    ('format = "json"',{'format':'json'}),
    ('format = "md"',{'format':'md'}),
    ('provider = "generic"',{'provider':'generic'}),
    ('provider = "auto"',{'provider':'auto'}),
    ('tokenizer = "estimate"',{'tokenizer':'estimate'}),
    ('what_if = ["dense", "dedup"]',{'what_if':['dense','dedup']}),
    ('policy = "custom-family"',{'policy':'custom-family'}),
    ('rules = "my-rules.json"',{'rules':'my-rules.json'}),
])
def test_valid_values_and_custom_policy_stay_supported(tmp_path,entry,expected):
    cfg=tmp_path/'config.toml';cfg.write_text(entry+'\n')
    assert load_config(str(cfg))==expected

@pytest.mark.parametrize('value',['"false"','0','[]'])
def test_invalid_isolation_never_changes_real_analysis(tmp_path,capsys,value):
    import json
    rows=[{'id':str(i),'model':'gpt-6-astra','prompt':'policy '*2000,'session_id':str(i)} for i in range(2)]
    log=tmp_path/'logs.jsonl';log.write_text('\n'.join(json.dumps(r) for r in rows))
    cfg=tmp_path/'config.toml';cfg.write_text('isolate_sessions = '+value+'\n')
    with pytest.raises(SystemExit,match='boolean'):
        main(['analyze',str(log),'--config',str(cfg),'--tokenizer','estimate','--format','json'])
    assert not capsys.readouterr().out


def test_invalid_format_cannot_silently_write_markdown(tmp_path,capsys):
    import json
    log=tmp_path/'log.jsonl';log.write_text(json.dumps({'model':'gpt-6-astra','prompt':'hello'})+'\n')
    cfg=tmp_path/'config.toml';cfg.write_text('format = "jsno"\n')
    with pytest.raises(SystemExit,match='format'):
        main(['analyze',str(log),'--config',str(cfg),'--tokenizer','estimate'])
    assert not capsys.readouterr().out
