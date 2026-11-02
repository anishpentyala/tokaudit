"""The resolved config destination must govern where sensitive reports go."""
from pathlib import Path
import json
import pytest
from tokaudit.cli import main

@pytest.mark.parametrize('implicit',[False,True])
@pytest.mark.parametrize('fmt',['json','md'])
def test_config_output_is_honored(tmp_path,monkeypatch,capsys,implicit,fmt):
    monkeypatch.chdir(tmp_path)
    log=tmp_path/'requests.jsonl';log.write_text(json.dumps({'model':'gpt-6-astra','prompt':'private marker '*2000}))
    target=tmp_path/'configured-report.txt'
    config=tmp_path/'.tokaudit.toml';config.write_text(f'output = "{target}"\nformat = "{fmt}"\ntokenizer = "estimate"\n')
    assert main(['analyze',str(log),* ([] if implicit else ['--config',str(config)])])==0
    assert target.exists()
    stdout=capsys.readouterr().out
    assert stdout==f'wrote {target}\n'
    assert 'private marker' not in stdout
    if fmt=='json':assert json.loads(target.read_text())['summary']['requests']==1
    else:assert target.read_text().startswith('# tokaudit report')


def test_explicit_output_overrides_config_without_touching_it(tmp_path,capsys):
    log=tmp_path/'requests.jsonl';log.write_text(json.dumps({'model':'gpt-6-astra','prompt':'test'}))
    configured=tmp_path/'configured.json';configured.write_text('existing private report')
    selected=tmp_path/'chosen.json'
    config=tmp_path/'config.toml';config.write_text(f'output = "{configured}"\nformat = "json"\n')
    assert main(['analyze',str(log),'--config',str(config),'--output',str(selected)])==0
    assert configured.read_text()=='existing private report'
    assert json.loads(selected.read_text())['summary']['requests']==1
    assert capsys.readouterr().out==f'wrote {selected}\n'
