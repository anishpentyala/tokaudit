"""CLI users can choose the duplicate index coverage/work tradeoff."""
import json

import pytest

from tokaudit.cli import main
from tokaudit.config import load_config


def log(tmp_path):
    path=tmp_path/'log.jsonl'
    path.write_text('\n'.join(json.dumps({'id':str(i),'model':'gpt-6-astra','prompt':'policy '*80}) for i in range(2)))
    return str(path)


def run(path,capsys,*args):
    assert main(['analyze',path,'--tokenizer','estimate','--format','json','--dup-min-tokens','20',*args])==0
    return json.loads(capsys.readouterr().out)


def test_zero_budget_disables_index_not_other_analysis(tmp_path,capsys):
    data=run(log(tmp_path),capsys,'--dup-max-shingles','0')
    assert data['duplicate_search']['indexed_shingles']==0
    assert not data['duplicate_search']['index_complete']
    assert not data['duplicate_blocks']
    assert data['summary']['requests_simulated']==2


def test_larger_budget_recovers_duplicate_missed_by_small_budget(tmp_path,capsys):
    path=log(tmp_path)
    small=run(path,capsys,'--dup-max-shingles','1')
    large=run(path,capsys,'--dup-max-shingles','1000')
    assert not small['duplicate_blocks'] and not small['duplicate_search']['index_complete']
    assert large['duplicate_blocks'] and large['duplicate_search']['index_complete']
    assert small['summary']==large['summary']
    assert small['cache_simulation']==large['cache_simulation']


def test_budget_defaults_preserve_current_behavior(tmp_path,capsys):
    data=run(log(tmp_path),capsys)
    assert data['duplicate_search']['max_shingles']==500000
    assert data['duplicate_search']['index_complete']


def test_config_budget_and_cli_override(tmp_path,capsys):
    path=log(tmp_path);cfg=tmp_path/'config.toml';cfg.write_text('dup_max_shingles = 0\n')
    assert load_config(str(cfg))=={'dup_max_shingles':0}
    assert not run(path,capsys,'--config',str(cfg))['duplicate_blocks']
    assert run(path,capsys,'--config',str(cfg),'--dup-max-shingles','1000')['duplicate_blocks']


@pytest.mark.parametrize('raw',['true','false','1.0','"100"','-1'])
def test_invalid_config_budget_rejected(tmp_path,raw):
    cfg=tmp_path/'cfg.toml';cfg.write_text('dup_max_shingles = '+raw+'\n')
    with pytest.raises(SystemExit,match='dup_max_shingles.*nonnegative integer'):
        load_config(str(cfg))


def test_invalid_cli_budget_preserves_existing_output(tmp_path):
    output=tmp_path/'report.json';output.write_text('prior report')
    with pytest.raises(SystemExit,match='dup_max_shingles must be a nonnegative integer'):
        main(['analyze',log(tmp_path),'--dup-max-shingles','-1','--output',str(output)])
    assert output.read_text()=='prior report'


def test_budget_is_per_provider_group_not_a_global_scan_limit(tmp_path,capsys):
    path=tmp_path/'groups.jsonl'
    path.write_text('\n'.join(json.dumps({'id':str(i),'model':model,'prompt':'policy '*80})
                             for i,model in enumerate(['gpt-6-astra','claude-opus-5'])))
    data=run(str(path),capsys,'--dup-max-shingles','1')
    assert len(data['groups'])==2
    assert all(g['duplicate_search']['indexed_shingles']==1 for g in data['groups'])
    assert all(g['duplicate_search']['max_shingles']==1 for g in data['groups'])


def test_zero_budget_markdown_names_partial_coverage(tmp_path,capsys):
    assert main(['analyze',log(tmp_path),'--tokenizer','estimate','--dup-max-shingles','0'])==0
    out=capsys.readouterr().out
    assert 'partial' in out.lower() and '0 /' in out
