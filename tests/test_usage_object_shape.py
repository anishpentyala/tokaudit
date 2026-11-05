"""Malformed present normalized usage cannot quietly disappear from coverage."""
import json
import pytest
from tokaudit.logs import parse_record,load_log

@pytest.mark.parametrize('value',[True,False,0,1,1.5,'', 'SYNTHETIC_PRIVATE_USAGE',[],[{'prompt_tokens':4}]])
def test_malformed_actual_usage_is_rejected_without_echo(value):
    with pytest.raises(ValueError,match='actual_usage must be an object or null') as error:
        parse_record({'id':'r','prompt':'text','actual_usage':value})
    assert 'SYNTHETIC_PRIVATE_USAGE' not in str(error.value)

@pytest.mark.parametrize('value',[None,{}, {'prompt_tokens':0}, {'prompt_tokens':'4'}])
def test_missing_null_empty_or_valid_usage_remain_supported(value):
    request=parse_record({'prompt':'text','actual_usage':value})
    assert request.actual_usage==({'prompt_tokens':int(value['prompt_tokens'])}
                                   if value and 'prompt_tokens' in value else None if value is None else {})
    assert parse_record({'prompt':'text'}).actual_usage is None

def test_loader_rejects_whole_log_with_physical_line_context(tmp_path):
    path=tmp_path/'log.jsonl'
    path.write_text('\n'+json.dumps({'prompt':'first','actual_usage':{'prompt_tokens':1}})+'\n\n'+
                    json.dumps({'prompt':'second','actual_usage':['SYNTHETIC_PRIVATE_USAGE']})+'\n')
    with pytest.raises(SystemExit) as error:load_log(str(path))
    message=str(error.value)
    assert ':4:' in message and 'actual_usage must be an object or null' in message
    assert 'SYNTHETIC_PRIVATE_USAGE' not in message

@pytest.mark.parametrize('mode',['json','md'])
def test_cli_does_not_publish_partial_or_no_usage_report(tmp_path,capsys,mode):
    from tokaudit.cli import main
    path=tmp_path/'log.jsonl'
    path.write_text(json.dumps({'model':'gpt-6-astra','prompt':'first','actual_usage':{'prompt_tokens':1}})+'\n'+
                    json.dumps({'model':'gpt-6-astra','prompt':'second','actual_usage':True}))
    with pytest.raises(SystemExit):
        main(['analyze',str(path),'--tokenizer','estimate','--format',mode])
    output=capsys.readouterr()
    assert not output.out


def test_malformed_shape_does_not_change_other_metadata_or_content():
    row={'id':'r','prompt':'hello','actual_usage':{'prompt_tokens':3},'trace':{'extra':'value'}}
    original=json.dumps(row)
    req=parse_record(row)
    assert req.actual_usage=={'prompt_tokens':3} and req.messages[0].text=='hello'
    assert json.dumps(row)==original
