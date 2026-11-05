"""Canonical message identity must not stringify malformed roles into aliases."""
import json
import pytest
from tokaudit.logs import parse_record
from tokaudit.adapters import adapt

BAD=[None,True,False,0,1,1.5,[],{},['SYNTHETIC_PRIVATE_ROLE'],{'SYNTHETIC_PRIVATE_ROLE':1},'']

@pytest.mark.parametrize('role',BAD)
def test_present_malformed_role_is_rejected_without_value_echo(role):
    with pytest.raises(ValueError,match='message role must be a nonempty string') as error:
        parse_record({'messages':[{'role':role,'content':'same'}]})
    assert 'SYNTHETIC_PRIVATE_ROLE' not in str(error.value)

@pytest.mark.parametrize('role',['user','assistant','system','developer','custom-role','None','True','1','[]','{}'])
def test_nonempty_string_roles_keep_exact_identity(role):
    req=parse_record({'messages':[{'role':role,'content':'same'}]})
    assert req.messages[0].role==role


def test_omitted_role_retains_user_default():
    req=parse_record({'messages':[{'content':'same'}]})
    assert req.messages[0].role=='user'
    assert parse_record({'prompt':'same'}).messages[0].role=='user'

@pytest.mark.parametrize('kind',['plain','litellm','langfuse','helicone'])
def test_adapted_malformed_role_cannot_enter_cache_identity(kind):
    messages=[{'role':None,'content':'same'}]
    if kind=='plain':obj={'messages':messages}
    elif kind=='litellm':obj={'request_id':'l','startTime':0,'messages':messages}
    elif kind=='langfuse':obj={'id':'l','type':'GENERATION','startTime':0,'input':{'messages':messages}}
    else:obj={'request_id':'h','request_body':{'messages':messages}}
    with pytest.raises(ValueError,match='message role'):
        parse_record(adapt(obj))

@pytest.mark.parametrize('mode',['json','md'])
def test_cli_stops_at_physical_line_before_partial_report(tmp_path,capsys,mode):
    from tokaudit.cli import main
    path=tmp_path/'rows.jsonl'
    path.write_text(json.dumps({'prompt':'first'})+'\n\n'+json.dumps({'messages':[{'role':{},'content':'same'}]}))
    with pytest.raises(SystemExit) as error:
        main(['analyze',str(path),'--tokenizer','estimate','--format',mode])
    assert ':3:' in str(error.value) and 'message role must be a nonempty string' in str(error.value)
    assert capsys.readouterr().out==''

@pytest.mark.parametrize('role',['tool','function'])
def test_existing_unsupported_tool_role_rejection_unchanged(role):
    with pytest.raises(ValueError,match='unsupported tool/function'):
        parse_record({'messages':[{'role':role,'content':'same'}]})


def test_valid_request_not_mutated_or_normalized():
    obj={'messages':[{'role':'custom Role','content':'same'}]}
    before=json.dumps(obj)
    req=parse_record(obj)
    assert req.canonical()=='<custom Role>\nsame'
    assert json.dumps(obj)==before
