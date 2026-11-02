"""Reported token counts must not be truncated or converted from booleans."""
import json
import pytest
from tokaudit.logs import parse_record,load_log

@pytest.mark.parametrize('value',[True,False,-1,-1.5,1.5,'1.5',None,[],{},float('nan'),float('inf')])
@pytest.mark.parametrize('field',['prompt_tokens','cache_read_tokens','cache_write_tokens'])
def test_invalid_normalized_usage_is_rejected(value,field):
    with pytest.raises(ValueError,match='actual_usage'):
        parse_record({'prompt':'test','actual_usage':{field:value}})

@pytest.mark.parametrize('value',[0,1,100,'0','100',1.0])
def test_integral_nonnegative_counts_preserved(value):
    assert parse_record({'prompt':'test','actual_usage':{'prompt_tokens':value}}).actual_usage=={'prompt_tokens':int(value)}

@pytest.mark.parametrize('value',[True,-1,1.5])
def test_log_error_has_line_context(tmp_path,value):
    path=tmp_path/'log.jsonl';path.write_text(json.dumps({'prompt':'test','actual_usage':{'prompt_tokens':value}}))
    with pytest.raises(SystemExit,match=r':1:.*actual_usage'):
        load_log(str(path))

@pytest.mark.parametrize('value',[True,False,-1,-1.5,1.5,float('nan'),float('inf')])
@pytest.mark.parametrize('field',['prompt_tokens','prompt_cache_read_tokens','prompt_cache_write_tokens'])
def test_helicone_invalid_optional_counts_are_not_manufactured(value,field):
    from tokaudit.adapters import adapt
    source={'request_id':'h','request_body':{'prompt':'test'},field:value}
    normalized=adapt(source)
    assert not normalized.get('actual_usage')

@pytest.mark.parametrize('value',[0,1,1.0,200])
def test_helicone_valid_integral_counts_survive(value):
    from tokaudit.adapters import adapt
    normalized=adapt({'request_id':'h','request_body':{'prompt':'test'},'prompt_cache_read_tokens':value})
    assert normalized['actual_usage']=={'cache_read_tokens':int(value)}
