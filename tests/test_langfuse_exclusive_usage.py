"""Standard exclusive text/cache usage buckets must not lose cached input."""
import pytest
from tokaudit.adapters import adapt


def actual(details):
    return adapt({'id':'lf','traceId':'t','type':'GENERATION','startTime':'2026-11-05',
                  'model':'gpt-6-astra','input':'text','usageDetails':details}).get('actual_usage',{})


def test_official_documented_openai_exclusive_bucket_example():
    assert actual({'input':86,'input_cached_tokens':17817,'output':188,'total':18091}) == {
        'prompt_tokens':17903,'cache_read_tokens':17817}


def test_distinct_ttl_write_buckets_sum_not_max():
    assert actual({'input':100,'input_cache_creation_5m':200,'input_cache_creation_1h':300}) == {
        'prompt_tokens':600,'cache_write_tokens':500}


def test_generic_write_remainder_adds_to_ttl_buckets():
    assert actual({'input':100,'cache_creation_input_tokens':500,
                   'input_cache_creation_5m':200,'input_cache_creation_1h':300}) == {
        'prompt_tokens':1100,'cache_write_tokens':1000}


def test_aggregate_input_without_cache_remains_same():
    assert actual({'input':100,'output':5}) == {'prompt_tokens':100}


@pytest.mark.parametrize('key',['input_cache_creation','cache_creation_input_tokens',
                               'cache_write_tokens','input_cache_write_tokens'])
def test_generic_write_aliases_add_to_read_and_input(key):
    assert actual({'input':10,'input_cache_read':20,key:30}) == {
        'prompt_tokens':60,'cache_read_tokens':20,'cache_write_tokens':30}


def test_same_bucket_aliases_are_not_double_counted():
    assert actual({'input':10,'input_tokens':10,'input_cached_tokens':20,
                   'cache_read_input_tokens':20,'input_cache_creation':30,
                   'cache_creation_input_tokens':30,'input_cache_creation_5m':40,
                   'input_cache_creation_1h':50}) == {
        'prompt_tokens':150,'cache_read_tokens':20,'cache_write_tokens':120}


@pytest.mark.parametrize('details',[
    {'input_cached_tokens':20},
    {'input':10,'input_audio_tokens':20},
    {'input':10,'input_custom_units':20},
    {'input':10,'input_cache_read':None},
    {'input':10,'input_cache_creation_5m':-1},
    {'input':10,'input_cache_creation_1h':1.5},
    {'input':10,'cache_creation_input_tokens':True},
])
def test_incomplete_or_unsupported_input_has_no_invented_prompt_total(details):
    assert 'prompt_tokens' not in actual(details)


@pytest.mark.parametrize('key',['input_cache_creation_5m','input_cache_creation_1h'])
@pytest.mark.parametrize('value',[None,-1,True,1.5,'20'])
def test_invalid_ttl_does_not_publish_partial_write_total(key,value):
    usage=actual({'input':10,'input_cache_creation':30,key:value})
    assert 'cache_write_tokens' not in usage
    assert 'prompt_tokens' not in usage


def test_output_and_total_are_not_input_buckets():
    assert actual({'input':10,'input_cached_tokens':20,'output':50,
                   'output_reasoning_tokens':30,'total':110}) == {
        'prompt_tokens':30,'cache_read_tokens':20}


def test_zero_buckets_are_observed_not_missing():
    assert actual({'input':0,'input_cached_tokens':0,
                   'input_cache_creation_5m':0,'input_cache_creation_1h':0}) == {
        'prompt_tokens':0,'cache_read_tokens':0,'cache_write_tokens':0}


@pytest.mark.parametrize('value',[None,-1,True,1.5,'20'])
def test_invalid_generic_write_does_not_hide_behind_valid_ttl(value):
    usage=actual({'input':10,'input_cache_creation':value,'input_cache_creation_5m':20})
    assert 'cache_write_tokens' not in usage
    assert 'prompt_tokens' not in usage


@pytest.mark.parametrize('fmt',['json','md'])
def test_cli_compares_inclusive_input_with_cache_coverage(tmp_path,capsys,fmt):
    import json
    from tokaudit.cli import main
    row={'id':'lf','traceId':'t','type':'GENERATION','startTime':'2026-11-05T00:00:00Z',
         'model':'opus-4-6','input':'policy '*4000,
         'usageDetails':{'input':100,'input_cached_tokens':200,
                         'input_cache_creation':30,'input_cache_creation_5m':40,
                         'input_cache_creation_1h':50,'output':25,'total':445}}
    path=tmp_path/'log.jsonl';path.write_text(json.dumps(row)+'\n')
    assert main(['analyze',str(path),'--tokenizer','estimate','--provider','anthropic',
                 '--format',fmt])==0
    output=capsys.readouterr().out
    if fmt=='json':
        usage=json.loads(output)['provider_reported']
        assert usage['actual_prompt_tokens']==420
        assert usage['actual_cache_read_tokens']==200
        assert usage['actual_cache_write_tokens']==120
        assert usage['coverage']=={'prompt_tokens':1,'cache_read_tokens':1,'cache_write_tokens':1}
    else:
        assert '420' in output
        assert '120' in output
