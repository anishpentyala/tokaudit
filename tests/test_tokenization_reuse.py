"""Local reuse must preserve token identity, output and caller isolation."""
import json
from pathlib import Path

import pytest
from tokaudit import tokenizers


def cache(encode, **kwargs):
    assert hasattr(tokenizers, '_BoundedEncoder'), 'bounded token reuse missing'
    return tokenizers._BoundedEncoder(encode, **kwargs)


def test_hits_do_not_reencode_and_mutation_cannot_poison_cache():
    calls = []
    def encode(text):
        calls.append(text)
        return [ord(c) for c in text]
    reuse = cache(encode)
    first = reuse('same'); first.append(-1)
    second = reuse('same'); second.clear()
    assert reuse('same') == list(map(ord, 'same'))
    assert calls == ['same']


@pytest.mark.parametrize('limits', [dict(max_entries=1), dict(max_tokens=3), dict(max_chars=3)])
def test_lru_limits_evict_without_changing_results(limits):
    calls = []
    def encode(text):
        calls.append(text); return list(map(ord,text))
    reuse = cache(encode, **limits)
    assert reuse('aa') == [97,97]
    assert reuse('bb') == [98,98]
    assert reuse('aa') == [97,97]
    assert calls == ['aa','bb','aa']


def test_hit_refreshes_lru_and_empty_tokens_are_cached():
    calls = []
    def encode(text):
        calls.append(text); return []
    reuse = cache(encode, max_entries=2)
    for value in ['a','b','a','c','a','b']:
        assert reuse(value) == []
    assert calls == ['a','b','c','b']


@pytest.mark.parametrize('limits', [dict(max_entries=0),dict(max_tokens=1),dict(max_chars=1)])
def test_oversize_or_disabled_entries_bypass(limits):
    calls=[]
    reuse=cache(lambda text: calls.append(text) or [1,2], **limits)
    assert reuse('large') == reuse('large') == [1,2]
    assert calls == ['large','large']
    assert not reuse._entries


def test_exact_text_identity_and_arbitrary_integer_token_values():
    values={'a':[2**200], 'A':[-2**200], 'a ':[]}
    reuse=cache(values.__getitem__)
    for _ in range(2):
        for text, expected in values.items():
            assert reuse(text) == expected


def test_encoder_error_is_not_cached():
    calls=[]
    def bad(text):
        calls.append(text); raise ValueError('no tokens')
    reuse=cache(bad)
    for _ in range(2):
        with pytest.raises(ValueError,match='no tokens'): reuse('a')
    assert calls == ['a','a']


@pytest.mark.parametrize('field', ['max_entries','max_tokens','max_chars'])
@pytest.mark.parametrize('value', [-1,True,1.5])
def test_bad_cache_limits_rejected(field,value):
    assert hasattr(tokenizers,'_BoundedEncoder'), 'bounded token reuse missing'
    with pytest.raises(ValueError,match=field): tokenizers._BoundedEncoder(lambda _:[],**{field:value})


def test_cli_reuses_completed_encodes_across_scenario_passes(tmp_path, monkeypatch, capsys):
    """Runs on baseline too: exact duplicate input must encode once per analysis."""
    from tokaudit import cli
    from collections import Counter
    calls=Counter()
    class Backend:
        name='estimate'
        @staticmethod
        def encode(text):
            calls[text]+=1
            return list(map(ord,text))
    monkeypatch.setattr(cli,'get_backend',lambda _:Backend())
    log=tmp_path/'log.jsonl'
    prompt='stable text '*50
    log.write_text('\n'.join(json.dumps({'id':str(i),'model':'gpt-6-astra','prompt':prompt,'ts':i*60}) for i in range(3)))
    args=['analyze',str(log),'--format','json','--what-if','dense','--what-if','dedup','--what-if','openai-12h']
    assert cli.main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report['summary']['requests']==3 and len(report['what_if'])==3
    assert calls and max(calls.values())==1
    calls.clear()
    assert cli.main(args)==0
    assert max(calls.values())==1  # No cross-analysis global state.
    assert json.loads(capsys.readouterr().out)==report


@pytest.mark.parametrize('mode',['estimate','exact'])
def test_cli_cached_report_matches_uncached_report(tmp_path,monkeypatch,capsys,mode):
    from tokaudit import cli
    if mode=='exact': pytest.importorskip('tiktoken')
    fixture=str(Path(__file__).resolve().parents[1]/'examples'/'quickstart.jsonl')
    args=['analyze',fixture,'--tokenizer',mode,'--format','json','--what-if','dense','--what-if','dedup']
    assert cli.main(args)==0
    cached=capsys.readouterr().out
    if hasattr(cli,'_BoundedEncoder'): monkeypatch.setattr(cli,'_BoundedEncoder',lambda encode:encode)
    assert cli.main(args)==0
    assert capsys.readouterr().out==cached


def test_retained_totals_stay_inside_all_limits():
    reuse=cache(lambda text:list(range(len(text))),max_entries=3,max_tokens=7,max_chars=7)
    for text in ['a','bb','ccc','dddd','e','ff']*10:
        assert reuse(text)==list(range(len(text)))
        assert len(reuse._entries)<=3 and reuse._tokens<=7 and reuse._chars<=7
        assert reuse._tokens==sum(map(len,reuse._entries.values()))
        assert reuse._chars==sum(map(len,reuse._entries))


def test_oversized_item_does_not_evict_reusable_small_item():
    calls=[]
    def encode(text):
        calls.append(text); return list(map(ord,text))
    reuse=cache(encode,max_tokens=3,max_chars=3)
    assert reuse('a')==[97]
    assert reuse('oversize')==list(map(ord,'oversize'))
    assert reuse('a')==[97]
    assert calls==['a','oversize']


def test_boundaries_accept_exact_budget_and_zero_budget_empty_values():
    reuse=cache(lambda text:[1]*len(text),max_entries=1,max_tokens=2,max_chars=2)
    assert reuse('aa')==[1,1] and reuse._tokens==2 and reuse._chars==2
    calls=[]
    empty=cache(lambda text:calls.append(text) or [],max_tokens=0,max_chars=0)
    assert empty('')==empty('')==[] and calls==['']
