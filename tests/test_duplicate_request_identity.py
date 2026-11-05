"""Exported IDs are labels, not unique identities for separate log rows."""
import json

import pytest
from tokaudit.dedup import analyze_duplicates
from tokaudit.logs import Request,Message


def analyze(ids, messages=None):
    rows=[Request(rid,'m','openai',[Message('user',text) for text in (messages[i] if messages else ['shared'])])
          for i,rid in enumerate(ids)]
    return analyze_duplicates(rows,lambda _:list(range(20)),min_tokens=10,shingle=5)


@pytest.mark.parametrize('ids',[['same','same'],['',''],['same','other','same']])
def test_duplicate_labels_do_not_hide_distinct_rows(ids):
    result=analyze(ids)
    assert len(result.blocks)==1
    block=result.blocks[0]
    assert block.token_count==20 and len(block.occurrences)==len(ids)
    assert block.request_indices==list(range(len(ids)))
    assert block.distinct_requests==len(ids)


def test_one_request_with_multiple_messages_is_not_cross_request_reuse():
    assert analyze(['only'],[['a','a']]).blocks==[]


def test_multiple_messages_count_occurrences_separately_from_requests():
    block=analyze(['same','same'],[['a','a'],['a']]).blocks[0]
    assert len(block.occurrences)==3 and block.distinct_requests==2
    assert block.request_indices==[0,0,1]
    assert block.repeated_tokens==40  # Existing repeated-volume semantics.


def test_suppression_uses_input_positions_not_colliding_labels():
    values={'a':list(range(20)),'b':list(range(100,120))}
    rows=[Request('same','m','openai',[Message('user',x)]) for x in ['a','a','b','b']]
    blocks=analyze_duplicates(rows,values.__getitem__,min_tokens=10,shingle=5).blocks
    assert len(blocks)==2
    assert {tuple(b.request_indices) for b in blocks}=={(0,1),(2,3)}


@pytest.mark.parametrize('fmt',['json','md'])
def test_cli_duplicate_ids_keep_record_coverage_and_honest_counts(tmp_path,capsys,fmt):
    from tokaudit.cli import main
    prompt='repeated policy sentence '*100
    path=tmp_path/'log.jsonl'
    path.write_text('\n'.join(json.dumps({'id':'same','model':'custom','messages':messages}) for messages in [
        [{'role':'user','content':prompt},{'role':'user','content':prompt}],
        [{'role':'user','content':prompt}]]))
    assert main(['analyze',str(path),'--tokenizer','estimate','--format',fmt])==0
    out=capsys.readouterr().out
    if fmt=='json':
        block=json.loads(out)['duplicate_blocks'][0]
        assert block['requests']==['same']  # Labels remain labels.
        assert block['occurrences']==3 and block['distinct_requests']==2
        assert block['request_indices']==[0,0,1]
    else:
        assert '3 message occurrences across 2 request records' in out


def test_legacy_manually_constructed_block_keeps_label_based_fallback():
    from tokaudit.dedup import DuplicateBlock
    block=DuplicateBlock(20,[('a',0),('b',0)],'sample')
    assert block.distinct_requests==2 and block.request_indices==[]
