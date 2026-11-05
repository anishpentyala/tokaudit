"""Reuse proven equal runs, never a hash or an unverified candidate."""
import json
import pytest
from tokaudit.dedup import analyze_duplicates
from tokaudit.logs import Message,Request


class CountedInt(int):
    comparisons=0
    __hash__=int.__hash__
    def __eq__(self,other):
        type(self).comparisons+=1
        return int.__eq__(self,other)


def test_repeated_long_block_does_not_rescan_every_suffix():
    values={str(i):[CountedInt(x) for x in range(1000)] for i in range(3)}
    rows=[Request(str(i),'m','openai',[Message('user',str(i))]) for i in range(3)]
    CountedInt.comparisons=0
    result=analyze_duplicates(rows,values.__getitem__,min_tokens=100,shingle=20)
    assert len(result.blocks)==1 and result.blocks[0].token_count==1000
    assert len(result.blocks[0].occurrences)==3
    assert CountedInt.comparisons < 100_000, 'equal suffixes were repeatedly rescanned'


@pytest.mark.parametrize('offset',[0,3,20])
def test_mismatch_after_equal_run_is_not_skipped(offset):
    common=list(range(50))
    values={'a':common+[100]+list(range(50,100)),
            'b':[-1]*offset+common+[101]+list(range(50,100))}
    rows=[Request(x,'m','openai',[Message('user',x)]) for x in values]
    result=analyze_duplicates(rows,values.__getitem__,min_tokens=10,shingle=5)
    assert result.blocks and result.blocks[0].token_count==50


def test_run_cache_eviction_does_not_drop_late_matching_pairs():
    values={str(i):list(range(12)) for i in range(4100)}
    rows=[Request(x,'m','openai',[Message('user',x)]) for x in values]
    result=analyze_duplicates(rows,values.__getitem__,min_tokens=6,shingle=6,max_shingles=100_000)
    assert result.index_complete and result.blocks[0].token_count==12
    assert len(result.blocks[0].occurrences)==4100


def test_multiple_message_alignments_keep_distinct_runs():
    values={'a':list(range(20)), 'b':list(range(7))+[99]+list(range(8,20))}
    rows=[Request('one','m','openai',[Message('user','a'),Message('user','b')]),
          Request('two','m','openai',[Message('user','a')])]
    result=analyze_duplicates(rows,values.__getitem__,min_tokens=5,shingle=3)
    # Existing greedy suppression selects the12-token three-occurrence block.
    # Optimization preserves that heuristic, not a new longest-block policy.
    assert result.blocks[0].token_count==12
    assert set(result.blocks[0].occurrences)=={('one',0),('one',1),('two',0)}
