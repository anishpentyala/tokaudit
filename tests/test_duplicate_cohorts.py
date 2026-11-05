"""Short or divergent seed witnesses must not hide full duplicate cohorts."""
import itertools
import pytest
from tokaudit.dedup import analyze_duplicates
from tokaudit.logs import Request, Message


def analyze(values, order, min_tokens=6, shingle=3):
    rows = [Request(x, 'm', 'openai', [Message('user', x)]) for x in order]
    return analyze_duplicates(rows, values.__getitem__, min_tokens=min_tokens, shingle=shingle)


@pytest.mark.parametrize('order', list(itertools.permutations('abc')))
@pytest.mark.parametrize('distractor', [[1,2,3,99], [1,2,3,99,98,97], [1,2,3]])
def test_early_seed_distractor_does_not_hide_longer_pair(order, distractor):
    values = {'a': distractor, 'b': [1,2,3,4,5,6], 'c': [1,2,3,4,5,6]}
    result = analyze(values, order)
    assert result.index_complete
    assert len(result.blocks) == 1
    assert result.blocks[0].token_count == 6
    assert set(result.blocks[0].occurrences) == {('b',0),('c',0)}


def test_two_divergent_full_cohorts_both_survive():
    values = {'a':[1,2,3,4,5,6], 'b':[1,2,3,7,8,9],
              'c':[1,2,3,4,5,6], 'd':[1,2,3,7,8,9]}
    blocks = analyze(values, 'abcd').blocks
    assert {frozenset(x.occurrences) for x in blocks} == {
        frozenset({('a',0),('c',0)}), frozenset({('b',0),('d',0)})}


def test_short_earlier_offset_in_same_message_does_not_hide_later_match():
    values = {'a':[1,2,3,99,1,2,3,4,5,6], 'b':[1,2,3,4,5,6]}
    blocks = analyze(values, 'ab').blocks
    assert len(blocks) == 1 and blocks[0].token_count == 6


def test_shared_seed_alone_still_is_not_full_block():
    values = {'a':[1,2,3,4,5,6], 'b':[1,2,3,7,8,9]}
    assert analyze(values, 'ab').blocks == []


@pytest.mark.parametrize('minimum,shingle', [(2,3),(3,3),(6,3),(9,3)])
def test_cohort_partition_respects_exact_thresholds(minimum, shingle):
    values = {'a':[1,2,3,99], 'b':list(range(1,10)), 'c':list(range(1,10))}
    blocks = analyze(values, 'abc', min_tokens=minimum, shingle=shingle).blocks
    assert blocks
    assert all(block.token_count >= max(minimum, shingle) for block in blocks)


def test_unindexed_real_pair_is_not_synthesized_by_cohort_split():
    values = {'a':[1,2,3,99], 'b':[1,2,3,4,5,6], 'c':[1,2,3,4,5,6]}
    rows = [Request(x,'m','openai',[Message('user',x)]) for x in 'abc']
    result = analyze_duplicates(rows, values.__getitem__, min_tokens=6, shingle=3,
                                max_shingles=6)
    assert not result.index_complete and result.blocks == []
