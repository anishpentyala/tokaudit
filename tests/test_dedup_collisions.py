"""Shingle hashes are an index, never evidence of token equality."""
import sys
import pytest
from tokaudit.dedup import find_duplicates
from tokaudit.logs import Request,Message

@pytest.mark.parametrize('shingle',[2,5,10])
def test_distinct_shingles_with_identical_hash_do_not_match(shingle):
    # Python integer hashes are modulo this value; these positive token IDs
    # produce equal tuple hashes but unequal token sequences deterministically.
    p=sys.hash_info.modulus
    seq={'a':[1]*20,'b':[1+p]*20}
    assert hash(tuple(seq['a'][:shingle]))==hash(tuple(seq['b'][:shingle]))
    rows=[Request(x,'m','generic',[Message('user',x)]) for x in seq]
    assert find_duplicates(rows,seq.__getitem__,min_tokens=shingle,shingle=shingle)==[]

@pytest.mark.parametrize('shingle',[2,5,10])
def test_hash_collision_does_not_hide_real_duplicate_group(shingle):
    p=sys.hash_info.modulus
    seq={'a':[1]*20,'b':[1+p]*20,'c':[1+p]*20}
    rows=[Request(x,'m','generic',[Message('user',x)]) for x in seq]
    blocks=find_duplicates(rows,seq.__getitem__,min_tokens=shingle,shingle=shingle)
    assert blocks
    assert set(blocks[0].occurrences)=={('b',0),('c',0)}
    assert blocks[0].token_count==20

@pytest.mark.parametrize('placement', ['same-message', 'prior-message', 'prior-request'])
def test_exact_groups_across_message_and_request_boundaries(placement):
    p = sys.hash_info.modulus
    seq = {'collision': [1] * 20, 'real': [1 + p] * 20,
           'combined': [1] * 20 + [2] * 5 + [1 + p] * 20}
    if placement == 'same-message':
        rows = [Request('a', 'm', 'generic', [Message('user', 'combined')])]
    elif placement == 'prior-message':
        rows = [Request('a', 'm', 'generic', [Message('user', 'collision'), Message('user', 'real')])]
    else:
        rows = [Request('collision', 'm', 'generic', [Message('user', 'collision')]),
                Request('a', 'm', 'generic', [Message('user', 'real')])]
    rows.append(Request('b', 'm', 'generic', [Message('user', 'real')]))
    blocks = find_duplicates(rows, seq.__getitem__, min_tokens=10, shingle=5)
    assert len(blocks) == 1
    a_message = 1 if placement == 'prior-message' else 0
    assert set(blocks[0].occurrences) == {('a', a_message), ('b', 0)}
    assert blocks[0].token_count == 20
