"""Findings must distinguish source context from the matched token region."""
import json
import pytest
from tokaudit.dedup import find_duplicates, DuplicateBlock
from tokaudit.logs import Message, Request
from tokaudit.prefix import simulate
from tokaudit.report import build_report, to_markdown
from tokaudit.rules import load_rules
from tokaudit.tokenizers import EstimatorBackend, TiktokenBackend


def rows(texts, repeated_ids=False):
    return [Request('same' if repeated_ids else str(i), 'gpt-6-astra', 'openai',
                    [Message('user', 'unique message ' + str(i)), Message('user', t)])
            for i, t in enumerate(texts)]


@pytest.mark.parametrize('backend', [EstimatorBackend, TiktokenBackend])
@pytest.mark.parametrize('repeated_ids', [False, True])
def test_exact_ranges_bind_every_observed_occurrence(backend, repeated_ids):
    if backend is TiktokenBackend:
        pytest.importorskip('tiktoken')
    enc = backend().encode
    shared = 'shared policy retention access review and reporting. ' * 45
    reqs = rows(['FIRST unrelated introduction. '*20 + shared + ' first suffix',
                 'SECOND unrelated introduction. '*7 + shared + ' second suffix',
                 'THIRD unrelated introduction. '*12 + shared + ' third suffix'], repeated_ids)
    blocks = find_duplicates(reqs, enc, min_tokens=100, shingle=20)
    assert blocks
    for block in blocks:
        spans = block.token_spans
        assert len(spans) == len(block.occurrences) == len(block.request_indices)
        slices = []
        for (rid, msg), position, span in zip(block.occurrences, block.request_indices, spans):
            assert span['request_index'] == position
            assert reqs[position].rid == rid
            assert span['message_index'] == msg == 1
            assert 0 < span['token_start'] < span['token_end']
            assert span['token_end'] - span['token_start'] == block.token_count
            tokens = enc(reqs[position].messages[msg].text)
            assert span['token_end'] <= len(tokens)
            slices.append(tokens[span['token_start']:span['token_end']])
        assert all(part == slices[0] for part in slices)


@pytest.mark.parametrize('omit', [False, True])
def test_report_labels_source_context_and_keeps_content_free_locations(omit):
    enc = EstimatorBackend().encode
    shared = 'shared policy retention access review and reporting. '*45
    reqs = rows(['PRIVATE_FIRST introduction. '*20+shared,
                 'OTHER introduction. '*10+shared], True)
    blocks = find_duplicates(reqs, enc, min_tokens=100, shingle=20)
    ruleset = load_rules(); rules = ruleset.for_provider('openai')
    report = build_report(reqs, simulate(reqs, enc, rules), blocks, ruleset, rules,
                          'estimate', omit_content_previews=omit)
    md = to_markdown(report)
    assert 'zero-based' in md and 'not the matched-span preview' in md
    for block, exported in zip(blocks, report['duplicate_blocks']):
        assert exported['token_spans'] == block.token_spans
        assert exported['preview_scope'] == 'source_message_start_not_matched_span'
        for span in exported['token_spans']:
            assert f"tokens [{span['token_start']}, {span['token_end']})" in md
    if omit:
        assert 'PRIVATE_FIRST' not in json.dumps(report) + md
        assert all('preview' not in d for d in report['duplicate_blocks'])
    else:
        assert 'source message context "PRIVATE_FIRST' in md


def test_legacy_manual_block_has_no_fabricated_location():
    block = DuplicateBlock(2, [('a', 0), ('b', 0)], 'context')
    assert block.token_spans == []


def test_locations_within_message_do_not_change_match_counts():
    seq = {'a': [99]*9+list(range(20)), 'b': [98]*4+list(range(20))}
    reqs = [Request(k, 'm', 'generic', [Message('user', k)]) for k in seq]
    block, = find_duplicates(reqs, seq.__getitem__, min_tokens=15, shingle=5)
    assert block.token_count == 20 and block.repeated_tokens == 20
    assert block.token_spans == [dict(request_index=0,message_index=0,token_start=9,token_end=29),
                                dict(request_index=1,message_index=0,token_start=4,token_end=24)]
