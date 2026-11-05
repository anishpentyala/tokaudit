"""A bounded index must not masquerade as complete duplicate-search evidence."""
import copy
import json

import pytest

from tokaudit import dedup
from tokaudit.logs import Message, Request


def rows():
    return [Request(str(i), 'custom', 'openai', [Message('user', 'same')]) for i in range(2)]


def analyze(*args, **kwargs):
    assert hasattr(dedup, 'analyze_duplicates'), 'structured duplicate-search coverage is missing'
    return dedup.analyze_duplicates(*args, **kwargs)


@pytest.mark.parametrize('cap,complete', [(0, False), (16, False), (31, False), (32, True), (33, True)])
def test_exact_index_budget_coverage(cap, complete):
    result = analyze(rows(), lambda _: list(range(20)), min_tokens=10, shingle=5, max_shingles=cap)
    assert result.indexed_shingles == min(cap, 32)
    assert result.eligible_shingles == 32 and result.max_shingles == cap
    assert result.index_complete is complete
    if cap <= 16:
        assert result.blocks == []  # Every second-request window was omitted.
    if cap >= 32:
        assert result.blocks[0].token_count == 20
    assert result.blocks == dedup.find_duplicates(rows(), lambda _: list(range(20)),
                                                 min_tokens=10, shingle=5, max_shingles=cap)


def test_short_and_empty_messages_have_zero_eligible_windows():
    calls = []
    def encode(text):
        calls.append(text)
        return [1] if text else []
    requests = [Request('a', 'custom', 'openai', [Message('user', ''), Message('user', 'short')])]
    result = analyze(requests, encode, shingle=5, max_shingles=0)
    assert calls == ['', 'short']
    assert result.index_complete and result.indexed_shingles == result.eligible_shingles == 0


def test_capped_search_still_counts_and_tokenizes_later_messages_without_mutation():
    requests = rows(); requests[1].messages.append(Message('user', 'extra'))
    original = copy.deepcopy(requests); calls = []
    def encode(text):
        calls.append(text)
        return list(range(20))
    result = analyze(requests, encode, shingle=5, max_shingles=1)
    assert calls == ['same', 'same', 'extra']
    assert result.indexed_shingles == 1 and result.eligible_shingles == 48
    assert requests == original


@pytest.mark.parametrize('cap', [0, 16, 32])
@pytest.mark.parametrize('fmt', ['json', 'md'])
def test_cli_reports_partial_or_complete_index(tmp_path, monkeypatch, capsys, cap, fmt):
    from tokaudit import cli
    # Production CLI default is500k. Small cap and deterministic token IDs
    # reproduce truncation cheaply while using real detection and reporting.
    assert hasattr(dedup, 'analyze_duplicates'), 'structured duplicate-search coverage is missing'
    # CLI supplies shingle by keyword, so use a wrapper to select a small window.
    def bounded(requests, encode, **kwargs):
        kwargs.update(shingle=5, max_shingles=cap)
        return dedup.analyze_duplicates(requests, encode, **kwargs)
    monkeypatch.setattr(cli, 'analyze_duplicates', bounded)
    monkeypatch.setattr(cli, 'get_backend', lambda _: type('Backend', (), {'name': 'estimate', 'encode': staticmethod(lambda _: list(range(20)))})())
    log = tmp_path / 'log.jsonl'
    log.write_text('\n'.join(json.dumps({'id':str(i),'model':'custom','prompt':'same'}) for i in range(2)))
    assert cli.main(['analyze', str(log), '--dup-min-tokens', '10', '--format', fmt]) == 0
    out = capsys.readouterr().out
    if fmt == 'json':
        report = json.loads(out); coverage = report['duplicate_search']
        assert coverage['index_complete'] == (cap >= 32)
        assert coverage['eligible_shingles'] == 32
        assert coverage['indexed_shingles'] == min(cap,32)
    else:
        assert 'Shingle index:' in out
        if cap < 32:
            assert 'CAPPED - partial search' in out
            assert 'unindexed windows may contain duplicates' in out
            assert 'No repeated blocks above the detection threshold.' not in out
        else:
            assert '(complete)' in out


def test_partial_index_can_still_return_real_blocks():
    result = analyze(rows(), lambda _: list(range(20)), min_tokens=10, shingle=5, max_shingles=17)
    assert not result.index_complete and result.blocks
    assert result.blocks[0].token_count == 20


def test_complete_index_does_not_claim_exhaustive_detection(tmp_path, capsys):
    from tokaudit.cli import main
    log = tmp_path / 'log.jsonl'; log.write_text('{"model":"custom","prompt":"tiny"}')
    assert main(['analyze', str(log), '--tokenizer', 'estimate']) == 0
    text = capsys.readouterr().out
    assert 'not exhaustive duplicate discovery' in text
    assert 'heuristic search' in text


@pytest.mark.parametrize('field', ['min_tokens', 'shingle', 'top', 'max_shingles'])
@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_bad_search_limits_are_not_silent_partial_scans(field, value):
    with pytest.raises(ValueError, match=field):
        dedup.find_duplicates(rows(), lambda _: list(range(20)), **{field:value})


@pytest.mark.parametrize('via_config', [False, True])
def test_cli_invalid_threshold_is_clean_and_preserves_output(tmp_path, capsys, via_config):
    from tokaudit.cli import main
    log = tmp_path / 'log.jsonl'; log.write_text('{"prompt":"tiny"}')
    output = tmp_path / 'report.json'; output.write_text('preserve')
    args = ['analyze', str(log), '--output', str(output)]
    if via_config:
        config = tmp_path / 'config.toml'; config.write_text('dup_min_tokens = 0')
        args += ['--config', str(config)]
    else:
        args += ['--dup-min-tokens', '0']
    with pytest.raises(SystemExit, match='positive integer'):
        main(args)
    assert output.read_text() == 'preserve' and not capsys.readouterr().out


def test_real_estimator_cli_wires_coverage_without_test_backends(tmp_path, capsys):
    from tokaudit.cli import main
    from tokaudit.tokenizers import EstimatorBackend
    prompt = 'A shared instruction paragraph. ' * 100
    log = tmp_path / 'log.jsonl'
    log.write_text('\n'.join(json.dumps({'id':str(i),'model':'custom','prompt':prompt}) for i in range(2)))
    assert main(['analyze', str(log), '--tokenizer', 'estimate', '--format', 'json']) == 0
    report = json.loads(capsys.readouterr().out)
    expected = 2 * (len(EstimatorBackend().encode(prompt)) - 42 + 1)
    assert report['duplicate_search']['eligible_shingles'] == expected
    assert report['duplicate_search']['index_complete']
    assert report['duplicate_blocks']


def test_capped_empty_markdown_cannot_claim_no_duplicates(tmp_path, monkeypatch, capsys):
    """Baseline-compatible semantic witness: cap omits identical later input."""
    from tokaudit import cli
    requests = rows()
    encode = lambda _: list(range(20))
    assert dedup.find_duplicates(requests, encode, min_tokens=10, shingle=5, max_shingles=16) == []
    assert dedup.find_duplicates(requests, encode, min_tokens=10, shingle=5, max_shingles=32)
    name = 'analyze_duplicates' if hasattr(cli, 'analyze_duplicates') else 'find_duplicates'
    detector = getattr(cli, name)
    def capped(requests, encode, **kwargs):
        kwargs.update(shingle=5, max_shingles=16)
        return detector(requests, encode, **kwargs)
    monkeypatch.setattr(cli, name, capped)
    monkeypatch.setattr(cli, 'get_backend', lambda _: type('Backend', (), {'name':'estimate', 'encode':staticmethod(encode)})())
    log = tmp_path / 'log.jsonl'
    log.write_text('\n'.join(json.dumps({'id':str(i),'model':'custom','prompt':'same'}) for i in range(2)))
    assert cli.main(['analyze', str(log), '--dup-min-tokens', '10']) == 0
    out = capsys.readouterr().out
    assert 'CAPPED - partial search' in out
    assert 'unindexed windows may contain duplicates' in out
