"""Missing timestamps preserve log order, not invented cache chronology."""
from copy import deepcopy
import json

import pytest

from tokaudit.logs import parse_record
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.whatif import run_scenarios

RULES = ProviderRules('anthropic', 1, 300, 1.25, .1, True, False)


def encode(text):
    return list(map(ord, text))


def record(rid, timestamp, texts):
    value = {'id': rid, 'model': 'custom', 'provider': 'anthropic',
             'messages': [{'role': 'user', 'content': [
                 {'type': 'text', 'text': text,
                  **({'cache_control': {'type': 'ephemeral'}} if i == len(texts) - 1 else {})}
                 for i, text in enumerate(texts)]}]}
    if timestamp != 'absent':
        value['ts'] = timestamp
    return value


@pytest.mark.parametrize('missing', ['absent', None, True, 'invalid', float('nan'), float('inf')])
@pytest.mark.parametrize('timed', [-10, 0, 10, 2**53 + 1])
def test_missing_time_does_not_move_later_write_before_earlier_request(missing, timed):
    rows = [parse_record(record('long-first', missing, ['alpha', 'beta'])),
            parse_record(record('short-second', timed, ['alpha']))]
    before = deepcopy(rows)
    assert rows[0].ts is None
    result = simulate(rows, encode, RULES)
    assert [cost.rid for cost in result.per_request] == ['long-first', 'short-second']
    assert [cost.read for cost in result.per_request] == [0, 0]
    assert [cost.written for cost in result.per_request] == [16, 12]
    # Only the final marked block is written. A future shorter write cannot
    # supply an earlier read. 16*1.25 + 12*1.25 = 35, not the sorted 21.2.
    assert result.simulated_units == 35
    assert result.wasted_write_tokens == 28
    assert any('log order' in note for note in result.notes)
    assert rows == before


@pytest.mark.parametrize('missing_index', [0, 1, 2])
def test_any_missing_time_preserves_all_log_positions_and_disables_expiry(missing_index):
    times = [900, 1, 600]
    times[missing_index] = None
    rows = [parse_record(record(str(i), timestamp, ['same']))
            for i, timestamp in enumerate(times)]
    result = simulate(rows, encode, RULES)
    assert [cost.rid for cost in result.per_request] == ['0', '1', '2']
    assert [cost.read for cost in result.per_request] == [0, 11, 11]


def test_all_valid_times_still_sort_and_apply_expiry_without_mutation():
    rows = [parse_record(record('late', 301, ['alpha'])),
            parse_record(record('early', 0, ['alpha']))]
    before = deepcopy(rows)
    result = simulate(rows, encode, RULES)
    assert [cost.rid for cost in result.per_request] == ['early', 'late']
    assert [cost.read for cost in result.per_request] == [0, 0]
    assert rows == before


def test_all_valid_times_still_sort_before_prefix_reuse():
    rows = [parse_record(record('long-later', 10, ['alpha', 'beta'])),
            parse_record(record('short-earlier', 0, ['alpha']))]
    result = simulate(rows, encode, RULES)
    assert [cost.rid for cost in result.per_request] == ['short-earlier', 'long-later']
    assert [cost.read for cost in result.per_request] == [0, 12]
    assert result.simulated_units == 21.2


def test_equal_usable_times_keep_stable_log_order():
    rows = [parse_record(record('long-first', 10, ['alpha', 'beta'])),
            parse_record(record('short-second', 10, ['alpha']))]
    assert simulate(rows, encode, RULES).simulated_units == 35


def test_all_missing_times_keep_existing_log_order():
    rows = [parse_record(record('long-first', None, ['alpha', 'beta'])),
            parse_record(record('short-second', None, ['alpha']))]
    result = simulate(rows, encode, RULES)
    assert [cost.rid for cost in result.per_request] == ['long-first', 'short-second']
    assert result.simulated_units == 35


def test_dense_missing_time_noop_uses_same_log_order_contract():
    rows = [parse_record(record('long-first', None, ['alpha', 'beta'])),
            parse_record(record('short-second', 10, ['alpha']))]
    scenario, = run_scenarios(['dense'], rows, encode, RULES)
    assert scenario['result']['simulated_units'] == 35
    assert 'identical' in ' '.join(scenario['notes'])


@pytest.mark.parametrize('mode', ['json', 'md'])
def test_real_cli_reports_mixed_time_cost_in_log_order(tmp_path, capsys, mode):
    from tokaudit.cli import main
    from tokaudit.tokenizers import EstimatorBackend
    rows = [record('long-first', None, ['alpha ' * 20, 'beta ' * 20]),
            record('short-second', 10, ['alpha ' * 20])]
    path = tmp_path / 'requests.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'as_of': '2026-11-09', 'providers': {'anthropic': {
        'min_prefix_tokens': 1, 'ttl_seconds': 300, 'write_multiplier': 1.25,
        'read_multiplier': .1, 'refresh_on_read': True, 'enabled_by_default': False}}}))
    # Independent expected result: both designated prefixes are first writes.
    parsed = [parse_record(row) for row in rows]
    expected_tokens = sum(len(EstimatorBackend().encode(row.canonical())) for row in parsed)
    args = ['analyze', str(path), '--rules', str(rules), '--tokenizer', 'estimate', '--format', mode]
    assert main(args) == 0
    output = capsys.readouterr().out
    if mode == 'json':
        report = json.loads(output)
        assert [row['id'] for row in report['cache_simulation']] == ['long-first', 'short-second']
        assert [row['cache_read'] for row in report['cache_simulation']] == [0, 0]
        assert report['summary']['simulated_units'] == round(expected_tokens * 1.25, 1)
        assert report['summary']['wasted_write_tokens'] == expected_tokens
    else:
        assert 'log order' in output
        assert f'With caching: {round(expected_tokens * 1.25, 1):,.0f} units' in output
        assert f'Cache writes never re-read: {expected_tokens:,} tokens' in output
