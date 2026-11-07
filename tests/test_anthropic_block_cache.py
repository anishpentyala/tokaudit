"""Synthetic text-only protocol controls, not measured provider billing."""
from dataclasses import replace

import pytest

from tokaudit.logs import parse_record
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.whatif import run_scenarios

RULES = ProviderRules(name='anthropic', min_prefix_tokens=1, ttl_seconds=300,
                      write_multiplier=1.25, read_multiplier=.1,
                      refresh_on_read=True, enabled_by_default=False)


def encode(text):
    # Exact character tokens make boundary arithmetic transparent, including
    # Unicode. Provider tokens still use the selected tokenizer in real runs.
    return list(map(ord, text))


def block(text, marked=False):
    value = {'type': 'text', 'text': text}
    if marked:
        value['cache_control'] = {'type': 'ephemeral'}
    return value


def request(rid, system, messages=(), ts=0):
    return parse_record({'id': rid, 'model': 'claude-test', 'ts': ts,
                         'system': system, 'messages': list(messages)})


def user(text, marked=False):
    return {'role': 'user', 'content': [block(text, marked)]}


def run(rows):
    return simulate(rows, encode, RULES)


def test_changed_designated_block_does_not_read_unwritten_stable_prefix():
    rows = [request(str(i), 'stable instructions', [user(tail, True)], i)
            for i, tail in enumerate(('unique A', 'unique B'))]
    result = run(rows)
    assert result.per_request[1].read == 0
    assert result.per_request[1].written == len(rows[1].canonical())
    assert result.wasted_write_tokens == sum(row.written for row in result.per_request)


@pytest.mark.parametrize('place', ['system', 'user'])
@pytest.mark.parametrize('suffix', ['volatile', 'different suffix'])
def test_write_stops_at_marked_text_block(place, suffix):
    content = [block('static Ω', True), block(suffix)]
    row = (request('a', content) if place == 'system'
           else request('a', 'instructions', [{'role': 'user', 'content': content}]))
    cost, = run([row]).per_request
    prefix = ('<system>\nstatic Ω' if place == 'system'
              else '<system>\ninstructions\n<user>\nstatic Ω')
    assert cost.written == len(prefix)
    assert cost.uncached == len(suffix)
    assert cost.read == 0
    assert cost.total_input == len(row.canonical())


@pytest.mark.parametrize('place', ['system', 'user'])
def test_prior_marked_block_reused_with_changed_suffix(place):
    rows = []
    for i, tail in enumerate(('first', 'second')):
        content = [block('shared Ω', True), block(tail)]
        rows.append(request(str(i), content, ts=i) if place == 'system'
                    else request(str(i), 'instructions', [{'role': 'user', 'content': content}], i))
    first, second = run(rows).per_request
    assert second.read == first.written
    assert second.read > 0
    assert second.written == 0
    assert second.uncached == len('second')


@pytest.mark.parametrize('extra,hit', [(19, True), (20, False), (21, False)])
@pytest.mark.parametrize('within_message', [False, True])
def test_lookback_counts_text_blocks_including_designated_block(extra, hit, within_message):
    first = request('a', [block('stable', True)])
    if within_message:
        content = [block('stable')] + [block(str(i), i == extra - 1) for i in range(extra)]
        second = request('b', content, ts=1)
    else:
        second = request('b', 'stable', [user(str(i), i == extra - 1) for i in range(extra)], 1)
    cost = run([first, second]).per_request[1]
    assert bool(cost.read) is hit
    if hit:
        assert cost.read == len('<system>\nstable')


def test_additional_breakpoint_opens_another_lookback_window():
    first = request('a', [block('stable', True)])
    second = request('b', [block('stable', True)],
                     [user(str(i), i == 24) for i in range(25)], 1)
    cost = run([first, second]).per_request[1]
    assert cost.read == len('<system>\nstable')
    assert cost.written == cost.total_input - cost.read


def test_only_marked_internal_prefix_can_become_a_later_hit():
    first = request('a', [block('one'), block('two', True), block('three')])
    second = request('b', [block('one', True), block('other')], ts=1)
    assert run([first, second]).per_request[1].read == 0


def test_expired_marked_block_is_rewritten_not_read():
    first = request('a', [block('stable', True), block('tail')])
    second = replace(first, rid='b', ts=301)
    cost = run([first, second]).per_request[1]
    assert cost.read == 0
    assert cost.written == len('<system>\nstable')
    assert cost.uncached == len('tail')


def test_string_and_single_text_block_share_boundary():
    first = request('a', 'stable', ts=0)
    second = request('b', [block('stable', True)], ts=1)
    assert run([first, second]).per_request[1].read == len(first.canonical())


def test_distinct_block_marker_positions_are_not_deduped():
    first = request('a', [block('one', True), block('two')])
    second = request('b', [block('one'), block('two', True)], ts=1)
    assert first.canonical() == second.canonical()
    scenario, = run_scenarios(['dedup'], [first, second], encode, RULES)
    assert any('No exact duplicate' in note for note in scenario['notes'])


def test_identical_block_markers_still_dedup():
    first = request('a', [block('one', True), block('two')])
    scenario, = run_scenarios(['dedup'], [first, replace(first, rid='b', ts=1)], encode, RULES)
    assert any('1 exact duplicate' in note for note in scenario['notes'])


def test_openai_keeps_existing_intermediate_message_approximation():
    rows = [request(str(i), 'stable instructions', [user(tail, True)], i)
            for i, tail in enumerate(('unique A', 'unique B'))]
    rules = replace(RULES, name='openai')
    assert simulate(rows, encode, rules).per_request[1].read == len('<system>\nstable instructions')


@pytest.mark.parametrize('wrapper', ['plain', 'litellm', 'helicone', 'langfuse'])
@pytest.mark.parametrize('fmt', ['json', 'md'])
def test_cli_preserves_internal_markers_through_export_adapters(tmp_path, capsys, wrapper, fmt):
    import json
    from tokaudit.cli import main
    from tokaudit.logs import load_log
    from tokaudit.rules import load_rules
    from tokaudit.tokenizers import EstimatorBackend

    rows = []
    for i, tail in enumerate(('first suffix', 'second suffix')):
        # Above the bundled model's minimum; no provider rates are guessed.
        messages = [{'role': 'user', 'content': [block('stable ' * 4000, True), block(tail)]}]
        body = {'id': str(i), 'model': 'claude-fable-5', 'ts': i, 'messages': messages}
        if wrapper == 'litellm':
            body = {'request_id': str(i), 'startTime': i, 'model': body['model'],
                    'messages': messages, 'custom_llm_provider': 'anthropic'}
        elif wrapper == 'helicone':
            body = {'request_id': str(i), 'request_created_at': i,
                    'request_model': body['model'], 'provider': 'anthropic',
                    'request_body': json.dumps({'messages': messages})}
        elif wrapper == 'langfuse':
            body = {'id': str(i), 'startTime': i, 'type': 'GENERATION',
                    'model': body['model'], 'input': json.dumps({'messages': messages})}
        rows.append(body)
    path = tmp_path / 'traffic.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    parsed = load_log(str(path))
    enc = EstimatorBackend().encode
    expected_read = len(enc('<user>\n' + 'stable ' * 4000))
    expected = simulate(parsed, enc, load_rules().for_provider('anthropic'))
    assert expected.per_request[0].written == expected_read
    assert expected.per_request[1].read == expected_read
    assert expected.per_request[1].written == 0
    out = tmp_path / ('report.' + fmt)
    assert main(['analyze', str(path), '--tokenizer', 'estimate', '--format', fmt,
                 '--dup-max-shingles', '0', '--output', str(out)]) == 0
    assert 'wrote ' in capsys.readouterr().out
    text = out.read_text()
    assert main(['analyze', str(path), '--tokenizer', 'estimate', '--format', fmt,
                 '--dup-max-shingles', '0']) == 0
    assert capsys.readouterr().out.strip() == text.strip()
    assert '20 block positions' in text
    if fmt == 'json':
        data = json.loads(text)
        assert data['cache_simulation'][0]['cache_write'] == expected_read
        assert data['cache_simulation'][1]['cache_read'] == expected_read
        assert data['cache_simulation'][1]['cache_write'] == 0
        assert json.loads(out.read_text()) == data
    else:
        assert 'SIMULATED' in text
        assert out.read_text().strip() == text.strip()


def test_nonmonotonic_canonical_boundary_stops_instead_of_clipping():
    row = request('a', [block('prefix', True), block('suffix')])
    def nonmonotonic(text):
        return [1, 2] if text.endswith('prefix') else [1]
    with pytest.raises(SystemExit, match='canonical token counts decrease'):
        simulate([row], nonmonotonic, RULES)


def test_exact_tokenizer_boundary_merge_is_not_valid_cache_accounting():
    import tiktoken
    enc = tiktoken.get_encoding('o200k_base').encode
    row = request('a', [block('establis', True), block('hment')])
    assert len(enc('<system>\nestablis')) > len(enc(row.canonical()))
    with pytest.raises(SystemExit, match='canonical token counts decrease'):
        simulate([row], enc, RULES)


def test_nonmonotonic_below_minimum_is_uncached_not_a_false_charge():
    row = request('a', [block('prefix', True), block('suffix')])
    def nonmonotonic(text):
        return [1, 2] if text.endswith('prefix') else [1]
    cost, = simulate([row], nonmonotonic, replace(RULES, min_prefix_tokens=10)).per_request
    assert cost.read == cost.written == 0
    assert cost.uncached == cost.total_input == 1


def test_cli_boundary_merge_preserves_existing_output(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tokaudit import cli
    path, output = tmp_path / 'traffic.jsonl', tmp_path / 'report.json'
    body = {'model': 'claude-fable-5', 'system': [block('prefix', True), block('suffix')],
            'messages': []}
    path.write_text(json.dumps(body))
    output.write_text('KEEP')
    def nonmonotonic(text):
        return [1] * (10000 if text.endswith('prefix') else 5000)
    monkeypatch.setattr(cli, 'get_backend', lambda _: SimpleNamespace(name='synthetic', encode=nonmonotonic))
    with pytest.raises(SystemExit, match='canonical token counts decrease'):
        cli.main(['analyze', str(path), '--format', 'json', '--output', str(output)])
    assert output.read_text() == 'KEEP'
