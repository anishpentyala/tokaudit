from dataclasses import replace

import pytest

from tokaudit.logs import Message, Request
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.tokenizers import EstimatorBackend
from tokaudit.whatif import run_scenarios

ENC = EstimatorBackend().encode
RULES = ProviderRules(name='anthropic', min_prefix_tokens=8, ttl_seconds=300,
                     write_multiplier=1.25, read_multiplier=0.1,
                     refresh_on_read=True, enabled_by_default=False)


def req(rid, markers, ts=0):
    return Request(rid, 'claude-test', 'anthropic',
                   [Message('system', 'prefix ' * 50),
                    Message('user', 'question ' * 40)],
                   breakpoints=markers, ts=ts)


@pytest.mark.parametrize('left,right', [([], [0]), ([0], [1]), ([0, 1], [1]),
                                      ([1], []), ([0, 1], [0])])
def test_distinct_breakpoint_layouts_not_collapsed(left, right):
    rows = [req('a', left), req('b', right, 10)]
    expected = simulate(rows, ENC, RULES)
    scenario, = run_scenarios(['dedup'], rows, ENC, RULES)
    assert scenario['result']['baseline_units'] == round(expected.baseline_units, 1)
    assert scenario['result']['simulated_units'] == round(expected.simulated_units, 1)
    assert any('No exact duplicate' in n for n in scenario['notes'])
    assert [r.breakpoints for r in rows] == [left, right]


@pytest.mark.parametrize('markers', [[], [0], [1], [0, 1]])
def test_same_breakpoints_still_collapse(markers):
    a = req('a', markers)
    b = replace(a, rid='b', ts=10, breakpoints=list(markers))
    scenario, = run_scenarios(['dedup'], [a, b], ENC, RULES)
    expected = simulate([a], ENC, RULES)
    assert scenario['result']['baseline_units'] == round(expected.baseline_units, 1)
    assert any('1 exact duplicate' in n for n in scenario['notes'])


@pytest.mark.parametrize('fmt', ['json', 'md'])
def test_cli_different_block_markers_preserve_both_requests(tmp_path, capsys, fmt):
    import json
    from tokaudit.cli import main
    from tokaudit.logs import load_log
    from tokaudit.rules import load_rules

    system = 'shared ' * 1000
    question = 'question ' * 1000
    rows = [
        {'id': 'a', 'model': 'claude-fable-5', 'provider': 'anthropic',
         'ts': 0, 'system': [{'type': 'text', 'text': system,
                            'cache_control': {'type': 'ephemeral'}}],
         'messages': [{'role': 'user', 'content': question}]},
        {'id': 'b', 'model': 'claude-fable-5', 'provider': 'anthropic',
         'ts': 10, 'system': system,
         'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': question,
                                                 'cache_control': {'type': 'ephemeral'}}]}]},
    ]
    path = tmp_path / 'traffic.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    parsed = load_log(str(path))
    assert parsed[0].canonical() == parsed[1].canonical()
    assert parsed[0].breakpoints != parsed[1].breakpoints
    rules = load_rules().for_provider('anthropic')
    expected = simulate(parsed, ENC, rules)
    assert len(expected.per_request) == 2
    assert expected.per_request[0].simulated_units != expected.per_request[1].simulated_units
    assert main(['analyze', str(path), '--tokenizer', 'estimate', '--format', fmt,
                 '--what-if', 'dedup']) == 0
    out = capsys.readouterr().out
    if fmt == 'json':
        report = json.loads(out)
        scenario, = report['what_if']
        assert scenario['result']['baseline_units'] == round(expected.baseline_units, 1)
        assert scenario['result']['simulated_units'] == round(expected.simulated_units, 1)
        assert report['summary']['requests'] == 2
    else:
        assert 'No exact duplicate requests found' in out


def test_caveat_names_breakpoint_identity():
    scenario, = run_scenarios(['dedup'], [req('a', [0])], ENC, RULES)
    assert 'breakpoint' in ' '.join(scenario['caveats'])
