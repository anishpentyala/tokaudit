"""Policy input must not silently change cache behavior or cost arithmetic."""
import json
from pathlib import Path

import pytest

from tokaudit.cli import main
from tokaudit.rules import load_rules


def write_rules(tmp_path, provider):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'providers': {'openai': provider}}), encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('field', ['min_prefix_tokens', 'ttl_seconds',
                                  'max_explicit_breakpoints', 'read_refresh_seconds'])
@pytest.mark.parametrize('value', [-1, True, False, 1.5, '1.5', None, [], {}, float('inf'), 10**400])
def test_invalid_provider_counts_rejected(tmp_path, field, value):
    if field == 'read_refresh_seconds' and value is None:
        assert load_rules(write_rules(tmp_path, {field: value})).for_provider('openai').read_refresh_seconds is None
        return  # Explicit null preserves optional read-refresh behavior.
    with pytest.raises(SystemExit, match=field):
        load_rules(write_rules(tmp_path, {field: value}))


@pytest.mark.parametrize('field', ['min_prefix_tokens', 'ttl_seconds'])
@pytest.mark.parametrize('value', [True, -1, .5, [], float('nan')])
def test_invalid_family_counts_rejected(tmp_path, field, value):
    with pytest.raises(SystemExit, match=field):
        load_rules(write_rules(tmp_path, {'families': {'custom': {field: value}}}))


@pytest.mark.parametrize('field', ['write_multiplier', 'read_multiplier'])
@pytest.mark.parametrize('value', [True, False, -1, 'NaN', float('inf'), '-Infinity', 10**400, [], None])
@pytest.mark.parametrize('family', [False, True])
def test_invalid_cost_multipliers_rejected(tmp_path, field, value, family):
    provider = {'families': {'custom': {field: value}}} if family else {field: value}
    with pytest.raises(SystemExit, match=field):
        load_rules(write_rules(tmp_path, provider))


@pytest.mark.parametrize('field', ['refresh_on_read', 'enabled_by_default'])
@pytest.mark.parametrize('value', ['false', 0, 1, None, [], {}])
def test_flags_are_not_truthiness_coerced(tmp_path, field, value):
    with pytest.raises(SystemExit, match=field):
        load_rules(write_rules(tmp_path, {field: value}))


@pytest.mark.parametrize('value', ['false', 1, None])
def test_family_support_is_boolean(tmp_path, value):
    with pytest.raises(SystemExit, match='savings_supported'):
        load_rules(write_rules(tmp_path, {'families': {'custom': {'savings_supported': value}}}))


@pytest.mark.parametrize('provider,field', [
    ({'families': []}, 'families'), ({'families': {'x': []}}, 'family'),
    ({'model_minimums': []}, 'model_minimums'),
    ({'model_minimums': {'x': True}}, 'model_minimums'),
    ({'model_minimums': {'x': -1}}, 'model_minimums'),
    ({'families': {'x': {'match': 'gpt-custom'}}}, 'match'),
    ({'families': {'x': {'match': [42]}}}, 'match'),
    ({'notes': 'note'}, 'notes'), ({'notes': [42]}, 'notes'),
    ({'families': {'x': {'notes': {}}}}, 'notes'),
])
def test_malformed_containers_exit_cleanly(tmp_path, provider, field):
    with pytest.raises(SystemExit, match=field):
        load_rules(write_rules(tmp_path, provider))


@pytest.mark.parametrize('text', [
    '{"providers":{"openai":{},"openai":{}}}',
    '{"providers":{"openai":{"ttl_seconds":30,"ttl_seconds":300}}}',
    '{"providers":{"openai":{"read_multiplier":0.1,"read_multiplie\\u0072":0.9}}}',
    '{"providers":{"openai":{"families":{"x":{"match":[],"match":[]}}}}}',
    '{"providers":{"openai":{}},"private-marker":1,"private-marker":2}',
])
def test_duplicate_policy_members_not_silently_overwritten(tmp_path, text):
    path = tmp_path / 'policy.json'; path.write_text(text)
    with pytest.raises(SystemExit, match='duplicate JSON object member') as caught:
        load_rules(str(path))
    assert 'private-marker' not in str(caught.value)


def test_valid_custom_policy_preserves_defaults_and_supported_numeric_forms(tmp_path):
    provider = {'min_prefix_tokens': '32', 'ttl_seconds': 60.0,
                'read_refresh_seconds': None, 'max_explicit_breakpoints': 0,
                'write_multiplier': '0', 'read_multiplier': 0.1,
                'refresh_on_read': False, 'enabled_by_default': True,
                'model_minimums': {'CUSTOM': '16'},
                'families': {'custom': {'match': ['CUSTOM'], 'notes': ['ok'],
                                        'savings_supported': False}}}
    rules = load_rules(write_rules(tmp_path, provider)).for_provider('openai')
    assert (rules.min_prefix_tokens, rules.ttl_seconds, rules.max_explicit_breakpoints) == (32, 60, 0)
    assert rules.resolve_min_prefix('custom') == 16
    assert rules.read_refresh_seconds is None
    family = rules.families['custom']
    assert (family.ttl_seconds, family.write_multiplier, family.read_multiplier) == (60, 0., .1)
    assert family.match == ['custom'] and not family.savings_supported


@pytest.mark.parametrize('provider', [{'read_multiplier': 'NaN'}, {'refresh_on_read': 'false'},
                                     {'families': []}, {'ttl_seconds': 1.5}])
def test_bad_policy_preserves_cli_output(tmp_path, capsys, provider):
    policy = write_rules(tmp_path, provider)
    output = tmp_path / 'report.json'; output.write_text('preserve')
    log = str(Path(__file__).resolve().parents[1] / 'examples' / 'quickstart.jsonl')
    with pytest.raises(SystemExit):
        main(['analyze', log, '--rules', policy, '--tokenizer', 'estimate',
              '--format', 'json', '--output', str(output)])
    assert output.read_text() == 'preserve'
    captured = capsys.readouterr()
    assert not captured.out and 'Traceback' not in captured.err


def test_loaded_valid_policy_matches_direct_simulation(tmp_path):
    from tokaudit.logs import Message, Request
    from tokaudit.prefix import simulate
    from tokaudit.rules import ProviderRules
    from tokaudit.tokenizers import EstimatorBackend
    fields = dict(min_prefix_tokens=8, ttl_seconds=30, write_multiplier=1.25,
                  read_multiplier=.1, refresh_on_read=True, enabled_by_default=True,
                  read_refresh_seconds=10)
    loaded = load_rules(write_rules(tmp_path, fields)).for_provider('openai')
    direct = ProviderRules(name='openai', **fields)
    requests = [Request(rid=str(i), model='custom', provider='openai',
                        messages=[Message('user', 'stable prefix ' * 30)], ts=ts)
                for i, ts in enumerate([0., 20., 35., 50.])]
    encode = EstimatorBackend().encode
    actual, expected = simulate(requests, encode, loaded), simulate(requests, encode, direct)
    assert actual == expected
    assert actual.per_request[0].written > 0
    assert actual.per_request[1].read > 0
    assert actual.per_request[2].read == 0


@pytest.mark.parametrize('value', [[], None, {'x': 1}])
def test_sources_container_error_is_clean(tmp_path, value):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'sources': value, 'providers': {'openai': {}}}))
    with pytest.raises(SystemExit, match='sources'):
        load_rules(str(path))


def test_defaults_and_separate_object_members_are_valid(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text('{"providers":{"openai":{"ttl_seconds":30},"anthropic":{"ttl_seconds":60}}}')
    rules = load_rules(str(path))
    assert rules.for_provider('openai').ttl_seconds == 30
    assert rules.for_provider('anthropic').ttl_seconds == 60
    assert rules.for_provider('openai').read_multiplier == .1
