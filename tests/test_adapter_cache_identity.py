"""Structured exported request bodies must retain represented cache identity."""
import json

import pytest

from tokaudit.adapters import adapt
from tokaudit.logs import parse_record, load_log
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.whatif import run_scenarios

RULES = ProviderRules('openai', 8, 1800, 1.25, 0.1, False, True)
ENC = lambda text: list(text.encode('utf-8'))


def exported(kind, body, rid='a', ts=0, serialized=False):
    body = json.dumps(body) if serialized else body
    if kind == 'helicone':
        return {'request_id': rid, 'request_created_at': ts,
                'request_model': 'gpt-test', 'provider': 'openai',
                'request_body': body}
    return {'id': rid, 'startTime': ts, 'traceId': 't', 'type': 'GENERATION',
            'model': 'gpt-test', 'input': body}


def body(key='tenant-a', field='prompt_cache_key'):
    return {'messages': [{'role': 'user', 'content': 'shared ' * 100}], field: key}


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
@pytest.mark.parametrize('serialized', [False, True])
@pytest.mark.parametrize('field', ['prompt_cache_key', 'cache_key'])
def test_normalization_retains_cache_identity_without_mutation(kind, serialized, field):
    raw = exported(kind, body(field=field), serialized=serialized)
    before = json.dumps(raw, sort_keys=True)
    normalized = adapt(raw)
    assert normalized[field] == 'tenant-a'
    assert parse_record(normalized).cache_key == 'tenant-a'
    assert json.dumps(raw, sort_keys=True) == before


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
@pytest.mark.parametrize('serialized', [False, True])
def test_distinct_exported_keys_do_not_create_false_reads_or_dedup(kind, serialized):
    rows = [parse_record(adapt(exported(kind, body(key), str(i), i,
                                       serialized=serialized)))
            for i, key in enumerate(['tenant-a', 'tenant-b', 'tenant-a'])]
    result = simulate(rows, ENC, RULES)
    assert [cost.read > 0 for cost in result.per_request] == [False, False, True]
    scenario, = run_scenarios(['dedup'], rows, ENC, RULES)
    expected = simulate(rows[:2], ENC, RULES)
    assert scenario['result']['baseline_units'] == round(expected.baseline_units, 1)
    assert any('1 exact duplicate' in note and '(2 of 3 kept)' in note
               for note in scenario['notes'])


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
def test_native_key_precedence_matches_plain_request(kind):
    request = body(field='cache_key')
    request['prompt_cache_key'] = 'native'
    assert parse_record(adapt(exported(kind, request))).cache_key == 'native'
    request['prompt_cache_key'] = ''
    assert parse_record(adapt(exported(kind, request))).cache_key == 'tenant-a'


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
def test_missing_key_retains_existing_shared_namespace(kind):
    request = body()
    del request['prompt_cache_key']
    rows = [parse_record(adapt(exported(kind, request, str(i), i))) for i in range(2)]
    assert all(row.cache_key is None for row in rows)
    assert simulate(rows, ENC, RULES).per_request[1].read > 0


@pytest.mark.parametrize('serialized', [False, True])
def test_helicone_prompt_body_retains_key(serialized):
    assert parse_record(adapt(exported('helicone', {'prompt': 'text',
           'prompt_cache_key': 'tenant'}, serialized=serialized))).cache_key == 'tenant'


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
def test_jsonl_adaptation_keeps_represented_distinctions(tmp_path, kind):
    path = tmp_path / 'traffic.jsonl'
    path.write_text('\n'.join(json.dumps(exported(kind, body(key), str(i), i,
                                                serialized=True))
                              for i, key in enumerate(['a', 'b'])), encoding='utf-8')
    rows = load_log(str(path))
    assert [row.cache_key for row in rows] == ['a', 'b']
    assert simulate(rows, ENC, RULES).per_request[1].read == 0


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
def test_export_metadata_does_not_invent_missing_body_key(kind):
    request = body()
    del request['prompt_cache_key']
    raw = exported(kind, request)
    raw['prompt_cache_key'] = 'export-metadata-not-a-request-key'
    assert parse_record(adapt(raw)).cache_key is None


@pytest.mark.parametrize('kind', ['helicone', 'langfuse'])
@pytest.mark.parametrize('tokenizer', ['estimate', 'exact'])
def test_cli_export_matches_native_report_scope(tmp_path, capsys, kind, tokenizer):
    from tokaudit.cli import main
    native, exports = [], []
    for i, key in enumerate(['tenant-a', 'tenant-b', 'tenant-a']):
        request = body(key)
        request['messages'][0]['content'] = 'shared policy ' * 1500
        native.append({'id': str(i), 'model': 'gpt-6-astra', 'ts': i, **request})
        row = exported(kind, request, str(i), i, serialized=True)
        row['request_model' if kind == 'helicone' else 'model'] = 'gpt-6-astra'
        exports.append(row)
    reports = []
    for name, rows in [('native', native), ('export', exports)]:
        path = tmp_path / (name + '.jsonl')
        path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
        assert main(['analyze', str(path), '--tokenizer', tokenizer, '--format', 'json',
                     '--what-if', 'dedup']) == 0
        reports.append(json.loads(capsys.readouterr().out))
    assert reports[0]['summary'] == reports[1]['summary']
    assert reports[0]['what_if'] == reports[1]['what_if']
    assert any('(2 of 3 kept)' in note for note in reports[1]['what_if'][0]['notes'])
