"""Conflicting raw JSON members are not usable request/accounting evidence."""
import json

import pytest

from tokaudit.cli import main
from tokaudit.integrity import build_integrity_report
from tokaudit.logs import load_log


@pytest.mark.parametrize('record', [
    '{"model":"gpt-5.6","prompt":"first","prompt":"second"}',
    '{"model":"gpt-5.6","messages":[{"role":"user","content":"first","content":"second"}]}',
    '{"model":"gpt-5.6","prompt":"safe","private-marker":1,"private-marker":2}',
    '{"model":"gpt-5.6","prompt":"same","prompt":"same"}',
    '{"model":"gpt-5.6","prompt":"first","pro\\u006dpt":"second"}',
])
def test_analysis_rejects_ambiguous_members(tmp_path, record):
    path = tmp_path / 'log.jsonl'
    path.write_text('\n' + record + '\n')
    with pytest.raises(SystemExit, match=r'log.jsonl:2:.*duplicate JSON object member') as caught:
        load_log(str(path))
    assert 'private-marker' not in str(caught.value)


@pytest.mark.parametrize('record', [
    '{"request_id":"r","startTime":"t","spend":0,"spend":1,"prompt_tokens":10}',
    '{"id":"r","usageDetails":{"input":10,"input":20}}',
    '{"request_id":"r","startTime":"t","spend":1,"request_id":"s"}',
])
def test_integrity_cannot_report_clean_ambiguous_accounting(tmp_path, record):
    path = tmp_path / 'accounting.jsonl'
    path.write_text(record)
    with pytest.raises(SystemExit, match='duplicate JSON object member'):
        build_integrity_report(str(path))


@pytest.mark.parametrize('shape', ['langfuse', 'helicone'])
def test_embedded_export_json_rejects_ambiguous_members(tmp_path, shape):
    body = '{"model":"gpt-5.6","prompt":"first","prompt":"second"}'
    if shape == 'langfuse':
        row = {'id': 'r', 'type': 'GENERATION', 'startTime': '2026-01-01T00:00:00Z', 'usageDetails': {}, 'input': body}
    else:
        row = {'request_id': 'r', 'request_body': body}
    path = tmp_path / 'export.jsonl'
    path.write_text(json.dumps(row))
    with pytest.raises(SystemExit, match=r'export.jsonl:1:.*duplicate JSON object member'):
        load_log(str(path))


@pytest.mark.parametrize('command,record', [
    ('analyze', '{"prompt":"first","prompt":"second"}'),
    ('integrity', '{"id":"r","usageDetails":{"input":10,"input":20}}'),
])
def test_cli_error_preserves_existing_output(tmp_path, capsys, command, record):
    path = tmp_path / 'log.jsonl'; path.write_text(record)
    output = tmp_path / 'report.json'; output.write_text('preserve')
    with pytest.raises(SystemExit, match='duplicate JSON object member'):
        main([command, str(path), '--format', 'json', '--output', str(output)])
    assert output.read_text() == 'preserve'
    captured = capsys.readouterr()
    assert not captured.out
    assert 'Traceback' not in captured.err


def test_equal_members_in_different_objects_are_not_duplicates(tmp_path):
    path = tmp_path / 'log.jsonl'
    path.write_text(json.dumps({'model': 'gpt-5.6', 'messages': [
        {'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]}))
    assert len(load_log(str(path))[0].messages) == 2


def test_duplicate_looking_plain_prompt_is_still_plain_text(tmp_path):
    row = {'id': 'r', 'type': 'GENERATION', 'startTime': '2026-01-01T00:00:00Z', 'usageDetails': {},
           'input': 'Explain "x": 1, "x": 2, please'}
    path = tmp_path / 'log.jsonl'; path.write_text(json.dumps(row))
    assert load_log(str(path))[0].messages[0].text == row['input']
