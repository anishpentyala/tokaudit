import json

import pytest

from tokaudit.logs import parse_record, load_log
from tokaudit.cli import main


@pytest.mark.parametrize('content', [
    [{'type': 'image_url', 'image_url': {'url': 'private-image'}}],
    [{'type': 'text', 'text': 'same'}, {'type': 'image', 'source': {'data': 'private'}}],
    [{'type': 'input_audio', 'input_audio': {'data': 'private'}}],
    [{'type': 'tool_use', 'name': 'search', 'input': {'q': 'private'}}],
    [{'type': 'tool_result', 'content': 'private'}],
    [{'type': 'thinking', 'thinking': 'private'}],
    [{'type': 'text', 'text': {'not': 'text'}}],
    [{'type': 'text'}], ['private'], 42, {'text': 'private'}, None,
])
def test_rejects_unrepresented_content_without_echoing_payload(content):
    with pytest.raises(ValueError, match='unsupported.*content|text.*string') as exc:
        parse_record({'id': 'x', 'messages': [{'role': 'user', 'content': content}]})
    assert 'private' not in str(exc.value)


@pytest.mark.parametrize('field,value', [('tools', [{'name': 'private'}]),
                                        ('functions', [{'name': 'private'}]),
                                        ('tool_choice', 'auto')])
def test_rejects_request_tool_configuration(field, value):
    with pytest.raises(ValueError, match='unsupported.*tool|unsupported.*function'):
        parse_record({'prompt': 'same', field: value})


@pytest.mark.parametrize('field', ['tool_calls', 'function_call'])
def test_rejects_message_tool_payload(field):
    with pytest.raises(ValueError, match='unsupported.*tool|unsupported.*function'):
        parse_record({'messages': [{'role': 'assistant', 'content': '', field: [{'id': 'private'}]}]})


@pytest.mark.parametrize('content', ['plain', '', [{'type': 'text', 'text': 'plain'}],
                                   [{'type': 'input_text', 'text': 'plain'}],
                                   [{'text': 'plain'}]])
def test_supported_text_stays_supported(content):
    assert parse_record({'messages': [{'role': 'user', 'content': content}]}).messages[0].text in ('', 'plain')


def test_mixed_log_fails_on_unsupported_row_before_report(tmp_path, capsys):
    path = tmp_path / 'traffic.jsonl'
    rows = [{'prompt': 'safe'}, {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'private'}}]}]}]
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    out = tmp_path / 'report.json'
    with pytest.raises(SystemExit, match=':2:'):
        main(['analyze', str(path), '--format', 'json', '--output', str(out)])
    assert not out.exists()
    assert not capsys.readouterr().out


@pytest.mark.parametrize('kind', ['litellm', 'langfuse', 'helicone'])
@pytest.mark.parametrize('field', ['tools', 'functions'])
def test_export_adapters_do_not_drop_unsupported_request_fields(kind, field):
    from tokaudit.adapters import adapt
    body = {'messages': [{'role': 'user', 'content': 'same'}],
            field: [{'name': 'private'}]}
    if kind == 'litellm':
        row = {'request_id': 'x', 'startTime': 0, **body}
    elif kind == 'langfuse':
        row = {'id': 'x', 'startTime': 0, 'traceId': 't', 'input': json.dumps(body)}
    else:
        row = {'request_id': 'x', 'request_body': json.dumps(body)}
    normalized = adapt(row)
    assert normalized[field] == body[field]
    with pytest.raises(ValueError, match='unsupported'):
        parse_record(normalized)


@pytest.mark.parametrize('kind', ['litellm', 'langfuse', 'helicone'])
def test_export_adapter_multimodal_parts_rejected(kind):
    from tokaudit.adapters import adapt
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'same'},
                                            {'type': 'image_url', 'image_url': {'url': 'private'}}]}]
    if kind == 'litellm':
        row = {'request_id': 'x', 'startTime': 0, 'messages': messages}
    elif kind == 'langfuse':
        row = {'id': 'x', 'startTime': 0, 'traceId': 't', 'input': json.dumps({'messages': messages})}
    else:
        row = {'request_id': 'x', 'request_body': json.dumps({'messages': messages})}
    with pytest.raises(ValueError, match='unsupported'):
        parse_record(adapt(row))


@pytest.mark.parametrize('role', ['tool', 'function'])
def test_tool_and_function_roles_cannot_lose_identity(role):
    with pytest.raises(ValueError, match='unsupported tool/function'):
        parse_record({'messages': [{'role': role, 'content': 'same'}]})


@pytest.mark.parametrize('field', ['tools', 'functions', 'tool_choice', 'function_call'])
def test_empty_optional_tool_configuration_allowed(field):
    assert parse_record({'prompt': 'safe', field: []}).messages[0].text == 'safe'


def test_unsupported_system_content_rejected():
    with pytest.raises(ValueError, match='unsupported'):
        parse_record({'system': [{'type': 'image', 'source': 'private'}], 'prompt': 'safe'})


def test_supported_text_cache_marker_preserved():
    row = {'system': [{'type': 'text', 'text': 'safe', 'cache_control': {'type': 'ephemeral'}}],
           'prompt': 'question'}
    request = parse_record(row)
    assert request.breakpoints == [0]
    assert request.messages[0].text == 'safe'
