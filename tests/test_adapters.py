import json

import pytest

from tokaudit.logs import load_log, parse_record
from tokaudit.adapters import adapt


def test_litellm_spendlog():
    req = parse_record(adapt({
        "request_id": "req_abc", "call_type": "acompletion",
        "model": "gpt-5.6", "custom_llm_provider": "openai",
        "startTime": "2026-10-28T10:00:00+00:00",
        "endTime": "2026-10-28T10:00:02+00:00",
        "messages": [{"role": "user", "content": "hello"}],
        "session_id": "sess-1", "prompt_tokens": 12,
    }))
    assert req.rid == "req_abc"
    assert req.provider == "openai"
    assert req.ts is not None
    assert req.session_id == "sess-1"
    assert req.messages[0].text == "hello"


def test_litellm_string_messages():
    req = parse_record(adapt({
        "request_id": "r2", "startTime": "2026-10-28T10:00:00Z",
        "model": "gpt-5.6", "messages": "plain string prompt",
    }))
    assert req.messages[0].role == "user"
    assert req.messages[0].text == "plain string prompt"


def test_langfuse_observation():
    req = parse_record(adapt({
        "id": "obs-9", "model": "claude-fable-5-1",
        "startTime": "2026-10-28T10:00:00Z",
        "input": {"system": "sys", "messages": [{"role": "user", "content": "hi"}]},
    }))
    assert req.provider == "anthropic"
    assert req.messages[0].role == "system"
    assert req.messages[1].text == "hi"


def test_langfuse_input_list():
    req = parse_record(adapt({
        "id": "obs-10", "model": "gpt-5.6",
        "startTime": "2026-10-28T10:00:00Z",
        "input": [{"role": "user", "content": "hi"}],
    }))
    assert req.provider == "openai"


def test_helicone_row():
    req = parse_record(adapt({
        "request_id": "h-1", "created_at": "2026-10-28T10:00:00Z",
        "request": {"model": "gpt-5.6", "body": {"model": "gpt-5.6",
                    "messages": [{"role": "user", "content": "hey"}]}},
        "response": {"status": 200, "body": {}},
    }))
    assert req.rid == "h-1"
    assert req.messages[0].text == "hey"


def test_unknown_shape_passes_through():
    obj = {"model": "gpt-5.6", "messages": [{"role": "user", "content": "x"}]}
    assert adapt(obj) is obj


def test_bad_line_reports_line_number(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('{"model": "gpt-5.6", "messages": []}\n{not json}\n')
    with pytest.raises(SystemExit) as exc:
        load_log(str(f))
    assert ":2:" in str(exc.value)


def test_adapter_mismatch_is_loud(tmp_path):
    f = tmp_path / "odd.jsonl"
    f.write_text(json.dumps({
        "request_id": "x", "startTime": "2026-10-28T10:00:00Z",
        "model": "gpt-5.6", "messages": 42,
    }) + "\n")
    with pytest.raises(SystemExit) as exc:
        load_log(str(f))
    assert "unexpected type" in str(exc.value)
