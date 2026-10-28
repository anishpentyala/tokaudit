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


def _langfuse_generation(**kw):
    # Minimal ObservationV2 (official Fern schema, validated 2026-10-28):
    # core fields always present; model/sessionId/input are optional groups.
    base = {
        "id": "obs-1", "traceId": "tr-1", "projectId": "proj-1",
        "type": "GENERATION", "startTime": "2026-10-28T10:00:00.000Z",
        "endTime": "2026-10-28T10:00:01.200Z", "parentObservationId": None,
    }
    base.update(kw)
    return base


def test_langfuse_observation():
    req = parse_record(adapt(_langfuse_generation(
        id="obs-9", model="claude-fable-5-1",
        input={"system": "sys", "messages": [{"role": "user", "content": "hi"}]},
    )))
    assert req.provider == "anthropic"
    assert req.messages[0].role == "system"
    assert req.messages[1].text == "hi"


def test_langfuse_input_list():
    req = parse_record(adapt(_langfuse_generation(
        id="obs-10", model="gpt-5.6",
        input=[{"role": "user", "content": "hi"}],
    )))
    assert req.provider == "openai"


def test_langfuse_v2_raw_string_json_input():
    # The v2 API returns input as a raw string; structured inputs arrive
    # JSON-serialized and must be recovered as messages.
    req = parse_record(adapt(_langfuse_generation(
        id="obs-11", model="gpt-5.6", sessionId="sess-7",
        input='[{"role": "user", "content": "hello there"}]',
    )))
    assert req.messages[0].text == "hello there"
    assert req.session_id == "sess-7"


def test_langfuse_v2_plain_string_prompt():
    req = parse_record(adapt(_langfuse_generation(
        id="obs-12", model="gpt-5.6", input="just a plain prompt",
    )))
    assert req.messages[0].text == "just a plain prompt"


def test_langfuse_non_generation_rejected():
    with pytest.raises(ValueError, match="not an LLM generation"):
        adapt(_langfuse_generation(id="obs-13", type="SPAN", input="x"))


def test_langfuse_missing_io_group_rejected():
    with pytest.raises(ValueError, match="io"):
        adapt(_langfuse_generation(id="obs-14", model="gpt-5.6"))


def _helicone_row(**kw):
    # Minimal flat HeliconeRequest row (packages/llm-mapper/types.ts,
    # validated 2026-10-28): request_id, request_created_at, request_body,
    # response_body, provider, request_model are the load-bearing fields.
    base = {
        "request_id": "h-1", "response_id": "resp-1",
        "request_created_at": "2026-10-28T10:00:00Z",
        "response_created_at": "2026-10-28T10:00:01Z",
        "request_model": "gpt-5.6", "response_model": "gpt-5.6",
        "provider": "OPENAI", "response_status": 200,
        "request_body": {"model": "gpt-5.6",
                         "messages": [{"role": "user", "content": "hey"}]},
        "response_body": {},
    }
    base.update(kw)
    return base


def test_helicone_row():
    req = parse_record(adapt(_helicone_row()))
    assert req.rid == "h-1"
    assert req.messages[0].text == "hey"
    assert req.model == "gpt-5.6"


def test_helicone_string_request_body():
    req = parse_record(adapt(_helicone_row(
        request_id="h-2",
        request_body=json.dumps({"model": "gpt-5.6", "prompt": "one long prompt"}),
    )))
    assert req.messages[0].text == "one long prompt"


def test_helicone_anthropic_style_body_keeps_system():
    req = parse_record(adapt(_helicone_row(
        request_id="h-3", request_model="claude-fable-5-1", provider="ANTHROPIC",
        request_body={"model": "claude-fable-5-1", "system": "be brief",
                      "messages": [{"role": "user", "content": "hi"}]},
    )))
    assert req.provider == "anthropic"
    assert req.messages[0].role == "system"


def test_helicone_missing_messages_rejected():
    with pytest.raises(ValueError, match="no messages/prompt"):
        adapt(_helicone_row(request_id="h-4", request_body={"model": "gpt-5.6"}))


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


def test_langfuse_fixture_end_to_end():
    import pathlib
    path = pathlib.Path(__file__).parent / "fixtures" / "langfuse_observations_v2.jsonl"
    reqs = load_log(str(path))
    assert len(reqs) == 2
    assert reqs[0].session_id == "sess-1"
    assert reqs[0].messages[0].role == "user"
    assert reqs[1].messages[0].text == "plain single string prompt body"


def test_helicone_usage_fields_carried_through():
    req = parse_record(adapt(_helicone_row(
        request_id="h-usage",
        request_body={"model": "gpt-5.6", "prompt": "p"},
    )) | {})
    assert req.actual_usage is None  # base row has no usage fields


def test_helicone_usage_extracted():
    row = _helicone_row(request_id="h-u2",
                        request_body={"model": "gpt-5.6", "prompt": "p"})
    row["prompt_tokens"] = 1200
    row["prompt_cache_read_tokens"] = 900
    row["prompt_cache_write_tokens"] = 300
    req = parse_record(adapt(row))
    assert req.actual_usage == {"prompt_tokens": 1200,
                                "cache_read_tokens": 900,
                                "cache_write_tokens": 300}
