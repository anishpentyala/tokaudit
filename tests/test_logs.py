import pytest

from tokaudit.logs import parse_record


def test_openai_style():
    req = parse_record({
        "id": "a1", "model": "gpt-5.6", "created": 1761724800,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    })
    assert req.rid == "a1"
    assert req.provider == "openai"
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.ts == 1761724800.0
    assert req.breakpoints == []
    assert "<system>" in req.canonical()


def test_anthropic_style_with_system_and_breakpoint():
    req = parse_record({
        "model": "claude-opus-5",
        "system": [
            {"type": "text", "text": "Big instructions.", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Question?"}]}],
    })
    assert req.provider == "anthropic"
    assert req.messages[0].role == "system"
    assert req.breakpoints == [0]


def test_openai_explicit_breakpoint_part():
    req = parse_record({
        "model": "gpt-5.6",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "static part", "prompt_cache_breakpoint": {"mode": "explicit"}},
                {"type": "text", "text": "volatile part"},
            ],
        }],
    })
    assert req.breakpoints == [0]
    assert "static part" in req.messages[0].text


def test_generic_prompt_and_iso_ts():
    req = parse_record({"prompt": "just text", "ts": "2026-10-28T09:00:00Z"})
    assert req.provider == "generic"
    assert req.messages[0].role == "user"
    assert req.ts is not None


def test_missing_content_raises():
    with pytest.raises(ValueError):
        parse_record({"model": "gpt-5.6"})
