"""Fail-closed input handling: bad inputs exit with a clean message,
never a traceback or silent mis-parse."""

import json

import pytest

from tokaudit.cli import main
from tokaudit.logs import load_log
from tokaudit.rules import load_rules


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_log_invalid_json_line(tmp_path):
    path = _write(tmp_path, "bad.jsonl", '{"model": "gpt-test", "prompt": "hi"}\nnot json\n')
    with pytest.raises(SystemExit) as e:
        load_log(path)
    assert "bad.jsonl:2" in str(e.value) and "invalid JSON" in str(e.value)


def test_log_non_object_line(tmp_path):
    path = _write(tmp_path, "arr.jsonl", '["not", "an", "object"]\n')
    with pytest.raises(SystemExit) as e:
        load_log(path)
    assert "expected a JSON object" in str(e.value)


def test_log_record_without_prompt(tmp_path):
    path = _write(tmp_path, "empty.jsonl", '{"model": "gpt-test"}\n')
    with pytest.raises(SystemExit) as e:
        load_log(path)
    assert "no messages/prompt field" in str(e.value)


def test_log_message_not_object(tmp_path):
    path = _write(tmp_path, "msg.jsonl", '{"messages": ["just a string"]}\n')
    with pytest.raises(SystemExit) as e:
        load_log(path)
    assert "message is not an object" in str(e.value)


def test_log_empty_file(tmp_path):
    path = _write(tmp_path, "empty.jsonl", "\n\n")
    with pytest.raises(SystemExit) as e:
        load_log(path)
    assert "no requests found" in str(e.value)


def test_rules_missing_file(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_rules(str(tmp_path / "nope.json"))
    assert "not found" in str(e.value)


def test_rules_invalid_json(tmp_path):
    path = _write(tmp_path, "rules.json", "{not json")
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "invalid JSON" in str(e.value)


def test_rules_wrong_shape(tmp_path):
    path = _write(tmp_path, "rules.json", json.dumps(["openai"]))
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "'providers' object" in str(e.value)


def test_rules_provider_not_object(tmp_path):
    path = _write(tmp_path, "rules.json", json.dumps({"providers": {"openai": 42}}))
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "'openai' must be an object" in str(e.value)


def test_rules_bad_numeric_field(tmp_path):
    path = _write(tmp_path, "rules.json", json.dumps(
        {"providers": {"openai": {"ttl_seconds": "soon"}}}))
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "'openai'" in str(e.value)


def test_cli_bad_log_exits_cleanly(tmp_path, capsys):
    path = _write(tmp_path, "bad.jsonl", "not json\n")
    with pytest.raises(SystemExit):
        main(["analyze", path])
    assert "Traceback" not in capsys.readouterr().err


def test_rules_negative_multiplier_rejected(tmp_path):
    path = _write(tmp_path, "rules.json", json.dumps(
        {"providers": {"openai": {"read_multiplier": -0.5}}}))
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "non-negative" in str(e.value)


def test_rules_negative_family_ttl_rejected(tmp_path):
    path = _write(tmp_path, "rules.json", json.dumps(
        {"providers": {"openai": {"families": {"f1": {"ttl_seconds": -1}}}}}))
    with pytest.raises(SystemExit) as e:
        load_rules(path)
    assert "family 'f1'" in str(e.value) and "non-negative" in str(e.value)
