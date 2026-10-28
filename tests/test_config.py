import json
import os

import pytest

from tokaudit.cli import main
from tokaudit.config import load_config

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_log.jsonl")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_no_config_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == {}


def test_config_values_loaded(tmp_path):
    path = _write(tmp_path, ".tokaudit.toml",
                  'tokenizer = "estimate"\nformat = "json"\ndup_min_tokens = 256\n'
                  'isolate_sessions = true\nwhat_if = ["dense"]\n')
    cfg = load_config(path)
    assert cfg == {"tokenizer": "estimate", "format": "json",
                   "dup_min_tokens": 256, "isolate_sessions": True,
                   "what_if": ["dense"]}


def test_unknown_key_fails_closed(tmp_path):
    path = _write(tmp_path, ".tokaudit.toml", 'tokeniser = "estimate"\n')
    with pytest.raises(SystemExit) as e:
        load_config(path)
    assert "unknown keys" in str(e.value) and "tokeniser" in str(e.value)


def test_invalid_toml_fails_closed(tmp_path):
    path = _write(tmp_path, ".tokaudit.toml", "tokenizer = = oops\n")
    with pytest.raises(SystemExit) as e:
        load_config(path)
    assert "invalid TOML" in str(e.value)


def test_missing_explicit_config_fails_closed(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_config(str(tmp_path / "nope.toml"))
    assert "not found" in str(e.value)


def test_wrong_types_fail_closed(tmp_path):
    path = _write(tmp_path, ".tokaudit.toml", 'dup_min_tokens = "lots"\n')
    with pytest.raises(SystemExit):
        load_config(path)
    path = _write(tmp_path, ".tokaudit.toml", 'what_if = "dense"\n')
    with pytest.raises(SystemExit):
        load_config(path)


def test_cli_uses_config_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".tokaudit.toml", 'tokenizer = "estimate"\nformat = "json"\n')
    rc = main(["analyze", FIXTURE])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "summary" in data or "groups" in data


def test_cli_flag_overrides_config(tmp_path, capsys):
    cfg = _write(tmp_path, "cfg.toml", 'format = "json"\n')
    rc = main(["analyze", FIXTURE, "--config", cfg, "--format", "md"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("# tokaudit report")  # markdown, not JSON


def test_cli_config_what_if(tmp_path, capsys):
    cfg = _write(tmp_path, "cfg.toml", 'format = "json"\nwhat_if = ["dedup"]\n')
    rc = main(["analyze", FIXTURE, "--config", cfg, "--tokenizer", "estimate"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    groups = data["groups"] if "groups" in data else [data]
    for g in groups:
        assert [s["name"] for s in g["what_if"]] == ["dedup"]
