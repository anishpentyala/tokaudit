import json

import pytest

from tokaudit.cli import main
from tokaudit.prices import cost_usd, load_price_sheet


def _sheet(tmp_path, models):
    p = tmp_path / "prices.json"
    p.write_text(json.dumps({"models": models}))
    return str(p)


GOOD = {"gpt-6-astra": {"input": 10.0, "cached_input": 1.0, "cache_write": 12.5}}


def test_valid_sheet_loads(tmp_path):
    sheet = load_price_sheet(_sheet(tmp_path, GOOD))
    assert sheet["gpt-6-astra"]["cache_write"] == 12.5


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_price_sheet(str(tmp_path / "nope.json"))
    assert "not found" in str(e.value)


def test_wrong_shape_fails_closed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(["gpt-6-astra"]))
    with pytest.raises(SystemExit):
        load_price_sheet(str(p))


def test_missing_rate_fails_closed(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_price_sheet(_sheet(tmp_path, {"m": {"input": 1.0, "cached_input": 0.1}}))
    assert "cache_write" in str(e.value)


def test_typo_rate_key_fails_closed(tmp_path):
    bad = {"m": {"input": 1.0, "cached_imput": 0.1, "cache_write": 1.25}}
    with pytest.raises(SystemExit) as e:
        load_price_sheet(_sheet(tmp_path, bad))
    assert "cached_imput" in str(e.value)


def test_negative_rate_fails_closed(tmp_path):
    bad = {"m": {"input": -1.0, "cached_input": 0.1, "cache_write": 1.25}}
    with pytest.raises(SystemExit) as e:
        load_price_sheet(_sheet(tmp_path, bad))
    assert "non-negative" in str(e.value)


def test_cost_usd_math():
    # 1M uncached + 2M read + 0.5M written at 10/1/12.5 per 1M.
    usd = cost_usd(1_000_000, 2_000_000, 500_000,
                   {"input": 10.0, "cached_input": 1.0, "cache_write": 12.5})
    assert usd == pytest.approx(10.0 + 2.0 + 6.25)


def test_cli_priced_report(tmp_path, capsys):
    import os
    ex = os.path.join(os.path.dirname(__file__), "..", "examples", "quickstart.jsonl")
    sheet = _sheet(tmp_path, GOOD)
    rc = main(["analyze", ex, "--tokenizer", "estimate", "--format", "json",
               "--prices", sheet])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    p = data["pricing"]
    assert p["priced_requests"] == 6 and p["unpriced_requests"] == []
    assert p["baseline_usd"] > p["simulated_usd"] > 0
    assert "not observed billing" in p["note"]


def test_cli_unpriced_models_excluded(tmp_path, capsys):
    import os
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_log.jsonl")
    sheet = _sheet(tmp_path, {"no-such-model-zzz": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25}})
    rc = main(["analyze", fixture, "--tokenizer", "estimate", "--format", "json",
               "--prices", sheet])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    groups = data["groups"] if "groups" in data else [data]
    for g in groups:
        assert g["pricing"]["priced_requests"] == 0
        assert g["pricing"]["unpriced_requests"]


def test_markdown_shows_dollars(tmp_path):
    import os
    ex = os.path.join(os.path.dirname(__file__), "..", "examples", "quickstart.jsonl")
    sheet = _sheet(tmp_path, GOOD)
    out = tmp_path / "r.md"
    rc = main(["analyze", ex, "--tokenizer", "estimate", "--prices", sheet,
               "--output", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "From your price sheet" in text and "$" in text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                 "NaN", "Infinity", "-Infinity", "1e309"])
@pytest.mark.parametrize("key", ["input", "cached_input", "cache_write"])
def test_nonfinite_rates_fail_closed(tmp_path, value, key):
    rates = {"input": 1., "cached_input": .1, "cache_write": 1.25}
    rates[key] = value
    with pytest.raises(SystemExit, match="finite"):
        load_price_sheet(_sheet(tmp_path, {"m": rates}))


@pytest.mark.parametrize("value", [True, False])
def test_boolean_rates_fail_closed(tmp_path, value):
    with pytest.raises(SystemExit, match="must be a number"):
        load_price_sheet(_sheet(tmp_path, {"m": {"input": value, "cached_input": .1, "cache_write": 1.25}}))


def test_overflowing_integer_rate_fails_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="must be a number"):
        load_price_sheet(_sheet(tmp_path, {"m": {"input": 10**400, "cached_input": .1, "cache_write": 1.25}}))


def test_zero_and_finite_numeric_string_rates_remain_valid(tmp_path):
    rates = load_price_sheet(_sheet(tmp_path, {"m": {"input": "1.5", "cached_input": 0, "cache_write": "2.0"}}))
    assert cost_usd(1_000_000, 2, 0, rates["m"]) == pytest.approx(1.5)


def test_cli_rejects_invalid_prices_before_writing_dollar_report(tmp_path, capsys):
    from pathlib import Path
    ex = str(Path(__file__).resolve().parents[1] / "examples" / "quickstart.jsonl")
    prices = _sheet(tmp_path, {"gpt-6-astra": {"input": "NaN", "cached_input": .1, "cache_write": 1.25}})
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="finite"):
        main(["analyze", ex, "--tokenizer", "estimate", "--prices", prices,
              "--format", "json", "--output", str(output)])
    assert not output.exists()
    assert "baseline_usd" not in capsys.readouterr().out
