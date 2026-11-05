"""Price files cannot silently select one of several conflicting rates."""
import json
import pytest

from tokaudit.cli import main
from tokaudit.prices import load_price_sheet


@pytest.mark.parametrize('raw', [
    '{"models":{"m":{"input":10,"input":0,"cached_input":1,"cache_write":12}}}',
    '{"models":{"m":{"input":10,"\\u0069nput":0,"cached_input":1,"cache_write":12}}}',
    '{"models":{"m":{"input":10,"cached_input":1,"cache_write":12},"m":{"input":0,"cached_input":0,"cache_write":0}}}',
    '{"models":{"m":{"input":10,"cached_input":1,"cache_write":12}},"models":{"m":{"input":0,"cached_input":0,"cache_write":0}}}',
    '{"models":{"m":{"input":10,"input":10,"cached_input":1,"cache_write":12}}}',
    '{"models":{"m":{"input":10,"cached_input":1,"cache_write":12}},"note":"one","note":"two"}',
])
def test_duplicate_members_fail_closed(tmp_path, raw):
    path = tmp_path / 'prices.json'
    path.write_text(raw)
    with pytest.raises(SystemExit, match='duplicate JSON object member'):
        load_price_sheet(str(path))


@pytest.mark.parametrize('existing', [False, True])
def test_cli_duplicate_price_does_not_publish_dollars(tmp_path, capsys, existing):
    log = tmp_path / 'log.jsonl'
    log.write_text(json.dumps({'id':'a','model':'gpt-6-astra','prompt':'policy '*2000}))
    price = tmp_path / 'prices.json'
    price.write_text('{"models":{"gpt-6-astra":{"input":10,"input":0,"cached_input":1,"cache_write":12}}}')
    output = tmp_path / 'report.json'
    if existing:
        output.write_text('existing verified report')
    with pytest.raises(SystemExit, match='duplicate JSON object member'):
        main(['analyze',str(log),'--tokenizer','estimate','--prices',str(price),
              '--format','json','--output',str(output)])
    if existing:
        assert output.read_text() == 'existing verified report'
    else:
        assert not output.exists()
    assert 'baseline_usd' not in capsys.readouterr().out


def test_duplicate_diagnostic_does_not_echo_keys_or_values(tmp_path):
    path = tmp_path / 'prices.json'
    path.write_text('{"models":{},"private customer marker":1,"private customer marker":2}')
    with pytest.raises(SystemExit) as exc:
        load_price_sheet(str(path))
    assert 'private customer marker' not in str(exc.value)
    assert str(path) in str(exc.value)
    assert 'duplicate JSON object member' in str(exc.value)


def test_same_rate_names_in_different_model_objects_are_not_duplicates(tmp_path):
    path = tmp_path / 'prices.json'
    path.write_text(json.dumps({'models':{'a':{'input':1,'cached_input':0,'cache_write':2},
                                         'b':{'input':2,'cached_input':0,'cache_write':3}}}))
    assert load_price_sheet(str(path)) == {'a':{'input':1.,'cached_input':0.,'cache_write':2.},
                                         'b':{'input':2.,'cached_input':0.,'cache_write':3.}}


def test_excessively_nested_price_json_has_clean_diagnostic(tmp_path):
    path = tmp_path / 'prices.json'
    path.write_text('[' * 10000 + '0' + ']' * 10000)
    with pytest.raises(SystemExit, match='JSON nested too deeply to parse'):
        load_price_sheet(str(path))
