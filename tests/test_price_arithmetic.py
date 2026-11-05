"""Finite rates must not produce nonfinite dollar reports silently."""
import json
import math

import pytest

from tokaudit.cli import main
from tokaudit.prices import cost_usd


def rates(value):return {'input':value,'cached_input':value,'cache_write':value}


@pytest.mark.parametrize('counts,rate,expected',[
    ((1000000,0,0),1e308,1e308),
    ((500000,250000,250000),1e308,1e308),
    ((10**310,0,0),1e-308,1e-4),
    ((10**400,0,0),0.,0.),
])
def test_representable_cost_survives_intermediate_overflow(counts,rate,expected):
    actual=cost_usd(*counts,rates(rate))
    assert math.isfinite(actual) and actual==pytest.approx(expected)


@pytest.mark.parametrize('counts',[(10**400,0,0),(0,10**400,0),(0,0,10**400)])
def test_unrepresentable_cost_is_explicit_error(counts):
    with pytest.raises(ValueError,match='finite dollar range'):
        cost_usd(*counts,rates(1.))


@pytest.mark.parametrize('existing',[False,True])
def test_cli_aggregate_overflow_does_not_write_nonfinite_report(tmp_path,existing):
    log=tmp_path/'log.jsonl'
    # One request is representable, but the four-request baseline total is not.
    log.write_text('\n'.join(json.dumps({'id':str(i),'model':'gpt-6-astra','prompt':'word '*400000}) for i in range(4)))
    prices=tmp_path/'prices.json';prices.write_text(json.dumps({'models':{'gpt-6-astra':rates(1e308)}}))
    output=tmp_path/'out.json'
    if existing:output.write_text('prior verified report')
    with pytest.raises(SystemExit,match='finite dollar range'):
        main(['analyze',str(log),'--tokenizer','estimate','--prices',str(prices),
              '--format','json','--dup-max-shingles','0','--output',str(output)])
    if existing:assert output.read_text()=='prior verified report'
    else:assert not output.exists()


def test_normal_cost_math_unchanged():
    r={'input':10.,'cached_input':1.,'cache_write':12.5}
    assert cost_usd(1000000,2000000,500000,r)==18.25


def test_nonfinite_individual_cli_cost_is_clean_error(tmp_path):
    log=tmp_path/'log.jsonl'
    log.write_text(json.dumps({'id':'a','model':'gpt-6-astra','prompt':'word '*1500000}))
    prices=tmp_path/'prices.json';prices.write_text(json.dumps({'models':{'gpt-6-astra':rates(1e308)}}))
    output=tmp_path/'out.json'
    with pytest.raises(SystemExit,match='finite dollar range'):
        main(['analyze',str(log),'--tokenizer','estimate','--prices',str(prices),
              '--format','json','--dup-max-shingles','0','--output',str(output)])
    assert not output.exists()
