"""Finite cache TTLs must not be widened or shortened by epoch arithmetic."""
import math
import pytest
from fractions import Fraction
from tokaudit.logs import Request,Message
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules


def run(times,ttl=10000,refresh=False,refresh_seconds=None):
    rules=ProviderRules('custom',1,ttl,1.25,.1,refresh,True,
                         read_refresh_seconds=refresh_seconds)
    rows=[Request(str(i),'m','generic',[Message('user','constant')],ts=t) for i,t in enumerate(times)]
    return simulate(rows,lambda text:list(text.encode()),rules)


@pytest.mark.parametrize('epoch',[1e20,-1e20,1e100,-1e100,1e308,-1e308])
@pytest.mark.parametrize('refresh',[False,True])
def test_rounding_ttl_up_does_not_make_late_request_a_hit(epoch,refresh):
    later=math.nextafter(epoch,math.inf)
    gap=Fraction(later)-Fraction(epoch)
    ttl=int(gap*Fraction(3,5))
    assert gap>ttl
    result=run([epoch,later],ttl=ttl,refresh=refresh)
    assert result.per_request[1].read==0


@pytest.mark.parametrize('refresh',[False,True])
def test_exact_boundary_remains_inclusive_at_large_epoch(refresh):
    t=1e20;later=math.nextafter(t,math.inf)
    result=run([t,later],ttl=int(Fraction(later)-Fraction(t)),refresh=refresh)
    assert result.per_request[1].read>0


def test_refresh_deadline_is_not_rounded_up():
    t=1e20
    # Initial lifetime covers the first hit. Its shorter refresh must not
    # shrink coverage or extend beyond the exact later max deadline.
    times=[t,t+16384,t+32768,t+49152]
    result=run(times,ttl=30000,refresh=True,refresh_seconds=20000)
    assert result.per_request[1].read>0
    assert result.per_request[2].read>0
    assert result.per_request[3].read>0


def test_refresh_does_not_invent_extra_lifetime_beyond_last_exact_deadline():
    t=1e20
    result=run([t,t+16384,t+49152],ttl=30000,refresh=True,refresh_seconds=20000)
    assert result.per_request[1].read>0
    assert result.per_request[2].read==0


@pytest.mark.parametrize('times',[[0,10000,10001],[1e20,1e20,None]])
def test_ordinary_boundaries_and_missing_timestamp_policy_preserved(times):
    result=run(times)
    if times[-1] is None:
        assert all(c.read>0 for c in result.per_request[1:])
        assert any('timestamps' in n for n in result.notes)
    else:
        assert result.per_request[1].read>0
        assert result.per_request[2].read==0
