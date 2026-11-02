"""A read of a rewritten prefix must not credit an expired unused write."""
import pytest
from tokaudit.logs import Request,Message
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules

@pytest.mark.parametrize('refresh',[False,True])
@pytest.mark.parametrize('times,expected',[
    ([0,11,12],8),
    ([0,1,12,13],0),
    ([0,11,22,23],16),
    ([0,1,12],8),
    ([0,10],0),
])
def test_wasted_writes_are_per_cache_generation(refresh,times,expected):
    rows=[Request(str(i),'m','generic',[Message('user','repeat')],ts=t) for i,t in enumerate(times)]
    rules=ProviderRules('generic',2,10,1.25,0.1,refresh,True)
    result=simulate(rows,lambda text:[1]*8,rules)
    assert result.wasted_write_tokens==expected

@pytest.mark.parametrize('refresh',[False,True])
def test_reads_of_short_prefix_credit_only_current_generation(refresh):
    # Different tails force new writes while the shared prefix is read.
    seq={'<system>\nshared':[1]*8,
         '<system>\nshared\n<user>\na':[1]*8+[2]*8,
         '<system>\nshared\n<user>\nb':[1]*8+[3]*8,
         '<system>\nshared\n<user>\nc':[1]*8+[4]*8}
    rows=[Request(str(i),'m','generic',[Message('system','shared'),Message('user',tail)],ts=t) for i,(tail,t) in enumerate([('a',0),('b',11),('c',12)])]
    result=simulate(rows,seq.__getitem__,ProviderRules('generic',2,10,1.25,0.1,refresh,True))
    assert [c.read for c in result.per_request]==[0,0,8]
    # First16-token write expired unused; second16 shares read prefix, last8 unused.
    assert result.wasted_write_tokens==24

@pytest.mark.parametrize('isolate',[False,True])
def test_namespace_generations_do_not_credit_other_namespace(isolate):
    rows=[Request(str(i),'m','generic',[Message('user','repeat')],ts=t,cache_key=key,session_id=key) for i,(t,key) in enumerate([(0,'x'),(11,'y'),(12,'y')])]
    result=simulate(rows,lambda text:[1]*8,ProviderRules('generic',2,10,1.25,0.1,False,True),isolate_sessions=isolate)
    assert [c.read for c in result.per_request]==[0,0,8]
    assert result.wasted_write_tokens==8


def test_read_refresh_preserves_current_write_generation():
    rows=[Request(str(i),'m','generic',[Message('user','repeat')],ts=t) for i,t in enumerate([0,9,18,27])]
    result=simulate(rows,lambda text:[1]*8,ProviderRules('generic',2,10,1.25,0.1,True,True))
    assert [c.read for c in result.per_request]==[0,8,8,8]
    assert result.wasted_write_tokens==0
