"""A hash collision is not a cache hit."""
import sys
import pytest
from tokaudit.logs import Request,Message
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules

@pytest.mark.parametrize('refresh',[False,True])
@pytest.mark.parametrize('shape',['equal-length','real-repeat'])
def test_colliding_prefixes_do_not_share_cache(refresh,shape):
    p=sys.hash_info.modulus
    # Collision reproducible without relying on random hash seeds.
    a=[1]*8;b=[1+p]*8
    seq={'a':a,'b':b}
    enc=lambda text:seq[text.rsplit('\n',1)[-1]]
    rows=[Request('a','m','generic',[Message('user','a')],ts=0),Request('b','m','generic',[Message('user','b')],ts=1)]
    if shape=='real-repeat':rows.append(Request('c','m','generic',[Message('user','b')],ts=2))
    rules=ProviderRules('generic',2,10,1.25,0.1,refresh,True)
    assert hash(tuple(a))==hash(tuple(b)) and a!=b
    result=simulate(rows,enc,rules)
    assert result.per_request[1].read==0
    assert result.per_request[1].written==8
    if shape=='real-repeat':
        assert result.per_request[2].read==8
        assert result.wasted_write_tokens==8
    else:
        assert result.wasted_write_tokens==16

@pytest.mark.parametrize('offset',[0,1<<80,-(1<<80)])
def test_arbitrary_size_custom_token_ids_keep_exact_equality(offset):
    seq={'a':[offset+1]*8,'b':[offset+2]*8}
    enc=lambda text:seq[text.rsplit('\n',1)[-1]]
    rows=[Request(x,'m','generic',[Message('user',text)],ts=i) for i,(x,text) in enumerate([('a','a'),('b','b'),('c','a')])]
    rules=ProviderRules('generic',2,10,1.25,0.1,False,True)
    result=simulate(rows,enc,rules)
    assert [c.read for c in result.per_request]==[0,0,8]
    assert result.wasted_write_tokens==8

@pytest.mark.parametrize('values',[[0,1,-1],[-(1<<63),(1<<63)-1],[1<<80,-(1<<80)]])
def test_exact_key_keeps_shape_order_and_full_integer_values(values):
    from tokaudit.prefix import _prefix_key
    assert _prefix_key(values)==_prefix_key(list(values))
    assert _prefix_key(values)!=_prefix_key(values[::-1])
    assert _prefix_key(values)!=_prefix_key(values+[0])
    assert _prefix_key(values)!=_prefix_key([v+1 for v in values])


def test_expired_prefix_does_not_read_despite_exact_match():
    enc=lambda text:[1]*8
    rows=[Request(str(i),'m','generic',[Message('user','a')],ts=ts) for i,ts in enumerate([0,11])]
    result=simulate(rows,enc,ProviderRules('generic',2,10,1.25,0.1,False,True))
    assert [c.read for c in result.per_request]==[0,0]
    assert result.wasted_write_tokens==16
