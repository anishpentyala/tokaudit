"""Multi-turn conversation and adversarial-prefix scenarios."""

from tokaudit.logs import Message, Request
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.tokenizers import EstimatorBackend

ENC = EstimatorBackend().encode
RULES = ProviderRules(
    name="openai", min_prefix_tokens=32, ttl_seconds=1800,
    write_multiplier=1.25, read_multiplier=0.1,
    refresh_on_read=False, enabled_by_default=True,
)
SYSTEM = "You are a careful assistant. " + ("Follow the house style guide precisely. " * 30)


def turn(rid, history, ts):
    return Request(rid=rid, model="gpt-5.6", provider="openai", messages=list(history), ts=ts)


def test_growing_conversation_hits_hard():
    history = [Message("system", SYSTEM)]
    reqs = []
    for i in range(5):
        history = history + [Message("user", f"question {i}"), Message("assistant", f"answer {i} " * 20)]
        reqs.append(turn(f"t{i}", history, float(i * 120)))
    sim = simulate(reqs, ENC, RULES)
    # Turn 1 writes the whole thing; later turns re-read nearly everything.
    assert sim.per_request[0].read == 0
    for c in sim.per_request[1:]:
        assert c.read / c.total_input > 0.5
    assert sim.savings_pct > 40


def test_volatile_prefix_kills_cache():
    reqs = [
        Request(rid=f"v{i}", model="gpt-5.6", provider="openai",
                messages=[Message("system", SYSTEM + f" Current time: 2026-10-28T10:0{i}:00"),
                          Message("user", "same question")],
                ts=float(i * 60))
        for i in range(3)
    ]
    sim = simulate(reqs, ENC, RULES)
    for c in sim.per_request:
        assert c.read == 0  # the timestamp inside the prefix breaks every match
    assert sim.wasted_write_tokens > 0


def test_volatile_suffix_preserves_cache():
    reqs = [
        Request(rid=f"s{i}", model="gpt-5.6", provider="openai",
                messages=[Message("system", SYSTEM),
                          Message("user", f"question at 2026-10-28T10:0{i}:00")],
                ts=float(i * 60))
        for i in range(3)
    ]
    sim = simulate(reqs, ENC, RULES)
    for c in sim.per_request[1:]:
        assert c.read > 0  # stable prefix survives; only the tail is new

def _ns_turn(rid, history, ts, model="gpt-5.6", session=None, cache_key=None):
    return Request(rid=rid, model=model, provider="openai", messages=list(history),
                   ts=ts, session_id=session, cache_key=cache_key)


def test_models_never_share_cache_entries():
    # OpenAI routes cache reuse per model within an org+region, so an identical
    # prefix under another model is a cold miss, not a read.
    history = [Message("system", SYSTEM), Message("user", "hi " * 40)]
    reqs = [
        _ns_turn("m1", history, 0.0, model="gpt-5.6"),
        _ns_turn("m2", history, 10.0, model="gpt-5.6-mini"),
        _ns_turn("m3", history + [Message("assistant", "ok " * 40)], 20.0, model="gpt-5.6"),
    ]
    sim = simulate(reqs, ENC, RULES)
    first, second, third = sim.per_request
    assert first.read == 0
    assert second.read == 0
    assert second.written > 0
    assert third.read > 0


def test_prompt_cache_keys_separate_reuse_groups():
    # A supplied prompt_cache_key separates cache reuse between groups.
    history = [Message("system", SYSTEM), Message("user", "hi " * 40)]
    reqs = [
        _ns_turn("k1", history, 0.0, cache_key="tenant-a"),
        _ns_turn("k2", history, 10.0, cache_key="tenant-b"),
        _ns_turn("k3", history + [Message("assistant", "ok " * 40)], 20.0, cache_key="tenant-a"),
    ]
    sim = simulate(reqs, ENC, RULES)
    first, second, third = sim.per_request
    assert first.read == 0
    assert second.read == 0
    assert third.read > 0


def test_sessions_share_cache_by_default():
    # Provider docs give no app-session cache boundary: identical prefixes in
    # different sessions of the same model DO reuse by default.
    history = [Message("system", SYSTEM), Message("user", "hi " * 40)]
    reqs = [
        _ns_turn("a1", history, 0.0, session="session-a"),
        _ns_turn("b1", history, 10.0, session="session-b"),
    ]
    sim = simulate(reqs, ENC, RULES)
    assert sim.per_request[1].read > 0


def test_isolate_sessions_is_an_explicit_conservative_scenario():
    history = [Message("system", SYSTEM), Message("user", "hi " * 40)]
    reqs = [
        _ns_turn("a1", history, 0.0, session="session-a"),
        _ns_turn("b1", history, 10.0, session="session-b"),
    ]
    sim = simulate(reqs, ENC, RULES, isolate_sessions=True)
    assert sim.per_request[1].read == 0
    assert sim.per_request[1].written > 0
    assert any("conservative scenario" in n for n in sim.notes)


def test_requests_without_session_share_one_namespace():
    # Backwards-compatible: no session_id behaves like a single shared cache.
    history = [Message("system", SYSTEM), Message("user", "hi " * 40)]
    reqs = [
        _ns_turn("n1", history, 0.0),
        _ns_turn("n2", history + [Message("assistant", "ok " * 40)], 10.0),
    ]
    sim = simulate(reqs, ENC, RULES)
    assert sim.per_request[1].read > 0
