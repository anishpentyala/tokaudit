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
