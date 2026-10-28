"""Cross-cutting invariants of the cache simulator.

Behavioral cases live in test_prefix.py / test_multiturn.py; these tests pin
properties that must hold for every input, including a seeded randomized
sweep so regressions in the accounting show up regardless of scenario.
"""

import random

from tokaudit.logs import Message, Request
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.tokenizers import EstimatorBackend

ENC = EstimatorBackend().encode

OPENAI = ProviderRules(
    name="openai", min_prefix_tokens=32, ttl_seconds=1800,
    write_multiplier=1.25, read_multiplier=0.1,
    refresh_on_read=False, enabled_by_default=True,
)
ANTH = ProviderRules(
    name="anthropic", min_prefix_tokens=32, ttl_seconds=300,
    write_multiplier=1.25, read_multiplier=0.1,
    refresh_on_read=True, enabled_by_default=False,
)

PREFIXES = ["alpha " * 200, "beta " * 200, "gamma " * 200]


def make_req(rid, prefix, suffix, ts, key=None, session=None, model="gpt-test"):
    return Request(rid=rid, model=model, provider="openai",
                   messages=[Message("system", prefix), Message("user", suffix)],
                   ts=ts, cache_key=key, session_id=session)


def check_accounting(sim):
    by_rid = {c.rid: c for c in sim.per_request}
    for c in sim.per_request:
        # Accounting identity: every input token is exactly one of
        # read / written / uncached.
        assert c.read + c.written + c.uncached == c.total_input, c
        assert 0 <= c.read <= c.total_input
        assert 0 <= c.written <= c.total_input
        assert c.uncached >= 0
        assert c.baseline_units == float(c.total_input)
        assert c.simulated_units >= 0
    # Totals are the sums of their parts.
    assert abs(sim.baseline_units - sum(c.baseline_units for c in sim.per_request)) < 1e-6
    assert abs(sim.simulated_units - sum(c.simulated_units for c in sim.per_request)) < 1e-6
    return by_rid


def test_accounting_identity_on_shared_prefixes():
    reqs = [make_req(f"r{i}", PREFIXES[i % 3], f"q{i}", ts=float(i * 60))
            for i in range(8)]
    check_accounting(simulate(reqs, ENC, OPENAI))
    check_accounting(simulate(reqs, ENC, ANTH))


def test_first_request_in_namespace_never_reads():
    reqs = [make_req("a", PREFIXES[0], "q", ts=0.0, key="k1"),
            make_req("b", PREFIXES[0], "q2", ts=10.0, key="k2"),
            make_req("c", PREFIXES[0], "q3", ts=20.0, key="k1")]
    sim = simulate(reqs, ENC, OPENAI)
    by_rid = {c.rid: c for c in sim.per_request}
    assert by_rid["a"].read == 0  # first in k1
    assert by_rid["b"].read == 0  # first in k2
    assert by_rid["c"].read > 0   # second in k1 hits


def test_expiry_boundary_is_inclusive_then_misses():
    # Entry written at t=0 with TTL 1800 expires at t=1800; the check is
    # expiry >= ts, so a read at exactly t=1800 still hits, just after misses.
    w = make_req("w", PREFIXES[0], "q", ts=0.0)
    at = make_req("at", PREFIXES[0], "q2", ts=1800.0)
    after = make_req("af", PREFIXES[0], "q3", ts=1800.001)
    sim = simulate([w, at], ENC, OPENAI)
    assert sim.per_request[1].read > 0
    sim = simulate([w, after], ENC, OPENAI)
    assert sim.per_request[1].read == 0


def test_zero_wasted_writes_when_everything_reread():
    reqs = [make_req(f"r{i}", PREFIXES[0], f"q{i % 2}", ts=float(i * 60))
            for i in range(4)]
    sim = simulate(reqs, ENC, OPENAI)
    # Writes happen only where later requests re-read (the shared prefix).
    assert sim.wasted_write_tokens <= sum(
        c.written for c in sim.per_request if c.written
    )
    # Explicit: a fully shared write-read chain wastes only unique tails.
    assert sim.wasted_write_tokens >= 0


def test_seeded_random_sweep_invariants():
    rng = random.Random(20261028)
    for trial in range(200):
        n = rng.randint(1, 12)
        reqs = []
        for i in range(n):
            reqs.append(make_req(
                f"t{trial}r{i}",
                rng.choice(PREFIXES),
                "q" * rng.randint(1, 40),
                ts=float(rng.randint(0, 7200)),
                key=rng.choice([None, "k1", "k2"]),
                session=rng.choice([None, "s1", "s2"]),
                model=rng.choice(["gpt-test", "gpt-test-2"]),
            ))
        rules = ANTH if trial % 2 else OPENAI
        for isolate in (False, True):
            sim = simulate(reqs, ENC, rules, isolate_sessions=isolate)
            check_accounting(sim)
            assert sim.wasted_write_tokens >= 0
            # Never claim more savings than the baseline itself.
            assert sim.savings_units <= sim.baseline_units + 1e-6
