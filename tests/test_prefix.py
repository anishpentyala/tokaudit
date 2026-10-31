from tokaudit.logs import Request, Message
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

SYSTEM = "word " * 600  # long stable system prompt (~150 estimator tokens)


def make_req(rid, user_text, ts):
    return Request(rid=rid, model="gpt-test", provider="openai",
                   messages=[Message("system", SYSTEM), Message("user", user_text)], ts=ts)


def test_shared_prefix_gets_read_hits():
    reqs = [make_req(f"r{i}", f"question number {i}", ts=float(i * 60)) for i in range(4)]
    sim = simulate(reqs, ENC, OPENAI)
    assert sim.per_request[0].read == 0
    assert sim.per_request[0].written > 0
    for c in sim.per_request[1:]:
        assert c.read >= 32  # the shared system prefix is reused
    assert sim.savings_units > 0
    # Each later request still writes its unique short tail, which is never
    # re-read; those tail writes are correctly counted as wasted.
    assert sim.wasted_write_tokens == sum(c.written for c in sim.per_request[1:])


def test_below_minimum_nothing_cached():
    reqs = [Request(rid="s1", model="gpt-test", provider="openai",
                    messages=[Message("user", "hi")], ts=0.0)]
    sim = simulate(reqs, ENC, OPENAI)
    assert sim.per_request[0].read == 0
    assert sim.per_request[0].written == 0
    assert sim.savings_units == 0


def test_ttl_expiry_blocks_read():
    reqs = [make_req("a", "first", 0.0), make_req("b", "second", 10_000.0)]
    sim = simulate(reqs, ENC, OPENAI)  # ttl 1800s < 10000s
    assert sim.per_request[1].read == 0


def test_anthropic_refresh_on_read_extends():
    reqs = [make_req("a", "one", 0.0), make_req("b", "two", 200.0), make_req("c", "three", 400.0)]
    sim = simulate(reqs, ENC, ANTH)  # ttl 300, refresh keeps it alive
    assert sim.per_request[1].read > 0
    assert sim.per_request[2].read > 0


def test_single_use_write_costs_extra():
    reqs = [make_req("only", "once", 0.0)]
    sim = simulate(reqs, ENC, OPENAI)
    c = sim.per_request[0]
    assert c.written > 0 and c.read == 0
    assert sim.simulated_units > sim.baseline_units  # 1.25x write, no reuse
    assert sim.wasted_write_tokens == c.written
    assert any("never re-read" in n for n in sim.notes)


def test_no_timestamps_notes_assumption():
    r = make_req("x", "q", 0.0)
    r.ts = None
    sim = simulate([r, r], ENC, OPENAI)
    assert any("timestamps" in n for n in sim.notes)


def test_legacy_family_is_fail_closed():
    from tokaudit.rules import load_rules

    rules = load_rules().for_provider("openai")
    big_system = "word " * 5000
    reqs = [
        Request(rid=f"old{i}", model="gpt-4o", provider="openai",
                messages=[Message("system", big_system), Message("user", f"q{i}")],
                ts=float(i * 60))
        for i in range(3)
    ]
    sim = simulate(reqs, ENC, rules)
    # No verified rates for pre-GPT-5.6: excluded from savings, not guessed.
    assert sim.per_request == []
    assert len(sim.skipped) == 3
    assert all("no verified rates" in reason for _, _, reason in sim.skipped)
    assert any("excluded from savings simulation" in n for n in sim.notes)




def test_anthropic_unknown_model_excluded():
    from tokaudit.rules import load_rules

    rules = load_rules().for_provider("anthropic")
    reqs = [
        Request(rid=f"a{i}", model="claude-unannounced-9", provider="anthropic",
                messages=[Message("system", SYSTEM), Message("user", f"q{i}")],
                ts=float(i * 60))
        for i in range(2)
    ]
    sim = simulate(reqs, ENC, rules)
    assert sim.per_request == []
    assert len(sim.skipped) == 2
    assert all("not in the anthropic minimums table" in r for _, _, r in sim.skipped)


def test_anthropic_known_model_still_simulates():
    from tokaudit.rules import load_rules

    rules = load_rules().for_provider("anthropic")
    big = "word " * 5000  # above the 4096 strictest minimum
    reqs = [
        Request(rid=f"a{i}", model="claude-opus-4-5-20251101", provider="anthropic",
                messages=[Message("system", big), Message("user", f"q{i}")],
                ts=float(i * 60))
        for i in range(2)
    ]
    sim = simulate(reqs, ENC, rules)
    assert len(sim.per_request) == 2
    assert sim.per_request[1].read > 0


def test_force_family_paths():
    import pytest

    from tokaudit.rules import load_rules

    rules = load_rules().for_provider("openai")
    reqs = [make_req("u1", "q", 0.0), make_req("u2", "q2", 60.0)]
    reqs[0].model = reqs[1].model = "some-unknown-model"

    # Unknown model, no override: excluded from savings (fail closed).
    sim = simulate(reqs, ENC, rules)
    assert sim.per_request == []
    assert len(sim.skipped) == 2

    # Explicit user-chosen hypothetical: verified family forced on unknown model.
    sim2 = simulate(reqs, ENC, rules, force_family="gpt6_verified_01x")
    assert len(sim2.per_request) == 2

    with pytest.raises(SystemExit):
        simulate(reqs, ENC, rules, force_family="nope")
    with pytest.raises(SystemExit):
        simulate(reqs, ENC, rules, force_family="legacy")  # no verified rates
    with pytest.raises(SystemExit):
        # Generic GPT-5.6+ family: base policy documented, per-model read rate not.
        simulate(reqs, ENC, rules, force_family="gpt56_plus_generic")


def test_openai_shipped_rules_refresh_extends_past_first_ttl():
    """End-to-end with the SHIPPED OpenAI rules: write at t=0, read at
    t=1500s (25m, hit refreshes to t=3300s), read at t=2400s (40m) must still
    hit. With hard expiry the entry dies at t=1800s and the third request
    misses. Verifies the 2026-10-31 doc correction end to end."""
    from tokaudit.rules import load_rules

    long_system = "word " * 900  # ~1200 estimator tokens, above the 1024 minimum
    rules = load_rules().for_provider("openai")

    def shipped_req(rid, ts):
        return Request(rid=rid, model="gpt-6-luna", provider="openai",
                       messages=[Message("system", long_system), Message("user", "q")], ts=ts)

    reqs = [shipped_req("w", 0.0), shipped_req("r1", 1500.0), shipped_req("r2", 2400.0)]
    sim = simulate(reqs, ENC, rules)
    assert sim.per_request[1].read > 0
    assert sim.per_request[2].read > 0
