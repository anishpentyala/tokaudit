import json

import pytest

from tokaudit.cli import main
from tokaudit.prefix import simulate
from tokaudit.whatif import run_scenarios

from tokaudit.logs import Message, Request
from tokaudit.rules import ProviderRules
from tokaudit.tokenizers import EstimatorBackend

ENC = EstimatorBackend().encode
OPENAI = ProviderRules(
    name="openai", min_prefix_tokens=32, ttl_seconds=1800,
    write_multiplier=1.25, read_multiplier=0.1,
    refresh_on_read=False, enabled_by_default=True,
)
SYSTEM = "word " * 600


def make_req(rid, user_text, ts):
    return Request(rid=rid, model="gpt-test", provider="openai",
                   messages=[Message("system", SYSTEM), Message("user", user_text)], ts=ts)


def test_dense_compressed_timing_recovers_expired_hits():
    # Two requests sharing a long prefix, arriving 2h apart (TTL is 1800s):
    # the second misses as-is but would hit if traffic were dense.
    reqs = [make_req("a", "first question", ts=0.0),
            make_req("b", "second question", ts=7200.0)]
    asis = simulate(reqs, ENC, OPENAI)
    assert asis.per_request[1].read == 0  # expired

    (sc,) = run_scenarios(["dense"], reqs, ENC, OPENAI)
    assert sc["result"]["simulated_units"] < asis.simulated_units
    assert any("rescaled" in n for n in sc["notes"])
    # Non-mutating: the as-is simulation is untouched.
    asis2 = simulate(reqs, ENC, OPENAI)
    assert asis2.simulated_units == asis.simulated_units


def test_dense_is_ceiling_never_worse():
    reqs = [make_req(f"r{i}", f"q{i}", ts=float(i * 2400)) for i in range(5)]
    asis = simulate(reqs, ENC, OPENAI)
    (sc,) = run_scenarios(["dense"], reqs, ENC, OPENAI)
    assert sc["result"]["simulated_units"] <= asis.simulated_units


def test_dense_noop_when_already_dense():
    reqs = [make_req("a", "q1", ts=0.0), make_req("b", "q2", ts=60.0)]
    (sc,) = run_scenarios(["dense"], reqs, ENC, OPENAI)
    asis = simulate(reqs, ENC, OPENAI)
    assert sc["result"]["simulated_units"] == round(asis.simulated_units, 1)
    assert any("already fits" in n for n in sc["notes"])


def test_dense_noop_without_timestamps():
    reqs = [make_req("a", "q1", ts=None), make_req("b", "q2", ts=None)]
    (sc,) = run_scenarios(["dense"], reqs, ENC, OPENAI)
    assert any("without timestamps" in n for n in sc["notes"])


def test_dense_caveats_scoped_to_simulator_model():
    reqs = [make_req("a", "q1", ts=0.0), make_req("b", "q2", ts=7200.0)]
    (sc,) = run_scenarios(["dense"], reqs, ENC, OPENAI)
    joined = " ".join(sc["caveats"])
    assert "not a real-world" in joined  # bound claim scoped to the model
    assert "does not claim" in joined    # no rescheduling-safety implication


def test_dedup_collapses_only_exact_duplicates():
    reqs = [make_req("a", "same question", ts=0.0),
            make_req("b", "same question", ts=10.0),   # exact duplicate
            make_req("c", "same question!", ts=20.0)]  # near-duplicate
    asis = simulate(reqs, ENC, OPENAI)
    (sc,) = run_scenarios(["dedup"], reqs, ENC, OPENAI)
    assert any("b" in n and "1 exact duplicate" in n for n in sc["notes"])
    # Baseline drops by exactly the duplicate's token count.
    dropped_tokens = asis.per_request[1].total_input
    assert sc["result"]["baseline_units"] == round(asis.baseline_units - dropped_tokens, 1)


def test_dedup_noop_without_duplicates():
    reqs = [make_req("a", "q1", ts=0.0), make_req("b", "q2", ts=10.0)]
    (sc,) = run_scenarios(["dedup"], reqs, ENC, OPENAI)
    assert any("No exact duplicate" in n for n in sc["notes"])


def test_dedup_preserves_cache_key_and_session_distinctions():
    base = dict(rid="x", user_text="same", ts=0.0)
    reqs = [
        make_req("a", "same", ts=0.0),
        make_req("b", "same", ts=1.0),
    ]
    # Identical prompt but different cache keys / sessions: not duplicates.
    import tokaudit.logs as L
    k1 = L.Request(rid="k1", model="gpt-test", provider="openai",
                   messages=[L.Message("system", SYSTEM), L.Message("user", "same")],
                   ts=2.0, cache_key="one")
    k2 = L.Request(rid="k2", model="gpt-test", provider="openai",
                   messages=[L.Message("system", SYSTEM), L.Message("user", "same")],
                   ts=3.0, cache_key="two")
    s1 = L.Request(rid="s1", model="gpt-test", provider="openai",
                   messages=[L.Message("system", SYSTEM), L.Message("user", "same")],
                   ts=4.0, session_id="sess")
    (sc,) = run_scenarios(["dedup"], reqs + [k1, k2, s1], ENC, OPENAI)
    # Only "b" collapses (exact twin of "a"); k1/k2/s1 are kept distinct.
    assert any("1 exact duplicate" in n and "b" in n for n in sc["notes"])


def test_dedup_caveat_flags_sampling_params():
    reqs = [make_req("a", "q", ts=0.0), make_req("b", "q", ts=1.0)]
    (sc,) = run_scenarios(["dedup"], reqs, ENC, OPENAI)
    joined = " ".join(sc["caveats"]).lower()
    assert "sampling" in joined and "organization" in joined
    assert any("Non-mutating" in c for c in sc["caveats"])


def test_unknown_scenario_fails_closed():
    with pytest.raises(SystemExit):
        run_scenarios(["teleport"], [make_req("a", "q", ts=0.0)], ENC, OPENAI)


def test_cli_what_if_json_and_markdown(tmp_path, capsys):
    import os
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_log.jsonl")
    rc = main(["analyze", fixture, "--tokenizer", "estimate", "--format", "json"])
    assert rc == 0
    base = json.loads(capsys.readouterr().out)

    rc = main(["analyze", fixture, "--tokenizer", "estimate", "--format", "json",
               "--what-if", "dense", "--what-if", "dedup"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    groups = data["groups"] if "groups" in data else [data]
    for g in groups:
        assert [s["name"] for s in g["what_if"]] == ["dense", "dedup"]
        for s in g["what_if"]:
            assert s["caveats"], "every scenario must carry caveats"
    # The as-is simulation is unchanged by adding scenarios.
    base_groups = base["groups"] if "groups" in base else [base]
    for b, g in zip(base_groups, groups):
        assert b["summary"]["simulated_units"] == g["summary"]["simulated_units"]

    out = tmp_path / "r.md"
    rc = main(["analyze", fixture, "--tokenizer", "estimate",
               "--what-if", "dense", "--output", str(out)])
    assert rc == 0
    assert "What-if scenarios (hypothetical, non-mutating)" in out.read_text()


ANTH = ProviderRules(
    name="anthropic", min_prefix_tokens=32, ttl_seconds=300,
    write_multiplier=1.25, read_multiplier=0.1,
    refresh_on_read=True, enabled_by_default=False,
)


def make_anth(rid, user_text, ts):
    from tokaudit.logs import Message, Request
    return Request(rid=rid, model="claude-test", provider="anthropic",
                   messages=[Message("system", SYSTEM), Message("user", user_text)], ts=ts)


def test_anthropic_1h_recovers_hits_between_5_and_60_min():
    # Reuse 10 minutes after the write: expired under the default 5-minute
    # cache, alive under the documented 1-hour option.
    reqs = [make_anth("w", "q1", ts=0.0), make_anth("r", "q2", ts=600.0)]
    asis = simulate(reqs, ENC, ANTH)
    assert asis.per_request[1].read == 0
    (sc,) = run_scenarios(["anthropic-1h"], reqs, ENC, ANTH)
    assert sc["result"]["simulated_units"] != round(asis.simulated_units, 1)
    assert any("2x base input" in n for n in sc["notes"])


def test_anthropic_1h_charges_pricier_writes():
    # Single-use prefix: 1h writes cost strictly more (2x vs 1.25x).
    reqs = [make_anth("w", "q1", ts=0.0)]
    asis = simulate(reqs, ENC, ANTH)
    (sc,) = run_scenarios(["anthropic-1h"], reqs, ENC, ANTH)
    assert sc["result"]["simulated_units"] > round(asis.simulated_units, 1)


def test_anthropic_1h_rejected_for_other_providers():
    reqs = [make_req("a", "q1", ts=0.0)]
    with pytest.raises(SystemExit) as e:
        run_scenarios(["anthropic-1h"], reqs, ENC, OPENAI)
    assert "Anthropic rules only" in str(e.value)


def test_anthropic_1h_caveats_flag_refresh_assumption():
    reqs = [make_anth("w", "q1", ts=0.0)]
    (sc,) = run_scenarios(["anthropic-1h"], reqs, ENC, ANTH)
    joined = " ".join(sc["caveats"])
    assert "refresh" in joined.lower() and "Non-mutating" in joined


def test_openai_12h_recovers_hits_between_30m_and_12h():
    # Reuse 2 hours after the write: expired under the default 30-minute
    # cache, alive under the documented 12-hour option.
    reqs = [make_req("w", "q1", ts=0.0), make_req("r", "q2", ts=7200.0)]
    asis = simulate(reqs, ENC, OPENAI)
    assert asis.per_request[1].read == 0
    (sc,) = run_scenarios(["openai-12h"], reqs, ENC, OPENAI)
    assert sc["result"]["simulated_units"] != round(asis.simulated_units, 1)
    assert any("2.5x ordinary input" in n for n in sc["notes"])


def test_openai_12h_charges_pricier_writes():
    # Single-use prefix: 12h writes cost strictly more (2.5x vs 1.25x).
    reqs = [make_req("w", "q1", ts=0.0)]
    asis = simulate(reqs, ENC, OPENAI)
    (sc,) = run_scenarios(["openai-12h"], reqs, ENC, OPENAI)
    assert sc["result"]["simulated_units"] > round(asis.simulated_units, 1)


def test_openai_12h_rejected_for_other_providers():
    from tokaudit.whatif import SCENARIOS
    assert "openai-12h" in SCENARIOS
    with pytest.raises(SystemExit) as e:
        run_scenarios(["openai-12h"], [make_anth("w", "q1", ts=0.0)], ENC, ANTH)
    assert "OpenAI rules only" in str(e.value)


def test_openai_12h_caveats_flag_retention_assumption():
    reqs = [make_req("w", "q1", ts=0.0)]
    (sc,) = run_scenarios(["openai-12h"], reqs, ENC, OPENAI)
    joined = " ".join(sc["caveats"])
    assert "at least" in joined and "Non-mutating" in joined


def test_openai_12h_read_preserves_write_expiry_without_restarting_clock():
    from dataclasses import replace
    # 11h read preserves the original 12h coverage, not a new 12h clock.
    rules = replace(OPENAI, refresh_on_read=True)
    reqs = [make_req(str(i), "same", ts=t * 3600) for i, t in enumerate([0, 11, 13])]
    (scenario,) = run_scenarios(["openai-12h"], reqs, ENC, rules)
    # Write, one discounted read, then another write after expiry.
    token_count = simulate(reqs[:1], ENC, rules).per_request[0].total_input
    assert scenario["result"]["simulated_units"] == round(token_count * (2.5 + 0.1 + 2.5), 1)



def test_openai_12h_read_near_expiry_extends_only_thirty_minutes():
    from dataclasses import replace
    rules = replace(OPENAI, refresh_on_read=True, ttl_seconds=43200,
                    read_refresh_seconds=1800)
    reqs = [make_req(str(i), "same", ts=t * 60) for i, t in enumerate([0, 710, 730, 761])]
    costs = simulate(reqs, ENC, rules).per_request
    assert costs[1].read > 0  # keep coverage through 740 minutes
    assert costs[2].read > 0  # extend coverage to 760, not another 12h
    assert costs[3].read == 0


def test_openai_read_cannot_shorten_long_write_coverage():
    from dataclasses import replace
    rules = replace(OPENAI, refresh_on_read=True, ttl_seconds=43200,
                    read_refresh_seconds=1800)
    reqs = [make_req(str(i), "same", ts=t * 3600) for i, t in enumerate([0, 1, 11])]
    assert all(c.read > 0 for c in simulate(reqs, ENC, rules).per_request[1:])
