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
