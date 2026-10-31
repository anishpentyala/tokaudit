from tokaudit.rules import load_rules


def test_rules_load_and_date():
    rs = load_rules()
    assert rs.as_of == "2026-10-31"
    assert "openai" in rs.sources and "anthropic" in rs.sources


def test_anthropic_model_minimums():
    rules = load_rules().for_provider("anthropic")
    assert rules.resolve_min_prefix("claude-opus-4-5-20251101") == 4096
    assert rules.resolve_min_prefix("claude-opus-4-7") == 2048
    assert rules.resolve_min_prefix("claude-fable-5-1") == 512
    assert rules.resolve_min_prefix("claude-unknown-9") == 4096  # strictest known, conservative


def test_generic_falls_back_to_openai_rules():
    rs = load_rules()
    assert rs.for_provider("generic").name == "openai"


def test_gpt61_reduced_read_family_wins_over_gpt6_prefix():
    # "gpt-6.1" must resolve to the 0.05x-read family, not the gpt-6 catch-all.
    rules = load_rules().for_provider("openai")
    fam = rules.resolve_family("gpt-6.1-sol").family
    assert fam.read_multiplier == 0.05
    assert fam.write_multiplier == 1.25
    assert not rules.resolve_family("gpt-6.1-sol").assumed


def test_other_gpt6_models_keep_point_one_read_rate():
    rules = load_rules().for_provider("openai")
    for model in ("gpt-6-astra", "gpt-6-luna", "gpt-6"):
        fam = rules.resolve_family(model).family
        assert fam.read_multiplier == 0.1, model


def test_gpt61_reduced_read_family_wins_over_gpt6_prefix():
    # Sol/Terra resolve to the 0.05x-read family, not the generic GPT-6 family.
    rules = load_rules().for_provider("openai")
    for model in ("gpt-6.1-sol", "gpt-6.1-terra"):
        fam = rules.resolve_family(model).family
        assert fam.read_multiplier == 0.05, model
        assert not rules.resolve_family(model).assumed


def test_astra_luna_verified_at_point_one_read_rate():
    rules = load_rules().for_provider("openai")
    for model in ("gpt-6-astra", "gpt-6-luna"):
        fam = rules.resolve_family(model).family
        assert fam.read_multiplier == 0.1, model
        assert not rules.resolve_family(model).assumed


def test_unknown_model_variants_fail_closed():
    # Unverified variants must NOT inherit rates from nearby verified names.
    rules = load_rules().for_provider("openai")
    unknowns = (
        "gpt-6.1-mini",       # near the Sol/Terra exception but unverified
        "gpt-6.1",            # bare version, not a documented alias
        "gpt-6-mini",         # near Astra/Luna but unverified
        "gpt-6",              # bare generation name
        "gpt-5.6",            # generic family: base policy only, read rate unverified
        "gpt-7",              # future generation
        "gpt-6-astral",       # typo-squatting the Astra alias
        "gpt-6-astra-2027-01-01",  # dated snapshot: no documented rate inheritance
        "gpt-6.1-sol-2026-10-01",  # same: even a real-looking snapshot fails closed
        "o4-mini",            # legacy family
    )
    for model in unknowns:
        res = rules.resolve_family(model)
        assert res.family.savings_supported is False, model


def test_openai_refresh_on_read_per_current_doc():
    """OpenAI's prompt-caching guide (fetched 2026-10-31): a 30-minute read
    refreshes coverage to at least 30 minutes after that read, so the shipped
    rules must refresh on read. See rules/EVIDENCE.md for the verbatim excerpt."""
    rules = load_rules().for_provider("openai")
    assert rules.refresh_on_read is True


def test_shipped_openai_read_refresh_is_thirty_minutes():
    assert load_rules().for_provider("openai").read_refresh_seconds == 1800
    assert load_rules().for_provider("anthropic").read_refresh_seconds is None


def test_negative_read_refresh_is_rejected(tmp_path):
    import json
    import pytest
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"providers": {"x": {"read_refresh_seconds": -1}}}))
    with pytest.raises(SystemExit, match="read_refresh_seconds must be non-negative"):
        load_rules(str(path))
