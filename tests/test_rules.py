from tokaudit.rules import load_rules


def test_rules_load_and_date():
    rs = load_rules()
    assert rs.as_of == "2026-10-28"
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
