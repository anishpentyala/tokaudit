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
