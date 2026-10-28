# Roadmap

Prioritized next steps. Each item should ship with tests.

## v0.2 - core depth
- [x] QUALITY PASS (main, 2026-10-28): Anthropic unknown models excluded
      from savings (done in v0.2.0).
- [x] Parse real proxy/gateway export formats (LiteLLM, Helicone, Langfuse
      traces) in addition to raw JSONL. (v0.2.0 adapters)
- [ ] Per-model minimum cacheable lengths for OpenAI models (model
      comparison data, sourced and dated like the Anthropic map).
- [x] Multi-turn conversation simulation via full-history prefix matching
      (tests incl. adversarial volatile-prefix case, v0.2.0).
- [ ] Group simulation by session_id so unrelated sessions don't share a
      cache namespace.
- [ ] `--what-if`: re-simulate with volatile fields (timestamps, request
      IDs) stripped from prefixes to show the upside of reordering.

## v0.3 - usability
- [ ] HTML report with per-request table.
- [ ] `tokaudit watch` mode for a growing log file.
- [ ] Config file support (`.tokaudit.toml`) for rules overrides.

## Ideas to evaluate (not committed)
- Anthropic 1-hour cache cost-benefit modeling.
- Cost estimation in dollars when the user supplies a price sheet
  (never ship dollar prices in the tool; they go stale).
- GitHub Action that audits a fixture log on PRs.

## Explicitly out of scope
- Rewriting or compressing prompts (changes model behavior; other tools
  do this, e.g. LLMLingua).
- Chasing download/star counts. Usefulness first.
