# Roadmap

Prioritized next steps. Each item should ship with tests.

## v0.2 - core depth
- [x] QUALITY PASS (main, 2026-10-28): Anthropic unknown models excluded
      from savings (done in v0.2.0).
- [x] Parse real proxy/gateway export formats (LiteLLM, Helicone, Langfuse
      traces) in addition to raw JSONL. (v0.2.0 adapters)
- [x] Per-model minimum cacheable lengths for OpenAI models: closed as
      no-variation. Doc (2026-10-28) states a uniform 1,024-token minimum
      for GPT-5.6 and later; the compare page lists no per-model minimums.
      Earlier models vary by request settings and stay fail-closed
      (rules/EVIDENCE.md).
- [x] Multi-turn conversation simulation via full-history prefix matching
      (tests incl. adversarial volatile-prefix case, v0.2.0).
- [x] Cache namespaces grounded in documented provider scope (model +
      prompt_cache_key); `--isolate-sessions` ships as a labeled
      conservative scenario, not provider semantics (v0.2.x).
- [x] `--what-if` ships as non-mutating `dense` (compressed timing) and
      `dedup` (exact-duplicate collapse) scenarios. The original
      volatile-field-stripping variant is dropped: it would imply
      reordered prompts preserve semantics/quality, which we cannot
      evidence.

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
