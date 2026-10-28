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
- [x] `.tokaudit.toml` config file support (fail-closed validation).
- [ ] HTML report with per-request table. (on hold: no concrete need)
- [ ] `tokaudit watch` mode for a growing log file. (on hold: no concrete need)
- [ ] `--prices SHEET.json`: convert units to dollars from a user-supplied
      price sheet (model -> {input, cached_input, cache_write} per 1M).
      Evaluated 2026-10-28: accepted for next session. No prices ship in
      the tool; the sheet is user data, validated fail-closed.

## Evaluated and closed
- Anthropic 1-hour cache cost-benefit modeling -> shipped as
  `--what-if anthropic-1h` (documented rates, caveated assumptions).
- GitHub Action that audits a fixture log on PRs -> redundant: CI already
  runs `analyze` over every fixture via pytest on each push.

## Explicitly out of scope
- Rewriting or compressing prompts (changes model behavior; other tools
  do this, e.g. LLMLingua).
- Chasing download/star counts. Usefulness first.
