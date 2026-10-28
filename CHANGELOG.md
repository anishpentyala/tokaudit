# Changelog

All notable changes. Dates are commit dates; savings figures are always
simulated estimates against dated provider rules, never observed billing.

## Unreleased

### Added

- `--what-if dense|dedup` scenarios: non-mutating hypothetical simulations
  reported next to the as-is result. `dense` is a hypothetical compressed-timing
  scenario (timestamps rescaled so nothing expires; the never-worse
  property is proven only for the simulator's model, and the output says
  so - no real-world bound, no rescheduling-safety claim). `dedup`
  collapses exact duplicate requests across every distinction the log
  represents (model, prompt, prompt_cache_key, session_id), caveated
  that organization and sampling parameters are not in the log.
  Unknown scenario names fail closed.
- Provider-reported vs simulated usage comparison for Helicone exports
  (schema-verified `prompt_cache_read_tokens` / `prompt_cache_write_tokens`
  / `prompt_tokens` carried through as `actual_usage`; never fed back into
  the simulation). Langfuse/LiteLLM comparison deferred with the grounding
  gap documented rather than guessed.
- `--prices SHEET.json`: optional user-supplied price sheet converting
  simulated units to dollar figures (per-1M rates for input/cached_input/
  cache_write). No prices ship with the tool; sheets fail closed on
  unknown keys, missing or negative rates; models absent from the sheet
  are excluded from dollar totals. Labeled as simulated, not observed
  billing. Also available via `.tokaudit.toml`.
- `.tokaudit.toml` config file for `analyze` defaults (auto-discovered in
  the current directory or via `--config`); CLI flags win; unknown keys
  and bad values fail closed. Needs tomllib (Python 3.11+) or `tomli`.
- `--what-if anthropic-1h`: re-simulates Anthropic logs with the
  documented 1-hour cache TTL (3600s, writes 2x base input vs 1.25x,
  reads unchanged) to weigh pricier writes against the longer reuse
  window. Anthropic-only (clean error elsewhere); caveats flag that
  refresh-on-read for 1-hour entries is assumed from the default-cache
  documentation.

### Fixed

- **Packaging:** the default provider rules are now bundled inside the
  installed package (`tokaudit/providers.json`, package-data). Previously
  a pip-installed `tokaudit` looked for `rules/providers.json` relative to
  the source tree and failed with "rules file not found" outside a
  checkout. A sync test pins the bundled copy to the repo copy.
  `tomli>=2` is now a runtime dependency on Python < 3.11 so
  `.tokaudit.toml` works out of the box there; without it (or without
  tomllib) config use exits with a clean message, not a traceback.
- Custom rules files (`--rules`) now fail closed with clean errors:
  missing file, invalid JSON, wrong top-level shape, non-object provider
  entries, non-numeric fields, and negative TTL/multiplier values are all
  rejected with the provider named, instead of leaking tracebacks or
  simulating with nonsense rates.

### Changed (behavioral, estimates can differ from v0.2.0) (behavioral, estimates can differ from v0.2.0)

- **Cache simulation namespaces corrected to documented provider scope.**
  v0.2.0 shared one cache across all requests in a log. Simulation now
  partitions entries by model and by a record's `prompt_cache_key`,
  matching OpenAI's documented routing ("caches are not shared across
  organizations... routing for a given model", 2026-10-28). One shared
  organization + region is assumed because logs carry no such fields, so
  mixed-account logs can still overstate reuse. New `--isolate-sessions`
  flag partitions by `session_id` as an explicit conservative scenario
  (clearly labeled: not provider-documented behavior). Identical logs can
  produce different (usually lower) simulated savings than v0.2.0.

### Added

- Model matching accepts exact documented model identifiers only; dated
  snapshots and other suffixed variants do NOT inherit rates (no documented
  provider guarantee), so they fail closed like any unknown model unless the
  user explicitly forces a family with --policy. Verified per model on
  2026-10-28 via the prompt-caching guide and model comparison list prices:
  GPT-6.1 Sol/Terra read 0.05x, GPT-6 Astra/Luna read 0.1x, all at 1.25x
  writes. The generic
  GPT-5.6+ family keeps the documented base policy (1024-token minimum,
  30-minute default lifetime, 1.25x writes) but is excluded from savings
  because per-model read rates are unverified; unknown variants fail closed.
  Rates remain list-price-derived simulation parameters; savings stay
  simulated, never observed billing.

### Fixed

- Langfuse adapter: detection no longer requires `input` (it lives in the
  optional `io` field group); GENERATION-only with loud rejection of
  SPAN/EVENT/etc.; missing io group raises a re-export hint; raw-string
  inputs are JSON-recovered to messages; `sessionId` is mapped. Validated
  against the official Fern schema (ObservationV2), 2026-10-28.
- Helicone adapter: parses the real FLAT `HeliconeRequest` row
  (`request_id`, `request_created_at`, `request_body`, `response_body`,
  `provider`, `request_model`) instead of an assumed nested wrapper;
  string `request_body` is JSON-parsed; missing messages/prompt is a loud
  error. Validated against packages/llm-mapper/types.ts, 2026-10-28.

## v0.2.0 - 2026-10-28

- Adapters for LiteLLM SpendLogs, Langfuse observations, Helicone rows.
- Multi-turn conversation simulation via full-history prefix matching.
- Fail-closed savings: unknown/assumed policy families (incl. Anthropic
  unknown models) are excluded from savings but still token/dedup-analyzed.
- Policy families with per-model minimum cacheable lengths.

## v0.1.3 and earlier - 2026-10-28

- Initial audit engine: token waste, duplicate detection, prompt-caching
  savings simulation with dated, sourced provider rules; fail-closed rules
  hardening; OpenAI and Anthropic provider families from official docs.
