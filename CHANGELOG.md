# Changelog

## Unreleased

### Added

- `tokaudit integrity`: offline accounting-integrity checks over the
  supported log formats - consistency checks with stated schema
  assumptions (zero spend on token-bearing non-error calls,
  spend/cost_breakdown disagreement, usage-sum mismatch, duplicate-billed
  request IDs, provider-reported cost fields in usage chunks, non-numeric
  accounting fields), each finding anchored to the documented upstream
  failure class it matches and naming the legitimate alternatives
  (zero-priced models, cached or discounted calls, retry re-emissions,
  provider cost-field contracts) that the record cannot rule out.
  cost_breakdown and usage_object are read from both the top level and
  the SpendLogs metadata JSON (where real rows nest them), and the
  gateway's own zero_cost_diagnostic field is surfaced as a finding when
  a record carries it.
  Documented legitimate patterns stay quiet: cache-inclusive usage sums
  are not flagged when the record's cache fields exactly explain the gap,
  and OpenRouter's documented usage cost field is not flagged on records
  that name OpenRouter as provider.
  (litellm #48937/#40100/#35691-class, #35829, #25351; langfuse #14987)
  with a link, never asserted as proof of misbilling.
  `--fail-on-findings` for CI gating. 16 tests with synthetic fixtures.

### Fixed

- Benchmark corpus seeding: `random.Random((seed, category))` seeded
  with a tuple. Before Python 3.11 any hashable object was accepted and
  hashed; the tuple's hash includes the salted string hash of the
  category, so the "deterministic" corpus differed between processes
  (reproduced: PYTHONHASHSEED=1 vs 999 give different streams on 3.10)
  and the old README table measured one arbitrary process's corpus.
  Python 3.11 restricts seed types to None/int/float/str/bytes/
  bytearray (CPython docs, `random.Random`: "Formerly the seed could be
  any hashable object"), so the same code raises TypeError there. Now
  seeded with a string, stable across processes and versions; the
  corpus content changes as a result, so the README's measured error
  table and the tokenizers docstring figures are regenerated
  (prose/code/JSON mean abs err now 17.5%/14.8%/20.1%). Cross-process
  determinism is pinned by a subprocess regression test.

- Fail-closed logging parity: six malformed-input classes that crashed
  with an uncaught traceback now exit cleanly with file:line context -
  a non-list `messages` field, non-integer `actual_usage` values,
  non-UTF-8 log files, JSON nested beyond the parser's recursion limit,
  unreadable/missing log paths, and non-finite (`NaN`/`Infinity`)
  reported-usage numbers in Helicone rows (now skipped like the other
  adapters' malformed values). Non-finite timestamps are treated as
  missing (conservative ordering) instead of silently poisoning TTL
  comparisons.

### Changed

- CI: added the 3.13 leg. The 3.9 leg is pinned to the ubuntu-24.04
  runner: actions/python-versions publishes 3.9 builds only for linux
  22.04/24.04, none for Ubuntu 26.04 (which ubuntu-latest now resolves
  to), so setup-python failed there with "The version '3.9' with
  architecture 'x64' was not found for Ubuntu 26.04". 3.10-3.13 have
  26.04 builds and run on ubuntu-latest. Python 3.9 support is
  retained (`requires-python` stays >=3.9).

### Added

- `tests/test_robustness.py`: hostile-input sweeps - record-mutation
  and random-structure fuzz over the log parser (SystemExit or success,
  never a traceback; 500 random records), estimator crash-resistance
  over unicode soup incl. lone surrogates/control chars (2000 samples),
  huge-input completion, count monotonicity at word boundaries (2000
  random texts), and a pinned characterization of the known mid-word
  cut anomaly (documented in the tokenizers docstring).

All notable changes. Dates are commit dates; savings figures are always
simulated estimates against dated provider rules, never observed billing.

## Unreleased

### Added

- `tokaudit.benchmark --file PATH`: measure the estimator against
  tiktoken on your own UTF-8 text (chunked at configurable sizes),
  so estimate-mode error can be checked on real logs before trusting
  it. Fails closed on missing, empty, undecodable, or too-short input.
- `tokaudit.benchmark`: deterministic synthetic corpus benchmark
  (`python -m tokaudit.benchmark`) measuring EstimatorBackend against
  tiktoken o200k_base per content category (prose/code/JSON) with
  mean/median/p90/max absolute error. Seeded template generation, no
  network, fails closed on empty categories or samples. Results are
  labeled synthetic; they describe the estimator on this corpus only.

### Changed

- EstimatorBackend reworked from fixed 4-character blocks to per-word
  scoring in thirds of a pseudo-token: short natural-language words
  (85%+ letters, <=20 chars) score 4/3 (the standard ~0.75 words/token
  English heuristic); symbol-heavy words score ~2/3 token per structural
  character ({}[]",:) plus ~1 token per 4 remaining characters;
  dense-script characters (CJK, kana, Hangul, Thai, Khmer, Myanmar)
  score ~2/3 token each; emoji 1 each; long alphanumeric ID/hash runs
  ~1 token per 2 characters. Exact prefix semantics are now documented
  and property-tested: exact prefix matching holds at completed-word
  boundaries only; mid-word cuts share up to the last completed word.
  Measured on the synthetic benchmark corpus: prose 17.6%, code 15.8%,
  JSON 20.4% mean absolute error. Held-out validation on independently
  sourced texts (Gutenberg public-domain prose, cpython PSF-2.0 code,
  GitHub public API JSON, unicode.org UDHR translations) is tabulated
  in the README, including the remaining weak spots (Vietnamese,
  Russian/Arabic, Thai, repetitive text).

- `tokaudit.benchmark --file PATH`: measure the estimator against
  tiktoken on your own UTF-8 text (chunked at configurable sizes),
  so estimate-mode error can be checked on real logs before trusting
  it. Fails closed on missing, empty, undecodable, or too-short input.
- `tokaudit.benchmark`: deterministic synthetic corpus benchmark
  (`python -m tokaudit.benchmark`) measuring EstimatorBackend against
  tiktoken o200k_base per content category (prose/code/JSON) with
  mean/median/p90/max absolute error. Seeded template generation, no
  network, fails closed on empty categories or samples. Results are
  labeled synthetic; they describe the estimator on this corpus only.

### Changed

- EstimatorBackend reworked from fixed 4-character blocks to per-word
  scoring in thirds of a pseudo-token: natural-language words (85%+
  ASCII letters) score 4/3 (the standard ~0.75 words/token English
  heuristic); symbol-heavy words score ~2/3 token per structural
  character ({}[]",:) plus ~1 token per 4 remaining characters.
  Prefix stability preserved up to boundary-word rounding; count() ==
  len(encode()) still holds. Measured mean absolute error on the
  synthetic benchmark corpus: prose 18.1% (was ~48% on short texts),
  code 16.2%, JSON 19.4%.

- `--what-if dense|dedup` scenarios: non-mutating hypothetical simulations
  reported next to the as-is result. `dense` is a hypothetical compressed-timing
  scenario (timestamps rescaled so nothing expires; the never-worse
  property is proven only for the simulator's model, and the output says
  so - no real-world bound, no rescheduling-safety claim). `dedup`
  collapses exact duplicate requests across every distinction the log
  represents (model, prompt, prompt_cache_key, session_id), caveated
  that organization and sampling parameters are not in the log.
  Unknown scenario names fail closed.
- `--what-if openai-12h`: documented OpenAI 12-hour cache option
  (TTL 43200s, 2.5x writes vs 1.25x default, reads unchanged) as a
  non-mutating scenario, mirroring `anthropic-1h`.
- Provider-reported vs simulated usage comparison for Helicone exports
  (schema-verified `prompt_cache_read_tokens` / `prompt_cache_write_tokens`
  / `prompt_tokens` carried through as `actual_usage`; never fed back into
  the simulation). Extended to Langfuse `usageDetails` using the key
  families pinned by Langfuse's provider-usage-key matrix (aliases
  de-duplicated by max, TTL-specific write keys included). LiteLLM
  SpendLogs compare typed `prompt_tokens` only; cache comparison stays
  closed (`cache_hit` is LiteLLM's application-level cache flag, not
  provider prompt-cache usage).
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
