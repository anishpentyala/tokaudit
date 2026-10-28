# Work log

Factual record of work sessions. Newest first.

## 2026-10-28 (session 2, ~08:19 PDT, Instinct assistant) - v0.2.0
- Direction change from Anish via main: 70% daily usage is a TARGET for
  meaningful work on the GitHub EC, not just a ceiling. Work sessions
  should be substantive, not one short daily run.
- Adapter grounding: LiteLLM SpendLogs fields verified against current
  litellm/proxy/schema.prisma (model LiteLLM_SpendLogs) fetched today;
  Langfuse Observations API v2 and Helicone /v1/request/query endpoint
  shapes verified at endpoint level; field drift fails loudly, never
  silently.
- Built: adapters.py (litellm/langfuse/helicone normalization + strict
  unknown-shape errors), session_id capture, line-numbered JSON errors,
  Anthropic unknown-model exclusion (main's quality pass), multi-turn
  simulation tests (growing history hits >40% savings; volatile timestamp
  inside a prefix destroys all cache hits - the key adversarial case),
  LiteLLM-shaped end-to-end fixture run.
- 42 tests passing. Ran CLI on LiteLLM fixture: honest 0% (prompts below
  the 1024 minimum) with correct duplicate and hoist recommendations.
- Note: discovered test_force_family_and_unknown_model_fail_closed had
  been silently dropped in last night's patch; re-added as
  test_force_family_paths. Lesson: grep test names after bulk edits.

## 2026-10-28 (session 1d, ~00:27 PDT, Instinct assistant) - fail-closed release blocker
- Main ruled: no guessed billing numbers, no provider defaults on unknown
  models, even labeled. Removed the unverified legacy read-discount (0.5x)
  and TTL (600s) entirely.
- Behavior now: pre-GPT-5.6 OpenAI models and any unrecognized model are
  EXCLUDED from savings simulation (listed in the report with reasons);
  token counts and duplicate analysis still cover them. --policy
  gpt56_plus is the explicit-hypothetical escape hatch; forcing the
  rate-less legacy family errors out and asks for --rules with verified
  rates. Custom rules files count as explicit user parameters and run.
- Anthropic unknown models assume the strictest known minimum (4096),
  so savings can only be understated.
- 28 tests passing, including fail-closed paths. Release blocker cleared
  pending main's confirmation.

## 2026-10-28 (session 1c, ~00:26 PDT, Instinct assistant) - policy families
- Main's requirement implemented: models resolve to dated policy families
  instead of GPT-5.6+ defaults applying to every log. openai now has
  gpt56_plus (verified) and legacy (pre-GPT-5.6, no write charge per
  official model-comparison table). Unrecognized models fall back to
  provider defaults and are flagged "assumed" in the report; --policy
  forces a family.
- Legacy read discount (0.5x) and TTL (600s) are explicitly labeled
  ASSUMPTION in rules notes + report: the live pricing page is
  JS-rendered and could not be verified today. Only unverified numbers
  in the tool; both overridable.
- 3 new tests (legacy no-write-charge, forced family, unknown-model
  note). 29 passing.

## 2026-10-28 (session 1b, ~00:25 PDT, Instinct assistant) - correctness review response
- Re-verified every shipped provider rule against the official pages
  fetched today; saved verbatim dated excerpts in rules/EVIDENCE.md.
  Reviewer suspicion of cross-provider conflation resolved: OpenAI's
  GPT-5.6+ terms genuinely include 1.25x writes / 30-min minimum
  lifetime / 0.1x reads (new-style terms; earlier OpenAI models have NO
  write charge - now noted).
- Confirmed delta billing is provider-documented: OpenAI "each input
  token uses exactly one of uncached/cache-write/cached-read rate";
  Anthropic total = cache_read + cache_creation + input tokens.
- New documented caveats added to rules notes + README: OpenAI lifetime
  is a minimum (simulation conservative), 128-token rounding on cached
  reporting, cache-key settings (reasoning.effort, text.verbosity,
  context_management), Anthropic billing breakdown.
- Report labels changed: "Simulated savings (not observed billing)";
  JSON keys renamed simulated_savings_*; new observation_note assumption.
- README: bounded landscape claim (no more "nothing exists" phrasing),
  new Privacy section (local-only, no uploads, redact secrets).
- Added test_rules.py (dated rules, Anthropic per-model minimums,
  generic fallback). 27 tests passing. Committed as v0.1.1 (66b83b2).

## 2026-10-28 (session 1, ~00:20-01:00 PDT, Instinct assistant)
- Researched the landscape: existing token tools are single-prompt
  counters/minifiers (tokensaver, tok, tokentrimmer, tokenshrink) or
  heavy neural compressors (LLMLingua family). No polished workload-level
  prompt-cache savings analyzer found -> differentiation confirmed.
- Verified current provider caching rules from live docs:
  platform.openai.com/docs/guides/prompt-caching and
  docs.claude.com/en/docs/build-with-claude/prompt-caching (both fetched
  2026-10-28; rules with as_of date and sources in rules/providers.json).
- Built v0.1.0: log parsing (OpenAI/Anthropic/generic JSONL, explicit
  cache breakpoints), cache simulation with TTL/refresh/delta-write
  billing, shingle-based duplicate detection, dated+overridable provider
  rules, markdown + JSON reports, CLI (`analyze`, `rules`).
- 24 tests, all passing (parsing, simulation correctness incl. TTL
  expiry, refresh-on-read, wasted writes, dedup, estimator tolerance vs
  tiktoken, CLI end-to-end on a fixture).
- Fixed two real bugs found by tests: prefix-hash misalignment at message
  boundaries and dedup min-length collapsing on periodic text.
- Name check: `tokaudit` free on PyPI (404) as of 2026-10-28.
- Status: private, local only. Not published anywhere. Awaiting Anish's
  review before any repo goes public.

## 2026-10-28 (publish + supervision diagnosis)

- Published tokaudit v0.2.0 to https://github.com/anishpentyala/tokaudit (public). Route: cloud browser; GitHub /upload commit flow crashed the remote tab on every attempt, so files were committed one-by-one via the new-file editor flow (~40 granular commits). Verified by fresh anonymous clone: tree identical to local source (one cosmetic trailing-newline difference in local rules/providers.json), and fresh-clone test run passed 42/42.
- Investigated roboflow/supervision #2674 (NumPy fallback thick rectangle borders) as an external contribution candidate. Diagnosis: already fixed on develop with pixel-equality parity tests; verified locally (158/158 tests/cv2/test_drawing.py pass against opencv-headless 5.0.0). Issue thread shows the remaining fallback-parity primitives claimed/completed by the issue author and another contributor; the recent _cv2 bugs (#2840, #2841) are already open PRs. No unclaimed, non-duplicative work there; dropped.
- v0.2.1 work started: cache simulation is now scoped per session_id (separate namespaces; requests without one share the default namespace as before). 2 new tests (cross-session cold miss, no-session backwards compatibility); 44/44 pass. Not yet published to GitHub.
- Queued (main, via outreach feedback): validate the Langfuse adapter against Langfuse's official published schema (fern API definitions at github.com/langfuse/langfuse fern/apis/server/definition/observations.yml + commons.yml). Early finding to verify next run: the v2 observations API returns input/output as RAW STRINGS (JSON serialized), and `input` is only in the optional `io` field group, so the adapter's detection (`input` + startTime present) and message handling must tolerate string-JSON input and missing input, with loud skips for unsupported shapes. Add synthetic fixtures built from official docs/examples incl. unsupported formats; report evidence and limitations.
- Langfuse adapter validated against the authoritative Fern schema (github.com/langfuse/langfuse fern/apis/server/definition/commons.yml, ObservationV2, fetched 2026-10-28). Fixes: detection no longer requires `input` (it sits in the optional `io` field group; core fields drive detection); GENERATION-only with a loud error for SPAN/EVENT/etc.; missing io group is a loud re-export hint; raw-string inputs are JSON-recovered to messages when structured (v2 returns input/output as raw strings); sessionId now mapped. New synthetic schema-grounded tests (raw-string JSON input, plain prompt, SPAN rejection, missing-io rejection, session mapping) plus an end-to-end JSONL fixture (tests/fixtures/langfuse_observations_v2.jsonl). 52/52 pass.
- Helicone adapter validated against the authoritative HeliconeRequest type (github.com/Helicone/helicone packages/llm-mapper/types.ts, fetched 2026-10-28). Found and fixed a real shape error: rows are FLAT (request_id, request_created_at, request_body, response_body, provider, request_model), not the previously assumed nested request/response wrapper. request_body as a JSON string is handled; missing messages/prompt is a loud error. Schema-grounded synthetic tests added. 55/55 pass.
- OpenAI per-model rates verified via browser render of the JS-only model comparison page (2026-10-28): GPT-6 Astra 0.1x read / 1.25x write, GPT-6.1 Sol 0.05x / 1.25x, GPT-6 Luna 0.1x / 1.25x. Added gpt61_reduced_read family (match "gpt-6.1", resolves before the gpt-6 prefix catch-all, order pinned by tests). Legacy pre-GPT-5.6 family stays fail-closed (page lists current models only). 57/57 pass.
- Tightened per main: model matching is now exact-alias (m == alias or alias + dash-suffix); verified read rates only for named models (Sol/Terra 0.05x, Astra/Luna 0.1x); generic GPT-5.6+ family keeps documented base policy but is excluded from savings (per-model read rate unverified); unknown variants (gpt-6.1-mini, gpt-6-mini, gpt-7, "gpt-6-astral" typo-squat, gpt-5.6 itself) fail closed, pinned by adversarial tests. Fixed a real alias-boundary bug the adversarial test caught ("gpt-6-astral" substring-matched "gpt-6-astra"). 59/59 pass.
- Final tightening per main: matching is exact documented identifiers only; dated snapshots and suffixed variants fail closed (no documented rate inheritance) unless the user forces a family with --policy. Tests moved accordingly (dated variants now assert fail-closed). README accuracy pass: --policy example uses a supported family, documents --isolate-sessions, exact-match policy, cache namespaces and the single org+region assumption. 59/59 pass.
- Added --what-if dense|dedup scenarios (whatif.py): non-mutating hypothetical simulations reported next to the as-is result, per main's gate - no reordering/quality-safety claims, upper-bound and sampling-parameter caveats attached, unknown scenario names fail closed. 10 new tests (ceiling recovers expired hits, never-worse invariant, no-op cases, exact-only dedup, caveat presence, CLI integration, as-is result unchanged). 69/69 pass. README + CHANGELOG updated.
- Added tests/test_invariants.py: seeded randomized sweep (200 logs x both providers x isolate_sessions on/off) pinning the accounting identity (read+written+uncached==total), nonnegativity, totals consistency, and savings<=baseline; plus first-in-namespace-never-reads, inclusive expiry boundary, and zero-waste checks.
- Main's scope gate applied: dense renamed to "hypothetical compressed-timing scenario" - never-worse stated as a proven property of the simulator's model only, explicitly not a real-world bound (routing/concurrency/write pricing). dedup now requires cache_key and session_id equality too; caveats note organization and sampling parameters are not represented in the log. 75/75 pass.
- Hardened load_rules: missing/invalid/mis-shaped custom rules files and negative TTL/multiplier values now fail closed with clean SystemExit messages naming the provider (was raw tracebacks/nonsense rates). 13 new fail-closed tests (test_failclosed.py) covering log + rules + CLI paths. 88/88 pass. Held locally for the next batched publish per main.
- Added GitHub Actions CI (.github/workflows/ci.yml): pytest across Python 3.9-3.12 with dev extras. Fixed pyproject version drift (0.1.0 -> 0.2.0 to match the published v0.2.0 changelog references).
- Closed the OpenAI per-model minimum-length roadmap item as no-variation: doc verbatim (2026-10-28) gives uniform 1,024 tokens for GPT-5.6+; compare page (re-rendered) shows rates but no minimum column; earlier models vary by request settings and stay fail-closed. Evidence appended to rules/EVIDENCE.md; cached rates on the page still match EVIDENCE values (Astra 1.00/12.50, Sol 0.10/2.50, Luna 0.01).
- Added --what-if anthropic-1h: documented 1-hour TTL cost-benefit scenario for Anthropic logs (ttl 3600s, writes 2x vs 1.25x, reads 0.1x; provider-guarded, refresh-on-read assumption caveated). 4 new tests. 92/92 pass.
- Added .tokaudit.toml config support for analyze (auto-discovery + --config; CLI flags win; unknown keys/bad TOML/bad types fail closed; tomllib on 3.11+, tomli backport otherwise and in dev extras for CI parity). 9 new tests. 101/101 pass. README config section + CHANGELOG entry.
- Clean-install validation (main's steer) caught a real bug: pip-installed tokaudit could not find default rules (path relative to source tree). Fixed by bundling providers.json as package data + sync test pinning bundled==repo copy. tomli>=2 moved to runtime deps for Python<3.11. Validated in fresh venvs on 3.10: --no-deps install analyzes + lists rules cleanly; config use without tomli fails closed with a clear message (no traceback); config honored when tomli present. HTML report work reverted per main (no concrete need). 103/103 pass.
- Wheel-level install validation: built tokaudit-0.2.0 wheel (build isolation); verified providers.json inside the wheel and METADATA (Requires-Dist: tomli>=2; python_version<"3.11"). Installed the wheel into a fresh venv from a neutral cwd (/tmp/outside, no checkout) and reproduced the quickstart analysis (6 requests, 70.2% simulated savings). Added examples/quickstart.jsonl + captured report + examples/README.md, pinned by a CLI test. 104/104 pass.
- README accuracy sweep: executed every documented command against the wheel install from a neutral cwd - all pass. Caught and fixed a stale "# upper-bound ceiling" comment on --what-if dense in the usage block (renamed to compressed-timing scenario).
