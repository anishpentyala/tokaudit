# tokaudit report

## Summary

- Requests analyzed: 5
- Provider rules: openai (rules as of 2026-10-28)
- Tokenizer: estimate
- Requests in savings simulation: 5
- Total input tokens: 5,413
- Baseline cost: 5,413 units
- With caching: 1,830 units
- Simulated savings (not observed billing): 3,583 units (66.2%)
- Cache writes never re-read: 40 tokens

## Duplicate content across requests

- ~1070 tokens x 5 requests (4280 repeated tokens): "You are the support agent for Northwind Books. Policy: orders ship within two business days; returns..."

## Provider-reported vs simulated

- Requests with provider usage fields: 5
- Cache read tokens: provider 18,000 vs simulated 4,292
- Cache write tokens: provider 4,500 vs simulated 1,121
- Provider prompt tokens: 23,070
- Provider-reported fields come from the export (Helicone prompt_cache_read_tokens / prompt_cache_write_tokens / prompt_tokens), counted with the provider's tokenizer and hidden context; the simulation re-tokenizes a canonical render. Exact equality is not expected - treat large divergence as a signal to question coverage or model assumptions, not as a billing discrepancy by itself.

## Recommendations

- Hoist 1 repeated block(s) (~4280 repeated tokens) into one stable cached prefix (system or developer message) instead of resending them per request.
- Some cached prefixes were never re-used. Only write prefixes you expect to reuse within the cache lifetime; a single-use write costs more than sending uncached.
- Keep volatile fields (timestamps, request IDs) out of the cached prefix to preserve the hit rate this simulation estimates.

## Assumptions and sources

- Estimated counts (about 4 chars/token, +/-25% by content type); install tokaudit[exact] for exact OpenAI counts.
- Requests are rendered with a canonical template, not the provider's exact chat template; relative reuse patterns are preserved.
- Units are multiples of the base input-token rate, not dollars, so results stay valid as dollar prices change.
- Savings are SIMULATED from your log against the stated provider rules; they are not observed billing data. Confirm against the provider's usage fields (cache_read_input_tokens / cached_tokens) in production.
- Some tokens were written to cache but never re-read; single-use prefixes cost extra whenever the family charges for writes.
- Cache namespace model (rules/EVIDENCE.md, OpenAI 2026-10-28): entries shared within one organization + region per model; a record's prompt_cache_key separates reuse groups. Logs carry no org/region fields, so one shared org+region is assumed - mixed-account logs can overstate reuse.
- [gpt6_verified_01x] GPT-6 Astra and GPT-6 Luna specifically: 0.1x read / 1.25x write, confirmed by model comparison page list prices 2026-10-28 (Astra $10.00/$1.00/$12.50, Luna $0.10/$0.01/$0.125 per 1M).
- Policy families applied: gpt-6-astra -> gpt6_verified_01x
- Minimum cacheable prefix 1024 tokens for GPT-5.6 and later; varies by model and request settings for earlier models.
- Default minimum cache lifetime is 30 minutes; a 12-hour minimum lifetime writes at 2.5x base input (Responses API).
- The lifetime is a MINIMUM: OpenAI may retain entries longer, so a hard-expiry simulation is conservative (real hit rates can be higher).
- Cache reads 0.1x base input on most GPT-5.6+ models; 0.05x on GPT-6.1 Sol and GPT-6.1 Terra.
- Write rates are totals, not surcharges: each input token is billed at exactly one of uncached / cache-write / cached-read rate.
- Models before GPT-5.6 have no cache-write charge (read discount only) and use prompt_cache_retention instead of ttl.
- Cached-token reporting rounds down to a multiple of 128 tokens.
- The rendered prefix is the cache key: settings such as reasoning.effort, text.verbosity and context_management compaction can change it.
- Caching is on by default (implicit); explicit breakpoints available via prompt_cache_options (up to 4 writes per request).
- Source (openai): https://platform.openai.com/docs/guides/prompt-caching
- Source (anthropic): https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- Source (openai_compare): https://platform.openai.com/docs/models/compare
