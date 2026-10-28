# Evidence for provider caching rules

Verbatim excerpts from official documentation, fetched 2026-10-28.
Re-fetch both pages before relying on these after that date; providers
change terms without notice.

## OpenAI
Source: https://platform.openai.com/docs/guides/prompt-caching

- "The minimum cacheable prompt length is 1,024 tokens for GPT-5.6 and
  later and varies by request settings for earlier models."
- "For GPT-5.6 and later, cache writes cost 1.25x the standard, uncached
  input-token rate for the default 30-minute lifetime. In the Responses
  API, you can choose a 12-hour minimum lifetime at a write rate of 2.5x
  ordinary input. Subsequent reads cost 0.1x the ordinary input rate on
  most of these models and 0.05x on GPT-6.1 Sol and GPT-6.1 Terra,
  regardless of the selected lifetime."
- "Cache-write pricing is not an additive fee: input tokens use the
  uncached-input, cached-input, or cache-write rate."
- "These are total write rates, not surcharges on top of ordinary input."
- "OpenAI may retain entries longer than the requested minimum."
- Model comparison table: for earlier models, "No additional cache-write
  charge"; cached-token reporting "rounds down to a multiple of 128".
- "Cache reuse requires the entire rendered prefix to match."
- "Implicit mode: OpenAI chooses breakpoint locations out of the box."
- Explicit mode: "Each request can create up to four cache writes."

## Anthropic
Source: https://docs.claude.com/en/docs/build-with-claude/prompt-caching

- "By default, the cache has a 5-minute lifetime. The cache is refreshed
  for no additional cost each time the cached content is used."
- "5-minute cache write tokens are 1.25 times the base input tokens
  price"; "1-hour cache write tokens are 2 times"; "Cache read tokens
  are 0.1 times the base input tokens price".
- "The minimum cacheable prompt length is: 512 tokens for Claude Fable
  5.1, Claude Mythos 5.1, Claude Opus 5, Claude Fable 5, and Claude
  Mythos 5; 2,048 tokens for Claude Mythos Preview and Claude Opus 4.7;
  4,096 tokens for Claude Opus 4.6 and Claude Opus 4.5."
- "total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens
  + input_tokens" with "input_tokens: Number of input tokens which were
  not read from or used to create a cache (that is, tokens after the
  last cache breakpoint)."
- "For concurrent requests, note that a cache entry only becomes
  available after the first response begins."

## Cache scope and routing (OpenAI, fetched 2026-10-28, same URL)

Verbatim excerpts grounding the simulation's cache-namespace model:

> "Caches are not shared across organizations and cannot be reused across regional processing boundaries."

> "OpenAI handles routing automatically. Within an organization and processing region, routing for a given model depends on: ... A supplied `prompt_cache_key`, which separates cache reuse between groups of requests and helps optimize cache routing on models before GPT-5.6."

> "On GPT-5.6 and later, OpenAI handles cache routing automatically; the key is not needed to optimize caching. You can use separate keys to maintain separate cache accounting for customers or users within your application."

Implication for simulation: entries are shared within one organization + region
per model; `prompt_cache_key` partitions reuse. App session IDs are NOT a
documented cache boundary. `--isolate-sessions` is an explicit conservative
scenario, labeled as such in output.

## Per-model list prices (OpenAI model comparison, fetched 2026-10-28 via browser render; page is JS-only)

URL: https://platform.openai.com/docs/models/compare

Per-model pricing cards, verbatim values:

- GPT-6 Astra ("Our most capable model for the most demanding work"):
  Input $10.00, Cached input $1.00, Cache writes $12.50, Output $50.00 per 1M tokens.
  -> read 0.1x, write 1.25x.
- GPT-6.1 Sol ("Near-Astra performance for complex work at a lower cost"):
  Input $2.00, Cached input $0.10, Cache writes $2.50, Output $10.00 per 1M tokens.
  -> read 0.05x, write 1.25x.
- GPT-6 Luna ("Our most efficient model for focused, high-volume tasks"):
  Input $0.10, Cached input $0.01, Cache writes $0.125, Output $0.50 per 1M tokens.
  -> read 0.1x, write 1.25x.

These confirm the documented multipliers per model. The page lists current
models only; pre-GPT-5.6 cached-input rates remain unverified, so the legacy
family stays fail-closed.
