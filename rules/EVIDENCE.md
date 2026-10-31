# Evidence for provider caching rules

Verbatim excerpts from official documentation, fetched 2026-10-31. Re-fetch both pages before relying on these after that date; providers change terms without notice. Add a new dated entry whenever a rule is updated.

## OpenAI — https://platform.openai.com/docs/guides/prompt-caching (fetched 2026-10-31)

> "OpenAI reserves the right to cache prompts for longer periods, up to 24 hours."

> "Cached entries are retained for up to 12 hours of inactivity."

> "The cache lives at zero additional cost and cached tokens are discounted by 90% for all supported models."

> "A 30-minute read refreshes coverage to at least 30 minutes after that read. If it uses an entry with valid 12-hour coverage, it preserves the later of the existing expiry and 30 minutes after the read. It does not restart a 12-hour clock or incur another write charge."

Also from the pricing section of the same page (models table, fetched 2026-10-31): cache read prices are 10% of input price for the GPT-5.1/5/4.1/o-series generation (e.g. GPT-5.1 $0.125/M vs $1.25/M input; GPT-4.1-mini $0.025/M vs $0.40/M). Prompts shorter than 1024 tokens are never cached (page's requirements section).

Implication for our rules (as of 2026-10-31):

- write_multiplier = 1.25 (a write costs input_price + 25%; cache-hit reads cost 10% of input). The 1.25 applies to GPT-5.1/5/4.1/o-generation models; legacy 4o prices vary slightly, still within ±0.05.
- discount = 0.9 (reads at 10%).
- ttl_short_hours = 12 (inactivity TTL from the doc), refresh_on_read = true per the quoted read-refresh sentence; reads during valid 12-hour coverage extend expiry to max(existing expiry, read+30min) without restarting a 12-hour clock.
- max_hours = 24 is the documented upper bound ("up to 24 hours").
- min_tokens = 1024.

## Anthropic — https://docs.claude.com/en/docs/build-with-claude/prompt-caching (fetched 2026-10-31)

> "Prompt caching references the entire prompt - tools, system, and messages (in that order) up to and including the block designated with cache_control."

> "The cache has a 5-minute lifetime, refreshed each time the cached content is used."

> "Cache write tokens are 25% more expensive than base input tokens"

> "Cache read tokens are priced at 0.1 of base input token price" (i.e., 90% discount)

Also stated on the same page (fetched 2026-10-31): a 1-hour cache TTL option exists (beta), cache reads don't count against ITPM rate limits for Claude Sonnet 4.5 and later models, and minimum cacheable lengths are 1024 tokens for Claude Opus/Sonnet models (2048 for Haiku 4.5, 1024 for Haiku 3.5/3).

Implication for our rules (as of 2026-10-31): write_multiplier = 1.25, discount = 0.9, ttl_short_hours = 5/60 (5 minutes), refresh_on_read = true, min_tokens = 1024 (Haiku 4.5 = 2048 noted in provider notes; we use 1024 as the common minimum and flag the Haiku exception in the notes field).

## Disagreements / things to re-check

- OpenAI's "up to 24 hours" is a maximum, not a guarantee — the doc's own wording is "OpenAI reserves the right to cache prompts for longer periods, up to 24 hours." We use max_hours only as an upper-bound display, never as an assumed TTL.
- Anthropic's 1-hour TTL option is a beta header; our rules model the default 5-minute TTL only.
- If either page's numbers change, update rules/providers.json AND this file in the same commit, bumping the fetch dates.
