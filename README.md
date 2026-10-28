# tokaudit

Audit LLM request logs for token waste and quantify prompt-caching savings.

Most teams running LLM workloads pay for the same prompt prefix over and
over: system prompts, tool definitions, few-shot examples, retrieved
documents. Both major providers offer prompt caching with steep discounts
on reused prefixes, but it is hard to know how much of your own traffic
would actually hit the cache. tokaudit answers that from your logs.

## What it does

Given a JSONL request log, tokaudit:

- **Simulates prompt caching** against provider rules (minimum cacheable
  length, cache lifetime, write/read multipliers) and reports estimated
  savings in units of the base input-token rate. Rules ship in a dated,
  sourced data file (`rules/providers.json`) and can be overridden with
  `--rules`, because providers change their terms.
- **Finds repeated content** sent in multiple requests (the same document,
  examples or boilerplate) and sizes the waste, so you can hoist it into
  one stable cached prefix.
- **Flags requests below the minimum cacheable length** and wasted cache
  writes (prefixes written but never re-used, which cost extra).
- **Counts tokens exactly** with `tiktoken` when installed, or with a
  dependency-free estimator (~4 chars/token, +/-25% by content type).

What it is not: a neural prompt compressor. It never rewrites your
prompts, so there is no risk of changing model behavior. It measures
workload-level reuse, which is where the reliable savings are.

Why another tool: in a bounded scan (2026-10-28) of public token tools,
we found single-prompt counters/minifiers (tokensaver, tok, tokentrimmer,
tokenshrink) and neural compressors (Microsoft's LLMLingua family), but
none focused on workload-level prompt-cache simulation from request
logs. That scan was not exhaustive; if a maintained tool does this well,
use it and tell us.

## Privacy

Request logs can contain sensitive prompts, personal data and leaked
secrets. tokaudit runs entirely locally: it makes no network calls and
uploads nothing. Still, treat reports as sensitive (they contain short
content previews), redact secrets from logs before sharing any output,
and prefer analyzing aggregated or synthetic logs when demonstrating
the tool.

## Install

```bash
pip install -e .          # estimator only, no dependencies
pip install -e ".[exact]" # + tiktoken for exact OpenAI counts
```

## Usage

```bash
tokaudit analyze requests.jsonl                 # markdown report to stdout
tokaudit analyze requests.jsonl --format json   # machine-readable
tokaudit analyze requests.jsonl --tokenizer estimate
tokaudit rules                                  # show provider rules in use
tokaudit analyze requests.jsonl --policy legacy   # force pre-GPT-5.6 policy family
```

Log format: one JSON request per line. Adapters normalize LiteLLM
SpendLogs rows, Langfuse observations and Helicone request rows
(field signatures grounded 2026-10-28; unknown shapes fail loudly,
never silently mis-parse). Raw formats: OpenAI chat style
(`{"model", "messages": [{"role", "content"}]}`), Anthropic messages style
(`{"model", "system", "messages"}` with string or block content), or a
generic `{"prompt": "..."}`. Optional `id`, `ts`/`timestamp`/`created`
(epoch or ISO-8601). Explicit `cache_control` / `prompt_cache_breakpoint`
markers in the log are honored.

## Method and honest limits

- Requests are rendered with a canonical template, not each provider's
  exact chat template. Relative reuse across requests from the same
  workload is preserved; absolute prefix lengths are approximate.
- Cache writes are charged on the new tokens beyond the longest cached
  prefix, matching how providers bill `cache_creation` tokens. Implicit
  caching is approximated as one write at the end of the prompt with
  intermediate message boundaries becoming readable.
- Requests without timestamps are treated as if nothing expires.
- Savings are in multiples of the base input rate, not dollars, so the
  result stays correct as prices change. Multiply by your current price.
- The estimator backend is a heuristic; use `--tokenizer exact` for
  decisions that depend on exact counts.

Multi-turn conversations are handled naturally: when each request
carries its full message history, later turns re-read the cached
earlier turns (tested in `tests/test_multiturn.py`, including the
adversarial case where a volatile timestamp inside the system prompt
destroys every cache match).

See `rules/providers.json` for the dated provider rules and their source
URLs, and `PROVENANCE.md` for how this project is built and credited.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

Roadmap and contribution ideas: `ROADMAP.md`.
