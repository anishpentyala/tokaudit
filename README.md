# tokaudit

[![tests](https://github.com/anishpentyala/tokaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/anishpentyala/tokaudit/actions/workflows/ci.yml)

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
  sourced data file (bundled as package data; the repo copy at
  `rules/providers.json` is the editable source) and can be overridden with
  `--rules`, because providers change their terms.
- **Finds repeated content** sent in multiple requests (the same document,
  examples or boilerplate) and sizes the waste, so you can hoist it into
  one stable cached prefix.
- **Flags requests below the minimum cacheable length** and wasted cache
  writes (prefixes written but never re-used, which cost extra).
- **Counts tokens exactly** with `tiktoken` when installed, or with a
  dependency-free estimator (word-based heuristic: ~4/3 token per short
  natural-language word; per-character scoring on symbol-heavy words,
  dense scripts, emoji, and long ID/hash runs).

Estimator accuracy, measured against tiktoken o200k_base on the deterministic
SYNTHETIC corpus in `tokaudit.benchmark` (60 generated samples per category,
seed 20261029; not real workload data):

| Category | Samples | Mean exact tokens | Mean abs err | Median abs err | p90 abs err | Max abs err |
| --- | --- | --- | --- | --- | --- | --- |
| prose | 60 | 60 | 17.5% | 17.9% | 20.2% | 24.1% |
| code | 60 | 45 | 14.8% | 12.1% | 23.3% | 27.8% |
| json | 60 | 340 | 20.1% | 22.2% | 26.2% | 31.9% |

## Accounting integrity checks

`tokaudit analyze` finds token waste. `tokaudit integrity` answers a
different question: are the accounting fields themselves internally
consistent? Upstream bugs regularly say no - and when spend reads $0,
budgets silently stop enforcing
([litellm#48937](https://github.com/BerriAI/litellm/issues/48937)).

```bash
python -m tokaudit.cli integrity your_log.jsonl
# add --fail-on-findings to exit 1 when anything fires (CI gating)
```

Offline consistency checks over the raw records (LiteLLM SpendLogs,
Helicone rows, Langfuse exports), each with a stated schema assumption:
zero spend on token-bearing non-error calls, spend vs cost_breakdown
disagreement, usage sums that don't add up (assuming the OpenAI/LiteLLM
usage shape; records whose own fields document a cache-inclusive sum are
not flagged), duplicate-billed request IDs, provider-reported cost
fields inside usage chunks (suppressed on providers where that field is
a documented contract, e.g. OpenRouter), and non-numeric/non-finite
accounting fields. Offline; no request content is read.

Every finding is labelled with the documented upstream failure class
whose signature it matches (e.g. litellm #40100/#48937/#35829/#25351,
langfuse #14987) and a link to the issue. A finding means "consistent
with that class under the stated schema assumption" - whether it applies
depends on your upstream versions, and it is a signal, never proof of
misbilling; legitimate $0 spends, discounts, cached calls and retry
re-emissions exist and are named in the finding text where they apply. The
test fixtures are synthetic records built to reproduce each documented
signature, not observed provider data.

Reproduce with `python -m tokaudit.benchmark --samples 60`, or measure the
estimator on your own text with
`python -m tokaudit.benchmark --file your_log_excerpt.txt` (chunks the file
and reports the same error statistics; requires `tokaudit[exact]`).

Held-out validation on independently sourced texts (chunked at 200/800/3000
characters, mean absolute error vs tiktoken o200k_base; sources are public
domain, permissively licensed, or public API responses):

| Text | Source / license | Before rework | After |
| --- | --- | --- | --- |
| English prose (Pride and Prejudice) | Project Gutenberg, public domain | 6.7% | 6.8% |
| French prose (Les Miserables I) | Project Gutenberg, public domain | 15.9% | 15.9% |
| Python code (cpython Lib/argparse.py) | PSF-2.0 | 12.9% | 7.8% |
| Real API JSON (GitHub REST responses) | api.github.com public responses | 30.9% | 19.0% |
| UDHR English/French/Spanish/German | unicode.org/udhr (free with attribution) | 3.7-16.8% | unchanged |
| UDHR Hindi | same | 6.9% | 6.9% |
| UDHR Russian / Arabic | same | 22.8% / 23.3% | unchanged (known limit) |
| UDHR Vietnamese | same | 47.1% | unchanged (known limit) |
| UDHR Japanese / Chinese | same | 96.4% / 93.1% | 26.1% / 20.4% |
| UDHR Korean / Khmer / Myanmar | same | 43.6% / 57.7% / 50.6% | 14.5% / 5.7% / 18.0% |
| UDHR Thai | same | 49.8% | 52.5% over (known limit) |
| Emoji-heavy mixed text | constructed sample | 15% under | matches exactly |

Known weak spots, stated plainly: Vietnamese (~47% under), Russian and Arabic
(~23% under), Thai (~52% over), and highly repetitive text (~27% over on a
repeated-sentence fixture - BPE compresses repetition a length heuristic
cannot see). Unspaced scripts were the worst case before this rework
(93-97% under) and are now bounded but still approximate. Use
`--tokenizer exact` for exact counts.

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
tokaudit analyze requests.jsonl --policy gpt6_verified_01x  # force a verified family
tokaudit analyze requests.jsonl --isolate-sessions          # conservative scenario
tokaudit analyze requests.jsonl --what-if dense             # compressed-timing scenario
tokaudit analyze requests.jsonl --what-if dedup             # exact-duplicate collapse
tokaudit analyze requests.jsonl --what-if anthropic-1h      # 1-hour TTL cost-benefit
                                                # (Anthropic logs; use --provider anthropic on mixed logs)
tokaudit analyze requests.jsonl --what-if openai-12h        # 12-hour TTL cost-benefit
                                                # (OpenAI logs; use --provider openai on mixed logs)
```

`--prices SHEET.json` adds dollar figures from your own price sheet -
`{"models": {"gpt-6-astra": {"input": 10.0, "cached_input": 1.0,
"cache_write": 12.5}}}` (per 1M tokens). No prices ship with tokaudit;
sheets fail closed on unknown keys, missing or negative rates, and
requests whose model is absent from the sheet are excluded from dollar
totals. Dollar figures apply your sheet to simulated token flows: still
simulated, not observed billing.

`--what-if` scenarios are hypothetical and non-mutating: they re-run the
simulation on a transformed copy of the log and report their own numbers
next to the as-is simulation. `dense` rescales timestamps so nothing
expires. Within the simulator's model this never reduces hits (a proven
model property), but it is a hypothetical timing scenario, not a
real-world bound - real hits also depend on best-effort provider prefix
routing and concurrency. `dedup` collapses exact duplicate requests
(same model, rendered prompt, prompt_cache_key and session_id - only
safe if the duplicates were redundant, which sampling parameters and
organization, not present in the log, can decide).
`anthropic-1h` (Anthropic logs only) re-simulates with the documented
1-hour cache option - TTL 3600s, writes at 2x base input instead of
1.25x, reads unchanged - so pricier writes and the longer reuse window
are weighed together. `openai-12h` (OpenAI logs only) does
the same for the documented 12-hour cache option - TTL 43200s, writes at
2.5x ordinary input instead of 1.25x for the default 30-minute cache,
reads unchanged (including per-model read discounts). No scenario claims
reordering or collapsing real traffic is safe for your application.

Savings are estimated only for exactly matched, documented model
identifiers (e.g. `gpt-6-astra`, `gpt-6.1-sol`). Unknown models, dated
snapshots and lookalike variants fail closed: they are excluded from
savings but still analyzed for tokens and duplicates. Force a family with
`--policy` to run an explicit hypothetical, or supply your own verified
rates with `--rules`. Cache reuse is simulated per model and
`prompt_cache_key` (documented provider scope); `--isolate-sessions`
additionally partitions by `session_id` as a clearly labeled conservative
scenario, not provider behavior.

Log format: one JSON request per line. Adapters normalize LiteLLM
SpendLogs rows, Langfuse observations and Helicone request rows
(field signatures grounded 2026-10-28; unknown shapes fail loudly,
never silently mis-parse). Raw formats: OpenAI chat style
(`{"model", "messages": [{"role", "content"}]}`), Anthropic messages style
(`{"model", "system", "messages"}` with string or block content), or a
generic `{"prompt": "..."}`. Optional `id`, `ts`/`timestamp`/`created`
(epoch or ISO-8601). Explicit `cache_control` / `prompt_cache_breakpoint`
markers in the log are honored.

## Configuration

`tokaudit analyze` reads defaults from `./.tokaudit.toml` when present
(or from `--config PATH`). Known keys: `rules`, `policy`, `tokenizer`,
`provider`, `format`, `dup_min_tokens`, `isolate_sessions`, `what_if`,
`output`, `prices`. Unknown keys, invalid TOML and unreadable files fail
closed with a clear error. Explicit CLI flags always win over the file.
TOML parsing uses the standard library on Python 3.11+ and the bundled
`tomli` dependency on older versions.

## Provider-reported usage

When an export carries real usage fields, the report adds a
"Provider-reported vs simulated" section. Supported today: Helicone rows
(`prompt_cache_read_tokens` / `prompt_cache_write_tokens` /
`prompt_tokens`, typed fields on HeliconeRequest), Langfuse
`usageDetails` (cache key families pinned by Langfuse's own
provider-usage-key matrix, excerpt in `rules/EVIDENCE.md`; aliases
within one bucket are de-duplicated by taking the maximum, not the sum),
and LiteLLM SpendLogs (`prompt_tokens` only). Reported usage is never
fed back into the simulation. Providers count with their own tokenizers
and hidden context, so exact equality is not expected; large divergence
is a signal to question log coverage or model assumptions.

Two deliberate exclusions: LiteLLM cache usage stays uncompared because
SpendLogs has no cache-read token field, and its `cache_hit` flag marks
LiteLLM's own application-level cache, not the provider's prompt cache.
And `examples/helicone_usage.jsonl` is a SYNTHETIC hand-written fixture
whose usage values are tiktoken-derived reference numbers, not observed
provider usage or billing data.

## Method and honest limits

- Requests are rendered with a canonical template, not each provider's
  exact chat template. Relative reuse across requests from the same
  workload is preserved; absolute prefix lengths are approximate.
- Cache writes are charged on the new tokens beyond the longest cached
  prefix, matching how providers bill `cache_creation` tokens. Implicit
  caching is approximated as one write at the end of the prompt with
  intermediate message boundaries becoming readable.
- Requests without timestamps are treated as if nothing expires.
- One shared organization and region is assumed (logs carry no such
  fields); mixed-account logs can overstate reuse.
- Savings are in multiples of the base input rate, not dollars, so the
  result stays correct as prices change. Multiply by your current price.
- The estimator backend is a heuristic; use `--tokenizer exact` for
  decisions that depend on exact counts. Estimator error propagates into
  simulated savings in either direction - the error in a savings figure
  depends on the estimator's error on BOTH the baseline and the
  optimized run, so no general conservative-direction claim holds. On
  the two shipped example logs, `--tokenizer estimate` overstates saved
  units by 10-12% versus `--tokenizer exact`; on JSON-heavy logs the
  estimator undercounts tokens instead. Treat estimator-mode savings as
  approximate in magnitude, not bounded in direction.

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
