# tokaudit

[![tests](https://github.com/anishpentyala/tokaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/anishpentyala/tokaudit/actions/workflows/ci.yml)

Audit LLM request logs for token waste and quantify prompt-caching savings.

Most teams running LLM workloads pay for the same prompt prefix over and
over: system prompts, few-shot examples, retrieved
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
# try it: python -m tokaudit.cli integrity examples/integrity_spendlogs.jsonl
```

Offline consistency checks over the raw records (LiteLLM SpendLogs,
LiteLLM standard-logging payloads, Helicone rows, Langfuse exports),
each with a stated schema assumption:
zero spend on token-bearing non-error calls, spend vs cost_breakdown
disagreement (SpendLogs rows nest cost_breakdown inside the metadata
JSON - both levels are read), usage sums that don't add up (assuming the
OpenAI/LiteLLM usage shape; records whose own fields document a
cache-inclusive sum are not flagged), duplicate-billed request IDs,
provider-reported cost fields inside usage chunks (suppressed on
providers where that field is a documented contract, e.g. OpenRouter),
non-numeric/non-finite accounting fields, negative or fractional supported
token counts (even when totals balance), and the gateway's OWN
zero-cost diagnostic when a record carries it (standard-logging-payload
shape; LiteLLM itself computed $0 and recorded why). On
standard-logging-payload records, response_cost is read as the spend
field. Offline; no request content is read.

Analysis and integrity reject repeated decoded JSON member names in any raw
log object, including nested objects and supported JSON-encoded Langfuse input
and Helicone request bodies used for analysis. Even equal repeated values are
ambiguous input, not evidence to choose a winner. Errors identify the physical
log line without printing the repeated name or its values. Repeated names in
separate objects and duplicate-looking text in plain prompts remain valid.
Price sheets also reject duplicate decoded JSON object members at any depth,
including equal-valued duplicates. Dictionary-based Python APIs cannot recover
members already discarded by an earlier parser.

A clean integrity result requires accounting coverage. Empty files, unrelated
JSON, request-only rows, paginated envelopes and rows with no supported
accounting fields stop with file/line guidance before writing a report. Mixed
logs fail on the unsupported row rather than silently checking a subset.
Supported partial accounting rows remain valid; missing optional fields are
not manufactured as zero. Present non-numeric/non-finite values in LiteLLM
(including `response_cost`), Helicone and Langfuse accounting fields produce
findings. A zero-finding result means only that the available supported fields
passed these checks, not that every billing field was present or correct.

For CI artifacts or downstream processing, request a machine-readable report:

```bash
python -m tokaudit.cli integrity your_log.jsonl --format json --output integrity.json --fail-on-findings
```

JSON schema version 1 includes `summary.checked_records`, `format_counts`,
`finding_count`, `invariant_counts`, and the complete `findings` array. Each
finding retains its invariant, physical JSONL line number, request ID, detail
and source anchor. Unlike the text preview, JSON does not truncate finding
families after five rows. `--fail-on-findings` keeps the same exit policy in
both formats: exit 1 for findings when requested, otherwise exit 0 after a
valid check. Input errors still stop before overwriting the output file.

Coverage counts describe recognized rows, not the completeness of every
optional billing field or verified billing correctness.
The integrity reader does not retain complete input rows after checking them.
Cross-row duplicate checks keep only the first positive-spend amount and line
for each qualifying request ID. The returned report still keeps all findings,
and the duplicate-ID map grows with distinct billable IDs. Memory is therefore
not constant: one large input line, many findings, or many unique IDs can still
require substantial memory. Request bodies are parsed as part of each JSON row
but are not copied into the report or retained for cross-row checks. JSON has no request
content, but request IDs and diagnostic values can still be sensitive. Review
before sharing. The Python `run_integrity(path)` API still returns findings;
`build_integrity_report(path)` returns the structured report.

Every finding is labelled with the documented upstream failure class
whose signature it matches (e.g. litellm #40100/#48937/#35829/#25351,
langfuse #14987) and a link to the issue. Some anchored issues are
closed upstream (#40100 closed 2026-09, #25351 closed 2026-07,
#35829 closed 2026-10, langfuse #14987 closed 2026-08); the checks
remain useful because exported logs from affected versions still
circulate - and the failure class stays live: litellm #49369 (open)
reports the same zero-cost signature on the SDK stream wrapper. A finding means "consistent
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

## Message structure and cache identity

The canonical role-header rendering is used for token estimates, not as an
unambiguous serialization of a request. A literal `<user>` header in content
must not become a real message boundary. Cache identity therefore combines
exact token IDs with the roles and text lengths at message boundaries; exact
request duplicate scenarios compare structured role/text messages as well as
model, cache key, session and breakpoint layout. Token counts and the rendering
template are unchanged. This prevents these structural aliases, not every
difference between the canonical template and a provider's real rendering.

## Text-only input coverage

Analysis requires strings or lists of text/input_text parts with string text.
Non-text parts (images, audio, tool-use/results, thinking) and malformed content
are rejected instead of being silently discarded or stringified. Nonempty tool
and function configuration, tool-call payloads and tool/function-role messages
are also rejected. The LiteLLM, Langfuse and Helicone adapters preserve these
fields for this check. A mixed JSONL log fails at the unsupported row before
writing a report; it does not silently analyze only the text subset.

This is a text-only audit, not multimodal token counting or a full provider
request validator. Ordinary text and text-part cache markers remain supported.
Other request settings still need the simulation's existing assumptions and
duplicate-scenario caveats. Re-export a deliberately scoped text-only workload
rather than deleting unsupported fields and treating the result as the original.

## Provider-usage comparison coverage

Provider-reported token fields are compared only with the same uniquely
identified requests in the simulation. Each field has its own coverage count:
a record with prompt tokens but no cache-read field does not establish zero
cache reads. Missing comparable fields are JSON `null` and Markdown "unknown";
explicit zero remains zero. Requests excluded by an unverified model policy,
and duplicate request IDs that cannot be paired uniquely, are excluded from
this comparison. Simulation totals elsewhere still describe the full selected
workload. These are coverage-scoped diagnostics, not billing verification.

## Privacy

Request logs can contain sensitive prompts, personal data and leaked
secrets. tokaudit runs entirely locally: it makes no network calls and
uploads nothing. Still, treat reports as sensitive (they contain short
content previews), redact secrets from logs before sharing any output,
and prefer analyzing aggregated or synthetic logs when demonstrating
the tool.

Use `--omit-content-previews` (or TOML `omit_content_previews = true`) to
remove raw duplicate-content previews from JSON and Markdown reports. Counts,
request IDs, model names, file paths, metadata and simulation results remain.
This is not anonymization or a guarantee that a report contains no sensitive
information: identifiers and user-supplied metadata can be sensitive too.
In opt-in JSON, duplicate blocks omit the `preview` key and assumptions record
`content_previews_omitted: true`. Default reports keep their existing shape.

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

Price rates must be finite, nonnegative numbers. NaN, infinity, overflowed
numbers and booleans are rejected before a dollar report is written. Finite
numeric strings and zero rates remain supported.

`--prices SHEET.json` adds dollar figures from your own price sheet -
`{"models": {"gpt-6-astra": {"input": 10.0, "cached_input": 1.0,
"cache_write": 12.5}}}` (per 1M tokens). No prices ship with tokaudit;
sheets fail closed on duplicate JSON object members, excessively nested JSON,
unknown keys, missing or negative rates, and
requests whose model is absent from the sheet are excluded from dollar
totals. Each simulated request retains its own model for pricing, even
when request IDs repeat across models or timestamps reorder the log. Legacy
programmatic simulation rows without a model use ID lookup only when the
request ID is unique; ambiguous rows are unpriced rather than guessed.
Dollar figures apply your sheet to simulated token flows: still simulated,
not observed billing.

`--what-if` scenarios are hypothetical and non-mutating: they re-run the
simulation on a transformed copy of the log and report their own numbers
next to the as-is simulation. `dense` rescales timestamps so nothing
expires. Within the simulator's model this never reduces hits (a proven
model property), but it is a hypothetical timing scenario, not a
real-world bound - real hits also depend on best-effort provider prefix
routing and concurrency. `dedup` collapses exact duplicate requests
(same model, rendered prompt, prompt_cache_key, session_id and explicit
cache-breakpoint layout - only
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
(or from `--config PATH`). TOML values are validated before analysis:
`isolate_sessions` must be a real boolean, `dup_min_tokens` a real integer
and `dup_max_shingles` a nonnegative integer
(no string/bool/fraction coercion), and provider/tokenizer/format/scenario
names must match the CLI choices. Custom policy names and file paths remain
supported. Invalid values stop with config-file guidance before report writes.
Known keys: `rules`, `policy`, `tokenizer`,
`provider`, `format`, `dup_min_tokens`, `dup_max_shingles`, `isolate_sessions`, `what_if`,
`output`, `prices`, `omit_content_previews`. Unknown keys, invalid TOML and unreadable files fail
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

OpenAI read refresh is separate from write retention: a read extends coverage
only to the later of its existing expiry and 30 minutes after the read. The
`openai-12h` what-if does not restart 12 hours on each hit. Custom provider rules
can set `read_refresh_seconds`; omitting it keeps same-TTL refresh behavior.
Regenerate earlier 12-hour what-if estimates, which could overstate savings.

### Custom policy input validation

Rules loaded from `--rules` (and the bundled rules file) reject repeated decoded
JSON member names instead of keeping the last value. Counts and lifetimes must
be non-negative integers representable as finite numbers; integral floats and
base-10 integer strings are accepted for existing exports. Booleans are not
counts. Cost multipliers must be finite and non-negative; finite numeric strings
remain accepted. Policy flags require JSON `true` or `false`, not strings or
numbers. An omitted or null `read_refresh_seconds` retains same-TTL refresh.
Zero counts, lifetimes and multipliers remain supported custom parameters.

Malformed family/minimum containers and non-string match/note lists stop with
file and field guidance, before a report is written. An invalid override cannot
silently truncate a TTL, turn the string `"false"` into enabled refresh, or
introduce NaN cost arithmetic. This validates policy input, not whether custom
rates match a provider contract. Unknown fields remain ignored for compatibility;
constructed Python `ProviderRules` objects do not go through this file loader.
No bundled provider values or simulation behavior for valid policies change.

### Reported token-count validation

Normalized `actual_usage` fields must be nonnegative integer counts. Boolean,
negative, fractional and non-finite values fail with the record and line
context instead of being truncated or converted into token counts. Integral
floats and base-10 integer strings remain accepted for normalized exports.
Helicone's optional numeric usage fields keep their existing missing-data
policy: invalid fields are omitted, not changed to zero, and therefore do not
contribute to that field's comparison coverage. No provider pricing or
simulation rules are changed.

### Exact prefix-cache identity

Cache simulation compares complete token sequences, not a hash alone. Keys
retain signed64-bit token IDs as compact bytes; arbitrary-size custom token
IDs use exact tuples without truncation. This adds memory proportional to the
unique cached prefix content, including expired entries retained for write
attribution. On a single synthetic500-request workload with2000distinct tokens
per request, tracemalloc peak was8.07MiB versus0.46MiB for the old hash-only
index (full Python tuples used34.67MiB). Runtime was0.150s versus0.121s on that
run. This is a bounded resource check, not a general performance guarantee.
The regression uses custom positive token IDs with deterministic equal tuple
hashes; it does not establish a collision rate for built-in tokenizers.

### Wasted-write attribution across expiry

Wasted cache writes are tracked per prefix-write generation. A read of a
rewritten prefix does not retroactively credit an earlier write that expired
unused. Read refresh extends the same generation; it does not create a new
write. The existing boundary approximation remains: if any stored boundary
from one charged write is later read during its lifetime, the whole charged
delta is treated as used. This is not token-by-token attribution or a provider
billing verdict.

### Per-request reported usage diagnostics

When reported usage is available, JSON retains field-specific reported,
simulated and signed `simulated_minus_reported` token counts for each uniquely
matched request. Aggregate agreement can hide opposite differences across
requests, so inspect these rows before trusting an aggregate comparison.
Missing fields remain unknown, not zero; duplicate/ambiguous IDs and requests
excluded by simulation stay excluded. Markdown shows the first 20 comparable
requests, with an explicit truncation note; JSON keeps every comparable row.
Counts use different tokenizers/templates and are not evidence of misbilling.
Rows expose request IDs and model names, even with content previews omitted.

### Duplicate-search coverage

Analysis reports `duplicate_search` in each provider group's JSON report and a
shingle-index coverage line in Markdown: indexed windows, total eligible windows,
the index budget (default 500,000 per provider group), and `index_complete`. Once the budget is
used, later windows are not indexed. A capped search is labelled partial, even
when it returns real blocks, and an empty result does not claim there are no
duplicates in the unindexed portion.

`index_complete` means every eligible token window entered the index. It does
not mean every duplicate was discovered: extension, greedy suppression and the
top-block limit remain heuristics. All messages are still tokenized and retained
for comparison, so the index cap is not a total memory bound. Set
`--dup-max-shingles N` (or `dup_max_shingles = N` in config) to choose this
per-provider-group budget. CLI overrides config; the default remains 500,000.
Zero skips duplicate indexing while cache simulation and tokenization still
run, and coverage stays partial whenever eligible windows exist. Increasing it can
use more time and memory. Token/cost simulation coverage is separate and is not
changed by this index budget.

Python callers can use `analyze_duplicates(...)` for a `DuplicateSearchResult`
with `blocks` and these coverage fields. `find_duplicates(...)` keeps its list
return for compatibility. Both accept positive integer `min_tokens`, `shingle`
and `top`, plus non-negative integer `max_shingles` (zero indexes no windows).
Invalid limits fail instead of silently disabling detection or returning a
misleading partial list. The CLI requires a positive `--dup-min-tokens` value.

### Tokenization reuse during analysis

One CLI `analyze` invocation reuses exact-text tokenizations across simulation,
what-if scenarios and duplicate search. The local LRU retains at most 128
entries, 100,000 token IDs and 1,000,000 text characters. Oversize entries bypass
it without evicting smaller reusable entries. Retained tokens are immutable and
returned lists are independent, so a consumer cannot change later tokenizations.
The cache disappears after that analysis; tokenization backends and reports are
unchanged. Standalone Python simulation/dedup APIs are not automatically cached.

These limits bound retained cache contents, not the whole analysis's memory or
peak allocations. Cold or mostly unique workloads can cost extra time and
memory. A synthetic local exact-tokenizer check with 100 requests, one shared
system message and three what-if passes reduced backend calls from 800 to 401;
three paired runs had median simulation/scenario time 0.261s without reuse and
0.156s with reuse. This excludes CLI startup, parsing and duplicate search. It is
not a general benchmark, provider billing saving or quality claim. Eviction can
require re-encoding and does not change the resulting token IDs or reports.

Duplicate extension reuses already verified equal token runs along the same
message-pair alignment instead of comparing every overlapping suffix again.
The local memo holds at most 4,096 alignments; eviction only repeats work.
Candidate windows still require exact token equality, including hash-collision
controls. Detection, greedy suppression, top-block selection and index coverage
are unchanged. The memo limit is not a total analysis-memory bound, and short or
mostly unique inputs can cost extra work.

A synthetic local check of three identical 2,000-token messages, shingle size20,
minimum100, had three-paired-run median duplicate-search time0.594s on the prior
code and0.00968s with verified-run reuse, with identical blocks and coverage.
This excludes tokenization/CLI startup and is not a general benchmark or a
provider-billing saving.

### Duplicate content and repeated request IDs

Duplicate search identifies request records by their position in the supplied
request list, not by the exported ID label. Separate records with equal or empty
IDs can contain real repeated content; their labels no longer hide those blocks
or collapse separate blocks during suppression. Repeated log rows can also be
retry emissions or duplicated exports, so detection does not prove separate
provider calls or billed waste.

Each JSON duplicate block includes `request_indices`, aligned one-for-one with
its message occurrences, and `distinct_requests`. Indices are zero-based in the
provider group's input list (or the Python caller's list), not physical JSONL
line numbers and not global cross-provider indices. `requests` remains the
sorted set of ID labels. Markdown separates message occurrences from distinct
request records. Repeated-token volume still counts message occurrences; it is
not a separate-request or billing estimate. Manually constructed legacy
`DuplicateBlock` values without indices retain an ID-based count fallback.

Integrity checks treat supported LiteLLM/Helicone prompt/completion/total token counts as nonnegative integers. An integral JSON float
remains accepted. Negative/fractional values produce `invalid_token_count`,
not a correction or inferred replacement. This rule does not apply to cost
fields: refunds or credits may be negative. Provider-specific nested detail
counts and Langfuse flat usage buckets (which may represent other units) are
outside this check. Related issue links describe failure classes,
not proof that an export has that upstream bug or was misbilled.

The exact tokenizer treats literal special-token spellings in exported prompt
text (for example `<|endoftext|>`) as ordinary text, rather than rejecting
the log or inserting model-control token IDs. Counts still describe the
canonical text render, not a provider's hidden chat-template control tokens.
