# Quickstart example

`quickstart.jsonl` is six OpenAI support requests that share one long
system prompt (a bookstore policy), sent two minutes apart. Reproduce the
report from a repo checkout:

```bash
pip install .
tokaudit analyze examples/quickstart.jsonl --tokenizer estimate --what-if dedup
```

Expected highlights (full captured output in `quickstart-report.md`,
regenerated 2026-10-28 with the estimator tokenizer):

- 6 requests, 7,886 total input tokens
- Simulated savings 5,532 units (70.2%) - the shared prefix is written
  once and read five times; units are multiples of the base input rate,
  not dollars, and savings are simulated, not observed billing
- The duplicate-content section flags the repeated policy text
  (~1,303 tokens x 6 requests) as a hoist candidate
- 45 cache-write tokens are never re-read (the per-question tails)

The log uses `gpt-6-astra`, an exactly documented model identifier with
verified cache rates (see `rules/EVIDENCE.md`). Models without verified
rates are excluded from savings and listed under
"skipped, unverified policy".

## Provider-reported usage example

`helicone_usage.jsonl` is a SYNTHETIC five-request Helicone export
(typed HeliconeRequest fields), written by hand for demonstration: no
real traffic, provider account, or billing system produced it. Its
`prompt_cache_read_tokens` / `prompt_cache_write_tokens` /
`prompt_tokens` values are tokenizer-derived reference values (see
below) standing in for the fields a real Helicone export would carry.
Run:

```bash
tokaudit analyze examples/helicone_usage.jsonl --tokenizer estimate
```

The report gains a "Provider-reported vs simulated" section (captured in
`helicone-usage-report.md`, regenerated 2026-10-28). The usage values in
this synthetic fixture were counted with a real tokenizer (tiktoken
`o200k_base`) over the fixture's own prompts - the shared prefix is
1,119 tokens, read four times and written once - so they are internally
consistent reference values, NOT observed provider usage or billing
data. tokaudit's estimator counts the same prefix at ~1,394, so the
section shows a ~1.25x divergence between the fixture's reference
values and the estimator; that gap measures this fixture only, not
estimator accuracy across workloads. The section exists to catch
coverage or modeling mistakes, not to reproduce billing arithmetic, and
exact equality is not expected even with matching tokenizers because
providers count hidden context. See the note printed under the
section.
