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

`helicone_usage.jsonl` is a five-request Helicone export (typed
HeliconeRequest fields) where the provider reported cache reads and one
cache write. Run:

```bash
tokaudit analyze examples/helicone_usage.jsonl --tokenizer estimate
```

The report gains a "Provider-reported vs simulated" section (captured in
`helicone-usage-report.md`, regenerated 2026-10-28). The provider numbers
(18,000 read / 4,500 written) and the simulated numbers (4,292 / 1,121
with the estimator) differ by design: the provider counts with its own
tokenizer, and the estimator is crude on repetitive prose. The section
exists to catch coverage or modeling mistakes, not to reproduce billing
arithmetic - see the note printed under the section.
