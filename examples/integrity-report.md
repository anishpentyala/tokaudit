# Integrity example (SYNTHETIC)

`integrity_spendlogs.jsonl` is a SYNTHETIC ten-row LiteLLM SpendLogs export,
written by hand for demonstration: no real traffic, provider account, or
billing system produced it. The records are shaped like real SpendLogs rows
(accounting fields top-level; cost_breakdown and usage_object nested inside
the metadata JSON, matching the current SpendLogsMetadata schema), plus one
record carrying the gateway's own zero_cost_diagnostic field.

Captured output (regenerated 2026-10-31, `tokaudit integrity examples/integrity_spendlogs.jsonl`):

```
integrity: 11 finding(s) in examples/integrity_spendlogs.jsonl

duplicate_spend_record: 1 record(s) - consistent with https://github.com/BerriAI/litellm/issues/25351
  line 6 [demo-dup-1]: request_id already billed $0.002 at line 5; this row bills $0.002 again. Verify these are distinct billable calls - retried or streamed re-emissions sharing one request_id can look identical without being double billing.

non_numeric_accounting_field: 1 record(s) - consistent with https://github.com/BerriAI/litellm/issues/25351
  line 9 [demo-nonnumeric-1]: spend='0.01' is not a finite number

spend_cost_breakdown_disagreement: 1 record(s) - consistent with https://github.com/BerriAI/litellm/issues/40100
  line 3 [demo-mismatch-1]: spend=0 but cost_breakdown.total_cost=0.0042 in the same record (beyond float tolerance). The record's own fields disagree; legitimate explanations include discounts or a unit mismatch - a signal to inspect, not proof of misbilling.

upstream_cost_field_in_usage: 1 record(s) - consistent with https://github.com/BerriAI/litellm/issues/35829
  line 8 [demo-upstream-1]: usage object carries a provider-reported 'cost' field; a legitimate contract on some providers (e.g. OpenRouter usage accounting), but affected LiteLLM versions trusted it verbatim as USD without unit validation - check which applies to this record

usage_sum_mismatch: 1 record(s) - consistent with https://github.com/langfuse/langfuse/issues/14987
  line 4 [demo-sum-1]: total_tokens=999 but prompt+completion=150 (prompt=100, completion=50). Assumes the OpenAI/LiteLLM usage shape (total = prompt + completion, detail fields are subsets); if this record's schema counts tokens outside that sum, verify the schema before treating this as an error.

zero_cost_diagnostic_flagged: 1 record(s) - consistent with https://github.com/BerriAI/litellm/issues/48937
  line 10 [demo-zcd-1]: the gateway's own zero-cost diagnostic fired on this record (reason='pricing_not_applied'): LiteLLM itself determined the call was priced at $0 anomalously - the strongest form of evidence this failure class produces

zero_spend_with_usage: 5 record(s) - consistent with https://github.com/BerriAI/litellm/issues/48937
  line 2 [demo-zero-1]: spend=0 on a non-error call with 280 usage tokens; budgets enforce against spend, so such records are uncounted. Also legitimate for zero-priced models, fully cached calls billed at $0, or fully discounted calls - check whether this model has a nonzero price in your cost map before treating this as the documented failure class.
  line 3 [demo-mismatch-1]: spend=0 on a non-error call with 130 usage tokens; budgets enforce against spend, so such records are uncounted. Also legitimate for zero-priced models, fully cached calls billed at $0, or fully discounted calls - check whether this model has a nonzero price in your cost map before treating this as the documented failure class.
  line 7 [demo-or-1]: spend=0 on a non-error call with 100 usage tokens; budgets enforce against spend, so such records are uncounted. Also legitimate for zero-priced models, fully cached calls billed at $0, or fully discounted calls - check whether this model has a nonzero price in your cost map before treating this as the documented failure class.
  line 8 [demo-upstream-1]: spend=0 on a non-error call with 100 usage tokens; budgets enforce against spend, so such records are uncounted. Also legitimate for zero-priced models, fully cached calls billed at $0, or fully discounted calls - check whether this model has a nonzero price in your cost map before treating this as the documented failure class.
  line 10 [demo-zcd-1]: spend=0 on a non-error call with 180 usage tokens; budgets enforce against spend, so such records are uncounted. Also legitimate for zero-priced models, fully cached calls billed at $0, or fully discounted calls - check whether this model has a nonzero price in your cost map before treating this as the documented failure class.

A finding means 'consistent with a documented failure class' - check the linked issue against your upstream versions. It is a signal, not proof of misbilling. The usage-sum check assumes the OpenAI/LiteLLM usage shape (total = prompt + completion; detail fields are subsets); records whose schema documents a different sum are suppressed when the fields show it.
```
