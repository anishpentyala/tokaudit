"""Offline accounting-integrity checks for LLM gateway/tracing logs.

tokaudit analyze finds token WASTE. This module answers a different question:
are the accounting fields themselves internally consistent? A steady stream
of upstream bugs shows they often are not - and when spend reads $0, budgets
silently stop enforcing (BerriAI/litellm#48937).

Method: offline consistency checks over the raw log records, each with a
stated schema assumption. No live calls, no content capture, no price
sheet. Each check is anchored to a documented upstream failure class (the
GitHub issue it reproduces the SIGNATURE of), which also provides the
synthetic-fixture reference.

Epistemics (fail closed): a finding says "this record is consistent with
documented failure class X", never "your deployment has bug X" and never
"you were misbilled". Whether a class applies depends on the user's upstream
versions; the report prints the issue link so the user can check. Legitimate
$0 spends exist (free/custom-priced models, fully cached or discounted
calls), so checks are signals with stated assumptions, not verdicts: where
a documented legitimate pattern explains the record, the check stays
quiet; otherwise the finding text names both the failure class and the
legitimate alternatives the record cannot rule out. Records we cannot read fail closed (the loader exits), matching
the analyzer's robustness contract.

Format coverage mirrors the adapters: LiteLLM SpendLogs rows (full
check set; accounting fields are top-level columns while cost_breakdown
and usage_object nest inside the row's metadata JSON - both levels are
read), LiteLLM standard-logging-payload records (zero_cost_diagnostic
check; that field exists only on that shape), Helicone flat rows and
Langfuse ObservationV2 exports (usage-sum check where the fields exist).
Request text is never read.

Issue anchors (all fetched 2026-10-30):
  #48937 streamed chat completions logged at $0 when upstream reports
        usage.cost: 0 - budgets read 0 for all streaming traffic
  #40100 streamed azure_ai spend=0 while cost_breakdown holds the real cost
  #35691 $0 spend for custom models absent from the cost map
  #35829 provider-reported usage.cost trusted verbatim as USD, no unit check
  #25351 /spend/logs inflated totals vs /spend/keys (duplicate counting)
  langfuse#14987 output count wrong when completion_tokens_details exceeds
        completion_tokens
All under https://github.com/BerriAI/litellm/issues/ resp.
https://github.com/langfuse/langfuse/issues/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LITELLM = "https://github.com/BerriAI/litellm/issues/"
LANGFUSE = "https://github.com/langfuse/langfuse/issues/"


@dataclass
class Finding:
    invariant: str
    line: int
    rid: str
    detail: str
    anchor: str  # issue URL documenting the failure class


def _num(v: Any) -> Optional[float]:
    """A real, finite, non-bool number, else None (fail closed on strings,
    NaN/Infinity - json.loads accepts those literals)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return float(v)


def _is_litellm_spendlog(obj: Dict[str, Any]) -> bool:
    return "request_id" in obj and "startTime" in obj


def _is_helicone_row(obj: Dict[str, Any]) -> bool:
    return "request_id" in obj and "request_body" in obj


def _is_langfuse_observation(obj: Dict[str, Any]) -> bool:
    return "usageDetails" in obj or "usage_details" in obj


def _usage_triple(obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(prompt, completion, total) token numbers from any of the supported
    record shapes; None where absent or non-numeric."""
    if _is_litellm_spendlog(obj):
        return (_num(obj.get("prompt_tokens")),
                _num(obj.get("completion_tokens")),
                _num(obj.get("total_tokens")))
    if _is_helicone_row(obj):
        return (_num(obj.get("prompt_tokens")),
                _num(obj.get("completion_tokens")),
                None)
    details = obj.get("usageDetails") or obj.get("usage_details") or {}
    if isinstance(details, dict):
        prompt = _num(details.get("input"))
        completion = _num(details.get("output"))
        total = _num(details.get("total"))
        return prompt, completion, total
    return None, None, None


def check_record(obj: Dict[str, Any], line: int) -> List[Finding]:
    """Apply every applicable check to one raw record."""
    findings: List[Finding] = []
    rid = str(obj.get("request_id") or obj.get("id") or f"line-{line}")

    prompt, completion, total = _usage_triple(obj)

    # I3: usage sum mismatch - total disagrees with prompt+completion.
    # (langfuse#14987 class; also fires on LiteLLM/Helicone rows, where a
    # mismatch means the gateway's own fields disagree.)
    # Schema assumption (stated in the finding): the OpenAI/LiteLLM usage
    # shape, where total = prompt + completion and detail fields (cached,
    # reasoning, audio) are SUBSETS of prompt/completion - verified against
    # LiteLLM's anthropic transformation, which folds cache_read/cache_creation
    # into prompt_tokens and sets total = prompt + completion. A different
    # mapper may count cache tokens OUTSIDE that sum; when explicit cache
    # fields exactly explain the gap, that documented alternative holds and
    # the check stays quiet instead of accusing.
    if prompt is not None and completion is not None and total is not None:
        gap = total - (prompt + completion)
        if abs(gap) > 1:
            cache_extra = 0.0
            for k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                cache_extra += _num(obj.get(k)) or 0.0
            details = obj.get("usageDetails") or obj.get("usage_details")
            if isinstance(details, dict):
                for k, v in details.items():
                    if "cache" in k:
                        cache_extra += _num(v) or 0.0
            if not (cache_extra > 0 and abs(gap - cache_extra) <= 1):
                findings.append(Finding(
                    "usage_sum_mismatch", line, rid,
                    f"total_tokens={total:g} but prompt+completion="
                    f"{prompt + completion:g} (prompt={prompt:g}, "
                    f"completion={completion:g}). Assumes the OpenAI/LiteLLM "
                    "usage shape (total = prompt + completion, detail fields "
                    "are subsets); if this record's schema counts tokens "
                    "outside that sum, verify the schema before treating "
                    "this as an error.",
                    LANGFUSE + "14987"))

    if not _is_litellm_spendlog(obj):
        return findings

    spend = _num(obj.get("spend"))

    # I1: zero spend on an apparently successful, token-bearing call.
    # Budgets enforce against spend, so $0 records bypass them (#48937).
    # Not flagged: errored calls (a $0 error record is correct - the #4604
    # fix direction) and records with no usage to bill.
    failed = bool(obj.get("error") or obj.get("status") == "failure")
    billed_tokens = (total if total is not None else None) or (
        (prompt or 0) + (completion or 0)
        if (prompt is not None or completion is not None) else None)
    if spend == 0.0 and billed_tokens and billed_tokens > 0 and not failed:
        findings.append(Finding(
            "zero_spend_with_usage", line, rid,
            f"spend=0 on a non-error call with {billed_tokens:g} usage "
            "tokens; budgets enforce against spend, so such records are "
            "uncounted. Also legitimate for zero-priced models, fully "
            "cached calls billed at $0, or fully discounted calls - "
            "check whether this model has a nonzero price in your cost "
            "map before treating this as the documented failure class.",
            LITELLM + "48937"))

    # I2: spend disagrees with the record's own cost_breakdown (#40100).
    # Real SpendLogs rows nest cost_breakdown inside the metadata JSON;
    # standard-logging-payload records carry it top-level. Read both.
    metadata = obj.get("metadata")
    breakdown = obj.get("cost_breakdown")
    if breakdown is None and isinstance(metadata, dict):
        breakdown = metadata.get("cost_breakdown")
    if isinstance(breakdown, dict):
        cb_total = _num(breakdown.get("total_cost"))
        if spend is not None and cb_total is not None:
            if abs(spend - cb_total) > max(1e-9, 1e-6 * abs(cb_total)):
                findings.append(Finding(
                    "spend_cost_breakdown_disagreement", line, rid,
                    f"spend={spend:g} but cost_breakdown.total_cost="
                    f"{cb_total:g} in the same record (beyond float "
                    "tolerance). The record's own fields disagree; "
                    "legitimate explanations include discounts or a unit "
                    "mismatch - a signal to inspect, not proof of "
                    "misbilling.",
                    LITELLM + "40100"))

    # I6: an upstream-reported cost field riding inside the usage chunk
    # (#35829: trusted verbatim as USD with no unit validation). Presence
    # alone is the flag - no magnitude math, no false-positive arithmetic.
    # Legitimate contract on some providers: OpenRouter returns a usage cost
    # field as part of its usage-accounting API, so a record that names
    # OpenRouter as its provider is suppressed rather than flagged.
    # Real SpendLogs rows nest the usage dict at metadata.usage_object.
    usage_obj = obj.get("usage")
    if usage_obj is None and isinstance(metadata, dict):
        usage_obj = metadata.get("usage_object")
    if isinstance(usage_obj, dict) and "cost" in usage_obj:
        provider = str(obj.get("custom_llm_provider")
                       or obj.get("llm_provider") or "").lower()
        if "openrouter" not in provider:
            findings.append(Finding(
                "upstream_cost_field_in_usage", line, rid,
                "usage object carries a provider-reported 'cost' field; "
                "a legitimate contract on some providers (e.g. OpenRouter "
                "usage accounting), but affected LiteLLM versions trusted "
                "it verbatim as USD without unit validation - check which "
                "applies to this record",
                LITELLM + "35829"))

    # I8: litellm's OWN zero-cost diagnostic is present on the record.
    # This is provider-recorded evidence, not a heuristic: the gateway
    # itself computed $0 and flagged why (#48937: spend and budgets read
    # 0 for all streaming traffic while the diagnostic says
    # "pricing_not_applied"). The field exists on standard-logging
    # payloads; SpendLogs rows do not carry it.
    zcd = obj.get("zero_cost_diagnostic")
    if zcd is None and isinstance(metadata, dict):
        zcd = metadata.get("zero_cost_diagnostic")
    if isinstance(zcd, dict) and isinstance(zcd.get("reason"), str):
        findings.append(Finding(
            "zero_cost_diagnostic_flagged", line, rid,
            "the gateway's own zero-cost diagnostic fired on this record "
            f"(reason={zcd['reason']!r}): LiteLLM itself determined the "
            "call was priced at $0 anomalously - the strongest form of "
            "evidence this failure class produces",
            LITELLM + "48937"))

    # I7: accounting fields present but non-numeric/non-finite (strings,
    # NaN, Infinity) - ingestion-corruption class; the analyzer already
    # skips such usage conservatively, here it is surfaced as a finding.
    for field_name in ("spend", "prompt_tokens", "completion_tokens", "total_tokens"):
        raw = obj.get(field_name)
        if raw is not None and _num(raw) is None:
            findings.append(Finding(
                "non_numeric_accounting_field", line, rid,
                f"{field_name}={raw!r} is not a finite number",
                LITELLM + "25351"))
    return findings


def check_duplicate_spend(records: List[Tuple[int, Dict[str, Any]]]) -> List[Finding]:
    """I5: the same request_id counted with spend>0 on more than one row
    (#25351-class double counting inflates spend totals)."""
    seen: Dict[str, Tuple[int, float]] = {}
    findings: List[Finding] = []
    for line, obj in records:
        if not _is_litellm_spendlog(obj):
            continue
        rid = obj.get("request_id")
        spend = _num(obj.get("spend"))
        if not isinstance(rid, str) or not rid or spend is None or spend <= 0:
            continue
        if rid in seen:
            first_line, first_spend = seen[rid]
            findings.append(Finding(
                "duplicate_spend_record", line, rid,
                f"request_id already billed ${first_spend:g} at line "
                f"{first_line}; this row bills ${spend:g} again. Verify "
                "these are distinct billable calls - retried or streamed "
                "re-emissions sharing one request_id can look identical "
                "without being double billing.",
                LITELLM + "25351"))
        else:
            seen[rid] = (line, spend)
    return findings


def run_integrity(path: str) -> List[Finding]:
    """Check one JSONL log; unreadable structure fails closed (SystemExit),
    matching the analyzer's robustness contract."""
    records: List[Tuple[int, Dict[str, Any]]] = []
    findings: List[Finding] = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{path}: {exc.strerror or exc}")
    with fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{idx + 1}: invalid JSON: {exc}")
            except RecursionError:
                raise SystemExit(f"{path}:{idx + 1}: JSON nested too deeply to parse")
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{idx + 1}: expected a JSON object per line")
            records.append((idx + 1, obj))
            findings.extend(check_record(obj, idx + 1))
    findings.extend(check_duplicate_spend(records))
    return findings
