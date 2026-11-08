"""Report building and rendering (JSON + Markdown)."""

from __future__ import annotations

from typing import Dict, List, Optional

from .dedup import DuplicateBlock, DuplicateSearchResult
from .logs import Request
from .prefix import SimulationResult
from .rules import ProviderRules, Ruleset


def build_report(
    requests: List[Request],
    sim: SimulationResult,
    dups: List[DuplicateBlock],
    ruleset: Ruleset,
    rules: ProviderRules,
    tokenizer_name: str,
    scenarios: Optional[List[Dict]] = None,
    price_sheet: Optional[Dict[str, Dict[str, float]]] = None,
    price_source: Optional[str] = None,
    omit_content_previews: bool = False,
    duplicate_search: Optional[DuplicateSearchResult] = None,
) -> Dict:
    total_input = sum(c.total_input for c in sim.per_request)
    below_min = [
        c.rid for c in sim.per_request if c.total_input < rules.min_prefix_tokens
    ]
    recommendations = _recommendations(sim, dups, rules, below_min)
    pricing = None
    if price_sheet is not None:
        from .prices import cost_usd
        from collections import Counter
        id_counts = Counter(req.rid for req in requests)
        models_by_rid = {req.rid: req.model for req in requests
                         if id_counts[req.rid] == 1}
        priced = {"baseline_usd": 0.0, "simulated_usd": 0.0, "priced_requests": 0,
                  "unpriced": []}
        for cst in sim.per_request:
            model = cst.model or models_by_rid.get(cst.rid, "")
            rates = price_sheet.get(model)
            if rates is None:
                priced["unpriced"].append(cst.rid)
                continue
            priced["priced_requests"] += 1
            priced["baseline_usd"] += cost_usd(cst.total_input, 0, 0, rates)
            priced["simulated_usd"] += cost_usd(cst.uncached, cst.read, cst.written, rates)
        import math
        if not all(math.isfinite(priced[key]) for key in ("baseline_usd", "simulated_usd")):
            raise ValueError("simulated price aggregate exceeds finite dollar range")
        pricing = {
            "source": price_source,
            "baseline_usd": round(priced["baseline_usd"], 4),
            "simulated_usd": round(priced["simulated_usd"], 4),
            "simulated_savings_usd": round(priced["baseline_usd"] - priced["simulated_usd"], 4),
            "priced_requests": priced["priced_requests"],
            "unpriced_requests": priced["unpriced"],
            "note": ("Dollar figures apply the user-supplied price sheet to "
                     "simulated token flows; still simulated, not observed "
                     "billing, and the sheet itself is not provider-verified."),
        }
    return {
        "summary": {
            "requests": len(requests),
            "requests_simulated": len(sim.per_request),
            "requests_skipped_unverified_policy": len(sim.skipped),
            "provider": sim.provider,
            "tokenizer": tokenizer_name,
            "total_input_tokens": total_input,
            "baseline_units": round(sim.baseline_units, 1),
            "simulated_units": round(sim.simulated_units, 1),
            "simulated_savings_units": round(sim.savings_units, 1),
            "simulated_savings_pct": round(sim.savings_pct, 1),
            "wasted_write_tokens": sim.wasted_write_tokens,
        },
        "cache_simulation": [
            {
                "id": c.rid,
                "total_input": c.total_input,
                "cache_read": c.read,
                "cache_write": c.written,
                "uncached": c.uncached,
            }
            for c in sim.per_request
        ],
        "duplicate_blocks": [
            {
                "tokens": d.token_count,
                "occurrences": len(d.occurrences),
                "distinct_requests": d.distinct_requests,
                "request_indices": d.request_indices,
                "token_spans": d.token_spans,
                "preview_scope": "source_message_start_not_matched_span",
                "repeated_tokens": d.repeated_tokens,
                "requests": sorted({rid for rid, _ in d.occurrences}),
                **({} if omit_content_previews else {"preview": d.preview}),
            }
            for d in dups
        ],
        **({"duplicate_search": {
            "indexed_shingles": duplicate_search.indexed_shingles,
            "eligible_shingles": duplicate_search.eligible_shingles,
            "max_shingles": duplicate_search.max_shingles,
            "index_complete": duplicate_search.index_complete,
            "note": ("Coverage describes indexed token windows, not exhaustive duplicate "
                     "discovery or total memory use. Detection is heuristic and returns "
                     "only the top blocks after suppression. A capped index can miss "
                     "later requests; all messages are still tokenized."),
        }} if duplicate_search is not None else {}),
        "skipped_unverified_policy": [
            {"id": rid, "model": model, "reason": reason} for rid, model, reason in sim.skipped
        ],
        "below_min_cacheable": below_min,
        "what_if": scenarios or [],
        "pricing": pricing,
        "provider_reported": _provider_reported(requests, sim),
        "recommendations": recommendations,
        "assumptions": {
            **({"content_previews_omitted": True} if omit_content_previews else {}),
            "rules_as_of": ruleset.as_of,
            "sources": ruleset.sources,
            "provider_notes": rules.notes,
            "simulation_notes": sim.notes,
            "tokenizer_note": (
                "Exact counts (tiktoken)." if tokenizer_name == "exact"
                else "Estimated counts (about 4 chars/token, +/-25% by content type); "
                     "install tokaudit[exact] for exact OpenAI counts."
            ),
            "render_note": (
                "Requests are rendered with a canonical template, not the provider's "
                "exact chat template; relative reuse patterns are preserved."
            ),
            "cost_note": (
                "Units are multiples of the base input-token rate, not dollars, so "
                "results stay valid as dollar prices change."
            ),
            "observation_note": (
                "Savings are SIMULATED from your log against the stated provider rules; "
                "they are not observed billing data. Confirm against the provider's "
                "usage fields (cache_read_input_tokens / cached_tokens) in production."
            ),
        },
    }


def _provider_reported(requests: List[Request], sim: SimulationResult) -> Optional[Dict]:
    """Compare provider-reported usage with the simulation when the export
    carried real usage fields (Helicone rows today). Providers count with
    their own tokenizers and hidden context, so exact equality is not
    expected; large divergence means the simulation or the log's coverage
    should be questioned."""
    from collections import Counter

    with_usage = [r for r in requests if r.actual_usage]
    if not with_usage:
        return None
    request_counts = Counter(r.rid for r in requests)
    simulation_counts = Counter(c.rid for c in sim.per_request)
    ambiguous = sorted(rid for rid in request_counts.keys() | simulation_counts.keys()
                       if request_counts[rid] > 1 or simulation_counts[rid] > 1)
    costs = {c.rid: c for c in sim.per_request if c.rid not in ambiguous}
    matched = [(r, costs[r.rid]) for r in with_usage
               if r.rid in costs and r.rid not in ambiguous]
    result = {
        "requests_with_usage": len(with_usage),
        "requests_compared": len(matched),
        "ambiguous_request_ids": ambiguous,
        "coverage": {},
        "per_request": [],
        "note": ("Provider-reported fields come from the export; missing fields are unknown, "
                 "not zero. Each field compares only requests with that explicit field and "
                 "one uniquely matching simulated request. Requests skipped by simulation "
                 "and ambiguous duplicate IDs are excluded. Provider tokenizers and hidden "
                 "context differ from the canonical render, so exact equality is not expected. "
                 "Divergence alone is not evidence of a billing discrepancy."),
    }
    # Aggregates can hide equal-and-opposite errors. Keep field-specific
    # request evidence; absence is not zero and never contributes a delta.
    fields = (("cache_read_tokens", "read"), ("cache_write_tokens", "written"),
              ("prompt_tokens", "total_input"))
    for request, cost in matched:
        comparisons = {}
        for field, attribute in fields:
            if field in request.actual_usage:
                actual = request.actual_usage[field]
                simulated = getattr(cost, attribute)
                comparisons[field] = {"reported": actual, "simulated": simulated,
                                      "simulated_minus_reported": simulated - actual}
        if comparisons:
            result["per_request"].append({"id": request.rid, "model": request.model,
                                           "fields": comparisons})
    for field, actual_key, simulated_key, attribute in (
        ("cache_read_tokens", "actual_cache_read_tokens", "simulated_cache_read_tokens", "read"),
        ("cache_write_tokens", "actual_cache_write_tokens", "simulated_cache_write_tokens", "written"),
        ("prompt_tokens", "actual_prompt_tokens", "simulated_prompt_tokens", "total_input"),
    ):
        covered = [(r, c) for r, c in matched if field in r.actual_usage]
        result["coverage"][field] = len(covered)
        result[actual_key] = sum(r.actual_usage[field] for r, _ in covered) if covered else None
        result[simulated_key] = sum(getattr(c, attribute) for _, c in covered) if covered else None
    return result


def _recommendations(
    sim: SimulationResult,
    dups: List[DuplicateBlock],
    rules: ProviderRules,
    below_min: List[str],
) -> List[str]:
    recs: List[str] = []
    total_repeated = sum(d.repeated_tokens for d in dups)
    if dups:
        recs.append(
            f"Hoist {len(dups)} repeated block(s) (~{total_repeated} repeated tokens) "
            "into one stable cached prefix (system or developer message) instead of "
            "resending them per request."
        )
    if sim.wasted_write_tokens:
        recs.append(
            "Some cached prefixes were never re-used. Only write prefixes you expect "
            "to reuse within the cache lifetime; a single-use write costs more than "
            "sending uncached."
        )
    if below_min:
        recs.append(
            f"{len(below_min)} request(s) fall below the minimum cacheable length "
            f"({rules.min_prefix_tokens} tokens). If they share a stable prefix, "
            "reaching the minimum can unlock cache pricing."
        )
    volatile = any("timestamp" in n.lower() or "volatile" in n.lower() for n in sim.notes)
    if sim.savings_pct < 5 and not recs:
        recs.append(
            "Little reusable prefix found. Put static content (instructions, docs, "
            "few-shot examples) first and volatile content (timestamps, per-request "
            "context) last so the shared prefix is as long as possible."
        )
    if not volatile and sim.savings_pct >= 5:
        recs.append(
            "Keep volatile fields (timestamps, request IDs) out of the cached prefix "
            "to preserve the hit rate this simulation estimates."
        )
    return recs


def to_markdown(report: Dict) -> str:
    s = report["summary"]
    lines = [
        "# tokaudit report",
        "",
        "## Summary",
        "",
        f"- Requests analyzed: {s['requests']}",
        f"- Provider rules: {s['provider']} (rules as of {report['assumptions']['rules_as_of']})",
        f"- Tokenizer: {s['tokenizer']}",
        f"- Requests in savings simulation: {s['requests_simulated']}"
        + (f" ({s['requests_skipped_unverified_policy']} skipped, unverified policy)"
           if s['requests_skipped_unverified_policy'] else ""),
        f"- Total input tokens: {s['total_input_tokens']:,}",
        f"- Baseline cost: {s['baseline_units']:,.0f} units",
        f"- With caching: {s['simulated_units']:,.0f} units",
        f"- Simulated savings (not observed billing): {s['simulated_savings_units']:,.0f} units "
        f"({s['simulated_savings_pct']}%)",
    ]
    if report.get("pricing"):
        p = report["pricing"]
        lines.append(
            f"- From your price sheet ({p['source']}): baseline \${p['baseline_usd']:,.4f}, "
            f"with caching \${p['simulated_usd']:,.4f}, simulated savings "
            f"\${p['simulated_savings_usd']:,.4f} "
            f"({p['priced_requests']} requests priced"
            + (f", {len(p['unpriced_requests'])} skipped - model not in sheet"
               if p['unpriced_requests'] else "") + ")"
        )
        lines.append(f"  - {p['note']}")
    if s["wasted_write_tokens"]:
        lines.append(f"- Cache writes never re-read: {s['wasted_write_tokens']:,} tokens")
    lines += ["", "## Duplicate content across requests", ""]
    if report["assumptions"].get("content_previews_omitted"):
        lines.append("Content previews omitted. IDs, metadata and aggregate statistics remain; this is not anonymization.")
    coverage = report.get("duplicate_search")
    if coverage is not None:
        state = "complete" if coverage["index_complete"] else "CAPPED - partial search"
        lines.append(f"Shingle index: {coverage['indexed_shingles']:,} / "
                     f"{coverage['eligible_shingles']:,} eligible windows ({state}).")
        lines.append(coverage["note"])
    if report["duplicate_blocks"]:
        lines.append("Locations use zero-based request-record and message positions within this "
                     "analysis group and half-open token ranges for the selected tokenizer, not "
                     "character offsets. Source message context is not the matched-span preview.")
        lines.append("")
        for d in report["duplicate_blocks"]:
            lines.append(
                f"- ~{d['tokens']} tokens x {d['occurrences']} message occurrences "
                f"across {d.get('distinct_requests', len(d['requests']))} request records "
                f"({d['repeated_tokens']} repeated tokens)"
                + (f": source message context \"{d['preview']}\"" if 'preview' in d else ": content preview omitted")
            )
            for span in d.get("token_spans", []):
                lines.append(
                    f"    - Request record {span['request_index']}, message {span['message_index']}: "
                    f"tokens [{span['token_start']}, {span['token_end']})"
                )
    else:
        if coverage is not None and not coverage["index_complete"]:
            lines.append("No repeated blocks found in the indexed subset; "
                         "unindexed windows may contain duplicates.")
        else:
            lines.append("No repeated blocks found above the detection threshold "
                         "by the heuristic search.")
    if report["what_if"]:
        lines += ["", "## What-if scenarios (hypothetical, non-mutating)", ""]
        for sc in report["what_if"]:
            r = sc["result"]
            lines.append(f"### {sc['title']}")
            lines.append("")
            lines.append(
                f"- Simulated cost in this scenario: {r['simulated_units']:,.0f} units "
                f"(savings {r['simulated_savings_units']:,.0f} units, "
                f"{r['simulated_savings_pct']}%)"
            )
            for n in sc["notes"]:
                lines.append(f"- {n}")
            for c in sc["caveats"]:
                lines.append(f"- Caveat: {c}")
            lines.append("")
    if report.get("provider_reported"):
        pr = report["provider_reported"]
        lines += ["", "## Provider-reported vs simulated", ""]
        lines.append(f"- Requests with provider usage fields: {pr['requests_with_usage']}; "
                     f"uniquely matched to simulation: {pr['requests_compared']}")
        for label, field, actual, simulated in (
            ("Cache read tokens", "cache_read_tokens", "actual_cache_read_tokens", "simulated_cache_read_tokens"),
            ("Cache write tokens", "cache_write_tokens", "actual_cache_write_tokens", "simulated_cache_write_tokens"),
            ("Prompt tokens", "prompt_tokens", "actual_prompt_tokens", "simulated_prompt_tokens"),
        ):
            count = pr['coverage'][field]
            values = (f"provider {pr[actual]:,} vs simulated {pr[simulated]:,}" if count
                      else "unknown (no comparable reported field)")
            lines.append(f"- {label}: {values}; coverage {count} request(s)")
        if pr['ambiguous_request_ids']:
            lines.append(f"- Ambiguous request IDs excluded: {len(pr['ambiguous_request_ids'])}")
        lines.append(f"- {pr['note']}")
        details = pr.get("per_request", [])
        if details:
            # Escape untrusted identifiers as one literal Markdown table cell.
            import html
            def cell(value):
                return "<code>" + (html.escape(str(value), quote=False)
                    .replace("`", "&#96;").replace("|", "&#124;")
                    .replace("\r", r"\r").replace("\n", r"\n")) + "</code>"
            lines += ["", "### Per-request usage comparison", "",
                      "Signed delta = simulated minus provider-reported tokens. "
                      "Missing fields stay unknown. These are diagnostic differences, not billing findings.", ""]
            if len(details) > 20:
                lines.append(f"First 20 of {len(details)} comparable requests shown; "
                             "JSON retains all per-request comparisons.")
                lines.append("")
            lines += ["| Request | Model | Field | Reported | Simulated | Delta |",
                      "| --- | --- | --- | ---: | ---: | ---: |"]
            for item in details[:20]:
                for field, values in item["fields"].items():
                    lines.append(f"| {cell(item['id'])} | {cell(item['model'])} | {field} | "
                                 f"{values['reported']:,} | {values['simulated']:,} | "
                                 f"{values['simulated_minus_reported']:+,} |")
    lines += ["", "## Recommendations", ""]
    for r in report["recommendations"]:
        lines.append(f"- {r}")
    lines += ["", "## Assumptions and sources", ""]
    a = report["assumptions"]
    lines.append(f"- {a['tokenizer_note']}")
    lines.append(f"- {a['render_note']}")
    lines.append(f"- {a['cost_note']}")
    lines.append(f"- {a['observation_note']}")
    for note in a["simulation_notes"]:
        lines.append(f"- {note}")
    for note in a["provider_notes"]:
        lines.append(f"- {note}")
    for name, url in a["sources"].items():
        lines.append(f"- Source ({name}): {url}")
    lines.append("")
    return "\n".join(lines)
