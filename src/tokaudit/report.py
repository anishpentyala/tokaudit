"""Report building and rendering (JSON + Markdown)."""

from __future__ import annotations

from typing import Dict, List, Optional

from .dedup import DuplicateBlock
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
) -> Dict:
    total_input = sum(c.total_input for c in sim.per_request)
    below_min = [
        c.rid for c in sim.per_request if c.total_input < rules.min_prefix_tokens
    ]
    recommendations = _recommendations(sim, dups, rules, below_min)
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
                "repeated_tokens": d.repeated_tokens,
                "requests": sorted({rid for rid, _ in d.occurrences}),
                "preview": d.preview,
            }
            for d in dups
        ],
        "skipped_unverified_policy": [
            {"id": rid, "model": model, "reason": reason} for rid, model, reason in sim.skipped
        ],
        "below_min_cacheable": below_min,
        "recommendations": recommendations,
        "assumptions": {
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
    if s["wasted_write_tokens"]:
        lines.append(f"- Cache writes never re-read: {s['wasted_write_tokens']:,} tokens")
    lines += ["", "## Duplicate content across requests", ""]
    if report["duplicate_blocks"]:
        for d in report["duplicate_blocks"]:
            lines.append(
                f"- ~{d['tokens']} tokens x {d['occurrences']} requests "
                f"({d['repeated_tokens']} repeated tokens): \"{d['preview']}\""
            )
    else:
        lines.append("No repeated blocks above the detection threshold.")
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
