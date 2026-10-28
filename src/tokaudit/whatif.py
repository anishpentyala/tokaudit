"""What-if scenarios: non-mutating simulations over hypothetical conditions.

Each scenario re-runs the cache simulation on a transformed *copy* of the
request list and reports its own numbers next to the as-is simulation.
Nothing here rewrites the log, and no scenario claims that changing real
traffic is safe for the application: whether requests may be reordered,
collapsed, or rescheduled depends on the caller, which the log cannot see.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

from .logs import Request
from .prefix import SimulationResult, simulate
from .rules import ProviderRules

SCENARIOS = ("anthropic-1h", "dense", "dedup")


def _summarize(sim: SimulationResult) -> Dict:
    return {
        "baseline_units": round(sim.baseline_units, 1),
        "simulated_units": round(sim.simulated_units, 1),
        "simulated_savings_units": round(sim.savings_units, 1),
        "simulated_savings_pct": round(sim.savings_pct, 1),
    }


def _dense(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    rules: ProviderRules,
    force_family: Optional[str],
    isolate_sessions: bool,
) -> Dict:
    """Hypothetical compressed-timing scenario: same traffic, nothing expires.

    Timestamps are rescaled (order and relative spacing preserved) so the
    log's full span fits within the shortest cache lifetime among the
    families in use. Within this simulator's model, shrinking gaps while
    preserving order can only keep entries alive longer (a hit at gap g
    survives any gap <= g, and refresh-on-read only extends lifetimes), so
    its savings are never below the as-is simulation. That is a property of
    the model, not of real traffic.
    """
    caveats = [
        "Hypothetical timing scenario, not a prediction and not a real-world "
        "upper bound: the never-worse property is proven only for this "
        "simulator's model. Real hit rates also depend on provider prefix "
        "routing (documented as best-effort), concurrency, and write pricing.",
        "Whether requests may safely be rescheduled or sent closer together is "
        "the caller's decision; the log cannot show dependencies between them. "
        "This scenario does not claim any rescheduling preserves application "
        "behavior.",
    ]
    timed = [r for r in requests if r.ts is not None]
    if len(timed) != len(requests):
        sim = simulate(requests, encode, rules, force_family=force_family,
                       isolate_sessions=isolate_sessions)
        return {
            "name": "dense",
            "title": "Compressed-timing scenario",
            "result": _summarize(sim),
            "caveats": caveats,
            "notes": ["Log has requests without timestamps, so the as-is "
                      "simulation already treats nothing as expiring; the "
                      "dense scenario is identical to it."],
        }

    # Shortest TTL among the families that would actually be simulated.
    ttls: List[float] = []
    for r in requests:
        if force_family is not None:
            fam = rules.families[force_family]
        else:
            fam = rules.resolve_family(r.model).family
        if fam.savings_supported:
            ttls.append(float(fam.ttl_seconds))
    span = max(r.ts for r in timed) - min(r.ts for r in timed)
    if not ttls or span <= 0 or span <= min(ttls):
        sim = simulate(requests, encode, rules, force_family=force_family,
                       isolate_sessions=isolate_sessions)
        return {
            "name": "dense",
            "title": "Compressed-timing scenario",
            "result": _summarize(sim),
            "caveats": caveats,
            "notes": ["Log span already fits inside the shortest cache "
                      "lifetime in use; the dense scenario is identical to "
                      "the as-is simulation."],
        }

    t0 = min(r.ts for r in timed)
    scale = min(ttls) / span
    compressed = [replace(r, ts=t0 + (r.ts - t0) * scale) for r in requests]
    sim = simulate(compressed, encode, rules, force_family=force_family,
                   isolate_sessions=isolate_sessions)
    return {
        "name": "dense",
        "title": "Compressed-timing scenario",
        "result": _summarize(sim),
        "caveats": caveats,
        "notes": [
            f"Timestamps rescaled by {scale:.4g}x (span {span:.0f}s -> "
            f"{min(ttls):.0f}s, the shortest cache lifetime in use); request "
            "order and relative spacing preserved."
        ],
    }


def _dedup(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    rules: ProviderRules,
    force_family: Optional[str],
    isolate_sessions: bool,
) -> Dict:
    """Scenario: exact duplicate requests collapsed to their first occurrence.

    A duplicate is exact only when everything the log represents matches:
    model, full canonically rendered prompt, prompt_cache_key, and
    session_id. Sampling parameters are not part of the log format, so two
    identical requests may still have been intentional independent samples;
    that judgment stays with the user.
    """
    caveats = [
        "Collapsing duplicates is only safe when the duplicates are "
        "interchangeable. Distinctions the log represents (model, prompt, "
        "prompt_cache_key, session_id) are preserved, but the log does not "
        "record organization, region, or sampling parameters (e.g. "
        "temperature), so identical requests may have been deliberate "
        "independent samples or billed to different accounts. Treat the "
        "saving as available only after confirming the duplicates were "
        "redundant.",
        "Non-mutating: the as-is simulation above is unchanged; this scenario "
        "shows what the cost would have been without the exact repeats.",
    ]
    seen: Dict[Tuple[str, str], str] = {}
    kept: List[Request] = []
    dropped: List[str] = []
    for r in requests:
        key = (r.model, r.canonical(), r.cache_key, r.session_id)
        if key in seen:
            dropped.append(r.rid)
        else:
            seen[key] = r.rid
            kept.append(r)
    sim = simulate(kept, encode, rules, force_family=force_family,
                   isolate_sessions=isolate_sessions)
    notes = []
    if dropped:
        notes.append(
            f"{len(dropped)} exact duplicate request(s) collapsed "
            f"({len(kept)} of {len(requests)} kept): {', '.join(dropped)}."
        )
    else:
        notes.append("No exact duplicate requests found; the scenario is "
                     "identical to the as-is simulation.")
    return {
        "name": "dedup",
        "title": "Exact-duplicate collapse",
        "result": _summarize(sim),
        "caveats": caveats,
        "notes": notes,
    }


def _anthropic_1h(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    rules: ProviderRules,
    force_family: Optional[str],
    isolate_sessions: bool,
) -> Dict:
    """Scenario: Anthropic's documented 1-hour cache TTL instead of 5 minutes.

    Anthropic documents 5-minute writes at 1.25x base input and 1-hour
    writes at 2x base input, reads at 0.1x either way (rules/EVIDENCE.md).
    The scenario re-simulates with ttl 3600s and write multiplier 2.0 so
    the pricier writes and the longer reuse window are weighed together.
    """
    if rules.name != "anthropic":
        raise SystemExit(
            "--what-if anthropic-1h applies to Anthropic rules only "
            f"(this group uses '{rules.name}')."
        )
    caveats = [
        "Using the 1-hour TTL is a real request-time option in Anthropic's "
        "API, but whether its writes pay off depends on actual reuse landing "
        "5-60 minutes after each write; this simulation of your log cannot "
        "predict future traffic.",
        "Refresh-on-read is applied to 1-hour entries exactly as documented "
        "for the default 5-minute cache; our evidence does not separately "
        "document refresh behavior for 1-hour entries, so treat hits near "
        "the end of the window as optimistic.",
        "Non-mutating: the as-is simulation above is unchanged.",
    ]
    one_hour = replace(
        rules,
        ttl_seconds=3600,
        write_multiplier=2.0,
        families={
            name: replace(fam, ttl_seconds=3600, write_multiplier=2.0)
            for name, fam in rules.families.items()
        },
    )
    sim = simulate(requests, encode, one_hour, force_family=force_family,
                   isolate_sessions=isolate_sessions)
    notes = [
        "Re-simulated with the documented 1-hour cache option: TTL 3600s, "
        "writes at 2x base input (vs 1.25x for the default 5-minute cache), "
        "reads unchanged at 0.1x (rules/EVIDENCE.md, Anthropic prompt-caching "
        "doc)."
    ]
    if sim.skipped:
        notes.append(
            f"{len(sim.skipped)} request(s) remain excluded (unverified "
            "policy), same as the as-is simulation."
        )
    return {
        "name": "anthropic-1h",
        "title": "Anthropic 1-hour cache scenario",
        "result": _summarize(sim),
        "caveats": caveats,
        "notes": notes,
    }


_RUNNERS = {"anthropic-1h": _anthropic_1h, "dense": _dense, "dedup": _dedup}


def run_scenarios(
    names: List[str],
    requests: List[Request],
    encode: Callable[[str], List[int]],
    rules: ProviderRules,
    force_family: Optional[str] = None,
    isolate_sessions: bool = False,
) -> List[Dict]:
    out = []
    for name in names:
        if name not in _RUNNERS:
            raise SystemExit(
                f"unknown what-if scenario '{name}'; known: {sorted(_RUNNERS)}"
            )
        out.append(_RUNNERS[name](requests, encode, rules, force_family,
                                  isolate_sessions))
    return out
