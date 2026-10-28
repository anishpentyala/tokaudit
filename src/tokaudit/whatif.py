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

SCENARIOS = ("dense", "dedup")


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
    """Ceiling scenario: the same traffic compressed so nothing expires.

    Timestamps are rescaled (order and relative spacing preserved) so the
    log's full span fits within the shortest cache lifetime among the
    families in use. The result is an upper bound on cache benefit if
    traffic were dense enough that entries never expired.
    """
    caveats = [
        "Upper-bound scenario, not a prediction. Real hit rates also depend on "
        "provider prefix routing, which is documented as best-effort.",
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
            "title": "Dense-traffic ceiling",
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
            "title": "Dense-traffic ceiling",
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
        "title": "Dense-traffic ceiling",
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

    A duplicate is exact only when the model and the full canonically
    rendered prompt match. Sampling parameters are not part of the log
    format, so two identical requests may still have been intentional
    independent samples; that judgment stays with the user.
    """
    caveats = [
        "Collapsing duplicates is only safe when the duplicates are "
        "interchangeable. The log records model and prompt but not sampling "
        "parameters (e.g. temperature), so identical requests may have been "
        "deliberate independent samples. Treat the saving as available only "
        "after confirming the duplicates were redundant.",
        "Non-mutating: the as-is simulation above is unchanged; this scenario "
        "shows what the cost would have been without the exact repeats.",
    ]
    seen: Dict[Tuple[str, str], str] = {}
    kept: List[Request] = []
    dropped: List[str] = []
    for r in requests:
        key = (r.model, r.canonical())
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


_RUNNERS = {"dense": _dense, "dedup": _dedup}


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
