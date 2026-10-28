"""Prompt-cache simulation over a sequence of requests.

Model (documented approximation, not a provider oracle):

- Each request is rendered canonically (see logs.py) and tokenized.
- Eligible breakpoints are cumulative token counts at message boundaries
  that meet the provider's minimum cacheable length. Explicit breakpoints
  recorded in the log restrict writes to the marked boundaries; otherwise
  one implicit write is assumed at the full prompt (implicit OpenAI
  caching / Anthropic top-level automatic caching).
- A read hits when an unexpired cached entry is a prefix of the request;
  the longest match wins. Reads are charged at ``read_multiplier``.
- Writes are charged on the *delta*: tokens beyond the longest cached
  prefix up to the furthest new breakpoint, at ``write_multiplier``.
  Intermediate boundary entries become readable without extra charge
  (approximation of provider breakpoint selection).
- Entries expire ``ttl_seconds`` after writing; providers with
  ``refresh_on_read`` renew the TTL on every read.
- Cost is in units of the base input-token rate (uncached 1.0x). No
  dollar prices are baked in, so results survive price changes.

Requests without timestamps are treated as one cache lifetime apart from
expiry concerns (log order, nothing expires); a note records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .logs import Request
from .rules import ProviderRules


@dataclass
class RequestCost:
    rid: str
    total_input: int
    read: int
    written: int
    uncached: int
    baseline_units: float
    simulated_units: float


@dataclass
class SimulationResult:
    provider: str
    per_request: List[RequestCost]
    baseline_units: float
    simulated_units: float
    wasted_write_tokens: int
    notes: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def savings_units(self) -> float:
        return self.baseline_units - self.simulated_units

    @property
    def savings_pct(self) -> float:
        if self.baseline_units <= 0:
            return 0.0
        return 100.0 * self.savings_units / self.baseline_units


def simulate(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    rules: ProviderRules,
    force_family: Optional[str] = None,
) -> SimulationResult:
    ordered = sorted(requests, key=lambda r: (r.ts is None, r.ts or 0.0))
    no_ts = any(r.ts is None for r in ordered)
    notes: List[str] = []
    if no_ts:
        notes.append(
            "Some requests lack timestamps; treated as arriving in log order with nothing expiring."
        )
    if not rules.enabled_by_default:
        notes.append(
            f"{rules.name}: caching needs cache_control breakpoints; requests without markers "
            "were simulated in automatic mode (breakpoint at the end of the prompt)."
        )

    if force_family is not None and force_family not in rules.families:
        raise SystemExit(
            f"unknown policy family '{force_family}' for {rules.name}; "
            f"known: {sorted(rules.families)}"
        )
    if force_family is not None and not rules.families[force_family].savings_supported:
        raise SystemExit(
            f"policy family '{force_family}' has no verified rates, so savings cannot be "
            "estimated with it. Supply verified rates with --rules."
        )
    skipped: List[Tuple[str, str, str]] = []
    families_used: Dict[str, str] = {}
    assumed_notes: List[str] = []
    family_note_added: set = set()

    cache: Dict[int, float] = {}          # prefix-hash -> expiry
    written_marks: Dict[int, List[int]] = {}  # request index -> boundary hashes written
    hash_read: Dict[int, bool] = {}
    per_request: List[RequestCost] = []
    charged_delta: List[Tuple[int, int]] = []  # (request index, charged delta tokens)

    for i, req in enumerate(ordered):
        ts = req.ts if req.ts is not None else float(i)
        if force_family is not None:
            fam = rules.families[force_family]
            fam_assumed = False
        else:
            resolution = rules.resolve_family(req.model)
            fam = resolution.family
            fam_assumed = resolution.assumed
        families_used[req.model] = fam.name + (" (assumed)" if fam_assumed else "")
        if fam_assumed:
            reason = fam.notes[0] if fam.notes else "no verified policy family for this model"
            skipped.append((req.rid, req.model, reason))
            continue
        if not fam.savings_supported:
            skipped.append((req.rid, req.model, f"family '{fam.name}' has no verified rates"))
            continue
        for note in fam.notes:
            if note not in family_note_added:
                family_note_added.add(note)
                assumed_notes.append(f"[{fam.name}] {note}")
        ttl = float("inf") if no_ts else float(fam.ttl_seconds)
        # Family minimum is the base; a provider-level per-model minimum
        # (e.g. Anthropic's model table) overrides it when it matches.
        min_pref = fam.min_prefix_tokens
        mm = rules.resolve_min_prefix(req.model)
        if mm != rules.min_prefix_tokens:
            min_pref = mm

        # Boundary k is the prefix covering the first k+1 messages, hashed
        # from its own rendered text so shared text prefixes always hash
        # equal across requests.
        parts = [f"<{m.role}>\n{m.text}" for m in req.messages]
        bounds: List[Tuple[int, int]] = []  # (token_count, prefix_hash)
        for k in range(len(parts)):
            prefix_text = "\n".join(parts[: k + 1])
            toks = encode(prefix_text)
            bounds.append((len(toks), hash(tuple(toks))))
        total = bounds[-1][0] if bounds else 0
        eligible = [(b, h) for b, h in bounds if b >= min_pref]

        # Longest unexpired cached prefix.
        read_tokens = 0
        for b, h in sorted(eligible, reverse=True):
            expiry = cache.get(h)
            if expiry is not None and expiry >= ts:
                read_tokens = b
                hash_read[h] = True
                if rules.refresh_on_read:
                    cache[h] = ts + ttl
                break

        # Choose write breakpoints.
        if req.breakpoints:
            marked = {bounds[idx] for idx in req.breakpoints if idx < len(bounds)}
            write_bounds = sorted(b for b, h in marked if b >= min_pref and b > read_tokens)
            write_bounds = write_bounds[: rules.max_explicit_breakpoints]
        else:
            write_bounds = [total] if total >= min_pref and total > read_tokens else []

        written_delta = 0
        marks: List[int] = []
        if write_bounds:
            furthest = max(write_bounds)
            written_delta = furthest - read_tokens
            # Store every eligible boundary up to the furthest write so later
            # requests can match shorter shared prefixes (approximation).
            for b, h in eligible:
                if read_tokens < b <= furthest:
                    expiry = cache.get(h)
                    if expiry is None or expiry < ts:
                        cache[h] = ts + ttl
                        hash_read.setdefault(h, False)
                    marks.append(h)
        written_marks[i] = marks
        if written_delta:
            charged_delta.append((i, written_delta))

        uncached = max(total - read_tokens - written_delta, 0)
        per_request.append(
            RequestCost(
                rid=req.rid,
                total_input=total,
                read=read_tokens,
                written=written_delta,
                uncached=uncached,
                baseline_units=float(total),
                simulated_units=(
                    uncached * 1.0
                    + written_delta * fam.write_multiplier
                    + read_tokens * fam.read_multiplier
                ),
            )
        )

    # Wasted writes: charged deltas whose written prefixes no later request read.
    wasted = 0
    for i, delta in charged_delta:
        if not any(hash_read.get(h, False) for h in written_marks.get(i, [])):
            wasted += delta
    if wasted:
        notes.append(
            "Some tokens were written to cache but never re-read; single-use "
            "prefixes cost extra whenever the family charges for writes."
        )
    notes.extend(assumed_notes)
    if skipped:
        models = sorted({m for _, m, _ in skipped})
        notes.append(
            f"{len(skipped)} request(s) excluded from savings simulation because their "
            f"policy is unverified (models: {', '.join(models)}). Their tokens still count "
            "in duplicate/token analysis. Pass --policy for a known family or --rules with "
            "verified rates to include them."
        )

    family_summary = "; ".join(f"{m} -> {f}" for m, f in sorted(families_used.items()))
    if family_summary:
        notes.append(f"Policy families applied: {family_summary}")

    return SimulationResult(
        provider=rules.name,
        per_request=per_request,
        baseline_units=sum(c.baseline_units for c in per_request),
        simulated_units=sum(c.simulated_units for c in per_request),
        wasted_write_tokens=wasted,
        notes=notes,
        skipped=skipped,
    )
