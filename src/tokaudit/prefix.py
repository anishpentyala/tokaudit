"""Prompt-cache simulation over a sequence of requests.

Model (documented approximation, not a provider oracle):

- Each request is rendered canonically (see logs.py) and tokenized.
- Eligible breakpoints are cumulative token counts at message boundaries
  (individual text blocks for Anthropic) meeting the provider minimum.
  Explicit markers restrict writes; otherwise one implicit write is assumed
  at the full prompt (OpenAI caching / Anthropic automatic-cache scenario).
- A read hits when an unexpired cached entry is a prefix of the request;
  the longest match wins. Reads are charged at ``read_multiplier``.
- Writes are charged on the *delta*: tokens beyond the longest cached
  prefix up to the furthest new breakpoint, at ``write_multiplier``.
  Anthropic writes only designated text blocks; reads look back at most 20
  blocks per breakpoint for prior writes. Other providers retain the
  intermediate-message-boundary approximation.
- Entries expire ``ttl_seconds`` after writing; providers with
  ``refresh_on_read`` extend coverage on reads by ``read_refresh_seconds``
  (or the write TTL when unspecified), never shortening existing coverage.
- Cost is in units of the base input-token rate (uncached 1.0x). No
  dollar prices are baked in, so results survive price changes.

Requests without timestamps are treated as one cache lifetime apart from
expiry concerns (log order, nothing expires); a note records it.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union

from .logs import Request
from .rules import ProviderRules


TokenKey = Union[bytes, Tuple[int, ...]]
PrefixKey = Tuple[TokenKey, tuple]


def _prefix_key(tokens: List[int]) -> TokenKey:
    """Retain exact token identity without keeping one Python int per token.

    Signed 64-bit packing covers built-in tokenizers. Arbitrary-size custom
    token IDs keep exact tuple equality instead of truncating or hashing.
    Keys only live within this process, so native byte order is immaterial.
    """
    try:
        return array("q", tokens).tobytes()
    except OverflowError:
        return tuple(tokens)


@dataclass
class RequestCost:
    rid: str
    total_input: int
    read: int
    written: int
    uncached: int
    baseline_units: float
    simulated_units: float
    model: str = ""


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
    isolate_sessions: bool = False,
) -> SimulationResult:
    ordered = sorted(requests, key=lambda r: (r.ts is None, r.ts or 0.0))
    no_ts = any(r.ts is None for r in ordered)
    notes: List[str] = []
    if no_ts:
        notes.append(
            "Some requests lack timestamps; treated as arriving in log order with nothing expiring."
        )
    if rules.name == "anthropic":
        notes.append(
            "Anthropic text-only model: writes only at designated blocks; reads "
            "search prior writes within 20 block positions per breakpoint. "
            "Configured lifetime/rates still apply, not mixed per-block TTLs; "
            "provider routing and concurrency are not modeled. "
            "Protocol checked 2026-11-06: "
            "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"
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

    # Cache namespaces mirror documented provider scope: entries are shared
    # within one organization and region for a given model, and a supplied
    # prompt_cache_key separates reuse between groups (rules/EVIDENCE.md).
    # Records carry no org/region fields, so one shared org+region is assumed;
    # that can overstate reuse when a log mixes accounts. --isolate-sessions
    # is an explicit conservative scenario (not provider semantics) that also
    # partitions by the app's session_id.
    caches: Dict[Tuple[str, Optional[str], Optional[str]], Dict[PrefixKey, Union[float, Fraction]]] = {}
    read_marks: Dict[Tuple[str, Optional[str], Optional[str]], Dict[Tuple[PrefixKey, int], bool]] = {}
    generations: Dict[Tuple[str, Optional[str], Optional[str]], Dict[PrefixKey, int]] = {}
    written_marks: Dict[int, List[Tuple[PrefixKey, int]]] = {}  # request index -> boundary keys written

    per_request: List[RequestCost] = []
    charged_delta: List[Tuple[int, int]] = []  # (request index, charged delta tokens)

    for i, req in enumerate(ordered):
        # A finite timestamp plus TTL can round up (false hit), down (false
        # miss), or overflow. Compare exact represented times internally.
        ts = (Fraction(req.ts) if not no_ts else
              req.ts if req.ts is not None else float(i))
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
        ttl = float("inf") if no_ts else Fraction(fam.ttl_seconds)
        # Family minimum is the base; a provider-level per-model minimum
        # (e.g. Anthropic's model table) overrides it when it matches.
        min_pref = fam.min_prefix_tokens
        mm = rules.resolve_min_prefix(req.model)
        if mm != rules.min_prefix_tokens:
            min_pref = mm

        namespace = (req.model, req.cache_key, req.session_id if isolate_sessions else None)
        cache = caches.setdefault(namespace, {})
        hash_read = read_marks.setdefault(namespace, {})
        cache_generations = generations.setdefault(namespace, {})

        # Anthropic retains text-block boundaries inside each message.
        # Other providers retain the existing message-boundary approximation.
        # Keys preserve tokens and structural layout, not just a raw hash.
        parts = [f"<{m.role}>\n{m.text}" for m in req.messages]
        bounds: List[Tuple[int, PrefixKey]] = []
        marked_indices = []
        block_mode = rules.name == "anthropic"
        layout = []
        for k, message in enumerate(req.messages):
            ends = ((message.block_ends if message.block_ends is not None
                     else (len(message.text),)) if block_mode else (len(message.text),))
            for block_index, end in enumerate(ends):
                prefix_text = "\n".join(parts[:k] + [f"<{message.role}>\n{message.text[:end]}"])
                toks = encode(prefix_text)
                prefix_layout = tuple(layout + [(message.role, end, tuple(ends[:block_index + 1]))])
                bounds.append((len(toks), (_prefix_key(toks), prefix_layout)))
                if k in req.breakpoints and (
                    not block_mode or block_index in message.cache_points
                    or (not message.cache_points and block_index == len(ends) - 1)
                ):
                    marked_indices.append(len(bounds) - 1)
            layout.append((message.role, len(message.text), tuple(ends)))
        total = len(encode(req.canonical())) if req.messages else 0
        eligible = [(b, h) for b, h in bounds if b >= min_pref]
        counts = [b for b, _ in bounds] + [total]
        if block_mode and any(a >= min_pref and a > b
                              for a, b in zip(counts, counts[1:])):
            # Some tokenizers merge across a canonical text-block boundary.
            # Never charge a cached prefix longer than the later prompt or
            # silently clip it into a made-up accounting result.
            raise SystemExit(
                "Anthropic canonical token counts decrease across content "
                "blocks; cannot simulate block-cache costs with this tokenizer"
            )
        # Anthropic writes at designated blocks only. Reads inspect up to 20
        # block positions per breakpoint, looking for prior actual writes.
        # No markers retains the existing labeled automatic-cache scenario.
        if not req.breakpoints and bounds:
            marked_indices = [len(bounds) - 1]
        if block_mode:
            marked_indices = marked_indices[:rules.max_explicit_breakpoints]
            read_indices = {j for i in marked_indices for j in range(max(0, i - 19), i + 1)}
            read_bounds = [bound for j, bound in enumerate(bounds)
                           if j in read_indices and bound[0] >= min_pref]
        else:
            read_bounds = eligible

        # Longest unexpired cached prefix.
        read_tokens = 0
        for b, h in sorted(read_bounds, key=lambda bound: bound[0], reverse=True):
            expiry = cache.get(h)
            if expiry is not None and expiry >= ts:
                read_tokens = b
                hash_read[(h, cache_generations[h])] = True
                if rules.refresh_on_read:
                    refresh = ttl if rules.read_refresh_seconds is None else (
                        rules.read_refresh_seconds if no_ts else Fraction(rules.read_refresh_seconds))
                    cache[h] = max(expiry, ts + refresh)
                break

        # Choose write breakpoints.
        if block_mode or req.breakpoints:
            marked = {bounds[idx] for idx in marked_indices}
            write_bounds = sorted(b for b, h in marked if b >= min_pref and b > read_tokens)
            write_bounds = write_bounds[: rules.max_explicit_breakpoints]
        else:
            write_bounds = [total] if total >= min_pref and total > read_tokens else []

        written_delta = 0
        marks: List[Tuple[PrefixKey, int]] = []
        if write_bounds:
            furthest = max(write_bounds)
            written_delta = furthest - read_tokens
            # Anthropic stores only designated block entries. Other providers
            # keep the existing intermediate-boundary approximation.
            stored_bounds = marked if block_mode else eligible
            for b, h in stored_bounds:
                if read_tokens < b <= furthest:
                    expiry = cache.get(h)
                    if expiry is None or expiry < ts:
                        cache[h] = ts + ttl
                        cache_generations[h] = cache_generations.get(h, 0) + 1
                        hash_read[(h, cache_generations[h])] = False
                    marks.append((h, cache_generations[h]))
        written_marks[i] = marks
        if written_delta:
            charged_delta.append((i, written_delta))

        uncached = max(total - read_tokens - written_delta, 0)
        per_request.append(
            RequestCost(
                rid=req.rid,
                model=req.model,
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
        req = ordered[i]
        ns = (req.model, req.cache_key, req.session_id if isolate_sessions else None)
        ns_reads = read_marks.get(ns, {})
        if not any(ns_reads.get(h, False) for h in written_marks.get(i, [])):
            wasted += delta
    if wasted:
        notes.append(
            "Some tokens were written to cache but never re-read; single-use "
            "prefixes cost extra whenever the family charges for writes."
        )
    notes.append(
        "Cache namespace model (rules/EVIDENCE.md, OpenAI 2026-10-28): entries shared "
        "within one organization + region per model; a record's prompt_cache_key "
        "separates reuse groups. Logs carry no org/region fields, so one shared "
        "org+region is assumed - mixed-account logs can overstate reuse."
    )
    if isolate_sessions:
        notes.append(
            "--isolate-sessions active: cache entries are additionally partitioned by "
            "session_id. This is a conservative scenario, not provider-documented "
            "behavior; real providers may reuse eligible prefixes across sessions."
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
