"""Detect repeated content blocks across requests.

Repeated blocks are text sent in two or more distinct requests: few-shot
examples, the same retrieved document, boilerplate instructions. They are
candidates for hoisting into one stable, cacheable prefix.

Method: shingle hashing. Each message is tokenized; every ``shingle``-token
window is hashed. Windows seen in at least two distinct requests seed a
match, which is extended token-by-token across its occurrences. Blocks
shorter than ``min_tokens`` are discarded, overlapping reports are
suppressed greedily, and the top ``top`` blocks by repeated volume are
returned.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .logs import Request


@dataclass
class DuplicateBlock:
    token_count: int
    occurrences: List[Tuple[str, int]]  # (request id, message index)
    preview: str
    request_indices: List[int] = field(default_factory=list)
    # Zero-based input request/message positions; half-open selected-encoder
    # token ranges. No claim of character offsets or complete occurrence search.
    token_spans: List[Dict[str, int]] = field(default_factory=list)

    @property
    def distinct_requests(self) -> int:
        return len(set(self.request_indices)) if self.request_indices else len({rid for rid, _ in self.occurrences})

    @property
    def repeated_tokens(self) -> int:
        return self.token_count * max(len(self.occurrences) - 1, 0)


@dataclass
class DuplicateSearchResult:
    """Search coverage for the bounded shingle index, not proof of completeness."""
    blocks: List[DuplicateBlock]
    indexed_shingles: int
    eligible_shingles: int
    max_shingles: int

    @property
    def index_complete(self) -> bool:
        return self.indexed_shingles == self.eligible_shingles


def find_duplicates(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    min_tokens: int = 128,
    shingle: int = 50,
    top: int = 10,
    max_shingles: int = 500_000,
) -> List[DuplicateBlock]:
    """Compatibility list API; use analyze_duplicates for index coverage."""
    return analyze_duplicates(requests, encode, min_tokens, shingle, top,
                              max_shingles).blocks


def analyze_duplicates(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    min_tokens: int = 128,
    shingle: int = 50,
    top: int = 10,
    max_shingles: int = 500_000,
) -> DuplicateSearchResult:
    for label, value in (("min_tokens", min_tokens), ("shingle", shingle),
                         ("top", top), ("max_shingles", max_shingles)):
        minimum = 0 if label == "max_shingles" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    index: Dict[int, List[Tuple[int, int, int]]] = {}
    tokenized: List[List[List[int]]] = []
    seen = 0
    eligible_shingles = 0

    for ridx, req in enumerate(requests):
        per_msg: List[List[int]] = []
        for midx, msg in enumerate(req.messages):
            toks = encode(msg.text)
            per_msg.append(toks)
            if len(toks) < shingle:
                continue
            eligible_shingles += len(toks) - shingle + 1
            for start in range(0, len(toks) - shingle + 1):
                if seen >= max_shingles:
                    break
                h = hash(tuple(toks[start : start + shingle]))
                index.setdefault(h, []).append((ridx, midx, start))
                seen += 1
        tokenized.append(per_msg)

    def exact_groups():
        for bucket in index.values():
            if len({r for r, _, _ in bucket}) < 2:
                continue
            # Hashes narrow candidates; only equal token windows share a seed.
            # Partition lazily so unique windows do not retain full token tuples.
            groups: Dict[Tuple[int, ...], List[Tuple[int, int, int]]] = {}
            for ridx, midx, start in bucket:
                seed = tuple(tokenized[ridx][midx][start : start + shingle])
                groups.setdefault(seed, []).append((ridx, midx, start))
            # A short/divergent first occurrence must not hide another pair
            # sharing a full minimum-length block. Split off verified cohorts,
            # reusing the same exact-run cache used by extension below.
            pending = list(reversed(list(groups.values())))
            required = max(shingle, min_tokens)
            while pending:
                occs = [(r, m, start) for r, m, start in pending.pop()
                        if start + required <= len(tokenized[r][m])]
                if len({r for r, _, _ in occs}) < 2:
                    continue
                anchor = occs[0]
                good, rest = [], []
                for occurrence in occs:
                    if equal_run(anchor, occurrence) >= required:
                        good.append(occurrence)
                    else:
                        rest.append(occurrence)
                if len({r for r, _, _ in good}) >= 2:
                    yield good
                if len({r for r, _, _ in rest}) >= 2:
                    pending.append(rest)

    # A prior extension proves an equal run along this pair's alignment.
    # Reuse only inside that run; its right edge is a known mismatch/end.
    # Bound memo entries independently of the shingle index. Eviction only
    # costs repeated comparisons, never changes the candidates or matches.
    runs = OrderedDict()
    def equal_run(anchor, occurrence):
        r0, m0, s0 = anchor
        ridx, midx, start = occurrence
        toks0, toks = tokenized[r0][m0], tokenized[ridx][midx]
        alignment = (r0, m0, ridx, midx, start - s0)
        run = runs.get(alignment)
        if run is not None and run[0] <= s0 and s0 + shingle <= run[1]:
            runs.move_to_end(alignment)
            return run[1] - s0
        length = shingle
        while (s0 + length < len(toks0) and start + length < len(toks)
               and toks0[s0 + length] == toks[start + length]):
            length += 1
        runs[alignment] = (s0, s0 + length)
        runs.move_to_end(alignment)
        if len(runs) > 4096:
            runs.popitem(last=False)
        return length

    blocks: List[DuplicateBlock] = []
    for occs in exact_groups():
        req_ids = {o[0] for o in occs}
        if len(req_ids) < 2:
            continue
        r0, m0, s0 = occs[0]
        good: List[Tuple[int, int, int]] = []  # (request position, midx, length)
        seen_occ = set()
        starts = {}
        for ridx, midx, start in occs:
            if (ridx, midx) in seen_occ:
                continue
            seen_occ.add((ridx, midx))
            length = equal_run((r0, m0, s0), (ridx, midx, start))
            if length >= min_tokens:
                good.append((ridx, midx, length))
                starts[(ridx, midx)] = start
        if len({g[0] for g in good}) < 2:
            continue
        min_len = min(g[2] for g in good)
        occurrences = [(requests[ridx].rid, midx) for ridx, midx, _ in good]
        text = requests[r0].messages[m0].text.strip().replace("\n", " ")
        blocks.append(
            DuplicateBlock(
                token_count=min_len,
                occurrences=occurrences,
                preview=text[:100] + ("..." if len(text) > 100 else ""),
                request_indices=[ridx for ridx, _, _ in good],
                token_spans=[{
                    "request_index": ridx,
                    "message_index": midx,
                    "token_start": starts[(ridx, midx)],
                    "token_end": starts[(ridx, midx)] + min_len,
                } for ridx, midx, _ in good],
            )
        )

    # Suppress overlapping/nested reports: keep the largest blocks first.
    # Message membership alone cannot show overlap: independent shared regions
    # in the same messages should remain separate findings. Suppress only when
    # every candidate occurrence overlaps a span in one accepted finding.
    blocks.sort(key=lambda b: b.repeated_tokens, reverse=True)
    accepted: List[DuplicateBlock] = []
    accepted_spans: List[Dict[Tuple[int, int], Tuple[int, int]]] = []
    for b in blocks:
        spans = {(span["request_index"], span["message_index"]):
                 (span["token_start"], span["token_end"])
                 for span in b.token_spans}
        if any(all(key in previous and start < previous[key][1]
                   and previous[key][0] < end
                   for key, (start, end) in spans.items())
               for previous in accepted_spans):
            continue
        accepted.append(b)
        accepted_spans.append(spans)
        if len(accepted) >= top:
            break
    return DuplicateSearchResult(accepted, seen, eligible_shingles, max_shingles)
