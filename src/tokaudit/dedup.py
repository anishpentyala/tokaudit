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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .logs import Request


@dataclass
class DuplicateBlock:
    token_count: int
    occurrences: List[Tuple[str, int]]  # (request id, message index)
    preview: str

    @property
    def repeated_tokens(self) -> int:
        return self.token_count * max(len(self.occurrences) - 1, 0)


def find_duplicates(
    requests: List[Request],
    encode: Callable[[str], List[int]],
    min_tokens: int = 128,
    shingle: int = 50,
    top: int = 10,
    max_shingles: int = 500_000,
) -> List[DuplicateBlock]:
    index: Dict[int, List[Tuple[int, int, int]]] = {}
    tokenized: List[List[List[int]]] = []
    seen = 0

    for ridx, req in enumerate(requests):
        per_msg: List[List[int]] = []
        for midx, msg in enumerate(req.messages):
            toks = encode(msg.text)
            per_msg.append(toks)
            if shingle <= 0 or len(toks) < shingle:
                continue
            for start in range(0, len(toks) - shingle + 1):
                if seen >= max_shingles:
                    break
                h = hash(tuple(toks[start : start + shingle]))
                index.setdefault(h, []).append((ridx, midx, start))
                seen += 1
        tokenized.append(per_msg)

    blocks: List[DuplicateBlock] = []
    for h, occs in index.items():
        req_ids = {o[0] for o in occs}
        if len(req_ids) < 2:
            continue
        r0, m0, s0 = occs[0]
        toks0 = tokenized[r0][m0]
        good: List[Tuple[str, int, int]] = []  # (rid, midx, length)
        seen_occ = set()
        for ridx, midx, start in occs:
            if (ridx, midx) in seen_occ:
                continue
            seen_occ.add((ridx, midx))
            toks = tokenized[ridx][midx]
            length = shingle
            while (
                s0 + length < len(toks0)
                and start + length < len(toks)
                and toks0[s0 + length] == toks[start + length]
            ):
                length += 1
            if length >= min_tokens:
                good.append((requests[ridx].rid, midx, length))
        if len({g[0] for g in good}) < 2:
            continue
        min_len = min(g[2] for g in good)
        occurrences = [(rid, midx) for rid, midx, _ in good]
        text = requests[r0].messages[m0].text.strip().replace("\n", " ")
        blocks.append(
            DuplicateBlock(
                token_count=min_len,
                occurrences=occurrences,
                preview=text[:100] + ("..." if len(text) > 100 else ""),
            )
        )

    # Suppress overlapping/nested reports: keep the largest blocks first,
    # drop blocks whose (rid, midx) set is a subset of an accepted block's.
    blocks.sort(key=lambda b: b.repeated_tokens, reverse=True)
    accepted: List[DuplicateBlock] = []
    accepted_sets: List[set] = []
    for b in blocks:
        occ_set = set(b.occurrences)
        if any(occ_set <= prev for prev in accepted_sets):
            continue
        accepted.append(b)
        accepted_sets.append(occ_set)
        if len(accepted) >= top:
            break
    return accepted
