# SPDX-License-Identifier: MIT
"""Versioned priority/rotating candidate admission policy (r1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lechonk.core.schemas import Candidate, CandidateTable, CandidateTableCapExceeded, estimate_tokens, _render_line

CANDIDATE_SURFACE_POLICY = "candidate_surface_v3"
CANDIDATE_SURFACE_ADMISSION_VERSION = "candidate_surface_v3-r1"
CANDIDATE_SURFACE_TOKEN_BUDGET = 12000
DOMAIN_ORDER = ("unit", "city", "tech", "dipl")

def _joined(items: Iterable[Candidate]) -> int:
    return estimate_tokens("\n".join(_render_line(c) for c in items))

def _line(c: Candidate) -> int:
    return estimate_tokens(_render_line(c)) + 1

def _counts(items: Iterable[Candidate]) -> dict[str, int]:
    out = {d: 0 for d in DOMAIN_ORDER}
    for c in items: out[c.domain] = out.get(c.domain, 0) + 1
    return out

def _disclosure(n: int) -> str:
    return f"candidates omitted: {n} (beyond budget; priority/rotating admission)"

@dataclass(frozen=True)
class AdmissionResult:
    offered: CandidateTable
    policy: str
    cap_tokens: int
    total_candidate_count: int
    offered_candidate_count: int
    omitted_candidate_count: int
    total_by_domain: dict[str, int]
    offered_by_domain: dict[str, int]
    omitted_by_domain: dict[str, int]
    total_tokens: int
    offered_tokens: int
    omitted_tokens: int

    def render(self) -> str:
        lines = [_render_line(c) for c in self.offered.candidates]
        if self.omitted_candidate_count:
            lines.append(_disclosure(self.omitted_candidate_count))
        return "\n".join(lines)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())

def admit_candidate_surface(table: CandidateTable, *, cap_tokens: int = CANDIDATE_SURFACE_TOKEN_BUDGET) -> AdmissionResult:
    if cap_tokens < 1: raise ValueError("cap_tokens must be positive")
    candidates = list(table.candidates)
    ids = [c.id for c in candidates]
    if len(ids) != len(set(ids)): raise ValueError("candidate IDs must be unique")
    total = _joined(candidates)
    counts = _counts(candidates)
    if total <= cap_tokens:
        return AdmissionResult(CandidateTable(turn=table.turn, candidates=candidates), CANDIDATE_SURFACE_POLICY,
            cap_tokens, len(candidates), len(candidates), 0, counts, counts.copy(), {d:0 for d in DOMAIN_ORDER}, total,total,0)
    # Reserve a truthful disclosure line. Contact candidates are mandatory.
    protected = [c for c in candidates if c.in_contact]
    reserved = estimate_tokens(_disclosure(len(candidates))) + 1
    used = reserved + sum(_line(c) for c in protected)
    if used > cap_tokens:
        raise CandidateTableCapExceeded(f"protected candidates exceed candidate-surface-v3 cap: {used} > {cap_tokens}")
    kept = {c.id for c in protected}
    # Immediate research choices are always preferred over goals.
    for c in candidates:
        if c.domain == "tech" and c.verb.startswith("research_tech_") and used + _line(c) <= cap_tokens:
            kept.add(c.id); used += _line(c)
    # Give each city one production choice, rotating deterministic source order.
    cities = {}
    for c in candidates:
        if c.domain == "city" and c.verb.startswith("produce_"):
            cities.setdefault(c.actor_id, []).append(c)
    for actor, queue in sorted(cities.items()):
        if queue:
            c = queue[table.turn % len(queue)]
            if used + _line(c) <= cap_tokens: kept.add(c.id); used += _line(c)
    # True lane/actor round-robin for remaining candidates.
    queues = {d: {} for d in DOMAIN_ORDER}
    for c in candidates:
        if c.id in kept: continue
        queues.setdefault(c.domain, {}).setdefault(c.actor_id, []).append(c)
    pos = {d:{a:0 for a in q} for d,q in queues.items()}
    actor_cursor = {d: (table.turn % len(queues[d])) if queues[d] else 0 for d in DOMAIN_ORDER}
    start = table.turn % len(DOMAIN_ORDER)
    while True:
        progress = False
        for offset in range(len(DOMAIN_ORDER)):
            d = DOMAIN_ORDER[(start + offset) % len(DOMAIN_ORDER)]
            actors = sorted(queues.get(d, {}))
            if not actors: continue
            # One actor per lane per cycle; rotate the starting actor by turn.
            for step in range(len(actors)):
                actor = actors[(actor_cursor[d] + step) % len(actors)]
                i = pos[d][actor]
                if i >= len(queues[d][actor]): continue
                c = queues[d][actor][i]; pos[d][actor] = i + 1; progress = True
                if used + _line(c) <= cap_tokens: kept.add(c.id); used += _line(c)
                break
            actor_cursor[d] = (actor_cursor[d] + 1) % len(actors)
        if not progress: break
    offered = [c for c in candidates if c.id in kept]
    omitted = [c for c in candidates if c.id not in kept]
    # Revisit omitted candidates using the actual omission count.  The first
    # pass reserves a conservative disclosure line; this pass prevents that
    # reservation from needlessly hiding a candidate when the final line is
    # shorter (and makes admission maximal in source order).
    changed = True
    while changed:
        changed = False
        for c in list(omitted):
            trial_offered = [*offered, c]
            trial_omitted = len(omitted) - 1
            if estimate_tokens("\n".join([*(_render_line(x) for x in trial_offered), _disclosure(trial_omitted)])) <= cap_tokens:
                omitted.remove(c)
                offered.append(c)
                changed = True
    # Ensure final disclosure itself remains inside the cap; protected data is never dropped.
    protected_ids = {
        c.id for c in offered
        if c.in_contact
        or (c.domain == "tech" and c.verb.startswith("research_tech_"))
        or (c.domain == "city" and c.verb.startswith("produce_"))
    }
    while estimate_tokens("\n".join([*(_render_line(c) for c in offered), _disclosure(len(omitted))])) > cap_tokens:
        removable = next((c for c in reversed(offered) if c.id not in protected_ids), None)
        if removable is None: raise CandidateTableCapExceeded("priority candidates plus disclosure exceed cap")
        offered.remove(removable); omitted.append(removable)
    offered.sort(key=lambda c: candidates.index(c))
    return AdmissionResult(CandidateTable(turn=table.turn, candidates=offered), CANDIDATE_SURFACE_POLICY,
        cap_tokens, len(candidates), len(offered), len(omitted), counts, _counts(offered), _counts(omitted),
        total, _joined(offered), _joined(omitted))

__all__ = ["AdmissionResult", "CANDIDATE_SURFACE_POLICY", "CANDIDATE_SURFACE_ADMISSION_VERSION", "CANDIDATE_SURFACE_TOKEN_BUDGET", "admit_candidate_surface"]
