# SPDX-License-Identifier: MIT
"""Versioned, deterministic candidate-surface admission policy.

The runner selects this policy explicitly through its versioned configuration;
the implementation remains isolated here so its admission and telemetry can
be tested independently of the active gameplay loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lechonk.core.schemas import (
    Candidate,
    CandidateTable,
    CandidateTableCapExceeded,
    estimate_tokens,
    _omission_line,
    _render_line,
)


CANDIDATE_SURFACE_POLICY = "candidate_surface_v2"
CANDIDATE_SURFACE_TOKEN_BUDGET = 8000
DOMAIN_ORDER = ("unit", "city", "tech", "dipl")


@dataclass(frozen=True)
class AdmissionResult:
    """The offered table and auditable admission accounting."""

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
        """Render the admitted candidates and an honest omission disclosure."""
        lines = [_render_line(candidate) for candidate in self.offered.candidates]
        if self.omitted_candidate_count:
            lines.append(_omission_line(self.omitted_candidate_count))
        return "\n".join(lines)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())


def _line_tokens(candidate: Candidate) -> int:
    # Account for the newline/join separator, matching CandidateTable.offered.
    return estimate_tokens(_render_line(candidate)) + 1


def _joined_tokens(candidates: Iterable[Candidate]) -> int:
    """Estimate exactly the joined candidate-line text, with empty = zero."""
    lines = [_render_line(candidate) for candidate in candidates]
    return estimate_tokens("\n".join(lines)) if lines else 0


def _domain_counts(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for candidate in candidates:
        counts[candidate.domain] = counts.get(candidate.domain, 0) + 1
    return counts


def admit_candidate_surface(
    table: CandidateTable,
    *,
    cap_tokens: int = CANDIDATE_SURFACE_TOKEN_BUDGET,
) -> AdmissionResult:
    """Admit a deterministic, coverage-oriented subset of ``table``.

    Armed-in-contact candidates are reserved first.  Remaining candidates are
    visited round-robin by domain and then actor, with source order preserved
    inside each actor queue.  An oversized candidate is omitted and the
    iterator continues, allowing smaller candidates in other queues to fit.
    The final offered table is restored to the original source order.
    """
    if cap_tokens < 1:
        raise ValueError("cap_tokens must be positive")

    candidates = list(table.candidates)
    candidate_ids = [candidate.id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique for deterministic admission")
    total_count = len(candidates)
    total_by_domain = _domain_counts(candidates)
    raw_tokens = _joined_tokens(candidates)
    # Do not reserve a disclosure line when it is unnecessary.  A table that
    # already fits is offered in full, including protected and ordinary lines.
    if raw_tokens <= cap_tokens:
        offered = CandidateTable(turn=table.turn, candidates=candidates)
        offered_by_domain = _domain_counts(candidates)
        return AdmissionResult(
            offered=offered,
            policy=CANDIDATE_SURFACE_POLICY,
            cap_tokens=cap_tokens,
            total_candidate_count=total_count,
            offered_candidate_count=total_count,
            omitted_candidate_count=0,
            total_by_domain=total_by_domain,
            offered_by_domain=offered_by_domain,
            omitted_by_domain={domain: 0 for domain in DOMAIN_ORDER},
            total_tokens=raw_tokens,
            offered_tokens=raw_tokens,
            omitted_tokens=0,
        )
    reserve = estimate_tokens(_omission_line(total_count)) + 1
    protected = [candidate for candidate in candidates if candidate.in_contact]
    protected_tokens = sum(_line_tokens(candidate) for candidate in protected)
    if protected_tokens + reserve > cap_tokens:
        raise CandidateTableCapExceeded(
            "armed-in-contact candidates plus mandatory rendering overhead "
            f"exceed the candidate-table cap: {protected_tokens + reserve} > "
            f"{cap_tokens} tokens"
        )

    kept_ids = {candidate.id for candidate in protected}
    # Reserve the disclosure line while admitting.  This guarantees that a
    # non-empty omission disclosure cannot push the rendered surface over the
    # cap; the small conservative reserve is released only when no candidate
    # is omitted (the final rendering then has no disclosure line).
    used = protected_tokens + reserve

    queues: dict[str, dict[str, list[Candidate]]] = {
        domain: {} for domain in DOMAIN_ORDER
    }
    for candidate in candidates:
        if candidate.in_contact:
            continue
        queues.setdefault(candidate.domain, {}).setdefault(candidate.actor_id, []).append(candidate)

    # Deterministic domain/actor cycling.  Each queue advances even when its
    # current candidate does not fit, so a large line cannot starve later,
    # smaller candidates.
    positions = {
        domain: {actor: 0 for actor in actors}
        for domain, actors in queues.items()
    }
    while True:
        progress = False
        for domain in DOMAIN_ORDER:
            actors = queues.get(domain, {})
            for actor in sorted(actors):
                index = positions[domain][actor]
                if index >= len(actors[actor]):
                    continue
                candidate = actors[actor][index]
                positions[domain][actor] = index + 1
                progress = True
                if used + _line_tokens(candidate) <= cap_tokens:
                    kept_ids.add(candidate.id)
                    used += _line_tokens(candidate)
        if not progress:
            break

    kept = [candidate for candidate in candidates if candidate.id in kept_ids]
    offered = CandidateTable(turn=table.turn, candidates=kept)
    omitted = [candidate for candidate in candidates if candidate.id not in kept_ids]
    offered_by_domain = _domain_counts(kept)
    omitted_by_domain = _domain_counts(omitted)
    # These fields describe actual rendered candidate-line text.  Omitted
    # tokens exclude the disclosure line by definition, so total and offered
    # may differ from their sum because each joined string is estimated
    # independently and the estimator rounds at string boundaries.
    total_tokens = _joined_tokens(candidates)
    offered_tokens = _joined_tokens(kept)
    omitted_tokens = _joined_tokens(omitted)
    return AdmissionResult(
        offered=offered,
        policy=CANDIDATE_SURFACE_POLICY,
        cap_tokens=cap_tokens,
        total_candidate_count=total_count,
        offered_candidate_count=len(kept),
        omitted_candidate_count=len(omitted),
        total_by_domain=total_by_domain,
        offered_by_domain=offered_by_domain,
        omitted_by_domain=omitted_by_domain,
        total_tokens=total_tokens,
        offered_tokens=offered_tokens,
        omitted_tokens=omitted_tokens,
    )


__all__ = [
    "AdmissionResult",
    "CANDIDATE_SURFACE_POLICY",
    "CANDIDATE_SURFACE_TOKEN_BUDGET",
    "admit_candidate_surface",
]
