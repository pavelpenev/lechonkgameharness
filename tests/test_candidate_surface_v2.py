from lechonk.core.candidate_surface_v2 import (
    CANDIDATE_SURFACE_POLICY,
    admit_candidate_surface,
)
from lechonk.core.schemas import Candidate, CandidateTable, CandidateTableCapExceeded, estimate_tokens


def candidate(domain, actor, verb="move", summary="ok", *, contact=False):
    return Candidate(
        id=f"{domain}:{actor}:{verb}",
        domain=domain,
        actor_id=str(actor),
        verb=verb,
        summary=summary,
        in_contact=contact,
    )


def test_round_robin_covers_domains_and_restores_source_order():
    table = CandidateTable(
        turn=4,
        candidates=[
            candidate("unit", 1, "a"), candidate("unit", 1, "b"),
            candidate("unit", 2, "a"), candidate("city", 3, "a"),
            candidate("tech", "player", "a"), candidate("dipl", 4, "a"),
        ],
    )
    result = admit_candidate_surface(table, cap_tokens=45)
    assert result.policy == CANDIDATE_SURFACE_POLICY
    assert result.offered_by_domain["city"] >= 1
    assert result.offered_by_domain["tech"] >= 1
    assert result.offered_by_domain["dipl"] >= 1
    assert [c.id for c in result.offered.candidates] == sorted(
        [c.id for c in result.offered.candidates],
        key=lambda ident: [x.id for x in table.candidates].index(ident),
    )


def test_protected_candidates_are_retained_and_overflow_is_explicit():
    protected = candidate("unit", 1, "attack", "x", contact=True)
    result = admit_candidate_surface(
        CandidateTable(turn=1, candidates=[protected, candidate("tech", "p", "research")]),
        cap_tokens=100,
    )
    assert protected.id in {item.id for item in result.offered.candidates}
    too_large = candidate("unit", 1, "attack", "x" * 1000, contact=True)
    try:
        admit_candidate_surface(CandidateTable(turn=1, candidates=[too_large]), cap_tokens=10)
    except CandidateTableCapExceeded:
        pass
    else:
        raise AssertionError("protected overflow must fail closed")


def test_deterministic_and_skips_large_candidate_for_smaller_later_choice():
    large = candidate("unit", 1, "large", "x" * 100)
    small = candidate("city", 2, "small", "x")
    table = CandidateTable(turn=2, candidates=[large, small])
    first = admit_candidate_surface(table, cap_tokens=20)
    second = admit_candidate_surface(table, cap_tokens=20)
    assert [c.id for c in first.offered.candidates] == [c.id for c in second.offered.candidates]
    assert small.id in {c.id for c in first.offered.candidates}
    assert first.omitted_candidate_count == 1


def test_accounting_and_cap_hold():
    table = CandidateTable(
        turn=3,
        candidates=[candidate("tech", "p", "research", "alphabet"), candidate("unit", 1, "move")],
    )
    result = admit_candidate_surface(table, cap_tokens=100)
    assert result.total_candidate_count == 2
    assert result.offered_candidate_count == 2
    assert result.omitted_candidate_count == 0
    assert result.omitted_tokens == 0
    assert result.estimated_tokens() <= 100


def test_duplicate_candidate_ids_fail_closed():
    duplicate = candidate("unit", 1, "move")
    table = CandidateTable(turn=1, candidates=[duplicate, duplicate.model_copy()])
    try:
        admit_candidate_surface(table)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")


def test_omission_disclosure_and_exact_rendered_cap_accounting():
    table = CandidateTable(
        turn=1,
        candidates=[candidate("unit", 1, "a", "x" * 100), candidate("city", 2, "b", "y")],
    )
    result = admit_candidate_surface(table, cap_tokens=20)
    assert result.omitted_candidate_count == 1
    assert "candidates omitted: 1 (beyond budget, table order)" in result.render()
    assert result.offered_tokens == estimate_tokens("\n".join(
        f"{item.id} — {item.summary}" for item in result.offered.candidates
    ))
    assert result.estimated_tokens() <= result.cap_tokens
    assert result.omitted_tokens > 0


def test_mixed_protected_and_unprotected_preserves_protected_counting():
    table = CandidateTable(
        turn=1,
        candidates=[
            candidate("unit", 1, "attack", "contact", contact=True),
            candidate("city", 2, "build", "ordinary"),
        ],
    )
    result = admit_candidate_surface(table, cap_tokens=30)
    assert result.offered_by_domain["unit"] == 1
    assert result.total_by_domain["city"] == 1
    assert result.total_candidate_count == result.offered_candidate_count + result.omitted_candidate_count


def test_near_cap_table_that_fits_is_not_shortened_for_disclosure_reserve():
    table = CandidateTable(
        turn=1,
        candidates=[candidate("unit", 1, "move", "x" * 20), candidate("city", 2, "build", "y")],
    )
    raw_tokens = estimate_tokens("\n".join(
        f"{item.id} — {item.summary}" for item in table.candidates
    ))
    result = admit_candidate_surface(table, cap_tokens=raw_tokens)
    assert result.omitted_candidate_count == 0
    assert result.offered_candidate_count == len(table.candidates)
    assert [item.id for item in result.offered.candidates] == [item.id for item in table.candidates]
