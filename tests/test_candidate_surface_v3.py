import pytest

from lechonk.core.candidate_surface_v3 import (
    CANDIDATE_SURFACE_POLICY, admit_candidate_surface,
)
from lechonk.core.schemas import Candidate, CandidateTable, CandidateTableCapExceeded


def c(domain, actor, verb, summary="x", contact=False):
    return Candidate(id=f"{domain}:{actor}:{verb}", domain=domain, actor_id=str(actor),
                     verb=verb, summary=summary, in_contact=contact)


def test_real_like_surface_fits_12000_and_budget_includes_disclosure():
    items = []
    for i in range(435): items.append(c("unit", i, "goto_n", "u"))
    for i in range(184): items.append(c("city", i, "produce_w", "city"))
    for i in range(75): items.append(c("tech", "cur_player", f"research_tech_t_{i}", "tech"))
    result = admit_candidate_surface(CandidateTable(turn=10, candidates=items))
    assert result.policy == CANDIDATE_SURFACE_POLICY
    assert result.omitted_candidate_count == 0
    assert result.estimated_tokens() <= 12_000


def test_contact_and_immediate_research_are_retained_or_fail_closed():
    protected = c("unit", 1, "attack_n", "contact", True)
    research = c("tech", "cur_player", "research_tech_seafaring_1", "research")
    huge = [c("unit", i, "goto_n", "z" * 1000) for i in range(20)]
    result = admit_candidate_surface(CandidateTable(turn=2, candidates=[protected, research, *huge]), cap_tokens=300)
    offered = {x.id for x in result.offered.candidates}
    assert protected.id in offered
    assert research.id in offered
    with pytest.raises(CandidateTableCapExceeded):
        admit_candidate_surface(CandidateTable(turn=2, candidates=[c("unit", 1, "attack_n", "z" * 10000, True)]), cap_tokens=30)


def test_each_city_gets_a_rotating_production_choice_and_order_is_deterministic():
    items = [c("city", city, f"produce_{kind}", kind * 30) for city in (1, 2, 3) for kind in ("warrior", "settler")]
    a = admit_candidate_surface(CandidateTable(turn=4, candidates=items), cap_tokens=200)
    b = admit_candidate_surface(CandidateTable(turn=4, candidates=items), cap_tokens=200)
    assert [x.id for x in a.offered.candidates] == [x.id for x in b.offered.candidates]
    for city in (1, 2, 3):
        assert any(x.id.startswith(f"city:{city}:produce_") for x in a.offered.candidates)
    cnext = admit_candidate_surface(CandidateTable(turn=5, candidates=items), cap_tokens=200)
    assert {x.id for x in a.offered.candidates} != {x.id for x in cnext.offered.candidates}


def test_domain_round_robin_keeps_non_unit_lanes_visible():
    items = [c("unit", i, "goto_n", "u" * 20) for i in range(100)]
    items += [c("tech", "cur_player", "research_tech_x_1", "t")]
    items += [c("city", 1, "produce_settler", "p")]
    result = admit_candidate_surface(CandidateTable(turn=3, candidates=items), cap_tokens=500)
    domains = {x.domain for x in result.offered.candidates}
    assert {"unit", "tech", "city"} <= domains
    assert result.estimated_tokens() <= 500


def test_overflow_round_robin_rotates_actor_start_by_turn():
    items = [c("unit", actor, f"goto_n_{idx}", f"actor-{actor}-" + "x" * 15)
             for actor in (1, 2, 3) for idx in range(3)]
    # A narrow budget admits only a few entries, making the starting actor
    # observable while preserving deterministic order for a fixed turn.
    turn0 = admit_candidate_surface(CandidateTable(turn=0, candidates=items), cap_tokens=40)
    turn1 = admit_candidate_surface(CandidateTable(turn=1, candidates=items), cap_tokens=40)
    assert {x.actor_id for x in turn0.offered.candidates} != {x.actor_id for x in turn1.offered.candidates}
    assert [x.id for x in turn0.offered.candidates] == [x.id for x in admit_candidate_surface(
        CandidateTable(turn=0, candidates=items), cap_tokens=40).offered.candidates]


def test_disclosure_is_truthful_and_rendered_budget_is_hard():
    result = admit_candidate_surface(CandidateTable(turn=1, candidates=[c("unit", i, "goto_n", "q" * 100) for i in range(30)]), cap_tokens=120)
    assert "priority/rotating admission" in result.render()
    assert result.estimated_tokens() <= result.cap_tokens


def test_disclosure_trim_never_silently_drops_selected_city_production():
    items = [c("city", city, "produce_warrior", "p" * 20) for city in (1, 2)]
    items.extend(c("unit", i, "goto_n", "u" * 40) for i in range(8))
    try:
        result = admit_candidate_surface(CandidateTable(turn=0, candidates=items), cap_tokens=110)
    except CandidateTableCapExceeded:
        # If the required production set plus accurate disclosure cannot fit,
        # fail closed instead of silently dropping a city choice.
        return
    assert {x.actor_id for x in result.offered.candidates
            if x.domain == "city" and x.verb.startswith("produce_")} == {"1", "2"}
    assert result.estimated_tokens() <= result.cap_tokens
