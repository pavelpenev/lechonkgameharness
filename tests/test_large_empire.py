# SPDX-License-Identifier: GPL-3.0-or-later
"""WP2 acceptance: the §3 truncation policy validated on a LARGE-EMPIRE
fixture, not only opening turns (design-plan-freeciv-v1.1.md §3 line 32).

The synthetic large-empire observation (tests/large_empire_fixture.py: 6 own
cities, 48 own units, 13 foreign units, 4 contacts, full 80x50 map breadth,
20 research candidates, 31 messages) is driven through the real
``lechonk.adapter.obs_extract.extract_snapshot`` under the default 8,000-token
budget. Asserted per §3:

- the rendered snapshot is ≤ 8,000 tokens under the policy (and the
  untruncated render is well over it — the policy actually engaged);
- idle/distant own units are elided FIRST (the elided set is an exact prefix
  of the extractor's elision order: idle before working, farthest from the
  capital first, id as the deterministic tie-break);
- armed-in-contact own units are NEVER elided; foreign units are never
  elided; working units survive while farther idle units are elided;
- candidate lists are capped by priority with the omitted disclosure set
  (research by (cost, name); production alphabetically; messages counted);
- the delta report is ≤ 800 tokens under a heavy turn-over-turn change load
  (the disclosure warning is budgeted too — regression: the warning event
  used to push the enforced delta back over its own budget);
- the strategy memory is ≤ 1,500 tokens under a heavy memory load (threat
  hypotheses elided first, then the notes, then the oldest commitments, with
  the elision disclosed).
"""

from __future__ import annotations

from lechonk.adapter.obs_extract import extract_snapshot
from lechonk.core.delta import compute_delta, enforce_delta_budget
from lechonk.core.memory import StrategyMemory
from lechonk.core.schemas import (
    DELTA_TOKEN_BUDGET,
    MEMORY_TOKEN_BUDGET,
    SNAPSHOT_TOKEN_BUDGET,
    Commitment,
    MemoryUpdate,
    estimate_tokens,
)

from tests.large_empire_fixture import (
    CAPITAL,
    MY_PLAYER,
    N_FAR_IDLE,
    UNIT_TYPE_IDS,
    elision_order_key,
    large_empire_world,
    large_ruleset,
)

NO_ELISION = 10 ** 9  # reference extraction: no truncation


def _extract(obs, info, *, token_budget=SNAPSHOT_TOKEN_BUDGET):
    return extract_snapshot(
        obs, info, my_player_id=MY_PLAYER, ruleset=large_ruleset(),
        token_budget=token_budget).snapshot


def _world(**kwargs):
    return large_empire_world(**kwargs)


# ---------------------------------------------------------------------------
# the snapshot budget under the policy
# ---------------------------------------------------------------------------


def test_large_empire_is_genuinely_large_and_over_budget_untruncated():
    obs, info = _world(turn=30)
    full = _extract(obs, info, token_budget=NO_ELISION)
    assert full.map_view.width == 80 and full.map_view.height == 50
    counts = full.map_view.tile_counts
    assert counts["visible"] > 0 and counts["last_known"] > 0 \
        and counts["unexplored"] > 0  # full map breadth, all three classes
    own_cities = [c for c in full.cities if c.is_own]
    assert len(own_cities) >= 6
    own_units = [u for u in full.units if u.is_own]
    foreign_units = [u for u in full.units if not u.is_own]
    assert len(own_units) >= 47
    assert len(foreign_units) >= 13
    assert len(full.diplomacy) >= 4  # several contacts
    assert full.estimated_tokens() > SNAPSHOT_TOKEN_BUDGET, \
        "the fixture must exceed the budget so the policy actually engages"


def test_large_empire_snapshot_within_8000_under_policy():
    obs, info = _world(turn=30)
    snap = _extract(obs, info)
    assert snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET
    assert snap.omitted.summary != "nothing omitted"  # the policy fired


def test_large_empire_policy_is_deterministic():
    obs, info = _world(turn=30)
    a = _extract(obs, info)
    b = _extract(obs, info)
    assert a.render() == b.render()


# ---------------------------------------------------------------------------
# the elision policy: idle/distant first, armed-in-contact never
# ---------------------------------------------------------------------------


def test_large_empire_elision_is_a_prefix_of_the_elision_order():
    obs, info = _world(turn=30)
    full = _extract(obs, info, token_budget=NO_ELISION)
    snap = _extract(obs, info)

    elided_ids = list(snap.omitted.elided_unit_ids)
    assert elided_ids, "budget pressure must elide something"

    # The elidable set (own, not in contact) in the extractor's elision
    # order: idle first, farthest from the capital first, id tie-break.
    elidable = [u for u in full.units if u.is_own and not u.in_contact]
    elidable.sort(key=lambda u: elision_order_key(u, CAPITAL))
    expected_prefix = [u.id for u in elidable[:len(elided_ids)]]
    assert elided_ids == expected_prefix, \
        "the elided set must be an exact prefix of the elision order"

    # Every elided unit is own, idle, and not in contact.
    by_id = {u.id: u for u in full.units}
    for uid in elided_ids:
        u = by_id[uid]
        assert u.is_own
        assert not u.in_contact
        assert u.activity in (None, 0), "working units are elided only " \
                                        "after every idle unit"

    # The disclosure lists exactly the elided ids.
    assert set(snap.omitted.elided_unit_ids) == set(elided_ids)
    assert len(snap.omitted.elided_unit_ids) == len(set(elided_ids))
    assert "elided" in snap.omitted.summary
    assert str(elided_ids[0]) in snap.omitted.summary


def test_large_empire_protected_units_never_elided():
    obs, info = _world(turn=30)
    full = _extract(obs, info, token_budget=NO_ELISION)
    snap = _extract(obs, info)
    elided = set(snap.omitted.elided_unit_ids)
    surviving = {u.id for u in snap.units}

    # Armed-in-contact own units: never elided.
    in_contact = [u for u in full.units if u.is_own and u.in_contact]
    assert len(in_contact) >= 3
    for u in in_contact:
        assert u.id not in elided
        assert u.id in surviving

    # Foreign units (visible AND revealed-but-not-visible): never elided.
    foreign = [u for u in full.units if not u.is_own]
    assert len(foreign) >= 13
    for u in foreign:
        assert u.id not in elided
        assert u.id in surviving

    # Working units survive while far idle units are elided (idle first).
    working = [u for u in full.units
               if u.is_own and u.activity not in (None, 0)]
    assert len(working) >= 2
    for u in working:
        assert u.id not in elided
        assert u.id in surviving
    assert elided, "far idle units must be elided for this to be meaningful"

    # Near-capital idle units survive while far idle units are elided
    # (distant first).
    near_idle = [u for u in full.units
                 if u.is_own and u.activity in (None, 0) and not u.in_contact
                 and max(abs(u.x - CAPITAL[0]), abs(u.y - CAPITAL[1])) <= 2]
    assert len(near_idle) >= 2
    for u in near_idle:
        assert u.id in surviving

    # The elided units are exactly the far idle settlers (ids 200..239).
    far_idle = {u.id for u in full.units
                if u.is_own and 200 <= u.id < 200 + N_FAR_IDLE}
    assert elided and elided.issubset(far_idle)


# ---------------------------------------------------------------------------
# candidate lists capped by priority, with the omitted disclosure set
# ---------------------------------------------------------------------------


def test_large_empire_candidate_caps_with_disclosure():
    obs, info = _world(turn=30)
    snap = _extract(obs, info)

    # Research candidates: capped, retained in (cost, name) priority order.
    assert snap.omitted.research_candidates_capped is True
    assert 0 < len(snap.research_candidates) <= 12
    full = _extract(obs, info, token_budget=NO_ELISION)
    by_priority = sorted(full.research_candidates,
                         key=lambda c: (c.cost is None, c.cost or 0, c.name))
    assert [c.name for c in snap.research_candidates] == \
        [c.name for c in by_priority[:len(snap.research_candidates)]]

    # Production candidates: capped at 8 per own city, alphabetical.
    assert snap.omitted.production_candidates_capped is True
    buildable = sorted(UNIT_TYPE_IDS)
    for city in snap.cities:
        if city.is_own:
            assert city.production_candidates == buildable[:8]

    # Messages: capped with the elided count disclosed (31 messages: 30
    # server events + the game-domain chat line; capped at 20).
    assert snap.omitted.messages_elided == 11
    assert len(snap.messages) == 20

    # The map grid is omitted on large maps; the trichotomy counts remain.
    assert snap.map_view.grid is None
    assert snap.map_view.tile_counts["visible"] > 0
    assert snap.map_view.tile_counts["last_known"] > 0
    assert snap.map_view.tile_counts["unexplored"] > 0

    # The disclosure summary carries every fired cap.
    assert "research candidate list capped" in snap.omitted.summary
    assert "message" in snap.omitted.summary


# ---------------------------------------------------------------------------
# the delta report ≤ 800 tokens under a heavy change load
# ---------------------------------------------------------------------------


def test_large_empire_delta_within_800_under_load():
    obs_cur, info_cur = _world(turn=30)
    obs_prev, info_prev = _world(turn=29, variant="prev")
    # Untruncated snapshots: the delta sees the full change set (40 moved
    # units, a city growth, research progress, a new contact).
    cur = _extract(obs_cur, info_cur, token_budget=NO_ELISION)
    prev = _extract(obs_prev, info_prev, token_budget=NO_ELISION)
    raw = compute_delta(prev, cur)
    assert raw.estimated_tokens() > DELTA_TOKEN_BUDGET, \
        "the fixture's turn-over-turn change load must exceed the budget"

    enforced = enforce_delta_budget(raw)
    assert enforced.estimated_tokens() <= DELTA_TOKEN_BUDGET, \
        "regression: the truncation warning must not push the enforced " \
        "delta back over its own budget"
    assert enforced.events[-1].kind == "warning"
    assert "delta truncated" in enforced.events[-1].detail
    assert enforce_delta_budget(raw).render() == enforced.render(), \
        "the delta budget enforcement is deterministic"


# ---------------------------------------------------------------------------
# the strategy memory ≤ 1,500 tokens under a heavy memory load
# ---------------------------------------------------------------------------


def test_large_empire_memory_within_1500_under_load():
    mem = StrategyMemory()
    update = MemoryUpdate(
        commitments=[Commitment(domain="expansion",
                                target=f"city site {i} near the northern river",
                                deadline_turn=31 + i)
                     for i in range(100)],
        threats=[f"unit:{300 + i}" for i in range(40)],
        notes=("hold the line and expand north, building roads toward the "
               "front. " * 258),
    )
    # Pre/post-advance separation: the update is staged during the turn and
    # committed post-advance; it first appears in the NEXT turn's prompt.
    mem.begin_turn(30)
    mem.apply(update, turn=30)
    mem.advance(30)
    mem.begin_turn(31)
    text = mem.render_for_prompt()

    assert estimate_tokens(text) <= MEMORY_TOKEN_BUDGET
    # The elision disclosure is present; threats go first, then the notes,
    # then the oldest commitments.
    assert "memory truncated" in text
    assert "oldest threat hypotheses" in text
    assert "oldest commitments elided" in text
    threat_lines = [l for l in text.splitlines() if l.startswith("- unit:")]
    assert not threat_lines, "all 40 threat hypotheses are elided first"
    commitment_lines = [l for l in text.splitlines() if "(by turn" in l]
    assert 0 < len(commitment_lines) < 100, \
        "the notes are shortened to nothing, then the oldest commitments " \
        "are elided"
    assert "oldest commitments elided" in text

    # Deterministic.
    mem2 = StrategyMemory()
    mem2.begin_turn(30)
    mem2.apply(update, turn=30)
    mem2.advance(30)
    mem2.begin_turn(31)
    assert mem2.render_for_prompt() == text
