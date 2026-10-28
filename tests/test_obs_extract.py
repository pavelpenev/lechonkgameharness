# SPDX-License-Identifier: GPL-3.0-or-later
"""Extractor behavior tests: empty-observation classification by returned
flags, research display cross-checked against the tech domain, embassy-gated
intel, and the token-budget truncation policy (idle/distant first, never
armed-in-contact)."""

from __future__ import annotations

import pytest

from lechonk.adapter.obs_extract import (
    ACTION_ATTACK,
    ExtractOutcome,
    MAX_PRODUCTION_CANDIDATES,
    _apply_token_budget,
    RulesetNames,
    extract_snapshot,
)
from lechonk.core.schemas import SNAPSHOT_TOKEN_BUDGET

from tests.fog_fixtures import (
    ENEMY,
    MY_PLAYER,
    NUM_IMPROVEMENTS,
    PLAINS,
    SETTLERS,
    VUT_NONE,
    VUT_UTYPE,
    WARRIORS,
    base_info,
    base_world,
    enemy_player,
    fixture_ruleset,
    foreign_unit,
    make_observation,
    my_player,
    own_city,
    own_unit,
    tech_entry,
)


def _extract(obs, info=None, **kwargs):
    return extract_snapshot(
        obs, info if info is not None else base_info(),
        my_player_id=MY_PLAYER, ruleset=fixture_ruleset(), **kwargs)


# ---------------------------------------------------------------------------
# Empty observations: classified by the RETURNED FLAGS, never the dict alone
# ---------------------------------------------------------------------------


def test_empty_observation_truncated_is_infra():
    result = _extract({}, truncated=True)
    assert result.outcome is ExtractOutcome.TRUNCATED_INFRA
    assert result.snapshot is None
    assert "truncated=True" in result.reason


def test_empty_observation_terminated_defeated():
    result = _extract({}, terminated=True, my_player_is_defeated=True)
    assert result.outcome is ExtractOutcome.TERMINATED_DEFEATED
    assert result.snapshot is None


def test_empty_observation_terminated_game_over():
    result = _extract({}, terminated=True, my_player_is_defeated=False)
    assert result.outcome is ExtractOutcome.TERMINATED_GAME_OVER
    assert result.snapshot is None


def test_empty_observation_without_flags_fails_closed():
    with pytest.raises(ValueError, match="never the dict alone|refusing"):
        _extract({})


def test_nonempty_observation_with_terminated_flag_still_extracts():
    """Flags are authoritative: terminated=True on a non-empty observation
    classifies the outcome but the snapshot is still extracted."""
    result = _extract(base_world(), terminated=True,
                      my_player_is_defeated=False)
    assert result.outcome is ExtractOutcome.TERMINATED_GAME_OVER
    assert result.snapshot is not None
    assert result.snapshot.turn == 1


# ---------------------------------------------------------------------------
# Research display: computed from the tech domain, cross-checked
# ---------------------------------------------------------------------------


def test_research_display_uses_tech_domain_not_research_name():
    """Plan §7.2: compute research display from the tech domain; the PR #7
    research_name fix is verified applied (docs/gate-record.md) but we
    cross-check rather than trust it. Here upstream research_name is stale
    ('Alphabet') while the tech domain says Pottery is being researched."""
    techs = {
        1: tech_entry(1, "Alphabet", inv_state=2, reqs={}),  # known
        2: tech_entry(2, "Pottery", inv_state=1, reqs={1: True},
                      is_researching=True),
    }
    obs = base_world(
        players=[my_player(researching=2, research_name="Alphabet",
                           bulbs_researched=5, researching_cost=30,
                           total_bulbs_prod=5),
                 enemy_player()],
        techs=techs,
    )
    snap = _extract(obs).snapshot
    assert snap.player.research.current_target == "Pottery"
    assert snap.player.research.current_target_id == 2
    assert snap.player.research.bulbs_researched == 5
    assert snap.player.research.cost_total == 30
    assert snap.player.research.turns_left == 5  # ceil((30-5)/5)


def test_research_candidates_enumerated_with_cost_and_notes():
    snap = _extract(base_world()).snapshot
    by_name = {c.name: c for c in snap.research_candidates}
    # inv_state==PREREQS_KNOWN only: Alphabet, Pottery, Agriculture
    # (Bronze Working is TECH_UNKNOWN, Writing is TECH_KNOWN)
    assert set(by_name) == {"Alphabet", "Pottery", "Agriculture"}
    assert by_name["Alphabet"].cost == 20
    assert by_name["Pottery"].note == ""  # req Alphabet is known
    assert by_name["Agriculture"].note == "requires: Pottery"
    # sorted by (cost, name)
    assert [c.name for c in snap.research_candidates] == [
        "Alphabet", "Pottery", "Agriculture"]


def test_no_research_target_renders_none():
    snap = _extract(base_world()).snapshot
    assert snap.player.research.current_target is None
    assert snap.player.research.current_target_id is None
    assert snap.player.research.turns_left is None


# ---------------------------------------------------------------------------
# Embassy-gated intel
# ---------------------------------------------------------------------------


def test_embassy_gated_intel_respected():
    # no embassy, zeroed packet placeholders (what the server actually sends
    # for opponents without embassy): nothing may be claimed as knowledge
    snap = _extract(base_world()).snapshot
    entry = next(d for d in snap.diplomacy if d.counterpart_id == ENEMY)
    assert entry.state == "No contact"
    assert "no embassy intel" in entry.intel
    assert "unknown" in entry.intel

    # hearsay leaks only positive gold / research / government
    obs = base_world(players=[
        my_player(),
        enemy_player(gold=0, tax=0, science=0, luxury=0, researching=None,
                     government=0),
    ])
    entry = next(d for d in _extract(obs).snapshot.diplomacy
                 if d.counterpart_id == ENEMY)
    assert entry.intel == "no embassy intel: gold/research/government unknown", \
        "zeroed placeholders must render as unknown, never as claimed values"
    assert "tax" not in entry.intel, "tax is embassy-only"

    # hearsay leaks: positive gold / research / government render as known
    obs = base_world(players=[
        my_player(),
        enemy_player(gold=123, research_name="Pottery", researching=2,
                     government=1, government_name="Monarchy"),
    ])
    entry = next(d for d in _extract(obs).snapshot.diplomacy
                 if d.counterpart_id == ENEMY)
    assert "hearsay intel" in entry.intel
    assert "gold 123" in entry.intel
    assert "researching Pottery" in entry.intel
    assert "government Monarchy" in entry.intel
    assert "tax" not in entry.intel, "tax is embassy-only, not hearsay"

    # embassy (tech_* fields filled by the embassy report): full intel
    obs = base_world(players=[
        my_player(),
        enemy_player(gold=123, research_name="Pottery", researching=2,
                     government=1, government_name="Monarchy", tax=10,
                     science=80, luxury=10,
                     **{"tech_1": True, "tech_2": False, "tech_7": True}),
    ])
    entry = next(d for d in _extract(obs).snapshot.diplomacy
                 if d.counterpart_id == ENEMY)
    assert "embassy intel" in entry.intel
    assert "tax 10% science 80% luxury 10%" in entry.intel
    assert "2 techs known" in entry.intel


def test_diplomacy_state_names_and_meeting():
    obs = base_world(dipl={ENEMY: {"diplomatic_state": 1,
                                   "diplomacy_clause_map": ["clause x"],
                                   "meeting_initializers": ENEMY}})
    snap = _extract(obs).snapshot
    entry = next(d for d in snap.diplomacy if d.counterpart_id == ENEMY)
    assert entry.state == "War"
    assert entry.meeting_active is True
    assert entry.clauses == ["clause x"]


# ---------------------------------------------------------------------------
# Combat odds (when exposed) and attack candidates
# ---------------------------------------------------------------------------


def test_attack_candidates_carry_odds_when_exposed():
    action_probs = {
        11: {6: {ACTION_ATTACK: {"min": 60.0, "max": 65.0}}},  # dir S=6
    }
    result = _extract(base_world(), action_probs=action_probs)
    snap = result.snapshot
    warrior = next(u for u in snap.units if u.id == 11)
    south = next(a for a in warrior.attack_candidates if a.direction == "S")
    assert south.target_unit_id == 20
    assert south.target_type == "Warriors"
    assert south.odds == 0.65


def test_attack_candidates_without_odds_when_not_exposed():
    snap = _extract(base_world()).snapshot
    warrior = next(u for u in snap.units if u.id == 11)
    assert warrior.attack_candidates, "target is enumerated"
    assert all(a.odds is None for a in warrior.attack_candidates)


# ---------------------------------------------------------------------------
# Token-budget truncation policy (§3)
# ---------------------------------------------------------------------------


def _crowded_world(n_idle: int, with_defender: bool = True):
    """Capital + n_idle idle settlers spread far from the capital + one armed
    warrior in contact with a visible enemy (never elidable)."""
    units = [own_unit(10, SETTLERS, 1, 1)]
    for i in range(n_idle):
        units.append(own_unit(100 + i, SETTLERS, 1 + (i % 4), 1 + (i % 4),
                              activity=0))
    units.append(own_unit(11, WARRIORS, 2, 1))  # adjacent to enemy at (2,2)
    units.append(foreign_unit(20, WARRIORS, 2, 2))
    return base_world(units=units)


def test_token_budget_elides_idle_distant_first_never_in_contact():
    obs = _crowded_world(n_idle=40)
    full = _extract(obs, token_budget=10**9).snapshot  # no elision reference
    assert full.estimated_tokens() > 2000  # crowded
    # WP1 package C: the CityView/UnitView schema fields added for the §7.4
    # policy (shield_stock/production_progress/terrain) make each remaining
    # unit cost slightly more and lengthen the elided-id disclosure summary,
    # so the arbitrary pressure budget below moved 2000 -> 2100 (the policy
    # budget SNAPSHOT_TOKEN_BUDGET is unchanged).
    result = _extract(obs, token_budget=2100)
    snap = result.snapshot
    assert snap.estimated_tokens() <= 2100
    # the armed in-contact warrior survives
    assert any(u.id == 11 and u.in_contact for u in snap.units)
    assert 11 not in snap.omitted.elided_unit_ids
    # the enemy unit survives (foreign units are never elided)
    assert any(u.id == 20 for u in snap.units)
    # elided units are all own, idle, non-contact; the protected units
    # (armed-in-contact warrior 11, foreign enemy 20) are never elided
    elided = set(snap.omitted.elided_unit_ids)
    assert elided, "budget pressure must elide something"
    assert elided.isdisjoint({11, 20})
    by_id = {u.id: u for u in full.units}
    for uid in elided:
        assert by_id[uid].is_own
        assert not by_id[uid].in_contact
        assert by_id[uid].activity in (None, 0)
    # disclosure is present and specific
    assert "elided" in snap.omitted.summary
    assert len(snap.omitted.elided_unit_ids) == len(elided)


def test_token_budget_elision_order_is_deterministic():
    obs = _crowded_world(n_idle=30)
    a = _extract(obs, token_budget=1800).snapshot
    b = _extract(obs, token_budget=1800).snapshot
    assert a.render() == b.render()
    # farthest idle units go first: with a moderate budget the elided set is
    # the farthest idle settlers
    full = _extract(obs, token_budget=10**9).snapshot
    capital = (1, 1)
    idle = [u for u in full.units
            if u.is_own and u.id not in (10, 11) and not u.in_contact]
    idle.sort(key=lambda u: (-max(abs(u.x - capital[0]), abs(u.y - capital[1])),
                             u.id))
    expected_first = idle[0].id
    assert expected_first in a.omitted.elided_unit_ids


def test_default_budget_is_8000():
    snap = _extract(base_world()).snapshot
    assert snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET


def test_protected_units_can_drop_low_priority_region_notes():
    """Protected units remain visible when aggregate prose must be dropped."""
    snap = _extract(_crowded_world(n_idle=0), token_budget=10**9).snapshot
    protected_ids = {
        u.id for u in snap.units if (not u.is_own) or u.in_contact
    }
    # Remove the one own idle unit from this synthetic protected-only case;
    # the remaining units are exactly the foreign/contact units the policy
    # must never elide.
    snap.units = [u for u in snap.units if u.id in protected_ids]
    snap.map_view.region_notes.extend(["aggregate region prose " * 20] * 80)

    _apply_token_budget(snap, SNAPSHOT_TOKEN_BUDGET)

    assert snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET
    assert {u.id for u in snap.units} == protected_ids
    assert "region note" in snap.omitted.summary


def test_disclosure_is_included_in_final_snapshot_budget():
    """The omission disclosure itself cannot push the final render over cap."""
    snap = _extract(_crowded_world(n_idle=40), token_budget=10**9).snapshot
    _apply_token_budget(snap, 2100)
    assert snap.estimated_tokens() <= 2100
    assert snap.omitted.summary != "nothing omitted"


def test_region_note_disclosure_survives_repeated_budget_passes():
    snap = _extract(_crowded_world(n_idle=0), token_budget=10**9).snapshot
    protected_ids = {
        u.id for u in snap.units if (not u.is_own) or u.in_contact
    }
    snap.units = [u for u in snap.units if u.id in protected_ids]
    snap.map_view.region_notes.extend(["aggregate region prose " * 20] * 80)

    _apply_token_budget(snap, SNAPSHOT_TOKEN_BUDGET)
    first_count = snap.omitted.region_notes_elided
    first_summary = snap.omitted.summary
    _apply_token_budget(snap, SNAPSHOT_TOKEN_BUDGET)

    assert first_count > 0
    assert snap.omitted.region_notes_elided == first_count
    assert snap.omitted.summary == first_summary
    assert "some region summaries elided" in snap.omitted.summary
    assert snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET
    assert {u.id for u in snap.units} == protected_ids


# ---------------------------------------------------------------------------
# Messages + misc
# ---------------------------------------------------------------------------


def test_messages_from_info_and_game_domain():
    obs = base_world(game={"messages": "chat: hello", "year": "4000 BC",
                           "turn": 1})
    info = base_info(messages=[{"text": "server event: barbarians seen"},
                               {"turn": 1, "other": "dict message"}])
    snap = _extract(obs, info=info).snapshot
    assert "server event: barbarians seen" in snap.messages
    assert any("other" in m for m in snap.messages)  # dict messages str()'d
    assert "chat: hello" in snap.messages  # game-domain chat string appended


def test_messages_capped_with_disclosure():
    info = base_info(messages=[{"text": f"msg {i}"} for i in range(30)])
    snap = _extract(base_world(), info=info).snapshot
    assert len(snap.messages) == 20
    assert snap.omitted.messages_elided == 10
    assert "message" in snap.omitted.summary


def test_production_candidates_capped_with_disclosure():
    """§3 overflow policy: truncating the own-city buildable list at
    MAX_PRODUCTION_CANDIDATES sets the disclosure; a city under the cap
    leaves it unset."""
    ruleset = RulesetNames(
        unit_types={i: f"Unit{i}" for i in range(MAX_PRODUCTION_CANDIDATES + 4)})
    city = own_city(can_build_unit=[1] * (MAX_PRODUCTION_CANDIDATES + 4))
    snap = extract_snapshot(
        make_observation(cities=[city], units=[]), base_info(),
        my_player_id=MY_PLAYER, ruleset=ruleset).snapshot
    (view,) = snap.cities
    assert len(view.production_candidates) == MAX_PRODUCTION_CANDIDATES
    assert snap.omitted.production_candidates_capped is True

    snap_under = _extract(make_observation(cities=[own_city()],
                                           units=[])).snapshot
    assert len(snap_under.cities[0].production_candidates) == 2
    assert snap_under.omitted.production_candidates_capped is False


def test_my_player_id_inferred_when_not_passed():
    result = extract_snapshot(base_world(), base_info(),
                              ruleset=fixture_ruleset())
    assert result.outcome is ExtractOutcome.OK
    assert result.snapshot.player.player_id == MY_PLAYER
    assert result.snapshot.player.name == "Caesar"


def test_map_grid_omitted_on_large_maps():
    # 40x40 = 1600 cells > GRID_MAX_CELLS (1024): grid omitted, counts kept
    status = {(x, y): 2 for x in range(6) for y in range(6)}
    obs = make_observation(
        width=40, height=40, status=status,
        terrain={xy: 1 for xy in status},
        players=[my_player()], cities=[], units=[])
    snap = _extract(obs).snapshot
    assert snap.map_view.grid is None
    assert snap.map_view.width == 40
    assert snap.map_view.tile_counts["visible"] == 36
    assert snap.map_view.tile_counts["unexplored"] == 1600 - 36


# ---------------------------------------------------------------------------
# WP1 package C: CityView shield stock / production progress + UnitView
# terrain (the §7.4 buy rule + improve-by-terrain-yield inputs)
# ---------------------------------------------------------------------------


def test_city_shield_stock_and_production_progress_populated():
    # own city producing Warriors (VUT_UTYPE value carries the num_impr_types
    # offset per the fixture ruleset's city_production_names), 40% paid for.
    city = own_city(production_kind=VUT_UTYPE,
                    production_value=WARRIORS + NUM_IMPROVEMENTS,
                    shield_stock=12, prod_process=0.4, buy_cost=20)
    snap = _extract(make_observation(cities=[city], units=[])).snapshot
    (view,) = snap.cities
    assert view.production == "Warriors"
    assert view.shield_stock == 12
    assert view.production_progress == 0.4
    assert view.buy_cost == 20


def test_city_progress_none_when_nothing_in_production():
    city = own_city(production_kind=VUT_NONE, production_value=-1,
                    shield_stock=7, prod_process=32767)  # FC_INFINITY sentinel
    snap = _extract(make_observation(cities=[city], units=[])).snapshot
    (view,) = snap.cities
    assert view.production is None
    assert view.shield_stock == 7  # stored shields are still reported
    assert view.production_progress is None  # nothing in production -> no fraction


def test_unit_terrain_populated_when_visible_none_when_fogged():
    # (1,1) visible Plains (own unit); (2,2) visible Plains (foreign unit);
    # (3,3) last-known Forest — the fixture deliberately carries stale
    # terrain data there; the extractor must NOT leak it (fog-gated).
    obs = base_world(units=[own_unit(10, SETTLERS, 1, 1),
                            foreign_unit(20, WARRIORS, 3, 3),
                            foreign_unit(21, WARRIORS, 2, 2)])
    snap = _extract(obs).snapshot
    by_id = {u.id: u for u in snap.units}
    assert by_id[10].terrain == "Plains"   # own unit, visible tile
    assert by_id[21].terrain == "Plains"   # foreign unit, visible tile
    assert by_id[20].terrain is None       # last-known tile: unknown


def test_map_arrays_are_x_major_like_map_state():
    """The real map_state arrays are X-MAJOR: map_allocate(x_size, y_size)
    -> np.full((x_size, y_size)) and status[x, y] = ... (map_state.py:52-63,
    120-132) — verified against the live map (78 x 52 arrays, units at
    x=75). A NON-SQUARE map pins the orientation: a unit at x=6 on an
    8-wide/4-tall map must read its own tile (visible, terrain known, all
    8 move candidates in bounds). Under a transposed [y][x] read the tile
    would fall outside the grid (unexplored, terrain None, no moves)."""
    status = {(x, y): 2 for x in range(8) for y in range(4)}
    obs = make_observation(
        width=8, height=4, status=status,
        terrain={xy: PLAINS for xy in status},
        players=[my_player()],
        cities=[], units=[own_unit(10, SETTLERS, 6, 1)],
    )
    snap = _extract(obs).snapshot
    assert snap.map_view.width == 8   # x extent = first array dimension
    assert snap.map_view.height == 4  # y extent = second array dimension
    unit = next(u for u in snap.units if u.id == 10)
    assert unit.terrain == "Plains"
    assert len(unit.move_candidates) == 8  # every neighbor in bounds
    by_dir = {m.direction: m for m in unit.move_candidates}
    assert by_dir["E"].terrain == "Plains"
    assert by_dir["E"].note == "visible"
    assert by_dir["W"].note == "visible"
    # the ASCII grid renders rows=y, chars=x from the same x-major arrays
    rows = snap.map_view.grid.split("\n")
    assert len(rows) == 4 and all(len(r) == 8 for r in rows)
    assert rows[1][6] == "P"  # tile (6,1): visible Plains
