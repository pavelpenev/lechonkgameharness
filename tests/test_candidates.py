# SPDX-License-Identifier: GPL-3.0-or-later
"""Candidate-table tests: offline, from synthetic available_actions structures.

Covers (WP1 package B acceptance):
- every domain (unit/city/tech/dipl) enumerated from info['available_actions'];
- the §7.1 unit gating rules (movesleft>0, not done_moving,
  ssa_controller==SSA_NONE, not keep_activity);
- the NOT-exposed verbs absent (chat, set_sci_lux_tax, change_gov,
  spy/trade/transport verbs, sell verbs, ActAutoSettler, ActExplore, clause
  construction, specialists, city_keep_production);
- candidate-ID regex conformance (CANDIDATE_ID_RE) and uniqueness;
- combat-odds surfacing into attack candidates (percent min/max keyed by
  freeciv DIR8 indices NW=0..SE=7 -> payload fractions);
- determinism (same input -> same table).
"""

from __future__ import annotations

from typing import Any

import pytest

from lechonk.adapter.candidates import (
    UNIT_FIXED_VERBS,
    build_candidates,
)
from lechonk.core.schemas import CANDIDATE_ID_RE, CandidateTable

from tests.fog_fixtures import (
    MY_PLAYER,
    SETTLERS,
    base_world,
    own_unit,
)

# freeciv DIR8 indices (map_const.py): NW=0, N=1, NE=2, W=3, E=4, SW=5, S=6, SE=7.
ACTION_ATTACK = 45  # version/v14.py

# A synthetic available_actions structure exercising the whole surface,
# including NOT-exposed verbs that must be filtered out.
FULL_ACTIONS: dict[str, dict[Any, dict[str, bool]]] = {
    "unit": {
        10: {  # own Settlers at (1,1)
            "goto_0": True, "goto_4": True, "goto_1": False,  # NW ok, E ok, N invalid
            "attack_2": True,  # NE (no enemy there, but upstream says valid)
            "build_city": True, "fortify": True, "sentry": True,
            "build_road": True, "irrigation": True, "mine": True,
            "join_city": False, "set_homecity": True, "cancel_order": True,
            "keep_activity": True,
            # NOT exposed:
            "autosettlers": True, "explore": True,
            "spy_steal_tech_3": True, "spy_bribe_unit_2": True,
            "trade_route_1": True, "marketplace_4": True,
            "conquer_city_2": True, "hut_enter_1": True, "disembark_0": True,
            "paradrop": True, "airlift": True, "upgrade": True, "board": True,
            "deboard": True, "unit_unload": True, "embassy_stay_1": True,
            "investigate_spend_2": True, "transform": True, "cultivate": True,
            "plant": True, "fortress": True, "airbase": True,
            "build_railroad": True, "pillage": True, "fallout": True,
            "disband": True, "no_orders": True, "action_selection": True,
            "wait": True, "nuke": True, "get_action_pro_0": True,
        },
        11: {  # own Warriors at (2,1)
            "goto_3": True, "attack_6": True, "sentry": True, "fortify": True,
            "autosettlers": True,
        },
        20: {  # FOREIGN unit id in the action map must be gated out
            "goto_0": True, "sentry": True,
        },
    },
    "city": {
        1: {  # own city
            "produce_Settlers": True, "produce_City Walls": True,
            "produce_Granary": True, "produce_Warriors": False,
            "city_buy_production": True,
            "city_work_0_1_0": True, "city_work_2_-1_1": True,
            "city_unwork_0_1_0": True, "city_unwork_1_-1_-1": True,
            # NOT exposed:
            "city_sell_improvement_Granary": True,
            "city_change_specialist_0": True, "city_keep_production": True,
        },
    },
    "tech": {
        "cur_player": {
            "research_tech_Alphabet_0": True,
            "research_tech_Pottery_1": True,
            "research_tech_Bronze Working_2": False,  # invalid upstream
            "set_tech_goal_Bronze Working_2": True,
            "set_tech_goal_Alphabet_0": False,  # invalid upstream
        },
    },
    "dipl": {
        2: {
            "stop_negotiation_2": True, "cancel_treaty_2": True,
            "accept_treaty_2": False,  # invalid upstream
            # NOT exposed:
            "cancel_vision_2": True, "start_negotiation_2": True,
            "add_clause_1_2_0_0": True, "remove_clause_1_2_0_0": True,
            "trade_tech_clause_2": True, "trade_gold_clause_2": True,
            "trade_city_clause_2": True,
        },
    },
    # Entire ctrl_types that are NOT exposed:
    "player": {"message": {"chat": True}},
    "gov": {0: {"change_gov_Republic": True,
                "set_sci_lux_tax_90_0_10": True, "gov_do_nothing": True}},
}


def _info(actions=None, turn=1):
    return {"turn": turn, "mini_game_messages": [],
            "available_actions": FULL_ACTIONS if actions is None else actions}


def _build(obs=None, info=None, **kwargs):
    return build_candidates(
        base_world() if obs is None else obs,
        _info() if info is None else info,
        my_player_id=MY_PLAYER,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# unit domain
# ---------------------------------------------------------------------------


def test_unit_exposed_verbs_present():
    table = _build()
    unit10 = {c.verb for c in table.candidates
              if c.domain == "unit" and c.actor_id == "10"}
    expected = {"goto_nw", "goto_e", "attack_ne"} | set(UNIT_FIXED_VERBS)
    expected -= {"join_city"}  # invalid upstream for unit 10
    assert unit10 == expected


def test_unit_directional_verbs_use_compass_names():
    table = _build()
    by_id = table.by_id()
    assert "unit:10:goto_nw" in by_id   # raw goto_0 -> NW
    assert "unit:10:goto_e" in by_id    # raw goto_4 -> E
    assert "unit:10:attack_ne" in by_id  # raw attack_2 -> NE
    assert "unit:10:goto_n" not in by_id  # goto_1 was invalid upstream
    assert by_id["unit:10:goto_nw"].payload["dir8"] == 0
    assert by_id["unit:10:goto_e"].payload["dir8"] == 4
    assert by_id["unit:10:goto_nw"].payload["dir"] == "NW"


def test_unit_invalid_actions_excluded():
    table = _build()
    verbs10 = {c.verb for c in table.candidates
               if c.domain == "unit" and c.actor_id == "10"}
    assert "join_city" not in verbs10  # available_actions value False


def test_unit_fixed_verb_payload_carries_raw_action_key():
    table = _build()
    by_id = table.by_id()
    cand = by_id["unit:10:sentry"]
    assert cand.payload["ctrl_type"] == "unit"
    assert cand.payload["actor_id"] == 10
    assert cand.payload["action_key"] == "sentry"
    assert cand.summary


def test_unit_gating_moves_left_zero():
    obs = base_world()
    obs["unit"][10] = own_unit(10, SETTLERS, 1, 1, moves_left=0)
    table = _build(obs=obs)
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "10"]


def test_unit_gating_ssa_controller_not_none():
    obs = base_world()
    obs["unit"][10] = own_unit(10, SETTLERS, 1, 1, ssa_controller=1)
    table = _build(obs=obs)
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "10"]


def test_unit_gating_done_moving_via_unit_flags():
    table = _build(unit_flags={10: {"done_moving": True, "keep_activity": False}})
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "10"]
    # unit 11 (not flagged) still enumerated
    assert [c for c in table.candidates
            if c.domain == "unit" and c.actor_id == "11"]


def test_unit_gating_keep_activity_via_unit_flags():
    table = _build(unit_flags={10: {"done_moving": False, "keep_activity": True}})
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "10"]


def test_unit_gating_foreign_actor_excluded():
    """A foreign unit id in the action map (should not happen upstream) is
    gated out by the own-unit check."""
    table = _build()
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "20"]


def test_unit_gating_without_observation_entry():
    """An actor missing from the observation cannot be gated -> excluded."""
    obs = base_world()
    del obs["unit"][10]
    table = _build(obs=obs)
    assert not [c for c in table.candidates
                if c.domain == "unit" and c.actor_id == "10"]


# ---------------------------------------------------------------------------
# NOT-exposed verbs
# ---------------------------------------------------------------------------

NOT_EXPOSED_MARKERS = [
    "chat", "set_sci_lux_tax", "change_gov", "gov_do_nothing",
    "spy_", "trade_route", "marketplace", "conquer_city", "hut_enter",
    "disembark", "paradrop", "airlift", "upgrade", "board", "deboard",
    "unit_unload", "embassy_stay", "investigate_spend", "transform",
    "cultivate", "plant", "fortress", "airbase", "build_railroad",
    "pillage", "fallout", "disband", "no_orders", "action_selection",
    "wait", "nuke", "get_action_pro", "autosettlers", "explore",
    "sell", "specialist", "keep_production", "cancel_vision",
    "start_negotiation", "add_clause", "remove_clause", "trade_tech_clause",
    "trade_gold_clause", "trade_city_clause",
]


@pytest.mark.parametrize("marker", NOT_EXPOSED_MARKERS)
def test_not_exposed_verbs_absent(marker):
    table = _build()
    offenders = [c.id for c in table.candidates if marker in c.id]
    assert not offenders, f"NOT-exposed verb leaked into the table: {offenders}"


def test_entire_ctrl_types_not_exposed():
    table = _build()
    domains = {c.domain for c in table.candidates}
    assert domains == {"unit", "city", "tech", "dipl"}
    # no candidate carries a player/gov ctrl_type
    assert not [c for c in table.candidates
                if c.payload["ctrl_type"] not in ("unit", "city", "tech", "dipl")]


# ---------------------------------------------------------------------------
# city domain
# ---------------------------------------------------------------------------


def test_city_verbs():
    table = _build()
    verbs = {c.verb for c in table.candidates
             if c.domain == "city" and c.actor_id == "1"}
    assert verbs == {
        "produce_settlers", "produce_city_walls", "produce_granary",
        "city_buy_production",
        "city_work_0_1_0", "city_work_2_m1_1",
        "city_unwork_0_1_0", "city_unwork_1_m1_m1",
    }


def test_city_produce_name_with_space_sanitized():
    table = _build()
    by_id = table.by_id()
    cand = by_id["city:1:produce_city_walls"]
    assert cand.payload["action_key"] == "produce_City Walls"
    assert cand.payload["production"] == "City Walls"


def test_city_work_negative_offsets_sanitized_and_parseable():
    table = _build()
    by_id = table.by_id()
    cand = by_id["city:1:city_work_2_m1_1"]
    assert cand.payload["action_key"] == "city_work_2_-1_1"
    assert cand.payload["dx"] == -1 and cand.payload["dy"] == 1
    assert cand.payload["output_idx"] == 2
    cand = by_id["city:1:city_unwork_1_m1_m1"]
    assert cand.payload["action_key"] == "city_unwork_1_-1_-1"
    assert cand.payload["dx"] == -1 and cand.payload["dy"] == -1


def test_city_invalid_produce_excluded():
    table = _build()
    assert "city:1:produce_warriors" not in table.by_id()


# ---------------------------------------------------------------------------
# tech domain
# ---------------------------------------------------------------------------


def test_tech_verbs_and_actor():
    table = _build()
    tech = {c.id: c for c in table.candidates if c.domain == "tech"}
    assert set(tech) == {
        "tech:cur_player:research_tech_alphabet_0",
        "tech:cur_player:research_tech_pottery_1",
        "tech:cur_player:set_tech_goal_bronze_working_2",
    }
    cand = tech["tech:cur_player:research_tech_alphabet_0"]
    assert cand.payload["ctrl_type"] == "tech"
    assert cand.payload["actor_id"] == "cur_player"
    assert cand.payload["action_key"] == "research_tech_Alphabet_0"
    assert cand.payload["tech_id"] == 0
    assert cand.payload["tech_name"] == "Alphabet"
    cand = tech["tech:cur_player:set_tech_goal_bronze_working_2"]
    assert cand.payload["action_key"] == "set_tech_goal_Bronze Working_2"
    assert cand.payload["tech_id"] == 2


# ---------------------------------------------------------------------------
# diplomacy domain
# ---------------------------------------------------------------------------


def test_dipl_verbs_fold_player_into_actor():
    table = _build()
    dipl = {c.id: c for c in table.candidates if c.domain == "dipl"}
    assert set(dipl) == {"dipl:2:stop_negotiation", "dipl:2:cancel_treaty"}
    cand = dipl["dipl:2:stop_negotiation"]
    assert cand.payload["ctrl_type"] == "dipl"
    assert cand.payload["actor_id"] == 2
    assert cand.payload["action_key"] == "stop_negotiation_2"
    assert cand.payload["counterpart"] == 2


# ---------------------------------------------------------------------------
# combat odds
# ---------------------------------------------------------------------------


def test_attack_odds_from_action_probs_dir8_percent_ranges():
    """action_prob[uid][dir8][ACTION_ATTACK] = {'min','max'} percent."""
    action_probs = {10: {2: {ACTION_ATTACK: {"min": 40, "max": 60}}}}
    table = _build(action_probs=action_probs)
    cand = table.by_id()["unit:10:attack_ne"]
    assert cand.payload["odds_min"] == 0.4
    assert cand.payload["odds_max"] == 0.6
    assert "40%-60%" in cand.summary


def test_attack_odds_absent_when_no_probs():
    table = _build()
    cand = table.by_id()["unit:10:attack_ne"]
    assert cand.payload["odds_min"] is None
    assert cand.payload["odds_max"] is None
    assert "odds unknown" in cand.summary


def test_attack_odds_missing_entry_for_dir():
    action_probs = {10: {5: {ACTION_ATTACK: {"min": 10, "max": 20}}}}
    table = _build(action_probs=action_probs)
    cand = table.by_id()["unit:10:attack_ne"]  # dir 2, not 5
    assert cand.payload["odds_min"] is None


# ---------------------------------------------------------------------------
# ID conformance, uniqueness, determinism
# ---------------------------------------------------------------------------


def test_all_candidate_ids_match_regex_and_are_unique():
    table = _build()
    assert isinstance(table, CandidateTable)
    assert table.candidates, "expected a non-empty table"
    ids = [c.id for c in table.candidates]
    for cid in ids:
        assert CANDIDATE_ID_RE.match(cid), f"id does not conform: {cid}"
    assert len(ids) == len(set(ids))
    assert len(table.by_id()) == len(ids)


def test_table_is_pydantic_valid():
    table = _build()
    dumped = table.model_dump(mode="json")
    assert dumped["turn"] == 1
    assert len(dumped["candidates"]) == len(table.candidates)


def test_deterministic_same_input_same_table():
    t1 = _build()
    t2 = _build()
    assert t1.model_dump(mode="json") == t2.model_dump(mode="json")


def test_empty_available_actions_yields_empty_table():
    table = build_candidates(base_world(), {"turn": 7, "available_actions": {}},
                             my_player_id=MY_PLAYER)
    assert table.candidates == []
    assert table.turn == 7


def test_verb_sanitization_collision_backstop():
    """Two raw keys sanitizing to the same verb get distinct IDs."""
    actions = {
        "city": {1: {"produce_City Walls": True, "produce_City  Walls": True}},
    }
    table = build_candidates(base_world(), _info(actions),
                             my_player_id=MY_PLAYER)
    city_ids = [c.id for c in table.candidates if c.domain == "city"]
    assert len(city_ids) == 2
    assert len(set(city_ids)) == 2
    for cid in city_ids:
        assert CANDIDATE_ID_RE.match(cid)
