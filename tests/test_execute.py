# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute tests: offline candidate->action mapping, read-back verifiers,
session-facing primitives (fake session), and the guarded-close source scan.

Covers (WP1 package B acceptance):
- execute maps every exposed verb to a valid CivRealm action tuple
  (candidate_to_action over a full synthetic table);
- per-action read-back verification for every verb class, including the
  special-cased build_city two-round-trip read-back;
- the finish-phase primitive (step(None)) and the documented 'pass' no-op;
- no unguarded env.close() anywhere in the adapter package (source scan);
- the adapter modules import without civrealm; the HTTP timeout guard
  injects <= 15 s; the session timeouts align with the §3 deadlines; the
  one-env-per-process guard fires.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

from lechonk.adapter import execute as exe
from lechonk.adapter import session as session_mod
from lechonk.adapter.execute import (
    CONTRADICTED,
    INCONCLUSIVE,
    PASS_ACTION,
    VERIFIED,
    candidate_to_action,
    execute_candidate,
    execute_pass,
    finish_phase,
    verify_readback,
)
from lechonk.adapter.session import (
    AdapterSession,
    SessionStateError,
    SessionTimeoutError,
    StepOutcome,
)
from lechonk.core.schemas import Candidate

from tests.fog_fixtures import (
    ENEMY,
    MY_PLAYER,
    NUM_IMPROVEMENTS,
    SETTLERS,
    VUT_IMPROVEMENT,
    VUT_UTYPE,
    WARRIORS,
    default_dipl,
    default_techs,
    enemy_player,
    foreign_unit,
    make_observation,
    my_player,
    own_city,
    own_unit,
)
from tests.test_candidates import FULL_ACTIONS, _build as build_table

REPO = Path(__file__).resolve().parents[1]

# production display name -> (production_kind, production_value) as reported
# in the city observation (fixture_ruleset: VUT_UTYPE values carry the
# num_impr_types offset).
PRODUCTION_NAMES = {
    "Warriors": (VUT_UTYPE, WARRIORS + NUM_IMPROVEMENTS),
    "Settlers": (VUT_UTYPE, SETTLERS + NUM_IMPROVEMENTS),
    "Granary": (VUT_IMPROVEMENT, 0),
}


def _table():
    return build_table()


def _cand(cid: str) -> Candidate:
    return _table().by_id()[cid]


def _raw_cand(cid: str, domain: str, actor: str, verb: str,
              payload: dict[str, Any]) -> Candidate:
    """A candidate for a verb that FULL_ACTIONS marks invalid (join_city,
    accept_treaty): the read-back verifiers are payload-driven, so they are
    testable without a table entry."""
    return Candidate(id=cid, domain=domain, actor_id=actor, verb=verb,
                     payload=payload)


def _world(units, cities, players=None, dipl=None):
    return make_observation(
        players=players if players is not None else [my_player(), enemy_player()],
        cities=cities, units=units,
        dipl=dipl if dipl is not None else default_dipl(),
    )


def _verify_cand(cand: Candidate, pre_obs, post_obs, *, post_info=None,
                 pre_info=None, production_names=PRODUCTION_NAMES,
                 my_player_id=MY_PLAYER):
    return verify_readback(
        cand, pre_obs, post_obs,
        my_player_id=my_player_id,
        pre_info=pre_info, post_info=post_info,
        production_names=production_names)


def _verify(cid: str, pre_obs, post_obs, *, post_info=None, pre_info=None,
            production_names=PRODUCTION_NAMES, my_player_id=MY_PLAYER):
    return _verify_cand(_cand(cid), pre_obs, post_obs, post_info=post_info,
                        pre_info=pre_info, production_names=production_names,
                        my_player_id=my_player_id)


# ---------------------------------------------------------------------------
# candidate -> action tuple (offline mapping coverage, every exposed verb)
# ---------------------------------------------------------------------------


def test_candidate_to_action_maps_every_exposed_verb():
    table = _table()
    assert table.candidates, "expected a non-empty table"
    for cand in table.candidates:
        ctrl_type, actor_id, action_key = candidate_to_action(cand)
        assert ctrl_type == cand.payload["ctrl_type"]
        assert ctrl_type in ("unit", "city", "tech", "dipl")
        assert actor_id == cand.payload["actor_id"]
        assert action_key == cand.payload["action_key"]
        assert isinstance(action_key, str) and action_key


def test_candidate_to_action_rejects_malformed_payload():
    bad = Candidate(id="unit:10:sentry", domain="unit", actor_id="10",
                    verb="sentry", payload={"ctrl_type": "unit"})
    with pytest.raises(ValueError, match="payload is missing"):
        candidate_to_action(bad)


def test_candidate_to_action_rejects_empty_action_key():
    bad = Candidate(id="unit:10:sentry", domain="unit", actor_id="10",
                    verb="sentry",
                    payload={"ctrl_type": "unit", "actor_id": 10,
                             "action_key": ""})
    with pytest.raises(ValueError, match="empty action_key"):
        candidate_to_action(bad)


def test_candidate_with_nonconforming_id_is_rejected_by_schema():
    with pytest.raises(Exception):
        Candidate(id="unit:10:goto north", domain="unit", actor_id="10",
                  verb="goto north")


# ---------------------------------------------------------------------------
# unit read-backs
# ---------------------------------------------------------------------------


def test_goto_readback_verified():
    pre = _world([own_unit(10, SETTLERS, 1, 1), own_unit(11, WARRIORS, 2, 1)],
                 [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 2, 1), own_unit(11, WARRIORS, 2, 1)],
                  [own_city(1)])
    rb = _verify("unit:10:goto_e", pre, post)  # dir8=4 -> E (+1,0)
    assert rb.status == VERIFIED
    assert rb.evidence["expected"] == [2, 1]


def test_goto_readback_contradicted():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    assert _verify("unit:10:goto_e", pre, post).status == CONTRADICTED


def test_attack_readback_verified_attacker_advances():
    pre = _world([own_unit(11, WARRIORS, 2, 1), foreign_unit(20, WARRIORS, 2, 2)],
                 [own_city(1)])
    post = _world([own_unit(11, WARRIORS, 2, 2)], [own_city(1)])
    rb = _verify("unit:11:attack_s", pre, post)  # dir8=6 -> S (0,+1)
    assert rb.status == VERIFIED
    assert rb.evidence["targets"] == [20]


def test_attack_readback_verified_target_destroyed():
    pre = _world([own_unit(11, WARRIORS, 2, 1), foreign_unit(20, WARRIORS, 2, 2)],
                 [own_city(1)])
    post = _world([own_unit(11, WARRIORS, 2, 1)], [own_city(1)])
    assert _verify("unit:11:attack_s", pre, post).status == VERIFIED


def test_attack_readback_contradicted():
    pre = _world([own_unit(11, WARRIORS, 2, 1), foreign_unit(20, WARRIORS, 2, 2)],
                 [own_city(1)])
    post = _world([own_unit(11, WARRIORS, 2, 1), foreign_unit(20, WARRIORS, 2, 2)],
                  [own_city(1)])
    assert _verify("unit:11:attack_s", pre, post).status == CONTRADICTED


def test_build_city_readback_verified_two_round_trip():
    """Special-cased: new own city at the unit's tile + unit consumed."""
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([], [own_city(1), own_city(3, name="New Rome", x=1, y=1)])
    rb = _verify("unit:10:build_city", pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["new_cities"] == [3]
    assert rb.evidence["tile"] == [1, 1]


def test_build_city_readback_inconclusive_uncertain_mutation_window():
    """Unit consumed but no new city observed: reconcile, never blind replay."""
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([], [own_city(1)])
    rb = _verify("unit:10:build_city", pre, post)
    assert rb.status == INCONCLUSIVE
    assert "uncertain-mutation" in rb.detail


def test_build_city_readback_contradicted():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    assert _verify("unit:10:build_city", pre, post).status == CONTRADICTED


@pytest.mark.parametrize("cid,activity", [
    ("unit:10:fortify", 4),
    ("unit:10:sentry", 5),
    ("unit:10:build_road", 13),
    ("unit:10:irrigation", 3),
    ("unit:10:mine", 2),
])
def test_activity_readbacks_verified(cid, activity):
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1, activity=activity)],
                  [own_city(1)])
    rb = _verify(cid, pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["activity"] == activity


@pytest.mark.parametrize("cid", ["unit:10:fortify", "unit:10:build_road",
                                 "unit:10:irrigation", "unit:10:mine"])
def test_activity_readback_contradicted(cid):
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    assert _verify(cid, pre, post).status == CONTRADICTED


def test_sentry_readback_never_contradicted_upstream_quirk():
    # unit_actions.py:183-226: the read path cancels ACTIVITY_SENTRY within
    # the same step (restore_activity never restores SENTRY), so an unchanged
    # activity is NOT evidence the sentry was not sent.
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    rb = _verify("unit:10:sentry", pre, post)
    assert rb.status == INCONCLUSIVE
    assert "sentry" in rb.detail.lower()
    # a sentry activity that survives into the post-observation is VERIFIED
    post_sentry = _world([own_unit(10, SETTLERS, 1, 1, activity=5)],
                         [own_city(1)])
    assert _verify("unit:10:sentry", pre, post_sentry).status == VERIFIED


def test_fortify_readback_verified_when_post_activity_is_fortifying():
    # freeciv's in-turn fortify state: a unit ordered to fortify with
    # partial moves left shows ACTIVITY_FORTIFYING (10) in the post-step
    # observation and completes to ACTIVITY_FORTIFIED (4) at the next turn
    # start (live evidence: the WP2 model run). The in-turn state is the
    # order in effect — VERIFIED, not contradicted.
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1, activity=10)], [own_city(1)])
    rb = _verify("unit:10:fortify", pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["activity"] == 10
    assert "fortify order in effect" in rb.detail


def test_fortify_readback_inconclusive_when_already_fortifying():
    # An already-fortifying unit re-fortified stays fortifying: unchanged
    # is not attributable to this action (the same cannot-tell semantics
    # as the sentry quirk above).
    pre = _world([own_unit(10, SETTLERS, 1, 1, activity=10)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1, activity=10)], [own_city(1)])
    rb = _verify("unit:10:fortify", pre, post)
    assert rb.status == INCONCLUSIVE
    assert "already fortifying" in rb.detail


def test_join_city_readback():
    cand = _raw_cand("unit:10:join_city", "unit", "10", "join_city",
                     {"ctrl_type": "unit", "actor_id": 10,
                      "action_key": "join_city"})
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    assert _verify_cand(cand, pre, _world([], [own_city(1)])).status \
        == VERIFIED
    assert _verify_cand(cand, pre, pre).status == CONTRADICTED


def test_set_homecity_readback():
    pre = _world([own_unit(10, SETTLERS, 1, 1, home_city=1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1, home_city=3)], [own_city(1)])
    assert _verify("unit:10:set_homecity", pre, post).status == VERIFIED
    assert _verify("unit:10:set_homecity", pre, pre).status == CONTRADICTED


def test_cancel_order_readback():
    pre = _world([own_unit(10, SETTLERS, 1, 1, activity=7)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1, activity=0)], [own_city(1)])
    assert _verify("unit:10:cancel_order", pre, post).status == VERIFIED
    # nothing to cancel: not observable -> inconclusive (never contradicted)
    assert _verify("unit:10:cancel_order", post, post).status == INCONCLUSIVE


def test_keep_activity_readback_via_refreshed_actions():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post_info = {"turn": 1, "available_actions": {"unit": {}}}
    assert _verify("unit:10:keep_activity", pre, post,
                   post_info=post_info).status == VERIFIED
    post_info_still_actable = {"turn": 1, "available_actions": {
        "unit": {10: {"goto_4": True}}}}
    assert _verify("unit:10:keep_activity", pre, post,
                   post_info=post_info_still_actable).status == CONTRADICTED
    assert _verify("unit:10:keep_activity", pre, post).status == INCONCLUSIVE


# ---------------------------------------------------------------------------
# city read-backs
# ---------------------------------------------------------------------------


def test_produce_readback_verified():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])
    post = _world([own_unit(10, SETTLERS, 1, 1)],
                  [own_city(1, production_kind=VUT_UTYPE,
                            production_value=SETTLERS + NUM_IMPROVEMENTS)])
    rb = _verify("city:1:produce_settlers", pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["production_value"] == SETTLERS + NUM_IMPROVEMENTS


def test_produce_readback_contradicted():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1)])
    assert _verify("city:1:produce_settlers", pre, post).status == CONTRADICTED


def test_produce_readback_fallback_without_ruleset_map():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1, production_kind=VUT_UTYPE,
                                  production_value=SETTLERS + NUM_IMPROVEMENTS)])
    rb = _verify("city:1:produce_settlers", pre, post, production_names=None)
    assert rb.status == VERIFIED  # "production changed" fallback


def test_buy_production_readback():
    pre = _world([], [own_city(1, shield_stock=0)])
    post = _world([], [own_city(1, shield_stock=10)])
    assert _verify("city:1:city_buy_production", pre, post).status == VERIFIED
    assert _verify("city:1:city_buy_production", pre, pre).status == CONTRADICTED


def test_city_work_readback_via_refreshed_actions():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1)])
    post_info = {"turn": 1, "available_actions": {"city": {1: {}}}}
    assert _verify("city:1:city_work_0_1_0", pre, post,
                   post_info=post_info).status == VERIFIED
    post_info_still = {"turn": 1, "available_actions": {
        "city": {1: {"city_work_0_1_0": True}}}}
    assert _verify("city:1:city_work_0_1_0", pre, post,
                   post_info=post_info_still).status == CONTRADICTED
    assert _verify("city:1:city_work_0_1_0", pre, post).status == INCONCLUSIVE


# ---------------------------------------------------------------------------
# tech read-backs
# ---------------------------------------------------------------------------


def test_research_tech_readback():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1)],
                  players=[my_player(researching=0), enemy_player()])
    rb = _verify("tech:cur_player:research_tech_alphabet_0", pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["researching"] == 0
    post_other = _world([], [own_city(1)],
                        players=[my_player(researching=1), enemy_player()])
    assert _verify("tech:cur_player:research_tech_alphabet_0", pre,
                   post_other).status == CONTRADICTED


def test_set_tech_goal_readback():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1)],
                  players=[my_player(tech_goal=2), enemy_player()])
    rb = _verify("tech:cur_player:set_tech_goal_bronze_working_2", pre, post)
    assert rb.status == VERIFIED
    assert rb.evidence["tech_goal"] == 2
    post_other = _world([], [own_city(1)],
                        players=[my_player(tech_goal=0), enemy_player()])
    assert _verify("tech:cur_player:set_tech_goal_bronze_working_2", pre,
                   post_other).status == CONTRADICTED


@pytest.mark.parametrize(
    ("verb_id", "field", "expected"),
    [
        ("tech:cur_player:research_tech_alphabet_7", "researching", 7),
        ("tech:cur_player:set_tech_goal_bronze_working_7", "tech_goal", 7),
    ],
)
def test_tech_readback_uses_active_player_id_with_multiple_players(
        verb_id, field, expected):
    """Opponent records have the same marker keys; identity must be explicit."""
    pre = _world([], [own_city(1)])
    opponent = enemy_player(player_id=1, **{field: 3})
    active = my_player(player_id=2, **{field: expected})
    post = _world([], [own_city(1)], players=[opponent, active])
    verb = verb_id.rsplit(":", 1)[-1]
    cand = _raw_cand(verb_id, "tech", "cur_player", verb,
                     {"ctrl_type": "tech", "actor_id": "cur_player",
                      "action_key": verb, "tech_id": expected})
    rb = _verify_cand(cand, pre, post, my_player_id=2)
    assert rb.status == VERIFIED
    assert rb.evidence[field] == expected


@pytest.mark.parametrize(
    ("verb_id", "field"),
    [
        ("tech:cur_player:research_tech_alphabet_7", "researching"),
        ("tech:cur_player:set_tech_goal_bronze_working_7", "tech_goal"),
    ],
)
def test_tech_readback_contradicts_active_player_not_opponent(
        verb_id, field):
    pre = _world([], [own_city(1)])
    opponent = enemy_player(player_id=1, **{field: 7})
    active = my_player(player_id=2, **{field: 3})
    post = _world([], [own_city(1)], players=[opponent, active])
    verb = verb_id.rsplit(":", 1)[-1]
    cand = _raw_cand(verb_id, "tech", "cur_player", verb,
                     {"ctrl_type": "tech", "actor_id": "cur_player",
                      "action_key": verb, "tech_id": 7})
    rb = _verify_cand(cand, pre, post, my_player_id=2)
    assert rb.status == CONTRADICTED


def test_tech_readback_is_inconclusive_when_active_player_missing():
    pre = _world([], [own_city(1)])
    opponent = enemy_player(player_id=1, researching=7, tech_goal=7)
    post = _world([], [own_city(1)], players=[opponent])
    cid = "tech:cur_player:research_tech_alphabet_7"
    cand = _raw_cand(cid, "tech", "cur_player", "research_tech_alphabet_7",
                     {"ctrl_type": "tech", "actor_id": "cur_player",
                      "action_key": "research_tech_alphabet_7", "tech_id": 7})
    assert _verify_cand(cand, pre, post, my_player_id=2).status == INCONCLUSIVE


# ---------------------------------------------------------------------------
# diplomacy read-backs
# ---------------------------------------------------------------------------


def test_stop_negotiation_readback():
    # the candidate counterpart is player 2 (FULL_ACTIONS dipl actor)
    pre = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 1, "diplomacy_clause_map": ["clause"],
            "meeting_initializers": -1}})
    post = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 1, "diplomacy_clause_map": [],
            "meeting_initializers": -1}})
    assert _verify("dipl:2:stop_negotiation", pre, post).status == VERIFIED
    assert _verify("dipl:2:stop_negotiation", pre, pre).status == CONTRADICTED
    # no meeting was active -> inconclusive
    assert _verify("dipl:2:stop_negotiation", post, post).status == INCONCLUSIVE


def test_cancel_treaty_readback():
    pre = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 4, "diplomacy_clause_map": [],
            "meeting_initializers": -1}})
    post = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 1, "diplomacy_clause_map": [],
            "meeting_initializers": -1}})
    assert _verify("dipl:2:cancel_treaty", pre, post).status == VERIFIED
    assert _verify("dipl:2:cancel_treaty", pre, pre).status == CONTRADICTED


def test_accept_treaty_readback():
    # accept_treaty_2 is invalid in FULL_ACTIONS; the verifier is
    # payload-driven, so build the candidate directly.
    cand = _raw_cand("dipl:2:accept_treaty", "dipl", "2", "accept_treaty",
                     {"ctrl_type": "dipl", "actor_id": 2,
                      "action_key": "accept_treaty_2", "counterpart": 2})
    pre = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 1, "diplomacy_clause_map": ["clause"],
            "meeting_initializers": -1}})
    post = _world([], [own_city(1)], dipl={
        2: {"diplomatic_state": 1, "diplomacy_clause_map": [],
            "meeting_initializers": -1}})
    assert _verify_cand(cand, pre, post).status == VERIFIED
    assert _verify_cand(cand, pre, pre).status == CONTRADICTED


# ---------------------------------------------------------------------------
# session-facing primitives (fake session)
# ---------------------------------------------------------------------------


class FakeSession:
    """Duck-typed AdapterSession: scripted step transitions."""

    def __init__(self, obs, info, *, player_id=MY_PLAYER,
                 production_names=None, transitions=()):
        self._obs = obs
        self._info = info
        self._player_id = player_id
        self._production_names = (PRODUCTION_NAMES if production_names is None
                                  else production_names)
        self._transitions = list(transitions)
        self.step_deadline = 30.0
        self.wait_deadline = 900.0
        self.calls: list[tuple[Any, Optional[float]]] = []

    @property
    def obs(self):
        return self._obs

    @property
    def info(self):
        return self._info

    @property
    def player_id(self):
        return self._player_id

    def production_name_map(self):
        return self._production_names

    def step(self, action, *, deadline=None):
        self.calls.append((action, deadline))
        if not self._transitions:
            raise AssertionError("FakeSession: no scripted transition left")
        post_obs, post_info, terminated, truncated = self._transitions.pop(0)
        self._obs = post_obs
        self._info = post_info
        return StepOutcome(observation=post_obs, reward=0.0,
                           terminated=terminated, truncated=truncated,
                           info=post_info, action=action, elapsed_s=0.01)


def test_execute_candidate_end_to_end_verified():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)],
                 players=[my_player(), enemy_player()])
    pre_info = {"turn": 1, "available_actions": FULL_ACTIONS}
    post = _world([own_unit(10, SETTLERS, 2, 1)], [own_city(1)])
    post_info = {"turn": 1, "available_actions": {}}
    sess = FakeSession(pre, pre_info,
                       transitions=[(post, post_info, False, False)])
    result = execute_candidate(sess, _cand("unit:10:goto_e"))
    assert result.status == "executed"
    assert result.action == ("unit", 10, "goto_4")
    assert result.readback.status == VERIFIED
    assert result.turn_before == 1 and result.turn_after == 1
    assert sess.calls == [(("unit", 10, "goto_4"), 30.0)]


def test_execute_candidate_propagates_timeout_and_marks_nothing():
    pre = _world([own_unit(10, SETTLERS, 1, 1)], [own_city(1)])

    class TimeoutSession(FakeSession):
        def step(self, action, *, deadline=None):
            self.calls.append((action, deadline))
            raise SessionTimeoutError("step", 30.0)

    sess = TimeoutSession(pre, {"turn": 1, "available_actions": {}})
    with pytest.raises(SessionTimeoutError):
        execute_candidate(sess, _cand("unit:10:goto_e"))


def test_finish_phase_sends_none_under_wait_deadline():
    pre = _world([], [own_city(1)])
    post = _world([], [own_city(1)])
    sess = FakeSession(pre, {"turn": 1, "available_actions": {}},
                       transitions=[(post, {"turn": 2, "available_actions": {}},
                                     False, False)])
    outcome = finish_phase(sess)
    assert sess.calls == [(None, 900.0)]
    assert outcome.info["turn"] == 2


def test_finish_phase_honours_explicit_deadline():
    sess = FakeSession(_world([], [own_city(1)]),
                       {"turn": 1, "available_actions": {}},
                       transitions=[(_world([], [own_city(1)]),
                                     {"turn": 2, "available_actions": {}},
                                     False, False)])
    finish_phase(sess, deadline=123.0)
    assert sess.calls == [(None, 123.0)]


def test_execute_pass_is_documented_noop():
    pre = _world([], [own_city(1)])
    sess = FakeSession(pre, {"turn": 1, "available_actions": {}},
                       transitions=[(pre, {"turn": 1, "available_actions": {}},
                                     False, False)])
    outcome = execute_pass(sess)
    assert sess.calls == [(PASS_ACTION, 30.0)]
    assert outcome.action == "pass"
    assert exe.PASS_ACTION == "pass"


# ---------------------------------------------------------------------------
# guarded close: source scan (acceptance criterion 3)
# ---------------------------------------------------------------------------


def _close_calls_in(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "close"):
            target = ast.unparse(node.func.value) if node.func.value else "?"
            calls.append((node.lineno, target))
    return calls


def test_no_unguarded_env_close_in_adapter_package():
    """Every .close() call in src/lechonk/adapter is the single guarded
    self._env.close() inside AdapterSession's worker dispatch; the public
    AdapterSession.close wraps it with a hard deadline and a skip path."""
    adapter = REPO / "src" / "lechonk" / "adapter"
    all_calls = []
    for path in sorted(adapter.glob("*.py")):
        for lineno, target in _close_calls_in(path):
            all_calls.append((path.name, lineno, target))
    # the ONLY .close() call in the whole adapter package is the guarded env
    # close (inside session.py, on self._env).
    assert all_calls, "expected the guarded env close call to exist"
    for fname, _lineno, target in all_calls:
        assert fname == "session.py", f"unguarded .close() in {fname}"
        assert target == "self._env", f"unexpected .close() target {target!r}"

    # the guarded wrapper: hard deadline + skip path + RPC to the worker.
    session_src = (adapter / "session.py").read_text(encoding="utf-8")
    tree = ast.parse(session_src)
    close_fn = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name == "close"
                and any(isinstance(p, ast.Attribute) and p.attr == "close_deadline"
                        for p in ast.walk(node))):
            close_fn = node
            break
    assert close_fn is not None, "AdapterSession.close not found"
    segment = ast.get_source_segment(session_src, close_fn)
    assert "_rpc(\"close\"" in segment, "close must go through the supervised RPC"
    assert "self.close_deadline" in segment, "close must use the hard deadline"
    assert "skipped_after_deadline" in segment, "close must have a skip path"
    # and the guarded call site is documented as such
    assert "guarded env.close() call site" in session_src


def test_smoke_script_never_calls_env_close_directly():
    smoke = REPO / "scripts" / "wp1_adapter_smoke.py"
    if not smoke.exists():
        pytest.skip("smoke script not written yet")
    # AST scan (docstring mentions of env.close() are fine; calls are not):
    # the smoke may only close through the guarded session API.
    for lineno, target in _close_calls_in(smoke):
        assert target == "session", f"smoke .close() on {target!r} at {lineno}"


# ---------------------------------------------------------------------------
# adapter hygiene: offline import purity, HTTP guard, §3-aligned timeouts
# ---------------------------------------------------------------------------

ADAPTER_IMPORT_PROBE = (
    "import sys\n"
    "import lechonk.adapter.session\n"
    "import lechonk.adapter.candidates\n"
    "import lechonk.adapter.execute\n"
    "bad = sorted(m for m in sys.modules\n"
    "             if m == 'civrealm' or m.startswith('civrealm.'))\n"
    "print('CIVREALM_MODULES:', bad)\n"
    "sys.exit(1 if bad else 0)\n"
)


def test_adapter_modules_import_without_civrealm():
    proc = subprocess.run([sys.executable, "-c", ADAPTER_IMPORT_PROBE],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(REPO))
    assert proc.returncode == 0, (
        f"adapter modules transitively imported civrealm:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "CIVREALM_MODULES: []" in proc.stdout


HTTP_GUARD_PROBE = (
    "import requests\n"
    "from lechonk.adapter.session import _install_http_timeout_guard\n"
    "seen = {}\n"
    "def spy(self, method, url, **kwargs):\n"
    "    seen['timeout'] = kwargs.get('timeout')\n"
    "    class R:\n"
    "        status_code = 200\n"
    "        text = ''\n"
    "    return R()\n"
    "requests.sessions.Session.request = spy\n"
    "_install_http_timeout_guard(15.0)\n"
    "requests.get('http://127.0.0.1:1/pubstatus')\n"
    "requests.post('http://127.0.0.1:1/deletesavegame')\n"
    "assert seen['timeout'] == 15.0, seen\n"
    "print('HTTP_GUARD_OK', seen['timeout'])\n"
)


def test_http_timeout_guard_injects_bounded_timeout():
    """The upstream no-timeout HTTP calls (scorelog, listsavegames,
    deletesavegame, create_pbem_user, gamesetting) are wrapped at <= 15 s."""
    proc = subprocess.run([sys.executable, "-c", HTTP_GUARD_PROBE],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(REPO))
    assert proc.returncode == 0, (
        f"HTTP guard probe failed:\nstdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}")
    assert "HTTP_GUARD_OK 15.0" in proc.stdout


def test_session_timeouts_align_with_plan_deadlines():
    """§3: ACTING <= 180 s, WAITING <= 720 s + margin, HTTP <= 15 s,
    per-mutation send+read-back <= 30 s, teardown <= 30 s."""
    assert session_mod.SERVER_TIMEOUT_S >= 180  # raised for ACTING
    assert session_mod.BEGIN_TURN_TIMEOUT_S >= 720  # raised for WAITING + margin
    assert session_mod.HTTP_TIMEOUT_S <= 15
    assert session_mod.DEFAULT_STEP_DEADLINE_S <= 30
    assert session_mod.DEFAULT_CLOSE_DEADLINE_S <= 30
    assert session_mod.DEFAULT_WAIT_DEADLINE_S >= 720
    # deterministic seeds: all three set explicitly before reset (module
    # docstring / _setup_env)
    setup_src = (REPO / "src" / "lechonk" / "adapter" / "session.py").read_text()
    for key in ("debug.randomly_generate_seeds", "debug.mapseed",
                "debug.gameseed", "debug.agentseed", "self_play"):
        assert f'fc_args["{key}"]' in setup_src, f"fc_args[{key!r}] not set"


def test_one_env_per_process_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(session_mod, "_ENV_EVER_CREATED", True)
    sess = AdapterSession(workdir=str(tmp_path), seed=1)
    with pytest.raises(SessionStateError, match="one env per process"):
        sess.start()
