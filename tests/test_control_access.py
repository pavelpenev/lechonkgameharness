import pytest

from lechonk.core.candidate_surface_v3 import admit_candidate_surface
from lechonk.core.control_access import assess_control_access
from lechonk.adapter.candidates import build_candidates
from lechonk.core.schemas import (
    Candidate,
    CandidateTable,
    CityView,
    MapView,
    PlayerView,
    Snapshot,
    UnitView,
)


def _candidate(domain, actor, verb):
    return Candidate(
        id=f"{domain}:{actor}:{verb}", domain=domain, actor_id=str(actor),
        verb=verb, payload={"actor_id": actor}, summary=verb,
    )


def _snapshot(*, include_settler=True, include_city=True):
    return Snapshot(
        turn=3,
        player=PlayerView(player_id=0, name="Le Chonk"),
        cities=([CityView(id=1, name="Roma", x=2, y=2, is_own=True,
                          owner_id=0)] if include_city else []),
        units=([UnitView(id=159, owner_id=0, is_own=True,
                         type_name="Settlers", x=3, y=3, moves_left=3)]
                if include_settler else []),
        map_view=MapView(width=8, height=8),
    )


def _table():
    return CandidateTable(turn=3, candidates=[
        _candidate("unit", 159, "build_city"),
        _candidate("unit", 159, "goto_n"),
        _candidate("city", 1, "produce_warrior"),
        _candidate("city", 1, "produce_settler"),
        _candidate("tech", "cur_player", "research_tech_alphabet_1"),
    ])


def test_control_access_passes_when_v3_keeps_classic_controls():
    table = _table()
    offered = admit_candidate_surface(table).offered
    report = assess_control_access(
        table, offered, _snapshot(),
        {"unit": {159: {"owner": 0, "type_rule_name": "Settlers"}},
         "city": {1: {"owner": 0}}},
        my_player_id=0,
    )
    assert report["passed"] is True
    assert report["missing_ref_ids"] == []


def test_control_access_fails_when_settler_state_is_elided():
    table = _table()
    report = assess_control_access(
        table, table, _snapshot(include_settler=False),
        {"unit": {159: {"owner": 0, "type_rule_name": "Settlers"}},
         "city": {1: {"owner": 0}}},
        my_player_id=0,
    )
    assert report["passed"] is False
    assert "159" in report["missing_unit_state_ids"]


def test_control_access_fails_when_a_required_ref_is_omitted():
    table = _table()
    offered = CandidateTable(turn=3, candidates=table.candidates[:-1])
    report = assess_control_access(
        table, offered, _snapshot(),
        {"unit": {159: {"owner": 0, "type_rule_name": "Settlers"}},
         "city": {1: {"owner": 0}}},
        my_player_id=0,
    )
    assert report["passed"] is False
    assert "tech:cur_player:research_tech_alphabet_1" in report["missing_ref_ids"]


def test_control_access_legacy_starvation_is_visible():
    table = CandidateTable(turn=3, candidates=[
        *_table().candidates,
        *[_candidate("unit", i, "goto_n") for i in range(200)],
    ])
    offered = table.__class__(turn=3, candidates=table.candidates[:2])
    report = assess_control_access(
        table, offered, _snapshot(),
        {"unit": {159: {"owner": 0, "type_rule_name": "Settlers"}},
         "city": {1: {"owner": 0}}},
        my_player_id=0,
    )
    assert report["passed"] is False
    assert report["missing_ref_ids"]


def _real_fixture(unit_count=50, city_count=8):
    unit_ids = [159, 171] + [i for i in range(unit_count + 2)
                              if i not in (159, 171)][:unit_count - 2]
    units = {
        uid: {"owner": 0, "moves_left": 3, "ssa_controller": 0,
              "type_rule_name": "Settlers" if uid in (159, 171) else "Warriors"}
        for uid in unit_ids
    }
    available_units = {
        str(uid): {**{f"goto_{i}": True for i in range(8)},
                   "keep_activity": True,
                   **({"build_city": True, "build_road": True,
                       "irrigation": True, "mine": True}
                      if uid in (159, 171) else {"fortify": True})}
        for uid in unit_ids
    }
    production_names = ["Warriors", "Settlers", "Workers", "Granary",
                        "Barracks", "Temple", "City Walls", "Phalanx",
                        "Archers", "Library", "Marketplace", "Harbor"]
    cities = {str(cid): {f"produce_{name}": True for name in production_names}
              for cid in range(city_count)}
    tech_names = ["Alphabet", "Bronze Working", "Pottery", "Masonry",
                  "Horseback Riding", "Warrior Code"]
    available = {"unit": available_units, "city": cities,
                 "tech": {"cur_player": {
                     f"research_tech_{name}_{i}": True
                     for i, name in enumerate(tech_names)}}}
    observation = {"unit": units,
                   "city": {str(cid): {"owner": 0} for cid in range(city_count)}}
    table = build_candidates(
        observation, {"turn": 3, "available_actions": available},
        my_player_id=0,
        unit_flags={uid: {} for uid in unit_ids},
        turn=3,
    )
    snapshot = Snapshot(
        turn=3, player=PlayerView(player_id=0, name="Le Chonk"),
        cities=[CityView(id=cid, name=f"C{cid}", x=cid % 80,
                         y=20 + cid // 80, is_own=True, owner_id=0)
                for cid in range(city_count)],
        units=[UnitView(id=uid, owner_id=0, is_own=True,
                        type_name=units[uid]["type_rule_name"], x=uid % 80,
                        y=uid // 80,
                        moves_left=3) for uid in unit_ids],
        map_view=MapView(width=80, height=50),
    )
    return observation, table, snapshot


def test_real_build_candidates_v3_keeps_rehearsal_critical_controls():
    observation, table, snapshot = _real_fixture()
    offered = admit_candidate_surface(table).offered
    report = assess_control_access(table, offered, snapshot, observation,
                                   my_player_id=0)
    assert report["passed"] is True
    assert {"159", "171"} <= set(report["required_unit_ids"])
    assert report["missing_ref_ids"] == []


def test_real_build_candidates_legacy_surface_fails_same_fixture():
    observation, table, snapshot = _real_fixture()
    offered = table.offered()
    report = assess_control_access(table, offered, snapshot, observation,
                                   my_player_id=0)
    assert report["passed"] is False
    assert report["missing_ref_ids"]


@pytest.mark.parametrize("remove", [
    "unit:159:goto_n", "unit:159:build_city",
    "city:0:produce_warriors", "tech:cur_player:research_tech_alphabet_0",
])
def test_real_build_candidates_gate_identifies_missing_critical_ref(remove):
    observation, table, snapshot = _real_fixture()
    offered = CandidateTable(turn=3, candidates=[
        c for c in table.candidates if c.id != remove])
    report = assess_control_access(table, offered, snapshot, observation,
                                   my_player_id=0)
    assert report["passed"] is False
    assert remove in report["missing_ref_ids"]


def test_busy_settler_with_no_legal_action_can_pass_if_state_is_present():
    observation, table, snapshot = _real_fixture()
    observation["unit"][159]["moves_left"] = 0
    table = CandidateTable(turn=3, candidates=[
        c for c in table.candidates if c.actor_id != "159"])
    report = assess_control_access(table, table, snapshot, observation,
                                   my_player_id=0)
    assert report["passed"] is True


def test_no_raw_production_is_reported_as_unavailable_not_failure():
    observation, table, snapshot = _real_fixture()
    table = CandidateTable(turn=3, candidates=[
        c for c in table.candidates
        if not (c.domain == "city" and c.actor_id == "0")])
    report = assess_control_access(table, table, snapshot, observation,
                                   my_player_id=0)
    assert report["passed"] is True
    assert "0" in report["unavailable_city_production_ids"]


def test_unknown_raw_maps_fail_but_valid_empty_world_passes():
    table = CandidateTable(turn=1, candidates=[])
    empty_snapshot = _snapshot(include_settler=False, include_city=False)
    empty_snapshot.units = []
    empty_snapshot.cities = []
    valid = assess_control_access(table, table, empty_snapshot,
                                  {"unit": {}, "city": {}}, my_player_id=0)
    assert valid["passed"] is True
    unknown = assess_control_access(table, table, empty_snapshot,
                                   {"unit": [], "city": {}}, my_player_id=0)
    assert unknown["passed"] is False
    assert "raw_observation_unknown:unit" in unknown["issues"]


@pytest.mark.parametrize("unit_count,city_count", [(100, 20), (512, 64)])
def test_real_admission_stress_reports_actual_critical_coverage(unit_count, city_count):
    observation, table, snapshot = _real_fixture(unit_count, city_count)
    offered = admit_candidate_surface(table).offered
    report = assess_control_access(table, offered, snapshot, observation,
                                   my_player_id=0)
    assert report["raw_total"] == len(table.candidates)
    assert report["offered_total"] == len(offered.candidates)
    if unit_count == 512:
        assert report["offered_total"] < report["raw_total"]
        assert report["passed"] is False
        assert any((ref.startswith("unit:159:goto_")
                    or ref.startswith("unit:171:goto_")
                    or ref.startswith("city:"))
                   for ref in report["missing_ref_ids"])
    else:
        assert report["passed"] is True
