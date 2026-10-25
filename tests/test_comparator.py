# SPDX-License-Identifier: MIT
"""Comparator + degraded-default policy tests (WP1 package C, plan §7.4).

Offline: synthetic core.schemas Snapshot + CandidateTable fixtures drive both
entry points (lechonk.comparator.scripted.decide and
lechonk.comparator.defaults.decide). Covers:
- research priority list + tie-breaks (lowest cost, then list order),
  trade-adjacent tier, the classic "The Wheel" alias, no-op when already on
  the priority target;
- the production ladder (defensive unit / builder unit / growth / science /
  wealth / fallback) incl. the builders < cities+1 rule and the buy rule
  (gold > 2x buy_cost and shield stock < 25% complete, simulated gold);
- production hysteresis (§7.4 AMENDMENT 2026-10-24): a satisfied top
  applicable tier does not switch, a lower-tier satisfaction never
  suppresses a higher-priority switch, and the degraded defaults are
  unaffected;
- unit defaults: found city (one founding per turn), improve by terrain
  yield, fortify, military defend-the-nearest-city, attack with favorable
  odds (and candidate-gated attack without odds data, plan §6);
- diplomacy default: stop all negotiations, never accept/cancel;
- the comparator emits only whitelisted verbs (excluded verbs —
  exploration/autosettler/etc. — never appear even when present in the
  table);
- degraded-default behavior: preserves orders/production, resolves only the
  new mandatory choices, logged with reason scripted_default;
- the two entry points are separate functions with a proven behavioral
  difference;
- determinism (same inputs -> identical decision log).
"""

from __future__ import annotations

from lechonk.comparator import defaults, scripted
from lechonk.comparator.scripted import (
    PROVENANCE_CLASS,
    is_exposed_verb,
)
from lechonk.core.schemas import (
    Candidate,
    CandidateTable,
    CityView,
    DiplomacyView,
    MapView,
    PlayerView,
    ResearchCandidate,
    ResearchView,
    Snapshot,
    UnitView,
)

# ---------------------------------------------------------------------------
# Synthetic Snapshot / CandidateTable builders
# ---------------------------------------------------------------------------


def _city(cid=1, name="Rome", x=1, y=1, *, production=None, buy_cost=20,
          shield_stock=0, production_progress=None, is_own=True):
    return CityView(id=cid, name=name, x=x, y=y, is_own=is_own,
                    production=production, buy_cost=buy_cost,
                    shield_stock=shield_stock,
                    production_progress=production_progress)


def _unit(uid=10, type_name="Settlers", x=1, y=1, *, terrain="Plains",
          activity_name="idle", is_own=True):
    return UnitView(id=uid, is_own=is_own, type_name=type_name, x=x, y=y,
                    terrain=terrain, activity_name=activity_name)


def _snap(*, turn=1, gold=50, target=None, target_id=None, cities=(),
          units=(), research_cands=(), dipl=()):
    return Snapshot(
        turn=turn,
        player=PlayerView(player_id=0, gold=gold,
                          research=ResearchView(current_target=target,
                                                current_target_id=target_id)),
        research_candidates=list(research_cands),
        cities=list(cities),
        units=list(units),
        map_view=MapView(width=10, height=10),
        diplomacy=list(dipl),
    )


def _cand(domain, actor, verb, **payload):
    cid = f"{domain}:{actor}:{verb}"
    return Candidate(id=cid, domain=domain, actor_id=str(actor), verb=verb,
                     summary="", payload=payload)


def _table(*cands, turn=1):
    return CandidateTable(turn=turn, candidates=list(cands))


def _tech_cand(name, tech_id, cost=None):
    verb = "research_tech_" + name.lower().replace(" ", "_").replace(".", "") \
        + f"_{tech_id}"
    return _cand("tech", "cur_player", verb, ctrl_type="tech",
                 actor_id="cur_player", action_key=f"research_tech_{name}_{tech_id}",
                 tech_id=tech_id, tech_name=name)


def _rc(tech_id, name, cost=None):
    return ResearchCandidate(tech_id=tech_id, name=name, cost=cost)


def _produce_cand(city_id, name):
    verb = "produce_" + name.lower().replace(" ", "_").replace(".", "")
    return _cand("city", str(city_id), verb, ctrl_type="city",
                 actor_id=city_id, action_key=f"produce_{name}",
                 production=name)


def _buy_cand(city_id):
    return _cand("city", str(city_id), "city_buy_production", ctrl_type="city",
                 actor_id=city_id, action_key="city_buy_production")


def _goto_cand(uid, compass, dir8):
    return _cand("unit", str(uid), f"goto_{compass}", ctrl_type="unit",
                 actor_id=uid, action_key=f"goto_{dir8}", dir=compass.upper(),
                 dir8=dir8)


def _attack_cand(uid, compass, dir8, odds_min=None, odds_max=None):
    payload = {"ctrl_type": "unit", "actor_id": uid,
               "action_key": f"attack_{dir8}", "dir": compass.upper(),
               "dir8": dir8}
    if odds_min is not None:
        payload["odds_min"] = odds_min
    if odds_max is not None:
        payload["odds_max"] = odds_max
    return _cand("unit", str(uid), f"attack_{compass}", **payload)


def _unit_cand(uid, verb):
    return _cand("unit", str(uid), verb, ctrl_type="unit", actor_id=uid,
                 action_key=verb)


def _dipl_cand(counterpart, verb):
    return _cand("dipl", str(counterpart), verb, ctrl_type="dipl",
                 actor_id=counterpart, action_key=f"{verb}_{counterpart}",
                 counterpart=counterpart)


def _decisions_by_verb(log):
    return [d.verb for d in log.decisions]


def _decisions_for_actor(log, domain, actor_id):
    return [d for d in log.decisions
            if d.domain == domain and d.actor_id == str(actor_id)]


# ---------------------------------------------------------------------------
# Research priority list (§7.4)
# ---------------------------------------------------------------------------


def test_research_priority_list_order_beats_cost():
    # Alphabet (cost 30) must beat Pottery (cost 10): the published list
    # order dominates the cost tie-break.
    snap = _snap(research_cands=[_rc(2, "Alphabet", 30), _rc(63, "Pottery", 10)])
    table = _table(_tech_cand("Pottery", 63), _tech_cand("Alphabet", 2))
    log = scripted.decide(snap, table)
    tech = [d for d in log.decisions if d.domain == "tech"]
    assert len(tech) == 1
    assert tech[0].reason == "research_priority_list"
    assert "alphabet" in tech[0].candidate_id


def test_research_tiebreak_lowest_cost_then_list_order():
    # Unlisted techs (tier 2): lowest cost wins...
    snap = _snap(research_cands=[_rc(40, "Map Making", 30),
                                 _rc(10, "Ceremonial Burial", 20)])
    table = _table(_tech_cand("Map Making", 40), _tech_cand("Ceremonial Burial", 10))
    log = scripted.decide(snap, table)
    tech = [d for d in log.decisions if d.domain == "tech"]
    assert len(tech) == 1
    assert "ceremonial_burial" in tech[0].candidate_id
    # ...and equal costs break by candidate-table (list) order.
    snap2 = _snap(research_cands=[_rc(40, "Map Making", 20),
                                  _rc(10, "Ceremonial Burial", 20)])
    table2 = _table(_tech_cand("Map Making", 40), _tech_cand("Ceremonial Burial", 10))
    log2 = scripted.decide(snap2, table2)
    tech2 = [d for d in log2.decisions if d.domain == "tech"]
    assert "map_making" in tech2[0].candidate_id  # first in table order


def test_research_trade_adjacent_tier_beats_cheaper_unlisted():
    # Trade (trade-adjacent) beats a cheaper unlisted tech.
    snap = _snap(research_cands=[_rc(80, "Trade", 50), _rc(40, "Map Making", 20)])
    table = _table(_tech_cand("Trade", 80), _tech_cand("Map Making", 40))
    log = scripted.decide(snap, table)
    tech = [d for d in log.decisions if d.domain == "tech"]
    assert "trade" in tech[0].candidate_id


def test_research_wheel_alias_matches_classic_the_wheel():
    # The published list says "Wheel"; classic names the tech "The Wheel".
    snap = _snap(research_cands=[_rc(86, "The Wheel", 40),
                                 _rc(35, "Horseback Riding", 30)])
    table = _table(_tech_cand("The Wheel", 86), _tech_cand("Horseback Riding", 35))
    log = scripted.decide(snap, table)
    tech = [d for d in log.decisions if d.domain == "tech"]
    assert "the_wheel" in tech[0].candidate_id


def test_research_no_action_when_already_on_priority_target():
    snap = _snap(target="Alphabet", target_id=2,
                 research_cands=[_rc(2, "Alphabet", 30)])
    table = _table(_tech_cand("Alphabet", 2))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "tech"] == []


def test_research_switches_off_non_priority_target():
    snap = _snap(target="Bronze Working", target_id=9,
                 research_cands=[_rc(9, "Bronze Working", 28),
                                 _rc(2, "Alphabet", 20)])
    table = _table(_tech_cand("Bronze Working", 9), _tech_cand("Alphabet", 2))
    log = scripted.decide(snap, table)
    tech = [d for d in log.decisions if d.domain == "tech"]
    assert len(tech) == 1
    assert "alphabet" in tech[0].candidate_id


# ---------------------------------------------------------------------------
# Production ladder + buy rule (§7.4)
# ---------------------------------------------------------------------------


def test_production_undefended_city_gets_defensive_unit():
    snap = _snap(cities=[_city(1, production=None)])
    table = _table(_produce_cand(1, "Settlers"), _produce_cand(1, "Warriors"),
                   _produce_cand(1, "Granary"))
    log = scripted.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "production_ladder:defensive_unit_undefended_city"
    assert city[0].verb == "produce_warriors"


def test_production_defended_city_builder_while_improvements_remain():
    # A military unit on the city tile defends it; improvements remain
    # (Granary candidate) and builders (0) < cities+1 (2) -> builder unit.
    snap = _snap(cities=[_city(1, production="Warriors")],
                 units=[_unit(11, "Warriors", 1, 1)])
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"),
                   _produce_cand(1, "Workers"), _produce_cand(1, "Granary"))
    log = scripted.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "production_ladder:builder_unit_improvements_remain"
    assert city[0].verb == "produce_settlers"  # Settlers before Workers


def test_production_builder_tier_skipped_when_builders_full():
    # builders (2 Settlers/Workers) >= cities+1 (2) -> growth building.
    snap = _snap(cities=[_city(1, production="Warriors")],
                 units=[_unit(11, "Warriors", 1, 1),
                        _unit(10, "Settlers", 2, 2),
                        _unit(12, "Workers", 3, 3)])
    table = _table(_produce_cand(1, "Settlers"), _produce_cand(1, "Granary"))
    log = scripted.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "production_ladder:growth_building"
    assert city[0].verb == "produce_granary"


def test_production_science_then_wealth_then_fallback():
    # Defended + builders full: science building beats wealth; wealth beats
    # the final fallback (any remaining legal production, e.g. City Walls).
    base_snap = _snap(cities=[_city(1, production="Warriors")],
                      units=[_unit(11, "Warriors", 1, 1),
                             _unit(10, "Settlers", 2, 2),
                             _unit(12, "Workers", 3, 3)])
    log = scripted.decide(base_snap, _table(_produce_cand(1, "Library"),
                                            _produce_cand(1, "Marketplace")))
    city = [d for d in log.decisions if d.domain == "city"]
    assert city[0].reason == "production_ladder:science_building"
    assert city[0].verb == "produce_library"

    log = scripted.decide(base_snap, _table(_produce_cand(1, "Marketplace")))
    city = [d for d in log.decisions if d.domain == "city"]
    assert city[0].reason == "production_ladder:wealth_building"
    assert city[0].verb == "produce_marketplace"

    log = scripted.decide(base_snap, _table(_produce_cand(1, "City Walls")))
    city = [d for d in log.decisions if d.domain == "city"]
    assert city[0].reason == "production_ladder:fallback_production"
    assert city[0].verb == "produce_city_walls"


def test_production_no_switch_when_already_set():
    snap = _snap(cities=[_city(1, production="Warriors")])
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "city"] == []


def test_buy_rule_fires_when_gold_rich_and_progress_low():
    snap = _snap(gold=100, cities=[_city(1, production="Warriors",
                                        buy_cost=20, production_progress=0.1)])
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1))
    log = scripted.decide(snap, table)
    buys = [d for d in log.decisions if d.verb == "city_buy_production"]
    assert len(buys) == 1
    assert buys[0].reason == "buy_production_rule"


def test_buy_rule_gold_threshold_is_strictly_greater_than_2x():
    # gold == 2x buy_cost is NOT > 2x buy_cost -> no buy.
    snap = _snap(gold=40, cities=[_city(1, production="Warriors",
                                       buy_cost=20, production_progress=0.1)])
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.verb == "city_buy_production"] == []
    # gold = 2x + 1 -> buy.
    snap2 = _snap(gold=41, cities=[_city(1, production="Warriors",
                                        buy_cost=20, production_progress=0.1)])
    log2 = scripted.decide(snap2, table)
    assert len([d for d in log2.decisions
                if d.verb == "city_buy_production"]) == 1


def test_buy_rule_progress_threshold_is_below_25_percent():
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1))
    snap_low = _snap(gold=100, cities=[_city(1, production="Warriors",
                                            buy_cost=20,
                                            production_progress=0.24)])
    snap_high = _snap(gold=100, cities=[_city(1, production="Warriors",
                                             buy_cost=20,
                                             production_progress=0.25)])
    assert len([d for d in scripted.decide(snap_low, table).decisions
                if d.verb == "city_buy_production"]) == 1
    assert [d for d in scripted.decide(snap_high, table).decisions
            if d.verb == "city_buy_production"] == []


def test_buy_rule_requires_a_production_target():
    # Nothing in production -> nothing to buy, whatever the gold.
    snap = _snap(gold=1000, cities=[_city(1, production=None, buy_cost=20,
                                         production_progress=None)])
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.verb == "city_buy_production"] == []


def test_buy_rule_skipped_when_ladder_switches_production():
    # The ladder switches Warriors -> Settlers this turn; the buy would
    # complete the NEW item, so it is skipped.
    snap = _snap(gold=1000, cities=[_city(1, production="Warriors",
                                         buy_cost=20, production_progress=0.1)],
                 units=[_unit(11, "Warriors", 1, 1)])  # defended
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"),
                   _produce_cand(1, "Granary"), _buy_cand(1))
    log = scripted.decide(snap, table)
    verbs = _decisions_by_verb(log)
    assert "produce_settlers" in verbs  # the ladder switch fires
    assert "city_buy_production" not in verbs  # ...and the buy is skipped


def test_buy_rule_simulates_gold_across_cities():
    # Two cities, buy_cost 20 each: gold 60 buys the first (60 > 40, sim 40)
    # but not the second (40 is not > 40).
    cities = [_city(1, "Rome", 1, 1, production="Warriors", buy_cost=20,
                    production_progress=0.1),
              _city(2, "Utica", 5, 5, production="Warriors", buy_cost=20,
                    production_progress=0.1)]
    units = [_unit(11, "Warriors", 1, 1), _unit(12, "Warriors", 5, 5)]
    snap = _snap(gold=60, cities=cities, units=units)
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1),
                   _produce_cand(2, "Warriors"), _buy_cand(2))
    log = scripted.decide(snap, table)
    buys = [d for d in log.decisions if d.verb == "city_buy_production"]
    assert [b.actor_id for b in buys] == ["1"]
    # gold 61: both buys fit (61 > 40 -> sim 41; 41 > 40 -> buy).
    snap2 = _snap(gold=61, cities=cities, units=units)
    log2 = scripted.decide(snap2, table)
    buys2 = [d for d in log2.decisions if d.verb == "city_buy_production"]
    assert [b.actor_id for b in buys2] == ["1", "2"]


# ---------------------------------------------------------------------------
# Production hysteresis (§7.4 AMENDMENT 2026-10-24)
# ---------------------------------------------------------------------------


def test_hysteresis_undefended_city_keeps_defensive_unit_production():
    # (a) The city already produces a defensive unit (Warriors) and is
    # undefended: the defensive tier — the top applicable tier — is
    # SATISFIED by the current target, so no switch. The candidate table
    # omits Warriors (upstream never lists the current production target);
    # without hysteresis the ladder would fall through and switch (the
    # Warriors/Barracks 2-cycle of artifacts/runs/wp1-comparator/).
    snap = _snap(cities=[_city(1, production="Warriors")])  # no units: undefended
    table = _table(_produce_cand(1, "Settlers"), _produce_cand(1, "Barracks"))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "city"] == []


def test_hysteresis_undefended_city_producing_barracks_switches_to_defensive():
    # (b) Producing Barracks on an undefended city: Barracks does not
    # satisfy the defensive tier, so the ladder switches to the defensive
    # unit (the hysteresis rule never suppresses the top tier's switch).
    snap = _snap(cities=[_city(1, production="Barracks")])  # no units: undefended
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"))
    log = scripted.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "production_ladder:defensive_unit_undefended_city"
    assert city[0].verb == "produce_warriors"


def test_hysteresis_does_not_suppress_higher_priority_switch():
    # (c) The city is defended and produces a growth building (Granary —
    # the growth tier, a LOWER priority than the builder tier). The builder
    # tier is the top applicable tier (improvements remain via the Barracks
    # candidate, builders 0 < cities+1) and Granary does not satisfy it ->
    # switch to the builder unit. A current target satisfying only a lower
    # tier never protects it.
    snap = _snap(cities=[_city(1, production="Granary")],
                 units=[_unit(11, "Warriors", 1, 1)])  # defended
    table = _table(_produce_cand(1, "Settlers"), _produce_cand(1, "Barracks"))
    log = scripted.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "production_ladder:builder_unit_improvements_remain"
    assert city[0].verb == "produce_settlers"


def test_hysteresis_keeps_satisfied_growth_building():
    # Defended and builders full: the growth tier is the top applicable
    # tier and the current target (Granary, absent from the candidate
    # table as the current production) satisfies it -> no switch to the
    # science building.
    snap = _snap(cities=[_city(1, production="Granary")],
                 units=[_unit(11, "Warriors", 1, 1),
                        _unit(10, "Settlers", 2, 2),
                        _unit(12, "Workers", 3, 3)])
    table = _table(_produce_cand(1, "Library"))
    log = scripted.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "city"] == []


def test_hysteresis_leaves_degraded_defaults_unaffected():
    # (d) The degraded-default entry point preserves production anyway and
    # only resolves EMPTY production; the hysteresis change in the shared
    # production_pick must not leak into it: a city already producing
    # Warriors (undefended, Warriors absent from the table) gets no
    # decision from defaults either.
    snap = _snap(cities=[_city(1, production="Warriors")])
    table = _table(_produce_cand(1, "Settlers"), _produce_cand(1, "Barracks"))
    log = defaults.decide(snap, table)
    assert log.policy == "degraded_default"
    assert [d for d in log.decisions if d.domain == "city"] == []
    # ...and EMPTY production still resolves via the unchanged ladder
    # (undefended, no defensive candidate in the table -> builder tier).
    snap2 = _snap(cities=[_city(1, production=None)])
    log2 = defaults.decide(snap2, table)
    city2 = [d for d in log2.decisions if d.domain == "city"]
    assert len(city2) == 1
    assert city2[0].reason == "scripted_default"
    assert city2[0].trigger == "production_empty"
    assert city2[0].verb == "produce_settlers"


# ---------------------------------------------------------------------------
# Unit defaults (§7.4)
# ---------------------------------------------------------------------------


def test_unit_settler_founds_first_legal_candidate_one_per_turn():
    snap = _snap(units=[_unit(10, "Settlers", 1, 1),
                        _unit(11, "Settlers", 2, 2)])
    table = _table(_unit_cand(10, "build_city"), _unit_cand(11, "build_city"))
    log = scripted.decide(snap, table)
    found = [d for d in log.decisions if d.verb == "build_city"]
    assert len(found) == 1  # "no city is being founded this turn"
    assert found[0].actor_id == "10"  # the first legal candidate tile
    assert found[0].reason == "unit_default:found_city"


def test_unit_second_settler_improves_instead_of_founding():
    snap = _snap(units=[_unit(10, "Settlers", 1, 1),
                        _unit(11, "Settlers", 2, 2, terrain="Plains")])
    table = _table(_unit_cand(10, "build_city"), _unit_cand(11, "build_city"),
                   _unit_cand(11, "irrigation"))
    log = scripted.decide(snap, table)
    unit11 = _decisions_for_actor(log, "unit", 11)
    assert len(unit11) == 1
    assert unit11[0].verb == "irrigation"
    assert unit11[0].reason == "unit_default:improve:irrigation"


def test_unit_improve_by_terrain_yield():
    # Plains -> irrigation, Hills -> mine, Tundra -> road.
    for terrain, verb in (("Plains", "irrigation"), ("Hills", "mine"),
                          ("Tundra", "build_road")):
        snap = _snap(units=[_unit(10, "Workers", 1, 1, terrain=terrain)])
        table = _table(_unit_cand(10, "irrigation"), _unit_cand(10, "mine"),
                       _unit_cand(10, "build_road"))
        log = scripted.decide(snap, table)
        unit = _decisions_for_actor(log, "unit", 10)
        assert len(unit) == 1, terrain
        assert unit[0].verb == verb, terrain
        assert unit[0].reason == f"unit_default:improve:{verb}"


def test_unit_improve_falls_back_through_road_irrigation_mine():
    # Plains prefers irrigation, but with no irrigation candidate the
    # fallback order is road, then irrigation, then mine.
    snap = _snap(units=[_unit(10, "Workers", 1, 1, terrain="Plains")])
    table = _table(_unit_cand(10, "mine"), _unit_cand(10, "build_road"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 10)
    assert unit[0].verb == "build_road"


def test_unit_already_active_gets_no_action():
    snap = _snap(units=[_unit(10, "Workers", 1, 1, terrain="Plains",
                              activity_name="irrigate")])
    table = _table(_unit_cand(10, "irrigation"))
    log = scripted.decide(snap, table)
    assert _decisions_for_actor(log, "unit", 10) == []


def test_unit_military_defends_nearest_city():
    # Warrior at (3,1), cities at (1,1) and (5,5): nearest is (1,1) (dist 2);
    # goto_w strictly reduces the distance -> defend move.
    snap = _snap(cities=[_city(1, "Rome", 1, 1), _city(2, "Utica", 5, 5)],
                 units=[_unit(11, "Warriors", 3, 1)])
    table = _table(_goto_cand(11, "n", 1), _goto_cand(11, "e", 4),
                   _goto_cand(11, "w", 3))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert len(unit) == 1
    assert unit[0].verb == "goto_w"
    assert unit[0].reason == "unit_default:defend_city"


def test_unit_military_defends_the_nearer_of_two_cities():
    # Warrior at (4,4): dist 1 to Utica (5,5), dist 3 to Rome (1,1).
    # goto_n does not reduce the distance to Utica; goto_se does.
    snap = _snap(cities=[_city(1, "Rome", 1, 1), _city(2, "Utica", 5, 5)],
                 units=[_unit(11, "Warriors", 4, 4)])
    table = _table(_goto_cand(11, "n", 1), _goto_cand(11, "e", 4),
                   _goto_cand(11, "se", 7))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert unit[0].verb == "goto_se"
    assert "Utica" in unit[0].detail


def test_unit_military_on_city_fortifies():
    snap = _snap(cities=[_city(1, "Rome", 1, 1)],
                 units=[_unit(11, "Warriors", 1, 1)])
    table = _table(_unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert len(unit) == 1
    assert unit[0].verb == "fortify"
    assert unit[0].reason == "unit_default:fortify"


def test_unit_military_on_city_already_fortified_holds():
    snap = _snap(cities=[_city(1, "Rome", 1, 1)],
                 units=[_unit(11, "Warriors", 1, 1, activity_name="fortified")])
    table = _table(_unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    assert _decisions_for_actor(log, "unit", 11) == []


def test_unit_military_attacks_adjacent_target_with_favorable_odds():
    snap = _snap(cities=[_city(1, "Rome", 1, 1)],
                 units=[_unit(11, "Warriors", 1, 1)])
    table = _table(_attack_cand(11, "se", 7, odds_min=0.6, odds_max=0.75),
                   _unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert len(unit) == 1
    assert unit[0].verb == "attack_se"
    assert unit[0].reason == "unit_default:attack_favorable"


def test_unit_military_does_not_attack_unfavorable_odds():
    # odds_max 0.3 < 0.5: no attack; the defender fortifies instead.
    snap = _snap(cities=[_city(1, "Rome", 1, 1)],
                 units=[_unit(11, "Warriors", 1, 1)])
    table = _table(_attack_cand(11, "se", 7, odds_min=0.2, odds_max=0.3),
                   _unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert len(unit) == 1
    assert unit[0].verb == "fortify"
    assert "attack" not in unit[0].reason


def test_unit_military_attacks_without_odds_data_when_unavailable():
    # Plan §6 residual: with odds unavailable the attack is candidate-gated.
    snap = _snap(cities=[_city(1, "Rome", 1, 1)],
                 units=[_unit(11, "Warriors", 1, 1)])
    table = _table(_attack_cand(11, "se", 7),  # no odds payload at all
                   _unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert len(unit) == 1
    assert unit[0].verb == "attack_se"
    assert unit[0].reason == "unit_default:attack_favorable"


def test_unit_military_without_cities_attacks_or_fortifies():
    snap = _snap(units=[_unit(11, "Warriors", 3, 3)])
    table = _table(_attack_cand(11, "n", 1, odds_min=0.9, odds_max=0.95),
                   _unit_cand(11, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 11)
    assert unit[0].verb == "attack_n"
    # No attack candidates and no cities -> fortify.
    table2 = _table(_unit_cand(11, "fortify"))
    log2 = scripted.decide(snap, table2)
    unit2 = _decisions_for_actor(log2, "unit", 11)
    assert unit2[0].verb == "fortify"


def test_unit_civilian_without_improve_verbs_fortifies():
    snap = _snap(units=[_unit(10, "Workers", 1, 1, terrain="Plains")])
    table = _table(_unit_cand(10, "fortify"))
    log = scripted.decide(snap, table)
    unit = _decisions_for_actor(log, "unit", 10)
    assert unit[0].verb == "fortify"
    assert unit[0].reason == "unit_default:fortify"


def test_unit_absent_from_table_gets_no_decision():
    snap = _snap(units=[_unit(10, "Settlers", 1, 1)])
    table = _table(_unit_cand(99, "build_city"))  # stale actor, not in snapshot
    log = scripted.decide(snap, table)
    assert log.decisions == ()


# ---------------------------------------------------------------------------
# Diplomacy default (§7.4)
# ---------------------------------------------------------------------------


def test_diplomacy_stops_all_negotiations():
    snap = _snap(dipl=[DiplomacyView(counterpart_id=2, state="War",
                                     meeting_active=True),
                       DiplomacyView(counterpart_id=3, state="War",
                                     meeting_active=True)])
    table = _table(_dipl_cand(2, "stop_negotiation"),
                   _dipl_cand(3, "stop_negotiation"))
    log = scripted.decide(snap, table)
    dipl = [d for d in log.decisions if d.domain == "dipl"]
    assert [d.actor_id for d in dipl] == ["2", "3"]
    assert all(d.reason == "diplomacy_default:stop_negotiation" for d in dipl)


def test_diplomacy_never_accepts_or_cancels_treaty():
    snap = _snap(dipl=[DiplomacyView(counterpart_id=2, state="Peace",
                                     meeting_active=True)])
    table = _table(_dipl_cand(2, "stop_negotiation"),
                   _dipl_cand(2, "accept_treaty"),
                   _dipl_cand(2, "cancel_treaty"))
    log = scripted.decide(snap, table)
    verbs = _decisions_by_verb(log)
    assert verbs == ["stop_negotiation"]  # decline path only; no proactive offers


# ---------------------------------------------------------------------------
# Whitelisted verbs only (plan §3 line 37 — excluded verbs never emitted)
# ---------------------------------------------------------------------------


EXCLUDED_VERBS = [
    "autosettlers", "explore", "spy_steal_tech_3", "spy_bribe_unit_2",
    "trade_route_1", "marketplace_4", "conquer_city_2", "hut_enter_1",
    "disembark_0", "paradrop", "airlift", "upgrade", "board", "deboard",
    "unit_unload", "embassy_stay_1", "investigate_spend_2", "transform",
    "cultivate", "plant", "fortress", "airbase", "build_railroad", "pillage",
    "fallout", "disband", "no_orders", "action_selection", "wait", "nuke",
    "get_action_pro_0", "city_sell_improvement_granary",
    "city_change_specialist_0", "city_keep_production", "cancel_vision_2",
    "start_negotiation_2", "add_clause_1_2_0_0", "remove_clause_1_2_0_0",
    "trade_tech_clause_2", "chat", "set_sci_lux_tax_90_0_10",
    "change_gov_republic", "sell_unit_10",
]


def test_is_exposed_verb_whitelist_surface():
    exposed = [
        "goto_n", "goto_se", "attack_w", "build_city", "fortify", "sentry",
        "build_road", "irrigation", "mine", "join_city", "set_homecity",
        "cancel_order", "keep_activity", "produce_warriors",
        "produce_city_walls", "city_buy_production", "city_work_0_1_0",
        "city_unwork_1_m1_m1", "research_tech_alphabet_2",
        "set_tech_goal_alphabet_2", "stop_negotiation", "cancel_treaty",
        "accept_treaty",
    ]
    for verb in exposed:
        assert is_exposed_verb(verb), verb
    for verb in EXCLUDED_VERBS:
        assert not is_exposed_verb(verb), verb


def test_comparator_emits_only_whitelisted_verbs():
    # A table stuffed with excluded verbs (exploration/autosettler are absent
    # from the real candidate table by construction — here they are present
    # to prove the comparator's own whitelist holds as defense in depth).
    snap = _snap(cities=[_city(1, production=None), _city(3, "Tyre", 4, 4,
                                                            production=None)],
                 units=[_unit(10, "Settlers", 1, 1),
                        _unit(20, "Workers", 2, 2)])
    table = _table(
        # unit 10: one legal verb + a pile of excluded ones
        _unit_cand(10, "build_city"),
        *(_unit_cand(10, v) for v in EXCLUDED_VERBS[:20]),
        # unit 20: ONLY excluded verbs -> no decision at all
        *(_unit_cand(20, v) for v in EXCLUDED_VERBS[20:]),
        # city 1: legal produce
        _produce_cand(1, "Warriors"),
        # city 3: only excluded city verbs -> no decision
        *(_cand("city", "3", v, ctrl_type="city", actor_id=3, action_key=v)
          for v in ("city_sell_improvement_granary",
                    "city_change_specialist_0", "city_keep_production")),
        # tech: only excluded/surface-but-unused verbs -> no research decision
        _cand("tech", "cur_player", "set_tech_goal_alphabet_2",
              ctrl_type="tech", actor_id="cur_player",
              action_key="set_tech_goal_Alphabet_2", tech_id=2,
              tech_name="Alphabet"),
        # diplomacy: only excluded verbs -> no decision
        _cand("dipl", "2", "start_negotiation_2", ctrl_type="dipl",
              actor_id=2, action_key="start_negotiation_2", counterpart=2),
        _cand("dipl", "2", "add_clause_1_2_0_0", ctrl_type="dipl",
              actor_id=2, action_key="add_clause_1_2_0_0", counterpart=2),
    )
    log = scripted.decide(snap, table)
    table_ids = {c.id for c in table.candidates}
    assert log.decisions, "the legal verbs must still produce decisions"
    for d in log.decisions:
        assert is_exposed_verb(d.verb), f"excluded verb emitted: {d.verb}"
        assert d.candidate_id in table_ids
    # the excluded-only actors got nothing
    assert _decisions_for_actor(log, "unit", 20) == []
    assert _decisions_for_actor(log, "city", 3) == []
    assert [d for d in log.decisions if d.domain == "tech"] == []
    assert [d for d in log.decisions if d.domain == "dipl"] == []
    # and the legal ones fired
    assert any(d.verb == "build_city" for d in log.decisions)
    assert any(d.verb == "produce_warriors" for d in log.decisions)


# ---------------------------------------------------------------------------
# Degraded defaults (§7.4 line 163) — distinct from the comparator
# ---------------------------------------------------------------------------


def test_defaults_resolve_research_completed_with_scripted_default_reason():
    snap = _snap(target=None, target_id=None,
                 research_cands=[_rc(2, "Alphabet", 20)])
    table = _table(_tech_cand("Alphabet", 2))
    log = defaults.decide(snap, table)
    assert log.policy == "degraded_default"
    assert len(log.decisions) == 1
    d = log.decisions[0]
    assert d.reason == "scripted_default"
    assert d.trigger == "research_completed"
    assert d.provenance_class == PROVENANCE_CLASS
    assert d.verb.startswith("research_tech_")


def test_defaults_preserve_in_progress_research():
    # Research in progress on a non-priority target: preserved, no action.
    snap = _snap(target="Bronze Working", target_id=9,
                 research_cands=[_rc(9, "Bronze Working", 28),
                                 _rc(2, "Alphabet", 20)])
    table = _table(_tech_cand("Bronze Working", 9), _tech_cand("Alphabet", 2))
    log = defaults.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "tech"] == []


def test_defaults_resolve_empty_city_production():
    snap = _snap(cities=[_city(1, production=None)])
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"))
    log = defaults.decide(snap, table)
    city = [d for d in log.decisions if d.domain == "city"]
    assert len(city) == 1
    assert city[0].reason == "scripted_default"
    assert city[0].trigger == "production_empty"
    assert city[0].verb == "produce_warriors"  # undefended -> defensive unit


def test_defaults_preserve_current_city_production():
    snap = _snap(cities=[_city(1, production="Warriors")])
    table = _table(_produce_cand(1, "Warriors"), _produce_cand(1, "Settlers"))
    log = defaults.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "city"] == []


def test_defaults_preserve_unit_orders():
    # A settler with a legal found-city candidate and a worker with a legal
    # irrigation candidate: the degraded defaults leave both alone.
    snap = _snap(units=[_unit(10, "Settlers", 1, 1, terrain="Plains"),
                        _unit(11, "Workers", 2, 2, terrain="Plains")])
    table = _table(_unit_cand(10, "build_city"), _unit_cand(11, "irrigation"))
    log = defaults.decide(snap, table)
    assert [d for d in log.decisions if d.domain == "unit"] == []


def test_defaults_never_buy_and_never_touch_diplomacy():
    # No research target and no tech candidates (nothing to resolve), city
    # production set (preserved), rich gold with a buy candidate (defaults
    # never buy), active meeting (defaults never touch diplomacy).
    snap = _snap(gold=1000, cities=[_city(1, production="Warriors",
                                         buy_cost=20, production_progress=0.1)],
                 dipl=[DiplomacyView(counterpart_id=2, state="War",
                                     meeting_active=True)])
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1),
                   _dipl_cand(2, "stop_negotiation"))
    log = defaults.decide(snap, table)
    assert log.decisions == ()


def test_defaults_no_buy_no_diplomacy_with_research_active():
    snap = _snap(gold=1000, target="Alphabet", target_id=2,
                 cities=[_city(1, production="Warriors", buy_cost=20,
                               production_progress=0.1)],
                 dipl=[DiplomacyView(counterpart_id=2, state="War",
                                     meeting_active=True)])
    table = _table(_produce_cand(1, "Warriors"), _buy_cand(1),
                   _dipl_cand(2, "stop_negotiation"))
    log = defaults.decide(snap, table)
    assert log.decisions == ()  # preserve everything: no buy, no diplomacy


def test_comparator_and_defaults_are_distinct_entry_points_with_different_behavior():
    # One rich scenario where the comparator acts on every domain but the
    # degraded defaults preserve everything (nothing mandatory is pending).
    snap = _snap(
        gold=500, target="Bronze Working", target_id=9,
        research_cands=[_rc(9, "Bronze Working", 28), _rc(2, "Alphabet", 20)],
        cities=[_city(1, "Rome", 1, 1, production="Granary", buy_cost=20,
                      production_progress=0.1)],
        units=[_unit(10, "Settlers", 2, 2, terrain="Plains"),
               _unit(11, "Warriors", 1, 1),          # defender on the city tile
               _unit(12, "Workers", 3, 3, terrain="Plains")],
        dipl=[DiplomacyView(counterpart_id=2, state="War", meeting_active=True)],
    )
    table = _table(
        _tech_cand("Bronze Working", 9), _tech_cand("Alphabet", 2),
        _produce_cand(1, "Warriors"), _produce_cand(1, "Granary"),
        _buy_cand(1),
        _unit_cand(10, "build_city"), _unit_cand(10, "irrigation"),
        _unit_cand(11, "fortify"),
        _unit_cand(12, "irrigation"),
        _dipl_cand(2, "stop_negotiation"), _dipl_cand(2, "accept_treaty"),
    )
    comp = scripted.decide(snap, table)
    degr = defaults.decide(snap, table)

    # separate callables, separate policies
    assert scripted.decide is not defaults.decide
    assert comp.policy == "comparator"
    assert degr.policy == "degraded_default"
    assert comp.turn == degr.turn == snap.turn
    assert comp.finish_phase and degr.finish_phase

    # the comparator acts on every domain...
    comp_verbs = set(_decisions_by_verb(comp))
    assert any(v.startswith("research_tech_") for v in comp_verbs)  # switch to Alphabet
    assert "city_buy_production" in comp_verbs                      # buy rule
    assert "build_city" in comp_verbs                               # settle-first
    assert "fortify" in comp_verbs                                  # defender holds
    assert "irrigation" in comp_verbs                               # worker improves
    assert "stop_negotiation" in comp_verbs                         # decline
    assert "accept_treaty" not in comp_verbs                        # never accept
    assert len(comp.decisions) >= 5

    # ...while the degraded defaults preserve everything (research in
    # progress, production set): zero decisions.
    assert degr.decisions == ()

    # every comparator decision is a table candidate with the placeholder
    # provenance class for the WP2 ledger
    table_ids = {c.id for c in table.candidates}
    for d in comp.decisions:
        assert d.candidate_id in table_ids
        assert d.provenance_class == PROVENANCE_CLASS == "scripted_default"
        assert d.trigger == ""


def test_decision_log_shape_and_determinism():
    # City 1: undefended, production empty -> ladder switch (produce).
    # City 2: defended, already producing Warriors (the ladder pick) at 10%
    # with rich gold -> the buy rule fires (no switch, so no skip).
    snap = _snap(gold=100, target="Bronze Working", target_id=9,
                 research_cands=[_rc(2, "Alphabet", 20)],
                 cities=[_city(1, "Rome", 1, 1, production=None),
                         _city(2, "Utica", 5, 5, production="Warriors",
                               buy_cost=20, production_progress=0.1)],
                 units=[_unit(10, "Settlers", 2, 2, terrain="Plains"),
                        _unit(11, "Warriors", 5, 5)])
    table = _table(_tech_cand("Alphabet", 2),
                   _produce_cand(1, "Warriors"),
                   _produce_cand(2, "Warriors"), _buy_cand(2),
                   _unit_cand(10, "build_city"))
    log = scripted.decide(snap, table)
    assert log.by_domain == {"tech": 1, "city": 2, "unit": 1}
    assert log.candidate_ids() == [d.candidate_id for d in log.decisions]
    assert len(set(log.candidate_ids())) == len(log.candidate_ids())
    # deterministic: same inputs -> identical log
    assert scripted.decide(snap, table) == log
    assert defaults.decide(snap, table) == defaults.decide(snap, table)
