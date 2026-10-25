# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic CivRealm observation dicts — offline fog fixtures (WP1 package A).

These builders produce the same 11-domain observation structure the vendored
CivRealm TurnManager returns (turn_manager.py:128-141; controller_list at
civ_controller.py:185-195): map, player, unit, city, tech, dipl, game (+ the
rules/options/client placeholders). No live server is needed.

Map array conventions (map_state.py): status 0=unknown, 1=known-fogged,
2=seen; terrain 255 = no terrain data; tile_owner/unit_owner/city_owner 255 =
no owner. The unit/unit_owner/city_owner arrays are a client-side cache that
map_state._update_state fills for ALL known units/cities regardless of tile
visibility — the fixtures use them to prove the extractor never leaks stale
fogged-tile data.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from lechonk.adapter.obs_extract import RulesetNames

MY_PLAYER = 0
ENEMY = 1

# terrain ids
OCEAN = 0
PLAINS = 1
GRASSLAND = 2
FOREST = 3
HILLS = 4

# unit type ids
WARRIORS = 0
SETTLERS = 1
PHALANX = 2

NUM_UNIT_TYPES = 3
NUM_IMPROVEMENTS = 1  # Granary only

# production kinds (version/v14.py)
VUT_NONE = 0
VUT_IMPROVEMENT = 3
VUT_UTYPE = 6


def fixture_ruleset() -> RulesetNames:
    return RulesetNames(
        terrains={OCEAN: "Ocean", PLAINS: "Plains", GRASSLAND: "Grassland",
                  FOREST: "Forest", HILLS: "Hills"},
        unit_types={WARRIORS: "Warriors", SETTLERS: "Settlers", PHALANX: "Phalanx"},
        improvements={0: "Granary"},
        nations={0: "Romans", 1: "Barbarians"},
        tech_costs={1: 20, 2: 30, 3: 40, 5: 50},
        # city_state.py adds num_impr_types to production_value for VUT_UTYPE.
        city_production_names={
            (VUT_UTYPE, SETTLERS + NUM_IMPROVEMENTS): "Settlers",
            (VUT_UTYPE, WARRIORS + NUM_IMPROVEMENTS): "Warriors",
            (VUT_IMPROVEMENT, 0): "Granary",
        },
    )


def my_player(**overrides: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "player_id": MY_PLAYER, "name": "Caesar", "score": 10, "team": 0,
        "is_alive": True, "nation": 0, "turns_alive": 1, "government": 1,
        "target_government": 0, "government_name": "Republic",
        "researching": None, "research_name": None, "tax": 10, "science": 90,
        "luxury": 0, "gold": 50, "culture": 0, "mood": 0,
        "revolution_finishes": 0, "science_cost": 0, "bulbs_researched": 0,
        "researching_cost": 0, "tech_goal": None, "tech_upkeep": 0,
        "techs_researched": 0, "total_bulbs_prod": 4, "embassy_txt": "-",
        "love": "-",
    }
    p.update(overrides)
    return p


def enemy_player(**overrides: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "player_id": ENEMY, "name": "Hannibal", "score": 8, "team": 1,
        "is_alive": True, "nation": 1, "turns_alive": 1, "government": None,
        "target_government": None, "government_name": None,
        "researching": None, "research_name": None, "tax": None,
        "science": None, "luxury": None, "gold": None, "love": "-",
    }
    p.update(overrides)
    return p


def own_city(city_id: int = 1, name: str = "Rome", x: int = 1, y: int = 1,
             **overrides: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "id": city_id, "owner": MY_PLAYER, "size": 1, "x": x, "y": y,
        "name": name,
        "food_stock": 2, "granary_size": 20, "granary_turns": 3,
        "production_kind": VUT_NONE, "production_value": -1,
        "city_radius_sq": 5, "buy_cost": 20, "shield_stock": 0,
        "disbanded_shields": 0, "caravan_shields": 0,
        "last_turns_shield_surplus": 1,
        "can_build_unit": [1, 1, 0],  # Warriors + Settlers buildable
        "improvements": [0],
        "turn_last_built": 1,
        "luxury": 0, "science": 4,
        "prod_food": 3, "surplus_food": 2, "prod_gold": 1, "surplus_gold": 1,
        "prod_shield": 2, "surplus_shield": 1, "prod_trade": 1,
        "surplus_trade": 1, "bulbs": 2, "city_waste": 0,
        "city_corruption": 0, "city_pollution": 0, "state": 2,
        "growth_in": "3 turns", "turns_to_prod_complete": 32767,
        "prod_process": 0,
        "ppl_angry": 0, "ppl_unhappy": 0, "ppl_content": 1, "ppl_happy": 0,
    }
    c.update(overrides)
    return c


def foreign_city(city_id: int = 2, name: str = "Carthage", x: int = 4,
                 y: int = 4, **overrides: Any) -> dict[str, Any]:
    """Foreign city: upstream sends -1/zero placeholders for own-city fields."""
    c: dict[str, Any] = {
        "id": city_id, "owner": ENEMY, "size": 5, "x": x, "y": y, "name": name,
        "food_stock": -1, "granary_size": -1, "granary_turns": -1,
        "production_kind": -1, "production_value": -1, "city_radius_sq": -1,
        "buy_cost": -1, "shield_stock": -1, "disbanded_shields": -1,
        "caravan_shields": -1, "last_turns_shield_surplus": -1,
        "can_build_unit": [0, 0, 0], "improvements": [0, 0],
        "turn_last_built": -1,
        "luxury": -1, "science": -1,
        "prod_food": -1, "surplus_food": -1, "prod_gold": -1,
        "surplus_gold": -1, "prod_shield": -1, "surplus_shield": -1,
        "prod_trade": -1, "surplus_trade": -1, "bulbs": -1, "city_waste": -1,
        "city_corruption": -1, "city_pollution": -1, "state": -1,
        "growth_in": None, "turns_to_prod_complete": -1, "prod_process": -1,
        "ppl_angry": -1, "ppl_unhappy": -1, "ppl_content": -1,
        "ppl_happy": -1,
    }
    c.update(overrides)
    return c


def own_unit(unit_id: int = 10, type_id: int = SETTLERS, x: int = 1, y: int = 1,
             **overrides: Any) -> dict[str, Any]:
    u: dict[str, Any] = {
        "id": unit_id,
        "owner": MY_PLAYER, "health": 100, "veteran": 0, "activity": 0,
        "ssa_controller": 0, "x": x, "y": y,
        "type_rule_name": fixture_ruleset().unit_types[type_id],
        "type_attack_strength": 3 if type_id == WARRIORS else 0,
        "type_defense_strength": 1, "type_firepower": 1, "type_build_cost": 30,
        "type_convert_time": 0, "type_converted_to": -1,
        "type_obsoleted_by": -1, "type_hp": 10, "type_move_rate": 3,
        "type_vision_radius_sq": 2, "type_worker": 1,
        "type_can_transport": False,
        "home_city": 1, "moves_left": 3,
        "upkeep_food": 1, "upkeep_shield": 0, "upkeep_gold": 0,
    }
    u.update(overrides)
    return u


def foreign_unit(unit_id: int = 20, type_id: int = WARRIORS, x: int = 2,
                 y: int = 2, **overrides: Any) -> dict[str, Any]:
    """Foreign unit: upstream sends -1 placeholders for own-unit fields."""
    u: dict[str, Any] = {
        "id": unit_id,
        "owner": ENEMY, "health": 80, "veteran": 0, "activity": 0,
        "ssa_controller": 0, "x": x, "y": y,
        "type_rule_name": fixture_ruleset().unit_types[type_id],
        "type_attack_strength": 3, "type_defense_strength": 1,
        "type_firepower": 1, "type_build_cost": 30, "type_convert_time": 0,
        "type_converted_to": -1, "type_obsoleted_by": -1, "type_hp": 10,
        "type_move_rate": 3, "type_vision_radius_sq": 2, "type_worker": 0,
        "type_can_transport": False,
        "home_city": -1, "moves_left": -1,
        "upkeep_food": -1, "upkeep_shield": -1, "upkeep_gold": -1,
    }
    u.update(overrides)
    return u


def tech_entry(tech_id: int, name: str, inv_state: int = 1,
               reqs: Optional[Mapping[int, bool]] = None,
               is_researching: bool = False) -> dict[str, Any]:
    return {
        "name": name, "inv_state": inv_state, "is_researching": is_researching,
        "is_tech_goal": False, "sup_units": [], "sup_improvements": [],
        "is_req_for_goal": False, "reqs": dict(reqs or {}),
    }


def default_techs() -> dict[int, dict[str, Any]]:
    return {
        1: tech_entry(1, "Alphabet", inv_state=1, reqs={}),
        2: tech_entry(2, "Pottery", inv_state=1, reqs={1: True}),
        3: tech_entry(3, "Agriculture", inv_state=1, reqs={2: False}),
        5: tech_entry(5, "Bronze Working", inv_state=0, reqs={1: False}),
        7: tech_entry(7, "Writing", inv_state=2, reqs={1: True, 2: True}),
    }


def default_dipl() -> dict[int, dict[str, Any]]:
    return {ENEMY: {"diplomatic_state": 5, "diplomacy_clause_map": [],
                    "meeting_initializers": -1}}


def make_observation(
    *,
    width: int = 6,
    height: int = 6,
    status: Optional[Mapping[tuple[int, int], int]] = None,
    terrain: Optional[Mapping[tuple[int, int], int]] = None,
    tile_owner: Optional[Mapping[tuple[int, int], int]] = None,
    stale_unit_owner: Optional[Mapping[tuple[int, int], int]] = None,
    stale_unit_counts: Optional[Mapping[tuple[int, int], int]] = None,
    stale_city_owner: Optional[Mapping[tuple[int, int], int]] = None,
    players: Optional[list[dict[str, Any]]] = None,
    cities: Optional[list[dict[str, Any]]] = None,
    units: Optional[list[dict[str, Any]]] = None,
    techs: Optional[Mapping[int, Mapping[str, Any]]] = None,
    dipl: Optional[Mapping[int, Mapping[str, Any]]] = None,
    game: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a full 11-domain observation dict.

    ``status``/``terrain``/``tile_owner`` default to unexplored/255/255.
    ``stale_*`` arrays model the client-side map cache that map_state fills
    for ALL known entities regardless of tile visibility (fog-hidden state).

    The map arrays are X-MAJOR, exactly like the real map_state:
    map_allocate(x_size, y_size) -> np.full((x_size, y_size)) and
    status[x, y] = ... (map_state.py:52-63, 120-132). So grid[x][y] carries
    tile (x, y): the first dimension is the x extent (``width``), the second
    the y extent (``height``). Non-square maps pin this orientation (see
    test_map_arrays_are_x_major_like_map_state).
    """
    status = status or {}
    terrain = terrain or {}
    tile_owner = tile_owner or {}
    stale_unit_owner = stale_unit_owner or {}
    stale_unit_counts = stale_unit_counts or {}
    stale_city_owner = stale_city_owner or {}

    status_grid = [[int(status.get((x, y), 0)) for y in range(height)]
                   for x in range(width)]
    terrain_grid = [[int(terrain.get((x, y), 255)) for y in range(height)]
                    for x in range(width)]
    owner_grid = [[int(tile_owner.get((x, y), 255)) for y in range(height)]
                  for x in range(width)]
    unit_owner_grid = [[int(stale_unit_owner.get((x, y), 255))
                        for y in range(height)] for x in range(width)]
    city_owner_grid = [[int(stale_city_owner.get((x, y), 255))
                        for y in range(height)] for x in range(width)]
    unit_count_grid = [[[int(stale_unit_counts.get((x, y), 0))]
                        * NUM_UNIT_TYPES for y in range(height)]
                       for x in range(width)]

    players = players if players is not None else [my_player(), enemy_player()]
    cities = cities if cities is not None else []
    units = units if units is not None else []
    techs = techs if techs is not None else default_techs()
    dipl = dipl if dipl is not None else default_dipl()
    game = game if game is not None else {"messages": "", "year": "4000 BC",
                                           "turn": 1}

    return {
        "map": {
            "status": status_grid,
            "terrain": terrain_grid,
            "extras": [[[0]] * height for _ in range(width)],
            "output": [[[0] * 6] * height for _ in range(width)],
            "tile_owner": owner_grid,
            "city_owner": city_owner_grid,
            "unit": unit_count_grid,
            "unit_owner": unit_owner_grid,
        },
        "player": {p["player_id"]: p for p in players},
        "unit": {int(u.get("id", i + 10)): u for i, u in enumerate(units)},
        "city": {c["id"]: c for c in cities},
        "tech": dict(techs),
        "dipl": dict(dipl),
        "game": dict(game),
        "gov": {"name": "Republic", "id": 1, "helptext": ""},
        "rules": {"turn": 1, "year": "4000 BC"},
        "options": {},
        "client": {},
    }


def base_world(**kwargs: Any) -> dict[str, Any]:
    """The canonical small world: capital Rome at (1,1), own Settlers (10) at
    (1,1) and Warriors (11) at (2,1), visible enemy Warriors (20) at (2,2),
    enemy city Carthage (2) at (4,4). Tiles: (0..3,0..2) visible around the
    capital; (3,3) last-known; (4,4) visible; the rest unexplored."""
    status: dict[tuple[int, int], int] = {}
    for x in range(4):
        for y in range(3):
            status[(x, y)] = 2
    status[(4, 4)] = 2
    status[(3, 3)] = 1
    terrain = {xy: PLAINS for xy in status}
    terrain[(4, 4)] = GRASSLAND
    terrain[(3, 3)] = FOREST
    kwargs.setdefault("status", status)
    kwargs.setdefault("terrain", terrain)
    kwargs.setdefault("players", [my_player(), enemy_player()])
    kwargs.setdefault("cities", [own_city(), foreign_city()])
    kwargs.setdefault("units", [own_unit(10, SETTLERS, 1, 1),
                                own_unit(11, WARRIORS, 2, 1),
                                foreign_unit(20, WARRIORS, 2, 2)])
    return make_observation(**kwargs)


def base_info(turn: int = 1,
              messages: Optional[list[Any]] = None) -> dict[str, Any]:
    return {"turn": turn, "mini_game_messages": messages or [],
            "available_actions": {}}
