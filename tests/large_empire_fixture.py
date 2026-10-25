# SPDX-License-Identifier: GPL-3.0-or-later
"""Large-empire truncation fixture (WP2 acceptance, plan §3 line 32).

The §3 overflow policy ("if a budget is exceeded, deterministic
visibility-safe truncation applies — idle/distant units elided first, never
armed-in-contact ones; candidate lists capped by priority — and the snapshot
header discloses what was omitted; the truncation policy is validated on a
large-empire fixture, not only opening turns") is validated here on a
synthetic LARGE-EMPIRE observation: many own cities and units, several
contacts, full map breadth — driven through the real
``lechonk.adapter.obs_extract.extract_snapshot`` under the default 8,000-token
budget.

The world (deterministic, no server): an 80x50 map (the real game map is
78x52), a large visible region around the capital plus a distant visible
enemy pocket, a last-known ring, unexplored breadth; 6 own cities (the
capital plus 5), 3 foreign cities (1 visible, 2 last-known); 47 own units
(3 armed-in-contact warriors, 2 working units, 2 near-capital idle settlers,
40 far idle settlers spread over distance rings 15..24); 13 foreign units
(10 visible, 3 revealed-but-not-visible); 4 contacts; 20 research
candidates (over the 12-cap); 30 messages (over the 20-cap); every own city
with 12 buildable unit types (over the 8-cap).

``large_empire_world(turn=..., variant=...)`` returns the (observation, info)
pair; ``variant="prev"`` is the same empire one turn earlier (far units one
tile away, one city smaller, fewer bulbs, one fewer contact) for the delta
load.
"""

from __future__ import annotations

from typing import Any, Optional

from lechonk.adapter.obs_extract import RulesetNames
from tests.fog_fixtures import make_observation

MY_PLAYER = 0
ENEMY_A = 1
ENEMY_B = 2
ENEMY_C = 3
ENEMY_D = 4  # appears only in the current world (a new contact)

# --- fixture ruleset (12 unit types / 12 improvements / 10 terrains) --------
TERRAIN_IDS = {
    "Ocean": 0, "Plains": 1, "Grassland": 2, "Forest": 3, "Hills": 4,
    "Desert": 5, "Tundra": 6, "Swamp": 7, "Jungle": 8, "Mountain": 9,
}
UNIT_TYPE_IDS = {
    "Warriors": 0, "Settlers": 1, "Workers": 2, "Phalanx": 3, "Archers": 4,
    "Legion": 5, "Musketeers": 6, "Riflemen": 7, "Cavalry": 8, "Catapult": 9,
    "Knights": 10, "Cannon": 11,
}
IMPROVEMENT_IDS = {
    "Granary": 0, "City Walls": 1, "Barracks": 2, "Library": 3,
    "Marketplace": 4, "Bank": 5, "Cathedral": 6, "University": 7,
    "Factory": 8, "Stock Exchange": 9, "Harbor": 10, "Palace": 11,
}
NUM_IMPROVEMENTS = len(IMPROVEMENT_IDS)
VUT_NONE = 0
VUT_IMPROVEMENT = 3
VUT_UTYPE = 6

TECH_NAMES = [
    "Alphabet", "Pottery", "Agriculture", "Bronze Working", "Writing",
    "Currency", "Masonry", "Wheel", "Horseback Riding", "Iron Working",
    "Mathematics", "Construction", "Literature", "Code of Laws", "Feudalism",
    "Theology", "Steel Working", "Architecture", "Astronomy", "Gunpowder",
]

# World geometry.
CAPITAL = (10, 10)
MAP_W = 80
MAP_H = 50
VISIBLE_RADIUS = 30          # Chebyshev distance from the capital
LASTKNOWN_MIN = 31           # ring: VISIBLE_RADIUS+1 .. LASTKNOWN_MIN+2
LASTKNOWN_MAX = 33
ENEMY_POCKET = (70, 40)      # distant visible enemy pocket
ENEMY_POCKET_RADIUS = 6

N_OWN_CITIES = 6
N_FAR_IDLE = 40              # distance rings 15..24, 4 per ring
FAR_IDLE_RINGS = range(15, 25)


def large_ruleset() -> RulesetNames:
    terrains = {tid: name for name, tid in TERRAIN_IDS.items()}
    unit_types = {tid: name for name, tid in UNIT_TYPE_IDS.items()}
    improvements = {tid: name for name, tid in IMPROVEMENT_IDS.items()}
    return RulesetNames(
        terrains=terrains,
        unit_types=unit_types,
        improvements=improvements,
        nations={0: "Romans", 1: "Carthaginians", 2: "Greeks", 3: "Celts",
                 4: "Etruscans"},
        tech_costs={i + 1: 10 * (i + 1) for i in range(len(TECH_NAMES))},
        # city_state.py adds num_impr_types to production_value for VUT_UTYPE.
        city_production_names=(
            {(VUT_UTYPE, uid + NUM_IMPROVEMENTS): name
             for name, uid in UNIT_TYPE_IDS.items()}
            | {(VUT_IMPROVEMENT, iid): name
               for name, iid in IMPROVEMENT_IDS.items()}
        ),
    )


def _chebyshev(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def _tile_status(x: int, y: int) -> int:
    """0=unexplored, 1=last-known, 2=visible (the map trichotomy)."""
    d = _chebyshev(x, y, *CAPITAL)
    if d <= VISIBLE_RADIUS:
        return 2
    if LASTKNOWN_MIN <= d <= LASTKNOWN_MAX:
        return 1
    if _chebyshev(x, y, *ENEMY_POCKET) <= ENEMY_POCKET_RADIUS:
        return 2
    return 0


def _terrain_id(x: int, y: int) -> int:
    """Deterministic terrain for seen tiles (255 = no data elsewhere)."""
    if _tile_status(x, y) == 0:
        return 255
    pattern = [1, 2, 3, 4, 5, 7, 8]
    return pattern[(x * 7 + y * 3) % len(pattern)]


# ---------------------------------------------------------------------------
# entity builders (the same field shapes fog_fixtures produces)
# ---------------------------------------------------------------------------


def _my_player(**overrides: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "player_id": MY_PLAYER, "name": "Caesar", "score": 420, "team": 0,
        "is_alive": True, "nation": 0, "turns_alive": 30, "government": 1,
        "target_government": 0, "government_name": "Republic",
        "researching": 5, "research_name": "Currency", "tax": 10,
        "science": 90, "luxury": 0, "gold": 340, "culture": 210, "mood": 1,
        "revolution_finishes": 0, "science_cost": 0, "bulbs_researched": 40,
        "researching_cost": 60, "tech_goal": None, "tech_upkeep": 0,
        "techs_researched": 8, "total_bulbs_prod": 12, "embassy_txt": "-",
        "love": "-",
    }
    p.update(overrides)
    return p


def _enemy_player(pid: int, name: str, nation: int, **overrides: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "player_id": pid, "name": name, "score": 300 + pid, "team": 1,
        "is_alive": True, "nation": nation, "turns_alive": 30,
        "government": None, "target_government": None,
        "government_name": None, "researching": None, "research_name": None,
        "tax": None, "science": None, "luxury": None, "gold": None,
        "love": "-",
    }
    p.update(overrides)
    return p


def _own_city(city_id: int, name: str, x: int, y: int, size: int,
              production: Optional[str] = "Warriors", **overrides: Any) -> dict[str, Any]:
    ruleset = large_ruleset()
    if production is None:
        prod_kind, prod_value = VUT_NONE, -1
    else:
        for (kind, value), pname in ruleset.city_production_names.items():
            if pname == production:
                prod_kind, prod_value = kind, value
                break
        else:  # pragma: no cover - the fixture only uses known names
            raise ValueError(f"unknown production {production!r}")
    c: dict[str, Any] = {
        "id": city_id, "owner": MY_PLAYER, "size": size, "x": x, "y": y,
        "name": name,
        "food_stock": 20, "granary_size": 60, "granary_turns": 1,
        "production_kind": prod_kind, "production_value": prod_value,
        "city_radius_sq": 5, "buy_cost": 40, "shield_stock": 12,
        "disbanded_shields": 0, "caravan_shields": 0,
        "last_turns_shield_surplus": 2,
        "can_build_unit": [1] * len(UNIT_TYPE_IDS),  # 12 buildable (> 8 cap)
        "improvements": [0, 1],
        "turn_last_built": 28,
        "luxury": 0, "science": 12,
        "prod_food": 8, "surplus_food": 5, "prod_gold": 3, "surplus_gold": 2,
        "prod_shield": 6, "surplus_shield": 4, "prod_trade": 4,
        "surplus_trade": 3, "bulbs": 6, "city_waste": 0,
        "city_corruption": 0, "city_pollution": 0, "state": 2,
        "growth_in": "2 turns", "turns_to_prod_complete": 4,
        "prod_process": 0.35,
        "ppl_angry": 0, "ppl_unhappy": 0, "ppl_content": 4, "ppl_happy": 2,
    }
    c.update(overrides)
    return c


def _foreign_city(city_id: int, name: str, x: int, y: int, owner: int,
                  size: int = 5) -> dict[str, Any]:
    return {
        "id": city_id, "owner": owner, "size": size, "x": x, "y": y,
        "name": name,
        "food_stock": -1, "granary_size": -1, "granary_turns": -1,
        "production_kind": -1, "production_value": -1, "city_radius_sq": -1,
        "buy_cost": -1, "shield_stock": -1, "disbanded_shields": -1,
        "caravan_shields": -1, "last_turns_shield_surplus": -1,
        "can_build_unit": [0] * len(UNIT_TYPE_IDS), "improvements": [0] * 12,
        "turn_last_built": -1,
        "luxury": -1, "science": -1,
        "prod_food": -1, "surplus_food": -1, "prod_gold": -1,
        "surplus_gold": -1, "prod_shield": -1, "surplus_shield": -1,
        "prod_trade": -1, "surplus_trade": -1, "bulbs": -1, "city_waste": -1,
        "city_corruption": -1, "city_pollution": -1, "state": -1,
        "growth_in": None, "turns_to_prod_complete": -1, "prod_process": -1,
        "ppl_angry": -1, "ppl_unhappy": -1, "ppl_content": -1, "ppl_happy": -1,
    }


def _own_unit(unit_id: int, type_name: str, x: int, y: int, *,
              activity: int = 0, moves_left: int = 3) -> dict[str, Any]:
    uid = UNIT_TYPE_IDS[type_name]
    return {
        "id": unit_id, "owner": MY_PLAYER, "health": 100, "veteran": 1,
        "activity": activity, "ssa_controller": 0, "x": x, "y": y,
        "type_rule_name": type_name,
        "type_attack_strength": 3 if type_name in ("Warriors", "Phalanx",
                                                  "Legion", "Musketeers",
                                                  "Riflemen", "Cavalry",
                                                  "Knights", "Cannon") else 0,
        "type_defense_strength": 2, "type_firepower": 1,
        "type_build_cost": 30, "type_convert_time": 0,
        "type_converted_to": -1, "type_obsoleted_by": -1, "type_hp": 10,
        "type_move_rate": 3, "type_vision_radius_sq": 2, "type_worker": 1,
        "type_can_transport": False,
        "home_city": 1, "moves_left": moves_left,
        "upkeep_food": 1, "upkeep_shield": 0, "upkeep_gold": 0,
    }


def _foreign_unit(unit_id: int, type_name: str, x: int, y: int,
                  owner: int) -> dict[str, Any]:
    return {
        "id": unit_id, "owner": owner, "health": 80, "veteran": 0,
        "activity": 0, "ssa_controller": 0, "x": x, "y": y,
        "type_rule_name": type_name,
        "type_attack_strength": 3, "type_defense_strength": 2,
        "type_firepower": 1, "type_build_cost": 30, "type_convert_time": 0,
        "type_converted_to": -1, "type_obsoleted_by": -1, "type_hp": 10,
        "type_move_rate": 3, "type_vision_radius_sq": 2, "type_worker": 0,
        "type_can_transport": False,
        "home_city": -1, "moves_left": -1,
        "upkeep_food": -1, "upkeep_shield": -1, "upkeep_gold": -1,
    }


def _far_idle_positions() -> list[tuple[int, int]]:
    """40 positions on distance rings 15..24 from the capital (4 per ring),
    all on visible tiles, ordered farthest ring first."""
    positions: list[tuple[int, int]] = []
    for d in FAR_IDLE_RINGS:
        positions.extend([(10 + d, 10), (10, 10 + d), (10 + d, 13),
                          (13, 10 + d)])
    return positions[:N_FAR_IDLE]


# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------


def large_empire_world(*, turn: int = 30, variant: str = "cur"
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The large-empire (observation, info) pair.

    ``variant="cur"`` (default) is turn ``turn``; ``variant="prev"`` is the
    same empire one turn earlier: the far idle units one tile east, one city
    one size smaller, fewer bulbs researched, one fewer contact — a heavy
    delta load for the ≤800-token delta budget check.
    """
    prev = variant == "prev"
    status = {(x, y): _tile_status(x, y)
              for x in range(MAP_W) for y in range(MAP_H)}
    terrain = {(x, y): _terrain_id(x, y) for x in range(MAP_W)
               for y in range(MAP_H)}
    # Owned tiles: a small area around each own city (seen tiles only).
    tile_owner: dict[tuple[int, int], int] = {}
    own_city_xy = [(10, 10), (16, 12), (12, 18), (22, 10), (18, 24), (28, 18)]
    for cx, cy in own_city_xy:
        for x in range(cx - 3, cx + 4):
            for y in range(cy - 3, cy + 4):
                if 0 <= x < MAP_W and 0 <= y < MAP_H \
                        and _tile_status(x, y) == 2:
                    tile_owner[(x, y)] = MY_PLAYER
    # The enemy pocket is owned by the enemy.
    for x in range(ENEMY_POCKET[0] - ENEMY_POCKET_RADIUS,
                   ENEMY_POCKET[0] + ENEMY_POCKET_RADIUS + 1):
        for y in range(ENEMY_POCKET[1] - ENEMY_POCKET_RADIUS,
                       ENEMY_POCKET[1] + ENEMY_POCKET_RADIUS + 1):
            if 0 <= x < MAP_W and 0 <= y < MAP_H \
                    and _tile_status(x, y) == 2:
                tile_owner[(x, y)] = ENEMY_A

    players = [
        _my_player(bulbs_researched=35 if prev else 40,
                   score=400 if prev else 420),
        _enemy_player(ENEMY_A, "Hannibal", 1),
        _enemy_player(ENEMY_B, "Dionysios", 2),
        _enemy_player(ENEMY_C, "Brennus", 3),
    ]
    if not prev:
        players.append(_enemy_player(ENEMY_D, "Lars", 4))

    cities = [
        _own_city(1, "Roma", 10, 10, 7 if prev else 8),
        _own_city(2, "Antium", 16, 12, 5, production="Granary"),
        _own_city(3, "Neapolis", 12, 18, 4, production="Library"),
        _own_city(4, "Capua", 22, 10, 3, production="Settlers"),
        _own_city(5, "Brundisium", 18, 24, 6, production="Barracks"),
        _own_city(6, "Syracusae", 28, 18, 3, production="Archers"),
        _foreign_city(20, "Karthago", 70, 40, ENEMY_A, size=9),
        _foreign_city(21, "Himera", 10, 42, ENEMY_B, size=4),
        _foreign_city(22, "Lilybaeum", 43, 12, ENEMY_C, size=6),
    ]

    units: list[dict[str, Any]] = []
    # Armed own warriors in contact with visible enemies (never elided).
    units.append(_own_unit(100, "Legion", 20, 14))
    units.append(_own_unit(101, "Phalanx", 22, 16))
    units.append(_own_unit(102, "Warriors", 18, 13))
    # Working own units (activity != idle: elided only after every idle unit).
    units.append(_own_unit(103, "Workers", 12, 11, activity=2))   # mine
    units.append(_own_unit(104, "Workers", 11, 12, activity=3))   # irrigate
    # Near-capital idle settlers.
    units.append(_own_unit(105, "Settlers", 11, 10))
    units.append(_own_unit(106, "Settlers", 10, 11))
    # Far idle settlers on distance rings 15..24 (the elision candidates).
    for i, (x, y) in enumerate(_far_idle_positions()):
        units.append(_own_unit(200 + i, "Settlers", x + (1 if prev else 0), y))
    # Visible foreign units (never elided): 3 in contact + 5 scattered +
    # 2 in the distant enemy pocket.
    units.append(_foreign_unit(300, "Warriors", 20, 15, ENEMY_A))
    units.append(_foreign_unit(301, "Archers", 21, 16, ENEMY_B))
    units.append(_foreign_unit(302, "Warriors", 19, 14, ENEMY_C))
    units.append(_foreign_unit(303, "Cavalry", 30, 20, ENEMY_A))
    units.append(_foreign_unit(304, "Legion", 33, 22, ENEMY_B))
    units.append(_foreign_unit(305, "Settlers", 25, 28, ENEMY_C))
    units.append(_foreign_unit(306, "Workers", 16, 20, ENEMY_A))
    units.append(_foreign_unit(307, "Phalanx", 28, 12, ENEMY_B))
    units.append(_foreign_unit(308, "Warriors", 69, 39, ENEMY_A))
    units.append(_foreign_unit(309, "Archers", 71, 41, ENEMY_A))
    # Revealed-but-not-visible foreign units on last-known tiles.
    units.append(_foreign_unit(310, "Warriors", 12, 42, ENEMY_B))
    units.append(_foreign_unit(311, "Legion", 43, 12, ENEMY_C))
    units.append(_foreign_unit(312, "Cavalry", 8, 41, ENEMY_A))

    techs = {i + 1: {"name": name, "inv_state": 1, "is_researching": i == 4,
                     "is_tech_goal": False, "sup_units": [],
                     "sup_improvements": [], "is_req_for_goal": False,
                     "reqs": {}}
             for i, name in enumerate(TECH_NAMES)}
    # 20 research candidates (over the 12-cap), sorted by (cost, name).

    dipl = {
        ENEMY_A: {"diplomatic_state": 1, "diplomacy_clause_map": [],
                  "meeting_initializers": -1},
        ENEMY_B: {"diplomatic_state": 3, "diplomacy_clause_map": [],
                  "meeting_initializers": -1},
        ENEMY_C: {"diplomatic_state": 5,
                  "diplomacy_clause_map": ["clause: share maps"],
                  "meeting_initializers": ENEMY_C},
    }
    if not prev:
        dipl[ENEMY_D] = {"diplomatic_state": 5, "diplomacy_clause_map": [],
                         "meeting_initializers": -1}

    obs = make_observation(
        width=MAP_W, height=MAP_H,
        status=status, terrain=terrain, tile_owner=tile_owner,
        players=players, cities=cities, units=units, techs=techs, dipl=dipl,
        game={"messages": "chat: the empire grows", "year": "1250 BC",
              "turn": turn},
    )
    messages = [{"text": f"server event {i}: legion musters in the provinces"}
                for i in range(30)]  # over the 20-cap
    info = {"turn": turn, "mini_game_messages": messages,
            "available_actions": {}}
    return obs, info


def elision_order_key(unit, capital_xy):
    """The extractor's elision key (obs_extract._apply_token_budget):
    (working, -dist, id) — idle first, farthest from the capital first."""
    working = 1 if (unit.activity not in (None, 0)
                    or unit.ssa_controller not in (None, 0)) else 0
    dist = max(abs(unit.x - capital_xy[0]), abs(unit.y - capital_xy[1]))
    return (working, -dist, unit.id)
