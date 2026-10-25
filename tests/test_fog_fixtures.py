# SPDX-License-Identifier: GPL-3.0-or-later
"""Fog fixtures (design-plan-freeciv-v1.1.md §3 line 36, §7.2; WP1 package A).

(a) status trichotomy: unexplored / last-known / visible distinguishable.
(b) vanished enemy: a previously-visible foreign unit disappearing yields
    "not currently visible" — never "destroyed".
(c) world pair: two worlds differing ONLY in hidden mutable state produce
    equivalent policy-facing snapshots.
(d) cold-start reload: revealed-but-not-visible mutable fields render as
    unknown, not stale.
"""

from __future__ import annotations

import pytest

from lechonk.adapter.obs_extract import (
    ExtractOutcome,
    extract_snapshot,
)
from lechonk.core.schemas import SNAPSHOT_TOKEN_BUDGET

from tests.fog_fixtures import (
    ENEMY,
    FOREST,
    GRASSLAND,
    MY_PLAYER,
    PLAINS,
    WARRIORS,
    base_info,
    base_world,
    enemy_player,
    fixture_ruleset,
    foreign_city,
    foreign_unit,
    make_observation,
    my_player,
    own_city,
    own_unit,
)


def _extract(obs, info=None, **kwargs):
    result = extract_snapshot(
        obs, info or base_info(), my_player_id=MY_PLAYER,
        ruleset=fixture_ruleset(), **kwargs)
    assert result.outcome is ExtractOutcome.OK
    assert result.snapshot is not None
    return result.snapshot


# ---------------------------------------------------------------------------
# (a) status trichotomy
# ---------------------------------------------------------------------------


def test_status_trichotomy_distinguishable():
    """0=unexplored / 1=last-known / 2=visible must be distinguishable in the
    Snapshot (plan §7.2: the trichotomy maps exactly)."""
    # 3x3 map: (0,0) unexplored, (1,1) last-known with known terrain,
    # (2,2) visible. No cities -> no capital -> no rel_to_capital.
    obs = make_observation(
        width=3, height=3,
        status={(0, 0): 0, (1, 1): 1, (2, 2): 2},
        terrain={(1, 1): PLAINS, (2, 2): GRASSLAND},
        players=[my_player()],
        cities=[], units=[],
    )
    snap = _extract(obs)

    assert snap.map_view.tile_counts == {"visible": 1, "last_known": 1,
                                         "unexplored": 7}
    grid = snap.map_view.grid
    assert grid is not None, "small maps render the ASCII grid"
    rows = grid.split("\n")
    assert len(rows) == 3
    # unexplored renders as '.'
    assert rows[0][0] == "."
    # last-known with known terrain renders as the lowercase terrain letter
    assert rows[1][1] == "p"  # plains, last known
    # visible renders as the UPPERCASE terrain letter
    assert rows[2][2] == "G"  # grassland, visible
    # the three categories are pairwise distinct at the tile level
    assert len({rows[0][0], rows[1][1], rows[2][2]}) == 3
    # region notes carry the counts too
    assert any("visible=1" in n and "last-known=1" in n and "unexplored=7" in n
               for n in snap.map_view.region_notes)


def test_status_trichotomy_last_known_without_terrain():
    """A last-known tile with no terrain data (never seen, only explored)
    renders as '~', distinct from both unexplored and visible."""
    obs = make_observation(
        width=2, height=2,
        status={(0, 0): 1, (1, 1): 2},
        terrain={(1, 1): PLAINS},  # (0,0) terrain stays 255 (unknown)
        players=[my_player()],
        cities=[], units=[],
    )
    snap = _extract(obs)
    rows = snap.map_view.grid.split("\n")
    assert rows[0][0] == "~"  # last known, terrain unknown
    assert rows[1][1] == "P"  # visible
    assert rows[0][1] == "." and rows[1][0] == "."  # unexplored
    assert snap.map_view.tile_counts["last_known"] == 1  # only (0,0)


# ---------------------------------------------------------------------------
# (b) vanished enemy
# ---------------------------------------------------------------------------


def test_vanished_enemy_is_not_destroyed():
    """A previously-visible foreign unit that disappears from the observation
    (its tile goes fogged) yields 'not currently visible' — never 'destroyed'.

    The unit is absent from the client's unit domain (the live-game case: the
    server removes fogged units). The snapshot must not claim destruction, must
    not list the unit, and must render the tile as last-known with no visible
    units."""
    before = base_world()
    snap_before = _extract(before)
    enemy = next(u for u in snap_before.units if u.id == 20)
    assert enemy.visibility == "visible"
    assert enemy.hp == 80

    # The unit vanishes: tile (2,2) goes fogged (last-known), the unit leaves
    # the client's unit domain, and the stale map cache still claims a unit
    # owner there (hidden state that must not leak).
    after = base_world(
        status={(x, y): 2 for x in range(4) for y in range(3)}
        | {(4, 4): 2, (3, 3): 1, (2, 2): 1},
        units=[own_unit(10, 1, 1, 1), own_unit(11, WARRIORS, 2, 1)],
        stale_unit_owner={(2, 2): ENEMY},
        stale_unit_counts={(2, 2): 1},
    )
    snap_after = _extract(after)

    # the unit is gone from the policy-facing surface
    assert all(u.id != 20 for u in snap_after.units)
    # no destruction claim anywhere in the rendered snapshot
    rendered = snap_after.render().lower()
    assert "destroy" not in rendered
    assert "unit 20" not in rendered
    # the tile is last-known (not visible, not unexplored):
    # 13 visible in base_world, (2,2) goes fogged -> 12 visible, 2 last-known
    assert snap_after.map_view.tile_counts["last_known"] == 2  # (3,3) + (2,2)
    assert snap_after.map_view.tile_counts["visible"] == 12
    # the enemy-units landmark for (2,2) is gone
    assert all(not (l.kind == "enemy_units" and l.x == 2 and l.y == 2)
               for l in snap_after.map_view.landmarks)
    # the grid shows the tile as last-known, not visible
    row = snap_after.map_view.grid.split("\n")[2]
    assert row[2] == "p"  # plains, last known (lowercase)


def test_revealed_but_not_visible_unit_is_not_destroyed():
    """Cold-start variant: the unit is still in the client's unit domain but on
    a fogged tile (stale entry after a checkpoint load). It renders as
    'not currently visible' with unknown mutable fields — never destroyed,
    never stale hp."""
    obs = base_world(
        status={xy: 2 for xy in
                [(x, y) for x in range(4) for y in range(3)]}
        | {(4, 4): 2, (3, 3): 1, (2, 2): 1},
        units=[own_unit(10, 1, 1, 1), own_unit(11, WARRIORS, 2, 1),
               foreign_unit(20, WARRIORS, 2, 2, health=55)],
    )
    snap = _extract(obs)
    enemy = next(u for u in snap.units if u.id == 20)
    assert enemy.visibility == "not_visible"
    assert enemy.hp is None  # unknown, not the stale 55
    assert "destroy" not in snap.render().lower()


# ---------------------------------------------------------------------------
# (c) world pair — hidden mutable state must not leak
# ---------------------------------------------------------------------------


def _world_pair(hidden: str):
    """Two worlds identical in every visible/policy-visible respect. They
    differ ONLY in hidden mutable state:

    - the fogged foreign city's stale size (5 vs 7) and stale owner arrays;
    - the fogged-tile foreign unit's stale health (30 vs 80);
    - the stale map-cache arrays on fogged tiles (tile_owner / unit /
      unit_owner / city_owner): present vs absent.
    """
    common = dict(
        status={xy: 2 for xy in
                [(x, y) for x in range(4) for y in range(3)]}
        | {(4, 4): 1, (3, 3): 1, (2, 2): 2},
        terrain={xy: PLAINS for xy in
                 [(x, y) for x in range(4) for y in range(3)]}
        | {(4, 4): GRASSLAND, (3, 3): FOREST, (2, 2): PLAINS},
        players=[my_player(), enemy_player()],
        cities=[own_city(),
                foreign_city(2, "Carthage", 4, 4,
                             size=5 if hidden == "a" else 7)],
        units=[own_unit(10, 1, 1, 1), own_unit(11, WARRIORS, 2, 1),
               foreign_unit(20, WARRIORS, 2, 2),
               # revealed-but-not-visible on the fogged tile (3,3)
               foreign_unit(21, WARRIORS, 3, 3,
                            health=30 if hidden == "a" else 80)],
    )
    if hidden == "a":
        common.update(
            tile_owner={(3, 3): ENEMY, (4, 4): ENEMY, (1, 1): MY_PLAYER},
            stale_unit_owner={(3, 3): ENEMY, (2, 2): ENEMY},
            stale_unit_counts={(3, 3): 1, (2, 2): 1},
            stale_city_owner={(4, 4): ENEMY, (1, 1): MY_PLAYER},
        )
    else:
        common.update(
            tile_owner={(1, 1): MY_PLAYER},
            stale_unit_owner={(2, 2): ENEMY},
            stale_unit_counts={(2, 2): 1},
            stale_city_owner={(1, 1): MY_PLAYER},
        )
    return make_observation(**common)


def test_world_pair_equivalent_snapshots():
    """Two synthetic worlds differing ONLY in hidden mutable state must
    produce equivalent policy-facing snapshots (plan §3 line 36 fixture
    class)."""
    snap_a = _extract(_world_pair("a"))
    snap_b = _extract(_world_pair("b"))
    assert snap_a.render() == snap_b.render()


def test_world_pair_hidden_fields_render_unknown():
    """The hidden mutable values themselves must not appear: the fogged city
    renders size/owner unknown; the fogged unit renders hp unknown."""
    for hidden in ("a", "b"):
        snap = _extract(_world_pair(hidden))
        carthage = next(c for c in snap.cities if c.id == 2)
        assert carthage.visibility == "last_known"
        assert carthage.size is None
        assert carthage.owner_id is None
        fogged_unit = next(u for u in snap.units if u.id == 21)
        assert fogged_unit.visibility == "not_visible"
        assert fogged_unit.hp is None
        # stale owner data must not leak into region notes either
        notes = " ".join(snap.map_view.region_notes)
        assert "owned by Hannibal" not in notes


# ---------------------------------------------------------------------------
# (d) cold-start reload policy
# ---------------------------------------------------------------------------


def test_cold_start_reload_renders_unknown_not_stale():
    """A freshly loaded world renders revealed-but-not-visible mutable fields
    as unknown, not stale: the fogged foreign city's size (7 in the client
    cache) and the fogged unit's hp (80) must NOT appear."""
    obs = make_observation(
        status={(1, 1): 2, (2, 1): 2, (2, 2): 2, (5, 5): 2,
                (4, 4): 1, (3, 3): 1},
        terrain={(1, 1): PLAINS, (2, 1): PLAINS, (2, 2): PLAINS,
                 (5, 5): GRASSLAND, (4, 4): GRASSLAND, (3, 3): FOREST},
        players=[my_player(), enemy_player()],
        cities=[
            own_city(1, "Rome", 1, 1),
            # revealed-but-not-visible: stale size 7, stale owner 1
            foreign_city(2, "Carthage", 4, 4, size=7, owner=ENEMY),
            # currently visible: size 3, owner 1
            foreign_city(3, "Tyre", 5, 5, size=3, owner=ENEMY),
        ],
        units=[
            own_unit(10, 1, 1, 1),
            own_unit(11, WARRIORS, 2, 1),
            # revealed-but-not-visible: stale hp 80
            foreign_unit(20, WARRIORS, 3, 3, health=80),
            # currently visible: hp 60
            foreign_unit(21, WARRIORS, 2, 2, health=60),
        ],
    )
    snap = _extract(obs)

    carthage = next(c for c in snap.cities if c.id == 2)
    assert carthage.visibility == "last_known"
    assert carthage.size is None, "stale size must render as unknown"
    assert carthage.owner_id is None, "stale owner must render as unknown"

    tyre = next(c for c in snap.cities if c.id == 3)
    assert tyre.visibility == "visible"
    assert tyre.size == 3
    assert tyre.owner_id == ENEMY

    fogged = next(u for u in snap.units if u.id == 20)
    assert fogged.visibility == "not_visible"
    assert fogged.hp is None, "stale hp must render as unknown"

    visible = next(u for u in snap.units if u.id == 21)
    assert visible.visibility == "visible"
    assert visible.hp == 60

    # the stale values must not appear anywhere in the rendered snapshot
    rendered = snap.render()
    assert '"size":7' not in rendered
    assert '"hp":80' not in rendered


def test_cold_start_own_entities_unaffected():
    """Own entities are always current knowledge: the own city renders its
    full production/economy fields even though the extractor fog-gates foreign
    entities."""
    obs = base_world()
    snap = _extract(obs)
    rome = next(c for c in snap.cities if c.is_own)
    assert rome.visibility == "visible"
    assert rome.size == 1
    assert rome.food_surplus == 2
    assert rome.shield_surplus == 1
    assert rome.buy_cost == 20
    assert rome.production_candidates == ["Settlers", "Warriors"]
    assert rome.rel_to_capital == "here (dist 0)"
    # a second own city gets a relative direction/distance to the capital
    obs2 = base_world(
        cities=[own_city(1, "Rome", 1, 1), own_city(4, "Antium", 3, 2)],
    )
    snap2 = _extract(obs2)
    antium = next(c for c in snap2.cities if c.id == 4)
    assert antium.rel_to_capital == "2E 1S (dist 2)"


# ---------------------------------------------------------------------------
# budget + general fog consistency on the base world
# ---------------------------------------------------------------------------


def test_base_world_snapshot_within_budget_and_deterministic():
    snap1 = _extract(base_world())
    snap2 = _extract(base_world())
    assert snap1.render() == snap2.render(), "extraction is deterministic"
    assert snap1.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET
    assert snap1.omitted.summary == "nothing omitted"


def test_base_world_fog_consistency():
    snap = _extract(base_world())
    # every visible foreign unit sits on a visible tile; every foreign city
    # on a non-visible tile is last_known with unknown mutable fields
    for u in snap.units:
        if not u.is_own:
            assert u.visibility in ("visible", "not_visible")
    carthage = next(c for c in snap.cities if c.id == 2)
    assert carthage.visibility == "visible"  # (4,4) is visible in base_world
    assert carthage.size == 5
    assert carthage.owner_id == ENEMY
    # foreign -1 placeholders never leak as literal values
    enemy = next(u for u in snap.units if u.id == 20)
    assert enemy.moves_left is None
    rendered = snap.render()
    assert '"moves_left":-1' not in rendered
    assert '"size":-1' not in rendered
    # own units carry per-unit move candidates with terrain notes
    own = next(u for u in snap.units if u.id == 10)
    assert {m.direction for m in own.move_candidates} == {
        "N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    assert all(m.terrain or m.note for m in own.move_candidates)
    # the armed own warrior at (2,1) is adjacent to the visible enemy at
    # (2,2) -> in contact, and has an attack candidate with a direction
    warrior = next(u for u in snap.units if u.id == 11)
    assert warrior.in_contact is True
    assert any(a.direction == "S" and a.target_unit_id == 20
               for a in warrior.attack_candidates)
