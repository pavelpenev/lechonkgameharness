# SPDX-License-Identifier: GPL-3.0-or-later
"""lechonk.adapter.obs_extract — CivRealm observation dict -> core Snapshot.

GPL-3.0-or-later (design-plan-freeciv-v1.1.md decision 7): this module is part
of the adapter package, the ONLY lechonk package that may import civrealm. It
deliberately does NOT import civrealm at module level (``import civrealm``
performs an HTTP pubstatus fetch at import time, port_utils.py:44-46); every
input is a plain dict, so the module stays unit-testable offline.

Fog gating (plan §3 line 36, §7.2; the pipeline invariant over the whole
policy-facing surface):
- map status trichotomy maps exactly: 0=unexplored, 1=last-known, 2=visible
  (map_state.py:119-132). Terrain is rendered current only when seen, last-known
  when known-but-fogged; owners are rendered only for seen tiles.
- A vanished enemy is NOT destroyed: foreign units render only from what the
  client currently knows, gated on tile visibility. A foreign unit present in
  the client's unit list but on a non-visible tile is rendered as
  ``not_visible`` with mutable fields unknown — never with stale values, never
  as "destroyed".
- Cold start after checkpoint load: revealed-but-not-visible mutable fields
  (foreign city size/owner, foreign unit hp) are unknown until observed.
- Embassy-gated intel is respected: fields the client does not know render as
  "unknown", never as 0/-1 placeholders.
- Empty observations are classified by the RETURNED FLAGS (terminated vs
  truncated; game over vs my_player_is_defeated) — never from the dict alone
  (freeciv_base_env.py step() + civ_controller.py:430-435).

Token budget: the rendered snapshot is capped at
``lechonk.core.schemas.SNAPSHOT_TOKEN_BUDGET`` (8,000) with deterministic
visibility-safe truncation (idle/distant units elided first; armed-in-contact
units never) and the ``omitted`` disclosure field (§3 line 32).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from lechonk.core.activities import ACTIVITY_NAMES
from lechonk.core.schemas import (
    SNAPSHOT_TOKEN_BUDGET,
    AttackCandidate,
    CityView,
    DiplomacyView,
    Landmark,
    MapView,
    MoveCandidate,
    OmittedDisclosure,
    PlayerView,
    ResearchCandidate,
    ResearchView,
    Snapshot,
    UnitView,
)

# --- CivRealm wire constants (vendor/civrealm, source-verified) -------------
# map_state.py:119-132
TILE_UNEXPLORED = 0
TILE_LAST_KNOWN = 1
TILE_VISIBLE = 2
TERRAIN_UNKNOWN = 255
NO_OWNER = 255
# map_const.py:1-16 — freeciv's DIR8 numbering (y grows southward on the
# rendered map). action_prob dicts are keyed by these indices.
DIR8_NAMES: tuple[str, ...] = ("NW", "N", "NE", "W", "E", "SW", "S", "SE")
DIR8_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
DIR8_INDEX = {name: i for i, name in enumerate(DIR8_NAMES)}
SSA_NONE = 0
# version/v14.py:472-478
VUT_NONE = 0
VUT_IMPROVEMENT = 3
VUT_UTYPE = 6
# tech_const.py:34-36
TECH_UNKNOWN = 0
TECH_PREREQS_KNOWN = 1
TECH_KNOWN = 2
# version/v14.py:387
ACTION_ATTACK = 45
# player_const.py:46-57
DS_NAMES = {
    -1: "No contact",  # no diplstate record for this counterpart
    0: "Armistice", 1: "War", 2: "Ceasefire", 3: "Peace", 4: "Alliance",
    5: "No contact", 6: "Team",
}

# Truncation caps (§3: candidate lists capped by priority).
MAX_RESEARCH_CANDIDATES = 12
MAX_PRODUCTION_CANDIDATES = 8
MAX_MESSAGES = 20
# The ASCII map grid is included only on small maps; region summaries +
# landmarks carry large maps (§7.3 map_view).
GRID_MAX_CELLS = 1024
# Per-tech invention fields as filled by the embassy report
# (player_state.py:128-129): ``tech_<numeric id>``. Distinct from the common
# ``tech_goal`` / ``tech_upkeep`` / ``techs_researched`` player fields.
_TECH_FIELD_RE = re.compile(r"^tech_\d+\Z")


@dataclass(frozen=True)
class RulesetNames:
    """Ruleset-derived display tables. Duck-typed so this module never imports
    civrealm; the caller (session/smoke) builds it from the ruleset controller.

    ``city_production_names`` is keyed by ``(production_kind, production_value)``
    AS REPORTED in the city domain (city_state.py adds ``num_impr_types`` to the
    value for VUT_UTYPE — the caller applies the same offset when building the
    table).
    """

    terrains: Mapping[int, str] = field(default_factory=dict)
    unit_types: Mapping[int, str] = field(default_factory=dict)
    improvements: Mapping[int, str] = field(default_factory=dict)
    nations: Mapping[int, str] = field(default_factory=dict)
    tech_costs: Mapping[int, Optional[int]] = field(default_factory=dict)
    city_production_names: Mapping[tuple[int, int], str] = field(default_factory=dict)


class ExtractOutcome(str, Enum):
    OK = "ok"
    TERMINATED_GAME_OVER = "terminated_game_over"  # game over, my player not defeated
    TERMINATED_DEFEATED = "terminated_defeated"  # my_player_is_defeated
    TRUNCATED_INFRA = "truncated_infra"  # infra timeout / horizon truncation


@dataclass
class Extraction:
    outcome: ExtractOutcome
    snapshot: Optional[Snapshot]
    reason: str = ""


# ---------------------------------------------------------------------------
# Small conversion helpers (observation dicts carry numpy arrays / BitVectors)
# ---------------------------------------------------------------------------


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _grid2d(arr: Any) -> list[list[Any]]:
    """2D map array (numpy or nested lists) -> nested lists."""
    arr = _to_list(arr)
    if not isinstance(arr, list) or not arr or not isinstance(arr[0], list):
        return []
    return arr


def _bit_list(value: Any) -> list[int]:
    """BitVector / numpy bool array / list -> list of 0/1 ints."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [int(bool(v)) for v in value]
    # BitVector: index by position.
    try:
        return [int(value[i]) for i in range(len(value))]
    except Exception:
        return []


def _int_key(key: Any) -> int:
    try:
        return int(key)
    except (TypeError, ValueError):
        return -1


def _rel(dx: int, dy: int) -> str:
    """Relative direction/distance statement: '2E 1S (dist 2)'."""
    parts = []
    if dx > 0:
        parts.append(f"{dx}E")
    elif dx < 0:
        parts.append(f"{-dx}W")
    if dy > 0:
        parts.append(f"{dy}S")
    elif dy < 0:
        parts.append(f"{-dy}N")
    dist = max(abs(dx), abs(dy))
    if not parts:
        return "here (dist 0)"
    return f"{' '.join(parts)} (dist {dist})"


def _terrain_letter(terrain_id: int, ruleset: RulesetNames) -> Optional[str]:
    if terrain_id is None or terrain_id == TERRAIN_UNKNOWN:
        return None
    name = ruleset.terrains.get(terrain_id)
    if not name:
        return None
    for ch in name:
        if ch.isalnum():
            return ch.upper()
    return None


def _msg_text(m: Any) -> str:
    if isinstance(m, dict):
        for key in ("text", "message", "msg"):
            if key in m and m[key] is not None:
                return str(m[key])
        return str(m)
    return str(m)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_snapshot(
    observation: Mapping[str, Any],
    info: Optional[Mapping[str, Any]] = None,
    *,
    terminated: bool = False,
    truncated: bool = False,
    my_player_is_defeated: bool = False,
    my_player_id: Optional[int] = None,
    ruleset: Optional[RulesetNames] = None,
    action_probs: Optional[Mapping[int, Mapping[int, Mapping[int, Mapping[str, float]]]]] = None,
    token_budget: int = SNAPSHOT_TOKEN_BUDGET,
) -> Extraction:
    """CivRealm observation dict -> fog-gated core Snapshot.

    ``terminated``/``truncated``/``my_player_is_defeated`` are the flags
    RETURNED BY THE ENV (freeciv_base_env.py step()); empty observations are
    classified by those flags, never from the dict alone.

    ``action_probs`` (optional) is ``unit_id -> dir8 -> action_key ->
    {'min': p, 'max': p}`` (percent), as surfaced client-side in
    ``unit_ctrl.prop_actions.unit_data[uid].action_prob`` (plan §7.2 combat
    odds). When absent, attack candidates carry no odds (disclosed).
    """
    ruleset = ruleset or RulesetNames()
    info = info or {}

    # --- classify by returned flags (never the dict alone) -----------------
    if truncated and not terminated:
        outcome = ExtractOutcome.TRUNCATED_INFRA
        reason = ("empty observation with truncated=True: infrastructure timeout "
                  "(server_timeout/begin_turn_timeout) or turn-horizon truncation; "
                  "classified from the returned flag, never the dict")
    elif terminated:
        if my_player_is_defeated:
            outcome = ExtractOutcome.TERMINATED_DEFEATED
            reason = "terminated=True with my_player_is_defeated=True"
        else:
            outcome = ExtractOutcome.TERMINATED_GAME_OVER
            reason = "terminated=True with my_player_is_defeated=False (game over / victory)"
    elif not observation:
        raise ValueError(
            "empty observation without terminated/truncated flags: refusing to "
            "classify from the dict alone (plan §7.1)")
    else:
        outcome = ExtractOutcome.OK
        reason = ""

    if not observation:
        return Extraction(outcome, None, reason)

    # --- parse domains ------------------------------------------------------
    players = {_int_key(k): v for k, v in (observation.get("player") or {}).items()}
    cities = {_int_key(k): v for k, v in (observation.get("city") or {}).items()}
    units = {_int_key(k): v for k, v in (observation.get("unit") or {}).items()}
    techs = {_int_key(k): v for k, v in (observation.get("tech") or {}).items()}
    dipl = {_int_key(k): v for k, v in (observation.get("dipl") or {}).items()}
    game = observation.get("game") or {}

    map_domain = observation.get("map") or {}
    status = _grid2d(map_domain.get("status"))
    terrain = _grid2d(map_domain.get("terrain"))
    tile_owner = _grid2d(map_domain.get("tile_owner"))
    # The map arrays are X-MAJOR, exactly as map_state allocates and writes
    # them: map_allocate(x_size, y_size) -> np.full((x_size, y_size)) and
    # status[x, y] = ... (map_state.py:52-63, 120-132). The grid's FIRST
    # dimension is the x extent (map width); the second is the y extent
    # (map height). Verified against the live map (78 x 52 arrays, units at
    # x=75): reading [y][x] transposes every tile on non-square maps.
    width = len(status)
    height = len(status[0]) if width else 0

    if my_player_id is None:
        my_player_id = _infer_my_player_id(players)

    def tile_status(x: int, y: int) -> int:
        if 0 <= y < height and 0 <= x < width:
            return int(status[x][y]) if status[x] else TILE_UNEXPLORED
        return TILE_UNEXPLORED

    my = players.get(my_player_id, {})
    player_names = {pid: (p.get("name") if isinstance(p, Mapping) else None)
                    for pid, p in players.items()}

    # --- capital (own city with the smallest id; freeciv assigns city ids in
    # --- creation order, so the starting city is the capital) ---------------
    own_cities = {cid: c for cid, c in cities.items()
                  if c.get("owner") == my_player_id}
    capital_id = min(own_cities) if own_cities else None
    capital_xy = None
    if capital_id is not None:
        capital_xy = (int(own_cities[capital_id].get("x", 0)),
                      int(own_cities[capital_id].get("y", 0)))

    def rel_to_capital(x: int, y: int) -> Optional[str]:
        if capital_xy is None:
            return None
        return _rel(x - capital_xy[0], y - capital_xy[1])

    # --- player + research (computed from the tech domain, cross-checked) ---
    player_view, research_candidates = _extract_player(
        my, techs, ruleset, my_player_id, player_names)

    # --- cities (fog-gated) -------------------------------------------------
    city_views, production_candidates_capped = _extract_cities(
        cities, my_player_id, player_names, tile_status, rel_to_capital, ruleset)

    # --- units (fog-gated) --------------------------------------------------
    unit_views = _extract_units(
        units, my_player_id, player_names, tile_status, width, height,
        terrain, ruleset, action_probs, capital_xy)

    # --- map view (trichotomy preserved) ------------------------------------
    map_view = _extract_map_view(
        width, height, status, terrain, tile_owner, ruleset,
        cities, units, my_player_id, player_names, rel_to_capital)

    # --- diplomacy (embassy-gated intel respected) ---------------------------
    diplomacy_views = _extract_diplomacy(dipl, players, player_names, my_player_id)

    # --- messages -------------------------------------------------------------
    messages = _extract_messages(info, game)

    # --- assemble + enforce the token budget ---------------------------------
    omitted = OmittedDisclosure()
    # Initial deterministic caps (§3: candidate lists capped); deeper caps under
    # budget pressure happen in _apply_token_budget.
    if len(research_candidates) > MAX_RESEARCH_CANDIDATES:
        research_candidates = research_candidates[:MAX_RESEARCH_CANDIDATES]
        omitted.research_candidates_capped = True
    if production_candidates_capped:
        omitted.production_candidates_capped = True
    if len(messages) > MAX_MESSAGES:
        omitted.messages_elided = len(messages) - MAX_MESSAGES
        messages = messages[:MAX_MESSAGES]
    snapshot = Snapshot(
        turn=int(info.get("turn", 0) or 0),
        player=player_view,
        research_candidates=research_candidates,
        cities=city_views,
        units=unit_views,
        map_view=map_view,
        diplomacy=diplomacy_views,
        messages=messages,
        omitted=omitted,
    )
    _apply_token_budget(snapshot, token_budget)
    return Extraction(outcome, snapshot, reason)


def _infer_my_player_id(players: Mapping[int, Mapping[str, Any]]) -> Optional[int]:
    """My player is the entry carrying own-player-only fields (player_state.py:
    hearsay/embassy intel never sets 'embassy_txt'/'tech_goal'/'mood')."""
    markers = ("embassy_txt", "tech_goal", "mood", "total_bulbs_prod")
    for pid in sorted(players):
        p = players[pid]
        if isinstance(p, Mapping) and any(m in p for m in markers):
            return pid
    return None


def _extract_player(
    my: Mapping[str, Any],
    techs: Mapping[int, Mapping[str, Any]],
    ruleset: RulesetNames,
    my_player_id: Optional[int],
    player_names: Mapping[int, Optional[str]],
) -> tuple[PlayerView, list[ResearchCandidate]]:
    researching_id = my.get("researching")
    # Cross-check the display name against the tech domain (plan §7.2: compute
    # research display from the tech domain; the PR #7 research_name fix is
    # verified applied per docs/gate-record.md, but we do not trust it alone).
    target_name = None
    if researching_id not in (None, 0):
        tech_entry = techs.get(int(researching_id)) if researching_id is not None else None
        if tech_entry is not None:
            target_name = tech_entry.get("name")
            if my.get("research_name") and my["research_name"] != target_name:
                # Upstream field disagrees with the tech domain; the tech domain
                # wins (it is derived from the ruleset table).
                pass
        else:
            target_name = my.get("research_name")
    bulbs = my.get("bulbs_researched")
    cost = my.get("researching_cost")
    total_bulbs = my.get("total_bulbs_prod")
    turns_left = None
    if researching_id not in (None, 0) and isinstance(bulbs, int) \
            and isinstance(cost, int) and isinstance(total_bulbs, int) and total_bulbs > 0:
        turns_left = max(1, math.ceil((cost - bulbs) / total_bulbs))
    nation = my.get("nation")
    player_view = PlayerView(
        player_id=my_player_id if my_player_id is not None else -1,
        name=my.get("name"),
        nation=(ruleset.nations.get(int(nation)) if nation is not None else None)
        or (f"nation#{nation}" if nation is not None else None),
        score=my.get("score"),
        gold=my.get("gold"),
        tax_rate=my.get("tax"),
        science_rate=my.get("science"),
        luxury_rate=my.get("luxury"),
        government=my.get("government_name"),
        is_alive=my.get("is_alive"),
        techs_researched=my.get("techs_researched"),
        research=ResearchView(
            current_target=target_name,
            current_target_id=(int(researching_id)
                               if researching_id not in (None, 0) else None),
            bulbs_researched=bulbs if isinstance(bulbs, int) else None,
            cost_total=cost if isinstance(cost, int) else None,
            turns_left=turns_left,
        ),
    )

    # Research candidates: techs whose prerequisites are known (tech domain
    # inv_state), sorted by (cost, name) — the §7.4 priority ordering is applied
    # by the comparator/candidates layer, not here.
    candidates: list[ResearchCandidate] = []
    for tech_id in sorted(techs):
        entry = techs[tech_id]
        if not isinstance(entry, Mapping):
            continue
        if entry.get("inv_state") != TECH_PREREQS_KNOWN:
            continue
        reqs = entry.get("reqs") or {}
        missing = []
        for req_value, active in sorted(reqs.items(), key=lambda kv: _int_key(kv[0])):
            if not active:
                req_tech = techs.get(_int_key(req_value))
                missing.append(req_tech.get("name") if req_tech else f"req#{_int_key(req_value)}")
        cost = ruleset.tech_costs.get(tech_id)
        candidates.append(ResearchCandidate(
            tech_id=tech_id,
            name=str(entry.get("name") or f"tech#{tech_id}"),
            cost=cost if isinstance(cost, int) else None,
            note=("requires: " + ", ".join(missing)) if missing else "",
        ))
    candidates.sort(key=lambda c: (c.cost is None, c.cost or 0, c.name))
    return player_view, candidates


def _extract_cities(
    cities: Mapping[int, Mapping[str, Any]],
    my_player_id: Optional[int],
    player_names: Mapping[int, Optional[str]],
    tile_status,
    rel_to_capital,
    ruleset: RulesetNames,
) -> tuple[list[CityView], bool]:
    views: list[CityView] = []
    production_candidates_capped = False
    for cid in sorted(cities):
        c = cities[cid]
        if not isinstance(c, Mapping):
            continue
        x, y = int(c.get("x", 0)), int(c.get("y", 0))
        owner = c.get("owner")
        is_own = owner == my_player_id
        visible = tile_status(x, y) == TILE_VISIBLE
        if is_own:
            # Own cities are always currently visible to their owner.
            prod_kind = c.get("production_kind")
            prod_value = c.get("production_value")
            production = None
            if prod_kind not in (None, VUT_NONE) and prod_value is not None:
                production = ruleset.city_production_names.get(
                    (int(prod_kind), int(prod_value)))
                if production is None:
                    production = f"kind{int(prod_kind)}#{int(prod_value)}"
            can_build = _bit_list(c.get("can_build_unit"))
            buildable = sorted(
                ruleset.unit_types[i] for i, bit in enumerate(can_build)
                if bit and i in ruleset.unit_types)
            if len(buildable) > MAX_PRODUCTION_CANDIDATES:
                production_candidates_capped = True
            turns_left = c.get("turns_to_prod_complete")
            views.append(CityView(
                id=cid,
                name=str(c.get("name") or f"city#{cid}"),
                x=x, y=y,
                is_own=True,
                owner_id=owner,
                owner_name=player_names.get(owner),
                visibility="visible",
                rel_to_capital=rel_to_capital(x, y),
                size=c.get("size") if isinstance(c.get("size"), int) else None,
                food_surplus=_nz(c.get("surplus_food")),
                shield_surplus=_nz(c.get("surplus_shield")),
                production=production,
                production_turns_left=(int(turns_left)
                                       if isinstance(turns_left, int) and turns_left >= 0
                                       else None),
                buy_cost=c.get("buy_cost") if isinstance(c.get("buy_cost"), int) else None,
                # WP1 package C (§7.4 buy rule): stored shields + fraction of
                # the current production already paid for (upstream
                # prod_process = shield_stock / build_cost, capped at 32767
                # which is the FC_INFINITY sentinel for "nothing in
                # production" — production is None in that case anyway).
                shield_stock=_nz(c.get("shield_stock")),
                production_progress=(
                    round(float(c["prod_process"]), 4)
                    if production is not None
                    and isinstance(c.get("prod_process"), (int, float))
                    and not isinstance(c.get("prod_process"), bool)
                    and 0 <= c["prod_process"] < 32767
                    else None),
                production_candidates=buildable[:MAX_PRODUCTION_CANDIDATES],
            ))
        else:
            # Foreign city: fog-gated. Mutable fields (size, owner) are current
            # only when the city tile is visible; otherwise unknown — never the
            # stale value (cold-start reload policy, §3 line 36).
            if visible:
                views.append(CityView(
                    id=cid,
                    name=str(c.get("name") or f"city#{cid}"),
                    x=x, y=y,
                    is_own=False,
                    owner_id=owner,
                    owner_name=player_names.get(owner),
                    visibility="visible",
                    rel_to_capital=rel_to_capital(x, y),
                    size=c.get("size") if isinstance(c.get("size"), int) else None,
                ))
            else:
                views.append(CityView(
                    id=cid,
                    name=str(c.get("name") or f"city#{cid}"),
                    x=x, y=y,
                    is_own=False,
                    owner_id=None,  # unknown: ownership can change while fogged
                    owner_name=None,
                    visibility="last_known",
                    rel_to_capital=rel_to_capital(x, y),
                    size=None,  # unknown: size can change while fogged
                ))
    return views, production_candidates_capped


def _nz(value: Any) -> Optional[int]:
    """Upstream -1/zero placeholders for foreign entities -> None (unknown)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return None if value < 0 else value


def _extract_units(
    units: Mapping[int, Mapping[str, Any]],
    my_player_id: Optional[int],
    player_names: Mapping[int, Optional[str]],
    tile_status,
    width: int,
    height: int,
    terrain: list[list[Any]],
    ruleset: RulesetNames,
    action_probs: Optional[Mapping[int, Mapping[int, Mapping[int, Mapping[str, float]]]]],
    capital_xy: Optional[tuple[int, int]],
) -> list[UnitView]:
    # Index visible foreign units per tile for contact + attack candidates.
    foreign_by_tile: dict[tuple[int, int], list[int]] = {}
    for uid, u in units.items():
        if not isinstance(u, Mapping) or u.get("owner") == my_player_id:
            continue
        x, y = int(u.get("x", 0)), int(u.get("y", 0))
        if tile_status(x, y) == TILE_VISIBLE:
            foreign_by_tile.setdefault((x, y), []).append(uid)

    views: list[UnitView] = []
    for uid in sorted(units):
        u = units[uid]
        if not isinstance(u, Mapping):
            continue
        x, y = int(u.get("x", 0)), int(u.get("y", 0))
        owner = u.get("owner")
        is_own = owner == my_player_id
        visible = tile_status(x, y) == TILE_VISIBLE
        # Terrain of the unit's own tile, fog-gated: terrain is written only
        # for seen tiles (map_state.py), so a non-visible tile renders None
        # (never a stale value). WP1 package C: the §7.4 improve-by-terrain
        # rule reads this field.
        tile_terrain = None
        if visible:
            t_id = (int(terrain[x][y])
                    if terrain and 0 <= x < len(terrain) and terrain[x]
                    and 0 <= y < len(terrain[x]) else TERRAIN_UNKNOWN)
            tile_terrain = (ruleset.terrains.get(t_id)
                            if t_id != TERRAIN_UNKNOWN else None)
        type_name = str(u.get("type_rule_name") or "unknown")
        activity = u.get("activity")
        ssa = u.get("ssa_controller")
        attack_strength = u.get("type_attack_strength")

        # Armed-in-contact: own armed unit adjacent to a visible foreign unit
        # or sitting next to a visible foreign city. Never elided by truncation.
        in_contact = False
        if is_own and isinstance(attack_strength, int) and attack_strength > 0:
            for dir8 in range(8):
                dx, dy = DIR8_DELTAS[dir8]
                if foreign_by_tile.get((x + dx, y + dy)):
                    in_contact = True
                    break

        move_candidates: list[MoveCandidate] = []
        attack_candidates: list[AttackCandidate] = []
        if is_own:
            moves_left = u.get("moves_left")
            can_move = isinstance(moves_left, int) and moves_left > 0
            if can_move:
                for dir8 in range(8):
                    dx, dy = DIR8_DELTAS[dir8]
                    name = DIR8_NAMES[dir8]
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    st = tile_status(nx, ny)
                    t_id = (int(terrain[nx][ny])
                            if terrain and 0 <= nx < len(terrain) and terrain[nx]
                            and 0 <= ny < len(terrain[nx]) else TERRAIN_UNKNOWN)
                    if st == TILE_VISIBLE:
                        note = "visible"
                    elif st == TILE_LAST_KNOWN:
                        note = "last known (units not visible)"
                    else:
                        note = "unexplored"
                    t_name = ruleset.terrains.get(t_id) if t_id != TERRAIN_UNKNOWN else None
                    move_candidates.append(MoveCandidate(
                        direction=name,
                        terrain=t_name,
                        note=note,
                    ))
            if isinstance(attack_strength, int) and attack_strength > 0:
                for dir8 in range(8):
                    dx, dy = DIR8_DELTAS[dir8]
                    name = DIR8_NAMES[dir8]
                    nx, ny = x + dx, y + dy
                    for target_id in foreign_by_tile.get((nx, ny), []):
                        target = units.get(target_id) or {}
                        odds = None
                        if action_probs:
                            probs = action_probs.get(uid, {}).get(dir8, {})
                            attack_prob = probs.get(ACTION_ATTACK) if probs else None
                            if isinstance(attack_prob, Mapping) and "max" in attack_prob:
                                odds = round(float(attack_prob["max"]) / 100.0, 3)
                        attack_candidates.append(AttackCandidate(
                            direction=name,
                            target_unit_id=target_id,
                            target_type=str(target.get("type_rule_name") or "unknown"),
                            odds=odds,
                        ))

        if is_own:
            # Own units: full current detail (the server always tells us about
            # our own units). -1 placeholders never occur for own units.
            views.append(UnitView(
                id=uid,
                owner_id=owner,
                owner_name=player_names.get(owner),
                is_own=True,
                type_name=type_name,
                x=x, y=y,
                terrain=tile_terrain,
                hp=u.get("health") if isinstance(u.get("health"), int) else None,
                moves_left=moves_left if isinstance(u.get("moves_left"), int) else None,
                activity=activity if isinstance(activity, int) else None,
                activity_name=(ACTIVITY_NAMES.get(activity)
                               if isinstance(activity, int) else None),
                ssa_controller=ssa if isinstance(ssa, int) else None,
                veteran=bool(u.get("veteran")) if u.get("veteran") is not None else None,
                visibility="visible",
                in_contact=in_contact,
                move_candidates=move_candidates,
                attack_candidates=attack_candidates,
            ))
        elif visible:
            # Foreign unit on a currently visible tile: current observation.
            # moves_left/home_city/upkeep arrive as -1 placeholders -> unknown.
            views.append(UnitView(
                id=uid,
                owner_id=owner,
                owner_name=player_names.get(owner),
                is_own=False,
                type_name=type_name,
                x=x, y=y,
                terrain=tile_terrain,
                hp=u.get("health") if isinstance(u.get("health"), int) else None,
                moves_left=None,
                activity=None,
                activity_name=None,
                ssa_controller=None,
                veteran=bool(u.get("veteran")) if u.get("veteran") is not None else None,
                visibility="visible",
                in_contact=False,
            ))
        else:
            # Revealed-but-not-visible (e.g. cold start after checkpoint load):
            # the unit is known to the client but its tile is not currently
            # visible. Mutable fields are unknown — never the stale value, and
            # never a claim of destruction or survival.
            views.append(UnitView(
                id=uid,
                owner_id=owner,
                owner_name=player_names.get(owner),
                is_own=False,
                type_name=type_name,
                x=x, y=y,
                terrain=tile_terrain,
                hp=None,  # unknown: health changes while fogged
                moves_left=None,
                activity=None,
                activity_name=None,
                ssa_controller=None,
                veteran=None,
                visibility="not_visible",
                in_contact=False,
            ))
    return views


def _extract_map_view(
    width: int,
    height: int,
    status: list[list[Any]],
    terrain: list[list[Any]],
    tile_owner: list[list[Any]],
    ruleset: RulesetNames,
    cities: Mapping[int, Mapping[str, Any]],
    units: Mapping[int, Mapping[str, Any]],
    my_player_id: Optional[int],
    player_names: Mapping[int, Optional[str]],
    rel_to_capital,
) -> MapView:
    counts = {"visible": 0, "last_known": 0, "unexplored": 0}
    for y in range(height):
        for x in range(width):
            st = int(status[x][y]) if status[x] else TILE_UNEXPLORED
            if st == TILE_VISIBLE:
                counts["visible"] += 1
            elif st == TILE_LAST_KNOWN:
                counts["last_known"] += 1
            else:
                counts["unexplored"] += 1

    region_notes = [
        f"map {width}x{height}: visible={counts['visible']} "
        f"last-known={counts['last_known']} unexplored={counts['unexplored']}"
    ]

    # Owners are rendered only for SEEN tiles (tile_owner is stale on fogged
    # tiles — map_state.py writes it only when known==2).
    owner_counts: dict[int, int] = {}
    for y in range(height):
        for x in range(width):
            st = int(status[x][y]) if status[x] else TILE_UNEXPLORED
            if st != TILE_VISIBLE:
                continue
            owner = (int(tile_owner[x][y])
                     if tile_owner and tile_owner[x] else NO_OWNER)
            if owner != NO_OWNER:
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
    for owner in sorted(owner_counts):
        name = player_names.get(owner) or f"player#{owner}"
        region_notes.append(f"visible tiles owned by {name}: {owner_counts[owner]}")

    # ASCII grid on small maps only (deterministic budget guard).
    grid = None
    if width * height <= GRID_MAX_CELLS and width > 0 and height > 0:
        rows = []
        for y in range(height):
            row = []
            for x in range(width):
                st = int(status[x][y]) if status[x] else TILE_UNEXPLORED
                t_id = (int(terrain[x][y])
                        if terrain and terrain[x] else TERRAIN_UNKNOWN)
                letter = _terrain_letter(t_id, ruleset)
                if st == TILE_VISIBLE:
                    row.append(letter or "#")
                elif st == TILE_LAST_KNOWN:
                    row.append(letter.lower() if letter else "~")
                else:
                    row.append(".")
            rows.append("".join(row))
        grid = "\n".join(rows)
        region_notes.append(
            "grid legend: '.'=unexplored, '~'=last-known (terrain unknown), "
            "lowercase=last-known terrain, UPPERCASE=visible terrain")

    # Landmarks: cities and visible enemy unit groups.
    def _st_at(x: int, y: int) -> int:
        if 0 <= y < height and 0 <= x < width:
            return int(status[x][y]) if status[x] else TILE_UNEXPLORED
        return TILE_UNEXPLORED

    landmarks: list[Landmark] = []
    for cid in sorted(cities):
        c = cities[cid]
        if not isinstance(c, Mapping):
            continue
        x, y = int(c.get("x", 0)), int(c.get("y", 0))
        st = _st_at(x, y)
        is_own = c.get("owner") == my_player_id
        name = str(c.get("name") or f"city#{cid}")
        if is_own:
            landmarks.append(Landmark(
                kind="own_city", label=f"city {name}", x=x, y=y,
                rel_to_capital=rel_to_capital(x, y), visibility="visible",
                note=f"size {c.get('size')}"))
        elif st == TILE_VISIBLE:
            landmarks.append(Landmark(
                kind="foreign_city", label=f"city {name}", x=x, y=y,
                rel_to_capital=rel_to_capital(x, y), visibility="visible",
                note=f"owner {player_names.get(c.get('owner')) or c.get('owner')}, "
                     f"size {c.get('size')}"))
        else:
            landmarks.append(Landmark(
                kind="foreign_city", label=f"city {name}", x=x, y=y,
                rel_to_capital=rel_to_capital(x, y), visibility="last_known",
                note="present (last known); size/owner unknown — not currently visible"))
    enemy_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for uid, u in units.items():
        if not isinstance(u, Mapping) or u.get("owner") == my_player_id:
            continue
        x, y = int(u.get("x", 0)), int(u.get("y", 0))
        if _st_at(x, y) == TILE_VISIBLE:
            enemy_groups.setdefault((x, y), []).append(u)
    for (x, y) in sorted(enemy_groups):
        group = enemy_groups[(x, y)]
        types = sorted({str(u.get("type_rule_name") or "unknown") for u in group})
        owners = sorted({u.get("owner") for u in group})
        owner_txt = ", ".join(player_names.get(o) or f"player#{o}" for o in owners)
        landmarks.append(Landmark(
            kind="enemy_units",
            label=f"{len(group)} enemy unit(s): {'/'.join(types)}",
            x=x, y=y,
            rel_to_capital=rel_to_capital(x, y), visibility="visible",
            note=f"owner {owner_txt}"))

    return MapView(
        width=width, height=height,
        tile_counts=counts, grid=grid,
        landmarks=landmarks, region_notes=region_notes,
    )


def _extract_diplomacy(
    dipl: Mapping[int, Mapping[str, Any]],
    players: Mapping[int, Mapping[str, Any]],
    player_names: Mapping[int, Optional[str]],
    my_player_id: Optional[int],
) -> list[DiplomacyView]:
    views: list[DiplomacyView] = []
    for pid in sorted(dipl):
        if pid == my_player_id:
            continue
        entry = dipl[pid]
        if not isinstance(entry, Mapping):
            continue
        state_id = entry.get("diplomatic_state", -1)
        clauses = [str(c) for c in (entry.get("diplomacy_clause_map") or [])]
        initializer = entry.get("meeting_initializers", -1)
        # Embassy-gated intel, respecting upstream's gating semantics exactly
        # (player_state.py:98-130): the hearsay report leaks government /
        # gold / research only when > 0; tax, science, luxury and the tech
        # list only with an embassy. The embassy report fills the per-tech
        # invention fields ``tech_<id>`` (numeric ids); without an embassy
        # they stay None, and the zeroed packet fields for opponents are
        # placeholders, not knowledge — they render as unknown, never as 0.
        p = players.get(pid) or {}

        def _invention_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in entry.items()
                    if _TECH_FIELD_RE.match(k) and v is not None}

        inventions = _invention_fields(p)
        has_embassy = bool(inventions)
        intel_parts = []
        gold = p.get("gold")
        if has_embassy or (isinstance(gold, int) and not isinstance(gold, bool)
                           and gold > 0):
            intel_parts.append(f"gold {gold}")
        researching = p.get("researching")
        if has_embassy or (isinstance(researching, int)
                           and not isinstance(researching, bool)
                           and researching > 0):
            if p.get("research_name") is not None:
                intel_parts.append(f"researching {p['research_name']}")
            elif isinstance(researching, int) and researching > 0:
                intel_parts.append(f"researching tech#{researching}")
        gov = p.get("government")
        if has_embassy or (isinstance(gov, int) and not isinstance(gov, bool)
                           and gov > 0):
            intel_parts.append(f"government {p.get('government_name') or gov}")
        if has_embassy:
            intel_parts.append(f"tax {p.get('tax')}% science {p.get('science')}% "
                               f"luxury {p.get('luxury')}%")
            known_techs = sum(1 for v in inventions.values() if v)
            intel_parts.append(f"{known_techs} techs known")
        if intel_parts:
            intel = (("embassy intel: " if has_embassy else "hearsay intel: ")
                     + ", ".join(intel_parts))
        else:
            intel = "no embassy intel: gold/research/government unknown"
        views.append(DiplomacyView(
            counterpart_id=pid,
            counterpart_name=player_names.get(pid),
            state=DS_NAMES.get(state_id if isinstance(state_id, int) else -1,
                               f"state#{state_id}"),
            meeting_active=(isinstance(initializer, int) and initializer >= 0),
            clauses=clauses,
            intel=intel,
        ))
    return views


def _extract_messages(
    info: Mapping[str, Any],
    game: Mapping[str, Any],
) -> list[str]:
    messages: list[str] = []
    for m in info.get("mini_game_messages") or []:
        messages.append(_msg_text(m))
    game_msgs = game.get("messages") if isinstance(game, Mapping) else None
    if game_msgs:
        messages.append(str(game_msgs))
    return messages


# ---------------------------------------------------------------------------
# Token-budget enforcement (§3: deterministic, visibility-safe truncation)
# ---------------------------------------------------------------------------


def _apply_token_budget(snapshot: Snapshot, budget: int) -> None:
    """Cap the rendered snapshot at ``budget`` tokens.

    Truncation order (deterministic): idle/distant own units elided first
    (farthest from the capital first); armed-in-contact units are NEVER elided;
    then candidate lists capped (research, production, messages); the ASCII map
    grid is dropped last-but-one. Every elision is disclosed in ``omitted``.
    Foreign units are never elided (visible enemies are policy-salient).
    """
    omitted = snapshot.omitted

    if snapshot.estimated_tokens() > budget:
        # 1) Elide own units: idle first, then farthest from the capital. Never
        #    armed-in-contact units, never foreign units.
        capital_xy = None
        for c in snapshot.cities:
            if c.is_own and capital_xy is None:
                capital_xy = (c.x, c.y)
        # capital = first own city in list order (list is sorted by id).

        def elision_key(u: UnitView) -> tuple[int, int, int]:
            working = 1 if (u.activity not in (None, 0)
                            or u.ssa_controller not in (None, SSA_NONE)) else 0
            if capital_xy is not None:
                dist = max(abs(u.x - capital_xy[0]), abs(u.y - capital_xy[1]))
            else:
                dist = 0
            return (working, -dist, u.id)

        while snapshot.estimated_tokens() > budget:
            elidable = [u for u in snapshot.units if u.is_own and not u.in_contact]
            if not elidable:
                break
            elidable.sort(key=elision_key)
            victim = elidable[0]
            snapshot.units.remove(victim)
            omitted.elided_unit_ids.append(victim.id)

        # 2) Cap research candidates.
        if snapshot.estimated_tokens() > budget \
                and len(snapshot.research_candidates) > 6:
            snapshot.research_candidates = snapshot.research_candidates[:6]
            omitted.research_candidates_capped = True
        if snapshot.estimated_tokens() > budget \
                and len(snapshot.research_candidates) > 3:
            snapshot.research_candidates = snapshot.research_candidates[:3]
            omitted.research_candidates_capped = True

        # 3) Cap messages.
        if snapshot.estimated_tokens() > budget and len(snapshot.messages) > 10:
            omitted.messages_elided += len(snapshot.messages) - 10
            snapshot.messages = snapshot.messages[:10]
        if snapshot.estimated_tokens() > budget and len(snapshot.messages) > 5:
            omitted.messages_elided += len(snapshot.messages) - 5
            snapshot.messages = snapshot.messages[:5]

        # 4) Drop the ASCII map grid.
        if snapshot.estimated_tokens() > budget \
                and snapshot.map_view.grid is not None:
            snapshot.map_view.grid = None
            omitted.map_grid_omitted = True

        # 5) Final check: fail loudly rather than ship an over-budget snapshot.
        if snapshot.estimated_tokens() > budget:
            raise ValueError(
                f"snapshot exceeds token budget {budget} even after truncation "
                f"({snapshot.estimated_tokens()} tokens; protected units remain)")

    # Disclosure summary (§3 line 32) — always computed, also when only the
    # initial deterministic caps (research/messages) fired.
    parts = []
    if omitted.elided_unit_ids:
        parts.append(f"elided {len(omitted.elided_unit_ids)} own unit(s) "
                     f"(idle/distant first): {omitted.elided_unit_ids}")
    if omitted.research_candidates_capped:
        parts.append("research candidate list capped")
    if omitted.messages_elided:
        parts.append(f"elided {omitted.messages_elided} message(s)")
    if omitted.map_grid_omitted:
        parts.append("map grid omitted (region summaries + landmarks retained)")
    omitted.summary = "; ".join(parts) if parts else "nothing omitted"
