# SPDX-License-Identifier: MIT
"""lechonk.comparator.scripted — the comparator policy (plan §7.4, verbatim).

The comparator is the ACTIVE scripted policy used for paired evaluation and it
doubles as the degraded-default engine (plan §3 line 44). It consumes
core.schemas Snapshot + CandidateTable and emits candidate-ID choices; all
execution goes through the adapter (the turn driver owns it). This module is
MIT and never imports civrealm or lechonk.adapter (test-enforced).

Published policy (design-plan-freeciv-v1.1.md §7.4, published verbatim in the
README):
- Research priority list (first affordable legal candidate wins):
  Alphabet -> Pottery -> Agriculture -> Bronze Working -> Writing -> Currency
  -> Masonry -> Wheel -> trade-adjacent techs; tie-break: lowest cost, then
  list order.
- Production priority list (per city, first legal): defensive unit if city
  undefended -> builder-type unit while improvements remain and builders <
  cities+1 -> growth building (Granary-class) -> science building -> wealth
  fallback. Buy-production when gold > 2x buy_cost and shield stock < 25%
  complete.
- Unit defaults: found city on the first legal candidate tile when a settler
  exists and no city is being founded this turn; else improve
  (road/irrigation/mine by terrain yield); else fortify. Military: defend the
  nearest own city; attack only adjacent visible targets with favorable odds
  (or without odds data when odds are unavailable, per §6).
- Diplomacy default: decline/stop all negotiations; no proactive offers
  (decline path = stop_negotiation).

Documented interpretation choices (none contradict the plan):
- "Affordable" for research = present in the legal candidate table (freeciv
  research targets are bulb-paced, not gold-gated; the table enumerates
  exactly the researchable techs). Tech costs are not exposed through the
  CivRealm ruleset packets (verified live: all None), so the cost tie-break
  degrades to list order in practice; the sort key implements it per the
  published policy.
- "City undefended" = no own military unit on the city tile (snapshot units,
  including units gated out of the candidate table — a fortified defender
  still defends).
- "Improvements remain" = the city has at least one legal produce_ candidate
  for a building (name classes, classic ruleset — the ruleset is pinned to
  CivRealm's default `classic`, decision 8; names grounded from the image's
  classic ruleset files).
- The military ladder is defend-first (the plan lists defend before attack):
  a military unit not on its nearest own city moves to defend it (or
  fortifies in place when blocked); a military unit already on its city (or
  with no cities at all) attacks an adjacent visible target with favorable
  odds, else fortifies. "Favorable" = odds_max >= 0.5 (the upper bound of the
  surfaced win-probability range); with odds unavailable the attack is
  candidate-gated (plan §6 residual).
- The buy rule is skipped for a city whose production the ladder switches
  this turn (the buy would complete the NEW item, not the assessed one), and
  the gold check simulates the spend across cities in id order.
- At most one action per unit per turn and at most one city founding per
  turn ("no city is being founded this turn").
- The comparator only ever considers candidates whose verb is on the exposed
  whitelist (defense in depth — the candidate table already excludes
  exploration/autosettler/etc. by construction, plan §3 line 37).
- The published list says "Wheel"; the classic ruleset names the tech
  "The Wheel" — matched by stripping a leading "The ". "Agriculture" has no
  classic counterpart (classic has no such tech); the entry is kept verbatim
  and simply never matches in classic.
- Production hysteresis (§7.4 AMENDMENT 2026-10-24): a production tier is
  SATISFIED when the city's current production target already belongs to
  that tier's output class, even when the target is absent from the
  candidate table (upstream never lists the current production target as a
  produce candidate — the root cause of the Warriors/Barracks 2-cycle
  observed in artifacts/runs/wp1-comparator/). A satisfied top applicable
  tier does not switch; a current target satisfying only a lower tier
  never suppresses a higher-priority switch. The degraded-default entry
  point is unaffected (it only resolves EMPTY production, where no target
  exists to satisfy a tier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from lechonk.core.schemas import Candidate, CandidateTable, Snapshot

# ---------------------------------------------------------------------------
# Published constants (plan §7.4)
# ---------------------------------------------------------------------------

# WP1 placeholder provenance class for the WP2 ledger: every decision this
# package emits is a scripted resolution in WP1 (the comparator doubles as
# the degraded-default engine, plan §3 line 44).
PROVENANCE_CLASS = "scripted_default"

# Research priority list (verbatim §7.4).
RESEARCH_PRIORITY: tuple[str, ...] = (
    "Alphabet", "Pottery", "Agriculture", "Bronze Working", "Writing",
    "Currency", "Masonry", "Wheel",
)
# Trade-adjacent techs (classic: the Trade/Banking/Economics line that leads
# to caravan trade routes and market income).
TRADE_ADJACENT_TECHS: tuple[str, ...] = ("Trade", "Banking", "Economics")

# Classic-ruleset unit classes (grounded from the image's classic
# units.ruleset; the ruleset is pinned, decision 8).
MILITARY_LAND_UNITS: tuple[str, ...] = (
    "Warriors", "Phalanx", "Archers", "Legion", "Pikemen", "Musketeers",
    "Partisan", "Alpine Troops", "Riflemen", "Marines", "Paratroopers",
    "Mech. Inf.", "Horsemen", "Chariot", "Knights", "Dragoons", "Cavalry",
    "Armor", "Catapult", "Cannon", "Artillery", "Howitzer",
)
MILITARY_NAVAL_AIR_UNITS: tuple[str, ...] = (
    "Fighter", "Bomber", "Helicopter", "Stealth Fighter", "Stealth Bomber",
    "Trireme", "Caravel", "Galleon", "Frigate", "Ironclad", "Destroyer",
    "Cruiser", "AEGIS Cruiser", "Battleship", "Submarine", "Carrier",
    "Transport", "Cruise Missile", "Nuclear",
)
MILITARY_UNIT_NAMES = frozenset(MILITARY_LAND_UNITS + MILITARY_NAVAL_AIR_UNITS)
BUILDER_UNITS: tuple[str, ...] = ("Settlers", "Workers", "Engineers")
BUILDER_UNIT_NAMES = frozenset(BUILDER_UNITS)

# Classic-ruleset production classes (grounded from the image's classic
# buildings.ruleset + units.ruleset).
DEFENSIVE_UNITS: tuple[str, ...] = MILITARY_LAND_UNITS
GROWTH_BUILDINGS: tuple[str, ...] = ("Granary", "Aqueduct")  # Granary-class
SCIENCE_BUILDINGS: tuple[str, ...] = ("Library", "University", "Research Lab")
WEALTH_BUILDINGS: tuple[str, ...] = ("Marketplace", "Bank", "Stock Exchange")
# All classic improvements (incl. wonders + Coinage): a city "has improvements
# remaining" while it has a legal produce_ candidate for any of these.
BUILDING_NAMES = frozenset({
    "Airport", "Aqueduct", "Bank", "Barracks", "Barracks II", "Barracks III",
    "Cathedral", "City Walls", "Coastal Defense", "Colosseum", "Courthouse",
    "Factory", "Granary", "Harbor", "Hydro Plant", "Library", "Marketplace",
    "Mass Transit", "Mfg. Plant", "Nuclear Plant", "Offshore Platform",
    "Palace", "Police Station", "Port Facility", "Power Plant",
    "Recycling Center", "Research Lab", "SAM Battery", "SDI Defense",
    "Sewer System", "Solar Plant", "Space Component", "Space Module",
    "Space Structural", "Stock Exchange", "Super Highways", "Supermarket",
    "Temple", "University", "Apollo Program", "A.Smith's Trading Co.",
    "Colossus", "Copernicus' Observatory", "Cure For Cancer",
    "Darwin's Voyage", "Eiffel Tower", "Great Library", "Great Wall",
    "Hanging Gardens", "Hoover Dam", "Isaac Newton's College",
    "J.S. Bach's Cathedral", "King Richard's Crusade", "Leonardo's Workshop",
    "Lighthouse", "Magellan's Expedition", "Manhattan Project",
    "Marco Polo's Embassy", "Michelangelo's Chapel", "Oracle", "Pyramids",
    "SETI Program", "Shakespeare's Theater", "Statue of Liberty",
    "Sun Tzu's War Academy", "United Nations", "Women's Suffrage", "Coinage",
})

# Improve-by-terrain-yield classes (classic terrain names).
IRRIGATION_TERRAINS = frozenset({"Plains", "Grassland", "River"})
MINE_TERRAINS = frozenset({"Hills", "Mountains"})

# Buy rule thresholds (§7.4: "gold > 2x buy_cost and shield stock < 25%
# complete").
BUY_GOLD_MULTIPLIER = 2
BUY_PROGRESS_MAX = 0.25

# "Favorable odds" = the upper bound of the surfaced win-probability range
# reaches this threshold (§7.4 military rule).
FAVORABLE_ODDS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Exposed verb whitelist (plan §3 line 37 / §7.2 — the v1 action surface)
# ---------------------------------------------------------------------------

_UNIT_FIXED_VERBS = frozenset({
    "build_city", "fortify", "sentry", "build_road", "irrigation", "mine",
    "join_city", "set_homecity", "cancel_order", "keep_activity",
})
_CITY_FIXED_VERBS = frozenset({"city_buy_production"})
_DIPL_VERBS = frozenset({"stop_negotiation", "cancel_treaty", "accept_treaty"})
_COMPASS = frozenset({"n", "ne", "e", "se", "s", "sw", "w", "nw"})


def is_exposed_verb(verb: str) -> bool:
    """True iff ``verb`` is on the exposed v1 action surface (§3 line 37).

    Everything else — exploration (ActExplore), autosettler/auto-worker,
    chat, tax/government tuning, spy/trade/transport verbs, sell verbs,
    clause construction, specialists, keep-production — is NOT exposed and
    the comparator never emits it (the candidate table already excludes these
    by construction; this whitelist is defense in depth).
    """
    if verb in _UNIT_FIXED_VERBS or verb in _CITY_FIXED_VERBS \
            or verb in _DIPL_VERBS:
        return True
    if verb.startswith("goto_") or verb.startswith("attack_"):
        return verb.split("_", 1)[1] in _COMPASS
    if verb.startswith("produce_"):
        return len(verb) > len("produce_")
    if verb.startswith("city_work_") or verb.startswith("city_unwork_"):
        return True
    if verb.startswith("research_tech_") or verb.startswith("set_tech_goal_"):
        return True
    return False


# ---------------------------------------------------------------------------
# Decision + decision log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """One emitted policy choice: a candidate-ID reference (never a raw
    coordinate — the batch schema's rule, §7.3)."""

    domain: str  # "unit" | "city" | "tech" | "dipl"
    candidate_id: str
    verb: str
    actor_id: str
    reason: str
    provenance_class: str = PROVENANCE_CLASS
    # Degraded-default trigger ("research_completed" / "production_empty");
    # empty for the comparator, whose reasons are the policy reasons.
    trigger: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DecisionLog:
    """The policy output for one turn: the emitted decisions + the finish
    marker. Both policies end the turn; the driver owns the finish primitive
    (§3: finish phase has exactly one owner — the pipeline)."""

    policy: str  # "comparator" | "degraded_default"
    turn: int
    decisions: tuple[Decision, ...] = ()
    finish_phase: bool = True

    @property
    def by_domain(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.domain] = out.get(d.domain, 0) + 1
        return out

    def candidate_ids(self) -> list[str]:
        return [d.candidate_id for d in self.decisions]


# ---------------------------------------------------------------------------
# Shared selection helpers (also used by defaults.py — "the same published
# priority lists", §7.4 line 163)
# ---------------------------------------------------------------------------


def own_cities(snapshot: Snapshot) -> list[Any]:
    """Own cities in deterministic (id) order."""
    return sorted((c for c in snapshot.cities if c.is_own), key=lambda c: c.id)


def _own_unit_views(snapshot: Snapshot) -> dict[int, Any]:
    return {u.id: u for u in snapshot.units if u.is_own}


def _by_actor(table: CandidateTable, domain: str) -> dict[str, list[Candidate]]:
    """Exposed candidates of one domain grouped by actor, table order kept."""
    out: dict[str, list[Candidate]] = {}
    for c in table.candidates:
        if c.domain == domain and is_exposed_verb(c.verb):
            out.setdefault(c.actor_id, []).append(c)
    return out


def _research_tier(name: str) -> tuple[int, int]:
    """(tier, sub-order) for a tech name: 0=priority list, 1=trade-adjacent,
    2=anything else. Matches classic's "The Wheel" for the published "Wheel"
    (leading article stripped)."""
    bare = name[4:] if name.startswith("The ") else name
    if bare in RESEARCH_PRIORITY:
        return 0, RESEARCH_PRIORITY.index(bare)
    if bare in TRADE_ADJACENT_TECHS:
        return 1, TRADE_ADJACENT_TECHS.index(bare)
    return 2, 0


def research_pick(snapshot: Snapshot,
                  table: CandidateTable) -> Optional[Candidate]:
    """First affordable legal research candidate (§7.4): priority list
    order, then lowest cost, then candidate-table order."""
    cands = [c for c in table.candidates
             if c.domain == "tech" and c.verb.startswith("research_tech_")
             and is_exposed_verb(c.verb)]
    if not cands:
        return None
    costs = {rc.tech_id: rc.cost for rc in snapshot.research_candidates}

    def key(item: tuple[int, Candidate]) -> tuple[int, int, float, int]:
        idx, cand = item
        tier, sub = _research_tier(str(cand.payload.get("tech_name") or ""))
        cost = costs.get(cand.payload.get("tech_id"))
        return (tier, sub, cost if isinstance(cost, (int, float)) else float("inf"), idx)

    return min(enumerate(cands), key=key)[1]


@dataclass(frozen=True)
class ProductionContext:
    """Snapshot-derived production-ladder context (per turn)."""

    military_tiles: frozenset[tuple[int, int]]  # tiles with an own military unit
    builders: int                                # own builder-type units
    own_city_count: int


def production_context(snapshot: Snapshot) -> ProductionContext:
    military_tiles = frozenset(
        (u.x, u.y) for u in snapshot.units
        if u.is_own and u.type_name in MILITARY_UNIT_NAMES)
    builders = sum(1 for u in snapshot.units
                   if u.is_own and u.type_name in BUILDER_UNIT_NAMES)
    return ProductionContext(military_tiles, builders, len(own_cities(snapshot)))


def production_pick(city: Any, produce_cands: Iterable[Candidate],
                    ctx: ProductionContext) -> tuple[Optional[Candidate], str]:
    """The §7.4 production ladder for one city: (pick, tier-reason).

    defensive unit if city undefended -> builder-type unit while improvements
    remain and builders < cities+1 -> growth building (Granary-class) ->
    science building -> wealth fallback -> any remaining legal production.

    Production hysteresis (§7.4 AMENDMENT 2026-10-24): walking the ladder in
    priority order, a tier whose output class already contains the city's
    current production target is SATISFIED — even when that target is absent
    from the candidate table (upstream never lists the current production
    target as a produce candidate) — and a satisfied top applicable tier
    does not switch (pick=None). A current target satisfying only a lower
    tier never suppresses a higher-priority switch.
    """
    produce_cands = list(produce_cands)
    current = city.production

    def first_in(names: Iterable[str]) -> Optional[Candidate]:
        for name in names:
            for c in produce_cands:
                if c.payload.get("production") == name:
                    return c
        return None

    classified = (frozenset(DEFENSIVE_UNITS) | BUILDER_UNIT_NAMES
                  | frozenset(GROWTH_BUILDINGS) | frozenset(SCIENCE_BUILDINGS)
                  | frozenset(WEALTH_BUILDINGS))

    def satisfied(names: Iterable[str]) -> bool:
        return current is not None and current in names

    undefended = (city.x, city.y) not in ctx.military_tiles
    improvements_remain = any(
        c.payload.get("production") in BUILDING_NAMES for c in produce_cands)
    builders_short = ctx.builders < ctx.own_city_count + 1

    tiers: tuple[tuple[str, Iterable[str], bool], ...] = (
        ("defensive_unit_undefended_city", DEFENSIVE_UNITS, undefended),
        ("builder_unit_improvements_remain", BUILDER_UNITS,
         improvements_remain and builders_short),
        ("growth_building", GROWTH_BUILDINGS, True),
        ("science_building", SCIENCE_BUILDINGS, True),
        ("wealth_building", WEALTH_BUILDINGS, True),
    )
    for reason, names, applicable in tiers:
        if not applicable:
            continue
        if satisfied(names):
            return None, f"production_hysteresis:{reason}"
        cand = first_in(names)
        if cand is not None:
            return cand, reason
    # Fallback tier: any remaining legal production. Its output class is
    # everything outside the classified sets above, so a current target in
    # that class satisfies it the same way.
    if current is not None and current not in classified:
        return None, "production_hysteresis:fallback_production"
    for c in produce_cands:
        if c.payload.get("production") not in classified:
            return c, "fallback_production"
    return None, ""


# ---------------------------------------------------------------------------
# Comparator domain policies
# ---------------------------------------------------------------------------


def decide_research(snapshot: Snapshot, table: CandidateTable) -> list[Decision]:
    """Research: enforce the published priority list (no action when the
    current target already is the priority pick)."""
    pick = research_pick(snapshot, table)
    if pick is None:
        return []
    current_id = snapshot.player.research.current_target_id
    if current_id is not None and current_id == pick.payload.get("tech_id"):
        return []  # already researching the priority target
    name = pick.payload.get("tech_name")
    return [Decision(
        domain="tech", candidate_id=pick.id, verb=pick.verb,
        actor_id=pick.actor_id, reason="research_priority_list",
        detail=f"research {name} per the published priority list")]


def decide_production(snapshot: Snapshot,
                      table: CandidateTable) -> list[Decision]:
    """Production ladder per own city + the buy rule (§7.4)."""
    decisions: list[Decision] = []
    ctx = production_context(snapshot)
    gold = snapshot.player.gold
    sim_gold = gold if isinstance(gold, int) else 0
    city_cands = _by_actor(table, "city")
    for city in own_cities(snapshot):
        cands = city_cands.get(str(city.id), [])
        produce_cands = [c for c in cands if c.verb.startswith("produce_")]
        buy_cands = [c for c in cands if c.verb == "city_buy_production"]
        switched = False
        if produce_cands:
            pick, tier = production_pick(city, produce_cands, ctx)
            if pick is not None \
                    and city.production != pick.payload.get("production"):
                decisions.append(Decision(
                    domain="city", candidate_id=pick.id, verb=pick.verb,
                    actor_id=pick.actor_id, reason=f"production_ladder:{tier}",
                    detail=(f"city {city.name} -> "
                            f"{pick.payload.get('production')} ({tier})")))
                switched = True
        # Buy rule — skipped when the ladder switches production this turn
        # (the buy would complete the NEW item, not the assessed one). The
        # gold check simulates the spend across cities in id order.
        if (not switched and buy_cands and city.production is not None
                and isinstance(city.buy_cost, int) and city.buy_cost > 0
                and sim_gold > BUY_GOLD_MULTIPLIER * city.buy_cost
                and city.production_progress is not None
                and city.production_progress < BUY_PROGRESS_MAX):
            buy = buy_cands[0]
            decisions.append(Decision(
                domain="city", candidate_id=buy.id, verb=buy.verb,
                actor_id=buy.actor_id, reason="buy_production_rule",
                detail=(f"city {city.name}: gold {sim_gold} > "
                        f"{BUY_GOLD_MULTIPLIER}x buy_cost {city.buy_cost} and "
                        f"shield stock {city.production_progress:.0%} < "
                        f"{BUY_PROGRESS_MAX:.0%} complete")))
            sim_gold -= city.buy_cost
    return decisions


# Compass -> (dx, dy); y grows southward on the rendered map (map_const.py).
_COMPASS_DELTAS = {
    "n": (0, -1), "ne": (1, -1), "e": (1, 0), "se": (1, 1),
    "s": (0, 1), "sw": (-1, 1), "w": (-1, 0), "nw": (-1, -1),
}
# Verb -> the activity the snapshot reports while the unit is doing it
# (obs_extract ACTIVITY_NAMES).
_ACTIVITY_FOR_VERB = {
    "irrigation": "irrigate", "mine": "mine", "build_road": "road",
    "fortify": "fortified",
}


def _chebyshev(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def _improve_pick(unit_view: Any, verbs: frozenset[str]) -> Optional[str]:
    """Improve verb by terrain yield (§7.4): irrigation on farmland,
    mine on hills/mountains, road otherwise; when the terrain-preferred verb
    is not legal, fall back through road/irrigation/mine in the listed
    order."""
    terrain = unit_view.terrain
    if terrain in IRRIGATION_TERRAINS:
        preferred = "irrigation"
    elif terrain in MINE_TERRAINS:
        preferred = "mine"
    else:
        preferred = "build_road"
    order = [preferred] + [v for v in ("build_road", "irrigation", "mine")
                           if v != preferred]
    for verb in order:
        if verb in verbs:
            return verb
    return None


def _goto_toward(unit_view: Any, target_xy: tuple[int, int],
                 goto_cands: list[Candidate]) -> Optional[Candidate]:
    """First goto candidate (table order = dir8 index order) that strictly
    reduces the Chebyshev distance to the target."""
    cur = _chebyshev(unit_view.x, unit_view.y, *target_xy)
    if cur == 0:
        return None
    for cand in goto_cands:
        compass = str(cand.payload.get("dir")
                      or cand.verb.split("_", 1)[1]).lower()
        dx, dy = _COMPASS_DELTAS[compass]
        if _chebyshev(unit_view.x + dx, unit_view.y + dy, *target_xy) < cur:
            return cand
    return None


def _best_attack(attack_cands: list[Candidate]) -> Optional[Candidate]:
    """The attack to take (§7.4): an adjacent visible target with favorable
    odds (odds_max >= FAVORABLE_ODDS_THRESHOLD) — or any attack candidate when
    odds data is unavailable (plan §6 residual: candidate-gated without
    odds). Among eligible targets the highest odds_max wins; targets without
    odds rank below measured ones; ties break by candidate-table order."""
    best: Optional[Candidate] = None
    best_key: Optional[tuple[int, float, int]] = None
    for idx, cand in enumerate(attack_cands):
        odds_max = cand.payload.get("odds_max")
        if odds_max is None:
            key = (1, 0.0, idx)  # eligible (candidate-gated), ranks last
        else:
            try:
                odds = float(odds_max)
            except (TypeError, ValueError):
                continue
            if odds < FAVORABLE_ODDS_THRESHOLD:
                continue
            key = (0, odds, idx)
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    return best


def _unit_decisions(snapshot: Snapshot, table: CandidateTable) -> list[Decision]:
    decisions: list[Decision] = []
    unit_cands = _by_actor(table, "unit")
    views = _own_unit_views(snapshot)
    cities = own_cities(snapshot)
    founding = False

    def actor_order(actor: str) -> tuple[int, Any]:
        return (0, int(actor)) if actor.isdigit() else (1, actor)

    for actor_id in sorted(unit_cands, key=actor_order):
        cands = unit_cands[actor_id]
        uid = int(actor_id) if actor_id.isdigit() else None
        view = views.get(uid) if uid is not None else None
        if view is None:
            continue  # actor not in the snapshot (stale table entry)
        verbs = frozenset(c.verb for c in cands)
        by_verb: dict[str, Candidate] = {}
        for c in cands:
            by_verb.setdefault(c.verb, c)

        def emit(cand: Candidate, reason: str, detail: str) -> None:
            decisions.append(Decision(
                domain="unit", candidate_id=cand.id, verb=cand.verb,
                actor_id=actor_id, reason=reason, detail=detail))

        # 1) Found city on the first legal candidate tile — one founding per
        #    turn ("no city is being founded this turn").
        if not founding and "build_city" in verbs:
            emit(by_verb["build_city"], "unit_default:found_city",
                 f"unit {uid} ({view.type_name}) founds a city on its tile")
            founding = True
            continue

        # 2) Military: defend the nearest own city; attack only adjacent
        #    visible targets with favorable odds (else fortify).
        if view.type_name in MILITARY_UNIT_NAMES:
            gotos = [c for c in cands if c.verb.startswith("goto_")]
            attacks = [c for c in cands if c.verb.startswith("attack_")]
            if cities:
                nearest = min(
                    cities,
                    key=lambda c: (_chebyshev(view.x, view.y, c.x, c.y), c.id))
                dist = _chebyshev(view.x, view.y, nearest.x, nearest.y)
                if dist > 0:
                    goto = _goto_toward(view, (nearest.x, nearest.y), gotos)
                    if goto is not None:
                        emit(goto, "unit_default:defend_city",
                             f"unit {uid} ({view.type_name}) moves toward "
                             f"city {nearest.name} (dist {dist})")
                        continue
                    if "fortify" in verbs and view.activity_name != "fortified":
                        emit(by_verb["fortify"],
                             "unit_default:defend_city_blocked",
                             f"unit {uid} ({view.type_name}) fortifies in "
                             f"place (blocked) while defending city "
                             f"{nearest.name}")
                        continue
                    continue  # no legal way to approach: hold
                # On the city: take a favorable adjacent attack, else fortify.
                attack = _best_attack(attacks)
                if attack is not None:
                    odds = attack.payload.get("odds_max")
                    emit(attack, "unit_default:attack_favorable",
                         f"unit {uid} ({view.type_name}) attacks "
                         f"{attack.verb} (odds_max "
                         f"{'unknown' if odds is None else f'{float(odds):.0%}'})")
                    continue
                if "fortify" in verbs and view.activity_name != "fortified":
                    emit(by_verb["fortify"], "unit_default:fortify",
                         f"unit {uid} ({view.type_name}) fortifies on city "
                         f"{nearest.name}")
                    continue
                continue
            # No cities at all: attack favorable adjacent targets, else fortify.
            attack = _best_attack(attacks)
            if attack is not None:
                odds = attack.payload.get("odds_max")
                emit(attack, "unit_default:attack_favorable",
                     f"unit {uid} ({view.type_name}) attacks {attack.verb} "
                     f"(odds_max "
                     f"{'unknown' if odds is None else f'{float(odds):.0%}'})")
                continue
            if "fortify" in verbs and view.activity_name != "fortified":
                emit(by_verb["fortify"], "unit_default:fortify",
                     f"unit {uid} ({view.type_name}) fortifies (no city to "
                     f"defend)")
                continue
            continue

        # 3) Civilian: improve by terrain yield, else fortify.
        improve = _improve_pick(view, verbs)
        if improve is not None:
            if view.activity_name == _ACTIVITY_FOR_VERB[improve]:
                continue  # already doing exactly this
            emit(by_verb[improve], f"unit_default:improve:{improve}",
                 f"unit {uid} ({view.type_name}) improves "
                 f"({improve} on {view.terrain or 'unknown terrain'})")
            continue
        if "fortify" in verbs and view.activity_name != "fortified":
            emit(by_verb["fortify"], "unit_default:fortify",
                 f"unit {uid} ({view.type_name}) fortifies")
            continue
        # No legal action for this unit: hold.
    return decisions


def decide_units(snapshot: Snapshot, table: CandidateTable) -> list[Decision]:
    """Unit defaults (§7.4)."""
    return _unit_decisions(snapshot, table)


def decide_diplomacy(snapshot: Snapshot,
                     table: CandidateTable) -> list[Decision]:
    """Diplomacy default (§7.4): decline/stop all negotiations, no proactive
    offers — the decline path is stop_negotiation. accept_treaty is never
    emitted (that would be accepting, not declining); cancel_treaty is not a
    negotiation and is left alone."""
    decisions: list[Decision] = []
    for cand in table.candidates:
        if cand.domain == "dipl" and cand.verb == "stop_negotiation" \
                and is_exposed_verb(cand.verb):
            decisions.append(Decision(
                domain="dipl", candidate_id=cand.id, verb=cand.verb,
                actor_id=cand.actor_id,
                reason="diplomacy_default:stop_negotiation",
                detail=(f"decline/stop the negotiation with player "
                        f"{cand.actor_id}; no proactive offers")))
    return decisions


# ---------------------------------------------------------------------------
# The comparator entry point
# ---------------------------------------------------------------------------


def decide(snapshot: Snapshot, table: CandidateTable) -> DecisionLog:
    """The comparator entry point (§7.4): the active scripted policy.

    Deterministic: same snapshot + table -> same decision log. Emits at most
    one action per unit, one founding per turn, one production switch per
    city, and buys under the simulated-gold rule.
    """
    decisions: list[Decision] = []
    decisions += decide_research(snapshot, table)
    decisions += decide_production(snapshot, table)
    decisions += decide_units(snapshot, table)
    decisions += decide_diplomacy(snapshot, table)
    return DecisionLog(policy="comparator", turn=snapshot.turn,
                       decisions=tuple(decisions))
