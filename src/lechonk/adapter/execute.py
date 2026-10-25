# SPDX-License-Identifier: GPL-3.0-or-later
"""lechonk.adapter.execute — candidate -> CivRealm action tuple + read-back.

GPL-3.0-or-later (design-plan-freeciv-v1.1.md decision 7; same package as the
civrealm-importing session module). This module never imports civrealm: the
read-back verifiers are pure functions over the plain observation/info dicts,
so they are unit-testable offline (tests/test_execute.py).

Per plan §7.1 (execute.py module contract):
- candidate_to_action maps a core.schemas.Candidate to the CivRealm action
  tuple (ctrl_type, actor_id, action_key) through the candidate PAYLOAD —
  never by re-parsing the opaque candidate ID.
- execute_candidate runs ONE action through the session and verifies it with
  a per-action read-back against the refreshed observation (§3: sequential
  execute with per-action read-back verification). Single-action scope:
  batch revalidation/grouping is the WP2 runner's job; the runner calls
  execute_candidate per action with refreshed state (session.step updates
  the cached obs/info).
- build_city is a two-round-trip action (name-suggestion wait, then the
  found-city packet; unit_actions.py:935-957 with the pid-44 handler at
  unit_ctrl.py:678-696 completing the dance inside one env.step) and gets a
  special-cased read-back: founding is verified by a NEW own city at the
  unit's tile with the unit consumed. A mid-dance disconnect leaves founding
  ambiguous -> INCONCLUSIVE -> WP2 full-state reconciliation (never blind
  replay, §3).
- finish_phase is the pipeline-owned finish primitive (step(None) ends the
  phase; §3: finish phase has exactly one owner — the pipeline; the model may
  request it, the pipeline sends it). WP2's runner calls it; this module
  only exposes it safely (WAITING deadline).
- 'pass' is a documented no-op (civ_controller.py:385-386 sends a debug
  message; no game-state change).

Read-back statuses: VERIFIED (post-state matches the intended effect),
CONTRADICTED (post-state proves non-execution), INCONCLUSIVE (cannot tell —
the WP2 reconciliation path owns these; they are never silently retried).
One upstream quirk: a sentry read-back can never be CONTRADICTED — the read
path cancels ACTIVITY_SENTRY within the same step (unit_actions.py:183-226),
so an unchanged activity is not evidence of non-execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from lechonk.adapter.obs_extract import DIR8_DELTAS
from lechonk.core.schemas import Candidate, is_candidate_id

# Upstream activity ids (version/v14.py:50-65) for activity read-backs.
ACTIVITY_IDLE = 0
ACTIVITY_MINE = 2
ACTIVITY_IRRIGATE = 3
ACTIVITY_FORTIFIED = 4
ACTIVITY_SENTRY = 5
ACTIVITY_GEN_ROAD = 13
# freeciv's in-turn fortify state: a unit ordered to fortify shows
# ACTIVITY_FORTIFYING within the same turn and completes to
# ACTIVITY_FORTIFIED at the next turn start (live evidence: the WP2 model
# run — units fortified with partial moves left show 10 in the
# post-observation).
ACTIVITY_FORTIFYING = 10

# The documented no-op action (civ_controller.py:385-386).
PASS_ACTION = "pass"

VERIFIED = "verified"
CONTRADICTED = "contradicted"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Readback:
    """Per-action read-back verification result."""

    status: str  # VERIFIED | CONTRADICTED | INCONCLUSIVE
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """One executed candidate: the action tuple, the step outcome, the read-back."""

    candidate_id: str
    action: tuple[str, Any, str]
    status: str  # "executed" (the step ran; readback carries verification)
    readback: Readback
    terminated: bool
    truncated: bool
    turn_before: int
    turn_after: int
    elapsed_s: float
    detail: str


# ---------------------------------------------------------------------------
# candidate -> action tuple
# ---------------------------------------------------------------------------


def candidate_to_action(candidate: Candidate) -> tuple[str, Any, str]:
    """Map a candidate to its CivRealm action tuple via the payload."""
    if not is_candidate_id(candidate.id):
        raise ValueError(f"candidate id {candidate.id!r} is not a valid "
                         f"candidate reference")
    payload = candidate.payload or {}
    missing = [k for k in ("ctrl_type", "actor_id", "action_key") if k not in payload]
    if missing:
        raise ValueError(
            f"candidate {candidate.id!r} payload is missing {missing}; cannot "
            f"map it to an action tuple")
    ctrl_type = payload["ctrl_type"]
    actor_id = payload["actor_id"]
    action_key = payload["action_key"]
    if not isinstance(action_key, str) or not action_key:
        raise ValueError(f"candidate {candidate.id!r} has an empty action_key")
    return (ctrl_type, actor_id, action_key)


# ---------------------------------------------------------------------------
# read-back verification (pure functions over observation/info dicts)
# ---------------------------------------------------------------------------


def _units(obs: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for k, v in (obs.get("unit") or {}).items():
        try:
            uid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, Mapping):
            out[uid] = v
    return out


def _cities(obs: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for k, v in (obs.get("city") or {}).items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, Mapping):
            out[cid] = v
    return out


def _players(obs: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for k, v in (obs.get("player") or {}).items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, Mapping):
            out[pid] = v
    return out


def _dipl(obs: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for k, v in (obs.get("dipl") or {}).items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, Mapping):
            out[pid] = v
    return out


def _pos(unit: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    x, y = unit.get("x"), unit.get("y")
    if isinstance(x, int) and isinstance(y, int) and not isinstance(x, bool) \
            and not isinstance(y, bool):
        return (x, y)
    return None


def _rb(status: str, detail: str, **evidence: Any) -> Readback:
    return Readback(status, detail, evidence)


def verify_readback(
    candidate: Candidate,
    pre_obs: Mapping[str, Any],
    post_obs: Mapping[str, Any],
    *,
    my_player_id: int,
    pre_info: Optional[Mapping[str, Any]] = None,
    post_info: Optional[Mapping[str, Any]] = None,
    production_names: Optional[Mapping[str, tuple[int, int]]] = None,
) -> Readback:
    """Verify one executed candidate against the pre/post observations."""
    payload = candidate.payload or {}
    if candidate.domain == "unit":
        return _verify_unit(candidate, payload, pre_obs, post_obs,
                            my_player_id=my_player_id, post_info=post_info)
    if candidate.domain == "city":
        return _verify_city(candidate, payload, pre_obs, post_obs,
                            production_names=production_names,
                            post_info=post_info)
    if candidate.domain == "tech":
        return _verify_tech(candidate, payload, pre_obs, post_obs)
    if candidate.domain == "dipl":
        return _verify_dipl(candidate, payload, pre_obs, post_obs)
    return _rb(INCONCLUSIVE, f"no verifier for domain {candidate.domain!r}")


# --- unit verifiers ----------------------------------------------------------


def _verify_unit(candidate: Candidate, payload: Mapping[str, Any],
                 pre_obs: Mapping[str, Any], post_obs: Mapping[str, Any],
                 *, my_player_id: int,
                 post_info: Optional[Mapping[str, Any]]) -> Readback:
    verb = candidate.verb
    uid = payload.get("actor_id")
    uid = int(uid) if isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit()) else None
    pre_units = _units(pre_obs)
    post_units = _units(post_obs)
    pre_u = pre_units.get(uid) if uid is not None else None
    post_u = post_units.get(uid) if uid is not None else None
    if uid is None or pre_u is None:
        return _rb(INCONCLUSIVE, f"unit actor {payload.get('actor_id')!r} not "
                                 f"in the pre-observation")
    pre_pos = _pos(pre_u)

    if verb.startswith("goto_"):
        dir8 = payload.get("dir8")
        if not isinstance(dir8, int) or pre_pos is None:
            return _rb(INCONCLUSIVE, "goto without dir8 or pre-position")
        dx, dy = DIR8_DELTAS[dir8]
        expected = (pre_pos[0] + dx, pre_pos[1] + dy)
        if post_u is None:
            return _rb(INCONCLUSIVE, "own unit missing after move (own units "
                                     "are always visible)")
        post_pos = _pos(post_u)
        if post_pos == expected:
            return _rb(VERIFIED, f"unit {uid} moved {pre_pos} -> {post_pos}",
                       expected=list(expected))
        if post_pos == pre_pos:
            return _rb(CONTRADICTED, f"unit {uid} did not move "
                                     f"({pre_pos} -> {post_pos})")
        return _rb(INCONCLUSIVE, f"unit {uid} at unexpected tile {post_pos}")

    if verb.startswith("attack_"):
        dir8 = payload.get("dir8")
        if not isinstance(dir8, int) or pre_pos is None:
            return _rb(INCONCLUSIVE, "attack without dir8 or pre-position")
        dx, dy = DIR8_DELTAS[dir8]
        target_tile = (pre_pos[0] + dx, pre_pos[1] + dy)
        targets = [tid for tid, tu in pre_units.items()
                   if _pos(tu) == target_tile and tu.get("owner") != my_player_id]
        if not targets:
            return _rb(INCONCLUSIVE, f"no pre-observed enemy on the attacked "
                                     f"tile {target_tile}")
        moved = post_u is not None and _pos(post_u) == target_tile
        attacker_gone = post_u is None
        targets_gone = all(tid not in post_units for tid in targets)
        if moved:
            return _rb(VERIFIED, f"unit {uid} advanced into {target_tile} "
                                 f"(attacker won)", targets=targets)
        if attacker_gone:
            return _rb(VERIFIED, f"unit {uid} lost the attack (unit consumed)",
                       targets=targets)
        if targets_gone:
            return _rb(VERIFIED, f"target(s) {targets} destroyed",
                       targets=targets)
        if post_u is not None and _pos(post_u) == pre_pos:
            return _rb(CONTRADICTED, f"unit {uid} still at {pre_pos} with "
                                     f"target(s) {targets} present")
        return _rb(INCONCLUSIVE, "attack outcome not observable")

    if verb == "build_city":
        # Special-cased two-round-trip read-back (module docstring).
        if pre_pos is None:
            return _rb(INCONCLUSIVE, "build_city without pre-position")
        pre_cities = _cities(pre_obs)
        post_cities = _cities(post_obs)
        pre_own = {cid for cid, c in pre_cities.items()
                   if c.get("owner") == my_player_id}
        new_at_tile = [cid for cid, c in post_cities.items()
                       if cid not in pre_own
                       and c.get("owner") == my_player_id
                       and _pos(c) == pre_pos]
        if post_u is None and new_at_tile:
            return _rb(VERIFIED,
                       f"city founded: unit {uid} consumed, new own city "
                       f"{new_at_tile} at {pre_pos}",
                       new_cities=new_at_tile, tile=list(pre_pos))
        if post_u is None and not new_at_tile:
            return _rb(INCONCLUSIVE,
                       f"unit {uid} consumed but no new own city at {pre_pos} "
                       f"observed — uncertain-mutation window; reconcile on "
                       f"full state before any resubmission (never blind replay)")
        if post_u is not None and not new_at_tile:
            return _rb(CONTRADICTED, f"unit {uid} still present at {pre_pos}, "
                                     f"no new city")
        return _rb(INCONCLUSIVE, f"unit {uid} present and new city "
                                 f"{new_at_tile} — unexpected combination")

    if verb in ("fortify", "sentry", "build_road", "irrigation", "mine"):
        expected = {
            "fortify": ACTIVITY_FORTIFIED,
            "sentry": ACTIVITY_SENTRY,
            "build_road": ACTIVITY_GEN_ROAD,
            "irrigation": ACTIVITY_IRRIGATE,
            "mine": ACTIVITY_MINE,
        }[verb]
        pre_activity = pre_u.get("activity")
        post_activity = post_u.get("activity") if post_u is not None else None
        if post_u is None:
            return _rb(INCONCLUSIVE, f"unit {uid} missing after {verb}")
        if post_activity == expected:
            return _rb(VERIFIED, f"unit {uid} activity {pre_activity} -> "
                                 f"{post_activity} ({verb})",
                       activity=post_activity)
        if verb == "fortify" and post_activity == ACTIVITY_FORTIFYING:
            # freeciv's in-turn fortify state (ACTIVITY_FORTIFYING): the
            # order took effect and completes to ACTIVITY_FORTIFIED at the
            # next turn start. An already-fortifying unit re-fortified stays
            # fortifying — unchanged is not attributable to this action
            # (the same cannot-tell semantics as the sentry quirk below).
            if pre_activity == ACTIVITY_FORTIFYING:
                return _rb(INCONCLUSIVE,
                           f"unit {uid} already fortifying ({pre_activity}) "
                           f"before fortify — unchanged activity is not "
                           f"attributable to this action")
            return _rb(VERIFIED, f"unit {uid} activity {pre_activity} -> "
                                 f"{post_activity} (fortify order in effect; "
                                 f"completes to fortified at the next turn "
                                 f"start)", activity=post_activity)
        if post_activity == pre_activity:
            if verb == "sentry":
                # Upstream quirk (unit_actions.py:183-226): the read path's
                # cancel_orders_before_query_pro cancels the unit's activity
                # before querying action probabilities, and restore_activity
                # deliberately never restores ACTIVITY_SENTRY — a sentry is
                # always undone within the same step, so the post-observation
                # shows the pre-sentry activity. Unchanged activity is not
                # evidence of non-execution: never CONTRADICTED for sentry.
                return _rb(INCONCLUSIVE,
                           f"unit {uid} activity unchanged ({pre_activity}) "
                           f"after sentry — upstream read path cancels "
                           f"ACTIVITY_SENTRY in the same step "
                           f"(unit_actions.py:183-226); sent-then-cancelled is "
                           f"indistinguishable from never-sent")
            return _rb(CONTRADICTED, f"unit {uid} activity unchanged "
                                     f"({pre_activity}) after {verb}")
        return _rb(INCONCLUSIVE, f"unit {uid} activity {post_activity} "
                                 f"after {verb} (expected {expected})")

    if verb == "join_city":
        if post_u is None:
            return _rb(VERIFIED, f"unit {uid} joined the city (consumed)")
        return _rb(CONTRADICTED, f"unit {uid} still present after join_city")

    if verb == "set_homecity":
        pre_home = pre_u.get("home_city")
        post_home = post_u.get("home_city") if post_u is not None else None
        if post_u is None:
            return _rb(INCONCLUSIVE, f"unit {uid} missing after set_homecity")
        if isinstance(post_home, int) and post_home != pre_home:
            return _rb(VERIFIED, f"unit {uid} home_city {pre_home} -> "
                                 f"{post_home}", home_city=post_home)
        if post_home == pre_home:
            return _rb(CONTRADICTED, f"unit {uid} home_city unchanged "
                                     f"({pre_home})")
        return _rb(INCONCLUSIVE, "home_city not observable")

    if verb == "cancel_order":
        pre_activity = pre_u.get("activity")
        post_activity = post_u.get("activity") if post_u is not None else None
        if post_u is None:
            return _rb(INCONCLUSIVE, f"unit {uid} missing after cancel_order")
        if isinstance(pre_activity, int) and pre_activity != ACTIVITY_IDLE \
                and post_activity == ACTIVITY_IDLE:
            return _rb(VERIFIED, f"unit {uid} activity {pre_activity} -> 0 "
                                 f"(orders cancelled)")
        return _rb(INCONCLUSIVE, f"cancel_order not observable "
                                 f"(activity {pre_activity} -> {post_activity})")

    if verb == "keep_activity":
        # keep_activity is not in the observation; verify via the refreshed
        # available_actions: the unit must no longer be actable.
        acts = ((post_info or {}).get("available_actions") or {}).get("unit") or {}
        unit_acts = acts.get(uid) or {}
        if post_info is None:
            return _rb(INCONCLUSIVE, "keep_activity needs the refreshed "
                                     "available_actions")
        if not unit_acts or not any(unit_acts.values()):
            return _rb(VERIFIED, f"unit {uid} no longer actable (hold)")
        return _rb(CONTRADICTED, f"unit {uid} still actable after keep_activity")

    return _rb(INCONCLUSIVE, f"no verifier for unit verb {verb!r}")


# --- city verifiers ----------------------------------------------------------


def _verify_city(candidate: Candidate, payload: Mapping[str, Any],
                 pre_obs: Mapping[str, Any], post_obs: Mapping[str, Any],
                 *, production_names: Optional[Mapping[str, tuple[int, int]]],
                 post_info: Optional[Mapping[str, Any]]) -> Readback:
    verb = candidate.verb
    cid = payload.get("actor_id")
    cid = int(cid) if isinstance(cid, int) or (isinstance(cid, str) and cid.isdigit()) else None
    pre_c = _cities(pre_obs).get(cid) if cid is not None else None
    post_c = _cities(post_obs).get(cid) if cid is not None else None
    if cid is None or pre_c is None:
        return _rb(INCONCLUSIVE, f"city actor {payload.get('actor_id')!r} not "
                                 f"in the pre-observation")

    if verb.startswith("produce_"):
        name = payload.get("production")
        if post_c is None:
            return _rb(INCONCLUSIVE, f"city {cid} missing after produce")
        post_kind = post_c.get("production_kind")
        post_value = post_c.get("production_value")
        if production_names and name in production_names:
            kind, value = production_names[name]
            if (post_kind, post_value) == (kind, value):
                return _rb(VERIFIED, f"city {cid} production -> {name}",
                           production_kind=post_kind,
                           production_value=post_value)
            pre_kind = pre_c.get("production_kind")
            pre_value = pre_c.get("production_value")
            if (post_kind, post_value) == (pre_kind, pre_value):
                return _rb(CONTRADICTED, f"city {cid} production unchanged "
                                         f"({pre_kind},{pre_value})")
            return _rb(INCONCLUSIVE, f"city {cid} production "
                                     f"({post_kind},{post_value}) != {name} "
                                     f"({kind},{value})")
        # No ruleset mapping: fall back to "production changed".
        pre_kind = pre_c.get("production_kind")
        pre_value = pre_c.get("production_value")
        if (post_kind, post_value) != (pre_kind, pre_value):
            return _rb(VERIFIED, f"city {cid} production changed "
                                 f"({pre_kind},{pre_value}) -> "
                                 f"({post_kind},{post_value})")
        return _rb(CONTRADICTED, f"city {cid} production unchanged")

    if verb == "city_buy_production":
        if post_c is None:
            return _rb(INCONCLUSIVE, f"city {cid} missing after buy")
        pre_stock = pre_c.get("shield_stock")
        post_stock = post_c.get("shield_stock")
        if isinstance(pre_stock, int) and isinstance(post_stock, int):
            if post_stock > pre_stock:
                return _rb(VERIFIED, f"city {cid} shield_stock {pre_stock} -> "
                                     f"{post_stock} (bought)",
                           shield_stock=post_stock)
            return _rb(CONTRADICTED, f"city {cid} shield_stock unchanged "
                                     f"({pre_stock}) after buy")
        return _rb(INCONCLUSIVE, "shield_stock not observable")

    if verb.startswith("city_work_") or verb.startswith("city_unwork_"):
        # Worked-tile state is not in the observation; verify via the
        # refreshed available_actions: the exact action key must no longer
        # be offered (the tile is now worked / no longer worked).
        if post_info is None:
            return _rb(INCONCLUSIVE, f"{verb} needs the refreshed "
                                     f"available_actions")
        acts = ((post_info.get("available_actions") or {}).get("city")
                or {}).get(cid) or {}
        still_valid = bool(acts.get(payload.get("action_key")))
        if not still_valid:
            return _rb(VERIFIED, f"city {cid} tile work state changed "
                                 f"({verb}); action no longer offered")
        return _rb(CONTRADICTED, f"city {cid} {verb} action still offered")

    return _rb(INCONCLUSIVE, f"no verifier for city verb {verb!r}")


# --- tech verifiers ----------------------------------------------------------


def _verify_tech(candidate: Candidate, payload: Mapping[str, Any],
                 pre_obs: Mapping[str, Any],
                 post_obs: Mapping[str, Any]) -> Readback:
    verb = candidate.verb
    tech_id = payload.get("tech_id")
    # The tech actor is 'cur_player' (our own player); find it via the
    # own-player marker fields (player_state.py).
    def my(obs):
        for pid, p in _players(obs).items():
            if any(m in p for m in ("tech_goal", "bulbs_researched", "embassy_txt")):
                return p
        return None
    post_p = my(post_obs)
    if post_p is None:
        return _rb(INCONCLUSIVE, "own player not found in the post-observation")
    if verb.startswith("research_tech_"):
        post_researching = post_p.get("researching")
        if post_researching == tech_id:
            return _rb(VERIFIED, f"researching -> {tech_id} "
                                 f"({payload.get('tech_name')})",
                       researching=post_researching)
        if isinstance(post_researching, int):
            return _rb(CONTRADICTED, f"researching is {post_researching}, "
                                     f"expected {tech_id}")
        return _rb(INCONCLUSIVE, f"researching not observable "
                                 f"({post_researching!r})")
    if verb.startswith("set_tech_goal_"):
        post_goal = post_p.get("tech_goal")
        if post_goal == tech_id:
            return _rb(VERIFIED, f"tech_goal -> {tech_id} "
                                 f"({payload.get('tech_name')})",
                       tech_goal=post_goal)
        if isinstance(post_goal, int):
            return _rb(CONTRADICTED, f"tech_goal is {post_goal}, "
                                     f"expected {tech_id}")
        return _rb(INCONCLUSIVE, f"tech_goal not observable ({post_goal!r})")
    return _rb(INCONCLUSIVE, f"no verifier for tech verb {verb!r}")


# --- diplomacy verifiers -----------------------------------------------------


def _verify_dipl(candidate: Candidate, payload: Mapping[str, Any],
                 pre_obs: Mapping[str, Any],
                 post_obs: Mapping[str, Any]) -> Readback:
    verb = candidate.verb
    p = payload.get("counterpart")
    p = int(p) if isinstance(p, int) or (isinstance(p, str) and str(p).isdigit()) else None
    pre_d = _dipl(pre_obs).get(p) if p is not None else None
    post_d = _dipl(post_obs).get(p) if p is not None else None
    if p is None or pre_d is None:
        return _rb(INCONCLUSIVE, f"diplomacy counterpart "
                                 f"{payload.get('counterpart')!r} not in the "
                                 f"pre-observation")

    if verb == "stop_negotiation":
        if post_d is None:
            return _rb(INCONCLUSIVE, f"counterpart {p} missing after "
                                     f"stop_negotiation")
        pre_clauses = pre_d.get("diplomacy_clause_map") or []
        post_clauses = post_d.get("diplomacy_clause_map") or []
        if not pre_clauses:
            return _rb(INCONCLUSIVE, f"no meeting with {p} was active to stop")
        if not post_clauses:
            return _rb(VERIFIED, f"meeting with {p} stopped "
                                 f"(clause map emptied)")
        return _rb(CONTRADICTED, f"clause map with {p} still non-empty "
                                 f"({post_clauses})")

    if verb in ("cancel_treaty", "accept_treaty"):
        if post_d is None:
            return _rb(INCONCLUSIVE, f"counterpart {p} missing after {verb}")
        pre_state = pre_d.get("diplomatic_state")
        post_state = post_d.get("diplomatic_state")
        pre_clauses = pre_d.get("diplomacy_clause_map") or []
        post_clauses = post_d.get("diplomacy_clause_map") or []
        if post_state != pre_state or post_clauses != pre_clauses:
            return _rb(VERIFIED, f"{verb} with {p}: state {pre_state} -> "
                                 f"{post_state}, clauses changed")
        return _rb(CONTRADICTED, f"{verb} with {p} left state {pre_state} "
                                 f"and clauses unchanged")

    return _rb(INCONCLUSIVE, f"no verifier for diplomacy verb {verb!r}")


# ---------------------------------------------------------------------------
# session-facing execution primitives
# ---------------------------------------------------------------------------


def execute_candidate(session: Any, candidate: Candidate,
                      *, deadline: Optional[float] = None) -> ExecutionResult:
    """Execute ONE candidate through the session with read-back verification.

    The session's cached obs/info are the pre-state (they refresh on every
    step); the runner calls this per action with refreshed state. Single-
    action scope: batch revalidation/grouping is the WP2 runner's job.
    """
    action = candidate_to_action(candidate)
    pre_obs = session.obs
    pre_info = session.info
    turn_before = int((pre_info or {}).get("turn", 0) or 0)
    t0 = time.monotonic()
    outcome = session.step(
        action,
        deadline=deadline if deadline is not None else session.step_deadline)
    elapsed = time.monotonic() - t0
    post_obs = outcome.observation
    post_info = outcome.info
    readback = verify_readback(
        candidate, pre_obs, post_obs,
        my_player_id=session.player_id,
        pre_info=pre_info, post_info=post_info,
        production_names=session.production_name_map())
    turn_after = int((post_info or {}).get("turn", turn_before) or turn_before)
    return ExecutionResult(
        candidate_id=candidate.id,
        action=action,
        status="executed",
        readback=readback,
        terminated=bool(outcome.terminated),
        truncated=bool(outcome.truncated),
        turn_before=turn_before,
        turn_after=turn_after,
        elapsed_s=elapsed,
        detail=f"{candidate.id}: read-back {readback.status} — "
               f"{readback.detail}",
    )


def finish_phase(session: Any, *, deadline: Optional[float] = None) -> Any:
    """Pipeline-owned finish primitive: step(None) ends the phase.

    §3: finish phase has exactly one owner — the pipeline (the model may
    request it; the pipeline sends it). Blocks through WAITING until the next
    own phase, under the WAITING deadline (<= 720 s + margin).
    """
    return session.step(
        None,
        deadline=deadline if deadline is not None else session.wait_deadline)


def execute_pass(session: Any, *, deadline: Optional[float] = None) -> Any:
    """Documented no-op: 'pass' (civ_controller.py:385-386) sends a debug
    message and changes no game state. Exposed for runner completeness; the
    model never needs it."""
    return session.step(
        PASS_ACTION,
        deadline=deadline if deadline is not None else session.step_deadline)
