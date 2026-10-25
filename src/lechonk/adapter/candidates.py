# SPDX-License-Identifier: GPL-3.0-or-later
"""lechonk.adapter.candidates — legal-candidate table from available_actions.

GPL-3.0-or-later (design-plan-freeciv-v1.1.md decision 7; same package as the
civrealm-importing session module). This module itself never imports civrealm:
its inputs are the plain info/observation dicts, so it stays unit-testable
offline (tests/test_candidates.py drives it with synthetic structures).

Builds core.schemas.CandidateTable per plan §7.2 (action surface inventory,
lines 142-149) from ``info['available_actions']`` (turn_manager.get_info();
structure ``{ctrl_type: {actor_id: {action_key: valid}}}``).

Exposed surface (a WHITELIST — everything else in available_actions is NOT
exposed, including chat, set_sci_lux_tax, change_gov, spy/trade/transport
verbs, sell verbs, ActAutoSettler ('autosettlers'), ActExplore ('explore'),
clause construction, specialists, city_keep_production):
- unit: goto_<dir> (8), attack_<dir> (8), build_city, fortify, sentry
  (ActSEntry — PR #7 re-enabled, docs/gate-record.md entry 5), build_road,
  irrigation, mine, join_city, set_homecity, cancel_order (housekeeping),
  keep_activity (hold);
- city: produce_<name> (units and improvements), city_buy_production,
  city_work_<i>_<dx>_<dy> / city_unwork_...;
- tech: research_tech_<name>_<id>, set_tech_goal_<name>_<id> (actor
  'cur_player');
- diplomacy: stop_negotiation_<p>, cancel_treaty_<p>, accept_treaty_<p>
  (actor = counterpart p).

Unit gating (plan §7.1, unit_actions.py:324-331): movesleft > 0, not
done_moving, ssa_controller == SSA_NONE, not keep_activity. moves_left and
ssa_controller come from the unit observation; done_moving/keep_activity are
NOT in the observation — the session supplies them via ``unit_flags`` (raw
punit fields). Upstream already omits non-actable actors from
available_actions (base_action.py get_action_info); our gating is
defense-in-depth on top of it.

Candidate IDs are opaque ``domain:actor:verb`` tokens matching
core.schemas.CANDIDATE_ID_RE. Verbs are derived from the raw CivRealm action
keys: dir indices become compass names (goto_4 -> goto_e), names are
sanitized to [a-z0-9_] (produce_City Walls -> produce_city_walls; a negative
tile offset '-' becomes 'm'), and the diplomacy player suffix folds into the
actor (stop_negotiation_2 with actor 2 -> dipl:2:stop_negotiation). The RAW
action key always travels in payload['action_key'] — execute.py maps a
candidate to its (ctrl_type, actor_id, action_key) tuple through the payload,
never by re-parsing the ID.

Combat odds (plan §7.2 line 148): attack candidates surface unit_data
action_prob percent min/max ranges keyed by freeciv DIR8 indices (NW=0..SE=7)
as payload odds_min/odds_max fractions — the package-A extractor pattern
(unit_ctrl.prop_actions.unit_data[uid].action_prob[dir8][ACTION_ATTACK]).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional

from lechonk.adapter.obs_extract import (
    ACTION_ATTACK,
    DIR8_NAMES,
    SSA_NONE,
)
from lechonk.core.schemas import CANDIDATE_ID_RE, Candidate, CandidateTable

# --- exposed verb surface (plan §7.2 lines 142-149) --------------------------

# Unit verbs without a direction suffix (unit_actions.py registration list).
UNIT_FIXED_VERBS = frozenset({
    "build_city", "fortify", "sentry", "build_road", "irrigation", "mine",
    "join_city", "set_homecity", "cancel_order", "keep_activity",
})
# Directional unit verbs: raw keys are '<prefix><dir8 index 0..7>'.
UNIT_DIR_PREFIXES = ("goto_", "attack_")
# City verbs.
CITY_BUY_VERB = "city_buy_production"
CITY_PRODUCE_PREFIX = "produce_"
CITY_WORK_PREFIXES = ("city_work_", "city_unwork_")
# Tech verbs (actor 'cur_player'): '<prefix><Name>_<tech id>'.
TECH_PREFIXES = ("research_tech_", "set_tech_goal_")
# Diplomacy verbs (actor = counterpart player id): '<verb>_<p>'.
DIPL_VERBS = ("stop_negotiation", "cancel_treaty", "accept_treaty")

# ctrl_type -> candidate domain (everything else is not exposed).
CTRL_DOMAIN = {"unit": "unit", "city": "city", "tech": "tech", "dipl": "dipl"}
# domain -> ctrl_type (inverse of CTRL_DOMAIN).
CTRL_DOMAIN_REVERSE = {v: k for k, v in CTRL_DOMAIN.items()}
# Deterministic domain ordering for the table.
_DOMAIN_ORDER = ("unit", "city", "tech", "dipl")

_DIR8_KEY_RE = re.compile(r"^(goto|attack)_([0-7])\Z")
_TECH_KEY_RE = re.compile(r"^(research_tech|set_tech_goal)_(.+)_(\d+)\Z")
_CITY_WORK_RE = re.compile(r"^city_(un)?work_(.+)\Z")
_DIPL_KEY_RE = re.compile(r"^(stop_negotiation|cancel_treaty|accept_treaty)_(\d+)\Z")


def _sanitize(text: str) -> str:
    """Verb-safe sanitization: [a-z0-9_] only.

    Lowercase; '-' becomes 'm' (negative tile offsets, e.g. city_work_2_-1_1
    -> city_work_2_m1_1); any other invalid run collapses to '_'. A
    table-level collision backstop (below) keeps IDs unique even if two raw
    keys sanitize identically.
    """
    text = text.lower().replace("-", "m")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return text.strip("_")


def _unique_id(domain: str, actor: str, verb: str, raw_key: str,
               used: set[str]) -> str:
    """Candidate ID with a deterministic collision backstop."""
    cid = f"{domain}:{actor}:{verb}"
    if cid not in used:
        return cid
    suffix = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:6]
    cid = f"{domain}:{actor}:{verb}_x{suffix}"
    if cid in used:  # pragma: no cover - sha1-prefix collision is not realistic
        raise ValueError(f"candidate id collision on {cid!r}")
    return cid


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unit_actor_gated(units: Mapping[int, Mapping[str, Any]], uid: int,
                      my_player_id: int,
                      unit_flags: Optional[Mapping[int, Mapping[str, Any]]]) -> bool:
    """Plan §7.1 unit gating (unit_actions.py:324-331), defense-in-depth.

    Own unit only; moves_left > 0 and ssa_controller == SSA_NONE from the
    observation; done_moving / keep_activity from the session-supplied
    unit_flags (raw punit fields, absent from the observation).
    """
    u = units.get(uid)
    if not isinstance(u, Mapping):
        return False
    if u.get("owner") != my_player_id:
        return False
    moves_left = u.get("moves_left")
    if isinstance(moves_left, bool) or not isinstance(moves_left, int):
        return False  # own units always report moves_left; missing = unusable
    if moves_left <= 0:
        return False
    ssa = u.get("ssa_controller")
    if isinstance(ssa, bool) or not isinstance(ssa, int) or ssa != SSA_NONE:
        return False
    flags = (unit_flags or {}).get(uid) or {}
    if flags.get("done_moving"):
        return False
    if flags.get("keep_activity"):
        return False
    return True


def _attack_odds(action_probs: Optional[Mapping[int, Mapping[int, Mapping[int, Mapping[str, float]]]]],
                 uid: int, dir8: int) -> tuple[Optional[float], Optional[float]]:
    """(odds_min, odds_max) fractions from percent min/max ranges, or (None, None)."""
    if not action_probs:
        return None, None
    probs = action_probs.get(uid, {}).get(dir8, {})
    ap = probs.get(ACTION_ATTACK) if probs else None
    if not isinstance(ap, Mapping) or "max" not in ap:
        return None, None
    try:
        odds_max = round(float(ap["max"]) / 100.0, 3)
        odds_min = round(float(ap.get("min", ap["max"])) / 100.0, 3)
    except (TypeError, ValueError):
        return None, None
    return odds_min, odds_max


def build_candidates(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    my_player_id: int,
    unit_flags: Optional[Mapping[int, Mapping[str, Any]]] = None,
    action_probs: Optional[Mapping[int, Mapping[int, Mapping[int, Mapping[str, float]]]]] = None,
    turn: Optional[int] = None,
) -> CandidateTable:
    """Build the legal-candidate table from info['available_actions'].

    Only actions whose available_actions entry is truthy (upstream
    is_action_valid) AND whose verb is on the exposed whitelist are included;
    unit actors must additionally pass the §7.1 gating. Deterministic:
    domains in fixed order, actors and action keys sorted.
    """
    observation = observation or {}
    info = info or {}
    available = info.get("available_actions") or {}
    if turn is None:
        turn = int(info.get("turn", 0) or 0)

    units = {_int(k): v for k, v in (observation.get("unit") or {}).items()
             if _int(k) is not None}
    unit_types = {uid: str(u.get("type_rule_name") or "unit")
                  for uid, u in units.items() if isinstance(u, Mapping)}

    candidates: list[Candidate] = []
    used_ids: set[str] = set()

    def add(domain: str, actor: str, verb: str, raw_key: str, summary: str,
            payload: dict[str, Any]) -> None:
        cid = _unique_id(domain, actor, verb, raw_key, used_ids)
        used_ids.add(cid)
        if CANDIDATE_ID_RE.match(cid) is None:
            raise ValueError(
                f"candidate id {cid!r} does not match CANDIDATE_ID_RE "
                f"(raw action key {raw_key!r})")
        if "actor_id" not in payload:
            raise ValueError(
                f"candidate {cid!r} payload is missing 'actor_id'; cannot "
                f"map it to an action tuple (raw action key {raw_key!r})")
        full_payload = {"ctrl_type": CTRL_DOMAIN_REVERSE[domain],
                        "actor_id": payload["actor_id"],
                        "action_key": raw_key}
        full_payload.update(payload)
        # W2-3 construction invariant: this mirrors execute.candidate_to_action's
        # validation set (valid id above, all three mapping keys, a non-empty
        # action_key), so an unmappable candidate cannot be CONSTRUCTED — a
        # mapping failure can never happen at execute time.
        missing = [k for k in ("ctrl_type", "actor_id", "action_key")
                   if k not in full_payload]
        if missing:
            raise ValueError(
                f"candidate {cid!r} payload is missing {missing}; cannot "
                f"map it to an action tuple")
        if not isinstance(full_payload["action_key"], str) \
                or not full_payload["action_key"]:
            raise ValueError(f"candidate {cid!r} has an empty action_key")
        candidates.append(Candidate(
            id=cid, domain=domain, actor_id=actor, verb=verb,
            summary=summary, payload=full_payload))

    for domain in _DOMAIN_ORDER:
        ctrl_type = CTRL_DOMAIN_REVERSE[domain]
        actors = available.get(ctrl_type) or {}
        for actor_key in sorted(actors, key=str):
            action_map = actors[actor_key] or {}
            for raw_key in sorted(action_map):
                if not action_map[raw_key]:
                    continue  # upstream is_action_valid == False
                if domain == "unit":
                    cand = _unit_candidate(
                        raw_key, actor_key, units, unit_types, my_player_id,
                        unit_flags, action_probs)
                elif domain == "city":
                    cand = _city_candidate(raw_key, actor_key)
                elif domain == "tech":
                    cand = _tech_candidate(raw_key, actor_key)
                else:
                    cand = _dipl_candidate(raw_key, actor_key)
                if cand is None:
                    continue  # not on the exposed whitelist
                verb, summary, payload = cand
                add(domain, str(actor_key), verb, raw_key, summary, payload)

    return CandidateTable(turn=turn, candidates=candidates)


def _unit_candidate(
    raw_key: str, actor_key: Any,
    units: Mapping[int, Mapping[str, Any]],
    unit_types: Mapping[int, str],
    my_player_id: int,
    unit_flags: Optional[Mapping[int, Mapping[str, Any]]],
    action_probs: Optional[Mapping[int, Mapping[int, Mapping[int, Mapping[str, float]]]]],
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """(verb, summary, payload) for an exposed unit action, or None."""
    uid = _int(actor_key)
    if uid is None:
        return None
    if not _unit_actor_gated(units, uid, my_player_id, unit_flags):
        return None
    utype = unit_types.get(uid, "unit")

    m = _DIR8_KEY_RE.match(raw_key)
    if m:
        prefix, dir8 = m.group(1), int(m.group(2))
        compass = DIR8_NAMES[dir8].lower()
        verb = f"{prefix}_{compass}"
        payload: dict[str, Any] = {
            "actor_id": uid, "dir8": dir8, "dir": DIR8_NAMES[dir8],
        }
        if prefix == "attack":
            odds_min, odds_max = _attack_odds(action_probs, uid, dir8)
            payload["odds_min"] = odds_min
            payload["odds_max"] = odds_max
            if odds_max is not None:
                summary = (f"unit {uid} ({utype}) attack {compass} "
                           f"(win odds {odds_min:.0%}-{odds_max:.0%})")
            else:
                summary = f"unit {uid} ({utype}) attack {compass} (odds unknown)"
        else:
            summary = f"unit {uid} ({utype}) move {compass}"
        return verb, summary, payload

    if raw_key in UNIT_FIXED_VERBS:
        return raw_key, f"unit {uid} ({utype}) {raw_key}", {"actor_id": uid}
    return None  # not exposed (spy/trade/transport/sell/autosettler/explore/...)


def _city_candidate(
    raw_key: str, actor_key: Any,
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """(verb, summary, payload) for an exposed city action, or None."""
    cid = _int(actor_key)
    if cid is None:
        return None

    if raw_key == CITY_BUY_VERB:
        return (raw_key, f"city {cid} buy production", {"actor_id": cid})

    if raw_key.startswith(CITY_PRODUCE_PREFIX):
        name = raw_key[len(CITY_PRODUCE_PREFIX):]
        verb = CITY_PRODUCE_PREFIX + _sanitize(name)
        return (verb, f"city {cid} produce {name}",
                {"actor_id": cid, "production": name})

    m = _CITY_WORK_RE.match(raw_key)
    if m:
        un = bool(m.group(1))
        tail = m.group(2)
        parts = tail.rsplit("_", 2)
        if len(parts) != 3:
            return None
        idx_s, dx_s, dy_s = parts
        dx, dy = _int(dx_s), _int(dy_s)
        if dx is None or dy is None:
            return None
        output_idx = None if idx_s == "None" else _int(idx_s)
        if output_idx is None and idx_s != "None":
            return None
        verb = _sanitize(raw_key)
        verb_kind = "unwork" if un else "work"
        return (verb, f"city {cid} {verb_kind} tile ({dx},{dy})",
                {"actor_id": cid, "dx": dx, "dy": dy,
                 "output_idx": output_idx})
    return None  # not exposed (sell/specialist/keep_production)


def _tech_candidate(
    raw_key: str, actor_key: Any,
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """(verb, summary, payload) for an exposed tech action, or None."""
    m = _TECH_KEY_RE.match(raw_key)
    if not m:
        return None
    prefix, name, tech_id_s = m.group(1), m.group(2), m.group(3)
    tech_id = _int(tech_id_s)
    if tech_id is None:
        return None
    verb = f"{prefix}_{_sanitize(name)}_{tech_id}"
    if prefix == "research_tech":
        summary = f"research {name} (tech {tech_id})"
    else:
        summary = f"set research goal {name} (tech {tech_id})"
    return (verb, summary,
            {"actor_id": str(actor_key), "tech_id": tech_id,
             "tech_name": name})


def _dipl_candidate(
    raw_key: str, actor_key: Any,
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """(verb, summary, payload) for an exposed diplomacy action, or None."""
    m = _DIPL_KEY_RE.match(raw_key)
    if not m:
        return None
    verb, counterpart_s = m.group(1), m.group(2)
    counterpart = _int(counterpart_s)
    if counterpart is None:
        return None
    return (verb, f"{verb} with player {counterpart}",
            {"actor_id": counterpart, "counterpart": counterpart})
