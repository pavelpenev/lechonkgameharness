# SPDX-License-Identifier: MIT
"""Deterministic release gate for the classic gameplay control surface.

This is intentionally a small, pure check.  It does not judge strategy and it
does not claim that the entire CivRealm action vocabulary is supported.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from lechonk.core.jsonutil import canonical_json
from lechonk.core.schemas import CandidateTable, Snapshot


_FOUNDER_NAMES = {"settler", "settlers", "engineer", "engineers"}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _records(value: Any) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    if not isinstance(value, Mapping):
        return result
    for key, item in value.items():
        ident = _as_int(key)
        if ident is not None and isinstance(item, Mapping):
            result[ident] = item
    return result


def _owner(item: Mapping[str, Any]) -> int | None:
    for key in ("owner", "owner_id", "player", "player_id"):
        value = _as_int(item.get(key))
        if value is not None:
            return value
    return None


def _candidate_hash(table: CandidateTable) -> str:
    data = [candidate.model_dump(mode="json", exclude_none=False)
            for candidate in table.candidates]
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def assess_control_access(
    table: CandidateTable,
    offered: CandidateTable,
    snapshot: Snapshot,
    observation: Mapping[str, Any],
    *,
    my_player_id: int,
) -> dict[str, Any]:
    """Check that the v3 policy did not hide classic critical controls.

    ``table`` is the uncapped, current candidate table and ``offered`` is the
    exact model-facing table.  The raw observation is used to discover actors
    that snapshot truncation may have hidden; snapshot-only checks would miss
    the original failure mode.
    """
    raw_issues: list[str] = []
    if "unit" not in observation or not isinstance(observation.get("unit"), Mapping):
        raw_issues.append("raw_observation_unknown:unit")
    if "city" not in observation or not isinstance(observation.get("city"), Mapping):
        raw_issues.append("raw_observation_unknown:city")
    raw_units = _records(observation.get("unit"))
    raw_cities = _records(observation.get("city"))
    raw = list(table.candidates)
    offered_ids = {candidate.id for candidate in offered.candidates}
    raw_ids = {candidate.id for candidate in raw}

    own_units = {uid: item for uid, item in raw_units.items()
                 if _owner(item) == my_player_id}
    own_cities = {cid: item for cid, item in raw_cities.items()
                  if _owner(item) == my_player_id}
    founders = {
        uid for uid, item in own_units.items()
        if str(item.get("type_rule_name") or item.get("type_name") or "")
        .casefold() in _FOUNDER_NAMES
    }
    build_city_ids = {
        _as_int(candidate.payload.get("actor_id"))
        for candidate in raw
        if candidate.domain == "unit" and candidate.verb == "build_city"
    }
    build_city_ids.discard(None)
    founders |= {uid for uid in build_city_ids if uid is not None}
    contact_ids = {
        candidate.actor_id for candidate in raw
        if candidate.domain == "unit" and candidate.in_contact
    }
    founder_ids = {str(uid) for uid in founders}

    required = [candidate for candidate in raw if (
        (candidate.domain == "city" and candidate.verb.startswith("produce_"))
        or (candidate.domain == "tech" and candidate.verb.startswith("research_tech_"))
        or (candidate.domain == "unit" and (
            candidate.verb == "build_city"
            or candidate.in_contact
            or (candidate.actor_id in founder_ids and candidate.verb.startswith("goto_"))))
    )]
    required_ids = sorted(candidate.id for candidate in required)
    missing_ids = sorted(ident for ident in required_ids if ident not in offered_ids)

    snapshot_units = {str(unit.id): unit for unit in snapshot.units
                      if unit.is_own}
    snapshot_cities = {str(city.id): city for city in snapshot.cities
                       if city.is_own}
    missing_unit_state = sorted({str(uid) for uid in founders | {
        _as_int(actor) for actor in contact_ids if _as_int(actor) is not None
    } if str(uid) not in snapshot_units})
    production_city_ids = {
        str(candidate.actor_id) for candidate in raw
        if candidate.domain == "city" and candidate.verb.startswith("produce_")
    }
    missing_city_state = sorted(str(cid) for cid in own_cities
                                if str(cid) not in snapshot_cities)
    missing_production_state = sorted(cid for cid in production_city_ids
                                      if cid not in snapshot_cities)
    unavailable_production_city_ids = sorted(
        str(cid) for cid in own_cities
        if str(cid) not in production_city_ids)

    # If a city has no raw production candidates, it may simply have no legal
    # production action; that is reported as informational rather than guessed
    # into a failure.  A raw candidate whose actor state is absent is unsafe.
    issues: list[str] = list(raw_issues)
    issues.extend(f"missing_ref:{ident}" for ident in missing_ids)
    issues.extend(f"missing_unit_state:{ident}" for ident in missing_unit_state)
    issues.extend(f"missing_city_state:{ident}" for ident in missing_city_state)
    issues.extend(f"missing_production_state:{ident}" for ident in missing_production_state)

    by_domain: dict[str, int] = {}
    offered_by_domain: dict[str, int] = {}
    for candidate in raw:
        by_domain[candidate.domain] = by_domain.get(candidate.domain, 0) + 1
    for candidate in offered.candidates:
        offered_by_domain[candidate.domain] = offered_by_domain.get(candidate.domain, 0) + 1

    return {
        "schema": "control-access-v1",
        "turn": table.turn,
        "scope": "classic-critical-controls",
        "passed": not issues,
        "issues": sorted(issues),
        "raw_total": len(raw),
        "offered_total": len(offered.candidates),
        "raw_by_domain": by_domain,
        "offered_by_domain": offered_by_domain,
        "required_ref_ids": required_ids,
        "missing_ref_ids": missing_ids,
        "required_unit_ids": sorted(founder_ids | set(contact_ids)),
        "missing_unit_state_ids": missing_unit_state,
        "required_city_ids": sorted(str(cid) for cid in own_cities),
        "missing_city_state_ids": missing_city_state,
        "missing_owned_city_production_ids": missing_production_state,
        "unavailable_city_production_ids": unavailable_production_city_ids,
        "raw_hash": _candidate_hash(table),
        "offered_hash": _candidate_hash(offered),
        "omitted_ordinary_count": len(raw_ids - offered_ids - set(missing_ids)),
    }


__all__ = ["assess_control_access"]
