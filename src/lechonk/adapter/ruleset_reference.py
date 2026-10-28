# SPDX-License-Identifier: GPL-3.0-or-later
"""A bounded, deterministic encyclopedia extracted from CivRealm RulesetCtrl.

This module belongs to the GPL-3.0-or-later ``lechonk.adapter`` boundary
because its input is the vendored GPL CivRealm ruleset implementation.

The adapter deliberately copies only stable, public ruleset data.  It never
serializes packet dictionaries wholesale: packet fields not listed in the
allow-list below are ignored, which keeps player, map, and server state out of
model-facing reference responses.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

REFERENCE_SCHEMA_VERSION = "ruleset_reference_v1"
REFERENCE_SOURCE = "CivRealm RulesetCtrl ruleset packets"
DEFAULT_MAX_TEXT_CHARS = 16_000
DEFAULT_RENDER_MAX_CHARS = 8_000

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACE_RE = re.compile(r"[^\S\n]+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = {"a", "an", "are", "does", "for", "how", "is", "me", "of", "please", "show", "stats", "tell", "the", "what", "which"}

# These are intentionally narrow.  Keys are the public names used by the
# Freeciv ruleset packets; aliases are accepted only when they map to the same
# documented concept.
_CATEGORY_FIELDS: dict[str, tuple[str, ...]] = {
    "tech": ("name", "helptext", "cost", "research_reqs"),
    "unit": (
        "name", "helptext", "attack_strength", "defense_strength",
        "move_rate", "vision_radius_sq", "hp", "firepower", "build_cost",
        "upkeep_food", "upkeep_gold", "build_reqs", "reqs", "obsoleted_by",
        "obsolete_by",
    ),
    "improvement": (
        "name", "helptext", "build_cost", "upkeep", "reqs",
        "obsolete_by",
    ),
    "terrain": ("name", "helptext", "output"),
    "government": ("name", "helptext"),
    "extra": ("name", "helptext"),
}

_CONTAINER_NAMES = {
    "tech": "techs",
    "unit": "unit_types",
    "improvement": "improvements",
    "terrain": "terrains",
    "government": "governments",
    "extra": "extras",
}
_NUMERIC_FIELDS = {
    "attack_strength", "defense_strength", "move_rate", "vision_radius_sq",
    "hp", "firepower", "build_cost", "cost", "upkeep_food", "upkeep_gold",
    "upkeep", "food", "shield", "trade", "move_cost", "movement_cost",
    "defense_bonus",
}
_REQ_FIELDS = {"research_reqs", "build_reqs", "reqs"}
_CATEGORY_ALIASES = {
    "tech": "tech", "techs": "tech", "technology": "tech", "technologies": "tech",
    "unit": "unit", "units": "unit",
    "building": "improvement", "buildings": "improvement",
    "improvement": "improvement", "improvements": "improvement",
    "terrain": "terrain", "terrains": "terrain",
    "government": "government", "governments": "government",
    "extra": "extra", "extras": "extra",
}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _clean_text(value: Any, limit: int) -> str:
    """Normalize packet text and cap it without emitting control characters."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(part) for part in value if part is not None)
    elif not isinstance(value, str):
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value)
    value = re.sub(r" *\n *", "\n", value).strip()
    if limit < 1:
        return ""
    if len(value) > limit:
        return value[: max(0, limit - 1)].rstrip() + "…"
    return value


def _safe_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return None


def _safe_requirements(value: Any) -> list[int | float | str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    result: list[int | float | str] = []
    for item in value:
        scalar = _safe_scalar(item)
        if scalar is not None and not isinstance(scalar, bool):
            result.append(scalar)
    return result


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(value)
                 if token.casefold() not in _STOP_WORDS and len(token) > 1)


def _singular_plural_equal(left: str, right: str) -> bool:
    """Recognize only the safe, regular trailing-``s`` alias."""
    if left == right:
        return True
    if len(left) < 3 or len(right) < 4:
        return False
    if right.endswith("s") and not right.endswith("ss") and right[:-1] == left:
        return True
    if left.endswith("s") and not left.endswith("ss") and left[:-1] == right:
        return True
    return False


def _safe_requirement_records(value: Any) -> list[dict[str, int | float | str | bool]]:
    """Copy only the public requirement tuple fields from Freeciv packets."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        record = {}
        for key in ("kind", "value", "present", "range", "survives"):
            scalar = _safe_scalar(item.get(key))
            if scalar is not None:
                record[key] = scalar
        if record:
            result.append(record)
    return result


def _safe_output(value: Any) -> list[int | float]:
    # RulesetCtrl converts terrain output to numpy arrays.  ``tolist`` is the
    # only non-JSON representation admitted here; preserve the packet's
    # positional order without assigning unverified labels.
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                result.append(item)
        return result
    if not isinstance(value, Mapping):
        return []
    result: list[int | float] = []
    for key in ("food", "shield", "trade"):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result.append(item)
    return result


def _record(category: str, packet: Any, text_limit: int) -> dict[str, Any] | None:
    name = _clean_text(_get(packet, "name"), text_limit)
    if not name:
        return None
    result: dict[str, Any] = {"category": category, "name": name}
    packet_id = _get(packet, "id")
    if isinstance(packet_id, int) and not isinstance(packet_id, bool):
        result["id"] = packet_id
    for key in _CATEGORY_FIELDS[category]:
        if key in {"name", "research_reqs", "build_reqs", "reqs", "obsolete_by", "output"}:
            continue
        value = _get(packet, key)
        if key == "helptext":
            text = _clean_text(value, text_limit)
            if text:
                result["helptext"] = text
        elif key in _NUMERIC_FIELDS:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = value
        else:
            scalar = _safe_scalar(value)
            if scalar is not None:
                result[key] = scalar
    for key in _REQ_FIELDS:
        values = _safe_requirement_records(_get(packet, key))
        if values:
            result["requirements"] = values
            break
    raw_obsolete = _get(packet, "obsolete_by")
    obsolete = (_safe_requirement_records(raw_obsolete)
                if isinstance(raw_obsolete, (list, tuple)) and
                any(isinstance(item, Mapping) for item in raw_obsolete)
                else _safe_requirements(raw_obsolete))
    if obsolete:
        result["obsolete_by"] = obsolete
    output = _safe_output(_get(packet, "output"))
    if output:
        result["output"] = output
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sorted_records(records: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in records:
        unique.setdefault((record.get("id"), record["name"].casefold()), record)
    return tuple(sorted(unique.values(), key=lambda item: (item["name"].casefold(), item.get("id", -1))))


def _resolve_tech_requirement_names(categories: dict[str, tuple[dict[str, Any], ...]]) -> None:
    """Add names only for requirement records explicitly typed as tech IDs."""
    tech_names = {
        record["id"]: record["name"]
        for record in categories.get("tech", ())
        if isinstance(record.get("id"), int)
    }
    if not tech_names:
        return
    for category in ("tech", "unit", "improvement"):
        updated = []
        for record in categories.get(category, ()):
            copy = dict(record)
            requirements = []
            for field_name in ("requirements", "obsolete_by"):
                field_values = copy.get(field_name, ())
                annotated = []
                for requirement in field_values:
                    requirement = dict(requirement) if isinstance(requirement, Mapping) else requirement
                    if not isinstance(requirement, dict):
                        annotated.append(requirement)
                        continue
                    # VUT_ADVANCE (1) plus REQ_RANGE_PLAYER (7) is the only
                    # scope where a tech ID is safely name-resolvable here.
                    if (requirement.get("kind") == 1 and
                            requirement.get("range") == 7 and
                            isinstance(requirement.get("value"), int)):
                        name = tech_names.get(requirement["value"])
                        if name is not None:
                            requirement["value_name"] = name
                    annotated.append(requirement)
                if annotated:
                    copy[field_name] = annotated
            updated.append(copy)
        categories[category] = tuple(updated)


def _resolve_unit_obsolescence(categories: dict[str, tuple[dict[str, Any], ...]]) -> None:
    unit_names = {record["id"]: record["name"] for record in categories.get("unit", ()) if isinstance(record.get("id"), int)}
    if not unit_names:
        return
    updated = []
    for record in categories.get("unit", ()):
        copy = dict(record)
        ids = copy.get("obsolete_by", ())
        if isinstance(ids, (list, tuple)) and any(isinstance(item, Mapping) for item in ids):
            ids = [item.get("value") for item in ids if isinstance(item, Mapping)]
        if isinstance(ids, (int, str)):
            ids = (ids,)
        names = [unit_names[item] for item in ids if item in unit_names]
        if names:
            copy["obsolete_by_names"] = names
        obsolete_name = copy.get("obsoleted_by")
        if isinstance(obsolete_name, int) and obsolete_name in unit_names:
            copy["obsoleted_by"] = unit_names[obsolete_name]
        updated.append(copy)
    categories["unit"] = tuple(updated)


@dataclass(frozen=True)
class RulesetReference:
    """Immutable model-facing ruleset snapshot."""

    ruleset_name: str
    summary: str
    description: str
    categories: Mapping[str, tuple[Mapping[str, Any], ...]]
    schema_version: str = REFERENCE_SCHEMA_VERSION
    source: str = REFERENCE_SOURCE
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = {
            category: tuple(dict(record) for record in self.categories.get(category, ()))
            for category in _CATEGORY_FIELDS
        }
        object.__setattr__(self, "categories", MappingProxyType(normalized))
        object.__setattr__(self, "_hash", hashlib.sha256(_canonical(self.to_dict(include_hash=False)).encode("utf-8")).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "ruleset_name": self.ruleset_name,
            "summary": self.summary,
            "description": self.description,
            "categories": {
                category: [dict(record) for record in self.categories.get(category, ())]
                for category in _CATEGORY_FIELDS
            },
        }
        if include_hash:
            result["content_hash"] = self.content_hash
        return result

    def canonical_json(self) -> str:
        return _canonical(self.to_dict(include_hash=False))


def extract_ruleset_reference(
    ruleset_ctrl: Any,
    *,
    ruleset_name: str | None = None,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> RulesetReference:
    """Extract whitelisted static data from a CivRealm ``RulesetCtrl``."""
    categories: dict[str, tuple[dict[str, Any], ...]] = {}
    for category, container_name in _CONTAINER_NAMES.items():
        container = _get(ruleset_ctrl, container_name, {})
        values = container.values() if isinstance(container, Mapping) else (container or ())
        records = [_record(category, packet, max_text_chars) for packet in values]
        categories[category] = _sorted_records(record for record in records if record is not None)
    _resolve_tech_requirement_names(categories)
    _resolve_unit_obsolescence(categories)
    control = _get(ruleset_ctrl, "ruleset_control", {})
    name = _clean_text(ruleset_name or _get(control, "name") or "unknown", max_text_chars)
    return RulesetReference(
        ruleset_name=name or "unknown",
        summary=_clean_text(_get(ruleset_ctrl, "ruleset_summary"), max_text_chars),
        description=_clean_text(_get(ruleset_ctrl, "ruleset_description"), max_text_chars),
        categories=categories,
    )


def read_entry(reference: RulesetReference, category: str, name_or_id: str | int) -> dict[str, Any] | None:
    """Read one exact-name (case-insensitive) or exact numeric-id entry."""
    category = _CATEGORY_ALIASES.get(category.casefold(), "") if isinstance(category, str) else ""
    if category not in _CATEGORY_FIELDS:
        return None
    for entry in reference.categories.get(category, ()):
        if (isinstance(name_or_id, int) and entry.get("id") == name_or_id) or (
            isinstance(name_or_id, str) and entry["name"].casefold() == name_or_id.casefold()
        ):
            return dict(entry)
    return None


def search_entries(
    reference: RulesetReference,
    query: str,
    *,
    categories: Sequence[str] | None = None,
    limit: int = 10,
) -> tuple[dict[str, Any], ...]:
    """Search natural-language queries with deterministic relevance ranking."""
    query = _clean_text(query, DEFAULT_MAX_TEXT_CHARS)
    query_tokens = set(_tokens(query))
    if not query_tokens or limit <= 0:
        return ()
    allowed = {
        _CATEGORY_ALIASES.get(category.casefold(), "")
        for category in categories
        if isinstance(category, str)
    } if categories is not None else set(_CATEGORY_FIELDS)
    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for category in _CATEGORY_FIELDS:
        if category not in allowed:
            continue
        for entry in reference.categories.get(category, ()):
            name = entry["name"].casefold()
            name_tokens = set(_tokens(name))
            help_tokens = set(_tokens(entry.get("helptext", "")))
            exact_overlap = len(query_tokens & name_tokens)
            alias_overlap = sum(
                1 for query_token in query_tokens
                if query_token not in name_tokens
                and any(_singular_plural_equal(query_token, name_token) for name_token in name_tokens)
            )
            help_overlap = len(query_tokens & help_tokens)
            exact = int(name == query.casefold())
            if exact_overlap or alias_overlap or help_overlap:
                # A title-level singular/plural alias outranks an incidental
                # exact token in an unrelated title (Warriors before Warrior
                # Code for a unit lookup).
                score = exact * 2000 + alias_overlap * 600 + exact_overlap * 100 + help_overlap * 10
                candidates.append((-score, -exact, category, name, dict(entry)))
    candidates.sort(key=lambda item: item[:4])
    return tuple(item[4] for item in candidates[:limit])


def render_search(
    reference: RulesetReference,
    query: str,
    *,
    categories: Sequence[str] | None = None,
    limit: int = 10,
    max_chars: int = DEFAULT_RENDER_MAX_CHARS,
) -> dict[str, Any]:
    """Return bounded search output with explicit provenance and omission count."""
    matches = search_entries(reference, query, categories=categories, limit=limit)
    entries: list[dict[str, Any]] = []
    used = 0
    omitted = 0
    for entry in matches:
        encoded = _canonical(entry)
        if used + len(encoded) + 1 > max_chars:
            omitted += 1
            continue
        entries.append(entry)
        used += len(encoded) + 1
    return {
        "schema_version": reference.schema_version,
        "source": reference.source,
        "ruleset_name": reference.ruleset_name,
        "content_hash": reference.content_hash,
        "query": query,
        "entries": entries,
        "omitted": omitted,
    }


__all__ = [
    "REFERENCE_SCHEMA_VERSION", "REFERENCE_SOURCE", "RulesetReference",
    "extract_ruleset_reference", "read_entry", "search_entries", "render_search",
]
