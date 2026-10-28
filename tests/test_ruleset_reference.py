# SPDX-License-Identifier: MIT
from types import SimpleNamespace

from lechonk.adapter.ruleset_reference import (
    RulesetReference,
    extract_ruleset_reference,
    read_entry,
    render_search,
    search_entries,
)


def _search_fixture():
    return RulesetReference(
        ruleset_name="classic", summary="", description="",
        categories={
            "unit": ({"category": "unit", "name": "Warriors", "id": 1, "helptext": "Basic unit"},),
            "tech": ({"category": "tech", "name": "Warrior Code", "id": 2, "helptext": ""}, {"category": "tech", "name": "Pottery", "id": 3, "helptext": ""}),
            "terrain": ({"category": "terrain", "name": "Grassland", "id": 4, "helptext": ""},),
        },
    )


class _Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def _ctrl(reverse=False):
    entries = [
        {"id": 2, "name": "Writing", "helptext": "Write books\x00\r\nfor science", "research_reqs": [{"kind": 1, "value": 1, "present": True, "range": 7, "survives": 0}], "cost": 9999, "owner": "secret"},
        {"id": 1, "name": "Alphabet", "helptext": "Letters", "research_reqs": [], "player_id": 7},
    ]
    if reverse:
        entries.reverse()
    return SimpleNamespace(
        ruleset_name="classic",
        ruleset_control={"name": "Classic ruleset", "num_unit_types": 1},
        ruleset_summary="Classic\x01 ruleset",
        ruleset_description="A\r\nsmall ruleset",
        techs={entry["id"]: entry for entry in entries},
        unit_types={3: {"id": 3, "name": "Warrior", "helptext": "Basic unit", "attack_strength": 1, "defense_strength": 1, "vision_radius_sq": 2, "build_reqs": [{"kind": 1, "value": 1, "present": True, "range": 2, "survives": 1}], "obsolete_by": [4], "secret": "no"}, 4: {"id": 4, "name": "Phalanx", "helptext": "Advanced unit"}},
        improvements={4: {"id": 4, "name": "Granary", "helptext": "Stores food", "build_cost": 60, "reqs": [{"kind": 1, "value": 1, "present": True, "range": 7, "survives": 0}], "obsolete_by": [{"kind": 1, "value": 1, "present": True, "range": 7, "survives": 0}]}},
        terrains={5: {"id": 5, "name": "Grassland", "helptext": "Green land", "output": _Array([2, 0, 1]), "hidden": 42}},
        governments={},
        extras={6: {"id": 6, "name": "Railroad", "helptext": "Track", "secret": 1}, "Railroad": {"id": 6, "name": "Railroad", "helptext": "Track", "secret": 1}},
        current_player={"name": "must not leak"},
    )


def test_whitelist_and_metadata():
    reference = extract_ruleset_reference(_ctrl())
    writing = read_entry(reference, "tech", "Writing")
    assert writing == {"category": "tech", "name": "Writing", "id": 2, "helptext": "Write books\nfor science", "cost": 9999, "requirements": [{"kind": 1, "value": 1, "present": True, "range": 7, "survives": 0, "value_name": "Alphabet"}]}
    assert "owner" not in writing and "player_id" not in writing
    assert reference.ruleset_name == "Classic ruleset"
    assert reference.summary == "Classic ruleset"
    assert reference.description == "A\nsmall ruleset"
    assert reference.source == "CivRealm RulesetCtrl ruleset packets"


def test_order_and_hash_are_deterministic():
    first = extract_ruleset_reference(_ctrl())
    second = extract_ruleset_reference(_ctrl(reverse=True))
    assert first.canonical_json() == second.canonical_json()
    assert first.content_hash == second.content_hash


def test_search_exact_name_precedes_text_and_read_by_id():
    reference = extract_ruleset_reference(_ctrl())
    assert search_entries(reference, "writing", categories=["tech"])[0]["name"] == "Writing"
    assert search_entries(reference, "What are Warrior stats?")[0]["name"] == "Warrior"
    assert read_entry(reference, "units", 3)["name"] == "Warrior"
    assert read_entry(reference, "tech", "missing") is None


def test_search_supports_only_conservative_singular_plural_title_aliases():
    reference = _search_fixture()
    assert search_entries(reference, "Warrior", categories=["units"], limit=1)[0]["name"] == "Warriors"
    assert search_entries(reference, "Warrior", limit=2)[0]["name"] == "Warriors"
    assert search_entries(reference, "Warrior", categories=["tech"])[0]["name"] == "Warrior Code"
    assert search_entries(reference, "Pottery", categories=["technologies"])[0]["name"] == "Pottery"
    assert search_entries(reference, "Grassland", categories=["terrain"], limit=1)[0]["name"] == "Grassland"
    assert search_entries(reference, "Warrior", limit=0) == ()


def test_render_budget_and_provenance():
    reference = extract_ruleset_reference(_ctrl())
    response = render_search(reference, "", max_chars=20)
    assert response["source"] == reference.source
    assert response["content_hash"] == reference.content_hash
    assert response["entries"] == []
    response = render_search(reference, "unit", max_chars=100, limit=10)
    assert response["omitted"] >= 1


def test_nested_output_is_whitelisted():
    reference = extract_ruleset_reference(_ctrl())
    terrain = read_entry(reference, "terrain", "Grassland")
    assert terrain["output"] == [2, 0, 1]
    assert len(reference.categories["extra"]) == 1


def test_non_player_advance_requirement_keeps_scope_without_name_annotation():
    reference = extract_ruleset_reference(_ctrl())
    warrior = read_entry(reference, "unit", "Warrior")
    requirement = warrior["requirements"][0]
    assert requirement["range"] == 2
    assert requirement["survives"] == 1
    assert "value_name" not in requirement


def test_obsolescence_vector_is_preserved():
    reference = extract_ruleset_reference(_ctrl())
    warrior = read_entry(reference, "unit", "Warrior")
    assert warrior["obsolete_by"] == [4]
    granary = read_entry(reference, "improvement", "Granary")
    assert granary["obsolete_by"][0]["range"] == 7
    assert granary["obsolete_by"][0]["value_name"] == "Alphabet"
