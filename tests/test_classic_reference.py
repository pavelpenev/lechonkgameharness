import json
from pathlib import Path

import pytest

from lechonk.adapter.classic_reference import (
    ClassicReferenceError,
    PINNED_IMAGE_REF,
    _tree_hash,
    load_classic_reference,
    parse_classic_ruleset,
    render_search,
)


SNIPPET = '''
[unit_warrior]
name = _( "Warrior" )
rule_name = "warrior"
helptext = _("A sturdy \\
front-line unit. Привет")
requirements = "must not be interpreted"

[unit_scout]
name = _("Scout")
helptext = _("Fast explorer")
'''

# Small verbatim-shaped excerpts from the extracted classic ruleset.  The
# fields between ``name`` and ``helptext`` are intentionally non-display
# assignments: the parser must use them as boundaries, not as help text.
CLASSIC_WARRIORS = '''
[unit_warriors]
name          = _("Warriors")
class         = "Land"
reqs          = { "type", "name", "None" }
graphic       = "u.warriors"
graphic_alt   = "-"
helptext      = _("This unit may be built from the start of the game.\\
It is the weakest offensive unit.")
'''

CLASSIC_POTTERY = '''
[advance_pottery]
name     = _("Pottery")
req1     = "None"
req2     = "None"
flags    = ""
graphic  = "a.pottery"
graphic_alt = "-"
'''


def _reference(tmp_path: Path, text: str = SNIPPET) -> Path:
    root = tmp_path / "reference"
    classic = root / "classic"
    classic.mkdir(parents=True)
    (classic / "units.ruleset").write_text(text, encoding="utf-8")
    (root / "reference.json").write_text(json.dumps({
        "schema_version": "freeciv-reference-v1",
        "image_ref": PINNED_IMAGE_REF,
        "image_version": "test",
        "classic_ruleset_tree_sha256": _tree_hash(classic),
    }), encoding="utf-8")
    return root


def test_parse_only_display_fields_and_unicode():
    entries = parse_classic_ruleset(SNIPPET, source_file="units.ruleset")
    assert [e.name for e in entries] == ["Warrior", "Scout"]
    assert "Привет" in entries[0].helptext
    assert "requirements" not in entries[0].helptext
    assert entries[0].authority == "static_ruleset_source_non_authoritative"


def test_classic_shaped_assignments_are_boundaries_and_markers_are_cleaned():
    entries = parse_classic_ruleset(
        CLASSIC_WARRIORS
        + '\n[unit_marker]\nname = _("?unit:Marked")\nclass = "Land"\n'
        + CLASSIC_POTTERY,
        source_file="units.ruleset",
    )
    warriors = next(entry for entry in entries if entry.key == "unit_warriors")
    marked = next(entry for entry in entries if entry.key == "unit_marker")
    pottery = next(entry for entry in entries if entry.key == "advance_pottery")
    assert warriors.name == "Warriors"
    assert warriors.helptext == "This unit may be built from the start of the game.It is the weakest offensive unit."
    assert "graphic" not in warriors.helptext
    assert marked.name == "Marked"
    assert pottery.name == "Pottery"


def test_extracted_classic_files_have_clean_display_entries():
    root = Path(__file__).parents[1] / "artifacts" / "reference" / "freeciv" / "classic"
    if not (root / "units.ruleset").exists():
        pytest.skip("extracted classic ruleset is not present")
    units = parse_classic_ruleset((root / "units.ruleset").read_text(encoding="utf-8"), source_file="units.ruleset")
    techs = parse_classic_ruleset((root / "techs.ruleset").read_text(encoding="utf-8"), source_file="techs.ruleset")
    warriors = next(entry for entry in units if entry.key == "unit_warriors")
    pottery = next(entry for entry in techs if entry.key == "advance_pottery")
    assert warriors.name == "Warriors"
    assert pottery.name == "Pottery"
    assert "graphic" not in warriors.name.lower()
    assert "None" not in pottery.name


def test_load_and_deterministic_search(tmp_path):
    ref = load_classic_reference(_reference(tmp_path))
    snapshot = ref.to_dict()
    assert snapshot["authority"] == "static_ruleset_source_non_authoritative"
    assert snapshot["content_hash"] == ref.content_hash
    assert snapshot["provenance"]["tree_sha256"] == ref.tree_sha256
    assert snapshot["entries"][0]["source_file"] == "units.ruleset"
    result = render_search(ref, "what is warrior?")
    assert result["entries"][0]["name"] == "Warrior"
    assert result["authority"] == "static_ruleset_source_non_authoritative"
    assert result["entries"][0]["source_file"] == "units.ruleset"


@pytest.mark.parametrize("category", ["unit", "units"])
def test_search_category_aliases_match_live_reference_vocabulary(tmp_path, category):
    ref = load_classic_reference(_reference(tmp_path))
    result = render_search(ref, "warrior", categories=[category])
    assert [entry["name"] for entry in result["entries"]] == ["Warrior"]


def test_checksum_tamper_rejected(tmp_path):
    root = _reference(tmp_path)
    path = root / "classic" / "units.ruleset"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ClassicReferenceError, match="checksum"):
        load_classic_reference(root)


def test_stale_manifest_reports_migration_action(tmp_path):
    root = _reference(tmp_path)
    manifest_path = root / "reference.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["classic_ruleset_tree_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ClassicReferenceError, match=r"re-run .*extract_freeciv_reference\.sh") as exc_info:
        load_classic_reference(root)

    message = str(exc_info.value)
    assert "reference.json records" in message
    assert "extracted files hash to" in message
    assert "do not edit the checksum by hand" in message


def test_manifest_image_must_be_exact_pinned_digest(tmp_path):
    root = _reference(tmp_path)
    manifest_path = root / "reference.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["image_ref"] = "civrealm/freeciv-web@sha256:" + "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ClassicReferenceError, match="not the pinned Freeciv image") as exc_info:
        load_classic_reference(root)

    assert PINNED_IMAGE_REF in str(exc_info.value)
    assert "do not edit image_ref by hand" in str(exc_info.value)


def test_symlink_rejected(tmp_path):
    root = _reference(tmp_path)
    target = root / "classic" / "units.ruleset"
    link = root / "classic" / "escape.ruleset"
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    with pytest.raises(ClassicReferenceError, match="symlink"):
        load_classic_reference(root)


def test_missing_reference_is_explicit(tmp_path):
    ref = load_classic_reference(tmp_path / "missing")
    assert ref.status == "unavailable"


def test_limits_are_enforced(tmp_path):
    root = _reference(tmp_path)
    with pytest.raises(ClassicReferenceError, match="file count"):
        load_classic_reference(root, max_files=0)
