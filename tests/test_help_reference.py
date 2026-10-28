# SPDX-License-Identifier: MIT
import hashlib
import json
from lechonk.adapter.help_reference import (
    load_helpdata,
    parse_helpdata,
    read_entry,
    render_search,
    search_entries,
)


SAMPLE = r'''\
; Small representative sample of Freeciv's INI-like helpdata format.
[help_overview]
name = _("?help:Overview")
text = _("A safe static overview.\nSecond line."), _("A second paragraph.")

[help_gen_units]
generate = " Units"
categories = "Military", "Civilian"

[help_strategy]
name = _(" Strategy")
text = _("Use roads \
to move efficiently.")
secret = _("must not be indexed")

[help_tables]
name = _("Tables")
text = _("Costs include $5 in gold.\n\n$TerrainAlterations")

[help_combat]
name = _("Combat")
text = _("Combat compares the attack and defense strength of units.")
'''


def test_parses_static_nodes_and_marks_generated_nodes():
    reference = parse_helpdata(SAMPLE, source_path="artifacts/reference/freeciv/helpdata.txt")
    assert reference.status == "available"
    overview = read_entry(reference, "overview")
    assert overview == {
        "key": "help_overview",
        "name": "Overview",
        "text": "A safe static overview.\nSecond line.\n\nA second paragraph.",
    }
    generated = read_entry(reference, "help_gen_units")
    assert generated["generated_content_unavailable"] is True
    assert generated["generate"] == "Units"
    assert generated["categories"] == ["Civilian", "Military"]
    assert generated["domains"] == ["unit"]
    assert "secret" not in reference.canonical_json()


def test_search_and_render_are_deterministic_and_bounded():
    first = parse_helpdata(SAMPLE)
    second = parse_helpdata(SAMPLE.replace("[help_strategy]", "[help_strategy]\n"))
    assert first.content_hash == second.content_hash
    assert search_entries(first, "overview")[0]["name"] == "Overview"
    rendered = render_search(first, "", max_chars=100)
    assert rendered["entries"] == []
    rendered = render_search(first, "safe", max_chars=20)
    assert rendered["omitted"] == 1
    assert rendered["source_sha256"] == first.source_sha256


def test_text_is_sanitized_and_capped_without_evaluation():
    reference = parse_helpdata('[help_x]\nname = "X\\x00"\ntext = "' + ("a" * 20) + '"', max_text_chars=8)
    entry = read_entry(reference, "X ") or read_entry(reference, "X")
    assert entry is not None
    assert "\\x00" not in entry["name"]
    assert entry["text"].endswith("…")
    assert entry["omitted"] is True


def test_inline_generated_table_is_disclosed_without_treating_prices_as_tables():
    reference = parse_helpdata(SAMPLE)
    entry = read_entry(reference, "Tables")
    assert entry["generated_tables"] == ["TerrainAlterations"]
    assert entry["generated_content_unavailable"] is True
    assert "$5 in gold" in entry["text"]


def test_natural_question_query_matches_meaningful_terms_and_exact_titles_rank_first():
    reference = parse_helpdata(SAMPLE)
    results = search_entries(reference, "What is combat?")
    assert results[0]["name"] == "Combat"
    generated = search_entries(reference, "What are terrain alterations?")[0]
    assert generated["generated_content_unavailable"] is True
    assert generated["generated_tables"] == ["TerrainAlterations"]


def test_domain_filter_is_applied_before_limit_and_does_not_use_generic_categories():
    reference = parse_helpdata(SAMPLE)
    assert search_entries(reference, "what are units", categories=["unit"], limit=1)[0]["domains"] == ["unit"]
    assert search_entries(reference, "what are units", categories=["Units"], limit=1)[0]["domains"] == ["unit"]
    assert search_entries(reference, "what are units", categories=["Natural"], limit=10) == ()


def test_missing_or_unsupported_reference_is_explicit():
    assert parse_helpdata("").status == "unavailable"
    assert parse_helpdata("[ruleset]\nname = \"classic\"").status == "unsupported"
    assert load_helpdata("/definitely/not/a/helpdata.txt").status == "unavailable"


def test_manifest_checksum_mismatch_is_not_loaded(tmp_path):
    path = tmp_path / "helpdata.txt"
    path.write_text(SAMPLE)
    (tmp_path / "reference.json").write_text(json.dumps({
        "image_ref": "pinned",
        "helpdata_sha256": hashlib.sha256(b"different").hexdigest(),
    }))
    reference = load_helpdata(path)
    assert reference.status == "unavailable"
    assert "checksum verification" in reference.warnings[0]
