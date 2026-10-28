# SPDX-License-Identifier: MIT
"""Prompt template tests — the WP2 package-A prompt surface
(design-plan-freeciv-v1.1.md §7.5):

- templates are versioned FILES (``_vN`` suffix) under model/prompts/, never
  inline strings;
- the deterministic prompt-hash function (sha256 over the UTF-8 file bytes)
  is stable and matches the recorded hashes (any template edit breaks these
  tests — that is the point: the manifest records these hashes, decision 8);
- the system prompt carries role, interface rules (candidate IDs only +
  budgets + commitment grammar), the fog disclaimer, and the cheat sheet;
- the per-turn prompt structure: [stable prefix] + memory (≤1,500) +
  snapshot (≤8,000) + delta (≤800) + candidate table summary, with budget
  annotations, and the ActionBatch output contract incl. the
  memory_update-on-ActionBatch rule;
- the three repair prompt variants per error class render with their
  placeholders;
- the fixed fingerprint probe prompt (decision 8);
- no template references the API key or any secret.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from lechonk.model import prompts
from lechonk.model.prompts import (
    ALL_PROMPTS,
    ASSEMBLED_SECTION_BODY_TOKEN_BUDGET,
    CANDIDATE_TABLE_TOKEN_BUDGET,
    FINGERPRINT_PROBE,
    MEMORY_TOKEN_BUDGET,
    PROMPT_DIR,
    REPAIR_CONVERSATION_TOKEN_BUDGET,
    REPAIR_TEMPLATES,
    SYSTEM_PROMPT,
    TURN_PREFIX,
    PromptBudgetExceeded,
    PromptConfigError,
    PromptNotFoundError,
    PromptRenderError,
    PromptError,
    all_prompt_hashes,
    active_prompt_hashes,
    assembled_section_body_budget,
    candidate_table_budget,
    check_fixed_prompt_budget,
    capture_prompt_snapshot,
    PromptSnapshot,
    fingerprint_probe_prompt,
    load_prompt,
    prompt_hash,
    prompt_manifest_hashes,
    repair_messages,
    repair_prompt,
    repair_conversation_budget,
    render_template,
    system_prompt,
    turn_prompt,
    validate_conversation_system_message,
)

# Recorded hashes (sha256 over the UTF-8 template bytes). The run manifest
# records these (decision 8); editing a template without bumping its version
# and these constants is a test failure by design.
# system_v1/turn_prefix_v1 are retained on disk for history (the WP2
# truncation evidence, 8/17 length-capped calls, ran on v1); system_v2 is
# retained for history (the brevity discipline, F1, ran on v2); the live
# registry points at the v3 system prompt (FW-10: typed commitment grammar).
# fingerprint_probe_v1 is retained for history (the verbatim-echo probe);
# the live registry points at the v2 capacity-revealing probe (FW-21) —
# the fingerprint client (mistral_client) loads it directly.
RECORDED_HASHES = {
    "system_v1":
        "sha256:28d1e070592b0fcc4fb92a85b87d5caf725dc14548f977e3767bf675a4f6cdd4",
    "turn_prefix_v1":
        "sha256:6605991d15db3560d448c8179dda10d4d428e16ffd74b9fdb83ef13aaded368f",
    "system_v2":
        "sha256:c6b059013fa5b04a09b1def3233c743517888fb19980fd04d2b5fed4b94b2075",
    "system_v3":
        "sha256:e549096dd6b3dec92aa12fd71588930dd3d4687a59aabb9da0887afcad1dd46b",
    "turn_prefix_v2":
        "sha256:c5ef73e490dc422fcb5e57c40c1420c183cc50db325c6a109bbebf250fa07a51",
    "repair_schema_error_v1":
        "sha256:e7335edc51fdb0215f7b552555805a2803c1c7ef6f861b6f67bbdeda69e1c025",
    "repair_state_fog_error_v1":
        "sha256:fef9623ab199a2b8132ee996203be55336773cbe8e7aeba9f4b9d418eeb7304d",
    "repair_illegal_in_rule_v1":
        "sha256:f99a1ff3730256c8f55efeb59cda0868ffb1271456143d666d9cbb292cf236ec",
    "fingerprint_probe_v1":
        "sha256:f1e5cf8937f94cabb84e0178debab9587cfa0d5f6ad7e12ad6012d010a3d05d1",
    "fingerprint_probe_v2":
        "sha256:29c72ee59512c80521b3d55975e3dd82e3ea624a4e186655649b968ef78035cb",
}


# ---------------------------------------------------------------------------
# Files, versions, hashes
# ---------------------------------------------------------------------------


def test_all_declared_template_files_exist():
    for name in ALL_PROMPTS:
        assert (PROMPT_DIR / f"{name}.txt").is_file(), (
            f"missing template file for {name}")


def test_template_filenames_are_versioned():
    for name in ALL_PROMPTS:
        assert re.search(r"_v\d+$", name), (
            f"template {name} is not versioned (_vN)")


def test_prompt_hash_format_and_determinism():
    for name in ALL_PROMPTS:
        digest = prompt_hash(name)
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64
        assert prompt_hash(name) == digest  # deterministic


def test_prompt_hash_matches_file_bytes():
    for name in ALL_PROMPTS:
        expected = "sha256:" + hashlib.sha256(
            (PROMPT_DIR / f"{name}.txt").read_bytes()).hexdigest()
        assert prompt_hash(name) == expected


def test_recorded_hashes_are_stable():
    """The manifest-recorded hashes: any template edit breaks this test."""
    for name, expected in RECORDED_HASHES.items():
        assert prompt_hash(name) == expected, (
            f"template {name} changed (recorded {expected}, "
            f"now {prompt_hash(name)}) — bump the version and the manifest")


def test_all_prompt_hashes_covers_the_directory():
    hashes = all_prompt_hashes()
    on_disk = {p.stem for p in PROMPT_DIR.glob("*.txt")}
    assert set(hashes) == on_disk  # no orphans, no missing
    for name in ALL_PROMPTS:
        assert name in hashes


def test_prompt_manifest_hashes_covers_declared_templates():
    manifest_hashes = prompt_manifest_hashes()
    for name in ALL_PROMPTS:
        assert manifest_hashes[name] == RECORDED_HASHES[name]


def test_load_prompt_unknown_name_raises():
    with pytest.raises(PromptNotFoundError):
        load_prompt("does_not_exist_v9")


# ---------------------------------------------------------------------------
# System prompt (§7.5): role, interface rules, fog disclaimer, cheat sheet
# ---------------------------------------------------------------------------


def test_system_prompt_role_and_interface_rules():
    text = system_prompt()
    assert "Le Chonk" in text
    assert "Freeciv" in text
    # candidate IDs only
    assert "candidate IDs only" in text
    assert "domain:actor:verb" in text
    assert "unit:7:goto_n" in text
    # budgets
    assert "20 actions" in text
    assert "1,500 tokens" in text
    assert "8,000 tokens" in text
    assert "800 tokens" in text
    # commitment grammar
    assert "domain" in text and "target" in text and "deadline_turn" in text
    assert "falsifiable" in text


def test_system_prompt_fog_disclaimer():
    text = system_prompt()
    assert "VANISHED" in text
    assert "NOT destroyed" in text
    assert "Unknowns are real" in text
    assert "last-known" in text.lower()


def test_system_prompt_action_surface_cheat_sheet():
    text = system_prompt()
    for domain in ("unit:", "city:", "tech:", "dipl:"):
        assert domain in text
    for verb in ("goto_", "attack_", "build_city", "produce_",
                 "research_", "stop_negotiation"):
        assert verb in text


def test_system_prompt_brevity_discipline():
    """The v2 output-discipline block (review finding F1: 8/17 live calls
    hit the 8,192 completion-token cap on v1 prompts)."""
    text = system_prompt()
    assert "OUTPUT DISCIPLINE" in text
    assert "notes" in text and "2 sentences" in text
    assert "falsifiable" in text and "intend to act" in text
    assert "memory_update" in text and "compact" in text
    assert "SNAPSHOT" in text and "DELTA" in text  # no restating them
    assert "filler" in text


def test_system_prompt_v3_typed_commitment_grammar():
    """FW-10: the live system prompt is v3 and documents the typed
    commitment grammar — the target_kind vocabulary with the canonical
    target forms, the exact-match rule, and the verb vocabulary with the
    commitment-verb -> action-verb compatibility mapping."""
    assert prompts.SYSTEM_PROMPT == "system_v3"
    assert SYSTEM_PROMPT == "system_v3"
    text = system_prompt()
    # the typed grammar fields
    assert "target_kind" in text and "deadline_turn" in text
    assert "Commitment grammar (v3, typed targets)" in text
    # the target_kind vocabulary
    for kind in ("tech", "production", "unit", "city", "player", "tile",
                 "any"):
        assert kind in text
    # the canonical typed target forms (the runner's action-side forms)
    for form in ('"unit:<id>"', '"city:<id>"', '"player:<id>"',
                 '"tile:<x>,<y>"'):
        assert form in text
    assert '"*" as the target is the wildcard' in text
    # exact matching, no substring
    assert "EXACT match" in text and "partial names never match" in text
    # the verb vocabulary + the compatibility mapping
    for verb in ("research", "build", "attack", "move", "negotiate",
                 "defend"):
        assert verb in text
    for action_verb in ("research_*", "build_city", "produce_*",
                        "build_road", "irrigation", "mine", "attack_*",
                        "goto_*", "stop_negotiation", "cancel_treaty",
                        "accept_treaty", "fortify", "sentry"):
        assert action_verb in text


def test_turn_prefix_brevity_discipline():
    text = load_prompt(TURN_PREFIX)
    assert "Brevity" in text
    assert "≤ 2 sentences" in text
    assert "never restate the SNAPSHOT or DELTA" in text
    assert "never restate current production, research, or" in text


# ---------------------------------------------------------------------------
# Per-turn prompt structure (§7.5)
# ---------------------------------------------------------------------------


def test_turn_prefix_is_the_stable_prefix_with_output_contract():
    text = load_prompt(TURN_PREFIX)
    # stable: rules + cheat sheet
    assert "RULES (stable)" in text
    assert "CHEAT SHEET (stable)" in text
    assert "candidate IDs ONLY" in text
    # output contract: the ActionBatch shape
    assert "OUTPUT CONTRACT (stable)" in text
    for key in ("turn_id", "actions", "commitments", "finish_requested",
                "notes", "memory_update"):
        assert key in text
    # the memory_update-on-ActionBatch rule (no separate memory call)
    assert "separate memory call" in text
    assert "last valid memory persists" in text
    # the section order
    assert "MEMORY, SNAPSHOT, DELTA, CANDIDATE" in text


def test_turn_prompt_composition_order_and_budget_annotations():
    rendered = turn_prompt(
        memory="MEM-BODY", snapshot="SNAP-BODY", delta="DELTA-BODY",
        candidate_summary="CAND-BODY")
    prefix = load_prompt(TURN_PREFIX)
    # the stable prefix comes first (prompt caching)
    assert rendered.startswith(prefix)
    # sections follow in the §7.5 order, each with its budget annotation
    i_memory = rendered.index("## MEMORY")
    i_snapshot = rendered.index("## SNAPSHOT")
    i_delta = rendered.index("## DELTA")
    i_candidates = rendered.index("## CANDIDATE TABLE")
    assert i_memory < i_snapshot < i_delta < i_candidates
    assert f"≤ {MEMORY_TOKEN_BUDGET:,} tokens" in rendered
    assert "≤ 8,000 tokens" in rendered
    assert "≤ 800 tokens" in rendered
    # the variable bodies are present
    for body in ("MEM-BODY", "SNAP-BODY", "DELTA-BODY", "CAND-BODY"):
        assert body in rendered


def test_turn_prompt_stable_prefix_identical_across_turns():
    """The prefix is byte-identical across turns (prompt caching,
    chonk-doom lesson): only the section bodies vary."""
    a = turn_prompt(memory="m1", snapshot="s1", delta="d1",
                    candidate_summary="c1")
    b = turn_prompt(memory="m2", snapshot="s2", delta="d2",
                    candidate_summary="c2")
    prefix = load_prompt(TURN_PREFIX)
    assert a.startswith(prefix)
    assert b.startswith(prefix)
    assert a[:len(prefix)] == b[:len(prefix)]


def test_hybrid_history_is_explicit_and_precedes_current_state():
    rendered = turn_prompt(
        memory="MEM", history='{"turn": 1, "status": "verified"}',
        snapshot="CURRENT-SNAPSHOT", delta="CURRENT-DELTA",
        candidate_summary="CURRENT-CANDIDATES")
    assert "HYBRID CONTEXT (stable)" in rendered
    assert "historical decision data" in rendered
    assert rendered.index("## HISTORY") < rendered.index("## SNAPSHOT")
    assert rendered.index("CURRENT-SNAPSHOT") > rendered.index("## HISTORY")
    assert '"status": "verified"' in rendered


def test_hybrid_history_budget_is_separate_from_legacy_budgets():
    with pytest.raises(PromptBudgetExceeded):
        turn_prompt(memory="m", history="h" *
                    (prompts.HISTORY_TOKEN_BUDGET * 4 + 100),
                    snapshot="s", delta="d", candidate_summary="c")
    # Legacy prompt remains unchanged when history is absent.
    legacy = turn_prompt(memory="m", snapshot="s", delta="d",
                         candidate_summary="c")
    assert "## HISTORY" not in legacy


def test_turn_prompt_budget_enforcement():
    big_memory = "x" * (MEMORY_TOKEN_BUDGET * 4 + 100)  # > 1,500 tokens
    with pytest.raises(PromptBudgetExceeded):
        turn_prompt(memory=big_memory, snapshot="s", delta="d",
                    candidate_summary="c")
    big_snapshot = "y" * (8000 * 4 + 100)
    with pytest.raises(PromptBudgetExceeded):
        turn_prompt(memory="m", snapshot=big_snapshot, delta="d",
                    candidate_summary="c")
    big_delta = "z" * (800 * 4 + 100)
    with pytest.raises(PromptBudgetExceeded):
        turn_prompt(memory="m", snapshot="s", delta=big_delta,
                    candidate_summary="c")
    # within budget: fine
    turn_prompt(memory="m", snapshot="s", delta="d", candidate_summary="c")
    # the candidate table has no §3 budget; enforcement can be disabled
    turn_prompt(memory="m", snapshot="s", delta="d",
                candidate_summary="c" * 100000, enforce_budgets=False)


# ---------------------------------------------------------------------------
# Repair prompts (three variants per error class, §7.5)
# ---------------------------------------------------------------------------


def test_three_repair_variants_exist_per_error_class():
    assert set(REPAIR_TEMPLATES) == {"schema_error", "state_fog_error",
                                     "illegal_in_rule"}


def test_repair_prompt_schema_error_variant():
    text = repair_prompt(
        "schema_error", detail="ref pattern violated",
        failed_refs="(none)", refreshed_candidates="FULL TABLE: ...")
    assert "SCHEMA ERROR" in text
    assert "ref pattern violated" in text
    assert "candidate IDs" in text
    assert "FULL TABLE: ..." in text


def test_repair_prompt_state_fog_error_variant():
    text = repair_prompt(
        "state_fog_error", detail="ref unknown in the current table",
        failed_refs="unit:7:goto_n",
        refreshed_candidates="REFRESHED: unit:7:goto_s")
    assert "STATE-FOG ERROR" in text
    assert "unit:7:goto_n" in text
    assert "state changed" in text
    assert "REFRESHED: unit:7:goto_s" in text


def test_repair_prompt_illegal_in_rule_variant():
    text = repair_prompt(
        "illegal_in_rule", detail="actor out of moves",
        failed_refs="city:3:produce_settlers",
        refreshed_candidates="REFRESHED: city:3:produce_warriors")
    assert "ILLEGAL-IN-RULE" in text
    assert "city:3:produce_settlers" in text
    assert "REFRESHED: city:3:produce_warriors" in text


def test_repair_prompt_unknown_class_raises():
    with pytest.raises(prompts.PromptError):
        repair_prompt("transport_error", detail="x", failed_refs="y",
                      refreshed_candidates="z")


def test_repair_messages_structure():
    original = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "turn"}]
    messages = repair_messages(
        original, "{bad output}", "schema_error",
        detail="no JSON object", failed_refs="(none)",
        refreshed_candidates="TABLE")
    assert len(messages) == len(original) + 2
    assert messages[:2] == original  # the original conversation is preserved
    assert messages[2] == {"role": "assistant", "content": "{bad output}"}
    assert messages[3]["role"] == "user"
    assert "SCHEMA ERROR" in messages[3]["content"]
    assert "TABLE" in messages[3]["content"]


def test_render_template_missing_field_raises():
    with pytest.raises(PromptRenderError):
        render_template("repair_schema_error_v1", error_detail="only one")


def test_render_template_json_braces_are_literal():
    """JSON braces in templates are literal text, never placeholders."""
    text = render_template(
        "repair_schema_error_v1",
        error_detail="d", failed_refs="f", refreshed_candidates="c")
    assert '{"ref": "<candidate_id>"}' in text


# ---------------------------------------------------------------------------
# Fingerprint probe prompt (decision 8)
# ---------------------------------------------------------------------------


def test_fingerprint_probe_prompt_is_fixed():
    text = fingerprint_probe_prompt()
    assert text == load_prompt(FINGERPRINT_PROBE)
    # v2 (capacity-revealing) field labels — not the v1 verbatim echo, which
    # is retained on disk for history (FW-21)
    for label in ("MODEL:", "LAB:", "PRODUCT:", "CAPITAL:", "SEQ:"):
        assert label in text
    # stable hash (recorded in the manifest)
    assert prompt_hash(FINGERPRINT_PROBE) == RECORDED_HASHES[FINGERPRINT_PROBE]


# ---------------------------------------------------------------------------
# No secrets in any template
# ---------------------------------------------------------------------------


def test_no_template_references_the_api_key_or_secrets():
    for path in sorted(PROMPT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        assert "MISTRAL_API_KEY" not in text, f"{path.name} references the key var"
        assert "api_key" not in text.lower(), f"{path.name} references api_key"
        assert "sk-" not in text, f"{path.name} contains a key-shaped string"


# ---------------------------------------------------------------------------
# FW-13 budget sub-items: the fixed-text startup invariant + the exception
# seam (canonical PromptBudgetExceeded in core, re-exported here)
# ---------------------------------------------------------------------------


def test_check_fixed_prompt_budget_fits_and_is_deterministic():
    """FW-13 startup invariant: the fixed text (system prompt + stable turn
    prefix) is trivially small today — the check returns its cost, within
    the repair-conversation budget (the only check whose sum includes the
    fixed text), deterministically."""
    cost = check_fixed_prompt_budget()
    assert 0 < cost <= REPAIR_CONVERSATION_TOKEN_BUDGET
    assert check_fixed_prompt_budget() == cost


def test_check_fixed_prompt_budget_raises_config_error(monkeypatch):
    """A fixed text that alone exceeds the repair-conversation budget is a
    startup configuration error — PromptConfigError, fail fast at startup,
    not a mid-game kill."""
    monkeypatch.setattr(prompts, "REPAIR_CONVERSATION_TOKEN_BUDGET", 1)
    with pytest.raises(PromptConfigError):
        check_fixed_prompt_budget()


def test_prompt_snapshot_freezes_turn_and_repair_after_source_change(monkeypatch):
    snapshot = capture_prompt_snapshot("gameplay_v4", bundle_id="gameplay_v4")
    before_turn = turn_prompt(memory="m", snapshot="s", delta="d",
                              candidate_summary="c", profile="gameplay_v4",
                              bundle_id="gameplay_v4",
                              prompt_snapshot=snapshot)
    before_repair = repair_prompt(
        "schema_error", detail="bad", failed_refs="x",
        refreshed_candidates="c", profile="gameplay_v4",
        bundle_id="gameplay_v4", prompt_snapshot=snapshot)
    # A live client must keep using captured bytes even if the loader/source
    # changes after startup.  This also avoids touching any active bundle.
    monkeypatch.setattr(prompts, "load_prompt",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("snapshot reread source")))
    assert turn_prompt(memory="m", snapshot="s", delta="d",
                       candidate_summary="c", profile="gameplay_v4",
                       bundle_id="gameplay_v4",
                       prompt_snapshot=snapshot) == before_turn
    assert repair_prompt(
        "schema_error", detail="bad", failed_refs="x",
        refreshed_candidates="c", profile="gameplay_v4",
        bundle_id="gameplay_v4", prompt_snapshot=snapshot) == before_repair
    assert snapshot.manifest_hashes == capture_prompt_snapshot(
        "gameplay_v4", bundle_id="gameplay_v4").manifest_hashes
    with pytest.raises(TypeError):
        snapshot.texts["system_gameplay_v4"] = "mutated"


def test_gameplay_snapshot_uses_global_hybrid_prefix_with_bundle_history():
    snapshot = capture_prompt_snapshot("gameplay_v4", bundle_id="gameplay_v4")
    assert snapshot.texts[prompts.HYBRID_TURN_PREFIX] == load_prompt(
        prompts.HYBRID_TURN_PREFIX)
    rendered = turn_prompt(memory="m", history="prior", snapshot="s",
                           delta="d", candidate_summary="c",
                           profile="gameplay_v4", bundle_id="gameplay_v4",
                           prompt_snapshot=snapshot)
    assert "HYBRID CONTEXT (stable)" in rendered


def test_selected_snapshot_budget_is_checked_not_default_profile():
    base = capture_prompt_snapshot("gameplay_v4", bundle_id="gameplay_v4")
    texts = dict(base.texts)
    texts["system_gameplay_v4"] = "x" * 100_000
    oversized = PromptSnapshot(base.profile, base.bundle_id, texts,
                               dict(base.hashes))
    with pytest.raises(PromptConfigError):
        check_fixed_prompt_budget("gameplay_v4", bundle_id="gameplay_v4",
                                  snapshot=oversized)


def test_assembled_budget_constants_split_documents_both_sums():
    """W1-7: the former single ASSEMBLED_PROMPT_TOKEN_BUDGET guarded two
    different sums. The section-body budget is the sum of the four
    per-section budgets (what the initial check sums); the repair-conversation
    budget keeps the pre-split value (what the repair check sums against —
    the complete conversation, fixed text included). No live behavior change."""
    assert ASSEMBLED_SECTION_BODY_TOKEN_BUDGET == (
        prompts.MEMORY_TOKEN_BUDGET + prompts.SNAPSHOT_TOKEN_BUDGET
        + prompts.DELTA_TOKEN_BUDGET + prompts.CANDIDATE_TABLE_TOKEN_BUDGET)
    # Semantics-preserving split: the repair check keeps the pre-split value,
    # currently equal to the section-body sum.
    assert REPAIR_CONVERSATION_TOKEN_BUDGET == 11_500
    assert REPAIR_CONVERSATION_TOKEN_BUDGET == ASSEMBLED_SECTION_BODY_TOKEN_BUDGET


def test_prompt_budget_exceeded_is_canonical_in_core():
    """The exception seam: lechonk.model.prompts re-exports the canonical
    core class (the runner catches the same object without importing the
    model package)."""
    from lechonk.core.model_client import PromptBudgetExceeded as canonical
    assert prompts.PromptBudgetExceeded is canonical
    assert issubclass(canonical, RuntimeError)


def test_gameplay_profile_is_opt_in_and_scoped_hashes_are_stable():
    legacy = prompt_manifest_hashes()
    gameplay = active_prompt_hashes("gameplay_v1")
    assert legacy == prompt_manifest_hashes("legacy_eval")
    assert "system_gameplay_v1" not in legacy
    assert set(gameplay) == {
        "system_gameplay_v1", "turn_prefix_gameplay_v1",
        "repair_schema_gameplay_v1", "repair_state_fog_gameplay_v1",
        "repair_illegal_in_rule_gameplay_v1", FINGERPRINT_PROBE,
    }
    assert system_prompt("gameplay_v1") != system_prompt()
    assert turn_prompt(memory="m", snapshot="s", delta="d",
                       candidate_summary="c") != turn_prompt(
                           memory="m", snapshot="s", delta="d",
                           candidate_summary="c", profile="gameplay_v1")


def test_gameplay_v3_profile_and_candidate_budget_are_versioned():
    hashes = active_prompt_hashes("gameplay_v3")
    assert set(hashes) == {
        "system_gameplay_v3", "turn_prefix_gameplay_v3",
        "repair_schema_gameplay_v3", "repair_state_fog_gameplay_v3",
        "repair_illegal_in_rule_gameplay_v3", FINGERPRINT_PROBE,
    }
    assert candidate_table_budget("gameplay_v2") == CANDIDATE_TABLE_TOKEN_BUDGET
    assert candidate_table_budget("gameplay_v3") == 8_000
    assert assembled_section_body_budget("gameplay_v3") == 18_300
    assert repair_conversation_budget("gameplay_v3") == 24_000
    assert repair_conversation_budget("gameplay_v3", hybrid=True) == 28_000
    assert check_fixed_prompt_budget("gameplay_v3") > 0
    assert hashes["system_gameplay_v3"] == (
        "sha256:19810489ed6b228632b7b4eb6c58b8a61052d1db33b42f2390f90b067c9f0603")
    assert hashes["turn_prefix_gameplay_v3"] == (
        "sha256:1e22c3034328fb8af01bc67806741f944ded83619292fa5f23c7c5aecee3d8fc")
    assert hashes["repair_schema_gameplay_v3"] == (
        "sha256:1649819b77a1154b0573bc059d5d8eefc40f8110b2bba7e721a5ba5230d2beba")
    assert hashes["repair_state_fog_gameplay_v3"] == (
        "sha256:1616affee2dec1e5357c6bdf89c022f472fa680cd64ce4ac588bb8ba37760b40")
    assert hashes["repair_illegal_in_rule_gameplay_v3"] == (
        "sha256:ece8374c1eb624d4fa21794ca13962005851bd4ac426c2304b3c2e67375a5f0a")
    with pytest.raises(PromptConfigError):
        repair_conversation_budget("gameplay_v3", context_policy="legacy")


def test_gameplay_v3_directory_bundle_reads_real_files():
    bundled = prompt_manifest_hashes("gameplay_v3", bundle_id="gameplay_v3")
    active = prompt_manifest_hashes("gameplay_v3")
    assert bundled == active
    assert system_prompt("gameplay_v3", bundle_id="gameplay_v3") == \
        system_prompt("gameplay_v3")
    assert turn_prompt(memory="m", snapshot="s", delta="d",
                       candidate_summary="c", profile="gameplay_v3",
                       bundle_id="gameplay_v3") == turn_prompt(
                           memory="m", snapshot="s", delta="d",
                           candidate_summary="c", profile="gameplay_v3")
    for error_class in ("schema_error", "state_fog_error", "illegal_in_rule"):
        kwargs = dict(error_class=error_class, detail="d", failed_refs="f",
                      refreshed_candidates="c", profile="gameplay_v3")
        assert repair_prompt(**kwargs, bundle_id="gameplay_v3") == \
            repair_prompt(**kwargs)


def test_gameplay_v4_profile_and_directory_bundle_are_pinned():
    hashes = active_prompt_hashes("gameplay_v4")
    assert set(hashes) == {
        "system_gameplay_v4", "turn_prefix_gameplay_v4",
        "repair_schema_gameplay_v4", "repair_state_fog_gameplay_v4",
        "repair_illegal_in_rule_gameplay_v4", FINGERPRINT_PROBE,
    }
    assert hashes["system_gameplay_v4"] == (
        "sha256:0d3ed059cda96f14cb10f8d455f4c821ae129b5220b0c903596442775346ca98")
    assert hashes["turn_prefix_gameplay_v4"] == (
        "sha256:7d25f8c9967127397d9edbf16d74c00f717c3893a5d12da66bd033513d309937")
    assert prompt_manifest_hashes("gameplay_v4", bundle_id="gameplay_v4") == hashes
    assert candidate_table_budget("gameplay_v4") == 8_000
    assert assembled_section_body_budget("gameplay_v4") == 18_300
    assert repair_conversation_budget("gameplay_v4", context_policy="conversation_v1") == 24_000
    assert system_prompt("gameplay_v4", bundle_id="gameplay_v4") == system_prompt("gameplay_v4")


def test_gameplay_v5_is_opt_in_and_differs_only_in_turn_guidance():
    v4 = capture_prompt_snapshot("gameplay_v4", bundle_id="gameplay_v4")
    v5 = capture_prompt_snapshot("gameplay_v5", bundle_id="gameplay_v5")
    assert {name.replace("gameplay_v5", "gameplay_v4") for name in v5.manifest_hashes} == set(v4.manifest_hashes)
    assert v5.texts["system_gameplay_v5"] == v4.texts["system_gameplay_v4"]
    assert v5.texts["repair_schema_gameplay_v5"] == v4.texts["repair_schema_gameplay_v4"]
    assert v5.texts["repair_state_fog_gameplay_v5"] == v4.texts["repair_state_fog_gameplay_v4"]
    assert v5.texts["repair_illegal_in_rule_gameplay_v5"] == v4.texts["repair_illegal_in_rule_gameplay_v4"]
    assert v5.texts["turn_prefix_gameplay_v5"] != v4.texts["turn_prefix_gameplay_v4"]
    assert "Use one bounded decision pass" in v5.texts["turn_prefix_gameplay_v5"]
    assert candidate_table_budget("gameplay_v5") == 8_000
    assert repair_conversation_budget("gameplay_v5", context_policy="conversation_v1") == 24_000


def test_gameplay_v3_accepts_larger_candidate_surface_but_v2_does_not():
    body = "candidate " * 3_000
    rendered = turn_prompt(memory="m", snapshot="s", delta="d",
                           candidate_summary=body, profile="gameplay_v3")
    assert "≤ 8,000 tokens" in rendered
    with pytest.raises(PromptBudgetExceeded):
        turn_prompt(memory="m", snapshot="s", delta="d",
                    candidate_summary=body, profile="gameplay_v2")


def test_frozen_gameplay_v2_bundles_dispatch_without_mutating_profile():
    original = system_prompt("gameplay_v2", bundle_id="gameplay_v2_original")
    fixed = system_prompt("gameplay_v2", bundle_id="gameplay_v2_turnid_fix")
    assert original != fixed
    assert "turn_id" in fixed
    assert prompt_manifest_hashes(
        "gameplay_v2", bundle_id="gameplay_v2_original")[
            "system_gameplay_v2"] == (
        "sha256:dde0063c27abce2405bf92676cf96bc297449a5cd7f90770541849a97482775b")
    assert prompt_manifest_hashes(
        "gameplay_v2", bundle_id="gameplay_v2_turnid_fix")[
            "system_gameplay_v2"] == prompt_manifest_hashes(
                "gameplay_v2")["system_gameplay_v2"]


def test_gameplay_v6_uses_frozen_v4_bytes_and_12000_candidate_budget():
    from pathlib import Path
    v4_dir = Path(__file__).parents[1] / "src/lechonk/model/prompts/gameplay_v4"
    v6_dir = Path(__file__).parents[1] / "src/lechonk/model/prompts/gameplay_v6"
    names = ("system.txt", "turn_prefix.txt", "repair_schema.txt",
             "repair_state_fog.txt", "repair_illegal_in_rule.txt")
    assert v6_dir.is_dir()
    assert all((v6_dir / name).read_bytes() == (v4_dir / name).read_bytes()
               for name in names)
    assert prompts.system_prompt("gameplay_v6", bundle_id="gameplay_v6") == prompts.system_prompt("gameplay_v4", bundle_id="gameplay_v4")
    v6_turn = prompts.turn_prompt(memory="m", snapshot="s", delta="d", candidate_summary="c", profile="gameplay_v6", bundle_id="gameplay_v6")
    v4_turn = prompts.turn_prompt(memory="m", snapshot="s", delta="d", candidate_summary="c", profile="gameplay_v4", bundle_id="gameplay_v4")
    assert v6_turn.replace("≤ 12,000 tokens", "≤ 8,000 tokens") == v4_turn
    assert prompts.candidate_table_budget("gameplay_v6") == 12_000
    assert "max_tokens" not in prompts.system_prompt("gameplay_v6", bundle_id="gameplay_v6")


def test_persisted_system_message_must_match_selected_bundle():
    messages = [{"role": "system", "content":
                 system_prompt("gameplay_v2",
                               bundle_id="gameplay_v2_original")}]
    validate_conversation_system_message(
        messages, profile="gameplay_v2", bundle_id="gameplay_v2_original")
    with pytest.raises(PromptError):
        validate_conversation_system_message(
            messages, profile="gameplay_v2",
            bundle_id="gameplay_v2_turnid_fix")
