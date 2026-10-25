# SPDX-License-Identifier: MIT
"""Schema tests: ActionBatch raw-coordinate rejection, candidate-ID refs,
TranscriptRecord union, RunManifest draft, Snapshot rendering/budgets."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from lechonk.core.schemas import (
    ACTION_BATCH_MAX_ACTIONS,
    CANDIDATE_TABLE_TOKEN_BUDGET,
    COMMITMENT_TARGET_KINDS,
    ActionBatch,
    ActionItem,
    Candidate,
    CandidateTable,
    CandidateTableCapExceeded,
    Commitment,
    CommitmentRecord,
    Delta,
    DeltaEvent,
    CityView,
    OmittedDisclosure,
    RunManifest,
    RunManifestRecord,
    MapView,
    OutcomeRecord,
    PlayerView,
    ResearchView,
    Snapshot,
    TranscriptRecord,
    TurnStartRecord,
    estimate_tokens,
    is_candidate_id,
)


# ---------------------------------------------------------------------------
# ActionBatch: raw coordinates are a schema violation (§7.3)
# ---------------------------------------------------------------------------


def test_action_batch_accepts_candidate_refs():
    batch = ActionBatch(
        turn_id="run1/t1",
        actions=[{"ref": "unit:7:goto_n"}, {"ref": "city:3:produce_settlers"},
                 {"ref": "tech:player:research_alphabet"},
                 {"ref": "dipl:2:stop_negotiation"}],
        commitments=[{"domain": "research", "target": "Pottery",
                      "deadline_turn": 5}],
        finish_requested=True,
        notes="settle north",
    )
    assert len(batch.actions) == 4
    assert batch.actions[0].ref == "unit:7:goto_n"
    assert batch.commitments[0].deadline_turn == 5


@pytest.mark.parametrize("bad_action", [
    {"ref": "(12, 34)"},                 # raw coordinate pair
    {"ref": "12,34"},                    # bare coordinates
    {"ref": "move to (12, 34)"},         # free-text move with coordinates
    {"ref": "goto 12 34"},               # free-text move
    {"ref": "unit:7:goto_n", "x": 12, "y": 34},   # ref + raw coordinate fields
    {"ref": "unit:7:goto_n", "to": [12, 34]},
    {"action": "goto", "target": {"x": 12, "y": 34}},  # no ref at all
    {"ref": "unit:7:goto_n", "target_tile": "12,34"},
    {"ref": ""},                          # empty
    {"ref": "unit:7"},                    # malformed id
    {"ref": "move:12:34"},                # wrong domain
])
def test_action_batch_rejects_raw_coordinate_actions(bad_action):
    with pytest.raises(ValidationError):
        ActionBatch(turn_id="run1/t1", actions=[bad_action])


def test_action_batch_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        ActionBatch(turn_id="run1/t1", actions=[],
                    **{"move_unit": {"id": 7, "to": [12, 34]}})


def test_action_batch_max_twenty_actions():
    ok = ActionBatch(turn_id="t", actions=[
        {"ref": f"unit:{i}:goto_n"} for i in range(ACTION_BATCH_MAX_ACTIONS)])
    assert len(ok.actions) == ACTION_BATCH_MAX_ACTIONS
    with pytest.raises(ValidationError):
        ActionBatch(turn_id="t", actions=[
            {"ref": f"unit:{i}:goto_n"}
            for i in range(ACTION_BATCH_MAX_ACTIONS + 1)])


def test_action_item_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ActionItem.model_validate({"ref": "city:1:produce_warriors",
                                   "priority": 1})


def test_is_candidate_id_helper():
    assert is_candidate_id("unit:7:goto_n")
    assert is_candidate_id("tech:player:research_alphabet")
    assert not is_candidate_id("(12, 34)")
    assert not is_candidate_id("12,34")
    assert not is_candidate_id("unit:7:goto n")
    assert not is_candidate_id(12)
    # C5: a trailing newline must not slip past the end anchor ($ vs \Z).
    assert not is_candidate_id("unit:7:goto_n\n")


def test_action_item_rejects_trailing_newline_ref():
    # C5: pydantic Field pattern must reject a newline-terminated ref.
    with pytest.raises(ValidationError):
        ActionItem.model_validate({"ref": "unit:7:goto_n\n"})


# ---------------------------------------------------------------------------
# CandidateTable
# ---------------------------------------------------------------------------


def test_candidate_table_roundtrip():
    table = CandidateTable(turn=3, candidates=[
        Candidate(id="unit:7:goto_n", domain="unit", actor_id="7",
                  verb="goto_n", summary="Settlers move N",
                  payload={"dir": "N"}),
        Candidate(id="city:3:produce_settlers", domain="city", actor_id="3",
                  verb="produce_settlers"),
    ])
    assert table.by_id()["unit:7:goto_n"].verb == "goto_n"
    with pytest.raises(ValidationError):
        Candidate(id="(1, 2)", domain="unit", actor_id="1", verb="goto_n")
    with pytest.raises(ValidationError):
        Candidate(id="unit:1:goto_n", domain="unit", actor_id="1",
                  verb="goto_n", bogus=1)  # extra="forbid"


# ---------------------------------------------------------------------------
# FW-13: the candidate-table cap — capping, in-contact retention, cap error
# ---------------------------------------------------------------------------

def _fw13_candidate(i, summary_chars, in_contact=False):
    return Candidate(
        id=f"unit:9{i:02d}:wait",
        domain="unit",
        actor_id=f"9{i:02d}",
        verb="wait",
        summary="y" * summary_chars,
        in_contact=in_contact,
    )


def test_oversized_candidate_table_renders_capped_with_omission_line():
    """FW-13 acceptance: an oversized table renders capped — the
    in-contact candidates are always retained, the rest are admitted in
    table order while they fit, and the trailing omission line discloses
    the elided count with the honest policy wording."""
    total = 10
    table = CandidateTable(turn=3, candidates=[
        _fw13_candidate(i, summary_chars=760, in_contact=i in (0, 5))
        for i in range(total)])
    offered = table.offered()
    assert len(offered.candidates) < total  # the cap engaged
    kept_ids = {c.id for c in offered.candidates}
    # in-contact candidates are never elided
    assert "unit:900:wait" in kept_ids
    assert "unit:905:wait" in kept_ids
    # the trailing omission line discloses the elision honestly
    omitted = total - len(offered.candidates)
    assert omitted > 0
    rendered = table.render()
    assert rendered.endswith(
        f"candidates omitted: {omitted} (beyond budget, table order)")
    assert "lowest priority" not in rendered
    # the capped table (with its omission line) fits the budget
    assert offered.estimated_tokens() <= CANDIDATE_TABLE_TOKEN_BUDGET


def test_candidate_cap_raises_when_in_contact_alone_exceed_budget():
    """FW-13: in-contact candidates alone over CANDIDATE_TABLE_TOKEN_BUDGET
    are an explicit misconfiguration — CandidateTableCapExceeded, never a
    silent drop. A non-protected candidate of the same size is instead
    elided deterministically."""
    huge = "x" * (CANDIDATE_TABLE_TOKEN_BUDGET * 5)
    table = CandidateTable(turn=1, candidates=[
        Candidate(id="unit:1:hold", domain="unit", actor_id="1",
                  verb="hold", summary=huge, in_contact=True),
    ])
    with pytest.raises(CandidateTableCapExceeded):
        table.offered()

    elided = CandidateTable(turn=1, candidates=[
        Candidate(id="unit:2:wait", domain="unit", actor_id="2",
                  verb="wait", summary=huge),
    ])
    offered = elided.offered()
    assert offered.candidates == []
    assert elided.render() == "candidates omitted: 1 (beyond budget, table order)"


def test_offered_counts_exactly_the_admitted_lines():
    """Finding-3 accounting fix: `used` counts exactly the admitted lines'
    tokens. Pre-fix, `used` started at the protected-lines sum and then
    re-added each in-contact line in the loop — double-counting contact
    candidates and eliding more non-contact candidates than the budget
    requires. Here the 1104-token in-contact line plus the first 54-token
    non-contact line (plus the 13-token omission reserve) fits the
    1,200-token cap with the fix; the second non-contact line does not fit
    and is elided. Pre-fix, the doubled contact total (2,208) alone
    exceeded the budget, so both non-contact lines were dropped."""
    table = CandidateTable(turn=1, candidates=[
        Candidate(id="unit:1:hold", domain="unit", actor_id="1",
                  verb="hold", summary="a" * 4400, in_contact=True),
        Candidate(id="unit:2:wait", domain="unit", actor_id="2",
                  verb="wait", summary="b" * 200),
        Candidate(id="unit:3:wait", domain="unit", actor_id="3",
                  verb="wait", summary="b" * 200),
        Candidate(id="unit:4:wait", domain="unit", actor_id="4",
                  verb="wait", summary="b" * 200),
    ])
    offered = table.offered()
    assert [c.id for c in offered.candidates] == ["unit:1:hold",
                                                  "unit:2:wait"]
    assert table.render().endswith(
        "candidates omitted: 2 (beyond budget, table order)")


# ---------------------------------------------------------------------------
# TranscriptRecord union + RunManifest (drafts, frozen in WP2)
# ---------------------------------------------------------------------------


def test_transcript_record_union_parses_each_kind():
    adapter = TypeAdapter(TranscriptRecord)
    outcome = adapter.validate_python({
        "seq": 1, "ts": "2026-10-24T12:00:00Z", "checksum": "abc",
        "kind": "outcome", "outcome": "TRUNCATED", "reason": "infra timeout"})
    assert isinstance(outcome, OutcomeRecord)
    assert outcome.outcome == "TRUNCATED"

    turn = adapter.validate_python({
        "seq": 2, "ts": "2026-10-24T12:00:01Z", "checksum": "def",
        "kind": "turn_start", "turn": 4})
    assert isinstance(turn, TurnStartRecord)
    assert turn.turn == 4

    manifest = RunManifest(run_id="run1", model_id="mistral-large-4",
                           seeds={"mapseed": 42, "gameseed": 42,
                                  "agentseed": 42},
                           extra_field_for_wp2="allowed (extra=allow)")
    rec = adapter.validate_python({
        "seq": 0, "ts": "2026-10-24T12:00:00Z", "checksum": "ghi",
        "kind": "run_manifest", "manifest": manifest.model_dump()})
    assert isinstance(rec, RunManifestRecord)
    assert rec.manifest.model_id == "mistral-large-4"
    assert rec.manifest.extra_field_for_wp2 == "allowed (extra=allow)"


def test_run_manifest_draft_fields():
    m = RunManifest(
        image_digest="sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f"
                     "37a8e943faf920cb373a",
        vendored_base_sha="1b144d6", patch_head_sha="36ae456",
        ruleset="classic",
        server_settings={"phasemode": "player", "timeout": 0},
    )
    assert m.seeds == {}
    assert m.model_id is None


# ---------------------------------------------------------------------------
# Snapshot / Delta rendering + budgets
# ---------------------------------------------------------------------------


def _mini_snapshot() -> Snapshot:
    return Snapshot(
        turn=1,
        player=PlayerView(player_id=0, name="Caesar", gold=50,
                          research=ResearchView(current_target="Alphabet",
                                                bulbs_researched=3,
                                                cost_total=20, turns_left=5)),
        research_candidates=[],
        cities=[],
        units=[],
        map_view=MapView(width=6, height=6,
                         tile_counts={"visible": 13, "last_known": 1,
                                      "unexplored": 22}),
        diplomacy=[],
        messages=["Welcome, Caesar ruler of the Romans empire."],
        omitted=OmittedDisclosure(),
    )


def test_snapshot_render_is_json_and_token_estimate():
    snap = _mini_snapshot()
    rendered = snap.render()
    parsed = json.loads(rendered)
    assert parsed["turn"] == 1
    assert parsed["player"]["research"]["current_target"] == "Alphabet"
    assert snap.estimated_tokens() == estimate_tokens(rendered)
    assert snap.estimated_tokens() >= 1


def test_snapshot_render_rejects_nan():
    # FW-09: a NaN in the payload (pydantic float fields admit it) raises a
    # deterministic ValueError on render — never a bare NaN token.
    snap = _mini_snapshot()
    snap.cities = [CityView(id=1, name="Roma", x=2, y=3, is_own=True,
                            production_progress=float("nan"))]
    with pytest.raises(ValueError):
        snap.render()


def test_snapshot_render_clean_payloads_byte_identical():
    # FW-09 regression: the strict helper must not change the bytes for
    # clean payloads (the policy-facing surface).
    snap = _mini_snapshot()
    assert snap.render() == json.dumps(
        snap.model_dump(mode="json"), separators=(",", ":"),
        ensure_ascii=False)


def test_delta_render_clean_payloads_byte_identical():
    delta = Delta(turn=2, events=[
        DeltaEvent(kind="unit_changed", subject="unit:7", detail="moved N")])
    assert delta.render() == json.dumps(
        delta.model_dump(mode="json"), separators=(",", ":"),
        ensure_ascii=False)


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_delta_draft_renders():
    delta = Delta(turn=2, events=[
        DeltaEvent(kind="unit_changed", subject="unit:7",
                   detail="moved N"),
        DeltaEvent(kind="research_complete", subject="Alphabet"),
    ])
    parsed = json.loads(delta.render())
    assert parsed["events"][0]["kind"] == "unit_changed"
    assert delta.estimated_tokens() >= 1
    # a foreign unit vanishing is never a "lost unit" event
    with pytest.raises(ValidationError):
        DeltaEvent(kind="unit_lost_foreign")


def test_commitment_schema():
    c = Commitment(domain="expansion", target="city:4", deadline_turn=9)
    assert c.domain == "expansion"
    with pytest.raises(ValidationError):
        Commitment(domain="banana", target="x", deadline_turn=1)


# ---------------------------------------------------------------------------
# FW-10: typed commitment targets + additive verb (dated addendum
# artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/)
# ---------------------------------------------------------------------------


def test_commitment_fw10_additivity_only_named_fields_added():
    """FW-10 additivity: exactly the two named fields were added to the
    frozen Commitment field set — target_kind (default "any") and verb
    (default None). Nothing removed or renamed; extra fields still
    forbidden."""
    assert set(Commitment.model_fields) == {
        "domain", "target", "deadline_turn", "target_kind", "verb"}
    assert Commitment.model_fields["target_kind"].default == "any"
    assert Commitment.model_fields["verb"].default is None
    assert COMMITMENT_TARGET_KINDS == (
        "tech", "production", "unit", "city", "player", "tile", "any")
    # unknown kinds and unknown fields are still schema errors
    with pytest.raises(ValidationError):
        Commitment(domain="research", target="Pottery", deadline_turn=5,
                   target_kind="banana")
    with pytest.raises(ValidationError):
        Commitment(domain="research", target="Pottery", deadline_turn=5,
                   bogus=1)


def test_commitment_fw10_legacy_payload_loads_with_legacy_defaults():
    """FW-10 legacy fallback: a pre-addendum payload (the old field shape,
    no target_kind/verb) loads unchanged and lands on the documented
    legacy defaults — kind "any", verb None."""
    payload = {"domain": "research", "target": "Pottery", "deadline_turn": 9}
    c = Commitment.model_validate(payload)
    assert (c.domain, c.target, c.deadline_turn) == ("research", "Pottery", 9)
    assert c.target_kind == "any"
    assert c.verb is None
    # the legacy shape also round-trips through the frozen transcript union
    rec = CommitmentRecord(turn=1, commitments=[payload], threats=[],
                           call_seq=None)
    loaded = rec.commitments[0]
    assert (loaded.target_kind, loaded.verb) == ("any", None)
    adapter = TypeAdapter(TranscriptRecord)
    parsed = adapter.validate_python(
        {"seq": 1, "ts": "2026-10-24T12:00:00Z", "checksum": "x",
         "kind": "commitment", "turn": 1, "commitments": [payload],
         "threats": [], "call_seq": None})
    assert parsed.commitments[0].target_kind == "any"


def test_commitment_fw10_typed_target_and_verb_roundtrip():
    """FW-10: typed targets + verb validate, round-trip through the dump,
    and ride both the ActionBatch and memory_update commitment lists."""
    c = Commitment(domain="military", target_kind="unit", target="unit:7",
                   verb="attack", deadline_turn=12)
    assert (c.target_kind, c.target, c.verb) == ("unit", "unit:7", "attack")
    dumped = c.model_dump()
    assert dumped["target_kind"] == "unit" and dumped["verb"] == "attack"
    batch = ActionBatch(
        turn_id="run1/t1", actions=[],
        commitments=[{"domain": "research", "target_kind": "tech",
                      "target": "Pottery", "verb": "research",
                      "deadline_turn": 5}],
        memory_update={"commitments": [dumped]})
    assert batch.commitments[0].target_kind == "tech"
    assert batch.commitments[0].verb == "research"
    assert batch.memory_update.commitments[0].verb == "attack"
    # the "*" wildcard target is retained on every kind
    for kind in COMMITMENT_TARGET_KINDS:
        assert Commitment(domain="other", target_kind=kind, target="*",
                          deadline_turn=3).target == "*"
