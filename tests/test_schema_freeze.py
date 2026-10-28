# SPDX-License-Identifier: MIT
"""Schema-freeze tests (WP2): the frozen TranscriptRecord union and the
schema-metric coverage contract.

The WP2 acceptance (design-plan-freeciv-v1.1.md §4 line 54) requires the
schema-metric coverage check BEFORE the freeze: every §3 metric's required
inputs must be schema-covered. The enumeration itself is persisted as an
artifact (artifacts/runs/wp2-schema-coverage/schema-metric-coverage.md);
these tests pin the machine-checkable part of that contract: the frozen
record kinds, the per-record seq/ts/checksum, the outcome taxonomy, and
the exact fields each metric reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from lechonk.core.schemas import (
    OUTCOME_KINDS,
    PROVENANCE_CLASSES,
    READBACK_STATUSES,
    REJECTION_CLASSES,
    TRANSCRIPT_KINDS,
    ActionResultRecord,
    CandidatesRefRecord,
    CommitmentRecord,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    RunManifest,
    RunManifestRecord,
    SnapshotRefRecord,
    ThreatMention,
    TranscriptRecord,
    TurnStartRecord,
)

_ADAPTER = TypeAdapter(TranscriptRecord)


def test_frozen_record_kinds_are_exactly_the_73_list():
    assert TRANSCRIPT_KINDS == (
        "run_manifest", "turn_start", "snapshot_ref", "candidates_ref",
        "model_call", "action_result", "commitment", "metric_event",
        "outcome")
    # Every kind parses through the union and round-trips.
    samples = {
        "run_manifest": {"kind": "run_manifest",
                         "manifest": RunManifest(run_id="r").model_dump()},
        "turn_start": {"kind": "turn_start", "turn": 4},
        "snapshot_ref": {"kind": "snapshot_ref", "turn": 4,
                         "snapshot_hash": "h", "token_count": 10},
        "candidates_ref": {"kind": "candidates_ref", "turn": 4,
                           "table_hash": "h", "candidate_count": 3,
                           "by_domain": {"unit": 2, "tech": 1},
                           "opportunity_domains": ["research", "unit_action"]},
        "model_call": {"kind": "model_call", "call_id": "c1", "attempt": 1,
                       "prompt_hash": "ph", "latency_ms": 5,
                       "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0,
                       "fingerprint": "fp"},
        "action_result": {"kind": "action_result", "turn": 4,
                          "ref": "unit:7:goto_n", "domain": "unit",
                          "verb": "goto_n", "target": "tile:2,1",
                          "affected": "tile:2,1", "actor_tile": [1, 1],
                          "outcome": "executed", "provenance_class": "model",
                          "readback": "verified"},
        "commitment": {"kind": "commitment", "turn": 4,
                       "commitments": [{"domain": "research",
                                        "target": "Pottery",
                                        "deadline_turn": 9}],
                       "threats": [{"entity_id": "unit:20", "turn": 4,
                                    "tile": [2, 2]}],
                       "call_seq": 3},
        "metric_event": {"kind": "metric_event", "name": "participation",
                         "value": 0.5},
        "outcome": {"kind": "outcome", "outcome": "TRUNCATED",
                    "reason": "infra"},
    }
    assert set(samples) == set(TRANSCRIPT_KINDS)
    for kind, payload in samples.items():
        record = _ADAPTER.validate_python(
            {"seq": 1, "ts": "2026-10-24T12:00:00Z", "checksum": "x",
             **payload})
        assert record.kind == kind
        assert record.seq == 1 and record.ts and record.checksum


def test_every_record_carries_seq_ts_checksum():
    """§3 line 34: one checksummed record per line; seq/ts/checksum are
    ledger-assigned (defaults exist so records construct without them)."""
    record = TurnStartRecord(turn=1)
    assert (record.seq, record.ts, record.checksum) == (0, "", "")
    for cls in (RunManifestRecord, TurnStartRecord, SnapshotRefRecord,
                CandidatesRefRecord, ModelCallRecord, ActionResultRecord,
                CommitmentRecord, MetricEventRecord, OutcomeRecord):
        assert {"seq", "ts", "checksum"} <= set(cls.model_fields)


def test_outcome_taxonomy_is_frozen():
    assert OUTCOME_KINDS == ("COMPLETED", "DEFEATED", "VICTORY", "TRUNCATED")
    with pytest.raises(ValidationError):
        OutcomeRecord(outcome="DRAW")
    rec = OutcomeRecord(outcome="DEFEATED", reason="eliminated")
    assert rec.outcome == "DEFEATED" and rec.reason == "eliminated"


def test_provenance_and_rejection_classes_are_frozen():
    assert PROVENANCE_CLASSES == (
        "model", "continued_model_order", "scripted_default",
        "engine_automatic", "housekeeping")
    assert REJECTION_CLASSES == (
        "schema_error", "state_fog_error", "illegal_in_rule")
    assert READBACK_STATUSES == (
        "verified", "contradicted", "inconclusive", "none")
    with pytest.raises(ValidationError):
        ActionResultRecord(turn=1, ref="unit:1:goto_n", outcome="executed",
                           provenance_class="wizard")
    with pytest.raises(ValidationError):
        ActionResultRecord(turn=1, ref="unit:1:goto_n", outcome="rejected",
                           rejection_class="rude_error")


# ---------------------------------------------------------------------------
# the schema-metric coverage contract: the exact fields each §3 metric reads
# ---------------------------------------------------------------------------


def test_participation_inputs_are_schema_covered():
    """Participation gate: per-turn offered consequential choices
    (candidates_ref.opportunity_domains + turn) and executed model-class
    decisions (action_result turn/outcome/provenance_class/domain/verb)."""
    ref = CandidatesRefRecord(turn=7, table_hash="h", candidate_count=2,
                              by_domain={"tech": 1, "unit": 1},
                              opportunity_domains=["research", "unit_action"])
    assert ref.turn == 7 and ref.opportunity_domains
    action = ActionResultRecord(
        turn=7, ref="tech:cur_player:research_tech_alphabet_0",
        domain="tech", verb="research_tech_alphabet_0", target="Alphabet",
        outcome="executed", provenance_class="model", readback="verified")
    assert action.turn == 7 and action.outcome == "executed"
    assert action.provenance_class == "model"


def test_commitment_followthrough_inputs_are_schema_covered():
    """Commitment follow-through: structured commitments with dedup keys
    and deadline-inside-run censoring (commitment record), matching
    executed actions with structured target + turn, and the
    continued_model_order origin linkage."""
    rec = CommitmentRecord(
        turn=3,
        commitments=[{"domain": "research", "target": "Pottery",
                      "deadline_turn": 9}],
        threats=[{"entity_id": "unit:20", "turn": 3, "tile": [2, 2]}],
        call_seq=11)
    c = rec.commitments[0]
    assert (c.domain, c.target, c.deadline_turn) == ("research", "Pottery", 9)
    action = ActionResultRecord(
        turn=5, ref="tech:cur_player:research_tech_pottery_1", domain="tech",
        verb="research_tech_pottery_1", target="Pottery", outcome="executed",
        provenance_class="continued_model_order", origin_seq=4,
        readback="verified")
    assert action.target == "Pottery" and action.turn == 5
    assert action.origin_seq == 4
    # Threats carry the structured entity ID + tile (no free text; the
    # structural pattern is enforced at emission — the runner drops
    # non-conforming entries rather than parsing them).
    assert rec.threats[0].entity_id == "unit:20"
    assert rec.threats[0].tile == (2, 2)


def test_reason_but_dont_act_inputs_are_schema_covered():
    """Reason-but-don't-act: structured affected/actor_tile descriptors for
    entity/tile/defensive-response matching within the ±2-turn window."""
    action = ActionResultRecord(
        turn=6, ref="unit:11:attack_ne", domain="unit", verb="attack_ne",
        target="unit:20", affected="unit:20", actor_tile=[2, 1],
        outcome="executed", provenance_class="model", readback="verified")
    assert action.affected == "unit:20"
    assert action.actor_tile == (2, 1)
    assert action.verb == "attack_ne"


def test_illegal_action_taxonomy_inputs_are_schema_covered():
    """Illegal-action taxonomy: rejection classes on model-provenance
    attempts, plus the model_call attempt counter (≤2 per turn incl. the
    repair)."""
    for cls in REJECTION_CLASSES:
        rec = ActionResultRecord(turn=1, ref="unit:9:goto_n", outcome="rejected",
                                 rejection_class=cls,
                                 provenance_class="model", readback="none")
        assert rec.rejection_class == cls
    call = ModelCallRecord(call_id="c2", attempt=2, prompt_hash="ph",
                           latency_ms=1, tokens_in=1, tokens_out=1,
                           cost_usd=0.0, fingerprint="fp",
                           error="timeout")
    assert call.attempt == 2 and call.error == "timeout"
    unknown = ModelCallRecord(call_id="c3", attempt=1, prompt_hash="ph",
                              latency_ms=1, tokens_in=None, tokens_out=None,
                              cost_usd=None, fingerprint="fp",
                              usage_status="unavailable", cost_status="unknown")
    assert unknown.tokens_in is None and unknown.cost_usd is None
    assert unknown.usage_status == "unavailable"
    assert unknown.cost_status == "unknown"
    # No attempt cap in the schema: the RUNNER enforces ≤2 total API
    # attempts per turn (no third API attempt, §3 line 33).


def test_provenance_linkage_fields_are_schema_covered():
    """§3 line 38: policy-choice level with snapshot ID, offered candidates,
    choice, call ID, execution result, verification evidence."""
    rec = ActionResultRecord(
        turn=2, ref="city:1:produce_warriors", domain="city",
        verb="produce_warriors", target="Warriors", outcome="executed",
        rejection_class=None, provenance_class="scripted_default",
        origin_seq=None, reason="model unresponsive after the repair cap",
        snapshot_seq=5, candidates_seq=6, call_seq=7, readback="verified",
        detail="read-back verified")
    assert (rec.snapshot_seq, rec.candidates_seq, rec.call_seq) == (5, 6, 7)
    assert rec.reason and rec.readback == "verified"


def test_run_manifest_field_set_is_the_decision_8_list():
    """The RunManifest field set is frozen (decision 8); optionality is
    retained until WP3/eval populates every field at run start."""
    expected = {
        "run_id", "image_digest", "server_commit", "capstring",
        "vendored_base_sha", "patch_head_sha", "harness_revision",
        "python_lockfile", "seeds", "ruleset", "server_settings",
        "model_id", "sampling", "prompt_hashes", "config_after_load",
        "checkpoint_checksums", "model_fingerprint", "arm_start_ts",
    }
    assert expected <= set(RunManifest.model_fields)
    manifest = RunManifest(run_id="r", model_id="mistral-large-4",
                           seeds={"mapseed": 1, "gameseed": 1, "agentseed": 1})
    record = RunManifestRecord(manifest=manifest)
    assert record.manifest.model_id == "mistral-large-4"


# ---------------------------------------------------------------------------
# FW-10 dated additive amendment (2026-10-24): the schema-coverage addendum
# ---------------------------------------------------------------------------

FW10_ADDENDUM = (Path(__file__).parents[1] / "artifacts" / "runs" /
                 "wp2-schema-coverage" / "2026-10-24-verb-typed-targets" /
                 "addendum.md")


def test_fw10_dated_schema_coverage_addendum_exists():
    """FW-10: the dated additive-amendment addendum artifact exists next to
    the frozen coverage check.

    Clone-safety (finding 4): evidence artifacts under artifacts/runs/** are
    LOCAL RECORDED EVIDENCE — untracked by user ruling (commit b1c521d), so
    a fresh clone does not have them. This test pins the addendum's content
    only when the local tree carries it; it is skipped otherwise and must
    never fail a clone."""
    if not FW10_ADDENDUM.is_file():
        pytest.skip(
            "local recorded evidence only: the FW-10 addendum under "
            "artifacts/runs/** is not part of a clone (untracked per user "
            "ruling); the artifact is pinned only when the local tree has it")
    text = FW10_ADDENDUM.read_text(encoding="utf-8")
    # the addendum records the additive change, the verb grammar, and the
    # legacy fallback
    for needle in ("target_kind", "verb", "additive", "legacy",
                   "exact", "wildcard"):
        assert needle in text, f"addendum missing {needle!r}"


def test_fw10_commitment_amendment_is_additive_over_the_freeze():
    """FW-10: the frozen record kinds are untouched and the Commitment
    amendment adds exactly the two named fields (the freeze test above
    still passes; this pins the amendment's field set)."""
    from lechonk.core.schemas import Commitment
    assert set(Commitment.model_fields) == {
        "domain", "target", "deadline_turn", "target_kind", "verb"}
    # a frozen-union commitment record with the legacy payload still parses
    record = _ADAPTER.validate_python(
        {"seq": 1, "ts": "2026-10-24T12:00:00Z", "checksum": "x",
         "kind": "commitment", "turn": 4,
         "commitments": [{"domain": "research", "target": "Pottery",
                          "deadline_turn": 9}],
         "threats": [], "call_seq": 3})
    assert isinstance(record, CommitmentRecord)
    assert record.commitments[0].target_kind == "any"
    assert record.commitments[0].verb is None
