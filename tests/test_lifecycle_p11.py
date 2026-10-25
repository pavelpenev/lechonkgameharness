# SPDX-License-Identifier: MIT
"""WP3 lifecycle tests: the P11 ledger-audit logic (scripts/
wp3_p11_audit.py::audit_transcript), unit-tested offline over synthetic
transcripts written through the real TranscriptLedger.

The audit's duplicate-mutation rule: a candidate ref executed more than
once is only acceptable as an origin-linked continued_model_order with
non-execution proof — the §3 line 35 disconnect-resubmission shape
(first execution uncertain + a preceding mutation_reconciliation
non_execution_verified event) or the standing-order shape (the origin is
a model action_result of a persistent verb on an earlier turn).
Everything else — a blind replay, a resend without proof, an
unorigin-linked repeat — must FAIL the audit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lechonk.core.ledger import TranscriptLedger
from lechonk.core.schemas import (
    ActionResultRecord,
    MetricEventRecord,
    OutcomeRecord,
    TurnStartRecord,
)

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "wp3_p11_audit", REPO / "scripts" / "wp3_p11_audit.py")
wp3_p11_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp3_p11_audit)

REF = "unit:10:goto_e"


def _write(tmp_path, name, records, *, torn_tail=None):
    path = tmp_path / name
    ledger = TranscriptLedger(path)
    for record in records:
        ledger.append(record, durable=True)
    ledger.close()
    if torn_tail:
        with open(path, "ab") as fh:
            fh.write(torn_tail)
    return path


def _action(turn=1, **kwargs):
    fields = dict(turn=turn, ref=REF, domain="unit", verb="goto_e",
                  outcome="executed", provenance_class="model",
                  readback="verified")
    fields.update(kwargs)
    return ActionResultRecord(**fields)


def _recon(ref, decision, seq_note="reconciliation read-back contradicted"):
    return MetricEventRecord(
        name="mutation_reconciliation",
        value=f"{ref}: {decision} ({seq_note})")


def _quarantine(ref, decision="unknown"):
    return MetricEventRecord(
        name="mutation_quarantine",
        value=f"{ref}: {decision} on reconnect — quarantined, never resent")


def test_audit_clean_transcript_passes(tmp_path):
    """One verified execution + a bounded outcome: every check passes."""
    path = _write(tmp_path, "clean.jsonl", [
        TurnStartRecord(turn=1),
        _action(),
        OutcomeRecord(outcome="COMPLETED", reason="horizon reached"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    failed = {n: e for n, e in audit["checks"].items() if not e["pass"]}
    assert not failed, f"unexpected failures: {failed}"
    assert audit["verdict"] == "PASS"
    assert audit["repeat_executions"] == []


def test_audit_blind_replay_fails(tmp_path):
    """The same ref executed twice as plain model actions (a blind
    replay) — no origin-linked continued_model_order, no non-execution
    proof → the audit FAILS it."""
    path = _write(tmp_path, "blind.jsonl", [
        _action(),
        _action(),
        OutcomeRecord(outcome="COMPLETED"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_duplicate_mutation"]["pass"] is False
    assert audit["verdict"] == "FAIL"
    repeat = audit["repeat_executions"][0]
    assert repeat["continued_model_order"] is False
    assert repeat["non_execution_proof"] is None


def test_audit_resend_without_proof_fails(tmp_path):
    """A continued_model_order repeat whose FIRST execution was verified
    (not uncertain) and with no reconciliation event — a resend without
    non-execution proof → FAILS."""
    path = _write(tmp_path, "noproof.jsonl", [
        _action(readback="verified"),
        _action(provenance_class="continued_model_order", origin_seq=1),
        OutcomeRecord(outcome="COMPLETED"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_duplicate_mutation"]["pass"] is False
    assert audit["verdict"] == "FAIL"


def test_audit_unorigin_linked_repeat_fails(tmp_path):
    """A continued_model_order repeat with NO origin linkage → FAILS."""
    path = _write(tmp_path, "noorigin.jsonl", [
        _action(readback="inconclusive"),
        _recon(REF, "non_execution_verified"),
        _action(provenance_class="continued_model_order", origin_seq=None),
        OutcomeRecord(outcome="COMPLETED"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_duplicate_mutation"]["pass"] is False
    assert audit["verdict"] == "FAIL"


def test_audit_reconciliation_resubmission_passes(tmp_path):
    """The §3 line 35 shape: the first execution is uncertain
    (inconclusive), a mutation_reconciliation non_execution_verified event
    settles it, and the ONE resubmission is an origin-linked
    continued_model_order → PASSES (no duplicate mutation)."""
    ledger = TranscriptLedger(tmp_path / "resend.jsonl")
    seq_first = ledger.append(_action(readback="inconclusive"), durable=True)
    ledger.append(_recon(REF, "non_execution_verified"), durable=True)
    ledger.append(
        _action(provenance_class="continued_model_order",
                origin_seq=seq_first, readback="verified"),
        durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED"), durable=True)
    ledger.close()
    audit = wp3_p11_audit.audit_transcript(tmp_path / "resend.jsonl")
    failed = {n: e for n, e in audit["checks"].items() if not e["pass"]}
    assert not failed, f"unexpected failures: {failed}"
    assert audit["verdict"] == "PASS"
    assert len(audit["repeat_executions"]) == 1
    assert audit["repeat_executions"][0]["ok"] is True
    assert "reconciliation" in audit["repeat_executions"][0][
        "non_execution_proof"]


def test_audit_standing_order_continuation_passes(tmp_path):
    """The WP2 standing-order shape: a model sentry on turn 1, the
    pipeline's origin-linked continued_model_order re-issue on turn 2 →
    PASSES (the standing-order non-execution proof)."""
    ledger = TranscriptLedger(tmp_path / "standing.jsonl")
    seq_origin = ledger.append(
        _action(turn=1, ref="unit:11:sentry", verb="sentry",
                readback="verified"),
        durable=True)
    ledger.append(
        _action(turn=2, ref="unit:11:sentry", verb="sentry",
                provenance_class="continued_model_order",
                origin_seq=seq_origin, readback="verified"),
        durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED"), durable=True)
    ledger.close()
    audit = wp3_p11_audit.audit_transcript(tmp_path / "standing.jsonl")
    failed = {n: e for n, e in audit["checks"].items() if not e["pass"]}
    assert not failed, f"unexpected failures: {failed}"
    assert audit["verdict"] == "PASS"
    assert "standing-order" in audit["repeat_executions"][0][
        "non_execution_proof"]


def test_audit_dangling_uncertain_mutation_fails(tmp_path):
    """An uncertain (inconclusive) execution with NO reconciliation event
    afterwards — never settled → FAILS."""
    path = _write(tmp_path, "dangling.jsonl", [
        _action(readback="inconclusive"),
        OutcomeRecord(outcome="TRUNCATED", reason="quarantined"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_dangling_uncertain_mutations"]["pass"] is False
    assert audit["verdict"] == "FAIL"


def test_audit_quarantined_uncertain_is_settled_and_passes(tmp_path):
    """The quarantine shape: the uncertain execution IS settled by a
    mutation_reconciliation event (decision executed/unknown) and a
    mutation_quarantine event — no resend, no dangling → PASSES."""
    path = _write(tmp_path, "quarantine.jsonl", [
        _action(readback="inconclusive"),
        _recon(REF, "executed"),
        _quarantine(REF, "executed"),
        OutcomeRecord(outcome="TRUNCATED",
                      reason="uncertain mutation quarantined after "
                             "disconnect — never resent"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    failed = {n: e for n, e in audit["checks"].items() if not e["pass"]}
    assert not failed, f"unexpected failures: {failed}"
    assert audit["verdict"] == "PASS"
    assert audit["repeat_executions"] == []


def test_audit_torn_tail_fails_intact_check(tmp_path):
    """A torn tail (the pre-recovery mid-write kill state) → the intact
    check FAILS (the audit runs on post-recovery transcripts)."""
    path = _write(tmp_path, "torn.jsonl", [
        _action(),
        OutcomeRecord(outcome="COMPLETED"),
    ], torn_tail=b'{"kind":"action_result","turn":2,"ref":"unit:10:g')
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_torn_tail"]["pass"] is False
    # the complete records still load (the torn tail is discarded).
    assert audit["checks"]["transcript_loads_clean"]["pass"] is True
    assert audit["verdict"] == "FAIL"


def test_audit_unbounded_outcome_fails(tmp_path):
    """No outcome record (or a TRUNCATED without a reason) → the bounded
    outcome check FAILS."""
    path = _write(tmp_path, "nooutcome.jsonl", [
        _action(),
        MetricEventRecord(name="state_transition", value="ACTING->FAILED"),
    ])
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["bounded_outcome_record"]["pass"] is False

    path2 = _write(tmp_path, "noreason.jsonl", [
        _action(),
        OutcomeRecord(outcome="TRUNCATED", reason=""),
    ])
    audit2 = wp3_p11_audit.audit_transcript(path2)
    assert audit2["checks"]["bounded_outcome_record"]["pass"] is False
