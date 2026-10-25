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


# ---------------------------------------------------------------------------
# W3-1 (D5/D15): the pure pending-intent scanner
# (scripts/wp3_p11_audit.py::scan_pending_intents) — resolved / cancelled
# / uncertain / legacy classifications
# ---------------------------------------------------------------------------

from lechonk.core.ledger import load_transcript  # noqa: E402
from lechonk.core.runner import (  # noqa: E402
    _dispatch_cancelled_payload,
    _dispatch_intent_payload,
    _dispatch_result_detail,
)


def _intent(turn=1, ref=REF, call_seq=None, provenance="model"):
    return MetricEventRecord(
        name="dispatch_intent",
        value=_dispatch_intent_payload(
            turn=turn, ref=ref, call_seq=call_seq, snapshot_seq=10,
            candidates_seq=11, provenance=provenance))


def _cancelled(intent_seq, turn=1, ref=REF,
               reason="ACTING deadline reached"):
    return MetricEventRecord(
        name="mutation_dispatch_cancelled",
        value=_dispatch_cancelled_payload(
            intent_seq=intent_seq, reason=reason, turn=turn, ref=ref,
            call_seq=None, snapshot_seq=10, candidates_seq=11,
            provenance="model"))


def _result(intent_seq, turn=1, ref=REF, readback="verified", detail=None):
    return ActionResultRecord(
        turn=turn, ref=ref, domain="unit", verb="goto_e",
        outcome="executed", provenance_class="model", readback=readback,
        detail=_dispatch_result_detail(
            intent_seq, detail or f"{ref}: read-back {readback}"))


def _scan(tmp_path, name, records):
    path = _write(tmp_path, name, records)
    return wp3_p11_audit.scan_pending_intents(load_transcript(path).records)


def test_scan_resolved_intent(tmp_path):
    """W3-1 scanner: intent + its correlated action_result (the
    dispatch-result/v1 detail envelope) → resolved."""
    ledger = TranscriptLedger(tmp_path / "resolved.jsonl")
    ledger.append(TurnStartRecord(turn=1), durable=True)
    iseq = ledger.append(_intent(), durable=True)
    rseq = ledger.append(_result(iseq), durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED", reason="horizon"),
                  durable=True)
    ledger.close()
    scan = wp3_p11_audit.scan_pending_intents(
        load_transcript(tmp_path / "resolved.jsonl").records)
    assert scan["legacy"] is False
    assert len(scan["intents"]) == 1
    entry = scan["intents"][0]
    assert entry["status"] == "resolved"
    assert entry["seq"] == iseq
    assert entry["correlated_seq"] == rseq
    assert entry["turn"] == 1
    assert entry["ref"] == REF
    assert entry["provenance_class"] == "model"
    assert scan["resolved_seqs"] == [iseq]
    assert scan["cancelled_seqs"] == []
    assert scan["uncertain_seqs"] == []


def test_scan_cancelled_intent(tmp_path):
    """W3-1 scanner: intent + its cancellation event → cancelled (known
    not sent)."""
    ledger = TranscriptLedger(tmp_path / "cancelled.jsonl")
    ledger.append(TurnStartRecord(turn=1), durable=True)
    iseq = ledger.append(_intent(), durable=True)
    cseq = ledger.append(_cancelled(iseq), durable=True)
    ledger.append(OutcomeRecord(outcome="TRUNCATED", reason="deadline"),
                  durable=True)
    ledger.close()
    scan = wp3_p11_audit.scan_pending_intents(
        load_transcript(tmp_path / "cancelled.jsonl").records)
    assert scan["legacy"] is False
    entry = scan["intents"][0]
    assert entry["status"] == "cancelled"
    assert entry["seq"] == iseq
    assert entry["correlated_seq"] == cseq
    assert scan["cancelled_seqs"] == [iseq]
    assert scan["resolved_seqs"] == []
    assert scan["uncertain_seqs"] == []


def test_scan_uncertain_intent(tmp_path):
    """W3-1 scanner: an intent alone (a crash inside the dispatch
    window) → uncertain (forensic, D15)."""
    ledger = TranscriptLedger(tmp_path / "uncertain.jsonl")
    ledger.append(TurnStartRecord(turn=1), durable=True)
    iseq = ledger.append(_intent(), durable=True)
    ledger.append(OutcomeRecord(outcome="TRUNCATED", reason="killed"),
                  durable=True)
    ledger.close()
    scan = wp3_p11_audit.scan_pending_intents(
        load_transcript(tmp_path / "uncertain.jsonl").records)
    assert scan["legacy"] is False
    entry = scan["intents"][0]
    assert entry["status"] == "uncertain"
    assert entry["seq"] == iseq
    assert entry["correlated_seq"] is None
    assert scan["uncertain_seqs"] == [iseq]
    assert scan["resolved_seqs"] == []
    assert scan["cancelled_seqs"] == []


def test_scan_legacy_transcript(tmp_path):
    """W3-1 scanner: a transcript without any dispatch_intent records
    (pre-W3-1) → legacy (not newly declared safe or complete)."""
    scan = _scan(tmp_path, "legacy.jsonl", [
        TurnStartRecord(turn=1),
        _action(),
        OutcomeRecord(outcome="COMPLETED", reason="horizon reached"),
    ])
    assert scan["legacy"] is True
    assert scan["intents"] == []
    assert scan["resolved_seqs"] == []
    assert scan["cancelled_seqs"] == []
    assert scan["uncertain_seqs"] == []


def test_scan_same_ref_twice_pairs_per_intent(tmp_path):
    """W3-1 scanner: the same ref dispatched twice — each intent pairs
    with its OWN result via intent_seq (no cross-pairing)."""
    ledger = TranscriptLedger(tmp_path / "twice.jsonl")
    ledger.append(TurnStartRecord(turn=1), durable=True)
    i1 = ledger.append(_intent(), durable=True)
    r1 = ledger.append(_result(i1, readback="inconclusive"), durable=True)
    i2 = ledger.append(
        _intent(provenance="continued_model_order"), durable=True)
    r2 = ledger.append(_result(i2, readback="verified"), durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED"), durable=True)
    ledger.close()
    scan = wp3_p11_audit.scan_pending_intents(
        load_transcript(tmp_path / "twice.jsonl").records)
    assert [e["status"] for e in scan["intents"]] == ["resolved", "resolved"]
    assert scan["intents"][0]["seq"] == i1
    assert scan["intents"][0]["correlated_seq"] == r1
    assert scan["intents"][1]["seq"] == i2
    assert scan["intents"][1]["correlated_seq"] == r2
    assert scan["intents"][0]["provenance_class"] == "model"
    assert scan["intents"][1]["provenance_class"] == "continued_model_order"
    assert scan["resolved_seqs"] == [i1, i2]


def test_scan_legacy_detail_does_not_correlate(tmp_path):
    """W3-1 scanner: a legacy detail string (no envelope) remains valid
    but correlates nothing — the intent stays uncertain."""
    scan = _scan(tmp_path, "legacydetail.jsonl", [
        TurnStartRecord(turn=1),
        _intent(),
        # A pre-W3-1-style action_result: a plain detail string.
        _action(),
        OutcomeRecord(outcome="COMPLETED", reason="horizon reached"),
    ])
    assert scan["intents"][0]["status"] == "uncertain"
    assert scan["uncertain_seqs"] == [1]
    assert scan["resolved_seqs"] == []


def test_audit_discloses_pending_intents(tmp_path):
    """W3-1: the P11 audit reports the pending-intent scan (D15: the
    disclosure is forensic — an unresolved intent does NOT fail the
    audit; the runner never refuses on it)."""
    # A resolved intent: disclosed, all checks pass.
    ledger = TranscriptLedger(tmp_path / "ok.jsonl")
    ledger.append(TurnStartRecord(turn=1), durable=True)
    iseq = ledger.append(_intent(), durable=True)
    ledger.append(_result(iseq), durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED", reason="horizon"),
                  durable=True)
    ledger.close()
    audit = wp3_p11_audit.audit_transcript(tmp_path / "ok.jsonl")
    assert audit["verdict"] == "PASS"
    assert audit["checks"]["pending_intents_disclosed"]["pass"] is True
    assert audit["pending_intents"]["resolved_seqs"] == [iseq]
    assert audit["pending_intents"]["uncertain_seqs"] == []
    # An unresolved (forensic) intent: disclosed, NOT failed (D15).
    ledger2 = TranscriptLedger(tmp_path / "forensic.jsonl")
    ledger2.append(TurnStartRecord(turn=1), durable=True)
    ledger2.append(_intent(), durable=True)
    ledger2.append(OutcomeRecord(outcome="TRUNCATED", reason="killed"),
                   durable=True)
    ledger2.close()
    audit2 = wp3_p11_audit.audit_transcript(tmp_path / "forensic.jsonl")
    assert audit2["verdict"] == "PASS"
    assert audit2["checks"]["pending_intents_disclosed"]["pass"] is True
    assert audit2["pending_intents"]["uncertain_seqs"] == [1]
    assert "forensic" in audit2["checks"]["pending_intents_disclosed"][
        "detail"]


def test_scan_flags_a_real_runner_transcript_cut_in_the_window(tmp_path):
    """W3-1 crash-window matrix (2) end-to-end: a REAL runner transcript
    cut right after a durable intent (a kill between the intent fsync
    and the send) — load keeps the unresolved intent and the scanner
    flags it uncertain (the earlier intent resolved)."""
    from tests.runner_fixtures import (
        FakeModelClient,
        FakeSession,
        make_runner,
        ok_response,
        turn_world,
    )

    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e",
                     "tech:cur_player:research_tech_alphabet_0"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    path = tmp_path / "transcript.jsonl"
    records = load_transcript(path).records
    intents = [r for r in records
               if isinstance(r, MetricEventRecord)
               and r.name == "dispatch_intent"]
    assert len(intents) == 2
    # Simulate the kill: cut the file right after the SECOND intent's
    # line (the first send's intent+result survive; the second intent is
    # the unresolved tail).
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    cut = sum(len(line) + 1 for line in lines[:intents[1].seq + 1])
    killed = tmp_path / "killed.jsonl"
    killed.write_bytes(raw[:cut])
    load = load_transcript(killed)
    assert not load.discarded_trailing_partial
    assert load.records[-1].seq == intents[1].seq
    scan = wp3_p11_audit.scan_pending_intents(load.records)
    assert scan["legacy"] is False
    assert scan["resolved_seqs"] == [intents[0].seq]
    assert scan["uncertain_seqs"] == [intents[1].seq]
    assert scan["intents"][1]["status"] == "uncertain"
    assert scan["intents"][1]["correlated_seq"] is None


def test_metrics_never_parse_the_result_detail_envelope():
    """W3-1: the metrics ignore action_result.detail content — the
    dispatch-result/v1 envelope changes no metric (the frozen
    populations read outcome/readback/provenance, never detail)."""
    from lechonk.core.metrics import compute_metrics
    from lechonk.core.schemas import CandidatesRefRecord

    def records_with(detail):
        return [
            TurnStartRecord(turn=1),
            CandidatesRefRecord(
                turn=1, table_hash="sha256:" + "01" * 32, candidate_count=1,
                by_domain={"unit": 1}, opportunity_domains=["unit_action"]),
            _intent(),
            ActionResultRecord(
                turn=1, ref=REF, domain="unit", verb="goto_e",
                outcome="executed", provenance_class="model",
                readback="verified", detail=detail),
        ]

    plain = records_with(f"{REF}: read-back verified — moved")
    enveloped = records_with(_dispatch_result_detail(
        1, f"{REF}: read-back verified — moved"))
    report_plain = compute_metrics(plain)
    report_enveloped = compute_metrics(enveloped)
    assert (report_plain.metric_events()
            == report_enveloped.metric_events())
    # Sanity: the executed verified model action DOES count as
    # participation (the records carry an opportunity-bearing turn).
    assert report_enveloped.participation.numerator == 1
