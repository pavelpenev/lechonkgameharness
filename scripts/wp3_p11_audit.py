# SPDX-License-Identifier: MIT
"""WP3 package B deliverable 4: the P11 gate check (plan §3 line 76,
P11) — a ledger audit over the WP3 lifecycle transcripts after the
injections.

P11: "timeout/disconnect/mid-mutation/teardown-kill injection → no
duplicate mutation, no takeover, bounded FAILED, transcript intact".

For every audited transcript (the latest kill-test transcript under
artifacts/runs/wp3-kill/, artifacts/runs/wp3-teardown/case-*/transcript.jsonl,
artifacts/runs/wp3-disconnect/transcript.jsonl, plus the WP2 fault run
as a control when present):

- transcript intact: every line schema-validated + checksum-verified
  (load_transcript), no torn tail, seq contiguous, the final record is a
  bounded OutcomeRecord (a TRUNCATED outcome carries its reason);
- no duplicate mutation: every REPEAT execution of a candidate ref is a
  continued_model_order with origin linkage AND carries non-execution
  proof — either (a) a preceding mutation_reconciliation
  non_execution_verified event for that ref with the first execution
  uncertain (inconclusive) — the §3 line 35 disconnect-resubmission
  shape — or (b) the standing-order shape (the origin is a model
  action_result of a persistent verb on an earlier turn);
- no dangling uncertain mutations: every executed action_result with
  readback=inconclusive is followed by a mutation_reconciliation event
  for the same ref (settled, never left dangling);
- pending dispatch intents (W3-1, D5/D15): a PURE scanner classifies
  every dispatch_intent record — resolved (a correlated action_result
  via the dispatch-result/v1 detail envelope), cancelled (a
  mutation_dispatch_cancelled event — known not sent), or uncertain
  (forensic only: a crash inside the dispatch window; the runner never
  refuses or quarantines on it, D15). Transcripts without intents are
  legacy. The scanner is audit-side ONLY: the runner never imports or
  calls it;
- provenance classes within the frozen set;
- no takeover / bounded close / bounded outcome: cross-checked against
  the run's run_meta.json (player identity stable, username never
  renamed, close attempted and bounded, outcome matches the
  transcript's outcome record) when present.

Usage:
    .venv/bin/python scripts/wp3_p11_audit.py
    .venv/bin/python scripts/wp3_p11_audit.py --kill-transcript PATH

Artifacts: artifacts/runs/wp3-p11/p11_audit.json + p11_audit.md.
The audit logic is unit-tested offline (tests/test_lifecycle_p11.py).

Note: the pytest suite no longer refreshes wp3-kill evidence (the kill
test writes a per-run unique tmp_path directory; the standalone kill
test writes a per-run unique subdir of artifacts/runs/wp3-kill/) — the
P11 audit of kill evidence is a standalone-CLI activity, and this
script resolves the LATEST kill-test run subdirectory by default
(--kill-transcript pins one specific run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUN = "wp3-p11"
RUNDIR = REPO / "artifacts" / "runs" / RUN

# The transcripts under audit (run dir -> transcript name). The WP2
# fault run is included as a control when present.
#
# The wp3-kill entry is the KILL_LATEST sentinel (None): the latest
# kill-test transcript is resolved at audit time (see
# _latest_kill_transcript). The suite no longer refreshes wp3-kill
# evidence — auditing kill evidence is a standalone-CLI activity, and
# every run (suite or standalone) writes a per-run unique subdirectory,
# so there is no fixed transcript path to hard-code. Pass
# --kill-transcript to audit one specific run instead of the latest.
AUDIT_RUNS = [
    ("wp3-kill", None),
    ("wp3-teardown/case-teardown-stop",
     "artifacts/runs/wp3-teardown/case-teardown-stop/transcript.jsonl"),
    ("wp3-teardown/case-finish-kill",
     "artifacts/runs/wp3-teardown/case-finish-kill/transcript.jsonl"),
    ("wp3-disconnect", "artifacts/runs/wp3-disconnect/transcript.jsonl"),
    ("wp2-fault-run (control)",
     "artifacts/runs/wp2-fault-run/transcript.jsonl"),
]

WALL_BOUND_S = 240.0

# ---------------------------------------------------------------------------
# W3-1 (D5/D15): the pure pending-intent scanner — audit-side ONLY
# ---------------------------------------------------------------------------
# The runner never imports or calls this module (D15: unresolved intents
# are forensic evidence, never a runner refusal). The scanner is a pure
# function over loaded transcript records: it pairs each dispatch_intent
# with its correlation BY LEDGER SEQ (the dispatch-result/v1 detail
# envelope carries the intent's seq — never positional (turn, ref)
# matching, so a same-ref resend pairs with its OWN result).

DISPATCH_INTENT_EVENT = "dispatch_intent"
DISPATCH_CANCELLED_EVENT = "mutation_dispatch_cancelled"
DISPATCH_INTENT_PROTOCOL = "dispatch-intent/v1"
DISPATCH_CANCELLED_PROTOCOL = "dispatch-cancelled/v1"
DISPATCH_RESULT_PROTOCOL = "dispatch-result/v1"

INTENT_RESOLVED = "resolved"    # intent + its correlated action_result
INTENT_CANCELLED = "cancelled"  # intent + its cancellation (known not sent)
INTENT_UNCERTAIN = "uncertain"  # intent alone (forensic, D15)
INTENT_STATUSES = (INTENT_RESOLVED, INTENT_CANCELLED, INTENT_UNCERTAIN)


def _parse_json_object(text):
    """Strictly parse a canonical-JSON payload string; None when the value
    is not a JSON object (a legacy or corrupt payload)."""
    if not isinstance(text, str):
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_int_seq(value):
    return isinstance(value, int) and not isinstance(value, bool)


def parse_dispatch_intent(value):
    """The dispatch-intent payload (dict), or None when the value does
    not carry the dispatch-intent/v1 protocol."""
    data = _parse_json_object(value)
    if data is None or data.get("protocol") != DISPATCH_INTENT_PROTOCOL:
        return None
    return data


def parse_dispatch_cancelled(value):
    """The cancellation payload (dict), or None when the value does not
    carry the dispatch-cancelled/v1 protocol."""
    data = _parse_json_object(value)
    if data is None or data.get("protocol") != DISPATCH_CANCELLED_PROTOCOL:
        return None
    return data


def parse_dispatch_result_intent_seq(detail):
    """The intent seq carried by an action_result's dispatch-result/v1
    detail envelope, or None (legacy detail strings remain valid — they
    simply carry no correlation)."""
    data = _parse_json_object(detail)
    if data is None or data.get("protocol") != DISPATCH_RESULT_PROTOCOL:
        return None
    seq = data.get("intent_seq")
    return seq if _is_int_seq(seq) else None


def scan_pending_intents(records):
    """The PURE pending-intent scanner (W3-1, D5/D15) over loaded
    transcript records.

    Classification per dispatch_intent metric_event:
    - intent + a correlated action_result (a dispatch-result/v1 detail
      envelope with intent_seq == the intent's seq, at a HIGHER seq)
      -> "resolved";
    - intent + a cancellation event (a dispatch-cancelled/v1 payload with
      intent_seq == the intent's seq, at a higher seq) -> "cancelled"
      (known not sent);
    - intent alone -> "uncertain" (forensic: the send may or may not have
      gone out — a kill inside the dispatch window; D15: reported by this
      audit, never refused by the runner).
    A transcript with no dispatch_intent records at all is "legacy".

    Returns the report dict:
    {"legacy": bool,
     "intents": [{"seq", "turn", "ref", "provenance_class", "status",
                  "correlated_seq"}...],
     "resolved_seqs": [...], "cancelled_seqs": [...],
     "uncertain_seqs": [...]}
    """
    from lechonk.core.schemas import ActionResultRecord, MetricEventRecord

    intent_seqs = set()
    intent_events = []
    cancellations = {}  # intent_seq -> cancellation event's seq
    results = {}  # intent_seq -> correlated action_result's seq
    for rec in records:
        if isinstance(rec, MetricEventRecord):
            if rec.name == DISPATCH_INTENT_EVENT:
                intent_events.append(rec)
                intent_seqs.add(rec.seq)
            elif rec.name == DISPATCH_CANCELLED_EVENT:
                payload = parse_dispatch_cancelled(rec.value)
                if payload is None:
                    continue
                iseq = payload.get("intent_seq")
                if _is_int_seq(iseq) and iseq in intent_seqs \
                        and rec.seq > iseq:
                    cancellations.setdefault(iseq, rec.seq)
        elif isinstance(rec, ActionResultRecord):
            iseq = parse_dispatch_result_intent_seq(rec.detail)
            if iseq is not None and iseq in intent_seqs \
                    and rec.seq > iseq:
                results.setdefault(iseq, rec.seq)

    entries = []
    for rec in intent_events:
        payload = parse_dispatch_intent(rec.value)
        if rec.seq in cancellations:
            status, correlated = INTENT_CANCELLED, cancellations[rec.seq]
        elif rec.seq in results:
            status, correlated = INTENT_RESOLVED, results[rec.seq]
        else:
            status, correlated = INTENT_UNCERTAIN, None
        entries.append({
            "seq": rec.seq,
            "turn": payload.get("turn") if payload else None,
            "ref": payload.get("ref") if payload else None,
            "provenance_class": (payload.get("provenance_class")
                                 if payload else None),
            "status": status,
            "correlated_seq": correlated,
        })
    return {
        "legacy": not intent_events,
        "intents": entries,
        "resolved_seqs": [e["seq"] for e in entries
                          if e["status"] == INTENT_RESOLVED],
        "cancelled_seqs": [e["seq"] for e in entries
                           if e["status"] == INTENT_CANCELLED],
        "uncertain_seqs": [e["seq"] for e in entries
                           if e["status"] == INTENT_UNCERTAIN],
    }

# Sentinel for AUDIT_RUNS: resolve the latest kill-test transcript at
# audit time (the wp3-kill evidence root is frozen — the suite no longer
# refreshes it, and every run writes its own subdirectory).
KILL_LATEST = None


def latest_kill_transcript(kill_root: Path) -> Path | None:
    """The most recently modified kill-test transcript under kill_root:
    the frozen root's own transcript.jsonl (pre-isolation legacy
    evidence) competes with every per-run subdir's transcript.jsonl;
    the newest mtime wins. None when nothing exists."""
    candidates = [kill_root / "transcript.jsonl"]
    if kill_root.is_dir():
        candidates.extend(sorted(kill_root.glob("*-*/transcript.jsonl")))
    present = [p for p in candidates if p.is_file()]
    if not present:
        return None
    return max(present, key=lambda p: p.stat().st_mtime)


def audit_transcript(path) -> dict:
    """The P11 ledger audit over ONE transcript. Returns the evidence
    dict (checks + the repeat-execution analysis)."""
    from lechonk.core.ledger import LedgerCorruptionError, load_transcript
    from lechonk.core.runner import PERSISTENT_VERBS
    from lechonk.core.schemas import (
        ActionResultRecord,
        MetricEventRecord,
        OutcomeRecord,
        PROVENANCE_CLASSES,
    )

    path = Path(path)
    checks: dict = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        return bool(cond)

    # --- transcript intact ------------------------------------------------
    try:
        load = load_transcript(path)
        records = load.records
        check("transcript_loads_clean", True,
              f"{len(records)} records, every line schema-validated + "
              f"checksum-verified")
    except LedgerCorruptionError as exc:
        check("transcript_loads_clean", False, f"corruption: {exc}")
        records = []
        load = None
    if load is not None:
        check("no_torn_tail", not load.discarded_trailing_partial,
              f"discarded_trailing_partial={load.discarded_trailing_partial}")
    seqs = [r.seq for r in records]
    check("seq_contiguous", seqs == list(range(len(seqs))),
          f"seq 0..{len(seqs) - 1}" if seqs else "no records")

    # --- bounded outcome ----------------------------------------------------
    outcome_record = records[-1] if records else None
    bounded = (isinstance(outcome_record, OutcomeRecord)
               and (outcome_record.outcome != "TRUNCATED"
                    or bool(outcome_record.reason.strip())))
    if isinstance(outcome_record, OutcomeRecord):
        detail = (f"last record: outcome {outcome_record.outcome} "
                  f"reason={outcome_record.reason!r}")
    else:
        detail = (f"last record: "
                  f"{outcome_record.kind if outcome_record else None} "
                  f"(not an outcome record)")
    check("bounded_outcome_record", bounded, detail)

    # --- provenance within the frozen set ------------------------------------
    actions = [r for r in records if isinstance(r, ActionResultRecord)]
    bad_prov = sorted({a.provenance_class for a in actions
                       if a.provenance_class not in PROVENANCE_CLASSES})
    check("provenance_classes_frozen", not bad_prov,
          f"classes seen: {sorted({a.provenance_class for a in actions})}")

    # --- no duplicate mutation ------------------------------------------------
    events = [r for r in records if isinstance(r, MetricEventRecord)]
    by_seq = {r.seq: r for r in records}
    executed = [a for a in actions if a.outcome == "executed"]
    by_ref: dict[str, list] = {}
    for a in executed:
        by_ref.setdefault(a.ref, []).append(a)

    def reconciliation_event(ref, after_seq, before_seq=None):
        for e in events:
            if e.name != "mutation_reconciliation":
                continue
            if not str(e.value).startswith(f"{ref}: "):
                continue
            if e.seq <= after_seq:
                continue
            if before_seq is not None and e.seq >= before_seq:
                continue
            return e
        return None

    repeats = []
    duplicate_ok = True
    for ref, recs in sorted(by_ref.items()):
        if len(recs) <= 1:
            continue
        for repeat in recs[1:]:
            origin = by_seq.get(repeat.origin_seq) \
                if repeat.origin_seq is not None else None
            origin_ok = (isinstance(origin, ActionResultRecord)
                         and origin.outcome == "executed")
            recon = reconciliation_event(ref, recs[0].seq, repeat.seq)
            recon_ok = (recon is not None
                        and "non_execution_verified" in str(recon.value))
            first_uncertain = recs[0].readback == "inconclusive"
            standing_ok = (
                origin_ok
                and origin.provenance_class == "model"
                and origin.verb in PERSISTENT_VERBS
                and repeat.turn > origin.turn)
            is_cmo = repeat.provenance_class == "continued_model_order"
            ok = is_cmo and origin_ok \
                and ((recon_ok and first_uncertain) or standing_ok)
            proof = ("reconciliation (non_execution_verified, first "
                     "execution uncertain)" if recon_ok and first_uncertain
                     else ("standing-order continuation (state-proven "
                           "non-execution)" if standing_ok else None))
            repeats.append({
                "ref": ref, "repeat_seq": repeat.seq,
                "first_seq": recs[0].seq,
                "continued_model_order": is_cmo,
                "origin_linked": origin_ok,
                "non_execution_proof": proof,
                "ok": ok,
            })
            duplicate_ok = duplicate_ok and ok
    check("no_duplicate_mutation", duplicate_ok,
          (f"{len(repeats)} repeat execution(s), all origin-linked "
           f"continued_model_order with non-execution proof: "
           f"{[(r['ref'], r['non_execution_proof']) for r in repeats]}")
          if repeats else "no candidate was executed twice")

    # --- no dangling uncertain mutations ---------------------------------------
    dangling = []
    for a in executed:
        if a.readback != "inconclusive":
            continue
        if reconciliation_event(a.ref, a.seq) is None:
            dangling.append(a.ref)
    check("no_dangling_uncertain_mutations", not dangling,
          (f"unsettled uncertain mutations: {dangling}" if dangling
           else "every uncertain (inconclusive) execution was settled by a "
                "mutation_reconciliation event"))

    # --- pending dispatch intents (W3-1, D5/D15: forensic report) ------------
    scan = scan_pending_intents(records)
    check("pending_intents_disclosed",
          all(e["status"] in INTENT_STATUSES for e in scan["intents"]),
          (f"{len(scan['intents'])} dispatch intent(s): "
           f"{len(scan['resolved_seqs'])} resolved, "
           f"{len(scan['cancelled_seqs'])} cancelled (known not sent), "
           f"{len(scan['uncertain_seqs'])} uncertain (forensic, D15 — the "
           f"runner never refuses on them)"
           if not scan["legacy"]
           else "legacy transcript: no dispatch_intent records (pre-W3-1)"))

    return {
        "path": str(path),
        "records": len(records),
        "outcome": ({"outcome": outcome_record.outcome,
                     "reason": outcome_record.reason}
                    if isinstance(outcome_record, OutcomeRecord) else None),
        "executed_actions": len(executed),
        "repeat_executions": repeats,
        "pending_intents": scan,
        "checks": checks,
        "verdict": ("PASS" if all(c["pass"] for c in checks.values())
                    else "FAIL"),
    }


def audit_run(run_dir: Path) -> dict:
    """Cross-check one run dir's run_meta.json against its transcript
    (no takeover, bounded close, outcome match, bounded wall). Checks
    are strict where the run recorded the data and honestly N/A where
    the run has no such artifact (e.g. the kill test has no session)."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    checks: dict = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        return bool(cond)

    if not meta_path.exists():
        check("run_meta_present", False, f"missing {meta_path}")
        return {"checks": checks, "verdict": "FAIL"}
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    check("run_meta_present", True, str(meta_path))

    identity = meta.get("identity") or {}
    start, end = identity.get("at_start"), identity.get("at_end")
    if start is None and end is None:
        check("no_takeover_identity_stable", True,
              "n/a — this run has no game session (the kill test audits "
              "the ledger writer only)")
    else:
        stable = (isinstance(start, dict) and start == end
                  and start.get("username") == "myagent")
        check("no_takeover_identity_stable", stable,
              f"identity at start={start} at end={end} (self_play=False: "
              f"the player was never renamed or handed off)")

    outcome = meta.get("outcome") or {}
    close_status = outcome.get("close_status")
    if close_status is None:
        check("close_bounded", True,
              "n/a — this run has no session teardown (the kill test "
              "audits the ledger writer only)")
    else:
        check("close_bounded", close_status != "not_attempted",
              f"close_status={close_status!r} (the teardown was attempted "
              f"and bounded — skip path or failed-session refusal, never "
              f"a hang)")
    wall = meta.get("wall_clock_s")
    if wall is None:
        check("wall_bounded", True, "n/a — no wall clock recorded")
    else:
        check("wall_bounded", isinstance(wall, (int, float))
              and wall < WALL_BOUND_S,
              f"wall_clock_s={wall} (< {WALL_BOUND_S:.0f}s: no unbounded "
              f"wait)")
    return {
        "meta": {k: meta.get(k) for k in
                 ("run", "case", "seed", "outcome", "wall_clock_s")},
        "checks": checks,
        "verdict": ("PASS" if all(c["pass"] for c in checks.values())
                    else "FAIL"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kill-transcript", default=None,
                        help="explicit wp3-kill transcript path (default: "
                             "the latest kill-test transcript under "
                             "artifacts/runs/wp3-kill/)")
    args = parser.parse_args()
    RUNDIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNDIR / "audit_stdout.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    results = {}
    all_ok = True
    for name, rel in AUDIT_RUNS:
        if rel is KILL_LATEST:
            if args.kill_transcript is not None:
                transcript = Path(args.kill_transcript)
            else:
                transcript = latest_kill_transcript(
                    REPO / "artifacts" / "runs" / "wp3-kill")
            if transcript is None:
                log("  MISSING: no kill-test transcript under "
                    "artifacts/runs/wp3-kill/ — skipped (not produced "
                    "this session)")
                results[name] = {"missing": True}
                continue
            rel = transcript.relative_to(REPO).as_posix()
        else:
            transcript = REPO / rel
        run_dir = transcript.parent
        log(f"=== auditing {name}: {rel}")
        if not transcript.exists():
            log(f"  MISSING: {transcript} — skipped (not produced this "
                f"session)")
            results[name] = {"missing": True}
            continue
        t_audit = audit_transcript(transcript)
        r_audit = audit_run(run_dir)
        # the run-meta outcome must match the transcript's outcome record
        meta_outcome = (r_audit.get("meta", {}).get("outcome") or {})
        if meta_outcome.get("outcome") is None:
            r_audit["checks"]["outcome_matches_transcript"] = {
                "pass": True,
                "detail": "n/a — the run recorded no outcome in run_meta"}
        else:
            match = (meta_outcome.get("outcome") ==
                     (t_audit["outcome"] or {}).get("outcome"))
            r_audit["checks"]["outcome_matches_transcript"] = {
                "pass": match,
                "detail": (f"run_meta outcome="
                           f"{meta_outcome.get('outcome')!r} vs transcript "
                           f"outcome record="
                           f"{(t_audit['outcome'] or {}).get('outcome')!r}")}
        ok = (t_audit["verdict"] == "PASS" and r_audit["verdict"] == "PASS")
        all_ok = all_ok and ok
        results[name] = {"transcript": t_audit, "run_meta": r_audit,
                         "ok": ok}
        log(f"  transcript audit: {t_audit['verdict']} "
            f"({sum(1 for c in t_audit['checks'].values() if c['pass'])}/"
            f"{len(t_audit['checks'])} checks, "
            f"{t_audit['executed_actions']} executed actions, "
            f"{len(t_audit['repeat_executions'])} repeat execution(s))")
        log(f"  run-meta audit:   {r_audit['verdict']} "
            f"({sum(1 for c in r_audit['checks'].values() if c['pass'])}/"
            f"{len(r_audit['checks'])} checks)")
        for cname, entry in t_audit["checks"].items():
            log(f"    [{'PASS' if entry['pass'] else 'FAIL'}] {cname}: "
                f"{entry['detail']}")
        for cname, entry in r_audit["checks"].items():
            log(f"    [{'PASS' if entry['pass'] else 'FAIL'}] {cname}: "
                f"{entry['detail']}")

    verdict = "PASS" if all_ok else "FAIL"
    evidence = {
        "run": RUN,
        "gate": "P11 (design-plan-freeciv-v1.1.md §3 line 76): "
                "timeout/disconnect/mid-mutation/teardown-kill injection "
                "→ no duplicate mutation, no takeover, bounded FAILED, "
                "transcript intact",
        "audited_runs": {name: ({"missing": True} if res.get("missing")
                                else {"ok": res["ok"],
                                      "transcript": res["transcript"],
                                      "run_meta": res["run_meta"]})
                         for name, res in results.items()},
        "verdict": verdict,
    }
    with open(RUNDIR / "p11_audit.json", "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    lines = [
        "# WP3 P11 gate check — ledger audit after the lifecycle "
        "injections",
        "",
        "P11: timeout/disconnect/mid-mutation/teardown-kill injection → "
        "no duplicate mutation, no takeover, bounded FAILED, transcript "
        "intact.",
        "",
    ]
    for name, res in results.items():
        if res.get("missing"):
            lines.append(f"## `{name}` — MISSING (skipped)")
            lines.append("")
            continue
        t_audit, r_audit = res["transcript"], res["run_meta"]
        lines.append(f"## `{name}` — "
                     f"{'PASS' if res['ok'] else 'FAIL'}")
        lines.append("")
        lines.append(f"- transcript: {t_audit['records']} records, "
                     f"{t_audit['executed_actions']} executed actions, "
                     f"{len(t_audit['repeat_executions'])} repeat "
                     f"execution(s)")
        lines.append(f"- outcome: {t_audit['outcome']}")
        for cname, entry in t_audit["checks"].items():
            lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                         f"`{cname}`: {entry['detail']}")
        for cname, entry in r_audit["checks"].items():
            lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                         f"`{cname}`: {entry['detail']}")
        lines.append("")
    with open(RUNDIR / "p11_audit.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"wrote p11_audit.json/.md under {RUNDIR}")
    log(f"P11 GATE VERDICT: {verdict}")
    log_fh.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
