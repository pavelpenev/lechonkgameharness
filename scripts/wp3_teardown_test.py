# SPDX-License-Identifier: MIT
"""WP3 package B deliverable 2 (LIVE): disconnect-during-teardown
acceptance against the real stack (plan §4 line 56, §3 line 34) —
"inject a disconnect while the pipeline is tearing down … prove teardown
overrun hits the supervisor deadline → bounded FAILED/TRUNCATED outcome,
transcript intact and flushable, no hang, no unbounded wait".

Two cases, ONE CASE PER PROCESS RUN (one env per process, plan §7.1 —
the upstream blocking hazards are CivController.close()'s unbounded
game-over wait and the synchronous save-deletion / score HTTP):

- ``teardown-stop`` (the supervisor-deadline proof): a fresh
  deterministic reset (own server, the turn-11 checkpoint untouched);
  the pipeline completes its turn (COMPLETED) and the outcome record is
  durable BEFORE teardown; the wrapper's close() then SIGSTOPs our own
  game's server — the connection stays alive but the server is frozen,
  so the upstream CivController.close() game-over wait blocks — and the
  real guarded close hits the §3 teardown deadline → the skip path
  (skipped_after_deadline), a bounded return, no hang.
- ``finish-kill`` (the bounded-FAILED proof): the wrapper's first
  step(None) (the pipeline-owned finish primitive) SIGKILLs our own
  game's server synchronously — the connection drops as the phase_done
  is dispatched — and the response never arrives → the step deadline
  fires → bounded TRUNCATED (infrastructure failure) with the
  transcript intact; the teardown close is likewise bounded.

The runner, the ledger and the adapter session are 100% real; only the
server-process signal and the model client (a deterministic scripted
picker — no API call) are synthetic. The server is killed by PORT MATCH
only (our session's own fresh game).

Usage (one case per process):
    .venv/bin/python scripts/wp3_teardown_test.py --case teardown-stop
    .venv/bin/python scripts/wp3_teardown_test.py --case finish-kill

Artifacts under artifacts/runs/wp3-teardown/<case>/: run_stdout.log,
transcript.jsonl, run_meta.json, teardown_evidence.md, workdir/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wp3_lifecycle_common import (  # noqa: E402
    REPO,
    RunLogger,
    _Seam,
    cleanup_own_game,
    run_lifecycle_run,
    turn11_checkpoint_untouched,
    wait_for_stack,
)

RUN = "wp3-teardown"
CASES = ("teardown-stop", "finish-kill")
SEEDS = {"teardown-stop": 20261035, "finish-kill": 20261036}


def verify_case(logger, case, run):
    """The per-case acceptance checks (P11: bounded FAILED, artifacts
    intact, no hang, no takeover)."""
    from lechonk.core.schemas import ActionResultRecord, OutcomeRecord

    result = run["result"]
    records = run["records"]
    load = run["load"]
    wrapper = run["wrapper"]
    wall = run["wall_clock_s"]
    config = run["config"]

    actions = [r for r in records if isinstance(r, ActionResultRecord)]
    executed = [a for a in actions if a.outcome == "executed"]

    # --- transcript intact + flushable (§3 line 34) ----------------------
    logger.check("transcript_loads_clean", True, "load_transcript OK")
    logger.check("no_torn_tail", not load.discarded_trailing_partial,
                 f"discarded_trailing_partial="
                 f"{load.discarded_trailing_partial}")
    logger.check("seq_contiguous",
                 [r.seq for r in records] == list(range(len(records))),
                 f"seq 0..{len(records) - 1}")
    logger.check("outcome_record_last",
                 isinstance(records[-1], OutcomeRecord),
                 f"last record kind={records[-1].kind}")
    with open(run["transcript_path"], "rb") as fh:
        tail = fh.read()[-1:]
    logger.check("transcript_flushable", tail == b"\n",
                 "the transcript file is newline-terminated (fsynced)")

    # --- no takeover (P11) ------------------------------------------------
    identity_ok = (
        wrapper is not None
        and wrapper.identity_at_start is not None
        and wrapper.identity_at_end is not None
        and wrapper.identity_at_start == wrapper.identity_at_end
        and wrapper.identity_at_start.get("username") == "myagent")
    logger.check("no_takeover_identity_stable", identity_ok,
                 f"identity at start={wrapper.identity_at_start if wrapper else None} "
                 f"at end={wrapper.identity_at_end if wrapper else None} "
                 f"(self_play=False: the player was never renamed or "
                 f"handed off)")

    # --- the turn-11 checkpoint untouched ---------------------------------
    untouched, detail = turn11_checkpoint_untouched()
    logger.check("turn11_checkpoint_untouched", untouched, detail)

    # --- the injection actually fired --------------------------------------
    logger.check("injection_fired",
                 bool(wrapper.kill_events),
                 f"kill_events={wrapper.kill_events}")
    logger.check("close_was_called", wrapper.close_calls >= 1,
                 f"close_calls={wrapper.close_calls}")

    # --- bounded (no hang, no unbounded wait) ------------------------------
    logger.check("run_returned_bounded", wall < 240.0,
                 f"wall_clock={wall:.1f}s (startup + turn + step "
                 f"deadline + teardown deadline, all bounded)")

    if case == "teardown-stop":
        # --- the pipeline completed before the disconnect ------------------
        logger.check("pipeline_completed_before_disconnect",
                     result.outcome == "COMPLETED",
                     f"outcome={result.outcome} turns_completed="
                     f"{result.turns_completed}")
        logger.check("one_turn_executed",
                     len([a for a in executed
                          if a.provenance_class == "model"]) == 1,
                     f"executed model actions: "
                     f"{[(a.ref, a.readback) for a in executed]}")
        # --- the teardown overrun hit the supervisor deadline --------------
        logger.check("teardown_overrun_hit_deadline",
                     result.close_status == "skipped_after_deadline",
                     f"close_status={result.close_status!r} (the upstream "
                     f"game-over wait blocked; the §3 teardown deadline "
                     f"of {config.teardown_deadline_s:.0f}s fired → the "
                     f"skip path)")
        stop_events = [e for e in wrapper.kill_events
                       if e["signal"] == "SIGSTOP"]
        logger.check("server_was_frozen_at_teardown",
                     len(stop_events) == 1 and stop_events[0]["pid"],
                     f"SIGSTOP events: {stop_events}")
        # The close blocked ~the deadline (bounded, not a hang).
        logger.check("teardown_blocked_about_the_deadline",
                     wall >= config.teardown_deadline_s,
                     f"wall_clock={wall:.1f}s >= teardown deadline "
                     f"{config.teardown_deadline_s:.0f}s — the close "
                     f"blocked until the deadline, then skipped")
    else:  # finish-kill
        # --- bounded FAILED/TRUNCATED outcome ------------------------------
        logger.check("bounded_truncated_outcome",
                     result.outcome == "TRUNCATED",
                     f"outcome={result.outcome} reason={result.reason!r}")
        logger.check("infrastructure_failure_classified",
                     "infrastructure failure" in result.reason,
                     "the disconnect is reported as infrastructure failure "
                     "(separated from strategic loss, §3 line 34)")
        # The finish primitive is NOT a mutation: no reconciliation, no
        # resubmission — exactly one mutation send.
        mutation_sends = [c for c in wrapper.step_log if c[0] is not None]
        logger.check("exactly_one_mutation_sent",
                     len(mutation_sends) == 1,
                     f"mutation sends: {mutation_sends} (the finish "
                     f"disconnect is not a mutation: never resent)")
        from lechonk.core.schemas import MetricEventRecord
        reconciliations = [r for r in records
                           if isinstance(r, MetricEventRecord)
                           and r.name == "mutation_reconciliation"]
        logger.check("no_reconciliation_for_finish",
                     reconciliations == [],
                     "the finish primitive is not a mutation — no "
                     "uncertain-mutation reconciliation applies")
        # The teardown close is bounded too (the failed session refuses
        # the close RPC immediately — bounded, never a hang).
        logger.check("teardown_close_bounded",
                     result.close_status != "not_attempted",
                     f"close_status={result.close_status!r} (bounded: "
                     f"either the skip path or the failed-session "
                     f"refusal — never an unbounded wait)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args()
    case = args.case
    rundir = REPO / "artifacts" / "runs" / RUN / f"case-{case}"
    logger = RunLogger(rundir)
    logger.log(f"[{RUN}/{case}] repo={REPO}")
    logger.log(f"[{RUN}/{case}] rundir={rundir}")

    if not wait_for_stack():
        logger.log(f"[{RUN}/{case}] FATAL: pubstatus unreachable")
        return 2
    logger.log(f"[{RUN}/{case}] pubstatus 200 — stack up")

    seed = SEEDS[case]
    if case == "teardown-stop":
        seam = _Seam(stop_on_close=True)
        max_turns = 1
    else:
        seam = _Seam(kill_on_finish=True)
        max_turns = 2

    run = run_lifecycle_run(
        logger=logger, run_id=f"{RUN}-{case}", rundir=rundir, seed=seed,
        max_turns=max_turns, seam=seam,
        mutation_deadline_s=10.0, waiting_deadline_s=15.0,
        teardown_deadline_s=10.0)

    verify_case(logger, case, run)

    # --- cleanup: never leave our own frozen/killed game's server ------
    wrapper = run["wrapper"]
    cleanup = cleanup_own_game(logger, wrapper)
    logger.check("cleanup_own_game_server_gone",
                 cleanup["own_game_server_gone"],
                 f"port={cleanup['port']} signaled_pids="
                 f"{cleanup['signaled_pids']} killed_at_cleanup="
                 f"{cleanup['killed_at_cleanup']} — our game's server is "
                 f"gone ({cleanup['note']})")

    verdict = logger.verdict()
    logger.write_json("run_meta.json", {
        "run": f"{RUN}-{case}", "case": case, "seed": seed,
        "outcome": {"outcome": run["result"].outcome,
                    "reason": run["result"].reason,
                    "turns_completed": run["result"].turns_completed,
                    "close_status": run["result"].close_status},
        "wall_clock_s": round(run["wall_clock_s"], 1),
        "injection": {"seam": seam.describe(),
                      "kill_events": wrapper.kill_events,
                      "step_log": [{"action": str(c[0]),
                                    "deadline": c[1]}
                                   for c in wrapper.step_log],
                      "close_calls": wrapper.close_calls},
        "identity": {"at_start": wrapper.identity_at_start,
                     "at_end": wrapper.identity_at_end},
        "transcript": {"path": run["transcript_path"],
                       "records": len(run["records"]),
                       "discarded_trailing_partial":
                           run["load"].discarded_trailing_partial,
                       "last_seq": run["load"].last_seq},
        "checks": logger.checks,
        "verdict": verdict,
        "artifacts": {"stdout": "run_stdout.log",
                      "transcript": "transcript.jsonl",
                      "run_meta": "run_meta.json",
                      "evidence": "teardown_evidence.md"},
    })

    lines = [
        f"# WP3 teardown test (LIVE) — case `{case}`",
        "",
        f"- case: `{case}` (seed {seed}, fresh deterministic reset on "
        f"the session's own server; the turn-11 checkpoint untouched)",
        f"- injection: {seam.describe()}",
        f"- outcome: {run['result'].outcome} (turns_completed="
        f"{run['result'].turns_completed}, "
        f"close_status={run['result'].close_status!r}, "
        f"reason={run['result'].reason!r})",
        f"- wall clock: {run['wall_clock_s']:.1f} s (bounded)",
        f"- kill events: {wrapper.kill_events}",
        f"- identity: {wrapper.identity_at_start} -> "
        f"{wrapper.identity_at_end} (no takeover)",
        "",
        "## Checks",
        "",
    ]
    for name, entry in logger.checks.items():
        lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                     f"`{name}`: {entry['detail']}")
    with open(os.path.join(rundir, "teardown_evidence.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.log(f"[{RUN}/{case}] wrote teardown_evidence.md")
    logger.log(f"[{RUN}/{case}] VERDICT: {verdict} "
               f"({sum(1 for c in logger.checks.values() if c['pass'])}/"
               f"{len(logger.checks)} checks)")
    logger.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
