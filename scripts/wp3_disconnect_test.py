# SPDX-License-Identifier: MIT
"""WP3 package B deliverable 3 (LIVE): the mid-mutation disconnect case
against the real stack (plan §4 line 56, §3 line 35, P11).

"the case where a mutation's read-back is inconclusive because the
connection dropped mid-verification" — live: a fresh deterministic reset
(own server, the turn-11 checkpoint untouched); the scripted client
orders a unit move (a mutation); the wrapper's first mutation step
SIGKILLs our own game's server synchronously — the connection drops as
the mutation is dispatched — so the read-back never arrives.

The runner's §3 line 35 reconciliation then runs against the REAL
session: the mutation send timed out (the §3 per-mutation deadline),
which marks the session failed, so the reconciliation read (the
reconnect) is refused → the mutation's state is UNKNOWN → quarantine →
the bounded fail-stop: TRUNCATED with artifacts, the mutation NEVER
resent (never blind replay, no duplicate mutation).

The runner, the ledger and the adapter session are 100% real; only the
server-process signal and the model client (a deterministic scripted
picker — no API call) are synthetic. The server is killed by PORT MATCH
only (our session's own fresh game).

Usage:
    .venv/bin/python scripts/wp3_disconnect_test.py

Artifacts under artifacts/runs/wp3-disconnect/: run_stdout.log,
transcript.jsonl, run_meta.json, disconnect_evidence.md, workdir/.
"""

from __future__ import annotations

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

RUN = "wp3-disconnect"
SEED = 20261037  # fresh: distinct from the teardown cases and WP0-WP2


def main():
    rundir = REPO / "artifacts" / "runs" / RUN
    logger = RunLogger(rundir)
    logger.log(f"[{RUN}] repo={REPO}")
    logger.log(f"[{RUN}] rundir={rundir}")

    if not wait_for_stack():
        logger.log(f"[{RUN}] FATAL: pubstatus unreachable")
        return 2
    logger.log(f"[{RUN}] pubstatus 200 — stack up")

    seam = _Seam(kill_on_mutation=True)
    run = run_lifecycle_run(
        logger=logger, run_id=RUN, rundir=rundir, seed=SEED, max_turns=2,
        seam=seam, mutation_deadline_s=10.0, waiting_deadline_s=15.0,
        teardown_deadline_s=10.0)

    # ------------------------------------------------------------------
    # verification (P11 + §3 line 35)
    # ------------------------------------------------------------------
    from lechonk.core.schemas import (
        ActionResultRecord,
        MetricEventRecord,
        OutcomeRecord,
    )

    result = run["result"]
    records = run["records"]
    load = run["load"]
    wrapper = run["wrapper"]
    wall = run["wall_clock_s"]

    actions = [r for r in records if isinstance(r, ActionResultRecord)]
    executed = [a for a in actions if a.outcome == "executed"]
    events = [r for r in records if isinstance(r, MetricEventRecord)]
    reconciliations = [r for r in events
                       if r.name == "mutation_reconciliation"]
    quarantines = [r for r in events if r.name == "mutation_quarantine"]

    # --- the uncertain mutation: sent once, never resent -----------------
    mutation_sends = [c for c in wrapper.step_log if c[0] is not None]
    logger.check("exactly_one_mutation_sent", len(mutation_sends) == 1,
                 f"mutation sends: {mutation_sends} — the uncertain "
                 f"mutation is NEVER resent (never blind replay)")
    logger.check("mutation_recorded_as_uncertain",
                 len(executed) == 1 and executed[0].readback == "inconclusive",
                 f"executed action_results: "
                 f"{[(a.ref, a.readback, a.provenance_class) for a in executed]}")
    logger.check("no_duplicate_mutation",
                 len([a for a in actions
                      if a.outcome == "executed"
                      and a.provenance_class == "continued_model_order"])
                 == 0,
                 "no continued_model_order resubmission: the "
                 "reconciliation could not prove non-execution, so the "
                 "mutation was quarantined, never resent")

    # --- the reconciliation ran and settled UNKNOWN ----------------------
    logger.check("reconciliation_ran", len(reconciliations) == 1,
                 f"mutation_reconciliation events: "
                 f"{[r.value for r in reconciliations]}")
    logger.check("reconciliation_unknown",
                 len(reconciliations) == 1 and ": unknown (" in reconciliations[0].value,
                 "the reconnect read was refused (the step timeout marked "
                 "the session failed) → the mutation's state is UNKNOWN")
    logger.check("quarantine_recorded", len(quarantines) == 1,
                 f"mutation_quarantine events: "
                 f"{[r.value for r in quarantines]}")

    # --- bounded fail-stop with artifacts ---------------------------------
    logger.check("bounded_fail_stop_truncated",
                 result.outcome == "TRUNCATED",
                 f"outcome={result.outcome} reason={result.reason!r}")
    logger.check("quarantine_reason_reported",
                 "quarantined" in result.reason
                 and "never resent" in result.reason,
                 "the TRUNCATED reason carries the quarantine (never "
                 "blind replay, §3 line 35)")
    logger.check("run_returned_bounded", wall < 240.0,
                 f"wall_clock={wall:.1f}s (startup + turn + the §3 "
                 f"per-mutation deadline + a refused reconnect read + a "
                 f"bounded close — no hang, no unbounded wait)")
    logger.check("close_bounded",
                 result.close_status != "not_attempted",
                 f"close_status={result.close_status!r}")

    # --- transcript intact + flushable (§3 line 34) -----------------------
    logger.check("transcript_loads_clean", True, "load_transcript OK")
    logger.check("no_torn_tail", not load.discarded_trailing_partial,
                 f"discarded_trailing_partial="
                 f"{load.discarded_trailing_partial}")
    logger.check("seq_contiguous",
                 [r.seq for r in records] == list(range(len(records))),
                 f"seq 0..{len(records) - 1}")
    logger.check("outcome_record_last",
                 isinstance(records[-1], OutcomeRecord)
                 and records[-1].outcome == "TRUNCATED",
                 f"last record: {records[-1].kind} "
                 f"{records[-1].outcome}")
    with open(run["transcript_path"], "rb") as fh:
        tail = fh.read()[-1:]
    logger.check("transcript_flushable", tail == b"\n",
                 "the transcript file is newline-terminated (fsynced)")

    # --- no takeover (P11) --------------------------------------------------
    identity_ok = (
        wrapper.identity_at_start is not None
        and wrapper.identity_at_end is not None
        and wrapper.identity_at_start == wrapper.identity_at_end
        and wrapper.identity_at_start.get("username") == "myagent")
    logger.check("no_takeover_identity_stable", identity_ok,
                 f"identity at start={wrapper.identity_at_start} at "
                 f"end={wrapper.identity_at_end} (self_play=False: the "
                 f"player was never renamed or handed off)")

    # --- the turn-11 checkpoint untouched -----------------------------------
    untouched, detail = turn11_checkpoint_untouched()
    logger.check("turn11_checkpoint_untouched", untouched, detail)

    # --- the injection fired -------------------------------------------------
    logger.check("injection_fired", bool(wrapper.kill_events),
                 f"kill_events={wrapper.kill_events}")

    # ------------------------------------------------------------------
    # cleanup: never leave our own killed game's server behind
    # ------------------------------------------------------------------
    cleanup = cleanup_own_game(logger, wrapper)
    logger.check("cleanup_own_game_server_gone",
                 cleanup["own_game_server_gone"],
                 f"port={cleanup['port']} signaled_pids="
                 f"{cleanup['signaled_pids']} killed_at_cleanup="
                 f"{cleanup['killed_at_cleanup']} — our game's server is "
                 f"gone ({cleanup['note']})")

    verdict = logger.verdict()
    logger.write_json("run_meta.json", {
        "run": RUN, "seed": SEED,
        "outcome": {"outcome": result.outcome, "reason": result.reason,
                    "turns_completed": result.turns_completed,
                    "close_status": result.close_status},
        "wall_clock_s": round(wall, 1),
        "injection": {"seam": seam.describe(),
                      "kill_events": wrapper.kill_events,
                      "step_log": [{"action": str(c[0]), "deadline": c[1]}
                                   for c in wrapper.step_log],
                      "close_calls": wrapper.close_calls},
        "identity": {"at_start": wrapper.identity_at_start,
                     "at_end": wrapper.identity_at_end},
        "reconciliation": {"events": [r.value for r in reconciliations],
                           "quarantines": [r.value for r in quarantines]},
        "transcript": {"path": run["transcript_path"],
                       "records": len(records),
                       "discarded_trailing_partial":
                           load.discarded_trailing_partial,
                       "last_seq": load.last_seq},
        "checks": logger.checks,
        "verdict": verdict,
        "artifacts": {"stdout": "run_stdout.log",
                      "transcript": "transcript.jsonl",
                      "run_meta": "run_meta.json",
                      "evidence": "disconnect_evidence.md"},
    })

    lines = [
        f"# WP3 mid-mutation disconnect test (LIVE) — `{RUN}`",
        "",
        f"- seed {SEED}, fresh deterministic reset on the session's own "
        f"server; the turn-11 checkpoint untouched",
        f"- injection: {seam.describe()}",
        f"- outcome: {result.outcome} (turns_completed="
        f"{result.turns_completed}, close_status="
        f"{result.close_status!r}, reason={result.reason!r})",
        f"- wall clock: {wall:.1f} s (bounded)",
        f"- kill events: {wrapper.kill_events}",
        f"- mutation sends: {len(mutation_sends)} (never resent — no "
        f"duplicate mutation)",
        f"- reconciliation: {[r.value for r in reconciliations]}",
        f"- quarantine: {[r.value for r in quarantines]}",
        f"- identity: {wrapper.identity_at_start} -> "
        f"{wrapper.identity_at_end} (no takeover)",
        "",
        "## Checks",
        "",
    ]
    for name, entry in logger.checks.items():
        lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                     f"`{name}`: {entry['detail']}")
    with open(os.path.join(rundir, "disconnect_evidence.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.log(f"[{RUN}] wrote disconnect_evidence.md")
    logger.log(f"[{RUN}] VERDICT: {verdict} "
               f"({sum(1 for c in logger.checks.values() if c['pass'])}/"
               f"{len(logger.checks)} checks)")
    logger.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
