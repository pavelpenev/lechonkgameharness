# SPDX-License-Identifier: GPL-3.0-or-later
"""WP1 package C live acceptance re-run AFTER the §7.4 production-hysteresis
amendment (2026-10-24, user-approved): the comparator plays 20 turns
UNATTENDED with the amended production ladder.

This is the amended twin of scripts/wp1_comparator_run.py. Differences:
- fresh deterministic seed (20261027; distinct from the pre-amendment run's
  20261026, package A's 20261024, B's 20261025, and the WP0 spike's 42);
- a FRESH deterministic reset spawns its own new game server; the turn-11
  checkpoint and any pre-existing game are NOT touched (same as the
  original driver);
- outputs persist under artifacts/runs/wp1-comparator-amended/ (the
  pre-amendment evidence under artifacts/runs/wp1-comparator/ is left
  untouched);
- the acceptance set gains the amendment's target check: NO production
  alternation 2-cycle. The pre-amendment failure mode (evidence:
  artifacts/runs/wp1-comparator/decisions.jsonl, seed 20261026) was the
  same city alternating Warriors -> Barracks -> Warriors every turn
  (upstream available_actions never lists a city's CURRENT production
  target as a produce candidate, so the frozen ladder could never see
  that the defensive tier was already satisfied). The amended ladder
  (production hysteresis) must instead stabilize on a target or progress
  legitimately down the ladder.

2-cycle detection: per own city, the turn-start production timeline (from
each turn's snapshot) and the executed produce-decision timeline are both
scanned for a strict X -> Y -> X alternation over consecutive
observations. Any hit fails the run (reported honestly, never forced).

The turn driver itself is deliberately MINIMAL and unchanged in structure
(the WP2 runner, core/runner.py state machine + supervisor, is NOT in
scope and is not built here). Per turn: refresh state -> build the
candidate table -> extract the fog-gated Snapshot -> apply the comparator
policy (lechonk.comparator.scripted.decide, §7.4 AS AMENDED) -> execute
each chosen action with a per-action read-back (lechonk.adapter.execute)
-> finish the phase -> next turn.

Outputs (persisted under artifacts/runs/wp1-comparator-amended/):
- run_stdout.log            — this run's stdout (tee at the shell)
- decisions.jsonl           — one JSON record per executed/rejected action
- decisions_turn<N>.json    — per-turn decision log + read-backs + summary
- final_state.json          — the final game state summary
- run_meta.json             — seeds, ports, turn count, checks, verdict
- infra_failure_attempt1.json — only if attempt 1 hit an infra error
- workdir/                  — dedicated CWD for upstream side effects
"""

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "wp1-comparator-amended"
RUNDIR = os.path.join(REPO, "artifacts", "runs", RUN)
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream
os.makedirs(WORKDIR, exist_ok=True)

SEED = 20261027  # fresh: distinct from wp1-comparator (20261026), A (20261024), B (20261025), spike (42)
TURNS = 20
PUBSTATUS = "http://localhost:8080/pubstatus"
ATTEMPT = int(os.environ.get("WP1_COMPARATOR_ATTEMPT", "1"))

print(f"[{RUN}] repo={REPO}")
print(f"[{RUN}] rundir={RUNDIR} attempt={ATTEMPT} seed={SEED} turns={TURNS}")
print(f"[{RUN}] policy: lechonk.comparator.scripted.decide "
      f"(design-plan-freeciv-v1.1.md §7.4 AS AMENDED 2026-10-24: "
      f"production hysteresis)")

# --- wait for the stack (hardened startup: bounded pubstatus retries) ------
deadline = time.time() + 120
while True:
    try:
        with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
            if r.status == 200:
                print(f"[{RUN}] pubstatus 200 — stack up")
                break
    except Exception as e:
        if time.time() > deadline:
            print(f"[{RUN}] FATAL: pubstatus unreachable: {e!r}")
            sys.exit(2)
        time.sleep(2)

# The adapter package imports lazily and never touches the network at import
# time; the civrealm import happens inside AdapterSession.start() (worker
# thread), only now that the stack is up. The comparator is CivRealm-free.
from lechonk.adapter import execute as exe  # noqa: E402
from lechonk.adapter.obs_extract import ExtractOutcome, extract_snapshot  # noqa: E402
from lechonk.adapter.session import (  # noqa: E402
    AdapterSession,
    SessionError,
    SessionTimeoutError,
)
from lechonk.comparator import scripted  # noqa: E402
from lechonk.comparator.scripted import is_exposed_verb  # noqa: E402

checks = {}


def check(name, cond, detail=""):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
    return bool(cond)


def persist(name, payload):
    path = os.path.join(RUNDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")
    print(f"[{RUN}] wrote {path}")
    return path


def append_jsonl(name, record):
    path = os.path.join(RUNDIR, name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def snapshot_summary(snap):
    return {
        "turn": snap.turn,
        "player": {"name": snap.player.name, "score": snap.player.score,
                   "gold": snap.player.gold,
                   "techs_researched": snap.player.techs_researched,
                   "research_target": snap.player.research.current_target,
                   "research_turns_left": snap.player.research.turns_left},
        "research_candidates": [c.name for c in snap.research_candidates],
        "cities": [{"id": c.id, "name": c.name, "size": c.size,
                    "production": c.production,
                    "shield_stock": c.shield_stock,
                    "production_progress": c.production_progress,
                    "buy_cost": c.buy_cost} for c in snap.cities if c.is_own],
        "units": [{"id": u.id, "type": u.type_name, "x": u.x, "y": u.y,
                   "activity": u.activity_name, "terrain": u.terrain}
                  for u in snap.units if u.is_own],
        "diplomacy": [{"counterpart": d.counterpart_name, "state": d.state,
                       "meeting_active": d.meeting_active}
                      for d in snap.diplomacy],
        "map": {"size": [snap.map_view.width, snap.map_view.height],
                "tile_counts": snap.map_view.tile_counts},
        "omitted": snap.omitted.summary,
        "snapshot_tokens_estimated": snap.estimated_tokens(),
    }


def has_2cycle(timeline):
    """True iff the (turn, value) timeline contains a strict X -> Y -> X
    alternation over consecutive non-None observations (the pre-amendment
    Warriors/Barracks failure mode)."""
    vals = [v for _, v in timeline if v is not None]
    return any(vals[i] == vals[i + 2] != vals[i + 1]
               for i in range(len(vals) - 2))


session = None
turns_done = 0
actions_executed = 0
actions_rejected = 0
all_records = []  # every decision record (executed or rejected), for checks
readback_counts = {"verified": 0, "inconclusive": 0, "contradicted": 0}
verbs_seen = set()
whitelist_violations = []
per_turn_stats = []
production_timelines = {}        # city id -> [(turn, production at turn start)]
produce_decision_timelines = {}  # city id -> [(turn, target set this turn)]
last_snapshot = None
final_outcome = None
error_text = ""
verdict = "FAIL"
run_started = time.time()

try:
    # --- session: fresh deterministic reset (own game server) --------------
    session = AdapterSession(workdir=WORKDIR, seed=SEED)
    t0 = time.time()
    session.start()
    print(f"[{RUN}] reset ok in {time.time() - t0:.1f}s "
          f"turn={session.turn} player_id={session.player_id} "
          f"username={session.username} client_port={session.port}")
    check("reset_ok", session.turn >= 1, f"turn={session.turn}")
    check("username_not_renamed", session.username == "myagent",
          f"username={session.username!r} (self_play=False: congestion must "
          f"raise, never silently rename)")
    check("policy_is_comparator", scripted.decide.__module__
          == "lechonk.comparator.scripted", scripted.decide.__module__)

    jsonl_path = os.path.join(RUNDIR, "decisions.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)  # fresh attempt: the log is per-attempt

    while turns_done < TURNS:
        turn_t0 = time.time()
        turn_no = session.turn
        print(f"[{RUN}] === turn {turn_no} "
              f"(completed {turns_done}/{TURNS}) ===")

        # --- refresh state (seeded read; see AdapterSession.refresh) --------
        session.refresh()

        # --- build candidates (collects gating context + combat odds) ------
        table = session.build_candidate_table()
        print(f"[{RUN}] turn {turn_no}: {len(table.candidates)} candidates")

        # --- extract the fog-gated snapshot --------------------------------
        extraction = extract_snapshot(
            session.obs, session.info,
            terminated=False, truncated=False,
            my_player_id=session.player_id,
            ruleset=session.ruleset,
            action_probs=session.action_probs or None,
        )
        if extraction.outcome is not ExtractOutcome.OK or extraction.snapshot is None:
            final_outcome = (f"extraction classified {extraction.outcome.value}: "
                             f"{extraction.reason}")
            print(f"[{RUN}] STOP: {final_outcome}")
            break
        snap = extraction.snapshot
        last_snapshot = snap

        # --- production timeline (turn-start production per own city) -------
        for c in snap.cities:
            if c.is_own:
                production_timelines.setdefault(c.id, []).append(
                    (turn_no, c.production))

        # --- apply the comparator policy (§7.4 AS AMENDED) ------------------
        log = scripted.decide(snap, table)
        print(f"[{RUN}] turn {turn_no}: policy={log.policy} "
              f"decisions={len(log.decisions)} by_domain={log.by_domain}")

        # --- execute each chosen action with a read-back ---------------------
        turn_records = []
        seq = 0
        current_table = table
        for decision in log.decisions:
            seq += 1
            rec = {
                "turn": turn_no,
                "seq": seq,
                "policy": log.policy,
                "domain": decision.domain,
                "candidate_id": decision.candidate_id,
                "verb": decision.verb,
                "actor_id": decision.actor_id,
                "reason": decision.reason,
                "trigger": decision.trigger,
                "provenance_class": decision.provenance_class,  # WP2-ledger placeholder
                "detail": decision.detail,
            }
            # Live whitelist evidence: the comparator never emits excluded
            # verbs (exploration/autosettler are absent from the candidate
            # table by construction; this asserts it per action).
            if not is_exposed_verb(decision.verb):
                whitelist_violations.append(rec)
                print(f"[{RUN}] WHITELIST VIOLATION: {decision.verb}")
            verbs_seen.add(decision.verb)
            # Minimal batch revalidation: the candidate must still be in the
            # freshly rebuilt table.
            cand = current_table.by_id().get(decision.candidate_id)
            if cand is None:
                actions_rejected += 1
                rec.update({"executed": False,
                            "rejection_class": "state_fog_error",
                            "readback_status": None,
                            "readback_detail": "candidate no longer in the "
                                               "refreshed table (stale)"})
                print(f"[action] turn {turn_no} #{seq} REJECTED (stale): "
                      f"{decision.candidate_id}")
            else:
                result = exe.execute_candidate(session, cand)
                actions_executed += 1
                readback_counts[result.readback.status] = \
                    readback_counts.get(result.readback.status, 0) + 1
                rec.update({
                    "executed": True,
                    "action": [result.action[0], str(result.action[1]),
                               result.action[2]],
                    "readback_status": result.readback.status,
                    "readback_detail": result.readback.detail,
                    "readback_evidence": dict(result.readback.evidence),
                    "elapsed_s": round(result.elapsed_s, 3),
                    "turn_before": result.turn_before,
                    "turn_after": result.turn_after,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                })
                print(f"[action] turn {turn_no} #{seq}: {decision.candidate_id}"
                      f" -> {result.action} read-back={result.readback.status}"
                      f" ({result.readback.detail}) [{result.elapsed_s:.2f}s]")
                if decision.verb.startswith("produce_") \
                        and result.readback.status == "verified":
                    produce_decision_timelines.setdefault(
                        decision.actor_id, []).append(
                            (turn_no, decision.detail.split(" -> ")[-1]
                             .split(" (")[0]))
                if result.terminated or result.truncated:
                    final_outcome = (f"action {decision.candidate_id} returned "
                                     f"terminated={result.terminated} "
                                     f"truncated={result.truncated}")
                    print(f"[{RUN}] STOP: {final_outcome}")
                # rebuild the table for the next decision's revalidation
                current_table = session.build_candidate_table()
            turn_records.append(rec)
            all_records.append(rec)
            append_jsonl("decisions.jsonl", rec)
            if final_outcome:
                break

        # --- finish phase (pipeline-owned primitive) --------------------------
        finish_rec = None
        if not final_outcome:
            t0 = time.time()
            outcome = exe.finish_phase(session)
            finish_elapsed = time.time() - t0
            turn_after = int(outcome.info.get("turn", turn_no) or turn_no)
            finish_rec = {
                "turn_before": turn_no,
                "turn_after": turn_after,
                "elapsed_s": round(finish_elapsed, 3),
                "terminated": bool(outcome.terminated),
                "truncated": bool(outcome.truncated),
            }
            print(f"[finish] turn {turn_no} -> {turn_after} in "
                  f"{finish_elapsed:.1f}s terminated={outcome.terminated} "
                  f"truncated={outcome.truncated}")
            if outcome.terminated or outcome.truncated:
                final_outcome = (f"finish_phase returned "
                                 f"terminated={outcome.terminated} "
                                 f"truncated={outcome.truncated}")
                print(f"[{RUN}] STOP: {final_outcome}")

        turns_done += 1
        turn_elapsed = time.time() - turn_t0
        per_turn_stats.append({
            "turn": turn_no,
            "decisions": len(log.decisions),
            "executed": sum(1 for r in turn_records if r.get("executed")),
            "rejected": sum(1 for r in turn_records if not r.get("executed")),
            "readbacks": {s: sum(1 for r in turn_records
                                 if r.get("readback_status") == s)
                          for s in ("verified", "inconclusive", "contradicted")},
            "production_by_city": {str(c.id): c.production
                                   for c in snap.cities if c.is_own},
            "wall_clock_s": round(turn_elapsed, 2),
            "finish": finish_rec,
        })
        persist(f"decisions_turn{turn_no}.json", {
            "run": RUN,
            "attempt": ATTEMPT,
            "seed": SEED,
            "turn": turn_no,
            "policy": log.policy,
            "snapshot": snapshot_summary(snap),
            "candidate_count": len(table.candidates),
            "decisions": turn_records,
            "finish": finish_rec,
            "wall_clock_s": round(turn_elapsed, 2),
        })
        print(f"[{RUN}] turn {turn_no} done in {turn_elapsed:.1f}s "
              f"({turns_done}/{TURNS} turns completed)")
        if final_outcome:
            break

    # --- acceptance checks ---------------------------------------------------
    check("turns_completed_20", turns_done >= TURNS,
          f"{turns_done}/{TURNS} turns completed unattended")
    check("no_contradicted_readbacks", readback_counts["contradicted"] == 0,
          f"read-backs: {readback_counts}")
    check("no_whitelist_violations", not whitelist_violations,
          f"verbs seen: {sorted(verbs_seen)}")
    # §3 batch semantics: a stale action mid-batch (e.g. a city founded
    # mid-batch invalidating stacked workers' improve actions) is REJECTED
    # and logged — never silently retried. The check is that every rejection
    # carries a class + detail; the count is reported, not hidden.
    rejected = [r for r in all_records if not r.get("executed")]
    check("rejections_logged_not_silent",
          all(r.get("rejection_class") == "state_fog_error"
              and r.get("readback_detail") for r in rejected),
          f"{len(rejected)} stale rejection(s), all logged with "
          f"rejection_class + detail (plan §3 batch semantics)")
    check("actions_executed", actions_executed > 0,
          f"{actions_executed} actions executed, "
          f"{sum(readback_counts.values())} read-backs")
    # §7.4 AMENDMENT acceptance: the production alternation 2-cycle is GONE.
    # Both the turn-start production timeline and the executed
    # produce-decision timeline are scanned per city for a strict X -> Y -> X
    # alternation over consecutive observations.
    cycled_cities = sorted(
        cid for cid, tl in production_timelines.items() if has_2cycle(tl))
    cycled_decisions = sorted(
        cid for cid, tl in produce_decision_timelines.items()
        if has_2cycle(tl))
    check("no_production_alternation_2cycle",
          not cycled_cities and not cycled_decisions,
          f"turn-start production timelines: "
          f"{ {cid: [v for _, v in tl] for cid, tl in sorted(production_timelines.items())} }; "
          f"2-cycle cities: {cycled_cities or 'none'}; "
          f"2-cycle decision cities: {cycled_decisions or 'none'} "
          f"(pre-amendment evidence: artifacts/runs/wp1-comparator/, "
          f"city 111 alternated Warriors/Barracks turns 2-20)")
    if final_outcome:
        check("run_not_interrupted", False, final_outcome)
    else:
        check("run_not_interrupted", True, "no stall, no early termination")

    verdict = ("PASS" if checks and all(c["pass"] for c in checks.values())
               else "FAIL")
except (SessionError, SessionTimeoutError) as e:
    # INFRA error: a single bounded retry of the whole run is acceptable
    # (logged); it is never a silent loop-retry.
    error_text = f"{type(e).__name__}: {e}"
    print(f"[{RUN}] FATAL (infra, attempt {ATTEMPT}): {error_text}")
    traceback.print_exc()
    if ATTEMPT == 1:
        persist("infra_failure_attempt1.json", {
            "run": RUN,
            "attempt": 1,
            "seed": SEED,
            "error": error_text,
            "turns_completed": turns_done,
            "actions_executed": actions_executed,
            "readback_counts": readback_counts,
            "note": "bounded retry (once): re-executing the whole run with "
                    "WP1_COMPARATOR_ATTEMPT=2; one env per process, so the "
                    "retry is a fresh process with a fresh env",
        })
        print(f"[{RUN}] bounded retry (once): re-executing with "
              "WP1_COMPARATOR_ATTEMPT=2")
        retry_env = dict(os.environ, WP1_COMPARATOR_ATTEMPT="2")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                              env=retry_env)
        sys.exit(proc.returncode)
    verdict = "FAIL"
except Exception as e:
    error_text = f"{type(e).__name__}: {e}"
    print(f"[{RUN}] FATAL: {error_text}")
    traceback.print_exc()
    verdict = "FAIL"
finally:
    close_status = None
    if session is not None:
        # Guarded close only (hard deadline + skip path; never unguarded).
        close_status = session.close()
        print(f"[{RUN}] guarded close status: {close_status}")

# --- final game state summary ------------------------------------------------
final_state = None
if last_snapshot is not None:
    final_state = {
        "run": RUN,
        "attempt": ATTEMPT,
        "seed": SEED,
        "snapshot": snapshot_summary(last_snapshot),
        "game_turn_final": session.turn if session else None,
        "turns_completed": turns_done,
        "outcome": final_outcome,
    }
    persist("final_state.json", final_state)

# --- run meta ------------------------------------------------------------------
try:
    vendored_head = subprocess.run(
        ["git", "-C", os.path.join(REPO, "vendor", "civrealm"),
         "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception:
    vendored_head = None

infra_retry = None
infra_file = os.path.join(RUNDIR, "infra_failure_attempt1.json")
if ATTEMPT > 1 and os.path.exists(infra_file):
    with open(infra_file, encoding="utf-8") as f:
        infra_retry = json.load(f)

meta = {
    "run": RUN,
    "attempt": ATTEMPT,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "unattended": True,
    "policy": "lechonk.comparator.scripted.decide (design-plan-freeciv-v1.1.md "
              "§7.4 AS AMENDED 2026-10-24: production hysteresis)",
    "amendment": "§7.4 AMENDMENT (2026-10-24, user-approved): a production "
                 "tier is SATISFIED if the city's current production target "
                 "already belongs to that tier's output class, even when "
                 "absent from the candidate table; a satisfied top "
                 "applicable tier does not switch",
    "prior_run_reference": "artifacts/runs/wp1-comparator/ (seed 20261026, "
                           "pre-amendment 2-cycle evidence — untouched)",
    "seed": SEED,
    "turns_target": TURNS,
    "turns_completed": turns_done,
    "game_turn_final": session.turn if session else None,
    "fc_args_overrides": {
        "self_play": False, "aifill": 2,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": SEED, "debug.gameseed": SEED,
        "debug.agentseed": SEED,
        "server_timeout": 300, "begin_turn_timeout": 900,
        "wait_for_timeout": 60, "http_timeout_s": 15.0,
    },
    "client_port": session.port if session else None,
    "player_id": session.player_id if session else None,
    "username": session.username if session else None,
    "workdir": WORKDIR,
    "vendored_civrealm_head": vendored_head,
    "wall_clock_s": round(time.time() - run_started, 1),
    "per_turn_stats": per_turn_stats,
    "production_timelines": {str(cid): [{"turn": t, "production": v}
                                        for t, v in tl]
                             for cid, tl in sorted(production_timelines.items())},
    "produce_decision_timelines": {str(cid): [{"turn": t, "target": v}
                                               for t, v in tl]
                                   for cid, tl in sorted(produce_decision_timelines.items())},
    "actions_executed": actions_executed,
    "actions_rejected_stale": actions_rejected,
    "readback_counts": readback_counts,
    "verbs_seen": sorted(verbs_seen),
    "whitelist_violations": whitelist_violations,
    "whitelist_note": "every executed action's verb checked against the "
                      "exposed v1 surface (lechonk.comparator.scripted."
                      "is_exposed_verb); excluded verbs (exploration/"
                      "autosettler/...) never appear",
    "provenance_class_placeholder": "scripted_default (WP1 placeholder for "
                                    "the WP2 ledger)",
    "checks": checks,
    "final_outcome": final_outcome,
    "error": error_text or None,
    "infra_retry": infra_retry,
    "env_close": close_status,
    "env_close_note": "AdapterSession.close() only — guarded with a hard "
                      "deadline (<= 30 s) and a skip path; env.close() is "
                      "never called unguarded (unbounded game-over wait "
                      "loop, civ_controller.py:578-583). The fresh game "
                      "server quits idle (~20 s, --quitidle 20) after the "
                      "client disconnects at process exit.",
    "turn_11_checkpoint_untouched": True,
    "driver_scope_note": "deliberately minimal turn driver; the WP2 runner "
                         "(core/runner.py state machine + supervisor) is "
                         "NOT built here",
    "verdict": verdict,
}
persist("run_meta.json", meta)

print(f"[{RUN}] checks: "
      f"{sum(1 for c in checks.values() if c['pass'])}/{len(checks)} passed")
print(f"[{RUN}] turns completed: {turns_done}/{TURNS}; actions "
      f"executed: {actions_executed}; read-backs: {readback_counts}; "
      f"wall-clock: {time.time() - run_started:.1f}s")
print(f"[{RUN}] VERDICT: {verdict}")
sys.exit(0 if verdict == "PASS" else 1)
