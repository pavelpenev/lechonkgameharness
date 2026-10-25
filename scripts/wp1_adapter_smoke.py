# SPDX-License-Identifier: GPL-3.0-or-later
"""WP1 package B live smoke: AdapterSession + candidate table + execute.

Behavioral evidence for the package-B adapter on REAL data (offline tests
live in tests/test_candidates.py and tests/test_execute.py). Per
design-plan-freeciv-v1.1.md §7.1:

- a FRESH deterministic reset spawns its own new game server; the turn-11
  checkpoint and any pre-existing game are NOT touched; the fresh server
  quits idle (~20 s, --quitidle 20) after the client disconnects — expected;
- run from a dedicated working directory (upstream writes logs/,
  recordings/, game_scores/ into CWD);
- env.close() is only ever called through AdapterSession.close() — the
  guarded wrapper with a hard deadline and a skip path (the upstream
  CivController.close() contains an unbounded game-over wait loop,
  civ_controller.py:578-583);
- the scripted sequence (per-action read-backs via lechonk.adapter.execute):
  set research, found a city (special-cased two-round-trip read-back —
  executed BEFORE production because this scenario starts without a city),
  set city production, move one unit, sentry one unit (live PR #7 ActSEntry
  send-verification — docs/gate-record.md entry 5; the upstream read path
  cancels ACTIVITY_SENTRY within the same step, so the sentry read-back is
  verified or inconclusive-by-quirk, never contradicted), then end the
  phase with the pipeline-owned finish primitive (step(None)) and observe
  the phase advance.

Outputs (persisted under artifacts/runs/wp1-adapter/):
- smoke_stdout.log               — this run's stdout (tee at the shell)
- candidates_turn1_initial.json  — the CandidateTable right after reset
- candidates_turn1_after_founding.json — the table after the city is founded
- readbacks.json                 — per-action execution results + read-backs
- run_meta.json                  — seeds, ports, checks, verdict
- workdir/                       — dedicated CWD for upstream side effects
"""

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, "artifacts", "runs", "wp1-adapter")
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream
os.makedirs(WORKDIR, exist_ok=True)

SEED = 20261025  # distinct from the WP0 spike (42) and package A (20261024)
PUBSTATUS = "http://localhost:8080/pubstatus"

print(f"[wp1-adapter-smoke] repo={REPO}")
print(f"[wp1-adapter-smoke] rundir={RUNDIR}")

# --- wait for the stack (hardened startup: bounded pubstatus retries) ------
deadline = time.time() + 120
while True:
    try:
        with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
            if r.status == 200:
                print("[wp1-adapter-smoke] pubstatus 200 — stack up")
                break
    except Exception as e:
        if time.time() > deadline:
            print(f"[wp1-adapter-smoke] FATAL: pubstatus unreachable: {e!r}")
            sys.exit(2)
        time.sleep(2)

# The adapter package imports lazily and never touches the network at import
# time; the civrealm import happens inside AdapterSession.start() (worker
# thread), only now that the stack is up.
from lechonk.adapter import execute as exe  # noqa: E402
from lechonk.adapter.session import (  # noqa: E402
    AdapterSession,
    SessionError,
    SessionTimeoutError,
)

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
    print(f"[wp1-adapter-smoke] wrote {path}")
    return path


session = None
readbacks = []
verdict = "FAIL"
error_text = ""
by_domain1 = {}
by_domain2 = {}
raw_dipl = None
raw_sentry = None
sentry_cands = []
close_status = None
refresh_elapsed = None

try:
    # --- session: fresh deterministic reset (own game server) --------------
    session = AdapterSession(workdir=WORKDIR, seed=SEED)
    t0 = time.time()
    session.start()
    print(f"[wp1-adapter-smoke] reset ok in {time.time() - t0:.1f}s "
          f"turn={session.turn} player_id={session.player_id} "
          f"username={session.username} client_port={session.port}")
    check("reset_ok", session.turn >= 1, f"turn={session.turn}")
    check("username_not_renamed", session.username == "myagent",
          f"username={session.username!r} (self_play=False: congestion must "
          f"raise, never silently rename)")

    # --- standalone read path (seeded; upstream _get_info() opens with a
    # lock_control() that blocks until a server packet arrives) -------------
    t0 = time.time()
    snap = session.refresh()
    refresh_elapsed = time.time() - t0
    print(f"[wp1-adapter-smoke] refresh() in {refresh_elapsed:.2f}s "
          f"turn={snap['info'].get('turn')}")
    check("refresh_read_path_ok", refresh_elapsed < 30.0,
          f"{refresh_elapsed:.2f}s (seeded read; unseeded it blocks ~30 s "
          f"on the next server ping)")

    # --- enumerate the candidate table (turn 1, right after reset) ---------
    table1 = session.build_candidate_table()
    persist("candidates_turn1_initial.json", table1.model_dump(mode="json"))
    by_domain1 = {}
    for c in table1.candidates:
        by_domain1[c.domain] = by_domain1.get(c.domain, 0) + 1
    print(f"[wp1-adapter-smoke] turn-1 candidates by domain: {by_domain1} "
          f"(total {len(table1.candidates)})")
    check("table_turn1_nonempty", len(table1.candidates) > 0,
          f"{len(table1.candidates)} candidates")
    check("table_turn1_unit_domain", by_domain1.get("unit", 0) > 0,
          f"{by_domain1.get('unit', 0)} unit candidates")
    check("table_turn1_tech_domain", by_domain1.get("tech", 0) > 0,
          f"{by_domain1.get('tech', 0)} tech candidates")

    # PR #7 gate re-verification (docs/gate-record.md entry 5): the sentry
    # verb (ActSEntry) must be available LIVE in the candidate table.
    sentry_cands = [c for c in table1.candidates
                    if c.domain == "unit" and c.verb == "sentry"]
    raw_avail = (session.info.get("available_actions") or {}).get("unit", {})
    raw_sentry = {uid: am.get("sentry") for uid, am in raw_avail.items()
                  if isinstance(am, dict) and "sentry" in am}
    check("sentry_available_live_pr7", len(sentry_cands) > 0,
          f"{len(sentry_cands)} sentry candidate(s); raw upstream "
          f"available_actions sentry flags: {raw_sentry}")

    # The dipl domain at turn 1: no contact yet (first contact defaults to
    # war, plan §3), so no legal diplomacy action exists — persist the raw
    # upstream dipl structure as evidence that the domain was enumerated.
    raw_dipl = (session.info.get("available_actions") or {}).get("dipl", {})
    print(f"[wp1-adapter-smoke] raw available_actions['dipl'] at turn 1: "
          f"{json.dumps(raw_dipl, default=str)[:400]}")

    def pick(table, domain, predicate):
        for c in table.candidates:
            if c.domain == domain and predicate(c):
                return c
        return None

    def run_action(label, cand):
        """Execute one candidate with a per-action read-back; record evidence."""
        result = exe.execute_candidate(session, cand)
        rec = {
            "label": label,
            "candidate_id": result.candidate_id,
            "action": [result.action[0], str(result.action[1]),
                       result.action[2]],
            "status": result.status,
            "readback_status": result.readback.status,
            "readback_detail": result.readback.detail,
            "readback_evidence": dict(result.readback.evidence),
            "terminated": result.terminated,
            "truncated": result.truncated,
            "turn_before": result.turn_before,
            "turn_after": result.turn_after,
            "elapsed_s": round(result.elapsed_s, 3),
        }
        readbacks.append(rec)
        print(f"[action] {label}: {result.candidate_id} -> "
              f"{result.action} read-back={result.readback.status} "
              f"({result.readback.detail}) [{result.elapsed_s:.2f}s]")
        return result

    # --- scripted sequence --------------------------------------------------
    # 1. set research (tech domain)
    table = session.build_candidate_table()
    cand = pick(table, "tech",
                lambda c: c.verb.startswith("research_tech_")
                and c.payload.get("tech_name") == "Alphabet")
    if cand is None:
        cand = pick(table, "tech", lambda c: c.verb.startswith("research_tech_"))
    if cand is not None:
        result = run_action("set_research", cand)
        check("research_readback_verified",
              result.readback.status == "verified", result.readback.detail)
    else:
        check("research_readback_verified", False, "no research candidate")

    # 2. found a city (special-cased two-round-trip read-back). Executed
    #    before production: this scenario starts WITHOUT a city, so the
    #    production step needs the city founded first.
    table = session.build_candidate_table()
    cand = pick(table, "unit", lambda c: c.verb == "build_city")
    if cand is not None:
        result = run_action("found_city", cand)
        check("build_city_readback_verified",
              result.readback.status == "verified", result.readback.detail)
    else:
        check("build_city_readback_verified", False,
              "no build_city candidate (not legal this run)")

    # --- re-enumerate: the city domain is now covered ----------------------
    table2 = session.build_candidate_table()
    persist("candidates_turn1_after_founding.json",
            table2.model_dump(mode="json"))
    by_domain2 = {}
    for c in table2.candidates:
        by_domain2[c.domain] = by_domain2.get(c.domain, 0) + 1
    print(f"[wp1-adapter-smoke] post-founding candidates by domain: "
          f"{by_domain2} (total {len(table2.candidates)})")
    check("table_city_domain_covered", by_domain2.get("city", 0) > 0,
          f"{by_domain2.get('city', 0)} city candidates after founding")

    # 3. set city production (city domain)
    table = session.build_candidate_table()
    cand = pick(table, "city", lambda c: c.verb.startswith("produce_"))
    if cand is not None:
        result = run_action("set_city_production", cand)
        check("production_readback_verified",
              result.readback.status == "verified", result.readback.detail)
    else:
        check("production_readback_verified", False, "no produce candidate")

    # 4. move one unit (goto; prefer a unit that is not a Settlers so the
    #    founders keep their moves)
    table = session.build_candidate_table()
    moved_uid = None
    cand = pick(table, "unit", lambda c: (
        c.verb.startswith("goto_")
        and session.obs.get("unit", {}).get(int(c.actor_id), {})
        .get("type_rule_name") not in ("Settlers",)))
    if cand is None:
        cand = pick(table, "unit", lambda c: c.verb.startswith("goto_"))
    if cand is not None:
        moved_uid = int(cand.actor_id)
        result = run_action("move_unit", cand)
        check("move_readback_verified",
              result.readback.status == "verified", result.readback.detail)
    else:
        check("move_readback_verified", False, "no goto candidate")

    # 5. sentry one unit (live PR #7 ActSEntry send-verification; a unit
    #    different from the one just moved). The upstream read path cancels
    #    ACTIVITY_SENTRY within the same step (unit_actions.py:183-226), so
    #    the observation cannot confirm the sentry: verified or
    #    inconclusive-by-quirk both prove the packet was sent and the step
    #    completed cleanly.
    table = session.build_candidate_table()
    cand = pick(table, "unit", lambda c: (
        c.verb == "sentry" and int(c.actor_id) != moved_uid))
    if cand is None:
        cand = pick(table, "unit", lambda c: c.verb == "sentry")
    if cand is not None:
        result = run_action("sentry_unit", cand)
        check("sentry_sent_live_pr7",
              result.readback.status in ("verified", "inconclusive"),
              f"readback={result.readback.status} ({result.readback.detail})")
    else:
        check("sentry_sent_live_pr7", False, "no sentry candidate")

    # 6. finish the phase (pipeline-owned primitive) and observe the advance
    turn_before = session.turn
    t0 = time.time()
    outcome = exe.finish_phase(session)
    finish_elapsed = time.time() - t0
    turn_after = int(outcome.info.get("turn", turn_before) or turn_before)
    print(f"[finish] step(None) in {finish_elapsed:.1f}s: turn "
          f"{turn_before} -> {turn_after} terminated={outcome.terminated} "
          f"truncated={outcome.truncated}")
    check("finish_phase_advanced",
          turn_after > turn_before or outcome.terminated or outcome.truncated,
          f"turn {turn_before} -> {turn_after}")

    persist("readbacks.json", {
        "run": "wp1-adapter-smoke",
        "seed": SEED,
        "actions": readbacks,
        "finish_phase": {
            "turn_before": turn_before,
            "turn_after": turn_after,
            "elapsed_s": round(finish_elapsed, 3),
            "terminated": bool(outcome.terminated),
            "truncated": bool(outcome.truncated),
        },
    })

    verdict = ("PASS" if checks and all(c["pass"] for c in checks.values())
               else "FAIL")
except (SessionError, SessionTimeoutError) as e:
    error_text = f"{type(e).__name__}: {e}"
    print(f"[wp1-adapter-smoke] FATAL (session): {error_text}")
    traceback.print_exc()
    verdict = "FAIL"
except Exception as e:
    error_text = f"{type(e).__name__}: {e}"
    print(f"[wp1-adapter-smoke] FATAL: {error_text}")
    traceback.print_exc()
    verdict = "FAIL"
finally:
    close_status = None
    if session is not None:
        # Guarded close only (hard deadline + skip path; never unguarded).
        close_status = session.close()
        print(f"[wp1-adapter-smoke] guarded close status: {close_status}")

try:
    vendored_head = subprocess.run(
        ["git", "-C", os.path.join(REPO, "vendor", "civrealm"),
         "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception:
    vendored_head = None

meta = {
    "run": "wp1-adapter-smoke",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": SEED,
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
    "domain_coverage_turn1_initial": by_domain1,
    "domain_coverage_turn1_after_founding": by_domain2,
    "raw_available_actions_dipl_turn1": raw_dipl,
    "raw_available_actions_sentry_flags": raw_sentry,
    "sentry_candidates_turn1": [c.id for c in sentry_cands],
    "checks": checks,
    "refresh_elapsed_s": (round(refresh_elapsed, 3)
                          if refresh_elapsed is not None else None),
    "readbacks_file": os.path.join(RUNDIR, "readbacks.json"),
    "error": error_text or None,
    "env_close": close_status,
    "env_close_note": "AdapterSession.close() only — guarded with a hard "
                      "deadline (<= 30 s) and a skip path; env.close() is "
                      "never called unguarded (unbounded game-over wait "
                      "loop, civ_controller.py:578-583). The fresh game "
                      "server quits idle (~20 s, --quitidle 20) after the "
                      "client disconnects at process exit.",
    "turn_11_checkpoint_untouched": True,
    "verdict": verdict,
}
persist("run_meta.json", meta)

print(f"[wp1-adapter-smoke] checks: "
      f"{sum(1 for c in checks.values() if c['pass'])}/{len(checks)} passed")
print(f"[wp1-adapter-smoke] VERDICT: {verdict}")
sys.exit(0 if verdict == "PASS" else 1)
