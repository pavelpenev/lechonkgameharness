"""WP2 acceptance artifact: the §3 truncation policy on a large-empire
fixture (design-plan-freeciv-v1.1.md §3 line 32 — "the truncation policy is
validated on a large-empire fixture, not only opening turns").

Drives the synthetic large-empire observation (tests/large_empire_fixture.py:
6 own cities, 47 own units, 13 foreign units, 4 contacts, full 80x50 map
breadth, 20 research candidates, 31 messages) through the real
``lechonk.adapter.obs_extract.extract_snapshot`` under the default 8,000-token
budget and persists, under artifacts/runs/wp2-large-empire/:

- fixture_observation.json / fixture_info.json — the fixture itself (the
  exact observation/info dicts driven through the extractor);
- untruncated_snapshot.json — the reference render with no budget (the
  elision reference);
- rendered_snapshot.json — the policy-facing output under the 8,000-token
  budget (≤ 8,000 tokens, with the omitted disclosure);
- elision_evidence.json — the elision/disclosure evidence: the elided ids in
  elision order, the survival classes (armed-in-contact / working / foreign
  never elided), the candidate-list caps, the delta ≤ 800 and memory ≤ 1,500
  budget evidence under heavy loads, and the acceptance checks;
- run_meta.json — the run record (checks + verdict);
- run_stdout.log — this script's stdout.

Offline: no server, no API, no civrealm import (the extractor is pure over
plain dicts). The same policy is asserted by tests/test_large_empire.py.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # the tests package lives at the repo root
RUNDIR = os.path.join(REPO, "artifacts", "runs", "wp2-large-empire")
os.makedirs(RUNDIR, exist_ok=True)

LOG_PATH = os.path.join(RUNDIR, "run_stdout.log")
_log_file = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")
    _log_file.flush()


def write_json(name, obj):
    path = os.path.join(RUNDIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False,
                  default=str)
        fh.write("\n")
    log(f"wrote {name}")
    return path


from lechonk.adapter.obs_extract import extract_snapshot  # noqa: E402
from lechonk.core.delta import compute_delta, enforce_delta_budget  # noqa: E402
from lechonk.core.memory import StrategyMemory  # noqa: E402
from lechonk.core.schemas import (  # noqa: E402
    DELTA_TOKEN_BUDGET,
    MEMORY_TOKEN_BUDGET,
    SNAPSHOT_TOKEN_BUDGET,
    Commitment,
    MemoryUpdate,
    estimate_tokens,
)

from tests.large_empire_fixture import (  # noqa: E402
    CAPITAL,
    MY_PLAYER,
    N_FAR_IDLE,
    UNIT_TYPE_IDS,
    elision_order_key,
    large_empire_world,
    large_ruleset,
)

NO_ELISION = 10 ** 9
checks = {}


def check(name, cond, detail=""):
    checks[name] = {"pass": bool(cond), "detail": detail}
    log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
    return bool(cond)


def main():
    log("wp2-large-empire start (offline fixture artifact)")
    t0 = time.time()

    obs, info = large_empire_world(turn=30)
    ruleset = large_ruleset()
    write_json("fixture_observation.json", obs)
    write_json("fixture_info.json", info)

    full = extract_snapshot(
        obs, info, my_player_id=MY_PLAYER, ruleset=ruleset,
        token_budget=NO_ELISION).snapshot
    snap = extract_snapshot(
        obs, info, my_player_id=MY_PLAYER, ruleset=ruleset).snapshot
    write_json("untruncated_snapshot.json", json.loads(full.render()))
    write_json("rendered_snapshot.json", json.loads(snap.render()))
    log(f"untruncated render: {full.estimated_tokens()} tokens "
        f"(> {SNAPSHOT_TOKEN_BUDGET} budget)")
    log(f"policy render: {snap.estimated_tokens()} tokens "
        f"(<= {SNAPSHOT_TOKEN_BUDGET} budget)")

    # --- elision evidence ---------------------------------------------------
    elided_ids = list(snap.omitted.elided_unit_ids)
    elidable = [u for u in full.units if u.is_own and not u.in_contact]
    elidable.sort(key=lambda u: elision_order_key(u, CAPITAL))
    expected_prefix = [u.id for u in elidable[:len(elided_ids)]]
    by_id = {u.id: u for u in full.units}
    surviving = {u.id for u in snap.units}

    in_contact = [u for u in full.units if u.is_own and u.in_contact]
    foreign = [u for u in full.units if not u.is_own]
    working = [u for u in full.units
               if u.is_own and u.activity not in (None, 0)]
    near_idle = [u for u in full.units
                 if u.is_own and u.activity in (None, 0) and not u.in_contact
                 and max(abs(u.x - CAPITAL[0]), abs(u.y - CAPITAL[1])) <= 2]
    far_idle = {u.id for u in full.units
                if u.is_own and 200 <= u.id < 200 + N_FAR_IDLE}

    elision_evidence = {
        "fixture": {
            "map": [snap.map_view.width, snap.map_view.height],
            "tile_counts": snap.map_view.tile_counts,
            "own_cities": sum(1 for c in full.cities if c.is_own),
            "own_units": sum(1 for u in full.units if u.is_own),
            "foreign_units": len(foreign),
            "contacts": len(full.diplomacy),
            "research_candidates_untruncated": len(full.research_candidates),
            "messages_untruncated": len(full.messages),
        },
        "budgets": {
            "snapshot_budget": SNAPSHOT_TOKEN_BUDGET,
            "snapshot_untruncated_tokens": full.estimated_tokens(),
            "snapshot_policy_tokens": snap.estimated_tokens(),
            "delta_budget": DELTA_TOKEN_BUDGET,
            "memory_budget": MEMORY_TOKEN_BUDGET,
        },
        "elision": {
            "elided_count": len(elided_ids),
            "elided_unit_ids_in_elision_order": elided_ids,
            "is_exact_prefix_of_elision_order": elided_ids == expected_prefix,
            "all_elided_are_own_idle_not_in_contact": all(
                by_id[uid].is_own and not by_id[uid].in_contact
                and by_id[uid].activity in (None, 0) for uid in elided_ids),
            "elided_are_far_idle_only": set(elided_ids).issubset(far_idle),
            "armed_in_contact_never_elided": all(
                u.id in surviving for u in in_contact),
            "foreign_never_elided": all(u.id in surviving for u in foreign),
            "working_survive_while_idle_elided": (
                all(u.id in surviving for u in working) and bool(elided_ids)),
            "near_capital_idle_survive": all(
                u.id in surviving for u in near_idle),
        },
        "disclosure": snap.omitted.model_dump(mode="json"),
        "candidate_caps": {
            "research_candidates_capped":
                snap.omitted.research_candidates_capped,
            "research_candidates_kept": [
                c.name for c in snap.research_candidates],
            "production_candidates_capped":
                snap.omitted.production_candidates_capped,
            "production_candidates_kept_per_own_city": {
                str(c.id): c.production_candidates
                for c in snap.cities if c.is_own},
            "messages_elided": snap.omitted.messages_elided,
            "messages_kept": len(snap.messages),
        },
    }
    write_json("elision_evidence.json", elision_evidence)

    check("snapshot_within_8000_under_policy",
          snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET,
          f"{snap.estimated_tokens()} <= {SNAPSHOT_TOKEN_BUDGET}")
    check("policy_engaged", full.estimated_tokens() > SNAPSHOT_TOKEN_BUDGET,
          f"untruncated {full.estimated_tokens()} > {SNAPSHOT_TOKEN_BUDGET}")
    check("elision_is_prefix_of_elision_order",
          elided_ids == expected_prefix,
          f"{len(elided_ids)} elided, exact prefix of the elision order")
    check("armed_in_contact_never_elided",
          all(u.id in surviving for u in in_contact),
          f"{len(in_contact)} in-contact units survive")
    check("foreign_never_elided", all(u.id in surviving for u in foreign),
          f"{len(foreign)} foreign units survive")
    check("idle_elided_before_working",
          all(u.id in surviving for u in working) and bool(elided_ids),
          f"{len(working)} working units survive, {len(elided_ids)} idle "
          f"elided")
    check("disclosure_lists_elided_ids",
          set(snap.omitted.elided_unit_ids) == set(elided_ids)
          and "elided" in snap.omitted.summary,
          snap.omitted.summary)
    check("candidate_caps_disclosed",
          snap.omitted.research_candidates_capped
          and snap.omitted.production_candidates_capped
          and snap.omitted.messages_elided == 11,
          f"research capped at {len(snap.research_candidates)}, messages "
          f"elided {snap.omitted.messages_elided}")

    # --- delta ≤ 800 under a heavy change load ------------------------------
    obs_prev, info_prev = large_empire_world(turn=29, variant="prev")
    prev = extract_snapshot(
        obs_prev, info_prev, my_player_id=MY_PLAYER, ruleset=ruleset,
        token_budget=NO_ELISION).snapshot
    raw_delta = compute_delta(prev, full)
    enforced = enforce_delta_budget(raw_delta)
    elision_evidence["delta"] = {
        "raw_tokens": raw_delta.estimated_tokens(),
        "raw_events": len(raw_delta.events),
        "enforced_tokens": enforced.estimated_tokens(),
        "enforced_events": len(enforced.events),
        "truncation_warning": enforced.events[-1].detail
        if enforced.events else None,
    }
    write_json("elision_evidence.json", elision_evidence)
    check("delta_within_800_under_load",
          enforced.estimated_tokens() <= DELTA_TOKEN_BUDGET,
          f"raw {raw_delta.estimated_tokens()} -> enforced "
          f"{enforced.estimated_tokens()} <= {DELTA_TOKEN_BUDGET}")

    # --- memory ≤ 1,500 under a heavy memory load ---------------------------
    mem = StrategyMemory()
    update = MemoryUpdate(
        commitments=[Commitment(domain="expansion",
                                target=f"city site {i} near the northern river",
                                deadline_turn=31 + i)
                     for i in range(100)],
        threats=[f"unit:{300 + i}" for i in range(40)],
        notes=("hold the line and expand north, building roads toward the "
               "front. " * 258),
    )
    mem.begin_turn(30)
    mem.apply(update, turn=30)
    mem.advance(30)
    mem.begin_turn(31)
    mem_text = mem.render_for_prompt()
    mem_tokens = estimate_tokens(mem_text)
    elision_evidence["memory"] = {
        "rendered_tokens": mem_tokens,
        "truncation_disclosed": "memory truncated" in mem_text,
        "threat_hypotheses_elided_first": (
            not [l for l in mem_text.splitlines()
                 if l.startswith("- unit:")]),
        "commitments_kept": sum(1 for l in mem_text.splitlines()
                                if "(by turn" in l),
        "render": mem_text,
    }
    write_json("elision_evidence.json", elision_evidence)
    check("memory_within_1500_under_load", mem_tokens <= MEMORY_TOKEN_BUDGET,
          f"{mem_tokens} <= {MEMORY_TOKEN_BUDGET}, truncation disclosed")

    verdict = "PASS" if checks and all(c["pass"] for c in checks.values()) \
        else "FAIL"
    write_json("run_meta.json", {
        "run": "wp2-large-empire",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "WP2 acceptance: §3 truncation policy on a large-empire "
                 "fixture (plan §3 line 32); offline — no server, no API, "
                 "no civrealm import",
        "fixture": "tests/large_empire_fixture.py (deterministic synthetic "
                   "large-empire observation)",
        "extractor": "lechonk.adapter.obs_extract.extract_snapshot "
                     "(default 8,000-token budget)",
        "companion_test": "tests/test_large_empire.py (8 tests)",
        "wall_clock_s": round(time.time() - t0, 2),
        "checks": checks,
        "verdict": verdict,
    })
    log(f"checks: {sum(1 for c in checks.values() if c['pass'])}/"
        f"{len(checks)} passed")
    log(f"VERDICT: {verdict}")
    _log_file.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
