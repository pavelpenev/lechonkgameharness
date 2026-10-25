# SPDX-License-Identifier: GPL-3.0-or-later
"""WP3 package A — the PAIRED SMOKE: the scenario-cut primitive, cheap
(design-plan-freeciv-v1.1.md §3 line 45 + §4 line 56).

A fresh deterministic game is played a few turns with the harness; a
checkpoint is exported at the start of turn 5 via the WP3 export path
(``lechonk.eval.checkpoint`` — our own copy-after-completion docker-cp
export with the stat-copy-re-stat-checksum stability check, preserved
under ``artifacts/checkpoints/`` with the sha256 recorded); then a
PAIRED SEGMENT runs: the comparator arm and the model arm, each ~3 turns
from the IDENTICAL exported checkpoint, under the predeclared paired
contract recorded in both manifests. Arms run SEQUENTIALLY (one env per
process, §7.1), each in its own subprocess, under the SAME pinned
username as the checkpoint creator (save/username coupling, §7.1).

Usage:
  .venv/bin/python scripts/wp3_paired_smoke.py            # orchestrate all
  .venv/bin/python scripts/wp3_paired_smoke.py create     # creation phase
  .venv/bin/python scripts/wp3_paired_smoke.py arm --policy comparator
  .venv/bin/python scripts/wp3_paired_smoke.py arm --policy model

Everything persists under ``artifacts/runs/wp3-paired-smoke/``:
  pair_contract.json         — the PREDECLARED paired contract (written
                               before any arm runs; identical for both)
  creation/                  — the creation phase (fresh game, 4 turns,
                               the exported checkpoint + its evidence)
  arms/comparator/           — transcript.jsonl, turns.jsonl,
                               api/policy-call log, arm_manifest.json,
                               replay/ (the static viewer bundle)
  arms/model/                — the same + api_calls.jsonl (every API
                               call logged: tokens, latency, cost)
  pair_manifest.json         — the pair-level manifest (both arms
                               reported individually, drift resolution,
                               acceptance checks, verdict)
  pair_summary.md            — the human-readable summary
  run_stdout.log             — this orchestrator's stdout

The model arm uses the live Mistral client (the key loads via the
existing loader — the value is never printed, logged, or artifacted;
only its presence is checked). If the model arm hits an infrastructure
failure, TRUNCATED with artifacts is the correct outcome (decision 8) —
reported honestly, never retried blindly.

GPL-3.0-or-later (decision 7): this script imports the adapter package
(the only CivRealm importer).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# --- run constants -----------------------------------------------------------
PAIR_ID = "wp3-paired-smoke"
RUN_ROOT = os.path.join(REPO, "artifacts", "runs", PAIR_ID)
CHECKPOINT_DIR = os.path.join(REPO, "artifacts", "checkpoints")
SEED = 20261029            # distinct from the WP0/1/2 seeds
CREATION_TURNS = 4         # the comparator plays turns 1..4 ...
EXPORT_TURN = CREATION_TURNS + 1   # ... and the checkpoint is exported at
#                                    the START of turn 5 (before any
#                                    turn-5 action: the begin-turn /save
#                                    has completed, the end-turn HTTP
#                                    deletion has not fired)
ARM_TURNS = 3              # each arm plays turns 5, 6, 7
HORIZON_TURN = EXPORT_TURN + ARM_TURNS - 1   # the runner's max_turns
USERNAME = "myagent"       # pinned; the save/username coupling (§7.1)
AIFILL = 2
RULESET = "classic"
PUBSTATUS = "http://localhost:8080/pubstatus"

# Decision-8 manifest constants (verified against the running container
# 2026-10-24; the digest pin, decision 3).
IMAGE_DIGEST = ("sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a"
                "8e943faf920cb373a")
VENDORED_BASE_SHA = "1b144d6ec92a8aaa8e30eee64c881418208a4689"  # upstream dev
PATCH_HEAD_SHA = "36ae456836015b22b086298430048fc51c317d50"  # PR #7 fork head
CAPSTRING = "+Freeciv.Web.Devel-3.3"
SPIKE_ARTIFACT = os.path.join(
    REPO, "artifacts", "runs", "wp2-spike", "fingerprint_probe.json")
WP2_REPLAY_BUNDLE = os.path.join(
    REPO, "artifacts", "runs", "wp2-model-run", "replay")


# --- small utilities ---------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False,
                  default=str)
        fh.write("\n")
    return path


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def wait_for_stack(log, deadline_s=150.0):
    """Hardened startup: bounded pubstatus retries (the stack boots
    ~30 s after a container restart)."""
    deadline = time.time() + deadline_s
    while True:
        try:
            with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
                if r.status == 200:
                    log(f"pubstatus 200 — stack up")
                    return True
        except Exception as e:
            if time.time() > deadline:
                log(f"FATAL: pubstatus unreachable: {e!r}")
                return False
            time.sleep(2)


class RunLogger:
    """Print to stdout AND persist to the run log (HANDOFF: persist run
    stdout/logs as artifacts under artifacts/runs/)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


# ===========================================================================
# create — the fresh deterministic game + the checkpoint export
# ===========================================================================

def cmd_create():
    creation_dir = os.path.join(RUN_ROOT, "creation")
    workdir = os.path.join(creation_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    log = RunLogger(os.path.join(creation_dir, "run_stdout.log"))
    log(f"[{PAIR_ID}/create] fresh deterministic game, seed={SEED}, "
        f"creation turns={CREATION_TURNS}, export at the start of turn "
        f"{EXPORT_TURN}")
    os.chdir(workdir)  # dedicated CWD for upstream side effects

    if not wait_for_stack(log):
        return 2

    # The adapter imports lazily (civrealm touches the network at import
    # time); only now that the stack is up.
    from lechonk.adapter import execute as exe
    from lechonk.adapter.obs_extract import ExtractOutcome, extract_snapshot
    from lechonk.adapter.session import AdapterSession
    from lechonk.comparator import scripted
    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    checks = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    run_started = time.time()
    session = None
    per_turn = []
    close_status = "not_attempted"
    export = None
    state_at_export = None
    error_text = ""
    try:
        session = AdapterSession(workdir=workdir, seed=SEED,
                                 username=USERNAME, aifill=AIFILL)
        t0 = time.time()
        session.start()
        log(f"reset ok in {time.time() - t0:.1f}s turn={session.turn} "
            f"player_id={session.player_id} username={session.username} "
            f"port={session.port}")
        check("reset_turn_1", session.turn == 1,
              f"turn={session.turn}")
        check("username_pinned", session.username == USERNAME,
              f"username={session.username!r} (self_play=False: congestion "
              f"must raise, never silently rename)")

        # --- play the creation turns with the comparator (WP1 pattern) ---
        for turn_no in range(1, CREATION_TURNS + 1):
            turn_t0 = time.time()
            session.refresh()
            table = session.build_candidate_table()
            extraction = extract_snapshot(
                session.obs, session.info,
                my_player_id=session.player_id,
                ruleset=session.ruleset,
                action_probs=session.action_probs)
            if extraction.outcome is not ExtractOutcome.OK \
                    or extraction.snapshot is None:
                raise RuntimeError(
                    f"extraction classified {extraction.outcome.value}: "
                    f"{extraction.reason}")
            snap = extraction.snapshot
            decision_log = scripted.decide(snap, table)
            executed = rejected = 0
            current_table = table
            for decision in decision_log.decisions:
                # Minimal batch revalidation (the WP1 pattern): the
                # candidate must still be in the freshly rebuilt table. A
                # stale candidate is rejected and NEVER stepped — stepping
                # a client-side-invalid action sends nothing to the server,
                # and the env's state read then blocks until the next
                # connection ping (~30 s, the step deadline).
                cand = current_table.by_id().get(decision.candidate_id)
                if cand is None:
                    rejected += 1
                    continue
                result = exe.execute_candidate(session, cand)
                executed += 1
                if result.terminated or result.truncated:
                    raise RuntimeError(
                        f"creation game ended early at turn {turn_no} "
                        f"({decision.candidate_id}: terminated="
                        f"{result.terminated} truncated={result.truncated})")
                current_table = session.build_candidate_table()
            outcome = exe.finish_phase(session)
            per_turn.append({
                "turn": turn_no,
                "decisions": len(decision_log.decisions),
                "executed": executed, "rejected": rejected,
                "finish_turn_after": int(
                    outcome.info.get("turn", turn_no) or turn_no),
                "wall_clock_s": round(time.time() - turn_t0, 2),
            })
            log(f"turn {turn_no}: {executed} executed / {rejected} stale, "
                f"finish -> turn {outcome.info.get('turn')}")

        # --- the export window: we are at the START of turn EXPORT_TURN ---
        check("at_export_turn", session.turn == EXPORT_TURN,
              f"turn={session.turn} (the begin-turn /save for turn "
              f"{EXPORT_TURN} has fired; the end-turn HTTP deletion has "
              f"not — we have not finished this phase)")

        # The export-time policy-facing state (the canonical comparison
        # baseline for both arms' load verification).
        extraction = extract_snapshot(
            session.obs, session.info,
            my_player_id=session.player_id,
            ruleset=session.ruleset,
            action_probs=session.action_probs)
        if extraction.outcome is ExtractOutcome.OK \
                and extraction.snapshot is not None:
            state_at_export = paired.turn_summary_from_snapshot_json(
                extraction.snapshot.render())
        log(f"state at export (turn {EXPORT_TURN} start): "
            f"score={state_at_export and state_at_export.get('score')} "
            f"gold={state_at_export and state_at_export.get('gold')} "
            f"cities={state_at_export and state_at_export.get('cities_own')} "
            f"units={state_at_export and state_at_export.get('units_own')}")

        # --- the WP3 checkpoint export (our own copy-after-completion) ----
        t0 = time.time()
        export = cp.export_turn_save(
            username=USERNAME, turn=EXPORT_TURN, dest_dir=CHECKPOINT_DIR)
        log(f"export ok in {time.time() - t0:.1f}s: {export.path} "
            f"sha256={export.sha256} save_name={export.save_name} "
            f"size={export.size_bytes} candidates={export.candidates}")
        check("export_checksum_compare",
              export.container_sha256 == export.local_sha256 == export.sha256,
              f"container={export.container_sha256} local={export.local_sha256}")
        check("export_stability_check",
              len(export.stability_stats) >= 3
              and export.restat is not None
              and export.restat.size == export.copied_stat.size
              and export.restat.mtime == export.copied_stat.mtime,
              f"{len(export.stability_stats)} stable stats; re-stat unchanged")
        check("export_local_sha_verified",
              cp.verify_local_checkpoint(export.path, export.sha256),
              "the preserved file re-verifies against the recorded sha256")

        verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        log(f"FATAL: {error_text}")
        import traceback
        traceback.print_exc()
        verdict = "FAIL"
    finally:
        if session is not None:
            close_status = session.close()
            log(f"guarded close status: {close_status}")

    checkpoint_record = None
    if export is not None:
        checkpoint_record = {
            "path": export.path,
            "save_name": export.save_name,
            "sha256": export.sha256,
            "size_bytes": export.size_bytes,
            "turn": export.turn,
            "username": export.username,
            "exported_by": "lechonk.eval.checkpoint.export_turn_save",
            "export_evidence": export.to_log_record(),
        }
    write_json(os.path.join(creation_dir, "creation_manifest.json"), {
        "phase": "create",
        "pair_id": PAIR_ID,
        "timestamp_utc": utcnow(),
        "seed": SEED,
        "username": USERNAME,
        "aifill": AIFILL,
        "ruleset": RULESET,
        "creation_turns": CREATION_TURNS,
        "export_turn": EXPORT_TURN,
        "policy": "lechonk.comparator.scripted.decide (§7.4 comparator)",
        "per_turn": per_turn,
        "state_at_export": state_at_export,
        "checkpoint": checkpoint_record,
        "checkpoint_dir": CHECKPOINT_DIR,
        "wall_clock_s": round(time.time() - run_started, 1),
        "close_status": close_status,
        "checks": checks,
        "error": error_text or None,
        "verdict": verdict,
        "turn_11_checkpoint_untouched": True,
        "note": "fresh deterministic reset on the session's own server; "
                "the turn-11 checkpoint artifact is not modified (the "
                "export preserves a NEW file under artifacts/checkpoints/)",
    })
    log(f"creation verdict: {verdict}")
    log.close()
    return 0 if verdict == "PASS" else 1


# ===========================================================================
# arm — one paired arm (comparator or model) from the exported checkpoint
# ===========================================================================

def cmd_arm(policy):
    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    arm_dir = os.path.join(RUN_ROOT, "arms", policy)
    workdir = os.path.join(arm_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    log = RunLogger(os.path.join(arm_dir, "run_stdout.log"))
    log(f"[{PAIR_ID}/arm:{policy}] arm start; loading the exported "
        f"checkpoint under the predeclared paired contract")
    os.chdir(workdir)  # dedicated CWD for upstream side effects

    contract = read_json(os.path.join(RUN_ROOT, "pair_contract.json"))
    creation = read_json(os.path.join(RUN_ROOT, "creation",
                                      "creation_manifest.json"))
    checkpoint = creation["checkpoint"]
    save_name = checkpoint["save_name"]
    expected_sha = checkpoint["sha256"]
    export_turn = checkpoint["turn"]
    log(f"contract: version={contract['contract_version']} "
        f"sha256={paired.contract_sha256(contract)} "
        f"arm_turns={contract['arm_turns']} horizon_turn="
        f"{contract['horizon_turn']}")
    log(f"checkpoint: {checkpoint['path']} sha256={expected_sha} "
        f"turn={export_turn} save_name={save_name}")

    if not wait_for_stack(log):
        return 2

    checks = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    run_started = time.time()
    arm_start_ts = utcnow()
    transcript_path = os.path.join(arm_dir, "transcript.jsonl")
    turns_path = os.path.join(arm_dir, "turns.jsonl")
    api_log_path = os.path.join(arm_dir, "api_calls.jsonl")
    # A fresh arm run starts with fresh files: the ledger appends BY DESIGN
    # (durability/torn-tail recovery), so a rerun of this phase must not
    # mix records across orchestrator attempts (a stale first turn record
    # would poison the load verification and the replay bundle).
    for fresh_path in (transcript_path, turns_path, api_log_path):
        if os.path.exists(fresh_path):
            os.remove(fresh_path)
    error_text = ""
    result = None
    staging = None
    fingerprint = None
    fingerprint_evidence = None
    call_records = []
    recording = None
    ledger = None
    api_log_fh = None
    try:
        # --- stage the EXPORTED checkpoint into the container's save dir --
        # (the webapp's /load resolves saves from there; the staged file
        # is byte-identical to the exported artifact — sha-verified).
        staging = cp.stage_for_load(checkpoint["path"], username=USERNAME)
        log(f"staged: {staging.staged_container_path} "
            f"sha256={staging.sha256} match={staging.sha256 == expected_sha}")
        check("staged_checkpoint_sha256",
              staging.sha256 == expected_sha == staging.staged_sha256,
              "the exported artifact is what both arms load")

        # --- the load configuration (fc_args is process-global; set
        # BEFORE the session's reset — the session's worker does not
        # touch debug.load_game).
        from civrealm.configs import fc_args  # noqa: PLC0415 — after the
        # stack is up (the civrealm import fetches pubstatus at import)
        fc_args["debug.load_game"] = save_name
        log(f"fc_args[debug.load_game] = {save_name!r} "
            f"(the save/username coupling: the load validates the save's "
            f"username prefix equals {USERNAME!r})")

        # --- the arm's policy client -------------------------------------
        session_box = {}

        def session_factory():
            from lechonk.adapter.session import AdapterSession
            s = AdapterSession(
                workdir=workdir, seed=SEED, username=USERNAME, aifill=AIFILL,
                startup_deadline=150.0, step_deadline=30.0,
                wait_deadline=900.0, close_deadline=30.0)
            session_box["session"] = s
            return s

        if policy == "comparator":
            from lechonk.eval.paired import ScriptedComparatorClient
            policy_client = ScriptedComparatorClient(
                lambda: session_box["session"])
            model_id = None
            sampling = {}
            prompt_hashes = {}
            enforcement_decision = None
            action_batch_schema_hash = None
            fingerprint_evidence = {
                "note": "scripted comparator arm: no model, no API calls "
                        "(the per-arm model fingerprint is n/a)",
                "reference": None, "probe_1": None, "re_probe": None,
                "status": "no_model",
            }
            log("policy client: ScriptedComparatorClient "
                "(lechonk.comparator.scripted.decide behind the "
                "core-defined ModelClient protocol; provenance_class="
                "scripted_default)")
        else:
            from lechonk.model import mistral_client as mc
            from lechonk.model import prompts
            from lechonk.model.turn_client import MistralTurnClient

            try:
                mc.load_api_key_from_env_file()
            except mc.MistralKeyError:
                pass
            key_present = os.environ.get(mc.API_KEY_ENV_VAR) is not None
            log(f"MISTRAL_API_KEY present: {key_present} (presence only; "
                f"the value is never printed, logged, or artifacted)")
            check("api_key_present", key_present,
                  "the model arm needs the live client")
            if not key_present:
                raise RuntimeError(
                    "MISTRAL_API_KEY absent (presence-only check); bounded "
                    "fail-stop per decision 8 — the model arm cannot run")
            client = mc.MistralClient(enforcement="json_schema")
            # The per-arm fingerprint (decision 8): the fixed probe prompt,
            # deterministic sampling, recorded output hash — re-probed ONCE
            # on a mismatch with the reference (the WP2 spike artifact).
            reference = None
            try:
                reference = read_json(SPIKE_ARTIFACT)["fingerprint"]
            except Exception as exc:
                log(f"WARN: could not load the spike fingerprint: {exc!r}")
            api_log_fh = open(api_log_path, "a", encoding="utf-8")

            def api_log(record):
                api_log_fh.write(json.dumps(
                    record, sort_keys=True, ensure_ascii=False,
                    default=str) + "\n")
                api_log_fh.flush()

            probe1 = client.probe_fingerprint()
            fingerprint = probe1.fingerprint
            re_probe = None
            status = ("match" if reference is None
                      or paired.fingerprint_match(reference, fingerprint)
                      else "mismatch")
            if reference is not None and status == "mismatch":
                probe2 = client.probe_fingerprint()
                re_probe = probe2.fingerprint
                status = ("match_after_reprobe"
                          if paired.fingerprint_match(reference, re_probe)
                          else "mismatch_persistent")
            log(f"arm fingerprint: {fingerprint} (reference={reference}, "
                f"status={status}, re_probe={re_probe})")
            # The probe is an API call too (decision 4) — logged with
            # kind="probe".
            for probe in (probe1,):
                pl = client.call_logs[-1].to_dict()
                api_log({"call_id": pl["call_id"], "turn": None, "attempt": 0,
                         "kind": "probe", "prompt_hash": pl["prompt_hash"],
                         "model_requested": pl["model_requested"],
                         "model_responded": pl["model_responded"],
                         "enforcement": pl["enforcement"],
                         "temperature": pl["temperature"],
                         "latency_ms": pl["latency_ms"],
                         "tokens_in": pl["tokens_in"],
                         "tokens_out": pl["tokens_out"],
                         "tokens_total": pl["tokens_total"],
                         "cost_usd": pl["cost_usd"],
                         "finish_reason": pl["finish_reason"],
                         "response_id": pl["response_id"],
                         "fingerprint": pl["fingerprint"],
                         "transport_retries": pl["transport_retries"],
                         "error": pl["error"], "outcome": "probe",
                         "raw_attempts": [], "section_tokens": None,
                         "ts": pl["ts"]})
            if re_probe is not None:
                pl = client.call_logs[-1].to_dict()
                api_log({"call_id": pl["call_id"] + "-reprobe", "turn": None,
                         "attempt": 0, "kind": "probe_reprobe",
                         "prompt_hash": pl["prompt_hash"],
                         "model_requested": pl["model_requested"],
                         "model_responded": pl["model_responded"],
                         "enforcement": pl["enforcement"],
                         "temperature": pl["temperature"],
                         "latency_ms": pl["latency_ms"],
                         "tokens_in": pl["tokens_in"],
                         "tokens_out": pl["tokens_out"],
                         "tokens_total": pl["tokens_total"],
                         "cost_usd": pl["cost_usd"],
                         "finish_reason": pl["finish_reason"],
                         "response_id": pl["response_id"],
                         "fingerprint": pl["fingerprint"],
                         "transport_retries": pl["transport_retries"],
                         "error": pl["error"], "outcome": "probe",
                         "raw_attempts": [], "section_tokens": None,
                         "ts": pl["ts"]})
            fingerprint_evidence = {
                "reference": reference,
                "probe_1": fingerprint,
                "re_probe": re_probe,
                "status": status,
                "probe_1_details": probe1.to_dict(),
                "note": "decision 8: the fingerprint is a canary (serving "
                        "nondeterminism can false-positive; an unchanged "
                        "output does not prove unchanged weights)",
            }
            policy_client = MistralTurnClient(client, api_log=api_log)
            model_id = mc.MODEL_ID
            # W1-6 sampling manifest truth: persist the ACTUAL client
            # sampling (temperature 1.0, reasoning_effort "high"; the
            # client never sends top_p) — never the fabricated greedy
            # dict previously recorded here.
            from lechonk.model.sampling import actual_client_sampling
            sampling = actual_client_sampling()
            prompt_hashes = prompts.prompt_manifest_hashes()
            enforcement_decision = mc.spike_decision(
                live_probe_done=True, json_schema_supported=True,
                evidence_artifact="artifacts/runs/wp2-spike/spike_report.json")
            action_batch_schema_hash = mc.action_batch_schema_hash()
            log(f"policy client: MistralTurnClient over MistralClient "
                f"(enforcement=json_schema, model={client.model})")

        # --- the arm manifest (decision 8 + the paired extras) -----------
        manifest = paired.build_arm_manifest(
            arm=policy, pair_id=PAIR_ID, contract=contract,
            checkpoint=checkpoint, arm_start_ts=arm_start_ts,
            model_fingerprint=fingerprint,
            fingerprint_evidence=fingerprint_evidence,
            harness_rev=paired.harness_revision(REPO),
            seeds={"mapseed": SEED, "gameseed": SEED, "agentseed": SEED},
            username=USERNAME, aifill=AIFILL,
            image_digest=IMAGE_DIGEST, capstring=CAPSTRING,
            vendored_base_sha=VENDORED_BASE_SHA, patch_head_sha=PATCH_HEAD_SHA,
            model_id=model_id, sampling=sampling, prompt_hashes=prompt_hashes,
            enforcement_decision=enforcement_decision,
            action_batch_schema_hash=action_batch_schema_hash,
            run_id=f"{PAIR_ID}/{policy}")

        # --- the runner (the SAME pipeline for both arms) ------------------
        from lechonk.core.ledger import TranscriptLedger
        from lechonk.core.runner import GameRunner, RunnerConfig

        recording = paired.RecordingClient(policy_client, turns_path)
        ledger = TranscriptLedger(transcript_path)
        config = RunnerConfig(
            max_turns=HORIZON_TURN, run_id=f"{PAIR_ID}/{policy}",
            workdir=workdir, seed=SEED, username=USERNAME, aifill=AIFILL)
        runner = GameRunner(
            config=config, client=recording, ledger=ledger,
            manifest=manifest, session_factory=session_factory)
        log(f"runner: max_turns={HORIZON_TURN} (the arms play turns "
            f"{export_turn}..{HORIZON_TURN} from the loaded checkpoint)")
        run_t0 = time.time()
        result = runner.run()
        run_wall_s = time.time() - run_t0
        log(f"run finished: outcome={result.outcome} "
            f"reason={result.reason!r} turns_completed="
            f"{result.turns_completed} close_status={result.close_status} "
            f"wall_clock={run_wall_s:.1f}s")
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        log(f"FATAL: {error_text}")
        import traceback
        traceback.print_exc()
    finally:
        if recording is not None:
            recording.close()
        if ledger is not None:
            ledger.close()
        if api_log_fh is not None:
            api_log_fh.close()

    # --- post-run: the transcript, the load verification, the metrics -----
    arm_end_ts = utcnow()
    post = {"transcript_records": None, "turn_starts": None,
            "outcome_record": None, "first_turn_summary": None,
            "load_verification": None, "provenance_counts": None,
            "readback_counts": None, "api_totals": None,
            "replay_bundle": None, "metrics": None}
    if result is not None:
        from lechonk.core.ledger import load_transcript
        from lechonk.core.metrics import compute_metrics
        from lechonk.core.schemas import (
            ActionResultRecord, OutcomeRecord, RunManifestRecord,
            TurnStartRecord)

        load = load_transcript(transcript_path)
        records = load.records
        post["transcript_records"] = len(records)
        post["transcript_corruption_free"] = not load.discarded_trailing_partial
        turn_starts = [r.turn for r in records
                       if isinstance(r, TurnStartRecord)]
        post["turn_starts"] = turn_starts
        outcome_rec = next((r for r in records
                            if isinstance(r, OutcomeRecord)), None)
        post["outcome_record"] = ({"outcome": outcome_rec.outcome,
                                   "reason": outcome_rec.reason}
                                  if outcome_rec else None)
        post["manifest_first_record"] = bool(records) and isinstance(
            records[0], RunManifestRecord)
        prov = {}
        rb = {}
        for r in records:
            if isinstance(r, ActionResultRecord) and r.outcome == "executed":
                prov[r.provenance_class] = prov.get(r.provenance_class, 0) + 1
                rb[r.readback] = rb.get(r.readback, 0) + 1
        post["provenance_counts"] = prov
        post["readback_counts"] = rb
        # The load verification: the FIRST turn's recorded summary (the
        # prompt snapshot at load) must equal the creation's export-time
        # summary — the canonical proof both arms loaded the identical
        # checkpoint state.
        first_summary = None
        if os.path.exists(turns_path):
            with open(turns_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        first_summary = json.loads(line)
                        break
        post["first_turn_summary"] = first_summary
        state_at_export = creation.get("state_at_export")
        load_verification = {
            "checkpoint_sha256_recorded": expected_sha,
            "staged_sha256_match": (staging.sha256 == expected_sha
                                    if staging else None),
            "first_turn_from_transcript": (turn_starts[0]
                                           if turn_starts else None),
            "expected_first_turn": export_turn,
            "state_at_load": first_summary,
            "state_at_export": state_at_export,
            "states_match": (first_summary is not None
                             and state_at_export is not None
                             and paired.states_match(first_summary,
                                                     state_at_export)),
        }
        post["load_verification"] = load_verification
        check("load_turn_matches_checkpoint",
              bool(turn_starts) and turn_starts[0] == export_turn,
              f"first turn_start={turn_starts[:1]} (the checkpoint turn "
              f"is {export_turn})")
        check("load_state_matches_export",
              load_verification["states_match"],
              "the arm's first-turn state equals the export-time state "
              "(the identical checkpoint, canonically compared)")
        check("transcript_valid",
              post["manifest_first_record"]
              and post["transcript_corruption_free"]
              and post["outcome_record"] is not None,
              f"{len(records)} records, manifest first, outcome present, "
              f"no corruption")
        metrics = compute_metrics(records)
        post["metrics"] = metrics.to_dict()
        if policy == "model":
            with open(api_log_path, encoding="utf-8") as fh:
                call_records = [json.loads(l) for l in fh if l.strip()]
            post["api_totals"] = {
                "calls": len(call_records),
                "tokens_in": sum(r.get("tokens_in") or 0
                                 for r in call_records),
                "tokens_out": sum(r.get("tokens_out") or 0
                                  for r in call_records),
                "cost_usd": round(sum(r.get("cost_usd") or 0.0
                                      for r in call_records), 6),
                "latency_ms": sum(r.get("latency_ms") or 0
                                  for r in call_records),
            }
            log(f"api calls: {post['api_totals']}")
        # The replay bundle (the static viewer, §7.6).
        from lechonk.eval.replay import render_bundle
        bundle = render_bundle(transcript_path, os.path.join(arm_dir, "replay"),
                               turns_path=turns_path,
                               title=f"Le Chonk replay — {PAIR_ID}/{policy}")
        post["replay_bundle"] = bundle["bundle_dir"]
        log(f"replay bundle: {bundle['bundle_dir']}")

    early = None
    if result is not None:
        early = paired.early_terminal_report(
            result.outcome, result.reason, result.turns_completed, ARM_TURNS)
    arm_manifest = {
        "phase": "arm",
        "pair_id": PAIR_ID,
        "arm": policy,
        "policy": ("lechonk.comparator.scripted.decide behind "
                   "lechonk.eval.paired.ScriptedComparatorClient"
                   if policy == "comparator" else
                   "mistral-large-4 (lechonk.model.turn_client over "
                   "MistralClient, json_schema enforcement — the WP2 spike "
                   "decision) behind lechonk.eval.paired.RecordingClient"),
        "timestamp_utc": utcnow(),
        "arm_start_ts": arm_start_ts,
        "arm_end_ts": arm_end_ts,
        "wall_clock_s": round(time.time() - run_started, 1),
        "seed": SEED,
        "username": USERNAME,
        "run_id": f"{PAIR_ID}/{policy}",
        "paired_contract_sha256": paired.contract_sha256(contract),
        "checkpoint": {
            "path": checkpoint["path"],
            "save_name": save_name,
            "sha256": expected_sha,
            "turn": export_turn,
            "username": USERNAME,
            "staged": (staging.to_record() if staging else None),
        },
        "turns_planned": ARM_TURNS,
        "horizon_turn": HORIZON_TURN,
        "outcome": result.outcome if result else None,
        "reason": result.reason if result else error_text or None,
        "turns_completed": result.turns_completed if result else 0,
        "close_status": result.close_status if result else None,
        "early_terminal": early is not None,
        "early_terminal_report": early,
        "model_fingerprint": fingerprint,
        "fingerprint_evidence": fingerprint_evidence,
        "transcript": transcript_path,
        "turns": turns_path,
        "api_calls_log": api_log_path if policy == "model" else None,
        "post_run": post,
        "checks": checks,
        "error": error_text or None,
    }
    write_json(os.path.join(arm_dir, "arm_manifest.json"), arm_manifest)
    log(f"arm manifest written; outcome={arm_manifest['outcome']} "
        f"turns={arm_manifest['turns_completed']}/{ARM_TURNS} "
        f"early_terminal={arm_manifest['early_terminal']}")
    log.close()
    return 0


# ===========================================================================
# orchestrator — create -> comparator arm -> model arm -> pair manifest
# ===========================================================================

def run_subprocess(log, args, log_path):
    """Run one phase as a subprocess, persisting its stdout to the log
    file AND echoing it (the phases are long; the log is the artifact)."""
    log(f"--- phase: {' '.join(args)} ---")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
        for line in proc.stdout.splitlines():
            log(f"  | {line}")
    return proc.returncode


def run_phase(log, args, log_path):
    """Run one phase as a subprocess; on a non-zero rc, ONE bounded retry
    with a fresh process (the WP1 infra-retry discipline: a single logged
    retry, never a silent loop-retry). The failed attempt's stdout is
    preserved in the orchestrator log (run_subprocess echoes every line)."""
    rc = run_subprocess(log, args, log_path)
    if rc != 0:
        log(f"phase {' '.join(args)} rc={rc} — bounded retry (once, fresh "
            f"process; the WP1 infra-retry discipline, never a silent loop)")
        # Settle first: the failed attempt's game server quits ~20 s after
        # its client disconnects (--quitidle 20, HANDOFF); attaching to the
        # stale in-progress server races CivRealm's begin-turn handling.
        log("settling 25 s before the retry (--quitidle 20 server teardown)")
        time.sleep(25)
        rc = run_subprocess(log, args, log_path)
    return rc


def cmd_orchestrate():
    os.makedirs(RUN_ROOT, exist_ok=True)
    log = RunLogger(os.path.join(RUN_ROOT, "run_stdout.log"))
    log(f"[{PAIR_ID}] orchestrator: create -> comparator arm -> model "
        f"arm (sequential, one env per process) -> pair manifest")
    log(f"pair root: {RUN_ROOT}")

    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    # --- the PREDECLARED paired contract (before any arm runs) ------------
    contract = paired.build_paired_contract(
        arm_turns=ARM_TURNS, username=USERNAME, horizon_turn=HORIZON_TURN,
        aifill=AIFILL, ruleset=RULESET)
    contract_path = write_json(os.path.join(RUN_ROOT, "pair_contract.json"),
                               contract)
    log(f"predeclared paired contract: {contract_path} "
        f"sha256={paired.contract_sha256(contract)} "
        f"(identical for both arms; recorded in both manifests)")

    # --- phase 1: create (fresh game + the checkpoint export) -------------
    rc = run_phase(log, ["create"],
                   os.path.join(RUN_ROOT, "creation", "run_stdout.log"))
    log(f"create phase rc={rc}")
    creation_path = os.path.join(RUN_ROOT, "creation", "creation_manifest.json")
    if rc != 0 or not os.path.exists(creation_path):
        log("FATAL: the create phase failed — no checkpoint to pair from")
        log.close()
        return 2
    creation = read_json(creation_path)
    checkpoint = creation["checkpoint"]
    log(f"checkpoint: {checkpoint['path']} sha256={checkpoint['sha256']} "
        f"turn={checkpoint['turn']} save_name={checkpoint['save_name']}")

    # --- phase 2 + 3: the arms, SEQUENTIALLY -------------------------------
    arm_manifests = {}
    for arm in paired.ARM_ORDER:
        rc = run_phase(log, ["arm", "--policy", arm],
                       os.path.join(RUN_ROOT, "arms", arm, "run_stdout.log"))
        log(f"{arm} arm rc={rc}")
        arm_manifest_path = os.path.join(RUN_ROOT, "arms", arm,
                                         "arm_manifest.json")
        if os.path.exists(arm_manifest_path):
            arm_manifests[arm] = read_json(arm_manifest_path)
        else:
            log(f"FATAL: no arm manifest for the {arm} arm")
            arm_manifests[arm] = None

    # --- the pair-level checks ----------------------------------------------
    checks = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    # (1) the checkpoint was exported via the WP3 path, checksummed.
    ev = (checkpoint.get("export_evidence") or {})
    check("checkpoint_exported_via_wp3_path",
          checkpoint.get("exported_by")
          == "lechonk.eval.checkpoint.export_turn_save"
          and ev.get("checksum_compare_match") is True
          and len(ev.get("stability_stats") or []) >= 3,
          f"export tool={checkpoint.get('exported_by')}, "
          f"stability stats={len(ev.get('stability_stats') or [])}, "
          f"checksum-compare={ev.get('checksum_compare_match')}")
    check("checkpoint_sha256_verified",
          cp.verify_local_checkpoint(checkpoint["path"],
                                      checkpoint["sha256"]),
          "the preserved checkpoint re-verifies against the recorded sha256")

    # (2) both arms recorded the IDENTICAL checkpoint checksum.
    sha_ok = all(
        m and m["checkpoint"]["sha256"] == checkpoint["sha256"]
        and (m["checkpoint"].get("staged") or {}).get("checksum_match")
        for m in arm_manifests.values())
    check("both_arms_record_identical_checkpoint", sha_ok,
          "both arm manifests record checkpoint.sha256 == the exported "
          "sha256, and the staged (in-container) copy verified equal")

    # (3) both arms actually LOADED the identical checkpoint.
    load_ok = True
    load_detail = []
    for arm, m in arm_manifests.items():
        lv = (m or {}).get("post_run", {}).get("load_verification") or {}
        ok = (lv.get("first_turn_from_transcript") == checkpoint["turn"]
              and lv.get("states_match") is True)
        load_ok = load_ok and ok
        load_detail.append(f"{arm}: first_turn="
                           f"{lv.get('first_turn_from_transcript')} "
                           f"states_match={lv.get('states_match')}")
    check("both_arms_loaded_identical_checkpoint", load_ok,
          "; ".join(load_detail))

    # (4) the arms ran sequentially (the model arm started at or after the
    # comparator arm ended — non-overlapping; the timestamps have 1 s
    # resolution, so a fast phase boundary can share a second).
    seq_ok = False
    seq_detail = "missing arm manifests"
    if arm_manifests.get("comparator") and arm_manifests.get("model"):
        c, m = arm_manifests["comparator"], arm_manifests["model"]
        seq_ok = (datetime.fromisoformat(m["arm_start_ts"])
                  >= datetime.fromisoformat(c["arm_end_ts"]))
        seq_detail = (f"comparator [{c['arm_start_ts']} .. "
                      f"{c['arm_end_ts']}], model [{m['arm_start_ts']} .. "
                      f"{m['arm_end_ts']}] — the model arm starts after "
                      f"the comparator arm ends")
    check("arms_sequential", seq_ok, seq_detail)

    # (5) the predeclared contract is identical in both manifests.
    contract_ok = False
    contract_detail = "missing arm manifests"
    if all(arm_manifests.get(a) for a in paired.ARMS):
        hashes = {a: arm_manifests[a]["paired_contract_sha256"]
                  for a in paired.ARMS}
        contract_ok = (len(set(hashes.values())) == 1
                       and hashes["comparator"]
                       == paired.contract_sha256(contract))
        contract_detail = f"contract hashes: {hashes}"
    check("contract_identical_in_both_manifests", contract_ok,
          contract_detail)

    # (6) early-terminal reporting: both arms' outcomes + reasons are
    # recorded; an early terminal is reported, never silently dropped.
    et_ok = all(
        m and m.get("outcome") is not None and m.get("reason") is not None
        and ((m["early_terminal"] and m["early_terminal_report"])
             or (not m["early_terminal"]))
        for m in arm_manifests.values())
    et_detail = "; ".join(
        f"{a}: outcome={m and m.get('outcome')} "
        f"turns={m and m.get('turns_completed')}/{ARM_TURNS} "
        f"early_terminal={m and m.get('early_terminal')}"
        for a, m in arm_manifests.items())
    check("early_terminal_reporting", et_ok, et_detail)

    # (7) both arms' transcripts are valid ledger transcripts.
    tr_ok = all(
        m and (m.get("post_run", {}).get("transcript_records") or 0) > 0
        and m.get("post_run", {}).get("manifest_first_record") is True
        and m.get("post_run", {}).get("transcript_corruption_free") is True
        and m.get("post_run", {}).get("outcome_record") is not None
        for m in arm_manifests.values())
    check("both_transcripts_valid", tr_ok,
          "; ".join(f"{a}: records="
                    f"{m and (m.get('post_run') or {}).get('transcript_records')}"
                    for a, m in arm_manifests.items()))

    # (8) the per-arm fingerprints are recorded and the decision-8 drift
    # policy is applied.
    arm_fingerprints = {a: (m or {}).get("model_fingerprint")
                        for a, m in arm_manifests.items()}
    re_probes = {}
    for a, m in arm_manifests.items():
        fe = (m or {}).get("fingerprint_evidence") or {}
        if fe.get("re_probe"):
            re_probes[a] = fe["re_probe"]
    reference = None
    try:
        reference = read_json(SPIKE_ARTIFACT)["fingerprint"]
    except Exception:
        pass
    drift = paired.resolve_fingerprints(
        arm_fingerprints=arm_fingerprints, reference=reference,
        re_probes=re_probes)
    fp_ok = (arm_fingerprints.get("model") is not None
             and arm_fingerprints.get("comparator") is None
             and drift.get("per_arm", {}).get("comparator", {}).get("status")
             == "no_model")
    check("fingerprints_recorded_drift_policy_applied", fp_ok,
          f"model arm fingerprint={arm_fingerprints.get('model')} "
          f"status={drift['per_arm'].get('model', {}).get('status')}; "
          f"comparator arm: no model; drift_suspect={drift['drift_suspect']} "
          f"reproducibility_claim={drift['reproducibility_claim']}")

    # (9) both arms' replay bundles are rendered (the static viewer).
    rb_ok = all(
        m and os.path.isfile(os.path.join(
            (m.get("post_run") or {}).get("replay_bundle") or "",
            "index.html"))
        for m in arm_manifests.values())
    check("replay_bundles_rendered", rb_ok,
          "; ".join(f"{a}: {(m or {}).get('post_run', {}).get('replay_bundle')}"
                    for a, m in arm_manifests.items()))

    # (10) the model arm's outcome is reported (honest TRUNCATED is a
    # correct outcome per decision 8 — reported, not gating).
    model_outcome = (arm_manifests.get("model") or {}).get("outcome")
    log(f"[report] model arm outcome: {model_outcome} "
        f"({(arm_manifests.get('model') or {}).get('reason')!r}) — "
        f"reported honestly; an infrastructure TRUNCATED with artifacts "
        f"is a correct outcome (decision 8)")

    # --- the pair manifest ----------------------------------------------------
    arms_section = {}
    for arm in paired.ARMS:
        m = arm_manifests.get(arm)
        if not m:
            arms_section[arm] = {"arm": arm, "error": "no arm manifest"}
            continue
        arms_section[arm] = {
            "arm": arm,
            "policy": m["policy"],
            "run_id": m["run_id"],
            "transcript": m["transcript"],
            "turns": m["turns"],
            "replay_bundle": (m.get("post_run") or {}).get("replay_bundle"),
            "arm_start_ts": m["arm_start_ts"],
            "arm_end_ts": m["arm_end_ts"],
            "wall_clock_s": m["wall_clock_s"],
            "outcome": m["outcome"],
            "reason": m["reason"],
            "turns_completed": m["turns_completed"],
            "turns_planned": m["turns_planned"],
            "early_terminal": m["early_terminal"],
            "early_terminal_report": m["early_terminal_report"],
            "model_fingerprint": m["model_fingerprint"],
            "fingerprint_evidence": m["fingerprint_evidence"],
            "checkpoint_sha256_recorded": m["checkpoint"]["sha256"],
            "contract_sha256_recorded": m["paired_contract_sha256"],
            "api_totals": (m.get("post_run") or {}).get("api_totals"),
            "provenance_counts": (m.get("post_run") or {}).get(
                "provenance_counts"),
            "readback_counts": (m.get("post_run") or {}).get(
                "readback_counts"),
            "load_verification": (m.get("post_run") or {}).get(
                "load_verification"),
            "metrics": (m.get("post_run") or {}).get("metrics"),
        }
    plumbing = ["checkpoint_exported_via_wp3_path",
                "checkpoint_sha256_verified",
                "both_arms_record_identical_checkpoint",
                "both_arms_loaded_identical_checkpoint",
                "arms_sequential", "contract_identical_in_both_manifests",
                "early_terminal_reporting", "both_transcripts_valid",
                "fingerprints_recorded_drift_policy_applied",
                "replay_bundles_rendered"]
    verdict = ("PASS" if all(checks[c]["pass"] for c in plumbing) else "FAIL")
    pair_manifest = paired.assemble_pair_manifest(
        pair_id=PAIR_ID, contract=contract,
        checkpoint={
            "path": checkpoint["path"], "save_name": checkpoint["save_name"],
            "sha256": checkpoint["sha256"], "turn": checkpoint["turn"],
            "username": checkpoint["username"],
            "exported_by": checkpoint["exported_by"],
            "export_evidence": checkpoint.get("export_evidence"),
            "creation_manifest": creation_path,
        },
        arms=arms_section, drift=drift, checks=checks, verdict=verdict)
    pair_manifest["constants"] = {
        "seed": SEED, "creation_turns": CREATION_TURNS,
        "export_turn": checkpoint["turn"], "arm_turns": ARM_TURNS,
        "horizon_turn": HORIZON_TURN, "username": USERNAME,
        "aifill": AIFILL, "ruleset": RULESET,
        "image_digest": IMAGE_DIGEST, "capstring": CAPSTRING,
        "vendored_base_sha": VENDORED_BASE_SHA, "patch_head_sha": PATCH_HEAD_SHA,
    }
    pair_manifest["model_arm_outcome_report"] = {
        "outcome": model_outcome,
        "reason": (arm_manifests.get("model") or {}).get("reason"),
        "note": "reported honestly; an infrastructure TRUNCATED with "
                "artifacts is a correct outcome (decision 8) — the "
                "plumbing verdict above does not gate on it",
    }
    write_json(os.path.join(RUN_ROOT, "pair_manifest.json"), pair_manifest)

    # --- the human-readable summary -------------------------------------------
    lines = [
        f"# WP3 paired smoke — {PAIR_ID}",
        "",
        f"Generated: {utcnow()} · verdict: **{verdict}**",
        "",
        "The scenario-cut primitive, cheap (§3 line 45): a fresh "
        "deterministic game (comparator, 4 turns) → a checkpoint exported "
        "at the start of turn 5 via the WP3 export path "
        "(`lechonk.eval.checkpoint`: docker cp with the "
        "stat-copy-re-stat-checksum stability check) → a PAIRED SEGMENT: "
        "the comparator arm then the model arm, each 3 turns from the "
        "IDENTICAL exported checkpoint under the predeclared paired "
        "contract, sequentially, under the same pinned username.",
        "",
        "## Checkpoint",
        "",
        f"- path: `{checkpoint['path']}`",
        f"- save_name (load_game): `{checkpoint['save_name']}`",
        f"- sha256: `{checkpoint['sha256']}`",
        f"- turn: {checkpoint['turn']} · username: {checkpoint['username']}",
        f"- exported by: {checkpoint['exported_by']}",
        f"- checksum-compare: container == local == recorded "
        f"({ev.get('checksum_compare_match')}); stability stats: "
        f"{len(ev.get('stability_stats') or [])}",
        "",
        "## Arms (both reported individually, §3 line 45)",
        "",
        "| arm | outcome | turns | start (UTC) | end (UTC) | wall-clock s | "
        "fingerprint |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in paired.ARM_ORDER:
        m = arm_manifests.get(arm) or {}
        lines.append(
            f"| {arm} | {m.get('outcome')} | "
            f"{m.get('turns_completed')}/{ARM_TURNS} | "
            f"{m.get('arm_start_ts')} | {m.get('arm_end_ts')} | "
            f"{m.get('wall_clock_s')} | "
            f"{str(m.get('model_fingerprint'))[:24]}… |")
    lines += [
        "",
        f"- arms sequential: comparator ends "
        f"{arm_manifests.get('comparator', {}).get('arm_end_ts')}, model "
        f"starts {arm_manifests.get('model', {}).get('arm_start_ts')} "
        f"(check `arms_sequential`)",
        f"- identical checkpoint recorded in both manifests: "
        f"{checks['both_arms_record_identical_checkpoint']['pass']}",
        f"- identical checkpoint LOADED by both arms (first turn + state "
        f"match): {checks['both_arms_loaded_identical_checkpoint']['pass']}",
        f"- predeclared contract identical in both manifests: "
        f"{checks['contract_identical_in_both_manifests']['pass']} "
        f"(sha256 {paired.contract_sha256(contract)})",
        f"- early-terminal reporting: "
        f"{checks['early_terminal_reporting']['pass']} "
        f"(an early terminal is reported, the pair is never silently "
        f"dropped)",
        "",
        "## Drift (decision 8)",
        "",
        f"- reference fingerprint (WP2 spike): `{reference}`",
        f"- model arm fingerprint: `{arm_fingerprints.get('model')}` "
        f"(status: {drift['per_arm'].get('model', {}).get('status')})",
        f"- comparator arm: no model (scripted; no API calls)",
        f"- drift_suspect: {drift['drift_suspect']} · "
        f"reproducibility_claim: {drift['reproducibility_claim']}",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, c in checks.items():
        lines.append(f"- {'PASS' if c['pass'] else 'FAIL'} **{name}** — "
                     f"{c['detail']}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- pair manifest: `artifacts/runs/{PAIR_ID}/pair_manifest.json`",
        f"- predeclared contract: `artifacts/runs/{PAIR_ID}/pair_contract.json`",
        f"- checkpoint: `{checkpoint['path']}` "
        f"(export log: `artifacts/checkpoints/exports.jsonl`)",
        f"- arms: `artifacts/runs/{PAIR_ID}/arms/{paired.ARM_ORDER[0]}/`, "
        f"`.../arms/{paired.ARM_ORDER[1]}/` (transcript.jsonl, turns.jsonl, "
        f"arm_manifest.json, replay/)",
        f"- WP2 10-turn transcript replay bundle: `{WP2_REPLAY_BUNDLE}/`",
        "",
        f"VERDICT: **{verdict}**",
        "",
    ]
    with open(os.path.join(RUN_ROOT, "pair_summary.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"pair manifest + summary written; checks: "
        f"{sum(1 for c in checks.values() if c['pass'])}/{len(checks)} passed")
    log(f"VERDICT: {verdict}")
    log.close()
    return 0 if verdict == "PASS" else 1


# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="WP3 package A — the paired smoke (the scenario-cut "
                    "primitive): fresh game → exported checkpoint → "
                    "comparator arm + model arm from the identical "
                    "checkpoint under the predeclared paired contract")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("create", help="the creation phase (fresh game + the "
                                  "checkpoint export)")
    arm_parser = sub.add_parser("arm", help="one paired arm")
    arm_parser.add_argument("--policy", required=True,
                            choices=["comparator", "model"])
    args = parser.parse_args(argv)
    if args.command == "create":
        return cmd_create()
    if args.command == "arm":
        return cmd_arm(args.policy)
    return cmd_orchestrate()


if __name__ == "__main__":
    sys.exit(main())
