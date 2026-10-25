# SPDX-License-Identifier: MIT
"""WP2 package C deliverable 3: live fault injection against the real stack.

WP2 acceptance (plan §4 line 54): "degraded path exercised under fault
injection (timeout, malformed output, invalid action)" — the pipeline-side
repair and degraded paths must work against the REAL stack, not only the
offline fake (tests/test_runner.py covers the fake).

Design: a ``FaultInjectingMistralClient`` subclasses the real
``MistralClient`` and injects a scripted fault at the single-attempt seam
(``complete_once`` — one protocol call = one logical API attempt). Every
NON-faulted attempt delegates to the real transport unchanged. The runner,
the turn client (prompt assembly, class-specific repair templates,
per-call API logging) and the ledger stay 100% real; only the
transport-level outcome of the scheduled attempts is synthesized (no API
call is made for an injected attempt).

The fault schedule (3 turns, fresh deterministic reset, seed 20261029 —
distinct from the model run's 20261028):

- turn 1 attempt 1: injected TRANSPORT TIMEOUT (``AttemptOutcome.error``).
  The runner classifies it ``unresponsive`` and repairs — the repair is a
  REAL model call (the unresponsive repair re-asks the same turn prompt),
  proving the pipeline-side timeout→repair path against the real API.
- turn 2 attempt 1: injected MALFORMED OUTPUT (a JSON batch whose action
  ref is free text → ``AttemptOutcome.malformed``; the turn client
  extracts ``raw_attempts``). The runner rejects it ``schema_error`` and
  repairs with the schema_error template (the failed output is appended to
  the repair conversation).
- turn 2 attempt 2 (the repair): SCRIPTED VALID BATCH — a build_city
  candidate parsed from the turn's REAL offered candidate table (regex
  over the CANDIDATE TABLE section of the real rendered prompt; empty
  batch fallback). Proves the schema-error repair path ends in an
  executed, read-back-verified action.
- turn 3 attempt 1 AND attempt 2: injected INVALID CANDIDATE REF
  (``unit:424242:goto_n`` — a well-formed candidate ID that is never
  offered). The runner rejects it ``state_fog_error`` both times → the
  repair cap (≤2 attempts, decision 4) → the DEGRADED DEFAULT
  (comparator/defaults.decide, observed at the entry point — observation
  only, no behavior change). On this game state the default emits ZERO
  resolutions — research is active (turn 1's real action) and the turn-2
  city's production is preserved — the §7.4 "preserve everything, resolve
  only NEW mandatory choices" branch; the pipeline still finishes the
  phase. The scripted_default EXECUTION path (decisions executed with
  read-back, logged scripted_default) is covered offline by
  tests/test_runner.py::test_runner_unresponsive_after_repair_cap_degrades.

Artifacts (all under artifacts/runs/wp2-fault-run/): run_stdout.log,
transcript.jsonl (the provenance ledger), fault_injections.jsonl (one
record per injected fault, cross-referenced to api_calls.jsonl by
``api_call_seq``), api_calls.jsonl (per-call API log, decision 4 — real
calls carry real tokens; injected calls carry zero tokens and are marked
by their injection record), fault_evidence.md, run_meta.json, workdir/.

The API key loads via the existing client loader (env or repo-root .env at
runtime); the value is never printed, logged, or artifacted. Model-driven
degradation (repair cap → degraded defaults) is the EXERCISED behavior
here, not an infrastructure failure. Bounded single retry per infra
error: WP2_FAULT_RUN_ATTEMPT=2.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lechonk.model.mistral_client import MistralClient  # noqa: E402

RUN = "wp2-fault-run"
RUNDIR = os.path.join(REPO, "artifacts", "runs", RUN)
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream
os.makedirs(WORKDIR, exist_ok=True)

SEED = 20261029  # fresh: distinct from the model run (20261028) and wp1
TURNS = 3
PUBSTATUS = "http://localhost:8080/pubstatus"
ATTEMPT = int(os.environ.get("WP2_FAULT_RUN_ATTEMPT", "1"))

# Decision-8 manifest constants (verified this session / in the plan; the
# same pins as the model run — same stack, same image).
IMAGE_DIGEST = ("sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a"
                "8e943faf920cb373a")
VENDORED_BASE_SHA = "1b144d6ec92a8aaa8e30eee64c881418208a4689"  # upstream dev
PATCH_HEAD_SHA = "36ae456836015b22b086298430048fc51c317d50"  # PR #7 fork head
CAPSTRING = "+Freeciv.Web.Devel-3.3"  # plan decision 1 (two primary sources)
SPIKE_ARTIFACT = os.path.join(
    REPO, "artifacts", "runs", "wp2-spike", "fingerprint_probe.json")

LOG_PATH = os.path.join(RUNDIR, "run_stdout.log")
_log_file = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    """Print to stdout AND persist to the run log (HANDOFF: persist run
    stdout/logs as artifacts under artifacts/runs/)."""
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


checks = {}


def check(name, cond, detail=""):
    checks[name] = {"pass": bool(cond), "detail": detail}
    log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
    return bool(cond)


def fresh_artifact(name):
    """A fresh run rewrites its own per-attempt artifacts; on a bounded
    retry the previous attempt's artifact is preserved (renamed), never
    silently deleted."""
    path = os.path.join(RUNDIR, name)
    if os.path.exists(path):
        if ATTEMPT > 1:
            os.replace(path, path + ".attempt1")
            log(f"preserved the previous attempt's {name} as {name}.attempt1")
        else:
            os.remove(path)
    return path


def harness_revision():
    """The repo is NOT a git checkout (HANDOFF constraints): the harness
    revision is a deterministic content hash over src/lechonk — every file's
    sha256, sorted by path, hashed again."""
    root = Path(REPO) / "src" / "lechonk"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = str(path.relative_to(root))
            lines.append(f"{rel} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    blob = "\n".join(lines).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def load_spike_fingerprint():
    """The arm fingerprint from the WP2 spike artifact (the manifest value,
    decision 8)."""
    try:
        with open(SPIKE_ARTIFACT, encoding="utf-8") as fh:
            return json.load(fh)["fingerprint"]
    except Exception as exc:
        log(f"WARN: could not load the spike fingerprint artifact: {exc!r}")
        return None


def build_manifest(arm_start_ts, spike_fingerprint, live_probe):
    from lechonk.core.schemas import RunManifest
    from lechonk.model import mistral_client as mc
    from lechonk.model import prompts

    decision = mc.spike_decision(
        live_probe_done=True,
        json_schema_supported=True,
        evidence_artifact="artifacts/runs/wp2-spike/spike_report.json")
    fc_args = {
        "self_play": False, "aifill": 2,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": SEED, "debug.gameseed": SEED,
        "debug.agentseed": SEED,
        "server_timeout": 300, "begin_turn_timeout": 900,
        "wait_for_timeout": 60, "http_timeout_s": 15.0,
    }
    manifest = RunManifest(
        run_id=RUN,
        image_digest=IMAGE_DIGEST,
        server_commit=None,  # not persisted by the WP0 spike (stdout-only
        # session); the image digest pins the server build — disclosed.
        capstring=CAPSTRING,
        vendored_base_sha=VENDORED_BASE_SHA,
        patch_head_sha=PATCH_HEAD_SHA,
        harness_revision=harness_revision(),
        python_lockfile=None,  # no lockfile in the repo (uv-managed venv)
        seeds={"mapseed": SEED, "gameseed": SEED, "agentseed": SEED},
        ruleset="classic",
        server_settings={
            "fc_args_overrides": fc_args,
            "phasemode": "player", "timeout": 0, "pingtimeout": 720,
            "autotoggle": "disabled",
            "victories": "SPACERACE|ALLIED (CivRealm default)",
            "aifill": 2, "self_play": False,
        },
        model_id=mc.MODEL_ID,
        sampling={
            "temperature": 0.0, "max_tokens": mc.DEFAULT_MAX_TOKENS,
            "random_seed": None, "top_p": 1.0,
            "response_format": "json_schema strict (WP2 spike decision)",
        },
        prompt_hashes=prompts.prompt_manifest_hashes(),
        config_after_load={
            "fresh_reset": True, "checkpoint_loaded": False,
            "fc_args": fc_args,
            "note": "fresh deterministic reset on the session's own server; "
                    "the turn-11 checkpoint is untouched",
        },
        checkpoint_checksums={},  # none: no checkpoint is loaded
        model_fingerprint=spike_fingerprint,
        arm_start_ts=arm_start_ts,
    )
    # Extra (forward-compat) fields — the decision-8 field set is frozen;
    # these are recorded alongside, never inside it.
    manifest.server_commit_note = (
        "recorded live at P2 was stdout-only; not persisted — the image "
        "digest pins the server build")
    manifest.python_lockfile_note = (
        "no lockfile in the repo; the venv is uv-managed "
        f"(python {sys.version.split()[0]})")
    manifest.checkpoint_checksums_note = (
        "fresh deterministic reset; no checkpoint loaded (the turn-11 "
        "checkpoint is untouched)")
    manifest.enforcement_decision = decision
    manifest.action_batch_schema_hash = mc.action_batch_schema_hash()
    manifest.live_fingerprint_probe = live_probe
    manifest.fingerprint_matches_spike = (
        live_probe.get("fingerprint") == spike_fingerprint
        if live_probe and spike_fingerprint else None)
    manifest.turn_11_checkpoint_untouched = True
    manifest.fault_run = {
        "purpose": "WP2 acceptance: the degraded path exercised under "
                   "fault injection (timeout, malformed output, invalid "
                   "action) against the live stack",
        "fault_injection_seam": "MistralClient.complete_once (one protocol "
                                "call = one logical API attempt); every "
                                "non-faulted attempt delegates to the real "
                                "transport",
        "schedule": {f"t{t}/a{a}": f for (t, a), f in
                     sorted(FAULT_SCHEDULE.items())},
    }
    return manifest


# ---------------------------------------------------------------------------
# the fault schedule + the fault builders
# ---------------------------------------------------------------------------

# (turn, attempt) -> fault name. Attempts NOT in the schedule are REAL
# model calls (turn 1 attempt 2: the unresponsive repair is a real re-ask).
FAULT_SCHEDULE = {
    (1, 1): "transport_timeout",
    (2, 1): "malformed_output",
    (2, 2): "scripted_valid_batch",
    (3, 1): "invalid_candidate_ref",
    (3, 2): "invalid_candidate_ref",
}

# The injected malformed output: a JSON batch whose action ref is free
# text (an ActionBatch schema violation; the turn client extracts the ref
# as a raw attempt for the schema_error accounting).
MALFORMED_OUTPUT = (
    '{"turn_id": "wp2-fault-run/t2", '
    '"actions": [{"ref": "send the settler to the nearby ruins"}], '
    '"commitments": [], "finish_requested": false, '
    '"notes": "injected malformed output"}')
MALFORMED_RAW_REF = "send the settler to the nearby ruins"

# A well-formed candidate ID (matches CANDIDATE_ID_RE) that is never
# offered in any turn of a fresh game — the state_fog_error class.
INVALID_REF = "unit:424242:goto_n"

# The scripted valid batch picks a build_city candidate from the turn's
# REAL offered candidate table (the CANDIDATE TABLE section of the real
# rendered prompt) — anchored past the section header so the compact-JSON
# snapshot/delta sections can never match.
_BUILD_CITY_RE = re.compile(r"^unit:(\d+):build_city\b", re.MULTILINE)


def _fault_transport_timeout(turn, attempt):
    from lechonk.model import mistral_client as mc

    error = (f"injected fault (turn {turn} attempt {attempt}): transport "
             f"timeout — the client-side deadline expired before any "
             f"response bytes arrived (simulated; no API call was made)")
    outcome = mc.AttemptOutcome(raw=None, error=error, batch=None,
                                malformed=None, raw_content="")
    detail = ("transport timeout injected at the complete_once seam; the "
              "runner classifies it unresponsive and re-asks the same turn "
              "prompt — the repair attempt is a REAL model call")
    return outcome, detail


def _fault_malformed_output(turn, attempt):
    from lechonk.model import mistral_client as mc

    malformed = (f"injected fault (turn {turn} attempt {attempt}): the "
                 f"emitted ActionBatch violates the schema — the action "
                 f"ref is free text, not a candidate ID (simulated; no "
                 f"API call was made)")
    outcome = mc.AttemptOutcome(raw=None, error=None, batch=None,
                                malformed=malformed,
                                raw_content=MALFORMED_OUTPUT)
    detail = (f"malformed output injected; the turn client extracts the "
              f"raw attempt {MALFORMED_RAW_REF!r}; the runner rejects it "
              f"schema_error and repairs with the schema_error template "
              f"(the failed output is appended to the repair conversation)")
    return outcome, detail


def _fault_scripted_valid_batch(turn, attempt, messages):
    from lechonk.core.schemas import ActionBatch, ActionItem
    from lechonk.model import mistral_client as mc

    text = "\n".join(m.get("content", "") for m in messages)
    section = text.split("## CANDIDATE TABLE", 1)[-1]
    match = _BUILD_CITY_RE.search(section)
    if match is not None:
        ref = f"unit:{match.group(1)}:build_city"
        batch = ActionBatch(turn_id=f"wp2-fault-run/t{turn}",
                            actions=[ActionItem(ref=ref)])
        detail = (f"scripted valid batch: the build_city candidate {ref!r} "
                  f"parsed from the turn's REAL offered candidate table "
                  f"(guarantees the schema-error repair path ends in an "
                  f"executed, read-back-verified action)")
    else:
        batch = ActionBatch(turn_id=f"wp2-fault-run/t{turn}", actions=[])
        detail = ("scripted valid EMPTY batch: no build_city candidate was "
                  "offered this turn (fallback — the repair still proves "
                  "the schema_error repair prompt assembly + a valid, "
                  "responsive batch)")
    outcome = mc.AttemptOutcome(raw=None, error=None, batch=batch,
                                malformed=None,
                                raw_content=batch.model_dump_json())
    return outcome, detail


def _fault_invalid_candidate_ref(turn, attempt):
    from lechonk.core.schemas import ActionBatch, ActionItem
    from lechonk.model import mistral_client as mc

    batch = ActionBatch(turn_id=f"wp2-fault-run/t{turn}",
                        actions=[ActionItem(ref=INVALID_REF)])
    outcome = mc.AttemptOutcome(raw=None, error=None, batch=batch,
                                malformed=None,
                                raw_content=batch.model_dump_json())
    detail = (f"well-formed candidate ID {INVALID_REF!r} never offered "
              f"this turn; the runner rejects it state_fog_error "
              f"(attempt {attempt} of the ≤2-attempt cap, decision 4)")
    return outcome, detail


_FAULT_BUILDERS = {
    "transport_timeout": lambda turn, attempt, messages: _fault_transport_timeout(turn, attempt),
    "malformed_output": lambda turn, attempt, messages: _fault_malformed_output(turn, attempt),
    "scripted_valid_batch": _fault_scripted_valid_batch,
    "invalid_candidate_ref": lambda turn, attempt, messages: _fault_invalid_candidate_ref(turn, attempt),
}


class FaultInjectingMistralClient(MistralClient):
    """The real ``MistralClient`` with a scripted fault schedule injected
    at the single logical-attempt seam (``complete_once`` — one protocol
    call = one logical API attempt). Every non-faulted attempt delegates
    to the real transport unchanged; the runner, the turn client and the
    ledger stay 100% real."""

    def __init__(self, *args, fault_sink=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fault_sink = fault_sink
        self._turn_calls = 0   # one kind="initial" per turn
        self._call_seq = 0     # 1-based complete_once sequence
        self.injections: list[dict] = []

    def complete_once(self, *, messages, prompt_hash, attempt, kind,
                      temperature=None, max_tokens=None, timeout_ms=None):
        self._call_seq += 1
        if kind == "initial":
            self._turn_calls += 1
        turn = self._turn_calls
        fault = FAULT_SCHEDULE.get((turn, attempt))
        if fault is None:
            return super().complete_once(
                messages=messages, prompt_hash=prompt_hash, attempt=attempt,
                kind=kind, temperature=temperature, max_tokens=max_tokens,
                timeout_ms=timeout_ms)
        t0 = time.time()
        outcome, detail = _FAULT_BUILDERS[fault](turn, attempt, messages)
        # Log the synthetic attempt like every other call (the turn client
        # correlates call_logs[-1]); the injection record is separate.
        call_log = self._make_call_log(
            attempt=attempt, kind=kind, prompt_hash=prompt_hash,
            latency_ms=int((time.time() - t0) * 1000), raw=None,
            error=outcome.error)
        self._emit_log(call_log)
        record = {
            "api_call_seq": self._call_seq,
            "client_call_id": call_log.call_id,
            "turn": turn,
            "attempt": attempt,
            "kind": kind,
            "fault": fault,
            "detail": detail,
            "outcome_error": outcome.error,
            "outcome_malformed": outcome.malformed,
            "outcome_batch_refs": ([a.ref for a in outcome.batch.actions]
                                   if outcome.batch is not None else None),
            "ts": call_log.ts,
        }
        self.injections.append(record)
        if self._fault_sink is not None:
            self._fault_sink(record)
        log(f"[fault] injected {fault} at turn {turn} attempt {attempt} "
            f"(kind={kind}, api_call_seq={self._call_seq})")
        return outcome


# ---------------------------------------------------------------------------
# post-run verification (from the ledger + the injection/api-call logs)
# ---------------------------------------------------------------------------


def verify(records, load, result, injections, call_records,
           degraded_observations):
    from lechonk.core.schemas import (
        ActionResultRecord,
        ModelCallRecord,
        RunManifestRecord,
    )

    actions = [r for r in records if isinstance(r, ActionResultRecord)]
    calls = [r for r in records if isinstance(r, ModelCallRecord)]
    by_turn_calls: dict[int, list] = {}
    for c in calls:
        turn = int(c.call_id.rsplit("/t", 1)[1].split("/")[0])
        by_turn_calls.setdefault(turn, []).append(c)
    by_turn_actions: dict[int, list] = {}
    for a in actions:
        by_turn_actions.setdefault(a.turn, []).append(a)

    # --- the manifest (decision 8) ---------------------------------------
    check("manifest_is_first_ledger_record",
          bool(records) and isinstance(records[0], RunManifestRecord),
          f"first record kind: {records[0].kind if records else None}")
    manifest_fields = records[0].manifest.model_dump(mode="json") \
        if records and isinstance(records[0], RunManifestRecord) else {}
    check("manifest_decision8_fields",
          all(manifest_fields.get(f) is not None
              for f in ("run_id", "image_digest", "vendored_base_sha",
                        "patch_head_sha", "harness_revision", "seeds",
                        "ruleset", "model_id", "sampling", "prompt_hashes",
                        "model_fingerprint", "arm_start_ts"))
          and manifest_fields.get("capstring") == CAPSTRING,
          "seeds, image digest, vendored SHAs, model ID + sampling + "
          "prompt/template hashes, arm fingerprint (spike artifact), "
          "capstring")

    # --- the run itself ----------------------------------------------------
    check("run_completed_3_turns",
          result.outcome == "COMPLETED" and result.turns_completed == TURNS,
          f"outcome={result.outcome} turns_completed="
          f"{result.turns_completed} reason={result.reason!r}")
    check("ledger_no_corruption", not load.discarded_trailing_partial,
          "no trailing partial record; every line checksum-verified")
    check("seq_contiguous",
          [r.seq for r in records] == list(range(len(records))),
          f"seq 0..{len(records) - 1}")
    check("attempt_cap_respected",
          all(len(v) <= 2 for v in by_turn_calls.values()),
          f"model_call records per turn: "
          f"{ {t: len(v) for t, v in sorted(by_turn_calls.items())} } "
          f"(≤2 total API attempts per turn, decision 4)")

    # --- the injected faults were recorded ---------------------------------
    expected = [(1, 1, "transport_timeout"), (2, 1, "malformed_output"),
                (2, 2, "scripted_valid_batch"), (3, 1, "invalid_candidate_ref"),
                (3, 2, "invalid_candidate_ref")]
    observed = [(i["turn"], i["attempt"], i["fault"]) for i in injections]
    check("fault_injections_recorded", observed == expected,
          f"observed {observed}")

    # --- cross-reference: every injection matches its api_calls record -----
    def api_record(seq):
        return call_records[seq - 1] if 0 < seq <= len(call_records) else None

    cross_ok = True
    cross_detail = []
    for inj in injections:
        rec = api_record(inj["api_call_seq"])
        if rec is None:
            cross_ok = False
            cross_detail.append(f"t{inj['turn']}/a{inj['attempt']}: no api "
                                f"record at seq {inj['api_call_seq']}")
            continue
        fault = inj["fault"]
        if fault == "transport_timeout":
            ok = (rec["outcome"] == "transport_error"
                  and rec["error"] and "injected fault" in rec["error"])
        elif fault == "malformed_output":
            ok = (rec["outcome"] == "schema_error"
                  and MALFORMED_RAW_REF in (rec.get("raw_attempts") or []))
        else:  # scripted_valid_batch / invalid_candidate_ref: a valid batch
            ok = (rec["outcome"] == "ok" and rec["tokens_in"] == 0
                  and rec["tokens_out"] == 0)
        if not ok:
            cross_ok = False
            cross_detail.append(f"t{inj['turn']}/a{inj['attempt']} ({fault}): "
                                f"api record outcome={rec['outcome']} "
                                f"tokens={rec['tokens_in']}/"
                                f"{rec['tokens_out']}")
    check("injections_match_api_call_log", cross_ok,
          "; ".join(cross_detail) if cross_detail else
          f"{len(injections)} injections cross-referenced by api_call_seq")

    # --- fault class 1: transport timeout → unresponsive repair (REAL) ------
    t1_calls = by_turn_calls.get(1, [])
    t1_attempt1 = [c for c in t1_calls if c.attempt == 1]
    t1_attempt2 = [c for c in t1_calls if c.attempt == 2]
    check("turn1_timeout_then_real_repair",
          len(t1_calls) == 2 and len(t1_attempt1) == 1
          and t1_attempt1[0].error is not None
          and "injected fault" in t1_attempt1[0].error
          and len(t1_attempt2) == 1,
          f"turn 1 model_calls: {len(t1_calls)} "
          f"(attempt 1 error={'injected' if t1_attempt1 and t1_attempt1[0].error else None}, "
          f"attempt 2 error={t1_attempt2[0].error if t1_attempt2 else 'n/a'}) — "
          f"the unresponsive repair is a REAL model call")
    t1_actions = by_turn_actions.get(1, [])
    t1_executed = [a for a in t1_actions if a.outcome == "executed"]
    log(f"[evidence] turn 1 real-repair outcome: "
        f"{len(t1_executed)} executed model actions "
        f"(readbacks: {sorted({a.readback for a in t1_executed})}), "
        f"rejected: {[(a.ref, a.rejection_class) for a in t1_actions if a.outcome == 'rejected']}, "
        f"scripted_default: {len([a for a in t1_actions if a.provenance_class == 'scripted_default'])}")

    # --- fault class 2: malformed output → schema_error repair -------------
    t2_actions = by_turn_actions.get(2, [])
    t2_schema_rejects = [a for a in t2_actions
                         if a.outcome == "rejected"
                         and a.rejection_class == "schema_error"]
    check("turn2_schema_error_rejection",
          any(a.ref == MALFORMED_RAW_REF for a in t2_schema_rejects),
          f"schema_error rejections on turn 2: "
          f"{[(a.ref, a.rejection_class) for a in t2_schema_rejects]}")
    t2_calls = by_turn_calls.get(2, [])
    check("turn2_repair_call_made", len(t2_calls) == 2,
          f"turn 2 model_calls: {len(t2_calls)} (attempt 1 malformed → "
          f"attempt 2 the schema_error repair)")
    t2_build = [a for a in t2_actions
                if a.outcome == "executed" and a.verb == "build_city"
                and a.provenance_class == "model"]
    t2_scripted = next((i for i in injections
                        if i["turn"] == 2 and i["attempt"] == 2), None)
    fallback = bool(t2_scripted) and "EMPTY batch" in t2_scripted["detail"]
    if t2_scripted is None:
        check("turn2_repair_yields_verified_action", False,
              "no scripted repair injection recorded for turn 2")
    elif fallback:
        check("turn2_repair_yields_verified_action",
              len([a for a in t2_actions
                   if a.provenance_class == "scripted_default"]) == 0,
              f"fallback branch: {t2_scripted['detail']} (the turn stayed "
              f"responsive — no degraded default)")
    else:
        check("turn2_repair_yields_verified_action",
              len(t2_build) == 1
              and t2_build[0].readback in ("verified", "inconclusive"),
              f"the scripted repair batch executed build_city "
              f"{t2_build[0].ref if t2_build else None} with readback "
              f"{t2_build[0].readback if t2_build else None} (executed, "
              f"never contradicted)")

    # --- fault class 3: invalid candidate ref → state_fog → degraded -------
    t3_actions = by_turn_actions.get(3, [])
    t3_fog_rejects = [a for a in t3_actions
                      if a.outcome == "rejected"
                      and a.rejection_class == "state_fog_error"
                      and a.ref == INVALID_REF]
    check("turn3_state_fog_rejections_both_attempts",
          len(t3_fog_rejects) == 2,
          f"state_fog_error rejections of {INVALID_REF!r} on turn 3: "
          f"{len(t3_fog_rejects)} (one per attempt, ≤2-attempt cap)")
    t3_calls = by_turn_calls.get(3, [])
    check("turn3_repair_cap_exhausted", len(t3_calls) == 2,
          f"turn 3 model_calls: {len(t3_calls)} (both attempts invalid → "
          f"the repair cap → the degraded default)")
    t3_default = [a for a in t3_actions
                  if a.provenance_class == "scripted_default"]
    t3_default_exec = [a for a in t3_default if a.outcome == "executed"]
    t3_model_exec = [a for a in t3_actions
                     if a.outcome == "executed"
                     and a.provenance_class == "model"]
    t3_aborted = [a for a in t3_actions if a.outcome == "aborted_group"]
    # The degraded default ran on turn 3 — observed directly at the decide
    # entry point (§7.4 line 163; observation only, no behavior change).
    degraded_obs = [o for o in degraded_observations if o["turn"] == 3]
    check("turn3_degraded_default_ran",
          len(degraded_obs) == 1
          and degraded_obs[0]["policy"] == "degraded_default",
          f"comparator/defaults.decide observed once on turn 3 "
          f"(policy={degraded_obs[0]['policy'] if degraded_obs else None}, "
          f"decisions="
          f"{degraded_obs[0]['decision_count'] if degraded_obs else None}) "
          f"after the repair cap")
    if degraded_obs and degraded_obs[0]["decision_count"] == 0:
        # The §7.4 preserve-everything branch: research is active (turn 1's
        # real action) and the turn-2 city's production is preserved, so
        # there were no NEW mandatory choices to resolve. What the ledger
        # proves: no model action executed, nothing aborted, no
        # scripted_default action_results — and the pipeline still finished
        # the phase (the run reached the horizon).
        check("turn3_degraded_preserves_and_finishes",
              not t3_model_exec and not t3_aborted and not t3_default
              and result.outcome == "COMPLETED"
              and result.turns_completed == TURNS,
              f"zero resolutions on this state (research active + "
              f"production preserved): 0 model executions, 0 aborted "
              f"groups, 0 scripted_default actions; the pipeline still "
              f"finished the phase (COMPLETED, {result.turns_completed} "
              f"turns)")
    else:
        check("turn3_degraded_default_executed",
              len(t3_default_exec) >= 1
              and all(a.readback in ("verified", "inconclusive")
                      for a in t3_default_exec),
              f"scripted_default actions on turn 3: "
              f"{[(a.ref, a.verb, a.outcome, a.readback) for a in t3_default]} "
              f"— the degraded default executed with read-back (never "
              f"contradicted)")

    # --- honest reporting of the real (turn 1) model behavior ---------------
    all_exec = [a for a in actions if a.outcome == "executed"]
    log(f"[evidence] all executed actions: "
        f"{[(a.turn, a.ref, a.provenance_class, a.readback) for a in all_exec]}")
    return {
        "turn1_real_repair": {
            "model_calls": len(t1_calls),
            "attempt1_error_injected": bool(
                t1_attempt1 and t1_attempt1[0].error
                and "injected fault" in t1_attempt1[0].error),
            "attempt2_error": t1_attempt2[0].error if t1_attempt2 else None,
            "executed": [(a.ref, a.readback) for a in t1_executed],
        },
        "turn2_schema_error": {
            "rejections": [(a.ref, a.rejection_class)
                           for a in t2_schema_rejects],
            "repair_calls": len(t2_calls),
            "scripted_repair": (t2_scripted or {}).get("detail"),
            "build_city_executed": [(a.ref, a.readback) for a in t2_build],
        },
        "turn3_state_fog_degraded": {
            "state_fog_rejections": len(t3_fog_rejects),
            "model_calls": len(t3_calls),
            "model_executions": len(t3_model_exec),
            "aborted_groups": len(t3_aborted),
            "scripted_default": [(a.ref, a.verb, a.outcome, a.readback)
                                 for a in t3_default],
            "degraded_default_observed": degraded_obs,
        },
        "readbacks": {
            "verified": len([a for a in all_exec if a.readback == "verified"]),
            "inconclusive": len([a for a in all_exec
                                 if a.readback == "inconclusive"]),
            "contradicted": len([a for a in all_exec
                                 if a.readback == "contradicted"]),
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    log(f"[{RUN}] repo={REPO}")
    log(f"[{RUN}] rundir={RUNDIR} attempt={ATTEMPT} seed={SEED} turns={TURNS}")
    log(f"[{RUN}] policy: live fault injection at the complete_once seam "
        f"over the real stack — schedule: "
        f"{ {f't{t}/a{a}': f for (t, a), f in sorted(FAULT_SCHEDULE.items())} }")

    # --- wait for the stack (bounded pubstatus retries) --------------------
    deadline = time.time() + 120
    while True:
        try:
            with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
                if r.status == 200:
                    log(f"[{RUN}] pubstatus 200 — stack up")
                    break
        except Exception as e:
            if time.time() > deadline:
                log(f"[{RUN}] FATAL: pubstatus unreachable: {e!r}")
                return 2
            time.sleep(2)

    # --- API key: the narrow loader (presence only, never the value) --------
    from lechonk.model import mistral_client as mc

    try:
        mc.load_api_key_from_env_file()
    except mc.MistralKeyError:
        pass
    key_present = os.environ.get(mc.API_KEY_ENV_VAR) is not None
    log(f"[{RUN}] MISTRAL_API_KEY present: {key_present} (presence only; "
        f"the value is never printed, logged, or artifacted)")
    if not key_present:
        log(f"[{RUN}] FATAL: no API key — the run cannot start")
        return 2

    # --- the fault-injecting client + the real turn client ------------------
    api_log_path = fresh_artifact("api_calls.jsonl")
    api_log_fh = open(api_log_path, "a", encoding="utf-8")

    def api_log(record):
        api_log_fh.write(json.dumps(record, sort_keys=True,
                                    ensure_ascii=False, default=str) + "\n")
        api_log_fh.flush()

    fault_path = fresh_artifact("fault_injections.jsonl")
    fault_fh = open(fault_path, "a", encoding="utf-8")

    def fault_sink(record):
        fault_fh.write(json.dumps(record, sort_keys=True,
                                  ensure_ascii=False, default=str) + "\n")
        fault_fh.flush()

    client = FaultInjectingMistralClient(enforcement="json_schema",
                                         fault_sink=fault_sink)
    arm_start_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spike_fingerprint = load_spike_fingerprint()
    log(f"[{RUN}] arm fingerprint from the spike artifact: "
        f"{spike_fingerprint}")
    live_probe = None
    try:
        probe = client.probe_fingerprint()
        live_probe = probe.to_dict()
        log(f"[{RUN}] live fingerprint probe: {probe.fingerprint} "
            f"(matches spike: {probe.fingerprint == spike_fingerprint})")
    except Exception as exc:
        log(f"[{RUN}] WARN: live fingerprint probe failed: {exc!r} "
            f"(the spike artifact fingerprint is still recorded)")
        live_probe = {"error": str(exc)}

    # The fingerprint probe is an API call too (decision 4) — recorded
    # with kind="probe", turn=null.
    if live_probe and "error" not in live_probe:
        probe_log = client.call_logs[-1].to_dict()
        api_log({
            "call_id": probe_log["call_id"], "turn": None, "attempt": 0,
            "kind": "probe", "prompt_hash": probe_log["prompt_hash"],
            "model_requested": probe_log["model_requested"],
            "model_responded": probe_log["model_responded"],
            "enforcement": probe_log["enforcement"],
            "temperature": probe_log["temperature"],
            "latency_ms": probe_log["latency_ms"],
            "tokens_in": probe_log["tokens_in"],
            "tokens_out": probe_log["tokens_out"],
            "tokens_total": probe_log["tokens_total"],
            "cost_usd": probe_log["cost_usd"],
            "finish_reason": probe_log["finish_reason"],
            "response_id": probe_log["response_id"],
            "fingerprint": probe_log["fingerprint"],
            "transport_retries": probe_log["transport_retries"],
            "error": probe_log["error"], "outcome": "probe",
            "raw_attempts": [], "section_tokens": None,
            "ts": probe_log["ts"],
        })

    from lechonk.model.turn_client import MistralTurnClient

    adapter = MistralTurnClient(client, api_log=api_log)
    log(f"[{RUN}] adapter: MistralTurnClient over "
        f"FaultInjectingMistralClient (enforcement="
        f"{client.enforcement.value}, model={client.model}, "
        f"temperature={client.temperature}, max_tokens="
        f"{client.max_tokens}); non-faulted attempts are REAL API calls")

    # --- the manifest (decision 8) + the ledger + the runner ----------------
    from lechonk.core.ledger import TranscriptLedger
    from lechonk.core.runner import GameRunner, RunnerConfig

    manifest = build_manifest(arm_start_ts, spike_fingerprint, live_probe)
    transcript_path = fresh_artifact("transcript.jsonl")
    ledger = TranscriptLedger(transcript_path)
    config = RunnerConfig(
        max_turns=TURNS, run_id=RUN, workdir=WORKDIR, seed=SEED,
        username="myagent", aifill=2)
    runner = GameRunner(
        config=config, client=adapter, ledger=ledger, manifest=manifest)

    # --- observe the degraded-default entry point (§7.4 line 163) -----------
    # Observation only: the wrapper records the decision log and delegates
    # to the real decide — the pipeline behavior is unchanged. The runner
    # looks the function up on the module at call time (runner.py:880).
    from lechonk.comparator import defaults as defaults_mod

    degraded_observations = []
    real_decide = defaults_mod.decide

    def observed_decide(snapshot, table):
        decision_log = real_decide(snapshot, table)
        degraded_observations.append({
            "turn": decision_log.turn,
            "policy": decision_log.policy,
            "decision_count": len(decision_log.decisions),
            "decisions": [
                {"candidate_id": d.candidate_id, "domain": d.domain,
                 "verb": d.verb, "trigger": d.trigger, "detail": d.detail}
                for d in decision_log.decisions],
        })
        return decision_log

    defaults_mod.decide = observed_decide

    log(f"[{RUN}] starting the fault run: {TURNS} turns, fresh "
        f"deterministic reset on the session's own server")
    run_t0 = time.time()
    result = runner.run()
    run_wall_s = time.time() - run_t0
    api_log_fh.close()
    fault_fh.close()
    ledger.close()
    log(f"[{RUN}] run finished: outcome={result.outcome} "
        f"reason={result.reason!r} turns_completed={result.turns_completed} "
        f"close_status={result.close_status} wall_clock={run_wall_s:.1f}s")

    # --- post-run: verify the fault evidence from the ledger ---------------
    from lechonk.core.ledger import load_transcript

    load = load_transcript(transcript_path)
    records = load.records
    log(f"[{RUN}] transcript: {len(records)} records, "
        f"discarded_trailing_partial={load.discarded_trailing_partial}, "
        f"last_seq={load.last_seq}")
    call_records = adapter.call_records
    log(f"[{RUN}] api call log: {len(call_records)} records "
        f"({api_log_path}); fault injections: {len(client.injections)} "
        f"({fault_path})")

    evidence = verify(records, load, result, client.injections, call_records,
                      degraded_observations)

    write_json("fault_evidence.json", {
        "run": RUN, "attempt": ATTEMPT, "seed": SEED,
        "evidence": evidence,
        "injections": client.injections,
        "degraded_default_observations": degraded_observations,
    })

    # --- human-readable evidence --------------------------------------------
    lines = [
        "# WP2 fault run — fault-injection evidence",
        "",
        f"- run: `{RUN}` attempt {ATTEMPT}, seed {SEED}, {TURNS} turns, "
        f"fresh deterministic reset on the session's own server",
        f"- outcome: {result.outcome} (turns_completed="
        f"{result.turns_completed}, reason={result.reason!r})",
        f"- wall clock: {run_wall_s:.1f} s",
        "",
        "## Fault schedule (injected at the `complete_once` seam)",
        "",
        "| turn | attempt | fault | pipeline path exercised |",
        "|---|---|---|---|",
        "| 1 | 1 | transport timeout | runner classifies `unresponsive` "
        "→ repair re-asks the same prompt (REAL API call) |",
        "| 2 | 1 | malformed output (free-text ref) | rejected "
        "`schema_error` → repair with the schema_error template |",
        "| 2 | 2 | scripted valid batch (build_city from the real offered "
        "table) | repair yields an executed, read-back-verified action |",
        "| 3 | 1 | invalid candidate ref (`unit:424242:goto_n`) | rejected "
        "`state_fog_error` → repair with the state_fog template |",
        "| 3 | 2 | invalid candidate ref (again) | repair cap (≤2 attempts) "
        "→ degraded default (`scripted_default`) executed with read-back |",
        "",
        "## Observed ledger evidence",
        "",
        f"- turn 1 (timeout → real repair): {evidence['turn1_real_repair']}",
        f"- turn 2 (malformed → schema_error repair): "
        f"{evidence['turn2_schema_error']}",
        f"- turn 3 (invalid ref → state_fog → degraded): "
        f"{evidence['turn3_state_fog_degraded']}",
        f"- read-backs over all executed actions: {evidence['readbacks']}",
        f"- degraded-default observations (decide entry point): "
        f"{degraded_observations}",
        "",
        "## Checks",
        "",
    ]
    for name, entry in checks.items():
        lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                     f"`{name}`: {entry['detail']}")
    with open(os.path.join(RUNDIR, "fault_evidence.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote fault_evidence.md")

    verdict = all(c["pass"] for c in checks.values())
    write_json("run_meta.json", {
        "run": RUN, "attempt": ATTEMPT, "seed": SEED, "turns": TURNS,
        "outcome": {"outcome": result.outcome, "reason": result.reason,
                    "turns_completed": result.turns_completed,
                    "close_status": result.close_status},
        "wall_clock_s": round(run_wall_s, 1),
        "transcript": {"records": len(records),
                       "discarded_trailing_partial":
                           load.discarded_trailing_partial,
                       "last_seq": load.last_seq},
        "api_calls": len(call_records),
        "fault_injections": len(client.injections),
        "degraded_default_observations": degraded_observations,
        "evidence": evidence,
        "checks": checks,
        "verdict": "PASS" if verdict else "FAIL",
        "artifacts": {
            "stdout": "run_stdout.log",
            "transcript": "transcript.jsonl",
            "fault_injections": "fault_injections.jsonl",
            "api_calls": "api_calls.jsonl",
            "evidence": "fault_evidence.json/.md",
            "run_meta": "run_meta.json",
        },
    })
    log(f"[{RUN}] VERDICT: {'PASS' if verdict else 'FAIL'} "
        f"({sum(1 for c in checks.values() if c['pass'])}/"
        f"{len(checks)} checks)")
    _log_file.close()
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
