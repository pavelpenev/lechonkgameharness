"""WP2 package C — the 10-turn model-driven integration run.

design-plan-freeciv-v1.1.md §4 line 54 (WP2 acceptance): "10 model-driven
turns with read-back-verified effects; measured snapshot/delta/memory token
counts over those turns within the §3 budgets". This driver wires the FULL
pipeline for 10 turns against the live stack:

  AdapterSession (fresh deterministic reset, own game server — the turn-11
  checkpoint and pre-existing games are NOT touched)
    -> obs_extract (fog-gated snapshot, §3 truncation policy)
    -> candidates (legal-candidate table)
    -> prompts (versioned templates, §7.5)
    -> MistralClient (json_schema enforcement — the WP2 spike decision,
       decided live, recorded in the run manifest)
    -> core.runner state machine (ACTING -> FINISH_SENT -> WAITING -> ...)
    -> ledger (append-only checksummed JSONL transcript)
    -> memory (strategy memory, pre/post-advance separation)
    -> metrics (the §3 honesty metrics over the ledger)

Requirements implemented here:
- every model action is read-back-verified by the pipeline (the per-turn
  read-back evidence is aggregated from the ledger; honest INCONCLUSIVE is
  acceptable only where documented quirks apply — the sentry quirk);
- per-call API logging (decision 4: tokens, latency, cost) — api_calls.jsonl;
- TOKEN MEASUREMENT: the API-reported prompt/completion token counts per call
  are the ground truth; the rendered snapshot/delta/memory sizes are recorded
  alongside, and the per-turn numbers are compared against the §3 budgets
  (memory ≤ 1,500, snapshot ≤ 8,000, delta ≤ 800);
- the run manifest (decision 8 fields) is the FIRST ledger record (the
  runner emits it durable before anything else);
- the provenance ledger is complete for all 10 turns (verified post-run);
- the WP2-exit feasibility inputs (plan line 54 / probe P12): the measured
  mean per-turn wall-clock and the decision arithmetic (the ladder triggers
  plus the wall-clock budget cross-check with comparator arms and the ≥1.5
  contingency) — the DECISION itself is the orchestrator's; this run provides
  the measured inputs honestly.

Everything persists under artifacts/runs/wp2-model-run/:
  run_stdout.log, transcript.jsonl, api_calls.jsonl, token_budget_table.json,
  token_budget_table.md, metrics_report.json, metrics_report.md, run_meta.json,
  workdir/ (the dedicated CWD for upstream side effects).

The API key loads via the existing client loader (env var or the repo-root
.env at runtime); the key value is never printed, logged, or artifacted.

Bounded retry: one re-execution per infrastructure failure (TRUNCATED with an
infrastructure reason), logged — never a loop-retry, never a fabricated
completion. Model-driven degradation (repair cap -> degraded defaults) is NOT
an infrastructure failure and is reported, not retried.
"""

import argparse
import atexit
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from artifact_lifecycle import (ArtifactCollisionError, claim_run_directory,
                                prepare_attempt_artifact, prepare_retry_artifacts)
from model_heartbeat import ModelCallHeartbeat
from model_repetition import RepetitionDetector
from lechonk.capture_gateway import CaptureGateway

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lechonk.core.recovery import validate_player_name


def _known_number(value):
    """Return whether an API usage value is safe to include in a sum.

    Provider error records deliberately retain ``None`` usage fields.  They
    are useful evidence, but must not make post-run reporting crash (or be
    rendered as a fabricated zero on the per-call record).
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _report_value(value):
    """Return a stable human-readable value for incomplete telemetry.

    A failed or truncated turn can have no rendered prompt sections (or only
    a subset of them).  Reports must preserve that distinction instead of
    crashing or implying that the measurement was zero.
    """
    return "unknown" if value is None else value


def _section_report_value(section_tokens, name):
    """Render one section token count, including missing section keys."""
    if not isinstance(section_tokens, dict):
        return "unknown"
    return _report_value(section_tokens.get(name))


def resolve_player_name(requested, stored, *, fresh):
    """Resolve visible name while preserving legacy nameless checkpoints."""
    if fresh:
        return requested or "Le Chonk"
    if stored is not None and requested is not None and requested != stored:
        raise ValueError("--player-name conflicts with checkpoint")
    return stored if stored is not None else requested


def _token_usage_totals(records):
    """Aggregate token usage while preserving an explicit unknown count.

    The numeric fields are safe totals of known values (zero when all values
    are unavailable); the ``*_unknown_calls`` fields make that zero honest.
    Individual records are never modified.
    """
    in_values = [record.get("tokens_in") for record in records]
    out_values = [record.get("tokens_out") for record in records]
    in_known = [value for value in in_values if _known_number(value)]
    out_known = [value for value in out_values if _known_number(value)]
    in_unknown = len(in_values) - len(in_known)
    out_unknown = len(out_values) - len(out_known)
    return {
        "tokens_in": sum(in_known),
        "tokens_out": sum(out_known),
        "tokens_in_unknown_calls": in_unknown,
        "tokens_out_unknown_calls": out_unknown,
        "token_measurement_status": (
            "unknown" if in_unknown or out_unknown else "complete"),
    }


def _numeric_total(records, field):
    """Sum a numeric API field, treating unavailable values as zero."""
    return sum(value for value in (record.get(field) for record in records)
               if _known_number(value))


def _cost_usage_totals(records):
    """Return known cost subtotal plus explicit unknown-call accounting."""
    values = [record.get("cost_usd") for record in records]
    known = [value for value, record in zip(values, records)
             if _known_number(value) and record.get("cost_status") != "unknown"]
    return {
        "cost_usd": sum(known),
        "cost_unknown_calls": len(values) - len(known),
        "cost_measurement_status": "unknown" if len(known) != len(values)
        else "complete",
    }


def _numeric_max(values):
    """Return the maximum known measurement, or None if all are unknown."""
    known = [value for value in values if _known_number(value)]
    return max(known) if known else None


def _section_max(per_turn, section):
    """Maximum rendered section size, retaining an unknown-only result."""
    return _numeric_max(
        (turn.get("section_tokens", {}).get(section)
         if isinstance(turn.get("section_tokens"), dict) else None
         for turn in per_turn))


def _is_compaction_call(record):
    """Whether an API log record is context compaction, not a turn attempt."""
    compaction_kinds = {"conversation_compaction",
                        "conversation_compaction_repair"}
    return (record.get("kind") in compaction_kinds
            or record.get("physical_kind") in compaction_kinds)


def _snapshot_within_8000(section_tokens, ledger_tokens):
    """Validate both prompt and ledger snapshot measurements.

    A missing ledger reference is unknown, not zero; this deliberately fails
    the acceptance check while retaining the missing value in the report.
    """
    return (_known_number(section_tokens) and section_tokens <= 8000
            and _known_number(ledger_tokens) and ledger_tokens <= 8000)


def _parse_args(argv=None):
    """CLI: --turns (default 10, the WP2 acceptance count); --run-id and
    --seed so a short spot-check run persists under its own
    artifacts/runs/<id>/ directory with its own seed, never touching
    another run's artifacts."""
    parser = argparse.ArgumentParser(
        description="WP2 package C — model-driven integration run")
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("--turns", type=int, default=None,
                         help="number of model-driven turns (default: 10)")
    horizon.add_argument("--until-game-over", action="store_true",
                         help="continue until Freeciv reports game over")
    parser.add_argument("--run-id", default="wp2-model-run",
                        help="run id / artifact directory name under "
                             "artifacts/runs/ (default: wp2-model-run)")
    parser.add_argument("--seed", type=int, default=20261028,
                        help="deterministic map/game/agent seed "
                             "(default: 20261028)")
    parser.add_argument("--player-name", default=None,
                        help="visible Freeciv ruler name (default fresh run: Le Chonk; resume adopts checkpoint)")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        "--aifill", type=int, default=None,
        help="raw Freeciv aifill setting (total player slots, including the "
             "human; 1-10; default: 2)")
    scenario.add_argument(
        "--opponents", type=int, default=None,
        help="number of AI opponents (converted to aifill=opponents+1; "
             "default: 1)")
    parser.add_argument(
        "--resume", metavar="CHECKPOINT_MANIFEST", default=None,
        help="resume from a verified start-of-turn recovery manifest; use a new --run-id")
    parser.add_argument(
        "--api-timeout-s", type=float, default=120.0,
        help="per-request API timeout in seconds (default: 120)")
    parser.add_argument(
        "--acting-deadline-s", type=float, default=180.0,
        help="maximum model/action phase per turn in seconds (default: 180)")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="high",
        help="Mistral reasoning effort; 'none' disables the thinking trace "
             "(default: high)")
    parser.add_argument(
        "--stream", dest="stream", action="store_true",
        help="stream reasoning/output deltas as the response arrives")
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help="disable response streaming (default for direct runs)")
    parser.add_argument(
        "--context-policy", choices=("legacy", "conversation_v1"),
        default="legacy",
        help="prompt context policy (default: legacy; interactive launcher uses conversation_v1)")
    parser.add_argument(
        "--prompt-profile", choices=("legacy_eval", "gameplay_v1", "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"),
        default="legacy_eval",
        help="prompt profile (default: legacy_eval; interactive launcher uses gameplay_v2)")
    parser.add_argument(
        "--open-observer", action="store_true",
        help="open the read-only observer URL once after session startup")
    parser.add_argument(
        "--capture", action="store_true",
        help="capture structured model/progress events (requires --stream)")
    parser.add_argument(
        "--capture-dir", default=None, metavar="DIR",
        help="new, empty capture directory (requires --capture and --stream)")
    parser.add_argument(
        "--capture-sync", action="store_true",
        help="wait briefly for milestone screenshots (requires --capture)")
    parser.add_argument(
        "--capture-sync-timeout-ms", type=int, default=2000,
        help="maximum wait for one recording checkpoint (default: 2000 ms)")
    parser.add_argument(
        "--capture-sync-budget-ms", type=int, default=6000,
        help="maximum recording wait per turn (default: 6000 ms)")
    parser.set_defaults(stream=False)
    args = parser.parse_args(argv)
    # AdapterSession changes the process-wide cwd to the run workdir during
    # startup. Keep the checkpoint anchored to the caller's cwd so a bounded
    # retry (which re-executes this script after that cwd change) cannot
    # reinterpret a relative --resume path under artifacts/runs/<id>/workdir.
    if args.resume is not None:
        args.resume = str(Path(args.resume).expanduser().resolve())
    if args.api_timeout_s <= 0:
        parser.error("--api-timeout-s must be positive")
    if args.player_name is not None:
        try:
            args.player_name = validate_player_name(args.player_name)
        except ValueError as exc:
            parser.error(str(exc))
    if args.acting_deadline_s <= 0:
        parser.error("--acting-deadline-s must be positive")
    if args.acting_deadline_s < args.api_timeout_s + 25:
        parser.error("--acting-deadline-s must be at least "
                     "--api-timeout-s + 25 seconds")
    # Freeciv's pingtimeout has a hard 1800-second maximum; AdapterSession
    # derives it as acting deadline plus a 60-second margin.  Reject an
    # impossible request instead of silently clamping it and creating a run
    # whose declared deadline is not the one the server can enforce.
    if args.acting_deadline_s + 60 > 1800:
        parser.error("--acting-deadline-s + 60 seconds must be at most "
                     "1800 seconds (maximum acting deadline: 1740 seconds)")
    if args.capture and not args.stream:
        parser.error("--capture requires --stream")
    if args.capture_dir is not None and not args.capture:
        parser.error("--capture-dir requires --capture")
    if args.capture_sync and not args.capture:
        parser.error("--capture-sync requires --capture")
    if args.capture_sync_timeout_ms <= 0:
        parser.error("--capture-sync-timeout-ms must be positive")
    if args.capture_sync_budget_ms <= 0:
        parser.error("--capture-sync-budget-ms must be positive")
    if args.capture_sync and args.acting_deadline_s < (
            args.api_timeout_s + 25 + args.capture_sync_budget_ms / 1000):
        parser.error("--acting-deadline-s must be at least "
                     "--api-timeout-s + 25 seconds + capture sync budget")
    # The bundled CivRealm server is configured with maxplayers=10 and this
    # client reserves one slot for the controlled human player.  Keep this
    # check here (rather than letting the server silently clamp the setting)
    # so a run's declared scenario is always the scenario that starts.
    if args.aifill is None and args.opponents is None:
        args.aifill, args.opponents = 2, 1
    elif args.opponents is not None:
        if not 1 <= args.opponents <= 9:
            parser.error("--opponents must be between 1 and 9 "
                         "(maxplayers=10, one human slot reserved)")
        args.aifill = args.opponents + 1
    elif not 1 <= args.aifill <= 10:
        parser.error("--aifill must be between 1 and 10 "
                     "(maxplayers=10, including the human slot)")
    if not args.until_game_over and args.turns is None:
        args.turns = 10
    if args.turns is not None and args.turns <= 0:
        parser.error("--turns must be positive")
    if args.resume is not None and args.run_id == "wp2-model-run":
        parser.error("--resume requires an explicit new --run-id")
    if args.resume is not None and args.context_policy != "conversation_v1":
        parser.error("--resume requires --context-policy conversation_v1")
    if (args.prompt_profile != "legacy_eval"
            and args.context_policy != "conversation_v1"):
            parser.error("--prompt-profile gameplay_v1/gameplay_v2/gameplay_v3/gameplay_v4/gameplay_v5/gameplay_v6 requires "
                     "--context-policy conversation_v1")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id)
            or args.run_id in {".", ".."}):
        parser.error("--run-id must be a safe artifact directory name")
    return args


_ARGS = _parse_args()
RUN = _ARGS.run_id
RUNDIR = os.path.join(REPO, "artifacts", "runs", RUN)
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream

SEED = _ARGS.seed  # default 20261028: distinct from wp1 runs and the spike
TURNS = None if _ARGS.until_game_over else _ARGS.turns
UNTIL_GAME_OVER = TURNS is None
API_TIMEOUT_S = _ARGS.api_timeout_s
ACTING_DEADLINE_S = _ARGS.acting_deadline_s
REASONING_EFFORT = _ARGS.reasoning_effort
AIFILL = _ARGS.aifill
OPPONENTS = _ARGS.opponents
STREAM = _ARGS.stream
CONTEXT_POLICY = _ARGS.context_policy
PROMPT_PROFILE = _ARGS.prompt_profile
PLAYER_NAME = _ARGS.player_name
OPEN_OBSERVER = _ARGS.open_observer
CAPTURE_ENABLED = _ARGS.capture
CAPTURE_DIR = _ARGS.capture_dir
CAPTURE_SYNC = _ARGS.capture_sync
CAPTURE_SYNC_TIMEOUT_MS = _ARGS.capture_sync_timeout_ms
CAPTURE_SYNC_BUDGET_MS = _ARGS.capture_sync_budget_ms
PROMPT_BUNDLE_ID = PROMPT_PROFILE if PROMPT_PROFILE in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} else None
_PROMPT_SNAPSHOT = None
RECOVERY_DIGEST_SCHEMA = ("interactive-config-v3"
                          if PROMPT_PROFILE in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} else None)
CANDIDATE_SURFACE_POLICY = ("candidate_surface_v3" if PROMPT_PROFILE == "gameplay_v6" else "candidate_surface_v2"
                            if PROMPT_PROFILE in {"gameplay_v3", "gameplay_v4", "gameplay_v5"}
                            else "legacy_v1")
CONVERSATION_COMPACTION_LIMIT_TOKENS = 500_000
CONVERSATION_TAIL_TARGET_TOKENS = 100_000
STATIC_REFERENCE_PATH = os.path.join(
    REPO, "artifacts", "reference", "freeciv", "reference.json")
STATIC_MANUAL_REFERENCE_PATH = os.path.join(
    REPO, "artifacts", "reference", "freeciv-manual", "reference.json")
CONTEXT_JOURNAL_PATH = (
    os.path.join(RUNDIR, "conversation.jsonl")
    if CONTEXT_POLICY == "conversation_v1" else None)
CONVERSATION_STATE_PATH = (
    os.path.join(RUNDIR, "state.json")
    if CONTEXT_POLICY == "conversation_v1" else None)
# AdapterSession derives the controller server timeout as the greater of its
# historical 300-second floor and acting deadline plus a 60-second margin.
SERVER_TIMEOUT_EFFECTIVE_S = max(300.0, ACTING_DEADLINE_S + 60.0)
PINGTIMEOUT_CONFIGURED_S = max(
    1800.0 if UNTIL_GAME_OVER else 720.0, ACTING_DEADLINE_S + 60.0)
PUBSTATUS = "http://localhost:8080/pubstatus"
ATTEMPT = int(os.environ.get("WP2_MODEL_RUN_ATTEMPT", "1"))
RESUME_MANIFEST = _ARGS.resume
RESUME_CHECKPOINT = None
# Fresh roots get an isolated Freeciv save namespace.  A resumed lineage is
# replaced with the verified manifest value below, preserving it unchanged
# across generations (including explicit retry attempts).
from lechonk.core.recovery import derive_freeciv_username
FREECIV_USERNAME = derive_freeciv_username(RUN)


def _retry_argv_with_absolute_resume(argv, resume_path):
    """Return *argv* with every resume spelling anchored to *resume_path*.

    ``sys.argv`` still contains the user's original relative path even though
    argument parsing canonicalizes ``_ARGS.resume``. The bounded retry is a
    fresh process and must receive the canonical value explicitly; otherwise
    its import-time resume validation runs after the previous process changed
    cwd and resolves the path against the wrong directory.
    """
    if resume_path is None:
        return list(argv)
    absolute = str(Path(resume_path).expanduser().resolve())
    result = list(argv)
    for index, value in enumerate(result):
        if value == "--resume" and index + 1 < len(result):
            result[index + 1] = absolute
        elif value.startswith("--resume="):
            result[index] = "--resume=" + absolute
    return result


def _capture_retry_block_reason(capture_enabled):
    """Explain why an implicit retry is unsafe for a recorded run.

    The presenter attaches to one capture manifest for the lifetime of a
    process.  Re-executing with the same explicit ``--capture-dir`` collides
    with that manifest; choosing a new directory would instead produce a
    second, unobserved game unless a new presenter were attached.  Fail closed
    so a capture never silently loses its visual continuity.
    """
    if not capture_enabled:
        return None
    return (
        "implicit bounded retry disabled because capture is enabled; the "
        "observer cannot safely reattach to a new capture directory. "
        "Start a new run with a new --run-id and an empty --capture-dir "
        "(or rerun without --capture)"
    )


def _static_classic_reference_provenance():
    """Read the extracted classic tree identity, if provisioned.

    Extraction is optional, so an absent/invalid metadata file is represented
    explicitly as ``present=False`` rather than as an empty digest.  This is
    used for checkpoint binding only; the active ruleset snapshot remains the
    authoritative runtime reference.
    """
    try:
        with open(STATIC_REFERENCE_PATH, encoding="utf-8") as handle:
            metadata = json.load(handle)
        schema = metadata.get("schema_version")
        tree = metadata.get("classic_ruleset_tree_sha256")
        if (not isinstance(schema, str) or not schema or
                not isinstance(tree, str) or
                not re.fullmatch(r"[0-9a-f]{64}", tree)):
            return {"present": False, "schema_version": None, "tree_sha256": None,
                    "status": "unavailable"}
        return {"present": True, "schema_version": schema,
                "tree_sha256": tree, "status": "available"}
    except (OSError, ValueError, TypeError):
        return {"present": False, "schema_version": None, "tree_sha256": None,
                "status": "unavailable"}


STATIC_CLASSIC_REFERENCE = _static_classic_reference_provenance()


def _classic_runtime_schema():
    """Return the semantic schema emitted by the runtime reference loader."""
    if not STATIC_CLASSIC_REFERENCE["present"]:
        return None
    try:
        from lechonk.adapter.classic_reference import load_classic_reference
        reference = load_classic_reference(os.path.dirname(STATIC_REFERENCE_PATH))
        schema = getattr(reference, "schema_version", None)
        return schema if isinstance(schema, str) and schema else None
    except Exception:
        return None


CLASSIC_RUNTIME_SCHEMA = _classic_runtime_schema()


def _static_manual_reference_provenance():
    """Read and verify generated-manual provenance, if provisioned.

    The manual is optional.  Missing, malformed, or stale files are reported
    as unavailable and never represented by a fabricated digest.  When the
    extractor did not publish a semantic content hash, the ordered file
    digest map is hashed as a stable fallback identity.
    """
    try:
        # Use the exact production loader used by AdapterSession.  This keeps
        # startup/resume identity (parsed-entry content hash and source hash)
        # identical to the values captured in a checkpoint.
        from lechonk.adapter.manual_reference import load_manual_reference
        reference = load_manual_reference(os.path.dirname(STATIC_MANUAL_REFERENCE_PATH))
        status = getattr(reference, "status", None)
        schema = getattr(reference, "schema_version", None)
        digest = getattr(reference, "content_hash", None)
        source = getattr(reference, "source_sha256", None)
        if status not in ("available", "complete") or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (digest, source)):
            raise ValueError("manual loader reported unavailable")
        if not isinstance(schema, str) or not schema:
            raise ValueError("manual loader returned no schema")
        return {"present": True, "schema_version": schema, "hash": digest,
                "source_sha256": source, "status": "available"}
    except (OSError, ValueError, TypeError, ImportError):
        return {"present": False, "schema_version": None, "hash": None,
                "source_sha256": None, "status": "unavailable"}


STATIC_MANUAL_REFERENCE = _static_manual_reference_provenance()


def _active_prompt_hashes(prompts_module=None, *, bundle_id=None):
    """Return hashes for the selected prompt profile.

    New prompt implementations expose a profile-aware manifest function.  The
    fallback keeps this runner importable while that implementation is being
    rolled out and preserves the historical legacy hash set.
    """
    global _PROMPT_SNAPSHOT
    if prompts_module is None:
        from lechonk.model import prompts as prompts_module
    if (_PROMPT_SNAPSHOT is not None and
            _PROMPT_SNAPSHOT.profile == PROMPT_PROFILE and
            _PROMPT_SNAPSHOT.bundle_id == bundle_id):
        return _PROMPT_SNAPSHOT.manifest_hashes
    capture = getattr(prompts_module, "capture_prompt_snapshot", None)
    if capture is not None:
        _PROMPT_SNAPSHOT = capture(PROMPT_PROFILE, bundle_id=bundle_id)
        return _PROMPT_SNAPSHOT.manifest_hashes
    fn = getattr(prompts_module, "prompt_manifest_hashes")
    try:
        return fn(prompt_profile=PROMPT_PROFILE, bundle_id=bundle_id)
    except TypeError:
        try:
            return fn(PROMPT_PROFILE, bundle_id=bundle_id)
        except TypeError:
            hashes = fn()
            return hashes


def _active_system_prompt(prompts_module):
    """Render the selected system prompt, with legacy compatibility."""
    if (_PROMPT_SNAPSHOT is not None and
            _PROMPT_SNAPSHOT.profile == PROMPT_PROFILE and
            _PROMPT_SNAPSHOT.bundle_id == PROMPT_BUNDLE_ID):
        return prompts_module.system_prompt(
            PROMPT_PROFILE, bundle_id=PROMPT_BUNDLE_ID,
            snapshot=_PROMPT_SNAPSHOT)
    fn = prompts_module.system_prompt
    try:
        return fn(prompt_profile=PROMPT_PROFILE,
                  bundle_id=PROMPT_BUNDLE_ID)
    except TypeError:
        try:
            return fn(PROMPT_PROFILE, bundle_id=PROMPT_BUNDLE_ID)
        except TypeError:
            return fn()


def _active_prompt_hash_digest():
    """Compact stable identity for the selected prompt bundle."""
    return hashlib.sha256(json.dumps(
        _active_prompt_hashes(bundle_id=PROMPT_BUNDLE_ID), sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def recovery_config_digest():
    """Digest every model/prompt/game setting that must match on resume."""
    from lechonk.model import mistral_client as _mc
    from lechonk.model import prompts as _prompts
    if PROMPT_PROFILE in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} or RECOVERY_DIGEST_SCHEMA == "interactive-config-v3":
        from lechonk.core.recovery_identity import IdentityInput, digest_for, interactive_v3_spec
        identity = IdentityInput(
            context_policy=CONTEXT_POLICY, prompt_profile=PROMPT_PROFILE,
            model_id=_mc.MODEL_ID,
            prompt_hashes=_active_prompt_hashes(
                _prompts, bundle_id=PROMPT_BUNDLE_ID),
            ruleset="classic", seed=SEED, action_schema="ActionBatch-v1",
            classic_reference_present=STATIC_CLASSIC_REFERENCE["present"],
            classic_reference_schema_version=STATIC_CLASSIC_REFERENCE["schema_version"],
            classic_ruleset_tree_sha256=STATIC_CLASSIC_REFERENCE["tree_sha256"],
            manual_reference_present=STATIC_MANUAL_REFERENCE["present"],
            manual_reference_schema_version=STATIC_MANUAL_REFERENCE["schema_version"],
            manual_reference_hash=STATIC_MANUAL_REFERENCE["hash"],
            manual_reference_source_sha256=STATIC_MANUAL_REFERENCE["source_sha256"],
            aifill=AIFILL, configured_ai_opponents=AIFILL - 1,
            prompt_bundle_id=PROMPT_BUNDLE_ID,
            candidate_surface_policy=CANDIDATE_SURFACE_POLICY,
            candidate_surface_cap=(12000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else 8000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
            candidate_admission_version=("candidate_surface_v3-r1" if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else "candidate_surface_v2-r1" if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
            compaction_schema="conversation-compaction-v1",
            prompt_budget_schema=(f"{PROMPT_PROFILE.replace('_', '-')}-budget-v1"
                                  if CANDIDATE_SURFACE_POLICY in {"candidate_surface_v2", "candidate_surface_v3"}
                                  else "legacy-budget-v1"))
        return digest_for(interactive_v3_spec(), identity)
    payload = {
        "context_policy": CONTEXT_POLICY,
        "prompt_profile": PROMPT_PROFILE,
        "model_id": _mc.MODEL_ID,
        "prompt_hashes": _active_prompt_hashes(_prompts, bundle_id=PROMPT_BUNDLE_ID),
        "ruleset": "classic",
        "aifill": AIFILL,
        "configured_ai_opponents": AIFILL - 1,
        "seed": SEED,
        "action_schema": "ActionBatch-v1",
        "classic_reference_present": STATIC_CLASSIC_REFERENCE["present"],
        "classic_reference_schema_version": STATIC_CLASSIC_REFERENCE["schema_version"],
        "classic_ruleset_tree_sha256": STATIC_CLASSIC_REFERENCE["tree_sha256"],
        "manual_reference_present": STATIC_MANUAL_REFERENCE["present"],
        "manual_reference_schema_version": STATIC_MANUAL_REFERENCE["schema_version"],
        "manual_reference_hash": STATIC_MANUAL_REFERENCE["hash"],
        "manual_reference_source_sha256": STATIC_MANUAL_REFERENCE["source_sha256"],
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

# Validate the parent manifest before creating any child artifacts.  A
# continuation must never silently overwrite/resume its own lineage.
if RESUME_MANIFEST:
    from lechonk.core.recovery import load_checkpoint
    from lechonk.core.recovery_identity import (IdentityInput, digest_for,
                                                interactive_v3_spec,
                                                resolve_lineage)
    _parent_root = str(Path(RESUME_MANIFEST).resolve().parent)
    try:
        # Manifest and bound files are verified before any digest/lineage
        # decision.  This ordering prevents an untrusted sidecar from
        # influencing prompt or bundle selection.
        RESUME_CHECKPOINT = load_checkpoint(
            RESUME_MANIFEST, artifact_root=_parent_root,
            expected_config_digest=None,
            expected_config_digest_schema=None,
            expected_prompt_bundle_id=None,
            expected_classic_reference_schema_version=CLASSIC_RUNTIME_SCHEMA,
            expected_classic_ruleset_tree_sha256=(
                STATIC_CLASSIC_REFERENCE["tree_sha256"]),
            expected_classic_reference_present=(
                STATIC_CLASSIC_REFERENCE["present"]),
            expected_manual_reference_schema_version=(
                STATIC_MANUAL_REFERENCE["schema_version"]),
            expected_manual_reference_hash=STATIC_MANUAL_REFERENCE["hash"],
            expected_manual_reference_source_sha256=(
                STATIC_MANUAL_REFERENCE["source_sha256"]),
            expected_manual_reference_present=STATIC_MANUAL_REFERENCE["present"],
            verify_files=True)
    except Exception as exc:
        raise SystemExit(f"resume rejected: {exc}") from exc
    # The manifest checksum and bound-file checks above are authoritative for
    # the resumed namespace.  Do not derive a new name from the child run id.
    FREECIV_USERNAME = RESUME_CHECKPOINT.username
    if RESUME_CHECKPOINT.config_digest_schema == "interactive-config-v3":
        if RESUME_CHECKPOINT.prompt_bundle_id not in {
                "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6", "gameplay_v2_original",
                "gameplay_v2_turnid_fix"}:
            raise SystemExit("resume rejected: unknown prompt bundle")
        _bundle_profile = (RESUME_CHECKPOINT.prompt_bundle_id
                           if RESUME_CHECKPOINT.prompt_bundle_id in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
                           else "gameplay_v2")
        if PROMPT_PROFILE != _bundle_profile:
            raise SystemExit(
                "resume rejected: prompt profile conflicts with manifest bundle")
        PROMPT_BUNDLE_ID = RESUME_CHECKPOINT.prompt_bundle_id
        RECOVERY_DIGEST_SCHEMA = "interactive-config-v3"
        CANDIDATE_SURFACE_POLICY = (
            ("candidate_surface_v3" if PROMPT_BUNDLE_ID == "gameplay_v6" else "candidate_surface_v2") if PROMPT_BUNDLE_ID in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
            else "legacy_v1")
        identity = IdentityInput(
            context_policy=CONTEXT_POLICY, prompt_profile=PROMPT_PROFILE,
            model_id=__import__("lechonk.model.mistral_client",
                                fromlist=["MODEL_ID"]).MODEL_ID,
            prompt_hashes=_active_prompt_hashes(bundle_id=PROMPT_BUNDLE_ID),
            ruleset="classic",
            seed=SEED, action_schema="ActionBatch-v1",
            classic_reference_present=STATIC_CLASSIC_REFERENCE["present"],
            classic_reference_schema_version=STATIC_CLASSIC_REFERENCE["schema_version"],
            classic_ruleset_tree_sha256=STATIC_CLASSIC_REFERENCE["tree_sha256"],
            manual_reference_present=STATIC_MANUAL_REFERENCE["present"],
            manual_reference_schema_version=STATIC_MANUAL_REFERENCE["schema_version"],
            manual_reference_hash=STATIC_MANUAL_REFERENCE["hash"],
            manual_reference_source_sha256=STATIC_MANUAL_REFERENCE["source_sha256"],
            aifill=AIFILL, configured_ai_opponents=AIFILL - 1,
            prompt_bundle_id=PROMPT_BUNDLE_ID,
            candidate_surface_policy=CANDIDATE_SURFACE_POLICY,
            candidate_surface_cap=(12000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else 8000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
            candidate_admission_version=("candidate_surface_v3-r1" if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else "candidate_surface_v2-r1" if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
            compaction_schema="conversation-compaction-v1",
            prompt_budget_schema=(f"{PROMPT_PROFILE.replace('_', '-')}-budget-v1"
                                  if CANDIDATE_SURFACE_POLICY in {"candidate_surface_v2", "candidate_surface_v3"}
                                  else "legacy-budget-v1"))
        try:
            resolve_lineage(RESUME_CHECKPOINT, identity,
                            (interactive_v3_spec(),))
        except Exception as exc:
            raise SystemExit(f"resume rejected: {exc}") from exc
        if RESUME_CHECKPOINT.prompt_bundle_id != PROMPT_BUNDLE_ID:
            raise SystemExit("resume rejected: prompt bundle mismatch")
    elif RESUME_CHECKPOINT.config_digest_schema is None:
        if PROMPT_PROFILE != "gameplay_v2":
            raise SystemExit("resume rejected: legacy checkpoint requires --prompt-profile gameplay_v2")
        from lechonk.model import prompts as _legacy_prompts
        from lechonk.core.recovery_identity import IdentityInput, legacy_specs
        _legacy_matches = []
        for _bundle_id in ("gameplay_v2_original", "gameplay_v2_turnid_fix"):
            _hashes = _active_prompt_hashes(_legacy_prompts, bundle_id=_bundle_id)
            _identity = IdentityInput(
                context_policy=CONTEXT_POLICY, prompt_profile="gameplay_v2",
                model_id=__import__("lechonk.model.mistral_client", fromlist=["MODEL_ID"]).MODEL_ID,
                prompt_hashes=_hashes, ruleset="classic", seed=SEED,
                action_schema="ActionBatch-v1",
                classic_reference_present=STATIC_CLASSIC_REFERENCE["present"],
                classic_reference_schema_version=STATIC_CLASSIC_REFERENCE["schema_version"],
                classic_ruleset_tree_sha256=STATIC_CLASSIC_REFERENCE["tree_sha256"],
                manual_reference_present=STATIC_MANUAL_REFERENCE["present"],
                manual_reference_schema_version=STATIC_MANUAL_REFERENCE["schema_version"],
                manual_reference_hash=STATIC_MANUAL_REFERENCE["hash"],
                manual_reference_source_sha256=STATIC_MANUAL_REFERENCE["source_sha256"],
                aifill=AIFILL, configured_ai_opponents=AIFILL - 1,
                prompt_bundle_id=_bundle_id)
            _spec = legacy_specs(original_prompt_hashes=_hashes if _bundle_id == "gameplay_v2_original" else _active_prompt_hashes(_legacy_prompts, bundle_id="gameplay_v2_original"),
                                 corrected_prompt_hashes=_hashes if _bundle_id == "gameplay_v2_turnid_fix" else _active_prompt_hashes(_legacy_prompts, bundle_id="gameplay_v2_turnid_fix"))
            _chosen = _spec[0 if _bundle_id == "gameplay_v2_original" else 1]
            try:
                from lechonk.core.recovery_identity import digest_for
                if digest_for(_chosen, _identity) == RESUME_CHECKPOINT.config_digest:
                    _legacy_matches.append((_bundle_id, _chosen))
            except Exception:
                pass
        if len(_legacy_matches) != 1:
            raise SystemExit("resume rejected: legacy checkpoint does not match exactly one frozen lineage")
        PROMPT_BUNDLE_ID = _legacy_matches[0][0]
        RECOVERY_DIGEST_SCHEMA = "interactive-config-v3"
        CANDIDATE_SURFACE_POLICY = "legacy_v1"
    else:
        raise SystemExit(
            f"resume rejected: unsupported config digest schema "
            f"{RESUME_CHECKPOINT.config_digest_schema!r}")
    if RESUME_CHECKPOINT.run_id == RUN:
        raise SystemExit("resume rejected: --run-id must differ from parent run")
    if os.path.exists(RUNDIR) and ATTEMPT <= 1:
        raise SystemExit(
            f"resume rejected: target run directory already exists: {RUNDIR} "
            "(use a new --run-id or explicit retry attempt)")

# Fresh runs use the showcase name.  Resumes adopt the checkpoint's pinned
# display name when present; legacy checkpoints have no such field and retain
# their historical behavior unless the caller explicitly supplies a name.
try:
    PLAYER_NAME = resolve_player_name(
        PLAYER_NAME,
        RESUME_CHECKPOINT.player_name if RESUME_CHECKPOINT is not None else None,
        fresh=RESUME_CHECKPOINT is None)
except ValueError as exc:
    raise SystemExit(f"resume rejected: {exc}") from exc

# A first attempt may reuse an empty directory created by provisioning, but it
# must never silently delete a user's prior run artifacts. Retries are the only
# process allowed to rotate an existing artifact set.
try:
    RUN_TOKEN = claim_run_directory(
        RUNDIR, attempt=ATTEMPT, resuming=RESUME_MANIFEST is not None,
        token=os.environ.get("LECHONK_RUN_TOKEN"))
    os.environ["LECHONK_RUN_TOKEN"] = RUN_TOKEN
    prepare_retry_artifacts(RUNDIR, attempt=ATTEMPT)
except ArtifactCollisionError as exc:
    raise SystemExit(f"{exc} (use a new --run-id or explicit retry attempt)") from exc

os.makedirs(WORKDIR, exist_ok=True)

# Decision-8 manifest constants (verified this session / in the plan).
IMAGE_DIGEST = ("sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a"
                "8e943faf920cb373a")  # verified against the running
# container (docker inspect freeciv-web) 2026-10-24; the digest pin (decision 3).
VENDORED_BASE_SHA = "1b144d6ec92a8aaa8e30eee64c881418208a4689"  # upstream dev
PATCH_HEAD_SHA = "36ae456836015b22b086298430048fc51c317d50"  # PR #7 fork head
CAPSTRING = "+Freeciv.Web.Devel-3.3"  # plan decision 1 (two primary sources)
SPIKE_ARTIFACT = os.path.join(
    REPO, "artifacts", "runs", "wp2-spike", "fingerprint_probe.json")
COMPARATOR_RUN_META = os.path.join(
    REPO, "artifacts", "runs", "wp1-comparator-amended", "run_meta.json")

LOG_PATH = os.path.join(RUNDIR, "run_stdout.log")
TRACE_PATH = None
_log_file = open(LOG_PATH, "a", encoding="utf-8")
_LOG_LOCK = threading.RLock()
_STDOUT_BROKEN = False
_MODEL_HEARTBEAT = None
_MODEL_HEARTBEAT_GENERATION = None
_MODEL_HEARTBEAT_CALL_ID = None
_MODEL_CALL_EVENTS_ENABLED = False
_MODEL_REPETITION = None
_CAPTURE_GATEWAY = None
_CAPTURE_SYNC_TURN = None
_CAPTURE_SYNC_WAIT_USED_MS = 0.0
_CAPTURE_SYNC_ROUTINE_CHECKPOINTED = False


def log(msg):
    """Print to stdout AND persist to the run log (HANDOFF: persist run
    stdout/logs as artifacts under artifacts/runs/)."""
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    global _STDOUT_BROKEN
    with _LOG_LOCK:
        if not _STDOUT_BROKEN:
            try:
                print(line, flush=True)
            except (BrokenPipeError, OSError):
                # A closed terminal must not strand the heartbeat worker while
                # it is trying to report progress. The durable run log remains
                # the source of truth when stdout is unavailable.
                _STDOUT_BROKEN = True
        _log_file.write(line + "\n")
        _log_file.flush()


def _atomic_json(path, obj):
    """Durably replace a small startup artifact without partial JSON."""
    fd, tmp_path = tempfile.mkstemp(prefix=".ruleset-", dir=RUNDIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True,
                      ensure_ascii=False, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_STREAM_OUTPUT_SEEN = False
_OBSERVER_URL_PRINTED = False
_OBSERVER_OPEN_ATTEMPTED = False

# The decision-8 manifest is emitted by core.runner before the session starts,
# so its initial pingtimeout entry can only describe the requested setting.
# Keep the post-start read-back separately and use it in all post-run reports.
_SESSION_RUNTIME = {
    "pingtimeout_status": "pending_start",
    "pingtimeout_actual": None,
    "ruleset_reference_hash": None,
    "ruleset_reference_status": "pending_start",
}


def _pingtimeout_metadata():
    """Return truthful requested/observed pingtimeout metadata.

    AdapterSession supplies the verified values; old/fake sessions may not,
    in which case we explicitly preserve an unknown status rather than
    claiming either success or failure.
    """
    status = _SESSION_RUNTIME.get("pingtimeout_status") or "unknown"
    actual = _SESSION_RUNTIME.get("pingtimeout_actual")
    if status == "confirmed" and actual is not None:
        effective = f"verified server value {float(actual):g}"
    elif status == "pending_start":
        effective = "pending session start verification"
    else:
        effective = f"server readback status: {status}"
    return {
        "requested": PINGTIMEOUT_CONFIGURED_S,
        "status": status,
        "actual": actual,
        "effective": effective,
    }


def _record_session_runtime(event):
    """Capture optional session observability fields from a progress event."""
    if event.get("event") != "session_ready":
        return
    if "pingtimeout_status" in event:
        _SESSION_RUNTIME["pingtimeout_status"] = event["pingtimeout_status"]
    if "pingtimeout_actual" in event:
        _SESSION_RUNTIME["pingtimeout_actual"] = event["pingtimeout_actual"]
    for key in ("ruleset_reference_hash", "ruleset_reference_status"):
        if key in event:
            _SESSION_RUNTIME[key] = event[key]
    reference = event.get("ruleset_reference_json")
    if reference is not None:
        # Persist the exact startup snapshot as soon as the lifecycle callback
        # arrives, before the first model call.  This is an audit artifact,
        # not prompt context and never contains mutable game state.
        try:
            _atomic_json(os.path.join(RUNDIR, "ruleset_reference.json"),
                         reference)
            _atomic_json(os.path.join(RUNDIR, "ruleset_reference_meta.json"), {
                    "schema": "ruleset_reference_meta_v1",
                    "hash": _SESSION_RUNTIME["ruleset_reference_hash"],
                    "status": _SESSION_RUNTIME["ruleset_reference_status"],
                    "source": "AdapterSession startup snapshot",
                    "mutable_game_state_included": False,
                })
        except OSError as exc:
            log(f"[{RUN}] could not persist ruleset reference: {exc!r}")
    elif event.get("event") == "session_ready":
        # An unavailable optional reference is still explicit run metadata;
        # absence must never look like an unrecorded startup.
        try:
            _atomic_json(os.path.join(RUNDIR, "ruleset_reference_meta.json"), {
                    "schema": "ruleset_reference_meta_v1",
                    "hash": _SESSION_RUNTIME["ruleset_reference_hash"],
                    "status": _SESSION_RUNTIME["ruleset_reference_status"],
                    "source": "AdapterSession startup snapshot",
                    "mutable_game_state_included": False,
                })
        except OSError as exc:
            log(f"[{RUN}] could not persist ruleset metadata: {exc!r}")


def _persist_ruleset_reference_from_session(session):
    """Backfill the startup artifact for older runner event payloads."""
    if os.path.exists(os.path.join(RUNDIR, "ruleset_reference.json")):
        return
    reference = getattr(session, "ruleset_reference_json", None)
    if callable(reference):
        reference = reference()
    if reference is None:
        return
    digest = getattr(session, "ruleset_reference_hash", None)
    if callable(digest):
        digest = digest()
    status = getattr(session, "ruleset_reference_status", "available")
    if callable(status):
        status = status()
    _record_session_runtime({
        "event": "session_ready",
        "ruleset_reference_json": reference,
        "ruleset_reference_hash": digest,
        "ruleset_reference_status": status,
    })


def _safe_progress_text(value):
    """Keep arbitrary event text single-line and terminal-safe."""
    text = _ANSI_ESCAPE.sub("", str(value))
    text = _CONTROL_CHARS.sub(" ", text)
    text = " ".join(text.split())
    return text if len(text) <= 240 else text[:237] + "..."


def _safe_trace_text(value):
    """Remove terminal control sequences while preserving complete output."""
    text = _ANSI_ESCAPE.sub("", str(value))
    # Keep newlines and tabs so a reasoning trace remains readable. Remove
    # other C0/C1 controls, which could otherwise manipulate the terminal.
    # Strip carriage returns as well as other terminal controls: CR can move
    # the cursor back to column zero and let model output overwrite the log.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f-\x9f]", "", text)
    return text


def _pretty_model_answer(value):
    """Pretty-print a JSON answer without changing or truncating its fields."""
    text = _safe_trace_text(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    return json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)


def _render_model_output(event):
    """Render the complete returned reasoning and answer after one API call."""
    # Keep an unabridged, machine-readable copy alongside the human log. This
    # is deliberately separate from api_calls.jsonl, which must remain free of
    # prompt/response bodies for provenance and size reasons.
    global _STREAM_OUTPUT_SEEN
    streamed_this_call = _STREAM_OUTPUT_SEEN
    try:
        trace_event = dict(event)
        # A transport error can leave a partial reasoning trace. Mark it
        # explicitly so consumers never mistake it for a complete answer.
        trace_event["complete"] = event.get("error") is None
        with open(TRACE_PATH, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(trace_event, ensure_ascii=False,
                                        default=str) + "\n")
            trace_file.flush()
    except Exception as exc:
        log(f"[{RUN}] [model output] could not persist trace: {exc!r}")
    turn = _safe_progress_text(event.get("turn", "?"))
    attempt = _safe_progress_text(event.get("attempt", "?"))
    # Streaming has already printed every reasoning/final delta. Keep the
    # durable complete trace, but avoid printing the same response twice.
    if streamed_this_call:
        status = "partial (stream ended with error)" if event.get("error") else "complete"
        log(f"[{RUN}] [model output turn={turn} attempt={attempt}] "
            f"{status} (streamed above)")
        _STREAM_OUTPUT_SEEN = False
        return
    log(f"[{RUN}] [model output turn={turn} attempt={attempt}] BEGIN")
    reasoning = event.get("reasoning_text", event.get("reasoning", ""))
    answer = event.get("answer_text", event.get("answer", event.get("output", "")))
    if reasoning:
        log(f"[{RUN}] [model reasoning turn={turn} attempt={attempt}]\n"
            + _safe_trace_text(reasoning))
    else:
        log(f"[{RUN}] [model reasoning turn={turn} attempt={attempt}] (none returned)")
    log(f"[{RUN}] [model answer turn={turn} attempt={attempt}]\n"
        + _pretty_model_answer(answer))
    tool_calls = event.get("tool_calls")
    if tool_calls:
        rendered = _safe_trace_text(json.dumps(
            tool_calls, indent=2, ensure_ascii=False, default=str))
        if len(rendered) > 4000:
            rendered = rendered[:4000] + "\n[tool call display truncated]"
        log(f"[{RUN}] [model tool calls turn={turn} attempt={attempt}]\n"
            + rendered)
    chunks = event.get("content_chunks")
    if chunks:
        # ThinkChunk/TextChunk data is already rendered above. Print any
        # additional chunk types verbatim so SDK extensions cannot disappear
        # from the human view, while the JSONL trace retains every chunk.
        extras = []
        for chunk in chunks if isinstance(chunks, list) else [chunks]:
            chunk_type = (chunk.get("type") if isinstance(chunk, dict)
                          else getattr(chunk, "type", None))
            if chunk_type in {"thinking", "text"}:
                continue
            extras.append(chunk)
        if extras:
            log(f"[{RUN}] [model extra chunks turn={turn} attempt={attempt}]\n"
                + _safe_trace_text(json.dumps(
                    extras, indent=2, ensure_ascii=False, default=str)))
    log(f"[{RUN}] [model output turn={turn} attempt={attempt}] END")


_CAPTURE_SYNC_MILESTONE_WORDS = frozenset({
    "attack", "attacked", "bombard", "combat", "conquer", "conquest",
    "diplomacy", "diplomatic", "research", "wonder", "build_city",
    "found_city", "founding", "settle",
})
_CAPTURE_SYNC_NON_MODEL_PROVENANCE = frozenset({
    "engine_automatic", "engine_automatic_action", "automatic", "system",
    "observer", "ai_automatic",
})


def _capture_sync_action_kind(event):
    """Return (is_milestone, is_routine_move) for a bounded checkpoint."""
    if not isinstance(event, dict) or event.get("event") != "action_result":
        return False, False
    provenance = str(event.get("provenance_class") or "").lower()
    if provenance in _CAPTURE_SYNC_NON_MODEL_PROVENANCE:
        return False, False
    fields = []
    # Only action identity fields determine salience.  Target/readback text
    # can mention a city or research while describing an engine-generated
    # side effect and must not turn that side effect into a checkpoint.
    for key in ("domain", "verb", "action", "action_ref"):
        value = event.get(key)
        if value is not None:
            fields.append(str(value).lower())
    text = " ".join(fields)
    milestone = any(word in text for word in _CAPTURE_SYNC_MILESTONE_WORDS)
    routine = any(word in text for word in ("goto", "move", "explore"))
    return milestone, routine


def _capture_sync_wait(event):
    """Request and bounded-wait for one durable presenter checkpoint.

    This is deliberately opt-in and fail-open.  An acknowledgement proves
    that the PNG and image metadata were written; it does not prove that the
    requested game consequence is visible in the pixels.
    """
    global _CAPTURE_SYNC_TURN, _CAPTURE_SYNC_WAIT_USED_MS
    global _CAPTURE_SYNC_ROUTINE_CHECKPOINTED
    if not CAPTURE_SYNC or _CAPTURE_GATEWAY is None:
        return False
    milestone, routine = _capture_sync_action_kind(event)
    if not milestone and not (routine and not _CAPTURE_SYNC_ROUTINE_CHECKPOINTED):
        return False
    # Most action records carry their turn.  If a producer omits it, retain
    # the last explicit turn_started/turn_decision boundary rather than
    # creating a new budget bucket for every event.
    turn_from_event = event.get("turn") is not None
    turn = str(event.get("turn") or _CAPTURE_SYNC_TURN or "?")
    turn_resolution = ("event" if turn_from_event else
                       ("boundary" if _CAPTURE_SYNC_TURN is not None
                        else "unknown"))
    if turn != _CAPTURE_SYNC_TURN:
        _CAPTURE_SYNC_TURN = turn
        _CAPTURE_SYNC_WAIT_USED_MS = 0.0
        _CAPTURE_SYNC_ROUTINE_CHECKPOINTED = False
    if routine and not milestone:
        _CAPTURE_SYNC_ROUTINE_CHECKPOINTED = True
    # Count only time spent waiting for the presenter. Model/API latency is
    # outside this budget and must not suppress the action checkpoint.
    wait_used_before_ms = _CAPTURE_SYNC_WAIT_USED_MS
    remaining_ms = CAPTURE_SYNC_BUDGET_MS - wait_used_before_ms
    if remaining_ms <= 0:
        result = "budget_exhausted"
        seq = None
    else:
        checkpoint = dict(event)
        checkpoint["capture_checkpoint"] = True
        checkpoint["capture_checkpoint_turn"] = turn
        seq = _CAPTURE_GATEWAY.offer_control_receipt("progress", checkpoint)
        if seq is None:
            result = "gateway_full"
        else:
            ack_path = (Path(_CAPTURE_GATEWAY.capture_dir) / "checkpoint-acks"
                        / f"{seq}.json")
            deadline = time.monotonic() + min(
                CAPTURE_SYNC_TIMEOUT_MS, max(0.0, remaining_ms)) / 1000
            wait_started = time.monotonic()
            result = "timeout"
            while time.monotonic() < deadline:
                try:
                    with open(ack_path, encoding="utf-8") as fh:
                        ack = json.load(fh)
                    if (str(ack.get("run_id")) == str(RUN)
                            and str(ack.get("capture_id")) == str(
                                _CAPTURE_GATEWAY.capture_id)
                            and int(ack.get("capture_seq")) == int(seq)):
                        result = ("captured" if ack.get("status") == "captured"
                                  else "failed")
                        break
                except (FileNotFoundError, ValueError, TypeError,
                        OSError, json.JSONDecodeError):
                    pass
                time.sleep(0.02)
            _CAPTURE_SYNC_WAIT_USED_MS += (
                time.monotonic() - wait_started) * 1000
    payload = {
        "capture_sync_result": result,
        "checkpoint_capture_seq": seq,
        "turn": event.get("turn"),
        "action_ref": event.get("action_ref"),
        "milestone": milestone,
        "routine": routine,
        "wait_budget_ms": CAPTURE_SYNC_BUDGET_MS,
        "wait_used_ms": round(_CAPTURE_SYNC_WAIT_USED_MS, 3),
        "wait_remaining_ms": round(
            max(0.0, CAPTURE_SYNC_BUDGET_MS - _CAPTURE_SYNC_WAIT_USED_MS), 3),
        "turn_resolution": turn_resolution,
        "turn_uncertainty": turn_resolution != "event",
    }
    try:
        _CAPTURE_GATEWAY.offer_control("capture_sync_result", payload)
    except Exception:
        pass
    return seq is not None
    _STREAM_OUTPUT_SEEN = False


def _render_model_output_delta(event):
    """Print one response delta immediately, preserving embedded newlines."""
    global _STREAM_OUTPUT_SEEN
    reasoning = _safe_trace_text(event.get("reasoning", ""))
    answer = _safe_trace_text(event.get("answer", ""))
    if not reasoning and not answer:
        return
    _STREAM_OUTPUT_SEEN = True
    if reasoning:
        log(f"[{RUN}] [model reasoning delta]\n{reasoning}")
    if answer:
        log(f"[{RUN}] [model answer delta]\n{answer}")


def _observer_url(port):
    """Build the read-only webclient URL for a validated game port."""
    if isinstance(port, bool):
        return None
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        return None
    if not 1 <= port_number <= 65535 or str(port_number) != str(port).strip():
        return None
    return ("http://localhost:8080/webclient/?action=observe&"
            f"civserverport={port_number}&civserverhost=unknown&"
            "multi=true&type=multiplayer")


def _observer_is_headless():
    """Return true when no ordinary graphical browser display is available."""
    return not any(os.environ.get(name) for name in
                   ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET", "BROWSER"))


def _handle_session_ready_observer(event):
    """Print and optionally open the observer URL once per harness run.

    Browser failures are deliberately informational: the game and model run
    must continue, and the URL remains available for manual use.
    """
    global _OBSERVER_URL_PRINTED, _OBSERVER_OPEN_ATTEMPTED
    if event.get("event") != "session_ready" or _OBSERVER_URL_PRINTED:
        return
    _OBSERVER_URL_PRINTED = True
    url = _observer_url(event.get("port"))
    if url is None:
        log(f"[{RUN}] [observer] " + json.dumps({
            "status": "unavailable", "reason": "session_ready had no valid port",
            "open_requested": bool(OPEN_OBSERVER)}, sort_keys=True))
        return
    log(f"[{RUN}] [progress] observer URL: {url}")
    if not OPEN_OBSERVER:
        log(f"[{RUN}] [observer] " + json.dumps({
            "status": "ready", "url": url, "open_requested": False},
            sort_keys=True))
        return
    if _OBSERVER_OPEN_ATTEMPTED:
        return
    _OBSERVER_OPEN_ATTEMPTED = True
    if _observer_is_headless():
        log(f"[{RUN}] [observer] " + json.dumps({
            "status": "headless", "url": url, "open_requested": True,
            "warning": "no graphical browser display detected"}, sort_keys=True))
        return
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
        status = "opened" if opened else "open_failed"
        record = {"status": status, "url": url, "open_requested": True}
        if not opened:
            record["warning"] = "webbrowser.open returned false"
    except Exception as exc:  # browser integration is strictly non-fatal
        record = {"status": "open_failed", "url": url,
                  "open_requested": True,
                  "warning": f"{type(exc).__name__}: {exc}"}
    log(f"[{RUN}] [observer] " + json.dumps(record, sort_keys=True))


def _heartbeat_sink(event):
    """Forward metadata-only heartbeat records through the normal renderer."""
    # A worker can already be inside its sink when a repair replaces it. Do
    # not let that stale record appear as current-run progress.
    if event.get("generation") != _MODEL_HEARTBEAT_GENERATION:
        return
    progress({"event": event.get("kind", "model_call_heartbeat"), **event})


def _start_model_heartbeat(event):
    """Start the watchdog for one logical attempt and return its generation."""
    global _MODEL_HEARTBEAT, _MODEL_HEARTBEAT_GENERATION
    if _MODEL_HEARTBEAT is None:
        _MODEL_HEARTBEAT = ModelCallHeartbeat(
            interval_s=10.0, sink=_heartbeat_sink, failure_sink=_heartbeat_sink)
    turn = event.get("turn", "?")
    attempt = event.get("attempt", 1)
    call_id = event.get("call_id") or f"{RUN}/t{turn}/call{attempt}"
    _MODEL_HEARTBEAT_GENERATION = _MODEL_HEARTBEAT.start(
        run_id=str(event.get("run_id") or RUN), turn=turn,
        call_id=str(call_id), attempt=attempt)


def _stop_model_heartbeat():
    """Stop the active watchdog; safe to call from normal and error paths."""
    global _MODEL_HEARTBEAT_GENERATION, _MODEL_HEARTBEAT_CALL_ID
    if _MODEL_HEARTBEAT is not None and _MODEL_HEARTBEAT_GENERATION is not None:
        _MODEL_HEARTBEAT.stop(_MODEL_HEARTBEAT_GENERATION)
    _MODEL_HEARTBEAT_GENERATION = None
    _MODEL_HEARTBEAT_CALL_ID = None


def _repetition_warning(event):
    """Render detector metadata without including model text."""
    progress(event)


def _start_model_repetition(event):
    global _MODEL_REPETITION
    if _MODEL_REPETITION is None:
        _MODEL_REPETITION = RepetitionDetector(warning_sink=_repetition_warning)
    _MODEL_REPETITION.reset(
        call_id=str(event.get("call_id")), turn=event.get("turn"),
        attempt=event.get("attempt"), run_id=str(event.get("run_id") or RUN))


def _stop_model_repetition():
    global _MODEL_REPETITION
    if _MODEL_REPETITION is not None:
        _MODEL_REPETITION.reset()


def _model_call_event(event):
    """Correlate heartbeat ownership to one physical provider request.

    Runner ``model_call_*`` events are logical and arrive too late for tool
    calls/compaction.  These local client events are metadata-only; raw model
    traces continue through their existing callbacks unchanged.
    """
    global _MODEL_HEARTBEAT_GENERATION, _MODEL_HEARTBEAT_CALL_ID
    if not isinstance(event, dict):
        return
    # This is the physical provider lifecycle boundary.  Capture before any
    # heartbeat/repetition handling so the nested raw delta remains complete
    # and is not replaced by the human renderer's compact representation.
    if _CAPTURE_GATEWAY is not None:
        try:
            physical_id = event.get("physical_call_id") or event.get("call_id")
            logical_id = event.get("logical_call_id") or event.get("logical_call")
            raw_delta = event.get("delta") if "delta" in event else None
            payload = {
                "physical_call_id": physical_id,
                "logical_call_id": logical_id,
                "raw_delta": raw_delta,
                "event": event,
            }
            # Mirror common display fields at payload top level while keeping
            # the complete, nested raw delta and original callback intact.
            if isinstance(raw_delta, dict):
                for key in ("reasoning", "answer", "tool", "text", "message"):
                    if key in raw_delta:
                        payload[key] = raw_delta[key]
            _CAPTURE_GATEWAY.offer_trace(
                str(event.get("event") or "model_lifecycle"), payload)
        except Exception:
            pass
    call_id = event.get("call_id")
    if not call_id:
        return
    kind = event.get("event")
    if kind == "model_request_started":
        _start_model_repetition(event)
        if _MODEL_HEARTBEAT is None:
            _start_model_heartbeat(event)
        else:
            if (_MODEL_HEARTBEAT_CALL_ID is not None and
                    _MODEL_HEARTBEAT_GENERATION is not None):
                _MODEL_HEARTBEAT.stop(_MODEL_HEARTBEAT_GENERATION)
            _MODEL_HEARTBEAT_GENERATION = _MODEL_HEARTBEAT.start(
                run_id=str(event.get("run_id") or RUN),
                turn=event.get("turn", "?"), call_id=str(call_id),
                attempt=event.get("attempt", 0))
        _MODEL_HEARTBEAT_CALL_ID = str(call_id)
    elif kind == "model_request_delta":
        if (str(call_id) == _MODEL_HEARTBEAT_CALL_ID and
                _MODEL_HEARTBEAT_GENERATION is not None and
                _MODEL_HEARTBEAT is not None):
            _MODEL_HEARTBEAT.note_stream_activity(_MODEL_HEARTBEAT_GENERATION)
        if (_MODEL_REPETITION is not None and
                str(call_id) == _MODEL_REPETITION.call_id):
            delta = event.get("delta")
            if not isinstance(delta, dict):
                delta = event
            _MODEL_REPETITION.observe(
                reasoning=delta.get("reasoning", ""),
                answer=delta.get("answer", ""))
    elif kind == "model_request_finished":
        if (str(call_id) == _MODEL_HEARTBEAT_CALL_ID and
                _MODEL_HEARTBEAT_GENERATION is not None):
            _stop_model_heartbeat()
        if (_MODEL_REPETITION is not None and
                str(call_id) == _MODEL_REPETITION.call_id):
            _stop_model_repetition()
    # Keep request lifecycle visible as structured, metadata-labelled output;
    # this does not enter model_traces.jsonl or alter raw stream callbacks.
    progress(event)


def progress(event):
    """Render the runner's safe, structured progress events for humans.

    Model output events are rendered in full, while ordinary progress metadata
    remains compact. Prompts and API keys are never included in these events.
    """
    if not isinstance(event, dict):
        log(f"[{RUN}] [progress] {_safe_progress_text(event)}")
        return
    global _CAPTURE_SYNC_TURN, _CAPTURE_SYNC_WAIT_USED_MS
    global _CAPTURE_SYNC_ROUTINE_CHECKPOINTED
    if CAPTURE_SYNC and event.get("event") in {"turn_started", "turn_decision"}:
        boundary_turn = event.get("turn")
        if boundary_turn is not None:
            boundary_turn = str(boundary_turn)
            if boundary_turn != _CAPTURE_SYNC_TURN:
                _CAPTURE_SYNC_TURN = boundary_turn
                _CAPTURE_SYNC_WAIT_USED_MS = 0.0
                _CAPTURE_SYNC_ROUTINE_CHECKPOINTED = False
    if _CAPTURE_GATEWAY is not None:
        try:
            event_name = event.get("event", "update")
            if event_name == "session_ready":
                _CAPTURE_GATEWAY.session_ready(event)
            elif event_name in {
                    "model_request_started", "model_request_delta",
                    "model_request_finished"}:
                # Already emitted at the physical callback boundary.
                pass
            elif event_name in {"model_output_delta", "model_output"}:
                _CAPTURE_GATEWAY.offer_trace("model_output", event)
            else:
                checkpoint_enqueued = (
                    _capture_sync_wait(event)
                    if event_name == "action_result" else False)
                if not checkpoint_enqueued:
                    _CAPTURE_GATEWAY.offer_control("progress", event)
        except Exception:
            # Capture is strictly observational and must never perturb the run.
            pass
    _record_session_runtime(event)
    name = _safe_progress_text(event.get("event", "update"))
    # Logical runner events remain informational once physical lifecycle
    # events are wired. Keep the legacy fallback for offline callers that
    # have not installed the client callback yet.
    if (name == "model_call_started" and
            not _MODEL_CALL_EVENTS_ENABLED):
        _start_model_heartbeat(event)
    elif (name == "model_output_delta" and
          not _MODEL_CALL_EVENTS_ENABLED and
          _MODEL_HEARTBEAT is not None and
          _MODEL_HEARTBEAT_GENERATION is not None):
        _MODEL_HEARTBEAT.note_stream_activity(_MODEL_HEARTBEAT_GENERATION)
    if name == "session_ready":
        _handle_session_ready_observer(event)
    if name == "model_output_delta":
        _render_model_output_delta(event)
        return
    if name == "model_output":
        _render_model_output(event)
        return
    if name in {"model_call_heartbeat", "model_call_heartbeat_failure"}:
        fields = {key: value for key, value in event.items()
                  if key != "event" and key != "kind" and value is not None}
        log(f"[{RUN}] [progress] {name} "
            + json.dumps(fields, sort_keys=True, default=str))
        return
    if name == "model_call_finished" and not _MODEL_CALL_EVENTS_ENABLED:
        _stop_model_heartbeat()
    if name == "ruleset_lookup_finished" and event.get("result_preview"):
        preview = _safe_trace_text(str(event["result_preview"]))
        log(f"[{RUN}] [ruleset lookup result] {preview}")
    fields = []
    for key, value in event.items():
        if key == "event" or value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True, default=str)
        fields.append(f"{_safe_progress_text(key)}={_safe_progress_text(value)}")
    log(f"[{RUN}] [progress] {name}" + (" " + " ".join(fields) if fields else ""))


def write_json(name, obj):
    if name == "recovery.json":
        raise RuntimeError(
            "recovery.json is reserved for the checkpoint manifest; "
            "write run summaries to recovery-report.json")
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
        try:
            prepare_attempt_artifact(path, attempt=ATTEMPT)
        except ArtifactCollisionError:
            raise
        if ATTEMPT > 1:
            log(f"preserved the previous attempt's {name} as {name}.attempt1")
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


def build_manifest(arm_start_ts, spike_fingerprint, live_probe,
                   resume_checkpoint=None):
    from lechonk.core.schemas import RunManifest
    from lechonk.model import mistral_client as mc
    from lechonk.model import prompts
    from lechonk.model.sampling import actual_client_sampling

    decision = mc.spike_decision(
        live_probe_done=True,
        json_schema_supported=True,
        evidence_artifact="artifacts/runs/wp2-spike/spike_report.json")
    fc_args = {
        "self_play": False, "aifill": AIFILL,
        "player_name": PLAYER_NAME,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": SEED, "debug.gameseed": SEED,
        "debug.agentseed": SEED,
        "server_timeout": SERVER_TIMEOUT_EFFECTIVE_S,
        "begin_turn_timeout": 900,
        "wait_for_timeout": 60, "http_timeout_s": 15.0,
        "max_turns": 100000 if UNTIL_GAME_OVER else TURNS + 1,
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
            "phasemode": "player", "timeout": 0,
            "pingtimeout": int(PINGTIMEOUT_CONFIGURED_S),
            "pingtimeout_override_requested": PINGTIMEOUT_CONFIGURED_S,
            # The first ledger record is emitted before session startup, so
            # post-start verification is intentionally pending here.
            "pingtimeout_status": "pending_start",
            "pingtimeout_actual": None,
            "pingtimeout_effective": "pending session start verification",
            "autotoggle": "disabled",
            "victories": "SPACERACE|ALLIED (CivRealm default)",
            "aifill": AIFILL, "configured_ai_opponents": AIFILL - 1,
            "player_name": PLAYER_NAME,
            "self_play": False,
        },
        model_id=mc.MODEL_ID,
        # W1-6 sampling manifest truth: the ACTUAL client sampling
        # (temperature 1.0, configured reasoning effort; the client never
        # sends top_p) — never the fabricated greedy dict.
        # Keep the shared actual_client_sampling() helper as the single
        # manifest source of truth while recording this run's override.
        sampling=actual_client_sampling(reasoning_effort=REASONING_EFFORT),
        prompt_hashes=_active_prompt_hashes(prompts, bundle_id=PROMPT_BUNDLE_ID),
        config_after_load={
            "freeciv_username": FREECIV_USERNAME,
            "fresh_reset": resume_checkpoint is None,
            "checkpoint_loaded": resume_checkpoint is not None,
            "fc_args": fc_args,
            "api_timeout_s": API_TIMEOUT_S,
            "acting_deadline_s": ACTING_DEADLINE_S,
            "reasoning_effort": REASONING_EFFORT,
            "max_turns": TURNS,
            "until_game_over": UNTIL_GAME_OVER,
            "aifill": AIFILL,
            "configured_ai_opponents": AIFILL - 1,
            "player_name": PLAYER_NAME,
            "context_policy": CONTEXT_POLICY,
            "prompt_profile": PROMPT_PROFILE,
            "static_manual_reference": {
                "present": STATIC_MANUAL_REFERENCE["present"],
                "status": STATIC_MANUAL_REFERENCE["status"],
                "schema_version": STATIC_MANUAL_REFERENCE["schema_version"],
                "hash": STATIC_MANUAL_REFERENCE["hash"],
                "source_sha256": STATIC_MANUAL_REFERENCE["source_sha256"],
            },
            "conversation_compaction_limit_tokens": (
                CONVERSATION_COMPACTION_LIMIT_TOKENS
                if CONTEXT_POLICY == "conversation_v1" else None),
            "conversation_tail_target_tokens": (
                CONVERSATION_TAIL_TARGET_TOKENS
                if CONTEXT_POLICY == "conversation_v1" else None),
            "context_journal_path": CONTEXT_JOURNAL_PATH,
            "conversation_state_path": CONVERSATION_STATE_PATH,
            "note": ("continuation from a verified start-of-turn checkpoint"
                     if resume_checkpoint is not None else
                     "fresh deterministic reset on the session's own server; "
                     "the turn-11 checkpoint is untouched"),
        },
        checkpoint_checksums=({"conversation_state":
                              resume_checkpoint.conversation_state_sha256,
                              "save": resume_checkpoint.checkpoint_sha256}
                             if resume_checkpoint is not None else {}),
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
        (f"resumed from verified checkpoint turn {resume_checkpoint.turn} "
         f"from parent run {resume_checkpoint.run_id}")
        if resume_checkpoint is not None else
        "fresh deterministic reset; no checkpoint loaded (the turn-11 "
        "checkpoint is untouched)")
    manifest.enforcement_decision = decision
    manifest.action_batch_schema_hash = mc.action_batch_schema_hash()
    manifest.live_fingerprint_probe = live_probe
    manifest.fingerprint_matches_spike = (
        live_probe.get("fingerprint") == spike_fingerprint
        if live_probe and spike_fingerprint else None)
    manifest.turn_11_checkpoint_untouched = True
    return manifest


def validate_persisted_conversation_before_probe(state_path, *, profile,
                                                 bundle_id, trigger_tokens,
                                                 tail_tokens):
    """Load and validate resume conversation state before any provider call."""
    from lechonk.model.conversation import ConversationConfig, ConversationManager
    from lechonk.model import prompts as prompt_templates
    manager = ConversationManager.load(
        state_path,
        config=ConversationConfig(trigger_tokens=trigger_tokens,
                                  tail_tokens=tail_tokens))
    prompt_templates.validate_conversation_system_message(
        manager.messages, profile=profile, bundle_id=bundle_id)
    return manager


# ---------------------------------------------------------------------------
# post-run analysis (from the ledger + the per-call API log)
# ---------------------------------------------------------------------------


def parse_ts(ts):
    return datetime.fromisoformat(ts)


def analyse(records, call_records, result, run_wall_s):
    """The per-turn token/budget table + the read-back evidence, computed
    from the ledger (the provenance transcript) and the per-call API log."""
    from lechonk.core.schemas import (
        ActionResultRecord, CandidatesRefRecord, OutcomeRecord,
        SnapshotRefRecord, TurnStartRecord,
    )

    turn_starts = {}   # turn -> ts
    snapshot_refs = {}  # turn -> token_count
    candidates_refs = {}
    actions = {}       # turn -> [ActionResultRecord]
    outcome = None
    for rec in records:
        if isinstance(rec, TurnStartRecord):
            turn_starts[rec.turn] = rec.ts
        elif isinstance(rec, SnapshotRefRecord):
            snapshot_refs[rec.turn] = rec.token_count
        elif isinstance(rec, CandidatesRefRecord):
            candidates_refs[rec.turn] = rec
        elif isinstance(rec, ActionResultRecord):
            actions.setdefault(rec.turn, []).append(rec)
        elif isinstance(rec, OutcomeRecord):
            outcome = rec

    # The per-call API records (decision 4) keyed by turn.
    api_by_turn = {}
    for rec in call_records:
        if rec.get("turn") is not None and not _is_compaction_call(rec):
            api_by_turn.setdefault(rec["turn"], []).append(rec)

    turns = sorted(turn_starts)
    per_turn = []
    for i, turn in enumerate(turns):
        if i + 1 < len(turns):
            wall_s = (parse_ts(turn_starts[turns[i + 1]])
                      - parse_ts(turn_starts[turn])).total_seconds()
            wall_note = f"turn_start({turn}) -> turn_start({turns[i + 1]})"
        elif outcome is not None:
            wall_s = (parse_ts(outcome.ts)
                      - parse_ts(turn_starts[turn])).total_seconds()
            wall_note = f"turn_start({turn}) -> outcome record"
        else:
            wall_s = None
            wall_note = "no following turn_start/outcome record"
        api_calls = api_by_turn.get(turn, [])
        # The rendered section sizes come from the turn's prompt (the first
        # call of the turn renders it; repair calls repeat the same sections).
        raw_sections = (api_calls[0].get("section_tokens")
                        if api_calls else None)
        # Error/truncated call records may have no section telemetry at all,
        # or may contain only the sections rendered before the failure.
        # Normalize the shape once so analysis and reporting remain total.
        sections = {
            name: (raw_sections.get(name) if isinstance(raw_sections, dict)
                   else None)
            for name in ("memory", "snapshot", "delta", "candidates")
        }
        turn_actions = actions.get(turn, [])
        readbacks = {}
        for a in turn_actions:
            if a.outcome == "executed":
                readbacks[a.readback] = readbacks.get(a.readback, 0) + 1
        token_usage = _token_usage_totals(api_calls)
        cost_usage = _cost_usage_totals(api_calls)
        per_turn.append({
            "turn": turn,
            "wall_clock_s": round(wall_s, 2) if wall_s is not None else None,
            "wall_clock_note": wall_note,
            "api_calls": len(api_calls),
            "api_attempts": [c["attempt"] for c in api_calls],
            "api_outcomes": [c["outcome"] for c in api_calls],
            # TOKEN MEASUREMENT: the API-reported counts are the ground truth.
            "api_tokens_in": token_usage["tokens_in"],
            "api_tokens_out": token_usage["tokens_out"],
            "api_tokens_in_unknown_calls": token_usage[
                "tokens_in_unknown_calls"],
            "api_tokens_out_unknown_calls": token_usage[
                "tokens_out_unknown_calls"],
            "token_measurement_status": token_usage[
                "token_measurement_status"],
            "api_latency_ms": _numeric_total(api_calls, "latency_ms"),
            "api_cost_usd": round(cost_usage["cost_usd"], 6),
            "api_cost_unknown_calls": cost_usage["cost_unknown_calls"],
            "cost_measurement_status": cost_usage["cost_measurement_status"],
            # The rendered section sizes (estimate_tokens) for the §3 budgets.
            "section_tokens": sections,
            "snapshot_tokens_ledger": snapshot_refs.get(turn),
            "snapshot_within_8000": _snapshot_within_8000(
                sections.get("snapshot"), snapshot_refs.get(turn)),
            "delta_within_800": (
                _known_number(sections.get("delta"))
                and sections["delta"] <= 800),
            "memory_within_1500": (
                _known_number(sections.get("memory"))
                and sections["memory"] <= 1500),
            "candidate_count": (candidates_refs[turn].candidate_count
                                if turn in candidates_refs else None),
            "opportunity_domains": (candidates_refs[turn].opportunity_domains
                                    if turn in candidates_refs else []),
            "actions_executed": sum(1 for a in turn_actions
                                    if a.outcome == "executed"),
            "actions_rejected": sum(1 for a in turn_actions
                                    if a.outcome == "rejected"),
            "actions_aborted_group": sum(1 for a in turn_actions
                                         if a.outcome == "aborted_group"),
            "readbacks": readbacks,
            "provenance": {
                cls: sum(1 for a in turn_actions
                         if a.outcome == "executed"
                         and a.provenance_class == cls)
                for cls in ("model", "continued_model_order",
                            "scripted_default", "engine_automatic",
                            "housekeeping")},
        })
    return per_turn, outcome


def expected_ledger_turns(turns, resume_turn=None):
    """Return the absolute turn numbers expected in this run's ledger.

    ``--turns`` is a count of phases for both fresh and resumed runs.  A
    resumed checkpoint already identifies the first phase, so acceptance
    checks must not assume that every ledger starts at turn one.
    """
    if turns is None:
        return None
    start = resume_turn if resume_turn is not None else 1
    return list(range(start, start + turns))


def _turn_of_call(call_id):
    """The runner's call ids are '<run>/t<turn>/call<attempt>'."""
    try:
        return int(call_id.split("/t")[1].split("/")[0])
    except (IndexError, ValueError):
        return None


def readback_evidence(records):
    """The read-back verification summary over all executed actions, with
    the INCONCLUSIVE breakdown by verb (the sentry quirk is the documented
    INCONCLUSIVE case: the read path cancels ACTIVITY_SENTRY within the same
    step, so an unchanged activity is not evidence of non-execution)."""
    from lechonk.core.schemas import ActionResultRecord

    counts = {"verified": 0, "inconclusive": 0, "contradicted": 0, "none": 0}
    inconclusive_by_verb = {}
    contradicted = []
    for rec in records:
        if not isinstance(rec, ActionResultRecord):
            continue
        if rec.outcome != "executed":
            continue
        counts[rec.readback] = counts.get(rec.readback, 0) + 1
        if rec.readback == "inconclusive":
            inconclusive_by_verb[rec.verb] = \
                inconclusive_by_verb.get(rec.verb, 0) + 1
        if rec.readback == "contradicted":
            contradicted.append({
                "turn": rec.turn, "ref": rec.ref, "verb": rec.verb,
                "detail": rec.detail})
    return {
        "counts": counts,
        "inconclusive_by_verb": inconclusive_by_verb,
        "inconclusive_documented_quirk_only": (
            set(inconclusive_by_verb) <= {"sentry"}),
        "contradicted_actions": contradicted,
    }


def feasibility_inputs(per_turn, run_wall_s):
    """The WP2-exit feasibility decision inputs (plan line 54 / probe P12).

    The ladder triggers are NECESSARY, not sufficient; the pairs/horizon are
    additionally computed from a wall-clock budget: measured arm runtime ×
    turns × ≥1.5 contingency (retries + late-game turn growth) must fit one
    idle day (24 h), comparator arms included. The DECISION is the
    orchestrator's; this run records the measured inputs + the arithmetic.
    """
    walls = [t["wall_clock_s"] for t in per_turn
             if t["wall_clock_s"] is not None]
    mean_turn_s = (sum(walls) / len(walls)) if walls else None
    # The comparator arm's measured per-turn cost (the amended WP1 run).
    comparator_s_per_turn = None
    comparator_source = None
    try:
        with open(COMPARATOR_RUN_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        comparator_s_per_turn = (meta["wall_clock_s"]
                                 / meta["turns_completed"])
        comparator_source = ("artifacts/runs/wp1-comparator-amended/"
                             "run_meta.json")
    except Exception as exc:
        log(f"WARN: comparator run meta unavailable: {exc!r}")
    idle_day_s = 24 * 3600
    contingency = 1.5
    shapes = []
    for pairs, horizon in ((2, 50), (1, 50), (2, 25), (1, 25)):
        model_s = (pairs * horizon * mean_turn_s * contingency
                   if mean_turn_s is not None else None)
        comp_s = (pairs * horizon * comparator_s_per_turn * contingency
                  if comparator_s_per_turn is not None else None)
        total = (model_s + comp_s) if (model_s is not None
                                       and comp_s is not None) else None
        shapes.append({
            "pairs": pairs, "horizon_turns": horizon,
            "model_arm_seconds": round(model_s, 1) if model_s else None,
            "comparator_arm_seconds": round(comp_s, 1) if comp_s else None,
            "total_seconds": round(total, 1) if total is not None else None,
            "total_hours": round(total / 3600, 2) if total is not None else None,
            "fits_one_idle_day": (total <= idle_day_s
                                  if total is not None else None),
        })
    return {
        "measured": {
            "mean_per_turn_wall_clock_s": (round(mean_turn_s, 2)
                                           if mean_turn_s is not None
                                           else None),
            "per_turn_wall_clock_s": walls,
            "run_wall_clock_s": round(run_wall_s, 1),
            "turns_measured": len(walls),
            "comparator_arm_s_per_turn": (
                round(comparator_s_per_turn, 3)
                if comparator_s_per_turn is not None else None),
            "comparator_source": comparator_source,
        },
        "ladder_triggers": {
            "rule": "measured mean per-turn wall-clock > 6 min/turn -> 1 "
                    "evaluation pair; > 10 min/turn -> 1 pair AND the "
                    "25-turn horizon (necessary, not sufficient)",
            "mean_over_6_min": (mean_turn_s > 360
                                if mean_turn_s is not None else None),
            "mean_over_10_min": (mean_turn_s > 600
                                 if mean_turn_s is not None else None),
        },
        "wall_clock_budget": {
            "rule": "measured arm runtime x turns x >=1.5 contingency must "
                    "fit one idle day (24 h), comparator arms included",
            "idle_day_s": idle_day_s,
            "contingency": contingency,
            "shapes": shapes,
        },
        "note": "the DECISION itself is the orchestrator's; this run "
                "provides the measured inputs honestly",
    }


def render_table_md(per_turn, feasibility, metrics_dict):
    lines = [
        "# WP2 model run — per-turn tokens / budgets / wall-clock",
        "",
        "TOKEN MEASUREMENT: the API-reported prompt/completion token counts "
        "per call are the ground truth; the section sizes are the rendered "
        "memory/snapshot/delta/candidate texts (estimate_tokens).",
        "",
        "| turn | wall-clock s | api calls | api tokens in | api tokens out "
        "| api cost $ | memory tok (≤1500) | snapshot tok (≤8000) | delta "
        "tok (≤800) | cands | executed | rejected | read-backs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in per_turn:
        rb = t["readbacks"]
        rb_txt = ", ".join(f"{k}:{v}" for k, v in sorted(rb.items())) or "-"
        lines.append(
            f"| {t['turn']} | {t['wall_clock_s']} | {t['api_calls']} "
            f"({'+'.join(str(a) for a in t['api_attempts'])}) "
            f"| {t['api_tokens_in']} | {t['api_tokens_out']} "
            f"| {t['api_cost_usd']:.6f} "
            f"| {_section_report_value(t.get('section_tokens'), 'memory')} "
            f"| {_section_report_value(t.get('section_tokens'), 'snapshot')} "
            f"(ledger {_report_value(t.get('snapshot_tokens_ledger'))}) "
            f"| {_section_report_value(t.get('section_tokens'), 'delta')} "
            f"| {_report_value(t.get('candidate_count'))} "
            f"| {t['actions_executed']} | {t['actions_rejected']} "
            f"| {rb_txt} |")
    lines += [
        "",
        "## §3 budget comparison",
        "",
        f"- snapshot ≤ 8,000 tokens: "
        f"{'ALL WITHIN' if all(t['snapshot_within_8000'] for t in per_turn) else 'OVERRUN (see table)'}",
        f"- delta ≤ 800 tokens: "
        f"{'ALL WITHIN' if all(t['delta_within_800'] for t in per_turn) else 'OVERRUN (see table)'}",
        f"- memory ≤ 1,500 tokens: "
        f"{'ALL WITHIN' if all(t['memory_within_1500'] for t in per_turn) else 'OVERRUN (see table)'}",
        "",
        "## Feasibility inputs (WP2 exit, plan line 54 / P12)",
        "",
        f"- measured mean per-turn wall-clock: "
        f"{feasibility['measured']['mean_per_turn_wall_clock_s']} s "
        f"over {feasibility['measured']['turns_measured']} turns",
        f"- > 6 min/turn (1-pair trigger): "
        f"{feasibility['ladder_triggers']['mean_over_6_min']}; "
        f"> 10 min/turn (1-pair + 25-turn-horizon trigger): "
        f"{feasibility['ladder_triggers']['mean_over_10_min']}",
        f"- comparator arm: "
        f"{feasibility['measured']['comparator_arm_s_per_turn']} s/turn "
        f"({feasibility['measured']['comparator_source']})",
        "",
        "| pairs | horizon | model-arm s (×1.5) | comparator-arm s (×1.5) "
        "| total s | total h | fits one idle day (24 h) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in feasibility["wall_clock_budget"]["shapes"]:
        lines.append(
            f"| {s['pairs']} | {s['horizon_turns']} "
            f"| {s['model_arm_seconds']} | {s['comparator_arm_seconds']} "
            f"| {s['total_seconds']} | {s['total_hours']} "
            f"| {s['fits_one_idle_day']} |")
    lines += [
        "",
        "The DECISION itself is the orchestrator's; this table records the "
        "measured inputs and the arithmetic.",
        "",
        "## Metrics (core.metrics over the ledger)",
        "",
        f"- participation: {json.dumps(metrics_dict['participation'])}",
        f"- participation gate pass: {metrics_dict['participation_gate_pass']}",
        f"- domain model shares: "
        f"{json.dumps(metrics_dict['domain_model_shares'])}",
        f"- commitment follow-through (any): "
        f"{json.dumps(metrics_dict['commitment_followthrough_any'])}",
        f"- commitment follow-through (model): "
        f"{json.dumps(metrics_dict['commitment_followthrough_model'])}",
        f"- reason-but-don't-act: "
        f"{json.dumps(metrics_dict['reason_but_dont_act'])}",
        f"- illegal-action shares: "
        f"{json.dumps(metrics_dict['illegal_action_shares'])}",
        "",
    ]
    return "\n".join(lines)


def render_metrics_md(metrics_dict, feasibility, readback, per_turn,
                      result, call_records):
    m = metrics_dict
    lines = [
        "# WP2 model run — provenance + metrics report",
        "",
        "Generated from the 10-turn ledger via `lechonk.core.metrics` "
        "(the §3 frozen operational definitions; minimum-denominator rule "
        "applied).",
        "",
        "## Run",
        "",
        f"- outcome: {result.outcome} ({result.reason})",
        f"- turns completed: {result.turns_completed}",
        f"- run wall-clock: {round(feasibility['measured']['run_wall_clock_s'], 1)} s",
        f"- ledger: {result.ledger_path}",
        f"- close status: {result.close_status}",
        "",
        "## Participation (§3 line 38)",
        "",
        f"- turns-with-opportunities / model participation: "
        f"{m['participation']['numerator']}/{m['participation']['denominator']}"
        f" = {m['participation']['value']} "
        f"({m['participation']['status']})",
        f"- participation gate (≥80%): {m['participation_gate_pass']}",
        "- per-domain model-control shares:",
    ]
    for domain, share in m["domain_model_shares"].items():
        lines.append(f"  - {domain}: {share['numerator']}/"
                     f"{share['denominator']} = {share['value']} "
                     f"({share['status']})")
    lines += [
        "",
        "## Commitment follow-through (§3 line 40)",
        "",
        f"- commitments: {m['commitment_total']} total, "
        f"{m['commitment_deduped']} duplicates, "
        f"{m['commitment_censored']} censored at horizon",
        f"- intent realized by any policy: "
        f"{m['commitment_followthrough_any']['numerator']}/"
        f"{m['commitment_followthrough_any']['denominator']} = "
        f"{m['commitment_followthrough_any']['value']} "
        f"({m['commitment_followthrough_any']['status']})",
        f"- model-attributed: "
        f"{m['commitment_followthrough_model']['numerator']}/"
        f"{m['commitment_followthrough_model']['denominator']} = "
        f"{m['commitment_followthrough_model']['value']} "
        f"({m['commitment_followthrough_model']['status']})",
        "",
        "## Reason-but-don't-act (§3 line 41)",
        "",
        f"- threat mentions: {m['reason_mentions_total']} total, "
        f"{m['reason_mentions_deduped']} duplicates, "
        f"{m['reason_mentions_censored']} censored",
        f"- share (1 − acted-rate): "
        f"{m['reason_but_dont_act']['numerator']}/"
        f"{m['reason_but_dont_act']['denominator']} = "
        f"{m['reason_but_dont_act']['value']} "
        f"({m['reason_but_dont_act']['status']})",
        "",
        "## Illegal-action taxonomy (§3 line 43)",
        "",
        f"- model-emitted attempts: {m['illegal_attempts_total']}",
    ]
    for cls, share in m["illegal_action_shares"].items():
        lines.append(f"- {cls}: {share['numerator']}/{share['denominator']}"
                     f" = {share['value']} ({share['status']})")
    lines += [
        "",
        "## Read-back verification (per executed action)",
        "",
        f"- {json.dumps(readback['counts'])}",
        f"- INCONCLUSIVE by verb: "
        f"{json.dumps(readback['inconclusive_by_verb']) or '{}'}",
        f"- INCONCLUSIVE only where documented quirks apply (sentry): "
        f"{readback['inconclusive_documented_quirk_only']}",
        f"- CONTRADICTED actions: {len(readback['contradicted_actions'])}",
        "",
        "## Per-turn tokens / cost / latency / wall-clock",
        "",
        "| turn | wall-clock s | api calls | tokens in | tokens out | "
        "latency ms | cost $ |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in per_turn:
        lines.append(
            f"| {t['turn']} | {t['wall_clock_s']} | {t['api_calls']} "
            f"| {t['api_tokens_in']} | {t['api_tokens_out']} "
            f"| {t['api_latency_ms']} | {t['api_cost_usd']:.6f} |")
    tot_in = sum(t["api_tokens_in"] for t in per_turn)
    tot_out = sum(t["api_tokens_out"] for t in per_turn)
    unknown_in = sum(t["api_tokens_in_unknown_calls"] for t in per_turn)
    unknown_out = sum(t["api_tokens_out_unknown_calls"] for t in per_turn)
    tot_cost = sum(t["api_cost_usd"] for t in per_turn)
    cost_unknown = sum(t.get("api_cost_unknown_calls", 0) for t in per_turn)
    lines += [
        f"| **total** | | {sum(t['api_calls'] for t in per_turn)} "
        f"| {tot_in}{' (unknown calls: ' + str(unknown_in) + ')' if unknown_in else ''} "
        f"| {tot_out}{' (unknown calls: ' + str(unknown_out) + ')' if unknown_out else ''} | "
        f"{sum(t['api_latency_ms'] for t in per_turn)} "
        f"| {tot_cost:.6f}{' (unknown calls: ' + str(cost_unknown) + ')' if cost_unknown else ''} |",
        "",
        "## Feasibility inputs (WP2 exit — the decision is the "
        "orchestrator's)",
        "",
        f"- measured mean per-turn wall-clock: "
        f"{feasibility['measured']['mean_per_turn_wall_clock_s']} s "
        f"({feasibility['measured']['turns_measured']} turns)",
        f"- ladder triggers: >6 min/turn -> "
        f"{feasibility['ladder_triggers']['mean_over_6_min']}; "
        f">10 min/turn -> "
        f"{feasibility['ladder_triggers']['mean_over_10_min']}",
        f"- comparator arm: "
        f"{feasibility['measured']['comparator_arm_s_per_turn']} s/turn",
        "",
        "| pairs | horizon | model-arm s (×1.5) | comparator-arm s (×1.5) "
        "| total s | total h | fits 24 h |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in feasibility["wall_clock_budget"]["shapes"]:
        lines.append(
            f"| {s['pairs']} | {s['horizon_turns']} "
            f"| {s['model_arm_seconds']} | {s['comparator_arm_seconds']} "
            f"| {s['total_seconds']} | {s['total_hours']} "
            f"| {s['fits_one_idle_day']} |")
    lines += ["", f"- api call log records: {len(call_records)}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    global _PROMPT_SNAPSHOT, _CAPTURE_GATEWAY
    from lechonk.model import prompts as _prompt_module
    if (_PROMPT_SNAPSHOT is None or
            _PROMPT_SNAPSHOT.profile != PROMPT_PROFILE or
            _PROMPT_SNAPSHOT.bundle_id != PROMPT_BUNDLE_ID):
        _PROMPT_SNAPSHOT = _prompt_module.capture_prompt_snapshot(
            PROMPT_PROFILE, bundle_id=PROMPT_BUNDLE_ID)
    global _MODEL_CALL_EVENTS_ENABLED
    global CONTEXT_JOURNAL_PATH, CONVERSATION_STATE_PATH, RESUME_CHECKPOINT, TRACE_PATH
    log(f"[{RUN}] repo={REPO}")
    horizon_label = "until-game-over" if UNTIL_GAME_OVER else str(TURNS)
    log(f"[{RUN}] rundir={RUNDIR} attempt={ATTEMPT} seed={SEED} turns={horizon_label} "
        f"api_timeout_s={API_TIMEOUT_S:g} "
        f"acting_deadline_s={ACTING_DEADLINE_S:g}")
    log(f"[{RUN}] policy: mistral-large-4 via lechonk.model.turn_client "
        f"(json_schema, the WP2 spike decision decided live) over "
        f"core.runner's state machine")
    if PROMPT_PROFILE not in {"gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}:
        log(f"[{RUN}] WARNING: prompt profile {PROMPT_PROFILE!r} has no "
            "encyclopedia lookup tool; use scripts/launch.sh (or "
            "--prompt-profile gameplay_v2/gameplay_v3/gameplay_v4/gameplay_v5/gameplay_v6 with conversation_v1) for the "
            "interactive ruleset lookup profile")

    if CAPTURE_ENABLED:
        # A caller-supplied directory is an exact no-overwrite target.  The
        # default is unique even for bounded retries under one run id.
        capture_target = CAPTURE_DIR or os.path.join(
            RUNDIR, f"capture-{time.time_ns()}")
        _CAPTURE_GATEWAY = CaptureGateway.create(
            capture_target, run_id=RUN,
            expected_username=FREECIV_USERNAME, child_pid=os.getpid())
        # Fail-open cleanup also covers startup/API errors before runner.run.
        atexit.register(lambda: _CAPTURE_GATEWAY and
                        _CAPTURE_GATEWAY.finish(status="process_exit"))
        log(f"[{RUN}] capture enabled: dir={_CAPTURE_GATEWAY.capture_dir}")
        if CAPTURE_SYNC:
            log(f"[{RUN}] capture sync enabled: timeout_ms={CAPTURE_SYNC_TIMEOUT_MS} "
                f"budget_ms={CAPTURE_SYNC_BUDGET_MS}")

    # A reused run id must not silently mix a previous context journal into a
    # fresh transcript.  Preserve attempt-1 data on explicit retries using the
    # same convention as the other per-run artifacts.
    if CONTEXT_POLICY == "conversation_v1":
        CONTEXT_JOURNAL_PATH = fresh_artifact("conversation.jsonl")
        CONVERSATION_STATE_PATH = fresh_artifact("state.json")
        log(f"[{RUN}] context policy: conversation_v1, journal={CONTEXT_JOURNAL_PATH}, "
            f"compaction_limit_tokens={CONVERSATION_COMPACTION_LIMIT_TOKENS}, "
            f"tail_target_tokens={CONVERSATION_TAIL_TARGET_TOKENS}")
    else:
        log(f"[{RUN}] context policy: legacy (no context journal)")

    # --- wait for the stack (hardened startup: bounded pubstatus retries) --
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

    if RESUME_MANIFEST:
        from lechonk.eval.checkpoint import stage_for_load
        staged = stage_for_load(RESUME_CHECKPOINT.checkpoint_path,
                                username=RESUME_CHECKPOINT.username)
        log(f"[{RUN}] staged verified checkpoint turn={RESUME_CHECKPOINT.turn} "
                f"save={staged.save_name} parent_run={RESUME_CHECKPOINT.run_id}")

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

    # Validate the persisted conversation before constructing a client or
    # issuing its fingerprint probe.  A mismatched frozen prompt bundle must
    # fail before any provider call.
    if RESUME_CHECKPOINT is not None and CONTEXT_POLICY == "conversation_v1":
        try:
            validate_persisted_conversation_before_probe(
                RESUME_CHECKPOINT.conversation_state_path,
                profile=PROMPT_PROFILE, bundle_id=PROMPT_BUNDLE_ID,
                trigger_tokens=CONVERSATION_COMPACTION_LIMIT_TOKENS,
                tail_tokens=CONVERSATION_TAIL_TARGET_TOKENS)
        except Exception as exc:
            raise SystemExit(f"resume rejected: {exc}") from exc

    # --- the client (json_schema per the spike decision) + the adapter ------
    from lechonk.model.turn_client import MistralTurnClient

    client_kwargs = {
        "enforcement": "json_schema",
        "timeout_ms": round(API_TIMEOUT_S * 1000),
        "reasoning_effort": REASONING_EFFORT,
        "stream": STREAM,
        "stream_callback": (
            lambda event: progress({"event": "model_output_delta", **event})
            if STREAM else None),
        "call_event_callback": _model_call_event,
    }
    # Prompt-cache routing is an optional SDK/client capability.  Keep legacy
    # runs request-compatible while allowing conversation_v1 to use one stable
    # non-secret key for the run/compaction epoch.
    cache_lineage = (RESUME_CHECKPOINT.run_id
                     if RESUME_CHECKPOINT is not None else RUN)
    PROMPT_CACHE_KEY = (
        f"lechonk:{cache_lineage}:conversation-v1:{PROMPT_PROFILE}:"
        f"prompts-{_active_prompt_hash_digest()}:epoch-0"
        if CONTEXT_POLICY == "conversation_v1" else None)
    # Restore and validate the persisted conversation before constructing the
    # provider client or probing it.  On resume, the persisted manager's cache
    # epoch is authoritative; deriving a key from the immediate parent would
    # break the common-prefix cache after compaction/forking.
    conversation_manager = None
    initial_memory_state = None
    resume_load_game = None
    resume_turn = None
    if CONTEXT_POLICY == "conversation_v1":
        from lechonk.model.conversation import ConversationConfig, ConversationManager
        from lechonk.model import prompts as prompt_templates
        conversation_config = ConversationConfig(
            trigger_tokens=CONVERSATION_COMPACTION_LIMIT_TOKENS,
            tail_tokens=CONVERSATION_TAIL_TARGET_TOKENS)
        if RESUME_CHECKPOINT is not None:
            conversation_manager = ConversationManager.load(
                RESUME_CHECKPOINT.conversation_state_path,
                config=conversation_config)
            try:
                prompt_templates.validate_conversation_system_message(
                    conversation_manager.messages, profile=PROMPT_PROFILE,
                    bundle_id=PROMPT_BUNDLE_ID,
                    snapshot=_PROMPT_SNAPSHOT)
            except Exception as exc:
                raise SystemExit(f"resume rejected: {exc}") from exc
            conversation_manager.fork_persistence(
                CONVERSATION_STATE_PATH, CONTEXT_JOURNAL_PATH)
            initial_memory_state = RESUME_CHECKPOINT.strategy_memory
            from lechonk.eval.checkpoint import save_name_for
            resume_load_game = save_name_for(
                Path(RESUME_CHECKPOINT.checkpoint_path).name)
            resume_turn = RESUME_CHECKPOINT.turn
        else:
            conversation_manager = ConversationManager(
                system_message={"role": "system", "content": _active_system_prompt(prompt_templates)},
                config=conversation_config,
                state_path=CONVERSATION_STATE_PATH,
                journal_path=CONTEXT_JOURNAL_PATH,
                prompt_cache_key=PROMPT_CACHE_KEY,
            )
            conversation_manager._persist()
        PROMPT_CACHE_KEY = conversation_manager.prompt_cache_key
    if PROMPT_CACHE_KEY is not None and "prompt_cache_key" in inspect.signature(
            mc.MistralClient).parameters:
        client_kwargs["prompt_cache_key"] = PROMPT_CACHE_KEY
    client = mc.MistralClient(**client_kwargs)
    log(f"[{RUN}] reasoning_effort={REASONING_EFFORT}")
    _MODEL_CALL_EVENTS_ENABLED = True
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

    api_log_path = fresh_artifact("api_calls.jsonl")
    api_log_fh = open(api_log_path, "a", encoding="utf-8")
    # Model traces are per-attempt artifacts too. Rotate them before the first
    # model output so a bounded retry cannot append into attempt 1's trace.
    TRACE_PATH = fresh_artifact("model_traces.jsonl")

    def api_log(record):
        api_log_fh.write(json.dumps(record, sort_keys=True,
                                    ensure_ascii=False, default=str) + "\n")
        api_log_fh.flush()

    # The fingerprint probe is an API call too (decision 4: every call is
    # logged) — recorded with kind="probe", turn=null.
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

    adapter = MistralTurnClient(
        client, api_log=api_log, output_callback=progress,
        context_policy=CONTEXT_POLICY,
        prompt_profile=PROMPT_PROFILE,
        prompt_bundle_id=PROMPT_BUNDLE_ID,
        prompt_snapshot=_PROMPT_SNAPSHOT,
        conversation_manager=conversation_manager)
    log(f"[{RUN}] adapter: MistralTurnClient over MistralClient "
        f"(enforcement={client.enforcement.value}, model={client.model}, "
        f"temperature={client.temperature}, max_tokens={client.max_tokens}, "
        f"stream={STREAM})")

    # --- the manifest (decision 8) + the ledger + the runner ----------------
    from lechonk.core.ledger import TranscriptLedger
    from lechonk.core.runner import GameRunner, RunnerConfig

    manifest = build_manifest(arm_start_ts, spike_fingerprint, live_probe,
                              RESUME_CHECKPOINT)
    transcript_path = fresh_artifact("transcript.jsonl")
    ledger = TranscriptLedger(transcript_path)
    config = RunnerConfig(
        max_turns=(resume_turn + TURNS - 1 if resume_turn is not None and
                   TURNS is not None else TURNS),
        run_id=RUN, workdir=WORKDIR, seed=SEED,
        username=FREECIV_USERNAME, aifill=AIFILL,
        player_name=PLAYER_NAME,
        acting_deadline_s=ACTING_DEADLINE_S,
        context_policy=CONTEXT_POLICY,
        prompt_profile=PROMPT_PROFILE,
        context_journal_path=CONTEXT_JOURNAL_PATH,
        load_game=resume_load_game,
        resume_turn=resume_turn,
        resume_player_id=(RESUME_CHECKPOINT.player_id
                          if RESUME_CHECKPOINT is not None else None),
        initial_memory_state=initial_memory_state,
        checkpoint_dir=(os.path.join(RUNDIR, "checkpoints")
                        if RESUME_MANIFEST or CONTEXT_POLICY == "conversation_v1" else None),
        recovery_manifest_path=(os.path.join(RUNDIR, "recovery.json")
                                if RESUME_MANIFEST or CONTEXT_POLICY == "conversation_v1" else None),
        conversation_state_path=CONVERSATION_STATE_PATH,
        parent_run_id=(RESUME_CHECKPOINT.run_id if RESUME_CHECKPOINT else None),
        recovery_config_digest=recovery_config_digest(),
        recovery_config_digest_schema=RECOVERY_DIGEST_SCHEMA,
        prompt_bundle_id=PROMPT_BUNDLE_ID,
        candidate_surface_policy=CANDIDATE_SURFACE_POLICY,
        candidate_surface_cap=(12000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else 8000 if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
        candidate_admission_version=("candidate_surface_v3-r1" if CANDIDATE_SURFACE_POLICY == "candidate_surface_v3" else "candidate_surface_v2-r1"
                                     if CANDIDATE_SURFACE_POLICY == "candidate_surface_v2" else None),
        compaction_schema=("conversation-compaction-v1"
                           if RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
                           else None),
        prompt_budget_schema=(f"{PROMPT_PROFILE.replace('_', '-')}-budget-v1"
                              if CANDIDATE_SURFACE_POLICY in {"candidate_surface_v2", "candidate_surface_v3"}
                              else "legacy-budget-v1"),
        validate_reference_provenance=(
            RESUME_CHECKPOINT is not None and PROMPT_PROFILE in {
                "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}),
        expected_ruleset_reference_hash=(
            RESUME_CHECKPOINT.ruleset_reference_hash
            if RESUME_CHECKPOINT is not None else None),
        expected_help_reference_hash=(
            RESUME_CHECKPOINT.help_reference_hash
            if RESUME_CHECKPOINT is not None else None),
        expected_help_reference_source_sha256=(
            RESUME_CHECKPOINT.help_reference_source_sha256
            if RESUME_CHECKPOINT is not None else None),
        expected_classic_reference_schema_version=(
            CLASSIC_RUNTIME_SCHEMA
            if RESUME_CHECKPOINT is not None else None),
        expected_classic_ruleset_tree_sha256=(
            RESUME_CHECKPOINT.classic_ruleset_tree_sha256
            if RESUME_CHECKPOINT is not None else None),
        expected_classic_reference_present=(
            RESUME_CHECKPOINT.classic_reference_present
            if RESUME_CHECKPOINT is not None else None),
        expected_manual_reference_schema_version=(
            RESUME_CHECKPOINT.manual_reference_schema_version
            if RESUME_CHECKPOINT is not None else None),
        expected_manual_reference_hash=(
            RESUME_CHECKPOINT.manual_reference_hash
            if RESUME_CHECKPOINT is not None else None),
        expected_manual_reference_source_sha256=(
            RESUME_CHECKPOINT.manual_reference_source_sha256
            if RESUME_CHECKPOINT is not None else None),
        expected_manual_reference_present=(
            RESUME_CHECKPOINT.manual_reference_present
            if RESUME_CHECKPOINT is not None else None))
    runner = GameRunner(
        config=config, client=adapter, ledger=ledger, manifest=manifest,
        progress=progress)

    if RESUME_CHECKPOINT is None:
        log(f"[{RUN}] starting the run: {horizon_label} model-driven turns, "
            "fresh deterministic reset on the session's own server")
    else:
        log(f"[{RUN}] starting the run: {horizon_label} model-driven turns, "
            f"resumed from parent={RESUME_CHECKPOINT.run_id} "
            f"checkpoint_turn={RESUME_CHECKPOINT.turn}")
    run_t0 = time.time()
    try:
        result = runner.run()
    finally:
        # A transport exception, cancellation, or runner bug must not leave a
        # daemon heartbeat reporting against a dead run.
        _stop_model_heartbeat()
    _persist_ruleset_reference_from_session(getattr(runner, "_session", None))
    run_wall_s = time.time() - run_t0
    api_log_fh.close()
    ledger.close()
    log(f"[{RUN}] run finished: outcome={result.outcome} "
        f"reason={result.reason!r} turns_completed={result.turns_completed} "
        f"close_status={result.close_status} wall_clock={run_wall_s:.1f}s")
    if _CAPTURE_GATEWAY is not None:
        _CAPTURE_GATEWAY.ledger_result({
            "ledger_path": result.ledger_path,
            "outcome": result.outcome,
            "turns_completed": result.turns_completed,
        })
        _CAPTURE_GATEWAY.finish(
            status=str(result.outcome).lower(),
            payload={"outcome": result.outcome,
                     "reason": result.reason,
                     "turns_completed": result.turns_completed,
                     "close_status": result.close_status,
                     "wall_clock_s": run_wall_s})

    # --- post-run: the ledger, the per-turn table, the metrics --------------
    from lechonk.core.ledger import load_transcript
    from lechonk.core.metrics import compute_metrics

    load = load_transcript(transcript_path)
    records = load.records
    log(f"[{RUN}] transcript: {len(records)} records, "
        f"discarded_trailing_partial={load.discarded_trailing_partial}, "
        f"last_seq={load.last_seq}")

    call_records = adapter.call_records
    log(f"[{RUN}] api call log: {len(call_records)} records "
        f"({api_log_path})")

    per_turn, outcome_rec = analyse(records, call_records, result, run_wall_s)
    readback = readback_evidence(records)
    metrics = compute_metrics(records)
    metrics_dict = metrics.to_dict()
    feasibility = feasibility_inputs(per_turn, run_wall_s)

    write_json("token_budget_table.json", {
        "run": RUN, "attempt": ATTEMPT, "seed": SEED,
        "measurement_note": "the API-reported prompt/completion token "
                            "counts per call are the ground truth; the "
                            "section sizes are the rendered "
                            "memory/snapshot/delta/candidate texts",
        "budgets": {"snapshot": 8000, "delta": 800, "memory": 1500},
        "per_turn": per_turn,
        "totals": {
            "api_calls": sum(t["api_calls"] for t in per_turn),
            "api_tokens_in": sum(t["api_tokens_in"] for t in per_turn),
            "api_tokens_out": sum(t["api_tokens_out"] for t in per_turn),
            "api_tokens_in_unknown_calls": sum(
                t["api_tokens_in_unknown_calls"] for t in per_turn),
            "api_tokens_out_unknown_calls": sum(
                t["api_tokens_out_unknown_calls"] for t in per_turn),
            "token_measurement_status": (
                "unknown" if any(
                    t["token_measurement_status"] == "unknown"
                    for t in per_turn) else "complete"),
            "api_cost_usd": round(sum(t["api_cost_usd"] for t in per_turn),
                                  6),
            "api_cost_unknown_calls": sum(
                t.get("api_cost_unknown_calls", 0) for t in per_turn),
            "cost_measurement_status": (
                "unknown" if any(t.get("cost_measurement_status") == "unknown"
                                 for t in per_turn) else "complete"),
            "api_latency_ms": sum(t["api_latency_ms"] for t in per_turn),
            "run_wall_clock_s": round(run_wall_s, 1),
        },
    })
    with open(os.path.join(RUNDIR, "token_budget_table.md"), "w",
              encoding="utf-8") as fh:
        fh.write(render_table_md(per_turn, feasibility, metrics_dict))
    log("wrote token_budget_table.md")

    write_json("metrics_report.json", {
        "run": RUN, "attempt": ATTEMPT, "seed": SEED,
        "generated_by": "lechonk.core.metrics.compute_metrics over the "
                        "10-turn ledger (the §3 frozen definitions)",
        "metrics": metrics_dict,
        "readback_evidence": readback,
        "per_turn": per_turn,
        "feasibility": feasibility,
        "api_totals": {
            "calls": len(call_records),
            **_token_usage_totals(call_records),
            **_cost_usage_totals(call_records),
            "latency_ms": _numeric_total(call_records, "latency_ms"),
        },
        "outcome": {"outcome": result.outcome, "reason": result.reason,
                    "turns_completed": result.turns_completed,
                    "close_status": result.close_status},
    })
    with open(os.path.join(RUNDIR, "metrics_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write(render_metrics_md(metrics_dict, feasibility, readback,
                                   per_turn, result, call_records))
    log("wrote metrics_report.json + metrics_report.md")

    # --- acceptance checks ---------------------------------------------------
    from lechonk.core.schemas import RunManifestRecord

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
    check("ledger_complete_target_turns",
          ((not UNTIL_GAME_OVER
            and [t["turn"] for t in per_turn] ==
                expected_ledger_turns(TURNS,
                                       (RESUME_CHECKPOINT.turn
                                        if RESUME_CHECKPOINT is not None
                                        else None)))
           or (UNTIL_GAME_OVER and bool(per_turn))),
          f"{len(per_turn)} turn_start records"
          + (f" (turns {per_turn[0]['turn']}..{per_turn[-1]['turn']})"
             if per_turn else ""))
    check("ledger_no_corruption",
          not load.discarded_trailing_partial,
          "no trailing partial record; every line checksum-verified")
    check("seq_contiguous",
          [r.seq for r in records] == list(range(len(records))),
          f"seq 0..{len(records) - 1}")
    check("run_completed_target_horizon",
          ((result.outcome == "COMPLETED"
            and (UNTIL_GAME_OVER or result.turns_completed == TURNS))
           or (UNTIL_GAME_OVER
               and result.outcome in {"VICTORY", "DEFEATED"})),
          f"outcome={result.outcome} turns_completed="
          f"{result.turns_completed} reason={result.reason!r}")
    check("every_turn_has_model_call",
          all(t["api_calls"] >= 1 for t in per_turn),
          f"api calls per turn: {[t['api_calls'] for t in per_turn]}")
    check("attempt_cap_respected",
          all(len(t["api_attempts"]) <= 2 for t in per_turn),
          f"max attempts per turn: "
          f"{max((len(t['api_attempts']) for t in per_turn), default=0)} "
          f"(≤2 total API attempts per turn, decision 4)")
    check("no_contradicted_readbacks",
          readback["counts"].get("contradicted", 0) == 0,
          f"read-backs: {readback['counts']}")
    check("inconclusive_only_documented_quirks",
          readback["inconclusive_documented_quirk_only"],
          f"INCONCLUSIVE by verb: {readback['inconclusive_by_verb']} "
          f"(the sentry quirk: the read path cancels ACTIVITY_SENTRY "
          f"within the same step — documented in adapter/execute.py)")
    check("snapshot_within_8000_all_turns",
          all(t["snapshot_within_8000"] for t in per_turn),
          f"max snapshot tokens: "
          f"{_section_max(per_turn, 'snapshot')}; "
          f"missing/unknown ledger turns: "
          f"{sum(t['snapshot_tokens_ledger'] is None for t in per_turn)}")
    check("delta_within_800_all_turns",
          all(t["delta_within_800"] for t in per_turn),
          f"max delta tokens: "
          f"{_section_max(per_turn, 'delta')}")
    check("memory_within_1500_all_turns",
          all(t["memory_within_1500"] for t in per_turn),
          f"max memory tokens: "
          f"{_section_max(per_turn, 'memory')}")
    turn_call_records = [r for r in call_records
                         if r.get("turn") is not None
                         and not _is_compaction_call(r)]
    compaction_call_records = [r for r in call_records
                               if _is_compaction_call(r)]
    check("per_call_api_logged",
          len(turn_call_records) == sum(t["api_calls"] for t in per_turn)
          and all({"tokens_in", "tokens_out", "latency_ms", "cost_usd"}
                  <= set(r) for r in turn_call_records),
          f"{len(turn_call_records)} turn call records with tokens/latency/cost "
          f"(decision 4); compaction calls={len(compaction_call_records)}")
    check("feasibility_inputs_recorded",
          feasibility["measured"]["mean_per_turn_wall_clock_s"] is not None
          and len(feasibility["wall_clock_budget"]["shapes"]) == 4,
          f"mean per-turn wall-clock: "
          f"{feasibility['measured']['mean_per_turn_wall_clock_s']} s")

    # --- run meta -------------------------------------------------------------
    verdict = "PASS" if checks and all(c["pass"] for c in checks.values()) \
        else "FAIL"
    ping_meta = _pingtimeout_metadata()
    # The first ledger manifest was emitted before startup. This companion
    # artifact records the post-start AdapterSession read-back without
    # mutating that immutable first record.
    write_json("manifest_runtime.json", {
        "run": RUN,
        "source": "AdapterSession session_ready read-back",
        "server_settings": {
            "pingtimeout_override_requested": ping_meta["requested"],
            "pingtimeout_status": ping_meta["status"],
            "pingtimeout_actual": ping_meta["actual"],
            "pingtimeout_effective": ping_meta["effective"],
        },
    })
    write_json("run_meta.json", {
        "run": RUN,
        "attempt": ATTEMPT,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unattended": True,
        "policy": "mistral-large-4 (lechonk.model.turn_client over "
                  "MistralClient, json_schema enforcement — the WP2 spike "
                  "decision decided live) driven by core.runner's state "
                  "machine",
        "seed": SEED,
        "freeciv_username": FREECIV_USERNAME,
        "config": {"freeciv_username": FREECIV_USERNAME},
        "aifill": AIFILL,
        "configured_ai_opponents": AIFILL - 1,
        "turns_target": TURNS,
        "until_game_over": UNTIL_GAME_OVER,
        "context_policy": CONTEXT_POLICY,
        "prompt_profile": PROMPT_PROFILE,
        "prompt_hashes": _active_prompt_hashes(bundle_id=PROMPT_BUNDLE_ID),
        "conversation_compaction_limit_tokens": (
            CONVERSATION_COMPACTION_LIMIT_TOKENS
            if CONTEXT_POLICY == "conversation_v1" else None),
        "conversation_tail_target_tokens": (
            CONVERSATION_TAIL_TARGET_TOKENS
            if CONTEXT_POLICY == "conversation_v1" else None),
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "context_journal_path": CONTEXT_JOURNAL_PATH,
        "conversation_state_path": CONVERSATION_STATE_PATH,
        "api_timeout_s": API_TIMEOUT_S,
        "acting_deadline_s": ACTING_DEADLINE_S,
        "pingtimeout_status": ping_meta["status"],
        "pingtimeout_actual": ping_meta["actual"],
        "ruleset_reference_hash": _SESSION_RUNTIME["ruleset_reference_hash"],
        "ruleset_reference_status": _SESSION_RUNTIME["ruleset_reference_status"],
        "turns_completed": result.turns_completed,
        "outcome": result.outcome,
        "reason": result.reason,
        "fc_args_overrides": {
            "self_play": False, "aifill": AIFILL,
            "configured_ai_opponents": AIFILL - 1,
            "debug.randomly_generate_seeds": False,
            "debug.mapseed": SEED, "debug.gameseed": SEED,
            "debug.agentseed": SEED,
            "server_timeout": SERVER_TIMEOUT_EFFECTIVE_S,
            "begin_turn_timeout": 900,
            "wait_for_timeout": 60, "http_timeout_s": 15.0,
            "pingtimeout": int(PINGTIMEOUT_CONFIGURED_S),
            "pingtimeout_override_requested": PINGTIMEOUT_CONFIGURED_S,
            "pingtimeout_status": ping_meta["status"],
            "pingtimeout_actual": ping_meta["actual"],
            "pingtimeout_effective": ping_meta["effective"],
            "max_turns": 100000 if UNTIL_GAME_OVER else TURNS + 1,
        },
        "workdir": WORKDIR,
        "wall_clock_s": round(run_wall_s, 1),
        "api_totals": {
            "calls": len(call_records),
            **_token_usage_totals(call_records),
            **_cost_usage_totals(call_records),
            "latency_ms": _numeric_total(call_records, "latency_ms"),
        },
        "readback_evidence": readback,
        "feasibility": feasibility,
        "metrics_summary": {
            "participation": metrics_dict["participation"],
            "participation_gate_pass": metrics_dict["participation_gate_pass"],
            "domain_model_shares": metrics_dict["domain_model_shares"],
            "commitment_followthrough_any":
                metrics_dict["commitment_followthrough_any"],
            "commitment_followthrough_model":
                metrics_dict["commitment_followthrough_model"],
            "reason_but_dont_act": metrics_dict["reason_but_dont_act"],
            "illegal_action_shares": metrics_dict["illegal_action_shares"],
        },
        "turn_11_checkpoint_untouched": True,
        "artifacts": {
            "transcript": "transcript.jsonl",
            "api_calls": "api_calls.jsonl",
            "token_budget_table": "token_budget_table.json/.md",
            "metrics_report": "metrics_report.json/.md",
            "manifest_runtime": "manifest_runtime.json",
            "ruleset_reference": "ruleset_reference.json (when available)",
            "ruleset_reference_meta": "ruleset_reference_meta.json",
            "stdout": "run_stdout.log",
        },
        "checks": checks,
        "verdict": verdict,
    })

    log(f"[{RUN}] checks: {sum(1 for c in checks.values() if c['pass'])}/"
        f"{len(checks)} passed")
    log(f"[{RUN}] VERDICT: {verdict}")

    # --- bounded retry on infrastructure failure (once, logged) --------------
    if (verdict == "FAIL" and result.outcome == "TRUNCATED"
            and ATTEMPT == 1):
        capture_retry_block_reason = _capture_retry_block_reason(
            CAPTURE_ENABLED)
        write_json("infra_failure_attempt1.json", {
            "run": RUN, "attempt": 1, "seed": SEED,
            "outcome": result.outcome, "reason": result.reason,
            "turns_completed": result.turns_completed,
            "checks": checks,
            "retry_suppressed": capture_retry_block_reason is not None,
            "retry_suppression_reason": capture_retry_block_reason,
            "note": (capture_retry_block_reason
                      or
                      "bounded retry (once): re-executing the whole run "
                      "with WP2_MODEL_RUN_ATTEMPT=2; one env per process, "
                      "so the retry is a fresh process with a fresh env"),
        })
        if capture_retry_block_reason is not None:
            log(f"[{RUN}] bounded retry suppressed: "
                f"{capture_retry_block_reason}")
            _log_file.close()
            return 1
        log(f"[{RUN}] bounded retry (once): re-executing with "
            f"WP2_MODEL_RUN_ATTEMPT=2")
        retry_env = dict(os.environ, WP2_MODEL_RUN_ATTEMPT="2",
                         LECHONK_RUN_TOKEN=RUN_TOKEN)
        try:
            # Preserve the exact invocation on the bounded retry.  Dropping
            # argv here silently changed a short/conversation run into the
            # script's legacy 10-turn default and lost its policy/timeout
            # settings.
            retry_argv = _retry_argv_with_absolute_resume(
                sys.argv[1:], RESUME_MANIFEST)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), *retry_argv],
                env=retry_env)
        finally:
            _log_file.close()
        return proc.returncode
    _log_file.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
