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
import hashlib
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


def _parse_args(argv=None):
    """CLI: --turns (default 10, the WP2 acceptance count); --run-id and
    --seed so a short spot-check run persists under its own
    artifacts/runs/<id>/ directory with its own seed, never touching
    another run's artifacts."""
    parser = argparse.ArgumentParser(
        description="WP2 package C — model-driven integration run")
    parser.add_argument("--turns", type=int, default=10,
                        help="number of model-driven turns (default: 10)")
    parser.add_argument("--run-id", default="wp2-model-run",
                        help="run id / artifact directory name under "
                             "artifacts/runs/ (default: wp2-model-run)")
    parser.add_argument("--seed", type=int, default=20261028,
                        help="deterministic map/game/agent seed "
                             "(default: 20261028)")
    return parser.parse_args(argv)


_ARGS = _parse_args()
RUN = _ARGS.run_id
RUNDIR = os.path.join(REPO, "artifacts", "runs", RUN)
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream
os.makedirs(WORKDIR, exist_ok=True)

SEED = _ARGS.seed  # default 20261028: distinct from wp1 runs and the spike
TURNS = _ARGS.turns  # default 10: the WP2 acceptance turn count
PUBSTATUS = "http://localhost:8080/pubstatus"
ATTEMPT = int(os.environ.get("WP2_MODEL_RUN_ATTEMPT", "1"))

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
    from lechonk.model.sampling import actual_client_sampling

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
        # W1-6 sampling manifest truth: the ACTUAL client sampling
        # (temperature 1.0, reasoning_effort "high"; the client never
        # sends top_p) — never the fabricated greedy dict.
        sampling=actual_client_sampling(),
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
    return manifest


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
        if rec.get("turn") is not None:
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
        sections = (api_calls[0]["section_tokens"] if api_calls else
                    {"memory": None, "snapshot": None, "delta": None,
                     "candidates": None})
        turn_actions = actions.get(turn, [])
        readbacks = {}
        for a in turn_actions:
            if a.outcome == "executed":
                readbacks[a.readback] = readbacks.get(a.readback, 0) + 1
        per_turn.append({
            "turn": turn,
            "wall_clock_s": round(wall_s, 2) if wall_s is not None else None,
            "wall_clock_note": wall_note,
            "api_calls": len(api_calls),
            "api_attempts": [c["attempt"] for c in api_calls],
            "api_outcomes": [c["outcome"] for c in api_calls],
            # TOKEN MEASUREMENT: the API-reported counts are the ground truth.
            "api_tokens_in": sum(c["tokens_in"] for c in api_calls),
            "api_tokens_out": sum(c["tokens_out"] for c in api_calls),
            "api_latency_ms": sum(c["latency_ms"] for c in api_calls),
            "api_cost_usd": round(sum(c["cost_usd"] or 0.0
                                      for c in api_calls), 6),
            # The rendered section sizes (estimate_tokens) for the §3 budgets.
            "section_tokens": sections,
            "snapshot_tokens_ledger": snapshot_refs.get(turn),
            "snapshot_within_8000": (
                sections["snapshot"] is not None
                and sections["snapshot"] <= 8000
                and snapshot_refs.get(turn, 0) <= 8000),
            "delta_within_800": (
                sections["delta"] is not None and sections["delta"] <= 800),
            "memory_within_1500": (
                sections["memory"] is not None and sections["memory"] <= 1500),
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
            f"| {t['section_tokens']['memory']} "
            f"| {t['section_tokens']['snapshot']} "
            f"(ledger {t['snapshot_tokens_ledger']}) "
            f"| {t['section_tokens']['delta']} | {t['candidate_count']} "
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
    tot_cost = sum(t["api_cost_usd"] for t in per_turn)
    lines += [
        f"| **total** | | {sum(t['api_calls'] for t in per_turn)} "
        f"| {tot_in} | {tot_out} | "
        f"{sum(t['api_latency_ms'] for t in per_turn)} "
        f"| {tot_cost:.6f} |",
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
    log(f"[{RUN}] repo={REPO}")
    log(f"[{RUN}] rundir={RUNDIR} attempt={ATTEMPT} seed={SEED} turns={TURNS}")
    log(f"[{RUN}] policy: mistral-large-4 via lechonk.model.turn_client "
        f"(json_schema, the WP2 spike decision decided live) over "
        f"core.runner's state machine")

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

    # --- the client (json_schema per the spike decision) + the adapter ------
    from lechonk.model.turn_client import MistralTurnClient

    client = mc.MistralClient(enforcement="json_schema")
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

    adapter = MistralTurnClient(client, api_log=api_log)
    log(f"[{RUN}] adapter: MistralTurnClient over MistralClient "
        f"(enforcement={client.enforcement.value}, model={client.model}, "
        f"temperature={client.temperature}, max_tokens={client.max_tokens})")

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

    log(f"[{RUN}] starting the run: {TURNS} model-driven turns, fresh "
        f"deterministic reset on the session's own server")
    run_t0 = time.time()
    result = runner.run()
    run_wall_s = time.time() - run_t0
    api_log_fh.close()
    ledger.close()
    log(f"[{RUN}] run finished: outcome={result.outcome} "
        f"reason={result.reason!r} turns_completed={result.turns_completed} "
        f"close_status={result.close_status} wall_clock={run_wall_s:.1f}s")

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
            "api_cost_usd": round(sum(t["api_cost_usd"] for t in per_turn),
                                  6),
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
            "tokens_in": sum(r["tokens_in"] for r in call_records),
            "tokens_out": sum(r["tokens_out"] for r in call_records),
            "cost_usd": round(sum(r["cost_usd"] or 0.0
                                  for r in call_records), 6),
            "latency_ms": sum(r["latency_ms"] for r in call_records),
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
    check("ledger_complete_10_turns",
          len(per_turn) == TURNS
          and all(t["turn"] in range(1, TURNS + 1) for t in per_turn),
          f"{len(per_turn)} turn_start records"
          + (f" (turns {per_turn[0]['turn']}..{per_turn[-1]['turn']})"
             if per_turn else ""))
    check("ledger_no_corruption",
          not load.discarded_trailing_partial,
          "no trailing partial record; every line checksum-verified")
    check("seq_contiguous",
          [r.seq for r in records] == list(range(len(records))),
          f"seq 0..{len(records) - 1}")
    check("run_completed_10_turns",
          result.outcome == "COMPLETED" and result.turns_completed == TURNS,
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
          f"{max((t['section_tokens']['snapshot'] for t in per_turn), default=0)}")
    check("delta_within_800_all_turns",
          all(t["delta_within_800"] for t in per_turn),
          f"max delta tokens: "
          f"{max((t['section_tokens']['delta'] for t in per_turn), default=0)}")
    check("memory_within_1500_all_turns",
          all(t["memory_within_1500"] for t in per_turn),
          f"max memory tokens: "
          f"{max((t['section_tokens']['memory'] for t in per_turn), default=0)}")
    check("per_call_api_logged",
          len(call_records) == sum(t["api_calls"] for t in per_turn)
          and all({"tokens_in", "tokens_out", "latency_ms", "cost_usd"}
                  <= set(r) for r in call_records),
          f"{len(call_records)} call records with tokens/latency/cost "
          f"(decision 4)")
    check("feasibility_inputs_recorded",
          feasibility["measured"]["mean_per_turn_wall_clock_s"] is not None
          and len(feasibility["wall_clock_budget"]["shapes"]) == 4,
          f"mean per-turn wall-clock: "
          f"{feasibility['measured']['mean_per_turn_wall_clock_s']} s")

    # --- run meta -------------------------------------------------------------
    verdict = "PASS" if checks and all(c["pass"] for c in checks.values()) \
        else "FAIL"
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
        "turns_target": TURNS,
        "turns_completed": result.turns_completed,
        "outcome": result.outcome,
        "reason": result.reason,
        "fc_args_overrides": {
            "self_play": False, "aifill": 2,
            "debug.randomly_generate_seeds": False,
            "debug.mapseed": SEED, "debug.gameseed": SEED,
            "debug.agentseed": SEED,
            "server_timeout": 300, "begin_turn_timeout": 900,
            "wait_for_timeout": 60, "http_timeout_s": 15.0,
        },
        "workdir": WORKDIR,
        "wall_clock_s": round(run_wall_s, 1),
        "api_totals": {
            "calls": len(call_records),
            "tokens_in": sum(r["tokens_in"] for r in call_records),
            "tokens_out": sum(r["tokens_out"] for r in call_records),
            "cost_usd": round(sum(r["cost_usd"] or 0.0
                                  for r in call_records), 6),
            "latency_ms": sum(r["latency_ms"] for r in call_records),
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
            "stdout": "run_stdout.log",
        },
        "checks": checks,
        "verdict": verdict,
    })

    log(f"[{RUN}] checks: {sum(1 for c in checks.values() if c['pass'])}/"
        f"{len(checks)} passed")
    log(f"[{RUN}] VERDICT: {verdict}")
    _log_file.close()

    # --- bounded retry on infrastructure failure (once, logged) --------------
    if (verdict == "FAIL" and result.outcome == "TRUNCATED"
            and ATTEMPT == 1):
        write_json("infra_failure_attempt1.json", {
            "run": RUN, "attempt": 1, "seed": SEED,
            "outcome": result.outcome, "reason": result.reason,
            "turns_completed": result.turns_completed,
            "checks": checks,
            "note": "bounded retry (once): re-executing the whole run with "
                    "WP2_MODEL_RUN_ATTEMPT=2; one env per process, so the "
                    "retry is a fresh process with a fresh env",
        })
        log(f"[{RUN}] bounded retry (once): re-executing with "
            f"WP2_MODEL_RUN_ATTEMPT=2")
        retry_env = dict(os.environ, WP2_MODEL_RUN_ATTEMPT="2")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                              env=retry_env)
        return proc.returncode
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
