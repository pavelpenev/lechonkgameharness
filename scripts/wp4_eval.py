# SPDX-License-Identifier: GPL-3.0-or-later
"""WP4 package A — the EVALUATION DRIVER: paired runs from identical
checkpoints, generalized from the WP3 paired smoke (design-plan-
freeciv-v1.1.md §4 line 58).

Per pair (PAIRS of them, target 2 x 50 turns; the cut ladder — second
pair → horizon 50→25 → scenario-cut — is applied by the orchestrator
BETWEEN pairs, never by cutting the same-start same-horizon pair):

  1. a fresh deterministic create game (comparator-driven, CREATE_TURNS
     turns) → the checkpoint exported at the start of turn
     CREATE_TURNS+1 via the WP3 export path (``lechonk.eval.checkpoint``,
     checksummed, preserved under artifacts/checkpoints/);
  2. the COMPARATOR arm (HORIZON turns from the identical exported
     checkpoint);
  3. the MODEL arm (the same horizon, the same predeclared paired
     contract, the same pinned username), SEQUENTIALLY, one env per
     process.

An arm that terminates early (DEFEATED/VICTORY before the horizon) is
REPORTED, never dropped; crashes/stalls stay separated from strategic
losses via the outcome taxonomy (TRUNCATED with reason vs
DEFEATED/VICTORY/COMPLETED, §3 line 34).

Reporting (§4 line 58): per pair and overall — score/turn curves
(descriptive), cities/population/research progress, survival, the
invalid-action taxonomy shares, retries (repair attempts), tokens/cost/
latency per turn, wall-clock per turn, all honesty metrics (the §3
participation gate + per-domain shares, commitment follow-through both
shares, reason-but-don't-act, illegal-action shares with the minimum-
denominator rule), provenance shares, and the AI-state divergence-over-
turns diagnostic per pair (§7.7). Persisted as metrics_report.json +
metrics_report.md per pair and overall under artifacts/runs/<run-id>/.

HONEST PILOT FRAMING (§4 line 58): descriptive reporting only — no
model-ranking claims anywhere in the generated artifacts (enforced at
render time by ``assert_no_ranking_language``).

Usage:
  .venv/bin/python scripts/wp4_eval.py                 # the full run
  .venv/bin/python scripts/wp4_eval.py create --pair-index 1
  .venv/bin/python scripts/wp4_eval.py arm --policy comparator --pair-index 1

Everything persists under ``artifacts/runs/<run-id>/``:
  eval_manifest.json         — the run-level manifest (parameters, pairs,
                               drift decisions, framing)
  metrics_report.json / .md  — the OVERALL report (all pairs)
  run_stdout.log             — this orchestrator's stdout
  p1/, p2/, …/               — per pair:
    pair_contract.json         — the PREDECLARED paired contract
    creation/                  — the fresh game + the exported checkpoint
    arms/comparator/           — transcript.jsonl, turns.jsonl (enriched),
                                 arm_manifest.json, replay/
    arms/model/                — the same + api_calls.jsonl
    divergence.json            — the §7.7 per-turn divergence curve
    pair_manifest.json         — the pair-level manifest
    metrics_report.json / .md  — the per-pair report
  spectate/                  — optional zero-perturbation observer
                               footage (the model arm of pair 1,
                               --spectate; the proven WP0 attach flow)

The model arm uses the live Mistral client (the key loads via the
existing loader — the value is never printed, logged, or artifacted;
only its presence is checked). An infrastructure failure is TRUNCATED
with artifacts — reported honestly, never retried blindly (decision 8).

GPL-3.0-or-later (decision 7): this script's arm/create phases import
the adapter package (the only CivRealm importer).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# --- run defaults -----------------------------------------------------------
DEFAULT_RUN_ID = "wp4-eval"
DEFAULT_SEED = 20261104   # distinct from the WP0/1/2/3 seeds
DEFAULT_PAIRS = 2         # the WP2-exit feasibility decision: 2 pairs
DEFAULT_HORIZON = 50      # x 50 turns
DEFAULT_CREATE_TURNS = 4  # the comparator plays turns 1..N; the checkpoint
#                             is exported at the start of turn N+1
CHECKPOINT_DIR = os.path.join(REPO, "artifacts", "checkpoints")
USERNAME = "myagent"      # pinned; the save/username coupling (§7.1)
AIFILL = 2
RULESET = "classic"
PUBSTATUS = "http://localhost:8080/pubstatus"

# Decision-8 manifest constants (the same pins as the WP3 paired smoke,
# verified against the running container 2026-10-24; the digest pin,
# decision 3).
IMAGE_DIGEST = ("sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a"
                "8e943faf920cb373a")
VENDORED_BASE_SHA = "1b144d6ec92a8aaa8e30eee64c881418208a4689"  # upstream dev
PATCH_HEAD_SHA = "36ae456836015b22b086298430048fc51c317d50"  # PR #7 fork head
CAPSTRING = "+Freeciv.Web.Devel-3.3"
SPIKE_ARTIFACT = os.path.join(
    REPO, "artifacts", "runs", "wp2-spike", "fingerprint_probe.json")
OBSERVE_JS = os.path.join(REPO, "scripts", "wp0_observe_attach.js")
_OBSERVE_OUT_LINE = ("const OUT = path.join(__dirname, '..', 'artifacts',"
                     " 'runs', 'wp0-observe');")

PILOT_FRAMING = (
    "Honest pilot evaluation: descriptive reporting only — no model-"
    "ranking claims. The two arms share an exported checkpoint but are "
    "not counterfactual-identical (plan §6: the AI opponents are not "
    "proven deterministic given fixed seeds). Infrastructure failures "
    "(TRUNCATED, with reason) are reported separately from strategic "
    "losses (DEFEATED/VICTORY); crashes and stalls are never scored as "
    "strategic losses.")

# The honest-pilot guard: generated artifacts must contain no
# model-ranking language (§4 line 58). Word-bounded, case-insensitive.
# Exemptions keep legitimate vocabulary out of the guard: the meta-phrase
# "model-ranking" (used by the framing itself), metric words like
# "window" (the ±2-turn reason-but-don't-act window) that merely contain
# a blocked stem, and the plan-mandated §3 line 38 gate-failure label
# "degraded-dominant" (a labeling rule, not a ranking claim — the
# lookbehind exempts only the hyphenated label; bare "dominant" stays
# blocked).
RANKING_TERMS_RE = re.compile(
    r"\b(winning|wins?|beats?\b|beat\b|outperform\w*|better|worse|best|"
    r"worst|(?<!model-)rank\w*|superior|inferior|(?<!degraded-)dominat\w*|"
    r"stronger|weaker)\b",
    re.IGNORECASE)


def assert_no_ranking_language(text: str, where: str) -> str:
    """§4 line 58: honest pilot framing — no model-ranking claims in any
    generated artifact. Raises with the offending term when one appears."""
    m = RANKING_TERMS_RE.search(text)
    if m:
        raise RuntimeError(
            f"{where}: model-ranking language {m.group(0)!r} is not "
            f"allowed in generated artifacts (honest pilot framing)")
    return text


# ---------------------------------------------------------------------------
# small utilities (the WP3 smoke patterns, proven)
# ---------------------------------------------------------------------------

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
                    log("pubstatus 200 — stack up")
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


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def read_turns_jsonl(path):
    """Read an arm's per-turn summaries file (None when absent)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# offline-computable analysis helpers (no live state; unit-tested)
# ---------------------------------------------------------------------------

OUTCOME_CLASS_STRATEGIC = "strategic_terminal"
OUTCOME_CLASS_INFRASTRUCTURE = "infrastructure_truncation"
OUTCOME_CLASS_COMPLETED = "horizon_completed"
OUTCOME_CLASS_UNKNOWN = "unknown"


def outcome_class(outcome):
    """Crashes/stalls stay separated from strategic losses (§3 line 34):
    TRUNCATED (with reason) is infrastructure; DEFEATED/VICTORY are
    strategic terminals; COMPLETED is the horizon."""
    if outcome == "TRUNCATED":
        return "infrastructure_truncation"
    if outcome in ("DEFEATED", "VICTORY"):
        return "strategic_terminal"
    if outcome == "COMPLETED":
        return "horizon_completed"
    return "unknown"


def survived(outcome):
    """Survival at/beyond the horizon: True for COMPLETED/VICTORY, False
    for DEFEATED, None (unknown — reported honestly) for TRUNCATED."""
    if outcome in ("COMPLETED", "VICTORY"):
        return True
    if outcome == "DEFEATED":
        return False
    return None


def per_turn_analysis(records):
    """The per-turn wall-clock / API / action table over one arm's
    transcript (the WP2 per-turn discipline, generalized): wall-clock
    from the ledger record timestamps (turn_start → turn_start, the last
    turn → the outcome record), API tokens/cost/latency per turn from
    the transcript's model_call records (repair attempts tracked
    separately), action/read-back counts per turn."""
    from lechonk.core.schemas import (
        ActionResultRecord, CandidatesRefRecord, CommitmentRecord,
        ModelCallRecord, OutcomeRecord, SnapshotRefRecord, TurnStartRecord)

    def diff_ts(a, b):
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)
                ).total_seconds()

    rows = []
    current = None
    for rec in records:
        if isinstance(rec, TurnStartRecord):
            if rows and rows[-1]["wall_clock_s"] is None:
                rows[-1]["wall_clock_s"] = round(
                    diff_ts(rows[-1]["ts_start"], rec.ts), 2)
            current = {
                "turn": rec.turn, "ts_start": rec.ts, "wall_clock_s": None,
                "api_calls": 0, "api_repair_attempts": 0,
                "api_tokens_in": 0, "api_tokens_out": 0,
                "api_latency_ms": 0, "api_cost_usd": 0.0,
                "actions_executed": 0, "actions_rejected": 0,
                "actions_aborted_group": 0, "rejections_by_class": {},
                "readbacks": {}, "provenance": {},
                "snapshot_tokens": None, "candidate_count": None,
                "opportunity_domains": [], "commitments": 0, "threats": 0,
            }
            rows.append(current)
        elif current is None:
            continue  # records before the first turn_start (the manifest)
        elif isinstance(rec, SnapshotRefRecord):
            current["snapshot_tokens"] = rec.token_count
        elif isinstance(rec, CandidatesRefRecord):
            current["candidate_count"] = rec.candidate_count
            current["opportunity_domains"] = list(rec.opportunity_domains)
        elif isinstance(rec, ModelCallRecord):
            current["api_calls"] += 1
            if rec.attempt == 2:
                current["api_repair_attempts"] += 1
            current["api_tokens_in"] += rec.tokens_in
            current["api_tokens_out"] += rec.tokens_out
            current["api_latency_ms"] += rec.latency_ms
            current["api_cost_usd"] += rec.cost_usd
        elif isinstance(rec, ActionResultRecord):
            key = rec.outcome  # executed | rejected | aborted_group
            current[f"actions_{key}"] = current.get(f"actions_{key}", 0) + 1
            if rec.outcome == "executed":
                current["readbacks"][rec.readback] = \
                    current["readbacks"].get(rec.readback, 0) + 1
                cls = rec.provenance_class
                current["provenance"][cls] = \
                    current["provenance"].get(cls, 0) + 1
            elif rec.outcome == "rejected" and rec.rejection_class:
                cls = rec.rejection_class
                current["rejections_by_class"][cls] = \
                    current["rejections_by_class"].get(cls, 0) + 1
        elif isinstance(rec, CommitmentRecord) and current:
            current["commitments"] += len(rec.commitments)
            current["threats"] += len(rec.threats)
        elif isinstance(rec, OutcomeRecord) and rows \
                and rows[-1]["wall_clock_s"] is None:
            rows[-1]["wall_clock_s"] = round(
                diff_ts(rows[-1]["ts_start"], rec.ts), 2)
    for row in rows:
        row["api_cost_usd"] = round(row["api_cost_usd"], 6)
    return rows


def progress_curves(summaries):
    """The descriptive per-turn progress curves from one arm's per-turn
    summaries (score/turn curves, cities/population/research progress,
    known tiles) — §4 line 58's metric list."""
    rows = sorted((s for s in summaries if s.get("turn") is not None),
                  key=lambda s: s["turn"])

    def column(key):
        return [r.get(key) for r in rows]

    return {
        "turns": column("turn"),
        "score": column("score"),
        "gold": column("gold"),
        "cities_own": column("cities_own"),
        "population": [sum((c.get("size") or 0)
                           for c in (r.get("cities") or [])
                           if isinstance(c, dict)) for r in rows],
        "units_own": column("units_own"),
        "techs_researched": column("techs_researched"),
        "research_target": column("research_target"),
        "research_turns_left": column("research_turns_left"),
        "tiles_known": column("tiles_known"),
    }


DIVERGENCE_DIMENSIONS = ("score", "cities_own", "units_own", "tiles_known")

DIVERGENCE_NOTE = (
    "descriptive curve only; the arms are NOT counterfactual-identical "
    "(plan §6: the AI opponents are not proven deterministic given fixed "
    "seeds) — no counterfactual claims are made from this curve")


def _diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b)


def divergence_curve(summaries_comparator, summaries_model):
    """The §7.7 AI-state divergence-over-turns diagnostic for ONE pair:
    the per-turn distance between the two arms over (score, city count,
    unit count, known tiles). DESCRIPTIVE CURVE ONLY — the arms are not
    counterfactual-identical (plan §6)."""
    by_turn_c = {s.get("turn"): s for s in summaries_comparator
                 if s.get("turn") is not None}
    by_turn_m = {s.get("turn"): s for s in summaries_model
                 if s.get("turn") is not None}
    common = sorted(set(by_turn_c) & set(by_turn_m))
    per_turn = []
    for turn in common:
        c = by_turn_c[turn]
        m = by_turn_m[turn]
        entry = {"turn": turn}
        for dim in DIVERGENCE_DIMENSIONS:
            entry[f"{dim}_diff"] = _diff(c.get(dim), m.get(dim))
        l1 = 0
        for dim in DIVERGENCE_DIMENSIONS:
            v = entry[f"{dim}_diff"]
            if v is None:
                l1 = None
                break
            l1 += v
        entry["l1"] = l1
        per_turn.append(entry)
    l1_vals = [e["l1"] for e in per_turn if e["l1"] is not None]
    summary = {
        "dimensions": list(DIVERGENCE_DIMENSIONS),
        "turns_compared": len(common),
        "turns_only_in_comparator": sorted(set(by_turn_c) - set(by_turn_m)),
        "turns_only_in_model": sorted(set(by_turn_m) - set(by_turn_c)),
        "mean_l1": (round(sum(l1_vals) / len(l1_vals), 3)
                    if l1_vals else None),
        "max_l1": (max(l1_vals) if l1_vals else None),
    }
    for dim in DIVERGENCE_DIMENSIONS:
        vals = [e[f"{dim}_diff"] for e in per_turn
                if e[f"{dim}_diff"] is not None]
        summary[f"mean_{dim}_diff"] = (round(sum(vals) / len(vals), 3)
                                       if vals else None)
        summary[f"max_{dim}_diff"] = max(vals) if vals else None
    return {
        "diagnostic": "AI-state divergence over turns (§7.7): the per-turn "
                      "distance between the pair's arms over (score, city "
                      "count, unit count, known tiles)",
        "note": DIVERGENCE_NOTE,
        "dimensions": list(DIVERGENCE_DIMENSIONS),
        "known_tiles_definition": "map_view.tile_counts visible + last_known "
                                  "(the policy-facing fog trichotomy)",
        "per_turn": per_turn,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# the enriched per-turn recorder — the base turn summary does not carry
# the known-tile counts the §7.7 divergence needs; this wrapper reuses
# paired.turn_summary_from_prompt (the proven parser) and adds
# map_view.tile_counts (visible + last_known). The replay viewer reads
# turns.jsonl and tolerates the extra keys.
# ---------------------------------------------------------------------------

class TurnSummaryRecorder:
    """Wrap any ModelClient and persist the per-turn summary
    (``turns.jsonl``) enriched with the known-tile counts (§7.7
    divergence input). Transparent: provenance_class and both client
    calls pass through."""

    def __init__(self, inner, turns_path):
        self._inner = inner
        self._turns_path = str(turns_path)
        Path(self._turns_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._turns_path, "a", encoding="utf-8")
        self._summaries = []

    @property
    def provenance_class(self):
        return getattr(self._inner, "provenance_class", "model")

    @staticmethod
    def _enrich(summary, snapshot_text):
        """Add tiles_visible/tiles_last_known/tiles_known from the
        snapshot's map_view.tile_counts (absent → None, never 0)."""
        try:
            snap = json.loads(snapshot_text)
            counts = (snap.get("map_view") or {}).get("tile_counts") or {}
        except (ValueError, TypeError):
            counts = {}
        visible = counts.get("visible")
        last_known = counts.get("last_known")
        summary["tiles_visible"] = visible
        summary["tiles_last_known"] = last_known
        summary["tiles_known"] = (visible + last_known
                                  if visible is not None
                                  and last_known is not None else None)

    def complete_turn(self, prompt):
        from lechonk.eval import paired  # noqa: PLC0415 — lazy (module
        # level imports stay CivRealm-free; lechonk.eval is eval-owned)
        resp = self._inner.complete_turn(prompt)
        summary = paired.turn_summary_from_prompt(
            prompt, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if summary is not None:
            self._enrich(summary, prompt.snapshot)
            self._summaries.append(summary)
            self._fh.write(json.dumps(summary, sort_keys=True,
                                      ensure_ascii=False, default=str)
                           + "\n")
            self._fh.flush()
        return resp

    def complete_repair(self, prompt):
        return self._inner.complete_repair(prompt)

    def summaries(self):
        return list(self._summaries)

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()


# ---------------------------------------------------------------------------
# OPTIONAL spectate (--spectate): the proven WP0 zero-perturbation
# observer (scripts/wp0_observe_attach.js) spawned against the model
# arm's civserver port during pair 1. The script's output directory is
# hardcoded to the WP0 run dir — the source is patched at spawn time to
# record under <run-root>/spectate/ instead. NEVER fails the arm: any
# problem is logged and skipped (WP5's primary video source is the
# replay viewer, which can re-record separately).
# ---------------------------------------------------------------------------

def spawn_observer(port, spectate_dir, log):
    """Spawn the WP0 observer (non-blocking) recording into
    ``spectate_dir``; returns the pid or None (skipped)."""
    try:
        os.makedirs(spectate_dir, exist_ok=True)
        with open(OBSERVE_JS, encoding="utf-8") as fh:
            source = fh.read()
        if _OBSERVE_OUT_LINE not in source:
            log(f"WARN: spectate skipped — {OBSERVE_JS} no longer carries "
                f"the expected OUT line")
            return None
        patched = source.replace(
            _OBSERVE_OUT_LINE, f"const OUT = {json.dumps(spectate_dir)};")
        js_path = os.path.join(spectate_dir, "observer_attach_wp4.js")
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write(patched)
        log_path = os.path.join(spectate_dir, "observer_stdout.log")
        log_fh = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            ["node", js_path, str(port)], stdout=log_fh,
            stderr=subprocess.STDOUT, start_new_session=True)
        log(f"spectate: zero-perturbation observer spawned (pid {proc.pid}, "
            f"port {port}) — footage under {spectate_dir} (zero-"
            f"perturbation proven at WP0-6)")
        return proc.pid
    except Exception as exc:
        log(f"WARN: spectate skipped: {exc!r}")
        return None


class _SpectateSessionProxy:
    """Transparent session proxy that spawns the observer once the arm's
    session has started (the civserver port exists only after start())."""

    def __init__(self, inner, spectate_dir, log):
        self._inner = inner
        self._spectate_dir = spectate_dir
        self._log = log
        self._spawned = False

    def start(self):
        result = self._inner.start()
        if not self._spawned:
            self._spawned = True
            spawn_observer(self._inner.port, self._spectate_dir, self._log)
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# run-layout helpers
# ---------------------------------------------------------------------------

def run_root_for(run_id):
    return os.path.join(REPO, "artifacts", "runs", run_id)


def pair_dir_for(run_id, pair_index):
    return os.path.join(run_root_for(run_id), f"p{pair_index}")


def pair_id_for(run_id, pair_index):
    return f"{run_id}-p{pair_index}"


def pair_seed(seed, pair_index):
    """The per-pair create-game seed: base seed + (pair_index - 1) —
    distinct fresh games per pair, each deterministic (recorded in the
    creation manifest and the pair manifest)."""
    return seed + pair_index - 1


# ===========================================================================
# create — the fresh deterministic game + the checkpoint export (per pair)
# ===========================================================================

def cmd_create(run_id, pair_index, seed, create_turns, horizon):
    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    pair_id = pair_id_for(run_id, pair_index)
    pair_dir = pair_dir_for(run_id, pair_index)
    creation_dir = os.path.join(pair_dir, "creation")
    workdir = os.path.join(creation_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    log = RunLogger(os.path.join(creation_dir, "run_stdout.log"))
    export_turn = create_turns + 1   # the begin-turn /save for turn N+1 has
    #                                     fired; the end-turn HTTP deletion
    #                                     has not fired
    log(f"[{pair_id}/create] fresh deterministic game, seed={seed}, "
        f"creation turns={create_turns}, export at the start of turn "
        f"{export_turn} (horizon {horizon})")
    os.chdir(workdir)  # dedicated CWD for upstream side effects

    if not wait_for_stack(log):
        log.close()
        return 2

    # The adapter imports lazily (civrealm touches the network at import
    # time); only now that the stack is up.
    from lechonk.adapter import execute as exe
    from lechonk.adapter.obs_extract import ExtractOutcome, extract_snapshot
    from lechonk.adapter.session import AdapterSession
    from lechonk.comparator import scripted

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
        session = AdapterSession(workdir=workdir, seed=seed,
                                 username=USERNAME, aifill=AIFILL)
        t0 = time.time()
        session.start()
        log(f"reset ok in {time.time() - t0:.1f}s turn={session.turn} "
            f"player_id={session.player_id} username={session.username} "
            f"port={session.port}")
        check("reset_turn_1", session.turn == 1, f"turn={session.turn}")
        check("username_pinned", session.username == USERNAME,
              f"username={session.username!r} (self_play=False: congestion "
              f"must raise, never silently rename)")

        # --- play the creation turns with the comparator (the WP1
        # pattern, incl. the mid-batch revalidation: a stale candidate is
        # rejected and NEVER stepped — stepping a client-side-invalid
        # action sends nothing to the server, and the env's state read
        # then blocks until the next connection ping)
        for turn_no in range(1, create_turns + 1):
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

        check("at_export_turn", session.turn == export_turn,
              f"turn={session.turn} (the begin-turn /save for turn "
              f"{export_turn} has fired; the end-turn HTTP deletion has "
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
        log(f"state at export (turn {export_turn} start): "
            f"score={state_at_export and state_at_export.get('score')} "
            f"gold={state_at_export and state_at_export.get('gold')} "
            f"cities={state_at_export and state_at_export.get('cities_own')} "
            f"units={state_at_export and state_at_export.get('units_own')}")

        # --- the WP3 checkpoint export (our own copy-after-completion) ----
        t0 = time.time()
        export = cp.export_turn_save(
            username=USERNAME, turn=export_turn, dest_dir=CHECKPOINT_DIR)
        log(f"export ok in {time.time() - t0:.1f}s: {export.path} "
            f"sha256={export.sha256} save_name={export.save_name} "
            f"size={export.size_bytes} candidates={export.candidates}")
        check("export_checksum_compare",
              export.container_sha256 == export.local_sha256 == export.sha256,
              f"container={export.container_sha256} "
              f"local={export.local_sha256}")
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
        "pair_id": pair_id,
        "pair_index": pair_index,
        "run_id": run_id,
        "timestamp_utc": utcnow(),
        "seed": seed,
        "username": USERNAME,
        "aifill": AIFILL,
        "ruleset": RULESET,
        "creation_turns": create_turns,
        "export_turn": export_turn,
        "horizon": horizon,
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

def cmd_arm(run_id, pair_index, policy, spectate):
    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    pair_id = pair_id_for(run_id, pair_index)
    pair_dir = pair_dir_for(run_id, pair_index)
    arm_dir = os.path.join(pair_dir, "arms", policy)
    workdir = os.path.join(arm_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    log = RunLogger(os.path.join(arm_dir, "run_stdout.log"))
    log(f"[{pair_id}/arm:{policy}] arm start; loading the exported "
        f"checkpoint under the predeclared paired contract")
    os.chdir(workdir)  # dedicated CWD for upstream side effects

    contract = read_json(os.path.join(pair_dir, "pair_contract.json"))
    creation = read_json(os.path.join(pair_dir, "creation",
                                      "creation_manifest.json"))
    checkpoint = creation["checkpoint"]
    save_name = checkpoint["save_name"]
    expected_sha = checkpoint["sha256"]
    export_turn = checkpoint["turn"]
    seed = creation["seed"]
    arm_turns = contract["arm_turns"]
    horizon_turn = contract["horizon_turn"]
    log(f"contract: version={contract['contract_version']} "
        f"sha256={paired.contract_sha256(contract)} arm_turns={arm_turns} "
        f"horizon_turn={horizon_turn}")
    log(f"checkpoint: {checkpoint['path']} sha256={expected_sha} "
        f"turn={export_turn} save_name={save_name}")

    if not wait_for_stack(log):
        log.close()
        return 2

    checks = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    # OPTIONAL spectate: the model arm of pair 1 only.
    spectate_active = bool(spectate) and pair_index == 1 and policy == "model"
    spectate_dir = os.path.join(run_root_for(run_id), "spectate")

    run_started = time.time()
    arm_start_ts = utcnow()
    run_wall_s = None
    transcript_path = os.path.join(arm_dir, "transcript.jsonl")
    turns_path = os.path.join(arm_dir, "turns.jsonl")
    api_log_path = os.path.join(arm_dir, "api_calls.jsonl")
    # A fresh arm run starts with fresh files: the ledger appends BY
    # DESIGN (durability/torn-tail recovery), so a rerun of this phase
    # must not mix records across orchestrator attempts.
    for fresh_path in (transcript_path, turns_path, api_log_path):
        if os.path.exists(fresh_path):
            os.remove(fresh_path)
    error_text = ""
    result = None
    staging = None
    fingerprint = None
    fingerprint_evidence = None
    recording = None
    ledger = None
    api_log_fh = None
    try:
        # --- stage the EXPORTED checkpoint into the container's save dir
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
        log(f"fc_args[debug.load_game] = {save_name!r} (the save/username "
            f"coupling: the load validates the save's username prefix)")

        # --- the arm's policy client -------------------------------------
        session_box = {}

        def session_factory():
            from lechonk.adapter.session import AdapterSession
            s = AdapterSession(
                workdir=workdir, seed=seed, username=USERNAME, aifill=AIFILL,
                startup_deadline=150.0, step_deadline=30.0,
                wait_deadline=900.0, close_deadline=30.0)
            session_box["session"] = s
            if spectate_active:
                return _SpectateSessionProxy(
                    s, os.path.join(run_root_for(run_id), "spectate"), log)
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
            status = ("match" if reference is None or fingerprint == reference
                      else "mismatch")
            if reference is not None and fingerprint != reference:
                probe2 = client.probe_fingerprint()
                re_probe = probe2.fingerprint
                status = ("match_after_reprobe"
                          if re_probe == reference else "mismatch_persistent")
            log(f"arm fingerprint: {fingerprint} (reference={reference}, "
                f"status={status}, re_probe={re_probe})")
            # The probe is an API call too (decision 4) — logged with
            # kind="probe" (the WP2 logging pattern). The first probe's
            # result is a FingerprintProbe (no call_id/enforcement);
            # a re-probe's record is the underlying CallLog (attempt 0).
            probe_records = [("probe", probe1.to_dict())]
            if re_probe is not None:
                probe_records.append(("probe_reprobe",
                                      client.call_logs[-1].to_dict()))
            for kind, pl in probe_records:
                api_log({"call_id": pl.get("call_id", "probe"),
                         "turn": None, "attempt": 0,
                         "kind": kind, "prompt_hash": pl["prompt_hash"],
                         "model_requested": pl["model_requested"],
                         "model_responded": pl["model_responded"],
                         "enforcement": pl.get("enforcement"),
                         "temperature": pl["temperature"],
                         "latency_ms": pl["latency_ms"],
                         "tokens_in": pl["tokens_in"],
                         "tokens_out": pl["tokens_out"],
                         "tokens_total": pl["tokens_total"],
                         "cost_usd": pl["cost_usd"],
                         "finish_reason": pl["finish_reason"],
                         "response_id": pl["response_id"],
                         "fingerprint": pl["fingerprint"],
                         "transport_retries": pl.get("transport_retries"),
                         "error": pl.get("error"), "outcome": "probe",
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
            sampling = {
                "temperature": 0.0, "max_tokens": mc.DEFAULT_MAX_TOKENS,
                "random_seed": None, "top_p": 1.0,
                "response_format": "json_schema strict (WP2 spike decision)",
            }
            prompt_hashes = prompts.prompt_manifest_hashes()
            enforcement_decision = mc.spike_decision(
                live_probe_done=True, json_schema_supported=True,
                evidence_artifact="artifacts/runs/wp2-spike/spike_report.json")
            action_batch_schema_hash = mc.action_batch_schema_hash()
            log(f"policy client: MistralTurnClient over MistralClient "
                f"(enforcement=json_schema, model={client.model})")

        # --- the arm manifest (decision 8 + the paired extras) -----------
        manifest = paired.build_arm_manifest(
            arm=policy, pair_id=pair_id, contract=contract,
            checkpoint=checkpoint, arm_start_ts=arm_start_ts,
            model_fingerprint=fingerprint,
            fingerprint_evidence=fingerprint_evidence,
            harness_rev=paired.harness_revision(REPO),
            seeds={"mapseed": seed, "gameseed": seed, "agentseed": seed},
            username=USERNAME, aifill=AIFILL,
            image_digest=IMAGE_DIGEST, capstring=CAPSTRING,
            vendored_base_sha=VENDORED_BASE_SHA, patch_head_sha=PATCH_HEAD_SHA,
            model_id=model_id, sampling=sampling, prompt_hashes=prompt_hashes,
            enforcement_decision=enforcement_decision,
            action_batch_schema_hash=action_batch_schema_hash,
            run_id=f"{pair_id}/{policy}")

        # --- the runner (the SAME pipeline for both arms) ------------------
        from lechonk.core.ledger import TranscriptLedger
        from lechonk.core.runner import GameRunner, RunnerConfig

        recording = TurnSummaryRecorder(policy_client, turns_path)
        ledger = TranscriptLedger(transcript_path)
        config = RunnerConfig(
            max_turns=horizon_turn, run_id=f"{pair_id}/{policy}",
            workdir=workdir, seed=seed, username=USERNAME, aifill=AIFILL)
        runner = GameRunner(
            config=config, client=recording, ledger=ledger,
            manifest=manifest, session_factory=session_factory)
        log(f"runner: max_turns={horizon_turn} (the arm plays turns "
            f"{export_turn}..{horizon_turn} from the loaded checkpoint)")
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

    # --- post-run: transcript analysis, metrics, the replay bundle ------
    arm_end_ts = utcnow()
    post = {"transcript_records": None, "turn_starts": None,
            "outcome_record": None, "load_verification": None,
            "provenance_counts": None, "readback_counts": None,
            "api_totals": None, "per_turn": None, "repair_attempts": None,
            "wall_clock_total_s": None, "progress": None,
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

        # The per-turn wall-clock / API / action table + retries.
        per_turn_rows = per_turn_analysis(records)
        post["per_turn"] = per_turn_rows
        post["repair_attempts"] = sum(r["api_repair_attempts"]
                                      for r in per_turn_rows)
        post["wall_clock_total_s"] = round(sum(
            r["wall_clock_s"] for r in per_turn_rows
            if r["wall_clock_s"] is not None), 2)

        # The descriptive progress curves from the arm's summaries.
        post["progress"] = progress_curves(recording.summaries())

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
        bundle = render_bundle(transcript_path,
                               os.path.join(arm_dir, "replay"),
                               turns_path=turns_path,
                               title=f"Le Chonk replay — {pair_id}/{policy}")
        post["replay_bundle"] = bundle["bundle_dir"]
        log(f"replay bundle: {bundle['bundle_dir']}")

    early = None
    if result is not None:
        early = paired.early_terminal_report(
            result.outcome, result.reason, result.turns_completed, arm_turns)
    outcome_final = result.outcome if result else None
    arm_manifest = {
        "phase": "arm",
        "pair_id": pair_id,
        "pair_index": pair_index,
        "run_id": run_id,
        "arm": policy,
        "policy": ("lechonk.comparator.scripted.decide behind "
                   "lechonk.eval.paired.ScriptedComparatorClient"
                   if policy == "comparator" else
                   "mistral-large-4 (lechonk.model.turn_client over "
                   "MistralClient, json_schema enforcement — the WP2 spike "
                   "decision) behind the WP4 TurnSummaryRecorder"),
        "timestamp_utc": utcnow(),
        "arm_start_ts": arm_start_ts,
        "arm_end_ts": arm_end_ts,
        "wall_clock_s": round(time.time() - run_started, 1),
        "run_wall_s": (round(run_wall_s, 1)
                       if run_wall_s is not None else None),
        "seed": seed,
        "username": USERNAME,
        "run_id": f"{pair_id}/{policy}",
        "paired_contract_sha256": paired.contract_sha256(contract),
        "checkpoint": {
            "path": checkpoint["path"],
            "save_name": save_name,
            "sha256": expected_sha,
            "turn": export_turn,
            "username": USERNAME,
            "staged": (staging.to_record() if staging else None),
        },
        "turns_planned": arm_turns,
        "horizon_turn": horizon_turn,
        "outcome": outcome_final,
        "reason": result.reason if result else error_text or None,
        "outcome_class": outcome_class(outcome_final),
        "survived": survived(outcome_final),
        "turns_completed": result.turns_completed if result else 0,
        "close_status": result.close_status if result else None,
        "early_terminal": early is not None,
        "early_terminal_report": early,
        "model_fingerprint": fingerprint,
        "fingerprint_evidence": fingerprint_evidence,
        "transcript": transcript_path,
        "turns": turns_path,
        "api_calls_log": api_log_path if policy == "model" else None,
        "spectate": (spectate_active or None),
        "post_run": post,
        "checks": checks,
        "error": error_text or None,
    }
    assert_no_ranking_language(
        json.dumps(arm_manifest, default=str), f"{pair_id}/{policy} manifest")
    write_json(os.path.join(arm_dir, "arm_manifest.json"), arm_manifest)
    log(f"arm manifest written; outcome={arm_manifest['outcome']} "
        f"turns={arm_manifest['turns_completed']}/{arm_turns} "
        f"early_terminal={arm_manifest['early_terminal']}")
    log.close()
    return 0


# ===========================================================================
# orchestrator — per pair: create -> comparator arm -> model arm ->
# pair manifest + divergence; then the overall report
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
    preserved in the orchestrator log. Before the retry the driver
    settles 25 s: the failed attempt's game server quits ~20 s after its
    client disconnects (--quitidle 20, HANDOFF) and attaching to the
    stale in-progress server races CivRealm's begin-turn handling."""
    rc = run_subprocess(log, args, log_path)
    if rc != 0:
        log(f"phase {' '.join(args)} rc={rc} — bounded retry (once, fresh "
            f"process; the WP1 infra-retry discipline, never a silent loop)")
        log("settling 25 s before the retry (--quitidle 20 server teardown)")
        time.sleep(25)
        rc = run_subprocess(log, args, log_path)
    return rc


def provenance_shares(counts):
    """Executed-action provenance counts → shares (empty when no
    executed actions — never fabricated)."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def crash_strategic_split(arms_section):
    """The crash-vs-strategic split (§3 line 34): TRUNCATED arms are
    infrastructure, never scored as strategic losses."""
    by_class = {}
    for arm, entry in arms_section.items():
        by_class.setdefault(
            entry.get("outcome_class") or outcome_class(entry.get("outcome")),
            []).append(arm)
    return {
        "infrastructure_truncations":
            by_class.get("infrastructure_truncation", []),
        "strategic_terminals": by_class.get("strategic_terminal", []),
        "horizon_completed": by_class.get("horizon_completed", []),
        "note": "crashes/stalls (TRUNCATED) are never scored as strategic "
                "losses (§3 line 34)",
    }


DEGRADED_DOMINANT_CONSEQUENCE = (
    "plan §3 line 38: the run is reported in full but labeled "
    "degraded-dominant and excluded from model-performance claims")


def degraded_dominant_label(model_gate_pass):
    """The §3 line 38 labeling rule: a FAILED participation gate labels
    the run ``degraded-dominant`` (reported in full, excluded from
    model-performance claims); a passing (or n/a) gate is not labeled."""
    if model_gate_pass is False:
        return "degraded-dominant"
    return None


def transport_timeout_summary(api_calls_log):
    """The exit-review INFO item: the model arm's turns lost to transport
    read-timeouts, summarized from api_calls.jsonl (None when the log is
    absent — e.g. the scripted comparator arm)."""
    if not api_calls_log or not os.path.exists(api_calls_log):
        return None
    with open(api_calls_log, encoding="utf-8") as fh:
        calls = [json.loads(line) for line in fh if line.strip()]
    initial_to = sorted({r.get("turn") for r in calls
                         if r.get("attempt") == 1
                         and r.get("outcome") == "transport_error"
                         and r.get("turn") is not None})
    repair_to = sorted({r.get("turn") for r in calls
                        if r.get("attempt") == 2
                        and r.get("outcome") == "transport_error"
                        and r.get("turn") is not None})
    lost = sorted(set(initial_to) | (set(repair_to) - set(initial_to)))
    return {
        "error": "transport failure: The read operation timed out",
        "timeout_calls_total": len([r for r in calls
                                    if r.get("outcome") == "transport_error"]),
        "turn_calls_total": len([r for r in calls
                                 if r.get("attempt") in (1, 2)]),
        "initial_attempts_timed_out": len(initial_to),
        "initial_timeout_turns": initial_to,
        "repair_attempts_timed_out": len([r for r in calls
                                          if r.get("attempt") == 2
                                          and r.get("outcome")
                                          == "transport_error"]),
        "repair_timeout_turns": repair_to,
        "turns_lost_to_read_timeouts": lost,
        "note": "each timed-out call left its turn to the degraded default "
                "(no model participation that turn); a repair attempt "
                "follows the failed initial attempt per the "
                "<=2-attempts-per-turn cap",
    }


def pair_metrics_report(*, pair_id, run_id, pair_index, parameters,
                        checkpoint, contract_sha256, arm_manifests,
                        summaries_comparator, summaries_model, divergence,
                        drift, checks, verdict):
    """The per-pair metrics report (§4 line 58's metric list) assembled
    offline from the persisted arm manifests + per-turn summaries."""
    arms = {}
    for arm in ("comparator", "model"):
        m = arm_manifests.get(arm) or {}
        post = m.get("post_run") or {}
        outcome = m.get("outcome")
        per_turn = post.get("per_turn") or []
        metrics = post.get("metrics") or {}
        summaries = (summaries_comparator if arm == "comparator"
                     else summaries_model) or []
        entry = {
            "arm": arm,
            "policy": m.get("policy"),
            "run_id": m.get("run_id"),
            "outcome": outcome,
            "reason": m.get("reason"),
            "outcome_class": (m.get("outcome_class") or outcome_class(outcome)),
            "survived": (m.get("survived")
                         if m.get("survived") is not None
                         else survived(outcome)),
            "turns_completed": m.get("turns_completed"),
            "turns_planned": m.get("turns_planned"),
            "early_terminal": m.get("early_terminal"),
            "early_terminal_report": m.get("early_terminal_report"),
            "wall_clock_s": m.get("wall_clock_s"),
            "transcript": m.get("transcript"),
            "replay_bundle": post.get("replay_bundle"),
            "progress_curves": progress_curves(summaries),
            "wall_clock_per_turn": [[r["turn"], r["wall_clock_s"]]
                                    for r in per_turn],
            "retries": {"repair_attempts": post.get("repair_attempts")
                        if post.get("repair_attempts") is not None
                        else sum(r["api_repair_attempts"] for r in per_turn)},
            "invalid_action_taxonomy": (metrics.get("illegal_action_shares")
                                        or {}),
            "honesty_metrics": metrics,
            "provenance_counts": post.get("provenance_counts") or {},
            "provenance_shares": provenance_shares(
                post.get("provenance_counts") or {}),
            "model_fingerprint": m.get("model_fingerprint"),
            "api_totals": post.get("api_totals") if arm == "model" else None,
        }
        if arm == "model":
            entry["api_per_turn"] = [
                [r["turn"], r["api_tokens_in"], r["api_tokens_out"],
                 r["api_latency_ms"], r["api_cost_usd"]] for r in per_turn]
            entry["transport_timeouts"] = transport_timeout_summary(
                m.get("api_calls_log"))
        arms[arm] = entry
    model_gate_pass = (arms["model"]["honesty_metrics"] or {}).get(
        "participation_gate_pass")
    degraded = degraded_dominant_label(model_gate_pass)
    return {
        "pair_id": pair_id,
        "run_id": run_id,
        "pair_index": pair_index,
        "generated_by": "scripts/wp4_eval.py pair_metrics_report (WP4 "
                        "package A; design-plan-freeciv-v1.1.md §4 line 58)",
        "generated_at": utcnow(),
        "pilot_framing": PILOT_FRAMING,
        "parameters": dict(parameters),
        "checkpoint": dict(checkpoint),
        "contract_sha256": contract_sha256,
        "degraded_dominant": degraded,
        "degraded_dominant_consequence": (
            DEGRADED_DOMINANT_CONSEQUENCE if degraded else None),
        "outcome_taxonomy": {
            arm: {"outcome": arms[arm]["outcome"],
                  "class": arms[arm]["outcome_class"],
                  "survived": arms[arm]["survived"],
                  "reason": arms[arm]["reason"]}
            for arm in arms},
        "crash_vs_strategic": crash_strategic_split(arms),
        "arms": arms,
        "divergence": dict(divergence),
        "drift": dict(drift),
        "checks": dict(checks),
        "verdict": verdict,
    }


def render_pair_report_md(r):
    """The human-readable per-pair report (descriptive only)."""
    lines = [
        f"# WP4 evaluation — {r['pair_id']} "
        f"(pair {r['pair_index']}/{r['parameters']['pairs']})",
        "",
        f"Generated: {r['generated_at']}",
        "",
        f"**{r['pilot_framing']}**",
        "",
        "## Parameters",
        "",
        f"- horizon: {r['parameters']['horizon']} turns per arm · create "
        f"turns: {r['parameters']['create_turns']} · pair seed: "
        f"{r['parameters']['seed']} (base {r['parameters']['seed_base']} "
        f"+ index − 1)",
        f"- username: {r['parameters']['username']} (pinned) · aifill: "
        f"{r['parameters']['aifill']} · ruleset: {r['parameters']['ruleset']}",
        f"- checkpoint: `{r['checkpoint']['path']}` · sha256: "
        f"`{r['checkpoint']['sha256']}` (turn {r['checkpoint']['turn']})",
        f"- predeclared paired contract sha256: `{r['contract_sha256']}`",
        "",
        "## Outcomes (crashes/stalls separated from strategic losses, "
        "§3 line 34)",
        "",
        "| arm | outcome | class | survival | turns | early terminal | "
        "wall-clock s |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("comparator", "model"):
        a = r["arms"][arm]
        lines.append(
            f"| {arm} | {a['outcome']} | {a['outcome_class']} "
            f"| {a['survived']} | {a['turns_completed']}/"
            f"{a['turns_planned']} | {a['early_terminal']} "
            f"| {a['wall_clock_s']} |")
    lines += [
        "",
        f"- model arm reason: {r['arms']['model']['reason']}",
        f"- comparator arm reason: {r['arms']['comparator']['reason']}",
        f"- crash-vs-strategic split: {json.dumps(r['crash_vs_strategic'])}",
    ]
    if r.get("degraded_dominant"):
        lines.append(
            f"- participation gate: FAIL — pair labeled "
            f"**{r['degraded_dominant']}** "
            f"({r['degraded_dominant_consequence']})")
    else:
        lines.append(
            "- participation gate: PASS — not labeled (§3 line 38)")
    tt = r["arms"]["model"].get("transport_timeouts")
    if tt and tt.get("timeout_calls_total"):
        initial_turns = ", ".join(str(t) for t in tt["initial_timeout_turns"])
        repair_turns = ", ".join(str(t) for t in tt["repair_timeout_turns"])
        lines.append(
            f"- model-arm transport read-timeouts: "
            f"{tt['timeout_calls_total']} of {tt['turn_calls_total']} turn "
            f"calls failed with {tt['error']!r} "
            f"({tt['initial_attempts_timed_out']} initial attempts on turns "
            f"{initial_turns}; {tt['repair_attempts_timed_out']} repair "
            f"attempts on turns {repair_turns}) — each such turn fell to "
            f"the degraded default, the main driver of the "
            f"{r['arms']['model']['honesty_metrics'].get('participation', {}).get('value')} "
            f"model participation below the 0.80 gate")
    lines += [
        "",
        "## Progress curves (descriptive)",
        "",
    ]
    for arm in ("comparator", "model"):
        p = r["arms"][arm]["progress_curves"]
        lines += [
            f"### {arm}",
            "",
            "| turn | score | cities | population | units | research | "
            "techs | known tiles |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, turn in enumerate(p["turns"]):
            lines.append(
                f"| {turn} | {p['score'][i]} | {p['cities_own'][i]} "
                f"| {p['population'][i]} | {p['units_own'][i]} "
                f"| {p['research_target'][i]} "
                f"({p['research_turns_left'][i]} left) "
                f"| {p['techs_researched'][i]} | {p['tiles_known'][i]} |")
        lines.append("")
    m = r["arms"]["model"]
    if m.get("api_per_turn"):
        lines += [
            "## Model arm — wall-clock / tokens / cost per turn",
            "",
            "| turn | wall-clock s | tokens in | tokens out | latency ms | "
            "cost $ |",
            "|---|---|---|---|---|---|",
        ]
        wall_by_turn = dict(m["wall_clock_per_turn"])
        for turn, tin, tout, lat, cost in m["api_per_turn"]:
            lines.append(
                f"| {turn} | {wall_by_turn.get(turn)} | {tin} | {tout} "
                f"| {lat} | {cost:.6f} |")
        tot = m.get("api_totals") or {}
        lines += [
            f"| **total** | | {tot.get('calls')} | {tot.get('tokens_in')} "
            f"| {tot.get('tokens_out')} | {tot.get('cost_usd')} |",
            f"",
            f"- retries (repair attempts): "
            f"{m['retries']['repair_attempts']}",
            "",
        ]
    lines += [
        "## Invalid-action taxonomy (§3 line 43; minimum-denominator rule)",
        "",
    ]
    for arm in ("comparator", "model"):
        tax = r["arms"][arm]["invalid_action_taxonomy"]
        if not tax:
            continue
        lines.append(f"### {arm}")
        for cls, share in tax.items():
            lines.append(
                f"- {cls}: {share['numerator']}/{share['denominator']} = "
                f"{share['value']} ({share['status']})")
    lines += [
        "",
        "## Honesty metrics (per arm; §3 frozen definitions)",
        "",
    ]
    for arm in ("comparator", "model"):
        hm = r["arms"][arm]["honesty_metrics"] or {}
        if not hm:
            continue
        lines.append(f"### {arm}")
        # every honesty field is rendered via .get(): an absent or
        # None-valued field (an honest gap) renders as null, it never
        # crashes the report
        lines.append(f"- participation: {json.dumps(hm.get('participation'))}")
        lines.append(f"- participation gate pass: "
                     f"{hm.get('participation_gate_pass')}")
        lines.append(f"- domain model shares: "
                     f"{json.dumps(hm.get('domain_model_shares'))}")
        lines.append(f"- commitment follow-through (any): "
                     f"{json.dumps(hm.get('commitment_followthrough_any'))}")
        lines.append(f"- commitment follow-through (model): "
                     f"{json.dumps(hm.get('commitment_followthrough_model'))}")
        lines.append(f"- reason-but-don't-act: "
                     f"{json.dumps(hm.get('reason_but_dont_act'))}")
        lines.append(f"- illegal-action shares: "
                     f"{json.dumps(hm.get('illegal_action_shares'))}")
        lines.append("")
    lines += [
        "## Provenance shares (executed actions)",
        "",
    ]
    for arm in ("comparator", "model"):
        a = r["arms"][arm]
        lines.append(f"- {arm}: counts {json.dumps(a['provenance_counts'])} "
                     f"· shares {json.dumps(a['provenance_shares'])}")
    lines += [
        "",
        "## AI-state divergence (§7.7, descriptive)",
        "",
        f"- {r['divergence']['note']}",
        f"- turns compared: {r['divergence']['summary']['turns_compared']} "
        f"· mean L1: {r['divergence']['summary']['mean_l1']} · max L1: "
        f"{r['divergence']['summary']['max_l1']}",
        "",
        "| turn | score Δ | cities Δ | units Δ | known-tiles Δ | L1 |",
        "|---|---|---|---|---|---|",
    ]
    for e in r["divergence"]["per_turn"]:
        lines.append(
            f"| {e['turn']} | {e['score_diff']} | {e['cities_own_diff']} "
            f"| {e['units_own_diff']} | {e['tiles_known_diff']} "
            f"| {e['l1']} |")
    lines += [
        "",
        f"- drift (decision 8): drift_suspect="
        f"{r['drift']['drift_suspect']} · reproducibility_claim="
        f"{r['drift']['reproducibility_claim']}",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, c in r["checks"].items():
        lines.append(f"- {'PASS' if c['pass'] else 'FAIL'} **{name}** — "
                     f"{c['detail']}")
    lines += ["", f"VERDICT: **{r['verdict']}**", ""]
    return "\n".join(lines)


def render_overall_md(r):
    """The human-readable OVERALL report (descriptive only)."""
    lines = [
        f"# WP4 evaluation — {r['run_id']} (overall)",
        "",
        f"Generated: {r['generated_at']}",
        "",
        f"**{r['pilot_framing']}**",
        "",
        "## Parameters",
        "",
        f"- pairs: {r['parameters']['pairs']} · horizon: "
        f"{r['parameters']['horizon']} turns · create turns: "
        f"{r['parameters']['create_turns']} · base seed: "
        f"{r['parameters']['seed']} (pair seed = base + index − 1)",
        f"- username: {r['parameters']['username']} (pinned) · aifill: "
        f"{r['parameters']['aifill']} · ruleset: {r['parameters']['ruleset']}",
        f"- harness revision: `{r['parameters']['harness_revision']}`",
        "",
        "## Pairs",
        "",
    ]
    for p in r["pairs"]:
        lines += [
            f"### {p['pair_id']} — verdict: **{p['verdict']}**",
            "",
            "| arm | outcome | class | survival | turns | wall-clock s |",
            "|---|---|---|---|---|---|",
        ]
        for arm in ("comparator", "model"):
            a = p["arms"][arm]
            lines.append(
                f"| {arm} | {a['outcome']} | {a['outcome_class']} "
                f"| {a['survived']} | {a['turns_completed']}/"
                f"{a['turns_planned']} | {a['wall_clock_s']} |")
        lines += [
            "",
            f"- checkpoint sha256: `{p['checkpoint_sha256']}`",
            f"- divergence: turns compared "
            f"{p['divergence']['summary']['turns_compared']} · mean L1 "
            f"{p['divergence']['summary']['mean_l1']} · max L1 "
            f"{p['divergence']['summary']['max_l1']}",
            f"- drift: drift_suspect={p['drift']['drift_suspect']} · "
            f"reproducibility_claim={p['drift']['reproducibility_claim']}",
            (f"- participation gate: FAIL ({p.get('model_participation')} "
             f"< 0.80) — pair labeled degraded-dominant (plan §3 line 38: "
             f"reported in full, excluded from model-performance claims)"
             if p.get("degraded_dominant") else
             f"- participation gate: PASS ({p.get('model_participation')} ≥ "
             f"0.80) — not labeled"),
            f"- report: `artifacts/runs/{p['run_id']}/p{p['pair_index']}/"
            f"metrics_report.md`",
            "",
        ]
    t = r["totals"]
    lines += [
        "## Totals (all pairs)",
        "",
        f"- pairs run: {t['pairs_run']} · verdicts: "
        f"{json.dumps(t['verdicts'])}",
        f"- arms truncated (infrastructure): "
        f"{json.dumps(t['infrastructure_truncations'])} · strategic "
        f"terminals: {json.dumps(t['strategic_terminals'])} · horizon "
        f"completions: {json.dumps(t['horizon_completions'])}",
        f"- model-arm API calls: {t['api_calls']} · tokens in/out: "
        f"{t['tokens_in']}/{t['tokens_out']} · cost: ${t['cost_usd']}",
        f"- total wall-clock: {t['wall_clock_s']} s",
        "",
        "## Cut ladder",
        "",
        f"- {r['cut_ladder']}",
        "",
        f"OVERALL VERDICT: **{r['verdict']}**",
        "",
    ]
    return "\n".join(lines)


# ===========================================================================
# orchestrator — per pair: contract -> create -> comparator arm -> model
# arm -> divergence + pair manifest + pair report; then the overall report
# ===========================================================================

def cmd_orchestrate(pairs, horizon, create_turns, seed, run_id, spectate):
    from lechonk.eval import checkpoint as cp
    from lechonk.eval import paired

    run_root = run_root_for(run_id)
    os.makedirs(run_root, exist_ok=True)
    log = RunLogger(os.path.join(run_root, "run_stdout.log"))
    log(f"[{run_id}] orchestrator: {pairs} pair(s) x horizon {horizon} "
        f"(create turns {create_turns}, base seed {seed}, per-pair seeds "
        f"base + index - 1) — sequential, idle machine")
    log(f"run root: {run_root}")
    if spectate:
        log("spectate: ENABLED (the model arm of pair 1 will spawn the "
            "WP0 zero-perturbation observer)")

    run_started = time.time()
    pair_reports = []
    for pair_index in range(1, pairs + 1):
        pair_id = pair_id_for(run_id, pair_index)
        pair_dir = pair_dir_for(run_id, pair_index)
        p_seed = pair_seed(seed, pair_index)
        horizon_turn = create_turns + horizon  # the runner's max_turns
        log(f"=== pair {pair_index}/{pairs}: {pair_id} (seed {p_seed}, "
            f"horizon {horizon} turns, horizon_turn {horizon_turn}) ===")

        # --- the PREDECLARED paired contract (before any phase runs) ----
        contract = paired.build_paired_contract(
            arm_turns=horizon, username=USERNAME, horizon_turn=horizon_turn,
            aifill=AIFILL, ruleset=RULESET)
        contract_path = write_json(
            os.path.join(pair_dir, "pair_contract.json"), contract)
        log(f"predeclared paired contract: {contract_path} "
            f"sha256={paired.contract_sha256(contract)}")

        # --- phase 1: create (fresh game + the checkpoint export) -------
        rc = run_phase(log, [
            "--run-id", run_id, "--seed", str(seed),
            "--horizon", str(horizon), "--create-turns", str(create_turns),
            "create", "--pair-index", str(pair_index)],
            os.path.join(pair_dir, "creation", "run_stdout.log"))
        log(f"create phase rc={rc}")
        creation_path = os.path.join(pair_dir, "creation",
                                     "creation_manifest.json")
        if rc != 0 or not os.path.exists(creation_path):
            log(f"FATAL: the create phase failed for {pair_id} — no "
                f"checkpoint to pair from; the pair is reported as "
                f"incomplete (never silently dropped)")
            pair_reports.append({
                "pair_id": pair_id, "run_id": run_id,
                "pair_index": pair_index, "verdict": "FAIL",
                "checkpoint_sha256": None, "drift": {},
                "divergence": {
                    "summary": {"turns_compared": 0, "mean_l1": None,
                                "max_l1": None},
                    "per_turn": [],
                },
                "arms": {},
            })
            continue

        # --- phases 2 + 3: the arms, SEQUENTIALLY ------------------------
        for arm in paired.ARM_ORDER:
            arm_args = [
                "--run-id", run_id, "--seed", str(seed),
                "--horizon", str(horizon), "--create-turns",
                str(create_turns), "arm", "--policy", arm,
                "--pair-index", str(pair_index)]
            if spectate and pair_index == 1 and arm == "model":
                arm_args.append("--spectate")
            rc = run_phase(log, arm_args,
                           os.path.join(pair_dir, "arms", arm,
                                        "run_stdout.log"))
            log(f"{arm} arm rc={rc}")

        arm_manifests = {}
        for arm in paired.ARMS:
            path = os.path.join(pair_dir, "arms", arm, "arm_manifest.json")
            arm_manifests[arm] = (read_json(path) if os.path.exists(path)
                                  else None)
            if arm_manifests[arm] is None:
                log(f"FATAL: no arm manifest for the {arm} arm")

        creation = read_json(creation_path)
        checkpoint = creation["checkpoint"]

        # The per-turn summaries (the progress + divergence input).
        turns_c = read_turns_jsonl(os.path.join(
            pair_dir, "arms", "comparator", "turns.jsonl")) or []
        turns_m = read_turns_jsonl(os.path.join(
            pair_dir, "arms", "model", "turns.jsonl")) or []

        # --- the pair-level checks ---------------------------------------
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
              "the preserved checkpoint re-verifies against the recorded "
              "sha256")

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

        # (4) the arms ran sequentially (the model arm started at or
        # after the comparator arm ended — non-overlapping; the
        # timestamps have 1 s resolution, so a fast phase boundary can
        # share a second).
        seq_ok = False
        seq_detail = "missing arm manifests"
        if arm_manifests.get("comparator") and arm_manifests.get("model"):
            c, m = arm_manifests["comparator"], arm_manifests["model"]
            seq_ok = (parse_ts(m["arm_start_ts"])
                      >= parse_ts(c["arm_end_ts"]))
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
            f"turns={m and m.get('turns_completed')}/{horizon} "
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

        # (8) the per-arm fingerprints are recorded and the decision-8
        # drift policy is applied.
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
                 and drift.get("per_arm", {}).get("comparator", {}).get(
                     "status") == "no_model")
        check("fingerprints_recorded_drift_policy_applied", fp_ok,
              f"model arm fingerprint={arm_fingerprints.get('model')} "
              f"status={drift['per_arm'].get('model', {}).get('status')}; "
              f"comparator arm: no model; "
              f"drift_suspect={drift['drift_suspect']} "
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

        # (10) the per-turn summaries are recorded for both arms (the
        # progress curves + the divergence input).
        turns_ok = bool(turns_c) and bool(turns_m)
        check("per_turn_summaries_recorded", turns_ok,
              f"comparator: {len(turns_c)} summaries · model: "
              f"{len(turns_m)} summaries")

        # (11) the model arm's outcome is reported (honest TRUNCATED is a
        # correct outcome per decision 8 — reported, not gating).
        model_outcome = (arm_manifests.get("model") or {}).get("outcome")
        log(f"[report] model arm outcome: {model_outcome} "
            f"({(arm_manifests.get('model') or {}).get('reason')!r}) — "
            f"reported honestly; an infrastructure TRUNCATED with "
            f"artifacts is a correct outcome (decision 8)")

        # --- the §7.7 divergence diagnostic -------------------------------
        divergence = divergence_curve(turns_c, turns_m)
        divergence_path = write_json(
            os.path.join(pair_dir, "divergence.json"), divergence)
        log(f"divergence curve: {divergence_path} "
            f"(turns compared={divergence['summary']['turns_compared']})")
        check("divergence_curve_recorded",
              divergence["summary"]["turns_compared"] >= 1,
              f"{divergence['summary']['turns_compared']} common turn(s); "
              f"descriptive only, no counterfactual claims")

        plumbing = ["checkpoint_exported_via_wp3_path",
                    "checkpoint_sha256_verified",
                    "both_arms_record_identical_checkpoint",
                    "both_arms_loaded_identical_checkpoint",
                    "arms_sequential", "contract_identical_in_both_manifests",
                    "early_terminal_reporting", "both_transcripts_valid",
                    "fingerprints_recorded_drift_policy_applied",
                    "replay_bundles_rendered", "per_turn_summaries_recorded",
                    "divergence_curve_recorded"]
        verdict = ("PASS" if all(checks[c]["pass"] for c in plumbing)
                   else "FAIL")
        pair_manifest = paired.assemble_pair_manifest(
            pair_id=pair_id, contract=contract,
            checkpoint={
                "path": checkpoint["path"],
                "save_name": checkpoint["save_name"],
                "sha256": checkpoint["sha256"], "turn": checkpoint["turn"],
                "username": checkpoint["username"],
                "exported_by": checkpoint["exported_by"],
                "export_evidence": checkpoint.get("export_evidence"),
                "creation_manifest": creation_path,
            },
            arms={a: dict(arm_manifests.get(a) or {})
                  for a in paired.ARMS},
            drift=drift, checks=checks, verdict=verdict)
        pair_manifest["constants"] = {
            "run_id": run_id, "pair_index": pair_index, "seed": p_seed,
            "creation_turns": create_turns, "export_turn": checkpoint["turn"],
            "horizon": horizon, "arm_turns": horizon,
            "horizon_turn": horizon_turn, "username": USERNAME,
            "aifill": AIFILL, "ruleset": RULESET,
            "image_digest": IMAGE_DIGEST, "capstring": CAPSTRING,
            "vendored_base_sha": VENDORED_BASE_SHA,
            "patch_head_sha": PATCH_HEAD_SHA,
        }
        pair_manifest["model_arm_outcome_report"] = {
            "outcome": model_outcome,
            "reason": (arm_manifests.get("model") or {}).get("reason"),
            "note": "reported honestly; an infrastructure TRUNCATED with "
                    "artifacts is a correct outcome (decision 8) — the "
                    "plumbing verdict above does not gate on it",
        }
        write_json(os.path.join(pair_dir, "pair_manifest.json"),
                   pair_manifest)

        # --- the per-pair metrics report (json + md) ----------------------
        report = pair_metrics_report(
            pair_id=pair_id, run_id=run_id, pair_index=pair_index,
            parameters={
                "pairs": pairs, "horizon": horizon,
                "create_turns": create_turns, "seed": p_seed,
                "seed_base": seed, "username": USERNAME, "aifill": AIFILL,
                "ruleset": RULESET,
            },
            checkpoint=checkpoint,
            contract_sha256=paired.contract_sha256(contract),
            arm_manifests=arm_manifests,
            summaries_comparator=turns_c, summaries_model=turns_m,
            divergence=divergence, drift=drift, checks=checks,
            verdict=verdict)
        assert_no_ranking_language(
            json.dumps(report, default=str), f"{pair_id} metrics_report")
        write_json(os.path.join(pair_dir, "metrics_report.json"), report)
        md = render_pair_report_md(report)
        assert_no_ranking_language(md, f"{pair_id} metrics_report.md")
        with open(os.path.join(pair_dir, "metrics_report.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(md)
        log(f"pair report written: {pair_dir}/metrics_report.json/.md")

        pair_reports.append({
            "pair_id": pair_id, "run_id": run_id, "pair_index": pair_index,
            "verdict": verdict,
            "checkpoint_sha256": checkpoint["sha256"],
            "degraded_dominant": report["degraded_dominant"],
            "model_participation": (
                report["arms"]["model"]["honesty_metrics"]
                .get("participation", {}).get("value")),
            "drift": drift,
            "divergence": divergence,
            "arms": {
                arm: {
                    "outcome": report["arms"][arm]["outcome"],
                    "outcome_class": report["arms"][arm]["outcome_class"],
                    "survived": report["arms"][arm]["survived"],
                    "reason": report["arms"][arm]["reason"],
                    "turns_completed": report["arms"][arm]["turns_completed"],
                    "turns_planned": report["arms"][arm]["turns_planned"],
                    "wall_clock_s": report["arms"][arm]["wall_clock_s"],
                    "retries": report["arms"][arm]["retries"],
                    "api_totals": report["arms"][arm]["api_totals"],
                }
                for arm in ("comparator", "model")},
        })

    # --- the overall report (run level) -----------------------------------
    verdicts = {p["pair_id"]: p["verdict"] for p in pair_reports}
    infrastructure, strategic, completions = [], [], []
    api_calls = tokens_in = tokens_out = 0
    cost = wall_total = 0.0
    for p in pair_reports:
        for arm, a in p["arms"].items():
            oc = a.get("outcome_class")
            if oc == "infrastructure_truncation":
                infrastructure.append(f"{p['pair_id']}/{arm}")
            elif oc == "strategic_terminal":
                strategic.append(f"{p['pair_id']}/{arm}")
            elif oc == "horizon_completed":
                completions.append(f"{p['pair_id']}/{arm}")
            wall_total += a.get("wall_clock_s") or 0.0
            api = a.get("api_totals") or {}
            api_calls += api.get("calls") or 0
            tokens_in += api.get("tokens_in") or 0
            tokens_out += api.get("tokens_out") or 0
            cost += api.get("cost_usd") or 0.0
    overall_verdict = ("PASS" if verdicts
                       and all(v == "PASS" for v in verdicts.values())
                       else "FAIL")
    overall = {
        "run_id": run_id,
        "generated_by": "scripts/wp4_eval.py cmd_orchestrate (WP4 package "
                        "A; design-plan-freeciv-v1.1.md §4 line 58)",
        "generated_at": utcnow(),
        "pilot_framing": PILOT_FRAMING,
        "parameters": {
            "pairs": pairs, "horizon": horizon,
            "create_turns": create_turns, "seed": seed,
            "seed_policy": "pair seed = base seed + (pair_index - 1)",
            "username": USERNAME, "aifill": AIFILL, "ruleset": RULESET,
            "harness_revision": paired.harness_revision(REPO),
            "image_digest": IMAGE_DIGEST, "capstring": CAPSTRING,
            "vendored_base_sha": VENDORED_BASE_SHA,
            "patch_head_sha": PATCH_HEAD_SHA,
            "spectate": bool(spectate),
        },
        "pairs": [{
            "pair_id": p["pair_id"], "run_id": p["run_id"],
            "pair_index": p["pair_index"], "verdict": p["verdict"],
            "checkpoint_sha256": p["checkpoint_sha256"],
            "degraded_dominant": p["degraded_dominant"],
            "model_participation": p["model_participation"],
            "drift": p["drift"],
            "divergence": p["divergence"],
            "arms": p["arms"],
        } for p in pair_reports],
        "totals": {
            "pairs_run": len(pair_reports),
            "verdicts": verdicts,
            "infrastructure_truncations": infrastructure,
            "strategic_terminals": strategic,
            "horizon_completions": completions,
            "api_calls": api_calls, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "cost_usd": round(cost, 6),
            "wall_clock_s": round(wall_total, 1),
        },
        "cut_ladder": (
            "cut ladder (plan §4 line 58), applied by the orchestrator "
            "BETWEEN pairs, in order: second pair → horizon 50→25 → "
            "scenario-cut; the same-start same-horizon pair is never cut; "
            "this run's --pairs/--horizon parameters reach the ladder "
            "shapes directly"),
        "verdict": overall_verdict,
    }
    assert_no_ranking_language(json.dumps(overall, default=str),
                               f"{run_id} overall report")
    write_json(os.path.join(run_root, "eval_manifest.json"), overall)
    write_json(os.path.join(run_root, "metrics_report.json"), overall)
    md = render_overall_md(overall)
    assert_no_ranking_language(md, f"{run_id} overall metrics_report.md")
    with open(os.path.join(run_root, "metrics_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    log(f"overall report written: {run_root}/metrics_report.json/.md + "
        f"eval_manifest.json")
    log(f"OVERALL VERDICT: {overall_verdict} "
        f"({round(time.time() - run_started, 1)} s)")
    log.close()
    return 0 if overall_verdict == "PASS" else 1


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="WP4 package A — the evaluation driver: paired runs "
                    "from identical checkpoints (target 2 pairs x 50 "
                    "turns; the cut ladder — second pair → horizon 50→25 "
                    "→ scenario-cut — is applied by the orchestrator "
                    "between pairs via these same --pairs/--horizon "
                    "parameters)")
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS,
                        help="number of evaluation pairs (default 2)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="arm horizon in turns (default 50; the ladder "
                             "shape 25 is reachable with --horizon 25)")
    parser.add_argument("--create-turns", type=int,
                        default=DEFAULT_CREATE_TURNS,
                        help="comparator-driven creation turns before the "
                             "checkpoint export (default 4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base create-game seed (pair seed = seed + "
                             "pair_index - 1)")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID,
                        help="run id / artifacts directory name "
                             "(default: wp4-eval)")
    parser.add_argument("--spectate", action="store_true",
                        help="spawn the WP0 zero-perturbation observer "
                             "during the model arm of pair 1 (optional "
                             "WP5 footage; never perturbs the game)")
    sub = parser.add_subparsers(dest="command")
    create_parser = sub.add_parser(
        "create", help="one pair's creation phase (fresh game + the "
                       "checkpoint export)")
    create_parser.add_argument("--pair-index", type=int, required=True)
    arm_parser = sub.add_parser("arm", help="one paired arm")
    arm_parser.add_argument("--policy", required=True,
                            choices=["comparator", "model"])
    arm_parser.add_argument("--pair-index", type=int, required=True)
    # the orchestrator passes the phase flags after the subcommand
    arm_parser.add_argument("--spectate", action="store_true",
                            help="spawn the WP0 observer (pair 1 model arm)")
    args = parser.parse_args(argv)
    if args.command == "create":
        return cmd_create(args.run_id, args.pair_index, args.seed,
                          args.create_turns, args.horizon)
    if args.command == "arm":
        return cmd_arm(args.run_id, args.pair_index, args.policy,
                       args.spectate)
    return cmd_orchestrate(args.pairs, args.horizon, args.create_turns,
                           args.seed, args.run_id, args.spectate)


if __name__ == "__main__":
    sys.exit(main())
