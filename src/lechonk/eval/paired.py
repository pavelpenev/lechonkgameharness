# SPDX-License-Identifier: MIT
"""lechonk.eval.paired — WP3 paired-run plumbing (MIT; CivRealm-free).

design-plan-freeciv-v1.1.md §4 line 56 (WP3 scope) + §3 line 45 (the
scenario-cut paired contract): an IDENTICAL exported start checkpoint
launches BOTH policies (model + comparator) under a predeclared paired
contract recorded in the manifest — equal observation format, candidate
restrictions, action budgets, lifecycle handling for both arms; arms run
SEQUENTIALLY under the same pinned username as the checkpoint creator
(save/username coupling, §7.1); early-terminal reporting (an arm that
terminates early reports its outcome + reason; the pair is never
silently dropped); checkpoint checksums; arm-start timestamps; per-arm
model fingerprint with the decision-8 drift policy (FW-21: the fingerprint
is the K=3 sample hash SET — match = non-empty intersection of the
reference set and the arm's set; mismatch → re-probe once; persistent
mismatch → drift-suspect pair, excluded from reproducibility claims,
both arms still reported individually).

Architecture (one env per process, §7.1): each arm runs in its OWN
subprocess (``scripts/wp3_paired_smoke.py arm --policy ...``) because
``fc_args`` is process-global and the session enforces one env per
process. This module is the shared, offline-testable plumbing those
arm processes and the orchestrator consume:

- ``build_paired_contract`` — the predeclared contract, identical for
  both arms, hashed (``contract_sha256``) so the manifests prove
  identity;
- ``build_arm_manifest`` — the decision-8 RunManifest for one arm
  (checkpoint checksums, arm-start timestamp, per-arm fingerprint,
  config_after_load with the checkpoint selection);
- ``ScriptedComparatorClient`` — the comparator policy behind the
  core-defined ModelClient protocol, so the comparator arm runs through
  the SAME runner pipeline (equal lifecycle handling) and is labeled
  ``scripted_default``, never ``model`` (the runner reads the client's
  declared ``provenance_class``);
- ``RecordingClient`` — wraps any ModelClient and persists the per-turn
  summary (turns.jsonl) the replay viewer's empire timeline renders;
- ``resolve_fingerprints`` — the decision-8 drift policy;
- ``early_terminal_report`` / ``assemble_pair_manifest`` — the
  pair-level reporting (both arms individually, early terminals
  reported, never dropped).

Licensing: MIT — this module NEVER imports civrealm. The only adapter
touchpoint is a LAZY, function-level import of the CivRealm-free
``lechonk.adapter.obs_extract`` inside
``ScriptedComparatorClient.complete_turn`` (the runner's own
lazy-import discipline, §7.1). Test-enforced.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from lechonk.core.jsonutil import strict_dumps
from lechonk.core.model_client import ModelResponse, RepairPrompt, TurnPrompt
# W2-1: the comparator arm reuses the runner's contact-protection helper so
# both arms reconstruct the identical offered candidate surface.
from lechonk.core.runner import mark_armed_in_contact
from lechonk.core.schemas import (
    ActionBatch,
    ActionItem,
    RunManifest,
    RunManifestRecord,
)
from lechonk.model.mistral_client import decode_fingerprint_set

# The paired contract version (bump on any contract change; the hash in
# the manifest proves the identity of the predeclared contract).
PAIRED_CONTRACT_VERSION = "wp3-paired-v1"

ARM_COMPARATOR = "comparator"
ARM_MODEL = "model"
ARMS: tuple[str, ...] = (ARM_COMPARATOR, ARM_MODEL)
# The scenario-cut's arm order (§3 line 45: both policies run the matched
# segment; the comparator arm runs first — it is free of API cost).
ARM_ORDER: tuple[str, ...] = (ARM_COMPARATOR, ARM_MODEL)

# The frozen §3 numeric budgets the contract predeclares.
from lechonk.core.schemas import (
    ACTION_BATCH_MAX_ACTIONS,
    DELTA_TOKEN_BUDGET,
    MEMORY_TOKEN_BUDGET,
    SNAPSHOT_TOKEN_BUDGET,
)
MAX_API_ATTEMPTS_PER_TURN = 2


def canonical_json(obj: Any) -> str:
    return strict_dumps(obj, sort_keys=True)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the predeclared paired contract
# ---------------------------------------------------------------------------

def build_paired_contract(*, arm_turns: int, username: str, horizon_turn: int,
                          aifill: int, ruleset: str,
                          observation_format: str = (
                              "lechonk.adapter.obs_extract.extract_snapshot "
                              "fog-gated Snapshot (§7.3; ≤8000 tokens, "
                              "visibility-safe truncation with omitted "
                              "disclosure)"),
                          candidate_restrictions: str = (
                              "lechonk.adapter.candidates.build_candidates "
                              "legal-candidate table over the exposed v1 "
                              "surface (§7.2; excluded verbs — exploration, "
                              "autosettler, naval, air, spies, trade — are "
                              "never offered)"),
                          lifecycle: str = (
                              "lechonk.core.runner.GameRunner state machine "
                              "+ supervisor (§3 line 34 deadlines; run-level "
                              "stall watchdog 600 s; guarded session close)"),
                          ) -> dict[str, Any]:
    """The predeclared paired contract (§3 line 45 / §4 line 56). It is
    built ONCE by the orchestrator and recorded IDENTICALLY in both arms'
    manifests; ``contract_sha256`` proves the identity."""
    return {
        "contract_version": PAIRED_CONTRACT_VERSION,
        "arms": list(ARMS),
        "arm_order": list(ARM_ORDER),
        "sequential": True,
        "username": username,
        "username_note": "both arms run under the SAME pinned username as "
                         "the checkpoint creator (save/username coupling, "
                         "§7.1; CivRealm validates the save's username "
                         "prefix on load)",
        "start_checkpoint": "identical exported checkpoint for both arms "
                            "(checksummed; see the manifest's "
                            "checkpoint_checksums + config_after_load)",
        "horizon_turn": horizon_turn,
        "arm_turns": arm_turns,
        "observation_format": observation_format,
        "candidate_restrictions": candidate_restrictions,
        "action_budgets": {
            "max_actions_per_turn": ACTION_BATCH_MAX_ACTIONS,
            "max_api_attempts_per_turn": MAX_API_ATTEMPTS_PER_TURN,
            "snapshot_tokens": SNAPSHOT_TOKEN_BUDGET,
            "delta_tokens": DELTA_TOKEN_BUDGET,
            "memory_tokens": MEMORY_TOKEN_BUDGET,
        },
        "lifecycle": lifecycle,
        "early_terminal": "an arm that terminates before the horizon "
                          "reports its outcome + reason in its manifest "
                          "and the pair manifest; the pair is never "
                          "silently dropped (§3 line 45)",
        "drift_policy": "decision 8 (FW-21): the model fingerprint is the "
                        "K=3 sample hash set; a between-arm/reference "
                        "mismatch (no shared hash) → re-probe once; a "
                        "persistent mismatch marks the pair drift-suspect "
                        "— excluded from reproducibility claims, both "
                        "arms still reported individually",
        "scenario": {"ruleset": ruleset, "aifill": aifill},
    }


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(contract)))


# ---------------------------------------------------------------------------
# harness revision (same content hash the WP2 manifest records)
# ---------------------------------------------------------------------------

def harness_revision(repo_root: str | Path) -> str:
    """The harness revision: a deterministic content hash over
    ``src/lechonk`` (every file's sha256, sorted by path, hashed again) —
    the repo is NOT a git checkout (HANDOFF constraints)."""
    root = Path(repo_root) / "src" / "lechonk"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = str(path.relative_to(root))
            lines.append(f"{rel} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    blob = "\n".join(lines).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# the arm manifest (decision 8 + the paired extras)
# ---------------------------------------------------------------------------

def build_arm_manifest(*, arm: str, pair_id: str, contract: Mapping[str, Any],
                       checkpoint: Mapping[str, Any], arm_start_ts: str,
                       model_fingerprint: Optional[str],
                       fingerprint_evidence: Optional[Mapping[str, Any]],
                       harness_rev: str, seeds: Mapping[str, Optional[int]],
                       username: str, aifill: int, image_digest: str,
                       capstring: str, vendored_base_sha: str,
                       patch_head_sha: str, model_id: Optional[str],
                       sampling: Mapping[str, Any],
                       prompt_hashes: Mapping[str, str],
                       enforcement_decision: Optional[Mapping[str, Any]],
                       action_batch_schema_hash: Optional[str],
                       run_id: str) -> RunManifest:
    """The decision-8 RunManifest for one paired arm. The frozen field
    set carries the checkpoint checksums, the arm-start timestamp and the
    per-arm model fingerprint; the forward-compat extras (``extra=
    "allow"``) carry the predeclared paired contract + its hash, the arm
    identity and the fingerprint evidence."""
    contract_hash = contract_sha256(contract)
    fc_args = {
        "self_play": False, "aifill": aifill,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": seeds.get("mapseed"),
        "debug.gameseed": seeds.get("gameseed"),
        "debug.agentseed": seeds.get("agentseed"),
        "server_timeout": 300, "begin_turn_timeout": 900,
        "wait_for_timeout": 60, "http_timeout_s": 15.0,
        "debug.load_game": checkpoint["save_name"],
    }
    manifest = RunManifest(
        run_id=run_id,
        image_digest=image_digest,
        server_commit=None,  # not persisted by the WP0 spike (stdout-only);
        # the image digest pins the server build — disclosed.
        capstring=capstring,
        vendored_base_sha=vendored_base_sha,
        patch_head_sha=patch_head_sha,
        harness_revision=harness_rev,
        python_lockfile=None,  # no lockfile in the repo (uv-managed venv)
        seeds=dict(seeds),
        ruleset=contract["scenario"]["ruleset"],
        server_settings={
            "fc_args_overrides": fc_args,
            "phasemode": "player", "timeout": 0, "pingtimeout": 720,
            "autotoggle": "disabled",
            "victories": "SPACERACE|ALLIED (CivRealm default)",
            "aifill": aifill, "self_play": False,
        },
        model_id=model_id,
        sampling=dict(sampling),
        prompt_hashes=dict(prompt_hashes),
        config_after_load={
            "checkpoint_loaded": True,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_save_name": checkpoint["save_name"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_turn": checkpoint["turn"],
            "checkpoint_username": checkpoint["username"],
            "checkpoint_exported_by": checkpoint.get("exported_by"),
            "fc_args": fc_args,
            "note": "load of the IDENTICAL exported checkpoint under the "
                    "predeclared paired contract; the load identity is "
                    "verified post-run in the arm manifest (turn, "
                    "username, state)",
        },
        checkpoint_checksums={"start": checkpoint["sha256"]},
        model_fingerprint=model_fingerprint,
        arm_start_ts=arm_start_ts,
    )
    # Forward-compat extras (recorded alongside the frozen set, never
    # inside it).
    manifest.paired_pair_id = pair_id
    manifest.paired_arm = arm
    manifest.paired_contract = dict(contract)
    manifest.paired_contract_sha256 = contract_hash
    manifest.paired_username = username
    manifest.fingerprint_evidence = (dict(fingerprint_evidence)
                                     if fingerprint_evidence else None)
    manifest.enforcement_decision = (dict(enforcement_decision)
                                     if enforcement_decision else None)
    manifest.action_batch_schema_hash = action_batch_schema_hash
    return manifest


def read_arm_manifest(transcript_path: str | Path) -> RunManifest:
    """Read the arm manifest (the FIRST ledger record, emitted durable
    before anything else)."""
    from lechonk.core.ledger import load_transcript

    load = load_transcript(transcript_path)
    if not load.records or not isinstance(load.records[0], RunManifestRecord):
        raise ValueError(
            f"{transcript_path}: the first ledger record is not a "
            f"run_manifest (records: {len(load.records)})")
    return load.records[0].manifest


# ---------------------------------------------------------------------------
# the decision-8 drift policy
# ---------------------------------------------------------------------------

def fingerprint_hash_set(fingerprint: Optional[str]) -> list[str]:
    """The decoded hash set of a persisted fingerprint (decision 8, FW-21).

    A legacy scalar fingerprint (a single ``sha256:<hex>``, the pre-FW-21
    encoding) decodes to a one-element set — the documented fallback;
    None/blank decodes to the empty set."""
    if not fingerprint:
        return []
    return decode_fingerprint_set(fingerprint)


def fingerprint_match(reference: Optional[str],
                      arm_fingerprint: Optional[str]) -> bool:
    """The decision-8 canary rule (FW-21): a reference fingerprint and an
    arm fingerprint MATCH iff their sample hash sets intersect (any shared
    hash). Documented explicitly as a canary rule, NOT an equivalence
    relation."""
    return bool(set(fingerprint_hash_set(reference))
                & set(fingerprint_hash_set(arm_fingerprint)))


def resolve_fingerprints(*, arm_fingerprints: Mapping[str, Optional[str]],
                         reference: Optional[str],
                         re_probes: Optional[Mapping[str, Optional[str]]] = None
                         ) -> dict[str, Any]:
    """The decision-8 drift policy over the per-arm model fingerprints.

    ``arm_fingerprints`` maps arm → the fingerprint the arm recorded (None
    for a scripted arm: no model). Each fingerprint is the persisted K=3
    sample hash SET (FW-21; a legacy scalar is a one-element set — the
    documented fallback). ``reference`` is the reference fingerprint (the
    WP2 spike artifact, itself a K=3 sample set re-established under the
    new sampling). ``re_probes`` maps arm → the ONE re-probe fingerprint
    the arm performed after a mismatch (the re-probe happens in the arm
    process; this function applies the policy). Match = non-empty
    intersection of the reference set and the arm's set — a canary rule,
    not an equivalence relation. A persistent mismatch (or a between-arm
    mismatch among model arms, judged by the same intersection rule)
    marks the pair drift-suspect: excluded from reproducibility claims,
    both arms still reported individually.
    """
    re_probes = dict(re_probes or {})
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        fp = arm_fingerprints.get(arm)
        if fp is None:
            per_arm[arm] = {
                "arm": arm, "fingerprint": None, "reference": reference,
                "status": "no_model",
                "note": "scripted arm: no model fingerprint (the "
                        "comparator policy makes no API calls)",
            }
            continue
        entry: dict[str, Any] = {
            "arm": arm, "fingerprint": fp, "reference": reference,
            "fingerprint_hashes": fingerprint_hash_set(fp),
            "reference_hashes": fingerprint_hash_set(reference),
            "status": ("no_reference" if reference is None else
                       "match" if fingerprint_match(reference, fp)
                       else "mismatch"),
        }
        if (reference is not None and not fingerprint_match(reference, fp)
                and arm in re_probes):
            rp = re_probes[arm]
            entry["re_probe_fingerprint"] = rp
            entry["re_probe_hashes"] = fingerprint_hash_set(rp)
            entry["status"] = ("match_after_reprobe"
                               if fingerprint_match(reference, rp)
                               else "mismatch_persistent")
        per_arm[arm] = entry
    model_arms = [a for a in ARMS if arm_fingerprints.get(a) is not None]
    between_arm_mismatch = False
    if len(model_arms) >= 2:
        # The same canary rule between model arms: any shared hash.
        arm_sets = [set(fingerprint_hash_set(arm_fingerprints[a]))
                    for a in model_arms]
        between_arm_mismatch = not any(
            arm_sets[i] & arm_sets[j]
            for i in range(len(arm_sets))
            for j in range(i + 1, len(arm_sets)))
    drift_suspect = (
        any(e["status"] == "mismatch_persistent" for e in per_arm.values())
        or between_arm_mismatch)
    return {
        "policy": "decision 8 (FW-21): the fingerprint is the K=3 sample "
                  "hash set; match = non-empty intersection of the "
                  "reference set and the arm's set; on a mismatch, "
                  "re-probe once; a persistent mismatch marks the pair "
                  "drift-suspect — excluded from reproducibility claims, "
                  "both arms still reported individually",
        "reference_fingerprint": reference,
        "per_arm": per_arm,
        "between_arm_mismatch": between_arm_mismatch,
        "drift_suspect": drift_suspect,
        "reproducibility_claim": (not drift_suspect
                                  and reference is not None),
        "both_arms_reported_individually": True,
        "note": "the fingerprint is a canary: serving nondeterminism can "
                "false-positive; an unchanged output does not prove "
                "unchanged weights",
    }


# ---------------------------------------------------------------------------
# early-terminal reporting (§3 line 45)
# ---------------------------------------------------------------------------

def early_terminal_report(outcome: str, reason: str, turns_completed: int,
                          turns_planned: int) -> Optional[str]:
    """The early-terminal report for an arm: None when the arm reached
    its planned horizon (any terminal outcome at the horizon is normal,
    not early — VICTORY/DEFEATED on the horizon turn included);
    otherwise the outcome + reason + the shortfall. The pair is never
    silently dropped — the report is recorded in the arm manifest and
    the pair manifest."""
    if turns_completed >= turns_planned:
        return None
    return (f"arm terminated early: outcome={outcome} "
            f"reason={reason!r} turns_completed={turns_completed}/"
            f"{turns_planned} (reported; the pair is not silently dropped, "
            f"§3 line 45)")


# ---------------------------------------------------------------------------
# the scripted comparator arm's ModelClient
# ---------------------------------------------------------------------------

class ScriptedComparatorClient:
    """The comparator policy behind the core-defined ModelClient
    protocol, so the comparator arm runs through the SAME runner
    pipeline as the model arm (equal lifecycle handling, §3 line 45).

    The client re-derives the turn's Snapshot + CandidateTable from the
    session (the same deterministic, read-only extraction the runner
    performed moments earlier — no state change in between) and converts
    the comparator's DecisionLog into an ActionBatch of candidate-ID
    refs. It declares ``provenance_class = "scripted_default"`` so the
    ledger labels the arm's decisions honestly (never ``model``).

    ``session_provider`` is a zero-argument callable returning the
    AdapterSession (the runner creates the session inside its
    ``session_factory``; the provider resolves it lazily at call time).

    The ``lechonk.adapter.obs_extract`` import is LAZY (function level):
    the module is CivRealm-free, and the import happens only once the
    Docker stack is up (the runner's lazy-import discipline, §7.1).
    """

    provenance_class = "scripted_default"

    def __init__(self, session_provider: Callable[[], Any],
                 *, policy: str = ARM_COMPARATOR):
        self._session_provider = session_provider
        self._policy = policy
        self.turn_prompts: list[TurnPrompt] = []
        self.calls = 0

    def _prompt_hash(self, prompt: TurnPrompt) -> str:
        blob = "\n--\n".join([prompt.memory, prompt.snapshot,
                              prompt.delta, prompt.candidates])
        return sha256_text(blob)

    def _decide(self, prompt: TurnPrompt) -> ModelResponse:
        from lechonk.adapter.obs_extract import (  # noqa: PLC0415 — lazy
            ExtractOutcome, extract_snapshot)
        from lechonk.comparator import scripted  # noqa: PLC0415

        session = self._session_provider()
        t0 = time.monotonic()
        extraction = extract_snapshot(
            session.obs, session.info,
            my_player_id=session.player_id,
            ruleset=session.ruleset,
            action_probs=session.action_probs,
        )
        if extraction.outcome is not ExtractOutcome.OK \
                or extraction.snapshot is None:
            return ModelResponse(
                call_id=prompt.call_id, prompt_hash=self._prompt_hash(prompt),
                error=(f"snapshot extraction failed: "
                       f"{extraction.outcome.value} {extraction.reason}"),
                latency_ms=int((time.monotonic() - t0) * 1000),
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                fingerprint="n/a:scripted-comparator")
        table = session.build_candidate_table()
        # W2-1: reproduce the runner's contact protection and offered set —
        # the scripted policy must see the same candidate surface the model
        # saw (armed-in-contact candidates never elided; the rest capped).
        mark_armed_in_contact(table, extraction.snapshot)
        log = scripted.decide(extraction.snapshot, table.offered())
        batch = ActionBatch(
            turn_id=prompt.turn_id,
            actions=[ActionItem(ref=d.candidate_id) for d in log.decisions],
            finish_requested=True,
            notes=(f"{self._policy} policy: {len(log.decisions)} decisions "
                   f"by domain {log.by_domain}"),
        )
        return ModelResponse(
            call_id=prompt.call_id, prompt_hash=self._prompt_hash(prompt),
            batch=batch,
            latency_ms=int((time.monotonic() - t0) * 1000),
            tokens_in=0, tokens_out=0, cost_usd=0.0,
            fingerprint="n/a:scripted-comparator")

    def complete_turn(self, prompt: TurnPrompt) -> ModelResponse:
        self.turn_prompts.append(prompt)
        self.calls += 1
        return self._decide(prompt)

    def complete_repair(self, prompt: RepairPrompt) -> ModelResponse:
        # The scripted policy never emits invalid refs (it picks from the
        # offered table); a repair request is a defensive no-op that
        # reports the (impossible) failure honestly.
        return ModelResponse(
            call_id=prompt.call_id, prompt_hash=self._prompt_hash(prompt),
            error="scripted comparator: repair not applicable (the policy "
                  "only emits offered candidate IDs)",
            latency_ms=0, tokens_in=0, tokens_out=0, cost_usd=0.0,
            fingerprint="n/a:scripted-comparator")


# ---------------------------------------------------------------------------
# the recording wrapper (per-turn summaries for the replay viewer)
# ---------------------------------------------------------------------------

def turn_summary_from_snapshot_json(snapshot_json: str,
                                    *, turn_id: Optional[str] = None,
                                    ts: Optional[str] = None
                                    ) -> Optional[dict[str, Any]]:
    """Parse a rendered Snapshot (its JSON text) into the per-turn
    summary the replay viewer's empire timeline renders (score, cities,
    units, gold, research). Returns None when the snapshot does not
    parse (recorded honestly as a gap, never fabricated)."""
    try:
        snap = json.loads(snapshot_json)
    except (ValueError, TypeError):
        return None
    player = snap.get("player") or {}
    research = player.get("research") or {}
    own_cities = [c for c in (snap.get("cities") or []) if c.get("is_own")]
    own_units = [u for u in (snap.get("units") or []) if u.get("is_own")]
    return {
        "turn": snap.get("turn"),
        "turn_id": turn_id,
        "ts": ts,
        "score": player.get("score"),
        "gold": player.get("gold"),
        "techs_researched": player.get("techs_researched"),
        "research_target": research.get("current_target"),
        "research_turns_left": research.get("turns_left"),
        "cities_own": len(own_cities),
        "units_own": len(own_units),
        "cities": [{"id": c.get("id"), "name": c.get("name"),
                    "size": c.get("size"), "production": c.get("production")}
                   for c in own_cities],
        "units": [{"id": u.get("id"), "type": u.get("type_name"),
                   "x": u.get("x"), "y": u.get("y"),
                   "activity": u.get("activity_name")}
                  for u in own_units],
    }


def turn_summary_from_prompt(prompt: TurnPrompt,
                             *, ts: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The turn prompt's rendered snapshot → the per-turn summary (the
    RecordingClient's record; see turn_summary_from_snapshot_json)."""
    return turn_summary_from_snapshot_json(
        prompt.snapshot, turn_id=prompt.turn_id, ts=ts)


class RecordingClient:
    """Wrap any ModelClient and persist the per-turn summary
    (``turns.jsonl``) parsed from each turn prompt's rendered snapshot —
    the replay viewer's empire-timeline source (§7.6). The wrapper is
    transparent: it delegates both calls and passes the inner client's
    declared ``provenance_class`` through."""

    def __init__(self, inner: Any, turns_path: str | Path):
        self._inner = inner
        self._turns_path = str(turns_path)
        Path(self._turns_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._turns_path, "a", encoding="utf-8")
        self._summaries: list[dict[str, Any]] = []

    @property
    def provenance_class(self) -> str:
        return getattr(self._inner, "provenance_class", "model")

    @property
    def inner(self) -> Any:
        return self._inner

    def complete_turn(self, prompt: TurnPrompt) -> ModelResponse:
        resp = self._inner.complete_turn(prompt)
        summary = turn_summary_from_prompt(
            prompt, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if summary is not None:
            self._summaries.append(summary)
            self._fh.write(strict_dumps(summary, sort_keys=True) + "\n")
            self._fh.flush()
        return resp

    def complete_repair(self, prompt: RepairPrompt) -> ModelResponse:
        # The repair repeats the same turn's sections; the primary call
        # already recorded the turn's summary.
        return self._inner.complete_repair(prompt)

    def summaries(self) -> list[dict[str, Any]]:
        return list(self._summaries)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()


# ---------------------------------------------------------------------------
# privileged state summary (verification only — never policy-facing)
# ---------------------------------------------------------------------------

def state_summary_from_obs(obs: Mapping[str, Any], player_id: Optional[int]
                           ) -> dict[str, Any]:
    """TEST HELPER (not production code): a privileged state summary used
    in the WP3 tests to VERIFY that both arms loaded the identical
    checkpoint (a privileged comparison, outside the policy pipeline —
    §3 line 36: privileged data never enters the policy pipeline;
    replay/evaluation views are separate)."""
    players = obs.get("player") or {}
    me = players.get(player_id) if player_id is not None else None
    me = me if isinstance(me, dict) else {}
    own_cities = sorted(
        (int(cid), (c or {}).get("size"))
        for cid, c in (obs.get("city") or {}).items()
        if isinstance(c, Mapping) and c.get("owner") == player_id)
    own_units = sorted(
        int(uid) for uid, u in (obs.get("unit") or {}).items()
        if isinstance(u, Mapping) and u.get("owner") == player_id)
    return {
        "score": me.get("score"),
        "gold": me.get("gold"),
        "techs_researched": me.get("techs_researched"),
        "researching": me.get("researching"),
        "cities_own": [{"id": cid, "size": size} for cid, size in own_cities],
        "units_own": own_units,
    }


# The canonical state fields of a verification summary (both the
# obs-based privileged summary and the snapshot-based turn summary).
# Record metadata (turn_id/ts) is NOT state: the export-time capture
# stamps None while an arm's first-turn record stamps its own run id and
# timestamp, so it never participates in the comparison.
_STATE_FIELDS = ("score", "gold", "techs_researched", "researching",
                 "research_target", "research_turns_left",
                 "cities_own", "units_own", "cities", "units")


def states_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Canonical state equality over the verification summaries (the
    score/gold/cities/units identity the paired arms must share at
    load). Every state field present in either summary must be equal;
    record metadata (turn_id/ts) is excluded by construction."""
    return all(a.get(f) == b.get(f) for f in _STATE_FIELDS)


# ---------------------------------------------------------------------------
# the pair manifest (assembled by the orchestrator after both arms)
# ---------------------------------------------------------------------------

def assemble_pair_manifest(*, pair_id: str, contract: Mapping[str, Any],
                           checkpoint: Mapping[str, Any],
                           arms: Mapping[str, Mapping[str, Any]],
                           drift: Mapping[str, Any],
                           checks: Mapping[str, Any],
                           verdict: str) -> dict[str, Any]:
    """The pair-level manifest: the predeclared contract, the checkpoint,
    BOTH arms reported individually (outcomes, turns, fingerprints,
    early terminals), the sequential-arm evidence, the drift resolution
    and the acceptance checks."""
    return {
        "pair_id": pair_id,
        "generated_by": "lechonk.eval.paired.assemble_pair_manifest "
                        "(WP3 package A; design-plan-freeciv-v1.1.md §4 "
                        "line 56 + §3 line 45)",
        "paired_contract": dict(contract),
        "paired_contract_sha256": contract_sha256(contract),
        "checkpoint": dict(checkpoint),
        "arms": {arm: dict(arms[arm]) for arm in ARMS if arm in arms},
        "sequential": {
            "order": list(ARM_ORDER),
            "evidence": "the orchestrator ran the arms sequentially, each "
                        "in its own subprocess (one env per process, "
                        "§7.1); the per-arm start/end timestamps are "
                        "recorded in each arm's manifest",
        },
        "drift": dict(drift),
        "reproducibility_claim": drift.get("reproducibility_claim"),
        "checks": dict(checks),
        "verdict": verdict,
    }
