# SPDX-License-Identifier: MIT
"""lechonk.core.runner — the turn pipeline, state machine and supervisor (WP2).

Implements design-plan-freeciv-v1.1.md §3 lines 32-35 and §7.1
(core/runner.py module contract). Turn-horizon note: when a run ends by
reaching the turn horizon rather than by game-over or failure, the outcome
is recorded as COMPLETED (an OUTCOME in the run result/manifest, not a
state); its terminal state is NEXT_OWN_PHASE — no terminal
state_transition event is emitted on horizon completion, by design.

- Turn state machine: ACTING -> FINISH_SENT -> WAITING -> NEXT_OWN_PHASE |
  GAME_OVER | FAILED, with per-state progress deadlines enforced by the
  supervisor (activity-independent): ACTING ≤ 180 s; FINISH_SENT/WAITING ≤
  the pingtimeout-derived bound (720 s + margin = 900 s); startup ≤ 120 s;
  per-mutation send+read-back ≤ 30 s; teardown ≤ 30 s. Any state exceeding
  its deadline → FAILED/TRUNCATED with artifacts.
- Run-level stall watchdog: no transcript event for 10 minutes → the
  ordered shutdown path (abort + guarded session close; the unwind emits
  TRUNCATED with the reason). Pure backstop (D14): it fires on a stall
  only once the current phase's deadline has also passed (terminal
  states publish no deadline — the plain stall rule applies); a legal
  WAITING interval is never aborted before its WAITING deadline, which
  ends the wait itself (D7).
- Batch semantics (§3 line 33): after each executed mutation the remaining
  batch actions are revalidated against refreshed state; actions are
  grouped per acting entity; a stale or illegal action aborts only that
  entity's remaining group (logged rejected / aborted_group); other
  entities proceed; the turn is never aborted; no third API attempt.
- Finish phase has exactly one owner: the pipeline (the model may request
  it; the pipeline sends phase_done).
- Repair flow: one repair attempt per turn max (≤2 total API attempts per
  turn), with the per-class repair semantics (schema-error / state-fog-error
  / illegal-in-rule → class-specific repair prompt; unresponsive →
  bounded re-ask). Model unresponsive after the repair cap → the degraded
  default (comparator/defaults.decide), logged scripted_default with reason.
- Outcome taxonomy: COMPLETED (horizon) / DEFEATED / VICTORY / TRUNCATED
  (with reason).
- The runner imports the adapter LAZILY (only after the stack is up, §7.1):
  ``import civrealm`` performs a network fetch at import time, so every
  lechonk.adapter import happens inside run()/the default session factory,
  never at module level (test-enforced by an AST scan in
  tests/test_runner.py). Model consumption goes through the core-defined
  ModelClient interface (core/model_client.py) — a fake client drives the
  offline tests.
- The runner drives TranscriptRecord emission through the ledger for every
  state transition and action.

Classification of model-emitted action attempts into the illegal-action
taxonomy (§3 line 43, frozen operationalization):
- schema_error: the ref is not a candidate ID (raw coordinates, free text)
  or the batch violated the ActionBatch schema (the client reports
  ``malformed``; one rejected attempt per emitted ref is recorded);
- state_fog_error: a well-formed candidate ID that was NOT in the turn's
  offered candidate table (the model acted on state it could not see);
- illegal_in_rule: a ref that WAS offered this turn but is no longer legal
  on mid-batch revalidation (became stale), or an execution whose read-back
  is CONTRADICTED (the rules engine proved non-execution).

continued_model_order (§3 line 38, frozen WP2 operationalization): the
pipeline re-issues a model-originated PERSISTENT order (fortify/sentry/
build_road/irrigation/mine/keep_activity) exactly once, on a later turn
where (a) the model's batch contains no action for that actor and (b) the
current state proves the order is no longer in effect (read-back proof of
non-execution — mutations are never re-sent without it, §3 line 32). The
record links to its origin (origin_seq = the originating model
action_result). The upstream sentry quirk (the read path cancels
ACTIVITY_SENTRY within the same step) makes the sentry case real.

Uncertain mutations (WP3, §3 line 35): a mutation whose send fails after
dispatch (disconnect or deadline — the read-back never arrives) has an
UNKNOWN execution state. It is settled ONLY by full-state reconciliation:
the runner re-reads through the session (the reconnect) and compares the
target postconditions (adapter.execute.verify_readback of the
pre-mutation state against the fresh state). Only verified NON-execution
(CONTRADICTED) permits resubmission — exactly once, as
continued_model_order with origin linkage (the origin is the uncertain
model attempt, or the standing order's own origin for continuations).
An executed (VERIFIED) or unknown (INCONCLUSIVE / reconnect read failed)
mutation is NEVER resent: it is quarantined (mutation_quarantine
metric_event) and the run takes the bounded fail-stop — TRUNCATED with
artifacts, never blind replay, no duplicate mutation. This quarantine rule
applies to send-failed-after-dispatch mutations (disconnect-context only).
A step-completed INCONCLUSIVE read-back is different: the send succeeded and
the execution state is known, only the read-back evidence is non-probative
(e.g. sentry/build_city). Such a mutation is NOT quarantined — a standing
order stored on it stays continuable, and the later resend by
_continue_standing_orders is the normal state-gated re-issue (read-back
proof of non-execution required, §3 line 32), not a blind replay. The resubmission
itself is never repeated: a second disconnect (or a contradicted
resend) quarantines the same way. Reconciliation decisions are ledgered
as mutation_reconciliation metric_events so the P11 audit can verify the
non-execution proof behind every repeat execution.

Dispatch-intent durability (W3-1, D5): every mutation send — at BOTH send
sites, _execute_one's dispatch and the verified resubmission — is preceded
by a durable ``dispatch_intent`` metric_event (fsynced BEFORE the send), so
a kill between dispatch and the action_result leaves a durable trace of
which action was sent. The pinned send-site order: pre-capture pre-state ->
deadline/game-over check -> append intent (durable) -> FINAL deadline
recheck -> if blocked: emit the durable ``mutation_dispatch_cancelled``
event (the same payload keys as the intent + intent_seq), NO send -> else
execute_candidate -> append the correlated action_result (durable; its
detail carries a ``dispatch-result/v1`` envelope with the intent's seq —
legacy detail strings remain valid). An intent with neither a correlated
result nor a cancellation (a crash inside the dispatch window) is FORENSIC
ONLY (D15): the P11 audit's pending-intent scanner reports it; the runner
never refuses or quarantines on it.

W3-2 stop-at-deadline (D6/D13, DEF-1/DEF-2): every mutation route stops
initiating dispatches at the ACTING deadline. The planned batch guards
the TOP of each _execute_planned loop iteration — before _execute_one's
intent write, so a NEW mutation is never dispatched after the deadline
(the W3-1 pre-send check inside _execute_one remains the final
authority); the degraded defaults are covered by each group's re-entry
into _execute_planned; the standing-order continuation guards before
each _execute_one; the verified resend is covered by W3-1's pre-send
check at the resend site (verified: no gap). When a loop guard finds the
deadline reached it emits the partial-batch accounting — one
aborted_group action_result per planned-but-undispatched action
(readback "none", detail "not dispatched: ACTING deadline reached") plus
ONE "acting_deadline_stop" summary metric event (a partial-batch/v1
canonical-JSON payload) — and raises StateDeadlineExceeded(ACTING); the
run-level handler truncates with artifacts. An in-flight send keeps its
FULL mutation deadline (D13: never bounded by the remaining ACTING
time), so the documented ACTING overshoot of up to one mutation
deadline is expected. A session timeout marks the session failed but
does NOT cancel an in-flight worker operation — a timeout is not
transport cancellation. An ALREADY-ESTABLISHED game-over wins over a
newly-noticed deadline expiry (DEF-1). ONE comparison convention (>=
— the deadline is reached when clock() >= the deadline) is pinned across
the guards and the phase checks.

W3-9 engine-automatic observation (D9/D12, Option A): engine-automatic
resolutions (§3 line 38 — counted, never hidden) are detected by
observation-event diffing over a PERSISTENT last-seen baseline (own-city
production + the current research target) fed at FOUR observation events:
(1) turn start, post-extraction, from the session's cached obs; (2) after
EVERY executed mutation, from the same cache (ExecutionResult carries no
obs); (3) after the verified resend — the resend completes inside
_execute_one, so the caller's post-step event covers it; (4) after
successful phase completion (the finish outcome), so final-boundary
transitions before the horizon exit are captured. Attribution is
per-interval: a change is the model's iff it matches a produce/research
action executed since the last observation event (a bounded retained
set); every other change — a corroborated new city's first non-None
production (ownership transfer / checkpoint resync / re-sighting after
absence is baseline-only), a B→A revert, a buy-completion transition
observed at the next event — emits engine_automatic with a ref carrying
the entity identity and an observation index (unique per record).
Records carry the turn current AT the observation event — never beyond
the counted horizon (the finish boundary attributes to the COMPLETING
turn). A capture gap emits the DEF-3 diagnostic (fail-open, disclosed);
readback credit is unchanged (observation history never upgrades a
contradicted action). Within each iteration the W3-2 guard records are
emitted BEFORE the observation-diff records, so a deadline stop never
half-records an observation event.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from lechonk.core.activities import ACTIVITY_NAMES
from lechonk.comparator import defaults as defaults_mod
from lechonk.core.ledger import TranscriptLedger
from lechonk.core.memory import StrategyMemory
from lechonk.core.history import DecisionEpisode, DecisionHistory
from lechonk.core.context_journal import ContextJournal
from lechonk.core.control_access import assess_control_access
from lechonk.core.metrics import compute_metrics, opportunity_domains
from lechonk.core.model_client import (
    REPAIR_CLASS_SCHEMA,
    REPAIR_CLASS_STATE_FOG,
    REPAIR_CLASS_UNRESPONSIVE,
    ModelClient,
    ModelResponse,
    PromptBudgetExceeded,
    RepairPrompt,
    TurnPrompt,
)
from lechonk.core.delta import compute_delta, enforce_delta_budget
from lechonk.core.jsonutil import canonical_json
from lechonk.core.recovery import validate_freeciv_username, validate_player_name
from lechonk.core.schemas import (
    ActionBatch,
    ActionResultRecord,
    Candidate,
    CandidateTable,
    CandidateTableCapExceeded,
    CommitmentRecord,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    RunManifest,
    RunManifestRecord,
    Snapshot,
    SnapshotRefRecord,
    CandidatesRefRecord,
    ThreatMention,
    TurnStartRecord,
    is_candidate_id,
)


class SessionError(RuntimeError):
    """Session lifecycle or resume provenance failed closed."""

# --- §3 supervision deadlines ------------------------------------------------
ACTING_DEADLINE_S = 180.0        # the model phase (§3 line 34)
PINGTIMEOUT_S = 720.0            # server pingtimeout (decision 8)
WAITING_MARGIN_S = 180.0         # 720 + margin = 900 (§3 line 34)
STARTUP_DEADLINE_S = 120.0       # §3 line 34
MUTATION_DEADLINE_S = 30.0       # per-mutation send+read-back (§3 line 34)
TEARDOWN_DEADLINE_S = 30.0       # §3 line 34
STALL_WATCHDOG_S = 600.0         # no transcript event for 10 min (§3 line 34)

# Persistent verbs whose model-originated orders the pipeline may continue
# once (continued_model_order, §3 line 38 — see module docstring).
PERSISTENT_VERBS = frozenset(
    {"fortify", "sentry", "build_road", "irrigation", "mine", "keep_activity"})

# Budget-aware repair gating (WP4): below this remaining ACTING budget at
# repair time the repair call is skipped entirely (degraded default applies).
# Below any observed successful repair latency (min ~18 s in wp4-eval), so
# an attempted repair could only time out.
REPAIR_BUDGET_FLOOR_S = 20.0
_MAX_CONVERSATION_REJECTIONS = 32
_MAX_CONVERSATION_REJECTION_REASON = 240

# The activity the snapshot reports while the order is in effect (used for
# the read-back proof of non-execution; obs_extract ACTIVITY_NAMES).
# freeciv's fortify has TWO in-effect states: ACTIVITY_FORTIFYING within the
# turn the order was given and ACTIVITY_FORTIFIED once it completes at the
# next turn start — both count as in effect (a fortifying unit must not be
# re-ordered: mutations are never re-sent while the order holds).
_ACTIVITY_FOR_VERB = {
    "fortify": ("fortified", "fortifying"), "sentry": "sentry",
    "build_road": "road", "irrigation": "irrigate", "mine": "mine",
}
# Structured threat-entity IDs (the metric never parses free text, §3
# line 41): non-conforming memory_update.threats entries are dropped at
# emission.
ENTITY_ID_RE = re.compile(r"^(unit|city|player):[0-9]+\Z")

# --- W3-1 dispatch-intent protocol (D5) ---------------------------------------
# The durable pre-send intent record and its correlation envelopes. The
# intent is a MetricEventRecord (no schema change, no freeze amendment);
# its value is a canonical-JSON string carrying the dispatch identity
# ONLY — nothing in it asserts that the send happened (the honesty
# requirement: a kill before the send and a kill after it leave the same
# durable trace). The action_result that follows an intent correlates it
# via a protocol-prefixed detail envelope; legacy detail strings (no
# envelope) remain valid.
DISPATCH_INTENT_EVENT = "dispatch_intent"
DISPATCH_CANCELLED_EVENT = "mutation_dispatch_cancelled"
DISPATCH_INTENT_PROTOCOL = "dispatch-intent/v1"
DISPATCH_CANCELLED_PROTOCOL = "dispatch-cancelled/v1"
DISPATCH_RESULT_PROTOCOL = "dispatch-result/v1"


def _dispatch_intent_payload(*, turn: int, ref: str,
                             call_seq: Optional[int], snapshot_seq: int,
                             candidates_seq: int, provenance: str) -> str:
    """The dispatch_intent event's canonical-JSON payload: the dispatch
    identity (protocol tag, turn, action ref, the provenance linkage seqs
    and the provenance class) — never an execution claim."""
    return canonical_json({
        "protocol": DISPATCH_INTENT_PROTOCOL,
        "turn": turn,
        "ref": ref,
        "call_seq": call_seq,
        "snapshot_seq": snapshot_seq,
        "candidates_seq": candidates_seq,
        "provenance_class": provenance,
    })


def _dispatch_cancelled_payload(*, intent_seq: int, reason: str,
                                turn: int, ref: str,
                                call_seq: Optional[int], snapshot_seq: int,
                                candidates_seq: int,
                                provenance: str) -> str:
    """The mutation_dispatch_cancelled event's canonical-JSON payload:
    the same payload keys as the intent, plus the intent's ledger seq
    (the correlation) and the blocking reason."""
    return canonical_json({
        "protocol": DISPATCH_CANCELLED_PROTOCOL,
        "intent_seq": intent_seq,
        "reason": reason,
        "turn": turn,
        "ref": ref,
        "call_seq": call_seq,
        "snapshot_seq": snapshot_seq,
        "candidates_seq": candidates_seq,
        "provenance_class": provenance,
    })


def _dispatch_result_detail(intent_seq: int, message: str) -> str:
    """The correlated action_result's detail: the protocol-prefixed
    dispatch-result/v1 envelope carrying the intent's ledger seq (the
    result->intent correlation). The existing detail rides verbatim in
    ``message``."""
    return canonical_json({
        "protocol": DISPATCH_RESULT_PROTOCOL,
        "intent_seq": intent_seq,
        "message": message,
    })


# --- W3-2 stop-at-deadline protocol (D6/D13, DEF-1/DEF-2) ----------------------
# The loop-level ACTING-deadline guards stop every mutation route at the
# deadline; the stop's partial-batch accounting is one aborted_group
# action_result per planned-but-undispatched action (the DEF-2
# compatibility extension of the frozen literal) plus ONE summary metric
# event (a partial-batch/v1 canonical-JSON payload — no schema change,
# no freeze amendment; the viewer's operational-event filter renders it
# as a diagnostic, never an aggregate).
ACTING_DEADLINE_STOP_EVENT = "acting_deadline_stop"
PARTIAL_BATCH_PROTOCOL = "partial-batch/v1"

# --- W3-9 engine-automatic observation events (D9/D12, Option A) -------------
# Observation-event diffing over a persistent last-seen baseline replaces
# the old turn-start-snapshot diff (_detect_engine_automatic, deleted) and
# the per-turn trackers — ONE detection path, no dual counting. The four
# events: turn start post-extraction; after every executed mutation (the
# session cache is the source — ExecutionResult carries no obs); after the
# verified resend (it completes inside _execute_one, so the caller's
# post-step event covers it); after successful phase completion (the
# finish outcome — attributed to the COMPLETING turn, never beyond the
# counted horizon).
ENGINE_OBSERVATION_GAP_EVENT = "engine_observation_gap"
# DEF-3: the retained per-interval executed-action set is bounded so it
# cannot grow unbounded; an overflow drops the entry and emits the
# observation-gap diagnostic (fail-open with mandatory disclosure).
ENGINE_OBSERVATION_INTERVAL_MAX = 4096
# freeciv's VUT_NONE production kind (version/v14.py; mirrors
# obs_extract.VUT_NONE — the adapter imports stay lazy, §7.1).
VUT_NONE = 0


def _json_object(value: Any) -> Optional[dict]:
    """The parsed JSON object behind a metric event's value or an
    action_result's detail (None when it is not a JSON object)."""
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _unresolved_dispatch_intents(records: Sequence[Any]) -> list[int]:
    """The W3-2 stop summary's unresolved_intent_seqs: the seqs of
    dispatch_intent metric events with neither a correlated
    action_result (a dispatch-result/v1 detail envelope carrying the
    intent's seq) nor a cancellation (a dispatch-cancelled/v1 payload
    carrying it). The correlation is BY SEQ, never positional (the same
    semantics as the P11 audit's pending-intent scanner — a crash inside
    the dispatch window leaves the intent forensic-only, D15; the runner
    reports it in the stop summary, never refuses on it)."""
    intents: set[int] = set()
    settled: set[int] = set()

    def _intent_seq(data: dict) -> Optional[int]:
        seq = data.get("intent_seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            return seq
        return None

    for rec in records:
        if isinstance(rec, MetricEventRecord):
            if rec.name == DISPATCH_INTENT_EVENT:
                intents.add(rec.seq)
            elif rec.name == DISPATCH_CANCELLED_EVENT:
                payload = _json_object(rec.value)
                if payload is not None and payload.get("protocol") \
                        == DISPATCH_CANCELLED_PROTOCOL:
                    seq = _intent_seq(payload)
                    if seq is not None:
                        settled.add(seq)
        elif isinstance(rec, ActionResultRecord):
            envelope = _json_object(rec.detail)
            if envelope is not None and envelope.get("protocol") \
                    == DISPATCH_RESULT_PROTOCOL:
                seq = _intent_seq(envelope)
                if seq is not None:
                    settled.add(seq)
    return sorted(intents - settled)


class RunState(str, Enum):
    STARTUP = "STARTUP"
    ACTING = "ACTING"
    FINISH_SENT = "FINISH_SENT"
    WAITING = "WAITING"
    NEXT_OWN_PHASE = "NEXT_OWN_PHASE"
    GAME_OVER = "GAME_OVER"
    FAILED = "FAILED"


class StateDeadlineExceeded(RuntimeError):
    """A state exceeded its per-state progress deadline (§3 line 34)."""

    def __init__(self, state: RunState, deadline_s: float):
        self.state = state
        self.deadline_s = deadline_s
        super().__init__(
            f"state {state.value} exceeded its deadline of {deadline_s:.1f}s")


class _AbortRun(RuntimeError):
    """Ordered shutdown requested (stall watchdog or supervisor)."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class _QuarantineFailStop(RuntimeError):
    """An uncertain mutation was quarantined after a disconnect (§3 line
    35): the bounded fail-stop — the mutation is NEVER resent (never
    blind replay), the run ends TRUNCATED with artifacts."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class RunnerConfig:
    """§3 supervision deadlines and run parameters."""

    # None means continue until the session reports victory or defeat.
    # Numeric values retain the bounded evaluation/smoke-run behavior.
    max_turns: Optional[int] = 50
    # ``legacy`` preserves the frozen evaluation prompt. ``hybrid_v1`` adds
    # bounded, completed-turn decision history for interactive runs.
    context_policy: str = "legacy"
    # Prompt presentation profile.  ``legacy_eval`` preserves the frozen
    # evaluation rendering; ``gameplay_v2`` opts into the richer gameplay
    # instructions and typed strategy-memory fields.
    prompt_profile: str = "legacy_eval"
    context_journal_path: Optional[str] = None
    # Optional continuation inputs.  These are validated before a session is
    # allowed to act; the CLI is responsible for checksum-validating the
    # parent checkpoint and staging its save.
    load_game: Optional[str] = None
    resume_turn: Optional[int] = None
    resume_player_id: Optional[int] = None
    initial_memory_state: Optional[Mapping[str, Any]] = None
    # Optional durable start-of-turn checkpointing.  A callback is not used
    # here so the runner can keep the export and manifest atomic from the
    # same turn boundary.
    checkpoint_dir: Optional[str] = None
    recovery_manifest_path: Optional[str] = None
    conversation_state_path: Optional[str] = None
    parent_run_id: Optional[str] = None
    recovery_config_digest: Optional[str] = None
    recovery_config_digest_schema: Optional[str] = None
    prompt_bundle_id: Optional[str] = None
    # Immutable lookup provenance expected on resume; checked after startup
    # and before any model call.
    expected_ruleset_reference_hash: Optional[str] = None
    expected_help_reference_hash: Optional[str] = None
    expected_help_reference_source_sha256: Optional[str] = None
    expected_classic_reference_schema_version: Optional[str] = None
    expected_classic_ruleset_tree_sha256: Optional[str] = None
    expected_classic_reference_present: Optional[bool] = None
    expected_manual_reference_schema_version: Optional[str] = None
    expected_manual_reference_hash: Optional[str] = None
    expected_manual_reference_source_sha256: Optional[str] = None
    expected_manual_reference_present: Optional[bool] = None
    validate_reference_provenance: bool = False
    run_id: str = "run"
    acting_deadline_s: float = ACTING_DEADLINE_S
    # FINISH_SENT/WAITING ≤ pingtimeout-derived bound (720 s + margin).
    waiting_deadline_s: float = PINGTIMEOUT_S + WAITING_MARGIN_S
    startup_deadline_s: float = STARTUP_DEADLINE_S
    mutation_deadline_s: float = MUTATION_DEADLINE_S
    teardown_deadline_s: float = TEARDOWN_DEADLINE_S
    stall_watchdog_s: float = STALL_WATCHDOG_S
    watchdog_interval_s: float = 5.0
    # Adapter-session parameters (the default session factory).
    workdir: str = "."
    seed: int = 42
    username: str = "myagent"
    player_name: Optional[str] = None
    aifill: int = 2
    # Candidate admission is deliberately independent from the prompt
    # profile.  The legacy policy remains the default so existing runs and
    # cache/recovery lineages are byte-for-byte unchanged.
    candidate_surface_policy: str = "legacy_v1"
    candidate_surface_cap: Optional[int] = None
    candidate_admission_version: Optional[str] = None
    compaction_schema: Optional[str] = None
    prompt_budget_schema: Optional[str] = None

    def __post_init__(self) -> None:
        validate_freeciv_username(self.username)
        if self.player_name is not None:
            self.player_name = validate_player_name(self.player_name)
        # CivRealm's bundled server uses maxplayers=10.  The controlled
        # human occupies one slot, so aifill is the total number of filled
        # player slots (the Freeciv server's semantics), and may not exceed
        # maxplayers=10. Fail before Docker/session startup rather
        # than allowing Freeciv to silently clamp the scenario.
        if not 1 <= int(self.aifill) <= 10:
            raise ValueError(
                "aifill must be between 1 and 10 (maxplayers=10, "
                "including the human slot)")
        self.aifill = int(self.aifill)
        if self.prompt_profile not in {"legacy_eval", "gameplay_v1", "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}:
            raise ValueError(
                "prompt_profile must be legacy_eval, gameplay_v1, gameplay_v2, gameplay_v3, gameplay_v4, gameplay_v5, or gameplay_v6")
        if self.candidate_surface_policy not in {"legacy_v1", "candidate_surface_v2", "candidate_surface_v3"}:
            raise ValueError(
                "candidate_surface_policy must be legacy_v1, candidate_surface_v2, or candidate_surface_v3")
        if self.prompt_profile in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}:
            policy = "candidate_surface_v3" if self.prompt_profile == "gameplay_v6" else "candidate_surface_v2"
            cap = 12000 if self.prompt_profile == "gameplay_v6" else 8000
            admission = "candidate_surface_v3-r1" if self.prompt_profile == "gameplay_v6" else "candidate_surface_v2-r1"
            expected = (self.prompt_profile, policy, cap,
                        admission, "conversation-compaction-v1",
                        f"{self.prompt_profile.replace('_', '-')}-budget-v1")
            actual = (self.prompt_bundle_id, self.candidate_surface_policy,
                      self.candidate_surface_cap, self.candidate_admission_version,
                      self.compaction_schema, self.prompt_budget_schema)
            if actual != expected:
                raise ValueError(f"{self.prompt_profile} requires its versioned prompt/policy/budget identity")
            if self.recovery_config_digest_schema != "interactive-config-v3":
                raise ValueError(f"{self.prompt_profile} requires interactive-config-v3 recovery schema")
        elif self.candidate_surface_policy != "legacy_v1":
            raise ValueError("candidate surface policy requires a versioned gameplay profile")


@dataclass
class RunResult:
    outcome: str  # COMPLETED | DEFEATED | VICTORY | TRUNCATED
    reason: str
    turns_completed: int
    ledger_path: str
    metrics: Any  # core.metrics.MetricsReport
    close_status: str


@dataclass
class _PlannedAction:
    ref: str
    candidate: Candidate


@dataclass
class _IngestResult:
    """One model response's ingestion: the valid batch (if any), the
    planned (executable) actions after group-abort filtering, and the
    invalid refs with their taxonomy classes."""

    batch: Optional[ActionBatch]
    planned: list[_PlannedAction] = field(default_factory=list)
    invalid: list[tuple[str, str]] = field(default_factory=list)
    failed: bool = False  # no valid batch at all (transport error / malformed)
    call_seq: Optional[int] = None  # the model_call record's seq
    why: str = ""  # why the response was unusable ("" when usable; the
    # repair/degraded reason — "no actions" only for an empty batch)
    degraded: bool = False  # the degraded default applies
    # FW-R2: every attempt's memory_update for the turn, in attempt order;
    # _model_phase attaches the shared list to the result it returns.
    attempt_memory_updates: list = field(default_factory=list)
    # FW-16: every attempt's batch-level ActionBatch.commitments, in
    # attempt order; _model_phase attaches the shared list to the result
    # it returns.
    attempt_batch_commitments: list = field(default_factory=list)


@dataclass
class _StandingOrder:
    """A model-originated persistent order the pipeline may continue once
    (continued_model_order, linked to its origin)."""

    actor_id: str
    verb: str
    ref: str
    origin_seq: int
    reissued: bool = False


class _AdapterModules:
    """The lazily-imported adapter modules (never at module level, §7.1)."""

    def __init__(self, candidates, execute, obs_extract):
        self.candidates = candidates
        self.execute = execute
        self.obs_extract = obs_extract


def _default_session_factory(config: RunnerConfig) -> Any:
    """The production session factory. The adapter import is LAZY: the
    civrealm import happens inside AdapterSession.start() (worker thread),
    only after the Docker stack is up — import failure is a startup failure,
    not a crash (§7.1)."""
    from lechonk.adapter.session import AdapterSession  # noqa: PLC0415

    return AdapterSession(
        workdir=config.workdir,
        seed=config.seed,
        username=config.username,
        player_name=config.player_name,
        aifill=config.aifill,
        # §3 startup ≤ 120 s + server-spawn margin (the session default).
        startup_deadline=config.startup_deadline_s + 30.0,
        step_deadline=config.mutation_deadline_s,
        wait_deadline=config.waiting_deadline_s,
        close_deadline=config.teardown_deadline_s,
        max_turns=config.max_turns,
        acting_deadline_s=config.acting_deadline_s,
        load_game=config.load_game,
    )


class StallWatchdog:
    """Run-level stall watchdog (§3 line 34): no transcript event for
    ``stall_s`` → the ordered shutdown path. Pure backstop (D14): it
    fires on a stall only once the current phase's deadline has also
    passed — or when the phase publishes no deadline (terminal states).
    A legal WAITING interval is never aborted before its WAITING
    deadline, which ends the wait itself (D7). Watches the ledger's
    last-event clock reading; it never enforces a phase deadline itself
    and never writes to the ledger."""

    def __init__(self, *, stall_s: float, interval_s: float,
                 clock: Callable[[], float], ledger: TranscriptLedger,
                 on_stall: Callable[[str], None],
                 phase_state: Callable[[], tuple[int, Optional[float]]]):
        self._stall_s = stall_s
        self._interval_s = interval_s
        self._clock = clock
        self._ledger = ledger
        self._on_stall = on_stall
        self._phase_state = phase_state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="lechonk-stall-watchdog", daemon=True)
        self._thread.start()

    def _tick(self) -> bool:
        """One watchdog evaluation: True iff the stall abort was
        requested. Fires iff stalled AND the phase deadline has passed
        (or none is published). The (generation, deadline) pair is
        captured at tick start and re-verified before the abort request:
        a stale tick cannot abort a newly entered phase."""
        now = self._clock()
        if now - self._ledger.last_event <= self._stall_s:
            return False
        generation, deadline = self._phase_state()
        if deadline is not None and now < deadline:
            return False
        if self._phase_state() != (generation, deadline):
            return False  # a phase was entered mid-tick: stale tick
        self._on_stall(
            f"run-level stall watchdog: no transcript event for "
            f"{self._stall_s:.0f}s")
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._tick():
                return
            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()


def mark_armed_in_contact(table: CandidateTable, snapshot: Snapshot) -> None:
    """FW-13: candidates of armed-in-contact own units are never elided
    by the candidate-table cap (the §3 overflow policy keeps the tactical
    surface intact; idle/distant candidates elide first). Shared with the
    comparator arm's reconstruction of the runner's offered surface
    (eval/paired) so both arms see the same candidate membership."""
    contact = {str(u.id) for u in snapshot.units if u.in_contact}
    for cand in table.candidates:
        if cand.domain == "unit" and cand.actor_id in contact:
            cand.in_contact = True


class GameRunner:
    """The turn pipeline + state machine + supervisor (§3 lines 32-34).

    Drives one run: startup → per-turn (ACTING → FINISH_SENT → WAITING →
    NEXT_OWN_PHASE) → GAME_OVER/FAILED, emitting TranscriptRecords through
    the ledger for every state transition and action.
    """

    def __init__(self, *, config: RunnerConfig, client: ModelClient,
                 ledger: TranscriptLedger, manifest: RunManifest,
                 session_factory: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 progress: Optional[Callable[[dict[str, Any]], None]] = None):
        self.config = config
        self.client = client
        self.ledger = ledger
        self.manifest = manifest
        self._session_factory = session_factory or (
            lambda: _default_session_factory(config))
        self._clock = clock
        # Human-facing observability is deliberately best-effort: a broken
        # progress consumer must never change game behavior or ledger output.
        self._progress_callback = progress
        if config.initial_memory_state is None:
            self.memory = StrategyMemory()
        else:
            self.memory = StrategyMemory.from_state(config.initial_memory_state)
        self.history = DecisionHistory()
        self._context_journal: Optional[ContextJournal] = None
        if self.config.context_policy == "hybrid_v1" and self.config.context_journal_path:
            journal_path = Path(self.config.context_journal_path)
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._context_journal = ContextJournal.attach(
                journal_path, policy_version="hybrid_v1")
        self._history_results: list[dict[str, Any]] = []
        self._history_rejections: list[dict[str, Any]] = []
        self._history_collecting = False
        self._adapter: Optional[_AdapterModules] = None
        self._state = RunState.STARTUP
        self._state_entered: float = clock()
        # D7/D14: the stall watchdog's phase view — the state generation
        # (bumped by every _transition; the stale-tick guard) and the
        # current phase's absolute deadline published by _transition
        # (None in terminal states: the plain stall rule applies). Plain
        # attributes shared with the watchdog thread (the
        # ledger.last_event precedent).
        self._state_generation = 0
        self._phase_deadline_at: Optional[float] = (
            self._state_entered + config.startup_deadline_s)
        # The ACTING deadline's single source (W3-2 reuses this
        # attribute; it creates no second source). Published by
        # _transition(ACTING); its value coincides with the acting_entered
        # _run_turn captures from the same transition instant.
        self._acting_deadline_at: Optional[float] = None
        self._abort = threading.Event()
        self._abort_reason: Optional[str] = None
        self._session: Any = None
        self._watchdog: Optional[StallWatchdog] = None
        # Standing orders (continued_model_order) + the W3-9
        # engine-automatic observation tracker.
        self._standing: dict[str, _StandingOrder] = {}
        # The persistent last-seen baseline: own-city production display
        # names + the current research target id, fed at the four
        # observation events (see the W3-9 constants above).
        self._engine_baseline_seen = False
        self._engine_baseline_cities: dict[int, Optional[str]] = {}
        self._engine_baseline_research: Optional[int] = None
        # Every city id ever baselined: a re-sighting after absence is
        # baseline-only, never a fresh choice (hardening b).
        self._engine_known_cities: set[int] = set()
        # Corroborated foundings (the founding verifier's new_cities):
        # a confirmed new own city gets an initial production-resolution
        # event even when the founding itself was model-directed.
        self._engine_founded: set[int] = set()
        # The per-interval attribution set: the produce/research actions
        # executed since the last observation event (bounded, DEF-3).
        self._engine_interval_actions: set[tuple] = set()
        # City ids founded (verified) since the last observation event.
        self._engine_interval_foundings: set[int] = set()
        # The observation-event counter (the engine-auto refs' uniqueness).
        self._engine_observation_index = 0
        self._game_over: Optional[tuple[bool, bool]] = None  # (terminated, truncated)
        # The TRUE count of turns that completed (reached a turn boundary /
        # NEXT_OWN_PHASE) — survives an exception unwind out of _run_turns,
        # so the run result never reports a stale 0 (WP4 fix).
        self._turns_completed = 0

    def _progress(self, event: str, **fields: Any) -> None:
        """Publish a sanitized, non-sensitive live progress event."""
        if self._progress_callback is None:
            return
        payload = {"event": event}
        payload.update(fields)
        try:
            self._progress_callback(payload)
        except Exception:
            # Observability is never allowed to interrupt a run.
            return

    def _admit_candidates(self, table: CandidateTable):
        """Return the offered table and optional versioned admission result.

        Keeping this switch in the runner makes the legacy path use the exact
        pre-policy ``CandidateTable`` behavior.  The v2 module is imported
        only when explicitly selected, and is intentionally not coupled to a
        prompt profile or recovery digest yet.
        """
        if self.config.candidate_surface_policy == "candidate_surface_v3":
            from lechonk.core.candidate_surface_v3 import admit_candidate_surface
            admission = admit_candidate_surface(table, cap_tokens=self.config.candidate_surface_cap or 12000)
            return admission.offered, admission
        if self.config.candidate_surface_policy == "candidate_surface_v2":
            from lechonk.core.candidate_surface_v2 import (  # noqa: PLC0415
                admit_candidate_surface,
            )
            admission = admit_candidate_surface(table)
            return admission.offered, admission
        return table.offered(), None

    # ------------------------------------------------------------------
    # adapter (lazy, §7.1)
    # ------------------------------------------------------------------

    def _load_adapter(self) -> _AdapterModules:
        if self._adapter is None:
            # Lazy: only after the stack is up. These modules are
            # civrealm-free; the civrealm-importing session module is
            # imported inside the default session factory.
            from lechonk.adapter import candidates as candidates_mod  # noqa: PLC0415
            from lechonk.adapter import execute as execute_mod  # noqa: PLC0415
            from lechonk.adapter import obs_extract  # noqa: PLC0415

            self._adapter = _AdapterModules(
                candidates_mod, execute_mod, obs_extract)
        return self._adapter

    # ------------------------------------------------------------------
    # ledger emission helpers
    # ------------------------------------------------------------------

    def _transition(self, state: RunState) -> None:
        prev = self._state
        previous_entered = self._state_entered
        self._state = state
        self._state_entered = self._clock()
        # D7/D14: publish the phase's absolute deadline for the stall
        # watchdog. FINISH_SENT is transient but gets the waiting
        # deadline, not the plain rule; terminal states publish None.
        if state is RunState.STARTUP:
            self._phase_deadline_at = (
                self._state_entered + self.config.startup_deadline_s)
        elif state is RunState.ACTING:
            self._phase_deadline_at = (
                self._state_entered + self.config.acting_deadline_s)
            self._acting_deadline_at = self._phase_deadline_at
        elif state in (RunState.FINISH_SENT, RunState.WAITING):
            self._phase_deadline_at = (
                self._state_entered + self.config.waiting_deadline_s)
        else:  # GAME_OVER / FAILED / NEXT_OWN_PHASE: terminal
            self._phase_deadline_at = None
        # Bumped AFTER the deadline publication: a watchdog tick that
        # observes the new generation also observes the new deadline.
        self._state_generation += 1
        self.ledger.append(MetricEventRecord(
            name="state_transition", value=f"{prev.value}->{state.value}"))
        self._progress(
            "state",
            from_state=prev.value,
            state=state.value,
            elapsed_s=round(self._state_entered - previous_entered, 3),
        )

    def _emit(self, record: Any, *, durable: bool = False) -> int:
        return self.ledger.append(record, durable=durable)

    def _emit_action_result(self, *, durable: bool = False, **fields: Any) -> int:
        # Construct once so the live progress event reflects schema defaults
        # (including provenance_class), rather than only fields explicitly
        # supplied by the caller.
        record = ActionResultRecord(**fields)
        seq = self._emit(record, durable=durable)
        # Rejections are feedback, rather than executable history. Keep a
        # bounded copy for the durable conversation report even when this
        # result was emitted during initial ingest (before the legacy
        # history-collection window opens), or during revalidation.
        if (fields.get("outcome") == "rejected"
                and len(self._history_rejections)
                < _MAX_CONVERSATION_REJECTIONS):
            reason = str(fields.get("detail") or fields.get("reason") or "")
            self._history_rejections.append({
                "ref": fields.get("ref"),
                "rejection_class": fields.get("rejection_class"),
                "provenance_class": fields.get("provenance_class"),
                "reason": reason[:_MAX_CONVERSATION_REJECTION_REASON],
            })
        if self._history_collecting:
            # Keep only normalized, non-executable evidence for the next
            # prompt.  The raw ActionResultRecord remains the provenance
            # source in the transcript.
            self._history_results.append({
                key: fields.get(key)
                for key in ("domain", "verb", "target", "affected",
                            "outcome", "rejection_class",
                            "provenance_class", "readback")
                if fields.get(key) is not None
            })
        self._progress(
            "action_result",
            turn=fields.get("turn"), ref=fields.get("ref"),
            domain=fields.get("domain"), verb=fields.get("verb"),
            # Preserve the structured location descriptors in the live
            # capture stream.  ``actor_tile`` is a tuple in the frozen
            # ledger schema; publish it as a JSON-native list so observers
            # can safely use it for camera focus without parsing repr().
            target=fields.get("target"),
            affected=fields.get("affected"),
            actor_tile=(list(fields["actor_tile"])
                        if fields.get("actor_tile") is not None else None),
            outcome=fields.get("outcome"),
            provenance_class=record.provenance_class,
            readback=fields.get("readback"),
        )
        return seq

    def _commit_history_episode(self, *, turn: int, result: _IngestResult,
                                repair_used: bool, status: str = "completed") -> None:
        """Publish one completed turn to the optional hybrid journal."""
        if self.config.context_policy != "hybrid_v1":
            return
        planned = []
        for item in result.planned:
            candidate = item.candidate
            planned.append({
                "domain": candidate.domain,
                "verb": candidate.verb,
                "target": candidate.actor_id,
            })
        notes = result.batch.notes if result.batch is not None else ""
        self.history.add(DecisionEpisode.from_parts(
            turn=turn, notes=notes, planned=planned,
            results=self._history_results, degraded=result.degraded,
            repair_used=repair_used,
            status=("completed_degraded" if result.degraded and status == "completed"
                    else status)))
        if self._context_journal is not None:
            rendered = self.history.render()
            rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            # This append is a durability boundary.  If it fails, surface the
            # error before another model turn can observe an unjournaled
            # episode; the run's normal infrastructure-failure path records
            # the failure and preserves the transcript.
            self._context_journal.append(
                turn=turn, episode=self.history.episodes[-1].as_dict(),
                transcript_seq=self.ledger.last_seq,
                rendered_history_hash=rendered_hash, durable=True)

    def _commit_conversation_turn(self, *, turn: int,
                                  result: _IngestResult,
                                  repair_used: bool) -> None:
        """Commit a completed conversation turn after execution is known.

        ``MistralTurnClient`` owns the exact message/chunk history.  The
        runner supplies only normalized execution evidence, and does so after
        ``memory.advance`` and the finish phase.  This keeps a planned action
        from being replayed as if it succeeded.  The duck-typed call keeps
        scripted/fake clients and the legacy policy model-agnostic.
        """
        if self.config.context_policy != "conversation_v1":
            return
        commit = getattr(self.client, "commit_turn", None)
        if not callable(commit):
            raise RuntimeError(
                "conversation_v1 requires a client with commit_turn()")
        planned = []
        for item in result.planned:
            candidate = item.candidate
            planned.append({
                "domain": candidate.domain,
                "verb": candidate.verb,
                "target": candidate.actor_id,
            })
        report = {
            "historical_turn": turn,
            "status": "completed_degraded" if result.degraded else "completed",
            "repair_used": repair_used,
            "intent": {
                "notes": result.batch.notes if result.batch is not None else "",
                "planned": planned,
            },
            "execution": list(self._history_results),
            "rejected": list(self._history_rejections),
            "authority": "CURRENT_SNAPSHOT_AND_CANDIDATES_OVERRIDE_HISTORY",
            "rejection_guidance": (
                "Rejected refs are unavailable unless explicitly re-offered "
                "in the current candidate table; choose exact refs from "
                "CURRENT_SNAPSHOT_AND_CANDIDATES."),
        }
        commit(report)

    def _abort_conversation_turn(self) -> None:
        """Discard a staged conversation turn on any incomplete run path."""
        if self.config.context_policy != "conversation_v1":
            return
        abort = getattr(self.client, "abort_turn", None)
        if callable(abort):
            abort()

    # ------------------------------------------------------------------
    # the W3-1 dispatch-intent boundary (D5)
    # ------------------------------------------------------------------

    def _dispatch_block_reason(self) -> Optional[str]:
        """The pre-send deadline/game-over check (the pinned send-site
        order's first check and its FINAL recheck). An ALREADY-ESTABLISHED
        game-over wins over a newly-noticed deadline expiry (DEF-1); the
        ACTING deadline is the single source published by _transition
        (W3-3). None when the dispatch may proceed (no deadline published:
        proceed)."""
        if self._game_over is not None:
            return "game over"
        deadline = self._acting_deadline_at
        if deadline is not None and self._clock() >= deadline:
            return "ACTING deadline reached"
        return None

    def _emit_dispatch_intent(self, *, turn: int, ref: str,
                              call_seq: Optional[int], snapshot_seq: int,
                              candidates_seq: int, provenance: str) -> int:
        """Append the durable dispatch_intent record (D5: the pre-send
        fsync — the ledger's durable append flushes AND fsyncs, so the
        intent is on disk before the send)."""
        return self._emit(MetricEventRecord(
            name=DISPATCH_INTENT_EVENT,
            value=_dispatch_intent_payload(
                turn=turn, ref=ref, call_seq=call_seq,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                provenance=provenance)), durable=True)

    def _emit_dispatch_cancelled(self, *, intent_seq: int, reason: str,
                                 turn: int, ref: str,
                                 call_seq: Optional[int], snapshot_seq: int,
                                 candidates_seq: int, provenance: str) -> int:
        """Append the durable mutation_dispatch_cancelled record (the
        cancellation contract: the same payload keys as the intent +
        intent_seq). Emitted when the FINAL pre-send check blocks the
        dispatch — NO send occurs in that branch."""
        return self._emit(MetricEventRecord(
            name=DISPATCH_CANCELLED_EVENT,
            value=_dispatch_cancelled_payload(
                intent_seq=intent_seq, reason=reason, turn=turn, ref=ref,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, provenance=provenance)),
            durable=True)

    def _blocked_dispatch(self, adapter: _AdapterModules, cand: Candidate,
                          *, block: str, pre_info: Mapping[str, Any],
                          intent_seq: Optional[int]) -> tuple[Any, int]:
        """A dispatch blocked by the pre-send check (W3-1): NO send and NO
        action_result are fabricated for it. The deadline arm raises the
        supervision deadline (the existing bounded truncation path); the
        game-over arm (DEF-1: an established game-over wins over a
        newly-noticed deadline expiry) returns a not-dispatched result
        carrying the termination, so the caller's W2-4 handling stops the
        batch and the outcome flows through the game-over path."""
        if block == "game over":
            terminated, truncated = self._game_over or (False, False)
            detail = ("not dispatched: the game is over — no send, no "
                      "action_result (the W3-1 dispatch-intent boundary)")
            turn_before = int((pre_info or {}).get("turn", 0) or 0)
            return adapter.execute.ExecutionResult(
                candidate_id=cand.id, action=("", "", ""),
                status="not_dispatched",
                readback=adapter.execute.Readback(status="none",
                                                  detail=detail),
                terminated=bool(terminated), truncated=bool(truncated),
                turn_before=turn_before, turn_after=turn_before,
                elapsed_s=0.0, detail=detail), (
                    intent_seq if intent_seq is not None
                    else self.ledger.last_seq)
        # The deadline arm: raise the supervision deadline (the existing
        # bounded truncation with artifacts).
        raise StateDeadlineExceeded(
            RunState.ACTING, self.config.acting_deadline_s)

    # ------------------------------------------------------------------
    # abort / watchdog
    # ------------------------------------------------------------------

    def _request_abort(self, reason: str) -> None:
        """Ordered shutdown (stall watchdog): record the reason, set the
        abort event, and close the session guarded (best effort — the
        runner unwinds at the next boundary)."""
        if self._abort_reason is None:
            self._abort_reason = reason
        self._abort.set()
        session = self._session
        if session is not None:
            try:
                session.close()
            except Exception:
                pass  # the unwind path reports the abort reason

    def _configure_ruleset_lookup(self) -> None:
        """Install the bounded local encyclopedia callable for gameplay.

        The core runner remains model-agnostic: the concrete turn client may
        opt into this hook (the legacy/fake clients simply do not implement
        it).  The callable closes over the immutable reference collected by
        ``AdapterSession`` and never reaches into the live game, so a model
        tool call cannot mutate state.  It is deliberately enabled only for
        the interactive conversation profile; legacy requests retain their
        exact request shape and attempt accounting.
        """
        if (self.config.context_policy != "conversation_v1" or
                self.config.prompt_profile not in {"gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}):
            return
        setter = getattr(self.client, "set_ruleset_lookup", None)
        if not callable(setter):
            # Keep a small compatibility seam while the model adapter rolls
            # out the named hook; no dynamic attribute assignment is used.
            setter = getattr(self.client, "configure_ruleset_lookup", None)
        reference = getattr(self._session, "ruleset_reference", None)
        # AdapterSession may expose the provisioned static help index under
        # either name while that lifecycle seam is rolled out.  It is a
        # separate, non-authoritative source: only the live ruleset reference
        # may describe the current game.
        generic_help = getattr(self._session, "help_reference", None)
        if generic_help is None:
            generic_help = getattr(self._session, "generic_help_reference", None)
        if generic_help is None:
            generic_help = getattr(self._session, "freeciv_help_reference", None)
        # The extracted classic ruleset index is optional and deliberately
        # separate from helpdata: it is useful historical/reference prose,
        # but is never authoritative for the live game's ruleset.
        classic = getattr(self._session, "classic_reference", None)
        if classic is None:
            classic = getattr(self._session, "classic_ruleset_reference", None)
        manual = getattr(self._session, "manual_reference", None)
        if manual is None:
            manual = getattr(self._session, "freeciv_manual_reference", None)
        if manual is None:
            manual = getattr(self._session, "manual_help_reference", None)
        if self.config.validate_reference_provenance:
            help_source_sha = getattr(generic_help, "source_sha256", None)
            help_status = getattr(generic_help, "status", None)
            session_help_hash = getattr(self._session, "help_reference_hash", None)
            actual_help_hash = (session_help_hash if session_help_hash is not None
                                else getattr(generic_help, "content_hash", None))
            if help_status not in (None, "available"):
                actual_help_hash = None
                help_source_sha = None
            classic_status = getattr(self._session, "classic_reference_status", None)
            if callable(classic_status):
                classic_status = classic_status()
            classic_status = (getattr(classic, "status", classic_status)
                              if classic is not None else classic_status)
            classic_schema = getattr(classic, "schema_version", None)
            classic_tree = getattr(classic, "tree_sha256", None)
            classic_present = (None if classic is None and
                               classic_status is None else
                               classic_status in ("available", "complete"))
            manual_status = getattr(self._session, "manual_reference_status", None)
            if callable(manual_status):
                manual_status = manual_status()
            manual_status = (getattr(manual, "status", manual_status)
                             if manual is not None else manual_status)
            manual_schema = getattr(manual, "schema_version", None)
            manual_source_sha = getattr(manual, "source_sha256", None)
            manual_hash = getattr(manual, "content_hash", None)
            manual_present = (None if manual is None and manual_status is None
                              else manual_status in ("available", "complete"))
            if not manual_present:
                manual_schema = None
                manual_hash = None
                manual_source_sha = None
            if (getattr(reference, "content_hash", None)
                    != self.config.expected_ruleset_reference_hash
                    or actual_help_hash
                    != self.config.expected_help_reference_hash
                    or help_source_sha !=
                    self.config.expected_help_reference_source_sha256
                    or classic_schema != self.config.expected_classic_reference_schema_version
                    or classic_tree != self.config.expected_classic_ruleset_tree_sha256
                    or classic_present != self.config.expected_classic_reference_present
                    or manual_schema != self.config.expected_manual_reference_schema_version
                    or manual_hash != self.config.expected_manual_reference_hash
                    or manual_source_sha != self.config.expected_manual_reference_source_sha256
                    or manual_present != self.config.expected_manual_reference_present):
                raise SessionError(
                    "encyclopedia reference provenance changed on resume")
        if not callable(setter):
            return
        reference_status = getattr(
            self._session, "ruleset_reference_status", None)
        if callable(reference_status):
            reference_status = reference_status()
        if reference_status is None:
            reference_status = "available" if reference is not None else "unavailable"
        self._progress("ruleset_reference_status", status=str(reference_status),
                       content_hash=getattr(reference, "content_hash", None))

        def lookup(query: str, categories=None, limit: int = 8):
            """Read-only, bounded lookup against the startup snapshot."""
            safe_query = query if isinstance(query, str) else str(query or "")
            safe_query = safe_query[:256]
            try:
                safe_limit = max(1, min(int(limit), 8))
            except (TypeError, ValueError):
                safe_limit = 8
            self._progress("ruleset_lookup_started", query=safe_query,
                           limit=safe_limit)
            manual_exposed = (manual is not None or
                              hasattr(self._session, "manual_reference_status"))
            if (reference is None and generic_help is None and classic is None
                    and manual is None):
                result = {"status": "unavailable", "query": safe_query,
                          "entries": [], "reason": str(reference_status),
                          "sources": {"active_ruleset": "unavailable",
                                      "static_help": "unavailable",
                                      "static_classic": "unavailable",
                                      **({"static_manual": "unavailable"}
                                         if manual_exposed else {})},
                          "provenance": {}}
            else:
                # Lazy import preserves the core module's no-network-at-
                # import invariant and keeps the reference implementation in
                # the adapter package.
                from lechonk.adapter.ruleset_reference import render_search
                from lechonk.adapter.help_reference import (
                    render_search as render_help_search)
                # Keep this import lazy: classic_index is an optional
                # provisioned artifact and must not affect core imports.
                classic_render = None
                if classic is not None:
                    try:
                        from lechonk.adapter.classic_reference import (
                            render_search as classic_render)
                    except (ImportError, AttributeError):
                        classic_render = None
                manual_render = None
                if manual is not None:
                    try:
                        from lechonk.adapter.manual_reference import (
                            render_search as manual_render)
                    except (ImportError, AttributeError):
                        manual_render = None
                try:
                    active = ({"status": "unavailable", "entries": []}
                              if reference is None else render_search(
                                  reference, safe_query,
                                  categories=(categories or None),
                                  limit=safe_limit, max_chars=4000))
                    if reference is not None and "status" not in active:
                        active["status"] = "available"
                    static = ({"status": "unavailable", "entries": []}
                              if generic_help is None else render_help_search(
                                  generic_help, safe_query,
                                  categories=(categories or None),
                                  limit=safe_limit,
                                  max_chars=4000))
                    classic_result = ({"status": "unavailable", "entries": []}
                                      if classic is None or classic_render is None
                                      else classic_render(
                                          classic, safe_query,
                                          categories=(categories or None),
                                          limit=safe_limit, max_chars=4000))
                    manual_result = ({"status": "unavailable", "entries": []}
                                     if manual is None or manual_render is None
                                     else manual_render(
                                         manual, safe_query,
                                         categories=(categories or None),
                                         limit=safe_limit, max_chars=4000))
                    # Renderers may return only entries/status; always expose
                    # immutable source provenance in the tool result.
                    for key, attr in (("content_hash", "content_hash"),
                                      ("source_sha256", "source_sha256"),
                                      ("source", "source"),
                                      ("source_path", "source_path"),
                                      ("tree_sha256", "tree_sha256"),
                                      ("schema_version", "schema_version")):
                        if key not in classic_result:
                            value = getattr(classic, attr, None)
                            if value is not None:
                                classic_result[key] = value
                    for key, attr in (("content_hash", "content_hash"),
                                      ("source_sha256", "source_sha256"),
                                      ("source", "source"),
                                      ("source_path", "source_path"),
                                      ("schema_version", "schema_version")):
                        if key not in manual_result:
                            value = getattr(manual, attr, None)
                            if value is not None:
                                manual_result[key] = value
                    entries = []
                    for entry in active.get("entries", []):
                        item = dict(entry)
                        item["authority"] = "active_ruleset"
                        entries.append(item)
                    for entry in manual_result.get("entries", []):
                        item = dict(entry)
                        item["authority"] = "static_manual_non_authoritative"
                        entries.append(item)
                    for entry in static.get("entries", []):
                        item = dict(entry)
                        item["authority"] = "static_help_non_authoritative"
                        entries.append(item)
                    for entry in classic_result.get("entries", []):
                        item = dict(entry)
                        item["authority"] = "static_classic_non_authoritative"
                        entries.append(item)
                    # Live data wins, followed by the generated manual and
                    # then other static indexes. Only records from the same
                    # source/category/name are deduplicated.
                    unique = {}
                    authority_rank = {"active_ruleset": 0,
                                      "static_manual_non_authoritative": 1,
                                      "static_help_non_authoritative": 2,
                                      "static_classic_non_authoritative": 3}
                    for item in entries:
                        identity = (
                            item.get("authority", ""),
                            str(item.get("category", "")).casefold(),
                            (str(item.get("name", "")).casefold()
                             or str(item.get("key", "")).casefold()),
                        )
                        old = unique.get(identity)
                        if (old is None or authority_rank.get(
                                item.get("authority"), 99) < authority_rank.get(
                                old.get("authority"), 99)):
                            unique[identity] = item
                    entries = list(unique.values())
                    query_folded = safe_query.casefold()
                    entries.sort(key=lambda item: (
                        0 if str(item.get("name", "")).casefold() == query_folded
                        or str(item.get("key", "")).casefold() == query_folded
                        else 1,
                        authority_rank.get(item.get("authority"), 99),
                        str(item.get("name", "")).casefold(),
                        str(item.get("key", ""))))
                    active_status = active.get("status", "unavailable")
                    static_status = static.get("status", "unavailable")
                    result = {
                        "status": ("ok" if active.get("entries") or
                                   static.get("entries") or
                                   classic_result.get("entries") or
                                   manual_result.get("entries") or
                                   active_status == "available" or
                                   static_status == "available" or
                                   classic_result.get("status") == "available" or
                                   manual_result.get("status") == "available"
                                   else "unavailable"),
                        "query": safe_query,
                        "entries": entries[:safe_limit],
                        "sources": {
                            "active_ruleset": active_status,
                            "static_help": static_status,
                            "static_classic": classic_result.get(
                                "status", "unavailable"),
                            **({"static_manual": manual_result.get(
                                "status", "unavailable")} if manual_exposed else {}),
                        },
                        "provenance": {
                            "active_ruleset": {
                                key: active.get(key)
                                for key in ("source", "source_path", "source_sha256",
                                            "content_hash")
                                if active.get(key) is not None
                            },
                            "static_help": {
                                key: static.get(key)
                                for key in ("source", "source_path", "source_sha256",
                                            "content_hash")
                                if static.get(key) is not None
                            },
                            "static_classic": {
                                key: classic_result.get(key)
                                for key in ("source", "source_path",
                                            "source_sha256", "content_hash",
                                            "tree_sha256",
                                            "schema_version")
                                if classic_result.get(key) is not None
                            },
                            "static_manual": {
                                key: manual_result.get(key)
                                for key in ("source", "source_path",
                                             "source_sha256", "content_hash",
                                             "schema_version")
                                if manual_result.get(key) is not None
                            },
                        },
                    }
                    # Bound the complete callable result, not merely each
                    # source independently.  Keep it valid JSON because the
                    # turn client wraps this result in a role=tool message.
                    while (len(json.dumps(result, ensure_ascii=False,
                                          separators=(",", ":"))) > 4000
                           and result["entries"]):
                        result["entries"].pop()
                except Exception as exc:
                    # A malformed tool argument or unavailable reference is a
                    # normal lookup miss, never a reason to fail the turn.
                    result = {"status": "error", "query": safe_query,
                              "entries": [], "error": str(exc)[:240]}
            self._progress("ruleset_lookup_finished", query=safe_query,
                           status=result.get("status", "ok"),
                           entry_count=len(result.get("entries", [])),
                           result_preview=json.dumps(
                               result, ensure_ascii=False,
                               separators=(",", ":"))[:1200])
            return result

        try:
            setter(lookup)
        except TypeError:
            # A transitional adapter may accept the reference and build the
            # callable itself.  Keep this fallback bounded and explicit.
            setter(reference)
    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise _AbortRun(self._abort_reason or "abort requested")

    def _write_start_checkpoint(self, turn: int) -> None:
        """Export the begin-turn save before model/action work starts.

        This is deliberately fail-closed: a run configured for recovery must
        never continue after it failed to capture the rollback point and its
        matching conversation/memory state.
        """
        if not self.config.recovery_manifest_path:
            return
        if not self.config.checkpoint_dir or not self.config.conversation_state_path:
            raise SessionError("recovery checkpoint paths are incomplete")
        from lechonk.eval.checkpoint import export_turn_save
        from lechonk.core.recovery import build_checkpoint, write_checkpoint
        ruleset_ref = getattr(self._session, "ruleset_reference", None)
        help_ref = getattr(self._session, "help_reference", None)
        if help_ref is None:
            help_ref = getattr(self._session, "generic_help_reference", None)
        help_status = getattr(help_ref, "status", None)
        help_hash = getattr(self._session, "help_reference_hash", None)
        if help_hash is None:
            help_hash = getattr(help_ref, "content_hash", None)
        help_source_sha = getattr(help_ref, "source_sha256", None)
        if help_status not in (None, "available"):
            help_hash = None
            help_source_sha = None
        classic_ref = getattr(self._session, "classic_reference", None)
        classic_status = getattr(self._session, "classic_reference_status", None)
        if callable(classic_status):
            classic_status = classic_status()
        if classic_status is None:
            classic_status = getattr(classic_ref, "status", None)
        classic_present = (None if classic_ref is None and classic_status is None
                           else classic_status in ("available", "complete"))
        classic_provenance = getattr(
            self._session, "classic_reference_provenance", None)
        if callable(classic_provenance):
            classic_provenance = classic_provenance()
        classic_tree = getattr(classic_ref, "tree_sha256", None)
        if classic_tree is None and isinstance(classic_provenance, Mapping):
            classic_tree = classic_provenance.get("tree_sha256")
        # An unavailable reference object may retain its schema/default empty
        # digest.  Recovery manifests must encode absence as None, never an
        # empty or unverifiable checksum.
        classic_schema = (getattr(classic_ref, "schema_version", None)
                          if classic_present else None)
        if not classic_present:
            classic_tree = None
        manual_ref = getattr(self._session, "manual_reference", None)
        if manual_ref is None:
            manual_ref = getattr(self._session, "freeciv_manual_reference", None)
        manual_status = getattr(self._session, "manual_reference_status", None)
        if callable(manual_status):
            manual_status = manual_status()
        if manual_status is None:
            manual_status = getattr(manual_ref, "status", None)
        manual_present = (None if manual_ref is None and manual_status is None
                          else manual_status in ("available", "complete"))
        manual_schema = (getattr(manual_ref, "schema_version", None)
                         if manual_present else None)
        manual_source_sha = (getattr(manual_ref, "source_sha256", None)
                             if manual_present else None)
        manual_hash = (getattr(manual_ref, "content_hash", None)
                       if manual_present else None)
        exported = export_turn_save(
            username=self.config.username, turn=turn,
            dest_dir=self.config.checkpoint_dir)
        # Bind the checkpoint to an immutable conversation snapshot.  The
        # live state file continues changing as later turns commit and must
        # never mutate an earlier recovery manifest's hash.
        live_state = Path(self.config.conversation_state_path)
        snapshot_state = (Path(self.config.checkpoint_dir) /
                          f"conversation-state-turn-{turn}.json")
        # Checkpoints are standalone snapshots: they must not depend on the
        # live run's WAL, which may continue growing after this checkpoint.
        conversation = getattr(self.client, "_conversation_manager", None)
        if conversation is not None and callable(getattr(conversation, "export_standalone", None)):
            conversation.export_standalone(snapshot_state)
        else:
            snapshot_state.parent.mkdir(parents=True, exist_ok=True)
            tmp_state = snapshot_state.with_suffix(snapshot_state.suffix + ".tmp")
            shutil.copyfile(live_state, tmp_state)
            with open(tmp_state, "rb") as state_fh:
                os.fsync(state_fh.fileno())
            os.replace(tmp_state, snapshot_state)
        # Persist the rename itself before publishing the manifest.  Without
        # the directory fsync, a power loss could leave a manifest pointing
        # at a filename whose durable directory entry was never committed.
        try:
            dir_fd = os.open(snapshot_state.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            raise SessionError(
                f"cannot durably publish conversation checkpoint: {exc}") from exc
        checkpoint_kwargs = dict(
            checkpoint_path=exported.path,
            artifact_root=str(Path(self.config.checkpoint_dir).resolve().parent),
            conversation_state_path=snapshot_state,
            turn=turn, username=self.config.username, run_id=self.config.run_id,
            parent_run_id=self.config.parent_run_id,
            player_id=getattr(self._session, "player_id", None),
            config_digest=(self.config.recovery_config_digest or
                           f"context={self.config.context_policy};seed={self.config.seed}"),
            strategy_memory=self.memory.dump_state(),
            player_name=self.config.player_name,
            ruleset_reference_schema_version=getattr(
                ruleset_ref, "schema_version", None),
            ruleset_reference_hash=getattr(ruleset_ref, "content_hash", None),
            help_reference_schema_version=getattr(
                help_ref, "schema_version", None),
            help_reference_hash=help_hash,
            help_reference_source_sha256=help_source_sha,
            classic_reference_schema_version=classic_schema,
            classic_ruleset_tree_sha256=classic_tree,
            classic_reference_present=classic_present,
            checkpoint_sha256=exported.sha256)
        # Keep compatibility with recovery manifests created before the
        # optional generated-manual fields were introduced.
        import inspect
        build_parameters = inspect.signature(build_checkpoint).parameters
        for key, value in (
                ("manual_reference_schema_version", manual_schema),
                ("manual_reference_hash", manual_hash),
                ("manual_reference_source_sha256", manual_source_sha),
                ("manual_reference_present", manual_present)):
            if key in build_parameters:
                checkpoint_kwargs[key] = value
        for key, value in (("config_digest_schema", self.config.recovery_config_digest_schema),
                           ("prompt_bundle_id", self.config.prompt_bundle_id)):
            if key in build_parameters and value is not None:
                checkpoint_kwargs[key] = value
        checkpoint = build_checkpoint(**checkpoint_kwargs)
        write_checkpoint(self.config.recovery_manifest_path, checkpoint)
        self._progress("checkpoint_written", turn=turn,
                       path=self.config.recovery_manifest_path)

    def _watchdog_phase_state(self) -> tuple[int, Optional[float]]:
        """The stall watchdog's phase view: (state generation, phase
        deadline). Plain attributes read by the watchdog thread (the
        ledger.last_event precedent)."""
        return self._state_generation, self._phase_deadline_at

    # ------------------------------------------------------------------
    # supervision
    # ------------------------------------------------------------------

    def _terminated(self) -> bool:
        return self._game_over is not None

    def _check_acting_deadline(self, acting_entered: float) -> None:
        """The phase-level ACTING check (its finish-boundary call site
        guards leaving ACTING before the finish primitive). The single
        >= convention: the deadline is reached when
        clock() >= acting_entered + acting_deadline_s — the same instant
        _transition(ACTING) published as _acting_deadline_at. DEF-1: an
        ALREADY-ESTABLISHED game-over wins over a newly-noticed deadline
        expiry — when _game_over is set the check does not raise; the
        game-over path handles the outcome (mirrors
        _dispatch_block_reason's ordering)."""
        if self._game_over is not None:
            return
        if self._clock() - acting_entered >= self.config.acting_deadline_s:
            raise StateDeadlineExceeded(
                RunState.ACTING, self.config.acting_deadline_s)

    def _acting_deadline_reached(self) -> bool:
        """True when the ACTING deadline is reached — the single >=
        convention (clock() >= _acting_deadline_at, W3-3's single
        source) — and no game-over is established. DEF-1: an
        ALREADY-ESTABLISHED game-over wins over a newly-noticed
        deadline expiry (mirrors _dispatch_block_reason's ordering)."""
        if self._game_over is not None:
            return False
        deadline = self._acting_deadline_at
        return deadline is not None and self._clock() >= deadline

    def _emit_acting_deadline_stop(self, *, route: str, turn: int,
                                   remaining: Sequence[_PlannedAction],
                                   completed_result_seqs: Sequence[int],
                                   unstarted_refs: Sequence[str],
                                   provenance: str,
                                   call_seq: Optional[int],
                                   snapshot_seq: int,
                                   candidates_seq: int) -> None:
        """The W3-2 partial-batch accounting (D6, DEF-2): one
        aborted_group action_result per planned-but-undispatched action
        (readback "none", detail "not dispatched: ACTING deadline
        reached" — the DEF-2 compatibility extension of the frozen
        literal, amended openly in schemas.py), then ONE summary metric
        event "acting_deadline_stop" (a partial-batch/v1
        canonical-JSON payload: protocol, turn, reason, route, the
        completed result seqs, the unresolved dispatch-intent seqs, the
        unstarted refs and the remaining standing actors)."""
        for action in remaining:
            self._emit_aborted_group(
                turn=turn, cand=action.candidate, provenance=provenance,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq,
                detail="not dispatched: ACTING deadline reached")
        self._emit(MetricEventRecord(
            name=ACTING_DEADLINE_STOP_EVENT,
            value=canonical_json({
                "protocol": PARTIAL_BATCH_PROTOCOL,
                "turn": turn,
                "reason": "acting_deadline",
                "route": route,
                "completed_result_seqs": list(completed_result_seqs),
                "unresolved_intent_seqs": _unresolved_dispatch_intents(
                    self.ledger.records()),
                "unstarted_refs": list(unstarted_refs),
                "remaining_standing_actors": sorted(
                    actor for actor, order in self._standing.items()
                    if not order.reissued),
            })))

    def _acting_deadline_guard(self, *, route: str, turn: int,
                               remaining: Sequence[_PlannedAction],
                               completed_result_seqs: Sequence[int],
                               unstarted_refs: Sequence[str] = (),
                               provenance: str,
                               call_seq: Optional[int],
                               snapshot_seq: int,
                               candidates_seq: int) -> None:
        """The W3-2 loop-level ACTING-deadline guard (D6/D13). Called at
        the top of each mutation-route loop iteration, BEFORE
        _execute_one's intent write (the ORDER INVARIANT: never dispatch
        a NEW mutation after the deadline; the W3-1 pre-send check
        inside _execute_one remains the final authority). When the
        deadline is reached: emit the partial-batch accounting, then
        raise StateDeadlineExceeded(ACTING) — the existing run-level
        handler truncates with artifacts. D13: an in-flight send keeps
        its FULL mutation deadline (never bounded by the remaining
        ACTING time), so the documented ACTING overshoot of up to one
        mutation deadline is expected. A session timeout marks the
        session failed but does NOT cancel an in-flight worker
        operation — a timeout is not transport cancellation (the
        supervision deadline ends the RUN, never the server-side
        mutation)."""
        if not self._acting_deadline_reached():
            return
        self._emit_acting_deadline_stop(
            route=route, turn=turn, remaining=remaining,
            completed_result_seqs=completed_result_seqs,
            unstarted_refs=unstarted_refs, provenance=provenance,
            call_seq=call_seq, snapshot_seq=snapshot_seq,
            candidates_seq=candidates_seq)
        raise StateDeadlineExceeded(
            RunState.ACTING, self.config.acting_deadline_s)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        """Run the pipeline to an outcome (COMPLETED/DEFEATED/VICTORY/
        TRUNCATED) and return the result with the metric report."""
        outcome = "TRUNCATED"
        reason = "run did not complete"
        turns_completed = 0
        self._emit(RunManifestRecord(manifest=self.manifest), durable=True)
        self._watchdog = StallWatchdog(
            stall_s=self.config.stall_watchdog_s,
            interval_s=self.config.watchdog_interval_s,
            clock=self._clock, ledger=self.ledger,
            on_stall=self._request_abort,
            phase_state=self._watchdog_phase_state)
        self._watchdog.start()
        close_status = "not_attempted"
        try:
            adapter = self._load_adapter()
            self._transition(RunState.STARTUP)
            self._session = self._session_factory()
            self._session.start()  # §3 startup deadline (session-enforced)
            # The interactive gameplay client may expose a local,
            # read-only encyclopedia tool.  Configure it only after the
            # session has collected its immutable ruleset reference.
            self._configure_ruleset_lookup()
            if self.config.resume_turn is not None:
                actual_turn = int(getattr(self._session, "turn", -1))
                if actual_turn != int(self.config.resume_turn):
                    raise SessionError(
                        f"loaded checkpoint turn mismatch: expected "
                        f"{self.config.resume_turn}, got {actual_turn}")
                if (self.config.resume_player_id is not None and
                        int(getattr(self._session, "player_id", -1)) !=
                        int(self.config.resume_player_id)):
                    raise SessionError(
                        "loaded checkpoint player mismatch: expected "
                        f"{self.config.resume_player_id}, got "
                        f"{getattr(self._session, 'player_id', None)}")
            session_port = getattr(self._session, "port", None)
            if callable(session_port):
                session_port = session_port()
            # AdapterSession verifies the server's pingtimeout after reset.
            # Publish that read-back when available, while remaining
            # compatible with legacy/fake sessions that predate these
            # observability attributes.
            session_ready = {
                "port": session_port,
                "aifill": self.config.aifill,
                "configured_ai_opponents": self.config.aifill - 1,
                "configured_player_name": self.config.player_name,
            }
            for attr in ("pingtimeout_status", "pingtimeout_actual",
                         "scenario_readback_status", "aifill_actual",
                         "maxplayers_actual", "actual_total_players",
                         "actual_ai_players", "actual_ai_opponents",
                         "actual_barbarian_ai_opponents",
                         "actual_unknown_ai_opponents", "actual_player_roster",
                         "player_name_verified",
                         "ruleset_reference_status", "ruleset_reference_hash",
                         "help_reference_status", "help_reference_hash",
                         "classic_reference_status", "classic_reference_hash",
                         "manual_reference_status", "manual_reference_hash",
                         "manual_reference_source_sha256"):
                try:
                    value = getattr(self._session, attr)
                    if callable(value):
                        value = value()
                except Exception:
                    continue
                if value is not None:
                    session_ready[attr] = value
            # Persistable provenance is emitted before the first model call;
            # snapshots contain static ruleset/help data only, never live
            # observation or player state.
            for attr, key in (("ruleset_reference_json", "ruleset_reference_json"),
                              ("help_reference_json", "help_reference_json"),
                              ("classic_reference_json", "classic_reference_json"),
                              ("manual_reference_json", "manual_reference_json")):
                try:
                    value = getattr(self._session, attr)
                    if callable(value):
                        value = value()
                except Exception:
                    continue
                if value is not None:
                    session_ready[key] = value
            self._progress("session_ready", **session_ready)
            outcome, reason, turns_completed = self._run_turns(adapter)
        except _AbortRun as abort:
            self._transition(RunState.FAILED)
            outcome, reason = "TRUNCATED", abort.reason
            turns_completed = self._turns_completed
        except _QuarantineFailStop as exc:
            # §3 line 35: an uncertain mutation that reconciliation could
            # not prove non-executed is quarantined — bounded fail-stop.
            self._transition(RunState.FAILED)
            outcome = "TRUNCATED"
            reason = f"{exc.reason} (artifacts intact)"
            turns_completed = self._turns_completed
        except StateDeadlineExceeded as exc:
            self._transition(RunState.FAILED)
            outcome = "TRUNCATED"
            reason = (f"infrastructure deadline exceeded: {exc} "
                      f"(artifacts intact)")
            turns_completed = self._turns_completed
        except Exception as exc:  # session/infra failure: bounded fail-stop
            self._transition(RunState.FAILED)
            outcome = "TRUNCATED"
            if self._abort.is_set():
                # The stall watchdog requested the ordered shutdown while
                # the pipeline was blocked: its reason wins.
                reason = self._abort_reason or "abort requested"
            else:
                reason = f"infrastructure failure: {exc!r} (artifacts intact)"
            turns_completed = self._turns_completed
        finally:
            # A transport failure, phase timeout, or partial mutation must not
            # leave a staged assistant message in the next conversation.
            try:
                self._abort_conversation_turn()
            except Exception:
                pass
            metrics = compute_metrics(self.ledger.records())
            for name, value in metrics.metric_events():
                self._emit(MetricEventRecord(name=name, value=value))
            self.ledger.fsync()
            self._emit(OutcomeRecord(outcome=outcome, reason=reason),
                       durable=True)
            if self._session is not None:
                try:
                    close_status = self._session.close()
                except Exception as exc:
                    close_status = f"error: {exc!r}"
            if self._context_journal is not None:
                try:
                    self._context_journal.close()
                except Exception as exc:
                    # The journal append path is fail-closed; close errors
                    # are reported in the run status without masking the
                    # already-recorded outcome.
                    self._progress("context_journal_close_error", error=str(exc))
            if self._watchdog is not None:
                self._watchdog.stop()
        return RunResult(
            outcome=outcome, reason=reason, turns_completed=turns_completed,
            ledger_path=self.ledger.path, metrics=metrics,
            close_status=close_status)

    # ------------------------------------------------------------------
    # the turn loop
    # ------------------------------------------------------------------

    def _run_turns(self, adapter: _AdapterModules) -> tuple[str, str, int]:
        """The per-turn pipeline. Returns (outcome, reason, turns)."""
        prev_snapshot: Optional[Snapshot] = None
        turns = 0
        while True:
            self._check_abort()
            turn = int(self._session.turn)
            if (self.config.max_turns is not None
                    and turn > self.config.max_turns):
                # The regular turn checkpoint is written before model work,
                # so at a bounded horizon the latest manifest still points
                # at the turn that just completed.  Publish the next-turn
                # boundary now, after all effects and conversation commits
                # for the completed turn are durable.  This makes a resume
                # continue at ``turn`` without replaying the horizon turn.
                self._write_start_checkpoint(turn)
                # No terminal state_transition is emitted here by design:
                # COMPLETED is an outcome, terminal state NEXT_OWN_PHASE
                # (see the module docstring).
                return "COMPLETED", (
                    f"horizon reached after {turns} own phases "
                    f"(max_turns={self.config.max_turns})"), turns
            turns += 1
            outcome, reason, prev_snapshot = self._run_turn(
                adapter, turn, prev_snapshot)
            if outcome is not None:
                return outcome, reason, turns
            # The turn reached the NEXT_OWN_PHASE boundary: it completed.
            self._turns_completed = turns

    def _run_turn(self, adapter: _AdapterModules, turn: int,
                  prev_snapshot: Optional[Snapshot]
                  ) -> tuple[Optional[str], str, Optional[Snapshot]]:
        """One turn: ACTING → FINISH_SENT → WAITING → NEXT_OWN_PHASE |
        GAME_OVER. Returns (outcome-or-None, reason, prev_snapshot)."""
        self._transition(RunState.ACTING)
        acting_entered = self._state_entered
        self._history_results = []
        self._history_rejections = []
        self._history_collecting = False
        self._emit(TurnStartRecord(turn=turn))
        self._progress("turn_started", turn=turn)
        self._write_start_checkpoint(turn)

        # The client declares its provenance class (WP3 paired runs run a
        # scripted comparator arm through this same pipeline; its decisions
        # are scripted_default, never "model"). The default preserves the
        # model-arm labeling exactly.
        provenance = getattr(self.client, "provenance_class", "model")

        # --- snapshot + candidate table (the policy-facing surface) -----
        # The env's game-over flags ride the session cache: an empty
        # game-over observation is classified by them (DEFEATED /
        # VICTORY / TRUNCATED), never by the dict alone. D11: the defeat
        # verdict is RECORD-AUTHORITATIVE — the player record in the
        # observation decides; the env's defeat flag is only the
        # positive-only fallback inside _my_player_defeated, and an
        # unknown status is passed as False so extraction classifies
        # terminated_game_over and _game_over_outcome re-derives
        # (TRUNCATED — never a false VICTORY).
        terminated, truncated, _env_defeated = self._session_game_over_flags()
        defeated = self._my_player_defeated(self._session.obs)
        try:
            extraction = adapter.obs_extract.extract_snapshot(
                self._session.obs, self._session.info,
                terminated=terminated, truncated=truncated,
                my_player_is_defeated=bool(defeated),
                my_player_id=self._session.player_id,
                ruleset=self._session.ruleset,
                action_probs=self._session.action_probs,
            )
        except Exception:
            if not (terminated or truncated):
                # A non-terminal parse failure is a genuine
                # infrastructure failure — the generic handler.
                raise
            # D11 hardening (terminal-before-snapshot): a malformed
            # terminal observation must not raise into the generic
            # infra handler before the record-authoritative fallback
            # applies — classify the terminal flags/records directly.
            outcomes = adapter.obs_extract.ExtractOutcome
            if truncated and not terminated:
                outcome = outcomes.TRUNCATED_INFRA
            elif defeated:
                outcome = outcomes.TERMINATED_DEFEATED
            else:
                outcome = outcomes.TERMINATED_GAME_OVER
            return self._game_over_outcome(
                outcome, "malformed terminal observation: classified from "
                         "the env's game-over flags and the player record")
        if extraction.outcome != adapter.obs_extract.ExtractOutcome.OK:
            return self._game_over_outcome(extraction.outcome, extraction.reason)
        snapshot = extraction.snapshot
        assert snapshot is not None
        snapshot_seq = self._emit(SnapshotRefRecord(
            turn=turn,
            snapshot_hash=hashlib.sha256(
                snapshot.render().encode("utf-8")).hexdigest(),
            token_count=snapshot.estimated_tokens()))
        table = self._session.build_candidate_table()
        mark_armed_in_contact(table, snapshot)
        # W2-3: a cap overflow (armed-in-contact lines plus the mandatory
        # overhead exceeding the budget — D4/W2-2) is a dedicated bounded
        # failure like the prompt budget (FW-13): the metric event names
        # the condition, never a generic infrastructure truncation.
        try:
            offered, admission = self._admit_candidates(table)
            # Render exactly once, from the admitted surface.  In v2 the
            # admission result owns the expanded (8,000-token) rendering;
            # rendering its ``offered`` CandidateTable would silently route
            # back through the legacy 1,200-token cap.  Keep this canonical
            # text for both the ledger hash and the model prompt.
            candidates_text = (admission.render() if admission is not None
                               else table.render())
        except CandidateTableCapExceeded as exc:
            self._emit(MetricEventRecord(
                name="candidate_table_cap_exceeded",
                value=f"turn {turn}: {exc}"))
            self._transition(RunState.FAILED)
            return ("TRUNCATED", f"candidate table cap exceeded: {exc}",
                    prev_snapshot)
        by_domain: dict[str, int] = {}
        for cand in offered.candidates:
            by_domain[cand.domain] = by_domain.get(cand.domain, 0) + 1
        candidates_seq = self._emit(CandidatesRefRecord(
            turn=turn,
            table_hash=hashlib.sha256(
                candidates_text.encode("utf-8")).hexdigest(),
            candidate_count=len(offered.candidates),
            by_domain=by_domain,
            opportunity_domains=opportunity_domains(offered.candidates),
            candidate_surface_policy=(admission.policy if admission else "legacy_v1"),
            candidate_cap_tokens=(admission.cap_tokens if admission else None),
            total_candidate_count=(admission.total_candidate_count if admission else None),
            offered_candidate_count=(admission.offered_candidate_count if admission else None),
            omitted_candidate_count=(admission.omitted_candidate_count if admission else None),
            total_by_domain=(admission.total_by_domain if admission else {}),
            offered_by_domain=(admission.offered_by_domain if admission else {}),
            omitted_by_domain=(admission.omitted_by_domain if admission else {}),
            total_tokens=(admission.total_tokens if admission else None),
            offered_tokens=(admission.offered_tokens if admission else None),
            omitted_tokens=(admission.omitted_tokens if admission else None)))

        # Deadline release gate: gameplay_v6/candidate_surface_v3 must retain
        # the classic controls that a mature empire needs.  Keep this out of
        # the prompt and conversation so it cannot change cache prefixes.
        if self.config.candidate_surface_policy == "candidate_surface_v3":
            control_access = assess_control_access(
                table, offered, snapshot, self._session.obs,
                my_player_id=self._session.player_id)
            self._emit(MetricEventRecord(
                name="control_access",
                value=canonical_json(control_access)))
            self._progress(
                "control_access",
                turn=turn,
                passed=bool(control_access["passed"]),
                missing_ref_count=len(control_access["missing_ref_ids"]),
                missing_unit_state_count=len(control_access["missing_unit_state_ids"]),
                missing_city_state_count=len(control_access["missing_city_state_ids"]),
            )
            if not control_access["passed"]:
                self._transition(RunState.FAILED)
                missing = ",".join(control_access["missing_ref_ids"][:4]) or "state"
                return (
                    "TRUNCATED",
                    f"control access gate failed: {missing}",
                    prev_snapshot,
                )

        # W3-9 (D9/D12, Option A): the turn-start observation event — the
        # first of the four events feeding the persistent last-seen
        # baseline. Engine-automatic resolutions are counted, never
        # hidden (§3 line 38); the first event seeds the baseline
        # silently (turn 1's empty baseline: no records).
        self._observe_engine_automatic(
            turn=turn, snapshot_seq=snapshot_seq, candidates_seq=candidates_seq)

        delta = enforce_delta_budget(compute_delta(prev_snapshot, snapshot))
        delta_text = delta.render()
        memory_profile = ("gameplay"
                          if self.config.prompt_profile in {"gameplay_v1", "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
                          else "legacy")
        self.memory.begin_turn(turn, profile=memory_profile)
        # The intent-before-write durability boundary (§3 line 34): every
        # intent record of the turn is on disk before the first mutation.
        self.ledger.fsync()
        self._check_acting_deadline(acting_entered)

        # --- the model phase (≤2 API attempts incl. at most one repair) -
        # FW-13 failure policy: a prompt-budget failure is a dedicated,
        # bounded fail-stop — the metric event names the budget condition
        # and the run never reports it as a generic infrastructure
        # truncation.
        try:
            result, repair_used, why = self._model_phase(
                adapter, turn=turn, table=table, snapshot=snapshot,
                delta_text=delta_text, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, candidates_text=candidates_text,
                acting_entered=acting_entered, provenance=provenance)
        except PromptBudgetExceeded as exc:
            self._emit(MetricEventRecord(
                name="prompt_budget_exceeded",
                value=f"turn {turn}: {exc}"))
            self._transition(RunState.FAILED)
            return ("TRUNCATED", f"prompt budget exceeded: {exc}",
                    prev_snapshot)
        batch = result.batch
        self._progress(
            "turn_decision",
            turn=turn,
            actions=[item.ref for item in result.planned],
            action_count=len(result.planned),
            repair_used=repair_used,
            degraded=result.degraded,
            why=why,
            notes=(batch.notes if batch is not None else ""),
        )
        self._check_acting_deadline(acting_entered)
        self._check_abort()
        self._history_collecting = True

        # --- memory update (rides the ActionBatch, never a separate call);
        # FW-R2: the ledger carries the union of every attempt's
        # memory_update — a repair replaces the executed batch, never a
        # commitment the model already made this turn
        updates = result.attempt_memory_updates
        # FW-16: the batch-level ActionBatch.commitments are ingested too —
        # one CommitmentRecord per batch carrying the UNION of both sources
        # (and of every attempt), identical commitments deduplicated. No
        # source tagging: CommitmentRecord forbids extra fields.
        batch_commitments = result.attempt_batch_commitments
        if updates:
            self.memory.apply(updates[-1], turn=turn)
        if updates or batch_commitments:
            commitments: list = []
            threats: list = []
            for update in updates:
                for c in update.commitments:
                    if c not in commitments:
                        commitments.append(c)
                for t in update.threats:
                    if t not in threats:
                        threats.append(t)
            for c in batch_commitments:
                if c not in commitments:
                    commitments.append(c)
            self._emit(CommitmentRecord(
                turn=turn,
                commitments=commitments,
                threats=self._threat_mentions(threats, turn, snapshot),
                call_seq=result.call_seq))

        # --- execute ------------------------------------------------------
        executed_actors: set[str] = set()
        if result.planned:
            executed_actors = self._execute_planned(
                adapter, turn=turn, planned=result.planned, table=table,
                provenance=provenance, call_seq=result.call_seq,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                reason=result.why)
        elif result.degraded:
            executed_actors = self._execute_degraded(
                adapter, turn=turn, table=table, snapshot=snapshot,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                why=why)
        # else: the model answered with a valid empty batch — responsive,
        # nothing to execute, no degraded default.

        self._check_abort()
        # W2-4: the game-over check guards the continuation entry — a
        # mid-batch termination is checked BEFORE standing orders are
        # re-issued; no dispatch follows a dead game.
        if self._game_over is None:
            self._continue_standing_orders(
                adapter, turn=turn, table=table, snapshot=snapshot,
                acted_actors=executed_actors,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq)
        self._check_acting_deadline(acting_entered)
        self._check_abort()
        if self._game_over is not None:
            # A mutation ended the game mid-batch: no finish primitive.
            terminated, truncated = self._game_over
            self._game_over = None
            self._transition(RunState.GAME_OVER)
            if truncated and not terminated:
                return ("TRUNCATED",
                        "infrastructure truncation mid-batch — artifacts "
                        "intact", None)
            defeated = self._my_player_defeated(self._session.obs)
            if defeated is None:
                return ("TRUNCATED",
                        "game over mid-batch with unknown defeat status "
                        "(no well-formed player record, no positive env "
                        "defeat signal) — never a false VICTORY — "
                        "artifacts intact", None)
            return ("DEFEATED" if defeated else "VICTORY",
                    f"game over mid-batch (defeated={defeated})", None)

        # --- FINISH_SENT → WAITING (the pipeline owns the finish) --------
        self._transition(RunState.FINISH_SENT)
        self._transition(RunState.WAITING)
        finish = adapter.execute.finish_phase(
            self._session, deadline=self.config.waiting_deadline_s)
        # The single >= convention (the phase checks share it with the
        # W3-2 guards and _check_acting_deadline).
        if self._clock() - self._state_entered >= self.config.waiting_deadline_s:
            raise StateDeadlineExceeded(
                RunState.WAITING, self.config.waiting_deadline_s)
        self.memory.advance(turn)
        if finish.terminated or finish.truncated:
            self._transition(RunState.GAME_OVER)
            defeated = self._my_player_defeated(finish.observation)
            if finish.truncated and not finish.terminated:
                # The phase was interrupted by infrastructure; do not publish
                # a partial legacy history episode or context-journal record.
                self._history_collecting = False
                return ("TRUNCATED",
                        "infrastructure truncation at the phase boundary "
                        "(server timeout) — artifacts intact", None)
            # A truncated/infra phase is not a completed conversation turn;
            # only a terminal game-over boundary may be committed.  This
            # keeps a partial turn out of the next request's history.
            self._commit_history_episode(
                turn=turn, result=result, repair_used=repair_used)
            self._history_collecting = False
            self._commit_conversation_turn(
                turn=turn, result=result, repair_used=repair_used)
            if defeated is None:
                return ("TRUNCATED",
                        "game over at the phase boundary with unknown "
                        "defeat status (no well-formed player record, no "
                        "positive env defeat signal) — never a false "
                        "VICTORY — artifacts intact", None)
            return ("DEFEATED" if defeated else "VICTORY",
                    "game over at the phase boundary "
                    f"(defeated={defeated})", None)
        # A normal phase boundary is a completed conversation turn too.  The
        # next model request must see the exact assistant response plus this
        # turn's execution report; commit only after finish_phase and memory
        # advancement have both succeeded.
        self._commit_history_episode(
            turn=turn, result=result, repair_used=repair_used)
        self._history_collecting = False
        self._commit_conversation_turn(
            turn=turn, result=result, repair_used=repair_used)
        # W3-9: the finish-boundary observation event (the fourth) — the
        # post-finish observation is the first sight of the next phase's
        # state; final-boundary transitions before the horizon exit are
        # captured, attributed to THIS (completing) turn — never beyond
        # the counted horizon.
        self._observe_engine_automatic(
            turn=turn, snapshot_seq=snapshot_seq, candidates_seq=candidates_seq)
        self._transition(RunState.NEXT_OWN_PHASE)
        return None, "", snapshot

    # ------------------------------------------------------------------
    # game-over classification
    # ------------------------------------------------------------------

    def _session_game_over_flags(self) -> tuple[bool, bool, bool]:
        """The session-cached game-over flags: (terminated, truncated,
        my_player_is_defeated). The real AdapterSession caches them from
        the latest step (the defeat signal is read from the controller at
        the game-over boundary — it is not in the step tuple); a
        duck-typed session that does not expose them defaults to no
        game-over."""
        terminated = bool(getattr(self._session, "terminated", False))
        truncated = bool(getattr(self._session, "truncated", False))
        defeated = getattr(self._session, "my_player_is_defeated", None)
        return (terminated, truncated,
                defeated if isinstance(defeated, bool) else False)

    def _game_over_outcome(self, outcome: Any, reason: str
                           ) -> tuple[Optional[str], str, Optional[Snapshot]]:
        """Map the extraction outcome (classified by the env's game-over
        flags) to the run outcome. D11: the defeat verdict is
        RECORD-AUTHORITATIVE at every boundary — both terminated
        branches re-check the player record, so a well-formed alive
        record OVERRIDES the env's defeat flag (never a false DEFEATED)
        and the flag alone proves defeat only when the record is missing
        or malformed (the positive-only env fallback). The VICTORY branch
        stays evidence-gated: ``terminated_game_over`` means the env's
        defeat flag said not-defeated, but a False flag is not positive
        proof of survival (the vendor method returns False both when the
        player is provably alive AND when the player record is missing —
        civ_controller.py:303-330), so the defeat status must be provable
        or the run reports TRUNCATED — never a false VICTORY."""
        value = getattr(outcome, "value", outcome)
        self._transition(RunState.GAME_OVER)
        if value in ("terminated_defeated", "terminated_game_over"):
            # terminated=True: the player record decides (D11) — the
            # flag-sourced extraction verdict can never defeat a
            # well-formed alive record.
            defeated = self._my_player_defeated(self._session.obs)
            if defeated is None:
                return ("TRUNCATED",
                        "game over with unknown defeat status (no "
                        "well-formed player record, no positive env "
                        "defeat signal) — never a false VICTORY", None)
            if defeated:
                return ("DEFEATED", "my player is defeated", None)
            return ("VICTORY", f"game over (defeated={defeated})", None)
        if value == "truncated_infra":
            return ("TRUNCATED", reason or "infrastructure truncation", None)
        return ("TRUNCATED", reason or "unknown extraction outcome", None)

    def _my_player_defeated(self, obs: Any) -> Optional[bool]:
        """The defeat status at a game-over boundary: True/False when
        provable, None when unknown (never a false VICTORY).

        A well-formed player record (is_alive a bool) is authoritative
        (D11) — it OVERRIDES the env's defeat flag. When the record is
        missing or malformed, the env's defeat flag decides — but only
        positively: the vendor method returns False both when the
        player is provably alive and when the record is missing
        (civ_controller.py:303-330), so anything but a True flag leaves
        the status unknown.

        Hardened (D11): never raises on a malformed observation or
        player mapping — a malformed record counts as a missing record
        (the positive-only env fallback) — and the my-player entry is
        found whether the observation keys the player domain by integer
        or by serialized string."""
        players = obs.get("player") if isinstance(obs, Mapping) else None
        if not isinstance(players, dict):
            players = {}
        pid = self._session.player_id
        me = players.get(pid)
        if me is None and pid is not None:
            me = players.get(str(pid))
        if me is None and pid is not None:
            # Serialized-string player IDs (a JSON round-trip keys the
            # player domain by string): match by normalized integer.
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = None
            if pid_int is not None:
                for key, entry in players.items():
                    try:
                        if int(key) == pid_int:
                            me = entry
                            break
                    except (TypeError, ValueError):
                        continue
        if isinstance(me, dict) and isinstance(me.get("is_alive"), bool):
            return me["is_alive"] is False
        flag = getattr(self._session, "my_player_is_defeated", None)
        if flag is True:
            return True
        return None

    # ------------------------------------------------------------------
    # the model phase
    # ------------------------------------------------------------------

    def _emit_client_call_records(self, *, turn: int, prompt: TurnPrompt,
                                   response: Any, logical_attempt: int,
                                   before_records: int,
                                   next_attempt: int) -> tuple[int, int]:
        """Ledger physical API calls while preserving legacy fake clients.

        A lookup-enabled turn has one logical ``complete_turn`` call that
        may contain an initial tool-request API call and a tool-final API
        call.  The runner must expose both physical calls, then allow one
        repair (three physical calls total).  Legacy clients do not expose
        ``call_records`` and retain the historical one-record-per-protocol
        call behavior.
        """
        records = list(getattr(self.client, "call_records", ()))[before_records:]
        records = [record for record in records
                   if record.get("kind") != "conversation_compaction"]
        if not records:
            next_attempt += 1
            seq = self._emit(ModelCallRecord(
                call_id=prompt.call_id, attempt=next_attempt,
                prompt_hash=response.prompt_hash, latency_ms=response.latency_ms,
                tokens_in=response.tokens_in, tokens_out=response.tokens_out,
                cost_usd=response.cost_usd, fingerprint=response.fingerprint,
                error=response.error,
                usage_status=(response.usage_status or (
                    "partial" if response.tokens_in is not None or
                    response.tokens_out is not None else "unavailable")),
                cost_status=(response.cost_status or "unknown")))
            self._progress(
                "model_call_finished", turn=turn, attempt=next_attempt,
                logical_attempt=logical_attempt, physical=True,
                latency_ms=response.latency_ms, tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                outcome=("error" if response.error else "ok"),
            )
            return seq, next_attempt

        last_seq = None
        for record in records:
            next_attempt += 1
            seq = self._emit(ModelCallRecord(
                call_id=str(record.get("call_id") or prompt.call_id),
                attempt=next_attempt,
                prompt_hash=str(record.get("prompt_hash") or response.prompt_hash),
                latency_ms=int(record.get("latency_ms") or 0),
                tokens_in=(int(record["tokens_in"]) if record.get("tokens_in") is not None else None),
                tokens_out=(int(record["tokens_out"]) if record.get("tokens_out") is not None else None),
                cost_usd=(float(record["cost_usd"]) if record.get("cost_usd") is not None else None),
                fingerprint=str(record.get("fingerprint") or ""),
                error=record.get("error"),
                usage_status=record.get("usage_status"),
                cost_status=record.get("cost_status")))
            last_seq = seq
            self._progress(
                "model_call_finished", turn=turn, attempt=next_attempt,
                logical_attempt=logical_attempt,
                physical_kind=record.get("kind", "initial"),
                latency_ms=record.get("latency_ms", 0),
                tokens_in=record.get("tokens_in", 0),
                tokens_out=record.get("tokens_out", 0),
                outcome=("error" if record.get("error") else "ok"),
            )
        return last_seq, next_attempt

    def _model_phase(self, adapter: _AdapterModules, *, turn: int,
                     table: CandidateTable, snapshot: Snapshot,
                     delta_text: str, snapshot_seq: int,
                     candidates_seq: int, candidates_text: str,
                     acting_entered: float, provenance: str = "model"
                     ) -> tuple[_IngestResult, bool, str]:
        """Attempt 1 + at most one repair (≤2 total API attempts, decision
        4). Returns (result, repair_used, why)."""
        run_id = self.config.run_id
        # FW-R2: every attempt's memory_update, in attempt order — the
        # turn's CommitmentRecord unions them, so a repair never drops
        # what an earlier attempt already stated.
        memory_updates: list = []
        # FW-16: every attempt's batch-level ActionBatch.commitments, in
        # attempt order — the turn's CommitmentRecord unions both sources.
        batch_commitments: list = []
        physical_call_index = 0
        prompt = TurnPrompt(
            call_id=f"{run_id}/t{turn}/call1", turn_id=f"{run_id}/t{turn}",
            memory=self.memory.render_for_prompt(
                profile=("gameplay" if self.config.prompt_profile in
                         {"gameplay_v1", "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} else "legacy")),
            snapshot=snapshot.render(), delta=delta_text,
            candidates=candidates_text,
            history=(self.history.render()
                     if self.config.context_policy == "hybrid_v1" else ""),
            budget_s=max(0.0, self.config.acting_deadline_s
                         - (self._clock() - acting_entered)))
        self._progress("model_call_started", turn=turn, attempt=1)
        before_records = len(getattr(self.client, "call_records", ()))
        resp = self.client.complete_turn(prompt)
        call_seq, physical_call_index = self._emit_client_call_records(
            turn=turn, prompt=prompt, response=resp, logical_attempt=1,
            before_records=before_records, next_attempt=physical_call_index)
        result = self._ingest(
            resp, turn=turn, table=table, call_seq=call_seq,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            provenance=provenance)
        if result.batch is not None and result.batch.memory_update is not None:
            memory_updates.append(result.batch.memory_update)
        if result.batch is not None:
            batch_commitments.extend(result.batch.commitments)
        result.attempt_memory_updates = memory_updates
        result.attempt_batch_commitments = batch_commitments
        result.call_seq = call_seq
        result.why = self._why_unusable(resp, result)

        need_repair = (result.failed
                       or (result.batch is not None
                           and result.batch.actions and not result.planned))
        if not need_repair:
            return result, False, result.why

        # Budget-aware repair gating (WP4): the repair attempt must fit in
        # the remaining ACTING budget. Below the floor — under any observed
        # successful repair latency — the call can only time out, so it is
        # skipped entirely: the degraded default applies, the turn completes
        # within the deadline by construction.
        remaining_s = max(0.0, self.config.acting_deadline_s
                          - (self._clock() - acting_entered))
        if remaining_s < REPAIR_BUDGET_FLOOR_S:
            self._emit(MetricEventRecord(
                name="repair_skipped_insufficient_budget",
                value=(f"remaining ACTING budget {remaining_s:.1f}s below "
                       f"the {REPAIR_BUDGET_FLOOR_S:.0f}s repair floor; "
                       f"degraded default applies")))
            result2 = _IngestResult(batch=None, failed=True)
            result2.attempt_memory_updates = memory_updates
            result2.attempt_batch_commitments = batch_commitments
            result2.degraded = True
            why = (f"repair skipped: insufficient ACTING budget "
                   f"({remaining_s:.1f}s remaining < "
                   f"{REPAIR_BUDGET_FLOOR_S:.0f}s floor); degraded default "
                   f"applies")
            return result2, False, why

        # The one repair attempt, per error class (§7.5). budget_s carries
        # the remaining ACTING budget: the client derives the per-call hard
        # request timeout from it (remaining minus REPAIR_TIMEOUT_MARGIN_S),
        # so a timed-out repair surfaces as a transport failure on THIS
        # attempt — the repair-cap-exhausted path below applies, and the
        # turn never exceeds the deadline by construction.
        repair_prompt = RepairPrompt(
            call_id=f"{run_id}/t{turn}/call2", turn_id=f"{run_id}/t{turn}",
            memory=self.memory.render_for_prompt(
                profile=("gameplay" if self.config.prompt_profile in
                         {"gameplay_v1", "gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} else "legacy")),
            snapshot=snapshot.render(), delta=delta_text,
            candidates=candidates_text,
            budget_s=max(0.0, self.config.acting_deadline_s
                         - (self._clock() - acting_entered)),
            error_class=self._repair_class(resp, result),
            failed_refs=[ref for ref, _ in result.invalid],
            refreshed_candidates=candidates_text)
        self._progress("model_call_started", turn=turn, attempt=2)
        before_records = len(getattr(self.client, "call_records", ()))
        resp2 = self.client.complete_repair(repair_prompt)
        call_seq2, physical_call_index = self._emit_client_call_records(
            turn=turn, prompt=repair_prompt, response=resp2, logical_attempt=2,
            before_records=before_records, next_attempt=physical_call_index)
        result2 = self._ingest(
            resp2, turn=turn, table=table, call_seq=call_seq2,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            provenance=provenance)
        if result2.batch is not None and result2.batch.memory_update is not None:
            memory_updates.append(result2.batch.memory_update)
        if result2.batch is not None:
            batch_commitments.extend(result2.batch.commitments)
        result2.attempt_memory_updates = memory_updates
        result2.attempt_batch_commitments = batch_commitments
        result2.call_seq = call_seq2
        result2.why = self._why_unusable(resp2, result2)
        # The degraded default applies when the model is unresponsive after
        # the repair cap: no valid batch at all, or a final batch whose
        # every action was invalid. A valid EMPTY batch is responsive.
        result2.degraded = (
            not result2.planned
            and not (result2.batch is not None and not result2.batch.actions))
        physical_cap = (3 if self.config.context_policy == "conversation_v1"
                        and self.config.prompt_profile in {"gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
                        and hasattr(self.client, "call_records") else 2)
        if result2.degraded:
            why = (f"model unresponsive after the repair cap (≤{physical_cap} "
                   f"physical API attempts per turn): {result2.why}")
        else:
            # A successful repair is a usable model decision.  Preserve its
            # own diagnostic (normally empty) instead of labeling it as a
            # repair-cap failure in live progress and turn telemetry.
            why = result2.why
        return result2, True, why

    def _why_unusable(self, resp: ModelResponse, result: _IngestResult) -> str:
        """Why the response was unusable (the repair/degraded reason): ""
        when the response was usable (planned actions exist), "no actions"
        only for a genuinely empty batch."""
        if resp.error is not None:
            return f"API call failed: {resp.error}"
        if resp.malformed is not None or resp.batch is None:
            return f"malformed output: {resp.malformed or 'no batch'}"
        if result.batch is not None and result.batch.actions and not result.planned:
            classes = sorted({cls for _, cls in result.invalid})
            return (f"all {len(result.batch.actions)} emitted actions "
                    f"invalid (classes: {', '.join(classes)})")
        if result.planned:
            return ""
        return "no actions"

    def _repair_class(self, resp: ModelResponse, result: _IngestResult) -> str:
        if resp.error is not None:
            return REPAIR_CLASS_UNRESPONSIVE
        if resp.malformed is not None or resp.batch is None:
            return REPAIR_CLASS_SCHEMA
        if result.invalid:
            return result.invalid[0][1]
        return REPAIR_CLASS_UNRESPONSIVE

    # ------------------------------------------------------------------
    # ingestion: validate a model response against the offered table
    # ------------------------------------------------------------------

    def _ingest(self, resp: ModelResponse, *, turn: int,
                table: CandidateTable, call_seq: int, snapshot_seq: int,
                candidates_seq: int, provenance: str = "model") -> _IngestResult:
        """Validate one model response; emit action_result records for
        every model-emitted attempt (schema-error rejections for malformed
        outputs; table rejections for refs outside the OFFERED candidate
        set — cap-elided or never legal). Returns the planned (executable)
        actions after group-abort filtering."""
        if resp.error is not None:
            # Transport-level failure: no parseable attempts (the model_call
            # record carries the error).
            return _IngestResult(batch=None, failed=True)
        if resp.batch is None or resp.malformed is not None:
            # Schema violation: one rejected attempt per emitted ref (or a
            # single batch-level attempt when nothing was extractable).
            refs = list(resp.raw_attempts) or ["<malformed-batch>"]
            for raw in refs:
                detail = ("malformed output: "
                          + (resp.malformed or "no batch")
                          + "; ref rejected as schema_error")
                self._emit_rejection(
                    turn=turn, raw_ref=raw, cls="schema_error",
                    provenance=provenance,
                    call_seq=call_seq, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq,
                    detail=detail)
            return _IngestResult(batch=None, failed=True,
                                 invalid=[(r, "schema_error") for r in refs])
        batch = resp.batch
        # W2-1: validate against the OFFERED (deterministically capped) set —
        # the model could only ever see the capped rendering, so a well-formed
        # ref to a cap-elided candidate is state it could not see
        # (state_fog_error), never an executable action. Mid-batch legality is
        # a separate check (_execute_planned, full refreshed table).
        by_id = self._admit_candidates(table)[0].by_id()
        result = _IngestResult(batch=batch)
        aborted_actors: set[str] = set()
        for item in batch.actions:
            cand = by_id.get(item.ref)
            if cand is None:
                cls = ("schema_error" if not is_candidate_id(item.ref)
                       else "state_fog_error")
                self._emit_rejection(
                    turn=turn, raw_ref=item.ref, cls=cls,
                    provenance=provenance,
                    call_seq=call_seq, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq,
                    detail=(f"ref not in the turn's offered candidate table "
                            f"({'cap-elided or never legal this turn — state_fog_error' if cls == 'state_fog_error' else 'not a candidate ID — schema_error'})"))
                result.invalid.append((item.ref, cls))
                actor = _actor_of_ref(item.ref)
                if actor is not None:
                    aborted_actors.add(actor)
            elif cand.actor_id in aborted_actors:
                # §3 line 33: the invalid action aborts only this entity's
                # remaining group.
                self._emit_aborted_group(
                    turn=turn, cand=cand, provenance=provenance,
                    call_seq=call_seq, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq)
            else:
                result.planned.append(_PlannedAction(ref=item.ref,
                                                     candidate=cand))
        return result

    # ------------------------------------------------------------------
    # execution: batch semantics (§3 line 33)
    # ------------------------------------------------------------------

    def _execute_planned(self, adapter: _AdapterModules, *, turn: int,
                         planned: Sequence[_PlannedAction],
                         table: CandidateTable, provenance: str,
                         call_seq: Optional[int], snapshot_seq: int,
                         candidates_seq: int, reason: str = "",
                         route: str = "planned",
                         unstarted_refs: Sequence[str] = (),
                         completed_result_seqs: Optional[list[int]] = None
                         ) -> set[str]:
        """Execute planned actions sequentially with per-action read-back
        verification. After each executed mutation the remaining actions
        are revalidated against refreshed state; a stale or illegal action
        aborts only that entity's remaining group (logged rejected /
        aborted_group); other entities proceed; the turn is never aborted.
        Returns the set of actor IDs acted for.

        W3-2 (D6/D13): the loop-top ACTING-deadline guard runs at the top
        of every iteration, BEFORE _execute_one's intent write (never
        dispatch a NEW mutation after the deadline). The degraded route
        re-enters this loop per group (route="degraded", the later
        groups' refs carried as unstarted_refs, and the batch-level
        completed_result_seqs accumulator shared across the groups); the
        guard's accounting covers the batch's planned-but-undispatched
        actions."""
        remaining = list(planned)
        aborted_actors: set[str] = set()
        acted_actors: set[str] = set()
        executed_any = False
        # The batch's action_result seqs so far (shared across the
        # degraded route's groups when the caller passes its own list).
        completed_seqs = (completed_result_seqs
                           if completed_result_seqs is not None else [])
        while remaining:
            self._check_abort()
            # W3-2 (D6/D13): the loop-top guard — before _execute_one's
            # intent write (the ORDER INVARIANT); the W3-1 pre-send check
            # inside _execute_one remains the final authority.
            self._acting_deadline_guard(
                route=route, turn=turn, remaining=remaining,
                completed_result_seqs=completed_seqs,
                unstarted_refs=unstarted_refs, provenance=provenance,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq)
            planned_action = remaining[0]
            ref, cand = planned_action.ref, planned_action.candidate
            actor = cand.actor_id
            if actor in aborted_actors:
                self._emit_aborted_group(
                    turn=turn, cand=cand, provenance=provenance,
                    call_seq=call_seq, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq)
                remaining.pop(0)
                continue
            if executed_any:
                # Revalidate against refreshed state (a city founded
                # mid-batch was not in the original candidate set).
                fresh = self._session.build_candidate_table()
                fresh_by_id = fresh.by_id()
                if ref not in fresh_by_id:
                    self._emit_rejection(
                        turn=turn, raw_ref=ref, cls="illegal_in_rule",
                        cand=cand, provenance=provenance, call_seq=call_seq,
                        snapshot_seq=snapshot_seq,
                        candidates_seq=candidates_seq,
                        detail=("stale after a mid-batch mutation: no longer "
                                "in the refreshed candidate table; the "
                                "entity's remaining group is aborted (§3 "
                                "line 33)"))
                    aborted_actors.add(actor)
                    remaining.pop(0)
                    continue
                cand = fresh_by_id[ref]
            result, seq = self._execute_one(
                adapter, cand, turn=turn, provenance=provenance,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, reason=reason)
            completed_seqs.append(seq)
            readback = result.readback.status
            executed_any = True
            remaining.pop(0)
            acted_actors.add(actor)
            self._track_executed(cand, result, turn=turn)
            if provenance == "model":
                self._update_standing(actor, cand, seq, readback)
            if readback == adapter.execute.CONTRADICTED:
                # An illegal action aborts the entity's remaining group.
                aborted_actors.add(actor)
            if result.status == "executed":
                # W3-9: the post-step observation event — after the
                # mutation's own action_result (the emission-order
                # contract: a straddling completed mutation emits its
                # result first, then the observation diff). The verified
                # resend completes inside _execute_one, so this event
                # covers it too.
                self._observe_engine_automatic(
                    turn=turn, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq)
            if result.terminated or result.truncated:
                self._game_over = (bool(result.terminated),
                                   bool(result.truncated))
                # remaining already had the executed action popped off: it
                # IS the list of planned-but-unexecuted actions now.
                for rest in remaining:
                    self._emit_aborted_group(
                        turn=turn, cand=rest.candidate, provenance=provenance,
                        call_seq=call_seq, snapshot_seq=snapshot_seq,
                        candidates_seq=candidates_seq,
                        detail="game ended mid-batch")
                remaining.clear()
                break
        return acted_actors

    def _execute_degraded(self, adapter: _AdapterModules, *, turn: int,
                          table: CandidateTable, snapshot: Snapshot,
                          snapshot_seq: int, candidates_seq: int,
                          why: str) -> set[str]:
        """The degraded default (§3 line 44): the model is unresponsive
        after the repair cap → comparator/defaults.decide, logged
        scripted_default with reason. Selects from the OFFERED (capped)
        set so both arms act with identical capability (W2-1).

        W3-2 (D6/D13): the degraded route needs no guard of its own —
        each group re-enters _execute_planned, whose loop-top guard
        stops the next group before any dispatch (verified at HEAD: the
        group loop only calls _execute_planned). The later groups' refs
        ride into the stop summary as unstarted_refs."""
        offered, _ = self._admit_candidates(table)
        log = defaults_mod.decide(snapshot, offered)
        planned: list[_PlannedAction] = []
        for decision in log.decisions:
            cand = offered.by_id().get(decision.candidate_id)
            if cand is None:
                continue  # stale default resolution: logged, never executed
            planned.append(_PlannedAction(ref=decision.candidate_id,
                                          candidate=cand))
        reasons = {d.candidate_id: (f"{why}; {d.trigger}: {d.detail}")
                   for d in log.decisions}
        # Execute decision by decision (each defaulted choice carries its
        # own reason); the group semantics still apply per entity.
        acted: set[str] = set()
        remaining = list(planned)
        # W3-2: the degraded batch's action_result seqs so far — shared
        # across the groups so a stop summary reports the WHOLE batch.
        completed_seqs: list[int] = []
        while remaining:
            if self._terminated():
                # W2-4: the game ended mid-group — no further group is
                # dispatched (no dispatch follows a dead game).
                break
            group = [remaining[0]]
            actor = remaining[0].candidate.actor_id
            while len(remaining) > len(group) \
                    and remaining[len(group)].candidate.actor_id == actor:
                group.append(remaining[len(group)])
            executed = self._execute_planned(
                adapter, turn=turn, planned=group, table=table,
                provenance="scripted_default", call_seq=None,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                reason=reasons.get(group[0].ref, why),
                route="degraded",
                unstarted_refs=[a.ref for a in remaining[len(group):]],
                completed_result_seqs=completed_seqs)
            acted |= executed
            del remaining[:len(group)]
        return acted

    # ------------------------------------------------------------------
    # uncertain mutations: mid-mutation disconnect reconciliation
    # (§3 line 35)
    # ------------------------------------------------------------------

    def _execute_one(self, adapter: _AdapterModules, cand: Candidate, *,
                     turn: int, provenance: str,
                     call_seq: Optional[int], snapshot_seq: int,
                     candidates_seq: int, reason: str = "",
                     origin_seq: Optional[int] = None
                     ) -> tuple[Any, int]:
        """Execute ONE candidate and emit its action_result (§3 line 35).

        A mutation whose send fails after dispatch (disconnect or
        deadline — the read-back never arrives) has an UNKNOWN execution
        state: it is settled only by full-state reconciliation on
        reconnect, never by blind replay. Returns (ExecutionResult,
        action_result seq); raises _QuarantineFailStop when the mutation
        is executed-or-unknown (never resent).

        W3-1 (D5): the pinned send-site order — pre-capture pre-state ->
        deadline/game-over check -> durable intent -> FINAL deadline
        recheck -> send or cancel -> the correlated action_result."""
        target, affected, actor_tile = self._describe(cand)
        # Snapshot the caller-supplied pre-dispatch baseline BEFORE any
        # send (§3 line 35): reconciliation compares against the true
        # pre-mutation state, never a post-mutation re-read.
        pre_obs = self._session.obs
        pre_info = self._session.info
        # W2-3 pre-dispatch classification: map the candidate BEFORE any
        # send. candidate_to_action is pure, so a mapping failure is fully
        # classified here — outcome "rejected" (nothing was sent), no
        # reconciliation and no resend — and can never fall into the
        # disconnect path below. A failure raised after the send stays
        # with the generic handler: something WAS sent.
        try:
            adapter.execute.candidate_to_action(cand)
        except ValueError as exc:
            detail = (f"pre-dispatch mapping failed ({exc!r}): the "
                      f"candidate cannot be mapped to an action tuple — "
                      f"never sent, never reconciled, never resent")
            seq = self._emit_action_result(
                turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
                target=target, affected=affected, actor_tile=actor_tile,
                outcome="rejected", rejection_class="schema_error",
                provenance_class=provenance, origin_seq=origin_seq,
                reason=reason, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, call_seq=call_seq,
                readback="none", detail=detail)
            turn_before = int((pre_info or {}).get("turn", 0) or 0)
            return adapter.execute.ExecutionResult(
                candidate_id=cand.id, action=("", "", ""),
                status="rejected",
                readback=adapter.execute.Readback(status="none",
                                                  detail=detail),
                terminated=False, truncated=False,
                turn_before=turn_before, turn_after=turn_before,
                elapsed_s=0.0, detail=detail), seq
        # W3-1 (D5): the dispatch-intent boundary — the pinned send-site
        # order. The deadline/game-over check runs BEFORE the intent is
        # written (a blocked dispatch leaves no intent behind).
        block = self._dispatch_block_reason()
        if block is not None:
            return self._blocked_dispatch(
                adapter, cand, block=block, pre_info=pre_info,
                intent_seq=None)
        intent_seq = self._emit_dispatch_intent(
            turn=turn, ref=cand.id, call_seq=call_seq,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            provenance=provenance)
        # The FINAL pre-send recheck: the intent's fsync took time — a
        # deadline that passed during it cancels the dispatch durably
        # instead of sending past it.
        block = self._dispatch_block_reason()
        if block is not None:
            self._emit_dispatch_cancelled(
                intent_seq=intent_seq, reason=block, turn=turn,
                ref=cand.id, call_seq=call_seq,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                provenance=provenance)
            return self._blocked_dispatch(
                adapter, cand, block=block, pre_info=pre_info,
                intent_seq=intent_seq)
        try:
            result = adapter.execute.execute_candidate(
                self._session, cand,
                deadline=self.config.mutation_deadline_s)
        except Exception as exc:
            # The send's fate is unknown: reconcile, never blind replay.
            return self._reconcile_disconnect(
                adapter, cand, exc, turn=turn, provenance=provenance,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, reason=reason,
                origin_seq=origin_seq, target=target, affected=affected,
                actor_tile=actor_tile,
                pre_obs=pre_obs, pre_info=pre_info, intent_seq=intent_seq)
        seq = self._emit_action_result(
            turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
            target=target, affected=affected, actor_tile=actor_tile,
            outcome="executed",
            rejection_class=("illegal_in_rule"
                             if result.readback.status
                             == adapter.execute.CONTRADICTED else None),
            provenance_class=provenance, origin_seq=origin_seq,
            reason=reason, snapshot_seq=snapshot_seq,
            candidates_seq=candidates_seq, call_seq=call_seq,
            readback=result.readback.status,
            detail=_dispatch_result_detail(intent_seq, result.detail),
            durable=True)  # the action-boundary fsync (§3 line 34)
        return result, seq

    def _reconcile_disconnect(self, adapter: _AdapterModules,
                              cand: Candidate, exc: Exception, *,
                              turn: int, provenance: str,
                              call_seq: Optional[int], snapshot_seq: int,
                              candidates_seq: int, reason: str,
                              origin_seq: Optional[int],
                              target: Optional[str],
                              affected: Optional[str],
                              actor_tile: Optional[tuple[int, int]],
                              pre_obs: Mapping[str, Any],
                              pre_info: Mapping[str, Any],
                              intent_seq: int
                              ) -> tuple[Any, int]:
        """§3 line 35: settle a mutation whose read-back is inconclusive
        after a disconnect, ONLY by full-state reconciliation — the
        target postconditions are compared on reconnect (the pre-mutation
        state against a fresh read through the session). Verified
        non-execution permits ONE resubmission (continued_model_order,
        origin-linked); executed-or-unknown is quarantined → the bounded
        fail-stop. Never blind replay; no duplicate mutation.

        W3-1 (D5): the uncertain action_result below IS the correlated
        result of this dispatch's intent (the send's fate is durably
        recorded — the intent is resolved, not left forensic)."""
        # The uncertain attempt is ledgered first (the mutation WAS sent;
        # its verification is inconclusive).
        uncertain_seq = self._emit_action_result(
            turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
            target=target, affected=affected, actor_tile=actor_tile,
            outcome="executed", provenance_class=provenance,
            origin_seq=origin_seq, reason=reason,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            call_seq=call_seq, readback="inconclusive",
            detail=_dispatch_result_detail(
                intent_seq,
                f"mutation send failed after dispatch ({exc!r}): "
                f"the execution state is unknown — settling by "
                f"full-state reconciliation on reconnect (never "
                f"blind replay, §3 line 35)"),
            durable=True)
        # Full-state reconciliation: re-read (the reconnect) and compare
        # the target postconditions against the caller-captured
        # pre-mutation baseline (captured BEFORE the send — the session's
        # cached obs may already be post-mutation when the send failed,
        # e.g. a step that completed and an execute that raised after it).
        failure = getattr(self._session, "last_dispatch_failure", None)
        if isinstance(failure, Mapping) and failure.get("transport_uncertain"):
            # A websocket callback/write failure poisons the owning session;
            # refresh() would only re-enter the same connection and can also
            # clear standing orders.  Fail closed and quarantine without a
            # same-session reconnect or resend.
            diagnostic = failure.get("transport_diagnostic")
            if not isinstance(diagnostic, Mapping):
                diagnostic = {}
            decision = "unknown"
            evidence = (
                "transport callback failed; same-session refresh refused "
                f"(stage={failure.get('stage')!r}, "
                f"exception={failure.get('exception')!r}, "
                f"loop={failure.get('owned_loop')!r}, "
                f"vendor={diagnostic.get('vendor_module_sha256')!r}, "
                f"traceback={str(failure.get('traceback', ''))[-2048:]})")
            self._progress(
                "transport_failure_quarantined",
                stage=str(failure.get("stage", "unknown")),
                exception=str(failure.get("exception", "unknown")),
                loop=str(failure.get("owned_loop")),
                vendor_module=str(diagnostic.get("vendor_module", "")),
                vendor_module_sha256=str(diagnostic.get(
                    "vendor_module_sha256", "")),
                traceback=str(failure.get("traceback", ""))[-2048:],
            )
        else:
            try:
                fresh = self._session.refresh()
                readback = adapter.execute.verify_readback(
                    cand, pre_obs, fresh.get("observation") or {},
                    my_player_id=self._session.player_id,
                    pre_info=pre_info, post_info=fresh.get("info") or {},
                    production_names=self._session.production_name_map())
            except Exception as exc2:
                decision = "unknown"
                evidence = f"the reconnect read failed: {exc2!r}"
            else:
                if readback.status == adapter.execute.CONTRADICTED:
                    decision = "non_execution_verified"
                elif readback.status == adapter.execute.VERIFIED:
                    decision = "executed"
                else:
                    decision = "unknown"
                evidence = (f"reconciliation read-back {readback.status}: "
                            f"{readback.detail}")
        self._emit(MetricEventRecord(
            name="mutation_reconciliation",
            value=f"{cand.id}: {decision} ({evidence})"))
        if decision == "non_execution_verified":
            # Only verified non-execution permits resubmission — once.
            return self._resend_verified_non_execution(
                adapter, cand, turn=turn, provenance=provenance,
                snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq,
                origin_seq=(origin_seq if origin_seq is not None
                            else uncertain_seq),
                target=target, affected=affected, actor_tile=actor_tile)
        # Executed or unknown: NEVER resent — quarantine → bounded
        # fail-stop with artifacts.
        self._emit(MetricEventRecord(
            name="mutation_quarantine",
            value=(f"{cand.id}: {decision} on reconnect — quarantined, "
                   f"never resent (never blind replay, §3 line 35)")))
        raise _QuarantineFailStop(
            f"uncertain mutation quarantined after disconnect: {cand.id} "
            f"is {decision} on reconnect ({evidence}) — never resent "
            f"(never blind replay, §3 line 35)")

    def _resend_verified_non_execution(self, adapter: _AdapterModules,
                                       cand: Candidate, *, turn: int,
                                       provenance: str,
                                       snapshot_seq: int,
                                       candidates_seq: int,
                                       origin_seq: int,
                                       target: Optional[str],
                                       affected: Optional[str],
                                       actor_tile: Optional[tuple[int, int]]
                                       ) -> tuple[Any, int]:
        """The ONE permitted resubmission (§3 line 35): the mutation was
        verified NOT to have executed (reconciled on reconnect), so it is
        resent exactly once, linked to its origin. The provenance mirrors
        the origin attempt's class (``continued_model_order`` only for a
        model arm — a scripted arm's resubmission is never mislabeled as
        model-originated). A second disconnect, a contradicted resend, or
        an inconclusive resend read-back quarantines; the resubmission is
        never repeated.

        W3-1 (D5): the SAME pinned send-site order as _execute_one —
        pre-capture pre-state -> deadline/game-over check -> durable
        intent -> FINAL deadline recheck -> send or cancel -> the
        correlated action_result (each of the resubmission's possible
        action_results correlates this dispatch's intent)."""
        # W2-3: never resend an unmappable candidate. The pre-dispatch
        # classification in _execute_one keeps an unmappable candidate out
        # of the reconciliation path entirely; this guard is the backstop
        # at the resend site itself — the bounded fail-stop, never a
        # resubmission.
        try:
            adapter.execute.candidate_to_action(cand)
        except ValueError as exc:
            raise _QuarantineFailStop(
                f"unmappable candidate {cand.id} reached the resend path "
                f"({exc!r}) — never resent (pre-dispatch classification, "
                f"W2-3)") from None
        # Provenance honesty (§3 line 45): continued_model_order is a
        # MODEL-arm label — a scripted arm's resubmission mirrors its
        # origin attempt's provenance class, with the same origin linkage.
        resend_provenance = ("continued_model_order"
                             if provenance == "model" else provenance)
        # W3-1 (D5): the dispatch-intent boundary — the pinned send-site
        # order, identical to _execute_one's. Pre-capture the pre-state
        # (the pre-dispatch info baseline), check, then write the durable
        # intent, then the FINAL recheck.
        pre_info = self._session.info
        block = self._dispatch_block_reason()
        if block is not None:
            return self._blocked_dispatch(
                adapter, cand, block=block, pre_info=pre_info,
                intent_seq=None)
        intent_seq = self._emit_dispatch_intent(
            turn=turn, ref=cand.id, call_seq=None,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            provenance=resend_provenance)
        # The FINAL pre-send recheck (the intent's fsync took time).
        block = self._dispatch_block_reason()
        if block is not None:
            self._emit_dispatch_cancelled(
                intent_seq=intent_seq, reason=block, turn=turn,
                ref=cand.id, call_seq=None, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, provenance=resend_provenance)
            return self._blocked_dispatch(
                adapter, cand, block=block, pre_info=pre_info,
                intent_seq=intent_seq)
        try:
            result = adapter.execute.execute_candidate(
                self._session, cand,
                deadline=self.config.mutation_deadline_s)
        except Exception as exc:
            self._emit_action_result(
                turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
                target=target, affected=affected, actor_tile=actor_tile,
                outcome="executed",
                provenance_class=resend_provenance, origin_seq=origin_seq,
                reason="pipeline resubmission after verified non-execution",
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                call_seq=None, readback="inconclusive",
                detail=_dispatch_result_detail(
                    intent_seq,
                    f"the resubmission send also failed after dispatch "
                    f"({exc!r}) — quarantined, never resent again "
                    f"(never blind replay, §3 line 35)"),
                durable=True)
            self._emit(MetricEventRecord(
                name="mutation_quarantine",
                value=(f"{cand.id}: the verified-non-execution "
                       f"resubmission also failed ({exc!r}) — quarantined, "
                       f"never resent")))
            raise _QuarantineFailStop(
                f"uncertain mutation quarantined after disconnect: "
                f"{cand.id}'s resubmission failed ({exc!r}) — never "
                f"resent (never blind replay, §3 line 35)") from None
        readback = result.readback.status
        if readback == adapter.execute.CONTRADICTED:
            # The resend provably did not execute either; the one
            # resubmission is spent — never a second one.
            self._emit_action_result(
                turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
                target=target, affected=affected, actor_tile=actor_tile,
                outcome="executed", rejection_class="illegal_in_rule",
                provenance_class=resend_provenance, origin_seq=origin_seq,
                reason="pipeline resubmission after verified non-execution",
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                call_seq=None, readback=readback,
                detail=_dispatch_result_detail(intent_seq, result.detail),
                durable=True)
            self._emit(MetricEventRecord(
                name="mutation_quarantine",
                value=(f"{cand.id}: the resubmission read back contradicted "
                       f"execution — the one resubmission is spent, never "
                       f"resent again (§3 line 35)")))
            raise _QuarantineFailStop(
                f"uncertain mutation quarantined after disconnect: "
                f"{cand.id}'s resubmission read back contradicted "
                f"({result.detail}) — the one resubmission is spent, "
                f"never resent (never blind replay, §3 line 35)")
        if readback != adapter.execute.VERIFIED:
            # INCONCLUSIVE: the resend's execution state is unknown —
            # the same quarantine path as an exception or a contradicted
            # resend (§3 line 35 / the module docstring: an unknown
            # mutation is NEVER resent again; the run takes the bounded
            # fail-stop, never continues on an inconclusive settlement).
            self._emit_action_result(
                turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
                target=target, affected=affected, actor_tile=actor_tile,
                outcome="executed",
                provenance_class=resend_provenance, origin_seq=origin_seq,
                reason="pipeline resubmission after verified non-execution",
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                call_seq=None, readback=readback,
                detail=_dispatch_result_detail(intent_seq, result.detail),
                durable=True)
            self._emit(MetricEventRecord(
                name="mutation_quarantine",
                value=(f"{cand.id}: the resubmission read back {readback} "
                       f"— quarantined, never resent again (never blind "
                       f"replay, §3 line 35)")))
            raise _QuarantineFailStop(
                f"uncertain mutation quarantined after disconnect: "
                f"{cand.id}'s resubmission read back {readback} "
                f"({result.detail}) — never resent again (never blind "
                f"replay, §3 line 35)")
        seq = self._emit_action_result(
            turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
            target=target, affected=affected, actor_tile=actor_tile,
            outcome="executed",
            provenance_class=resend_provenance, origin_seq=origin_seq,
            reason=("pipeline resubmission after verified non-execution "
                    "(reconciled on reconnect, §3 line 35)"),
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            call_seq=None, readback=readback,
            detail=_dispatch_result_detail(intent_seq, result.detail),
            durable=True)
        if result.terminated or result.truncated:
            # W2-4: a resubmission that ends the game marks termination at
            # the resend site (mirrors the _execute_planned writer) — no
            # dispatch path continues past a dead game.
            self._game_over = (bool(result.terminated),
                               bool(result.truncated))
        return result, seq

    # ------------------------------------------------------------------
    # provenance emission helpers
    # ------------------------------------------------------------------

    def _emit_rejection(self, *, turn: int, raw_ref: str, cls: str,
                        provenance: str = "model",
                        cand: Optional[Candidate] = None,
                        call_seq: Optional[int], snapshot_seq: int,
                        candidates_seq: int, detail: str) -> int:
        domain, verb, target = "", "", None
        if is_candidate_id(raw_ref):
            parts = raw_ref.split(":")
            domain, verb = parts[0], parts[2]
        elif cand is not None:
            domain, verb = cand.domain, cand.verb
        if cand is not None:
            payload = cand.payload or {}
            target = payload.get("production") or payload.get("tech_name")
        return self._emit_action_result(
            turn=turn, ref=raw_ref, domain=domain or "unknown", verb=verb,
            target=target,
            outcome="rejected", rejection_class=cls,
            provenance_class=provenance, snapshot_seq=snapshot_seq,
            candidates_seq=candidates_seq, call_seq=call_seq,
            readback="none", detail=detail)

    def _emit_aborted_group(self, *, turn: int, cand: Candidate,
                            provenance: str, call_seq: Optional[int],
                            snapshot_seq: int, candidates_seq: int,
                            detail: str = ("the entity's group was aborted by "
                                           "an earlier stale/illegal action "
                                           "(§3 line 33)")) -> int:
        return self._emit_action_result(
            turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
            outcome="aborted_group", provenance_class=provenance,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            call_seq=call_seq, readback="none", detail=detail)

    def _describe(self, cand: Candidate) -> tuple[Optional[str], Optional[str], Optional[tuple[int, int]]]:
        """The structured (target, affected, actor_tile) descriptors for
        the ledger record — computed from the CURRENT session state (fresh
        at every execution point)."""
        payload = cand.payload or {}
        domain, verb = cand.domain, cand.verb
        actor_tile = self._actor_tile(cand)
        if domain == "tech":
            name = payload.get("tech_name")
            return name, name, actor_tile
        if domain == "dipl":
            descriptor = f"player:{payload.get('counterpart')}"
            return descriptor, descriptor, actor_tile
        if domain == "city":
            if verb.startswith("produce_"):
                name = payload.get("production")
                return name, name, actor_tile
            if verb.startswith("city_work_") or verb.startswith("city_unwork_"):
                if actor_tile is not None:
                    descriptor = (f"tile:{actor_tile[0] + int(payload.get('dx', 0))},"
                                  f"{actor_tile[1] + int(payload.get('dy', 0))}")
                    return descriptor, descriptor, actor_tile
            descriptor = f"city:{cand.actor_id}"
            return descriptor, descriptor, actor_tile
        # unit domain
        if (verb.startswith("goto_") or verb.startswith("attack_")) \
                and actor_tile is not None:
            adapter = self._load_adapter()
            dir8 = payload.get("dir8")
            dx, dy = adapter.obs_extract.DIR8_DELTAS[int(dir8)]
            dest = (actor_tile[0] + dx, actor_tile[1] + dy)
            descriptor = f"tile:{dest[0]},{dest[1]}"
            if verb.startswith("attack_"):
                enemy = self._foreign_unit_at(dest)
                if enemy is not None:
                    descriptor = f"unit:{enemy}"
            return descriptor, descriptor, actor_tile
        if verb == "build_city" and actor_tile is not None:
            descriptor = f"tile:{actor_tile[0]},{actor_tile[1]}"
            return descriptor, descriptor, actor_tile
        # Persistent orders and the remaining unit verbs: the order is on
        # the unit; the affected tile is the unit's tile.
        target = f"unit:{cand.actor_id}"
        affected = (f"tile:{actor_tile[0]},{actor_tile[1]}"
                    if actor_tile is not None else target)
        return target, affected, actor_tile

    def _actor_tile(self, cand: Candidate) -> Optional[tuple[int, int]]:
        obs = self._session.obs or {}
        try:
            actor = int(cand.actor_id)
        except (TypeError, ValueError):
            return None
        if cand.domain == "unit":
            unit = (obs.get("unit") or {}).get(actor)
            if isinstance(unit, dict) and isinstance(unit.get("x"), int) \
                    and isinstance(unit.get("y"), int):
                return (unit["x"], unit["y"])
        elif cand.domain == "city":
            city = (obs.get("city") or {}).get(actor)
            if isinstance(city, dict) and isinstance(city.get("x"), int) \
                    and isinstance(city.get("y"), int):
                return (city["x"], city["y"])
        return None

    def _foreign_unit_at(self, tile: tuple[int, int]) -> Optional[int]:
        obs = self._session.obs or {}
        my_id = self._session.player_id
        for key, unit in (obs.get("unit") or {}).items():
            if not isinstance(unit, dict):
                continue
            if unit.get("owner") == my_id:
                continue
            if (unit.get("x"), unit.get("y")) == tile:
                try:
                    return int(key)
                except (TypeError, ValueError):
                    return None
        return None

    # ------------------------------------------------------------------
    # standing orders (continued_model_order)
    # ------------------------------------------------------------------

    def _track_executed(self, cand: Candidate, result: Any, *,
                        turn: int) -> None:
        """Retain the executed produce/research actions (and the
        corroborated foundings) since the last observation event — the
        per-interval attribution set the next observation event diffs
        against (W3-9). A target change caused by our own action is not
        engine-automatic. A CONTRADICTED action is never attributed (the
        readback credit rules are unchanged: observation history never
        upgrades a contradicted action)."""
        readback = result.readback.status
        if readback not in ("verified", "inconclusive"):
            return
        payload = cand.payload or {}
        if cand.domain == "tech" and cand.verb.startswith("research_tech_"):
            tech_id = payload.get("tech_id")
            if isinstance(tech_id, int) and not isinstance(tech_id, bool):
                self._retain_interval_action(("research", tech_id),
                                             turn=turn, ref=cand.id)
        elif cand.domain == "city" and cand.verb.startswith("produce_"):
            name = payload.get("production")
            try:
                cid = int(cand.actor_id)
            except (TypeError, ValueError):
                cid = None
            if name is not None and cid is not None:
                self._retain_interval_action(("produce", cid, name),
                                             turn=turn, ref=cand.id)
        elif cand.domain == "unit" and cand.verb == "build_city":
            # Corroborated founding evidence: the founding verifier's
            # new_cities (execute.py) — only a VERIFIED founding counts
            # (an inconclusive founding is an uncertain mutation, not a
            # corroborated one).
            if readback == "verified":
                evidence = result.readback.evidence or {}
                for cid in (evidence.get("new_cities") or []):
                    try:
                        self._engine_interval_foundings.add(int(cid))
                    except (TypeError, ValueError):
                        continue

    def _retain_interval_action(self, entry: tuple, *, turn: int,
                                ref: str) -> None:
        """Retain one executed produce/research action in the bounded
        per-interval attribution set. DEF-3: the set is bounded so it
        cannot grow unbounded; an overflow drops the entry and emits the
        observation-gap diagnostic (fail-open with mandatory disclosure
        — the run remains valid, the affected turn's engine-auto
        accounting is potentially incomplete)."""
        if entry in self._engine_interval_actions:
            return
        if len(self._engine_interval_actions) >= ENGINE_OBSERVATION_INTERVAL_MAX:
            self._emit(MetricEventRecord(
                name=ENGINE_OBSERVATION_GAP_EVENT,
                value=(f"turn {turn}: the engine-observation attribution "
                       f"set overflowed (bounded at "
                       f"{ENGINE_OBSERVATION_INTERVAL_MAX} retained "
                       f"actions); {ref} is not retained — the "
                       f"engine-automatic accounting for turn {turn} is "
                       f"potentially incomplete (DEF-3: fail-open, "
                       f"disclosed)")))
            return
        self._engine_interval_actions.add(entry)

    def _update_standing(self, actor: str, cand: Candidate, seq: int,
                         readback: str) -> None:
        """Model acted for ``actor``: the fresh decision supersedes any
        standing order; a persistent verified/inconclusive order becomes
        the new standing order (continuable once, linked to its origin)."""
        self._standing.pop(actor, None)
        if cand.domain != "unit" or cand.verb not in PERSISTENT_VERBS:
            return
        if readback not in ("verified", "inconclusive"):
            return  # contradicted: the order is dead
        self._standing[actor] = _StandingOrder(
            actor_id=actor, verb=cand.verb, ref=cand.id, origin_seq=seq)

    def _continue_standing_orders(self, adapter: _AdapterModules, *,
                                  turn: int, table: CandidateTable,
                                  snapshot: Snapshot, acted_actors: set[str],
                                  snapshot_seq: int,
                                  candidates_seq: int) -> None:
        """continued_model_order (§3 line 38, WP2 operationalization — see
        module docstring): re-issue a model-originated persistent order
        exactly once, only when the model's batch contains no action for
        that actor AND the current state proves the order is no longer in
        effect (mutations are never re-sent without read-back proof of
        non-execution, §3 line 32). The re-issue resolves against the
        OFFERED (capped) candidate set (W2-1): a cap-elided ref is a lookup
        miss — re-resolve within the offered set or drop; never a
        state_fog_error and never executed from the uncapped table.

        W3-2 (D6/D13): the loop-level guard runs before each _execute_one
        — a standing-order re-issue is never dispatched after the ACTING
        deadline (the stop summary reports the outstanding orders as
        remaining_standing_actors, never as per-action records)."""
        offered, _ = self._admit_candidates(table)
        completed_seqs: list[int] = []
        for actor in sorted(self._standing):
            self._check_abort()
            standing = self._standing[actor]
            if actor in acted_actors:
                continue  # the model's fresh decision supersedes
            if self._standing_in_effect(standing, table):
                continue
            if standing.reissued:
                del self._standing[actor]  # one re-issue per order
                continue
            cand = offered.by_id().get(standing.ref)
            if cand is None:
                for candidate in offered.candidates:
                    if (candidate.domain == "unit"
                            and candidate.actor_id == actor
                            and candidate.verb == standing.verb):
                        cand = candidate
                        break
            if cand is None:
                del self._standing[actor]
                continue
            # W3-2 (D6/D13): the guard — before each _execute_one in the
            # continuation loop (a re-issue is never dispatched after the
            # deadline).
            self._acting_deadline_guard(
                route="standing", turn=turn, remaining=(),
                completed_result_seqs=completed_seqs,
                provenance="continued_model_order", call_seq=None,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq)
            result, seq = self._execute_one(
                adapter, cand, turn=turn,
                provenance="continued_model_order", call_seq=None,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                reason="pipeline continuation of a model-originated order "
                       "(linked to its origin)",
                origin_seq=standing.origin_seq)
            completed_seqs.append(seq)
            readback = result.readback.status
            standing.reissued = True
            if result.status == "executed":
                # W3-9: the post-step observation event — the standing
                # re-issue is a mutation like any other.
                self._observe_engine_automatic(
                    turn=turn, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq)
            if readback == adapter.execute.CONTRADICTED:
                del self._standing[actor]
            if result.terminated or result.truncated:
                # W2-4: a continuation that ends the game breaks the loop
                # and marks termination (mirrors the _execute_planned
                # writer) — no further standing order is dispatched.
                self._game_over = (bool(result.terminated),
                                   bool(result.truncated))
                break

    def _standing_in_effect(self, standing: _StandingOrder,
                            table: CandidateTable) -> bool:
        """Read-back proof of the order's presence in the current state."""
        obs = self._session.obs or {}
        try:
            actor = int(standing.actor_id)
        except (TypeError, ValueError):
            return False
        unit = (obs.get("unit") or {}).get(actor)
        if not isinstance(unit, dict):
            return False  # the unit is gone: the order cannot be in effect
        if standing.verb == "keep_activity":
            # The order holds when the unit is no longer actable.
            for candidate in table.candidates:
                if candidate.domain == "unit" \
                        and candidate.actor_id == standing.actor_id:
                    return False
            return True
        expected = _ACTIVITY_FOR_VERB.get(standing.verb)
        if expected is None:
            return False
        name = _activity_name(unit.get("activity"))
        if name is None:
            return False
        if isinstance(expected, tuple):
            return name in expected
        return name == expected

    # ------------------------------------------------------------------
    # engine-automatic observation (§3 line 38; W3-9, D9/D12 Option A)
    # ------------------------------------------------------------------

    def _observe_engine_automatic(self, *, turn: int, snapshot_seq: int,
                                  candidates_seq: int) -> None:
        """One W3-9 observation event: diff the session's cached
        observation against the persistent last-seen baseline and emit
        an engine_automatic record for every change no produce/research
        action of ours (executed since the last observation event)
        accounts for — counted, never hidden (§3 line 38).

        The four events: turn start post-extraction; after every
        executed mutation; after the verified resend (it completes
        inside _execute_one, so the caller's post-step event covers
        it); after successful phase completion (the finish outcome,
        attributed to the COMPLETING turn — never beyond the counted
        horizon). The first event seeds the baseline silently (the
        empty baseline: turn 1's pre-existing choices are not engine
        resolutions).

        Hardenings: entity/index identity in the refs (unique per
        record); corroborated founding evidence (an ownership transfer,
        checkpoint resynchronization or re-sighting after absence is
        baseline-only, never a fresh choice); duplicate/unchanged
        observations produce no record; a capture gap emits the DEF-3
        diagnostic (fail-open, disclosed); the readback credit rules are
        unchanged; buy-completion is captured only on an OBSERVED
        transition (never assumed at the buy's own post-step event)."""
        obs = self._session.obs
        cities, research_id, me, city_names = self._engine_observation_state(
            obs)
        if cities is None:
            # A capture gap (DEF-3): no observation to diff — disclose,
            # fail open, leave the baseline (and the interval) untouched.
            self._emit(MetricEventRecord(
                name=ENGINE_OBSERVATION_GAP_EVENT,
                value=(f"turn {turn}: no observation at this event — the "
                       f"engine-automatic accounting for turn {turn} is "
                       f"potentially incomplete (DEF-3: fail-open, "
                       f"disclosed)")))
            return
        self._engine_observation_index += 1
        index = self._engine_observation_index
        if not self._engine_baseline_seen:
            # The empty baseline seeds silently.
            self._engine_baseline_seen = True
            self._engine_baseline_cities = dict(cities)
            self._engine_baseline_research = research_id
            self._engine_known_cities.update(cities)
            self._engine_interval_actions.clear()
            self._engine_interval_foundings.clear()
            return
        my_id = self._session.player_id
        # --- the research target ---------------------------------------
        prev_research = self._engine_baseline_research
        if (research_id is not None and prev_research != research_id
                and ("research", research_id)
                not in self._engine_interval_actions):
            name = self._engine_research_name(obs, research_id, me)
            prev_name = (self._engine_research_name(obs, prev_research, me)
                         if prev_research is not None else None)
            self._emit_engine_automatic(
                turn=turn,
                ref=f"engine_automatic:research:player:{my_id}:obs{index}",
                domain="tech", verb="research_tech_auto", target=name,
                affected=name, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq,
                detail=(f"research target changed {prev_name} -> {name} "
                        f"between observation events with no executed tech "
                        f"action of ours — engine-automatic resolution "
                        f"(counted, never hidden, §3 line 38)"))
        self._engine_baseline_research = research_id
        # --- own-city production ----------------------------------------
        for cid in sorted(cities):
            production = cities[cid]
            if cid not in self._engine_baseline_cities:
                # A first sighting of an own city.
                self._engine_known_cities.add(cid)
                if cid in self._engine_founded \
                        or cid in self._engine_interval_foundings:
                    # Corroborated founding: founding and choosing the
                    # default production are distinct decisions — the
                    # engine's initial choice is a resolution even when
                    # the founding itself was model-directed.
                    self._engine_founded.add(cid)
                    self._engine_baseline_cities[cid] = production
                    if production is not None:
                        self._emit_engine_automatic(
                            turn=turn,
                            ref=(f"engine_automatic:production:city:{cid}"
                                 f":obs{index}"),
                            domain="city", verb="produce_auto",
                            target=production, affected=production,
                            snapshot_seq=snapshot_seq,
                            candidates_seq=candidates_seq,
                            detail=(f"city {city_names.get(cid, cid)} founded "
                                    f"with the engine's initial production "
                                    f"{production} — counted, never "
                                    f"hidden (§3 line 38)"))
                else:
                    # Ownership transfer / checkpoint resynchronization /
                    # re-sighting after absence: baseline-only, never
                    # mislabeled as a fresh choice.
                    self._engine_baseline_cities[cid] = production
                continue
            prev = self._engine_baseline_cities[cid]
            if (production is not None and production != prev
                    and ("produce", cid, production)
                    not in self._engine_interval_actions):
                self._emit_engine_automatic(
                    turn=turn,
                    ref=(f"engine_automatic:production:city:{cid}"
                         f":obs{index}"),
                    domain="city", verb="produce_auto", target=production,
                    affected=production, snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq,
                    detail=(f"city {city_names.get(cid, cid)} production "
                            f"changed {prev} -> {production} between "
                            f"observation events with no executed produce "
                            f"action of ours — engine-automatic resolution "
                            f"(counted, never hidden, §3 line 38)"))
            self._engine_baseline_cities[cid] = production
        # An own city absent from this observation is dropped (loss or
        # conquest); a re-sighting is baseline-only (the known-city set
        # keeps the id).
        for cid in list(self._engine_baseline_cities):
            if cid not in cities:
                del self._engine_baseline_cities[cid]
        # The interval ends at this observation event.
        self._engine_interval_actions.clear()
        self._engine_interval_foundings.clear()

    def _engine_observation_state(
            self, obs: Any) -> tuple[Optional[dict[int, Optional[str]]],
                                     Optional[int],
                                     Optional[Mapping[str, Any]],
                                     dict[int, str]]:
        """The baseline view of one observation: ({city_id: production
        display name}, the research target id, the own player entry,
        {city_id: name}). The cities are None when the observation
        carries no player domain (a capture gap — DEF-3)."""
        if not isinstance(obs, Mapping):
            return None, None, None, {}
        players = obs.get("player")
        if not isinstance(players, dict) or not players:
            return None, None, None, {}
        my_id = self._session.player_id
        me = players.get(my_id)
        if me is None and my_id is not None:
            me = players.get(str(my_id))
        if me is None and my_id is not None:
            # Serialized-string player IDs (a JSON round-trip keys the
            # player domain by string): match by normalized integer.
            try:
                pid_int = int(my_id)
            except (TypeError, ValueError):
                pid_int = None
            if pid_int is not None:
                for key, entry in players.items():
                    try:
                        if int(key) == pid_int:
                            me = entry
                            break
                    except (TypeError, ValueError):
                        continue
        researching = me.get("researching") if isinstance(me, Mapping) else None
        research_id = None
        if isinstance(researching, int) and not isinstance(researching, bool) \
                and researching != 0:
            research_id = researching
        names = self._engine_production_names()
        cities: dict[int, Optional[str]] = {}
        city_names: dict[int, str] = {}
        city_domain = obs.get("city")
        if isinstance(city_domain, dict):
            for key, entry in city_domain.items():
                if not isinstance(entry, Mapping):
                    continue
                try:
                    cid = int(key)
                except (TypeError, ValueError):
                    continue
                if entry.get("owner") != my_id:
                    continue  # own cities only (as today)
                kind = entry.get("production_kind")
                value = entry.get("production_value")
                production = None
                if kind not in (None, VUT_NONE) and value is not None:
                    try:
                        production = names.get((int(kind), int(value)))
                        if production is None:
                            production = f"kind{int(kind)}#{int(value)}"
                    except (TypeError, ValueError):
                        production = None
                cities[cid] = production
                city_names[cid] = str(entry.get("name") or f"city#{cid}")
        return (cities, research_id,
                me if isinstance(me, Mapping) else None, city_names)

    def _engine_production_names(self) -> Mapping[tuple[int, int], str]:
        """The (production_kind, production_value) -> display name map,
        mirroring the snapshot path's resolution (obs_extract): the
        ruleset's city_production_names, else the inverse of the
        session's production_name_map."""
        ruleset = getattr(self._session, "ruleset", None)
        names = getattr(ruleset, "city_production_names", None)
        if isinstance(names, Mapping) and names:
            return names
        out: dict[tuple[int, int], str] = {}
        for name, kv in (self._session.production_name_map() or {}).items():
            if isinstance(kv, tuple) and len(kv) == 2:
                try:
                    out[(int(kv[0]), int(kv[1]))] = name
                except (TypeError, ValueError):
                    continue
        return out

    def _engine_research_name(self, obs: Mapping[str, Any],
                              tech_id: Optional[int],
                              me: Optional[Mapping[str, Any]]) -> Optional[str]:
        """The research target's display name, mirroring the snapshot
        path's resolution (obs_extract._extract_player): the tech domain
        wins, then the player's research_name, then tech#<id>."""
        if tech_id is None:
            return None
        techs = obs.get("tech")
        entry = techs.get(tech_id) if isinstance(techs, dict) else None
        if isinstance(entry, Mapping):
            name = entry.get("name")
            if name:
                return str(name)
        name = me.get("research_name") if isinstance(me, Mapping) else None
        return str(name) if name else f"tech#{tech_id}"

    def _emit_engine_automatic(self, *, turn: int, ref: str, domain: str,
                               verb: str, target: str, affected: str,
                               snapshot_seq: int, candidates_seq: int,
                               detail: str) -> int:
        """The engine-automatic emission shape (unchanged): an
        ActionResultRecord, provenance_class "engine_automatic",
        readback "none", outcome "executed", turn = the observation
        event's turn. The ref carries the entity identity and the
        observation index — unique per record (name-only refs collide
        across cities and confuse the P11 audit's by-ref grouping)."""
        return self._emit_action_result(
            turn=turn, ref=ref, domain=domain, verb=verb, target=target,
            affected=affected, outcome="executed",
            provenance_class="engine_automatic", readback="none",
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            call_seq=None, detail=detail)

    # ------------------------------------------------------------------
    # memory / threats
    # ------------------------------------------------------------------

    def _threat_mentions(self, threats: Sequence[str], turn: int,
                         snapshot: Snapshot) -> list[ThreatMention]:
        """Structured threat hypotheses (entity IDs only — the metric never
        parses free text, §3 line 41); non-conforming entries are dropped
        at emission. A threat is an opponent/barbarian entity: mentions of
        own entities or of self are dropped, as are unresolvable ids
        (never counted). The tile is resolved from the snapshot when
        visible."""
        mentions: list[ThreatMention] = []
        seen: set[str] = set()
        self_id = snapshot.player.player_id
        for raw in threats:
            if not isinstance(raw, str) or not ENTITY_ID_RE.match(raw):
                continue  # not a structured entity ID: dropped, not parsed
            if raw in seen or not self._is_threat_entity(raw, snapshot,
                                                          self_id):
                continue
            seen.add(raw)
            mentions.append(ThreatMention(
                entity_id=raw, turn=turn,
                tile=self._entity_tile(raw, snapshot)))
        return mentions

    def _is_threat_entity(self, entity_id: str, snapshot: Snapshot,
                          self_id: int) -> bool:
        """§3 line 41: only opponent/barbarian entities count as threats —
        own entities and entities owned by self are dropped; ids that do
        not resolve in the current snapshot are dropped, never counted.
        Player ids resolve against self only."""
        kind, _, raw_id = entity_id.partition(":")
        try:
            entity = int(raw_id)
        except ValueError:
            return False
        if kind == "player":
            return entity != self_id
        if kind == "unit":
            for unit in snapshot.units:
                if unit.id == entity:
                    return not unit.is_own and unit.owner_id != self_id
        elif kind == "city":
            for city in snapshot.cities:
                if city.id == entity:
                    return not city.is_own and city.owner_id != self_id
        return False  # unresolvable id: dropped, never counted

    def _entity_tile(self, entity_id: str,
                     snapshot: Snapshot) -> Optional[tuple[int, int]]:
        kind, _, raw_id = entity_id.partition(":")
        try:
            entity = int(raw_id)
        except ValueError:
            return None
        if kind == "unit":
            for unit in snapshot.units:
                if unit.id == entity:
                    return (unit.x, unit.y)
        elif kind == "city":
            for city in snapshot.cities:
                if city.id == entity:
                    return (city.x, city.y)
        return None  # players have no tile; fogged entities: unknown


def _actor_of_ref(ref: str) -> Optional[str]:
    """The actor token of a well-formed candidate ref (None otherwise)."""
    if not is_candidate_id(ref):
        return None
    return ref.split(":")[1]


def _activity_name(activity_id: Any) -> Optional[str]:
    """The snapshot's activity name for a raw activity id (obs_extract
    ACTIVITY_NAMES)."""
    if isinstance(activity_id, bool) or not isinstance(activity_id, int):
        return None
    return ACTIVITY_NAMES.get(activity_id)
