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
"""

from __future__ import annotations

import hashlib
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

    max_turns: int = 50
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
    aifill: int = 2


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
        aifill=config.aifill,
        # §3 startup ≤ 120 s + server-spawn margin (the session default).
        startup_deadline=config.startup_deadline_s + 30.0,
        step_deadline=config.mutation_deadline_s,
        wait_deadline=config.waiting_deadline_s,
        close_deadline=config.teardown_deadline_s,
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
                 clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.client = client
        self.ledger = ledger
        self.manifest = manifest
        self._session_factory = session_factory or (
            lambda: _default_session_factory(config))
        self._clock = clock
        self.memory = StrategyMemory()
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
        # Standing orders (continued_model_order) + engine-automatic
        # detection trackers.
        self._standing: dict[str, _StandingOrder] = {}
        self._research_last_turn: Optional[int] = None
        self._produce_last_turn: dict[int, str] = {}
        self._game_over: Optional[tuple[bool, bool]] = None  # (terminated, truncated)
        # The TRUE count of turns that completed (reached a turn boundary /
        # NEXT_OWN_PHASE) — survives an exception unwind out of _run_turns,
        # so the run result never reports a stale 0 (WP4 fix).
        self._turns_completed = 0

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

    def _emit(self, record: Any, *, durable: bool = False) -> int:
        return self.ledger.append(record, durable=durable)

    def _emit_action_result(self, *, durable: bool = False, **fields: Any) -> int:
        return self._emit(ActionResultRecord(**fields), durable=durable)

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

    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise _AbortRun(self._abort_reason or "abort requested")

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
        if self._clock() - acting_entered > self.config.acting_deadline_s:
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
            if turn > self.config.max_turns:
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
        self._emit(TurnStartRecord(turn=turn))

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
            offered = table.offered()
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
                offered.render().encode("utf-8")).hexdigest(),
            candidate_count=len(offered.candidates),
            by_domain=by_domain,
            opportunity_domains=opportunity_domains(offered.candidates)))

        # Engine-automatic resolutions (counted, never hidden, §3 line 38):
        # a research/production target that changed during WAITING with no
        # executed action of ours is the engine's own resolution.
        self._detect_engine_automatic(
            turn=turn, prev_snapshot=prev_snapshot, snapshot=snapshot,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq)
        self._research_last_turn = None
        self._produce_last_turn = {}

        delta = enforce_delta_budget(compute_delta(prev_snapshot, snapshot))
        delta_text = delta.render()
        self.memory.begin_turn(turn)
        candidates_text = table.render()
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
        self._check_acting_deadline(acting_entered)
        self._check_abort()

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
        if self._clock() - self._state_entered > self.config.waiting_deadline_s:
            raise StateDeadlineExceeded(
                RunState.WAITING, self.config.waiting_deadline_s)
        self.memory.advance(turn)
        if finish.terminated or finish.truncated:
            self._transition(RunState.GAME_OVER)
            defeated = self._my_player_defeated(finish.observation)
            if finish.truncated and not finish.terminated:
                return ("TRUNCATED",
                        "infrastructure truncation at the phase boundary "
                        "(server timeout) — artifacts intact", None)
            if defeated is None:
                return ("TRUNCATED",
                        "game over at the phase boundary with unknown "
                        "defeat status (no well-formed player record, no "
                        "positive env defeat signal) — never a false "
                        "VICTORY — artifacts intact", None)
            return ("DEFEATED" if defeated else "VICTORY",
                    "game over at the phase boundary "
                    f"(defeated={defeated})", None)
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
        prompt = TurnPrompt(
            call_id=f"{run_id}/t{turn}/call1", turn_id=f"{run_id}/t{turn}",
            memory=self.memory.render_for_prompt(),
            snapshot=snapshot.render(), delta=delta_text,
            candidates=candidates_text,
            budget_s=max(0.0, self.config.acting_deadline_s
                         - (self._clock() - acting_entered)))
        resp = self.client.complete_turn(prompt)
        call_seq = self._emit(ModelCallRecord(
            call_id=prompt.call_id, attempt=1, prompt_hash=resp.prompt_hash,
            latency_ms=resp.latency_ms, tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out, cost_usd=resp.cost_usd,
            fingerprint=resp.fingerprint, error=resp.error))
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
            memory=self.memory.render_for_prompt(),
            snapshot=snapshot.render(), delta=delta_text,
            candidates=candidates_text,
            budget_s=max(0.0, self.config.acting_deadline_s
                         - (self._clock() - acting_entered)),
            error_class=self._repair_class(resp, result),
            failed_refs=[ref for ref, _ in result.invalid],
            refreshed_candidates=candidates_text)
        resp2 = self.client.complete_repair(repair_prompt)
        call_seq2 = self._emit(ModelCallRecord(
            call_id=repair_prompt.call_id, attempt=2,
            prompt_hash=resp2.prompt_hash, latency_ms=resp2.latency_ms,
            tokens_in=resp2.tokens_in, tokens_out=resp2.tokens_out,
            cost_usd=resp2.cost_usd, fingerprint=resp2.fingerprint,
            error=resp2.error))
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
        why = (f"model unresponsive after the repair cap (≤2 API attempts "
               f"per turn): {result2.why}")
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
        by_id = table.offered().by_id()
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
                         candidates_seq: int, reason: str = ""
                         ) -> set[str]:
        """Execute planned actions sequentially with per-action read-back
        verification. After each executed mutation the remaining actions
        are revalidated against refreshed state; a stale or illegal action
        aborts only that entity's remaining group (logged rejected /
        aborted_group); other entities proceed; the turn is never aborted.
        Returns the set of actor IDs acted for."""
        remaining = list(planned)
        aborted_actors: set[str] = set()
        acted_actors: set[str] = set()
        executed_any = False
        while remaining:
            self._check_abort()
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
            readback = result.readback.status
            executed_any = True
            remaining.pop(0)
            acted_actors.add(actor)
            self._track_executed(cand, readback)
            if provenance == "model":
                self._update_standing(actor, cand, seq, readback)
            if readback == adapter.execute.CONTRADICTED:
                # An illegal action aborts the entity's remaining group.
                aborted_actors.add(actor)
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
        set so both arms act with identical capability (W2-1)."""
        offered = table.offered()
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
                reason=reasons.get(group[0].ref, why))
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

    def _track_executed(self, cand: Candidate, readback: str) -> None:
        """Track executed research/produce actions for the
        engine-automatic detection (a target change caused by our own
        action is not engine-automatic)."""
        if readback not in ("verified", "inconclusive"):
            return
        if cand.domain == "tech" and cand.verb.startswith("research_tech_"):
            self._research_last_turn = cand.payload.get("tech_id")
        elif cand.domain == "city" and cand.verb.startswith("produce_"):
            try:
                self._produce_last_turn[int(cand.actor_id)] = \
                    cand.payload.get("production")
            except (TypeError, ValueError):
                pass

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
        state_fog_error and never executed from the uncapped table."""
        offered = table.offered()
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
            result, _seq = self._execute_one(
                adapter, cand, turn=turn,
                provenance="continued_model_order", call_seq=None,
                snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
                reason="pipeline continuation of a model-originated order "
                       "(linked to its origin)",
                origin_seq=standing.origin_seq)
            readback = result.readback.status
            standing.reissued = True
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
    # engine-automatic detection (§3 line 38)
    # ------------------------------------------------------------------

    def _detect_engine_automatic(self, *, turn: int,
                                 prev_snapshot: Optional[Snapshot],
                                 snapshot: Snapshot, snapshot_seq: int,
                                 candidates_seq: int) -> None:
        """Count engine-automatic resolutions of new research/production
        choices — never hidden (§3 line 38). A target change between turn
        starts that our own executed actions did not cause is the engine's
        resolution (the fork's research-continuation default). This
        includes the None -> target first choice, gated the same way: a
        first research/production action of ours attributes the choice to
        the model, never to the engine."""
        if prev_snapshot is None:
            return
        prev_research = prev_snapshot.player.research
        cur_research = snapshot.player.research
        prev_target_id = prev_research.current_target_id
        cur_target_id = cur_research.current_target_id
        if (cur_target_id is not None
                and prev_target_id != cur_target_id
                and self._research_last_turn != cur_target_id):
            name = cur_research.current_target or \
                f"tech#{cur_research.current_target_id}"
            self._emit_action_result(
                turn=turn, ref=f"engine_automatic:research:{name}",
                domain="tech", verb="research_tech_auto", target=name,
                affected=name, outcome="executed",
                provenance_class="engine_automatic", readback="none",
                snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, call_seq=None,
                detail=(f"research target changed "
                        f"{prev_research.current_target} -> {name} during "
                        f"WAITING with no executed tech action — "
                        f"engine-automatic continuation (counted, never "
                        f"hidden, §3 line 38)"))
        prev_cities = {c.id: c for c in prev_snapshot.cities if c.is_own}
        for city in snapshot.cities:
            if not city.is_own or city.id not in prev_cities:
                continue
            prev_city = prev_cities[city.id]
            if (city.production is not None
                    and prev_city.production != city.production
                    and self._produce_last_turn.get(city.id)
                    != city.production):
                self._emit_action_result(
                    turn=turn,
                    ref=f"engine_automatic:production:{city.production}",
                    domain="city", verb="produce_auto",
                    target=city.production, affected=city.production,
                    outcome="executed", provenance_class="engine_automatic",
                    readback="none", snapshot_seq=snapshot_seq,
                    candidates_seq=candidates_seq, call_seq=None,
                    detail=(f"city {city.name} production changed "
                            f"{prev_city.production} -> {city.production} "
                            f"during WAITING with no executed produce "
                            f"action — engine-automatic resolution "
                            f"(counted, never hidden, §3 line 38)"))

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
