# SPDX-License-Identifier: MIT
"""lechonk.core.runner — the turn pipeline, state machine and supervisor (WP2).

Implements design-plan-freeciv-v1.1.md §3 lines 32-35 and §7.1
(core/runner.py module contract):

- Turn state machine: ACTING -> FINISH_SENT -> WAITING -> NEXT_OWN_PHASE |
  GAME_OVER | FAILED, with per-state progress deadlines enforced by the
  supervisor (activity-independent): ACTING ≤ 180 s; FINISH_SENT/WAITING ≤
  the pingtimeout-derived bound (720 s + margin = 900 s); startup ≤ 120 s;
  per-mutation send+read-back ≤ 30 s; teardown ≤ 30 s. Any state exceeding
  its deadline → FAILED/TRUNCATED with artifacts.
- Run-level stall watchdog: no transcript event for 10 minutes → the
  ordered shutdown path (abort + guarded session close; the unwind emits
  TRUNCATED with the reason).
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
artifacts, never blind replay, no duplicate mutation. The resubmission
itself is never repeated: a second disconnect (or a contradicted
resend) quarantines the same way. Reconciliation decisions are ledgered
as mutation_reconciliation metric_events so the P11 audit can verify the
non-execution proof behind every repeat execution.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

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
    RepairPrompt,
    TurnPrompt,
)
from lechonk.core.delta import compute_delta, enforce_delta_budget
from lechonk.core.schemas import (
    ActionBatch,
    ActionResultRecord,
    Candidate,
    CandidateTable,
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
ENTITY_ID_RE = re.compile(r"^(unit|city|player):[0-9]+$")


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
    why: str = ""  # why the response was unusable (repair/degraded reason)
    degraded: bool = False  # the degraded default applies


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
    ``stall_s`` → the ordered shutdown path. Watches the ledger's
    last-event clock reading."""

    def __init__(self, *, stall_s: float, interval_s: float,
                 clock: Callable[[], float], ledger: TranscriptLedger,
                 on_stall: Callable[[str], None]):
        self._stall_s = stall_s
        self._interval_s = interval_s
        self._clock = clock
        self._ledger = ledger
        self._on_stall = on_stall
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="lechonk-stall-watchdog", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._clock() - self._ledger.last_event > self._stall_s:
                self._on_stall(
                    f"run-level stall watchdog: no transcript event for "
                    f"{self._stall_s:.0f}s")
                return
            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()


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
        self.ledger.append(MetricEventRecord(
            name="state_transition", value=f"{prev.value}->{state.value}"))

    def _emit(self, record: Any, *, durable: bool = False) -> int:
        return self.ledger.append(record, durable=durable)

    def _emit_action_result(self, *, durable: bool = False, **fields: Any) -> int:
        return self._emit(ActionResultRecord(**fields), durable=durable)

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

    # ------------------------------------------------------------------
    # supervision
    # ------------------------------------------------------------------

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
            on_stall=self._request_abort)
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
        # VICTORY / TRUNCATED), never by the dict alone.
        terminated, truncated, defeated = self._session_game_over_flags()
        extraction = adapter.obs_extract.extract_snapshot(
            self._session.obs, self._session.info,
            terminated=terminated, truncated=truncated,
            my_player_is_defeated=defeated,
            my_player_id=self._session.player_id,
            ruleset=self._session.ruleset,
            action_probs=self._session.action_probs,
        )
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
        by_domain: dict[str, int] = {}
        for cand in table.candidates:
            by_domain[cand.domain] = by_domain.get(cand.domain, 0) + 1
        candidates_seq = self._emit(CandidatesRefRecord(
            turn=turn,
            table_hash=hashlib.sha256(
                table.render().encode("utf-8")).hexdigest(),
            candidate_count=len(table.candidates),
            by_domain=by_domain,
            opportunity_domains=opportunity_domains(table.candidates)))

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
        result, repair_used, why = self._model_phase(
            adapter, turn=turn, table=table, snapshot=snapshot,
            delta_text=delta_text, snapshot_seq=snapshot_seq,
            candidates_seq=candidates_seq, candidates_text=candidates_text,
            acting_entered=acting_entered, provenance=provenance)
        self._check_acting_deadline(acting_entered)
        self._check_abort()

        # --- memory update (rides the ActionBatch, never a separate call)
        if result.batch is not None and result.batch.memory_update is not None:
            update = result.batch.memory_update
            self.memory.apply(update, turn=turn)
            self._emit(CommitmentRecord(
                turn=turn,
                commitments=list(update.commitments),
                threats=self._threat_mentions(update.threats, turn, snapshot),
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
        flags) to the run outcome. The VICTORY branch is evidence-gated:
        ``terminated_game_over`` means the env's defeat flag said
        not-defeated, but a False flag is not positive proof of survival
        (the vendor method returns False both when the player is
        provably alive AND when the player record is missing —
        civ_controller.py:303-330), so the defeat status must be provable
        or the run reports TRUNCATED — never a false VICTORY."""
        value = getattr(outcome, "value", outcome)
        self._transition(RunState.GAME_OVER)
        if value == "terminated_defeated":
            # terminated=True with the env's defeat flag True: positive
            # evidence of defeat.
            return ("DEFEATED", "my player is defeated", None)
        if value == "truncated_infra":
            return ("TRUNCATED", reason or "infrastructure truncation", None)
        if value == "terminated_game_over":
            defeated = self._my_player_defeated(self._session.obs)
            if defeated is None:
                return ("TRUNCATED",
                        "game over with unknown defeat status (no "
                        "well-formed player record, no positive env "
                        "defeat signal) — never a false VICTORY", None)
            return ("DEFEATED" if defeated else "VICTORY",
                    f"game over (defeated={defeated})", None)
        return ("TRUNCATED", reason or "unknown extraction outcome", None)

    def _my_player_defeated(self, obs: Any) -> Optional[bool]:
        """The defeat status at a game-over boundary: True/False when
        provable, None when unknown (never a false VICTORY).

        A well-formed player record (is_alive a bool) is authoritative.
        When the record is missing or malformed, the env's defeat flag
        decides — but only positively: the vendor method returns False
        both when the player is provably alive and when the record is
        missing (civ_controller.py:303-330), so anything but a True flag
        leaves the status unknown."""
        players = (obs or {}).get("player") or {}
        pid = self._session.player_id
        me = players.get(pid)
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
        if resp.error is not None:
            return f"API call failed: {resp.error}"
        if resp.malformed is not None or resp.batch is None:
            return f"malformed output: {resp.malformed or 'no batch'}"
        if result.batch is not None and result.batch.actions and not result.planned:
            classes = sorted({cls for _, cls in result.invalid})
            return (f"all {len(result.batch.actions)} emitted actions "
                    f"invalid (classes: {', '.join(classes)})")
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
        outputs; table rejections for invalid refs). Returns the planned
        (executable) actions after group-abort filtering."""
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
        by_id = table.by_id()
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
                    detail=(f"ref not in the offered candidate table "
                            f"({'never legal this turn — state_fog_error' if cls == 'state_fog_error' else 'not a candidate ID — schema_error'})"))
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
                         origin_seqs: Optional[dict[str, int]] = None
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
            origin_seq = (origin_seqs or {}).get(actor)
            result, seq = self._execute_one(
                adapter, cand, turn=turn, provenance=provenance,
                call_seq=call_seq, snapshot_seq=snapshot_seq,
                candidates_seq=candidates_seq, reason=reason,
                origin_seq=origin_seq)
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
        scripted_default with reason."""
        log = defaults_mod.decide(snapshot, table)
        planned: list[_PlannedAction] = []
        for decision in log.decisions:
            cand = table.by_id().get(decision.candidate_id)
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
        is executed-or-unknown (never resent)."""
        target, affected, actor_tile = self._describe(cand)
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
                actor_tile=actor_tile)
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
            readback=result.readback.status, detail=result.detail,
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
                              actor_tile: Optional[tuple[int, int]]
                              ) -> tuple[Any, int]:
        """§3 line 35: settle a mutation whose read-back is inconclusive
        after a disconnect, ONLY by full-state reconciliation — the
        target postconditions are compared on reconnect (the pre-mutation
        state against a fresh read through the session). Verified
        non-execution permits ONE resubmission (continued_model_order,
        origin-linked); executed-or-unknown is quarantined → the bounded
        fail-stop. Never blind replay; no duplicate mutation."""
        # The uncertain attempt is ledgered first (the mutation WAS sent;
        # its verification is inconclusive).
        uncertain_seq = self._emit_action_result(
            turn=turn, ref=cand.id, domain=cand.domain, verb=cand.verb,
            target=target, affected=affected, actor_tile=actor_tile,
            outcome="executed", provenance_class=provenance,
            origin_seq=origin_seq, reason=reason,
            snapshot_seq=snapshot_seq, candidates_seq=candidates_seq,
            call_seq=call_seq, readback="inconclusive",
            detail=(f"mutation send failed after dispatch ({exc!r}): "
                    f"the execution state is unknown — settling by "
                    f"full-state reconciliation on reconnect (never "
                    f"blind replay, §3 line 35)"),
            durable=True)
        # Full-state reconciliation: re-read (the reconnect) and compare
        # the target postconditions against the pre-mutation state.
        pre_obs = self._session.obs
        pre_info = self._session.info
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
        never repeated."""
        # Provenance honesty (§3 line 45): continued_model_order is a
        # MODEL-arm label — a scripted arm's resubmission mirrors its
        # origin attempt's provenance class, with the same origin linkage.
        resend_provenance = ("continued_model_order"
                             if provenance == "model" else provenance)
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
                detail=(f"the resubmission send also failed after dispatch "
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
                call_seq=None, readback=readback, detail=result.detail,
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
                call_seq=None, readback=readback, detail=result.detail,
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
            call_seq=None, readback=readback, detail=result.detail,
            durable=True)
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
        non-execution, §3 line 32)."""
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
            cand = table.by_id().get(standing.ref)
            if cand is None:
                for candidate in table.candidates:
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
        resolution (the fork's research-continuation default)."""
        if prev_snapshot is None:
            return
        prev_research = prev_snapshot.player.research
        cur_research = snapshot.player.research
        if (prev_research.current_target_id is not None
                and cur_research.current_target_id is not None
                and prev_research.current_target_id
                != cur_research.current_target_id
                and self._research_last_turn != cur_research.current_target_id):
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
            if (prev_city.production is not None
                    and city.production is not None
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
        at emission. The tile is resolved from the snapshot when visible."""
        mentions: list[ThreatMention] = []
        seen: set[str] = set()
        for raw in threats:
            if not isinstance(raw, str) or not ENTITY_ID_RE.match(raw):
                continue  # not a structured entity ID: dropped, not parsed
            if raw in seen:
                continue
            seen.add(raw)
            mentions.append(ThreatMention(
                entity_id=raw, turn=turn,
                tile=self._entity_tile(raw, snapshot)))
        return mentions

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
    names = {0: "idle", 1: "cultivate", 2: "mine", 3: "irrigate",
             4: "fortified", 5: "sentry", 6: "pillage", 7: "goto",
             8: "explore", 9: "transform", 10: "fortifying", 11: "clean",
             12: "base", 13: "road", 14: "convert", 15: "plant"}
    if isinstance(activity_id, bool) or not isinstance(activity_id, int):
        return None
    return names.get(activity_id)
