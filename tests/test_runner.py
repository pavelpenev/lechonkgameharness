# SPDX-License-Identifier: MIT
"""Runner tests (WP2): the turn pipeline, state machine, supervisor and
watchdog, driven offline by a fake session (real adapter read-backs) and a
fake model client — including the fault injection the WP2 acceptance
requires (timeout, malformed output, invalid action → the right repair
class or the degraded fallback; batch abort-group semantics; ledger
durability; the watchdog firing).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

from lechonk.adapter.session import StepOutcome
from lechonk.core.ledger import load_transcript
from lechonk.core.metrics import commitment_board
from lechonk.core.model_client import ModelResponse
from lechonk.core.runner import ENTITY_ID_RE
from lechonk.core.schemas import (
    ActionBatch,
    ActionResultRecord,
    CandidatesRefRecord,
    CityView,
    CommitmentRecord,
    MapView,
    MemoryUpdate,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    PlayerView,
    RunManifestRecord,
    Snapshot,
    SnapshotRefRecord,
    TurnStartRecord,
    UnitView,
)

from tests.lifecycle_fixtures import LifecycleSession
from tests.runner_fixtures import (
    FakeClock,
    FakeModelClient,
    FakeSession,
    advancing_response,
    error_response,
    make_runner,
    malformed_response,
    ok_response,
    turn_world,
)

REPO = Path(__file__).resolve().parents[1]


def _records(path):
    return load_transcript(path).records


def _actions(records):
    return [r for r in records if isinstance(r, ActionResultRecord)]


def _by_kind(records, cls):
    return [r for r in records if isinstance(r, cls)]


class _GameOverMidBatchSession(FakeSession):
    """A FakeSession whose named mutation ends the game (terminated=True)
    as its step outcome — the game-over-mid-batch injection."""

    def __init__(self, worlds, *, terminate_on):
        super().__init__(worlds)
        self._terminate_on = terminate_on

    def step(self, action, *, deadline=None):
        if action is not None and tuple(action) == self._terminate_on:
            outcome = super().step(action, deadline=deadline)
            return replace(outcome, terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


# ---------------------------------------------------------------------------
# happy path: a model-driven run to the horizon
# ---------------------------------------------------------------------------


def test_runner_happy_path_model_driven_run(tmp_path):
    """Two model-driven turns: the batch executes with read-back-verified
    effects, every state transition and action is ledgered, the memory
    update rides the batch, and the run completes at the horizon."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e",
                     "tech:cur_player:research_tech_alphabet_0"],
                    turn_id="t1",
                    commitments=[("research", "Alphabet", 5)],
                    threats=["unit:20"], notes="watch the enemy"),
        ok_response(["unit:11:sentry"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "COMPLETED"
    assert result.turns_completed == 2
    assert session.started and session.closed
    # The pipeline owns the finish: exactly one phase_done per turn.
    finishes = [c for c in session.calls if c[0] is None]
    assert len(finishes) == 2

    records = _records(runner.ledger.path)
    # Record-kind coverage per turn.
    assert isinstance(records[0], RunManifestRecord)
    assert len(_by_kind(records, TurnStartRecord)) == 2
    assert len(_by_kind(records, SnapshotRefRecord)) == 2
    assert len(_by_kind(records, CandidatesRefRecord)) == 2
    assert len(_by_kind(records, ModelCallRecord)) == 2
    # State transitions are ledgered.
    transitions = [r.value for r in _by_kind(records, MetricEventRecord)
                   if r.name == "state_transition"]
    assert "STARTUP->ACTING" in transitions
    assert "ACTING->FINISH_SENT" in transitions
    assert "FINISH_SENT->WAITING" in transitions
    assert "WAITING->NEXT_OWN_PHASE" in transitions
    # The executed actions: read-back verified, model provenance, with the
    # structured descriptors the metrics read.
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    assert {a.ref for a in executed} == {
        "unit:10:goto_e", "tech:cur_player:research_tech_alphabet_0",
        "unit:11:sentry"}
    goto = next(a for a in executed if a.verb == "goto_e")
    assert goto.readback == "verified"
    assert goto.provenance_class == "model"
    assert goto.actor_tile == (1, 1) and goto.affected == "tile:2,1"
    research = next(a for a in executed
                    if a.ref == "tech:cur_player:research_tech_alphabet_0")
    assert research.target == "Alphabet" and research.readback == "verified"
    # The memory update rode the batch: one commitment record carrying the
    # structured commitment + the threat mention (tile resolved from the
    # snapshot: unit 20 sits at (2,2)).
    commitments = _by_kind(records, CommitmentRecord)
    assert len(commitments) == 1
    assert commitments[0].turn == 1
    assert commitments[0].commitments[0].target == "Alphabet"
    assert commitments[0].threats[0].entity_id == "unit:20"
    assert commitments[0].threats[0].tile == (2, 2)
    # The memory advanced post-advance: turn 2's prompt carries it.
    assert "Alphabet" in client.turn_prompts[1].memory
    # The outcome record is last; the metrics were emitted before it.
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "COMPLETED"
    metric_names = {r.name for r in _by_kind(records, MetricEventRecord)
                    if r.name != "state_transition"}
    assert {"participation", "commitment_followthrough_any",
            "reason_but_dont_act"} <= metric_names
    # The metrics report is attached to the result.
    assert result.metrics.participation.denominator == 2
    assert result.metrics.participation.numerator == 2


# ---------------------------------------------------------------------------
# fault injection: timeout → repair (unresponsive) → executed
# ---------------------------------------------------------------------------


def test_runner_timeout_triggers_unresponsive_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        error_response("request timed out"),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # Two model calls: attempt 1 (error) + the one repair attempt.
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 2, 1]  # turn 1: 1+2, turn 2: 1
    assert calls[0].error == "request timed out"
    # The repair prompt carries the unresponsive class.
    assert len(client.repair_prompts) == 1
    assert client.repair_prompts[0].error_class == "unresponsive"
    # The repair's action executed with model provenance.
    actions = _actions(_records(runner.ledger.path))
    executed = [a for a in actions if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]
    assert executed[0].provenance_class == "model"
    assert executed[0].readback == "verified"
    assert executed[0].call_seq == calls[1].seq


# ---------------------------------------------------------------------------
# fault injection: malformed output → schema_error repair
# ---------------------------------------------------------------------------


def test_runner_malformed_output_triggers_schema_error_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        malformed_response(["(12, 34)", "move to (12, 34)"]),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    # One schema-error rejection per emitted raw ref (model attempts).
    rejected = [a for a in _actions(records) if a.outcome == "rejected"]
    assert [a.ref for a in rejected] == ["(12, 34)", "move to (12, 34)"]
    assert all(a.rejection_class == "schema_error" for a in rejected)
    assert all(a.provenance_class == "model" for a in rejected)
    # The repair prompt carries the schema_error class + the failed refs.
    assert client.repair_prompts[0].error_class == "schema_error"
    assert client.repair_prompts[0].failed_refs == ["(12, 34)",
                                                    "move to (12, 34)"]
    # The taxonomy counted the attempts (2 schema_error + 1 executed).
    assert result.metrics.illegal_attempts_total == 3
    assert result.metrics.illegal_action_shares["schema_error"].numerator == 2


# ---------------------------------------------------------------------------
# fault injection: invalid action → state_fog_error repair
# ---------------------------------------------------------------------------


def test_runner_invalid_action_triggers_state_fog_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:99:goto_n"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    rejected = [a for a in _actions(records) if a.outcome == "rejected"]
    # A well-formed ref that was never offered: state_fog_error.
    assert len(rejected) == 1
    assert rejected[0].ref == "unit:99:goto_n"
    assert rejected[0].rejection_class == "state_fog_error"
    assert client.repair_prompts[0].error_class == "state_fog_error"
    assert client.repair_prompts[0].failed_refs == ["unit:99:goto_n"]
    # The repair's valid action executed.
    executed = [a for a in _actions(records) if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]


# ---------------------------------------------------------------------------
# fault injection: unresponsive after the repair cap → degraded default
# ---------------------------------------------------------------------------


def test_runner_unresponsive_after_repair_cap_degrades(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        error_response("timeout 1"),
        error_response("timeout 2"),  # the repair also fails
        error_response("timeout 3"),
        error_response("timeout 4"),  # turn 2: the repair also fails
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # No third API attempt: exactly two calls per turn.
    assert client.calls == 4
    records = _records(runner.ledger.path)
    actions = _actions(records)
    defaults = [a for a in actions if a.provenance_class == "scripted_default"]
    # The degraded default resolved the new mandatory choices (research
    # completed + city production empty) via the published priority lists.
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.outcome == "executed" for a in defaults)
    assert all("model unresponsive" in a.reason for a in defaults)
    assert all(a.readback == "verified" for a in defaults)
    # The turn was still finished by the pipeline and the run completed.
    finishes = [c for c in session.calls if c[0] is None]
    assert len(finishes) == 2
    # The degraded decisions count in the per-domain shares, not as model
    # participation: turn 1 had opportunities but no model decision.
    assert result.metrics.participation.numerator == 0
    assert result.metrics.participation.denominator == 2


# ---------------------------------------------------------------------------
# batch semantics: a stale action aborts only its entity's group (§3 line 33)
# ---------------------------------------------------------------------------


def test_runner_mid_batch_stale_aborts_only_that_group(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    # unit 10: goto_e (executes, exhausting its moves), goto_nw (stale
    # after the move), build_city (aborted group); unit 11: sentry
    # (another entity — proceeds).
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:10:goto_nw",
                     "unit:10:build_city", "unit:11:sentry"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"  # the turn is never aborted
    actions = _actions(_records(runner.ledger.path))
    by_ref = {a.ref: a for a in actions}
    # The first action executed and was read-back verified.
    assert by_ref["unit:10:goto_e"].outcome == "executed"
    assert by_ref["unit:10:goto_e"].readback == "verified"
    # The stale action: rejected illegal_in_rule (it was offered, then
    # became illegal after the mid-batch mutation).
    stale = by_ref["unit:10:goto_nw"]
    assert stale.outcome == "rejected"
    assert stale.rejection_class == "illegal_in_rule"
    # The rest of unit 10's group: aborted_group (logged, never sent).
    aborted = by_ref["unit:10:build_city"]
    assert aborted.outcome == "aborted_group"
    # The other entity's action proceeded and executed.
    assert by_ref["unit:11:sentry"].outcome == "executed"
    assert by_ref["unit:11:sentry"].readback == "verified"
    # The taxonomy counted the model-emitted attempts only: goto_e (executed)
    # + goto_nw (rejected) + the turn-1 sentry (executed) = 3. The
    # aborted_group record is a pipeline-side group abort, not an attempt,
    # and the turn-2 continued_model_order re-issue is not model-emitted.
    assert result.metrics.illegal_attempts_total == 3


def test_runner_game_over_mid_batch_aborts_every_remaining_action(tmp_path):
    """A mutation ends the game mid-batch with >=2 pending actions: EVERY
    planned-but-unexecuted action — including the FIRST pending action
    after the executed one — gets its aborted_group record."""
    worlds = [turn_world(1), turn_world(2)]
    session = _GameOverMidBatchSession(worlds, terminate_on=("unit", 10,
                                                            "goto_4"))
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:10:build_city",
                     "unit:11:sentry"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "VICTORY"
    actions = _actions(_records(runner.ledger.path))
    # The first action executed and ended the game mid-batch.
    executed = [a for a in actions if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]
    # EVERY pending action got its aborted_group record — including the
    # first pending action after the executed one.
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    assert [a.ref for a in aborted] == ["unit:10:build_city",
                                        "unit:11:sentry"]
    assert all(a.detail == "game ended mid-batch" for a in aborted)


# ---------------------------------------------------------------------------
# continued_model_order: the pipeline re-issues a standing order once
# ---------------------------------------------------------------------------


def test_runner_continues_standing_order_linked_to_origin(tmp_path):
    # Turn 1: the model orders unit 11 to sentry (executed, verified).
    # Turn 2: the unit is idle again (the order is not in effect) and the
    # model's batch is empty → the pipeline re-issues the order once as
    # continued_model_order, linked to the originating model action.
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 0})]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    assert len(continuations) == 1
    cont = continuations[0]
    assert cont.turn == 2 and cont.verb == "sentry"
    assert cont.outcome == "executed" and cont.readback == "verified"
    # Linked to its origin: the turn-1 model action_result.
    by_seq = {a.seq: a for a in actions}
    origin = by_seq[cont.origin_seq]
    assert origin.provenance_class == "model"
    assert origin.turn == 1 and origin.verb == "sentry"
    # The continuation counts as model participation (model-originated).
    assert result.metrics.participation.numerator == 2


def test_runner_fortifying_unit_counts_as_standing_in_effect(tmp_path):
    # Turn 1: the model orders unit 11 to fortify (executed, verified —
    # the fake session applies ACTIVITY_FORTIFIED). Turn 2: the unit shows
    # ACTIVITY_FORTIFYING (10), freeciv's in-turn fortify state, which
    # completes to ACTIVITY_FORTIFIED at the turn start — the order is in
    # effect, so the pipeline must NOT re-issue it as
    # continued_model_order even with an empty model batch (live
    # evidence: the WP2 model run read-backs).
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 10})]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:11:fortify"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    by_turn_ref = {(a.turn, a.ref): a for a in actions}
    origin = by_turn_ref[(1, "unit:11:fortify")]
    assert origin.provenance_class == "model"
    assert origin.outcome == "executed" and origin.readback == "verified"
    # Fortifying = the order in effect: no continued_model_order re-issue.
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    assert continuations == []
    # The model participated exactly once (its own turn-1 action).
    assert result.metrics.participation.numerator == 1


# ---------------------------------------------------------------------------
# engine-automatic resolutions are counted, never hidden (§3 line 38)
# ---------------------------------------------------------------------------


def test_runner_counts_engine_automatic_research_continuation(tmp_path):
    # Turn 1: researching Alphabet (tech 1 per default_techs), the model
    # only moves a unit. Turn 2: the engine continued research to Pottery
    # (tech 2) during WAITING — no executed tech action → engine_automatic,
    # counted. (The snapshot's ResearchView resolves ids via the tech domain:
    # researching=1 -> "Alphabet", researching=2 -> "Pottery".)
    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 2
    assert automatic[0].domain == "tech"
    assert automatic[0].target == "Pottery"
    assert automatic[0].outcome == "executed"
    # Counted in the research domain share, not as model participation.
    research_share = result.metrics.domain_model_shares["research"]
    assert research_share.denominator == 1  # only the engine resolution
    assert research_share.numerator == 0


# ---------------------------------------------------------------------------
# supervision: the ACTING deadline and the stall watchdog
# ---------------------------------------------------------------------------


def test_runner_acting_deadline_exceeded_fails_bounded(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        advancing_response(
            lambda: ok_response(["unit:10:goto_e"], turn_id="t1",
                                with_memory=False),
            clock, seconds=200.0),  # the model call alone exceeds ACTING 180s
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    # Artifacts intact: the outcome record is ledgered.
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert session.closed


def test_runner_stall_watchdog_fires_on_silent_pipeline(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    # The finish step blocks until close() — no transcript event flows.
    session = FakeSession(worlds, block_step=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"stall_watchdog_s": 0.2,
                          "watchdog_interval_s": 0.02})
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "stall watchdog" in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "stall watchdog" in records[-1].reason
    assert session.closed  # the ordered shutdown closed the session


# ---------------------------------------------------------------------------
# outcome taxonomy: DEFEATED / VICTORY at the phase boundary
# ---------------------------------------------------------------------------


def test_runner_defeated_at_phase_boundary(tmp_path):
    obs, info = turn_world(1)
    obs["player"][0]["is_alive"] = False  # my player is defeated
    session = FakeSession([(obs, info)], final_terminated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    records = _records(runner.ledger.path)
    assert records[-1].outcome == "DEFEATED"


def test_runner_victory_at_phase_boundary(tmp_path):
    session = FakeSession([turn_world(1)], final_terminated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "VICTORY"


# ---------------------------------------------------------------------------
# ledger durability through the runner (§3 line 34)
# ---------------------------------------------------------------------------


def test_runner_transcript_is_durable_and_recovers(tmp_path):
    """The runner's transcript survives a simulated mid-write kill: the
    trailing partial record is discarded on load and the ledger recovers."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    path = Path(runner.ledger.path)
    good = path.read_bytes()
    good_seqs = [r.seq for r in load_transcript(path).records]

    # Simulate a kill mid-write of the NEXT record (torn bytes at EOF).
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"action_result","turn":3,"ref":"unit:10:g')
    load = load_transcript(path)
    assert load.discarded_trailing_partial
    # Every complete record survives the kill; only the torn tail is
    # discarded.
    assert [r.seq for r in load.records] == good_seqs
    valid_seqs = [r.seq for r in load.records]
    assert valid_seqs == sorted(valid_seqs) and len(valid_seqs) > 5

    # Recovery: attach truncates the torn tail and resumes the seq.
    from lechonk.core.ledger import TranscriptLedger

    recovered = TranscriptLedger.attach(path)
    assert recovered.next_seq == valid_seqs[-1] + 1
    recovered.close()
    assert path.read_bytes() == good  # the torn bytes are gone


# ---------------------------------------------------------------------------
# the runner imports the adapter lazily (§7.1)
# ---------------------------------------------------------------------------


def test_runner_imports_the_adapter_only_inside_functions():
    """No module-level lechonk.adapter import in core/runner.py: the
    adapter (and civrealm) import happens lazily, only after the stack is
    up."""
    source = (REPO / "src" / "lechonk" / "core" / "runner.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "lechonk.adapter" \
                        or alias.name.startswith("lechonk.adapter."):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "lechonk.adapter" \
                    or (node.module or "").startswith("lechonk.adapter."):
                offenders.append((node.lineno, node.module))
    assert not offenders, f"module-level adapter imports: {offenders}"


def test_runner_state_machine_and_deadlines_are_aligned_with_3():
    """The §3 supervision constants: ACTING ≤ 180 s, FINISH_SENT/WAITING ≤
    the pingtimeout-derived bound (720 s + margin), per-mutation ≤ 30 s,
    startup ≤ 120 s, teardown ≤ 30 s, stall watchdog 10 min."""
    from lechonk.core.runner import (
        ACTING_DEADLINE_S,
        MUTATION_DEADLINE_S,
        PINGTIMEOUT_S,
        STARTUP_DEADLINE_S,
        STALL_WATCHDOG_S,
        TEARDOWN_DEADLINE_S,
        WAITING_MARGIN_S,
        RunnerConfig,
    )

    assert ACTING_DEADLINE_S == 180.0
    assert PINGTIMEOUT_S == 720.0
    assert WAITING_MARGIN_S == 180.0  # 720 + margin = 900
    assert STARTUP_DEADLINE_S == 120.0
    assert MUTATION_DEADLINE_S == 30.0
    assert TEARDOWN_DEADLINE_S == 30.0
    assert STALL_WATCHDOG_S == 600.0
    config = RunnerConfig()
    assert config.waiting_deadline_s == 900.0
    assert config.acting_deadline_s == 180.0


# ---------------------------------------------------------------------------
# WP3: the client-declared provenance class (paired scripted arms)
# ---------------------------------------------------------------------------


def test_runner_client_provenance_class_passthrough(tmp_path):
    """A client MAY declare ``provenance_class`` (WP3 paired runs drive a
    scripted comparator arm through this same pipeline; its decisions
    are ``scripted_default``, never ``model``). The default when the
    attribute is absent is ``model`` (the pre-existing tests assert that
    path)."""
    class ScriptedClient(FakeModelClient):
        provenance_class = "scripted_default"

    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = ScriptedClient([
        # turn 1 attempt 1: a well-formed but never-offered ref — rejected
        # as state_fog_error, and (batch with actions, none planned) the
        # one repair fires
        ok_response(["unit:999:goto_n"], turn_id="t1", with_memory=False),
        # the one repair (state_fog_error class): a valid action
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    assert executed
    assert all(a.provenance_class == "scripted_default" for a in executed)
    rejected = [a for a in actions if a.outcome == "rejected"]
    assert [a.ref for a in rejected] == ["unit:999:goto_n"]
    assert rejected[0].rejection_class == "state_fog_error"
    assert rejected[0].provenance_class == "scripted_default"
    # The repair prompt was used (the unoffered ref triggered it) and the
    # repair's action executed with the client's provenance class.
    assert len(client.repair_prompts) == 1
    assert client.repair_prompts[0].error_class == "state_fog_error"
    assert client.repair_prompts[0].failed_refs == ["unit:999:goto_n"]
    # No standing orders are tracked for a non-model client (the
    # continued_model_order machinery is model-originated only).
    continued = [a for a in actions
                 if a.provenance_class == "continued_model_order"]
    assert continued == []


def test_runner_provenance_class_default_is_model(tmp_path):
    """The getattr default: a client WITHOUT the attribute labels its
    decisions ``model`` (the pre-existing behavior, pinned here)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    assert not hasattr(client, "provenance_class")
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    executed = [a for a in _actions(_records(runner.ledger.path))
                if a.outcome == "executed"]
    assert executed
    assert all(a.provenance_class == "model" for a in executed)


# ---------------------------------------------------------------------------
# WP4: budget-aware repair gating (the frozen 180 s ACTING deadline holds)
# ---------------------------------------------------------------------------


def test_runner_repair_timeout_within_budget_degrades_and_completes(tmp_path):
    """WP4 gating (a): the repair attempt times out inside the remaining
    ACTING budget (it surfaces as a transport failure on attempt 2 via the
    per-call hard request timeout) — the degraded default applies and the
    turn completes; no deadline exception."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 100 s of the 180 s ACTING budget.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=100.0),
        # The repair times out at its per-call bound (remaining - margin):
        # elapsed ends at 170 s, inside the deadline.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=70.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert "deadline" not in result.reason
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 2, 1]  # t1: 1+2, t2: 1
    assert calls[1].error == "request timed out"
    # Turn 1's decisions came from the degraded default.
    defaults = [a for a in _actions(_records(runner.ledger.path))
                if a.turn == 1 and a.outcome == "executed"]
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.provenance_class == "scripted_default" for a in defaults)
    # The repair prompt carried the remaining budget at repair time
    # (180 - 100 = 80 s) — the client's per-call timeout derives from it.
    assert round(client.repair_prompts[0].budget_s, 6) == 80.0


def test_runner_repair_skipped_below_budget_floor(tmp_path):
    """WP4 gating (b): remaining budget below the 20 s floor → the repair
    call is skipped entirely, the degraded default applies, and the ledger
    records the skip."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 165 s: remaining 15 s < 20 s floor.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=165.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # The repair was never called: one API call in turn 1, no repair prompt.
    assert len(client.repair_prompts) == 0
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 1]  # t1 attempt 1, t2 attempt 1
    # Turn 1 executed the degraded default.
    defaults = [a for a in _actions(_records(runner.ledger.path))
                if a.turn == 1 and a.outcome == "executed"]
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.provenance_class == "scripted_default" for a in defaults)
    # The skip is ledgered with the reason.
    skips = [r for r in _records(runner.ledger.path)
             if isinstance(r, MetricEventRecord)
             and r.name == "repair_skipped_insufficient_budget"]
    assert len(skips) == 1
    assert "15.0s" in skips[0].value


def test_runner_repair_within_budget_proceeds_normally(tmp_path):
    """WP4 gating (c)+(d): a first-attempt transport failure with a repair
    that fits the remaining budget → the repair proceeds exactly as before
    (no behavior change, no skip event)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=30.0),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert len(client.repair_prompts) == 1
    # The repair prompt carried the remaining budget (180 - 30 = 150 s).
    assert round(client.repair_prompts[0].budget_s, 6) == 150.0
    executed = [a for a in _actions(_records(runner.ledger.path))
                if a.outcome == "executed"]
    assert executed[0].ref == "unit:10:goto_e"
    assert executed[0].provenance_class == "model"
    assert executed[0].readback == "verified"
    skips = [r for r in _records(runner.ledger.path)
             if isinstance(r, MetricEventRecord)
             and r.name == "repair_skipped_insufficient_budget"]
    assert skips == []


def test_runner_reports_true_turns_completed_after_mid_turn_death(tmp_path):
    """WP4 fix: a run that dies mid-turn N+1 reports turns_completed == N
    (the pre-fix defect reported the stale local 0 on every exception
    unwind — the truncated wp4-eval model arms reported 0)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        # Turn 2 dies: the model call alone exceeds the ACTING deadline.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=200.0),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    assert result.turns_completed == 1
    # Artifacts intact: the outcome record is ledgered.
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


# ---------------------------------------------------------------------------
# FW-14a (A3): game-over flag plumbing — the acting boundary and the
# mid-batch / phase-boundary classifications
# ---------------------------------------------------------------------------

GOTO_REF = "unit:10:goto_e"
# The raw CivRealm action tuple for the goto_e candidate (action key
# goto_4 = dir8 E — the same table the read-back verifier uses).
GOTO_ACTION = ("unit", 10, "goto_4")


class _ActingBoundaryGameOverSession(FakeSession):
    """The session cache at the ACTING boundary holds the game-over
    state: an EMPTY observation plus the step flags the real
    AdapterSession caches (terminated/truncated + the env's defeat
    signal). The acting-boundary extraction must classify by the flags
    — never raise ValueError into the generic infrastructure handler."""

    def __init__(self, worlds, *, terminated, truncated, defeated,
                 empty_obs=True):
        super().__init__(worlds)
        self._terminated = terminated
        self._truncated = truncated
        self._defeated = defeated
        self._empty_obs = empty_obs

    @property
    def obs(self):
        return {} if self._empty_obs else super().obs

    @property
    def info(self):
        if self._empty_obs:
            return {"turn": 2, "available_actions": {}}
        return super().info

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated


def test_runner_acting_boundary_empty_game_over_obs_defeated(tmp_path):
    """A3a: an empty game-over observation at the acting boundary with
    terminated=True + the env's defeat signal True classifies as
    DEFEATED by the flags — not TRUNCATED "infrastructure failure" (the
    pre-fix ValueError path through the generic handler)."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=True)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert result.reason == "my player is defeated"
    assert "infrastructure failure" not in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "DEFEATED"


def test_runner_acting_boundary_empty_obs_truncated_stays_distinct(tmp_path):
    """The env flag values reach classification: truncated=True (not
    terminated) classifies as TRUNCATED infrastructure — distinct from
    DEFEATED, never confused with a defeat."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=False, truncated=True, defeated=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "DEFEATED"
    assert "truncated=True" in result.reason
    assert "infrastructure failure" not in result.reason


def test_runner_acting_boundary_missing_record_never_false_victory(tmp_path):
    """A3b: an empty game-over observation at the acting boundary with
    terminated=True but NO provable defeat status (no player record, no
    positive env defeat signal) is TRUNCATED/inconclusive — never a
    false VICTORY."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason


def test_runner_acting_boundary_victory_needs_a_live_record(tmp_path):
    """The VICTORY branch is evidence-gated: terminated=True with the
    env's defeat signal False AND a well-formed live player record in
    the observation classifies as VICTORY."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=False,
        empty_obs=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "VICTORY"
    assert "defeated=False" in result.reason


class _MidBatchEmptyGameOverSession(FakeSession):
    """A FakeSession caching the latest step's observation and game-over
    flags exactly like the real AdapterSession (FW-14a plumbing): the
    game-ending mutation returns an EMPTY observation (the real env's
    game-over observation) with terminated=True. The mid-batch
    classification (runner.py:632) must consult the env's defeat flag —
    never default to not-defeated on a missing player record."""

    def __init__(self, worlds, *, terminate_on, defeated):
        super().__init__(worlds)
        self._terminate_on = terminate_on
        self._defeated = defeated
        self._terminated = False
        self._truncated = False
        self._step_obs = None

    @property
    def obs(self):
        if self._step_obs is not None:
            return self._step_obs
        return super().obs

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated if (self._terminated or self._truncated) else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if action is not None and tuple(action) == self._terminate_on:
            outcome = replace(outcome, observation={}, terminated=True,
                              truncated=False)
        self._step_obs = outcome.observation
        self._terminated = bool(outcome.terminated)
        self._truncated = bool(outcome.truncated)
        return outcome


def test_runner_game_over_mid_batch_empty_obs_defeated(tmp_path):
    """A3b at runner.py:632: a mutation ends the game mid-batch with an
    EMPTY observation; the defeat status is proven by the env's defeat
    flag (True) -> DEFEATED — not a false VICTORY from the missing
    player record."""
    session = _MidBatchEmptyGameOverSession(
        [turn_world(1)], terminate_on=GOTO_ACTION, defeated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert "game over mid-batch (defeated=True)" in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "DEFEATED"


def test_runner_game_over_mid_batch_empty_obs_unknown_never_victory(tmp_path):
    """A3b at runner.py:632: the same empty game-over observation with
    NO provable defeat status (the env's defeat signal False — the
    vendor method returns False both when provably alive and when the
    record is missing) is TRUNCATED/inconclusive — never a false
    VICTORY."""
    session = _MidBatchEmptyGameOverSession(
        [turn_world(1)], terminate_on=GOTO_ACTION, defeated=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason
    assert "game over mid-batch" in result.reason


class _PhaseBoundaryEmptyGameOverSession(FakeSession):
    """The finish step ends the game with an EMPTY observation (the real
    env's game-over observation); the session caches the flags. The
    phase-boundary classification (runner.py:647) must consult the env's
    defeat flag — never default to not-defeated on a missing player
    record."""

    def __init__(self, worlds, *, defeated, truncated=False):
        super().__init__(worlds, final_terminated=not truncated,
                         final_truncated=truncated)
        self._defeated = defeated
        self._terminated = False
        self._truncated = False

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated if (self._terminated or self._truncated) else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if outcome.terminated or outcome.truncated:
            outcome = replace(outcome, observation={})
        self._terminated = bool(outcome.terminated)
        self._truncated = bool(outcome.truncated)
        return outcome


def test_runner_game_over_at_phase_boundary_empty_obs_defeated(tmp_path):
    """A3b at runner.py:647: the finish step ends the game with an
    EMPTY observation; the defeat status is proven by the env's defeat
    flag (True) -> DEFEATED — not a false VICTORY from the missing
    player record."""
    session = _PhaseBoundaryEmptyGameOverSession([turn_world(1)],
                                                 defeated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert "game over at the phase boundary (defeated=True)" in result.reason


def test_runner_game_over_at_phase_boundary_empty_obs_never_victory(tmp_path):
    """A3b at runner.py:647: the same empty game-over observation with
    NO provable defeat status is TRUNCATED/inconclusive — never a false
    VICTORY."""
    session = _PhaseBoundaryEmptyGameOverSession([turn_world(1)],
                                                 defeated=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason


def test_runner_truncated_at_phase_boundary_stays_truncated(tmp_path):
    """The env flag values reach classification at runner.py:647:
    truncated=True (not terminated) stays TRUNCATED infrastructure —
    distinct from DEFEATED, never confused with a defeat."""
    session = _PhaseBoundaryEmptyGameOverSession(
        [turn_world(1)], defeated=False, truncated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "infrastructure truncation at the phase boundary" in result.reason


# ---------------------------------------------------------------------------
# FW-14a (A5): an INCONCLUSIVE resend read-back quarantines (§3 line 35)
# ---------------------------------------------------------------------------


class _InconclusiveResendSession(LifecycleSession):
    """The resubmission's read-back is INCONCLUSIVE: the reconnect read
    proved non-execution (the resend is permitted), but the resend's own
    post-state can neither verify nor contradict the move (the unit
    vanished server-side). The §3 line 35 quarantine path must fire —
    the run never continues on an inconclusive settlement."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self._goto_sends = 0

    def step(self, action, *, deadline=None):
        if action == GOTO_ACTION:
            self._goto_sends += 1
            if self._goto_sends >= 2:
                # The resend: the unit vanished server-side — the
                # read-back can neither verify nor contradict the move.
                self.calls.append((action, deadline))
                obs, info = self._worlds[self._index]
                new_obs = copy.deepcopy(obs)
                del new_obs["unit"][10]
                self._worlds[self._index] = (new_obs, info)
                self._advance_cache(new_obs, info)
                return StepOutcome(observation=new_obs, reward=0.0,
                                   terminated=False, truncated=False,
                                   info=info, action=action,
                                   elapsed_s=0.01)
        return super().step(action, deadline=deadline)


def test_runner_inconclusive_resend_readback_quarantines(tmp_path):
    """A5: the ONE permitted resubmission's read-back is INCONCLUSIVE
    (the execution state is unknown) — the same quarantine path as an
    exception or a CONTRADICTED resend: the read-back record is
    emitted, a mutation_quarantine metric event is emitted, and the run
    takes the bounded fail-stop (TRUNCATED) — it never continues."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = _InconclusiveResendSession(
        worlds, disconnect_on=GOTO_ACTION, effect_applied=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    assert "read back inconclusive" in result.reason
    assert "never resent" in result.reason
    # Exactly two sends: the failed dispatch + the one resubmission —
    # never a third.
    assert session._goto_sends == 2
    assert len([c for c in session.calls if c[0] == GOTO_ACTION]) == 2

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 2
    # The uncertain attempt + the resubmission (both sent, both
    # inconclusive) — and nothing else.
    assert all(a.outcome == "executed" for a in by_ref)
    assert all(a.readback == "inconclusive" for a in by_ref)
    assert by_ref[0].provenance_class == "model"
    assert by_ref[1].provenance_class == "continued_model_order"
    assert by_ref[1].origin_seq == by_ref[0].seq
    # The first reconciliation proved non-execution (permitting the one
    # resubmission).
    reconciliations = [r for r in records
                       if isinstance(r, MetricEventRecord)
                       and r.name == "mutation_reconciliation"]
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: non_execution_verified" in reconciliations[0].value
    # The inconclusive resend read-back took the quarantine path: one
    # mutation_quarantine event naming the inconclusive read-back.
    quarantines = [r for r in records
                   if isinstance(r, MetricEventRecord)
                   and r.name == "mutation_quarantine"]
    assert len(quarantines) == 1
    assert "read back inconclusive" in quarantines[0].value
    assert GOTO_REF in quarantines[0].value
    # Bounded fail-stop with artifacts: the outcome record is ledgered;
    # the run did NOT continue (turn 1 never completed).
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "quarantined" in records[-1].reason
    assert result.turns_completed == 0
    assert session.closed


# ---------------------------------------------------------------------------
# threat mentions: opponent/barbarian conformance (B3, §3 line 41) and the
# ENTITY_ID_RE end anchor (C5 runner-side piece)
# ---------------------------------------------------------------------------


def _threat_snapshot() -> Snapshot:
    """Self is player 0; entities cover own, enemy-owned, owner-unknown
    (fogged foreign), and unresolvable ids."""
    return Snapshot(
        turn=1,
        player=PlayerView(player_id=0),
        map_view=MapView(width=8, height=8),
        units=[
            UnitView(id=10, is_own=True, type_name="Settlers", x=1, y=1),
            UnitView(id=20, is_own=False, owner_id=1,
                     type_name="Warriors", x=2, y=2),
            UnitView(id=30, is_own=False, owner_id=None,
                     type_name="Warriors", x=3, y=3),
        ],
        cities=[
            CityView(id=1, name="Rome", x=1, y=1, is_own=True),
            CityView(id=2, name="Carthage", x=4, y=4, is_own=False,
                     owner_id=1),
        ],
    )


def _threat_runner(tmp_path):
    return make_runner(tmp_path, client=FakeModelClient([]),
                       session=FakeSession([turn_world(1)]))


def test_threat_own_entity_mentions_not_counted(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    mentions = runner._threat_mentions(["unit:10", "city:1"], 1, snap)
    assert mentions == []


def test_threat_self_player_mention_dropped(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    assert runner._threat_mentions(["player:0"], 1, snap) == []


def test_threat_unresolvable_id_dropped(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    assert runner._threat_mentions(["unit:99", "city:99"], 1, snap) == []


def test_threat_opponent_barbarian_mentions_counted(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    mentions = runner._threat_mentions(
        ["unit:20", "city:2", "player:1", "unit:30"], 1, snap)
    assert [m.entity_id for m in mentions] == [
        "unit:20", "city:2", "player:1", "unit:30"]
    assert mentions[0].tile == (2, 2)
    assert mentions[1].tile == (4, 4)
    assert mentions[3].tile == (3, 3)


def test_entity_id_re_rejects_trailing_newline():
    assert ENTITY_ID_RE.match("unit:7\n") is None
    assert ENTITY_ID_RE.match("city:8\n") is None
    assert ENTITY_ID_RE.match("unit:7") is not None


# ---------------------------------------------------------------------------
# FW-R2: the CommitmentRecord carries the union of every attempt's
# memory_update (a repair replaces the executed batch, never an earlier
# attempt's commitments/threats)
# ---------------------------------------------------------------------------


def test_runner_two_attempt_turn_unions_memory_updates(tmp_path):
    """A repaired turn's CommitmentRecord carries both attempts'
    memory_update commitments."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        # Attempt 1 emits a commitment, then an unoffered ref -> repair.
        ok_response(["unit:99:goto_n"], turn_id="t1",
                    commitments=[("research", "Alphabet", 3)]),
        # The repair emits a different commitment.
        ok_response(["unit:10:goto_e"], turn_id="t1",
                    commitments=[("expansion", "city:2", 5)]),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target) for c in records[0].commitments] == [
        ("research", "Alphabet"), ("expansion", "city:2")]


def test_runner_repair_budget_skip_still_ledgers_attempt_memory_update(
        tmp_path):
    """FW-R2: the repair-budget-skip path still ledgers attempt 1's
    memory_update — the fabricated degraded result never hides it."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 165 s of the 180 s ACTING budget: remaining
        # 15 s sits below the 20 s repair floor -> the repair is skipped.
        advancing_response(
            lambda: ok_response(["unit:99:goto_n"], turn_id="t1",
                                commitments=[("research", "Alphabet", 3)]),
            clock, seconds=165.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target) for c in records[0].commitments] == [
        ("research", "Alphabet")]


# ---------------------------------------------------------------------------
# FW-16: batch-level ActionBatch.commitments are ingested alongside the
# memory_update ones; an engine-automatic resolution that fulfills a
# commitment credits fulfilled_other on the commitment board
# ---------------------------------------------------------------------------


def test_runner_batch_commitments_ingested_with_memory_update_ones(tmp_path):
    """FW-16: the batch-level ActionBatch.commitments reach the turn's
    CommitmentRecord alongside the memory_update ones — the union of both
    sources, identical commitments deduplicated."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    shared = {"domain": "research", "target": "Alphabet", "deadline_turn": 3}
    batch = ActionBatch(
        turn_id="t1",
        actions=[{"ref": "unit:10:goto_e"}],
        commitments=[shared,
                     {"domain": "expansion", "target": "city:2",
                      "deadline_turn": 5}],
        finish_requested=True,
        memory_update=MemoryUpdate(
            commitments=[shared,
                         {"domain": "military", "target": "unit:20",
                          "deadline_turn": 4}]),
    )
    client = FakeModelClient([
        ModelResponse(call_id="fake-call", prompt_hash="ph", batch=batch,
                      latency_ms=1, tokens_in=10, tokens_out=10,
                      cost_usd=0.0, fingerprint="fp"),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target, c.deadline_turn)
            for c in records[0].commitments] == [
        ("research", "Alphabet", 3),  # in both sources: deduplicated
        ("military", "unit:20", 4),   # memory_update only
        ("expansion", "city:2", 5),   # batch-level only
    ]


def test_runner_engine_automatic_resolution_credits_fulfilled_other(tmp_path):
    """FW-16: an engine-automatic resolution (readback "none" by
    construction) that fulfills a batch-level commitment produces a
    fulfilled_other — never fulfilled_model — credit on the commitment
    board, end to end."""
    # Turn 1: researching Alphabet, the model moves a unit and commits (at
    # batch level) to the Pottery research. Turn 2: the engine continued
    # research to Pottery (tech 2) during WAITING — no executed tech action
    # of ours -> engine_automatic, which fulfills the commitment.
    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False,
                    commitments=[("research", "Pottery", 2)]),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    # The batch-level commitment reached the turn's CommitmentRecord.
    commitments = [r for r in records if isinstance(r, CommitmentRecord)]
    assert len(commitments) == 1
    assert [(c.domain, c.target) for c in commitments[0].commitments] == [
        ("research", "Pottery")]
    # The engine-automatic research resolution fulfilled it: a
    # fulfilled_other credit on the board, never fulfilled_model.
    board = commitment_board(records)
    entry = next(e for e in board if e["target"] == "Pottery")
    assert entry["status"] == "fulfilled_other"
    assert entry["fulfilled_by"] == ["engine_automatic:research:Pottery"]
    assert entry["fulfilled_model_by"] == []
