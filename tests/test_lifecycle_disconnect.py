# SPDX-License-Identifier: MIT
"""WP3 lifecycle tests: the §3 line 35 uncertain-mutation reconciliation
(mid-mutation disconnect), offline over the fake session with the REAL
adapter read-back verifiers.

The four cases of the reconciliation contract:

1. transient disconnect + verified NON-execution on reconnect → the
   mutation is resent exactly ONCE, as continued_model_order linked to
   the uncertain model attempt; applied exactly once (no duplicate);
2. the mutation landed server-side but the read-back was lost →
   reconciliation proves EXECUTION → quarantined, NEVER resent, bounded
   fail-stop TRUNCATED with artifacts;
3. the reconnect read also fails → UNKNOWN → quarantined, never resent;
4. the one permitted resubmission disconnects again → quarantined,
   never a third send.

Plus the per-mutation deadline case: a send+read-back timeout marks the
session failed (the real AdapterSession refuses every later RPC) → the
reconciliation read is refused → unknown → quarantine.

And the FW-R1 narrow window: the step COMPLETES server-side (the cached
obs advances to the post-mutation state) and execute_candidate raises
afterwards → reconciliation compares against the pre-mutation baseline
captured before dispatch; the executed mutation reconciles to "executed"
and is NEVER resent (no duplicate mutation).
"""

from __future__ import annotations

from lechonk.adapter.session import SessionError
from lechonk.core.ledger import load_transcript
from lechonk.core.schemas import (
    ActionResultRecord,
    MetricEventRecord,
    OutcomeRecord,
)

from tests.lifecycle_fixtures import LifecycleSession
from tests.runner_fixtures import (
    FakeModelClient,
    make_runner,
    ok_response,
    turn_world,
)

GOTO_REF = "unit:10:goto_e"
# The raw CivRealm action tuple for the goto_e candidate (action key
# goto_4 = dir8 E — the same table the read-back verifier uses).
GOTO_ACTION = ("unit", 10, "goto_4")


class _ScriptedDefaultClient(FakeModelClient):
    """A comparator-arm client: declares the scripted provenance class
    (the runner reads the client's declared provenance_class)."""

    provenance_class = "scripted_default"


def _records(path):
    return load_transcript(path).records


def _actions(records):
    return [r for r in records if isinstance(r, ActionResultRecord)]


def _events(records, name):
    return [r for r in records
            if isinstance(r, MetricEventRecord) and r.name == name]


def _goto_sends(session):
    return [c for c in session.calls if c[0] == GOTO_ACTION]


# ---------------------------------------------------------------------------
# case 1: verified non-execution → the ONE permitted resubmission
# ---------------------------------------------------------------------------


def test_disconnect_verified_non_execution_resends_once(tmp_path):
    """The transient-disconnect case: the mutation's read-back is
    inconclusive (the connection dropped mid-verification), the reconnect
    read proves NON-execution (contradicted postconditions), and the
    pipeline resubmits exactly ONCE as continued_model_order linked to
    the uncertain model attempt. The mutation is applied exactly once
    (no duplicate), the run continues to the horizon."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "COMPLETED"
    # The disconnect fired once (the send); the resubmission went through.
    assert session.disconnects == [GOTO_ACTION]
    assert len(_goto_sends(session)) == 2  # the failed send + the resend
    # The mutation was APPLIED exactly once: world 0's unit 10 moved
    # (1,1) -> (2,1); a duplicate application would land at (3,1).
    unit = session._worlds[0][0]["unit"][10]
    assert (unit["x"], unit["y"]) == (2, 1)

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 2
    # The uncertain attempt: executed (the mutation was sent), the
    # read-back inconclusive, the execution state unknown.
    uncertain = by_ref[0]
    assert uncertain.outcome == "executed"
    assert uncertain.readback == "inconclusive"
    assert uncertain.provenance_class == "model"
    assert "execution state is unknown" in uncertain.detail
    # The reconciliation decision is ledgered (the non-execution proof).
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: non_execution_verified" in reconciliations[0].value
    # The resubmission: continued_model_order, origin-linked to the
    # uncertain model attempt, read-back verified.
    resend = by_ref[1]
    assert resend.outcome == "executed"
    assert resend.readback == "verified"
    assert resend.provenance_class == "continued_model_order"
    assert resend.origin_seq == uncertain.seq
    assert "verified non-execution" in resend.reason
    # No quarantine; the outcome record closed the run at the horizon.
    assert _events(records, "mutation_quarantine") == []
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "COMPLETED"


# ---------------------------------------------------------------------------
# case 2: executed server-side → quarantine, never resent
# ---------------------------------------------------------------------------


def test_disconnect_executed_server_side_quarantines(tmp_path):
    """The mutation landed server-side but the read-back was lost: the
    reconnect read proves EXECUTION — the mutation is NEVER resent
    (quarantine → bounded fail-stop TRUNCATED with artifacts; never blind
    replay; no duplicate mutation)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    assert "never resent" in result.reason
    # The mutation was sent exactly ONCE — never resent (no duplicate).
    assert len(_goto_sends(session)) == 1
    assert session.disconnects == [GOTO_ACTION]
    # The server-side effect is real (world 0's unit 10 moved once) but
    # the pipeline never re-sent: the transcript's only executed record
    # for the ref is the uncertain attempt.
    unit = session._worlds[0][0]["unit"][10]
    assert (unit["x"], unit["y"]) == (2, 1)

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 1
    assert by_ref[0].outcome == "executed"
    assert by_ref[0].readback == "inconclusive"
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: executed (" in reconciliations[0].value
    quarantines = _events(records, "mutation_quarantine")
    assert len(quarantines) == 1
    assert GOTO_REF in quarantines[0].value
    # Bounded fail-stop with artifacts: the outcome record is ledgered.
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "quarantined" in records[-1].reason
    assert session.closed


# ---------------------------------------------------------------------------
# case 3: the reconnect read also fails → unknown → quarantine
# ---------------------------------------------------------------------------


def test_disconnect_reconnect_fails_quarantines(tmp_path):
    """The permanent-disconnect case: the reconciliation read (the
    reconnect) also fails → the mutation's state is UNKNOWN →
    quarantined, never resent, bounded fail-stop with artifacts."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=False, refresh_fails=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    assert len(_goto_sends(session)) == 1  # never resent
    # The mutation never landed server-side.
    unit = session._worlds[0][0]["unit"][10]
    assert (unit["x"], unit["y"]) == (1, 1)

    records = _records(runner.ledger.path)
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: unknown (" in reconciliations[0].value
    assert "reconnect read failed" in reconciliations[0].value
    assert len(_events(records, "mutation_quarantine")) == 1
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


# ---------------------------------------------------------------------------
# provenance: a scripted arm's resubmission mirrors its origin's class
# ---------------------------------------------------------------------------


def test_resubmission_scripted_arm_mirrors_origin_provenance(tmp_path):
    """A comparator-arm (scripted_default) resubmission after verified
    non-execution mirrors its origin's provenance class with the same
    origin linkage — the model-only continued_model_order label is never
    applied to a scripted arm."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=False)
    client = _ScriptedDefaultClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 2
    assert by_ref[0].outcome == "executed"
    assert by_ref[0].readback == "inconclusive"
    assert by_ref[0].provenance_class == "scripted_default"
    # The resubmission mirrors the origin's provenance class (NOT the
    # model-only continued_model_order) with the same origin linkage.
    resend = by_ref[1]
    assert resend.outcome == "executed"
    assert resend.readback == "verified"
    assert resend.provenance_class == "scripted_default"
    assert resend.origin_seq == by_ref[0].seq
    assert "verified non-execution" in resend.reason
    assert _events(records, "mutation_quarantine") == []


# ---------------------------------------------------------------------------
# case 4: the resubmission disconnects again → never a third send
# ---------------------------------------------------------------------------


def test_resubmission_disconnect_quarantines_never_a_third_send(tmp_path):
    """The ONE permitted resubmission is itself interrupted by a
    disconnect: quarantined — the resubmission is never repeated (never a
    third send, never blind replay)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=False, disconnect_count=2)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    # Exactly two sends: the failed dispatch + the failed resubmission —
    # never a third.
    assert len(_goto_sends(session)) == 2
    assert session.disconnects == [GOTO_ACTION, GOTO_ACTION]
    # The mutation never landed server-side (both sends dropped before
    # the server applied anything).
    unit = session._worlds[0][0]["unit"][10]
    assert (unit["x"], unit["y"]) == (1, 1)

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    # The uncertain attempt + the uncertain resubmission (both sent,
    # both inconclusive) — and nothing else.
    assert len(by_ref) == 2
    assert all(a.outcome == "executed" for a in by_ref)
    assert all(a.readback == "inconclusive" for a in by_ref)
    assert by_ref[1].provenance_class == "continued_model_order"
    assert by_ref[1].origin_seq == by_ref[0].seq
    # The first reconciliation proved non-execution (permitting the one
    # resubmission); the resubmission's failure quarantined.
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: non_execution_verified" in reconciliations[0].value
    quarantines = _events(records, "mutation_quarantine")
    assert len(quarantines) == 1
    assert "resubmission also failed" in quarantines[0].value
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


# ---------------------------------------------------------------------------
# the per-mutation deadline: timeout → session failed → unknown
# ---------------------------------------------------------------------------


def test_mutation_deadline_timeout_quarantines(tmp_path):
    """The per-mutation send+read-back deadline (§3 line 34) exceeded:
    the session is marked failed (no blind retry, no silent
    continuation — the real AdapterSession refuses every later RPC), so
    the reconciliation read is refused → the mutation's state is UNKNOWN
    → quarantine → bounded fail-stop. Never resent."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               disconnect_timeout=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    assert len(_goto_sends(session)) == 1  # never resent
    assert session.failed is not None  # the session marked itself failed

    records = _records(runner.ledger.path)
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: unknown (" in reconciliations[0].value
    assert len(_events(records, "mutation_quarantine")) == 1
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


# ---------------------------------------------------------------------------
# case 5 (FW-R1): the narrow window — the step completes, the execute
# raises afterwards → reconciliation against the TRUE pre-mutation state
# ---------------------------------------------------------------------------


class _PostExecuteFailSession(LifecycleSession):
    """The FW-R1 narrow window: the mutation's step SUCCEEDS server-side
    (the cached obs/info advance to the post-mutation state, exactly like
    the real AdapterSession on a completed step) and then
    execute_candidate raises afterwards — injected as a ONE-SHOT
    production_name_map failure inside the read-back verification
    (execute.py raises after the successful step). The reconciliation's
    own production_name_map call (the reconnect) succeeds, so the
    comparison runs for real."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self._pnm_failures = 1

    def production_name_map(self):
        if self._pnm_failures > 0:
            self._pnm_failures -= 1
            raise SessionError(
                "injected post-step failure: the read-back verification "
                "raised after the step completed")
        return super().production_name_map()


def test_step_completed_execute_raised_after_reconciles_true_pre_state(
        tmp_path):
    """The narrow FW-R1 window: the step completes server-side (unit 10
    moved; the session's cached obs is POST-mutation) and
    execute_candidate raises afterwards. Reconciliation must compare
    against the PRE-mutation baseline captured before dispatch — the
    executed mutation reconciles to "executed" and is NEVER resent (a
    stale post-mutation baseline would reconcile non-execution and
    duplicate the move)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = _PostExecuteFailSession(worlds)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    # The mutation was sent exactly ONCE: reconciliation proved EXECUTION
    # against the true pre-state (unit at (1,1) before the send vs (2,1)
    # on reconnect) and quarantined — never resent (no duplicate move to
    # (3,1)).
    assert len(_goto_sends(session)) == 1
    unit = session._worlds[0][0]["unit"][10]
    assert (unit["x"], unit["y"]) == (2, 1)

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 1
    assert by_ref[0].outcome == "executed"
    assert by_ref[0].readback == "inconclusive"
    reconciliations = _events(records, "mutation_reconciliation")
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: executed (" in reconciliations[0].value
    quarantines = _events(records, "mutation_quarantine")
    assert len(quarantines) == 1
    assert GOTO_REF in quarantines[0].value
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "quarantined" in records[-1].reason
