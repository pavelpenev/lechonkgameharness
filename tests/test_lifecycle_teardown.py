# SPDX-License-Identifier: MIT
"""WP3 lifecycle tests: disconnect-during-teardown acceptance (plan §4
line 56, §3 line 34) — offline over the fake session.

The upstream teardown hazards are wrapped with hard deadlines
(§3 line 34): CivController.close() waits on a game-over message in an
unbounded loop, and the save-deletion / score retrieval are synchronous
HTTP. The offline cases prove the composition:

1. a disconnect at the pipeline's finish primitive (the pipeline is
   wrapping up) → bounded TRUNCATED (infrastructure failure), and the
   teardown then blocks on the dead connection → the §3 teardown
   deadline fires (the skip path) → the run returns BOUNDED, no hang,
   no unbounded wait — with the transcript intact and flushable DURING
   the stuck teardown (the outcome record was fsynced BEFORE close);
2. the same teardown overrun on the happy path (the pipeline completed
   before the disconnect) → the outcome record is already durable, the
   close overruns the deadline → skip path → bounded return.
"""

from __future__ import annotations

import threading
import time

from lechonk.core.ledger import load_transcript
from lechonk.core.schemas import OutcomeRecord

from tests.lifecycle_fixtures import LifecycleSession
from tests.runner_fixtures import (
    FakeModelClient,
    make_runner,
    ok_response,
    turn_world,
)

GOTO_REF = "unit:10:goto_e"


def test_disconnect_at_finish_then_blocking_teardown_is_bounded(tmp_path):
    """Disconnect while the pipeline is tearing down: the finish
    primitive's response never arrives → bounded TRUNCATED; the upstream
    close() game-over wait then blocks → the §3 teardown deadline fires
    (skip path) → the run returns bounded with the transcript intact
    and flushable — proven by loading the transcript WHILE the teardown
    is still stuck (the outcome record was fsynced before close)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_finish=True,
                               block_close=True, close_deadline=2.0)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"teardown_deadline_s": 2.0})

    t0 = time.monotonic()
    holder: dict = {}

    def run_it():
        holder["result"] = runner.run()

    thread = threading.Thread(target=run_it)
    thread.start()
    # Teardown is stuck (the upstream game-over wait overruns): while it
    # is, the transcript must already be intact and flushable.
    assert session.close_entered.wait(timeout=15.0)
    mid_load = load_transcript(runner.ledger.path)
    assert not mid_load.discarded_trailing_partial
    assert isinstance(mid_load.records[-1], OutcomeRecord)
    assert mid_load.records[-1].outcome == "TRUNCATED"
    assert [r.seq for r in mid_load.records] == \
        list(range(len(mid_load.records)))
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    wall = time.monotonic() - t0
    result = holder["result"]

    # Bounded FAILED/TRUNCATED outcome (the disconnect, not a hang).
    assert result.outcome == "TRUNCATED"
    assert "infrastructure failure" in result.reason
    # The teardown overrun hit the supervisor deadline: the skip path
    # (never an unbounded wait).
    assert result.close_status == "skipped_after_deadline"
    assert session.close_calls == 1
    assert session.close_blocked_s is not None
    assert session.close_blocked_s >= 2.0
    assert wall < 15.0  # bounded: the whole run, teardown included
    # Artifacts intact after the run: the transcript loads clean.
    load = load_transcript(runner.ledger.path)
    assert not load.discarded_trailing_partial
    assert isinstance(load.records[-1], OutcomeRecord)
    assert load.records[-1].outcome == "TRUNCATED"
    assert [r.seq for r in load.records] == list(range(len(load.records)))
    assert session.closed


def test_teardown_overrun_after_completion_is_bounded(tmp_path):
    """The happy path with a dead teardown: the pipeline completes and
    the outcome record is durable BEFORE the teardown; the blocking
    close then overruns the §3 teardown deadline → the skip path → a
    bounded return, no hang, artifacts intact."""
    worlds = [turn_world(1), turn_world(2)]
    session = LifecycleSession(worlds, block_close=True, close_deadline=0.5)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={"teardown_deadline_s": 0.5})

    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    assert result.outcome == "COMPLETED"
    assert result.close_status == "skipped_after_deadline"
    assert session.close_calls == 1
    assert session.close_blocked_s is not None
    assert session.close_blocked_s >= 0.5
    assert wall < 10.0  # bounded: no hang, no unbounded wait
    # Artifacts intact: the outcome record was durable before teardown.
    load = load_transcript(runner.ledger.path)
    assert not load.discarded_trailing_partial
    assert isinstance(load.records[-1], OutcomeRecord)
    assert load.records[-1].outcome == "COMPLETED"
    assert [r.seq for r in load.records] == list(range(len(load.records)))
    assert session.closed
