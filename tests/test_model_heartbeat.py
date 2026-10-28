from __future__ import annotations

import threading
import time

from scripts.model_heartbeat import ModelCallHeartbeat


def _wait_for(events: list[dict], count: int, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while len(events) < count and time.monotonic() < deadline:
        time.sleep(0.002)
    assert len(events) >= count


def test_emits_no_delta_heartbeat_with_call_identity() -> None:
    events: list[dict] = []
    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=events.append)
    token = heartbeat.start(run_id="r1", turn=3, call_id="c1", attempt=2)
    try:
        _wait_for(events, 1)
    finally:
        heartbeat.stop(token)

    event = events[0]
    assert event["kind"] == "model_call_heartbeat"
    assert event["run_id"] == "r1"
    assert event["turn"] == 3
    assert event["call_id"] == "c1"
    assert event["attempt"] == 2
    assert event["stream_state"] == "no_delta_yet"
    assert "messages" not in event and "reasoning" not in event


def test_activity_reports_age_since_last_delta() -> None:
    events: list[dict] = []
    heartbeat = ModelCallHeartbeat(interval_s=0.02, sink=events.append)
    token = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    try:
        time.sleep(0.005)
        assert heartbeat.note_stream_activity(token)
        _wait_for(events, 1)
    finally:
        heartbeat.stop(token)

    assert events[0]["last_stream_age_s"] >= 0
    assert "stream_state" not in events[0]


def test_replacement_invalidates_stale_token_and_old_worker_reports() -> None:
    events: list[dict] = []
    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=events.append)
    first = heartbeat.start(run_id="r", turn=1, call_id="old", attempt=1)
    second = heartbeat.start(run_id="r", turn=1, call_id="new", attempt=1)
    try:
        assert first != second
        assert not heartbeat.note_stream_activity(first)
        _wait_for(events, 1)
    finally:
        heartbeat.stop(second)

    assert events
    assert all(event["call_id"] == "new" for event in events)


def test_stop_joins_daemon_worker_without_leaking_thread() -> None:
    events: list[dict] = []
    heartbeat = ModelCallHeartbeat(interval_s=60, sink=events.append)
    token = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    heartbeat.stop(token)
    assert not heartbeat.active()
    assert not any(thread.name.startswith("model-heartbeat-") for thread in threading.enumerate())


def test_stop_in_finally_handles_api_exception() -> None:
    events: list[dict] = []
    heartbeat = ModelCallHeartbeat(interval_s=60, sink=events.append)
    token = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    try:
        raise RuntimeError("simulated API failure")
    except RuntimeError:
        pass
    finally:
        heartbeat.stop(token)
    assert not heartbeat.active()


def test_reentrant_sink_can_stop_without_deadlock() -> None:
    events: list[dict] = []
    holder: dict[str, int] = {}

    def sink(event: dict) -> None:
        events.append(event)
        heartbeat.stop(holder["token"])

    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=sink)
    holder["token"] = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    _wait_for(events, 1)
    deadline = time.monotonic() + 1
    while heartbeat.active() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not heartbeat.active()


def test_concurrent_stop_waits_for_sink_then_joins_worker() -> None:
    entered = threading.Event()
    release = threading.Event()
    events: list[dict] = []

    def sink(event: dict) -> None:
        events.append(event)
        entered.set()
        release.wait(1)

    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=sink)
    token = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    assert entered.wait(1)
    stopped = threading.Event()
    stopper = threading.Thread(target=lambda: (heartbeat.stop(token), stopped.set()))
    stopper.start()
    time.sleep(0.02)
    assert not stopped.is_set()
    release.set()
    stopper.join(1)
    assert stopped.is_set()
    assert not heartbeat.active()


def test_sink_exception_is_reported_and_call_can_restart() -> None:
    events: list[dict] = []
    failures: list[dict] = []
    should_fail = True

    def sink(event: dict) -> None:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("sink unavailable")
        events.append(event)

    heartbeat = ModelCallHeartbeat(
        interval_s=0.01, sink=sink, failure_sink=failures.append
    )
    first = heartbeat.start(run_id="r", turn=1, call_id="first", attempt=1)
    deadline = time.monotonic() + 1
    while heartbeat.active() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not heartbeat.active()
    assert failures and failures[0]["kind"] == "model_call_heartbeat_failure"

    second = heartbeat.start(run_id="r", turn=1, call_id="second", attempt=1)
    try:
        assert second != first
        _wait_for(events, 1)
    finally:
        heartbeat.stop(second)
    assert events[0]["call_id"] == "second"


def test_stop_is_bounded_when_sink_blocks_and_worker_is_daemon() -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_ref: list[threading.Thread] = []

    def sink(_: dict) -> None:
        entered.set()
        release.wait(5)

    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=sink)
    token = heartbeat.start(run_id="r", turn=1, call_id="c", attempt=1)
    assert entered.wait(1)
    worker_ref.extend(
        thread for thread in threading.enumerate() if thread.name.startswith("model-heartbeat-")
    )
    started = time.monotonic()
    assert heartbeat.stop(token)
    assert time.monotonic() - started < 0.5
    assert not heartbeat.active()
    assert worker_ref and all(thread.daemon for thread in worker_ref)

    # Let the blocked daemon finish so this test does not retain a live thread
    # in the test process after asserting the bounded shutdown behavior.
    release.set()
    deadline = time.monotonic() + 1
    while any(thread.is_alive() for thread in worker_ref) and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not any(thread.is_alive() for thread in worker_ref)


def test_inflight_old_sink_event_is_tagged_stale_after_replacement() -> None:
    entered = threading.Event()
    release = threading.Event()
    events: list[dict] = []

    def sink(event: dict) -> None:
        if not events:
            entered.set()
            release.wait(1)
        events.append(event)

    heartbeat = ModelCallHeartbeat(interval_s=0.01, sink=sink)
    old_token = heartbeat.start(run_id="r", turn=1, call_id="old", attempt=1)
    assert entered.wait(1)
    new_token = heartbeat.start(run_id="r", turn=1, call_id="new", attempt=1)
    assert new_token != old_token
    assert not heartbeat.note_stream_activity(old_token)
    release.set()
    _wait_for(events, 2)
    try:
        assert events[0]["generation"] == old_token
        assert events[1]["generation"] == new_token
    finally:
        heartbeat.stop(new_token)
