"""Small, thread-safe progress heartbeat for long-running model calls.

The helper deliberately reports metadata only.  It does not retain prompts,
responses, messages, or reasoning traces; callers remain responsible for
persisting the actual model transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping


HeartbeatSink = Callable[[Mapping[str, Any]], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class ModelCall:
    """Identity of one logical model call (including a repair attempt)."""

    run_id: str
    turn: int | str
    call_id: str
    attempt: int


class ModelCallHeartbeat:
    """Emit bounded progress events while one model call is in flight.

    ``start`` replaces an existing call, returning a generation token.  The
    token can be supplied to ``note_stream_activity`` so a late callback from
    a replaced call cannot update the new call's age.  ``stop`` wakes and
    makes a bounded join attempt; the worker is daemonized so a faulty sink
    cannot keep process shutdown alive.
    Every event carries its generation. A sink callback already in flight may
    finish after replacement; downstream consumers should filter stale
    generations when that distinction matters.
    """

    def __init__(
        self,
        *,
        interval_s: float = 10.0,
        clock: Clock = time.monotonic,
        sink: HeartbeatSink,
        failure_sink: HeartbeatSink | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(sink):
            raise TypeError("sink must be callable")
        if failure_sink is not None and not callable(failure_sink):
            raise TypeError("failure_sink must be callable")

        self._interval_s = float(interval_s)
        self._clock = clock
        self._sink = sink
        self._lock = threading.Lock()
        # Sink calls are serialized with one another.  Lifecycle operations do
        # not acquire this lock: a slow sink must never make stop unbounded.
        # RLock is intentional because a sink may synchronously call start/stop.
        self._sink_lock = threading.RLock()
        self._failure_sink = failure_sink
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._call: ModelCall | None = None
        self._started_at: float | None = None
        self._last_stream_at: float | None = None
        self._generation = 0

    def start(
        self,
        *,
        run_id: str,
        turn: int | str,
        call_id: str,
        attempt: int,
    ) -> int:
        """Start a heartbeat and replace any currently active heartbeat."""

        # Publish the replacement immediately, then make only a bounded effort
        # to join the old worker.  A sink is expected to be fast and nonblocking
        # (it should enqueue/print and return); daemon workers are a final
        # safeguard when a caller violates that contract.
        with self._lock:
            old_thread = self._deactivate_locked()
            now = self._clock()
            call = ModelCall(run_id=run_id, turn=turn, call_id=call_id, attempt=attempt)
            stop_event = threading.Event()
            generation = self._generation
            self._call = call
            self._started_at = now
            self._last_stream_at = None
            self._stop_event = stop_event
            thread = threading.Thread(
                target=self._run,
                args=(generation, call, now, stop_event),
                name=f"model-heartbeat-{generation}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        self._join_bounded(old_thread)
        return generation

    def note_stream_activity(self, generation: int) -> bool:
        """Record a response delta for the active generation.

        Returns ``False`` for a stale token (or when no call is active), which
        lets streaming adapters safely ignore callbacks after replacement.
        """

        now = self._clock()
        with self._lock:
            if self._thread is None or self._stop_event is None:
                return False
            if generation != self._generation:
                return False
            self._last_stream_at = now
            return True

    def stop(self, generation: int | None = None) -> bool:
        """Stop the active worker and make a bounded join attempt.

        A supplied stale generation is ignored.  The join happens outside the
        lock so a sink or worker cannot deadlock lifecycle callers.
        """

        with self._lock:
            if self._thread is None or self._stop_event is None:
                return False
            if generation is not None and generation != self._generation:
                return False
            old_thread = self._deactivate_locked()
        self._join_bounded(old_thread)
        return True

    close = stop

    def active(self) -> bool:
        with self._lock:
            return self._thread is not None

    def _run(
        self,
        generation: int,
        call: ModelCall,
        started_at: float,
        stop_event: threading.Event,
    ) -> None:
        try:
            while not stop_event.wait(self._interval_s):
                now = self._clock()
                with self._lock:
                    # Replacement/stop invalidates this worker before it can
                    # publish an event, even if it was already waking up.
                    if generation != self._generation or stop_event is not self._stop_event:
                        return
                    last_stream_at = self._last_stream_at
                    elapsed_s = max(0.0, now - started_at)
                    event: dict[str, Any] = {
                        "kind": "model_call_heartbeat",
                        "run_id": call.run_id,
                        "turn": call.turn,
                        "call_id": call.call_id,
                        "attempt": call.attempt,
                        "generation": generation,
                        "elapsed_s": elapsed_s,
                    }
                    if last_stream_at is None:
                        event["stream_state"] = "no_delta_yet"
                    else:
                        event["last_stream_age_s"] = max(0.0, now - last_stream_at)
                # Serialize sink callbacks. Generation is checked before
                # entering the sink, but an in-flight callback may finish after
                # replacement; consumers can filter its event.
                with self._sink_lock:
                    with self._lock:
                        if generation != self._generation or stop_event is not self._stop_event:
                            return
                    try:
                        self._sink(event)
                    except Exception as exc:
                        failure = {
                            "kind": "model_call_heartbeat_failure",
                            "run_id": call.run_id,
                            "turn": call.turn,
                            "call_id": call.call_id,
                            "attempt": call.attempt,
                            "generation": generation,
                            "error_type": type(exc).__name__,
                        }
                        if self._failure_sink is not None:
                            try:
                                self._failure_sink(failure)
                            except Exception:
                                pass
                        return
        finally:
            with self._lock:
                if generation == self._generation and self._thread is threading.current_thread():
                    self._thread = None
                    self._stop_event = None
                    self._call = None
                    self._started_at = None
                    self._last_stream_at = None

    def _deactivate_locked(self) -> threading.Thread | None:
        """Invalidate and detach the current worker; caller holds ``_lock``."""
        self._generation += 1
        thread = self._thread
        if thread is None:
            return None
        self._stop_event.set()  # type: ignore[union-attr]
        self._thread = None
        self._stop_event = None
        self._call = None
        self._started_at = None
        self._last_stream_at = None
        return thread

    @staticmethod
    def _join_bounded(thread: threading.Thread | None) -> None:
        if thread is not None and thread is not threading.current_thread():
            # Never let lifecycle shutdown wait on an arbitrary sink.
            thread.join(timeout=0.1)
