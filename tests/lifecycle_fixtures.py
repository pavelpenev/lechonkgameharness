# SPDX-License-Identifier: MIT
"""Lifecycle test fixtures (WP3 package B): a FakeSession subclass with
scripted lifecycle fault injection, driving the §3 line 35
uncertain-mutation reconciliation and the §3 teardown bound offline.

Injection seams (all deterministic, all through the real runner):

- mid-mutation disconnect: a scripted action's step raises SessionError
  AFTER dispatch — the read-back never arrives. ``effect_applied``
  controls whether the server applied the mutation before the connection
  dropped (the session's CACHED view stays stale either way, exactly
  like the real AdapterSession: the cache advances only on a successful
  RPC; the runner's reconciliation read (refresh) is what reveals the
  server-side truth). ``disconnect_count`` fires the injection more than
  once (a resubmission that disconnects again). ``disconnect_timeout``
  fires the per-mutation DEADLINE instead (SessionTimeoutError): the
  session is then marked failed — the real AdapterSession refuses every
  later RPC — so the reconciliation read is refused and the mutation
  quarantines as unknown.
- permanent disconnect: ``refresh_fails`` makes the reconnect read raise
  (the reconciliation cannot settle the mutation).
- disconnect at the finish primitive: ``disconnect_finish`` makes the
  first step(None) raise (the connection dropped as the pipeline wrapped
  up the turn — NOT a mutation: no reconciliation, the plain bounded
  infrastructure fail-stop).
- blocking close: ``block_close`` makes close() block past
  ``close_deadline`` and return the skip-path status — mirroring
  AdapterSession.close()'s guarded RPC: the upstream
  CivController.close() game-over wait overruns the §3 teardown
  deadline and the skip path applies (never an unbounded wait).
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Optional, Sequence

from lechonk.adapter.session import SessionError, SessionStateError, SessionTimeoutError

from tests.runner_fixtures import FakeSession


class LifecycleSession(FakeSession):
    """FakeSession with lifecycle fault injection (see module docstring)."""

    def __init__(self, worlds: Sequence[tuple[dict, dict]], *,
                 disconnect_on: Any = None,
                 effect_applied: bool = True,
                 disconnect_count: int = 1,
                 disconnect_timeout: bool = False,
                 refresh_fails: bool = False,
                 disconnect_finish: bool = False,
                 block_close: bool = False,
                 close_deadline: float = 30.0):
        super().__init__(worlds)
        self._disconnect_on = disconnect_on
        self._effect_applied = effect_applied
        self._disconnect_remaining = max(0, int(disconnect_count))
        self._disconnect_timeout = disconnect_timeout
        self._refresh_fails = refresh_fails
        self._disconnect_finish = disconnect_finish
        self._finish_disconnected = False
        self._block_close = block_close
        self._close_deadline = float(close_deadline)
        # The real AdapterSession marks itself failed on an operation
        # timeout (no blind retry, no silent continuation): every later
        # RPC is refused. Modelled here for the timeout injection.
        self._failed_msg: Optional[str] = None
        # The cached view: the session's contract is that obs/info expose
        # the latest COMPLETED operation's data (the real AdapterSession
        # advances the cache only on a successful RPC).
        self._cache_obs = copy.deepcopy(self._worlds[0][0])
        self._cache_info = copy.deepcopy(self._worlds[0][1])
        self._close_release = threading.Event()
        self.close_entered = threading.Event()
        self.disconnects: list = []
        self.refresh_calls = 0
        self.close_calls = 0
        self.close_blocked_s: Optional[float] = None

    # -- the cached view ----------------------------------------------------

    @property
    def obs(self):
        return self._cache_obs

    @property
    def info(self):
        return self._cache_info

    def _advance_cache(self, observation, info) -> None:
        self._cache_obs = observation
        self._cache_info = info

    # -- fault injection ----------------------------------------------------

    @property
    def failed(self):
        return self._failed_msg

    def _refuse_if_failed(self, kind: str) -> None:
        # Mirrors AdapterSession._rpc's guard: a failed session refuses
        # every operation.
        if self._failed_msg is not None:
            raise SessionStateError(
                f"session is failed ({self._failed_msg}); refusing {kind}")

    def _disconnect_matches(self, action) -> bool:
        if self._disconnect_remaining <= 0 or action is None:
            return False
        if self._disconnect_on == action:
            return True
        # Match by action key (the raw tuple's third element) for
        # convenience: disconnect_on="goto_4" matches ("unit", 10, "goto_4").
        return (isinstance(action, tuple) and len(action) == 3
                and action[2] == self._disconnect_on)

    def step(self, action, *, deadline=None):
        self._refuse_if_failed("step")
        if action is None and self._disconnect_finish \
                and not self._finish_disconnected:
            # The connection dropped at the pipeline's finish primitive
            # (not a mutation — no reconciliation applies).
            self._finish_disconnected = True
            self.calls.append((action, deadline))
            self.disconnects.append(action)
            raise SessionError(
                "injected disconnect at the finish primitive (the phase_done "
                "send's response never arrived)")
        if self._disconnect_matches(action):
            self._disconnect_remaining -= 1
            self.calls.append((action, deadline))
            obs, info = self._worlds[self._index]
            if self._effect_applied:
                # The server applied the mutation before the connection
                # dropped; the session's cached view stays stale (the
                # read-back never arrived).
                new_obs = copy.deepcopy(obs)
                self._apply_effect(new_obs, action)
                self._worlds[self._index] = (new_obs, info)
            self.disconnects.append(action)
            if self._disconnect_timeout:
                # The per-mutation send+read-back deadline (§3 line 34)
                # fired: the session is marked failed (the real
                # AdapterSession refuses every later RPC).
                self._failed_msg = ("step exceeded its deadline of "
                                    "30.0s (injected mutation timeout)")
                raise SessionTimeoutError("step", 30.0)
            raise SessionError(
                f"injected disconnect mid-mutation (action {action!r} was "
                f"dispatched; the read-back never arrived)")
        outcome = super().step(action, deadline=deadline)
        self._advance_cache(outcome.observation, outcome.info)
        return outcome

    def refresh(self):
        self._refuse_if_failed("refresh")
        self.refresh_calls += 1
        if self._refresh_fails:
            raise SessionError(
                "injected permanent disconnect: the reconnect read failed")
        # The reconnect read returns the SERVER-side truth (the world),
        # not the stale cache — and advances the cache, like the real
        # AdapterSession.refresh().
        obs, info = self._worlds[self._index]
        snap = {"info": info, "observation": obs}
        self._advance_cache(obs, info)
        return snap

    def close(self) -> str:
        self.close_calls += 1
        if self._block_close and not self._close_release.is_set():
            # Mirrors AdapterSession.close()'s guarded RPC: the upstream
            # CivController.close() game-over wait blocks; the §3 teardown
            # deadline fires and the skip path applies.
            self.close_entered.set()
            t0 = time.monotonic()
            released = self._close_release.wait(timeout=self._close_deadline)
            self.close_blocked_s = time.monotonic() - t0
            self._closed = True
            self._close_status = ("completed" if released
                                  else "skipped_after_deadline")
            return self._close_status
        return super().close()
