# SPDX-License-Identifier: MIT
"""AdapterSession tests (FW-14a): the game-over flag plumbing.

The real AdapterSession caches the latest step's game-over state — the
terminated/truncated flags from the env's step tuple plus the env's
defeat signal (``my_player_is_defeated``), which is NOT in the step
tuple (vendor: civ_controller.py:303, a read-only CivController property
reached via ``env.civ_controller``). The runner reads the cached flags
to classify an empty game-over observation (DEFEATED / VICTORY /
TRUNCATED) instead of raising into the generic infrastructure handler.

The session is exercised without a live env: ``step()``'s RPC is
stubbed, and ``_dispatch`` runs against a fake env/controller (the
worker-thread boundary is not under test here).
"""

from __future__ import annotations

from typing import Any, Optional

from lechonk.adapter.session import AdapterSession, StepOutcome


def _outcome(*, terminated: bool = False, truncated: bool = False,
             defeated: Optional[bool] = None,
             obs: Optional[dict] = None, info: Optional[dict] = None
             ) -> StepOutcome:
    return StepOutcome(
        observation=obs if obs is not None else {"player": {}},
        reward=0.0, terminated=terminated, truncated=truncated,
        info=info if info is not None else {"turn": 3,
                                            "available_actions": {}},
        action=("unit", 10, "goto_4"), elapsed_s=0.01,
        my_player_is_defeated=defeated)


def _session() -> AdapterSession:
    return AdapterSession(workdir="/tmp/lechonk-test-session", seed=42)


class _FakeController:
    """A duck-typed CivController exposing the defeat-signal property."""

    def __init__(self, defeated: Any = False):
        self._defeated = defeated
        self.calls = 0

    def my_player_is_defeated(self) -> bool:
        self.calls += 1
        if isinstance(self._defeated, BaseException):
            raise self._defeated
        return bool(self._defeated)


class _FakeEnv:
    """A duck-typed FreecivBaseEnv: a fixed step tuple + controller."""

    def __init__(self, *, terminated: bool, truncated: bool,
                 controller: _FakeController):
        self._terminated = terminated
        self._truncated = truncated
        self.civ_controller = controller

    def step(self, action):
        return ({"player": {}}, 0.0, self._terminated, self._truncated,
                {"turn": 3, "available_actions": {}})


# ---------------------------------------------------------------------------
# step(): the cached flags are exposed
# ---------------------------------------------------------------------------


def test_step_caches_and_exposes_the_game_over_flags():
    """step() advances the cached obs/info/turn AND the game-over flags
    (terminated/truncated + the defeat signal) so the runner can
    classify an empty game-over observation at the next decision
    point."""
    session = _session()
    assert session.terminated is False
    assert session.truncated is False
    assert session.my_player_is_defeated is None

    session._rpc = lambda kind, payload, timeout: _outcome(  # type: ignore[method-assign]
        terminated=True, defeated=True)
    outcome = session.step(("unit", 10, "goto_4"))

    assert outcome.terminated is True
    assert session.terminated is True
    assert session.truncated is False
    assert session.my_player_is_defeated is True
    assert session.obs == {"player": {}}
    assert session.turn == 3


def test_step_advances_and_clears_the_flags_per_step():
    """The cache describes the LATEST step: a game-over step's flags are
    replaced by the next step's (a mid-game step carries no defeat
    signal — None, never a stale True)."""
    session = _session()
    outcomes = [
        _outcome(terminated=True, defeated=True),
        _outcome(terminated=False, defeated=None),
    ]
    session._rpc = lambda kind, payload, timeout: outcomes.pop(0)  # type: ignore[method-assign]

    session.step(("unit", 10, "goto_4"))
    assert (session.terminated, session.truncated,
            session.my_player_is_defeated) == (True, False, True)

    session.step(("unit", 10, "goto_4"))
    assert (session.terminated, session.truncated,
            session.my_player_is_defeated) == (False, False, None)


def test_step_tolerates_an_outcome_without_the_defeat_field():
    """Backward compatibility: a duck-typed StepOutcome that predates
    the defeat-signal field caches None (unknown), never a fabricated
    signal."""
    session = _session()
    legacy = StepOutcome(
        observation={"player": {}}, reward=0.0, terminated=True,
        truncated=False, info={"turn": 4, "available_actions": {}},
        action=None, elapsed_s=0.01)
    assert legacy.my_player_is_defeated is None
    session._rpc = lambda kind, payload, timeout: legacy  # type: ignore[method-assign]
    session.step(None)
    assert session.terminated is True
    assert session.my_player_is_defeated is None


# ---------------------------------------------------------------------------
# _dispatch(): the defeat signal is read from the controller at the
# game-over boundary only
# ---------------------------------------------------------------------------


def test_dispatch_reads_the_defeat_signal_at_game_over():
    """The defeat signal is NOT in the env's step tuple: at a game-over
    step the worker reads it once from env.civ_controller (vendor:
    civ_controller.py:303) and carries it on the StepOutcome."""
    session = _session()
    controller = _FakeController(defeated=True)
    session._env = _FakeEnv(terminated=True, truncated=False,
                            controller=controller)

    outcome = session._dispatch("step", ("unit", 10, "goto_4"))

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.my_player_is_defeated is True
    assert controller.calls == 1


def test_dispatch_reads_the_defeat_signal_on_truncation():
    """A truncation step is also a game-over boundary for the signal
    read (the runner distinguishes TRUNCATED from DEFEATED by it)."""
    session = _session()
    controller = _FakeController(defeated=True)
    session._env = _FakeEnv(terminated=False, truncated=True,
                            controller=controller)

    outcome = session._dispatch("step", ("unit", 10, "goto_4"))

    assert outcome.truncated is True
    assert outcome.my_player_is_defeated is True
    assert controller.calls == 1


def test_dispatch_skips_the_defeat_read_mid_game():
    """Mid-game the controller is NOT consulted: the defeat signal is
    a game-over-boundary read, and the outcome carries None."""
    session = _session()
    controller = _FakeController(defeated=True)
    session._env = _FakeEnv(terminated=False, truncated=False,
                            controller=controller)

    outcome = session._dispatch("step", ("unit", 10, "goto_4"))

    assert outcome.terminated is False
    assert outcome.my_player_is_defeated is None
    assert controller.calls == 0


def test_dispatch_defeat_read_failure_is_unknown():
    """A failed signal read is None (unknown) — never a fabricated
    signal, never an exception escaping the step."""
    session = _session()
    controller = _FakeController(defeated=RuntimeError("client torn down"))
    session._env = _FakeEnv(terminated=True, truncated=False,
                            controller=controller)

    outcome = session._dispatch("step", ("unit", 10, "goto_4"))

    assert outcome.terminated is True
    assert outcome.my_player_is_defeated is None
    assert controller.calls == 1
