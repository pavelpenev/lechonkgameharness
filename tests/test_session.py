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
import asyncio
import math
import queue

from lechonk.adapter.session import (
    AdapterSession,
    SessionStateError,
    StepOutcome,
    _collect_ruleset_reference,
    _collect_help_reference,
    _collect_classic_reference,
    _collect_manual_reference,
)


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


class _FakeRulesetEnv:
    def __init__(self, rule_ctrl=None):
        self.civ_controller = type("Controller", (), {"rule_ctrl": rule_ctrl})()


def test_classic_reference_is_not_claimed_for_nonclassic_or_unknown_ruleset():
    reference, digest, status, provenance = _collect_classic_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "civ2civ3"}})()))
    assert reference is None and digest is None and provenance is None
    assert status == "not_active: active ruleset is civ2civ3"
    reference, digest, status, provenance = _collect_classic_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {})()))
    assert reference is None and digest is None and provenance is None
    assert status.startswith("unavailable: active ruleset identity")


def test_classic_reference_loader_is_optional_and_exposes_metadata(monkeypatch):
    import types

    reference = types.SimpleNamespace(
        status="available", content_hash="sha-classic",
        provenance={"source": "classic/*.ruleset"},
        to_dict=lambda: {"ruleset_name": "classic", "entries": {}})
    fake_module = types.SimpleNamespace(load_classic_reference=lambda path: reference)
    monkeypatch.setitem(__import__("sys").modules,
                        "lechonk.adapter.classic_reference", fake_module)
    captured, digest, status, provenance = _collect_classic_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "classic"}})()))
    assert captured is reference
    assert digest == "sha-classic"
    assert status == "available"
    assert provenance == {"source": "classic/*.ruleset"}


def test_classic_reference_accepts_freeciv_display_suffix(monkeypatch):
    import types

    reference = types.SimpleNamespace(status="available", content_hash="h",
                                      provenance={})
    fake_module = types.SimpleNamespace(
        load_classic_reference=lambda path: reference)
    monkeypatch.setitem(__import__("sys").modules,
                        "lechonk.adapter.classic_reference", fake_module)
    captured, digest, status, _ = _collect_classic_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "Classic ruleset"}})()))
    assert captured is reference
    assert (digest, status) == ("h", "available")


def test_classic_reference_properties_are_safe_before_start(tmp_path):
    session = AdapterSession(workdir=str(tmp_path), seed=1)
    assert session.classic_reference is None
    assert session.classic_reference_json is None
    assert session.classic_reference_hash is None
    assert session.classic_reference_status == "not_collected"
    assert session.classic_reference_provenance is None


def test_classic_reference_json_serializes_index_entries(tmp_path):
    from lechonk.adapter.classic_reference import ClassicEntry, ClassicReference

    session = AdapterSession(workdir=str(tmp_path), seed=1)
    session._classic_reference = ClassicReference(
        entries=(ClassicEntry(key="warrior", name="Warrior"),))
    payload = session.classic_reference_json
    assert payload["entries"][0]["name"] == "Warrior"
    assert payload["content_hash"] == session.classic_reference.content_hash


def test_manual_reference_is_not_claimed_for_nonclassic_or_unknown_ruleset():
    reference, digest, status, provenance = _collect_manual_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "civ2civ3"}})()))
    assert reference is None and digest is None and provenance is None
    assert status == "not_active: active ruleset is civ2civ3"
    reference, digest, status, provenance = _collect_manual_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {})()))
    assert reference is None and digest is None and provenance is None
    assert status.startswith("unavailable: active ruleset identity")


def test_manual_reference_loader_receives_fixed_artifact_and_exposes_metadata(monkeypatch):
    import sys
    import types

    captured_paths = []
    reference = types.SimpleNamespace(
        status="available", content_hash="sha-manual",
        provenance={"generated_by": "freeciv-manual"},
        to_dict=lambda: {"schema_version": 1, "entries": []})

    def load(path):
        captured_paths.append(path)
        return reference

    monkeypatch.setitem(sys.modules, "lechonk.adapter.manual_reference",
                        types.SimpleNamespace(load_manual_reference=load))
    captured, digest, status, provenance = _collect_manual_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "Classic ruleset"}})()))
    assert captured is reference
    assert digest == "sha-manual"
    assert status == "available"
    assert provenance == {"generated_by": "freeciv-manual"}
    assert captured_paths and str(captured_paths[0]).endswith(
        "artifacts/reference/freeciv-manual")


def test_manual_reference_properties_are_safe_before_start(tmp_path):
    session = AdapterSession(workdir=str(tmp_path), seed=1)
    assert session.manual_reference is None
    assert session.manual_reference_json is None
    assert session.manual_reference_hash is None
    assert session.manual_reference_status == "not_collected"
    assert session.manual_reference_provenance is None


def test_manual_reference_json_serializes_loader_value(tmp_path):
    import types

    session = AdapterSession(workdir=str(tmp_path), seed=1)
    session._manual_reference = types.SimpleNamespace(
        to_dict=lambda: {"schema_version": 1, "entries": [{"title": "Move"}]})
    payload = session.manual_reference_json
    assert payload == {"schema_version": 1, "entries": [{"title": "Move"}]}


def test_manual_reference_missing_artifact_is_nonfatal(monkeypatch):
    import sys
    import types

    def load(_path):
        raise FileNotFoundError("manual manifest is absent")

    monkeypatch.setitem(sys.modules, "lechonk.adapter.manual_reference",
                        types.SimpleNamespace(load_manual_reference=load))
    reference, digest, status, provenance = _collect_manual_reference(
        _FakeRulesetEnv(rule_ctrl=type("Rules", (), {
            "ruleset_control": {"name": "classic"}})()))
    assert reference is None and digest is None and provenance is None
    assert status == "unavailable: manual manifest is absent"


def test_ruleset_reference_is_optional_when_module_or_packet_is_absent():
    """A legacy/partial install remains playable without encyclopedia data."""
    reference, digest, status = _collect_ruleset_reference(
        _FakeRulesetEnv(rule_ctrl=object()))
    assert reference is None
    assert digest is None
    assert status.startswith("unavailable:")


def test_ruleset_reference_is_captured_without_changing_legacy_ruleset(monkeypatch):
    """Startup metadata is exposed separately from RulesetNames."""
    import types

    reference = {"ruleset": "classic", "technologies": {"alphabet": {"cost": 5}}}
    fake_module = types.SimpleNamespace(
        build_ruleset_reference=lambda rule_ctrl: reference)
    monkeypatch.setitem(__import__("sys").modules,
                        "lechonk.adapter.ruleset_reference", fake_module)
    captured, digest, status = _collect_ruleset_reference(
        _FakeRulesetEnv(rule_ctrl={"techs": {1: {"name": "alphabet"}}}))
    assert captured is reference
    assert digest is None
    assert status == "complete"


def test_session_reference_properties_are_safe_before_start(tmp_path):
    session = AdapterSession(workdir=str(tmp_path), seed=1)
    assert session.ruleset_reference is None
    assert session.ruleset_reference_json is None
    assert session.ruleset_reference_hash is None
    assert session.ruleset_reference_status == "not_collected"
    assert session.help_reference is None
    assert session.help_reference_json is None
    assert session.help_reference_hash is None
    assert session.help_reference_status == "not_collected"


def test_help_reference_loader_uses_fixed_provisioned_path(monkeypatch):
    import types

    seen = []
    fake = types.SimpleNamespace(
        load_helpdata=lambda path: seen.append(path) or types.SimpleNamespace(
            status="available", content_hash="abc", to_dict=lambda: {"status": "available"}))
    monkeypatch.setitem(__import__("sys").modules,
                        "lechonk.adapter.help_reference", fake)
    reference, digest, status = _collect_help_reference()
    assert reference is not None
    assert digest == "abc"
    assert status == "available"
    assert str(seen[0]).endswith("artifacts/reference/freeciv/helpdata.txt")


def test_ruleset_reference_json_boundary_and_content_hash(tmp_path):
    from lechonk.adapter.ruleset_reference import RulesetReference

    session = AdapterSession(workdir=str(tmp_path), seed=1)
    reference = RulesetReference(
        ruleset_name="classic", summary="", description="", categories={})
    session._ruleset_reference = reference
    session._ruleset_reference_hash = reference.content_hash
    session._ruleset_reference_status = "complete"
    assert session.ruleset_reference_json["ruleset_name"] == "classic"
    assert session.ruleset_reference_hash == reference.content_hash


def test_session_preserves_runner_horizon_for_controller_guard(tmp_path):
    bounded = AdapterSession(workdir=str(tmp_path / "bounded"), seed=1,
                             max_turns=12)
    unbounded = AdapterSession(workdir=str(tmp_path / "unbounded"), seed=1)
    assert bounded.max_turns == 12
    assert unbounded.max_turns is None
    assert bounded.controller_max_turns == 13
    assert unbounded.controller_max_turns == 100_000
    assert bounded.pingtimeout_s == 720.0
    assert unbounded.pingtimeout_s == 1800.0
    assert bounded.pingtimeout_status == "configured"
    assert unbounded.pingtimeout_status == "configured"
    assert bounded.pingtimeout_actual is None


def test_server_timeout_covers_long_model_phase(tmp_path):
    bounded = AdapterSession(
        workdir=str(tmp_path / "bounded-deadline"), seed=1,
        max_turns=2, acting_deadline_s=180.0)
    interactive = AdapterSession(
        workdir=str(tmp_path / "interactive-deadline"), seed=1,
        acting_deadline_s=1200.0)
    assert bounded.server_timeout_s == 300.0
    assert interactive.server_timeout_s == 1260.0
    assert interactive.server_timeout_s > interactive.acting_deadline_s
    assert interactive.pingtimeout_s == 1800.0


def test_pingtimeout_covers_custom_long_acting_deadline(tmp_path):
    session = AdapterSession(
        workdir=str(tmp_path / "long-deadline"), seed=1,
        acting_deadline_s=2000.0)
    assert session.pingtimeout_s == 2060.0


class _FakeOptions:
    def __init__(self, setting):
        self.server_settings = setting


class _FakePingController:
    def __init__(self, setting):
        self.opt_ctrl = _FakeOptions(setting)


class _FakePingEnv:
    def __init__(self, setting):
        self.civ_controller = _FakePingController(setting)


class _FakeScenarioEnv:
    def __init__(self, settings, players, nations=None):
        self.civ_controller = type(
            "Controller", (), {
                "opt_ctrl": _FakeOptions(settings),
                "player_ctrl": type("Players", (), {"players": players})(),
                "rule_ctrl": type("Rules", (), {"nations": nations or {}})(),
            })()


def test_scenario_readback_exposes_typed_settings_and_roster(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "scenario-ok"), seed=1)
    session._collect_scenario_readback(_FakeScenarioEnv(
        {"aifill": {"val": 3}, "maxplayers": {"val": 10}},
        {
            0: {"playerno": 0, "name": "myagent", "flags": [False]},
            1: {"playerno": 1, "name": "AI One", "flags": [True], "nation": 1},
            2: {"playerno": 2, "name": "AI Two", "flags": [True], "nation": 2},
        },
        {1: {"rule_name": "Romans"}, 2: {"rule_name": "Greeks"}},
    ))
    assert session.scenario_readback_status == "confirmed"
    assert session.aifill_actual == 3
    assert session.maxplayers_actual == 10
    assert session.actual_total_players == 3
    assert session.actual_ai_opponents == 2
    assert session.actual_ai_players == 2
    assert [item["is_ai"] for item in session.actual_player_roster] == [False, True, True]


def test_scenario_readback_excludes_barbarian_and_pirate_ai(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "scenario-barbarians"), seed=1)
    session._collect_scenario_readback(_FakeScenarioEnv(
        {"aifill": {"val": 5}, "maxplayers": {"val": 10}},
        {
            0: {"playerno": 0, "name": "myagent", "flags": [False]},
            1: {"playerno": 1, "name": "Civilization", "flags": [True], "nation": 1},
            2: {"playerno": 2, "name": "Barbarian", "flags": [True], "nation": 2},
            3: {"playerno": 3, "name": "Pirate", "flags": [True], "nation": 3},
            4: {"playerno": 4, "name": "Civilization 2", "flags": [True], "nation": 4},
        },
        {
            1: {"rule_name": "Romans"},
            2: {"rule_name": "Barbarian"},
            3: {"rule_name": "Pirate"},
            4: {"rule_name": "Greeks"},
        },
    ))
    assert session.scenario_readback_status == "confirmed"
    assert session.actual_total_players == 5
    assert session.actual_ai_players == 4
    assert session.actual_ai_opponents == 2
    assert session.actual_barbarian_ai_opponents == 2
    assert session.actual_unknown_ai_opponents == 0
    assert session.actual_player_roster[2]["is_barbarian"] is True
    assert session.actual_player_roster[2]["nation_rule_name"] == "Barbarian"
    assert session.actual_player_roster[3]["is_barbarian"] is True


def test_scenario_readback_unknown_ai_nation_is_partial(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "scenario-unknown"), seed=1)
    session._collect_scenario_readback(_FakeScenarioEnv(
        {"aifill": {"val": 2}, "maxplayers": {"val": 10}},
        {
            0: {"playerno": 0, "name": "myagent", "flags": [False]},
            1: {"playerno": 1, "name": "Unknown AI", "flags": [True], "nation": 99},
        },
        {},
    ))
    assert session.scenario_readback_status == "partial"
    assert session.actual_total_players == 2
    assert session.actual_ai_players == 1
    assert session.actual_ai_opponents is None
    assert session.actual_barbarian_ai_opponents == 0
    assert session.actual_unknown_ai_opponents == 1
    assert "nation_rule_name" not in session.actual_player_roster[1]
    assert "is_barbarian" not in session.actual_player_roster[1]


def test_scenario_readback_is_nonfatal_and_marks_missing_data(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "scenario-missing"), seed=1)
    session._collect_scenario_readback(_FakeScenarioEnv({}, {}))
    assert session.scenario_readback_status == "unavailable"
    assert session.aifill_actual is None
    assert session.actual_player_roster is None


def test_pingtimeout_readback_confirms_value_and_exposes_actual(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "ping-ok"), seed=1,
                             acting_deadline_s=731.1)
    expected = math.ceil(session.pingtimeout_s)
    session._verify_pingtimeout(
        _FakePingEnv({"pingtimeout": {"val": expected, "is_visible": True}}),
        expected,
    )
    assert session.pingtimeout_status == "confirmed"
    assert session.pingtimeout_actual == expected


def test_pingtimeout_readback_accepts_schema_without_visibility(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "ping-schema"), seed=1)
    expected = math.ceil(session.pingtimeout_s)
    session._verify_pingtimeout(
        _FakePingEnv({"pingtimeout": {"val": expected}}), expected)
    assert session.pingtimeout_actual == expected


def test_pingtimeout_readback_rejects_missing_setting(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "ping-missing"), seed=1)
    try:
        session._verify_pingtimeout(_FakePingEnv({}), math.ceil(session.pingtimeout_s))
    except SessionStateError as exc:
        assert "did not report pingtimeout" in str(exc)
    else:
        raise AssertionError("missing pingtimeout readback was accepted")


def test_pingtimeout_readback_rejects_mismatch_and_hidden_setting(tmp_path):
    session = AdapterSession(workdir=str(tmp_path / "ping-bad"), seed=1)
    expected = math.ceil(session.pingtimeout_s)
    for setting, fragment in (
        ({"val": expected - 1, "is_visible": True}, "mismatch"),
        ({"val": expected, "is_visible": False}, "not visible"),
    ):
        try:
            session._verify_pingtimeout(
                _FakePingEnv({"pingtimeout": setting}), expected)
        except SessionStateError as exc:
            assert fragment in str(exc)
        else:
            raise AssertionError(f"invalid pingtimeout readback accepted: {setting!r}")


def test_acting_deadline_must_be_finite_and_positive(tmp_path):
    for value in (0, -1, float("inf"), float("nan")):
        try:
            AdapterSession(
                workdir=str(tmp_path / f"invalid-{value}"), seed=1,
                acting_deadline_s=value)
        except ValueError as exc:
            assert "acting_deadline_s" in str(exc)
        else:
            raise AssertionError("invalid acting deadline was accepted")


def test_max_turns_must_be_positive_integer_or_none(tmp_path):
    for value in (0, -1, True, False, 2.5, "2"):
        try:
            AdapterSession(workdir=str(tmp_path / "invalid-max-turns"),
                           seed=1, max_turns=value)
        except ValueError as exc:
            assert "max_turns" in str(exc)
        else:
            raise AssertionError(f"invalid max_turns accepted: {value!r}")

    unlimited = AdapterSession(workdir=str(tmp_path / "unlimited"), seed=1,
                               max_turns=None)
    assert unlimited.max_turns is None
    bounded = AdapterSession(workdir=str(tmp_path / "bounded"), seed=1,
                             max_turns=1)
    assert bounded.max_turns == 1


def test_load_game_requires_staged_username_bound_save_name(tmp_path):
    loaded = AdapterSession(
        workdir=str(tmp_path / "loaded"), seed=1, username="myagent",
        load_game="myagent_turn_004")
    assert loaded.load_game == "myagent_turn_004"
    for invalid in ("", "/tmp/save", "other_turn_004", "myagent_../save"):
        try:
            AdapterSession(workdir=str(tmp_path / "invalid"), seed=1,
                           username="myagent", load_game=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid load_game accepted: {invalid!r}")


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


def test_dispatch_restores_captured_worker_loop_when_thread_local_is_missing():
    session = _session()
    controller = _FakeController(defeated=False)
    session._env = _FakeEnv(terminated=False, truncated=False,
                            controller=controller)
    loop = asyncio.new_event_loop()
    session._adapter_asyncio_loop = loop
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        asyncio.set_event_loop(None)
        outcome = session._dispatch("step", ("unit", 10, "goto_4"))
        assert outcome.terminated is False
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_dispatch_rejects_changed_worker_loop_before_env_step():
    session = _session()
    controller = _FakeController(defeated=False)
    env = _FakeEnv(terminated=False, truncated=False, controller=controller)
    called = False
    original_step = env.step

    def step(action):
        nonlocal called
        called = True
        return original_step(action)

    env.step = step
    session._env = env
    captured = asyncio.new_event_loop()
    current = asyncio.new_event_loop()
    session._adapter_asyncio_loop = captured
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        asyncio.set_event_loop(current)
        try:
            session._dispatch("step", ("unit", 10, "goto_4"))
        except SessionStateError as exc:
            assert "loop changed" in str(exc)
        else:
            raise AssertionError("changed worker loop was accepted")
        assert called is False
    finally:
        asyncio.set_event_loop(previous)
        current.close()
        captured.close()


def test_capture_event_loop_rejects_mismatched_websocket_loop():
    session = _session()
    worker_loop = asyncio.new_event_loop()
    socket_loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        asyncio.set_event_loop(worker_loop)
        websocket = type("Socket", (), {
            "_io_loop": type("IOLoop", (), {
                "_asyncio_loop": socket_loop})()})()
        env = type("Env", (), {
            "civ_controller": type("Controller", (), {
                "ws_client": websocket})()})()
        try:
            session._capture_event_loop_identity(env)
        except SessionStateError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("mismatched websocket loop was accepted")
    finally:
        asyncio.set_event_loop(previous)
        worker_loop.close()
        socket_loop.close()


def test_capture_event_loop_uses_public_tornado_asyncio_loop_property():
    session = _session()
    worker_loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        asyncio.set_event_loop(worker_loop)
        websocket = type("Socket", (), {
            "_io_loop": type("IOLoop", (), {
                "asyncio_loop": worker_loop})()})()
        env = type("Env", (), {
            "civ_controller": type("Controller", (), {
                "ws_client": websocket})()})()
        session._capture_event_loop_identity(env)
        assert session._adapter_socket_asyncio_loop is worker_loop
    finally:
        asyncio.set_event_loop(previous)
        worker_loop.close()


def test_worker_startup_failure_closes_published_env_and_preserves_error():
    session = _session()
    class Env:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    env = Env()
    failure = SessionStateError("loop identity failed")

    def setup():
        session._env = env
        raise failure

    session._setup_env = setup  # type: ignore[method-assign]
    session._worker_main()

    assert env.close_calls == 1
    assert session._startup_error is failure
    assert session._startup_cleanup_error is None
    assert session._startup_env_closed is True


def test_worker_clears_dispatch_failure_after_successful_rpc():
    session = _session()
    session._setup_env = lambda: None  # type: ignore[method-assign]
    calls = iter(("fail", "ok", "unrelated-fail"))

    def dispatch(kind, payload):
        outcome = next(calls)
        if outcome == "fail":
            error = RuntimeError("websocket callback failed")
            error.diagnostic = {"stage": "write"}
            raise error
        if outcome == "unrelated-fail":
            raise ValueError("ordinary later error")
        return "ok"

    session._dispatch = dispatch  # type: ignore[method-assign]
    results = [queue.Queue(maxsize=1) for _ in range(3)]
    session._commands.put(("step", None, results[0]))
    session._commands.put(("step", None, results[1]))
    session._commands.put(("step", None, results[2]))
    session._commands.put(None)
    session._worker_main()

    assert results[0].get()[0] == "error"
    assert results[1].get() == ("ok", "ok")
    assert results[2].get()[0] == "error"
    assert session.last_dispatch_failure["transport_uncertain"] is False
    assert session._ready.is_set()
