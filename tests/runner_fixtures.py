# SPDX-License-Identifier: MIT
"""Runner test fixtures (WP2): a fake AdapterSession over synthetic
CivRealm observation dicts and a fake ModelClient — the offline fault
injection harness for core/runner.py.

The fake session drives the REAL pure adapter functions (build_candidates
over its synthetic obs/info, the real per-action read-back verifiers via
execute_candidate), so batch revalidation, group-abort semantics and the
read-back evidence are exercised for real. Step effects mutate a deepcopy
of the observation (goto moves the unit and exhausts its moves; produce /
research / fortify / sentry set the post-state the verifiers check) AND the
mutated observation is persisted, so the next action's pre-state and the
mid-batch revalidation see the staleness a real session would show.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Optional, Sequence

from lechonk.adapter.candidates import build_candidates
from lechonk.adapter.obs_extract import DIR8_DELTAS
from lechonk.adapter.session import StepOutcome
from lechonk.core.model_client import ModelResponse
from lechonk.core.schemas import ActionBatch, MemoryUpdate, RunManifest

from tests.fog_fixtures import (
    MY_PLAYER,
    NUM_IMPROVEMENTS,
    SETTLERS,
    VUT_IMPROVEMENT,
    VUT_UTYPE,
    WARRIORS,
    base_world,
    fixture_ruleset,
)
from tests.test_candidates import FULL_ACTIONS

# production display name -> (production_kind, production_value) as the city
# observation reports it (VUT_UTYPE values carry the num_impr_types offset).
PRODUCTION_NAMES = {
    "Warriors": (VUT_UTYPE, WARRIORS + NUM_IMPROVEMENTS),
    "Settlers": (VUT_UTYPE, SETTLERS + NUM_IMPROVEMENTS),
    "Granary": (VUT_IMPROVEMENT, 0),
    "City Walls": (VUT_IMPROVEMENT, 1),
}

_ACTIVITY_IDS = {"fortify": 4, "sentry": 5, "build_road": 13,
                 "irrigation": 3, "mine": 2}


class FakeSession:
    """Duck-typed AdapterSession over a scripted list of turn worlds.

    ``worlds[i]`` is the (observation, info) pair for the (i+1)-th own
    phase. A finish step (action None) advances to the next world; past the
    last world it returns the scripted final flags (terminated/truncated)
    with the current observation — the game-over-at-the-phase-boundary path.
    """

    def __init__(self, worlds: Sequence[tuple[dict, dict]], *,
                 player_id: int = MY_PLAYER,
                 production_names: Optional[dict] = None,
                 final_terminated: bool = True,
                 final_truncated: bool = False,
                 block_step: bool = False):
        self._worlds = [(copy.deepcopy(obs), copy.deepcopy(info))
                        for obs, info in worlds]
        self._index = 0
        self._player_id = player_id
        self._production_names = (PRODUCTION_NAMES if production_names is None
                                  else production_names)
        self._final_terminated = final_terminated
        self._final_truncated = final_truncated
        self._block_step = block_step
        self._unblock = threading.Event()
        self.step_deadline = 30.0
        self.wait_deadline = 900.0
        self.calls: list[tuple[Any, Optional[float]]] = []
        self.started = False
        self._closed = False
        self._close_status = "not_attempted"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.started = True

    def close(self) -> str:
        self._closed = True
        self._close_status = "completed"
        self._unblock.set()  # unblock a step held for the stall test
        return self._close_status

    # -- state accessors ---------------------------------------------------

    @property
    def obs(self):
        return self._worlds[self._index][0]

    @property
    def info(self):
        return self._worlds[self._index][1]

    @property
    def turn(self) -> int:
        return int(self._worlds[self._index][1].get("turn", 1))

    @property
    def player_id(self):
        return self._player_id

    @property
    def username(self):
        return "myagent"

    @property
    def port(self):
        return 6300

    @property
    def ruleset(self):
        return fixture_ruleset()

    @property
    def action_probs(self):
        return {}

    @property
    def unit_flags(self):
        return {}

    @property
    def failed(self):
        return None

    @property
    def closed(self):
        return self._closed

    @property
    def close_status(self):
        return self._close_status

    def production_name_map(self):
        return dict(self._production_names)

    # -- supervised operations ---------------------------------------------

    def build_candidate_table(self):
        return build_candidates(
            self.obs, self.info, my_player_id=self._player_id,
            unit_flags={}, action_probs={}, turn=self.turn)

    def refresh(self):
        return {"info": self.info, "observation": self.obs}

    def step(self, action, *, deadline=None):
        self.calls.append((action, deadline))
        if self._block_step and action is None:
            # The stall test: hold the FINISH step (the pipeline-owned
            # phase_done) until close() unblocks it — no transcript event
            # flows while it is held.
            self._unblock.wait(timeout=30.0)
            raise RuntimeError(
                "blocked step released by close (simulated stall)")
        obs, info = self._worlds[self._index]
        if action is None:
            # The pipeline-owned finish primitive: advance the phase.
            if self._index + 1 < len(self._worlds):
                self._index += 1
                new_obs, new_info = self._worlds[self._index]
                return StepOutcome(observation=new_obs, reward=0.0,
                                   terminated=False, truncated=False,
                                   info=new_info, action=None, elapsed_s=0.01)
            # Past the last scripted world: the game ends at the boundary.
            return StepOutcome(observation=obs, reward=0.0,
                               terminated=self._final_terminated,
                               truncated=self._final_truncated,
                               info={"turn": info.get("turn", 1) + 1,
                                     "available_actions": {}},
                               action=None, elapsed_s=0.01)
        new_obs = copy.deepcopy(obs)
        self._apply_effect(new_obs, action)
        new_info = copy.deepcopy(info)
        # Persist the mutation: the next action's pre-state (session.obs,
        # read by execute_candidate) and the runner's mid-batch revalidation
        # (build_candidate_table over the refreshed obs) see the staleness.
        self._worlds[self._index] = (new_obs, new_info)
        return StepOutcome(observation=new_obs, reward=0.0,
                           terminated=False, truncated=False,
                           info=new_info, action=action, elapsed_s=0.01)

    # -- scripted step effects ----------------------------------------------

    def _apply_effect(self, obs: dict, action) -> None:
        ctrl_type, actor_id, action_key = action
        if ctrl_type == "unit":
            uid = int(actor_id)
            unit = obs["unit"].get(uid)
            if unit is None:
                return
            if action_key.startswith("goto_") or action_key.startswith("attack_"):
                # Raw action keys carry the freeciv dir8 index (goto_4 = E);
                # the same DIR8_DELTAS table the read-back verifier uses.
                dir8 = int(action_key.split("_", 1)[1])
                dx, dy = DIR8_DELTAS[dir8]
                unit["x"] = int(unit["x"]) + dx
                unit["y"] = int(unit["y"]) + dy
                unit["moves_left"] = 0  # the move exhausts the unit
            elif action_key in _ACTIVITY_IDS:
                unit["activity"] = _ACTIVITY_IDS[action_key]
            elif action_key == "build_city":
                del obs["unit"][uid]
        elif ctrl_type == "city":
            cid = int(actor_id)
            city = obs["city"].get(cid)
            if city is None:
                return
            if action_key.startswith("produce_"):
                name = action_key[len("produce_"):]
                kind, value = self._production_names.get(
                    name, (VUT_IMPROVEMENT, 0))
                city["production_kind"] = kind
                city["production_value"] = value
            elif action_key == "city_buy_production":
                city["shield_stock"] = int(city.get("shield_stock", 0)) + 10
        elif ctrl_type == "tech":
            me = obs["player"].get(self._player_id)
            if me is not None and action_key.startswith("research_tech_"):
                tech_id = int(action_key.rsplit("_", 1)[1])
                me["researching"] = tech_id
        elif ctrl_type == "dipl":
            entry = obs.get("dipl", {}).get(int(actor_id))
            if entry is not None:
                entry["diplomacy_clause_map"] = []


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------


def turn_world(turn: int, *, researching=None, unit_overrides=(), **obs_kwargs):
    """A synthetic turn world: the canonical base world with a full
    available_actions table and per-test overrides."""
    obs = base_world(**obs_kwargs)
    me = obs["player"][MY_PLAYER]
    if researching is not None:
        me["researching"] = researching
    for uid, overrides in unit_overrides:
        obs["unit"][uid].update(overrides)
    info = {"turn": turn, "mini_game_messages": [],
            "available_actions": copy.deepcopy(FULL_ACTIONS)}
    return obs, info


# ---------------------------------------------------------------------------
# fake model client
# ---------------------------------------------------------------------------


class FakeModelClient:
    """A scripted ModelClient: each call pops the next scripted response
    (a ModelResponse or a callable taking the prompt)."""

    def __init__(self, script: Sequence[Any] = ()):
        self._script = list(script)
        self.turn_prompts = []
        self.repair_prompts = []
        self.calls = 0

    def _next(self, prompt):
        self.calls += 1
        if not self._script:
            raise AssertionError("FakeModelClient: no scripted response left")
        item = self._script.pop(0)
        return item(prompt) if callable(item) else item

    def complete_turn(self, prompt):
        self.turn_prompts.append(prompt)
        return self._next(prompt)

    def complete_repair(self, prompt):
        self.repair_prompts.append(prompt)
        return self._next(prompt)


def ok_response(refs=(), *, turn_id="t", commitments=(), threats=(),
                notes="", finish_requested=True, with_memory=True):
    """A valid model response: an ActionBatch with the given candidate refs
    and a memory_update carrying the commitments/threats."""
    commitment_dicts = [{"domain": d, "target": t, "deadline_turn": dl}
                        for d, t, dl in commitments]
    batch = ActionBatch(
        turn_id=turn_id,
        actions=[{"ref": r} for r in refs],
        commitments=commitment_dicts,
        finish_requested=finish_requested,
        memory_update=(MemoryUpdate(commitments=commitment_dicts,
                                    threats=list(threats), notes=notes)
                       if with_memory else None),
    )
    return ModelResponse(call_id="fake-call", prompt_hash="ph", batch=batch,
                         latency_ms=1, tokens_in=10, tokens_out=10,
                         cost_usd=0.0, fingerprint="fp")


def error_response(message="request timed out"):
    """A transport-level failure (timeout / API error)."""
    return ModelResponse(call_id="fake-call", prompt_hash="ph",
                         error=message, latency_ms=1, tokens_in=10,
                         tokens_out=0, cost_usd=0.0, fingerprint="fp")


def malformed_response(raw_attempts, detail="output violated the batch schema"):
    """A schema-violating output: no valid batch, the raw emitted refs."""
    return ModelResponse(call_id="fake-call", prompt_hash="ph",
                         raw_attempts=list(raw_attempts), malformed=detail,
                         latency_ms=1, tokens_in=10, tokens_out=10,
                         cost_usd=0.0, fingerprint="fp")


class FakeClock:
    """An injectable monotonic clock for the runner/ledger/watchdog."""

    def __init__(self, start: Optional[float] = None):
        self.t = time.monotonic() if start is None else start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def advancing_response(factory: Callable[[], ModelResponse], clock: FakeClock,
                       seconds: float) -> Callable[[Any], ModelResponse]:
    """A scripted response that also advances the fake clock (simulating a
    slow model call for the ACTING-deadline test)."""
    def call(prompt):
        clock.advance(seconds)
        return factory()
    return call


# ---------------------------------------------------------------------------
# runner assembly
# ---------------------------------------------------------------------------


def make_runner(tmp_path, *, client, session, max_turns=2, clock=None,
                config_overrides: Optional[dict] = None, progress=None):
    """Assemble a GameRunner over the fake session + fake client."""
    from lechonk.core.ledger import TranscriptLedger
    from lechonk.core.runner import GameRunner, RunnerConfig

    overrides = dict(max_turns=max_turns, run_id="test-run",
                     watchdog_interval_s=0.02)
    if config_overrides:
        overrides.update(config_overrides)
    config = RunnerConfig(**overrides)
    clock = clock or time.monotonic
    ledger = TranscriptLedger(tmp_path / "transcript.jsonl", clock=clock)
    runner = GameRunner(
        config=config, client=client, ledger=ledger,
        manifest=RunManifest(run_id="test-run"),
        session_factory=lambda: session, clock=clock, progress=progress)
    return runner
