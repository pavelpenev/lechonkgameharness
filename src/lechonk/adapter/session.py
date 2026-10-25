# SPDX-License-Identifier: GPL-3.0-or-later
"""lechonk.adapter.session — FreecivBaseEnv lifecycle ownership (WP1 package B).

GPL-3.0-or-later (design-plan-freeciv-v1.1.md decision 7): this module is part
of the adapter package, the ONLY lechonk package that imports civrealm. The
civrealm import is LAZY (importlib inside the worker thread, only after the
Docker stack is up and after chdir into the dedicated working directory):
bringing civrealm into the process performs an HTTP pubstatus fetch at import
time (port_utils.py:44-46), and upstream creates logs/, recordings/ and
game_scores/ in CWD (freeciv_logging + freeciv_base_env.set_up_recording).

Operational notes implemented here (plan §7.1, source-verified):
- fc_args is PROCESS-GLOBAL and set BEFORE reset(): self_play=False (upstream
  default True silently renames the user on connection congestion — congestion
  must raise loudly instead), aifill, randomly_generate_seeds=False with
  explicit mapseed/gameseed/agentseed (reset(seed=) only sets mapseed +
  agentseed; gameseed is sent only when debug.gameseed is present,
  client_state.py:118-132).
- Upstream timers raised to align with §3 deadlines: server_timeout 30 -> 300
  (ACTING <= 180 s), begin_turn_timeout 30 -> 900 (WAITING <= 720 s + margin),
  wait_for_timeout 10000 -> 60 (the upstream wait-unstick bound aligned with
  the §3 per-mutation send+read-back <= 30 s budget; the 10000 s default would
  let a stuck action-response wait hang for ~2.8 h, contradicting §3's
  bounded-overrun rule — the session's own step deadline is the primary bound
  and fires first).
- One env per process in v1 (enforced by a module flag; fc_args is global).
- The env runs in a DEDICATED WORKER THREAD: the WebSocket client captures
  tornado's IOLoop.current() at connect() (web_socket_client.py:67) and
  start/stop_ioloop operate on that captured loop, so every env operation
  must run in the thread that called connect(). The calling thread
  supervises with hard deadlines (§3: startup <= 120 s, per-mutation
  send+read-back <= 30 s, teardown <= 30 s, WAITING <= 720 s + margin);
  Ports.get() spinning forever inside reset() is bounded by the startup
  deadline.
- env.close() is NEVER called unguarded: CivController.close() waits on a
  game-over message in an unbounded loop (civ_controller.py:578-583) and
  ws_client.close() restarts IOLoop.current() (web_socket_client.py:122-123).
  AdapterSession.close() wraps the call with a hard deadline and a skip path.
- All upstream HTTP calls (requests, no timeout) are wrapped at <= 15 s:
  scorelog, listsavegames, deletesavegame, create_pbem_user (login, every
  reset), gamesetting (login/load) — via a requests Session.request patch
  installed before the env is created.
"""

from __future__ import annotations

import importlib
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from lechonk.adapter import candidates as candidates_mod
from lechonk.adapter.obs_extract import RulesetNames
from lechonk.core.schemas import CandidateTable

# --- §3-aligned timeouts and deadlines --------------------------------------
HTTP_TIMEOUT_S = 15.0        # §3: HTTP <= 15 s (all upstream requests calls)
SERVER_TIMEOUT_S = 300       # ACTING <= 180 s (upstream default 30 truncates)
BEGIN_TURN_TIMEOUT_S = 900   # WAITING <= 720 s + margin (upstream default 30)
WAIT_FOR_TIMEOUT_S = 60      # 2x the §3 per-mutation <= 30 s bound (default 10000)
DEFAULT_STARTUP_DEADLINE_S = 150.0  # §3 startup <= 120 s + server-spawn margin
DEFAULT_STEP_DEADLINE_S = 30.0      # §3 per-mutation send+read-back <= 30 s
DEFAULT_WAIT_DEADLINE_S = 900.0     # §3 WAITING <= 720 s + margin
DEFAULT_CLOSE_DEADLINE_S = 30.0     # §3 teardown <= 30 s

# One env per process in v1 (plan §7.1: fc_args is process-global).
_ENV_EVER_CREATED = False


class SessionError(RuntimeError):
    """Session operation failed (worker-side exception or bad state)."""


class SessionTimeoutError(SessionError):
    """A supervised operation exceeded its hard deadline (§3 bounded overrun).

    The session is marked failed on timeout: no blind retry, no silent
    continuation — the runner fails the run with artifacts.
    """

    def __init__(self, kind: str, deadline: float):
        self.kind = kind
        self.deadline = deadline
        super().__init__(f"{kind} exceeded its deadline of {deadline:.1f}s")


class SessionStateError(SessionError):
    """The session is in a state that forbids the requested operation."""


@dataclass
class StepOutcome:
    """One env.step() result (gymnasium 5-tuple) plus the action and timing."""

    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    action: Any  # action tuple | None (finish) | 'pass'
    elapsed_s: float
    # The env's defeat signal (my_player_is_defeated), read from the
    # controller at a game-over step — it is NOT part of the env's step
    # tuple (vendor: civ_controller.py:303, a read-only CivController
    # property reached via env.civ_controller). None unless the step
    # ended the game.
    my_player_is_defeated: Optional[bool] = None


def _snapshot(value: Any) -> Any:
    """Freeze a CivRealm info/observation value for caching.

    Upstream hands out LIVE state objects (source-verified): the top-level
    observation/info dicts (turn_manager._turn_state / _turn_info) are reused
    across reads with their per-domain slots rebound, the player/diplomacy
    domain dicts are reused with per-entity dicts replaced, and the map
    domain's numpy arrays are allocated once (map_allocate) and mutated in
    place. A reference held across a step therefore sees the LATEST state,
    which corrupts pre/post read-back comparisons. This snapshot copies
    mappings recursively and numpy arrays one level, freezing every domain
    the adapter consumes.
    """
    if isinstance(value, Mapping):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if not isinstance(value, (str, bytes, bool, int, float, type(None))):
        copier = getattr(value, "copy", None)
        if callable(copier):
            try:
                return copier()  # numpy arrays and similar
            except Exception:
                return value
    return value


def _install_http_timeout_guard(timeout_s: float = HTTP_TIMEOUT_S) -> None:
    """Wrap every upstream requests call at <= timeout_s.

    CivRealm's reachable HTTP calls (scorelog, listsavegames, deletesavegame,
    create_pbem_user in login, gamesetting in login/load) use requests with
    NO timeout (plan §7.1). Patching Session.request injects a default
    timeout for every call site in this process; one env per process makes
    the process-global patch safe.
    """
    import requests  # lazy: only needed once civrealm is in the process

    original = requests.sessions.Session.request

    def _guarded(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("timeout", timeout_s)
        return original(self, method, url, **kwargs)

    requests.sessions.Session.request = _guarded  # type: ignore[method-assign]


def _collect_ruleset(env: Any) -> RulesetNames:
    """Ruleset display tables from the live ruleset controller (duck-typed)."""
    rule_ctrl = env.civ_controller.rule_ctrl
    n_impr = int(rule_ctrl.ruleset_control["num_impr_types"])
    return RulesetNames(
        terrains={tid: t["name"] for tid, t in rule_ctrl.terrains.items()},
        unit_types={uid: u["name"] for uid, u in rule_ctrl.unit_types.items()},
        improvements={iid: i["name"] for iid, i in rule_ctrl.improvements.items()},
        nations={nid: (n.get("name") or n.get("adjective") or f"nation#{nid}")
                 for nid, n in rule_ctrl.nations.items()},
        tech_costs={tid: (t.get("cost") if isinstance(t.get("cost"), int) else None)
                    for tid, t in rule_ctrl.techs.items()},
        # city_state.py adds num_impr_types to production_value for VUT_UTYPE (6);
        # VUT_IMPROVEMENT (3) values are improvement ids directly.
        city_production_names=(
            {(6, uid + n_impr): u["name"]
             for uid, u in rule_ctrl.unit_types.items()}
            | {(3, iid): i["name"]
               for iid, i in rule_ctrl.improvements.items()}
        ),
    )


def _collect_action_probs(env: Any) -> dict[int, dict[int, dict[int, dict[str, float]]]]:
    """unit_id -> dir8 -> action_id -> {'min','max'} percent (plan §7.2 line 148).

    Source: unit_ctrl.prop_actions.unit_data[uid].action_prob (reply pid 90),
    the same structure the package-A extractor consumes.
    """
    probs: dict[int, dict[int, dict[int, dict[str, float]]]] = {}
    unit_data = getattr(env.civ_controller.unit_ctrl.prop_actions, "unit_data", {})
    for uid, focus in unit_data.items():
        ap = getattr(focus, "action_prob", None)
        if ap:
            probs[int(uid)] = {int(d): dict(p) for d, p in ap.items()}
    return probs


def _collect_unit_flags(env: Any, my_player_id: int) -> dict[int, dict[str, bool]]:
    """punit gating flags not present in the observation (unit_actions.py:324-331).

    done_moving / keep_activity live on the raw punit, not in the unit
    observation domain; the candidate builder gates on them via this map.
    """
    flags: dict[int, dict[str, bool]] = {}
    units = getattr(env.civ_controller.unit_ctrl, "units", {})
    for uid, punit in units.items():
        if punit.get("owner") != my_player_id:
            continue
        flags[int(uid)] = {
            "done_moving": bool(punit.get("done_moving")),
            "keep_activity": bool(punit.get("keep_activity")),
        }
    return flags


class AdapterSession:
    """Owns one FreecivBaseEnv for the whole process (one env per process, v1).

    The env lives in a dedicated worker thread (the thread that called
    connect() owns the tornado ioloop); the calling thread sends commands
    over a queue and supervises each with a hard deadline. State properties
    (obs/info/turn/...) expose the latest completed step's data.
    """

    def __init__(
        self,
        *,
        workdir: str,
        seed: int,
        username: str = "myagent",
        aifill: int = 2,
        startup_deadline: float = DEFAULT_STARTUP_DEADLINE_S,
        step_deadline: float = DEFAULT_STEP_DEADLINE_S,
        wait_deadline: float = DEFAULT_WAIT_DEADLINE_S,
        close_deadline: float = DEFAULT_CLOSE_DEADLINE_S,
    ):
        self._workdir = os.path.abspath(workdir)
        self._seed = int(seed)
        self._username = username
        self._aifill = int(aifill)
        self.startup_deadline = float(startup_deadline)
        self.step_deadline = float(step_deadline)
        self.wait_deadline = float(wait_deadline)
        self.close_deadline = float(close_deadline)

        self._commands: "queue.Queue[tuple[str, Any, queue.Queue]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None

        # env-side handles (set by the worker, read cross-thread after steps)
        self._env: Any = None
        self._obs: Mapping[str, Any] = {}
        self._info: Mapping[str, Any] = {}
        self._turn = 0
        self._player_id: Optional[int] = None
        self._port: Optional[int] = None
        self._ruleset: Optional[RulesetNames] = None
        self._action_probs: Mapping[int, Any] = {}
        self._unit_flags: Mapping[int, Any] = {}
        # The latest completed step's game-over flags: terminated/truncated
        # come from the env's step tuple; the defeat signal is read from
        # the controller at the game-over boundary (see _dispatch).
        self._terminated = False
        self._truncated = False
        self._my_player_is_defeated: Optional[bool] = None

        self._failed: Optional[str] = None
        self._closed = False
        self._close_status = "not_attempted"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread and run the deterministic reset.

        §3 startup <= 120 s (plus server-spawn margin): bounded wait on the
        worker's ready event; a startup failure is a startup failure, not a
        crash (plan §7.1).
        """
        global _ENV_EVER_CREATED
        if _ENV_EVER_CREATED:
            raise SessionStateError(
                "one env per process in v1 (plan §7.1): this process already "
                "created a FreecivBaseEnv")
        _ENV_EVER_CREATED = True
        os.makedirs(self._workdir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._worker_main, name="lechonk-adapter-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.startup_deadline):
            self._failed = (f"startup exceeded its deadline of "
                            f"{self.startup_deadline:.1f}s")
            raise SessionTimeoutError("startup", self.startup_deadline)
        if self._startup_error is not None:
            self._failed = f"startup failed: {self._startup_error!r}"
            raise SessionError("startup failed") from self._startup_error

    def close(self) -> str:
        """Guarded env close: hard deadline + skip path. Returns the status.

        NEVER call env.close() unguarded: CivController.close() waits on a
        final game-over message in an unbounded loop (civ_controller.py:578-583)
        and ws_client.close() restarts IOLoop.current() (web_socket_client.py:
        122-123). The close runs in the worker thread under a §3 teardown
        deadline (<= 30 s); on timeout the skip path applies — the call is
        abandoned (the worker thread is a daemon), the process exits, and the
        fresh game server quits idle (~20 s, --quitidle 20). Skipping is
        recorded, never silent.
        """
        if self._closed:
            return self._close_status
        self._closed = True
        try:
            self._rpc("close", timeout=self.close_deadline)
            self._close_status = "completed"
        except SessionTimeoutError:
            # Skip path (see docstring): do not block the pipeline on the
            # upstream unbounded game-over wait loop.
            self._close_status = "skipped_after_deadline"
        except SessionError as exc:
            self._close_status = f"error: {exc!r}"
        return self._close_status

    # ------------------------------------------------------------------
    # state accessors (latest completed step)
    # ------------------------------------------------------------------

    @property
    def obs(self) -> Mapping[str, Any]:
        return self._obs

    @property
    def info(self) -> Mapping[str, Any]:
        return self._info

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def terminated(self) -> bool:
        """The latest completed step's terminated flag (game over)."""
        return self._terminated

    @property
    def truncated(self) -> bool:
        """The latest completed step's truncated flag (infra/horizon)."""
        return self._truncated

    @property
    def my_player_is_defeated(self) -> Optional[bool]:
        """The env's defeat signal read at the latest game-over step, or
        None (mid-game, or a game-over step whose signal read failed)."""
        return self._my_player_is_defeated

    @property
    def player_id(self) -> Optional[int]:
        return self._player_id

    @property
    def username(self) -> str:
        return self._username

    @property
    def port(self) -> Optional[int]:
        return self._port

    @property
    def ruleset(self) -> Optional[RulesetNames]:
        return self._ruleset

    @property
    def action_probs(self) -> Mapping[int, Any]:
        return self._action_probs

    @property
    def unit_flags(self) -> Mapping[int, Any]:
        return self._unit_flags

    @property
    def failed(self) -> Optional[str]:
        return self._failed

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_status(self) -> str:
        return self._close_status

    def production_name_map(self) -> dict[str, tuple[int, int]]:
        """production display name -> (production_kind, production_value) as
        reported in the city observation (the caller applies the same
        num_impr_types offset for VUT_UTYPE — RulesetNames docstring)."""
        if self._ruleset is None:
            return {}
        return {name: (kind, value)
                for (kind, value), name in self._ruleset.city_production_names.items()}

    # ------------------------------------------------------------------
    # supervised operations
    # ------------------------------------------------------------------

    def step(self, action: Any, *, deadline: Optional[float] = None) -> StepOutcome:
        """One env.step() under the §3 per-mutation deadline (<= 30 s).

        ``action`` is a CivRealm action tuple (ctrl_type, actor_id,
        action_key), None (finish phase), or 'pass' (no-op). The cached
        obs/info/turn advance to this step's result — and so do the
        game-over flags (terminated/truncated plus the env's defeat
        signal), so the runner can classify an empty game-over
        observation at the next decision point.
        """
        outcome = self._rpc("step", action,
                            timeout=deadline if deadline is not None
                            else self.step_deadline)
        self._obs = outcome.observation
        self._info = outcome.info
        self._turn = int(outcome.info.get("turn", self._turn) or self._turn)
        self._terminated = bool(outcome.terminated)
        self._truncated = bool(outcome.truncated)
        self._my_player_is_defeated = getattr(
            outcome, "my_player_is_defeated", None)
        return outcome

    def refresh(self) -> Mapping[str, Any]:
        """Re-read info+observation without acting.

        Upstream quirk (source-verified): CivController._get_info() opens with
        a lock_control() that only returns once a server packet batch arrives;
        with nothing pending, a STANDALONE read blocks until the next
        connection ping (~30 s), which would race the §3 per-read deadline.
        The read is therefore seeded with turn_manager.get_available_actions()
        — the action-probability queries give the leading lock_control
        something to wait for (upstream's own "seed a message" pattern,
        unit_actions.py:313-328). Only meaningful during our own phase
        (turn_manager._turn_ctrls is None while waiting).

        Read path caveat (plan §7.1): get_info_and_observation() is not
        perfectly non-mutating — probability queries cancel unit activities
        first and restore them after; a mid-read disconnect leaves activities
        cancelled (an uncertain-mutation window the WP2 reconciliation owns).
        """
        snap = self._rpc("refresh", timeout=self.step_deadline)
        self._obs = snap["observation"]
        self._info = snap["info"]
        self._turn = int(snap["info"].get("turn", self._turn) or self._turn)
        return snap

    def build_candidate_table(self) -> CandidateTable:
        """Build the legal-candidate table from the CURRENT state.

        No re-read here: the cached obs/info come from reset() or the latest
        step(), and env.step() recomputes info['available_actions'] after every
        action, so the cache is fresh at every decision point. (A standalone
        re-read would block on the upstream leading lock_control until the
        next server ping — see refresh().) The gating/odds context (raw punit
        flags, action probabilities) is collected read-only from the live
        controllers.
        """
        ctx = self._rpc("context", timeout=self.step_deadline)
        self._action_probs = ctx["action_probs"]
        self._unit_flags = ctx["unit_flags"]
        return candidates_mod.build_candidates(
            self._obs, self._info,
            my_player_id=self._player_id,
            unit_flags=self._unit_flags,
            action_probs=self._action_probs,
            turn=self._turn,
        )

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def _worker_main(self) -> None:
        try:
            self._setup_env()
        except BaseException as exc:  # startup failure, reported to start()
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            cmd = self._commands.get()
            if cmd is None:  # shutdown marker (no env close)
                return
            kind, payload, result_q = cmd
            try:
                result = self._dispatch(kind, payload)
            except BaseException as exc:
                result_q.put(("error", exc))
            else:
                result_q.put(("ok", result))

    def _setup_env(self) -> None:
        """Runs in the worker thread: chdir, lazy civrealm import, fc_args,
        HTTP guard, env creation, deterministic reset."""
        # Dedicated working directory FIRST: upstream writes logs/,
        # recordings/, game_scores/ into CWD (and the civrealm import itself
        # creates the log directory).
        os.chdir(self._workdir)

        configs = importlib.import_module("civrealm.configs")
        fc_args = configs.fc_args  # process-global (configs/__init__.py:112)
        fc_args["self_play"] = False  # congestion must raise, never rename
        fc_args["aifill"] = self._aifill
        fc_args["debug.randomly_generate_seeds"] = False
        fc_args["debug.mapseed"] = self._seed
        fc_args["debug.gameseed"] = self._seed
        fc_args["debug.agentseed"] = self._seed
        # Raised to align with §3 deadlines (plan §7.1; see module docstring).
        fc_args["server_timeout"] = SERVER_TIMEOUT_S
        fc_args["begin_turn_timeout"] = BEGIN_TURN_TIMEOUT_S
        fc_args["wait_for_timeout"] = WAIT_FOR_TIMEOUT_S

        # Wrap every upstream no-timeout HTTP call at <= 15 s BEFORE the env
        # is created (login's create_pbem_user/gamesetting run inside reset).
        _install_http_timeout_guard(HTTP_TIMEOUT_S)

        envs = importlib.import_module("civrealm.envs")
        env = envs.FreecivBaseEnv(username=self._username)
        obs, info = env.reset(seed=self._seed)
        self._env = env
        self._obs = _snapshot(obs)
        self._info = _snapshot(info)
        self._turn = int(info.get("turn", 0) or 0)
        self._player_id = int(env.get_playerid())
        self._port = int(env.get_port())
        self._ruleset = _collect_ruleset(env)

    def _dispatch(self, kind: str, payload: Any) -> Any:
        if kind == "step":
            t0 = time.monotonic()
            obs, reward, terminated, truncated, info = self._env.step(payload)
            defeated = None
            if terminated or truncated:
                # The defeat signal is NOT in the env's step tuple (vendor:
                # civ_controller.py:303 — a read-only CivController property
                # reached via env.civ_controller): read it once, at the
                # game-over boundary, and carry it on the outcome so the
                # session can cache it with the flags. A failed read is
                # None (unknown) — never a fabricated signal.
                try:
                    defeated = bool(
                        self._env.civ_controller.my_player_is_defeated())
                except Exception:
                    defeated = None
            return StepOutcome(
                observation=_snapshot(obs), reward=reward,
                terminated=terminated, truncated=truncated,
                info=_snapshot(info), action=payload,
                elapsed_s=time.monotonic() - t0,
                my_player_is_defeated=defeated)
        if kind == "refresh":
            # Seed the read first: _get_info()'s leading lock_control() only
            # returns once a server packet arrives; the action-probability
            # queries sent here give it something to wait for (see refresh()).
            self._env.civ_controller.turn_manager.get_available_actions()
            info, obs = self._env.civ_controller.get_info_and_observation()
            return {"info": _snapshot(info), "observation": _snapshot(obs)}
        if kind == "context":
            return {
                "action_probs": _collect_action_probs(self._env),
                "unit_flags": _collect_unit_flags(self._env, self._player_id),
            }
        if kind == "close":
            # THE guarded env.close() call site: AdapterSession.close() wraps
            # this with a hard deadline and a skip path (module docstring).
            self._env.close()
            return True
        raise SessionStateError(f"unknown command {kind!r}")

    # ------------------------------------------------------------------
    # main-thread RPC
    # ------------------------------------------------------------------

    def _rpc(self, kind: str, payload: Any = None, *, timeout: float) -> Any:
        if self._failed is not None:
            raise SessionStateError(
                f"session is failed ({self._failed}); refusing {kind}")
        if self._closed and kind != "close":
            raise SessionStateError(f"session is closed; refusing {kind}")
        result_q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)
        self._commands.put((kind, payload, result_q))
        try:
            status, value = result_q.get(timeout=timeout)
        except queue.Empty:
            self._failed = f"{kind} exceeded its deadline of {timeout:.1f}s"
            raise SessionTimeoutError(kind, timeout) from None
        if status == "error":
            raise SessionError(f"{kind} failed: {value!r}") from value
        return value
