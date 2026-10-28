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
- Upstream timers raised to align with §3 deadlines: bounded sessions use a
  300-second server_timeout (ACTING <= 180 s); long-game sessions derive a
  server_timeout above the configured acting deadline. begin_turn_timeout is
  900 (WAITING <= 720 s + margin),
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
import asyncio
import math
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from lechonk.adapter import candidates as candidates_mod
from lechonk.adapter.obs_extract import RulesetNames
from lechonk.core.recovery import validate_freeciv_username
from lechonk.core.schemas import CandidateTable

# --- §3-aligned timeouts and deadlines --------------------------------------
HTTP_TIMEOUT_S = 15.0        # §3: HTTP <= 15 s (all upstream requests calls)
SERVER_TIMEOUT_S = 300       # ACTING <= 180 s (upstream default 30 truncates)
SERVER_TIMEOUT_MARGIN_S = 60.0  # leave room for phase-boundary cleanup
BEGIN_TURN_TIMEOUT_S = 900   # WAITING <= 720 s + margin (upstream default 30)
WAIT_FOR_TIMEOUT_S = 60      # 2x the §3 per-mutation <= 30 s bound (default 10000)
PINGTIMEOUT_S = 1800         # Keep Freeciv alive across long streamed model calls.
# CivRealm's own controller truncates at fc_args['max_turns']; the runner owns
# the real horizon, so give an unbounded runner enough headroom here.
UNBOUNDED_MAX_TURNS = 100_000
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


def _collect_ruleset_reference(env: Any) -> tuple[Any, Optional[str], str]:
    """Best-effort immutable encyclopedia snapshot for the live ruleset.

    The reference builder is deliberately kept in a separate, whitelisted
    module.  It is optional adapter metadata: old/partial CivRealm packets
    must not make an otherwise playable session fail during startup.  The
    builder owns validation, freezing, JSON-safety, and the content hash.
    """
    try:
        rule_ctrl = env.civ_controller.rule_ctrl
        has_container = False
        for name in ("techs", "unit_types", "improvements", "terrains",
                     "governments", "extras"):
            value = rule_ctrl.get(name) if isinstance(rule_ctrl, Mapping) else getattr(rule_ctrl, name, None)
            if value:
                has_container = True
                break
        if not has_container:
            return None, None, "unavailable: ruleset packet containers are absent or empty"
        reference_mod = importlib.import_module("lechonk.adapter.ruleset_reference")
        builder = getattr(reference_mod, "build_ruleset_reference", None)
        if builder is None:
            builder = getattr(reference_mod, "extract_ruleset_reference")
        reference = builder(rule_ctrl)
    except Exception as exc:
        return None, None, f"unavailable: {exc}"
    if reference is None:
        return None, None, "unavailable: builder returned no reference"
    digest = getattr(reference, "reference_hash", None)
    if digest is None:
        digest = getattr(reference, "sha256", None)
    if digest is None:
        digest = getattr(reference, "content_hash", None)
    if digest is not None and not isinstance(digest, str):
        return None, None, "unavailable: reference hash is not a string"
    return reference, digest, "complete"


def _collect_help_reference() -> tuple[Any, Optional[str], str]:
    """Load only the provisioned, repository-local Freeciv helpdata file.

    The provision script extracts this optional artifact from the pinned game
    image.  A fixed path is intentional: startup never follows a model- or
    environment-supplied arbitrary filesystem path.
    """
    try:
        help_mod = importlib.import_module("lechonk.adapter.help_reference")
        loader = getattr(help_mod, "load_helpdata")
        path = Path(__file__).resolve().parents[3] / "artifacts" / "reference" / "freeciv" / "helpdata.txt"
        reference = loader(path)
    except Exception as exc:
        return None, None, f"unavailable: {exc}"
    status = getattr(reference, "status", "unavailable")
    digest = getattr(reference, "content_hash", None) if status == "available" else None
    return reference, digest, str(status)


def _classic_ruleset_name(env: Any) -> Optional[str]:
    """Return the server-reported ruleset identity, when present."""
    rule_ctrl = getattr(getattr(env, "civ_controller", None), "rule_ctrl", None)
    control = getattr(rule_ctrl, "ruleset_control", None)
    if isinstance(control, Mapping):
        value = control.get("name") or control.get("ruleset")
    else:
        value = getattr(control, "name", None) or getattr(control, "ruleset", None)
    if not isinstance(value, str) or not value.strip():
        return None
    # Freeciv's packet uses display names (currently ``Classic ruleset``),
    # while configuration and the extracted tree use the directory name.
    # Normalize only the official display suffix; do not accept arbitrary
    # names containing the word "classic".
    normalized = " ".join(value.casefold().split())
    if normalized.endswith(" ruleset"):
        normalized = normalized[:-len(" ruleset")].rstrip()
    return normalized


def _collect_classic_reference(env: Any) -> tuple[Any, Optional[str], str, Any]:
    """Load the optional static classic index only for the classic ruleset.

    The live RulesetCtrl snapshot remains authoritative.  This index is only a
    supplementary, repository-provisioned encyclopedia and must never be
    reported as classic when the server did not identify itself as classic.
    """
    active = _classic_ruleset_name(env)
    if active is None:
        return None, None, "unavailable: active ruleset identity was not reported", None
    if active != "classic":
        return None, None, f"not_active: active ruleset is {active}", None
    path = Path(__file__).resolve().parents[3] / "artifacts" / "reference" / "freeciv"
    try:
        module = importlib.import_module("lechonk.adapter.classic_reference")
        loader = getattr(module, "load_classic_reference", None)
        if loader is None:
            loader = getattr(module, "load_classic_index")
        reference = loader(path)
    except Exception as exc:
        return None, None, f"unavailable: {exc}", None
    status = str(getattr(reference, "status", "available"))
    if status not in ("available", "complete"):
        return reference, None, status, None
    digest = (getattr(reference, "content_hash", None)
              or getattr(reference, "reference_hash", None)
              or getattr(reference, "sha256", None))
    if digest is not None and not isinstance(digest, str):
        return None, None, "unavailable: classic reference hash is not a string", None
    provenance = getattr(reference, "provenance", None)
    if provenance is None:
        # Keep provenance explicit even though the reference object is a
        # frozen value object.  ``tree_sha256`` is the integrity identity of
        # the extracted source tree (and is intentionally not the semantic
        # content hash of the parsed index).
        provenance = {
            key: value for key in ("source_path", "image_ref", "image_version",
                                   "tree_sha256")
            if (value := getattr(reference, key, None)) not in (None, "")
        }
    return reference, digest, status, provenance


def _collect_manual_reference(env: Any) -> tuple[Any, Optional[str], str, Any]:
    """Load the provisioned, read-only Freeciv manual for classic games.

    The generated manual is deliberately optional startup metadata.  It is
    only eligible when the server identifies the active ruleset as exactly
    ``classic``; missing, stale, or malformed artifacts must never prevent a
    playable session from starting.  The loader owns manifest/hash
    verification, while this boundary keeps its provenance explicit.
    """
    active = _classic_ruleset_name(env)
    if active is None:
        return None, None, "unavailable: active ruleset identity was not reported", None
    if active != "classic":
        return None, None, f"not_active: active ruleset is {active}", None
    path = Path(__file__).resolve().parents[3] / "artifacts" / "reference" / "freeciv-manual"
    try:
        module = importlib.import_module("lechonk.adapter.manual_reference")
        loader = getattr(module, "load_manual_reference", None)
        if loader is None:
            loader = getattr(module, "load_manual_index", None)
        if loader is None:
            return None, None, "unavailable: manual reference loader is absent", None
        reference = loader(path)
    except Exception as exc:
        return None, None, f"unavailable: {exc}", None
    if reference is None:
        return None, None, "unavailable: loader returned no reference", None
    status = str(getattr(reference, "status", "available"))
    if status not in ("available", "complete"):
        return reference, None, status, None
    digest = (getattr(reference, "content_hash", None)
              or getattr(reference, "reference_hash", None)
              or getattr(reference, "sha256", None))
    if digest is not None and not isinstance(digest, str):
        return None, None, "unavailable: manual reference hash is not a string", None
    provenance = getattr(reference, "provenance", None)
    if provenance is None:
        provenance = {
            key: value for key in ("source_path", "image_ref", "image_version",
                                   "generated_by", "manifest_sha256")
            if (value := getattr(reference, key, None)) not in (None, "")
        }
    return reference, digest, status, provenance


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
        max_turns: Optional[int] = None,
        acting_deadline_s: Optional[float] = None,
        load_game: Optional[str] = None,
    ):
        self._workdir = os.path.abspath(workdir)
        self._seed = int(seed)
        self._username = validate_freeciv_username(username)
        self._aifill = int(aifill)
        self.startup_deadline = float(startup_deadline)
        self.step_deadline = float(step_deadline)
        self.wait_deadline = float(wait_deadline)
        self.close_deadline = float(close_deadline)
        if max_turns is not None:
            # Keep the session boundary unambiguous for direct callers too;
            # silently coercing 0, negatives, floats, or bools would either
            # disable the intended horizon or produce an invalid controller
            # guard.
            if isinstance(max_turns, bool) or not isinstance(max_turns, int):
                raise ValueError("max_turns must be a positive integer or None")
            if max_turns <= 0:
                raise ValueError("max_turns must be a positive integer or None")
        self.max_turns = max_turns
        if load_game is not None:
            if not isinstance(load_game, str) or not load_game.strip():
                raise ValueError("load_game must be a non-empty save name")
            if any(part in load_game for part in ("/", "\\", "..")):
                raise ValueError("load_game must be a save name, not a path")
            if not load_game.startswith(f"{username}_"):
                raise ValueError("load_game save name must belong to username")
        self.load_game = load_game
        if acting_deadline_s is not None:
            acting_deadline_s = float(acting_deadline_s)
            if not math.isfinite(acting_deadline_s) or acting_deadline_s <= 0:
                raise ValueError("acting_deadline_s must be a finite positive number")
        self.acting_deadline_s = acting_deadline_s
        # CivRealm's controller-side server timeout must exceed the runner's
        # model/action phase deadline.  Keep the historical 300s bounded-run
        # default when no runner deadline is supplied.
        self.server_timeout_s = max(
            float(SERVER_TIMEOUT_S),
            acting_deadline_s + SERVER_TIMEOUT_MARGIN_S
            if acting_deadline_s is not None else float(SERVER_TIMEOUT_S),
        )
        # CivRealm evaluates its turn guard after advancing the server turn.
        # Give bounded runs one extra server turn so the runner can own the
        # exact N-turn completion boundary.
        self.controller_max_turns = (
            UNBOUNDED_MAX_TURNS
            if self.max_turns is None else self.max_turns + 1
        )
        # The value is configured before reset(), while CivRealm's setup I/O
        # loop is active.  After reset, the typed server-settings packet is
        # inspected to verify that Freeciv accepted the value.
        # The websocket liveness window must cover the complete model/action
        # phase whenever a caller supplies an acting deadline.  Keep the
        # upstream 720-second default for ordinary bounded callers, but leave
        # a cleanup margin for longer configured phases.
        default_pingtimeout = PINGTIMEOUT_S if self.max_turns is None else 720.0
        self.pingtimeout_s = max(
            default_pingtimeout,
            acting_deadline_s + SERVER_TIMEOUT_MARGIN_S
            if acting_deadline_s is not None else 720.0,
        )
        self._pingtimeout_status = "configured"
        self._pingtimeout_actual: Optional[int] = None
        # Scenario read-back is best-effort.  Older CivRealm builds and test
        # doubles may not expose either controller, so startup must remain
        # compatible while clearly marking the requested values as unverified.
        self._scenario_readback_status = "unavailable"
        self._aifill_actual: Optional[int] = None
        self._maxplayers_actual: Optional[int] = None
        self._actual_total_players: Optional[int] = None
        # ``actual_ai_players`` is the raw PLRF_AI count.  Keep it separate
        # from ``actual_ai_opponents``, which deliberately excludes Freeciv's
        # Barbarian/Pirate nations.
        self._actual_ai_players: Optional[int] = None
        self._actual_ai_opponents: Optional[int] = None
        self._actual_barbarian_ai_opponents: Optional[int] = None
        self._actual_unknown_ai_opponents: Optional[int] = None
        self._actual_player_roster: Optional[list[Mapping[str, Any]]] = None

        self._commands: "queue.Queue[tuple[str, Any, queue.Queue]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._last_dispatch_failure: Optional[dict[str, Any]] = None
        self._startup_cleanup_error: Optional[BaseException] = None
        self._startup_env_closed = False

        # env-side handles (set by the worker, read cross-thread after steps)
        self._env: Any = None
        self._obs: Mapping[str, Any] = {}
        self._info: Mapping[str, Any] = {}
        self._turn = 0
        self._player_id: Optional[int] = None
        self._port: Optional[int] = None
        self._ruleset: Optional[RulesetNames] = None
        self._ruleset_reference: Any = None
        self._ruleset_reference_hash: Optional[str] = None
        self._ruleset_reference_status = "not_collected"
        self._help_reference: Any = None
        self._help_reference_hash: Optional[str] = None
        self._help_reference_status = "not_collected"
        self._classic_reference: Any = None
        self._classic_reference_hash: Optional[str] = None
        self._classic_reference_status = "not_collected"
        self._classic_reference_provenance: Any = None
        self._manual_reference: Any = None
        self._manual_reference_hash: Optional[str] = None
        self._manual_reference_status = "not_collected"
        self._manual_reference_provenance: Any = None
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

        # CivRealm's websocket captures the asyncio/Tornado loop created by
        # connect(). Keep those exact objects so a missing thread-local loop
        # can be restored without replacing a live websocket's loop.
        self._adapter_asyncio_loop: Any = None
        self._adapter_io_loop: Any = None
        self._adapter_socket_asyncio_loop: Any = None

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
            if isinstance(self._startup_error, SessionStateError):
                raise self._startup_error
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
    def pingtimeout_status(self) -> str:
        """Configuration/read-back status for the long-run pingtimeout."""
        return self._pingtimeout_status

    @property
    def pingtimeout_actual(self) -> Optional[int]:
        """The integer value reported by Freeciv after setup, if confirmed."""
        return self._pingtimeout_actual

    @property
    def scenario_readback_status(self) -> str:
        """Whether authoritative scenario settings/roster were observed."""
        return self._scenario_readback_status

    @property
    def aifill_actual(self) -> Optional[int]:
        return self._aifill_actual

    @property
    def maxplayers_actual(self) -> Optional[int]:
        return self._maxplayers_actual

    @property
    def actual_total_players(self) -> Optional[int]:
        return self._actual_total_players

    @property
    def actual_ai_opponents(self) -> Optional[int]:
        return self._actual_ai_opponents

    @property
    def actual_ai_players(self) -> Optional[int]:
        """Raw count of players carrying Freeciv's AI flag."""
        return self._actual_ai_players

    @property
    def actual_barbarian_ai_opponents(self) -> Optional[int]:
        """Count of AI players whose nation is Barbarian or Pirate."""
        return self._actual_barbarian_ai_opponents

    @property
    def actual_unknown_ai_opponents(self) -> Optional[int]:
        """Count of AI players whose nation classification is unavailable."""
        return self._actual_unknown_ai_opponents

    @property
    def actual_player_roster(self) -> Optional[list[Mapping[str, Any]]]:
        return self._actual_player_roster

    @property
    def ruleset(self) -> Optional[RulesetNames]:
        return self._ruleset

    @property
    def ruleset_reference(self) -> Any:
        """Immutable ruleset encyclopedia, when available.

        This is separate from ``ruleset``: the latter is the legacy display
        name snapshot consumed by action validation.  Missing or incomplete
        reference data is represented by ``None`` and does not prevent a game
        from starting.
        """
        return self._ruleset_reference

    @property
    def ruleset_reference_json(self) -> Optional[Mapping[str, Any]]:
        """JSON-native snapshot of the encyclopedia, when available.

        ``RulesetReference`` remains the immutable internal/public reference;
        this property is the explicit boundary for serializers and prompt
        code.  ``to_dict`` returns a fresh JSON-native snapshot each access.
        """
        reference = self._ruleset_reference
        if reference is None:
            return None
        to_dict = getattr(reference, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            return value if isinstance(value, Mapping) else None
        return reference if isinstance(reference, Mapping) else None

    @property
    def ruleset_reference_hash(self) -> Optional[str]:
        """SHA-256 content hash of ``ruleset_reference``, if available."""
        return self._ruleset_reference_hash

    @property
    def ruleset_reference_status(self) -> str:
        """Reference extraction status and any unavailable explanation."""
        return self._ruleset_reference_status

    @property
    def help_reference(self) -> Any:
        """Immutable static help index, or an unavailable-status reference."""
        return self._help_reference

    @property
    def help_reference_json(self) -> Optional[Mapping[str, Any]]:
        """JSON-native help index snapshot for serializers and prompts."""
        reference = self._help_reference
        if reference is None:
            return None
        to_dict = getattr(reference, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        elif hasattr(reference, "entries"):
            # ClassicReference deliberately keeps its value object minimal;
            # provide the documented JSON boundary here without exposing the
            # frozen object's internals or private hash field.
            value = {
                key: getattr(reference, key)
                for key in ("schema_version", "status", "source_path",
                            "image_ref", "image_version", "tree_sha256")
                if hasattr(reference, key)
            }
            value["content_hash"] = getattr(reference, "content_hash", None)
            value["entries"] = [
                entry.to_dict() if callable(getattr(entry, "to_dict", None))
                else dict(entry) if isinstance(entry, Mapping) else entry
                for entry in getattr(reference, "entries", ())
            ]
        else:
            value = reference
        return value if isinstance(value, Mapping) else None

    @property
    def help_reference_hash(self) -> Optional[str]:
        """Semantic content hash, only when helpdata was available."""
        return self._help_reference_hash

    @property
    def help_reference_status(self) -> str:
        """Helpdata availability status."""
        return self._help_reference_status

    @property
    def classic_reference(self) -> Any:
        """Optional static classic encyclopedia, never authoritative."""
        return self._classic_reference

    @property
    def classic_reference_json(self) -> Optional[Mapping[str, Any]]:
        reference = self._classic_reference
        if reference is None:
            return None
        to_dict = getattr(reference, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        elif hasattr(reference, "entries"):
            value = {
                key: getattr(reference, key)
                for key in ("schema_version", "status", "source_path",
                            "image_ref", "image_version", "tree_sha256")
                if hasattr(reference, key)
            }
            value["content_hash"] = getattr(reference, "content_hash", None)
            value["entries"] = [
                entry.to_dict() if callable(getattr(entry, "to_dict", None))
                else dict(entry) if isinstance(entry, Mapping) else entry
                for entry in getattr(reference, "entries", ())
            ]
        else:
            value = reference
        return value if isinstance(value, Mapping) else None

    @property
    def classic_reference_hash(self) -> Optional[str]:
        return self._classic_reference_hash

    @property
    def classic_reference_status(self) -> str:
        return self._classic_reference_status

    @property
    def classic_reference_provenance(self) -> Any:
        """JSON-safe provenance metadata, if the static index was loaded."""
        value = self._classic_reference_provenance
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @property
    def manual_reference(self) -> Any:
        """Optional generated Freeciv manual, never authoritative game state."""
        return self._manual_reference

    @property
    def manual_reference_json(self) -> Optional[Mapping[str, Any]]:
        """JSON-native manual index snapshot, when the verified artifact exists."""
        reference = self._manual_reference
        if reference is None:
            return None
        to_dict = getattr(reference, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        elif isinstance(reference, Mapping):
            value = reference
        else:
            value = None
        return value if isinstance(value, Mapping) else None

    @property
    def manual_reference_hash(self) -> Optional[str]:
        """Semantic content hash of the verified generated manual."""
        return self._manual_reference_hash

    @property
    def manual_reference_status(self) -> str:
        """Manual availability or non-fatal unavailability explanation."""
        return self._manual_reference_status

    @property
    def manual_reference_provenance(self) -> Any:
        """JSON-safe provenance for the generated manual, if available."""
        value = self._manual_reference_provenance
        return dict(value) if isinstance(value, Mapping) else value

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

    @property
    def last_dispatch_failure(self) -> Optional[Mapping[str, Any]]:
        """Bounded worker-side failure diagnostic, if the last RPC failed.

        This intentionally excludes action arguments and model output.  The
        runner uses the transport marker to avoid a same-session refresh when
        a websocket callback failed: that connection's mutation state is not
        safely reconnectable.
        """
        if self._last_dispatch_failure is None:
            return None
        return dict(self._last_dispatch_failure)

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
            # reset() may already have opened a websocket before a startup
            # verifier (including loop identity checks) failed.  Close that
            # env on this owning worker thread, preserving the original
            # startup exception if cleanup itself fails.
            env = self._env
            if env is not None and not self._startup_env_closed:
                try:
                    # Route cleanup through the sole guarded close dispatch;
                    # this preserves the package invariant that no caller
                    # invokes env.close() outside ``_dispatch('close')``.
                    self._dispatch("close", None)
                except BaseException as cleanup_exc:
                    self._startup_cleanup_error = cleanup_exc
                finally:
                    self._startup_env_closed = True
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
                # Keep a bounded, non-payload diagnostic for reconciliation
                # and failure artifacts.  Never include action arguments or
                # model text; the exception type/message and traceback are
                # sufficient to locate loop lifecycle failures.
                self._last_dispatch_failure = {
                    "stage": kind,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                    "worker_thread": threading.current_thread().name,
                    "owned_loop": (hex(id(self._adapter_asyncio_loop))
                                    if self._adapter_asyncio_loop is not None
                                    else None),
                    "transport_uncertain": bool(
                        getattr(exc, "diagnostic", None) is not None
                        or type(exc).__name__ == "WebSocketWriteError"
                        or "websocket" in str(exc).lower()
                    ),
                }
                transport_diagnostic = getattr(exc, "diagnostic", None)
                if isinstance(transport_diagnostic, Mapping):
                    # Keep only bounded scalar provenance/loop fields.  Do not
                    # persist payloads or arbitrary exception objects.
                    self._last_dispatch_failure["transport_diagnostic"] = {
                        str(key): (str(value)[-4096:] if key == "traceback"
                                   else value)
                        for key, value in transport_diagnostic.items()
                        if isinstance(value, (str, int, float, bool, type(None), dict))
                    }
                result_q.put(("error", exc))
            else:
                # A later successful RPC proves that the previous dispatch
                # diagnostic no longer describes the current operation.  Do
                # this only after publishing the failure above, so the
                # runner can inspect it while reconciling that failed step.
                self._last_dispatch_failure = None
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
        fc_args["server_timeout"] = self.server_timeout_s
        fc_args["begin_turn_timeout"] = BEGIN_TURN_TIMEOUT_S
        fc_args["wait_for_timeout"] = WAIT_FOR_TIMEOUT_S
        # CivRealm has an independent controller-side turn guard.  Keep it
        # above the runner's horizon. Bounded runs use one extra server turn
        # because CivRealm evaluates this guard after advancing the turn.
        fc_args["max_turns"] = self.controller_max_turns
        fc_args["pingtimeout"] = math.ceil(self.pingtimeout_s)
        # CivRealm consumes this before env.reset() and performs the load
        # while its setup I/O loop is active. The caller must stage and
        # checksum-validate the save before constructing this session.
        fc_args["debug.load_game"] = self.load_game or ""

        # Wrap every upstream no-timeout HTTP call at <= 15 s BEFORE the env
        # is created (login's create_pbem_user/gamesetting run inside reset).
        _install_http_timeout_guard(HTTP_TIMEOUT_S)

        envs = importlib.import_module("civrealm.envs")
        env = envs.FreecivBaseEnv(username=self._username)
        # Publish immediately: reset and post-reset verification may fail
        # after the websocket/server has already been created.
        self._env = env
        obs, info = env.reset(seed=self._seed)
        self._capture_event_loop_identity(env)
        self._verify_pingtimeout(env, fc_args["pingtimeout"])
        self._collect_scenario_readback(env)
        self._obs = _snapshot(obs)
        self._info = _snapshot(info)
        self._turn = int(info.get("turn", 0) or 0)
        self._player_id = int(env.get_playerid())
        self._port = int(env.get_port())
        self._ruleset = _collect_ruleset(env)
        (self._ruleset_reference, self._ruleset_reference_hash,
         self._ruleset_reference_status) = _collect_ruleset_reference(env)
        (self._help_reference, self._help_reference_hash,
         self._help_reference_status) = _collect_help_reference()
        (self._classic_reference, self._classic_reference_hash,
         self._classic_reference_status,
         self._classic_reference_provenance) = _collect_classic_reference(env)
        (self._manual_reference, self._manual_reference_hash,
         self._manual_reference_status,
         self._manual_reference_provenance) = _collect_manual_reference(env)

    def _verify_pingtimeout(self, env: Any, expected: int) -> None:
        """Verify Freeciv's typed server-settings read-back after reset.

        Packet 167 updates ``opt_ctrl.server_settings`` in place.  This is
        the authoritative value available to the adapter; a chat ``/set``
        line alone is not an acknowledgement.  Older controller schemas may
        omit ``is_visible``, so enforce it only when present.
        """
        controller = getattr(env, "civ_controller", None)
        options = getattr(getattr(controller, "opt_ctrl", None),
                          "server_settings", None)
        setting = options.get("pingtimeout") if isinstance(options, dict) else None
        if not isinstance(setting, dict) or "val" not in setting:
            raise SessionStateError(
                "server did not report pingtimeout in typed server settings"
            )
        if "is_visible" in setting and setting["is_visible"] is not True:
            raise SessionStateError(
                "server pingtimeout setting is not visible in typed server settings"
            )
        actual = setting["val"]
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise SessionStateError(
                f"server pingtimeout mismatch: expected {expected}, got {actual!r}"
            )
        self._pingtimeout_actual = actual
        self._pingtimeout_status = "confirmed"

    def _capture_event_loop_identity(self, env: Any) -> None:
        """Capture the loop already owned by the CivRealm websocket."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as exc:
            raise SessionStateError(
                "Freeciv worker has no current asyncio loop after reset"
            ) from exc
        if loop.is_closed() or loop.is_running():
            raise SessionStateError(
                "Freeciv worker asyncio loop is closed or running after reset"
            )
        controller = getattr(env, "civ_controller", None)
        socket = getattr(controller, "ws_client", None)
        io_loop = getattr(socket, "_io_loop", None)
        self._adapter_asyncio_loop = loop
        if io_loop is None:
            # Older controller schemas and test doubles do not expose loop
            # identity; the asyncio guard remains useful and compatible.
            return
        socket_loop = getattr(io_loop, "asyncio_loop", None)
        if socket_loop is None:
            socket_loop = getattr(io_loop, "_asyncio_loop", None)
        if socket_loop is not None and socket_loop is not loop:
            raise SessionStateError(
                "Freeciv websocket loop does not match worker asyncio loop"
            )
        self._adapter_io_loop = io_loop
        self._adapter_socket_asyncio_loop = socket_loop

    def _ensure_event_loop_identity(self) -> None:
        """Ensure the worker uses the websocket's captured loop for an RPC."""
        loop = self._adapter_asyncio_loop
        if loop is None:
            return
        if loop.is_closed():
            raise SessionStateError(
                "Freeciv websocket asyncio loop is closed "
                f"(owned={hex(id(loop))}, stage=dispatch)")
        if loop.is_running():
            raise SessionStateError(
                "Freeciv websocket asyncio loop is unexpectedly running "
                f"(owned={hex(id(loop))}, stage=dispatch)"
            )
        try:
            current = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(loop)
            current = loop
        if current is not loop:
            raise SessionStateError(
                "Freeciv worker asyncio loop changed while websocket was active "
                f"(owned={hex(id(loop))}, current={hex(id(current))}, "
                "stage=dispatch)"
            )
        if self._adapter_io_loop is not None:
            is_closed = getattr(self._adapter_io_loop, "closed", None)
            if callable(is_closed):
                is_closed = is_closed()
            if is_closed is True:
                raise SessionStateError("Freeciv websocket IOLoop is closed")
            socket_loop = getattr(self._adapter_io_loop, "asyncio_loop", None)
            if socket_loop is None:
                socket_loop = getattr(self._adapter_io_loop, "_asyncio_loop", None)
            if (self._adapter_socket_asyncio_loop is not None and
                    socket_loop is not self._adapter_socket_asyncio_loop):
                raise SessionStateError(
                    "Freeciv websocket IOLoop changed while connection was active"
                )

    def _collect_scenario_readback(self, env: Any) -> None:
        """Collect typed ``aifill``/``maxplayers`` and player roster data.

        This is intentionally non-fatal: these fields were not exposed by
        some older controller versions.  Values are included only when they
        are typed and therefore useful; a missing value remains unavailable
        instead of being mistaken for the requested configuration.
        """
        controller = getattr(env, "civ_controller", None)
        options = getattr(getattr(controller, "opt_ctrl", None),
                          "server_settings", None)
        setting_values: dict[str, int] = {}
        if isinstance(options, dict):
            for name in ("aifill", "maxplayers"):
                setting = options.get(name)
                value = setting.get("val") if isinstance(setting, dict) else None
                if (isinstance(value, int) and not isinstance(value, bool)
                        and value >= 0):
                    setting_values[name] = value
        self._aifill_actual = setting_values.get("aifill")
        self._maxplayers_actual = setting_values.get("maxplayers")

        player_ctrl = getattr(controller, "player_ctrl", None)
        players = getattr(player_ctrl, "players", None)
        rule_ctrl = getattr(controller, "rule_ctrl", None)
        nations = getattr(rule_ctrl, "nations", None)
        if not isinstance(nations, Mapping) and isinstance(rule_ctrl, Mapping):
            nations = rule_ctrl.get("nations")
        if not isinstance(nations, Mapping):
            nations = {}
        roster: list[Mapping[str, Any]] = []
        ai_count = 0
        barbarian_ai_count = 0
        unknown_ai_count = 0

        def nation_rule_name(nation_id: Any) -> Optional[str]:
            """Resolve the authoritative ruleset nation name, if present."""
            candidates = [nation_id]
            try:
                candidates.append(int(nation_id))
            except (TypeError, ValueError):
                pass
            for candidate in candidates:
                try:
                    nation = nations.get(candidate)
                except (TypeError, AttributeError):
                    nation = None
                if isinstance(nation, Mapping):
                    value = nation.get("rule_name")
                    if isinstance(value, str) and value:
                        return value
            return None

        if isinstance(players, dict):
            for playerno, player in sorted(players.items(), key=lambda item: str(item[0])):
                if not isinstance(player, dict):
                    continue
                flags = player.get("flags")
                is_ai: Optional[bool] = None
                try:
                    is_ai = bool(flags[0]) if flags is not None else None
                except (IndexError, KeyError, TypeError):
                    is_ai = None
                if is_ai is True:
                    ai_count += 1
                rule_name = nation_rule_name(player.get("nation"))
                is_barbarian: Optional[bool] = None
                if rule_name is not None:
                    is_barbarian = rule_name.lower() in {"barbarian", "pirate"}
                    if is_ai is True and is_barbarian:
                        barbarian_ai_count += 1
                elif is_ai is True:
                    unknown_ai_count += 1
                entry: dict[str, Any] = {
                    "playerno": player.get("playerno", playerno),
                    "name": player.get("name"),
                    "username": player.get("username"),
                    "nation": player.get("nation"),
                }
                if is_ai is not None:
                    entry["is_ai"] = is_ai
                if rule_name is not None:
                    entry["nation_rule_name"] = rule_name
                    entry["is_barbarian"] = is_barbarian
                roster.append(entry)
        if roster:
            self._actual_total_players = len(roster)
            all_ai_known = all("is_ai" in item for item in roster)
            self._actual_ai_players = ai_count if all_ai_known else None
            self._actual_barbarian_ai_opponents = (
                barbarian_ai_count if all_ai_known else None)
            self._actual_unknown_ai_opponents = (
                unknown_ai_count if all_ai_known else None)
            self._actual_ai_opponents = (
                ai_count - barbarian_ai_count - unknown_ai_count
                if all_ai_known and unknown_ai_count == 0 else None)
            self._actual_player_roster = roster
        complete_settings = self._aifill_actual is not None and self._maxplayers_actual is not None
        complete_roster = bool(roster) and self._aifill_actual is not None and len(roster) == self._aifill_actual
        complete_ai_classification = (
            bool(roster) and self._actual_unknown_ai_opponents == 0 and
            self._actual_ai_opponents is not None)
        if complete_settings and complete_roster and complete_ai_classification:
            self._scenario_readback_status = "confirmed"
        elif complete_settings or roster:
            self._scenario_readback_status = "partial"

    def _dispatch(self, kind: str, payload: Any) -> Any:
        self._ensure_event_loop_identity()
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
