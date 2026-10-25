# SPDX-License-Identifier: MIT
"""WP3 package B shared scaffolding for the LIVE lifecycle tests
(scripts/wp3_teardown_test.py, scripts/wp3_disconnect_test.py) against
the real stack.

Everything here is test scaffolding — the runner, the ledger and the
adapter session stay 100% real; only the disconnect injection (SIGSTOP /
SIGKILL of OUR OWN game's server process inside the container) and the
model client (a deterministic scripted picker — no API call) are
synthetic.

Injection model (one env per process, plan §7.1 — one case per process
run):

- ``stop_on_close``: close() SIGSTOPs our game's freeciv-web server
  first, then delegates to the real guarded close. The connection stays
  alive but the server is frozen: the upstream CivController.close()
  game-over wait (civ_controller.py:578-583) blocks → the §3 teardown
  deadline fires → the skip path (AdapterSession.close). This is the
  deterministic supervisor-deadline proof.
- ``kill_on_finish``: the first step(None) (the pipeline-owned finish
  primitive) SIGKILLs our game's server (blocking, before delegating),
  then delegates. The phase_done is dispatched into a just-dropped
  connection and the response never arrives → the step deadline fires →
  bounded infrastructure TRUNCATED (the finish is not a mutation: no
  reconciliation).
- ``kill_on_mutation``: the first mutation step SIGKILLs our game's
  server (blocking, before delegating), then delegates. The mutation is
  dispatched into a just-dropped connection and its read-back never
  arrives → the §3 line 35 uncertain-mutation reconciliation runs
  against the real session (the step timeout marks the session failed,
  so the reconnect read is refused) → unknown → quarantine → bounded
  fail-stop.

The kills are SYNCHRONOUS (the signal completes before the step is
delegated), which makes the injection deterministic: the server is
provably dead before the operation's response could arrive. The kill
helpers find OUR game's server precisely: the freeciv-web process whose
args contain ``--port <our session port>`` (the session's own fresh
game — the turn-11 checkpoint and any pre-existing game are never
touched; the server is killed by port match only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PUBSTATUS = "http://localhost:8080/pubstatus"
CONTAINER = "freeciv-web"

# Decision-8 manifest constants (the same stack pins as the WP2 runs).
IMAGE_DIGEST = ("sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a"
                "8e943faf920cb373a")
VENDORED_BASE_SHA = "1b144d6ec92a8aaa8e30eee64c881418208a4689"  # upstream dev
PATCH_HEAD_SHA = "36ae456836015b22b086298430048fc51c317d50"  # PR #7 fork head
CAPSTRING = "+Freeciv.Web.Devel-3.3"

# The turn-11 checkpoint of record (checksum per the handoff devlog) — the
# lifecycle tests
# never load it; its checksum is re-verified as the untouched proof.
TURN11_CHECKPOINT = ("artifacts/checkpoints/"
                     "myagent_T11_2026-10-24-10_24.sav.xz")
TURN11_SHA256 = ("1c24f05be17edfe640a2e4246369f2d8a5a3d1daaf66c330be82ef2a"
                 "36ddf5aa")

DEFAULT_PICK_RE = re.compile(r"^(unit:\d+:goto_\w+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# run-dir logging / checks / persistence
# ---------------------------------------------------------------------------


class RunLogger:
    """Stdout + persisted run log + a checks registry (wp2_fault_run
    pattern)."""

    def __init__(self, rundir):
        self.rundir = str(rundir)
        os.makedirs(self.rundir, exist_ok=True)
        self.log_path = os.path.join(self.rundir, "run_stdout.log")
        # Fresh per run: a rerun replaces its own previous artifacts
        # (wp2_fault_run's fresh_artifact pattern).
        self._fh = open(self.log_path, "w", encoding="utf-8")
        self.checks: dict = {}

    def log(self, msg):
        line = (f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}]"
                f" {msg}")
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def check(self, name, cond, detail=""):
        self.checks[name] = {"pass": bool(cond), "detail": detail}
        self.log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    def write_json(self, name, obj):
        path = os.path.join(self.rundir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False,
                      default=str)
            fh.write("\n")
        self.log(f"wrote {name}")
        return path

    def verdict(self):
        return ("PASS" if all(c["pass"] for c in self.checks.values())
                else "FAIL")

    def close(self):
        self._fh.close()


def wait_for_stack(timeout_s: float = 150.0) -> bool:
    """Bounded pubstatus retries (hardened startup, P3)."""
    deadline = time.time() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            if time.time() > deadline:
                return False
            time.sleep(2)


def harness_revision() -> str:
    """The repo is NOT a git checkout (HANDOFF constraints): the harness
    revision is a deterministic content hash over src/lechonk."""
    root = Path(REPO) / "src" / "lechonk"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = str(path.relative_to(root))
            lines.append(f"{rel} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    blob = "\n".join(lines).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def turn11_checkpoint_untouched() -> tuple[bool, str]:
    """Re-verify the turn-11 checkpoint's recorded sha256 (handoff devlog)."""
    path = Path(REPO) / TURN11_CHECKPOINT
    if not path.exists():
        return False, f"missing: {TURN11_CHECKPOINT}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == TURN11_SHA256, (
        f"sha256={digest} matches the HANDOFF-recorded checksum "
        f"{TURN11_SHA256}: {digest == TURN11_SHA256}")


# ---------------------------------------------------------------------------
# the container-side kill helpers (our own game's server, by port)
# ---------------------------------------------------------------------------


def _docker_exec(args: str, timeout: float = 30.0):
    return subprocess.run(
        ["sg", "docker", "-c", f"docker exec {CONTAINER} {args}"],
        capture_output=True, text=True, timeout=timeout)


def find_our_server_pid(port: int):
    """The pid of OUR game's server binary inside the container (the
    freeciv-web process whose args carry ``--port <port>``), or None."""
    r = _docker_exec(f"pgrep -f {port}")
    for pid in r.stdout.split():
        r2 = _docker_exec(f"ps -o args= -p {pid}")
        args = r2.stdout
        if "freeciv-web" in args and f"--port {port}" in args:
            return pid
    return None


def signal_our_server(port: int, sig: str):
    """Signal OUR game's server (STOP / CONT / KILL). Returns the pid
    signaled (or None)."""
    pid = find_our_server_pid(port)
    if pid is None:
        return None
    _docker_exec(f"kill -{sig} {pid}")
    return pid


def kill_our_server(port: int):
    """SIGKILL our own game's server (cleanup; the fresh game is ours)."""
    return signal_our_server(port, "KILL")


def pid_alive(pid) -> bool:
    """Whether a container pid still exists."""
    if pid is None:
        return False
    r = _docker_exec(f"ps -o stat= -p {pid}")
    return r.returncode == 0 and bool(r.stdout.strip())


def cleanup_own_game(logger, wrapper) -> dict:
    """Cleanup after a lifecycle run: resume any SIGSTOPped server (so
    it can observe the client leaving and exit) and kill our own game's
    server pid if it is still alive. The publite2 pool may respawn a
    fresh PREGAME server on the port afterwards — that is the stack's
    normal pool state (the same as the pre-existing 6000-6003 ports),
    not our game. Returns the cleanup evidence."""
    port = wrapper.identity_at_start["port"]
    signaled = [e for e in wrapper.kill_events if e.get("pid")]
    # Resume a frozen server first so it can exit on the client leaving.
    signal_our_server(port, "CONT")
    killed: list = []
    for event in signaled:
        pid = event["pid"]
        if pid_alive(pid):
            _docker_exec(f"kill -KILL {pid}")
            killed.append(pid)
    gone = all(not pid_alive(e["pid"]) for e in signaled)
    # Informational: the port's post-cleanup state (a respawned pregame
    # server is the pool's normal state).
    r = _docker_exec(f"pgrep -f {port}")
    remaining = r.stdout.split()
    evidence = {
        "port": port,
        "signaled_pids": [e["pid"] for e in signaled],
        "killed_at_cleanup": killed,
        "own_game_server_gone": gone,
        "port_pids_after_cleanup": remaining,
        "note": "a remaining pid is a fresh pregame server respawned by "
                "the publite2 pool (the stack's normal state, the same "
                "as the pre-existing ports) — not our game",
    }
    logger.log(f"[cleanup] {evidence}")
    return evidence


# ---------------------------------------------------------------------------
# the scripted model client (no API call)
# ---------------------------------------------------------------------------


class ScriptedClient:
    """A deterministic ModelClient: each turn's batch is the first unit
    goto candidate offered in the turn's candidate table (parsed from
    the rendered prompt). No API call is made."""

    def __init__(self, pick_re=DEFAULT_PICK_RE):
        self._pick_re = pick_re
        self.turn_prompts = []
        self.calls = 0

    def _response(self, prompt, batch):
        from lechonk.core.model_client import ModelResponse

        return ModelResponse(call_id=prompt.call_id, prompt_hash="scripted",
                             batch=batch, latency_ms=1, tokens_in=0,
                             tokens_out=0, cost_usd=0.0,
                             fingerprint="scripted-lifecycle")

    def complete_turn(self, prompt):
        from lechonk.core.schemas import ActionBatch, ActionItem

        self.calls += 1
        self.turn_prompts.append(prompt)
        match = self._pick_re.search(prompt.candidates)
        refs = [match.group(1)] if match else []
        batch = ActionBatch(turn_id=prompt.turn_id,
                            actions=[ActionItem(ref=r) for r in refs])
        return self._response(prompt, batch)

    def complete_repair(self, prompt):
        from lechonk.core.schemas import ActionBatch

        self.calls += 1
        return self._response(prompt, ActionBatch(turn_id=prompt.turn_id))


# ---------------------------------------------------------------------------
# the fault-injecting session wrapper
# ---------------------------------------------------------------------------


class _Seam:
    """The injection schedule for one lifecycle case."""

    def __init__(self, *, stop_on_close=False, kill_on_finish=False,
                 kill_on_mutation=False):
        self.stop_on_close = stop_on_close
        self.kill_on_finish = kill_on_finish
        self.kill_on_mutation = kill_on_mutation

    def describe(self):
        parts = []
        if self.stop_on_close:
            parts.append("close(): SIGSTOP our game's server before the "
                         "real guarded close (the upstream game-over wait "
                         "blocks → the §3 teardown deadline)")
        if self.kill_on_finish:
            parts.append("first step(None) (the finish primitive): SIGKILL "
                         "our game's server (synchronously, before "
                         "delegating) → the phase_done response never "
                         "arrives → the step deadline → bounded "
                         "infrastructure TRUNCATED")
        if self.kill_on_mutation:
            parts.append("first mutation step: SIGKILL our game's server "
                         "(synchronously, before delegating) → the "
                         "mutation's read-back never arrives → §3 line 35 "
                         "reconciliation (unknown → quarantine → bounded "
                         "fail-stop)")
        return "; ".join(parts)


class FaultInjectingSession:
    """Wraps the real AdapterSession (duck-typed delegation) and injects
    the disconnect at the scripted seam. The real session, the real
    env, the real guarded close — only the server process is signaled."""

    def __init__(self, inner, *, seam: _Seam):
        self._inner = inner
        self._seam = seam
        self._lock = threading.Lock()
        self._close_seam_fired = False
        self._finish_seam_fired = False
        self._mutation_seam_fired = False
        # evidence
        self.step_log: list = []
        self.kill_events: list = []
        self.close_calls = 0
        self.identity_at_start = None
        self.identity_at_end = None

    # -- delegation -------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def start(self):
        self._inner.start()
        self.identity_at_start = {
            "player_id": self._inner.player_id,
            "username": self._inner.username,
            "port": self._inner.port,
        }

    def _record_identity_end(self):
        try:
            self.identity_at_end = {
                "player_id": self._inner.player_id,
                "username": self._inner.username,
                "port": self._inner.port,
            }
        except Exception:
            self.identity_at_end = None

    # -- the seams --------------------------------------------------------

    def _kill_sync(self, why):
        """SIGKILL our game's server SYNCHRONOUSLY (the signal completes
        before the caller delegates the operation) — deterministic: the
        server is provably dead before the operation's response could
        arrive."""
        port = self._inner.port
        pid = signal_our_server(port, "KILL")
        self.kill_events.append({
            "why": why, "signal": "SIGKILL", "port": port, "pid": pid,
            "when": "synchronously before dispatching the operation (the "
                    "connection drops as the operation is sent)",
            "ts": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        })
        return pid

    # -- supervised operations ---------------------------------------------

    def step(self, action, *, deadline=None):
        self.step_log.append((action, deadline))
        if action is None and self._seam.kill_on_finish \
                and not self._finish_seam_fired:
            with self._lock:
                if not self._finish_seam_fired:
                    self._finish_seam_fired = True
                    self._kill_sync("finish primitive (step(None))")
        elif action is not None and self._seam.kill_on_mutation \
                and not self._mutation_seam_fired:
            with self._lock:
                if not self._mutation_seam_fired:
                    self._mutation_seam_fired = True
                    self._kill_sync(f"mutation {action!r}")
        return self._inner.step(action, deadline=deadline)

    def close(self):
        self.close_calls += 1
        if self._seam.stop_on_close and not self._close_seam_fired:
            with self._lock:
                if not self._close_seam_fired:
                    self._close_seam_fired = True
                    port = self._inner.port
                    pid = signal_our_server(port, "STOP")
                    self.kill_events.append({
                        "why": "teardown: the connection stays alive but "
                               "the server is frozen — the upstream "
                               "CivController.close() game-over wait "
                               "blocks → the §3 teardown deadline",
                        "signal": "SIGSTOP", "port": port, "pid": pid,
                        "ts": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"),
                    })
        status = self._inner.close()
        self._record_identity_end()
        return status


# ---------------------------------------------------------------------------
# the manifest (decision 8)
# ---------------------------------------------------------------------------


def build_manifest(*, run_id, seed, seam: _Seam, arm_start_ts):
    from lechonk.core.schemas import RunManifest

    fc_args = {
        "self_play": False, "aifill": 2,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": seed, "debug.gameseed": seed,
        "debug.agentseed": seed,
        "server_timeout": 300, "begin_turn_timeout": 900,
        "wait_for_timeout": 60, "http_timeout_s": 15.0,
    }
    manifest = RunManifest(
        run_id=run_id,
        image_digest=IMAGE_DIGEST,
        server_commit=None,  # not persisted by the WP0 spike (stdout-only
        # session); the image digest pins the server build — disclosed.
        capstring=CAPSTRING,
        vendored_base_sha=VENDORED_BASE_SHA,
        patch_head_sha=PATCH_HEAD_SHA,
        harness_revision=harness_revision(),
        python_lockfile=None,  # no lockfile in the repo (uv-managed venv)
        seeds={"mapseed": seed, "gameseed": seed, "agentseed": seed},
        ruleset="classic",
        server_settings={
            "fc_args_overrides": fc_args,
            "phasemode": "player", "timeout": 0, "pingtimeout": 720,
            "autotoggle": "disabled",
            "victories": "SPACERACE|ALLIED (CivRealm default)",
            "aifill": 2, "self_play": False,
        },
        model_id="scripted (WP3 lifecycle test — no model API call)",
        sampling={"client": "scripted first-unit-goto picker"},
        prompt_hashes={},
        config_after_load={
            "fresh_reset": True, "checkpoint_loaded": False,
            "fc_args": fc_args,
            "note": "fresh deterministic reset on the session's own server; "
                    "the turn-11 checkpoint is untouched",
        },
        checkpoint_checksums={},  # none: no checkpoint is loaded
        model_fingerprint=None,
        arm_start_ts=arm_start_ts,
    )
    manifest.server_commit_note = (
        "recorded live at P2 was stdout-only; not persisted — the image "
        "digest pins the server build")
    manifest.python_lockfile_note = (
        "no lockfile in the repo; the venv is uv-managed "
        f"(python {sys.version.split()[0]})")
    manifest.checkpoint_checksums_note = (
        "fresh deterministic reset; no checkpoint loaded (the turn-11 "
        "checkpoint is untouched)")
    manifest.lifecycle_test = {
        "purpose": "WP3 lifecycle hardening: live fault injection over "
                   "the real stack (runner + ledger + adapter session "
                   "100% real)",
        "injection": seam.describe(),
        "client": "ScriptedClient (deterministic first-unit-goto picker; "
                  "no API call)",
    }
    manifest.turn_11_checkpoint_untouched = True
    return manifest


# ---------------------------------------------------------------------------
# one lifecycle run over the real stack
# ---------------------------------------------------------------------------


def run_lifecycle_run(*, logger: RunLogger, run_id, rundir, seed, max_turns,
                      seam: _Seam, mutation_deadline_s=10.0,
                      waiting_deadline_s=15.0, teardown_deadline_s=10.0,
                      stall_watchdog_s=300.0):
    """Assemble and run ONE lifecycle run over the real stack. Returns
    the evidence dict (result, records, load, wrapper, timings)."""
    from lechonk.adapter.session import AdapterSession
    from lechonk.core.ledger import TranscriptLedger, load_transcript
    from lechonk.core.runner import GameRunner, RunnerConfig

    workdir = os.path.join(str(rundir), "workdir")
    os.makedirs(workdir, exist_ok=True)
    transcript_path = os.path.join(str(rundir), "transcript.jsonl")
    # Fresh transcript per run: the ledger appends, so a rerun must not
    # inherit the previous run's records (wp2_fault_run's fresh_artifact
    # pattern).
    if os.path.exists(transcript_path):
        os.remove(transcript_path)
    config = RunnerConfig(
        max_turns=max_turns, run_id=run_id, workdir=workdir, seed=seed,
        username="myagent", aifill=2,
        mutation_deadline_s=mutation_deadline_s,
        waiting_deadline_s=waiting_deadline_s,
        teardown_deadline_s=teardown_deadline_s,
        stall_watchdog_s=stall_watchdog_s)
    client = ScriptedClient()
    arm_start_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = build_manifest(run_id=run_id, seed=seed, seam=seam,
                             arm_start_ts=arm_start_ts)
    ledger = TranscriptLedger(transcript_path)
    holder: dict = {}

    def session_factory():
        inner = AdapterSession(
            workdir=workdir, seed=seed, username="myagent", aifill=2,
            # §3 startup ≤ 120 s + server-spawn margin (session default).
            startup_deadline=config.startup_deadline_s + 30.0,
            step_deadline=config.mutation_deadline_s,
            wait_deadline=config.waiting_deadline_s,
            close_deadline=config.teardown_deadline_s)
        wrapper = FaultInjectingSession(inner, seam=seam)
        holder["wrapper"] = wrapper
        return wrapper

    runner = GameRunner(config=config, client=client, ledger=ledger,
                        manifest=manifest, session_factory=session_factory)
    logger.log(f"[{run_id}] starting the lifecycle run: max_turns="
               f"{max_turns} seed={seed} seam: {seam.describe()}")
    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0
    ledger.close()
    load = load_transcript(transcript_path)
    wrapper = holder.get("wrapper")
    logger.log(f"[{run_id}] run finished: outcome={result.outcome} "
               f"reason={result.reason!r} turns_completed="
               f"{result.turns_completed} close_status="
               f"{result.close_status!r} wall_clock={wall:.1f}s")
    return {
        "result": result,
        "records": load.records,
        "load": load,
        "wrapper": wrapper,
        "client": client,
        "transcript_path": transcript_path,
        "wall_clock_s": wall,
        "config": config,
    }
