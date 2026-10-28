# SPDX-License-Identifier: MIT
"""lechonk.eval.checkpoint — the WP3 checkpoint export/preserve path (MIT).

design-plan-freeciv-v1.1.md §7.1 line 130 (the checkpoint export
primitive): "with Docker local, export = copy the save file out of the
container/webapp save directory (``docker cp``) after CivRealm's
begin-turn ``/save`` completes and before its end-turn HTTP deletion
(civ_controller.py:944-947, 979-980, 664-705); stability check per §3;
the exported file is the paired-run artifact (no HTTP download endpoint
exists)".

This module is our OWN copy-after-completion export, independent of
CivRealm's autosave HTTP deletion: while it is our turn (between the
begin-turn autosave and the end-turn ``deletesavegame``), the save file
for the current turn exists in the container at
``/var/lib/tomcat10/webapps/data/savegames/{username}/{username}_T{turn}_
{timestamp}.sav.{xz,zst}``. The export:

1. waits for the current turn's save file to appear (the begin-turn
   ``/save`` may still be writing when the phase starts);
2. runs the stability check — stat, re-stat until unchanged across
   ``stable_polls`` consecutive polls (the save completed);
3. copies the file out with ``docker cp``;
4. re-stats the container file (unchanged across the copy) and
   checksum-compares the container's ``sha256sum`` against the local
   copy's sha256;
5. preserves the verified file under ``artifacts/checkpoints/`` with the
   sha256 recorded (plus an append-only export log entry).

Licensing: MIT — this module NEVER imports civrealm (decision 7). The
export is docker-cp orchestration over subprocess only. Docker access:
plain ``docker`` when the session user has daemon access, otherwise
``sg docker -c "docker ..."`` (HANDOFF environment resume; the prefix is
probed once and cached).

The module also provides the inverse primitive the paired arms need:
``stage_for_load`` copies an exported checkpoint BACK into the
container's save directory (and verifies its sha256 in-container) so
the webapp's ``/load`` can see it — the exported artifact itself is what
both arms load (acceptance: a checkpoint exported via this path is
loadable by both arms).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from lechonk.core.jsonutil import strict_dumps
from lechonk.core.recovery import validate_freeciv_username

# The pinned container (HANDOFF: docker run --name freeciv-web ...).
DEFAULT_CONTAINER = "freeciv-web"
# The webapp save directory inside the container (HANDOFF session
# discoveries; civ_controller.py delete_save_game docstring).
SAVE_DIR_TEMPLATE = "/var/lib/tomcat10/webapps/data/savegames/{username}"
# The autosave extension is .sav.xz; .sav.zst appears from the
# interrupt-save path (civ_controller.py delete_save_game comment).
SAVE_SUFFIXES = (".sav.xz", ".sav.zst")
# The append-only export log under artifacts/checkpoints/.
EXPORT_LOG_NAME = "exports.jsonl"

DEFAULT_DEADLINE_S = 90.0     # bounded wait for the save to appear/stabilize
DEFAULT_STABLE_POLLS = 3      # consecutive identical stats = stable
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_DOCKER_TIMEOUT_S = 60.0


class CheckpointError(RuntimeError):
    """The checkpoint export/staging failed (bounded, reported, never
    retried blindly)."""


@dataclass(frozen=True)
class FileStat:
    """One stat reading of a container file (size + mtime + raw line)."""

    size: int
    mtime: float
    raw: str


@dataclass
class ExportResult:
    """One verified checkpoint export."""

    path: str                 # the preserved local file (artifacts/checkpoints/...)
    save_name: str            # the load_game value (filename minus suffix)
    sha256: str
    size_bytes: int
    turn: int
    username: str
    container_path: str
    stability_stats: list[FileStat] = field(default_factory=list)
    copied_stat: Optional[FileStat] = None   # the pre-copy stat
    restat: Optional[FileStat] = None        # the post-copy re-stat
    container_sha256: str = ""
    local_sha256: str = ""
    candidates: list[str] = field(default_factory=list)  # all matching files
    exported_at: str = ""
    tool: str = "lechonk.eval.checkpoint"

    def to_log_record(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "exported_at": self.exported_at,
            "path": self.path,
            "save_name": self.save_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "turn": self.turn,
            "username": self.username,
            "container_path": self.container_path,
            "candidates": self.candidates,
            "stability_stats": [s.raw for s in self.stability_stats],
            "copied_stat": self.copied_stat.raw if self.copied_stat else None,
            "restat": self.restat.raw if self.restat else None,
            "container_sha256": self.container_sha256,
            "local_sha256": self.local_sha256,
            "checksum_compare_match": (
                self.container_sha256 == self.local_sha256
                and self.container_sha256 == self.sha256),
        }


# ---------------------------------------------------------------------------
# docker access
# ---------------------------------------------------------------------------

_DOCKER_PREFIX: Optional[list[str]] = None


def _probe_docker_prefix() -> list[str]:
    """Plain ``docker`` when the session user has daemon access, else
    ``sg docker -c`` (HANDOFF environment resume). Probed once, cached."""
    global _DOCKER_PREFIX
    if _DOCKER_PREFIX is not None:
        return _DOCKER_PREFIX
    try:
        proc = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            _DOCKER_PREFIX = ["docker"]
            return _DOCKER_PREFIX
    except (OSError, subprocess.SubprocessError):
        pass
    _DOCKER_PREFIX = ["sg", "docker", "-c"]
    return _DOCKER_PREFIX


def run_docker(args: Sequence[str], *,
               timeout: float = DEFAULT_DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run one docker CLI invocation (plain or sg-wrapped). Raises
    CheckpointError on failure — the caller decides the bounded retry."""
    prefix = _probe_docker_prefix()
    if prefix == ["sg", "docker", "-c"]:
        cmd = ["sg", "docker", "-c", "docker " + shlex.join(list(args))]
    else:
        cmd = ["docker", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CheckpointError(f"docker {' '.join(args)} timed out: {exc}") from exc
    if proc.returncode != 0:
        raise CheckpointError(
            f"docker {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}")
    return proc


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_name_for(filename: str) -> str:
    """The ``debug.load_game`` value for a save filename: the filename
    minus its ``.sav.xz`` / ``.sav.zst`` suffix (wp0-5 load pattern)."""
    for suffix in SAVE_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def save_dir_for(username: str) -> str:
    return SAVE_DIR_TEMPLATE.format(username=validate_freeciv_username(username))


def _parse_stat(raw: str) -> Optional[FileStat]:
    """Parse ``stat -c '%s %Y'`` output; None when the file is absent."""
    text = raw.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        return FileStat(size=int(parts[0]), mtime=float(parts[1]), raw=text)
    except ValueError:
        return None


def _exec_stat(container: str, path: str) -> Optional[FileStat]:
    proc = run_docker(["exec", container, "stat", "-c", "%s %Y", path])
    return _parse_stat(proc.stdout)


def _exec_sha256(container: str, path: str) -> str:
    proc = run_docker(["exec", container, "sha256sum", path])
    out = proc.stdout.strip().split()
    if not out:
        raise CheckpointError(f"sha256sum returned nothing for {path}")
    return out[0]


def _list_save_dir(container: str, username: str) -> list[str]:
    proc = run_docker(["exec", container, "ls", "-1", save_dir_for(username)])
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _matching_saves(names: Sequence[str], *, username: str, turn: int
                  ) -> list[str]:
    """The save files for ``{username}_T{turn}_*`` (the begin-turn
    autosave name pattern, civ_controller.py save_game)."""
    prefix = f"{username}_T{turn}_"
    return sorted(n for n in names if n.startswith(prefix))


def _newest(container: str, names: Sequence[str], username: str) -> str:
    """The newest file (by mtime) among the candidates."""
    best, best_mtime = None, -1.0
    for name in names:
        stat = _exec_stat(container, f"{save_dir_for(username)}/{name}")
        if stat is not None and stat.mtime >= best_mtime:
            best, best_mtime = name, stat.mtime
    if best is None:
        raise CheckpointError(f"no stat-able save among {list(names)}")
    return best


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_turn_save(*, username: str, turn: int, dest_dir: str | Path,
                     container: str = DEFAULT_CONTAINER,
                     deadline_s: float = DEFAULT_DEADLINE_S,
                     stable_polls: int = DEFAULT_STABLE_POLLS,
                     poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep,
                     now_iso: Callable[[], str] = lambda: time.strftime(
                         "%Y-%m-%dT%H:%M:%SZ", time.gmtime())) -> ExportResult:
    """Export the current turn's autosave out of the container with the
    stat-copy-re-stat-checksum stability procedure (§3 / WP0 P8).

    The caller guarantees the export window: it is our turn (the begin-turn
    ``/save`` has completed or is completing) and the end-turn HTTP
    deletion has not fired (we have not finished the phase). The function
    itself re-verifies the window's assumptions: the file must appear,
    stabilize, copy, re-stat unchanged, and checksum-compare equal.

    The preserved file lands under ``dest_dir`` (artifacts/checkpoints/)
    with the sha256 recorded; an append-only log entry is written to
    ``dest_dir/exports.jsonl``.
    """
    username = validate_freeciv_username(username)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    deadline = clock() + deadline_s
    save_dir = save_dir_for(username)

    # 1-2. wait for the file, then for stability (stat, re-stat, ...).
    target: Optional[str] = None
    candidates: list[str] = []
    stats: list[FileStat] = []
    while clock() < deadline:
        names = _list_save_dir(container, username)
        candidates = _matching_saves(names, username=username, turn=turn)
        if candidates:
            target = _newest(container, candidates, username)
            stat = _exec_stat(container, f"{save_dir}/{target}")
            if stat is not None:
                stats.append(stat)
                # rolling baseline: the last stable_polls polls must agree
                # with each other (not with the first observation), so one
                # mid-series change cannot pin a stale stats[0] baseline.
                if len(stats) >= stable_polls and \
                        all(s.size == stats[-1].size and s.mtime == stats[-1].mtime
                            for s in stats[-stable_polls:]):
                    break  # stable: the begin-turn /save completed
        sleep(poll_interval_s)
    else:
        raise CheckpointError(
            f"no stable save for {username}_T{turn}_* in {container}:{save_dir} "
            f"within {deadline_s:.0f}s (candidates seen: {candidates})")
    assert target is not None
    # the stable window's stat (the file may have changed early in the series)
    copied_stat = stats[-1]

    # 3. copy out (to a temp name, then rename into place).
    container_path = f"{save_dir}/{target}"
    tmp_path = dest / f".exporting-{target}"
    if tmp_path.exists():
        tmp_path.unlink()
    run_docker(["cp", f"{container}:{container_path}", str(tmp_path)])

    # 4. re-stat the container file (unchanged across the copy) and
    #    checksum-compare container vs local.
    restat = _exec_stat(container, container_path)
    if restat is None or restat.size != copied_stat.size \
            or restat.mtime != copied_stat.mtime:
        raise CheckpointError(
            f"the container save changed across the copy: pre={copied_stat.raw} "
            f"post={restat.raw if restat else None}")
    local_sha = sha256_file(tmp_path)
    container_sha = _exec_sha256(container, container_path)
    if local_sha != container_sha:
        raise CheckpointError(
            f"checksum mismatch after docker cp: local={local_sha} "
            f"container={container_sha}")

    # 5. preserve under artifacts/checkpoints/ with the sha256 recorded.
    final_path = dest / target
    os.replace(tmp_path, final_path)
    result = ExportResult(
        path=str(final_path),
        save_name=save_name_for(target),
        sha256=local_sha,
        size_bytes=copied_stat.size,
        turn=turn,
        username=username,
        container_path=container_path,
        stability_stats=stats,
        copied_stat=copied_stat,
        restat=restat,
        container_sha256=container_sha,
        local_sha256=local_sha,
        candidates=candidates,
        exported_at=now_iso(),
    )
    append_export_log(dest, result.to_log_record())
    return result


def append_export_log(dest_dir: str | Path, record: dict[str, Any]) -> str:
    """Append one record to the append-only export log
    (``artifacts/checkpoints/exports.jsonl``)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / EXPORT_LOG_NAME
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(strict_dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return str(path)


# ---------------------------------------------------------------------------
# staging for load (the paired arms load the EXPORTED artifact)
# ---------------------------------------------------------------------------

@dataclass
class StagingResult:
    """One verified checkpoint staging into the container's save dir."""

    local_path: str
    save_name: str
    staged_container_path: str
    sha256: str
    staged_sha256: str
    staged_at: str
    tool: str = "lechonk.eval.checkpoint"

    def to_record(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "staged_at": self.staged_at,
            "local_path": self.local_path,
            "save_name": self.save_name,
            "staged_container_path": self.staged_container_path,
            "sha256": self.sha256,
            "staged_sha256": self.staged_sha256,
            "checksum_match": self.sha256 == self.staged_sha256,
        }


def stage_for_load(local_path: str | Path, *, username: str,
                   container: str = DEFAULT_CONTAINER,
                   now_iso: Callable[[], str] = lambda: time.strftime(
                       "%Y-%m-%dT%H:%M:%SZ", time.gmtime())) -> StagingResult:
    """Copy an exported checkpoint INTO the container's save directory so
    the webapp's ``/load`` resolves it, and verify the staged file's
    sha256 in-container. The staged name keeps the exported save name
    (the save/username coupling, §7.1: the load validates the save's
    username prefix equals ours)."""
    username = validate_freeciv_username(username)
    local = Path(local_path)
    if not local.is_file():
        raise CheckpointError(f"checkpoint not found: {local}")
    filename = local.name
    save_name = save_name_for(filename)
    if not save_name.startswith(f"{username}_"):
        raise CheckpointError(
            f"the checkpoint's save name {save_name!r} does not carry the "
            f"pinned username prefix {username!r} (save/username coupling, "
            f"civ_controller.py:707-723)")
    sha = sha256_file(local)
    container_path = f"{save_dir_for(username)}/{filename}"
    run_docker(["cp", str(local), f"{container}:{container_path}"])
    staged_sha = _exec_sha256(container, container_path)
    if staged_sha != sha:
        raise CheckpointError(
            f"staged checkpoint checksum mismatch: local={sha} "
            f"container={staged_sha}")
    return StagingResult(
        local_path=str(local),
        save_name=save_name,
        staged_container_path=container_path,
        sha256=sha,
        staged_sha256=staged_sha,
        staged_at=now_iso(),
    )


def verify_local_checkpoint(local_path: str | Path, expected_sha256: str) -> bool:
    """Re-verify a preserved checkpoint's sha256 against the recorded
    value (the paired-run acceptance reads this)."""
    return sha256_file(local_path) == expected_sha256
