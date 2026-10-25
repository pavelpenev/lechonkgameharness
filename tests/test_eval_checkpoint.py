# SPDX-License-Identifier: MIT
"""Tests for lechonk.eval.checkpoint — the WP3 checkpoint export/preserve
path (design-plan-freeciv-v1.1.md §7.1 line 130).

The docker layer is mocked (an injectable fake ``run_docker``): the export
procedure itself — appear → stabilize (stat/re-stat) → copy → re-stat →
checksum-compare → preserve + log — is exercised end to end, including
its bounded failure modes (no save appears, the save changes across the
copy, the checksum mismatches, the username prefix violates the
save/username coupling).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from lechonk.eval import checkpoint as cp


# ---------------------------------------------------------------------------
# a fake docker CLI
# ---------------------------------------------------------------------------

class FakeDocker:
    """A scriptable fake of ``run_docker``: serves ls/stat/cp/sha256sum
    against an in-memory container filesystem."""

    def __init__(self, files: dict[str, bytes] | None = None):
        # path -> content
        self.files = dict(files or {})
        self.calls: list[list[str]] = []
        # test hooks
        self.stat_mutations: list[str] = []  # paths whose stat changes once
        self._mutated: set[str] = set()
        self.cp_should_change_file: str | None = None  # path changed during cp

    def run(self, args, *, timeout=60.0):
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["exec", cp.DEFAULT_CONTAINER] and args[2] == "ls":
            # args: exec <container> ls -1 <dir>
            names = sorted(
                p.rsplit("/", 1)[-1] for p in self.files
                if p.rsplit("/", 1)[0] == args[4])
            return subprocess.CompletedProcess(args, 0, "\n".join(names) + "\n", "")
        if args[:2] == ["exec", cp.DEFAULT_CONTAINER] and args[2] == "stat":
            path = args[5]
            if path not in self.files:
                return subprocess.CompletedProcess(args, 1, "", "stat: cannot stat")
            content = self.files[path]
            mtime = 1000.0
            if path in self.stat_mutations and path not in self._mutated:
                # the save is still being written on the first poll
                self._mutated.add(path)
                mtime = 999.0
                content = content[:-1]  # a torn write
            size = len(content)
            return subprocess.CompletedProcess(
                args, 0, f"{size} {mtime}\n", "")
        if args[:2] == ["exec", cp.DEFAULT_CONTAINER] and args[2] == "sha256sum":
            path = args[3]
            digest = hashlib.sha256(self.files[path]).hexdigest()
            return subprocess.CompletedProcess(args, 0, f"{digest}  {path}\n", "")
        if args[0] == "cp" and args[1].startswith(cp.DEFAULT_CONTAINER + ":"):
            src = args[1].split(":", 1)[1]
            dest = args[2]
            if self.cp_should_change_file == src:
                # simulate the container file changing across the copy
                self.files[src] = self.files[src] + b"x"
            with open(dest, "wb") as fh:
                fh.write(self.files[src])
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "cp" and args[2].startswith(cp.DEFAULT_CONTAINER + ":"):
            # docker cp host -> container
            src, dest = args[1], args[2].split(":", 1)[1]
            with open(src, "rb") as fh:
                self.files[dest] = fh.read()
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"FakeDocker: unexpected args {args}")


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(cp, "run_docker", fake.run)
    return fake


def _save_bytes(tag: str = "save") -> bytes:
    return (f"freeciv-save-{tag}".encode() * 40)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def test_save_name_for_strips_suffixes():
    assert cp.save_name_for("myagent_T5_2026-10-24-17_32.sav.xz") == \
        "myagent_T5_2026-10-24-17_32"
    assert cp.save_name_for("myagent_T5_2026-10-24-17_32.sav.zst") == \
        "myagent_T5_2026-10-24-17_32"
    assert cp.save_name_for("plainname") == "plainname"


def test_parse_stat():
    assert cp._parse_stat("11696 1729785840\n") == cp.FileStat(
        11696, 1729785840.0, "11696 1729785840")
    assert cp._parse_stat("") is None
    assert cp._parse_stat("garbage") is None


def test_matching_saves():
    names = ["myagent_T5_2026-10-24-17_32.sav.xz",
             "myagent_T4_2026-10-24-17_31.sav.xz",
             "myagent_T11_2026-10-24-10_24.sav.xz",
             "other_T5_2026-10-24-17_32.sav.xz"]
    assert cp._matching_saves(names, username="myagent", turn=5) == \
        ["myagent_T5_2026-10-24-17_32.sav.xz"]


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert cp.sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


# ---------------------------------------------------------------------------
# export_turn_save (the stat-copy-re-stat-checksum procedure)
# ---------------------------------------------------------------------------

def test_export_turn_save_happy_path(tmp_path, fake_docker):
    save_dir = cp.save_dir_for("myagent")
    name = "myagent_T5_2026-10-24-17_32.sav.xz"
    content = _save_bytes()
    fake_docker.files[f"{save_dir}/{name}"] = content
    # the first stat sees a torn write (still saving), then it stabilizes
    fake_docker.stat_mutations.append(f"{save_dir}/{name}")

    # a fake clock: each sleep advances it by the poll interval
    now = [1000.0]

    def clock():
        return now[0]

    def sleep(s):
        now[0] += s

    result = cp.export_turn_save(
        username="myagent", turn=5, dest_dir=tmp_path,
        stable_polls=3, poll_interval_s=1.0, clock=clock, sleep=sleep)

    assert result.path == str(tmp_path / name)
    assert result.save_name == "myagent_T5_2026-10-24-17_32"
    assert result.sha256 == _sha(content)
    assert result.local_sha256 == result.container_sha256 == result.sha256
    assert result.size_bytes == len(content)
    assert result.turn == 5 and result.username == "myagent"
    assert result.container_path == f"{save_dir}/{name}"
    # the stability check: >= 3 stats, the re-stat unchanged across the copy
    assert len(result.stability_stats) >= 3
    assert result.restat is not None
    assert result.restat.size == result.copied_stat.size
    assert result.restat.mtime == result.copied_stat.mtime
    # the preserved file verifies
    assert cp.verify_local_checkpoint(result.path, result.sha256)
    # the export log entry was appended
    log_path = tmp_path / cp.EXPORT_LOG_NAME
    assert log_path.is_file()
    entries = [json.loads(l) for l in
               log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["sha256"] == result.sha256
    assert entries[0]["checksum_compare_match"] is True
    assert entries[0]["tool"] == "lechonk.eval.checkpoint"
    assert len(entries[0]["stability_stats"]) >= 3


def test_export_turn_save_waits_for_the_file(tmp_path, fake_docker,
                                             monkeypatch):
    """The save appears only after a few polls (the begin-turn /save is
    still being written when the phase starts)."""
    save_dir = cp.save_dir_for("myagent")
    name = "myagent_T5_2026-10-24-17_32.sav.xz"
    content = _save_bytes()
    now = [1000.0]
    polls = {"n": 0}
    real_run = fake_docker.run

    def run(args, *, timeout=60.0):
        if args[:2] == ["exec", cp.DEFAULT_CONTAINER] and args[2] == "ls":
            polls["n"] += 1
            if polls["n"] < 3:
                return subprocess.CompletedProcess(args, 0, "", "")
            fake_docker.files[f"{save_dir}/{name}"] = content
        return real_run(args, timeout=timeout)

    monkeypatch.setattr(cp, "run_docker", run)
    result = cp.export_turn_save(
        username="myagent", turn=5, dest_dir=tmp_path,
        stable_polls=2, poll_interval_s=0.5,
        clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    assert result.sha256 == _sha(content)
    assert polls["n"] >= 3


def test_export_turn_save_no_save_raises(tmp_path, fake_docker):
    now = [1000.0]
    with pytest.raises(cp.CheckpointError, match="no stable save"):
        cp.export_turn_save(
            username="myagent", turn=5, dest_dir=tmp_path,
            deadline_s=2.0, stable_polls=2, poll_interval_s=0.5,
            clock=lambda: now[0],
            sleep=lambda s: now.__setitem__(0, now[0] + s))


def test_export_turn_save_checksum_mismatch_raises(tmp_path, monkeypatch):
    save_dir = cp.save_dir_for("myagent")
    name = "myagent_T5_2026-10-24-17_32.sav.xz"
    fake = FakeDocker({f"{save_dir}/{name}": _save_bytes()})
    monkeypatch.setattr(cp, "run_docker", fake.run)
    # corrupt the local copy after docker cp writes it
    real_cp = fake.run

    def run(args, *, timeout=60.0):
        proc = real_cp(args, timeout=timeout)
        if args[0] == "cp" and args[1].startswith(cp.DEFAULT_CONTAINER + ":"):
            with open(args[2], "ab") as fh:
                fh.write(b"corruption")
        return proc

    monkeypatch.setattr(cp, "run_docker", run)
    with pytest.raises(cp.CheckpointError, match="checksum mismatch"):
        cp.export_turn_save(username="myagent", turn=5, dest_dir=tmp_path,
                            stable_polls=2, poll_interval_s=0.1,
                            clock=time.monotonic, sleep=lambda s: None)


def test_export_turn_save_file_changes_across_copy_raises(tmp_path, fake_docker):
    save_dir = cp.save_dir_for("myagent")
    name = "myagent_T5_2026-10-24-17_32.sav.xz"
    fake_docker.files[f"{save_dir}/{name}"] = _save_bytes()
    fake_docker.cp_should_change_file = f"{save_dir}/{name}"
    with pytest.raises(cp.CheckpointError, match="changed across the copy"):
        cp.export_turn_save(username="myagent", turn=5, dest_dir=tmp_path,
                            stable_polls=2, poll_interval_s=0.1,
                            clock=time.monotonic, sleep=lambda s: None)


def test_export_turn_save_picks_the_newest_same_turn_save(tmp_path, fake_docker):
    save_dir = cp.save_dir_for("myagent")
    older = "myagent_T5_2026-10-24-17_31.sav.xz"
    newer = "myagent_T5_2026-10-24-17_32.sav.xz"
    fake_docker.files[f"{save_dir}/{older}"] = _save_bytes("old")
    fake_docker.files[f"{save_dir}/{newer}"] = _save_bytes("new")
    # make the older file look newer by mtime is not possible via the fake
    # (mtime is fixed); instead verify the newest-by-mtime selection with
    # equal mtimes -> the last one wins deterministically.
    result = cp.export_turn_save(
        username="myagent", turn=5, dest_dir=tmp_path,
        stable_polls=2, poll_interval_s=0.1,
        clock=time.monotonic, sleep=lambda s: None)
    assert result.sha256 == _sha(_save_bytes("new"))
    assert sorted(result.candidates) == sorted([older, newer])


# ---------------------------------------------------------------------------
# stage_for_load
# ---------------------------------------------------------------------------

def test_stage_for_load_roundtrip(tmp_path, fake_docker):
    save_dir = cp.save_dir_for("myagent")
    name = "myagent_T5_2026-10-24-17_32.sav.xz"
    content = _save_bytes()
    local = tmp_path / name
    local.write_bytes(content)
    staging = cp.stage_for_load(local, username="myagent")
    assert staging.save_name == "myagent_T5_2026-10-24-17_32"
    assert staging.sha256 == _sha(content)
    assert staging.staged_sha256 == _sha(content)
    assert staging.to_record()["checksum_match"] is True
    # the staged file is in the container's save dir
    assert fake_docker.files[f"{save_dir}/{name}"] == content


def test_stage_for_load_rejects_foreign_username(tmp_path, fake_docker):
    local = tmp_path / "other_T5_2026-10-24-17_32.sav.xz"
    local.write_bytes(_save_bytes())
    with pytest.raises(cp.CheckpointError, match="username prefix"):
        cp.stage_for_load(local, username="myagent")


def test_stage_for_load_missing_file(tmp_path, fake_docker):
    with pytest.raises(cp.CheckpointError, match="not found"):
        cp.stage_for_load(tmp_path / "nope.sav.xz", username="myagent")


# ---------------------------------------------------------------------------
# docker prefix detection + command construction
# ---------------------------------------------------------------------------

def test_run_docker_plain_prefix(monkeypatch):
    monkeypatch.setattr(cp, "_DOCKER_PREFIX", None)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "29.7.2", "")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    proc = cp.run_docker(["version", "--format", "{{.Server.Version}}"])
    assert proc.returncode == 0
    assert seen["cmd"][:1] == ["docker"]
    # the probe cached the plain prefix
    assert cp._DOCKER_PREFIX == ["docker"]


def test_run_docker_sg_fallback(monkeypatch):
    monkeypatch.setattr(cp, "_DOCKER_PREFIX", None)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.setdefault("cmds", []).append(cmd)
        if cmd[0] == "docker":
            return subprocess.CompletedProcess(cmd, 1, "", "permission denied")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    cp.run_docker(["ps"])
    assert cp._DOCKER_PREFIX == ["sg", "docker", "-c"]
    # the sg invocation wraps the docker command in a single -c string
    sg_cmds = [c for c in seen["cmds"] if c[0] == "sg"]
    assert sg_cmds and sg_cmds[0][1:3] == ["docker", "-c"]
    assert sg_cmds[0][3].startswith("docker ")
    assert "ps" in sg_cmds[0][3]


def test_run_docker_failure_raises(monkeypatch):
    monkeypatch.setattr(cp, "_DOCKER_PREFIX", ["docker"])
    monkeypatch.setattr(
        cp.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(cp.CheckpointError, match="boom"):
        cp.run_docker(["ps"])


# ---------------------------------------------------------------------------
# strict JSON emission in the export log (FW-19)
# ---------------------------------------------------------------------------

def test_append_export_log_nan_raises(tmp_path):
    """The export log is written via strict_dumps: a NaN in the record
    raises ValueError deterministically — a bare NaN token never lands
    in the log (json.loads on the next read would die on it)."""
    with pytest.raises(ValueError):
        cp.append_export_log(tmp_path, {"sha256": float("nan")})


def test_append_export_log_lone_surrogate_escaped_and_round_trips(tmp_path):
    """A lone surrogate in the record is escaped as \\udXXX in the log
    line — valid JSON that never reaches the utf-8 encoder, and
    json.loads round-trips it to the surrogate string."""
    record = {"tool": "lechonk.eval.checkpoint",
              "note": "lone \ud800 surrogate"}
    path = Path(cp.append_export_log(tmp_path, record))
    text = path.read_text(encoding="utf-8")
    assert "\\ud800" in text
    assert json.loads(text) == record


def test_append_export_log_clean_record_round_trips(tmp_path):
    record = {"tool": "lechonk.eval.checkpoint", "turn": 5,
              "sha256": "sha256:" + "ab" * 32, "size_bytes": 120}
    path = Path(cp.append_export_log(tmp_path, record))
    assert json.loads(path.read_text(encoding="utf-8")) == record
