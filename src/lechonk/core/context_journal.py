# SPDX-License-Identifier: MIT
"""Durable journal for completed hybrid-context episodes.

This file is deliberately separate from the frozen transcript union.  A
context record is policy data used to reconstruct the next prompt; it is not a
game action or a substitute for the provenance transcript.  Records are
canonical JSON, checksummed, and fsynced before the caller starts another
model turn.  A torn final record is recoverable, while corruption before the
final record is reported rather than silently hidden.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lechonk.core.jsonutil import canonical_json


class ContextJournalCorruptionError(RuntimeError):
    """A non-trailing context record failed decoding or checksum validation."""


def _checksum(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "checksum"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable_episode(episode: Any) -> Any:
    """Convert a DecisionEpisode-like value to strict JSON data."""
    if hasattr(episode, "model_dump"):
        return episode.model_dump(mode="json")
    if hasattr(episode, "to_dict"):
        return episode.to_dict()
    if hasattr(episode, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(episode)
    if isinstance(episode, Mapping):
        return dict(episode)
    raise TypeError("episode must be a mapping or model/dataclass with a dump method")


@dataclass(frozen=True)
class ContextRecord:
    """One completed-turn context snapshot stored in ``context.jsonl``."""

    seq: int
    policy_version: str
    turn: int
    transcript_seq: int | None
    episode: Mapping[str, Any]
    rendered_history_hash: str
    ts: str = ""
    checksum: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "context_checkpoint",
            "seq": self.seq,
            "ts": self.ts,
            "policy_version": self.policy_version,
            "turn": self.turn,
            "transcript_seq": self.transcript_seq,
            "episode": dict(self.episode),
            "rendered_history_hash": self.rendered_history_hash,
            "checksum": self.checksum,
        }


@dataclass
class ContextJournalLoad:
    records: list[ContextRecord] = field(default_factory=list)
    discarded_trailing_partial: bool = False
    discarded_bytes: int = 0
    path: str = ""

    @property
    def last_seq(self) -> int:
        return self.records[-1].seq if self.records else -1


def _decode(raw: bytes) -> ContextRecord:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, Mapping) or data.get("kind") != "context_checkpoint":
        raise ValueError("invalid context record kind")
    if _checksum(data) != data.get("checksum"):
        raise ValueError("context checksum mismatch")
    episode = data.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("context episode is not an object")
    seq = data.get("seq")
    turn = data.get("turn")
    policy = data.get("policy_version")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError("invalid context sequence")
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
        raise ValueError("invalid context turn")
    if not isinstance(policy, str) or not policy:
        raise ValueError("invalid context policy version")
    transcript_seq = data.get("transcript_seq")
    if transcript_seq is not None and (
            not isinstance(transcript_seq, int) or isinstance(transcript_seq, bool)
            or transcript_seq < 0):
        raise ValueError("invalid transcript sequence")
    rendered_hash = data.get("rendered_history_hash")
    if not isinstance(rendered_hash, str) or not rendered_hash:
        raise ValueError("missing rendered history hash")
    return ContextRecord(
        seq=seq, policy_version=policy, turn=turn,
        transcript_seq=transcript_seq, episode=dict(episode),
        rendered_history_hash=rendered_hash, ts=str(data.get("ts", "")),
        checksum=str(data["checksum"]),
    )


def load_context_journal(path: str | Path) -> ContextJournalLoad:
    """Read a journal, discarding only a torn final record or unterminated tail."""
    path = str(path)
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    result = ContextJournalLoad(path=path)
    if not raw:
        return result
    *lines, tail = raw.split(b"\n")
    for index, line in enumerate(lines):
        try:
            result.records.append(_decode(line))
        except Exception:
            # Every entry in ``lines`` ended with a newline.  A malformed
            # newline-terminated record is therefore durable bytes with a
            # real integrity failure, not a recoverable short write.  Only
            # the unterminated ``tail`` below is eligible for recovery.
            raise ContextJournalCorruptionError(
                f"{path}: corrupt context record {index + 1}") from None
    if tail:
        result.discarded_trailing_partial = True
        result.discarded_bytes += len(tail)
    seqs = [record.seq for record in result.records]
    if seqs != list(range(len(seqs))):
        raise ContextJournalCorruptionError(
            f"{path}: context sequence violation (got {seqs})")
    return result


def _valid_prefix_bytes(path: str) -> int:
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    if not raw:
        return 0
    *lines, _tail = raw.split(b"\n")
    valid = 0
    for line in lines:
        try:
            _decode(line)
        except Exception:
            break
        valid += len(line) + 1
    return valid


class ContextJournal:
    """Append-only, fsynced journal of completed context episodes."""

    def __init__(self, path: str | Path, *, policy_version: str = "hybrid_v1"):
        self.path = str(path)
        self.policy_version = policy_version
        self._fh = open(self.path, "a", encoding="utf-8")
        self._next_seq = 0
        self._closed = False

    @classmethod
    def attach(cls, path: str | Path, *, policy_version: str = "hybrid_v1") -> "ContextJournal":
        loaded = load_context_journal(path)
        mismatched = {
            record.policy_version for record in loaded.records
            if record.policy_version != policy_version
        }
        if mismatched:
            raise ContextJournalCorruptionError(
                f"{path}: policy version mismatch (expected {policy_version!r}, "
                f"found {sorted(mismatched)!r})")
        if loaded.discarded_trailing_partial:
            with open(path, "r+b") as fh:
                fh.truncate(_valid_prefix_bytes(str(path)))
        journal = cls(path, policy_version=policy_version)
        journal._next_seq = loaded.last_seq + 1
        return journal

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def append(self, *, turn: int, episode: Any,
               transcript_seq: int | None = None,
               rendered_history_hash: str,
               durable: bool = True) -> ContextRecord:
        if self._closed:
            raise RuntimeError("context journal is closed")
        payload: dict[str, Any] = {
            "kind": "context_checkpoint",
            "seq": self._next_seq,
            "ts": _now(),
            "policy_version": self.policy_version,
            "turn": turn,
            "transcript_seq": transcript_seq,
            "episode": _jsonable_episode(episode),
            "rendered_history_hash": rendered_history_hash,
        }
        payload["checksum"] = _checksum(payload)
        line = canonical_json(payload) + "\n"
        self._fh.write(line)
        self._fh.flush()
        if durable:
            os.fsync(self._fh.fileno())
        record = _decode(line[:-1].encode("utf-8"))
        self._next_seq += 1
        return record

    def records(self) -> list[ContextRecord]:
        if not self._closed:
            self._fh.flush()
        return load_context_journal(self.path).records

    def fsync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "ContextJournal":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
