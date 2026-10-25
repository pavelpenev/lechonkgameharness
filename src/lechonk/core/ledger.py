# SPDX-License-Identifier: MIT
"""lechonk.core.ledger — the append-only provenance transcript (WP2).

§3 line 34 transcript durability: the JSONL is append-only with ONE
checksummed record per line (a frozen core.schemas.TranscriptRecord union
member), flushed (fsync) at the intent-before-write boundary and at each
action boundary; on restart a trailing partial record is discarded.

Provenance per §3 line 38 lives in the records themselves (see
core.schemas.ActionResultRecord); this module owns only the durability
mechanics:

- ``append(record, durable=True)`` assigns ``seq``/``ts``/``checksum``,
  validates the finalized record against the frozen union, writes exactly
  one canonical-JSON line, flushes, and (when ``durable``) fsyncs. The
  runner marks the two §3 durability boundaries with ``durable=True``:
  the intent-before-write boundary (all of a turn's intent records are on
  disk before the first mutation is sent — the runner calls ``fsync()``
  after appending them) and each action boundary (every ``action_result``
  of an executed mutation).
- ``fsync()`` forces the intent-before-write boundary explicitly.
- ``load_transcript(path)`` re-reads a transcript for the metrics: every
  line is schema-validated and checksum-verified, and seqs must be
  exactly 0..n-1 (a duplicate or gap raises LedgerCorruptionError);
  "the last newline-terminated line" is the last line that ends with a
  newline — the record preceding any unterminated tail — and a trailing
  partial record is DISCARDED and reported. The trailing partial is
  EITHER that final newline-terminated line failing validation/checksum
  OR unterminated bytes after the newline, and both may be discarded
  together when they co-occur. Unterminated tail bytes are ALWAYS a
  trailing partial — even when they decode and parse: their newline
  never landed, so they come from a torn-write window (never
  fsynced-complete); the mutation state is settled by read-back
  reconciliation, so discarding is safe (never "repair" by appending a
  newline). Corruption anywhere before the trailing record raises
  LedgerCorruptionError (never silently dropped).
- ``TranscriptLedger.attach(path)`` recovers a transcript after a kill and
  resumes appending at the next seq. ``load_transcript`` is read-only —
  it never modifies the file — while attach additionally truncates the
  recoverable working transcript to the valid prefix. Discarding is
  conservative by design: torn trailing bytes are treated as a torn
  write, not proven harmless; operational records preceding a discarded
  tail may remain uncorrelated, and the P11 audit's pending-intent
  scanner is the tool that reports them.

The ledger never imports civrealm or the adapter (core purity,
tests/test_core_purity.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Union

from pydantic import TypeAdapter

from lechonk.core.jsonutil import canonical_json
from lechonk.core.schemas import TranscriptRecord, TranscriptRecordBase

# One TypeAdapter for the frozen union (validation + discrimination by kind).
_RECORD_ADAPTER: TypeAdapter[TranscriptRecord] = TypeAdapter(TranscriptRecord)


class LedgerCorruptionError(RuntimeError):
    """A transcript line before the trailing partial record failed
    validation or checksum verification — the transcript is not silently
    truncated."""


def compute_checksum(record_dict: Mapping[str, Any]) -> str:
    """Per-record checksum: sha256 over the canonical JSON of the record
    with the ``checksum`` field excluded (the record carries its own
    checksum, so the checksum cannot cover itself)."""
    payload = {k: v for k, v in record_dict.items() if k != "checksum"}
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TranscriptLoad:
    """The result of reading a transcript back."""

    records: list[TranscriptRecord] = field(default_factory=list)
    discarded_trailing_partial: bool = False
    discarded_bytes: int = 0
    path: str = ""

    @property
    def last_seq(self) -> int:
        return self.records[-1].seq if self.records else -1


def _decode_line(line: bytes) -> TranscriptRecord:
    """Decode + schema-validate + checksum-verify one transcript line."""
    data = json.loads(line.decode("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("transcript line is not a JSON object")
    record = _RECORD_ADAPTER.validate_python(dict(data))
    if compute_checksum(dict(data)) != data.get("checksum"):
        raise ValueError("checksum mismatch")
    return record


def load_transcript(path: Union[str, Path]) -> TranscriptLoad:
    """Read a transcript, discarding a trailing partial record.

    A torn write at EOF (the mid-write kill) leaves either unterminated
    bytes after the last newline or a final newline-terminated line that
    fails validation/checksum — both are the trailing partial record and
    are discarded. "The last newline-terminated line" is the last line
    that actually ends with a newline: the record preceding any
    unterminated tail, and both trailing artifacts are discarded
    together when they co-occur. Unterminated bytes at EOF are ALWAYS a
    trailing partial, even when they decode and parse: the terminating
    newline never landed, so the record comes from a torn-write window
    (never fsynced-complete); its mutation state is settled by read-back
    reconciliation, so discarding is safe — never "repair" by appending
    a newline. The record seqs must be exactly 0..n-1 (no duplicates or
    gaps); a violation raises LedgerCorruptionError naming the file. Any
    other failure BEFORE the trailing record raises LedgerCorruptionError:
    the transcript is never silently truncated.

    This read is conservative by design and never modifies the file: a
    discarded trailing artifact is treated as a torn write, not proven
    harmless. Recovery that truncates the file to the valid prefix is
    TranscriptLedger.attach's job. Operational records preceding a
    discarded tail may remain uncorrelated; the P11 audit's
    pending-intent scanner is the tool that reports them.
    """
    path = str(path)
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    load = TranscriptLoad(path=path)
    if not raw:
        return load

    *body_lines, tail = raw.split(b"\n")
    # body_lines: newline-terminated lines; tail: bytes after the last
    # newline (b"" when the file ends with a newline).
    n = len(body_lines)
    for i, line in enumerate(body_lines):
        try:
            load.records.append(_decode_line(line))
        except Exception:
            if i == n - 1:
                # The final newline-terminated line is torn (killed between
                # the line write and the fsync): trailing partial, discard.
                load.discarded_trailing_partial = True
                load.discarded_bytes += len(line) + 1
                break
            raise LedgerCorruptionError(
                f"{path}: corrupt transcript line {i + 1} (before the "
                f"trailing record)") from None
    if tail:
        # Unterminated bytes after the last newline: ALWAYS a trailing
        # partial, even when they decode and parse — the terminating
        # newline never landed (torn-write window, never fsynced-complete);
        # the mutation state is settled by read-back reconciliation, so
        # discarding is safe. Never "repair" by appending a newline.
        load.discarded_trailing_partial = True
        load.discarded_bytes += len(tail)

    # Seq continuity: the kept records must be exactly 0..n-1.
    seqs = [r.seq for r in load.records]
    if seqs != list(range(len(seqs))):
        raise LedgerCorruptionError(
            f"{path}: transcript seq continuity violation "
            f"(expected 0..{len(seqs) - 1}, got {seqs})")
    return load


def valid_prefix_bytes(path: str) -> int:
    """The byte offset just after the last VALID record's line — the
    length of the transcript's valid prefix (used by attach() to truncate a
    trailing partial record out of the FILE)."""
    raw = Path(path).read_bytes()
    if not raw:
        return 0
    *body_lines, tail = raw.split(b"\n")
    valid_end = 0
    for line in body_lines:
        try:
            _decode_line(line)
        except Exception:
            break  # the trailing partial (load_transcript already validated
            # that any earlier failure would have raised)
        valid_end += len(line) + 1
    # An unterminated tail is ALWAYS a trailing partial (torn-write window,
    # never fsynced-complete; see load_transcript): truncate at the last
    # newline — never keep or repair it by appending a newline.
    return valid_end


class TranscriptLedger:
    """Append-only JSONL transcript with per-record checksums and explicit
    durability boundaries (§3 line 34).

    ``clock`` is a monotonic clock used only for the stall-watchdog's
    last-event tracking (inject a fake in tests); ``ts`` is wall time.
    """

    def __init__(self, path: Union[str, Path], *,
                 clock: Callable[[], float] = time.monotonic):
        self.path = str(path)
        self._clock = clock
        self._fh = open(self.path, "a", encoding="utf-8")
        self._next_seq = 0
        self._last_event: float = clock()
        self._closed = False

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------

    @classmethod
    def attach(cls, path: Union[str, Path], *,
               clock: Callable[[], float] = time.monotonic
               ) -> "TranscriptLedger":
        """Recover a transcript after a kill and resume appending.

        The trailing partial record is discarded — from the LOAD view and
        from the FILE (truncated at the valid prefix), so the next append
        lands right after the last valid record instead of after torn
        bytes. The next append continues at last valid seq + 1.
        """
        load = load_transcript(path)
        if load.discarded_trailing_partial:
            with open(path, "r+b") as fh:
                fh.truncate(valid_prefix_bytes(str(path)))
        ledger = cls(path, clock=clock)
        ledger._next_seq = load.last_seq + 1
        return ledger

    # ------------------------------------------------------------------
    # append
    # ------------------------------------------------------------------

    @property
    def last_seq(self) -> int:
        return self._next_seq - 1

    @property
    def last_event(self) -> float:
        """Monotonic clock reading of the last transcript event (the
        run-level stall watchdog watches this, §3 line 34)."""
        return self._last_event

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def append(self, record: TranscriptRecordBase, *,
               durable: bool = False) -> int:
        """Append one checksummed record; returns its seq.

        The ledger assigns ``seq``/``ts``/``checksum`` (the record's own
        values are overwritten), re-validates the finalized record against
        the frozen union, and writes exactly one canonical-JSON line.
        ``durable=True`` flushes AND fsyncs — the §3 durability boundaries
        (intent-before-write, each action boundary) pass durable=True.
        """
        if self._closed:
            raise RuntimeError("ledger is closed")
        data = record.model_dump(mode="json")
        data["seq"] = self._next_seq
        data["ts"] = _utc_now_iso()
        data["checksum"] = compute_checksum(data)
        # Re-validate the finalized record (the checksum must not be able
        # to smuggle an invalid record past the frozen union).
        finalized = _RECORD_ADAPTER.validate_python(data)
        line = canonical_json(finalized.model_dump(mode="json"))
        self._fh.write(line + "\n")
        self._fh.flush()
        self._last_event = self._clock()
        if durable:
            os.fsync(self._fh.fileno())
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def fsync(self) -> None:
        """The intent-before-write durability boundary: everything appended
        so far (the turn's intent records) is on disk before the first
        mutation is sent (§3 line 34)."""
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def records(self) -> list[TranscriptRecord]:
        """The transcript's records (re-read from the file; usable before
        or after close)."""
        if not self._closed:
            self._fh.flush()
        return load_transcript(self.path).records

    def close(self) -> None:
        if not self._closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "TranscriptLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
