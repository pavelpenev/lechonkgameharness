# SPDX-License-Identifier: MIT
"""Ledger tests (WP2): append-only JSONL transcript durability (§3 line 34).

- one checksummed record per line; the ledger assigns seq/ts/checksum and
  re-validates the finalized record against the frozen union;
- fsync at the durability boundaries (durable=True appends + the explicit
  intent-before-write fsync);
- a simulated mid-write kill leaves a trailing partial record, which is
  DISCARDED on load (never silently truncated mid-file: corruption before
  the trailing record raises);
- TranscriptLedger.attach recovers a transcript and resumes the seq.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from lechonk.core.ledger import (
    LedgerCorruptionError,
    TranscriptLedger,
    canonical_json,
    compute_checksum,
    load_transcript,
    valid_prefix_bytes,
)
from lechonk.core.schemas import (
    ActionResultRecord,
    MetricEventRecord,
    OutcomeRecord,
    TurnStartRecord,
)


def _action(turn: int = 1, **kwargs) -> ActionResultRecord:
    fields = dict(turn=turn, ref="unit:7:goto_n", domain="unit", verb="goto_n",
                  outcome="executed", provenance_class="model",
                  readback="verified")
    fields.update(kwargs)
    return ActionResultRecord(**fields)


# ---------------------------------------------------------------------------
# append + checksum
# ---------------------------------------------------------------------------


def test_append_assigns_seq_ts_checksum_and_validates(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    seq0 = ledger.append(TurnStartRecord(turn=1))
    seq1 = ledger.append(_action())
    ledger.close()
    assert (seq0, seq1) == (0, 1)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec0, rec1 = (json.loads(line) for line in lines)
    assert rec0["seq"] == 0 and rec0["ts"] and rec0["checksum"]
    assert rec1["seq"] == 1
    # The checksum covers the canonical JSON of the record minus itself.
    assert rec1["checksum"] == compute_checksum(
        {k: v for k, v in rec1.items() if k != "checksum"})
    # Round-trip through the frozen union.
    load = load_transcript(path)
    assert not load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]
    assert isinstance(load.records[1], ActionResultRecord)


def test_checksum_is_order_independent_and_canonical():
    rec = {"kind": "turn_start", "turn": 3, "seq": 0, "ts": "x", "checksum": ""}
    a = compute_checksum(rec)
    b = compute_checksum({"turn": 3, "kind": "turn_start", "ts": "x",
                          "seq": 0, "checksum": ""})
    assert a == b  # canonical JSON: sorted keys
    assert a != compute_checksum({**rec, "turn": 4})


# ---------------------------------------------------------------------------
# FW-09: strict canonical JSON (reject NaN, escape lone surrogates)
# ---------------------------------------------------------------------------


def test_canonical_json_rejects_nan_and_infinity():
    # allow_nan=False semantics: a deterministic ValueError, never a bare
    # NaN/Infinity token in the output.
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("-inf")})


def test_canonical_json_escapes_lone_surrogates():
    payload = {"s": "a\ud800b"}
    text = canonical_json(payload)
    # The surrogate is escaped, never emitted raw — valid JSON whose utf-8
    # encoding (the checksum path) cannot hit UnicodeEncodeError.
    assert text == '{"s":"a\\ud800b"}'
    assert json.loads(text) == payload
    checksum = compute_checksum(payload)
    assert checksum == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_canonical_json_clean_payloads_byte_identical():
    # Regression: for clean (non-NaN, surrogate-free) payloads the helper
    # must produce exactly the pre-FW-09 bytes (recorded-artifact
    # checksums depend on it).
    payload = {"b": 1, "a": [1, 2, {"c": "café — ü"}], "d": None, "e": True}
    assert canonical_json(payload) == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_append_record_with_lone_surrogate_roundtrips(tmp_path):
    # End to end: a surrogate-carrying record checksums on write, and the
    # read-back recomputes the same checksum (no UnicodeEncodeError).
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(MetricEventRecord(name="n\ud800", value="v"))
    ledger.close()
    load = load_transcript(path)
    assert not load.discarded_trailing_partial
    assert load.records[0].name == "n\ud800"


def test_append_rejects_records_outside_the_frozen_union(tmp_path):
    ledger = TranscriptLedger(tmp_path / "t.jsonl")
    with pytest.raises(Exception):
        ledger.append(MetricEventRecord(name="n", value="v", bogus=1))
    ledger.close()


def test_append_after_close_raises(tmp_path):
    ledger = TranscriptLedger(tmp_path / "t.jsonl")
    ledger.append(TurnStartRecord(turn=1))
    ledger.close()
    with pytest.raises(RuntimeError):
        ledger.append(TurnStartRecord(turn=2))


# ---------------------------------------------------------------------------
# durability boundaries (§3 line 34)
# ---------------------------------------------------------------------------


def test_durable_appends_and_fsync_call_os_fsync(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync

    def spy(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    ledger = TranscriptLedger(tmp_path / "t.jsonl")
    ledger.append(TurnStartRecord(turn=1))            # buffered
    assert calls == []
    ledger.fsync()                                     # intent-before-write
    assert len(calls) == 1
    ledger.append(_action(), durable=True)             # action boundary
    assert len(calls) == 2
    ledger.append(OutcomeRecord(outcome="COMPLETED"), durable=True)
    assert len(calls) == 3
    ledger.close()                                     # close fsyncs too
    assert len(calls) == 4


def test_last_event_tracks_the_injected_clock(tmp_path):
    ticks = [1000.0]
    ledger = TranscriptLedger(tmp_path / "t.jsonl", clock=lambda: ticks[0])
    assert ledger.last_event == 1000.0
    ledger.append(TurnStartRecord(turn=1))
    assert ledger.last_event == 1000.0
    ticks[0] = 1600.0  # the stall watchdog reads last_event
    ledger.append(TurnStartRecord(turn=2))
    assert ledger.last_event == 1600.0
    ledger.close()


# ---------------------------------------------------------------------------
# trailing partial record: the simulated mid-write kill (§3 line 34)
# ---------------------------------------------------------------------------


def test_trailing_partial_record_is_discarded(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_action(), durable=True)
    ledger.close()
    good_bytes = path.read_bytes()

    # Simulate a kill mid-write: torn bytes after the last newline.
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"action_result","turn":2,"ref":"unit:7:g')  # torn
    load = load_transcript(path)
    assert load.discarded_trailing_partial
    assert load.discarded_bytes > 0
    assert [r.seq for r in load.records] == [0, 1]  # prior records intact

    # A torn FINAL newline-terminated line (killed before the fsync) is
    # also a trailing partial.
    path2 = tmp_path / "t2.jsonl"
    path2.write_bytes(good_bytes + b'{"kind":"turn_start","turn":3,"seq":2\n')
    load = load_transcript(path2)
    assert load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]

    # A COMPLETE unterminated line (valid JSON + checksum) is STILL a
    # trailing partial: its terminating newline never landed, so it comes
    # from a torn-write window (never fsynced-complete) — always discarded,
    # never repaired by appending a newline.
    complete = json.loads(good_bytes.splitlines()[0])
    path3 = tmp_path / "t3.jsonl"
    path3.write_bytes(good_bytes + canonical_json(complete).encode("utf-8"))
    load = load_transcript(path3)
    assert load.discarded_trailing_partial
    assert load.discarded_bytes > 0
    assert [r.seq for r in load.records] == [0, 1]


def test_corrupt_final_line_and_unterminated_tail_both_discarded(tmp_path):
    # The exact combined torn-write case (D10, keep both): valid records,
    # then a corrupt FINAL newline-terminated line, then an unterminated
    # tail — both trailing artifacts are silently discarded together and
    # no LedgerCorruptionError is raised (only a failure BEFORE the
    # trailing record raises).
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_action(), durable=True)
    ledger.close()
    good_bytes = path.read_bytes()
    corrupt_line = b'{"kind":"turn_start","turn":2,"seq":2\n'  # no closing brace
    tail = b'{"kind":"action_result","turn":2,"ref":"unit:7:g'
    path.write_bytes(good_bytes + corrupt_line + tail)
    load = load_transcript(path)  # must not raise
    assert load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]
    assert load.discarded_bytes == len(corrupt_line) + len(tail)


def test_discarded_bytes_match_valid_prefix_bytes(tmp_path):
    # The discarded byte total is exactly what valid_prefix_bytes implies:
    # everything after the valid prefix — for each trailing-partial shape.
    def build(path, torn):
        ledger = TranscriptLedger(path)
        ledger.append(TurnStartRecord(turn=1), durable=True)
        ledger.append(_action(), durable=True)
        ledger.close()
        with open(path, "ab") as fh:
            fh.write(torn)

    cases = [
        b'{"kind":"turn_start","turn":2,"seq":2\n',   # corrupt final line
        b'{"kind":"action_result","turn":2,"ref":"x',  # unterminated tail
        b'{"kind":"turn_start","turn":2,"seq":2\n'     # both at once
        b'{"kind":"action_result","turn":2,"ref":"x',
    ]
    for i, torn in enumerate(cases):
        path = tmp_path / f"t{i}.jsonl"
        build(path, torn)
        size = path.stat().st_size
        load = load_transcript(path)
        assert load.discarded_trailing_partial
        assert load.discarded_bytes == size - valid_prefix_bytes(str(path))
        assert load.discarded_bytes > 0


def test_seq_duplicates_raise(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(TurnStartRecord(turn=2))
    ledger.close()
    lines = path.read_bytes().splitlines(keepends=True)
    # Rewrite the second line as a duplicate of seq 0 (recompute checksum).
    dup = {"kind": "turn_start", "turn": 1, "seq": 0, "ts": "x"}
    dup["checksum"] = compute_checksum(dup)
    lines[1] = canonical_json(dup).encode("utf-8") + b"\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(LedgerCorruptionError) as exc:
        load_transcript(path)
    assert str(path) in str(exc.value)


def test_seq_gap_raises(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(TurnStartRecord(turn=2))
    ledger.append(TurnStartRecord(turn=3))
    ledger.close()
    lines = path.read_bytes().splitlines(keepends=True)
    # Bump the middle record's seq to 3 (leaving a gap at 1); recompute its
    # checksum so only the continuity check can catch it.
    gap = {"kind": "turn_start", "turn": 2, "seq": 3, "ts": "x"}
    gap["checksum"] = compute_checksum(gap)
    lines[1] = canonical_json(gap).encode("utf-8") + b"\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(LedgerCorruptionError) as exc:
        load_transcript(path)
    assert str(path) in str(exc.value)


def test_trailing_blank_line_loads_cleanly_and_attach_truncates(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_action(), durable=True)
    ledger.close()
    good_bytes = path.read_bytes()
    path.write_bytes(good_bytes + b"\n")  # trailing blank line
    load = load_transcript(path)
    assert [r.seq for r in load.records] == [0, 1]  # loads cleanly

    recovered = TranscriptLedger.attach(path)
    recovered.close()
    assert path.read_bytes() == good_bytes  # attach truncated the blank line


def test_corruption_before_the_trailing_record_raises(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(TurnStartRecord(turn=2))
    ledger.append(TurnStartRecord(turn=3))
    ledger.close()
    lines = path.read_bytes().splitlines(keepends=True)
    lines[0] = b'{"kind":"turn_start","turn":1,"seq":0,"ts":"x","checksum":"bad"}\n'
    path.write_bytes(b"".join(lines))
    with pytest.raises(LedgerCorruptionError):
        load_transcript(path)


def test_attach_recovers_and_resumes_seq(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_action(), durable=True)
    ledger.close()
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"action_res')  # torn trailing write

    recovered = TranscriptLedger.attach(path)
    assert recovered.next_seq == 2  # resumes after the last valid record
    # The torn trailing record was truncated out of the FILE (not just the
    # load view): the next append lands right after the last valid line.
    seq = recovered.append(TurnStartRecord(turn=2), durable=True)
    assert seq == 2
    recovered.close()
    load = load_transcript(path)
    assert not load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1, 2]


def test_load_missing_or_empty_file(tmp_path):
    load = load_transcript(tmp_path / "missing.jsonl")
    assert load.records == [] and not load.discarded_trailing_partial
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    load = load_transcript(empty)
    assert load.records == []


# ---------------------------------------------------------------------------
# W3-1 (D5): the durable dispatch-intent record at the crash-window
# boundary — load-level behavior
# ---------------------------------------------------------------------------


def _dispatch_intent_event(turn=1, ref="unit:10:goto_e"):
    return MetricEventRecord(
        name="dispatch_intent",
        value=canonical_json({
            "protocol": "dispatch-intent/v1", "turn": turn, "ref": ref,
            "call_seq": None, "snapshot_seq": 0, "candidates_seq": 0,
            "provenance_class": "model"}))


def test_durable_intent_at_eof_is_kept_on_load(tmp_path):
    """W3-1 (D5): a transcript ending with a durable dispatch_intent (a
    kill between the intent fsync and the send) loads cleanly — the
    unresolved intent is KEPT (never discarded as a trailing partial)."""
    path = tmp_path / "intent.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    seq = ledger.append(_dispatch_intent_event(), durable=True)
    ledger.close()
    load = load_transcript(path)
    assert not load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]
    last = load.records[-1]
    assert isinstance(last, MetricEventRecord)
    assert last.name == "dispatch_intent"
    assert last.seq == seq


def test_torn_intent_tail_is_discarded(tmp_path):
    """W3-1 (D5): a kill mid-write of the intent line leaves torn bytes
    at EOF — the trailing partial is discarded; the earlier records
    (including earlier durable intents) load."""
    path = tmp_path / "torn-intent.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_dispatch_intent_event(), durable=True)
    ledger.close()
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"metric_event","name":"dispatch_inte')
    load = load_transcript(path)
    assert load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]
    assert load.records[-1].name == "dispatch_intent"


def test_torn_result_after_intent_leaves_it_unresolved(tmp_path):
    """W3-1 (D5): a kill after the send but mid-write of the
    action_result — the torn result is discarded on load and the intent
    remains the last record (unresolved: the send may or may not have
    gone out — the honest crash-window state)."""
    path = tmp_path / "torn-result.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    ledger.append(_dispatch_intent_event(), durable=True)
    ledger.close()
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"action_result","turn":1,"ref":"unit:10:g')
    load = load_transcript(path)
    assert load.discarded_trailing_partial
    assert [r.seq for r in load.records] == [0, 1]
    # The intent survives as the unresolved tail record.
    last = load.records[-1]
    assert isinstance(last, MetricEventRecord)
    assert last.name == "dispatch_intent"
