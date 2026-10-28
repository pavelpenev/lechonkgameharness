from __future__ import annotations

import json

import pytest

from lechonk.core.context_journal import (
    ContextJournal,
    ContextJournalCorruptionError,
    load_context_journal,
)


def _episode(turn: int) -> dict[str, object]:
    return {
        "turn": turn,
        "intent": {"notes": "scout", "planned": ["unit:1:move_north"]},
        "execution": [{"ref": "unit:1:move_north", "outcome": "verified"}],
    }


def test_round_trip_and_order(tmp_path):
    path = tmp_path / "context.jsonl"
    with ContextJournal(path) as journal:
        first = journal.append(turn=1, episode=_episode(1),
                               transcript_seq=8, rendered_history_hash="h1")
        second = journal.append(turn=2, episode=_episode(2),
                                transcript_seq=15, rendered_history_hash="h2")
        assert (first.seq, second.seq) == (0, 1)
    loaded = load_context_journal(path)
    assert [record.turn for record in loaded.records] == [1, 2]
    assert loaded.records[1].episode["execution"][0]["outcome"] == "verified"


def test_unterminated_tail_is_reported_and_attach_truncates(tmp_path):
    path = tmp_path / "context.jsonl"
    with ContextJournal(path) as journal:
        journal.append(turn=1, episode=_episode(1), rendered_history_hash="h")
    with path.open("ab") as fh:
        fh.write(b'{"kind":"context_checkpoint"')
    loaded = load_context_journal(path)
    assert loaded.discarded_trailing_partial
    with ContextJournal.attach(path) as journal:
        assert journal.next_seq == 1
        journal.append(turn=2, episode=_episode(2), rendered_history_hash="h2")
    assert len(load_context_journal(path).records) == 2


def test_nontrailing_corruption_is_not_hidden(tmp_path):
    path = tmp_path / "context.jsonl"
    with ContextJournal(path) as journal:
        journal.append(turn=1, episode=_episode(1), rendered_history_hash="h")
        journal.append(turn=2, episode=_episode(2), rendered_history_hash="h2")
    lines = path.read_bytes().splitlines()
    lines[0] = b"{}"
    path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ContextJournalCorruptionError):
        load_context_journal(path)


def test_newline_terminated_corrupt_final_record_is_not_recovered(tmp_path):
    path = tmp_path / "context.jsonl"
    with ContextJournal(path) as journal:
        journal.append(turn=1, episode=_episode(1), rendered_history_hash="h")
    with path.open("ab") as fh:
        fh.write(b'{"kind":"context_checkpoint"}\n')
    with pytest.raises(ContextJournalCorruptionError):
        load_context_journal(path)


def test_attach_rejects_policy_version_mismatch(tmp_path):
    path = tmp_path / "context.jsonl"
    with ContextJournal(path, policy_version="hybrid_v1") as journal:
        journal.append(turn=1, episode=_episode(1), rendered_history_hash="h")
    with pytest.raises(ContextJournalCorruptionError, match="policy version mismatch"):
        ContextJournal.attach(path, policy_version="hybrid_v2")
