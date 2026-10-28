# SPDX-License-Identifier: MIT
import json
import hashlib
import os
import stat
from pathlib import Path

import pytest
import lechonk.model.conversation as conversation_module

from lechonk.model.conversation import (
    CompactionRequired,
    ConversationConfig,
    ConversationError,
    ConversationPersistenceError,
    ConversationManager,
)


def _manager(tmp_path: Path | None = None) -> ConversationManager:
    return ConversationManager(
        system_message={"role": "system", "content": "stable instructions"},
        state_path=(tmp_path / "state.json" if tmp_path else None),
    )


def test_exact_prefix_and_full_thinking_chunk_round_trip(tmp_path):
    manager = _manager(tmp_path)
    first = manager.begin_turn({"role": "user", "content": "turn one"})
    manager.stage_assistant([{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                             {"type": "text", "text": "{\"actions\": []}"}])
    committed = manager.commit_turn(execution_report={"turn": 1, "outcome": "verified"})
    second = manager.begin_turn({"role": "user", "content": "turn two"})
    assert second[: len(first)] == first
    assert committed[2]["content"][0]["type"] == "thinking"


def test_tool_lookup_transaction_preserves_thinking_and_exact_prefix(tmp_path):
    manager = _manager(tmp_path)
    initial = manager.begin_turn({"role": "user", "content": "look up aqueduct"})
    exchange = initial + [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]}],
         "tool_calls": [{"id": "call-1", "type": "function",
                          "function": {"name": "lookup_freeciv", "arguments": "{\"topic\":\"aqueduct\"}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Aqueduct: improves city growth."},
        {"role": "assistant", "content": [{"type": "text", "text": "{\"actions\": []}"}]},
    ]
    committed = manager.commit_exchange(exchange)
    assert committed[:len(exchange)] == exchange
    follow_up = manager.begin_turn({"role": "user", "content": "next"})
    assert follow_up[:len(committed)] == committed


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
    [{"role": "assistant", "content": "x", "tool_calls": [{"id": "c", "type": "function"}]}],
    [{"role": "assistant", "content": "x", "tool_calls": [{"id": "c", "type": "function"}]},
     {"role": "tool", "tool_call_id": "wrong", "content": "x"},
     {"role": "assistant", "content": "final"}],
])
def test_invalid_or_incomplete_tool_transaction_is_rejected(messages):
    manager = _manager()
    manager.begin_turn({"role": "user", "content": "question"})
    with pytest.raises(ConversationError):
        manager.commit_exchange([{"role": "user", "content": "question"}, *messages])


def test_tool_transaction_recovery_repair_is_valid_and_compacts_as_one_group():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=200, tail_tokens=80),
    )
    initial = manager.begin_turn({"role": "user", "content": "question"})
    manager.commit_exchange(initial + [
        {"role": "assistant", "content": "bad", "tool_calls": [{"id": "c", "type": "function"}]},
        {"role": "tool", "tool_call_id": "c", "content": "lookup result"},
        {"role": "assistant", "content": "bad final"},
        {"role": "user", "content": "repair"},
        {"role": "assistant", "content": "good final"},
    ])
    manager.compact(lambda old: {"version": 1, "summary": "recovered", "commitments": [],
                                 "uncertainties": [], "facts": []}, force=True)
    assert "recovered" in manager.messages[1]["content"]


def test_tool_transaction_usage_calibration_counts_complete_exchange_once(monkeypatch):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=190, tail_tokens=50),
    )
    initial = manager.begin_turn({"role": "user", "content": "question"})
    exchange = initial + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "lookup_freeciv", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "result"},
        {"role": "assistant", "content": "final"},
    ]
    # Use deterministic message units to make the expected calibration
    # explicit: provider prompt 100 + (four suffix messages after the
    # request, 40 - assistant 10 + completion 50) = 180, then 10 new.
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(100, estimated_tokens=100, message_count=2,
                         completion_tokens=50)
    manager.commit_exchange(exchange, execution_report={"turn": 1})
    with pytest.raises(CompactionRequired):
        manager.begin_turn({"role": "user", "content": "next"})


def test_prefixed_request_estimate_uses_provider_only_for_verified_prefix(monkeypatch):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "u"},
                             {"role": "assistant", "content": "a"}])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(
        100, estimated_tokens=20, message_count=2,
        request_hash="sha256:" + hashlib.sha256(
            conversation_module.canonical_json(manager.messages[:2]).encode("utf-8")
        ).hexdigest())
    result = manager.estimate_prefixed_request(manager.messages[:2],
                                               [{"role": "user", "content": "instruction"}],
                                               guard_tokens=5)
    assert result["method"] == "provider_prefix"
    assert result["raw"] == 30 and result["effective"] == 115
    mismatch = manager.estimate_prefixed_request(
        [{"role": "system", "content": "different"}],
        [{"role": "user", "content": "instruction"}], guard_tokens=5)
    assert mismatch["method"] == "raw" and mismatch["effective"] == 25


def test_unknown_current_usage_invalidates_previous_calibration():
    manager = _manager()
    manager.record_usage(311_027, estimated_tokens=123,
                         message_count=1, completion_tokens=40_317,
                         request_hash="sha256:initial")
    assert manager.observed_prompt_tokens == 311_027
    assert manager.observed_completion_tokens == 40_317
    assert manager.observed_request_hash == "sha256:initial"
    manager.record_usage(None)
    assert manager.observed_prompt_tokens == 0
    assert manager.observed_completion_tokens is None
    assert manager.observed_request_hash is None


def test_missing_local_estimate_does_not_reuse_previous_estimate():
    manager = _manager()
    manager.record_usage(100, estimated_tokens=999, message_count=1,
                         completion_tokens=10,
                         request_hash="sha256:" + "b" * 64)
    manager.record_usage(200, estimated_tokens=None, message_count=1,
                         completion_tokens=20,
                         request_hash="sha256:" + "c" * 64)
    assert manager.observed_prompt_tokens == 200
    assert manager.observed_estimated_tokens == 0
    assert manager.observed_completion_tokens == 20


def test_prefixed_request_estimate_rejects_invalid_calibration_and_subset_is_safe(monkeypatch):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "u"},
                             {"role": "assistant", "content": "a"}])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(500_595, estimated_tokens=1_113_200, message_count=3)
    result = manager.estimate_prefixed_request(manager.messages[:2],
                                               [{"role": "user", "content": "instruction"}],
                                               guard_tokens=8_192)
    assert result["method"] == "raw"


def test_prefixed_request_actual_provider_shape_uses_proven_prefix(monkeypatch):
    manager = _manager()
    manager.messages.extend({"role": "user", "content": str(i)} for i in range(143))
    def shaped(messages):
        count = len(list(messages))
        return {144: 1_113_200, 137: 1_053_001, 1: 123}.get(count, count)
    monkeypatch.setattr(conversation_module, "_message_tokens", shaped)
    manager.record_usage(
        500_595, estimated_tokens=1_113_200, message_count=144,
        request_hash="sha256:" + hashlib.sha256(
            conversation_module.canonical_json(manager.messages[:144]).encode("utf-8")
        ).hexdigest())
    result = manager.estimate_prefixed_request(manager.messages[:137],
                                               [{"role": "user", "content": "instruction"}],
                                               guard_tokens=8_192)
    assert result["method"] == "provider_prefix"
    assert result["effective"] == 508_910


def test_prefixed_request_extends_verified_provider_prefix_through_response(monkeypatch):
    manager = _manager()
    manager.messages.extend([
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "long model response"},
        {"role": "user", "content": "[EXECUTION REPORT]\\n{}"},
    ])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    prefix_count = 2
    request_hash = "sha256:" + hashlib.sha256(
        conversation_module.canonical_json(manager.messages[:prefix_count]).encode("utf-8")
    ).hexdigest()
    manager.record_usage(100, estimated_tokens=20, message_count=prefix_count,
                         completion_tokens=40, request_hash=request_hash)
    result = manager.estimate_prefixed_request(
        manager.messages, [{"role": "user", "content": "compact"}], guard_tokens=5)
    assert result["method"] == "provider_extended_prefix"
    assert result["effective"] == 165


def test_prefixed_request_extended_provider_prefix_rejects_hash_mismatch(monkeypatch):
    manager = _manager()
    manager.messages.extend([
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "report"},
    ])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(100, estimated_tokens=20, message_count=2,
                         completion_tokens=40, request_hash="sha256:" + "0" * 64)
    result = manager.estimate_prefixed_request(
        manager.messages, [{"role": "user", "content": "compact"}], guard_tokens=5)
    assert result["method"] == "raw"


def test_prefixed_request_equal_size_changed_content_requires_hash(monkeypatch):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "same-size"}])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(100, estimated_tokens=20, message_count=2,
                         request_hash="sha256:" + "f" * 64)
    result = manager.estimate_prefixed_request(
        manager.messages[:2], [{"role": "user", "content": "next"}], guard_tokens=5)
    assert result["method"] == "raw"


def test_prefixed_request_legacy_calibration_without_hash_is_conservative(monkeypatch):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "u"}])
    monkeypatch.setattr(conversation_module, "_message_tokens",
                        lambda messages: len(list(messages)) * 10)
    manager.record_usage(100, estimated_tokens=20, message_count=2)
    result = manager.estimate_prefixed_request(
        manager.messages[:2], [{"role": "user", "content": "next"}], guard_tokens=5)
    assert result["method"] == "raw"


def test_oversized_compaction_summary_is_rejected_atomically():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
    )
    manager.messages.extend([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
    ])
    before = manager.request_messages()
    oversized = {"version": 1, "summary": "ok", "commitments": [],
                  "uncertainties": [], "facts": ["x" * 2_000] * 20}
    with pytest.raises(ConversationError, match="token budget"):
        manager.compact(lambda _: oversized, force=True)
    assert manager.request_messages() == before


def test_transaction_abort_does_not_publish_partial():
    manager = _manager()
    manager.begin_turn({"role": "user", "content": "x"})
    manager.stage_assistant("partial")
    manager.abort_turn()
    assert manager.request_messages() == [{"role": "system", "content": "stable instructions"}]


def test_engine_only_fallback_report_round_trips_and_compacts(tmp_path):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=40, tail_tokens=20),
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "conversation.jsonl",
    )
    manager.append_execution_report({"turn": 1, "outcome": "degraded"})
    restored = ConversationManager.load(tmp_path / "state.json")
    assert "[ENGINE-ONLY REPORT]" in restored.request_messages()[1]["content"]
    restored.compact(lambda _: {"version": 1, "summary": "fallback",
                                "commitments": [], "uncertainties": [], "facts": []},
                     force=True)


def test_calibration_hash_round_trips_and_legacy_state_without_hash_loads(tmp_path):
    state = tmp_path / "state.json"
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        state_path=state, journal_path=tmp_path / "conversation.jsonl")
    manager.record_usage(311_027, estimated_tokens=123,
                         message_count=1, completion_tokens=40_317,
                         request_hash="sha256:" + "a" * 64)
    manager._persist()
    restored = ConversationManager.load(state)
    assert restored.observed_request_hash == "sha256:" + "a" * 64
    payload = json.loads(state.read_text())
    payload.pop("observed_request_hash", None)
    payload.pop("checksum", None)
    payload["checksum"] = hashlib.sha256(
        conversation_module.canonical_json(payload).encode("utf-8")).hexdigest()
    state.write_text(json.dumps(payload) + "\n")
    legacy = ConversationManager.load(state)
    assert legacy.observed_request_hash is None


def test_fork_persistence_is_self_contained_and_parent_remains_unchanged(tmp_path):
    parent = ConversationManager(
        system_message={"role": "system", "content": "s"},
        state_path=tmp_path / "parent.json", journal_path=tmp_path / "parent.jsonl")
    parent._persist()
    parent.begin_turn({"role": "user", "content": "u"})
    parent.stage_assistant("a")
    parent.commit_turn()
    before = parent.request_messages()
    parent.fork_persistence(tmp_path / "child.json", tmp_path / "conversation.jsonl")
    child = ConversationManager.load(tmp_path / "child.json")
    assert child.request_messages() == before
    assert child._journal_records_applied == 1
    child.begin_turn({"role": "user", "content": "child"})
    child.stage_assistant("b")
    child.commit_turn()
    assert ConversationManager.load(tmp_path / "child.json").messages == child.messages
    assert parent.journal_path == tmp_path / "conversation.jsonl"


def test_fork_compaction_reloads_from_base_record(tmp_path):
    parent = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10))
    parent.messages.extend([{"role": "user", "content": "old"},
                            {"role": "assistant", "content": "answer"}])
    parent.fork_persistence(tmp_path / "child.json", tmp_path / "conversation.jsonl")
    child = ConversationManager.load(tmp_path / "child.json")
    child.compact(lambda _: {"version": 1, "summary": "kept", "commitments": [],
                             "uncertainties": [], "facts": []}, force=True)
    restored = ConversationManager.load(tmp_path / "child.json")
    assert "kept" in restored.messages[1]["content"]


def test_fork_rejects_staged_or_poisoned_parent(tmp_path):
    parent = _manager(tmp_path)
    parent.begin_turn({"role": "user", "content": "pending"})
    with pytest.raises(ConversationError):
        parent.fork_persistence(tmp_path / "child.json", tmp_path / "conversation.jsonl")
    parent.abort_turn()
    parent._poisoned = True
    with pytest.raises(ConversationPersistenceError):
        parent.fork_persistence(tmp_path / "child.json", tmp_path / "conversation.jsonl")


def test_fork_refuses_existing_destinations_and_nonstandard_journal(tmp_path):
    parent = _manager(tmp_path)
    state = tmp_path / "child.json"
    journal = tmp_path / "conversation.jsonl"
    state.write_bytes(b"keep-state")
    with pytest.raises(ConversationError, match="destinations"):
        parent.fork_persistence(state, journal)
    assert state.read_bytes() == b"keep-state"
    state.unlink()
    with pytest.raises(ConversationError, match="conversation.jsonl"):
        parent.fork_persistence(state, tmp_path / "other.jsonl")


def test_fork_reports_durable_journal_when_child_state_publish_fails(tmp_path, monkeypatch):
    parent = _manager(tmp_path)
    state = tmp_path / "child.json"
    journal = tmp_path / "conversation.jsonl"
    original = parent._persist_to
    def fail_state(path, *, journal_records_applied):
        if Path(path) == state:
            raise OSError("state publication failed")
        return original(path, journal_records_applied=journal_records_applied)
    monkeypatch.setattr(parent, "_persist_to", fail_state)
    with pytest.raises(ConversationPersistenceError, match="journal is durable"):
        parent.fork_persistence(state, journal)
    assert journal.exists() and not state.exists()
    assert parent.journal_path is None


def test_standalone_export_resets_wal_marker_and_reloads(tmp_path):
    manager = _manager(tmp_path)
    manager.begin_turn({"role": "user", "content": "u"})
    manager.stage_assistant("a")
    manager.commit_turn()
    standalone = tmp_path / "checkpoint" / "state.json"
    manager.export_standalone(standalone)
    data = __import__("json").loads(standalone.read_text())
    assert data["journal_records_applied"] == 0
    restored = ConversationManager.load(standalone)
    assert restored.messages == manager.messages


@pytest.mark.parametrize("kind", ["normal", "repair", "tool"])
def test_engine_only_after_completed_group_compacts_and_reloads(tmp_path, kind):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
        state_path=tmp_path / f"{kind}.json")
    initial = manager.begin_turn({"role": "user", "content": "question"})
    if kind == "normal":
        manager.stage_assistant("answer")
        manager.commit_turn(execution_report={"turn": 1})
    elif kind == "repair":
        manager.commit_exchange(initial + [
            {"role": "assistant", "content": "bad"},
            {"role": "user", "content": "repair"},
            {"role": "assistant", "content": "good"}],
            execution_report={"turn": 1})
    else:
        manager.commit_exchange(initial + [
            {"role": "assistant", "content": "lookup", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "result"},
            {"role": "assistant", "content": "final"}],
            execution_report={"turn": 1})
    manager.append_execution_report({"turn": 2, "outcome": "degraded"})
    manager.compact(lambda _: {"version": 1, "summary": "complete", "commitments": [],
                               "uncertainties": [], "facts": []}, force=True)
    restored = ConversationManager.load(tmp_path / f"{kind}.json")
    assert "complete" in restored.messages[1]["content"]


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "bare"}],
    [{"role": "user", "content": "q"},
     {"role": "assistant", "content": "lookup", "tool_calls": [
         {"id": "c", "type": "function"}]},
     {"role": "tool", "tool_call_id": "c", "content": "result"}],
])
def test_engine_only_after_incomplete_group_is_rejected(messages):
    manager = _manager()
    manager.messages.extend(messages)
    manager.messages.append({"role": "user", "content": "[ENGINE-ONLY REPORT]\n{}"})
    before = manager.request_messages()
    with pytest.raises(ConversationError, match="incomplete"):
        manager.compact(lambda _: {"version": 1, "summary": "x", "commitments": [],
                                   "uncertainties": [], "facts": []}, force=True)
    assert manager.request_messages() == before


def test_repeated_compaction_preserves_cache_epochs_tail_usage_and_wal(tmp_path):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=500, tail_tokens=150),
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "conversation.jsonl",
        prompt_cache_key="cache")
    manager._persist()
    for i in range(3):
        manager.begin_turn({"role": "user", "content": f"first-{i}"})
        manager.stage_assistant(f"answer-{i}")
        manager.commit_turn(execution_report={"turn": i})
    manager.compact(lambda _: {"version": 1, "summary": "first summary",
                               "commitments": ["hold research"],
                               "uncertainties": [], "facts": []}, force=True)
    assert manager.epoch == 1
    assert manager.prompt_cache_key == "cache:epoch-1"
    for i in range(3, 6):
        manager.begin_turn({"role": "user", "content": f"second-{i}"})
        manager.stage_assistant(f"answer-{i}")
        manager.commit_turn(execution_report={"turn": i})
    manager.compact(lambda _: {"version": 1, "summary": "second summary",
                               "commitments": ["hold expansion"],
                               "uncertainties": [], "facts": []}, force=True)
    assert manager.epoch == 2
    assert manager.prompt_cache_key == "cache:epoch-1:epoch-2"
    assert "second summary" in manager.messages[1]["content"]
    assert any(message.get("content") == "answer-5" for message in manager.messages)
    manager.record_usage(777, estimated_tokens=222, message_count=len(manager.messages),
                         completion_tokens=33)
    manager.append_execution_report({"turn": 99, "outcome": "calibration"})
    restored = ConversationManager.load(tmp_path / "state.json")
    assert restored.epoch == 2
    assert restored.prompt_cache_key == manager.prompt_cache_key
    assert restored.observed_prompt_tokens == 777
    records = ConversationManager.load_journal(tmp_path / "conversation.jsonl")
    assert sum(record["kind"] == "conversation_compaction" for record in records) == 2
    assert len(records) == len({record["seq"] for record in records})


def test_provider_completion_tokens_drive_next_projection_not_text_estimate():
    manager = ConversationManager(
        system_message={"role": "system", "content": "system"},
        config=ConversationConfig(trigger_tokens=1_000, tail_tokens=100),
    )
    manager.begin_turn({"role": "user", "content": "play"})
    manager.stage_assistant("brief")
    # The provider reports a large reasoning/completion payload even though
    # the retained answer text is short.  The next request must account for
    # that authoritative output count rather than chars/4 of "brief".
    manager.record_usage(100, estimated_tokens=100, message_count=2,
                         completion_tokens=5_000)
    manager.commit_turn()
    with pytest.raises(CompactionRequired):
        manager.begin_turn({"role": "user", "content": "next"})


def test_compaction_required_and_compactor_failure_is_atomic():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
    )
    with pytest.raises(CompactionRequired):
        manager.begin_turn({"role": "user", "content": "x" * 100})
    manager.messages.append({"role": "user", "content": "x" * 100})
    before = manager.request_messages()
    with pytest.raises(RuntimeError):
        manager.compact(lambda _: (_ for _ in ()).throw(RuntimeError("no summary")))
    assert manager.request_messages() == before


@pytest.mark.parametrize("summary", [{}, {"summary": "x"},
                                      {"summary": "x", "commitments": {},
                                       "uncertainties": [], "facts": []},
                                      {"summary": "", "commitments": [],
                                       "uncertainties": [], "facts": []}])
def test_invalid_compaction_summary_is_rejected_atomically(summary):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
    )
    manager.messages.extend([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
    ])
    before = manager.request_messages()
    with pytest.raises(ConversationError):
        manager.compact(lambda _: summary, force=True)
    assert manager.request_messages() == before


def test_typed_compaction_commitments_are_preserved_and_strict():
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    typed = {"version": 1, "summary": "keep", "commitments": [{
        "domain": "research", "target": "Writing", "target_kind": "tech",
        "verb": "research", "deadline_turn": 12}],
        "uncertainties": [], "facts": []}
    manager.compact(lambda _: typed, force=True)
    assert '"target_kind":"tech"' in manager.messages[1]["content"]
    before = manager.request_messages()
    invalid = dict(typed, commitments=[dict(typed["commitments"][0], extra="x")])
    with pytest.raises(ConversationError):
        manager.compact(lambda _: invalid, force=True)
    assert manager.request_messages() == before


def test_real_provider_compaction_historical_note_round_trips_and_reloads(tmp_path):
    manager = _manager(tmp_path)
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    # Exact historical_note emitted by the real provider response that exposed
    # the missing additive field (135 characters).
    note = (
        "All observations above are historical context as of turn 35; the next "
        "authoritative SNAPSHOT and CANDIDATE TABLE override this summary."
    )
    provider_summary = {
        "version": 1,
        "summary": "keep the strategic state",
        "commitments": [],
        "uncertainties": ["enemy intentions"],
        "facts": ["frontier city is defended"],
        "historical_note": note,
    }

    manager.compact(lambda _: provider_summary, force=True)
    payload = json.loads(manager.messages[1]["content"].split("\n", 1)[1])
    assert payload["summary"]["historical_note"] == note

    restored = ConversationManager.load(tmp_path / "state.json")
    restored_payload = json.loads(restored.messages[1]["content"].split("\n", 1)[1])
    assert restored_payload["summary"]["historical_note"] == note
    assert restored.request_messages() == manager.request_messages()


@pytest.mark.parametrize("historical_note", [None, 1, [], "", "   ", "x" * 2_001])
def test_compaction_historical_note_must_be_valid_when_present(historical_note):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    summary = {"version": 1, "summary": "keep", "commitments": [],
               "uncertainties": [], "facts": [], "historical_note": historical_note}
    before = manager.request_messages()
    with pytest.raises(ConversationError):
        manager.compact(lambda _: summary, force=True)
    assert manager.request_messages() == before


def test_compaction_historical_note_is_trimmed_and_unknown_fields_remain_rejected():
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    summary = {"version": 1, "summary": "keep", "commitments": [],
               "uncertainties": [], "facts": [], "historical_note": "  context  "}
    manager.compact(lambda _: summary, force=True)
    payload = json.loads(manager.messages[1]["content"].split("\n", 1)[1])
    assert payload["summary"]["historical_note"] == "context"

    manager.messages.extend([{"role": "user", "content": "later"},
                             {"role": "assistant", "content": "answer"}])
    invalid = dict(summary, extra="must not be ignored")
    before = manager.request_messages()
    with pytest.raises(ConversationError, match="unknown fields"):
        manager.compact(lambda _: invalid, force=True)
    assert manager.request_messages() == before


def test_compaction_historical_note_counts_toward_aggregate_budget(monkeypatch):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    summary = {"version": 1, "summary": "keep", "commitments": [],
               "uncertainties": [], "facts": [], "historical_note": "context"}
    original_estimate = conversation_module.estimate_tokens

    def estimate(text):
        if '"historical_note"' in text:
            return 8_001
        return original_estimate(text)

    monkeypatch.setattr(conversation_module, "estimate_tokens", estimate)
    with pytest.raises(ConversationError, match="token budget"):
        manager.compact(lambda _: summary, force=True)


@pytest.mark.parametrize("bad", [True, 1.0, {"version": 1}])
def test_compaction_summary_version_and_top_level_fields_are_strict(bad):
    manager = _manager()
    manager.messages.extend([{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "answer"}])
    summary = {"version": bad, "summary": "x", "commitments": [],
               "uncertainties": [], "facts": []}
    if isinstance(bad, dict):
        summary = {"version": 1, "summary": "x", "commitments": [],
                   "uncertainties": [], "facts": [], "extra": True}
    with pytest.raises(ConversationError):
        manager.compact(lambda _: summary, force=True)


def test_compaction_keeps_summary_and_tail(tmp_path):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
        state_path=tmp_path / "state.json",
    )
    for i in range(5):
        manager.messages.extend([
            {"role": "user", "content": f"old-{i}-" + "x" * 12},
            {"role": "assistant", "content": "answer"},
        ])
    manager.compact(lambda _: {"version": 1, "summary": "keep this", "commitments": [],
                               "uncertainties": [], "facts": []})
    assert manager.messages[1]["content"].startswith("[CONVERSATION COMPACTION]")
    assert "keep this" in manager.messages[1]["content"]
    assert (tmp_path / "state.json").exists()
    restored = ConversationManager.load(tmp_path / "state.json")
    assert restored.request_messages() == manager.request_messages()


def test_compaction_never_splits_turn_group(tmp_path):
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=300, tail_tokens=25),
        journal_path=tmp_path / "conversation.jsonl",
    )
    for i in range(5):
        manager.begin_turn({"role": "user", "content": f"turn {i}"})
        manager.stage_assistant([{"type": "thinking", "thinking": [{"type": "text", "text": "r"}]}])
        manager.commit_turn(execution_report={"turn": i, "ok": True})
    manager.messages.extend(manager.messages[1:] * 4)
    manager.compact(lambda _: {"version": 1, "summary": "durable", "commitments": [],
                               "uncertainties": [], "facts": []})
    tail = manager.messages[2:]
    assert all(message["role"] != "assistant" for message in tail[:1])
    assert all(
        not (message.get("role") == "user" and
             isinstance(message.get("content"), dict) and
             message["content"].get("type") == "execution_report")
        or index > 0 and tail[index - 1].get("role") == "assistant"
        for index, message in enumerate(tail)
    )


def test_journal_is_checksummed_and_rejects_corruption(tmp_path):
    path = tmp_path / "conversation.jsonl"
    manager = _manager(tmp_path)
    manager.journal_path = path
    manager.begin_turn({"role": "user", "content": "turn"})
    manager.stage_assistant("answer")
    manager.commit_turn()
    raw = path.read_text(encoding="utf-8")
    assert '"checksum"' in raw
    assert len(ConversationManager.load_journal(path)) == 1
    rebuilt = ConversationManager.reconstruct(
        {"role": "system", "content": "stable instructions"},
        ConversationManager.load_journal(path),
    )
    assert rebuilt.request_messages() == manager.request_messages()
    path.write_text(raw.replace('"epoch":0', '"epoch":99'), encoding="utf-8")
    with pytest.raises(ConversationError):
        ConversationManager.load_journal(path)


def test_repair_report_and_next_turn_compact_as_complete_groups():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=200, tail_tokens=40),
        prompt_cache_key="run",
    )
    for turn in range(3):
        initial = manager.begin_turn({"role": "user", "content": f"turn {turn}"})
        exchange = initial + [
            {"role": "assistant", "content": "bad"},
            {"role": "user", "content": "repair"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "ok"}]}]},
        ]
        manager.commit_exchange(exchange, execution_report={"turn": turn, "outcome": "verified"})
    manager.compact(lambda _: {"version": 1, "summary": "facts", "commitments": [],
                               "uncertainties": [], "facts": []})
    messages = manager.request_messages()
    assert messages[1]["content"].startswith("[CONVERSATION COMPACTION]")
    assert manager.prompt_cache_key.endswith("epoch-1")
    for index, message in enumerate(messages[2:], 2):
        if message["role"] == "assistant":
            assert messages[index - 1]["role"] in ("user",)


def test_compactor_receives_only_evicted_prefix():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=100, tail_tokens=50),
    )
    for label in ("old", "new"):
        manager.begin_turn({"role": "user", "content": label})
        manager.stage_assistant(label)
        manager.commit_turn(execution_report={"label": label})
    seen = []
    manager.compact(lambda messages: (seen.append(messages) or {
        "version": 1, "summary": "old", "commitments": [], "uncertainties": [], "facts": []
    }), force=True)
    flattened = " ".join(str(message.get("content")) for batch in seen for message in batch)
    assert "old" in flattened
    assert "new" not in flattened


def test_repeated_compaction_excludes_prior_checkpoint_and_requires_checksum(tmp_path):
    state = tmp_path / "state.json"
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=200, tail_tokens=20),
        state_path=state,
    )
    for i in range(3):
        manager.begin_turn({"role": "user", "content": f"turn-{i}"})
        manager.stage_assistant("answer")
        manager.commit_turn(execution_report={"turn": i})
    manager.compact(lambda _: {"version": 1, "summary": "first", "commitments": [],
                               "uncertainties": [], "facts": []}, force=True)
    manager.messages.extend([{"role": "user", "content": "later"}, {"role": "assistant", "content": "answer"}])
    manager.compact(lambda old: {"version": 1, "summary": "second", "commitments": [],
                                "uncertainties": [], "facts": []}, force=True)
    assert manager.messages[1]["content"].startswith("[CONVERSATION COMPACTION]")
    broken = state.read_text(encoding="utf-8").replace('"checksum"', '"missing_checksum"', 1)
    state.write_text(broken, encoding="utf-8")
    with pytest.raises(ConversationError):
        ConversationManager.load(state)


def test_provider_prompt_usage_calibrates_next_threshold():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=500_000, tail_tokens=100_000),
    )
    manager.record_usage(499_999)
    with pytest.raises(CompactionRequired):
        manager.begin_turn({"role": "user", "content": "new turn"})


def test_provider_usage_plus_post_request_assistant_report_is_counted():
    manager = ConversationManager(
        system_message={"role": "system", "content": "s"},
        config=ConversationConfig(trigger_tokens=500_000, tail_tokens=100_000),
    )
    manager.record_usage(490_000, estimated_tokens=10)
    manager.messages.extend([
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "x" * 60_000},
        {"role": "user", "content": "[EXECUTION REPORT] {\"ok\":true}"},
    ])
    with pytest.raises(CompactionRequired):
        manager.begin_turn({"role": "user", "content": "next"})


def test_provider_calibration_survives_state_round_trip(tmp_path):
    state = tmp_path / "state.json"
    manager = _manager(tmp_path)
    manager.record_usage(499_999, estimated_tokens=100, message_count=1,
                         completion_tokens=12)
    # Persist a completed state without inventing an assistant response.
    manager._persist()
    restored = ConversationManager.load(state,
                                        config=ConversationConfig(trigger_tokens=500_000))
    assert restored.observed_prompt_tokens == 499_999
    with pytest.raises(CompactionRequired):
        restored.begin_turn({"role": "user", "content": "next"})


def test_engine_only_execution_report_journal_round_trip(tmp_path):
    state = tmp_path / "state.json"
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        state_path=state, journal_path=journal,
    )
    manager.append_execution_report({"turn": 1, "outcome": "fallback"})
    records = ConversationManager.load_journal(journal)
    assert records[0]["kind"] == "execution_report"
    restored = ConversationManager.reconstruct(
        {"role": "system", "content": "stable"}, records)
    assert any("ENGINE-ONLY REPORT" in str(m["content"])
               for m in restored.messages)
    loaded = ConversationManager.load(state)
    assert any("ENGINE-ONLY REPORT" in str(m["content"])
               for m in loaded.messages)


def test_state_persist_propagates_parent_directory_fsync_failure(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    manager = _manager(tmp_path)
    with pytest.raises(OSError, match="directory fsync failed"):
        manager._persist()
    # The replacement may already be visible, but the caller receives the
    # durability failure and must not treat it as a durable checkpoint.
    assert state.exists()


def test_new_journal_propagates_parent_directory_fsync_failure(tmp_path, monkeypatch):
    journal = tmp_path / "conversation.jsonl"
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        journal_path=journal,
    )
    with pytest.raises(ConversationPersistenceError, match="ambiguous") as failure:
        manager.append_execution_report({"turn": 1, "outcome": "fallback"})
    assert "directory fsync failed" in str(failure.value.__cause__)
    assert journal.exists()


def test_journal_fsync_failure_rolls_back_in_memory_transition(tmp_path, monkeypatch):
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        journal_path=journal,
    )
    before = manager.request_messages()
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(ConversationPersistenceError, match="ambiguous") as failure:
        manager.append_execution_report({"turn": 1})
    assert "fsync failed" in str(failure.value.__cause__)
    assert manager.request_messages() != before


def test_ambiguous_journal_durability_poisoned_and_not_retried(tmp_path, monkeypatch):
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        journal_path=journal,
    )
    real_fsync = os.fsync
    calls = {"count": 0}

    def fail_after_write(fd):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("journal fsync uncertain")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_after_write)
    with pytest.raises(ConversationPersistenceError, match="ambiguous"):
        manager.append_execution_report({"turn": 1})
    first = journal.read_bytes()
    with pytest.raises(ConversationPersistenceError, match="poisoned"):
        manager.append_execution_report({"turn": 2})
    assert journal.read_bytes() == first


def test_unterminated_journal_tail_fails_closed(tmp_path):
    journal = tmp_path / "conversation.jsonl"
    journal.write_text('{"seq":0', encoding="utf-8")
    with pytest.raises(ConversationError, match="unterminated"):
        ConversationManager.load_journal(journal)


def test_journal_prewrite_failure_restores_staged_turn(tmp_path, monkeypatch):
    journal = tmp_path / "nested" / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        journal_path=journal,
    )
    manager.begin_turn({"role": "user", "content": "question"})
    manager.stage_assistant("answer")
    original_mkdir = Path.mkdir

    def fail_mkdir(self, *args, **kwargs):
        if self == journal.parent:
            raise OSError("journal mkdir failed")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(OSError, match="journal mkdir failed"):
        manager.commit_turn()
    assert manager.staged
    assert manager._staged_assistant == {"role": "assistant", "content": "answer"}


def test_journal_canonicalization_failure_restores_messages(tmp_path, monkeypatch):
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        journal_path=journal,
    )
    manager.begin_turn({"role": "user", "content": "question"})
    manager.stage_assistant("answer")
    before = manager.request_messages()
    original_canonical = conversation_module.canonical_json

    def fail_payload(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("kind") == "conversation_turn":
            raise TypeError("journal canonicalization failed")
        return original_canonical(value, *args, **kwargs)

    monkeypatch.setattr(conversation_module, "canonical_json", fail_payload)
    with pytest.raises(TypeError, match="journal canonicalization failed"):
        manager.commit_turn()
    assert manager.request_messages() == before
    assert manager.staged


def test_compaction_wal_replay_preserves_epoch_and_cache_generation(tmp_path):
    state = tmp_path / "state.json"
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        state_path=state, journal_path=journal, prompt_cache_key="cache",
        config=ConversationConfig(trigger_tokens=20, tail_tokens=10),
    )
    manager.messages.extend([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
    ])
    manager._persist()
    manager.compact(lambda _: {"version": 1, "summary": "summary",
                               "commitments": [], "uncertainties": [], "facts": []},
                    force=True)
    restored = ConversationManager.load(state)
    assert restored.epoch == manager.epoch
    assert restored.prompt_cache_key == manager.prompt_cache_key
    assert restored.messages == manager.messages


def test_state_restart_replays_wal_after_snapshot_failure(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    journal = tmp_path / "conversation.jsonl"
    manager = ConversationManager(
        system_message={"role": "system", "content": "stable"},
        state_path=state, journal_path=journal,
    )
    manager._persist()

    def fail_snapshot(*args, **kwargs):
        raise OSError("snapshot replace failed")

    monkeypatch.setattr(manager, "_persist", fail_snapshot)
    with pytest.raises(ConversationError, match="WAL committed") as failure:
        manager.append_execution_report({"turn": 1, "outcome": "fallback"})
    assert isinstance(failure.value.__cause__, OSError)
    assert "snapshot replace failed" in str(failure.value.__cause__)

    # The old snapshot still has journal_records_applied=0, while the WAL
    # contains the committed transition.  Restart must apply it exactly once.
    restored = ConversationManager.load(state)
    reports = [m for m in restored.messages
               if isinstance(m.get("content"), str)
               and "ENGINE-ONLY REPORT" in m["content"]]
    assert len(reports) == 1
    restored_again = ConversationManager.load(state)
    assert restored_again.messages == restored.messages
