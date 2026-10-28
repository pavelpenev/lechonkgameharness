# SPDX-License-Identifier: MIT
"""Offline tests for the forked-WAL checkpoint migration utility."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lechonk.model.conversation import ConversationManager
from lechonk.core.jsonutil import canonical_json
from lechonk.core.recovery import RecoveryError, build_checkpoint, write_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "migrate_fork_checkpoint", ROOT / "scripts" / "migrate_fork_checkpoint.py")
MIGRATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MIGRATE)


def _state_payload(messages, *, marker: int) -> dict:
    payload = {
        "version": 1, "policy_version": "conversation_v1", "epoch": 0,
        "messages": messages, "prompt_cache_key": None,
        "journal_records_applied": marker,
        "observed_prompt_tokens": 0, "observed_estimated_tokens": 0,
        "observed_request_message_count": None,
        "observed_completion_tokens": None,
    }
    payload["checksum"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def _record(messages) -> dict:
    record = {"kind": "conversation_turn", "seq": 0, "epoch": 0,
              "messages": messages}
    record["checksum"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")).hexdigest()
    return record


def _fixture(tmp_path: Path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir(parents=True); child.mkdir(parents=True)
    parent_save = parent / "save.sav.xz"
    parent_save.write_bytes(b"parent-save")
    parent_state = parent / "conversation-state.json"
    parent_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "parent"},
        {"role": "assistant", "content": "ack"},
    ]
    ConversationManager(system_message=parent_messages[0]).export_standalone(parent_state)
    # Replace the simple manager's one-message state with the intended valid
    # parent history and recompute its checksum through the same format.
    parent_state.write_text(canonical_json(_state_payload(parent_messages, marker=0)) + "\n")
    parent_manifest = build_checkpoint(
        checkpoint_path=parent_save, artifact_root=parent,
        conversation_state_path=parent_state, turn=1, username="player",
        run_id="parent", player_id=1, config_digest="cfg",
        strategy_memory={})
    parent_recovery = parent / "recovery.json"
    write_checkpoint(parent_recovery, parent_manifest)

    child_save = child / "save.sav.xz"
    child_save.write_bytes(b"child-save")
    suffix = [{"role": "user", "content": "child"},
              {"role": "assistant", "content": "done"}]
    full_messages = parent_messages + suffix
    child_state = child / "conversation-state.json"
    child_state.write_text(canonical_json(_state_payload(full_messages, marker=1)) + "\n")
    child_journal = child / "conversation.jsonl"
    child_journal.write_text(canonical_json(_record(suffix)) + "\n")
    child_manifest = build_checkpoint(
        checkpoint_path=child_save, artifact_root=child,
        conversation_state_path=child_state, turn=2, username="player",
        run_id="child", parent_run_id="parent", player_id=1,
        config_digest="cfg", strategy_memory={})
    child_recovery = child / "recovery.json"
    write_checkpoint(child_recovery, child_manifest)
    return parent_recovery, child_recovery, child_journal


def test_migrates_parent_plus_child_prefix_and_reloads(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    output = tmp_path / "migrated"
    manifest = MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                               child_journal=journal, output_dir=output,
                               run_id="migrated")
    loaded = ConversationManager.load(
        output / "checkpoints" / "conversation-state-turn-2.json")
    assert [m["content"] for m in loaded.messages] == ["system", "parent", "ack", "child", "done"]
    assert json.loads(manifest.read_text())["parent_run_id"] == "child"


def test_rejects_tampered_journal_and_preserves_output(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    journal.write_text(journal.read_text(encoding="utf-8").replace('"child"', '"tampered"'), encoding="utf-8")
    output = tmp_path / "migrated"
    with pytest.raises(RecoveryError):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=output)
    assert not output.exists()


def test_rejects_existing_output_directory(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    output = tmp_path / "migrated"
    output.mkdir()
    (output / "keep").write_bytes(b"keep")
    with pytest.raises(RecoveryError):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=output)
    assert (output / "keep").read_bytes() == b"keep"


def _rewrite_manifest(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json({k: v for k, v in payload.items()
                        if k != "manifest_sha256"}).encode("utf-8")).hexdigest()
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def test_rejects_wrong_parent_and_partial_marker(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    _rewrite_manifest(child, parent_run_id="other-parent")
    with pytest.raises(RecoveryError, match="parent_run_id"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-parent")

    parent, child, journal = _fixture(tmp_path / "partial")
    state_path = Path(json.loads(child.read_text())["conversation_state_path"])
    state = json.loads(state_path.read_text())
    state["journal_records_applied"] = 2
    state.pop("checksum", None)
    state["checksum"] = hashlib.sha256(canonical_json(state).encode()).hexdigest()
    state_path.write_text(canonical_json(state) + "\n", encoding="utf-8")
    _rewrite_manifest(child, conversation_state_sha256=hashlib.sha256(
        state_path.read_bytes()).hexdigest())
    with pytest.raises(RecoveryError, match="marker"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-marker")


def test_rejects_identity_mismatch_and_journal_symlink(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    _rewrite_manifest(child, config_digest="different")
    with pytest.raises(RecoveryError, match="identity mismatch"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-identity")

    parent, child, journal = _fixture(tmp_path / "symlink")
    target = journal.with_name("real-journal.jsonl")
    journal.rename(target)
    journal.symlink_to(target.name)
    with pytest.raises(RecoveryError, match="non-symlink"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-link")


def test_rejects_compaction_prefix(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    record = {"kind": "conversation_compaction", "seq": 0, "epoch": 1,
              "messages": [{"role": "user", "content": "summary"}]}
    record["checksum"] = hashlib.sha256(
        canonical_json({k: v for k, v in record.items() if k != "checksum"})
        .encode("utf-8")).hexdigest()
    journal.write_text(canonical_json(record) + "\n", encoding="utf-8")
    with pytest.raises(RecoveryError, match="compaction"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-compaction")


def test_ignores_parent_sibling_journal(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    parent_state = Path(json.loads(parent.read_text())['conversation_state_path'])
    later = {"kind": "conversation_turn", "seq": 0, "epoch": 0,
             "messages": [{"role": "assistant", "content": "must-not-leak"}]}
    later["checksum"] = hashlib.sha256(
        canonical_json(later).encode("utf-8")).hexdigest()
    parent_state.with_name("conversation.jsonl").write_text(
        canonical_json(later) + "\n")
    output = tmp_path / "isolated-parent"
    MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                    child_journal=journal, output_dir=output)
    loaded = ConversationManager.load(
        output / "checkpoints" / "conversation-state-turn-2.json")
    assert all(message.get("content") != "must-not-leak" for message in loaded.messages)


def test_accepts_legacy_parent_without_marker(tmp_path: Path) -> None:
    parent, child, journal = _fixture(tmp_path)
    parent_payload = json.loads(parent.read_text())
    parent_state = Path(parent_payload["conversation_state_path"])
    state = json.loads(parent_state.read_text())
    state.pop("journal_records_applied")
    state.pop("checksum")
    state["checksum"] = hashlib.sha256(canonical_json(state).encode()).hexdigest()
    parent_state.write_text(canonical_json(state) + "\n")
    _rewrite_manifest(parent, conversation_state_sha256=hashlib.sha256(
        parent_state.read_bytes()).hexdigest())
    output = tmp_path / "legacy-migrated"
    MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                    child_journal=journal, output_dir=output)
    assert ConversationManager.load(
        output / "checkpoints" / "conversation-state-turn-2.json").messages[-1]["content"] == "done"


def test_rejects_symlink_bound_save_and_cleans_failed_stage(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child, journal = _fixture(tmp_path)
    child_payload = json.loads(child.read_text())
    real_save = Path(child_payload["checkpoint_path"])
    link = real_save.with_name("save-link.sav.xz")
    real_save.rename(link)
    real_save.symlink_to(link.name)
    _rewrite_manifest(child, checkpoint_path=str(real_save))
    with pytest.raises(RecoveryError, match="non-symlink"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "bad-save-link")

    parent, child, journal = _fixture(tmp_path / "copy-failure")
    output = tmp_path / "copy-failure" / "migrated"
    original_copy = MIGRATE.shutil.copy2

    def fail_copy(*args, **kwargs):
        raise OSError("injected copy failure")

    monkeypatch.setattr(MIGRATE.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=output)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.migration-*"))
    monkeypatch.setattr(MIGRATE.shutil, "copy2", original_copy)


def test_fails_closed_when_verified_source_changes_during_copy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child, journal = _fixture(tmp_path)
    original_copy = MIGRATE.shutil.copy2

    def mutate_copy(source, destination, *args, **kwargs):
        result = original_copy(source, destination, *args, **kwargs)
        if Path(destination).name == "conversation-state.json":
            Path(destination).write_bytes(b"changed-after-copy")
        return result

    monkeypatch.setattr(MIGRATE.shutil, "copy2", mutate_copy)
    with pytest.raises(RecoveryError, match="changed while being copied"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "changed")
    assert not (tmp_path / "changed").exists()

    parent, child, journal = _fixture(tmp_path / "save-change")

    def mutate_save_copy(source, destination, *args, **kwargs):
        result = original_copy(source, destination, *args, **kwargs)
        if Path(destination).suffixes[-2:] == [".sav", ".xz"]:
            Path(destination).write_bytes(b"changed-save-after-copy")
        return result

    monkeypatch.setattr(MIGRATE.shutil, "copy2", mutate_save_copy)
    with pytest.raises(RecoveryError, match="changed while being copied"):
        MIGRATE.migrate(parent_manifest=parent, child_manifest=child,
                        child_journal=journal, output_dir=tmp_path / "save-changed")
    assert not (tmp_path / "save-changed").exists()
