#!/usr/bin/env python3
# SCRIPT-METADATA:
# name: migrate_fork_checkpoint
# description: Validate and migrate a forked conversation checkpoint into a standalone lineage.
# tags: recovery,wal,freeciv,checkpoint
# language: python
"""Safely migrate a resumed child checkpoint whose WAL contains only a suffix.

The command never edits either source lineage.  It verifies both manifests,
reconstructs the child state from the parent snapshot plus exactly the journal
prefix named by the child snapshot, and writes a new standalone checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from lechonk.model.conversation import ConversationConfig, ConversationError, ConversationManager
from lechonk.core.jsonutil import canonical_json
from lechonk.core.recovery import (
    RecoveryError,
    RecoveryCheckpoint,
    build_checkpoint,
    load_checkpoint,
    sha256_file,
    write_checkpoint,
)


_IDENTITY_FIELDS = (
    "username", "player_id", "config_digest", "config_digest_schema", "prompt_bundle_id",
    "ruleset_reference_schema_version", "ruleset_reference_hash",
    "help_reference_schema_version", "help_reference_hash",
    "help_reference_source_sha256", "classic_reference_schema_version",
    "classic_ruleset_tree_sha256", "classic_reference_present",
    "manual_reference_schema_version", "manual_reference_hash",
    "manual_reference_source_sha256", "manual_reference_present",
)


def _read_state(path: Path, *, require_marker: bool = True) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecoveryError(f"cannot read conversation state {path}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise RecoveryError("unsupported conversation state")
    checksum = data.pop("checksum", None)
    if not isinstance(checksum, str) or hashlib.sha256(
            canonical_json(data).encode("utf-8")).hexdigest() != checksum:
        raise RecoveryError("conversation state checksum mismatch")
    if not isinstance(data.get("messages"), list):
        raise RecoveryError("conversation state messages missing")
    marker = data.get("journal_records_applied")
    if marker is None and not require_marker:
        return data
    if not isinstance(marker, int) or isinstance(marker, bool) or marker < 0:
        raise RecoveryError("child state lacks a valid journal application marker")
    return data


def _identity_matches(parent: Any, child: Any) -> None:
    mismatches = [field for field in _IDENTITY_FIELDS
                  if getattr(parent, field) != getattr(child, field)]
    if mismatches:
        raise RecoveryError("parent/child identity mismatch: " + ", ".join(mismatches))


def _require_regular_source(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"{label} is not a regular non-symlink file")


def _reject_manifest_symlink(manifest_path: Path, field: str) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        bound = Path(payload[field])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RecoveryError(f"invalid {field} in recovery manifest") from exc
    if bound.is_symlink():
        raise RecoveryError(f"{field} is not a regular non-symlink file")


def _copy_verified(source: Path | str, destination: Path,
                   expected_sha256: str, label: str) -> None:
    shutil.copy2(source, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RecoveryError(f"{label} changed while being copied")


def _apply_suffix(parent: ConversationManager,
                  records: list[Mapping[str, Any]]) -> ConversationManager:
    """Apply validated child records to a parent snapshot in memory."""
    parent_messages = parent.messages
    if not parent_messages or parent_messages[0].get("role") != "system":
        raise RecoveryError("parent conversation has no system message")
    synthetic = [{
        "kind": "conversation_base", "seq": 0, "epoch": 0,
        "messages": parent_messages,
    }]
    for offset, record in enumerate(records, 1):
        if record.get("kind") == "conversation_base":
            raise RecoveryError("child WAL contains an unexpected conversation_base")
        if record.get("kind") == "conversation_compaction":
            # A compaction record's summary depends on the exact compaction
            # policy/context that produced it.  The offline migrator cannot
            # prove that context against a parent checkpoint, so it refuses
            # to guess rather than silently changing the model's history.
            raise RecoveryError("child WAL compaction records are unsupported")
        copied = dict(record)
        copied["seq"] = offset
        synthetic.append(copied)
    try:
        rebuilt = ConversationManager.reconstruct(
            parent_messages[0], synthetic, config=ConversationConfig())
    except ConversationError as exc:
        raise RecoveryError(f"child WAL cannot be applied to parent: {exc}") from exc
    # The synthetic base carries the parent's metadata so the suffix records
    # can be checked as a complete state transition, not just as messages.
    rebuilt.epoch = parent.epoch
    rebuilt.prompt_cache_key = parent.prompt_cache_key
    for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                  "observed_request_message_count", "observed_completion_tokens"):
        setattr(rebuilt, "_" + field, getattr(parent, "_" + field))
    for record in records:
        rebuilt.epoch = int(record.get("epoch", rebuilt.epoch))
        if "prompt_cache_key" in record:
            rebuilt.prompt_cache_key = record["prompt_cache_key"]
        for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                      "observed_request_message_count", "observed_completion_tokens"):
            if field in record:
                setattr(rebuilt, "_" + field, record[field])
    return rebuilt


def migrate(*, parent_manifest: Path, child_manifest: Path,
            child_journal: Path, output_dir: Path,
            run_id: str | None = None) -> Path:
    """Perform a fail-closed migration and return the new manifest path."""
    if output_dir.exists() and (
            output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir())):
        raise RecoveryError("output directory must be a new empty directory")
    if output_dir.is_symlink():
        raise RecoveryError("output directory must be a new empty directory")
    parent_root = parent_manifest.parent.resolve(strict=True)
    child_root = child_manifest.parent.resolve(strict=True)
    # Never follow a journal symlink, even when its target is inside the
    # child root.  This keeps the source lineage boundary explicit.
    if child_journal.is_symlink():
        raise RecoveryError("child journal is not a regular non-symlink file")
    journal_resolved = child_journal.resolve(strict=True)
    try:
        journal_resolved.relative_to(child_root)
    except ValueError as exc:
        raise RecoveryError("child journal is outside child artifact root") from exc
    if not journal_resolved.is_file() or journal_resolved.is_symlink():
        raise RecoveryError("child journal is not a regular non-symlink file")
    _reject_manifest_symlink(parent_manifest, "checkpoint_path")
    _reject_manifest_symlink(child_manifest, "checkpoint_path")
    _reject_manifest_symlink(parent_manifest, "conversation_state_path")
    _reject_manifest_symlink(child_manifest, "conversation_state_path")
    parent = load_checkpoint(parent_manifest, artifact_root=parent_root, verify_files=True)
    child = load_checkpoint(child_manifest, artifact_root=child_root, verify_files=True)
    _require_regular_source(Path(parent.checkpoint_path), "parent checkpoint")
    _require_regular_source(Path(child.checkpoint_path), "child checkpoint")
    if child.parent_run_id != parent.run_id:
        raise RecoveryError("child parent_run_id does not match parent run_id")
    _identity_matches(parent, child)
    parent_state = _read_state(Path(parent.conversation_state_path), require_marker=False)
    if parent_state.get("journal_records_applied") not in (None, 0):
        raise RecoveryError("parent checkpoint is not a standalone marker-zero state")
    child_state = _read_state(Path(child.conversation_state_path))
    try:
        records = ConversationManager.load_journal(child_journal)
    except ConversationError as exc:
        raise RecoveryError(f"invalid child conversation journal: {exc}") from exc
    marker = child_state["journal_records_applied"]
    if marker > len(records):
        raise RecoveryError("child state journal marker exceeds journal length")
    selected = records[:marker]
    # ConversationManager.load() probes a sibling conversation.jsonl.  The
    # parent checkpoint may have later WAL records beside it, so load only a
    # private copy of the bound snapshot and deliberately omit that sibling.
    with tempfile.TemporaryDirectory(prefix="migrate-parent-") as isolated:
        isolated_state = Path(isolated) / "conversation-state.json"
        _copy_verified(parent.conversation_state_path, isolated_state,
                       parent.conversation_state_sha256, "parent conversation state")
        parent_manager = ConversationManager.load(isolated_state)
    rebuilt = _apply_suffix(parent_manager, selected)
    if rebuilt.messages != child_state["messages"]:
        raise RecoveryError("parent plus child WAL prefix does not match child snapshot")
    for field in ("epoch", "prompt_cache_key", "observed_prompt_tokens",
                  "observed_estimated_tokens", "observed_request_message_count",
                  "observed_completion_tokens"):
        expected = child_state.get(field)
        actual = rebuilt.epoch if field == "epoch" else (
            rebuilt.prompt_cache_key if field == "prompt_cache_key" else
            getattr(rebuilt, "_" + field))
        if expected != actual:
            raise RecoveryError(f"child snapshot metadata mismatch: {field}")
    expected_messages = rebuilt.messages
    migrated_manager = ConversationManager.reconstruct(
        expected_messages[0], [{"kind": "conversation_base", "seq": 0,
                                "epoch": child_state.get("epoch", 0),
                                "messages": expected_messages}],
        config=ConversationConfig())
    for field in ("epoch", "prompt_cache_key", "observed_prompt_tokens",
                  "observed_estimated_tokens", "observed_request_message_count",
                  "observed_completion_tokens"):
        if field in child_state:
            value = child_state[field]
            if field == "epoch":
                migrated_manager.epoch = value
            elif field == "prompt_cache_key":
                migrated_manager.prompt_cache_key = value
            else:
                setattr(migrated_manager, "_" + field, value)
    new_run_id = run_id or f"migrated-{child.run_id}"
    if new_run_id in {parent.run_id, child.run_id}:
        raise RecoveryError("new run_id must be distinct from source lineages")
    # Stage all files in a private sibling directory, then publish the whole
    # run directory in one rename.  A failed copy/write cannot leave a partial
    # destination behind.
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise RecoveryError("output directory must be a new empty directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.migration-",
                                     dir=str(output_parent)))
    published = False
    try:
        checkpoint_dir = staging / "checkpoints"
        checkpoint_dir.mkdir()
        save_name = Path(child.checkpoint_path).name
        output_save = checkpoint_dir / save_name
        _copy_verified(child.checkpoint_path, output_save,
                       child.checkpoint_sha256, "child checkpoint")
        output_state = checkpoint_dir / f"conversation-state-turn-{child.turn}.json"
        migrated_manager.export_standalone(output_state)
        manifest = build_checkpoint(
            checkpoint_path=output_save, artifact_root=staging,
            conversation_state_path=output_state, turn=child.turn,
            username=child.username, run_id=new_run_id, parent_run_id=child.run_id,
            player_id=child.player_id, config_digest=child.config_digest,
            strategy_memory=child.strategy_memory,
            checkpoint_sha256=sha256_file(output_save),
            ruleset_reference_schema_version=child.ruleset_reference_schema_version,
            ruleset_reference_hash=child.ruleset_reference_hash,
            help_reference_schema_version=child.help_reference_schema_version,
            help_reference_hash=child.help_reference_hash,
            help_reference_source_sha256=child.help_reference_source_sha256,
            classic_reference_schema_version=child.classic_reference_schema_version,
            classic_ruleset_tree_sha256=child.classic_ruleset_tree_sha256,
            classic_reference_present=child.classic_reference_present,
            manual_reference_schema_version=child.manual_reference_schema_version,
            manual_reference_hash=child.manual_reference_hash,
            manual_reference_source_sha256=child.manual_reference_source_sha256,
            manual_reference_present=child.manual_reference_present,
            config_digest_schema=child.config_digest_schema,
            prompt_bundle_id=child.prompt_bundle_id)
        if output_dir.exists() or output_dir.is_symlink():
            raise RecoveryError("output directory appeared during migration")
        os.replace(staging, output_dir)
        published = True
        try:
            parent_fd = os.open(output_parent, os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            raise RecoveryError("cannot durably publish migrated run directory") from exc
        # Manifest paths are absolute, so rewrite them for the published
        # location after the atomic directory rename.
        final_manifest = RecoveryCheckpoint(
            **{**manifest.to_dict(),
               "checkpoint_path": str(output_dir / "checkpoints" / save_name),
               "conversation_state_path": str(output_dir / "checkpoints" /
                                                output_state.name)})
        manifest_path = output_dir / "recovery.json"
        write_checkpoint(manifest_path, final_manifest)
        return manifest_path
    except Exception:
        if published and output_dir.exists() and output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--child-manifest", type=Path, required=True)
    parser.add_argument("--child-journal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = migrate(parent_manifest=args.parent_manifest,
                           child_manifest=args.child_manifest,
                           child_journal=args.child_journal,
                           output_dir=args.output_dir, run_id=args.run_id)
    except (RecoveryError, ConversationError, OSError, ValueError) as exc:
        print(f"migration rejected: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
