# SPDX-License-Identifier: MIT
"""Durable, fail-closed metadata for a coordinated game checkpoint.

This module does not save or load Freeciv itself.  It binds an already verified
save export to the conversation state and committed strategy memory that must
be restored with it.  Paths are confined to the run artifact directory and
the manifest is written atomically so callers can safely use it as a resume
input after a process interruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lechonk.core.jsonutil import canonical_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FREECIV_USERNAME = re.compile(r"^[a-z]{3,31}$")
_VERSION = 1


class RecoveryError(RuntimeError):
    """Checkpoint metadata is missing, invalid, or does not match state."""


def validate_freeciv_username(value: Any) -> str:
    """Validate the restricted username form used for save-file isolation.

    Freeciv's save naming convention is vendor-parsed, so usernames are kept
    deliberately narrower than the protocol's general player-name surface:
    lowercase ASCII letters only, with the server's 3--31 character bounds.
    ``myagent`` remains valid for old manifests and lower-level callers.
    """
    if not isinstance(value, str) or not _FREECIV_USERNAME.fullmatch(value):
        raise RecoveryError(
            "invalid username: expected 3-31 lowercase ASCII letters")
    return value


def derive_freeciv_username(run_id: str) -> str:
    """Derive a deterministic, collision-resistant fresh-run username.

    The complete SHA-256 digest is interpreted as an integer, then reduced
    into a fixed-width 20-digit base-26 suffix.  Keeping the ``agent`` prefix
    makes generated names recognizable while remaining valid for Freeciv's
    save parser and old ``myagent`` manifests.
    """
    if not isinstance(run_id, str) or not run_id:
        raise RecoveryError("invalid run_id for username derivation")
    number = int.from_bytes(hashlib.sha256(run_id.encode("utf-8")).digest(),
                            "big") % (26 ** 20)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    suffix = ["a"] * 20
    for index in range(19, -1, -1):
        number, remainder = divmod(number, 26)
        suffix[index] = alphabet[remainder]
    return validate_freeciv_username("agent" + "".join(suffix))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise RecoveryError(f"cannot read checkpoint file {path}") from exc
    return digest.hexdigest()


def _inside(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RecoveryError(f"{label} is outside the artifact root") from exc
    if not resolved.is_file():
        raise RecoveryError(f"{label} is not a regular file")
    return resolved


def _text(value: Any, *, field: str, max_len: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise RecoveryError(f"invalid {field}")
    return value


@dataclass(frozen=True)
class RecoveryCheckpoint:
    """Validated binding between a save, model state, and committed memory."""

    checkpoint_path: str
    checkpoint_sha256: str
    conversation_state_path: str
    conversation_state_sha256: str
    turn: int
    player_id: int
    username: str
    run_id: str
    parent_run_id: str | None
    config_digest: str
    strategy_memory: dict[str, Any]
    version: int = _VERSION
    checkpoint_kind: str = "start_of_turn"
    # Optional model-knowledge provenance.  These fields were added after
    # the original manifest format; keeping them optional lets old manifests
    # remain readable, while load_checkpoint can fail closed when a caller
    # supplies an expected provenance value.
    ruleset_reference_schema_version: str | None = None
    ruleset_reference_hash: str | None = None
    help_reference_schema_version: str | None = None
    help_reference_hash: str | None = None
    help_reference_source_sha256: str | None = None
    # Optional static Freeciv classic tree provenance.  ``present`` is kept
    # separately from the digest so a deliberately unavailable encyclopedia
    # cannot be confused with an unrecorded/legacy manifest.
    classic_reference_schema_version: str | None = None
    classic_ruleset_tree_sha256: str | None = None
    classic_reference_present: bool | None = None
    # Optional generated Freeciv manual provenance.  The manual is an
    # auxiliary knowledge artifact, so an unavailable manual is represented
    # by ``present=False`` and no fabricated digest.
    manual_reference_schema_version: str | None = None
    manual_reference_hash: str | None = None
    manual_reference_source_sha256: str | None = None
    manual_reference_present: bool | None = None
    # Explicit identity metadata for versioned fresh-run lineages. Absent on
    # legacy manifests so they remain readable and fail closed at resolution.
    config_digest_schema: str | None = None
    prompt_bundle_id: str | None = None

    def __post_init__(self) -> None:
        # ``build_checkpoint`` and ``load_checkpoint`` validate at their
        # boundaries, but keep direct dataclass construction fail-closed too.
        validate_freeciv_username(self.username)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "conversation_state_path": self.conversation_state_path,
            "conversation_state_sha256": self.conversation_state_sha256,
            "turn": self.turn,
            "player_id": self.player_id,
            "username": self.username,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "config_digest": self.config_digest,
            "strategy_memory": self.strategy_memory,
            "ruleset_reference_schema_version": self.ruleset_reference_schema_version,
            "ruleset_reference_hash": self.ruleset_reference_hash,
            "help_reference_schema_version": self.help_reference_schema_version,
            "help_reference_hash": self.help_reference_hash,
            "help_reference_source_sha256": self.help_reference_source_sha256,
            "classic_reference_schema_version": self.classic_reference_schema_version,
            "classic_ruleset_tree_sha256": self.classic_ruleset_tree_sha256,
            "classic_reference_present": self.classic_reference_present,
            "manual_reference_schema_version": self.manual_reference_schema_version,
            "manual_reference_hash": self.manual_reference_hash,
            "manual_reference_source_sha256": self.manual_reference_source_sha256,
            "manual_reference_present": self.manual_reference_present,
            "config_digest_schema": self.config_digest_schema,
            "prompt_bundle_id": self.prompt_bundle_id,
        }

    def with_checksum(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["manifest_sha256"] = hashlib.sha256(
            canonical_json(payload).encode("utf-8")).hexdigest()
        return payload


def build_checkpoint(*, checkpoint_path: str | Path,
                     artifact_root: str | Path,
                     conversation_state_path: str | Path,
                     turn: int, username: str, run_id: str,
                     player_id: int,
                     config_digest: str,
                     strategy_memory: Mapping[str, Any],
                     parent_run_id: str | None = None,
                     checkpoint_sha256: str | None = None,
                     ruleset_reference_schema_version: str | None = None,
                     ruleset_reference_hash: str | None = None,
                     help_reference_schema_version: str | None = None,
                     help_reference_hash: str | None = None,
                     help_reference_source_sha256: str | None = None,
                     classic_reference_schema_version: str | None = None,
                     classic_ruleset_tree_sha256: str | None = None,
                     classic_reference_present: bool | None = None,
                     manual_reference_schema_version: str | None = None,
                     manual_reference_hash: str | None = None,
                     manual_reference_source_sha256: str | None = None,
                     manual_reference_present: bool | None = None,
                     config_digest_schema: str | None = None,
                     prompt_bundle_id: str | None = None) -> RecoveryCheckpoint:
    """Build a checkpoint from existing verified files and committed state."""
    root = Path(artifact_root).resolve(strict=True)
    save = _inside(Path(checkpoint_path), root, label="checkpoint path")
    conversation = _inside(Path(conversation_state_path), root,
                           label="conversation state path")
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
        raise RecoveryError("invalid checkpoint turn")
    if not isinstance(player_id, int) or isinstance(player_id, bool) or player_id < 0:
        raise RecoveryError("invalid checkpoint player_id")
    username = validate_freeciv_username(username)
    run_id = _text(run_id, field="run_id", max_len=256)
    config_digest = _text(config_digest, field="config_digest", max_len=256)
    if parent_run_id is not None:
        parent_run_id = _text(parent_run_id, field="parent_run_id", max_len=256)
    for value, field in ((config_digest_schema, "config_digest_schema"),
                         (prompt_bundle_id, "prompt_bundle_id")):
        if value is not None:
            _text(value, field=field, max_len=128)
    if not isinstance(strategy_memory, Mapping):
        raise RecoveryError("invalid strategy memory")
    if classic_reference_present is not None and not isinstance(
            classic_reference_present, bool):
        raise RecoveryError("invalid classic_reference_present")
    if manual_reference_present is not None and not isinstance(
            manual_reference_present, bool):
        raise RecoveryError("invalid manual_reference_present")
    for value, field in (
        (ruleset_reference_schema_version, "ruleset_reference_schema_version"),
        (help_reference_schema_version, "help_reference_schema_version"),
        (classic_reference_schema_version, "classic_reference_schema_version"),
        (manual_reference_schema_version, "manual_reference_schema_version"),
    ):
        if value is not None:
            _text(value, field=field, max_len=128)
    for value, field in (
        (ruleset_reference_hash, "ruleset_reference_hash"),
        (help_reference_hash, "help_reference_hash"),
        (help_reference_source_sha256, "help_reference_source_sha256"),
        (classic_ruleset_tree_sha256, "classic_ruleset_tree_sha256"),
        (manual_reference_hash, "manual_reference_hash"),
        (manual_reference_source_sha256, "manual_reference_source_sha256"),
    ):
        if value is not None and (not isinstance(value, str) or
                                  not _SHA256.fullmatch(value)):
            raise RecoveryError(f"invalid {field}")
    actual_save_sha = sha256_file(save)
    if checkpoint_sha256 is not None and checkpoint_sha256 != actual_save_sha:
        raise RecoveryError("checkpoint checksum mismatch")
    return RecoveryCheckpoint(
        checkpoint_path=str(save), checkpoint_sha256=actual_save_sha,
        conversation_state_path=str(conversation),
        conversation_state_sha256=sha256_file(conversation), turn=turn,
        player_id=player_id,
        username=username, run_id=run_id, parent_run_id=parent_run_id,
        config_digest=config_digest, strategy_memory=dict(strategy_memory),
        ruleset_reference_schema_version=ruleset_reference_schema_version,
        ruleset_reference_hash=ruleset_reference_hash,
        help_reference_schema_version=help_reference_schema_version,
        help_reference_hash=help_reference_hash,
        help_reference_source_sha256=help_reference_source_sha256,
        classic_reference_schema_version=classic_reference_schema_version,
        classic_ruleset_tree_sha256=classic_ruleset_tree_sha256,
        classic_reference_present=classic_reference_present,
        manual_reference_schema_version=manual_reference_schema_version,
        manual_reference_hash=manual_reference_hash,
        manual_reference_source_sha256=manual_reference_source_sha256,
        manual_reference_present=manual_reference_present,
        config_digest_schema=config_digest_schema,
        prompt_bundle_id=prompt_bundle_id)


def write_checkpoint(path: str | Path, checkpoint: RecoveryCheckpoint) -> str:
    """Atomically write a checksummed manifest and fsync its directory."""
    destination = Path(path)
    payload = checkpoint.with_checksum()
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = destination.parent.resolve(strict=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.",
                                     dir=str(destination.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            dir_fd = os.open(root, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            # The directory fsync is part of the durability contract: after
            # os.replace() the manifest may be visible, but without syncing
            # its parent directory a crash can lose the directory entry.
            # Report this as a failed checkpoint instead of claiming that the
            # manifest is durable.
            raise RecoveryError(
                f"cannot durably publish recovery manifest {destination}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return str(destination)


def load_checkpoint(path: str | Path, *, artifact_root: str | Path,
                    expected_turn: int | None = None,
                    expected_username: str | None = None,
                    expected_run_id: str | None = None,
                    expected_config_digest: str | None = None,
                    expected_player_id: int | None = None,
                    expected_ruleset_reference_schema_version: str | None = None,
                    expected_ruleset_reference_hash: str | None = None,
                    expected_help_reference_schema_version: str | None = None,
                    expected_help_reference_hash: str | None = None,
                    expected_help_reference_source_sha256: str | None = None,
                    expected_classic_reference_schema_version: str | None = None,
                    expected_classic_ruleset_tree_sha256: str | None = None,
                    expected_classic_reference_present: bool | None = None,
                    expected_manual_reference_schema_version: str | None = None,
                    expected_manual_reference_hash: str | None = None,
                    expected_manual_reference_source_sha256: str | None = None,
                    expected_manual_reference_present: bool | None = None,
                    expected_config_digest_schema: str | None = None,
                    expected_prompt_bundle_id: str | None = None,
                    verify_files: bool = True) -> RecoveryCheckpoint:
    """Load and verify metadata, optionally rehashing bound state files."""
    for value, field in (
        (expected_ruleset_reference_hash, "expected_ruleset_reference_hash"),
        (expected_help_reference_hash, "expected_help_reference_hash"),
        (expected_help_reference_source_sha256,
         "expected_help_reference_source_sha256"),
        (expected_classic_ruleset_tree_sha256,
         "expected_classic_ruleset_tree_sha256"),
        (expected_manual_reference_hash, "expected_manual_reference_hash"),
        (expected_manual_reference_source_sha256,
         "expected_manual_reference_source_sha256"),
    ):
        if value is not None and (not isinstance(value, str) or
                                  not _SHA256.fullmatch(value)):
            raise RecoveryError(f"invalid {field}")
    for value, field in (
        (expected_ruleset_reference_schema_version,
         "expected_ruleset_reference_schema_version"),
        (expected_help_reference_schema_version,
         "expected_help_reference_schema_version"),
        (expected_classic_reference_schema_version,
         "expected_classic_reference_schema_version"),
        (expected_manual_reference_schema_version,
         "expected_manual_reference_schema_version"),
    ):
        if value is not None:
            _text(value, field=field, max_len=128)
    for value, field in ((expected_config_digest_schema, "expected_config_digest_schema"),
                         (expected_prompt_bundle_id, "expected_prompt_bundle_id")):
        if value is not None:
            _text(value, field=field, max_len=128)
    manifest = Path(path)
    root = Path(artifact_root).resolve(strict=True)
    try:
        manifest.resolve(strict=True).relative_to(root)
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecoveryError("cannot read recovery manifest") from exc
    if not isinstance(data, dict) or data.get("version") != _VERSION:
        raise RecoveryError("unsupported recovery manifest version")
    if "player_id" not in data or data.get("player_id") is None:
        raise RecoveryError(
            "recovery manifest lacks player_id; regenerate the checkpoint "
            "before attempting resume")
    checksum = data.pop("manifest_sha256", None)
    if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
        raise RecoveryError("missing or invalid recovery manifest checksum")
    if hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest() != checksum:
        raise RecoveryError("recovery manifest checksum mismatch")
    try:
        checkpoint = RecoveryCheckpoint(**data)
    except (TypeError, ValueError) as exc:
        raise RecoveryError("invalid recovery manifest fields") from exc
    for value, field in ((checkpoint.config_digest_schema, "config_digest_schema"),
                         (checkpoint.prompt_bundle_id, "prompt_bundle_id")):
        if value is not None:
            _text(value, field=field, max_len=128)
    if checkpoint.checkpoint_kind != "start_of_turn":
        raise RecoveryError("unsupported checkpoint kind")
    if (not isinstance(checkpoint.player_id, int) or
            isinstance(checkpoint.player_id, bool) or checkpoint.player_id < 0):
        raise RecoveryError("invalid checkpoint player_id")
    validate_freeciv_username(checkpoint.username)
    if (checkpoint.classic_reference_present is not None and
            not isinstance(checkpoint.classic_reference_present, bool)):
        raise RecoveryError("invalid classic_reference_present")
    if (checkpoint.manual_reference_present is not None and
            not isinstance(checkpoint.manual_reference_present, bool)):
        raise RecoveryError("invalid manual_reference_present")
    if not _SHA256.fullmatch(checkpoint.checkpoint_sha256) or not _SHA256.fullmatch(
            checkpoint.conversation_state_sha256):
        raise RecoveryError("invalid bound file checksum")
    for value, field in (
        (checkpoint.ruleset_reference_hash, "ruleset_reference_hash"),
        (checkpoint.help_reference_hash, "help_reference_hash"),
        (checkpoint.help_reference_source_sha256, "help_reference_source_sha256"),
        (checkpoint.classic_ruleset_tree_sha256,
         "classic_ruleset_tree_sha256"),
        (checkpoint.manual_reference_hash, "manual_reference_hash"),
        (checkpoint.manual_reference_source_sha256,
         "manual_reference_source_sha256"),
    ):
        if value is not None and (not isinstance(value, str) or
                                  not _SHA256.fullmatch(value)):
            raise RecoveryError(f"invalid {field}")
    for value, field in (
        (checkpoint.ruleset_reference_schema_version, "ruleset_reference_schema_version"),
        (checkpoint.help_reference_schema_version, "help_reference_schema_version"),
        (checkpoint.classic_reference_schema_version,
         "classic_reference_schema_version"),
        (checkpoint.manual_reference_schema_version,
         "manual_reference_schema_version"),
    ):
        if value is not None:
            _text(value, field=field, max_len=128)
    if expected_turn is not None and checkpoint.turn != expected_turn:
        raise RecoveryError("checkpoint turn mismatch")
    if (expected_player_id is not None and
            checkpoint.player_id != expected_player_id):
        raise RecoveryError("checkpoint player mismatch")
    for expected, actual, label in (
        (expected_username, checkpoint.username, "username"),
        (expected_run_id, checkpoint.run_id, "run_id"),
        (expected_config_digest, checkpoint.config_digest, "config_digest"),
        (expected_ruleset_reference_schema_version,
         checkpoint.ruleset_reference_schema_version,
         "ruleset_reference_schema_version"),
        (expected_ruleset_reference_hash, checkpoint.ruleset_reference_hash,
         "ruleset_reference_hash"),
        (expected_help_reference_schema_version,
         checkpoint.help_reference_schema_version,
         "help_reference_schema_version"),
        (expected_help_reference_hash, checkpoint.help_reference_hash,
         "help_reference_hash"),
        (expected_help_reference_source_sha256,
         checkpoint.help_reference_source_sha256,
         "help_reference_source_sha256"),
        (expected_classic_reference_schema_version,
         checkpoint.classic_reference_schema_version,
         "classic_reference_schema_version"),
        (expected_classic_ruleset_tree_sha256,
         checkpoint.classic_ruleset_tree_sha256,
         "classic_ruleset_tree_sha256"),
        (expected_manual_reference_schema_version,
         checkpoint.manual_reference_schema_version,
         "manual_reference_schema_version"),
        (expected_manual_reference_hash, checkpoint.manual_reference_hash,
         "manual_reference_hash"),
        (expected_manual_reference_source_sha256,
         checkpoint.manual_reference_source_sha256,
         "manual_reference_source_sha256"),
        (expected_config_digest_schema, checkpoint.config_digest_schema,
         "config_digest_schema"),
        (expected_prompt_bundle_id, checkpoint.prompt_bundle_id,
         "prompt_bundle_id"),
    ):
        if expected is not None and actual is None:
            raise RecoveryError(f"checkpoint {label} is missing")
        if expected is not None and expected != actual:
            raise RecoveryError(f"checkpoint {label} mismatch")
    if (expected_classic_reference_present is not None and
            checkpoint.classic_reference_present is None):
        raise RecoveryError("checkpoint classic_reference_present is missing")
    if (expected_classic_reference_present is not None and
            checkpoint.classic_reference_present !=
            expected_classic_reference_present):
        raise RecoveryError("checkpoint classic_reference_present mismatch")
    if (expected_manual_reference_present is not None and
            checkpoint.manual_reference_present is None):
        raise RecoveryError("checkpoint manual_reference_present is missing")
    if (expected_manual_reference_present is not None and
            checkpoint.manual_reference_present !=
            expected_manual_reference_present):
        raise RecoveryError("checkpoint manual_reference_present mismatch")
    if verify_files:
        save = _inside(Path(checkpoint.checkpoint_path), root,
                       label="checkpoint path")
        conversation = _inside(Path(checkpoint.conversation_state_path), root,
                               label="conversation state path")
        if sha256_file(save) != checkpoint.checkpoint_sha256:
            raise RecoveryError("checkpoint file checksum mismatch")
        if sha256_file(conversation) != checkpoint.conversation_state_sha256:
            raise RecoveryError("conversation state checksum mismatch")
    return checkpoint
