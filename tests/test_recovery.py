# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lechonk.adapter.manual_reference import PINNED_IMAGE_REF, load_manual_reference

from lechonk.core.recovery import (
    RecoveryError,
    build_checkpoint,
    derive_freeciv_username,
    load_checkpoint,
    sha256_file,
    validate_freeciv_username,
    write_checkpoint,
)
from lechonk.core.ledger import TranscriptLedger
from lechonk.core.runner import GameRunner, RunnerConfig
from lechonk.core.schemas import RunManifest


def test_freeciv_username_derivation_is_pinned_and_distinct():
    assert derive_freeciv_username("run-alpha") == (
        "agentrdkypfwlojxzadrvnwkq")
    first = derive_freeciv_username("lineage-a")
    second = derive_freeciv_username("lineage-b")
    assert first != second
    assert first.startswith("agent") and len(first) == 25
    assert first.isascii() and first.islower() and first.isalpha()


@pytest.mark.parametrize("value", ["myagent", "player", "a" * 31])
def test_freeciv_username_accepts_legacy_and_valid_values(value):
    assert validate_freeciv_username(value) == value


@pytest.mark.parametrize("value", ["", "ab", "Aagent", "agent_1", "agent1",
                                    "../escape", "a" * 32])
def test_freeciv_username_rejects_path_or_parser_unsafe_values(value):
    with pytest.raises(RecoveryError, match="invalid username"):
        validate_freeciv_username(value)


def test_low_level_runner_rejects_unsafe_username():
    with pytest.raises(RecoveryError, match="invalid username"):
        RunnerConfig(username="../../escape")


def test_low_level_session_rejects_unsafe_username():
    from lechonk.adapter.session import AdapterSession
    with pytest.raises(RecoveryError, match="invalid username"):
        AdapterSession(workdir=".", seed=1, username="agent_1")


def _fixture(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    save = root / "checkpoints" / "player_T4.sav.xz"
    state = root / "conversation" / "state.json"
    save.parent.mkdir()
    state.parent.mkdir()
    save.write_bytes(b"verified save bytes")
    state.write_text('{"version":1,"epoch":0}\n', encoding="utf-8")
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", parent_run_id="run-3", player_id=1,
        config_digest="cfg-digest",
        strategy_memory={"version": 1, "commitments": [], "threats": [],
                         "notes": "hold the frontier"},
    )
    manifest = root / "recovery.json"
    write_checkpoint(manifest, checkpoint)
    return root, save, state, manifest, checkpoint


def test_round_trip_verifies_bound_files_and_identity(tmp_path):
    root, save, state, manifest, expected = _fixture(tmp_path)
    loaded = load_checkpoint(
        manifest, artifact_root=root, expected_turn=4,
        expected_username="player", expected_run_id="run-4",
        expected_config_digest="cfg-digest", expected_player_id=1)
    assert loaded == expected
    assert loaded.checkpoint_sha256 == sha256_file(save)
    assert loaded.conversation_state_sha256 == sha256_file(state)


def test_knowledge_provenance_round_trip_is_exposed_and_bound(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", parent_run_id="run-3", player_id=1,
        config_digest="cfg-digest", strategy_memory={},
        ruleset_reference_schema_version="ruleset_reference_v1",
        ruleset_reference_hash="a" * 64,
        help_reference_schema_version="freeciv_help_reference_v1",
        help_reference_hash="b" * 64,
        help_reference_source_sha256="c" * 64)
    write_checkpoint(manifest, checkpoint)

    loaded = load_checkpoint(
        manifest, artifact_root=root,
        expected_ruleset_reference_schema_version="ruleset_reference_v1",
        expected_ruleset_reference_hash="a" * 64,
        expected_help_reference_schema_version="freeciv_help_reference_v1",
        expected_help_reference_hash="b" * 64,
        expected_help_reference_source_sha256="c" * 64)
    assert loaded.ruleset_reference_hash == "a" * 64
    assert loaded.help_reference_hash == "b" * 64
    assert loaded.help_reference_source_sha256 == "c" * 64


def test_knowledge_provenance_mismatch_fails_closed(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", player_id=1, config_digest="cfg-digest",
        strategy_memory={}, ruleset_reference_hash="a" * 64,
        help_reference_hash="b" * 64)
    write_checkpoint(manifest, checkpoint)

    with pytest.raises(RecoveryError, match="ruleset_reference_hash mismatch"):
        load_checkpoint(manifest, artifact_root=root,
                        expected_ruleset_reference_hash="f" * 64)
    with pytest.raises(RecoveryError, match="help_reference_hash mismatch"):
        load_checkpoint(manifest, artifact_root=root,
                        expected_help_reference_hash="f" * 64)


def test_knowledge_provenance_missing_is_not_silent_on_resume(tmp_path):
    root, _save, _state, manifest, _ = _fixture(tmp_path)
    # The fixture is an old-style manifest with no knowledge fields.  It is
    # still readable for compatibility, but a lookup-enabled resume must say
    # explicitly that the binding is absent.
    with pytest.raises(RecoveryError,
                       match="ruleset_reference_hash is missing"):
        load_checkpoint(manifest, artifact_root=root,
                        expected_ruleset_reference_hash="a" * 64)


@pytest.mark.parametrize("field", [
    "ruleset_reference_hash", "help_reference_hash",
    "help_reference_source_sha256",
])
def test_knowledge_provenance_rejects_non_sha256_values(tmp_path, field):
    root, save, state, _manifest, _ = _fixture(tmp_path)
    kwargs = {field: "not-a-sha256"}
    with pytest.raises(RecoveryError, match=f"invalid {field}"):
        build_checkpoint(
            checkpoint_path=save, artifact_root=root,
            conversation_state_path=state, turn=4, username="player",
            run_id="run-4", player_id=1, config_digest="cfg-digest",
            strategy_memory={}, **kwargs)


def test_static_classic_reference_round_trip_and_tree_mismatch(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", player_id=1, config_digest="cfg-digest",
        strategy_memory={}, classic_reference_schema_version="freeciv-reference-v1",
        classic_ruleset_tree_sha256="d" * 64, classic_reference_present=True)
    write_checkpoint(manifest, checkpoint)
    loaded = load_checkpoint(
        manifest, artifact_root=root,
        expected_classic_reference_schema_version="freeciv-reference-v1",
        expected_classic_ruleset_tree_sha256="d" * 64,
        expected_classic_reference_present=True)
    assert loaded.classic_ruleset_tree_sha256 == "d" * 64
    with pytest.raises(RecoveryError, match="classic_ruleset_tree_sha256 mismatch"):
        load_checkpoint(manifest, artifact_root=root,
                        expected_classic_ruleset_tree_sha256="e" * 64,
                        expected_classic_reference_present=True)


def test_static_classic_reference_unavailable_is_distinct_from_legacy(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", player_id=1, config_digest="cfg-digest",
        strategy_memory={}, classic_reference_schema_version=None,
        classic_ruleset_tree_sha256=None, classic_reference_present=False)
    write_checkpoint(manifest, checkpoint)
    loaded = load_checkpoint(
        manifest, artifact_root=root, expected_classic_reference_present=False)
    assert loaded.classic_reference_present is False

    # A legacy checkpoint has no provenance at all and must be rejected when
    # the caller explicitly asks to bind the static reference state.
    (tmp_path / "legacy").mkdir()
    legacy_root, _save, _state, legacy_manifest, _ = _fixture(tmp_path / "legacy")
    with pytest.raises(RecoveryError, match="classic_reference_present is missing"):
        load_checkpoint(legacy_manifest, artifact_root=legacy_root,
                        expected_classic_reference_present=False)


def test_generated_manual_provenance_round_trip_and_mismatch(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", player_id=1, config_digest="cfg-digest",
        strategy_memory={}, manual_reference_schema_version="freeciv-manual-reference-v1",
        manual_reference_hash="e" * 64, manual_reference_present=True)
    write_checkpoint(manifest, checkpoint)
    loaded = load_checkpoint(
        manifest, artifact_root=root,
        expected_manual_reference_schema_version="freeciv-manual-reference-v1",
        expected_manual_reference_hash="e" * 64,
        expected_manual_reference_present=True)
    assert loaded.manual_reference_hash == "e" * 64
    with pytest.raises(RecoveryError, match="manual_reference_hash mismatch"):
        load_checkpoint(manifest, artifact_root=root,
                        expected_manual_reference_hash="f" * 64,
                        expected_manual_reference_present=True)


def test_generated_manual_unavailable_is_distinct_from_legacy(tmp_path):
    root, save, state, manifest, _ = _fixture(tmp_path)
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=4, username="player",
        run_id="run-4", player_id=1, config_digest="cfg-digest",
        strategy_memory={}, manual_reference_present=False)
    write_checkpoint(manifest, checkpoint)
    loaded = load_checkpoint(
        manifest, artifact_root=root, expected_manual_reference_present=False)
    assert loaded.manual_reference_present is False
    legacy_parent = tmp_path / "legacy"
    legacy_parent.mkdir()
    legacy_manifest = _fixture(legacy_parent)[3]
    with pytest.raises(RecoveryError, match="manual_reference_present is missing"):
        load_checkpoint(legacy_manifest,
                        artifact_root=legacy_parent / "run",
                        expected_manual_reference_present=False)


@pytest.mark.parametrize("field", [
    "manual_reference_hash",
])
def test_generated_manual_provenance_rejects_non_sha256_values(tmp_path, field):
    root, save, state, _manifest, _ = _fixture(tmp_path)
    with pytest.raises(RecoveryError, match=f"invalid {field}"):
        build_checkpoint(
            checkpoint_path=save, artifact_root=root,
            conversation_state_path=state, turn=4, username="player",
            run_id="run-4", player_id=1, config_digest="cfg-digest",
            strategy_memory={}, **{field: "not-a-sha256"})


def test_runtime_manual_artifact_identity_round_trips_into_checkpoint(tmp_path):
    manual = tmp_path / "manual"
    manual.mkdir()
    files = {}
    for number in range(1, 9):
        name = f"classic{number}.mediawiki"
        payload = f"= Classic {number} =\n=== Entry {number} ===\nUseful text.\n"
        (manual / name).write_text(payload, encoding="utf-8")
        files[name] = __import__("hashlib").sha256(payload.encode()).hexdigest()
    image_ref = PINNED_IMAGE_REF
    (manual / "reference.json").write_text(json.dumps({
        "schema_version": "freeciv-manual-reference-v1",
        "image_ref": image_ref,
        "image_id": image_ref.rsplit("@", 1)[-1],
        "ruleset": "classic", "files": files,
    }), encoding="utf-8")
    reference = load_manual_reference(manual)
    assert reference.status == "available"
    root = tmp_path / "run"
    root.mkdir()
    save = root / "save.sav.xz"
    state = root / "state.json"
    save.write_bytes(b"save")
    state.write_text("{}", encoding="utf-8")
    checkpoint = build_checkpoint(
        checkpoint_path=save, artifact_root=root,
        conversation_state_path=state, turn=1, username="player",
        run_id="run", player_id=1, config_digest="cfg", strategy_memory={},
        manual_reference_schema_version=reference.schema_version,
        manual_reference_hash=reference.content_hash,
        manual_reference_source_sha256=reference.source_sha256,
        manual_reference_present=True)
    manifest = root / "recovery.json"
    write_checkpoint(manifest, checkpoint)
    loaded = load_checkpoint(
        manifest, artifact_root=root,
        expected_manual_reference_schema_version=reference.schema_version,
        expected_manual_reference_hash=reference.content_hash,
        expected_manual_reference_source_sha256=reference.source_sha256,
        expected_manual_reference_present=True)
    assert loaded.manual_reference_hash == reference.content_hash
    config = RunnerConfig(
        expected_manual_reference_schema_version=reference.schema_version,
        expected_manual_reference_hash=reference.content_hash,
        expected_manual_reference_source_sha256=reference.source_sha256,
        expected_manual_reference_present=True)
    assert config.expected_manual_reference_hash == reference.content_hash
    assert config.expected_manual_reference_source_sha256 == reference.source_sha256


def test_tampered_save_fails_closed(tmp_path):
    root, save, _state, manifest, _expected = _fixture(tmp_path)
    save.write_bytes(b"tampered")
    with pytest.raises(RecoveryError, match="checkpoint file checksum"):
        load_checkpoint(manifest, artifact_root=root)


def test_path_traversal_and_identity_mismatch_fail(tmp_path):
    root, save, state, manifest, expected = _fixture(tmp_path)
    outside = tmp_path / "outside.sav"
    outside.write_bytes(b"outside")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["checkpoint_path"] = str(outside)
    payload = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    raw["manifest_sha256"] = __import__("hashlib").sha256(
        __import__("lechonk.core.jsonutil", fromlist=["canonical_json"]).canonical_json(
            payload).encode()).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecoveryError, match="outside the artifact root"):
        load_checkpoint(manifest, artifact_root=root)
    # The original manifest is rebuilt to exercise identity checks separately.
    write_checkpoint(manifest, expected)
    with pytest.raises(RecoveryError, match="turn mismatch"):
        load_checkpoint(manifest, artifact_root=root, expected_turn=5)


def test_manifest_checksum_and_atomic_write(tmp_path):
    root, _save, _state, manifest, _expected = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["run_id"] = "changed-without-checksum"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecoveryError, match="manifest checksum"):
        load_checkpoint(manifest, artifact_root=root)


def test_directory_fsync_failure_is_not_reported_as_durable(tmp_path, monkeypatch):
    root, _save, _state, manifest, checkpoint = _fixture(tmp_path)
    # The file itself is fsynced before replacement.  Fail only when the
    # parent-directory fsync is attempted, which is the durability boundary
    # that protects the rename across a crash.
    import lechonk.core.recovery as recovery_module

    original_open = recovery_module.os.open

    def fail_directory_open(path, flags, *args, **kwargs):
        if flags & recovery_module.os.O_DIRECTORY:
            raise OSError("directory fsync unavailable")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(recovery_module.os, "open", fail_directory_open)
    with pytest.raises(RecoveryError, match="durably publish"):
        write_checkpoint(manifest, checkpoint)
    # The temporary file is cleaned up even though publication was not
    # durable; the replaced manifest may exist, but callers must treat the
    # operation as failed and never rely on it as a checkpoint.
    assert not list(manifest.parent.glob(f".{manifest.name}.*"))


def test_missing_bound_state_is_rejected(tmp_path):
    root, save, state, manifest, _expected = _fixture(tmp_path)
    state.unlink()
    with pytest.raises(RecoveryError, match="conversation state path"):
        load_checkpoint(manifest, artifact_root=root)


def test_missing_player_id_is_rejected_as_legacy_manifest(tmp_path):
    root, _save, _state, manifest, _expected = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw.pop("player_id")
    payload = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    raw["manifest_sha256"] = __import__("hashlib").sha256(
        __import__("lechonk.core.jsonutil", fromlist=["canonical_json"]).canonical_json(
            payload).encode()).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecoveryError, match="lacks player_id"):
        load_checkpoint(manifest, artifact_root=root)


def test_old_manifest_version_and_invalid_player_id_fail_closed(tmp_path):
    root, _save, _state, manifest, _expected = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["version"] = 0
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecoveryError, match="unsupported recovery manifest version"):
        load_checkpoint(manifest, artifact_root=root)

    with pytest.raises(RecoveryError, match="invalid checkpoint player_id"):
        build_checkpoint(
            checkpoint_path=root / "checkpoints" / "player_T4.sav.xz",
            artifact_root=root,
            conversation_state_path=root / "conversation" / "state.json",
            turn=4, username="player", run_id="run-4", player_id=True,
            config_digest="cfg-digest", strategy_memory={})


def test_player_id_mismatch_fails_closed(tmp_path):
    root, _save, _state, manifest, _expected = _fixture(tmp_path)
    with pytest.raises(RecoveryError, match="checkpoint player mismatch"):
        load_checkpoint(manifest, artifact_root=root, expected_player_id=2)


def test_runner_fresh_capture_binds_session_player_identity(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints"
    state_path = run_root / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"version":1,"messages":[]}', encoding="utf-8")

    def fake_export(*, username, turn, dest_dir, **_kwargs):
        save = Path(dest_dir) / f"{username}_T{turn}_fake.sav.xz"
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_bytes(b"fresh save")
        return SimpleNamespace(path=str(save), sha256=sha256_file(save))

    import lechonk.eval.checkpoint as checkpoint_module
    monkeypatch.setattr(checkpoint_module, "export_turn_save", fake_export)
    runner = GameRunner(
        config=RunnerConfig(
            run_id="fresh", username="myagent", context_policy="conversation_v1",
            checkpoint_dir=str(checkpoint_dir),
            recovery_manifest_path=str(run_root / "recovery.json"),
            conversation_state_path=str(state_path)),
        client=object(), ledger=TranscriptLedger(tmp_path / "ledger.jsonl"),
        manifest=RunManifest(run_id="fresh"))
    runner._session = SimpleNamespace(player_id=7)
    runner._write_start_checkpoint(4)

    loaded = load_checkpoint(
        run_root / "recovery.json", artifact_root=run_root,
        expected_turn=4, expected_player_id=7)
    assert loaded.player_id == 7
