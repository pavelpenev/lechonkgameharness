# SPDX-License-Identifier: MIT
"""Offline executable checks for wp2_model_run's runtime wiring."""

import importlib.util
import hashlib
import io
import json
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wp2_model_run.py"


def load_runner(monkeypatch, profile):
    run_id = f"offline-wiring-{profile}-{uuid.uuid4().hex}"
    argv = ["wp2_model_run.py", "--run-id", run_id,
            "--prompt-profile", profile, "--context-policy", "conversation_v1"]
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = str(ROOT / "scripts")
    monkeypatch.syspath_prepend(scripts_dir)
    name = f"wp2_wiring_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prompt_snapshot_identity_is_shared_from_recovery_to_main_wiring(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v4")
    module._PROMPT_SNAPSHOT = None
    hashes = module._active_prompt_hashes(bundle_id="gameplay_v4")
    captured = module._PROMPT_SNAPSHOT
    assert captured is not None
    assert hashes == captured.manifest_hashes
    # A later phase must reuse the recovery snapshot rather than recapturing
    # files and risking identity A versus client prompt B.
    assert module._active_system_prompt(
        __import__("lechonk.model.prompts", fromlist=["system_prompt"])) == (
            __import__("lechonk.model.prompts", fromlist=["system_prompt"]).system_prompt(
                "gameplay_v4", bundle_id="gameplay_v4", snapshot=captured))
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_model_call_progress_drives_heartbeat_and_stops_first_physical_finish(
        monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    calls = []

    class FakeHeartbeat:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def start(self, **kwargs):
            calls.append(("start", kwargs))
            return len([item for item in calls if item[0] == "start"])

        def note_stream_activity(self, generation):
            calls.append(("delta", generation))
            return True

        def stop(self, generation=None):
            calls.append(("stop", generation))
            return True

    monkeypatch.setattr(module, "ModelCallHeartbeat", FakeHeartbeat)
    module._MODEL_HEARTBEAT = None
    module._MODEL_HEARTBEAT_GENERATION = None
    module.progress({"event": "model_call_started", "turn": 3, "attempt": 1})
    module.progress({"event": "model_output_delta", "reasoning": "delta"})
    module.progress({"event": "model_call_finished", "turn": 3,
                     "attempt": 1, "physical_kind": "initial"})
    # A lookup can emit another physical finish record; it must not stop a
    # newly-created watchdog because no new logical start occurred.
    module.progress({"event": "model_call_finished", "turn": 3,
                     "attempt": 1, "physical_kind": "tool_final"})
    assert [item[0] for item in calls] == ["init", "start", "delta", "stop"]
    assert calls[1][1]["call_id"].endswith("/t3/call1")
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_repair_start_replaces_generation_and_error_cleanup_stops_it(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    calls = []

    class FakeHeartbeat:
        def __init__(self, **kwargs):
            pass

        def start(self, **kwargs):
            calls.append(("start", kwargs))
            return len(calls)

        def stop(self, generation=None):
            calls.append(("stop", generation))
            return True

    monkeypatch.setattr(module, "ModelCallHeartbeat", FakeHeartbeat)
    module._MODEL_HEARTBEAT = None
    module._MODEL_HEARTBEAT_GENERATION = None
    module.progress({"event": "model_call_started", "turn": 4, "attempt": 1})
    module.progress({"event": "model_call_started", "turn": 4, "attempt": 2})
    module._stop_model_heartbeat()
    assert [item[0] for item in calls] == ["start", "start", "stop"]
    assert calls[-1][1] == 2
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_log_serializes_concurrent_multiline_output_and_survives_closed_stdout(
        monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    sink = io.StringIO()
    monkeypatch.setattr(module, "_log_file", sink)
    monkeypatch.setattr(module, "_STDOUT_BROKEN", True)
    threads = [threading.Thread(target=module.log, args=(f"line-{i}\nbody",))
               for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = sink.getvalue().splitlines()
    assert len(lines) == 40
    assert sum(line.endswith("body") for line in lines) == 20
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_fresh_v3_runtime_identity_is_complete(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    assert module.CLASSIC_RUNTIME_SCHEMA == "freeciv_classic_reference_v1"
    assert module.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
    assert module.PROMPT_BUNDLE_ID == "gameplay_v3"
    assert module.CANDIDATE_SURFACE_POLICY == "candidate_surface_v2"
    from lechonk.core.runner import RunnerConfig
    config = RunnerConfig(
        max_turns=1, run_id=module.RUN, workdir=module.WORKDIR,
        seed=module.SEED, username="myagent", aifill=module.AIFILL,
        prompt_profile=module.PROMPT_PROFILE,
        prompt_bundle_id=module.PROMPT_BUNDLE_ID,
        candidate_surface_policy=module.CANDIDATE_SURFACE_POLICY,
        candidate_surface_cap=8000,
        candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v3-budget-v1",
        recovery_config_digest_schema=module.RECOVERY_DIGEST_SCHEMA)
    assert config.prompt_bundle_id == "gameplay_v3"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_fresh_v4_runtime_identity_is_complete(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v4")
    assert module.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
    assert module.PROMPT_BUNDLE_ID == "gameplay_v4"
    assert module.CANDIDATE_SURFACE_POLICY == "candidate_surface_v2"
    assert module.recovery_config_digest().startswith("sha256:")
    from lechonk.core.runner import RunnerConfig
    config = RunnerConfig(
        max_turns=1, run_id=module.RUN, workdir=module.WORKDIR,
        seed=module.SEED, username="myagent", aifill=module.AIFILL,
        prompt_profile=module.PROMPT_PROFILE,
        prompt_bundle_id=module.PROMPT_BUNDLE_ID,
        candidate_surface_policy=module.CANDIDATE_SURFACE_POLICY,
        candidate_surface_cap=8000,
        candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v4-budget-v1",
        recovery_config_digest_schema=module.RECOVERY_DIGEST_SCHEMA)
    assert config.prompt_profile == "gameplay_v4"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_fresh_v5_runtime_identity_is_complete(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v5")
    assert module.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
    assert module.PROMPT_BUNDLE_ID == "gameplay_v5"
    assert module.CANDIDATE_SURFACE_POLICY == "candidate_surface_v2"
    from lechonk.core.runner import RunnerConfig
    config = RunnerConfig(
        max_turns=1, run_id=module.RUN, workdir=module.WORKDIR,
        seed=module.SEED, username="myagent", aifill=module.AIFILL,
        prompt_profile=module.PROMPT_PROFILE,
        prompt_bundle_id=module.PROMPT_BUNDLE_ID,
        candidate_surface_policy=module.CANDIDATE_SURFACE_POLICY,
        candidate_surface_cap=8000,
        candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v5-budget-v1",
        recovery_config_digest_schema=module.RECOVERY_DIGEST_SCHEMA)
    assert config.prompt_profile == "gameplay_v5"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_token_usage_totals_keep_null_compaction_usage_unknown(monkeypatch):
    """A structured preflight error must not crash post-run aggregation."""
    module = load_runner(monkeypatch, "gameplay_v3")
    compaction_error = {
        "kind": "compaction",
        "outcome": "error",
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": None,
        "latency_ms": 12,
    }
    result = module._token_usage_totals([
        {"tokens_in": 100, "tokens_out": 7}, compaction_error])
    assert result == {
        "tokens_in": 100,
        "tokens_out": 7,
        "tokens_in_unknown_calls": 1,
        "tokens_out_unknown_calls": 1,
        "token_measurement_status": "unknown",
    }
    # The raw structured record is evidence and remains unmodified.
    assert compaction_error["tokens_in"] is None
    assert compaction_error["tokens_out"] is None
    assert module._numeric_total([compaction_error], "latency_ms") == 12
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_token_usage_totals_reject_non_numeric_token_values(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    result = module._token_usage_totals([
        {"tokens_in": "unknown", "tokens_out": True},
        {"tokens_in": 3, "tokens_out": 4.5},
    ])
    assert result["tokens_in"] == 3
    assert result["tokens_out"] == 4.5
    assert result["tokens_in_unknown_calls"] == 1
    assert result["tokens_out_unknown_calls"] == 1
    assert result["token_measurement_status"] == "unknown"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_action_eligibility_distinguishes_milestone_and_routine(
        monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    assert module._capture_sync_action_kind({
        "event": "action_result", "domain": "unit", "verb": "attack",
    }) == (True, False)
    assert module._capture_sync_action_kind({
        "event": "action_result", "verb": "goto", "outcome": "verified",
    }) == (False, True)
    assert module._capture_sync_action_kind({
        "event": "turn_decision", "verb": "attack",
    }) == (False, False)
    assert module._capture_sync_action_kind({
        "event": "action_result", "verb": "research",
        "provenance_class": "engine_automatic",
    }) == (False, False)
    # A routine movement whose readback mentions a city remains routine.
    assert module._capture_sync_action_kind({
        "event": "action_result", "verb": "goto",
        "target": "city",
    }) == (False, True)
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


class _CheckpointGateway:
    def __init__(self, directory, *, capture_id="capture-test"):
        self.capture_dir = Path(directory)
        self.capture_id = capture_id
        self.receipts = []
        self.results = []
        self.next_seq = 17

    def offer_control_receipt(self, kind, payload):
        self.receipts.append((kind, payload))
        return self.next_seq

    def offer_control(self, kind, payload):
        self.results.append((kind, payload))
        return True


def _configure_checkpoint_module(module, gateway, *, budget=500,
                                 timeout=40):
    module._CAPTURE_GATEWAY = gateway
    module.CAPTURE_SYNC = True
    module.CAPTURE_SYNC_BUDGET_MS = budget
    module.CAPTURE_SYNC_TIMEOUT_MS = timeout
    module._CAPTURE_SYNC_TURN = None
    module._CAPTURE_SYNC_WAIT_USED_MS = 0.0
    module._CAPTURE_SYNC_ROUTINE_CHECKPOINTED = False


def test_capture_sync_accepts_only_matching_ack_identity(monkeypatch, tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    ack_dir = tmp_path / "checkpoint-acks"
    ack_dir.mkdir()
    _configure_checkpoint_module(module, gateway)
    (ack_dir / "17.json").write_text(json.dumps({
        "run_id": module.RUN, "capture_id": gateway.capture_id,
        "capture_seq": 17, "status": "captured",
    }))
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 1, "verb": "attack",
    }) is True
    assert gateway.results[-1][1]["capture_sync_result"] == "captured"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_budget_ignores_model_latency(monkeypatch, tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    ack_dir = tmp_path / "checkpoint-acks"
    ack_dir.mkdir()
    _configure_checkpoint_module(module, gateway, budget=40, timeout=40)
    # Simulate a slow reasoning/API phase before the action result arrives.
    module.progress({"event": "turn_started", "turn": 6})
    module.time.sleep(0.06)
    (ack_dir / "17.json").write_text(json.dumps({
        "run_id": module.RUN, "capture_id": gateway.capture_id,
        "capture_seq": 17, "status": "captured",
    }))
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 6, "verb": "attack",
    }) is True
    assert gateway.results[-1][1]["capture_sync_result"] == "captured"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_delayed_ack_records_wait_budget(monkeypatch, tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    ack_dir = tmp_path / "checkpoint-acks"
    ack_dir.mkdir()
    _configure_checkpoint_module(module, gateway, budget=200, timeout=120)

    def delayed_ack():
        module.time.sleep(0.04)
        (ack_dir / "17.json").write_text(json.dumps({
            "run_id": module.RUN, "capture_id": gateway.capture_id,
            "capture_seq": 17, "status": "captured",
        }))

    writer = threading.Thread(target=delayed_ack)
    writer.start()
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 7, "verb": "attack",
    }) is True
    writer.join()
    result = gateway.results[-1][1]
    assert result["capture_sync_result"] == "captured"
    assert result["wait_budget_ms"] == 200
    assert 20 <= result["wait_used_ms"] <= 200
    assert result["wait_remaining_ms"] < 200
    assert result["turn_resolution"] == "event"
    assert result["turn_uncertainty"] is False
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_rejects_wrong_identity_and_missing_presenter(monkeypatch,
                                                                    tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    ack_dir = tmp_path / "checkpoint-acks"
    ack_dir.mkdir()
    _configure_checkpoint_module(module, gateway, timeout=20)
    (ack_dir / "17.json").write_text(json.dumps({
        "run_id": "other-run", "capture_id": gateway.capture_id,
        "capture_seq": 17, "status": "captured",
    }))
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 2, "verb": "attack",
    }) is True
    assert gateway.results[-1][1]["capture_sync_result"] == "timeout"
    gateway.results.clear()
    gateway.next_seq = 18
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 2, "verb": "research",
    }) is True
    assert gateway.results[-1][1]["capture_sync_result"] == "timeout"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_turn_budget_exhaustion_and_no_sync_behavior(monkeypatch,
                                                                  tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    _configure_checkpoint_module(module, gateway, budget=1, timeout=20)
    module._CAPTURE_SYNC_TURN = "3"
    module._CAPTURE_SYNC_WAIT_USED_MS = 2
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 3, "verb": "attack",
    }) is False
    assert gateway.results[-1][1]["capture_sync_result"] == "budget_exhausted"
    module.CAPTURE_SYNC = False
    gateway.results.clear()
    assert module._capture_sync_wait({
        "event": "action_result", "turn": 4, "verb": "attack",
    }) is False
    assert gateway.results == []
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_uses_turn_boundary_when_action_omits_turn(monkeypatch,
                                                                tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    gateway = _CheckpointGateway(tmp_path)
    _configure_checkpoint_module(module, gateway, timeout=20)
    module.progress({"event": "turn_started", "turn": 8})
    assert module._CAPTURE_SYNC_TURN == "8"
    # No turn on this action: the established boundary is used for the
    # checkpoint rather than creating an unknown bucket.
    assert module._capture_sync_action_kind({
        "event": "action_result", "verb": "attack",
    }) == (True, False)
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_capture_sync_cli_requires_capture_and_reserves_deadline(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    with pytest.raises(SystemExit):
        module._parse_args(["--capture-sync", "--stream"])
    with pytest.raises(SystemExit):
        module._parse_args(["--capture", "--stream", "--capture-sync",
                            "--acting-deadline-s", "150"])
    parsed = module._parse_args([
        "--capture", "--stream", "--capture-sync", "--acting-deadline-s",
        "151", "--run-id", "sync-parse-test"])
    assert parsed.capture_sync is True
    assert parsed.capture_sync_budget_ms == 6000
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_section_max_preserves_unknown_only_measurements(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    assert module._numeric_max([None, "unknown", True]) is None
    assert module._section_max([
        {"section_tokens": {"snapshot": None}},
        {"section_tokens": {"snapshot": "unknown"}},
    ], "snapshot") is None
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_section_max_ignores_unknown_mixed_measurements(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    assert module._numeric_max([None, 1573, "unknown", 1200]) == 1573
    assert module._section_max([
        {"section_tokens": {"delta": None}},
        {"section_tokens": {"delta": 17}},
        {"section_tokens": {"delta": "unknown"}},
    ], "delta") == 17
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_snapshot_budget_distinguishes_missing_ledger_from_zero(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v3")
    assert not module._snapshot_within_8000(0, None)
    assert module._snapshot_within_8000(0, 0)
    assert not module._snapshot_within_8000(9000, 0)
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_model_output_callback_writes_bound_trace_and_retry_preserves_it(
        monkeypatch, tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    trace = tmp_path / "model_traces.jsonl"
    module.TRACE_PATH = str(trace)
    module._render_model_output({
        "turn": 1, "attempt": 1, "reasoning_text": "full reasoning",
        "answer_text": '{"turn_id":"1","actions":[]}',
        "tool_calls": [], "content_chunks": ["complete output"],
    })
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[0]["reasoning_text"] == "full reasoning"
    assert records[0]["complete"] is True
    trace.replace(trace.with_name("model_traces.jsonl.attempt1"))
    trace.write_text('{"turn":2,"complete":true}\n')
    assert "full reasoning" in trace.with_name("model_traces.jsonl.attempt1").read_text()
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_original_and_corrected_bundle_child_digests_differ(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v2")
    module.RECOVERY_DIGEST_SCHEMA = "interactive-config-v3"
    module.CANDIDATE_SURFACE_POLICY = "legacy_v1"
    digests = []
    for bundle in ("gameplay_v2_original", "gameplay_v2_turnid_fix"):
        module.PROMPT_BUNDLE_ID = bundle
        digests.append(module.recovery_config_digest())
    assert digests[0] != digests[1]
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_v2_frozen_bundle_child_identity_is_accepted(monkeypatch):
    module = load_runner(monkeypatch, "gameplay_v2")
    module.RECOVERY_DIGEST_SCHEMA = "interactive-config-v3"
    module.PROMPT_BUNDLE_ID = "gameplay_v2_original"
    module.CANDIDATE_SURFACE_POLICY = "legacy_v1"
    from lechonk.core.runner import RunnerConfig
    config = RunnerConfig(
        max_turns=1, run_id=module.RUN, workdir=module.WORKDIR,
        seed=module.SEED, username="myagent", aifill=2,
        prompt_profile="gameplay_v2", prompt_bundle_id="gameplay_v2_original",
        candidate_surface_policy="legacy_v1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="legacy-budget-v1",
        recovery_config_digest_schema="interactive-config-v3")
    assert config.recovery_config_digest_schema == "interactive-config-v3"
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_mismatched_resume_system_blocks_probe_before_boundary(monkeypatch, tmp_path):
    module = load_runner(monkeypatch, "gameplay_v3")
    from lechonk.model.conversation import ConversationConfig, ConversationManager
    state = tmp_path / "state.json"
    manager = ConversationManager(
        system_message={"role": "system", "content": "wrong bundle"},
        config=ConversationConfig(), state_path=state,
        journal_path=tmp_path / "conversation.jsonl")
    manager._persist()
    probe_calls = []
    with pytest.raises(Exception):
        module.validate_persisted_conversation_before_probe(
            state, profile="gameplay_v3", bundle_id="gameplay_v3",
            trigger_tokens=500_000, tail_tokens=100_000)
    assert probe_calls == []
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_historical_checkpoint_resolves_original_bundle_offline(monkeypatch, tmp_path):
    """Exercise the real CLI import/resume branch without entering main().

    The historical parent is copied into a temporary artifact root and its
    checksummed paths are rewritten to that copy.  Importing the script with
    ``__name__ != '__main__'`` runs manifest verification and lineage
    selection, but cannot start Docker, Freeciv, or an API call.
    """
    from lechonk.core.jsonutil import canonical_json

    source_root = ROOT / "artifacts" / "runs" / "night-20261027-2300"
    source_manifest = source_root / "recovery.json"
    if not source_manifest.is_file():
        pytest.skip("historical overnight checkpoint is unavailable")

    parent_root = tmp_path / "historical-parent"
    shutil.copytree(source_root, parent_root)
    manifest_path = parent_root / "recovery.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("checkpoint_path", "conversation_state_path"):
        original = Path(data[field])
        rewritten = parent_root / "checkpoints" / original.name
        assert rewritten.is_file(), rewritten
        data[field] = str(rewritten)
    data.pop("manifest_sha256", None)
    data["manifest_sha256"] = hashlib.sha256(
        canonical_json(data).encode("utf-8")).hexdigest()
    manifest_path.write_text(canonical_json(data) + "\n", encoding="utf-8")

    run_id = f"offline-historical-resume-{uuid.uuid4().hex}"
    argv = [
        "wp2_model_run.py", "--resume", str(manifest_path),
        "--run-id", run_id, "--seed", "20261028", "--aifill", "2",
        "--prompt-profile", "gameplay_v2", "--context-policy", "conversation_v1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = str(ROOT / "scripts")
    monkeypatch.syspath_prepend(scripts_dir)
    name = f"wp2_historical_resume_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        assert module.RESUME_CHECKPOINT is not None
        assert module.PROMPT_BUNDLE_ID == "gameplay_v2_original"
        assert module.PROMPT_PROFILE == "gameplay_v2"
        assert module.CANDIDATE_SURFACE_POLICY == "legacy_v1"
        assert module.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
    finally:
        shutil.rmtree(getattr(module, "RUNDIR", ""), ignore_errors=True)


def test_resume_rejects_changed_prompt_bundle_before_main(monkeypatch, tmp_path):
    """The real import/resume path fails closed when prompt identity changes."""
    from lechonk.core.jsonutil import canonical_json
    from lechonk.model import prompts
    source_root = ROOT / "artifacts" / "runs" / "night-20261027-2300"
    source_manifest = source_root / "recovery.json"
    if not source_manifest.is_file():
        pytest.skip("historical overnight checkpoint is unavailable")
    parent_root = tmp_path / "changed-parent"
    shutil.copytree(source_root, parent_root)
    manifest_path = parent_root / "recovery.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("checkpoint_path", "conversation_state_path"):
        original = Path(data[field])
        data[field] = str(parent_root / "checkpoints" / original.name)
    data.pop("manifest_sha256", None)
    data["manifest_sha256"] = hashlib.sha256(
        canonical_json(data).encode("utf-8")).hexdigest()
    manifest_path.write_text(canonical_json(data) + "\n", encoding="utf-8")
    original_capture = prompts.capture_prompt_snapshot
    def changed_capture(profile, *, bundle_id=None):
        snapshot = original_capture(profile, bundle_id=bundle_id)
        hashes = dict(snapshot.hashes)
        name = next(iter(hashes))
        hashes[name] = "sha256:" + "0" * 64
        return prompts.PromptSnapshot(snapshot.profile, snapshot.bundle_id,
                                     snapshot.texts, hashes)
    monkeypatch.setattr(prompts, "capture_prompt_snapshot", changed_capture)
    run_id = f"offline-changed-prompt-{uuid.uuid4().hex}"
    monkeypatch.setattr(sys, "argv", ["wp2_model_run.py", "--resume",
        str(manifest_path), "--run-id", run_id, "--seed", "20261028",
        "--aifill", "2", "--prompt-profile", "gameplay_v2",
        "--context-policy", "conversation_v1"])
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    name = f"wp2_changed_resume_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(Exception, match="legacy lineage provenance"):
        spec.loader.exec_module(module)


def test_synthetic_frozen_bundle_resume_and_wrong_seed_reject(monkeypatch, tmp_path):
    """Exercise unversioned parent -> explicit child -> child resume offline."""
    from lechonk.core.recovery import build_checkpoint, write_checkpoint
    from lechonk.core.recovery_identity import IdentityInput, digest_for, legacy_specs
    from lechonk.model.conversation import ConversationConfig, ConversationManager
    from lechonk.model import prompts

    source = load_runner(monkeypatch, "gameplay_v2")
    source.RECOVERY_DIGEST_SCHEMA = None
    source.PROMPT_BUNDLE_ID = "gameplay_v2_original"
    source.CANDIDATE_SURFACE_POLICY = "legacy_v1"
    refs = {
        # Checkpoint provenance stores the semantic loader schema; digest
        # identity below deliberately uses the extractor schema.
        "classic_reference_schema_version": source.CLASSIC_RUNTIME_SCHEMA,
        "classic_ruleset_tree_sha256": source.STATIC_CLASSIC_REFERENCE["tree_sha256"],
        "classic_reference_present": source.STATIC_CLASSIC_REFERENCE["present"],
        "manual_reference_schema_version": source.STATIC_MANUAL_REFERENCE["schema_version"],
        "manual_reference_hash": source.STATIC_MANUAL_REFERENCE["hash"],
        "manual_reference_source_sha256": source.STATIC_MANUAL_REFERENCE["source_sha256"],
        "manual_reference_present": source.STATIC_MANUAL_REFERENCE["present"],
    }
    parent_root = tmp_path / "parent"
    (parent_root / "checkpoints").mkdir(parents=True)
    save = parent_root / "checkpoints" / "player_T1.sav.xz"
    save.write_bytes(b"synthetic verified save")
    state = parent_root / "checkpoints" / "conversation-state.json"
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt(
            "gameplay_v2", bundle_id="gameplay_v2_original")},
        config=ConversationConfig(), state_path=state,
        journal_path=parent_root / "conversation.jsonl")
    manager._persist()
    original_hashes = source._active_prompt_hashes(bundle_id="gameplay_v2_original")
    original_spec, _ = legacy_specs(
        original_prompt_hashes=original_hashes,
        corrected_prompt_hashes=source._active_prompt_hashes(
            bundle_id="gameplay_v2_turnid_fix"))
    parent_identity = IdentityInput(
        context_policy="conversation_v1", prompt_profile="gameplay_v2",
        model_id="mistral-large-4", prompt_hashes=original_hashes,
        ruleset="classic", seed=20261028, action_schema="ActionBatch-v1",
            aifill=2, configured_ai_opponents=1,
            prompt_bundle_id="gameplay_v2_original",
            classic_reference_schema_version=source.STATIC_CLASSIC_REFERENCE["schema_version"],
            classic_ruleset_tree_sha256=source.STATIC_CLASSIC_REFERENCE["tree_sha256"],
            classic_reference_present=source.STATIC_CLASSIC_REFERENCE["present"],
            manual_reference_schema_version=refs["manual_reference_schema_version"],
            manual_reference_hash=refs["manual_reference_hash"],
            manual_reference_source_sha256=refs["manual_reference_source_sha256"],
            manual_reference_present=refs["manual_reference_present"])
    parent_digest = digest_for(original_spec, parent_identity)
    parent = build_checkpoint(
        checkpoint_path=save, artifact_root=parent_root,
        conversation_state_path=state, turn=1, username="myagent",
        run_id="synthetic-parent", player_id=1, config_digest=parent_digest,
        strategy_memory={}, **refs)
    manifest = parent_root / "recovery.json"
    write_checkpoint(manifest, parent)

    def import_resume(parent_manifest: Path, seed: int):
        run_id = f"offline-child-{seed}-{uuid.uuid4().hex}"
        argv = ["wp2_model_run.py", "--resume", str(parent_manifest),
                "--run-id", run_id, "--seed", str(seed), "--aifill", "2",
                "--prompt-profile", "gameplay_v2",
                "--context-policy", "conversation_v1"]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.syspath_prepend(str(ROOT / "scripts"))
        name = f"wp2_child_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        child = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(child)
        return child

    try:
        child = import_resume(manifest, 20261028)
        assert child.PROMPT_BUNDLE_ID == "gameplay_v2_original"
        assert child.CANDIDATE_SURFACE_POLICY == "legacy_v1"
        assert child.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
        child_root = tmp_path / "child"
        (child_root / "checkpoints").mkdir(parents=True)
        child_save = child_root / "checkpoints" / "player_T2.sav.xz"
        child_save.write_bytes(b"synthetic child save")
        child_state = child_root / "checkpoints" / "conversation-state.json"
        shutil.copy2(state, child_state)
        child_checkpoint = build_checkpoint(
            checkpoint_path=child_save, artifact_root=child_root,
            conversation_state_path=child_state, turn=2, username="myagent",
            run_id="synthetic-child", parent_run_id="synthetic-parent",
            player_id=1, config_digest=child.recovery_config_digest(),
            strategy_memory={}, config_digest_schema="interactive-config-v3",
            prompt_bundle_id="gameplay_v2_original", **refs)
        child_manifest = child_root / "recovery.json"
        write_checkpoint(child_manifest, child_checkpoint)
        second = import_resume(child_manifest, 20261028)
        assert second.PROMPT_BUNDLE_ID == "gameplay_v2_original"
        assert second.RESUME_CHECKPOINT.config_digest == child_checkpoint.config_digest
        shutil.rmtree(child.RUNDIR, ignore_errors=True)
        shutil.rmtree(second.RUNDIR, ignore_errors=True)
        with pytest.raises(SystemExit, match="resume rejected"):
            import_resume(child_manifest, 999999)
    finally:
        shutil.rmtree(getattr(source, "RUNDIR", ""), ignore_errors=True)


def test_synthetic_v4_checkpoint_child_resume_and_profile_seed_reject(
        monkeypatch, tmp_path):
    """Import-only v4 checkpoint/child lineage test.

    This exercises the real script resume gate, including the persisted v4
    system message, frozen bundle/policy/budget identity, and conversation
    cache lineage inputs, without starting Freeciv or constructing a client.
    """
    from lechonk.core.recovery import build_checkpoint, write_checkpoint
    from lechonk.model.conversation import ConversationConfig, ConversationManager
    from lechonk.model import prompts

    source = load_runner(monkeypatch, "gameplay_v4")
    refs = {
        "classic_reference_schema_version": source.CLASSIC_RUNTIME_SCHEMA,
        "classic_ruleset_tree_sha256": source.STATIC_CLASSIC_REFERENCE["tree_sha256"],
        "classic_reference_present": source.STATIC_CLASSIC_REFERENCE["present"],
        "manual_reference_schema_version": source.STATIC_MANUAL_REFERENCE["schema_version"],
        "manual_reference_hash": source.STATIC_MANUAL_REFERENCE["hash"],
        "manual_reference_source_sha256": source.STATIC_MANUAL_REFERENCE["source_sha256"],
        "manual_reference_present": source.STATIC_MANUAL_REFERENCE["present"],
    }
    parent_root = tmp_path / "v4-parent"
    (parent_root / "checkpoints").mkdir(parents=True)
    save = parent_root / "checkpoints" / "player_T1.sav.xz"
    save.write_bytes(b"synthetic v4 verified save")
    state = parent_root / "checkpoints" / "conversation-state.json"
    ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt(
            "gameplay_v4", bundle_id="gameplay_v4")},
        config=ConversationConfig(), state_path=state,
        journal_path=parent_root / "conversation.jsonl")._persist()
    parent = build_checkpoint(
        checkpoint_path=save, artifact_root=parent_root,
        conversation_state_path=state, turn=1, username="myagent",
        run_id="synthetic-v4-parent", player_id=1,
        config_digest=source.recovery_config_digest(), strategy_memory={},
        config_digest_schema="interactive-config-v3",
        prompt_bundle_id="gameplay_v4", **refs)
    manifest = parent_root / "recovery.json"
    write_checkpoint(manifest, parent)

    def import_resume(parent_manifest: Path, *, seed=20261028,
                      profile="gameplay_v4"):
        run_id = f"offline-v4-child-{uuid.uuid4().hex}"
        argv = ["wp2_model_run.py", "--resume", str(parent_manifest),
                "--run-id", run_id, "--seed", str(seed), "--aifill", "2",
                "--prompt-profile", profile,
                "--context-policy", "conversation_v1"]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.syspath_prepend(str(ROOT / "scripts"))
        name = f"wp2_v4_child_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        child = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(child)
        return child

    modules = []
    try:
        child = import_resume(manifest)
        modules.append(child)
        assert child.PROMPT_PROFILE == "gameplay_v4"
        assert child.PROMPT_BUNDLE_ID == "gameplay_v4"
        assert child.CANDIDATE_SURFACE_POLICY == "candidate_surface_v2"
        assert child.RECOVERY_DIGEST_SCHEMA == "interactive-config-v3"
        assert child.CONVERSATION_COMPACTION_LIMIT_TOKENS == 500_000
        assert child.CONVERSATION_TAIL_TARGET_TOKENS == 100_000
        assert child.RESUME_CHECKPOINT.prompt_bundle_id == "gameplay_v4"

        child_root = tmp_path / "v4-child"
        (child_root / "checkpoints").mkdir(parents=True)
        child_save = child_root / "checkpoints" / "player_T2.sav.xz"
        child_save.write_bytes(b"synthetic v4 child save")
        child_state = child_root / "checkpoints" / "conversation-state.json"
        shutil.copy2(state, child_state)
        child_checkpoint = build_checkpoint(
            checkpoint_path=child_save, artifact_root=child_root,
            conversation_state_path=child_state, turn=2, username="myagent",
            run_id="synthetic-v4-child", parent_run_id="synthetic-v4-parent",
            player_id=1, config_digest=child.recovery_config_digest(),
            strategy_memory={}, config_digest_schema="interactive-config-v3",
            prompt_bundle_id="gameplay_v4", **refs)
        child_manifest = child_root / "recovery.json"
        write_checkpoint(child_manifest, child_checkpoint)
        second = import_resume(child_manifest)
        modules.append(second)
        assert second.PROMPT_BUNDLE_ID == "gameplay_v4"
        assert second.RESUME_CHECKPOINT.config_digest == child_checkpoint.config_digest
        with pytest.raises(SystemExit, match="resume rejected"):
            import_resume(child_manifest, profile="gameplay_v3")
        with pytest.raises(SystemExit, match="resume rejected"):
            import_resume(child_manifest, seed=999999)
    finally:
        for module in modules:
            shutil.rmtree(getattr(module, "RUNDIR", ""), ignore_errors=True)
        shutil.rmtree(getattr(source, "RUNDIR", ""), ignore_errors=True)
