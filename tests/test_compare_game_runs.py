import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import compare_game_runs as cgr
from lechonk.core.jsonutil import canonical_json
from lechonk.model.conversation import ConversationManager
from lechonk.core.recovery import build_checkpoint, write_checkpoint


def checked(obj):
    payload = dict(obj)
    payload["checksum"] = hashlib.sha256(
        canonical_json({k: v for k, v in payload.items() if k != "checksum"}).encode()
    ).hexdigest()
    return payload


def write_run(root, *, parent=None, bad_transcript=False, zero=False, secret=False):
    root.mkdir()
    manifest = {"run_id": root.name, "config_digest": "same", "prompt_profile": "gameplay_v4"}
    if parent:
        manifest["parent_run_id"] = parent
    records = [checked({"seq": 0, "kind": "run_manifest", "manifest": manifest}),
               checked({"seq": 1, "kind": "turn_start", "turn": 1}),
               checked({"seq": 2, "kind": "action_result", "turn": 1,
                       "outcome": "executed"})]
    if bad_transcript:
        records[2]["checksum"] = "bad"
    (root / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
    api = {"turn": 1, "tokens_in": 0 if zero else 4, "tokens_out": 0 if zero else 2,
           "cached_tokens": 0 if zero else 1, "latency_ms": 0 if zero else 5,
           "attempt": 1}
    (root / "api_calls.jsonl").write_text(json.dumps(api) + "\n")
    cp = root / "checkpoints"
    cp.mkdir()
    snapshot = {"turn": 1, "player": {"score": 0 if zero else 12, "gold": 0,
                "techs_researched": 0}, "cities": [{"id": 1, "is_own": True,
                "production": "Warrior"}], "units": []}
    messages = [{"role": "user", "content": json.dumps(snapshot)}]
    if secret:
        messages.append({"role": "assistant", "content": "SECRET_REASONING_SHOULD_NOT_APPEAR"})
        (root / "model_traces.jsonl").write_text("SECRET_REASONING_SHOULD_NOT_APPEAR")
    state = {"version": 1, "policy_version": "x", "epoch": 2,
             "messages": messages, "prompt_cache_key": "x"}
    state["checksum"] = hashlib.sha256(canonical_json(state).encode()).hexdigest()
    (cp / "conversation-state-turn-1.json").write_text(json.dumps(state))


def test_missing_fields_and_true_zero_are_distinct(tmp_path):
    write_run(tmp_path / "a", zero=True)
    write_run(tmp_path / "b")
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    a = report["runs"][0]["turns"][0]
    assert a["state"]["score"] == 0
    assert a["provider"]["input_tokens"] == 0
    assert a["provider"]["cached_tokens"] == 0
    assert a["actions"].get("rejected") is None


def test_checksum_invalid_is_flagged_and_record_not_used(tmp_path):
    write_run(tmp_path / "bad", bad_transcript=True)
    write_run(tmp_path / "good")
    report = cgr.compare_runs([tmp_path / "bad", tmp_path / "good"])
    bad = report["runs"][0]
    assert bad["integrity"]["transcript"]["checksum_invalid"] == 1
    assert not bad["integrity"]["transcript"]["valid"]
    assert bad["turns"][0]["actions"] == {}


def test_fork_flag_and_no_reasoning_leak(tmp_path):
    write_run(tmp_path / "parent", secret=True)
    write_run(tmp_path / "child", parent="parent", secret=True)
    report = cgr.compare_runs([tmp_path / "parent", tmp_path / "child"])
    assert any("fork" in flag for flag in report["flags"])
    blob = json.dumps(report)
    assert "SECRET_REASONING" not in blob


def test_markdown_is_concise(tmp_path):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b")
    text = cgr.markdown(cgr.compare_runs([tmp_path / "a", tmp_path / "b"]))
    assert "| run | turn | score" in text
    assert "SECRET" not in text


def test_real_prompt_snapshot_section_is_bounded_and_tool_final_is_not_repair(tmp_path):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b")
    for root in (tmp_path / "a", tmp_path / "b"):
        state = next((root / "checkpoints").glob("*.json"))
        obj = json.loads(state.read_text())
        snapshot = json.loads(obj["messages"][0]["content"])
        obj["messages"][0]["content"] = (
            "intro\n## SNAPSHOT\n" + json.dumps(snapshot) +
            "\n## DELTA\nnot a snapshot")
        obj["checksum"] = hashlib.sha256(canonical_json(
            {k: v for k, v in obj.items() if k != "checksum"}).encode()).hexdigest()
        state.write_text(json.dumps(obj))
        (root / "api_calls.jsonl").write_text(json.dumps({
            "turn": 1, "kind": "tool_final", "attempt": 2,
            "tokens_in": 1, "tokens_out": 1, "latency_ms": 1}) + "\n")
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    assert report["runs"][0]["turns"][0]["state"]["cities_own"] == 1
    assert report["runs"][0]["aggregate"]["actions"].get("repairs") is None


def test_conflicting_checkpoint_snapshots_are_flagged(tmp_path):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b")
    for root in (tmp_path / "a", tmp_path / "b"):
        source = next((root / "checkpoints").glob("*.json"))
        copy = root / "checkpoints" / "conversation-state-turn-1-copy.json"
        obj = json.loads(source.read_text())
        obj["messages"][0]["content"] = json.dumps({
            "turn": 1, "player": {"score": 999, "gold": 0, "techs_researched": 0},
            "cities": [], "units": []})
        obj["checksum"] = hashlib.sha256(canonical_json(
            {k: v for k, v in obj.items() if k != "checksum"}).encode()).hexdigest()
        copy.write_text(json.dumps(obj))
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    assert any("conflicting checkpoint snapshots" in f for f in report["flags"])


def test_identical_snapshot_epoch_difference_is_not_conflict(tmp_path):
    write_run(tmp_path / "a")
    state = next((tmp_path / "a" / "checkpoints").glob("*.json"))
    obj = json.loads(state.read_text()); obj["epoch"] = 99
    obj["checksum"] = hashlib.sha256(canonical_json(
        {k: v for k, v in obj.items() if k != "checksum"}).encode()).hexdigest()
    (tmp_path / "a" / "checkpoints" / "later.json").write_text(json.dumps(obj))
    write_run(tmp_path / "b")
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    assert not any("conflicting checkpoint snapshots" in f for f in report["flags"])
    assert report["runs"][0]["turns"][0]["compaction_epoch"] in (2, 99)


def test_snapshot_symlink_escape_is_rejected_and_flagged(tmp_path):
    write_run(tmp_path / "a"); write_run(tmp_path / "b")
    outside = tmp_path / "outside.json"; outside.write_text("{}")
    (tmp_path / "a" / "checkpoints" / "conversation-state-escape.json").symlink_to(outside)
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    assert report["runs"][0]["integrity"]["snapshots"]["unsafe_paths"]
    assert any("symlinked" in f for f in report["flags"])


def test_missing_leading_and_trailing_state_turns_are_flagged(tmp_path):
    write_run(tmp_path / "a"); write_run(tmp_path / "b")
    root = tmp_path / "a"
    lines = [json.loads(x) for x in (root / "transcript.jsonl").read_text().splitlines()]
    lines.extend([checked({"seq": 3, "kind": "turn_start", "turn": 2}),
                  checked({"seq": 4, "kind": "turn_start", "turn": 3})])
    (root / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    report = cgr.compare_runs([root, tmp_path / "b"])
    assert any("missing state turns 2,3" in f for f in report["flags"])


def test_state_only_epoch_is_in_aggregate_metadata(tmp_path):
    write_run(tmp_path / "a"); write_run(tmp_path / "b")
    root = tmp_path / "a"
    state = root / "state.json"
    obj = {"version": 1, "policy_version": "x", "epoch": 17,
           "messages": [{"role": "user", "content": "no snapshot"}]}
    obj["checksum"] = hashlib.sha256(canonical_json(obj).encode()).hexdigest()
    state.write_text(json.dumps(obj))
    report = cgr.compare_runs([root, tmp_path / "b"])
    assert 17 in report["runs"][0]["compaction_epochs"]


def test_state_without_extractable_snapshot_is_unavailable(tmp_path):
    write_run(tmp_path / "a"); write_run(tmp_path / "b")
    state = tmp_path / "a" / "state.json"
    obj = {"version": 1, "policy_version": "x", "epoch": 1,
           "messages": [{"role": "user", "content": "no snapshot here"}]}
    obj["checksum"] = hashlib.sha256(canonical_json(obj).encode()).hexdigest()
    state.write_text(json.dumps(obj))
    report = cgr.compare_runs([tmp_path / "a", tmp_path / "b"])
    assert report["runs"][0]["aggregate"]["state_available"] is True
    assert report["runs"][0]["aggregate"]["final_state"] is None


def test_corrupt_wal_is_reported(tmp_path):
    root = tmp_path / "a"; root.mkdir()
    state = root / "state.json"; wal = root / "conversation.jsonl"
    manager = ConversationManager(system_message={"role": "system", "content": "x"},
                                   state_path=state, journal_path=wal)
    manager._persist()
    wal.touch()
    (root / "save.sav").write_bytes(b"save")
    cp = build_checkpoint(checkpoint_path=root / "save.sav", artifact_root=root,
                          conversation_state_path=state, turn=1, username="myagent",
                          run_id="a", player_id=0, config_digest="x", strategy_memory={})
    write_checkpoint(root / "recovery.json", cp)
    wal.write_text("{bad}\n")
    write_run(tmp_path / "b")
    report = cgr.compare_runs([root, tmp_path / "b"])
    assert report["runs"][0]["integrity"]["recovery"]["wal_valid"] is False
    assert any("WAL invalid" in f for f in report["flags"])


def test_production_layout_checkpoint_state_and_root_wal_are_verified(tmp_path):
    root = tmp_path / "a"; (root / "checkpoints").mkdir(parents=True)
    state = root / "checkpoints" / "conversation-state-turn-1.json"
    wal = root / "conversation.jsonl"
    manager = ConversationManager(system_message={"role": "system", "content": "x"},
                                   state_path=state, journal_path=wal)
    manager._persist()
    wal.touch()
    (root / "save.sav").write_bytes(b"save")
    cp = build_checkpoint(checkpoint_path=root / "save.sav", artifact_root=root,
                          conversation_state_path=state, turn=1, username="myagent",
                          run_id="a", player_id=0, config_digest="x", strategy_memory={})
    write_checkpoint(root / "recovery.json", cp)
    write_run(tmp_path / "b")
    report = cgr.compare_runs([root, tmp_path / "b"])
    recovery = report["runs"][0]["integrity"]["recovery"]
    assert recovery["wal_present"] is True
    assert recovery["wal_valid"] is True
    assert recovery["state_wal_relationship"] == "not_proven"
