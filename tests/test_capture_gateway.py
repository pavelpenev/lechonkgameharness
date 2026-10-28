import json
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import uuid
import time
from pathlib import Path

import pytest

from lechonk.capture_gateway import CaptureGateway, EVENT_VERSION


def read_events(path: Path):
    return [json.loads(line) for line in
            (path / "capture-events.jsonl").read_text().splitlines()]


def test_capture_manifest_and_session_ready_are_atomic_and_observable(tmp_path):
    target = tmp_path / "capture"
    gateway = CaptureGateway.create(target, run_id="run-1",
                                    expected_username="agent-abc")
    gateway.session_ready({"event": "session_ready", "port": 6551,
                           "actual_player_roster": [{"name": "agent-abc"}]})
    raw_reasoning = "keep every newline\nwithout truncation"
    lifecycle = {
        "physical_call_id": "physical-1",
        "logical_call_id": "logical-1",
        "raw_delta": {"reasoning": raw_reasoning,
                       "answer": '{"actions": []}'},
    }
    gateway.offer_trace("model_lifecycle", lifecycle)
    lifecycle["raw_delta"]["reasoning"] = "caller mutation after enqueue"
    gateway.finish(status="finished")

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["capture_id"]
    assert manifest["run_id"] == "run-1"
    assert manifest["expected_username"] == "agent-abc"
    assert manifest["session_ready"] is True
    assert manifest["session_port"] == 6551
    assert manifest["complete"] is True
    manifest_check = subprocess.run([
        "node", "-e",
        "const fs=require('fs'); const p=require('./scripts/capture_presenter.js'); const m=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); p.validateManifest(m); if(m.schema_version!=='capture-manifest-v1'||m.identity_assurance!=='provisional_username_witness'||m.production_ready!==false||m.nonce_evidence!=='not_provided') process.exit(1);",
        str(target / "manifest.json")], check=False)
    assert manifest_check.returncode == 0
    events = read_events(target)
    assert events
    assert all(event["version"] == EVENT_VERSION for event in events)
    assert all(event["schema_version"] == EVENT_VERSION for event in events)
    assert [event["capture_seq"] for event in events] == list(
        range(1, len(events) + 1))
    model = next(event for event in events
                  if event["event_kind"] == "model_lifecycle")
    assert model["payload"]["raw_delta"]["reasoning"] == raw_reasoning

    # Exercise the actual generated event envelope through the same
    # JavaScript runtime used by the presenter, rather than a hand-built
    # fixture.  The presenter requires schema_version and run_id.
    check = subprocess.run([
        "node", "-e",
        "const fs=require('fs'); const e=JSON.parse(fs.readFileSync(process.argv[1],'utf8').split('\\n')[0]); if(e.schema_version!=='capture-event-v1'||!e.run_id||!e.capture_id||!e.kind||!e.payload) process.exit(1);",
        str(target / "capture-events.jsonl")], check=False)
    assert check.returncode == 0


def test_capture_directory_never_overwrites_existing(tmp_path):
    target = tmp_path / "capture"
    target.mkdir()
    with pytest.raises(FileExistsError):
        CaptureGateway.create(target, run_id="run-1",
                              expected_username="agent-abc")


def test_trace_overflow_is_explicit_and_controls_are_separate(tmp_path):
    gateway = CaptureGateway.create(
        tmp_path / "capture", run_id="run-1", expected_username="agent-abc",
        trace_capacity=1, control_capacity=2)
    assert gateway.offer_trace("trace", {"n": 1})
    assert not gateway.offer_trace("trace", {"n": 2})
    gateway.offer_control("control", {"n": 3})
    gateway.finish(status="finished")
    manifest = json.loads((tmp_path / "capture" / "manifest.json").read_text())
    assert manifest["dropped_trace"] == 1
    assert manifest["partial"] is True
    assert any(event["event_kind"] == "control"
               for event in read_events(tmp_path / "capture"))


def test_control_receipt_returns_exact_written_sequence(tmp_path):
    target = tmp_path / "capture"
    gateway = CaptureGateway.create(target, run_id="run-1",
                                    expected_username="agent-abc",
                                    control_capacity=8)
    receipt = gateway.offer_control_receipt("checkpoint", {"token": "one"})
    assert isinstance(receipt, int)
    gateway.finish(status="finished")
    events = read_events(target)
    matching = next(event for event in events
                    if event["event_kind"] == "checkpoint")
    assert matching["capture_seq"] == receipt
    assert gateway.offer_control_receipt("after_close", {}) is None


def test_control_receipts_are_unique_under_concurrent_offers(tmp_path):
    target = tmp_path / "capture"
    gateway = CaptureGateway.create(target, run_id="run-1",
                                    expected_username="agent-abc",
                                    control_capacity=32)
    receipts = []
    lock = threading.Lock()

    def offer(index):
        receipt = gateway.offer_control_receipt("checkpoint", {"index": index})
        if receipt is not None:
            with lock:
                receipts.append(receipt)

    threads = [threading.Thread(target=offer, args=(index,))
               for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    gateway.finish(status="finished")
    events = read_events(target)
    written = {event["capture_seq"] for event in events
               if event["event_kind"] == "checkpoint"}
    assert len(receipts) == 16
    assert len(set(receipts)) == 16
    assert set(receipts) == written


def test_finish_is_idempotent_under_concurrent_callers(tmp_path):
    gateway = CaptureGateway.create(tmp_path / "capture", run_id="run-1",
                                    expected_username="agent-abc")
    threads = [threading.Thread(target=gateway.finish, kwargs={"status": "finished"})
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    events = read_events(tmp_path / "capture")
    assert sum(event["kind"] == "capture_finished" for event in events) == 1


def test_final_manifest_cannot_be_overwritten_by_concurrent_snapshot_writer(tmp_path):
    gateway = CaptureGateway.create(tmp_path / "capture", run_id="run-1",
                                    expected_username="agent-abc")
    workers = [threading.Thread(target=gateway._write_manifest)
               for _ in range(20)]
    for worker in workers:
        worker.start()
    gateway.finish(status="finished")
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    manifest = json.loads((tmp_path / "capture" / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["status"] == "finished"


def test_runner_progress_callback_binds_real_session_ready_shape(tmp_path, monkeypatch):
    run_id = f"capture-progress-{uuid.uuid4().hex}"
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.setattr(sys, "argv", ["wp2_model_run.py", "--run-id", run_id,
                                        "--prompt-profile", "gameplay_v3",
                                        "--context-policy", "conversation_v1"])
    script = Path(__file__).resolve().parents[1] / "scripts" / "wp2_model_run.py"
    spec = importlib.util.spec_from_file_location(
        f"wp2_capture_progress_{uuid.uuid4().hex}", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        gateway = CaptureGateway.create(
            "capture-relative", run_id=run_id,
            expected_username=module.FREECIV_USERNAME)
        module._CAPTURE_GATEWAY = gateway
        other_cwd = tmp_path / "runner-workdir"
        other_cwd.mkdir()
        os.chdir(other_cwd)
        module.progress({
            "event": "session_ready", "port": 6007,
            "actual_player_roster": [{"username": module.FREECIV_USERNAME}],
            "aifill": 3,
        })
        gateway.finish(status="finished")
    finally:
        os.chdir(old_cwd)
    manifest = json.loads((tmp_path / "capture-relative" / "manifest.json").read_text())
    assert manifest["session_ready"] is True
    assert manifest["session_port"] == 6007
    assert manifest["expected_username"] == module.FREECIV_USERNAME
    shutil.rmtree(module.RUNDIR, ignore_errors=True)


def test_subprocess_normal_finish_is_complete_but_atexit_process_exit_is_partial(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = (
        "import atexit, sys; from lechonk.capture_gateway import CaptureGateway; "
        "g=CaptureGateway.create(sys.argv[1], run_id='subprocess-run', expected_username='agent'); "
        "atexit.register(lambda: g.finish(status='process_exit')); "
        "g.offer_control('progress', {'event':'test'}); "
        "g.finish(status=sys.argv[2]) if sys.argv[2] != 'process_exit' else None"
    )
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    normal_dir = tmp_path / "normal"
    abnormal_dir = tmp_path / "abnormal"
    subprocess.run([sys.executable, "-c", script, str(normal_dir), "finished"],
                   cwd=root, env=env, check=True)
    subprocess.run([sys.executable, "-c", script, str(abnormal_dir), "process_exit"],
                   cwd=root, env=env, check=True)
    normal = json.loads((normal_dir / "manifest.json").read_text())
    abnormal = json.loads((abnormal_dir / "manifest.json").read_text())
    assert normal["complete"] is True
    assert normal["partial"] is False
    assert abnormal["complete"] is False
    assert abnormal["partial"] is True
    assert abnormal["status"] == "process_exit"
    assert any(json.loads(line)["kind"] == "capture_finished"
               for line in (abnormal_dir / "capture-events.jsonl").read_text().splitlines())
