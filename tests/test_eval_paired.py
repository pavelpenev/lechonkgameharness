# SPDX-License-Identifier: MIT
"""Tests for lechonk.eval.paired — the WP3 paired-run plumbing
(design-plan-freeciv-v1.1.md §4 line 56 + §3 line 45).

Covers: the predeclared paired contract (identity + hash), the arm
manifest (decision-8 fields + the paired extras), the decision-8 drift
policy (match / re-probe / persistent mismatch → drift-suspect), the
early-terminal reporting, the ScriptedComparatorClient (the comparator
arm behind the core-defined ModelClient protocol, over the runner's
fake session), the RecordingClient (the per-turn summaries), and the
eval package's CivRealm-free purity (subprocess import probe + source
scan, mirroring tests/test_core_purity.py).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lechonk.core.ledger import TranscriptLedger
from lechonk.core.model_client import ModelResponse, TurnPrompt
from lechonk.core.schemas import (
    ActionBatch,
    ActionItem,
    RunManifest,
    RunManifestRecord,
)
from lechonk.eval import paired

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "src" / "lechonk" / "eval"

CHECKPOINT = {
    "path": "/repo/artifacts/checkpoints/myagent_T5_2026-10-24-17_32.sav.xz",
    "save_name": "myagent_T5_2026-10-24-17_32",
    "sha256": "sha256:" + "ab" * 32,
    "turn": 5,
    "username": "myagent",
    "exported_by": "lechonk.eval.checkpoint.export_turn_save",
}


def _contract():
    return paired.build_paired_contract(
        arm_turns=3, username="myagent", horizon_turn=7, aifill=2,
        ruleset="classic")


# ---------------------------------------------------------------------------
# the predeclared paired contract
# ---------------------------------------------------------------------------

def test_contract_is_deterministic_and_hashed():
    c1 = _contract()
    c2 = _contract()
    assert c1 == c2
    assert paired.contract_sha256(c1) == paired.contract_sha256(c2)
    assert paired.contract_sha256(c1).startswith("sha256:")
    # a different contract hashes differently
    c3 = _contract()
    c3["arm_turns"] = 4
    assert paired.contract_sha256(c3) != paired.contract_sha256(c1)


def test_contract_declares_the_paired_semantics():
    c = _contract()
    assert c["arms"] == ["comparator", "model"]
    assert c["arm_order"] == ["comparator", "model"]
    assert c["sequential"] is True
    assert c["username"] == "myagent"
    assert c["horizon_turn"] == 7 and c["arm_turns"] == 3
    assert c["action_budgets"]["max_actions_per_turn"] == 20
    assert c["action_budgets"]["max_api_attempts_per_turn"] == 2
    assert c["action_budgets"]["snapshot_tokens"] == 8000
    assert c["action_budgets"]["delta_tokens"] == 800
    assert c["action_budgets"]["memory_tokens"] == 1500
    assert "observation_format" in c and "candidate_restrictions" in c
    assert "lifecycle" in c and "early_terminal" in c and "drift_policy" in c
    assert c["scenario"] == {"ruleset": "classic", "aifill": 2}


# ---------------------------------------------------------------------------
# the arm manifest
# ---------------------------------------------------------------------------

def _arm_manifest(arm="model", fingerprint="sha256:" + "cd" * 32):
    return paired.build_arm_manifest(
        arm=arm, pair_id="wp3-paired-smoke", contract=_contract(),
        checkpoint=CHECKPOINT, arm_start_ts="2026-10-24T18:00:00+00:00",
        model_fingerprint=fingerprint,
        fingerprint_evidence={"reference": None, "probe_1": fingerprint,
                              "re_probe": None, "status": "match"},
        harness_rev="sha256:" + "ef" * 32,
        seeds={"mapseed": 20261029, "gameseed": 20261029,
               "agentseed": 20261029},
        username="myagent", aifill=2,
        image_digest="sha256:" + "12" * 32, capstring="+Freeciv.Web.Devel-3.3",
        vendored_base_sha="1b144d6", patch_head_sha="36ae456",
        model_id="mistral-large-4" if arm == "model" else None,
        sampling={"temperature": 0.0} if arm == "model" else {},
        prompt_hashes={"system_v1": "sha256:x"} if arm == "model" else {},
        enforcement_decision=None, action_batch_schema_hash=None,
        run_id=f"wp3-paired-smoke/{arm}")


def test_arm_manifest_decision8_fields():
    m = _arm_manifest()
    dumped = m.model_dump(mode="json")
    for field in ("run_id", "image_digest", "capstring",
                  "vendored_base_sha", "patch_head_sha", "harness_revision",
                  "seeds", "ruleset", "server_settings", "model_id",
                  "sampling", "prompt_hashes", "config_after_load",
                  "checkpoint_checksums", "model_fingerprint",
                  "arm_start_ts"):
        assert dumped.get(field) is not None, field
    # the checkpoint selection + checksums (decision 8)
    assert dumped["checkpoint_checksums"] == {"start": CHECKPOINT["sha256"]}
    cal = dumped["config_after_load"]
    assert cal["checkpoint_loaded"] is True
    assert cal["checkpoint_sha256"] == CHECKPOINT["sha256"]
    assert cal["checkpoint_save_name"] == CHECKPOINT["save_name"]
    assert cal["checkpoint_turn"] == 5
    assert cal["fc_args"]["debug.load_game"] == CHECKPOINT["save_name"]
    # the paired extras (forward-compat, outside the frozen set)
    assert dumped["paired_pair_id"] == "wp3-paired-smoke"
    assert dumped["paired_arm"] == "model"
    assert dumped["paired_contract"] == _contract()
    assert dumped["paired_contract_sha256"] == paired.contract_sha256(
        _contract())
    assert dumped["fingerprint_evidence"]["status"] == "match"


def test_arm_manifest_contract_identical_across_arms():
    """The predeclared contract is recorded IDENTICALLY in both arms'
    manifests (the paired contract, §3 line 45)."""
    mc = _arm_manifest(arm="comparator", fingerprint=None)
    mm = _arm_manifest(arm="model")
    assert mc.paired_contract == mm.paired_contract
    assert mc.paired_contract_sha256 == mm.paired_contract_sha256
    assert mc.checkpoint_checksums == mm.checkpoint_checksums
    assert mc.paired_username == mm.paired_username == "myagent"


def test_read_arm_manifest(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(transcript)
    ledger.append(RunManifestRecord(manifest=_arm_manifest()))
    ledger.close()
    read = paired.read_arm_manifest(transcript)
    assert read.run_id == "wp3-paired-smoke/model"
    assert read.checkpoint_checksums == {"start": CHECKPOINT["sha256"]}


def test_read_arm_manifest_rejects_non_manifest_first(tmp_path):
    from lechonk.core.schemas import TurnStartRecord
    transcript = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(transcript)
    ledger.append(TurnStartRecord(turn=1))
    ledger.close()
    with pytest.raises(ValueError, match="run_manifest"):
        paired.read_arm_manifest(transcript)


# ---------------------------------------------------------------------------
# the decision-8 drift policy
# ---------------------------------------------------------------------------

def test_drift_policy_match():
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None, "model": "sha256:a"},
        reference="sha256:a")
    assert d["drift_suspect"] is False
    assert d["reproducibility_claim"] is True
    assert d["per_arm"]["model"]["status"] == "match"
    assert d["per_arm"]["comparator"]["status"] == "no_model"
    assert d["both_arms_reported_individually"] is True


def test_drift_policy_reprobe_then_match():
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None, "model": "sha256:b"},
        reference="sha256:a",
        re_probes={"model": "sha256:a"})
    assert d["per_arm"]["model"]["status"] == "match_after_reprobe"
    assert d["drift_suspect"] is False
    assert d["reproducibility_claim"] is True


def test_drift_policy_persistent_mismatch_marks_drift_suspect():
    """Decision 8: a persistent mismatch (after the ONE re-probe) marks
    the pair drift-suspect — excluded from reproducibility claims, both
    arms still reported individually."""
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None, "model": "sha256:b"},
        reference="sha256:a",
        re_probes={"model": "sha256:c"})
    assert d["per_arm"]["model"]["status"] == "mismatch_persistent"
    assert d["drift_suspect"] is True
    assert d["reproducibility_claim"] is False
    assert d["both_arms_reported_individually"] is True
    assert "canary" in d["note"]


def test_drift_policy_no_reference():
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None, "model": "sha256:a"},
        reference=None)
    assert d["reproducibility_claim"] is False
    assert d["drift_suspect"] is False
    # no reference: the fingerprint is recorded, no drift claim either way
    assert d["per_arm"]["model"]["status"] == "no_reference"


# ---------------------------------------------------------------------------
# the decision-8 drift policy, FW-21 (K=3 sample hash sets)
# ---------------------------------------------------------------------------

def test_fingerprint_match_helper_set_intersection():
    """FW-21: match = non-empty intersection of the two hash sets (a
    canary rule, not an equivalence relation)."""
    assert paired.fingerprint_match("sha256:a|sha256:b", "sha256:b|sha256:c")
    assert paired.fingerprint_match("sha256:a", "sha256:a")
    assert not paired.fingerprint_match("sha256:a", "sha256:b")
    assert not paired.fingerprint_match(None, "sha256:a")
    assert not paired.fingerprint_match("sha256:a", None)
    assert not paired.fingerprint_match(None, None)


def test_drift_policy_intersecting_sample_sets_match():
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None,
                          "model": "sha256:a|sha256:b"},
        reference="sha256:b|sha256:c")
    assert d["per_arm"]["model"]["status"] == "match"
    assert d["drift_suspect"] is False
    assert d["reproducibility_claim"] is True
    assert d["per_arm"]["model"]["fingerprint_hashes"] == [
        "sha256:a", "sha256:b"]
    assert d["per_arm"]["model"]["reference_hashes"] == [
        "sha256:b", "sha256:c"]


def test_drift_policy_disjoint_sets_reprobe_then_drift_suspect():
    """FW-21: disjoint sample sets → mismatch → ONE re-probe → an
    intersecting re-probe matches, a still-disjoint re-probe is a
    persistent mismatch (drift-suspect)."""
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None,
                          "model": "sha256:a|sha256:b"},
        reference="sha256:c|sha256:d",
        re_probes={"model": "sha256:e|sha256:c"})
    assert d["per_arm"]["model"]["status"] == "match_after_reprobe"
    assert d["per_arm"]["model"]["re_probe_hashes"] == [
        "sha256:c", "sha256:e"]
    assert d["drift_suspect"] is False
    assert d["reproducibility_claim"] is True
    d2 = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None,
                          "model": "sha256:a|sha256:b"},
        reference="sha256:c|sha256:d",
        re_probes={"model": "sha256:e|sha256:f"})
    assert d2["per_arm"]["model"]["status"] == "mismatch_persistent"
    assert d2["drift_suspect"] is True
    assert d2["reproducibility_claim"] is False
    assert d2["both_arms_reported_individually"] is True


def test_drift_policy_legacy_scalar_reference_fallback():
    """FW-21: a legacy scalar reference (a single hash, the pre-FW-21
    encoding) loads via the documented one-element-set fallback and
    matches an arm sample set containing that hash."""
    legacy = "sha256:" + "ab" * 32
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None,
                          "model": legacy + "|sha256:" + "cd" * 32},
        reference=legacy)
    assert d["per_arm"]["model"]["status"] == "match"
    assert d["per_arm"]["model"]["fingerprint_hashes"] == [
        legacy, "sha256:" + "cd" * 32]
    assert d["per_arm"]["model"]["reference_hashes"] == [legacy]
    assert d["drift_suspect"] is False
    # a legacy scalar reference that no arm sample shares is a mismatch
    d2 = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None,
                          "model": "sha256:" + "ef" * 32},
        reference=legacy)
    assert d2["per_arm"]["model"]["status"] == "mismatch"


def test_drift_policy_between_arm_sets_use_the_same_canary_rule():
    """FW-21: between-arm comparison uses the same intersection rule —
    model arms whose sample sets share a hash do not drift-suspect each
    other; disjoint sets do."""
    d = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": "sha256:a|sha256:b",
                          "model": "sha256:b|sha256:c"},
        reference=None)
    assert d["between_arm_mismatch"] is False
    assert d["drift_suspect"] is False
    d2 = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": "sha256:a",
                          "model": "sha256:b"},
        reference=None)
    assert d2["between_arm_mismatch"] is True
    assert d2["drift_suspect"] is True


# ---------------------------------------------------------------------------
# early-terminal reporting
# ---------------------------------------------------------------------------

def test_early_terminal_report_none_at_horizon():
    assert paired.early_terminal_report("COMPLETED", "horizon reached", 3, 3) is None


def test_early_terminal_report_on_early_stop():
    r = paired.early_terminal_report("TRUNCATED", "infrastructure failure: x", 1, 3)
    assert r is not None
    assert "TRUNCATED" in r and "1/3" in r and "not silently dropped" in r
    r2 = paired.early_terminal_report("DEFEATED", "my player is defeated", 2, 3)
    assert "DEFEATED" in r2 and "2/3" in r2


def test_early_terminal_report_terminal_at_horizon_is_not_early():
    """A VICTORY/DEFEATED on the horizon turn is a normal terminal, not
    an early termination: only a stop BEFORE the horizon is early."""
    assert paired.early_terminal_report("VICTORY", "game over mid-batch (defeated=False)", 3, 3) is None
    assert paired.early_terminal_report("DEFEATED", "my player is defeated", 3, 3) is None


# ---------------------------------------------------------------------------
# the scripted comparator client (over the runner's fake session)
# ---------------------------------------------------------------------------

def _fake_session():
    from tests.runner_fixtures import FakeSession, turn_world
    return FakeSession([turn_world(1), turn_world(2), turn_world(3)])


def test_scripted_comparator_client_decides_from_the_session():
    session = _fake_session()
    client = paired.ScriptedComparatorClient(lambda: session)
    assert client.provenance_class == "scripted_default"
    prompt = TurnPrompt(
        call_id="run/t1/call1", turn_id="run/t1",
        memory="mem", snapshot=session_obs_snapshot(session),
        delta="delta", candidates="cands")
    resp = client.complete_turn(prompt)
    assert resp.error is None
    assert resp.batch is not None
    assert isinstance(resp.batch, ActionBatch)
    # the batch references candidate IDs from the offered table only
    table = session.build_candidate_table()
    offered = table.by_id()
    assert resp.batch.actions
    for item in resp.batch.actions:
        assert isinstance(item, ActionItem)
        assert item.ref in offered
    assert resp.batch.finish_requested is True
    assert resp.fingerprint == "n/a:scripted-comparator"
    assert resp.tokens_in == 0 and resp.cost_usd == 0.0
    assert resp.prompt_hash.startswith("sha256:")
    assert client.calls == 1


def test_scripted_comparator_client_repair_is_defensive():
    from lechonk.core.model_client import RepairPrompt
    session = _fake_session()
    client = paired.ScriptedComparatorClient(lambda: session)
    resp = client.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1", memory="m",
        snapshot=session_obs_snapshot(session), delta="d", candidates="c",
        error_class="schema_error"))
    assert resp.error is not None
    assert "repair not applicable" in resp.error


def session_obs_snapshot(session):
    """The rendered snapshot the runner would build for the fake session
    (the client re-derives it from the session itself)."""
    from lechonk.adapter.obs_extract import extract_snapshot
    extraction = extract_snapshot(
        session.obs, session.info, my_player_id=session.player_id,
        ruleset=session.ruleset, action_probs=session.action_probs)
    assert extraction.snapshot is not None
    return extraction.snapshot.render()


# ---------------------------------------------------------------------------
# the recording client
# ---------------------------------------------------------------------------

class _InnerClient:
    def __init__(self, provenance_class=None):
        if provenance_class is not None:
            # set only when declared (the runner's getattr default is
            # "model" when the attribute is absent)
            self.provenance_class = provenance_class
        self.calls = 0

    def complete_turn(self, prompt):
        self.calls += 1
        return ModelResponse(call_id=prompt.call_id, prompt_hash="ph",
                             batch=ActionBatch(turn_id=prompt.turn_id),
                             latency_ms=5, tokens_in=10, tokens_out=20,
                             cost_usd=0.001, fingerprint="fp")

    def complete_repair(self, prompt):
        return ModelResponse(call_id=prompt.call_id, prompt_hash="ph",
                             error="inner repair", fingerprint="fp")


def test_recording_client_records_turn_summaries(tmp_path):
    session = _fake_session()
    snapshot_json = session_obs_snapshot(session)
    inner = _InnerClient(provenance_class="scripted_default")
    turns_path = tmp_path / "turns.jsonl"
    rec = paired.RecordingClient(inner, turns_path)
    assert rec.provenance_class == "scripted_default"  # passthrough
    prompt = TurnPrompt(call_id="r/t1/call1", turn_id="r/t1", memory="m",
                        snapshot=snapshot_json, delta="d", candidates="c")
    resp = rec.complete_turn(prompt)
    assert resp.batch is not None
    rec.close()
    lines = [json.loads(l) for l in
             turns_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    s = lines[0]
    assert s["turn"] == 1 and s["turn_id"] == "r/t1"
    assert s["cities_own"] >= 1 and s["units_own"] >= 1
    assert s["score"] is not None and s["gold"] is not None
    assert rec.summaries() == lines


def test_recording_client_provenance_default_is_model(tmp_path):
    inner = _InnerClient()  # no provenance_class attribute
    rec = paired.RecordingClient(inner, tmp_path / "turns.jsonl")
    assert rec.provenance_class == "model"
    assert rec.inner is inner


def test_turn_summary_from_snapshot_json_garbage_returns_none():
    assert paired.turn_summary_from_snapshot_json("not json") is None


# ---------------------------------------------------------------------------
# the privileged state summary (verification only)
# ---------------------------------------------------------------------------

def test_state_summary_and_match():
    obs = {
        "player": {1: {"score": 42, "gold": 30, "techs_researched": 1,
                       "researching": "Alphabet"}},
        "city": {7: {"owner": 1, "size": 3}, 8: {"owner": 2, "size": 2}},
        "unit": {10: {"owner": 1}, 11: {"owner": 1}, 20: {"owner": 2}},
    }
    s = paired.state_summary_from_obs(obs, 1)
    assert s["score"] == 42 and s["gold"] == 30
    assert s["cities_own"] == [{"id": 7, "size": 3}]
    assert s["units_own"] == [10, 11]
    assert paired.states_match(s, dict(s))
    assert not paired.states_match(s, {**s, "score": 43})


def test_states_match_ignores_record_metadata():
    """The snapshot-based summaries carry per-record metadata (turn_id/ts)
    that differs by construction between the export-time capture (stamped
    None) and an arm's first-turn record (stamped with the arm's run id and
    a timestamp) — only the canonical state fields participate in the
    load-identity comparison (live-verified 2026-10-24: the comparator arm
    loaded the identical checkpoint with all 10 state fields equal)."""
    export_side = {"turn": 5, "turn_id": None, "ts": None,
                   "score": 1, "gold": 54, "techs_researched": 1,
                   "research_target": "Alphabet", "research_turns_left": 24,
                   "cities_own": 1, "units_own": 5,
                   "cities": [{"id": 111, "name": "Niamey", "size": 1,
                               "production": "Warriors"}],
                   "units": [{"id": 103, "type": "Settlers", "x": 58,
                              "y": 24, "activity": "idle"}]}
    arm_side = {**export_side,
                "turn_id": "wp3-paired-smoke/comparator/t5",
                "ts": "2026-10-24T17:25:29Z"}
    assert paired.states_match(export_side, arm_side)
    assert paired.states_match(arm_side, export_side)
    # A state-field difference still fails the comparison.
    assert not paired.states_match(export_side, {**arm_side, "gold": 55})
    assert not paired.states_match(export_side,
                                   {**arm_side, "units_own": 6})


# ---------------------------------------------------------------------------
# the pair manifest
# ---------------------------------------------------------------------------

def test_assemble_pair_manifest_reports_both_arms_individually():
    contract = _contract()
    drift = paired.resolve_fingerprints(
        arm_fingerprints={"comparator": None, "model": "sha256:a"},
        reference="sha256:a")
    arms = {
        "comparator": {"arm": "comparator", "outcome": "COMPLETED",
                       "turns_completed": 3, "early_terminal": False,
                       "early_terminal_report": None,
                       "model_fingerprint": None},
        "model": {"arm": "model", "outcome": "TRUNCATED",
                  "reason": "infrastructure failure: timeout",
                  "turns_completed": 1, "early_terminal": True,
                  "early_terminal_report": "arm terminated early: ...",
                  "model_fingerprint": "sha256:a"},
    }
    pm = paired.assemble_pair_manifest(
        pair_id="wp3-paired-smoke", contract=contract, checkpoint=CHECKPOINT,
        arms=arms, drift=drift, checks={"x": {"pass": True, "detail": ""}},
        verdict="PASS")
    assert pm["pair_id"] == "wp3-paired-smoke"
    assert pm["paired_contract"] == contract
    assert pm["paired_contract_sha256"] == paired.contract_sha256(contract)
    assert pm["checkpoint"]["sha256"] == CHECKPOINT["sha256"]
    # both arms reported individually — even the TRUNCATED one (the pair
    # is never silently dropped)
    assert set(pm["arms"]) == {"comparator", "model"}
    assert pm["arms"]["model"]["outcome"] == "TRUNCATED"
    assert pm["arms"]["model"]["early_terminal_report"]
    assert pm["sequential"]["order"] == ["comparator", "model"]
    assert pm["drift"] == drift
    assert pm["reproducibility_claim"] is True
    assert pm["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# purity: eval never imports civrealm
# ---------------------------------------------------------------------------

IMPORT_PROBE = (
    "import sys\n"
    "import lechonk\n"
    "import lechonk.eval\n"
    "import lechonk.eval.checkpoint\n"
    "import lechonk.eval.paired\n"
    "import lechonk.eval.replay\n"
    "bad = sorted(m for m in sys.modules\n"
    "             if m == 'civrealm' or m.startswith('civrealm.'))\n"
    "print('CIVREALM_MODULES:', bad)\n"
    "sys.exit(1 if bad else 0)\n"
)


def test_importing_lechonk_eval_never_imports_civrealm():
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert proc.returncode == 0, (
        f"lechonk.eval transitively imported civrealm:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "CIVREALM_MODULES: []" in proc.stdout


def test_eval_sources_never_import_civrealm():
    import_re = re.compile(r"^\s*(import|from)\s+civrealm\b", re.MULTILINE)
    offenders = []
    for path in sorted(EVAL_DIR.rglob("*.py")):
        if import_re.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == [], f"eval imports civrealm: {offenders}"


def test_eval_module_level_adapter_imports_are_none():
    """The only adapter touchpoint is a lazy, function-level import of the
    CivRealm-free obs_extract (the runner's lazy-import discipline)."""
    import ast
    offenders = []
    for path in sorted(EVAL_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("lechonk.adapter") or \
                            alias.name == "civrealm":
                        offenders.append(f"{path}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("lechonk.adapter") or \
                        (node.module or "") == "civrealm":
                    offenders.append(f"{path}:{node.lineno} from {node.module}")
    assert offenders == [], f"module-level adapter imports: {offenders}"


# ---------------------------------------------------------------------------
# FW-18: the strict canonical-JSON rule in paired (the FW-09 rule)
# ---------------------------------------------------------------------------

def test_paired_canonical_json_nan_raises_deterministically():
    # C9b: NaN/Infinity are rejected (allow_nan=False) — a deterministic
    # ValueError, never a bare NaN/Infinity token (invalid JSON).
    with pytest.raises(ValueError):
        paired.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        paired.canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        paired.canonical_json({"x": float("-inf")})


def test_paired_canonical_json_lone_surrogate_escaped_not_unicode_encode_error():
    # C9b: a lone surrogate is escaped as \udXXX (valid JSON, round-trips
    # through json.loads); its utf-8 encoding can never fail with
    # UnicodeEncodeError.
    text = paired.canonical_json({"s": "a\ud800b"})  # must not raise
    assert "\\ud800" in text
    assert json.loads(text) == {"s": "a\ud800b"}
    text.encode("utf-8")


def test_paired_canonical_json_clean_payload_byte_identical_to_pre_fw18():
    # Regression: for clean (non-NaN, surrogate-free) payloads the output
    # is byte-identical to the pre-FW-18 json.dumps construction
    # (sort_keys, compact separators, ensure_ascii=False).
    payload = {"b": 1, "a": [1, 2, {"c": "café — ü"}], "d": None, "e": True}
    assert paired.canonical_json(payload) == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
