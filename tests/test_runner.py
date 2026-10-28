# SPDX-License-Identifier: MIT
"""Runner tests (WP2): the turn pipeline, state machine, supervisor and
watchdog, driven offline by a fake session (real adapter read-backs) and a
fake model client — including the fault injection the WP2 acceptance
requires (timeout, malformed output, invalid action → the right repair
class or the degraded fallback; batch abort-group semantics; ledger
durability; the deadline-aware stall watchdog (D7/D14: the legal-wait
and genuine-stall firing rule, the generation guard).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lechonk.adapter.session import StepOutcome
from lechonk.core.ledger import TranscriptLedger, load_transcript
from lechonk.core.metrics import commitment_board, compute_metrics
from lechonk.core.model_client import ModelResponse
from lechonk.core.runner import (
    ACTING_DEADLINE_STOP_EVENT,
    ENTITY_ID_RE,
    GameRunner,
    RunState,
    RunnerConfig,
    StallWatchdog,
    StateDeadlineExceeded,
    _dispatch_cancelled_payload,
    _dispatch_intent_payload,
    _dispatch_result_detail,
    _QuarantineFailStop,
    _StandingOrder,
    _unresolved_dispatch_intents,
    mark_armed_in_contact,
)
from lechonk.core.schemas import (
    ActionBatch,
    ActionResultRecord,
    Candidate,
    CandidateTable,
    CandidateTableCapExceeded,
    CandidatesRefRecord,
    CityView,
    CommitmentRecord,
    MapView,
    MemoryUpdate,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    PlayerView,
    RunManifest,
    RunManifestRecord,
    Snapshot,
    SnapshotRefRecord,
    TurnStartRecord,
    UnitView,
)
from lechonk.model.prompts import PromptBudgetExceeded
from lechonk.model.mistral_client import ProviderCapacityError

from tests.lifecycle_fixtures import LifecycleSession
from tests.fog_fixtures import (
    ENEMY,
    MY_PLAYER,
    NUM_IMPROVEMENTS,
    SETTLERS,
    VUT_IMPROVEMENT,
    VUT_UTYPE,
    WARRIORS,
    foreign_city,
    own_city,
)
from tests.runner_fixtures import (
    FakeClock,
    FakeModelClient,
    FakeSession,
    advancing_response,
    error_response,
    make_runner,
    malformed_response,
    ok_response,
    turn_world,
)

REPO = Path(__file__).resolve().parents[1]

# The P11 ledger audit (scripts/wp3_p11_audit.py), loaded the same way
# tests/test_lifecycle_p11.py loads it — for the W3-9 audit-side pins.
_audit_spec = importlib.util.spec_from_file_location(
    "wp3_p11_audit", REPO / "scripts" / "wp3_p11_audit.py")
wp3_p11_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(wp3_p11_audit)


def _records(path):
    return load_transcript(path).records


def _actions(records):
    return [r for r in records if isinstance(r, ActionResultRecord)]


def _by_kind(records, cls):
    return [r for r in records if isinstance(r, cls)]


def test_provider_capacity_fails_closed_before_repair_or_dispatch(tmp_path):
    """A backend-capacity 429 truncates at the start checkpoint.

    It must not be converted into a semantic repair or a degraded default
    action batch, and the durable turn-start record remains the recovery
    boundary.
    """
    class CapacityClient(FakeModelClient):
        def complete_turn(self, prompt):
            self.turn_prompts.append(prompt)
            self.calls += 1
            raise ProviderCapacityError("backend capacity", retry_after="5")

        def complete_repair(self, prompt):
            self.repair_prompts.append(prompt)
            raise AssertionError("capacity failure must not trigger repair")

    session = FakeSession([turn_world(1)], final_terminated=False,
                          final_truncated=False)
    client = CapacityClient()
    runner = make_runner(tmp_path, client=client, session=session, max_turns=1,
                         config_overrides={"context_policy": "conversation_v1"})

    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert client.calls == 1
    assert client.repair_prompts == []
    assert [action for action, _deadline in session.calls if action is not None] == []
    records = _records(runner.ledger.path)
    assert len(_by_kind(records, TurnStartRecord)) == 1
    assert _actions(records) == []
    assert "capacity" in result.reason.lower()


class _GameOverMidBatchSession(FakeSession):
    """A FakeSession whose named mutation ends the game (terminated=True)
    as its step outcome — the game-over-mid-batch injection."""

    def __init__(self, worlds, *, terminate_on):
        super().__init__(worlds)
        self._terminate_on = terminate_on

    def step(self, action, *, deadline=None):
        if action is not None and tuple(action) == self._terminate_on:
            outcome = super().step(action, deadline=deadline)
            return replace(outcome, terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


# ---------------------------------------------------------------------------
# happy path: a model-driven run to the horizon
# ---------------------------------------------------------------------------


def test_runner_happy_path_model_driven_run(tmp_path):
    """Two model-driven turns: the batch executes with read-back-verified
    effects, every state transition and action is ledgered, the memory
    update rides the batch, and the run completes at the horizon."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e",
                     "tech:cur_player:research_tech_alphabet_0"],
                    turn_id="t1",
                    commitments=[("research", "Alphabet", 5)],
                    threats=["unit:20"], notes="watch the enemy"),
        ok_response(["unit:11:sentry"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "COMPLETED"
    assert result.turns_completed == 2
    assert session.started and session.closed
    # The pipeline owns the finish: exactly one phase_done per turn.
    finishes = [c for c in session.calls if c[0] is None]
    assert len(finishes) == 2

    records = _records(runner.ledger.path)
    # Record-kind coverage per turn.
    assert isinstance(records[0], RunManifestRecord)
    assert len(_by_kind(records, TurnStartRecord)) == 2
    assert len(_by_kind(records, SnapshotRefRecord)) == 2
    assert len(_by_kind(records, CandidatesRefRecord)) == 2
    assert len(_by_kind(records, ModelCallRecord)) == 2
    # State transitions are ledgered.
    transitions = [r.value for r in _by_kind(records, MetricEventRecord)
                   if r.name == "state_transition"]
    assert "STARTUP->ACTING" in transitions
    assert "ACTING->FINISH_SENT" in transitions
    assert "FINISH_SENT->WAITING" in transitions
    assert "WAITING->NEXT_OWN_PHASE" in transitions
    # The executed actions: read-back verified, model provenance, with the
    # structured descriptors the metrics read.
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    assert {a.ref for a in executed} == {
        "unit:10:goto_e", "tech:cur_player:research_tech_alphabet_0",
        "unit:11:sentry"}
    goto = next(a for a in executed if a.verb == "goto_e")
    assert goto.readback == "verified"
    assert goto.provenance_class == "model"
    assert goto.actor_tile == (1, 1) and goto.affected == "tile:2,1"
    research = next(a for a in executed
                    if a.ref == "tech:cur_player:research_tech_alphabet_0")
    assert research.target == "Alphabet" and research.readback == "verified"
    # The memory update rode the batch: one commitment record carrying the
    # structured commitment + the threat mention (tile resolved from the
    # snapshot: unit 20 sits at (2,2)).
    commitments = _by_kind(records, CommitmentRecord)
    assert len(commitments) == 1
    assert commitments[0].turn == 1
    assert commitments[0].commitments[0].target == "Alphabet"
    assert commitments[0].threats[0].entity_id == "unit:20"
    assert commitments[0].threats[0].tile == (2, 2)
    # The memory advanced post-advance: turn 2's prompt carries it.
    assert "Alphabet" in client.turn_prompts[1].memory
    # The outcome record is last; the metrics were emitted before it.
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "COMPLETED"
    metric_names = {r.name for r in _by_kind(records, MetricEventRecord)
                    if r.name != "state_transition"}
    assert {"participation", "commitment_followthrough_any",
            "reason_but_dont_act"} <= metric_names
    # The metrics report is attached to the result.
    assert result.metrics.participation.denominator == 2
    assert result.metrics.participation.numerator == 2


def test_runner_gameplay_profile_renders_typed_commitments(tmp_path):
    """Gameplay prompts expose typed commitment metadata, while the
    default/legacy profile remains byte-compatible with an explicit legacy
    selection."""
    def typed_response(_prompt):
        response = ok_response(["unit:10:goto_e"], turn_id="t1",
                               commitments=[("research", "Alphabet", 5)])
        for commitment in response.batch.commitments:
            commitment.target_kind = "tech"
            commitment.verb = "research"
        for commitment in response.batch.memory_update.commitments:
            commitment.target_kind = "tech"
            commitment.verb = "research"
        return response

    def run(profile, path):
        path.mkdir()
        client = FakeModelClient([typed_response,
                                  ok_response(["unit:11:sentry"],
                                              turn_id="t2", with_memory=False)])
        overrides = {} if profile is None else {"prompt_profile": profile}
        runner = make_runner(
            path, client=client,
            session=FakeSession([turn_world(1), turn_world(2), turn_world(3)]),
            max_turns=2,
            config_overrides=overrides)
        assert runner.run().outcome == "COMPLETED"
        return client.turn_prompts[1].memory

    legacy_default = run(None, tmp_path / "default")
    legacy_explicit = run("legacy_eval", tmp_path / "explicit")
    gameplay = run("gameplay_v1", tmp_path / "gameplay")

    assert legacy_default == legacy_explicit
    assert "target_kind=" not in legacy_default
    assert " verb=" not in legacy_default
    assert "target_kind=tech" in gameplay
    assert "verb=research" in gameplay


def test_runner_installs_bounded_ruleset_lookup_only_for_gameplay_conversation(
        tmp_path, monkeypatch):
    """The session reference is exposed as a read-only local tool only for
    the interactive gameplay conversation profile."""
    reference = object()

    class LookupSession(FakeSession):
        ruleset_reference = reference

    class LookupClient(FakeModelClient):
        def __init__(self, script):
            super().__init__(script)
            self.lookup = None

        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    seen = []

    def fake_render(ref, query, *, categories=None, limit=10, max_chars=8000):
        seen.append((ref, query, categories, limit, max_chars))
        return {"status": "ok", "query": query, "entries": [{"name": "road"}]}

    monkeypatch.setattr(
        "lechonk.adapter.ruleset_reference.render_search", fake_render)
    worlds = [turn_world(1), turn_world(2)]
    client = LookupClient([ok_response([], turn_id="t1")])
    progress = []
    runner = make_runner(
        tmp_path, client=client, session=LookupSession(worlds), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"},
        progress=progress.append)
    runner.run()

    assert client.lookup is not None
    result = client.lookup("road", categories=["terrain"], limit=100)
    assert result["status"] == "ok"
    assert seen == [(reference, "road", ["terrain"], 8, 4000)]
    result = client.lookup("road")
    assert result["status"] == "ok"
    assert seen[-1] == (reference, "road", None, 8, 4000)
    assert [event["event"] for event in progress
            if event["event"].startswith("ruleset_lookup_")] == [
                "ruleset_lookup_started", "ruleset_lookup_finished",
                "ruleset_lookup_started", "ruleset_lookup_finished"]

    legacy_client = LookupClient([ok_response([], turn_id="t1")])
    legacy_path = tmp_path / "legacy"
    legacy_path.mkdir()
    runner = make_runner(legacy_path, client=legacy_client,
                         session=LookupSession(worlds), max_turns=1)
    runner.run()
    assert legacy_client.lookup is None


def test_runner_installs_ruleset_lookup_for_gameplay_v3(tmp_path):
    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client, session=FakeSession([turn_world(1), turn_world(2)]),
        max_turns=1, config_overrides={
            "context_policy": "conversation_v1", "prompt_profile": "gameplay_v3",
            "recovery_config_digest_schema": "interactive-config-v3",
            "prompt_bundle_id": "gameplay_v3",
            "candidate_surface_policy": "candidate_surface_v2",
            "candidate_surface_cap": 8000,
            "candidate_admission_version": "candidate_surface_v2-r1",
            "compaction_schema": "conversation-compaction-v1",
            "prompt_budget_schema": "gameplay-v3-budget-v1"})
    runner.run()
    assert callable(client.lookup)


def test_runner_installs_ruleset_lookup_for_gameplay_v4(tmp_path):
    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client, session=FakeSession([turn_world(1), turn_world(2)]),
        max_turns=1, config_overrides={
            "context_policy": "conversation_v1", "prompt_profile": "gameplay_v4",
            "recovery_config_digest_schema": "interactive-config-v3",
            "prompt_bundle_id": "gameplay_v4",
            "candidate_surface_policy": "candidate_surface_v2",
            "candidate_surface_cap": 8000,
            "candidate_admission_version": "candidate_surface_v2-r1",
            "compaction_schema": "conversation-compaction-v1",
            "prompt_budget_schema": "gameplay-v4-budget-v1"})
    runner.run()
    assert callable(client.lookup)


def test_runner_ledgers_tool_final_and_repair_as_three_physical_calls(tmp_path):
    class PhysicalClient(FakeModelClient):
        def __init__(self):
            super().__init__([malformed_response(["bad"]),
                              ok_response([], turn_id="t1")])
            self.call_records = []

        def complete_turn(self, prompt):
            response = super().complete_turn(prompt)
            self.call_records.extend([
                {"call_id": prompt.call_id, "kind": "initial",
                 "prompt_hash": "initial", "latency_ms": 1,
                 "tokens_in": 10, "tokens_out": 10, "cost_usd": 0,
                 "fingerprint": "fp"},
                {"call_id": prompt.call_id, "kind": "tool_final",
                 "prompt_hash": "final", "latency_ms": 2,
                 "tokens_in": 20, "tokens_out": 10, "cost_usd": 0,
                 "fingerprint": "fp"},
            ])
            return response

        def complete_repair(self, prompt):
            response = super().complete_repair(prompt)
            self.call_records.append(
                {"call_id": prompt.call_id, "kind": "repair",
                 "prompt_hash": "repair", "latency_ms": 3,
                 "tokens_in": 30, "tokens_out": 10, "cost_usd": 0,
                 "fingerprint": "fp"})
            return response

    client = PhysicalClient()
    runner = make_runner(
        tmp_path, client=client,
        session=FakeSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner.run()
    records = _by_kind(load_transcript(runner.ledger.path).records,
                       ModelCallRecord)
    assert len(records) == 3
    assert [record.attempt for record in records] == [1, 2, 3]


def test_runner_lookup_labels_static_help_non_authoritative(tmp_path, monkeypatch):
    class LookupSession(FakeSession):
        ruleset_reference = None
        ruleset_reference_status = "unavailable: no packet"
        help_reference = object()

    class LookupClient(FakeModelClient):
        def __init__(self, script):
            super().__init__(script)
            self.lookup = None

        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    monkeypatch.setattr(
        "lechonk.adapter.help_reference.render_search",
        lambda ref, query, *, categories=None, limit=10, max_chars=8000: {
            "status": "available",
            "entries": ([] if categories else [{"name": "static road"}]),
        })
    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client,
        session=LookupSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner.run()
    result = client.lookup("road")
    assert result["sources"] == {
        "active_ruleset": "unavailable", "static_help": "available",
        "static_classic": "unavailable"}
    assert result["entries"][0]["authority"] == "static_help_non_authoritative"
    assert client.lookup("road", categories=["terrain"])["entries"] == []


def test_runner_lookup_includes_classic_reference_with_active_precedence(tmp_path):
    from lechonk.adapter.classic_reference import ClassicEntry, ClassicReference

    class LookupSession(FakeSession):
        ruleset_reference = None
        help_reference = None
        classic_reference = ClassicReference(entries=(
            ClassicEntry("unit_archer", "Archer", category="unit"),
            ClassicEntry("road", "Road", category="terrain"),
        ), source_path="classic", tree_sha256="a" * 64)
        classic_reference_status = "available"

    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client,
        session=LookupSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner.run()
    result = client.lookup("what is archer", categories=["unit"], limit=8)
    assert result["status"] == "ok"
    assert result["sources"]["static_classic"] == "available"
    assert result["provenance"]["static_classic"]["content_hash"]
    assert result["entries"][0]["authority"] == (
        "static_classic_non_authoritative")
    assert result["entries"][0]["category"] == "unit"


def test_runner_lookup_includes_generated_manual_as_non_authoritative(
        tmp_path, monkeypatch):
    """The generated manual is static prose and ranks below live rules."""
    from types import ModuleType, SimpleNamespace

    manual = SimpleNamespace(
        status="available", schema_version="freeciv-manual-reference-v1",
        source="generated_freeciv_manual", source_path="manual/reference.json",
        source_sha256="b" * 64, content_hash="c" * 64)

    module = ModuleType("lechonk.adapter.manual_reference")
    def render_search(ref, query, *, categories=None, limit=10, max_chars=8000):
        return {"status": "available", "entries": [
            {"name": "road", "category": "terrain", "text": "manual"},
        ], "source": ref.source, "source_path": ref.source_path,
                "source_sha256": ref.source_sha256,
                "content_hash": ref.content_hash,
                "schema_version": ref.schema_version}
    module.render_search = render_search
    monkeypatch.setitem(__import__("sys").modules,
                        "lechonk.adapter.manual_reference", module)

    class LookupSession(FakeSession):
        ruleset_reference = None
        help_reference = None
        classic_reference = None
        manual_reference = manual
        manual_reference_status = "available"

    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client,
        session=LookupSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner.run()
    result = client.lookup("road", categories=["terrain"], limit=8)
    assert result["sources"]["static_manual"] == "available"
    assert result["entries"][0]["authority"] == (
        "static_manual_non_authoritative")
    assert result["provenance"]["static_manual"]["source_sha256"] == "b" * 64


def test_runner_lookup_keeps_active_and_manual_records_with_same_name(
        tmp_path, monkeypatch):
    """Different sources retain useful same-name records; exact source
    duplicates do not consume additional result slots."""
    from types import ModuleType, SimpleNamespace
    import sys

    manual = SimpleNamespace(status="available", schema_version="manual-v1",
                             source="generated_manual", source_path="manual",
                             source_sha256="d" * 64, content_hash="e" * 64)
    manual_module = ModuleType("lechonk.adapter.manual_reference")
    manual_module.render_search = lambda *args, **kwargs: {
        "status": "available", "entries": [
            {"name": "road", "category": "terrain", "text": "manual detail"},
            {"name": "road", "category": "terrain", "text": "manual detail"},
            {"name": "bridge", "category": "terrain", "text": "other"},
        ]}
    monkeypatch.setitem(sys.modules, "lechonk.adapter.manual_reference",
                        manual_module)

    active = SimpleNamespace(content_hash="f" * 64)
    monkeypatch.setattr(
        "lechonk.adapter.ruleset_reference.render_search",
        lambda *args, **kwargs: {
            "status": "available", "entries": [
                {"name": "road", "category": "terrain", "text": "live"},
            ]})

    class LookupSession(FakeSession):
        ruleset_reference = active
        ruleset_reference_status = "available"
        help_reference = None
        classic_reference = None
        manual_reference = manual
        manual_reference_status = "available"

    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    client = LookupClient([ok_response([], turn_id="t1")])
    runner = make_runner(
        tmp_path, client=client,
        session=LookupSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner.run()
    result = client.lookup("road", categories=["terrain"], limit=2)
    assert [entry["authority"] for entry in result["entries"]] == [
        "active_ruleset", "static_manual_non_authoritative"]
    assert result["entries"][0]["text"] == "live"


def test_runner_checkpoint_normalizes_unavailable_classic_reference(
        tmp_path, monkeypatch):
    from lechonk.adapter.classic_reference import ClassicReference

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    state_path = tmp_path / "conversation.json"
    state_path.write_text("{}\n", encoding="utf-8")
    save_path = checkpoint_dir / "save-1.sav"
    save_path.write_bytes(b"save")
    save_sha = hashlib.sha256(save_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "lechonk.eval.checkpoint.export_turn_save",
        lambda **kwargs: SimpleNamespace(path=str(save_path), sha256=save_sha))

    class CheckpointSession(FakeSession):
        classic_reference = ClassicReference(status="unavailable")
        classic_reference_status = "unavailable: absent"

    runner = make_runner(
        tmp_path, client=FakeModelClient(),
        session=CheckpointSession([turn_world(1), turn_world(2)]), max_turns=1,
        config_overrides={"checkpoint_dir": str(checkpoint_dir),
                          "recovery_manifest_path": str(tmp_path / "recovery.json"),
                          "conversation_state_path": str(state_path)})
    runner._session = CheckpointSession([turn_world(1), turn_world(2)])
    runner._write_start_checkpoint(1)
    manifest = json.loads((tmp_path / "recovery.json").read_text())
    assert manifest["classic_reference_present"] is False
    assert manifest["classic_reference_schema_version"] is None
    assert manifest["classic_ruleset_tree_sha256"] is None


def test_ruleset_lookup_configuration_does_not_close_live_session(tmp_path):
    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    session = FakeSession([turn_world(1), turn_world(2)])
    client = LookupClient()
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2"})
    runner._session = session
    runner._configure_ruleset_lookup()
    assert not session.closed
    runner._request_abort("test watchdog abort")
    assert session.closed


def test_resume_provenance_accepts_explicitly_unavailable_help(tmp_path):
    class MissingHelp:
        status = "unavailable"
        content_hash = "0" * 64
        source_sha256 = ""

    class LookupSession(FakeSession):
        ruleset_reference = None
        ruleset_reference_status = "unavailable"
        help_reference = MissingHelp()

    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    session = LookupSession([turn_world(1), turn_world(2)])
    runner = make_runner(
        tmp_path, client=LookupClient(), session=session, max_turns=1,
        config_overrides={"context_policy": "conversation_v1",
                          "prompt_profile": "gameplay_v2",
                          "validate_reference_provenance": True,
                          "expected_ruleset_reference_hash": None,
                          "expected_help_reference_hash": None,
                          "expected_help_reference_source_sha256": None})
    runner._session = session
    runner._configure_ruleset_lookup()


@pytest.mark.parametrize("profile", ["gameplay_v3", "gameplay_v4"])
def test_resume_provenance_binds_active_and_help_references_for_interactive_profiles(
        tmp_path, profile):
    from types import SimpleNamespace
    from lechonk.core.runner import SessionError

    active = SimpleNamespace(content_hash="a" * 64)
    help_ref = SimpleNamespace(status="available", content_hash="b" * 64,
                               source_sha256="c" * 64)

    class LookupSession(FakeSession):
        ruleset_reference = active
        ruleset_reference_status = "available"
        help_reference = help_ref
        classic_reference = None
        manual_reference = None

    class LookupClient(FakeModelClient):
        def set_ruleset_lookup(self, lookup):
            self.lookup = lookup

    overrides = {
        "context_policy": "conversation_v1", "prompt_profile": profile,
        "validate_reference_provenance": True,
        "prompt_bundle_id": profile,
        "candidate_surface_policy": "candidate_surface_v2",
        "candidate_surface_cap": 8000,
        "candidate_admission_version": "candidate_surface_v2-r1",
        "compaction_schema": "conversation-compaction-v1",
        "prompt_budget_schema": f"{profile.replace('_', '-')}-budget-v1",
        "recovery_config_digest_schema": "interactive-config-v3",
        "expected_ruleset_reference_hash": "a" * 64,
        "expected_help_reference_hash": "b" * 64,
        "expected_help_reference_source_sha256": "c" * 64,
    }
    (tmp_path / "same").mkdir()
    runner = make_runner(
        tmp_path / "same", client=LookupClient(),
        session=LookupSession([turn_world(1), turn_world(2)]),
        max_turns=1, config_overrides=overrides)
    runner._session = runner._session_factory()
    runner._configure_ruleset_lookup()

    active.content_hash = "d" * 64
    with pytest.raises(SessionError, match="provenance changed"):
        runner._configure_ruleset_lookup()
    active.content_hash = "a" * 64
    help_ref.content_hash = "e" * 64
    with pytest.raises(SessionError, match="provenance changed"):
        runner._configure_ruleset_lookup()
    help_ref.content_hash = "b" * 64
    LookupSession.ruleset_reference = None
    with pytest.raises(SessionError, match="provenance changed"):
        runner._configure_ruleset_lookup()
    LookupSession.ruleset_reference = active
    LookupSession.help_reference = None
    with pytest.raises(SessionError, match="provenance changed"):
        runner._configure_ruleset_lookup()


def test_runner_rejects_unknown_prompt_profile():
    with pytest.raises(ValueError, match="prompt_profile"):
        RunnerConfig(prompt_profile="unknown")


def test_runner_rejects_unknown_candidate_surface_policy():
    with pytest.raises(ValueError, match="candidate_surface_policy"):
        RunnerConfig(candidate_surface_policy="unknown")


def test_hybrid_context_carries_only_completed_turns(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    first = ok_response(["unit:10:goto_e"], turn_id="t1", notes="scout east")
    first.batch.notes = "scout east"
    second = ok_response(["unit:11:sentry"], turn_id="t2", notes="hold position")
    second.batch.notes = "hold position"
    client = FakeModelClient([first, second])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={
            "context_policy": "hybrid_v1",
            "context_journal_path": str(tmp_path / "context.jsonl"),
        })
    result = runner.run()

    assert result.outcome == "COMPLETED"
    assert "RECENT VERIFIED HISTORY" in client.turn_prompts[0].history
    assert '"episodes":[]' in client.turn_prompts[0].history
    assert '"turn":1' in client.turn_prompts[1].history
    assert "scout east" in client.turn_prompts[1].history
    # The history is updated only after turn 1 reaches its finish boundary.
    assert '"outcome":"executed"' in client.turn_prompts[1].history
    from lechonk.core.context_journal import load_context_journal
    journal = load_context_journal(tmp_path / "context.jsonl")
    assert [record.turn for record in journal.records] == [1, 2]
    assert all(record.transcript_seq is not None for record in journal.records)


def test_conversation_context_aborts_staged_turn_on_truncated_finish(tmp_path):
    """Infrastructure truncation must never commit a phantom conversation turn."""
    session = FakeSession(
        [turn_world(1)], final_terminated=False, final_truncated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", notes="scout east")
    ])
    commits: list[dict] = []
    aborts: list[bool] = []
    client.commit_turn = lambda report: commits.append(report)
    client.abort_turn = lambda: aborts.append(True)
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={"context_policy": "conversation_v1"})

    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert commits == []
    assert aborts == [True]


def test_hybrid_context_does_not_publish_truncated_finish(tmp_path):
    """An infrastructure-truncated phase cannot enter the legacy journal."""
    session = FakeSession(
        [turn_world(1)], final_terminated=False, final_truncated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", notes="scout east")
    ])
    journal_path = tmp_path / "context.jsonl"
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={
            "context_policy": "hybrid_v1",
            "context_journal_path": str(journal_path),
        })

    result = runner.run()

    assert result.outcome == "TRUNCATED"
    from lechonk.core.context_journal import load_context_journal
    assert load_context_journal(journal_path).records == []


def test_conversation_context_commits_each_completed_phase(tmp_path):
    """Nonterminal phase boundaries commit before the following turn."""
    session = FakeSession([turn_world(1), turn_world(2), turn_world(3)])
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", notes="scout east"),
        ok_response(["unit:11:sentry"], turn_id="t2", notes="hold position"),
    ])
    commits: list[dict] = []
    aborts: list[bool] = []
    client.commit_turn = lambda report: commits.append(report)
    client.abort_turn = lambda: aborts.append(True)
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"context_policy": "conversation_v1"})

    result = runner.run()

    assert result.outcome == "COMPLETED"
    assert [item["historical_turn"] for item in commits] == [1, 2]
    # Teardown may call the client's idempotent abort hook after the final
    # commit; it must not remove either committed turn.
    assert len(aborts) <= 1


def test_conversation_report_carries_rejections_into_next_request(tmp_path):
    """Initial-ingest rejection feedback is committed exactly once.

    The next request still receives the live candidate surface, so a stale
    ref cannot silently become an instruction to substitute another action.
    """
    session = FakeSession([turn_world(1), turn_world(2), turn_world(3)])
    client = FakeModelClient([
        ok_response(["unit:999:goto_w", "unit:10:goto_e"], turn_id="t1"),
        ok_response(["unit:11:sentry"], turn_id="t2", with_memory=False),
    ])
    commits: list[dict] = []
    client.commit_turn = lambda report: commits.append(report)
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"context_policy": "conversation_v1"})

    result = runner.run()

    assert result.outcome == "COMPLETED"
    assert len(commits) == 2
    rejected = commits[0]["rejected"]
    assert len(rejected) == 1
    assert rejected[0]["ref"] == "unit:999:goto_w"
    assert rejected[0]["rejection_class"] == "state_fog_error"
    assert rejected[0]["provenance_class"] == "model"
    assert "offered candidate table" in rejected[0]["reason"]
    assert "unit:999:goto_w" not in client.turn_prompts[1].candidates
    assert "unit:11:sentry" in client.turn_prompts[1].candidates
    assert commits[0]["rejected"] == rejected


def test_conversation_report_carries_mid_batch_stale_rejection_once(tmp_path):
    """A refreshed-table rejection is reported without hidden state."""
    session = FakeSession([turn_world(1), turn_world(2), turn_world(3)])
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:10:goto_nw",
                     "unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    commits: list[dict] = []
    client.commit_turn = lambda report: commits.append(report)
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"context_policy": "conversation_v1"})

    result = runner.run()

    assert result.outcome == "COMPLETED"
    stale = [item for item in commits[0]["rejected"]
             if item["ref"] == "unit:10:goto_nw"]
    assert len(stale) == 1
    assert stale[0]["rejection_class"] == "illegal_in_rule"
    assert stale[0]["provenance_class"] == "model"
    assert "refreshed candidate table" in stale[0]["reason"]
    assert set(stale[0]) == {
        "ref", "rejection_class", "provenance_class", "reason"}
    assert "unit:11:sentry" in client.turn_prompts[1].candidates


# ---------------------------------------------------------------------------
# fault injection: timeout → repair (unresponsive) → executed
# ---------------------------------------------------------------------------


def test_runner_timeout_triggers_unresponsive_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        error_response("request timed out"),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # Two model calls: attempt 1 (error) + the one repair attempt.
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 2, 1]  # turn 1: 1+2, turn 2: 1
    assert calls[0].error == "request timed out"
    # The repair prompt carries the unresponsive class.
    assert len(client.repair_prompts) == 1
    assert client.repair_prompts[0].error_class == "unresponsive"
    # The repair's action executed with model provenance.
    actions = _actions(_records(runner.ledger.path))
    executed = [a for a in actions if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]
    assert executed[0].provenance_class == "model"
    assert executed[0].readback == "verified"
    assert executed[0].call_seq == calls[1].seq


# ---------------------------------------------------------------------------
# fault injection: malformed output → schema_error repair
# ---------------------------------------------------------------------------


def test_runner_malformed_output_triggers_schema_error_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        malformed_response(["(12, 34)", "move to (12, 34)"]),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    progress = []
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, progress=progress.append)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    # One schema-error rejection per emitted raw ref (model attempts).
    rejected = [a for a in _actions(records) if a.outcome == "rejected"]
    assert [a.ref for a in rejected] == ["(12, 34)", "move to (12, 34)"]
    assert all(a.rejection_class == "schema_error" for a in rejected)
    assert all(a.provenance_class == "model" for a in rejected)
    # The repair prompt carries the schema_error class + the failed refs.
    assert client.repair_prompts[0].error_class == "schema_error"
    assert client.repair_prompts[0].failed_refs == ["(12, 34)",
                                                    "move to (12, 34)"]
    # The taxonomy counted the attempts (2 schema_error + 1 executed).
    assert result.metrics.illegal_attempts_total == 3
    assert result.metrics.illegal_action_shares["schema_error"].numerator == 2
    decisions = [event for event in progress if event.get("event") == "turn_decision"]
    assert decisions[0]["repair_used"] is True
    assert decisions[0]["degraded"] is False
    assert "repair cap" not in decisions[0]["why"]


# ---------------------------------------------------------------------------
# fault injection: invalid action → state_fog_error repair
# ---------------------------------------------------------------------------


def test_runner_invalid_action_triggers_state_fog_repair(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:99:goto_n"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    rejected = [a for a in _actions(records) if a.outcome == "rejected"]
    # A well-formed ref that was never offered: state_fog_error.
    assert len(rejected) == 1
    assert rejected[0].ref == "unit:99:goto_n"
    assert rejected[0].rejection_class == "state_fog_error"
    assert client.repair_prompts[0].error_class == "state_fog_error"
    assert client.repair_prompts[0].failed_refs == ["unit:99:goto_n"]
    # The repair's valid action executed.
    executed = [a for a in _actions(records) if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]


# ---------------------------------------------------------------------------
# fault injection: unresponsive after the repair cap → degraded default
# ---------------------------------------------------------------------------


def test_runner_unresponsive_after_repair_cap_degrades(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        error_response("timeout 1"),
        error_response("timeout 2"),  # the repair also fails
        error_response("timeout 3"),
        error_response("timeout 4"),  # turn 2: the repair also fails
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # No third API attempt: exactly two calls per turn.
    assert client.calls == 4
    records = _records(runner.ledger.path)
    actions = _actions(records)
    defaults = [a for a in actions if a.provenance_class == "scripted_default"]
    # The degraded default resolved the new mandatory choices (research
    # completed + city production empty) via the published priority lists.
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.outcome == "executed" for a in defaults)
    assert all("model unresponsive" in a.reason for a in defaults)
    assert all(a.readback == "verified" for a in defaults)
    # The turn was still finished by the pipeline and the run completed.
    finishes = [c for c in session.calls if c[0] is None]
    assert len(finishes) == 2
    # The degraded decisions count in the per-domain shares, not as model
    # participation: turn 1 had opportunities but no model decision.
    assert result.metrics.participation.numerator == 0
    assert result.metrics.participation.denominator == 2


# ---------------------------------------------------------------------------
# batch semantics: a stale action aborts only its entity's group (§3 line 33)
# ---------------------------------------------------------------------------


def test_runner_mid_batch_stale_aborts_only_that_group(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    # unit 10: goto_e (executes, exhausting its moves), goto_nw (stale
    # after the move), build_city (aborted group); unit 11: sentry
    # (another entity — proceeds).
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:10:goto_nw",
                     "unit:10:build_city", "unit:11:sentry"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"  # the turn is never aborted
    actions = _actions(_records(runner.ledger.path))
    by_ref = {a.ref: a for a in actions}
    # The first action executed and was read-back verified.
    assert by_ref["unit:10:goto_e"].outcome == "executed"
    assert by_ref["unit:10:goto_e"].readback == "verified"
    # The stale action: rejected illegal_in_rule (it was offered, then
    # became illegal after the mid-batch mutation).
    stale = by_ref["unit:10:goto_nw"]
    assert stale.outcome == "rejected"
    assert stale.rejection_class == "illegal_in_rule"
    # The rest of unit 10's group: aborted_group (logged, never sent).
    aborted = by_ref["unit:10:build_city"]
    assert aborted.outcome == "aborted_group"
    # The other entity's action proceeded and executed.
    assert by_ref["unit:11:sentry"].outcome == "executed"
    assert by_ref["unit:11:sentry"].readback == "verified"
    # The taxonomy counted the model-emitted attempts only: goto_e (executed)
    # + goto_nw (rejected) + the turn-1 sentry (executed) = 3. The
    # aborted_group record is a pipeline-side group abort, not an attempt,
    # and the turn-2 continued_model_order re-issue is not model-emitted.
    assert result.metrics.illegal_attempts_total == 3


def test_runner_game_over_mid_batch_aborts_every_remaining_action(tmp_path):
    """A mutation ends the game mid-batch with >=2 pending actions: EVERY
    planned-but-unexecuted action — including the FIRST pending action
    after the executed one — gets its aborted_group record."""
    worlds = [turn_world(1), turn_world(2)]
    session = _GameOverMidBatchSession(worlds, terminate_on=("unit", 10,
                                                            "goto_4"))
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:10:build_city",
                     "unit:11:sentry"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "VICTORY"
    actions = _actions(_records(runner.ledger.path))
    # The first action executed and ended the game mid-batch.
    executed = [a for a in actions if a.outcome == "executed"]
    assert [a.ref for a in executed] == ["unit:10:goto_e"]
    # EVERY pending action got its aborted_group record — including the
    # first pending action after the executed one.
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    assert [a.ref for a in aborted] == ["unit:10:build_city",
                                        "unit:11:sentry"]
    assert all(a.detail == "game ended mid-batch" for a in aborted)


# ---------------------------------------------------------------------------
# continued_model_order: the pipeline re-issues a standing order once
# ---------------------------------------------------------------------------


def test_runner_continues_standing_order_linked_to_origin(tmp_path):
    # Turn 1: the model orders unit 11 to sentry (executed, verified).
    # Turn 2: the unit is idle again (the order is not in effect) and the
    # model's batch is empty → the pipeline re-issues the order once as
    # continued_model_order, linked to the originating model action.
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 0})]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    assert len(continuations) == 1
    cont = continuations[0]
    assert cont.turn == 2 and cont.verb == "sentry"
    assert cont.outcome == "executed" and cont.readback == "verified"
    # Linked to its origin: the turn-1 model action_result.
    by_seq = {a.seq: a for a in actions}
    origin = by_seq[cont.origin_seq]
    assert origin.provenance_class == "model"
    assert origin.turn == 1 and origin.verb == "sentry"
    # The continuation counts as model participation (model-originated).
    assert result.metrics.participation.numerator == 2


class _SentryInconclusiveSession(FakeSession):
    """A step-completed (no disconnect) sentry whose read-back is
    INCONCLUSIVE: the send succeeded, but the upstream read path cancels
    ACTIVITY_SENTRY in the same step, so the post-observation shows an
    unchanged activity — sent-then-cancelled is indistinguishable from
    never-sent (the state is known, only the read-back is non-probative)."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self._sentry_sends = 0

    def step(self, action, *, deadline=None):
        if action == ("unit", 11, "sentry"):
            self._sentry_sends += 1
            if self._sentry_sends == 1:
                # Send completes normally; skip the scripted effect so the
                # read-back verifier sees unchanged activity (INCONCLUSIVE).
                return StepOutcome(
                    observation=self.obs, reward=0.0, terminated=False,
                    truncated=False, info=self.info, action=action,
                    elapsed_s=0.01)
        return super().step(action, deadline=deadline)


def test_runner_step_completed_inconclusive_standing_order_continues(tmp_path):
    """A step-completed INCONCLUSIVE read-back (the sentry quirk: the state
    is known, the read-back is merely non-probative) is NOT the §3 line 35
    disconnect quarantine: the standing order is stored on it and stays
    continuable — the later re-issue by _continue_standing_orders is the
    normal state-gated resend, never a quarantine."""
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 0})]),
        turn_world(3),
    ]
    session = _SentryInconclusiveSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    sentries = [a for a in actions if a.ref == "unit:11:sentry"]
    # The turn-1 model order: executed with an INCONCLUSIVE read-back.
    assert sentries[0].provenance_class == "model"
    assert sentries[0].outcome == "executed"
    assert sentries[0].readback == "inconclusive"
    # No disconnect occurred: nothing was quarantined.
    records = _records(runner.ledger.path)
    assert not [r for r in records if isinstance(r, MetricEventRecord)
                and r.name == "mutation_quarantine"]
    # The standing order was stored on the inconclusive read-back and the
    # pipeline re-issued it once as the normal state-gated continuation.
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    assert len(continuations) == 1
    cont = continuations[0]
    assert cont.turn == 2 and cont.verb == "sentry"
    assert cont.outcome == "executed" and cont.readback == "verified"
    assert cont.origin_seq == sentries[0].seq


def test_runner_fortifying_unit_counts_as_standing_in_effect(tmp_path):
    # Turn 1: the model orders unit 11 to fortify (executed, verified —
    # the fake session applies ACTIVITY_FORTIFIED). Turn 2: the unit shows
    # ACTIVITY_FORTIFYING (10), freeciv's in-turn fortify state, which
    # completes to ACTIVITY_FORTIFIED at the turn start — the order is in
    # effect, so the pipeline must NOT re-issue it as
    # continued_model_order even with an empty model batch (live
    # evidence: the WP2 model run read-backs).
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 10})]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:11:fortify"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    by_turn_ref = {(a.turn, a.ref): a for a in actions}
    origin = by_turn_ref[(1, "unit:11:fortify")]
    assert origin.provenance_class == "model"
    assert origin.outcome == "executed" and origin.readback == "verified"
    # Fortifying = the order in effect: no continued_model_order re-issue.
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    assert continuations == []
    # The model participated exactly once (its own turn-1 action).
    assert result.metrics.participation.numerator == 1


# ---------------------------------------------------------------------------
# engine-automatic resolutions are counted, never hidden (§3 line 38) —
# W3-9: observation-event diffing over a persistent last-seen baseline
# (D9/D12, Option A; the old turn-start-snapshot diff is replaced)
# ---------------------------------------------------------------------------


def test_runner_counts_engine_automatic_research_continuation(tmp_path):
    """W3-9 REPLACEMENT (the old test pinned the turn-start snapshot
    diff: the change was detected at the NEXT turn start and attributed
    to that turn). New path: the w1->w2 transition is captured at
    turn 1's FINISH-boundary observation event (the fourth event) and
    attributed to the COMPLETING turn."""
    # Turn 1: researching Alphabet (tech 1 per default_techs), the model
    # only moves a unit. The engine continues research to Pottery
    # (tech 2) during WAITING — no executed tech action -> engine_
    # automatic, counted at the finish-boundary event, attributed to
    # the completing turn. (The tech domain resolves the ids:
    # researching=1 -> "Alphabet", researching=2 -> "Pottery".)
    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1  # the completing turn (the finish
    # boundary), not the next turn start
    assert automatic[0].domain == "tech"
    assert automatic[0].target == "Pottery"
    assert automatic[0].outcome == "executed"
    assert automatic[0].readback == "none"
    # The ref carries the entity identity + the observation index.
    assert automatic[0].ref.startswith("engine_automatic:research:player:")
    assert ":obs" in automatic[0].ref
    # Counted in the research domain share, not as model participation.
    research_share = result.metrics.domain_model_shares["research"]
    assert research_share.denominator == 1  # only the engine resolution
    assert research_share.numerator == 0


def test_runner_counts_engine_automatic_first_research_choice(tmp_path):
    """W3-9 REPLACEMENT (the old test pinned the turn-start snapshot
    diff with the next-turn attribution). FW-R3 carried over: turn 1
    has NO research target (the player domain carries no "researching"
    — current_target_id None); the model only moves a unit. The engine
    auto-set an initial target (tech 2 = Pottery) during WAITING with
    no executed tech action of ours — a None -> target first choice is
    a new research choice: engine_automatic, counted at the
    finish-boundary event, attributed to the completing turn."""
    worlds = [
        turn_world(1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1  # the completing turn (the finish
    # boundary), not the next turn start
    assert automatic[0].domain == "tech"
    assert automatic[0].target == "Pottery"
    assert automatic[0].outcome == "executed"


def test_runner_first_research_choice_with_own_action_is_model_attributed(
        tmp_path):
    # FW-R3: turn 1 has NO research target and the model's OWN first
    # research action executes (verified: the scripted tech_id 1 lands as
    # "researching"). Turn 2: the world shows that same target id
    # (researching=1 = Alphabet) — the None -> target transition is
    # accounted for by our executed action: model-attributed, never
    # engine_automatic (the first choice must not bypass the own-action
    # suppression).
    worlds = [
        turn_world(1),
        turn_world(2, researching=1),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["tech:cur_player:research_tech_pottery_1"],
                    turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    research = [a for a in actions
                if a.ref == "tech:cur_player:research_tech_pottery_1"]
    assert len(research) == 1
    assert research[0].provenance_class == "model"
    assert research[0].outcome == "executed"
    assert research[0].readback == "verified"
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]


def test_runner_counts_engine_automatic_first_city_production(tmp_path):
    """W3-9 REPLACEMENT (the old test pinned the turn-start snapshot
    diff with the next-turn attribution). FW-R3 carried over: turn 1:
    # Rome has no production (production_kind VUT_NONE); the model only
    moves a unit. The engine auto-set the city's first production
    (Settlers) during WAITING with no executed produce action of ours —
    a None -> production first choice is a new production choice:
    engine_automatic, counted at the finish-boundary event, attributed
    to the completing turn."""
    worlds = [
        turn_world(1),
        turn_world(2, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=SETTLERS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1  # the completing turn (the finish
    # boundary), not the next turn start
    assert automatic[0].domain == "city"
    assert automatic[0].target == "Settlers"
    assert automatic[0].outcome == "executed"
    # The ref carries the entity identity + the observation index.
    assert automatic[0].ref == "engine_automatic:production:city:1:obs3"


def test_runner_first_city_production_with_own_action_is_model_attributed(
        tmp_path):
    # FW-R3: turn 1: Rome has no production and the model's OWN first
    # produce action (Settlers) executes (verified via the production name
    # map). Turn 2: the scripted world shows that same production — the
    # None -> production transition is accounted for by our executed
    # action: model-attributed, never engine_automatic.
    worlds = [
        turn_world(1),
        turn_world(2, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=SETTLERS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["city:1:produce_settlers"], turn_id="t1",
                    with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    produce = [a for a in actions if a.ref == "city:1:produce_settlers"]
    assert len(produce) == 1
    assert produce[0].provenance_class == "model"
    assert produce[0].outcome == "executed"
    assert produce[0].readback == "verified"
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]


# ---------------------------------------------------------------------------
# W3-9 acceptance matrix (D9/D12, Option A): the observation-event
# diffing hardenings — corroborated founding, A→B→A reverts, buy-
# completion conditionality, the empty baseline, the horizon rule, the
# emission-order contract, the DEF-3 gap diagnostic, entity-identity
# refs + the P11 audit grouping, and the unchanged readback credit
# ---------------------------------------------------------------------------


class _FoundingSession(FakeSession):
    """A FakeSession whose build_city step founds the city with the
    engine's default production choice landing with it — the founding
    verifier's corroborated evidence (execute.py: a new own city at
    the unit's tile, the unit consumed)."""

    def __init__(self, worlds, *, new_city_id=3, production="Warriors"):
        super().__init__(worlds)
        self._new_city_id = new_city_id
        self._production = production

    def _apply_effect(self, obs, action):
        ctrl_type, actor_id, action_key = action
        if ctrl_type == "unit" and action_key == "build_city":
            uid = int(actor_id)
            unit = obs["unit"].get(uid)
            if unit is None:
                return
            x, y = int(unit["x"]), int(unit["y"])
            del obs["unit"][uid]
            kind, value = self._production_names.get(
                self._production, (VUT_IMPROVEMENT, 0))
            obs["city"][self._new_city_id] = own_city(
                city_id=self._new_city_id, name="Newtown", x=x, y=y,
                production_kind=kind, production_value=value)
            return
        super()._apply_effect(obs, action)


def test_runner_engine_record_for_founded_city_initial_production(tmp_path):
    """W3-9 acceptance (1): a new city founded mid-batch — the post-step
    observation shows the new city with the engine's production choice.
    With the corroborated-founding rule the model-directed founding
    still emits the engine's production choice (founding and choosing
    the default production are distinct decisions): ONE engine record
    at that event, after the founding action's own result."""
    worlds = [turn_world(1), turn_world(2)]
    session = _FoundingSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:build_city"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    actions = _actions(records)
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    record = automatic[0]
    assert record.turn == 1
    assert record.domain == "city" and record.verb == "produce_auto"
    assert record.target == "Warriors"
    assert record.outcome == "executed" and record.readback == "none"
    # The ref carries the entity identity + the observation index.
    assert record.ref == "engine_automatic:production:city:3:obs2"
    # The founding action's own result precedes the observation diff
    # (the emission-order contract), and the founding verified.
    founding = [a for a in actions if a.ref == "unit:10:build_city"]
    assert len(founding) == 1
    assert founding[0].readback == "verified"
    assert record.seq > founding[0].seq


def test_runner_engine_record_for_a_b_a_revert(tmp_path):
    """W3-9 acceptance (2): A→B→A — the model produces B on turn 1
    (the post-step observation attributes the change to us: the
    baseline updates to B, no record), the engine reverts to A before
    the next turn start — the finish-boundary observation captures the
    revert: ONE engine record for the revert, none for the B change
    itself."""
    granary = dict(production_kind=VUT_IMPROVEMENT, production_value=0)
    worlds = [
        turn_world(1, cities=[own_city(**granary), foreign_city()]),
        turn_world(2, cities=[own_city(**granary), foreign_city()]),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["city:1:produce_settlers"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    # ONE engine record: the revert to Granary (A), captured at the
    # finish-boundary observation event, attributed to the completing
    # turn. None for the B (Settlers) change itself — our executed
    # produce action accounts for it.
    assert len(automatic) == 1
    assert automatic[0].turn == 1
    assert automatic[0].target == "Granary"
    assert automatic[0].verb == "produce_auto"
    assert not [a for a in automatic if a.target == "Settlers"]
    # Our produce action executed and was read-back verified.
    produce = [a for a in actions if a.ref == "city:1:produce_settlers"]
    assert len(produce) == 1 and produce[0].readback == "verified"


def test_runner_engine_record_for_buy_completion_at_next_observation(tmp_path):
    """W3-9 acceptance (6) + hardening (7): buy-completion capture is
    CONDITIONAL on an observed transition. The buy's own post-step
    observation shows the production UNCHANGED (the W3-8 live probe
    observed a bought Warrior completing at the turn change) — no
    record there; the completion transition is captured at the NEXT
    observation event (the finish boundary) as an engine choice with
    no produce action of ours matching."""
    worlds = [
        turn_world(1, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=SETTLERS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
        turn_world(2, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=WARRIORS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["city:1:city_buy_production"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    actions = _actions(records)
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1
    assert automatic[0].target == "Warriors"
    assert automatic[0].verb == "produce_auto"
    # The buy executed and was verified (the W3-8 pinned direction: a
    # successful buy raises shield_stock).
    buy = [a for a in actions if a.ref == "city:1:city_buy_production"]
    assert len(buy) == 1 and buy[0].readback == "verified"
    # The record was captured at the finish-boundary event (the next
    # observation event after the buy), NOT at the buy's own post-step
    # event: its seq follows the WAITING state transition (the finish
    # primitive already ran).
    waiting = [r for r in records
               if isinstance(r, MetricEventRecord)
               and r.name == "state_transition"
               and r.value == "FINISH_SENT->WAITING"]
    assert len(waiting) == 1
    assert automatic[0].seq > waiting[0].seq
    assert automatic[0].seq > buy[0].seq


def test_runner_first_turn_empty_baseline_emits_no_records(tmp_path):
    """W3-9 acceptance (7): the empty baseline seeds silently — turn
    1's PRE-EXISTING research/production choices are baselined, never
    emitted as engine resolutions."""
    cities = [
        own_city(production_kind=VUT_UTYPE,
                 production_value=SETTLERS + NUM_IMPROVEMENTS),
        foreign_city(),
    ]
    worlds = [
        turn_world(1, researching=2, cities=cities),
        turn_world(2, researching=2, cities=cities),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]


def test_runner_engine_records_load_and_render_in_replay(tmp_path):
    """W3-9 acceptance (8): a transcript containing engine records
    loads via load_transcript and renders in the replay viewer."""
    from lechonk.eval import replay

    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # loads via load_transcript
    records = load_transcript(runner.ledger.path).records
    engine = [a for a in records if isinstance(a, ActionResultRecord)
              and a.provenance_class == "engine_automatic"]
    assert len(engine) == 1
    # renders in replay
    out = tmp_path / "replay"
    summary = replay.render_bundle(runner.ledger.path, out)
    assert summary["meta"]["outcome"] == "COMPLETED"
    html = (out / "index.html").read_text(encoding="utf-8")
    assert engine[0].ref in html  # the engine record is embedded
    assert "pv-engine" in html  # the engine provenance class is styled


def test_runner_engine_records_never_attributed_beyond_horizon(tmp_path):
    """W3-9 acceptance (9) + the horizon rule: the finish-boundary
    event attributes to the COMPLETING turn — no engine record is
    attributed beyond max_turns (the runner exits before processing
    the next turn; the metric horizon is never extended)."""
    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert result.turns_completed == 1
    actions = _actions(_records(runner.ledger.path))
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    # The w1->w2 transition is captured at the finish boundary BEFORE
    # the horizon exit — attributed to the completing turn.
    assert len(automatic) == 1
    assert automatic[0].turn == 1
    assert all(a.turn <= 1 for a in automatic)
    # The metric horizon derives from turn-bearing records: (1, 1).
    assert result.metrics.run_turns == (1, 1)


class _EngineActingDeadlineSession(FakeSession):
    """A mutation step that consumes a scripted fake-clock interval per
    send (the D13 straddle) AND applies an engine production change on
    a scripted step — the engine acted during our own phase, invisible
    to any action of ours (the W3-9 post-step observation diff)."""

    def __init__(self, worlds, clock, *, step_delays_s, engine_step,
                 production=(VUT_UTYPE, WARRIORS + NUM_IMPROVEMENTS)):
        super().__init__(worlds)
        self._clock = clock
        self._step_delays_s = list(step_delays_s)
        self._engine_step = engine_step
        self._production = production
        self._mutation_steps = 0

    def step(self, action, *, deadline=None):
        if action is not None:
            self._mutation_steps += 1
            if self._step_delays_s:
                self._clock.advance(self._step_delays_s.pop(0))
            if self._mutation_steps == self._engine_step:
                obs, info = self._worlds[self._index]
                new_obs = copy.deepcopy(obs)
                city = new_obs["city"][1]
                city["production_kind"], city["production_value"] = \
                    self._production
                self._worlds[self._index] = (new_obs, info)
        return super().step(action, deadline=deadline)


def test_runner_deadline_stop_never_half_records_an_observation_event(tmp_path):
    """W3-9 acceptance (10) + the emission-order contract (glm §8): a
    deadline stop mid-batch does not half-record an observation event.
    The straddling completed mutation emits its own result first, then
    the observation diff (the engine's production change is captured);
    the W3-2 guard records (the aborted_group + the stop summary) come
    after — the observation event completed, never half-recorded."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _EngineActingDeadlineSession(
        worlds, clock, step_delays_s=[100.0, 100.0, 100.0], engine_step=2)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e", "unit:11:sentry",
                     "unit:10:goto_nw"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    records = _records(runner.ledger.path)
    actions = _actions(records)
    executed = [a for a in actions
                if a.outcome == "executed"
                and a.provenance_class == "model"]
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    automatic = [a for a in actions
                 if a.provenance_class == "engine_automatic"]
    # Two sends completed (the second straddled the deadline, D13); the
    # engine's production change on the straddling step is captured by
    # that mutation's post-step observation event.
    assert [a.ref for a in executed] == ["unit:10:goto_e", "unit:11:sentry"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1
    assert automatic[0].target == "Warriors"
    # The pinned order: the straddling mutation's own result, then the
    # observation diff, then the guard's records (never half-recorded).
    assert automatic[0].seq > executed[-1].seq
    assert aborted[0].seq > automatic[0].seq
    stops = _stop_events(records)
    assert len(stops) == 1
    assert stops[0].seq > aborted[0].seq


def test_runner_engine_observation_gap_is_disclosed_and_run_valid(
        tmp_path, monkeypatch):
    """W3-9 acceptance (11) + DEF-3: the bounded per-interval
    attribution set overflows -> the observation-gap diagnostic metric
    event is emitted (fail-open with mandatory disclosure) and the run
    remains valid."""
    monkeypatch.setattr(
        "lechonk.core.runner.ENGINE_OBSERVATION_INTERVAL_MAX", 0)
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["city:1:produce_settlers",
                     "tech:cur_player:research_tech_pottery_1"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    # Fail-open: the run remains valid.
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    gaps = [r for r in records
            if isinstance(r, MetricEventRecord)
            and r.name == "engine_observation_gap"]
    # Both retained actions dropped (the bound is 0): one diagnostic per
    # dropped action, each naming the affected turn.
    assert len(gaps) == 2
    assert all("turn 1" in g.value for g in gaps)
    # The dropped attribution is disclosed, never silent: with the
    # interval set empty the produce/research changes are NOT
    # attributed to our actions — the engine records are emitted and
    # the accounting is flagged potentially incomplete.
    automatic = [a for a in _actions(records)
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 2


def test_runner_ownership_transfer_is_baseline_only(tmp_path):
    """W3-9 hardening (2): a first sighting that is NOT a corroborated
    founding (an ownership transfer / checkpoint resynchronization) is
    baseline-only — never mislabeled as a fresh engine choice."""
    warriors = dict(production_kind=VUT_UTYPE,
                    production_value=WARRIORS + NUM_IMPROVEMENTS)
    worlds = [
        turn_world(1),
        turn_world(2, cities=[
            own_city(),
            own_city(city_id=3, name="Conquered", **warriors),
            foreign_city(),
        ]),
        turn_world(3, cities=[
            own_city(),
            own_city(city_id=3, name="Conquered", **warriors),
            foreign_city(),
        ]),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    # The transferred city's production is baselined, never recorded.
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]


def test_runner_resighting_after_absence_is_baseline_only(tmp_path):
    """W3-9 hardening (2): a city that disappears from an observation
    (loss/conquest) and is re-sighted later is baseline-only — never
    mislabeled as a fresh engine choice, even though its production
    differs from the pre-absence value."""
    warriors = dict(production_kind=VUT_UTYPE,
                    production_value=WARRIORS + NUM_IMPROVEMENTS)
    granary = dict(production_kind=VUT_IMPROVEMENT, production_value=0)
    worlds = [
        turn_world(1, cities=[
            own_city(),
            own_city(city_id=3, name="Faraway", **warriors),
            foreign_city(),
        ]),
        turn_world(2, cities=[own_city(), foreign_city()]),
        turn_world(3, cities=[
            own_city(),
            own_city(city_id=3, name="Faraway", **granary),
            foreign_city(),
        ]),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    actions = _actions(_records(runner.ledger.path))
    # The re-sighting (Warriors -> Granary across the absence) is
    # baseline-only: no record.
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]


class _NoEffectProduceSession(FakeSession):
    """A FakeSession whose produce step does NOT apply the effect —
    the mutation provably did not execute (the read-back is
    CONTRADICTED: the production is unchanged)."""

    def _apply_effect(self, obs, action):
        ctrl_type, _actor_id, action_key = action
        if ctrl_type == "city" and action_key.startswith("produce_"):
            return  # the production does not change
        super()._apply_effect(obs, action)


def test_runner_contradicted_produce_is_never_upgraded_by_observation(tmp_path):
    """W3-9 hardening (6): the readback credit rules are unchanged —
    observation history never upgrades a contradicted action. A
    produce action that reads back CONTRADICTED (the production is
    unchanged: the rules engine proved non-execution) is not
    attributed to the model, emits no engine record (the observation
    shows no change), and never credits."""
    worlds = [
        turn_world(1, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=SETTLERS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
        turn_world(2, cities=[
            own_city(production_kind=VUT_UTYPE,
                     production_value=SETTLERS + NUM_IMPROVEMENTS),
            foreign_city(),
        ]),
    ]
    session = _NoEffectProduceSession(worlds)
    client = FakeModelClient([
        ok_response(["city:1:produce_granary"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    actions = _actions(records)
    produce = [a for a in actions if a.ref == "city:1:produce_granary"]
    assert len(produce) == 1
    # Contradicted: provably non-executed (the production is unchanged).
    assert produce[0].outcome == "executed"
    assert produce[0].readback == "contradicted"
    assert produce[0].rejection_class == "illegal_in_rule"
    # No engine record: the observation shows no change, and the
    # observation history never upgrades the contradicted action.
    assert not [a for a in actions
                if a.provenance_class == "engine_automatic"]
    report = compute_metrics(records)
    assert report.domain_model_shares["production"].numerator == 0


def test_runner_engine_refs_distinguish_same_production_cities(tmp_path):
    """W3-9 acceptance (12) + hardening (1): two own cities whose
    production changes to the SAME name produce two DISTINGUISHABLE
    engine records — the refs carry the entity identity and the
    observation index — and the P11 audit's by-ref grouping keeps them
    apart (name-only refs would collide and fail the audit's
    no_duplicate_mutation check)."""
    settlers = dict(production_kind=VUT_UTYPE,
                    production_value=SETTLERS + NUM_IMPROVEMENTS)
    warriors = dict(production_kind=VUT_UTYPE,
                    production_value=WARRIORS + NUM_IMPROVEMENTS)

    def cities_with(prod):
        return [own_city(city_id=1, name="Rome", **prod),
                own_city(city_id=3, name="Alexandria", **prod),
                foreign_city()]

    worlds = [
        turn_world(1, cities=cities_with(settlers)),
        turn_world(2, cities=cities_with(warriors)),
        turn_world(3, cities=cities_with(warriors)),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    automatic = [a for a in _actions(records)
                 if a.provenance_class == "engine_automatic"]
    # Both cities changed to the SAME production name...
    assert len(automatic) == 2
    assert {a.target for a in automatic} == {"Warriors"}
    # ... yet the records are distinguishable: each ref carries its
    # city's identity and the observation index.
    assert {a.ref for a in automatic} == {
        "engine_automatic:production:city:1:obs2",
        "engine_automatic:production:city:3:obs2",
    }
    # The P11 audit groups executed results by ref: the entity-identity
    # refs keep the two same-production records apart — the audit's
    # no_duplicate_mutation check passes.
    audit = wp3_p11_audit.audit_transcript(runner.ledger.path)
    assert audit["checks"]["no_duplicate_mutation"]["pass"] is True


def test_p11_audit_groups_engine_records_by_entity_ref(tmp_path):
    """W3-9 hardening (1) counterfactual pin: two executed engine
    records sharing a NAME-ONLY ref (the old format) collide in the P11
    audit's by-ref grouping and FAIL its no_duplicate_mutation check —
    the entity-identity refs are what keep same-production records
    apart."""
    path = tmp_path / "collide.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    for _city_id in (1, 3):
        ledger.append(ActionResultRecord(
            turn=1, ref="engine_automatic:production:Warriors",
            domain="city", verb="produce_auto", target="Warriors",
            affected="Warriors", outcome="executed",
            provenance_class="engine_automatic", readback="none"),
            durable=True)
    ledger.append(OutcomeRecord(outcome="COMPLETED", reason="horizon"),
                  durable=True)
    ledger.close()
    audit = wp3_p11_audit.audit_transcript(path)
    assert audit["checks"]["no_duplicate_mutation"]["pass"] is False


# ---------------------------------------------------------------------------
# supervision: the ACTING deadline and the stall watchdog
# ---------------------------------------------------------------------------


def test_runner_acting_deadline_exceeded_fails_bounded(tmp_path):
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        advancing_response(
            lambda: ok_response(["unit:10:goto_e"], turn_id="t1",
                                with_memory=False),
            clock, seconds=200.0),  # the model call alone exceeds ACTING 180s
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    # Artifacts intact: the outcome record is ledgered.
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert session.closed


class _WatchdogLedger:
    """Duck-typed ledger for the watchdog: only ``last_event`` (the stall
    input). Any write attempt fails the test — the watchdog never writes
    to the ledger (D14)."""

    def __init__(self, last_event: float):
        self.last_event = last_event

    def append(self, *args, **kwargs):
        raise AssertionError("the stall watchdog must never write to the ledger")


class _PhaseState:
    """A mutable (generation, deadline) holder — the watchdog's
    phase_state callable."""

    def __init__(self, generation: int = 0, deadline=None):
        self.generation = generation
        self.deadline = deadline

    def __call__(self):
        return (self.generation, self.deadline)


class _TransitioningPhaseState:
    """A phase_state that transitions mid-tick: the first call (the
    tick's capture) reports the OLD phase; later calls (the re-verify)
    report the NEW phase — simulating a phase transition landing between
    the two reads."""

    def __init__(self, old, new):
        self._old = old
        self._new = new
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._old if self.calls == 1 else self._new


class _WatchdogDriver:
    """Drives a StallWatchdog tick-by-tick (no thread): a fake clock, a
    stub ledger, a mutable phase-state holder and a recorded stall
    list."""

    def __init__(self, *, stall_s=600.0, deadline=None, generation=0,
                 last_event=0.0):
        self.clock = FakeClock(start=last_event)
        self.ledger = _WatchdogLedger(last_event)
        self.phase = _PhaseState(generation=generation, deadline=deadline)
        self.stalls: list[str] = []
        self.watchdog = StallWatchdog(
            stall_s=stall_s, interval_s=0.02, clock=self.clock,
            ledger=self.ledger, on_stall=self.stalls.append,
            phase_state=self.phase)

    def tick(self) -> bool:
        return self.watchdog._tick()


class _SlowFinishSession(FakeSession):
    """A finish step that consumes a scripted fake-clock interval (a
    WAITING window: longer than the stall threshold, shorter than the
    WAITING deadline) before completing — the watchdog sees the stall
    while the window is legally open."""

    def __init__(self, worlds, clock, *, finish_delay_s):
        super().__init__(worlds)
        self._clock = clock
        self._finish_delay_s = finish_delay_s
        self.finish_delays = 0

    def step(self, action, *, deadline=None):
        if action is None:
            self._clock.advance(self._finish_delay_s)
            self.finish_delays += 1
        return super().step(action, deadline=deadline)


def test_runner_watchdog_ignores_legal_waiting_window(tmp_path):
    """D7: the watchdog must not abort a legal WAITING interval before
    its WAITING deadline. The finish step consumes 605 s of fake time —
    past the 600 s stall threshold, inside the 900 s WAITING deadline —
    and the watchdog never fires: the run completes normally."""
    clock = FakeClock()
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = _SlowFinishSession(worlds, clock, finish_delay_s=605.0)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert "stall watchdog" not in result.reason
    assert session.finish_delays == 2
    assert session.closed


def test_runner_waiting_deadline_ends_the_wait_not_the_watchdog(tmp_path):
    """D7: at the WAITING deadline it is the WAITING deadline mechanism —
    not the watchdog — that ends the wait. The finish step outlasts the
    900 s WAITING deadline (the stall threshold sits far above it, so the
    watchdog cannot fire): the run ends TRUNCATED with the WAITING
    deadline reason."""
    clock = FakeClock()
    session = _SlowFinishSession([turn_world(1)], clock,
                                 finish_delay_s=905.0)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock,
                         config_overrides={"stall_watchdog_s": 10000.0})
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "WAITING exceeded its deadline of 900.0s" in result.reason
    assert "stall watchdog" not in result.reason


def test_runner_stall_watchdog_fires_once_phase_deadline_passed(tmp_path):
    """D14: the watchdog is a pure backstop — a blocked finish whose
    WAITING deadline has ALSO passed is a genuine stall: the watchdog
    fires and the ordered shutdown closes the session."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    # The finish step blocks until close() — no transcript event flows.
    session = FakeSession(worlds, block_step=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=2,
        config_overrides={"stall_watchdog_s": 0.2,
                          "waiting_deadline_s": 0.1,
                          "watchdog_interval_s": 0.02})
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "stall watchdog" in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "stall watchdog" in records[-1].reason
    assert session.closed  # the ordered shutdown closed the session


def test_watchdog_legal_waiting_window_does_not_fire():
    """D7 at the firing rule: a stall inside a legal WAITING interval
    never fires — the WAITING deadline mechanism ends the wait, not the
    watchdog. A 605 s stall with the phase deadline at +900 s does not
    fire; at/after the deadline (>= 900 s) the backstop fires."""
    driver = _WatchdogDriver(deadline=900.0, last_event=0.0)
    driver.clock.advance(605.0)  # stalled past 600 s, inside the deadline
    assert driver.tick() is False
    assert driver.stalls == []
    driver.clock.advance(294.0)  # 899 s: still inside the deadline
    assert driver.tick() is False
    assert driver.stalls == []
    driver.clock.advance(1.0)  # 900 s: the deadline has passed
    assert driver.tick() is True
    assert len(driver.stalls) == 1
    assert "stall watchdog" in driver.stalls[0]


def test_watchdog_acting_hang_fires_after_deadline():
    """D14: an ACTING hang is a genuine stall once the ACTING deadline
    (180 s) has also passed — the watchdog fires at the 600 s stall
    threshold (the backstop is preserved)."""
    driver = _WatchdogDriver(deadline=180.0, last_event=0.0)
    driver.clock.advance(605.0)  # stalled; the deadline passed at 180 s
    assert driver.tick() is True
    assert len(driver.stalls) == 1


def test_watchdog_uses_the_new_deadline_after_a_transition():
    """A phase transition mid-stall republishes the deadline: the next
    tick evaluates the NEW phase's deadline (both directions)."""
    # Stalled inside a WAITING deadline: no fire; the wait ends and the
    # phase advances to a terminal state (no deadline) — the plain stall
    # rule applies again on the next tick.
    driver = _WatchdogDriver(deadline=900.0, last_event=0.0)
    driver.clock.advance(605.0)
    assert driver.tick() is False
    driver.phase.generation += 1
    driver.phase.deadline = None  # terminal: no deadline published
    assert driver.tick() is True
    assert len(driver.stalls) == 1
    # The reverse: stalled in a terminal state, then a transition into a
    # phase whose deadline is still 900 s out suspends the backstop.
    driver = _WatchdogDriver(deadline=None, last_event=0.0)
    driver.clock.advance(605.0)
    driver.phase.generation += 1
    driver.phase.deadline = driver.clock() + 900.0  # a fresh WAITING
    assert driver.tick() is False
    assert driver.stalls == []
    driver.clock.advance(900.0)  # the new deadline has passed
    assert driver.tick() is True


def test_watchdog_terminal_state_uses_the_plain_stall_rule():
    """D14: terminal states publish no deadline, so the plain stall rule
    applies — a stall fires (605 s would be a legal WAITING window)."""
    driver = _WatchdogDriver(deadline=None, last_event=0.0)
    driver.clock.advance(605.0)
    assert driver.tick() is True
    assert len(driver.stalls) == 1
    assert "stall watchdog" in driver.stalls[0]


def test_watchdog_stale_tick_cannot_abort_a_new_phase():
    """The generation guard: a stale tick cannot abort a newly entered
    phase. The tick captures a terminal phase (no deadline — it would
    fire), but a transition lands before the abort request: the
    re-verified generation differs, so the tick is discarded. The NEXT
    tick evaluates the new phase on its own terms and fires."""
    clock = FakeClock(start=0.0)
    stalls: list[str] = []
    # The tick's capture: terminal (no deadline). The re-verify: a fresh
    # ACTING phase whose deadline is still 180 s out.
    phase = _TransitioningPhaseState((0, None), (1, 180.0))
    watchdog = StallWatchdog(
        stall_s=600.0, interval_s=0.02, clock=clock,
        ledger=_WatchdogLedger(0.0), on_stall=stalls.append,
        phase_state=phase)
    clock.advance(605.0)  # stalled past the threshold
    assert watchdog._tick() is False
    assert stalls == []  # the stale tick did not abort
    assert phase.calls == 2  # captured, then re-verified
    # The next tick evaluates the new phase on its own terms: stalled and
    # past its deadline — the backstop fires.
    assert watchdog._tick() is True
    assert len(stalls) == 1


def test_watchdog_never_enforces_the_deadline_itself():
    """D14: no active deadline enforcement — the watchdog fires only on
    a stall (no transcript event for stall_s) once the deadline has
    passed. The deadline alone, with events flowing, never fires; and
    the watchdog never writes to the ledger (the stub ledger's append
    fails the test)."""
    driver = _WatchdogDriver(deadline=900.0, last_event=0.0)
    for _ in range(3):
        driver.clock.advance(300.0)  # 300 s, 600 s, 900 s — the deadline
        driver.ledger.last_event = driver.clock()  # an event just flowed
        assert driver.tick() is False
    assert driver.stalls == []


def test_runner_transition_publishes_phase_deadlines(tmp_path):
    """_transition publishes the current phase's absolute deadline for
    the watchdog: STARTUP +startup, ACTING +acting (also the single
    source _acting_deadline_at), FINISH_SENT/WAITING +waiting (FINISH_SENT
    gets the waiting deadline, not the plain rule), terminal states
    None. The generation bumps on every transition."""
    clock = FakeClock()
    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=FakeSession([turn_world(1)]),
                         max_turns=1, clock=clock)
    assert runner._acting_deadline_at is None  # before any ACTING

    def published(state):
        generation_before = runner._state_generation
        runner._transition(state)
        assert runner._state_generation == generation_before + 1
        return runner._phase_deadline_at

    entered = clock()
    assert published(RunState.STARTUP) == entered + 120.0
    clock.advance(10.0)
    entered = clock()
    deadline = published(RunState.ACTING)
    assert deadline == entered + 180.0
    assert runner._acting_deadline_at == deadline  # the single source
    clock.advance(10.0)
    entered = clock()
    assert published(RunState.FINISH_SENT) == entered + 900.0
    clock.advance(10.0)
    entered = clock()
    assert published(RunState.WAITING) == entered + 900.0
    for terminal in (RunState.GAME_OVER, RunState.FAILED,
                     RunState.NEXT_OWN_PHASE):
        assert published(terminal) is None


def test_runner_progress_callback_is_live_and_failure_isolated(tmp_path):
    events = []

    def progress(event):
        events.append(event)
        if event["event"] == "state":
            raise RuntimeError("observer disconnected")

    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=FakeSession([turn_world(1)]), max_turns=1,
                         progress=progress)
    runner._transition(RunState.ACTING)
    runner._progress("turn_started", turn=1)

    assert events[0]["event"] == "state"
    assert events[0]["state"] == "ACTING"
    # The callback exception above must not prevent later events.
    assert events[1] == {"event": "turn_started", "turn": 1}


def test_action_result_progress_exposes_json_native_focus_descriptors(tmp_path):
    """Capture observers receive the same structured location evidence as
    the durable action record, with tuples normalized for JSON consumers."""
    events = []
    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=FakeSession([turn_world(1)]), max_turns=1,
                         progress=events.append)

    runner._emit_action_result(
        turn=1, ref="unit:7:goto_e", domain="unit", verb="goto_e",
        target="tile:3,2", affected="tile:3,2", actor_tile=(2, 2),
        outcome="executed", readback="verified")

    assert events == [{
        "event": "action_result", "turn": 1, "ref": "unit:7:goto_e",
        "domain": "unit", "verb": "goto_e", "target": "tile:3,2",
        "affected": "tile:3,2", "actor_tile": [2, 2],
        "outcome": "executed", "provenance_class": "model",
        "readback": "verified",
    }]


def test_action_result_progress_preserves_engine_automatic_provenance(tmp_path):
    events = []
    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=FakeSession([turn_world(1)]), max_turns=1,
                         progress=events.append)

    runner._emit_action_result(
        turn=1, ref="city:7:produce", domain="city", verb="produce",
        target="city:7", outcome="executed",
        provenance_class="engine_automatic", readback="verified")

    assert events[-1]["event"] == "action_result"
    assert events[-1]["provenance_class"] == "engine_automatic"


def test_runner_session_ready_publishes_pingtimeout_readback(tmp_path):
    """Expose AdapterSession's verified timeout without breaking duck types."""
    session = FakeSession([turn_world(1)])
    session.pingtimeout_status = "confirmed"
    session.pingtimeout_actual = 731
    events = []
    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=session, max_turns=1, progress=events.append)

    runner.run()

    ready = next(event for event in events
                 if event.get("event") == "session_ready")
    assert ready["pingtimeout_status"] == "confirmed"
    assert ready["pingtimeout_actual"] == 731
    assert ready["aifill"] == 2
    assert ready["configured_ai_opponents"] == 1


def test_runner_session_ready_keeps_legacy_session_compatible(tmp_path):
    """A pre-readback fake session still receives the ordinary event shape."""
    events = []
    runner = make_runner(tmp_path, client=FakeModelClient(),
                         session=FakeSession([turn_world(1)]), max_turns=1,
                         progress=events.append)

    runner.run()

    ready = next(event for event in events
                 if event.get("event") == "session_ready")
    assert ready["event"] == "session_ready"
    assert "pingtimeout_status" not in ready
    assert "pingtimeout_actual" not in ready


# ---------------------------------------------------------------------------
# outcome taxonomy: DEFEATED / VICTORY at the phase boundary
# ---------------------------------------------------------------------------


def test_runner_defeated_at_phase_boundary(tmp_path):
    obs, info = turn_world(1)
    obs["player"][0]["is_alive"] = False  # my player is defeated
    session = FakeSession([(obs, info)], final_terminated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    records = _records(runner.ledger.path)
    assert records[-1].outcome == "DEFEATED"


def test_runner_victory_at_phase_boundary(tmp_path):
    session = FakeSession([turn_world(1)], final_terminated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "VICTORY"


def test_runner_unbounded_horizon_runs_until_game_over(tmp_path):
    """An interactive/unbounded run must not stop at an artificial turn cap."""
    session = FakeSession([turn_world(1), turn_world(2)],
                          final_terminated=True)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=None)

    result = runner.run()

    assert result.outcome == "VICTORY"
    assert result.turns_completed == 2
    assert "horizon" not in result.reason


# ---------------------------------------------------------------------------
# ledger durability through the runner (§3 line 34)
# ---------------------------------------------------------------------------


def test_runner_transcript_is_durable_and_recovers(tmp_path):
    """The runner's transcript survives a simulated mid-write kill: the
    trailing partial record is discarded on load and the ledger recovers."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    path = Path(runner.ledger.path)
    good = path.read_bytes()
    good_seqs = [r.seq for r in load_transcript(path).records]

    # Simulate a kill mid-write of the NEXT record (torn bytes at EOF).
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"action_result","turn":3,"ref":"unit:10:g')
    load = load_transcript(path)
    assert load.discarded_trailing_partial
    # Every complete record survives the kill; only the torn tail is
    # discarded.
    assert [r.seq for r in load.records] == good_seqs
    valid_seqs = [r.seq for r in load.records]
    assert valid_seqs == sorted(valid_seqs) and len(valid_seqs) > 5

    # Recovery: attach truncates the torn tail and resumes the seq.
    from lechonk.core.ledger import TranscriptLedger

    recovered = TranscriptLedger.attach(path)
    assert recovered.next_seq == valid_seqs[-1] + 1
    recovered.close()
    assert path.read_bytes() == good  # the torn bytes are gone


# ---------------------------------------------------------------------------
# the runner imports the adapter lazily (§7.1)
# ---------------------------------------------------------------------------


def test_runner_imports_the_adapter_only_inside_functions():
    """No module-level lechonk.adapter import in core/runner.py: the
    adapter (and civrealm) import happens lazily, only after the stack is
    up."""
    source = (REPO / "src" / "lechonk" / "core" / "runner.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "lechonk.adapter" \
                        or alias.name.startswith("lechonk.adapter."):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "lechonk.adapter" \
                    or (node.module or "").startswith("lechonk.adapter."):
                offenders.append((node.lineno, node.module))
    assert not offenders, f"module-level adapter imports: {offenders}"


def test_runner_state_machine_and_deadlines_are_aligned_with_3():
    """The §3 supervision constants: ACTING ≤ 180 s, FINISH_SENT/WAITING ≤
    the pingtimeout-derived bound (720 s + margin), per-mutation ≤ 30 s,
    startup ≤ 120 s, teardown ≤ 30 s, stall watchdog 10 min."""
    from lechonk.core.runner import (
        ACTING_DEADLINE_S,
        MUTATION_DEADLINE_S,
        PINGTIMEOUT_S,
        STARTUP_DEADLINE_S,
        STALL_WATCHDOG_S,
        TEARDOWN_DEADLINE_S,
        WAITING_MARGIN_S,
        RunnerConfig,
    )

    assert ACTING_DEADLINE_S == 180.0
    assert PINGTIMEOUT_S == 720.0
    assert WAITING_MARGIN_S == 180.0  # 720 + margin = 900
    assert STARTUP_DEADLINE_S == 120.0
    assert MUTATION_DEADLINE_S == 30.0
    assert TEARDOWN_DEADLINE_S == 30.0
    assert STALL_WATCHDOG_S == 600.0
    config = RunnerConfig()
    assert config.waiting_deadline_s == 900.0
    assert config.acting_deadline_s == 180.0


# ---------------------------------------------------------------------------
# WP3: the client-declared provenance class (paired scripted arms)
# ---------------------------------------------------------------------------


def test_runner_client_provenance_class_passthrough(tmp_path):
    """A client MAY declare ``provenance_class`` (WP3 paired runs drive a
    scripted comparator arm through this same pipeline; its decisions
    are ``scripted_default``, never ``model``). The default when the
    attribute is absent is ``model`` (the pre-existing tests assert that
    path)."""
    class ScriptedClient(FakeModelClient):
        provenance_class = "scripted_default"

    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = ScriptedClient([
        # turn 1 attempt 1: a well-formed but never-offered ref — rejected
        # as state_fog_error, and (batch with actions, none planned) the
        # one repair fires
        ok_response(["unit:999:goto_n"], turn_id="t1", with_memory=False),
        # the one repair (state_fog_error class): a valid action
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    assert executed
    assert all(a.provenance_class == "scripted_default" for a in executed)
    rejected = [a for a in actions if a.outcome == "rejected"]
    assert [a.ref for a in rejected] == ["unit:999:goto_n"]
    assert rejected[0].rejection_class == "state_fog_error"
    assert rejected[0].provenance_class == "scripted_default"
    # The repair prompt was used (the unoffered ref triggered it) and the
    # repair's action executed with the client's provenance class.
    assert len(client.repair_prompts) == 1
    assert client.repair_prompts[0].error_class == "state_fog_error"
    assert client.repair_prompts[0].failed_refs == ["unit:999:goto_n"]
    # No standing orders are tracked for a non-model client (the
    # continued_model_order machinery is model-originated only).
    continued = [a for a in actions
                 if a.provenance_class == "continued_model_order"]
    assert continued == []


def test_runner_provenance_class_default_is_model(tmp_path):
    """The getattr default: a client WITHOUT the attribute labels its
    decisions ``model`` (the pre-existing behavior, pinned here)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    assert not hasattr(client, "provenance_class")
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    executed = [a for a in _actions(_records(runner.ledger.path))
                if a.outcome == "executed"]
    assert executed
    assert all(a.provenance_class == "model" for a in executed)


# ---------------------------------------------------------------------------
# WP4: budget-aware repair gating (the frozen 180 s ACTING deadline holds)
# ---------------------------------------------------------------------------


def test_runner_repair_timeout_within_budget_degrades_and_completes(tmp_path):
    """WP4 gating (a): the repair attempt times out inside the remaining
    ACTING budget (it surfaces as a transport failure on attempt 2 via the
    per-call hard request timeout) — the degraded default applies and the
    turn completes; no deadline exception."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 100 s of the 180 s ACTING budget.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=100.0),
        # The repair times out at its per-call bound (remaining - margin):
        # elapsed ends at 170 s, inside the deadline.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=70.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert "deadline" not in result.reason
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 2, 1]  # t1: 1+2, t2: 1
    assert calls[1].error == "request timed out"
    # Turn 1's decisions came from the degraded default.
    defaults = [a for a in _actions(_records(runner.ledger.path))
                if a.turn == 1 and a.outcome == "executed"]
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.provenance_class == "scripted_default" for a in defaults)
    # The repair prompt carried the remaining budget at repair time
    # (180 - 100 = 80 s) — the client's per-call timeout derives from it.
    assert round(client.repair_prompts[0].budget_s, 6) == 80.0


def test_runner_repair_skipped_below_budget_floor(tmp_path):
    """WP4 gating (b): remaining budget below the 20 s floor → the repair
    call is skipped entirely, the degraded default applies, and the ledger
    records the skip."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 165 s: remaining 15 s < 20 s floor.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=165.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # The repair was never called: one API call in turn 1, no repair prompt.
    assert len(client.repair_prompts) == 0
    calls = _by_kind(_records(runner.ledger.path), ModelCallRecord)
    assert [c.attempt for c in calls] == [1, 1]  # t1 attempt 1, t2 attempt 1
    # Turn 1 executed the degraded default.
    defaults = [a for a in _actions(_records(runner.ledger.path))
                if a.turn == 1 and a.outcome == "executed"]
    assert {a.domain for a in defaults} == {"tech", "city"}
    assert all(a.provenance_class == "scripted_default" for a in defaults)
    # The skip is ledgered with the reason.
    skips = [r for r in _records(runner.ledger.path)
             if isinstance(r, MetricEventRecord)
             and r.name == "repair_skipped_insufficient_budget"]
    assert len(skips) == 1
    assert "15.0s" in skips[0].value


def test_runner_repair_within_budget_proceeds_normally(tmp_path):
    """WP4 gating (c)+(d): a first-attempt transport failure with a repair
    that fits the remaining budget → the repair proceeds exactly as before
    (no behavior change, no skip event)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=30.0),
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    assert len(client.repair_prompts) == 1
    # The repair prompt carried the remaining budget (180 - 30 = 150 s).
    assert round(client.repair_prompts[0].budget_s, 6) == 150.0
    executed = [a for a in _actions(_records(runner.ledger.path))
                if a.outcome == "executed"]
    assert executed[0].ref == "unit:10:goto_e"
    assert executed[0].provenance_class == "model"
    assert executed[0].readback == "verified"
    skips = [r for r in _records(runner.ledger.path)
             if isinstance(r, MetricEventRecord)
             and r.name == "repair_skipped_insufficient_budget"]
    assert skips == []


def test_runner_reports_true_turns_completed_after_mid_turn_death(tmp_path):
    """WP4 fix: a run that dies mid-turn N+1 reports turns_completed == N
    (the pre-fix defect reported the stale local 0 on every exception
    unwind — the truncated wp4-eval model arms reported 0)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        # Turn 2 dies: the model call alone exceeds the ACTING deadline.
        advancing_response(lambda: error_response("request timed out"),
                           clock, seconds=200.0),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    assert result.turns_completed == 1
    # Artifacts intact: the outcome record is ledgered.
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


# ---------------------------------------------------------------------------
# FW-14a (A3): game-over flag plumbing — the acting boundary and the
# mid-batch / phase-boundary classifications
# ---------------------------------------------------------------------------

GOTO_REF = "unit:10:goto_e"
# The raw CivRealm action tuple for the goto_e candidate (action key
# goto_4 = dir8 E — the same table the read-back verifier uses).
GOTO_ACTION = ("unit", 10, "goto_4")


class _ActingBoundaryGameOverSession(FakeSession):
    """The session cache at the ACTING boundary holds the game-over
    state: an EMPTY observation plus the step flags the real
    AdapterSession caches (terminated/truncated + the env's defeat
    signal). The acting-boundary extraction must classify by the flags
    — never raise ValueError into the generic infrastructure handler."""

    def __init__(self, worlds, *, terminated, truncated, defeated,
                 empty_obs=True):
        super().__init__(worlds)
        self._terminated = terminated
        self._truncated = truncated
        self._defeated = defeated
        self._empty_obs = empty_obs

    @property
    def obs(self):
        return {} if self._empty_obs else super().obs

    @property
    def info(self):
        if self._empty_obs:
            return {"turn": 2, "available_actions": {}}
        return super().info

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated


def test_runner_acting_boundary_empty_game_over_obs_defeated(tmp_path):
    """A3a: an empty game-over observation at the acting boundary with
    terminated=True + the env's defeat signal True classifies as
    DEFEATED by the flags — not TRUNCATED "infrastructure failure" (the
    pre-fix ValueError path through the generic handler)."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=True)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert result.reason == "my player is defeated"
    assert "infrastructure failure" not in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "DEFEATED"


def test_runner_acting_boundary_empty_obs_truncated_stays_distinct(tmp_path):
    """The env flag values reach classification: truncated=True (not
    terminated) classifies as TRUNCATED infrastructure — distinct from
    DEFEATED, never confused with a defeat."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=False, truncated=True, defeated=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "DEFEATED"
    assert "truncated=True" in result.reason
    assert "infrastructure failure" not in result.reason


def test_runner_acting_boundary_missing_record_never_false_victory(tmp_path):
    """A3b: an empty game-over observation at the acting boundary with
    terminated=True but NO provable defeat status (no player record, no
    positive env defeat signal) is TRUNCATED/inconclusive — never a
    false VICTORY."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason


def test_runner_acting_boundary_victory_needs_a_live_record(tmp_path):
    """The VICTORY branch is evidence-gated: terminated=True with the
    env's defeat signal False AND a well-formed live player record in
    the observation classifies as VICTORY."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=False,
        empty_obs=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "VICTORY"
    assert "defeated=False" in result.reason


def test_runner_game_over_victory_positive_pin(tmp_path):
    """F5 ruling A: the terminated_game_over branch reached with
    reliable provenance — the env's game-over flags plumbed through the
    session cache (terminated=True, truncated=False, defeat signal
    False) plus a well-formed ALIVE player record classify the run as
    VICTORY."""
    session = _ActingBoundaryGameOverSession(
        [turn_world(1)], terminated=True, truncated=False, defeated=False,
        empty_obs=False)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "VICTORY"
    assert "defeated=False" in result.reason


class _MidBatchEmptyGameOverSession(FakeSession):
    """A FakeSession caching the latest step's observation and game-over
    flags exactly like the real AdapterSession (FW-14a plumbing): the
    game-ending mutation returns an EMPTY observation (the real env's
    game-over observation) with terminated=True. The mid-batch
    classification (runner.py:632) must consult the env's defeat flag —
    never default to not-defeated on a missing player record."""

    def __init__(self, worlds, *, terminate_on, defeated):
        super().__init__(worlds)
        self._terminate_on = terminate_on
        self._defeated = defeated
        self._terminated = False
        self._truncated = False
        self._step_obs = None

    @property
    def obs(self):
        if self._step_obs is not None:
            return self._step_obs
        return super().obs

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated if (self._terminated or self._truncated) else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if action is not None and tuple(action) == self._terminate_on:
            outcome = replace(outcome, observation={}, terminated=True,
                              truncated=False)
        self._step_obs = outcome.observation
        self._terminated = bool(outcome.terminated)
        self._truncated = bool(outcome.truncated)
        return outcome


def test_runner_game_over_mid_batch_empty_obs_defeated(tmp_path):
    """A3b at runner.py:632: a mutation ends the game mid-batch with an
    EMPTY observation; the defeat status is proven by the env's defeat
    flag (True) -> DEFEATED — not a false VICTORY from the missing
    player record."""
    session = _MidBatchEmptyGameOverSession(
        [turn_world(1)], terminate_on=GOTO_ACTION, defeated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert "game over mid-batch (defeated=True)" in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "DEFEATED"


def test_runner_game_over_mid_batch_empty_obs_unknown_never_victory(tmp_path):
    """A3b at runner.py:632: the same empty game-over observation with
    NO provable defeat status (the env's defeat signal False — the
    vendor method returns False both when provably alive and when the
    record is missing) is TRUNCATED/inconclusive — never a false
    VICTORY."""
    session = _MidBatchEmptyGameOverSession(
        [turn_world(1)], terminate_on=GOTO_ACTION, defeated=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason
    assert "game over mid-batch" in result.reason


class _PhaseBoundaryEmptyGameOverSession(FakeSession):
    """The finish step ends the game with an EMPTY observation (the real
    env's game-over observation); the session caches the flags. The
    phase-boundary classification (runner.py:647) must consult the env's
    defeat flag — never default to not-defeated on a missing player
    record."""

    def __init__(self, worlds, *, defeated, truncated=False):
        super().__init__(worlds, final_terminated=not truncated,
                         final_truncated=truncated)
        self._defeated = defeated
        self._terminated = False
        self._truncated = False

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return self._truncated

    @property
    def my_player_is_defeated(self):
        return self._defeated if (self._terminated or self._truncated) else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if outcome.terminated or outcome.truncated:
            outcome = replace(outcome, observation={})
        self._terminated = bool(outcome.terminated)
        self._truncated = bool(outcome.truncated)
        return outcome


def test_runner_game_over_at_phase_boundary_empty_obs_defeated(tmp_path):
    """A3b at runner.py:647: the finish step ends the game with an
    EMPTY observation; the defeat status is proven by the env's defeat
    flag (True) -> DEFEATED — not a false VICTORY from the missing
    player record."""
    session = _PhaseBoundaryEmptyGameOverSession([turn_world(1)],
                                                 defeated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "DEFEATED"
    assert "game over at the phase boundary (defeated=True)" in result.reason


def test_runner_game_over_at_phase_boundary_empty_obs_never_victory(tmp_path):
    """A3b at runner.py:647: the same empty game-over observation with
    NO provable defeat status is TRUNCATED/inconclusive — never a false
    VICTORY."""
    session = _PhaseBoundaryEmptyGameOverSession([turn_world(1)],
                                                 defeated=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.outcome != "VICTORY"
    assert "unknown defeat status" in result.reason


def test_runner_truncated_at_phase_boundary_stays_truncated(tmp_path):
    """The env flag values reach classification at runner.py:647:
    truncated=True (not terminated) stays TRUNCATED infrastructure —
    distinct from DEFEATED, never confused with a defeat."""
    session = _PhaseBoundaryEmptyGameOverSession(
        [turn_world(1)], defeated=False, truncated=True)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=5)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "infrastructure truncation at the phase boundary" in result.reason


# ---------------------------------------------------------------------------
# FW-14a (A5): an INCONCLUSIVE resend read-back quarantines (§3 line 35)
# ---------------------------------------------------------------------------


class _InconclusiveResendSession(LifecycleSession):
    """The resubmission's read-back is INCONCLUSIVE: the reconnect read
    proved non-execution (the resend is permitted), but the resend's own
    post-state can neither verify nor contradict the move (the unit
    vanished server-side). The §3 line 35 quarantine path must fire —
    the run never continues on an inconclusive settlement."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self._goto_sends = 0

    def step(self, action, *, deadline=None):
        if action == GOTO_ACTION:
            self._goto_sends += 1
            if self._goto_sends >= 2:
                # The resend: the unit vanished server-side — the
                # read-back can neither verify nor contradict the move.
                self.calls.append((action, deadline))
                obs, info = self._worlds[self._index]
                new_obs = copy.deepcopy(obs)
                del new_obs["unit"][10]
                self._worlds[self._index] = (new_obs, info)
                self._advance_cache(new_obs, info)
                return StepOutcome(observation=new_obs, reward=0.0,
                                   terminated=False, truncated=False,
                                   info=info, action=action,
                                   elapsed_s=0.01)
        return super().step(action, deadline=deadline)


def test_runner_inconclusive_resend_readback_quarantines(tmp_path):
    """A5: the ONE permitted resubmission's read-back is INCONCLUSIVE
    (the execution state is unknown) — the same quarantine path as an
    exception or a CONTRADICTED resend: the read-back record is
    emitted, a mutation_quarantine metric event is emitted, and the run
    takes the bounded fail-stop (TRUNCATED) — it never continues."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = _InconclusiveResendSession(
        worlds, disconnect_on=GOTO_ACTION, effect_applied=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()

    assert result.outcome == "TRUNCATED"
    assert "quarantined" in result.reason
    assert "read back inconclusive" in result.reason
    assert "never resent" in result.reason
    # Exactly two sends: the failed dispatch + the one resubmission —
    # never a third.
    assert session._goto_sends == 2
    assert len([c for c in session.calls if c[0] == GOTO_ACTION]) == 2

    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 2
    # The uncertain attempt + the resubmission (both sent, both
    # inconclusive) — and nothing else.
    assert all(a.outcome == "executed" for a in by_ref)
    assert all(a.readback == "inconclusive" for a in by_ref)
    assert by_ref[0].provenance_class == "model"
    assert by_ref[1].provenance_class == "continued_model_order"
    assert by_ref[1].origin_seq == by_ref[0].seq
    # The first reconciliation proved non-execution (permitting the one
    # resubmission).
    reconciliations = [r for r in records
                       if isinstance(r, MetricEventRecord)
                       and r.name == "mutation_reconciliation"]
    assert len(reconciliations) == 1
    assert f"{GOTO_REF}: non_execution_verified" in reconciliations[0].value
    # The inconclusive resend read-back took the quarantine path: one
    # mutation_quarantine event naming the inconclusive read-back.
    quarantines = [r for r in records
                   if isinstance(r, MetricEventRecord)
                   and r.name == "mutation_quarantine"]
    assert len(quarantines) == 1
    assert "read back inconclusive" in quarantines[0].value
    assert GOTO_REF in quarantines[0].value
    # Bounded fail-stop with artifacts: the outcome record is ledgered;
    # the run did NOT continue (turn 1 never completed).
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "quarantined" in records[-1].reason
    assert result.turns_completed == 0
    assert session.closed


# ---------------------------------------------------------------------------
# threat mentions: opponent/barbarian conformance (B3, §3 line 41) and the
# ENTITY_ID_RE end anchor (C5 runner-side piece)
# ---------------------------------------------------------------------------


def _threat_snapshot() -> Snapshot:
    """Self is player 0; entities cover own, enemy-owned, owner-unknown
    (fogged foreign), and unresolvable ids."""
    return Snapshot(
        turn=1,
        player=PlayerView(player_id=0),
        map_view=MapView(width=8, height=8),
        units=[
            UnitView(id=10, is_own=True, type_name="Settlers", x=1, y=1),
            UnitView(id=20, is_own=False, owner_id=1,
                     type_name="Warriors", x=2, y=2),
            UnitView(id=30, is_own=False, owner_id=None,
                     type_name="Warriors", x=3, y=3),
        ],
        cities=[
            CityView(id=1, name="Rome", x=1, y=1, is_own=True),
            CityView(id=2, name="Carthage", x=4, y=4, is_own=False,
                     owner_id=1),
        ],
    )


def _threat_runner(tmp_path):
    return make_runner(tmp_path, client=FakeModelClient([]),
                       session=FakeSession([turn_world(1)]))


def test_threat_own_entity_mentions_not_counted(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    mentions = runner._threat_mentions(["unit:10", "city:1"], 1, snap)
    assert mentions == []


def test_threat_self_player_mention_dropped(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    assert runner._threat_mentions(["player:0"], 1, snap) == []


def test_threat_unresolvable_id_dropped(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    assert runner._threat_mentions(["unit:99", "city:99"], 1, snap) == []


def test_threat_opponent_barbarian_mentions_counted(tmp_path):
    runner = _threat_runner(tmp_path)
    snap = _threat_snapshot()
    mentions = runner._threat_mentions(
        ["unit:20", "city:2", "player:1", "unit:30"], 1, snap)
    assert [m.entity_id for m in mentions] == [
        "unit:20", "city:2", "player:1", "unit:30"]
    assert mentions[0].tile == (2, 2)
    assert mentions[1].tile == (4, 4)
    assert mentions[3].tile == (3, 3)


def test_entity_id_re_rejects_trailing_newline():
    assert ENTITY_ID_RE.match("unit:7\n") is None
    assert ENTITY_ID_RE.match("city:8\n") is None
    assert ENTITY_ID_RE.match("unit:7") is not None


# ---------------------------------------------------------------------------
# FW-R2: the CommitmentRecord carries the union of every attempt's
# memory_update (a repair replaces the executed batch, never an earlier
# attempt's commitments/threats)
# ---------------------------------------------------------------------------


def test_runner_two_attempt_turn_unions_memory_updates(tmp_path):
    """A repaired turn's CommitmentRecord carries both attempts'
    memory_update commitments."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        # Attempt 1 emits a commitment, then an unoffered ref -> repair.
        ok_response(["unit:99:goto_n"], turn_id="t1",
                    commitments=[("research", "Alphabet", 3)]),
        # The repair emits a different commitment.
        ok_response(["unit:10:goto_e"], turn_id="t1",
                    commitments=[("expansion", "city:2", 5)]),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target) for c in records[0].commitments] == [
        ("research", "Alphabet"), ("expansion", "city:2")]


def test_runner_repair_budget_skip_still_ledgers_attempt_memory_update(
        tmp_path):
    """FW-R2: the repair-budget-skip path still ledgers attempt 1's
    memory_update — the fabricated degraded result never hides it."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    clock = FakeClock()
    client = FakeModelClient([
        # Attempt 1 burns 165 s of the 180 s ACTING budget: remaining
        # 15 s sits below the 20 s repair floor -> the repair is skipped.
        advancing_response(
            lambda: ok_response(["unit:99:goto_n"], turn_id="t1",
                                commitments=[("research", "Alphabet", 3)]),
            clock, seconds=165.0),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target) for c in records[0].commitments] == [
        ("research", "Alphabet")]


# ---------------------------------------------------------------------------
# FW-16: batch-level ActionBatch.commitments are ingested alongside the
# memory_update ones; an engine-automatic resolution that fulfills a
# commitment credits fulfilled_other on the commitment board
# ---------------------------------------------------------------------------


def test_runner_batch_commitments_ingested_with_memory_update_ones(tmp_path):
    """FW-16: the batch-level ActionBatch.commitments reach the turn's
    CommitmentRecord alongside the memory_update ones — the union of both
    sources, identical commitments deduplicated."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    shared = {"domain": "research", "target": "Alphabet", "deadline_turn": 3}
    batch = ActionBatch(
        turn_id="t1",
        actions=[{"ref": "unit:10:goto_e"}],
        commitments=[shared,
                     {"domain": "expansion", "target": "city:2",
                      "deadline_turn": 5}],
        finish_requested=True,
        memory_update=MemoryUpdate(
            commitments=[shared,
                         {"domain": "military", "target": "unit:20",
                          "deadline_turn": 4}]),
    )
    client = FakeModelClient([
        ModelResponse(call_id="fake-call", prompt_hash="ph", batch=batch,
                      latency_ms=1, tokens_in=10, tokens_out=10,
                      cost_usd=0.0, fingerprint="fp"),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = [r for r in _records(runner.ledger.path)
               if isinstance(r, CommitmentRecord) and r.turn == 1]
    assert len(records) == 1
    assert [(c.domain, c.target, c.deadline_turn)
            for c in records[0].commitments] == [
        ("research", "Alphabet", 3),  # in both sources: deduplicated
        ("military", "unit:20", 4),   # memory_update only
        ("expansion", "city:2", 5),   # batch-level only
    ]


def test_runner_engine_automatic_resolution_credits_fulfilled_other(tmp_path):
    """FW-16: an engine-automatic resolution (readback "none" by
    construction) that fulfills a batch-level commitment produces a
    fulfilled_other — never fulfilled_model — credit on the commitment
    board, end to end."""
    # Turn 1: researching Alphabet, the model moves a unit and commits (at
    # batch level) to the Pottery research. Turn 2: the engine continued
    # research to Pottery (tech 2) during WAITING — no executed tech action
    # of ours -> engine_automatic, which fulfills the commitment.
    worlds = [
        turn_world(1, researching=1),
        turn_world(2, researching=2),
        turn_world(3),
    ]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False,
                    commitments=[("research", "Pottery", 2)]),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    # The batch-level commitment reached the turn's CommitmentRecord.
    commitments = [r for r in records if isinstance(r, CommitmentRecord)]
    assert len(commitments) == 1
    assert [(c.domain, c.target) for c in commitments[0].commitments] == [
        ("research", "Pottery")]
    # The engine-automatic research resolution fulfilled it: a
    # fulfilled_other credit on the board, never fulfilled_model.
    # (W3-9: the record is captured at turn 1's finish-boundary event,
    # and its ref carries the entity identity + the observation index.)
    automatic = [a for a in _actions(records)
                 if a.provenance_class == "engine_automatic"]
    assert len(automatic) == 1
    assert automatic[0].turn == 1
    assert automatic[0].target == "Pottery"
    board = commitment_board(records)
    entry = next(e for e in board if e["target"] == "Pottery")
    assert entry["status"] == "fulfilled_other"
    assert entry["fulfilled_by"] == [automatic[0].ref]
    assert entry["fulfilled_model_by"] == []


# ---------------------------------------------------------------------------
# FW-13: the candidate-table cap in the pipeline + the prompt-budget
# bounded fail-stop
# ---------------------------------------------------------------------------


class _SyntheticTableSession(FakeSession):
    """Serves a fixed synthetic candidate table (oversized for the cap)."""

    def __init__(self, worlds, table):
        super().__init__(worlds)
        self._table = table

    def build_candidate_table(self):
        return self._table


def test_runner_candidates_ref_hashes_the_offered_capped_text(tmp_path):
    """FW-13 acceptance: candidates_ref pins the OFFERED (capped) candidate
    text — sha256 over the capped rendering, with the capped count — not
    the pre-cap table."""
    # 20 unit candidates x ~80 tokens ~ 1600 tokens: over the 1,200 cap.
    table = CandidateTable(turn=1, candidates=[
        Candidate(id=f"unit:9{i:02d}:wait", domain="unit",
                  actor_id=f"9{i:02d}", verb="wait", summary="y" * 300)
        for i in range(20)])
    session = _SyntheticTableSession([turn_world(1), turn_world(2)], table)
    client = FakeModelClient([ok_response([], turn_id="t1",
                                          with_memory=False)])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    ref = _by_kind(_records(runner.ledger.path), CandidatesRefRecord)[0]
    # The same deterministic capped subset the runner computed.
    offered = table.offered()
    assert len(offered.candidates) < 20  # the cap engaged
    assert ref.candidate_count == len(offered.candidates)
    assert ref.by_domain == {"unit": len(offered.candidates)}
    assert ref.table_hash == hashlib.sha256(
        table.render().encode("utf-8")).hexdigest()
    assert ref.table_hash == hashlib.sha256(
        client.turn_prompts[0].candidates.encode("utf-8")).hexdigest()
    # The model-facing prompt carries the capped table + omission line.
    assert "candidates omitted" in client.turn_prompts[0].candidates


def test_runner_candidate_surface_v2_records_additive_admission(tmp_path):
    table = CandidateTable(turn=1, candidates=[
        Candidate(id=f"unit:{i}:wait", domain="unit", actor_id=str(i),
                  verb="wait", summary="u" * 300)
        for i in range(8)
    ] + [
        Candidate(id="tech:cur_player:research_tech_alphabet_0",
                  domain="tech", actor_id="cur_player", verb="research",
                  summary="research alphabet")
    ])
    session = _SyntheticTableSession([turn_world(1), turn_world(2)], table)
    client = FakeModelClient([ok_response([], turn_id="t1", with_memory=False)])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={
            "prompt_profile": "gameplay_v3",
            "candidate_surface_policy": "candidate_surface_v2",
            "prompt_bundle_id": "gameplay_v3",
            "recovery_config_digest_schema": "interactive-config-v3",
            "candidate_surface_cap": 8000,
            "candidate_admission_version": "candidate_surface_v2-r1",
            "compaction_schema": "conversation-compaction-v1",
            "prompt_budget_schema": "gameplay-v3-budget-v1",
        })
    assert runner.run().outcome == "COMPLETED"
    ref = _by_kind(_records(runner.ledger.path), CandidatesRefRecord)[0]
    assert ref.candidate_surface_policy == "candidate_surface_v2"
    assert ref.candidate_cap_tokens == 8000
    assert ref.total_candidate_count == 9
    assert ref.offered_candidate_count == ref.candidate_count
    assert ref.omitted_candidate_count == 0
    assert ref.total_by_domain["unit"] == 8
    assert ref.offered_by_domain["tech"] == 1
    assert ref.omitted_by_domain["unit"] == 0
    assert ref.total_tokens is not None
    assert ref.offered_tokens is not None
    assert "research alphabet" in client.turn_prompts[0].candidates


def test_runner_v2_uses_admission_render_for_large_protected_surface(tmp_path):
    """A protected surface between the legacy and v2 caps reaches the model.

    The ledger hash must pin the exact expanded text sent in the prompt; it
    must never re-render the admitted table through CandidateTable's legacy
    1,200-token cap.
    """
    from lechonk.core.candidate_surface_v2 import admit_candidate_surface

    candidate = Candidate(
        id="unit:10:sentry", domain="unit", actor_id="10", verb="sentry",
        summary="p" * 6200, in_contact=True)
    table = CandidateTable(turn=1, candidates=[candidate])
    admission = admit_candidate_surface(table)
    assert 1200 < admission.offered_tokens < 8000
    session = _SyntheticTableSession([turn_world(1), turn_world(2)], table)
    client = FakeModelClient([ok_response([], turn_id="t1", with_memory=False)])
    runner = make_runner(
        tmp_path, client=client, session=session, max_turns=1,
        config_overrides={
            "prompt_profile": "gameplay_v3",
            "candidate_surface_policy": "candidate_surface_v2",
            "prompt_bundle_id": "gameplay_v3",
            "recovery_config_digest_schema": "interactive-config-v3",
            "candidate_surface_cap": 8000,
            "candidate_admission_version": "candidate_surface_v2-r1",
            "compaction_schema": "conversation-compaction-v1",
            "prompt_budget_schema": "gameplay-v3-budget-v1",
        })
    assert runner.run().outcome == "COMPLETED"
    ref = _by_kind(_records(runner.ledger.path), CandidatesRefRecord)[0]
    expected = admission.render()
    assert ref.table_hash == hashlib.sha256(expected.encode()).hexdigest()
    assert client.turn_prompts[0].candidates == expected


def test_runner_prompt_budget_exceeded_bounded_fail_stop(tmp_path):
    """FW-13 failure policy: a prompt-budget failure is the dedicated
    bounded fail-stop — the prompt_budget_exceeded metric event, the
    ACTING->FAILED transition, and a TRUNCATED outcome whose reason names
    the budget — never a generic infrastructure truncation."""
    worlds = [turn_world(1), turn_world(2)]
    session = FakeSession(worlds)

    def over_budget(prompt):
        raise PromptBudgetExceeded(
            "assembled turn prompt exceeds its budget: 11000 > 10200 tokens")

    client = FakeModelClient([over_budget])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.reason.startswith("prompt budget exceeded")
    records = _records(runner.ledger.path)
    budget_events = [r for r in _by_kind(records, MetricEventRecord)
                     if r.name == "prompt_budget_exceeded"]
    assert len(budget_events) == 1
    assert "assembled turn prompt" in budget_events[0].value
    transitions = [r.value for r in _by_kind(records, MetricEventRecord)
                   if r.name == "state_transition"]
    assert "ACTING->FAILED" in transitions
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert records[-1].reason.startswith("prompt budget exceeded")
    assert "infrastructure" not in records[-1].reason.lower()
    # Bounded: the failed turn never reached a model_call record.
    assert _by_kind(records, ModelCallRecord) == []
    assert session.closed


# ---------------------------------------------------------------------------
# W2-3: pre-dispatch error classification + the run-loop cap catch
# ---------------------------------------------------------------------------


class _UnmappableTableSession(_SyntheticTableSession):
    """Serves a synthetic table AND fails loudly if any real mutation is
    sent (the pipeline-owned finish step, action None, still works)."""

    def step(self, action, *, deadline=None):
        if action is not None:
            raise AssertionError(
                "a mutation was sent for an unmappable candidate")
        return super().step(action, deadline=deadline)


def test_runner_pre_dispatch_mapping_failure_rejected_not_sent(tmp_path):
    """W2-3: a candidate that cannot be mapped to an action tuple is
    classified BEFORE any send — outcome rejected / schema_error /
    readback none — never ledgered as 'send failed after dispatch',
    never reconciled, never resent; the turn still completes."""
    table = CandidateTable(turn=1, candidates=[
        Candidate(id="unit:10:sentry", domain="unit", actor_id="10",
                  verb="sentry", summary="sentry (payload-less)")])
    session = _UnmappableTableSession([turn_world(1), turn_world(2)], table)
    client = FakeModelClient([
        ok_response(["unit:10:sentry"], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    by_ref = [a for a in _actions(records) if a.ref == "unit:10:sentry"]
    assert len(by_ref) == 1
    assert by_ref[0].outcome == "rejected"
    assert by_ref[0].rejection_class == "schema_error"
    assert by_ref[0].readback == "none"
    assert "pre-dispatch mapping failed" in by_ref[0].detail
    assert "after dispatch" not in by_ref[0].detail
    # No reconciliation, no quarantine: nothing was ever sent.
    assert [r for r in records
            if isinstance(r, MetricEventRecord)
            and r.name in ("mutation_reconciliation",
                           "mutation_quarantine")] == []
    # The session-level guard never fired: only the pipeline-owned
    # finish steps (action None) reached the session — no send, no resend.
    assert session.calls
    assert all(c[0] is None for c in session.calls)


def test_runner_resend_path_never_resends_unmappable_candidate(tmp_path):
    """W2-3: the resend path itself refuses an unmappable candidate —
    the bounded fail-stop (never a resubmission) and no send reaches
    the session."""
    session = FakeSession([turn_world(1), turn_world(2)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    adapter = runner._load_adapter()
    cand = Candidate(id="unit:10:sentry", domain="unit", actor_id="10",
                     verb="sentry")  # payload-less: unmappable
    with pytest.raises(_QuarantineFailStop, match="never resent"):
        runner._resend_verified_non_execution(
            adapter, cand, turn=1, provenance="model", snapshot_seq=0,
            candidates_seq=0, origin_seq=1, target=None, affected=None,
            actor_tile=None)
    assert [c for c in session.calls if c[0] is not None] == []


def test_runner_candidate_table_cap_exceeded_bounded_fail_stop(tmp_path):
    """W2-3: a candidate-table cap overflow (armed-in-contact lines plus
    the mandatory overhead exceeding the budget) is the dedicated bounded
    fail-stop — the candidate_table_cap_exceeded metric event, the
    ACTING->FAILED transition, and a TRUNCATED outcome naming the cap —
    never a generic infrastructure truncation."""
    table = CandidateTable(turn=1, candidates=[
        Candidate(id="unit:10:sentry", domain="unit", actor_id="10",
                  verb="sentry", summary="a" * 8000, in_contact=True)])
    session = _SyntheticTableSession([turn_world(1), turn_world(2)], table)
    client = FakeModelClient([])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=1)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert result.reason.startswith("candidate table cap exceeded")
    records = _records(runner.ledger.path)
    cap_events = [r for r in _by_kind(records, MetricEventRecord)
                  if r.name == "candidate_table_cap_exceeded"]
    assert len(cap_events) == 1
    assert "candidate-table cap" in cap_events[0].value
    transitions = [r.value for r in _by_kind(records, MetricEventRecord)
                   if r.name == "state_transition"]
    assert "ACTING->FAILED" in transitions
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"
    assert "infrastructure" not in records[-1].reason.lower()
    # Bounded: the failed turn never reached a model_call record.
    assert _by_kind(records, ModelCallRecord) == []
    assert session.closed


# ---------------------------------------------------------------------------
# W1-4: reason accounting — no "no actions" on success; dead origin_seqs gone
# ---------------------------------------------------------------------------


def test_runner_successful_batch_carries_no_misleading_reason(tmp_path):
    """W1-4: a successful planned batch's executed action_results carry no
    reason ("" — the record default); "no actions" is never ledgered for a
    response whose actions were planned and executed."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    executed = [a for a in _actions(records) if a.outcome == "executed"]
    assert {a.ref for a in executed} == {"unit:10:goto_e"}
    assert all(a.provenance_class == "model" for a in executed)
    assert all(a.reason == "" for a in executed)
    assert all(a.reason != "no actions" for a in executed)


def test_runner_why_unusable_success_and_empty_batch(tmp_path):
    """W1-4: _why_unusable returns "" for a usable response (planned
    actions) and "no actions" only for a genuinely empty batch."""
    session = FakeSession([turn_world(1), turn_world(2)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    table = session.build_candidate_table()
    resp_ok = ok_response(["unit:10:goto_e"], turn_id="t1", with_memory=False)
    result_ok = runner._ingest(resp_ok, turn=1, table=table, call_seq=0,
                               snapshot_seq=0, candidates_seq=0)
    assert result_ok.planned
    assert runner._why_unusable(resp_ok, result_ok) == ""
    resp_empty = ok_response([], turn_id="t1", with_memory=False)
    result_empty = runner._ingest(resp_empty, turn=1, table=table,
                                  call_seq=0, snapshot_seq=0, candidates_seq=0)
    assert not result_empty.planned
    assert not result_empty.batch.actions
    assert runner._why_unusable(resp_empty, result_empty) == "no actions"


def test_execute_planned_has_no_origin_seqs_parameter():
    """W1-4: the dead origin_seqs parameter is removed (no caller ever
    passed it; origin_seq stays None on _execute_one)."""
    params = inspect.signature(GameRunner._execute_planned).parameters
    assert "origin_seqs" not in params


# ---------------------------------------------------------------------------
# W2-1: capped-set (offered) membership — ingest, standing orders, the
# degraded default, and the shared contact protection
# ---------------------------------------------------------------------------


def _over_cap_table(turn, real_candidates):
    """A table over the 1,200-token cap: thirteen ~91-token filler
    candidates admitted first, the real fixture candidates appended after
    the budget is exhausted (elided — armed-in-contact candidates
    excepted, once the contact protection has marked them)."""
    fillers = [
        Candidate(id=f"unit:9{i:02d}:wait", domain="unit",
                  actor_id=f"9{i:02d}", verb="wait", summary="x" * 348)
        for i in range(13)]
    return CandidateTable(turn=turn, candidates=fillers + list(real_candidates))


def _fixture_snapshot(session):
    """The snapshot the runner would build for the fake session's current
    world (the real read-only extraction)."""
    from lechonk.adapter.obs_extract import extract_snapshot
    extraction = extract_snapshot(
        session.obs, session.info, my_player_id=session.player_id,
        ruleset=session.ruleset, action_probs=session.action_probs)
    assert extraction.snapshot is not None
    return extraction.snapshot


def test_runner_ingest_rejects_cap_elided_ref_as_state_fog_error(tmp_path):
    """W2-1: ingest validates against the OFFERED (capped) set — a
    well-formed ref to a cap-elided candidate is state the model could not
    see: rejected state_fog_error, never planned."""
    session = FakeSession([turn_world(1), turn_world(2)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    real = session.build_candidate_table()
    elided_ref = "unit:10:goto_e"
    big = _over_cap_table(1, real.candidates)
    # The cap genuinely engaged: the ref is in the FULL table but elided.
    assert elided_ref in real.by_id()
    assert elided_ref not in big.offered().by_id()

    resp = ok_response([elided_ref], turn_id="t1", with_memory=False)
    result = runner._ingest(resp, turn=1, table=big, call_seq=0,
                            snapshot_seq=0, candidates_seq=0)
    assert result.invalid == [(elided_ref, "state_fog_error")]
    assert not result.planned
    rejected = [a for a in _actions(_records(runner.ledger.path))
                if a.outcome == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].rejection_class == "state_fog_error"
    assert "cap-elided" in rejected[0].detail

    # Control: the SAME ref against the under-cap table is offered and
    # plans normally — only the cap elides it.
    result_ok = runner._ingest(
        ok_response([elided_ref], turn_id="t1", with_memory=False),
        turn=1, table=real, call_seq=0, snapshot_seq=0, candidates_seq=0)
    assert [p.ref for p in result_ok.planned] == [elided_ref]


def test_runner_standing_order_cap_elided_ref_is_a_lookup_miss(tmp_path):
    """W2-1: the standing-order re-issue resolves against the OFFERED set —
    a cap-elided ref is a lookup miss (re-resolve within the offered set,
    else drop): never executed from the uncapped table, never a
    state_fog_error."""
    worlds = [turn_world(1),
              turn_world(2, unit_overrides=[(11, {"activity": 0})])]
    session = FakeSession(worlds)
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    adapter = runner._load_adapter()
    runner._session = session  # direct-call tests: run() never ran
    session._index = 1  # the turn-2 world: unit 11 idle, order not in effect
    snapshot = _fixture_snapshot(session)
    real = session.build_candidate_table()
    assert "unit:11:sentry" in real.by_id()
    big = _over_cap_table(2, real.candidates)
    assert "unit:11:sentry" not in big.offered().by_id()  # cap-elided

    runner._standing["11"] = _StandingOrder(
        actor_id="11", verb="sentry", ref="unit:11:sentry", origin_seq=1)
    runner._continue_standing_orders(
        adapter, turn=2, table=big, snapshot=snapshot,
        acted_actors=set(), snapshot_seq=0, candidates_seq=0)
    # Lookup miss: the order was dropped, nothing was executed from the
    # uncapped table, and no state_fog_error was emitted.
    assert "11" not in runner._standing
    assert not [c for c in session.calls if c[0] == ("unit", 11, "sentry")]
    assert not [a for a in _actions(_records(runner.ledger.path))
                if a.rejection_class == "state_fog_error"]

    # Control: the same order against the under-cap table (the ref IS
    # offered) re-issues once as continued_model_order.
    runner._standing["11"] = _StandingOrder(
        actor_id="11", verb="sentry", ref="unit:11:sentry", origin_seq=1)
    runner._continue_standing_orders(
        adapter, turn=2, table=real, snapshot=snapshot,
        acted_actors=set(), snapshot_seq=0, candidates_seq=0)
    assert [c for c in session.calls if c[0] == ("unit", 11, "sentry")]
    continuations = [a for a in _actions(_records(runner.ledger.path))
                     if a.provenance_class == "continued_model_order"]
    assert len(continuations) == 1
    assert continuations[0].ref == "unit:11:sentry"
    assert continuations[0].readback == "verified"


def test_runner_degraded_default_selects_only_from_offered_set(tmp_path):
    """W2-1: the degraded default selects from the OFFERED set — a
    priority-list pick whose candidate the cap elided is never executed;
    over the full surface the same pick executes (control)."""
    session = FakeSession([turn_world(1), turn_world(2)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    adapter = runner._load_adapter()
    runner._session = session  # direct-call tests: run() never ran
    snapshot = _fixture_snapshot(session)
    real = session.build_candidate_table()
    big = _over_cap_table(1, real.candidates)
    tech_refs = {c.id for c in real.candidates if c.domain == "tech"}
    assert tech_refs
    assert not (tech_refs & set(big.offered().by_id()))  # all elided

    executed_big = runner._execute_degraded(
        adapter, turn=1, table=big, snapshot=snapshot,
        snapshot_seq=0, candidates_seq=0, why="test: degraded")
    assert executed_big == set()
    offered_ids = set(big.offered().by_id())
    for a in _actions(_records(runner.ledger.path)):
        if a.outcome == "executed":
            assert a.ref in offered_ids

    # Control: over the full (under-cap) surface the same default executes
    # the research pick — the cap was the only difference.
    executed_small = runner._execute_degraded(
        adapter, turn=1, table=real, snapshot=snapshot,
        snapshot_seq=0, candidates_seq=0, why="test: degraded")
    assert executed_small
    small_offered = set(real.offered().by_id())
    for a in _actions(_records(runner.ledger.path)):
        if a.outcome == "executed" and a.provenance_class == "scripted_default":
            assert a.ref in small_offered
    assert [a for a in _actions(_records(runner.ledger.path))
            if a.outcome == "executed" and a.domain == "tech"]


def test_mark_armed_in_contact_protects_offered_membership():
    """W2-1/FW-13: mark_armed_in_contact is the shared contact protection —
    an armed-in-contact unit's candidates stay in the offered set past the
    cap; without the marking the same candidates elide."""
    contact_unit = UnitView(id=7, owner_id=1, is_own=True,
                            type_name="Warriors", x=2, y=2, in_contact=True)
    snapshot = Snapshot(turn=1, player=PlayerView(player_id=1),
                        map_view=MapView(width=4, height=4),
                        units=[contact_unit])
    late = [
        Candidate(id="unit:7:attack_e", domain="unit", actor_id="7",
                  verb="attack_e", summary="attack east"),
        Candidate(id="unit:7:fortify", domain="unit", actor_id="7",
                  verb="fortify", summary="fortify"),
    ]
    big = _over_cap_table(1, late)
    # Unmarked: the cap elides unit 7's candidates.
    assert not {"unit:7:attack_e", "unit:7:fortify"} \
        <= set(big.offered().by_id())
    mark_armed_in_contact(big, snapshot)
    assert {"unit:7:attack_e", "unit:7:fortify"} \
        <= set(big.offered().by_id())


# ---------------------------------------------------------------------------
# W2-4: stop all dispatch paths on game termination (claim 9)
# ---------------------------------------------------------------------------


class _GameEndingContinuationSession(FakeSession):
    """The standing-order continuation ends the game: the SECOND send of
    the named action (the turn-1 model order is the first) returns
    terminated=True — the game-ending continuation injection."""

    def __init__(self, worlds, *, terminate_on):
        super().__init__(worlds)
        self._terminate_on = terminate_on
        self._terminate_sends = 0

    def step(self, action, *, deadline=None):
        if action is not None and tuple(action) == self._terminate_on:
            self._terminate_sends += 1
            if self._terminate_sends >= 2:
                outcome = super().step(action, deadline=deadline)
                return replace(outcome, terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


class _GameOverOnResearchSession(FakeSession):
    """The degraded default's research pick ends the game (terminated=True)
    as its step outcome — the game-over-mid-degraded-group injection."""

    def step(self, action, *, deadline=None):
        if action is not None and action[0] == "tech" \
                and str(action[2]).startswith("research_tech_"):
            outcome = super().step(action, deadline=deadline)
            return replace(outcome, terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


def test_runner_game_over_mid_batch_skips_standing_order_continuation(tmp_path):
    """W2-4(a): an ordinary batch that ends the game mid-batch guards the
    standing-order continuation entry — a continuable standing order
    pending from turn 1 is never re-issued (no dispatch follows a dead
    game)."""
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(11, {"activity": 0})]),
        turn_world(3),
    ]
    session = _GameOverMidBatchSession(worlds,
                                       terminate_on=("unit", 10, "goto_4"))
    client = FakeModelClient([
        ok_response(["unit:11:sentry"], turn_id="t1", with_memory=False),
        ok_response(["unit:10:goto_e"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "VICTORY"
    assert "game over mid-batch" in result.reason
    actions = _actions(_records(runner.ledger.path))
    # The turn-1 standing order for unit 11 was continuable (the unit is
    # idle again and the turn-2 batch holds no action for it) but the
    # continuation never ran.
    assert not [a for a in actions
                if a.provenance_class == "continued_model_order"]
    # Unit 11's sentry was sent exactly once (the turn-1 model order).
    assert len([c for c in session.calls
                if c[0] == ("unit", 11, "sentry")]) == 1


def test_runner_game_over_mid_degraded_group_stops_next_group(tmp_path):
    """W2-4(b): the degraded group loop stops before the next group once
    the game ended mid-group — the pending city-production default is
    never dispatched after the research pick terminated the game."""
    session = _GameOverOnResearchSession([turn_world(1)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    adapter = runner._load_adapter()
    runner._session = session  # direct-call tests: run() never ran
    snapshot = _fixture_snapshot(session)
    table = session.build_candidate_table()
    # Premise: the degraded default resolves BOTH mandatory choices
    # (research completed + city production empty) — two groups with two
    # distinct actors, the research pick first.
    from lechonk.comparator import defaults as defaults_mod
    log = defaults_mod.decide(snapshot, table.offered())
    assert [d.domain for d in log.decisions] == ["tech", "city"]

    acted = runner._execute_degraded(
        adapter, turn=1, table=table, snapshot=snapshot,
        snapshot_seq=0, candidates_seq=0, why="test: degraded")
    # The first group (the research pick) acted and ended the game.
    assert acted == {"cur_player"}
    assert runner._game_over == (True, False)
    # The second group (the city production pick) was never dispatched:
    # no city send, and the only scripted_default record is the research.
    assert not [c for c in session.calls
                if c[0] is not None and c[0][0] == "city"]
    scripted = [a for a in _actions(_records(runner.ledger.path))
                if a.provenance_class == "scripted_default"]
    assert [a.domain for a in scripted] == ["tech"]


def test_runner_game_ending_continuation_breaks_and_marks_game_over(tmp_path):
    """W2-4(c): a standing-order continuation whose result carries
    terminated breaks the continuation loop AND marks game over — the
    next standing order is never dispatched, and the turn ends through
    the game-over path (no finish primitive)."""
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(10, {"activity": 0}),
                                      (11, {"activity": 0})]),
        turn_world(3),
    ]
    session = _GameEndingContinuationSession(
        worlds, terminate_on=("unit", 10, "sentry"))
    client = FakeModelClient([
        ok_response(["unit:10:sentry", "unit:11:sentry"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "VICTORY"
    assert "game over mid-batch" in result.reason
    actions = _actions(_records(runner.ledger.path))
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    # Only unit 10's continuation was dispatched — the game ended on it,
    # so unit 11's standing order was never re-issued.
    assert [a.ref for a in continuations] == ["unit:10:sentry"]
    assert continuations[0].turn == 2
    assert continuations[0].outcome == "executed"
    assert continuations[0].readback == "verified"
    # The sends: unit 10 sentried twice (model order + continuation),
    # unit 11 exactly once (its model order; no continuation).
    assert len([c for c in session.calls
                if c[0] == ("unit", 10, "sentry")]) == 2
    assert len([c for c in session.calls
                if c[0] == ("unit", 11, "sentry")]) == 1
    # No finish primitive after the game ended mid-continuation (only
    # turn 1's finish advanced the phase).
    assert len([c for c in session.calls if c[0] is None]) == 1


def test_runner_game_ending_resend_marks_game_over(tmp_path):
    """W2-4(d): a verified resubmission whose result carries terminated
    marks game over at the resend site — the termination state is set
    before the result flows back to any caller."""
    session = _GameOverMidBatchSession([turn_world(1)],
                                       terminate_on=GOTO_ACTION)
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    adapter = runner._load_adapter()
    runner._session = session  # direct-call tests: run() never ran
    cand = session.build_candidate_table().by_id()[GOTO_REF]
    result, _seq = runner._resend_verified_non_execution(
        adapter, cand, turn=1, provenance="model", snapshot_seq=0,
        candidates_seq=0, origin_seq=0, target=None, affected=None,
        actor_tile=None)
    assert result.terminated and not result.truncated
    assert result.readback.status == "verified"
    assert runner._game_over == (True, False)


# ---------------------------------------------------------------------------
# W3-11 (D11): record-authoritative defeat precedence at EVERY boundary
# ---------------------------------------------------------------------------


def _d11_player_alive(obs):
    obs["player"][MY_PLAYER]["is_alive"] = True


def _d11_player_dead(obs):
    obs["player"][MY_PLAYER]["is_alive"] = False


def _d11_player_missing(obs):
    # The player domain carries no entry for my player at all.
    obs["player"] = {ENEMY: obs["player"][ENEMY]}


def _d11_player_malformed(obs):
    # A malformed my-player entry: no well-formed is_alive record AND a
    # non-numeric researching id — _extract_player's int(researching_id)
    # raises mid-parsing, so full snapshot construction cannot complete.
    obs["player"][MY_PLAYER] = {"player_id": MY_PLAYER, "name": "Caesar",
                                "researching": "not-a-tech-id"}


def _d11_player_malformed_alive(obs):
    # A well-formed ALIVE record plus a malformed field: the record wins
    # the classification even though snapshot construction raises.
    obs["player"][MY_PLAYER] = {"player_id": MY_PLAYER, "name": "Caesar",
                                "is_alive": True,
                                "researching": "not-a-tech-id"}


def _d11_player_string_key(obs, *, alive):
    # The player domain keyed by the SERIALIZED-STRING player id (a JSON
    # round-trip), as int-keyed observations do not always survive.
    me = obs["player"].pop(MY_PLAYER)
    me["is_alive"] = alive
    obs["player"][str(MY_PLAYER)] = me


_D11_CASES = [
    pytest.param(_d11_player_alive, True, "VICTORY",
                 id="alive-record_flag-true_record-wins"),
    pytest.param(_d11_player_dead, False, "DEFEATED",
                 id="dead-record_flag-false"),
    pytest.param(_d11_player_dead, True, "DEFEATED",
                 id="dead-record_flag-true"),
    pytest.param(_d11_player_missing, True, "DEFEATED",
                 id="missing-record_flag-true_positive-fallback"),
    pytest.param(_d11_player_missing, False, "TRUNCATED",
                 id="missing-record_flag-false_never-victory"),
    pytest.param(_d11_player_missing, None, "TRUNCATED",
                 id="missing-record_flag-none_never-victory"),
    pytest.param(_d11_player_malformed, True, "DEFEATED",
                 id="malformed-entry_flag-true_graceful"),
    pytest.param(_d11_player_malformed, False, "TRUNCATED",
                 id="malformed-entry_flag-false_graceful"),
    pytest.param(_d11_player_malformed_alive, True, "VICTORY",
                 id="malformed-entry_alive-record_flag-true_record-wins"),
    pytest.param(lambda obs: _d11_player_string_key(obs, alive=False),
                 False, "DEFEATED", id="string-key_dead-record"),
    pytest.param(lambda obs: _d11_player_string_key(obs, alive=True),
                 True, "VICTORY",
                 id="string-key_alive-record_flag-true_record-wins"),
]


class _D11TurnStartSession(FakeSession):
    """The session cache at the ACTING boundary holds the game-over
    state with the case's TERMINAL observation (non-empty, possibly
    malformed): the turn-start classification must be
    record-authoritative (D11) and a malformed player entry must never
    raise into the generic infra handler."""

    def __init__(self, worlds, *, player_mutator, env_defeated):
        super().__init__(worlds)
        terminal_obs = copy.deepcopy(worlds[0][0])
        player_mutator(terminal_obs)
        self._terminal_obs = terminal_obs
        self._env_defeated = env_defeated

    @property
    def obs(self):
        return self._terminal_obs

    @property
    def terminated(self):
        return True

    @property
    def truncated(self):
        return False

    @property
    def my_player_is_defeated(self):
        return self._env_defeated


class _D11MidBatchSession(FakeSession):
    """The named mutation ends the game mid-batch: its step returns the
    case's terminal observation with terminated=True and the session
    caches the env's defeat flag — the mid-batch classification must be
    record-authoritative (D11)."""

    def __init__(self, worlds, *, player_mutator, env_defeated):
        super().__init__(worlds)
        self._mutator = player_mutator
        self._env_defeated = env_defeated
        self._terminated = False
        self._terminal_obs = None

    @property
    def obs(self):
        if self._terminal_obs is not None:
            return self._terminal_obs
        return super().obs

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return False

    @property
    def my_player_is_defeated(self):
        return self._env_defeated if self._terminated else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if action is not None and tuple(action) == GOTO_ACTION:
            terminal_obs = copy.deepcopy(outcome.observation)
            self._mutator(terminal_obs)
            self._terminal_obs = terminal_obs
            self._terminated = True
            outcome = replace(outcome, observation=terminal_obs,
                              terminated=True, truncated=False)
        return outcome


class _D11ContinuationSession(FakeSession):
    """The standing-order continuation ends the game: the SECOND send of
    the sentry action (the turn-1 model order is the first) returns the
    case's terminal observation with terminated=True; the session caches
    the env's defeat flag."""

    def __init__(self, worlds, *, player_mutator, env_defeated):
        super().__init__(worlds)
        self._mutator = player_mutator
        self._env_defeated = env_defeated
        self._terminated = False
        self._terminal_obs = None
        self._sentry_sends = 0

    @property
    def obs(self):
        if self._terminal_obs is not None:
            return self._terminal_obs
        return super().obs

    @property
    def terminated(self):
        return self._terminated

    @property
    def truncated(self):
        return False

    @property
    def my_player_is_defeated(self):
        return self._env_defeated if self._terminated else None

    def step(self, action, *, deadline=None):
        if action is not None and tuple(action) == ("unit", 10, "sentry"):
            self._sentry_sends += 1
            if self._sentry_sends >= 2:
                outcome = super().step(action, deadline=deadline)
                terminal_obs = copy.deepcopy(outcome.observation)
                self._mutator(terminal_obs)
                self._terminal_obs = terminal_obs
                self._terminated = True
                return replace(outcome, observation=terminal_obs,
                               terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


class _D11ResendSession(LifecycleSession):
    """The verified resubmission ends the game: the first goto send
    disconnects with the effect NOT applied (reconciliation proves
    non-execution, permitting the one resend); the resend's step returns
    the case's terminal observation with terminated=True."""

    def __init__(self, worlds, *, player_mutator, env_defeated):
        super().__init__(worlds, disconnect_on=GOTO_ACTION,
                         effect_applied=False)
        self._mutator = player_mutator
        self._env_defeated = env_defeated
        self._terminated = False
        self._goto_sends = 0

    @property
    def my_player_is_defeated(self):
        return self._env_defeated if self._terminated else None

    def step(self, action, *, deadline=None):
        if action == GOTO_ACTION:
            self._goto_sends += 1
            if self._goto_sends >= 2:
                # The resend: the game ends with the terminal observation.
                outcome = super().step(action, deadline=deadline)
                terminal_obs = copy.deepcopy(outcome.observation)
                self._mutator(terminal_obs)
                self._advance_cache(terminal_obs, outcome.info)
                self._terminated = True
                return replace(outcome, observation=terminal_obs,
                               terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


class _D11FinishSession(FakeSession):
    """The finish step ends the game with the case's terminal observation
    (non-empty here, unlike the real env's empty game-over observation);
    the session caches the env's defeat flag — the phase-boundary
    classification must be record-authoritative (D11)."""

    def __init__(self, worlds, *, player_mutator, env_defeated):
        super().__init__(worlds)
        self._mutator = player_mutator
        self._env_defeated = env_defeated
        self._terminated = False

    @property
    def my_player_is_defeated(self):
        return self._env_defeated if self._terminated else None

    def step(self, action, *, deadline=None):
        outcome = super().step(action, deadline=deadline)
        if action is None and outcome.terminated:
            terminal_obs = copy.deepcopy(outcome.observation)
            self._mutator(terminal_obs)
            self._terminated = True
            outcome = replace(outcome, observation=terminal_obs)
        return outcome


@pytest.mark.parametrize("boundary", [
    "turn_start", "mid_batch", "standing_continuation", "verified_resend",
    "finish",
])
@pytest.mark.parametrize("player_mutator,env_defeated,expected", _D11_CASES)
def test_runner_d11_record_authoritative_at_every_boundary(
        tmp_path, boundary, player_mutator, env_defeated, expected):
    """W3-11 (D11): at EVERY game-over boundary (turn-start, an ordinary
    mid-batch mutation, a standing-order continuation, a verified resend,
    the finish phase) the defeat verdict is record-authoritative — a
    well-formed player record decides (an alive record OVERRIDES a
    positive env defeat flag; a dead record is DEFEATED), the env flag is
    a positive-only fallback when the record is missing or malformed,
    and an unknown status is TRUNCATED — never a false VICTORY. A
    malformed player entry in a terminal observation never raises into
    the generic infra handler, and serialized-string player IDs are
    honored."""
    if boundary == "turn_start":
        session = _D11TurnStartSession(
            [turn_world(1)], player_mutator=player_mutator,
            env_defeated=env_defeated)
        client = FakeModelClient([])
        max_turns = 1
    elif boundary == "mid_batch":
        session = _D11MidBatchSession(
            [turn_world(1)], player_mutator=player_mutator,
            env_defeated=env_defeated)
        client = FakeModelClient([
            ok_response([GOTO_REF], turn_id="t1", with_memory=False)])
        max_turns = 1
    elif boundary == "standing_continuation":
        worlds = [
            turn_world(1),
            turn_world(2, unit_overrides=[(10, {"activity": 0}),
                                          (11, {"activity": 0})]),
            turn_world(3),
        ]
        session = _D11ContinuationSession(
            worlds, player_mutator=player_mutator, env_defeated=env_defeated)
        client = FakeModelClient([
            ok_response(["unit:10:sentry", "unit:11:sentry"],
                        turn_id="t1", with_memory=False),
            ok_response([], turn_id="t2", with_memory=False),
        ])
        max_turns = 2
    elif boundary == "verified_resend":
        session = _D11ResendSession(
            [turn_world(1)], player_mutator=player_mutator,
            env_defeated=env_defeated)
        client = FakeModelClient([
            ok_response([GOTO_REF], turn_id="t1", with_memory=False)])
        max_turns = 2
    else:  # finish
        session = _D11FinishSession(
            [turn_world(1)], player_mutator=player_mutator,
            env_defeated=env_defeated)
        client = FakeModelClient([
            ok_response([GOTO_REF], turn_id="t1", with_memory=False)])
        max_turns = 5
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=max_turns)
    result = runner.run()
    assert result.outcome == expected
    # The classification is graceful at every boundary: never the generic
    # infrastructure handler, never a false VICTORY.
    assert "infrastructure failure" not in result.reason
    if expected == "VICTORY":
        assert "defeated=False" in result.reason
    elif expected == "DEFEATED":
        assert ("my player is defeated" in result.reason
                or "defeated=True" in result.reason)
    else:
        assert "unknown defeat status" in result.reason
    records = _records(runner.ledger.path)
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == expected


def test_d11_my_player_defeated_never_raises_on_malformed_obs(tmp_path):
    """W3-11 hardening: _my_player_defeated never raises on a malformed
    observation or player mapping — a malformed record counts as a
    missing record (the positive-only env fallback)."""
    session = FakeSession([turn_world(1)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    runner._session = session  # direct-call tests: run() never ran
    # A non-mapping observation.
    assert runner._my_player_defeated(None) is None
    assert runner._my_player_defeated([1, 2, 3]) is None
    assert runner._my_player_defeated("observation") is None
    # A player domain that is not a mapping.
    assert runner._my_player_defeated({"player": [1, 2]}) is None
    assert runner._my_player_defeated({"player": "players"}) is None
    # My entry not a mapping / is_alive not a bool.
    assert runner._my_player_defeated({"player": {MY_PLAYER: [1, 2]}}) is None
    assert runner._my_player_defeated(
        {"player": {MY_PLAYER: {"is_alive": "yes"}}}) is None
    # A well-formed record is still authoritative over the env flag.
    alive = {"player": {MY_PLAYER: {"is_alive": True}}}
    session.my_player_is_defeated = True
    assert runner._my_player_defeated(alive) is False
    session.my_player_is_defeated = False
    assert runner._my_player_defeated(alive) is False
    # The positive-only env fallback on a missing record.
    assert runner._my_player_defeated({}) is None
    session.my_player_is_defeated = True
    assert runner._my_player_defeated({}) is True
    session.my_player_is_defeated = None
    assert runner._my_player_defeated({}) is None


def test_d11_my_player_defeated_string_player_ids(tmp_path):
    """W3-11: serialized-string player IDs are honored — the my-player
    entry is found whether the observation keys the player domain by
    integer or by string (a JSON round-trip)."""
    session = FakeSession([turn_world(1)])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    runner._session = session  # direct-call tests: run() never ran
    dead = {"player": {str(MY_PLAYER): {"is_alive": False},
                       str(ENEMY): {"is_alive": True}}}
    assert runner._my_player_defeated(dead) is True
    alive = {"player": {str(MY_PLAYER): {"is_alive": True}}}
    session.my_player_is_defeated = True  # the record overrides the flag
    assert runner._my_player_defeated(alive) is False
    # Mixed key types in one domain.
    mixed = {"player": {str(MY_PLAYER): {"is_alive": False}, ENEMY: {}}}
    assert runner._my_player_defeated(mixed) is True


def test_d11_game_over_outcome_live_record_overrides_defeat_flag(tmp_path):
    """W3-11: the terminated_defeated branch re-checks the player
    record — a well-formed ALIVE record OVERRIDES the env's defeat flag
    that produced the extraction verdict (fall through to the game-over
    path), and the flag is honored only when the record is missing."""
    obs, info = turn_world(1)  # my player record: is_alive=True
    session = FakeSession([(obs, info)])
    session.my_player_is_defeated = True
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1)
    runner._session = session  # direct-call tests: run() never ran
    adapter = runner._load_adapter()
    outcomes = adapter.obs_extract.ExtractOutcome
    # The flag-sourced verdict (terminated_defeated) with an alive record.
    outcome, reason, snap = runner._game_over_outcome(
        outcomes.TERMINATED_DEFEATED,
        "terminated=True with my_player_is_defeated=True")
    assert outcome == "VICTORY"
    assert "defeated=False" in reason
    assert snap is None
    # The same verdict with a missing record: the positive flag decides.
    runner._session = _D11TurnStartSession(
        [turn_world(1)], player_mutator=_d11_player_missing,
        env_defeated=True)
    outcome, reason, _snap = runner._game_over_outcome(
        outcomes.TERMINATED_DEFEATED,
        "terminated=True with my_player_is_defeated=True")
    assert outcome == "DEFEATED"
    assert reason == "my player is defeated"


# ---------------------------------------------------------------------------
# W3-1 (D5): the durable dispatch-intent boundary + the cancellation
# contract + the D15 forensic-only regression pins
# ---------------------------------------------------------------------------


class _IntentDurabilitySession(FakeSession):
    """W3-1 (D5): on every mutation send, the dispatch intent for THAT
    send is already durable on disk — a complete, checksum-valid
    dispatch_intent line precedes the send (the pre-send fsync)."""

    def __init__(self, worlds, transcript_path):
        super().__init__(worlds)
        self._transcript_path = str(transcript_path)
        self.sends = 0
        self.intent_seqs_on_disk = []

    def step(self, action, *, deadline=None):
        if action is not None:
            self.sends += 1
            load = load_transcript(self._transcript_path)
            intents = [r for r in load.records
                       if isinstance(r, MetricEventRecord)
                       and r.name == "dispatch_intent"]
            # The intent for THIS send is already durable on disk, and
            # no action_result for it exists yet.
            assert not load.discarded_trailing_partial
            assert len(intents) == self.sends
            self.intent_seqs_on_disk.append(intents[-1].seq)
        return super().step(action, deadline=deadline)


def _intent_events(records):
    return [r for r in records
            if isinstance(r, MetricEventRecord)
            and r.name == "dispatch_intent"]


def _prefix_bytes_ending_at(path, seq):
    """The byte length of the transcript's prefix ending right after
    record ``seq``'s line (a simulated kill at that boundary)."""
    lines = Path(path).read_bytes().split(b"\n")
    assert lines[-1] == b""  # every record line is newline-terminated
    return sum(len(line) + 1 for line in lines[:seq + 1])


def test_runner_dispatch_intent_is_durable_before_every_send(tmp_path):
    """W3-1 (D5) crash-window matrix (1): the normal path — every send is
    preceded by its durable dispatch_intent (verified ON DISK inside the
    send itself), immediately followed by the correlated action_result
    (a dispatch-result/v1 detail envelope carrying the intent's seq)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = _IntentDurabilitySession(worlds, tmp_path / "transcript.jsonl")
    client = FakeModelClient([
        ok_response(["unit:10:goto_e",
                     "tech:cur_player:research_tech_alphabet_0"],
                    turn_id="t1", with_memory=False),
        ok_response(["unit:11:sentry"], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # The on-disk check inside every send: 3 sends, 3 durable intents.
    assert session.sends == 3
    assert len(session.intent_seqs_on_disk) == 3

    records = _records(runner.ledger.path)
    intents = _intent_events(records)
    executed = [a for a in _actions(records) if a.outcome == "executed"]
    assert len(intents) == 3 and len(executed) == 3
    # The pinned order per send: the intent is immediately followed by
    # its correlated action_result.
    intent_positions = [i for i, r in enumerate(records)
                         if r in intents]
    executed_positions = [i for i, r in enumerate(records)
                          if r in executed]
    assert len(intent_positions) == 3 == len(executed_positions)
    for ip, ep in zip(intent_positions, executed_positions):
        assert ep == ip + 1
    for intent, action in zip(intents, executed):
        payload = json.loads(intent.value)
        assert payload["protocol"] == "dispatch-intent/v1"
        assert payload["turn"] == action.turn
        assert payload["ref"] == action.ref
        assert payload["provenance_class"] == action.provenance_class
        assert payload["snapshot_seq"] == action.snapshot_seq
        assert payload["candidates_seq"] == action.candidates_seq
        assert payload["call_seq"] == action.call_seq
        envelope = json.loads(action.detail)
        assert envelope["protocol"] == "dispatch-result/v1"
        assert envelope["intent_seq"] == intent.seq
        assert action.ref in envelope["message"]
    # The intent payload carries identity only — no send/outcome claim.
    assert set(json.loads(intents[0].value)) == {
        "protocol", "turn", "ref", "call_seq", "snapshot_seq",
        "candidates_seq", "provenance_class"}


def test_runner_resend_pairs_each_intent_with_its_own_result(tmp_path):
    """W3-1 crash-window matrix (4): the same ref dispatched twice (the
    §3 line 35 verified resubmission) — each dispatch_intent pairs with
    its OWN action_result via intent_seq (no cross-pairing): the first
    intent with the uncertain attempt, the second with the resend."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = LifecycleSession(worlds, disconnect_on=GOTO_ACTION,
                               effect_applied=False)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # The two sends: the failed dispatch + the one resubmission.
    assert len([c for c in session.calls if c[0] == GOTO_ACTION]) == 2

    records = _records(runner.ledger.path)
    intents = _intent_events(records)
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(intents) == 2 and len(by_ref) == 2
    # Each intent pairs with its OWN result (no cross-pairing).
    env0 = json.loads(by_ref[0].detail)
    env1 = json.loads(by_ref[1].detail)
    assert env0["protocol"] == "dispatch-result/v1"
    assert env1["protocol"] == "dispatch-result/v1"
    assert env0["intent_seq"] == intents[0].seq
    assert env1["intent_seq"] == intents[1].seq
    assert env0["intent_seq"] != env1["intent_seq"]
    # The first result is the uncertain attempt; the second the verified
    # resubmission.
    assert by_ref[0].readback == "inconclusive"
    assert by_ref[1].readback == "verified"
    assert by_ref[1].provenance_class == "continued_model_order"
    # The resend's intent carries the resubmission's provenance class
    # (the intent is written at the resend site, before its send).
    payload1 = json.loads(intents[1].value)
    assert payload1["provenance_class"] == "continued_model_order"
    assert payload1["call_seq"] is None
    assert payload1["ref"] == GOTO_REF


def test_runner_final_check_blocked_cancels_the_dispatch(tmp_path,
                                                          monkeypatch):
    """W3-1 cancellation contract (crash-window matrix (5)): the FINAL
    pre-send recheck blocks the dispatch → a durable
    mutation_dispatch_cancelled event (the same payload keys as the
    intent + intent_seq) is emitted, NO send occurs, NO action_result is
    fabricated; the run takes the bounded deadline truncation."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    # The pre-send check passes once (the first check), then blocks (the
    # FINAL recheck after the intent's fsync).
    checks = {"n": 0}

    def fake_block_reason(self):
        checks["n"] += 1
        return None if checks["n"] == 1 else "ACTING deadline reached"

    monkeypatch.setattr(GameRunner, "_dispatch_block_reason",
                        fake_block_reason)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "deadline" in result.reason
    # NO send: the session saw no step at all.
    assert session.calls == []
    records = _records(runner.ledger.path)
    # NO action_result was fabricated for the blocked dispatch.
    assert _actions(records) == []
    intents = _intent_events(records)
    cancels = [r for r in records
               if isinstance(r, MetricEventRecord)
               and r.name == "mutation_dispatch_cancelled"]
    assert len(intents) == 1 and len(cancels) == 1
    intent_payload = json.loads(intents[0].value)
    cancel_payload = json.loads(cancels[0].value)
    # The same payload keys as the intent + intent_seq (+ the reason).
    assert cancel_payload["protocol"] == "dispatch-cancelled/v1"
    assert cancel_payload["intent_seq"] == intents[0].seq
    for key in ("turn", "ref", "call_seq", "snapshot_seq", "candidates_seq",
                "provenance_class"):
        assert cancel_payload[key] == intent_payload[key]
    assert cancel_payload["reason"] == "ACTING deadline reached"
    # Both records are durable on disk (they load back), intent first.
    assert intents[0].seq < cancels[0].seq


def test_runner_unresolved_intent_introduces_no_refusal(tmp_path):
    """W3-1 D15 regression pin: an unresolved (forensic) dispatch intent
    left by a crashed predecessor run introduces NO runner refusal — the
    runner attaches, continues appending normally and completes."""
    path = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1), durable=True)
    # A crashed predecessor's durable intent: no correlated result, no
    # cancellation — the forensic uncertain case.
    ledger.append(MetricEventRecord(
        name="dispatch_intent",
        value=_dispatch_intent_payload(
            turn=1, ref=GOTO_REF, call_seq=None, snapshot_seq=0,
            candidates_seq=0, provenance="model")), durable=True)
    dangling_seq = ledger.last_seq
    ledger.close()

    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    attached = TranscriptLedger.attach(path)
    config = RunnerConfig(max_turns=2, run_id="test-run",
                          watchdog_interval_s=0.02)
    runner = GameRunner(
        config=config, client=client, ledger=attached,
        manifest=RunManifest(run_id="test-run"),
        session_factory=lambda: session, clock=time.monotonic)
    result = runner.run()
    # The run starts and continues normally: no refusal, no quarantine,
    # the ordinary horizon completion.
    assert result.outcome == "COMPLETED"
    records = _records(runner.ledger.path)
    assert not [r for r in records
                if isinstance(r, MetricEventRecord)
                and r.name == "mutation_quarantine"]
    # The dangling intent survives untouched; the run's own records
    # continue the seq after it.
    dangling = _intent_events(records)
    assert dangling[0].seq == dangling_seq
    assert [r.seq for r in records] == list(range(len(records)))
    # The run's own dispatch was sent and correlated normally.
    assert len(dangling) == 2
    by_ref = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(by_ref) == 1
    envelope = json.loads(by_ref[0].detail)
    assert envelope["intent_seq"] == dangling[1].seq


def test_runner_never_imports_the_pending_intent_scanner():
    """W3-1 D15 regression pin (purity): no lechonk source imports or
    references the P11 audit's pending-intent scanner — unresolved
    intents are forensic (audit-side), never a runner input."""
    offenders = []
    for path in sorted((REPO / "src" / "lechonk").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "wp3_p11_audit" in text or "scan_pending_intents" in text:
            offenders.append(str(path))
    assert not offenders, (
        f"lechonk sources referencing the audit scanner: {offenders}")


def test_runner_intent_is_the_honest_crash_window_boundary(tmp_path):
    """W3-1 crash-window matrix (2)+(3) honesty pin: a kill between the
    intent fsync and the send, and a kill after the send before the
    read-back, leave the SAME durable trace — the transcript ends with
    the unresolved intent — and nothing in the intent record
    distinguishes them (the payload carries identity only, never an
    execution claim)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    session = FakeSession(worlds)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2)
    result = runner.run()
    assert result.outcome == "COMPLETED"
    # The send DID happen (the mutation was applied server-side) — yet
    # the durable trace below cannot show it.
    assert len([c for c in session.calls if c[0] == GOTO_ACTION]) == 1
    path = Path(runner.ledger.path)
    records = _records(path)
    intents = _intent_events(records)
    assert len(intents) == 1
    intent = intents[0]
    # The intent payload asserts identity ONLY — no send/outcome claim,
    # so a kill before the send and a kill after it are indistinguishable.
    payload = json.loads(intent.value)
    assert set(payload) == {"protocol", "turn", "ref", "call_seq",
                            "snapshot_seq", "candidates_seq",
                            "provenance_class"}
    assert payload["protocol"] == "dispatch-intent/v1"
    # Case (2): a kill between the intent fsync and the send — the
    # transcript ends with the unresolved intent; load keeps it.
    cut = _prefix_bytes_ending_at(path, intent.seq)
    killed_before = tmp_path / "killed-before-send.jsonl"
    killed_before.write_bytes(path.read_bytes()[:cut])
    load = load_transcript(killed_before)
    assert not load.discarded_trailing_partial
    assert load.records[-1].seq == intent.seq
    assert isinstance(load.records[-1], MetricEventRecord)
    # Case (3): a kill after the send before the read-back — the result
    # was never written, so the durable trace is IDENTICAL: the same
    # unresolved-intent tail (the honesty requirement, not a gap).
    killed_after = tmp_path / "killed-after-send.jsonl"
    killed_after.write_bytes(path.read_bytes()[:cut])
    load2 = load_transcript(killed_after)
    assert [r.seq for r in load2.records] == [r.seq for r in load.records]
    assert load2.records[-1].seq == intent.seq


# ---------------------------------------------------------------------------
# W3-2 (D6/D13, DEF-1/DEF-2): stop dispatch at the ACTING deadline with
# partial-batch accounting — one guard per mutation route
# ---------------------------------------------------------------------------


class _DeadlineStepSession(FakeSession):
    """A mutation step that consumes a scripted fake-clock interval per
    send (a slow send/read-back) — the ACTING deadline passes mid-batch
    while the in-flight send still completes (the D13 straddle)."""

    def __init__(self, worlds, clock, *, step_delays_s):
        super().__init__(worlds)
        self._clock = clock
        self._step_delays_s = list(step_delays_s)

    def step(self, action, *, deadline=None):
        if action is not None and self._step_delays_s:
            self._clock.advance(self._step_delays_s.pop(0))
        return super().step(action, deadline=deadline)


class _GameOverSlowStepSession(FakeSession):
    """The named mutation ends the game (terminated=True) AND consumes a
    scripted fake-clock interval — the game-over is established while the
    ACTING deadline also passes (the DEF-1 precedence injection)."""

    def __init__(self, worlds, clock, *, terminate_on, delay_s):
        super().__init__(worlds)
        self._clock = clock
        self._terminate_on = terminate_on
        self._delay_s = delay_s

    def step(self, action, *, deadline=None):
        if action is not None and tuple(action) == self._terminate_on:
            self._clock.advance(self._delay_s)
            outcome = super().step(action, deadline=deadline)
            return replace(outcome, terminated=True, truncated=False)
        return super().step(action, deadline=deadline)


class _DeadlineRefreshSession(LifecycleSession):
    """The reconnect read (refresh) consumes a scripted fake-clock
    interval — the ACTING deadline passes during the §3 line 35
    reconciliation, before the verified resubmission's pre-send check
    (the resend-route stop injection)."""

    def __init__(self, worlds, clock, *, disconnect_on, refresh_delay_s):
        super().__init__(worlds, disconnect_on=disconnect_on,
                         effect_applied=False)
        self._clock = clock
        self._refresh_delay_s = refresh_delay_s

    def refresh(self):
        self._clock.advance(self._refresh_delay_s)
        return super().refresh()


def _stop_events(records):
    return [r for r in records
            if isinstance(r, MetricEventRecord)
            and r.name == ACTING_DEADLINE_STOP_EVENT]


def _detail_envelope(record):
    """The dispatch-result/v1 detail envelope of an action_result (None
    for plain/legacy details)."""
    try:
        data = json.loads(record.detail)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _run_mid_batch_deadline_stop(tmp_path):
    """The W3-2 mid-batch stop scenario (the planned route): a 3-action
    batch whose sends each consume 100 s of fake time — the first two
    sends complete (the second straddles the 180 s ACTING deadline,
    D13), the loop-top guard stops the third. Returns (result, runner,
    session, records)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _DeadlineStepSession(worlds, clock,
                                   step_delays_s=[100.0, 100.0, 100.0])
    client = FakeModelClient([
        ok_response(["unit:10:goto_e",
                     "tech:cur_player:research_tech_alphabet_0",
                     "unit:11:sentry"],
                    turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    return result, runner, session, _records(runner.ledger.path)


def test_runner_acting_deadline_stop_mid_batch(tmp_path):
    """W3-2 (D6) acceptance (1): the mid-batch deadline stop — earlier
    actions have executed results, the remaining action gets an
    aborted_group record + the acting_deadline_stop summary event
    (correct fields), and the run truncates with the deadline reason."""
    result, runner, session, records = _run_mid_batch_deadline_stop(tmp_path)
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    # Two sends completed (the second straddled the deadline, D13); the
    # third action was never dispatched, and no finish primitive ran.
    assert [a.ref for a in executed] == [
        "unit:10:goto_e", "tech:cur_player:research_tech_alphabet_0"]
    assert len([c for c in session.calls if c[0] is not None]) == 2
    assert not [c for c in session.calls if c[0] is None]
    # The partial-batch record: the frozen literal, readback "none", the
    # pinned deadline detail, the batch's provenance linkage.
    assert [a.ref for a in aborted] == ["unit:11:sentry"]
    assert aborted[0].readback == "none"
    assert aborted[0].detail == "not dispatched: ACTING deadline reached"
    assert aborted[0].provenance_class == "model"
    assert aborted[0].turn == 1
    assert aborted[0].call_seq == executed[0].call_seq
    # The summary event: exactly one, the pinned partial-batch/v1 shape.
    stops = _stop_events(records)
    assert len(stops) == 1
    payload = json.loads(stops[0].value)
    assert payload == {
        "protocol": "partial-batch/v1",
        "turn": 1,
        "reason": "acting_deadline",
        "route": "planned",
        "completed_result_seqs": [executed[0].seq, executed[1].seq],
        "unresolved_intent_seqs": [],
        "unstarted_refs": [],
        "remaining_standing_actors": [],
    }
    # The accounting rides after the executed results, before the
    # run-level metrics/outcome records.
    assert stops[0].seq > aborted[0].seq > executed[-1].seq
    assert isinstance(records[-1], OutcomeRecord)
    assert records[-1].outcome == "TRUNCATED"


def test_runner_acting_deadline_stop_reconciles_dispatched(tmp_path):
    """W3-2 acceptance (7): reconciliation-by-construction — at the
    deadline stop no dispatched action lacks its result: every
    dispatch_intent pairs with its correlated action_result (or its
    cancellation), and the summary's unresolved_intent_seqs is empty."""
    result, runner, session, records = _run_mid_batch_deadline_stop(tmp_path)
    intents = _intent_events(records)
    assert len(intents) == 2  # one per dispatched send
    correlated = set()
    for action in _actions(records):
        envelope = _detail_envelope(action)
        if envelope is not None \
                and envelope.get("protocol") == "dispatch-result/v1":
            correlated.add(envelope["intent_seq"])
    assert correlated == {i.seq for i in intents}
    stops = _stop_events(records)
    assert json.loads(stops[0].value)["unresolved_intent_seqs"] == []
    assert _unresolved_dispatch_intents(records) == []


def test_unresolved_dispatch_intents_scans_by_seq(tmp_path):
    """The W3-2 stop summary's unresolved_intent_seqs scan: an intent
    alone is unresolved (the D15 forensic case); a correlated
    action_result (a dispatch-result/v1 envelope) or a cancellation (a
    dispatch-cancelled/v1 payload) settles it — by SEQ, never
    positional."""
    path = tmp_path / "scan.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    intent_a = ledger.append(MetricEventRecord(
        name="dispatch_intent",
        value=_dispatch_intent_payload(
            turn=1, ref=GOTO_REF, call_seq=None, snapshot_seq=0,
            candidates_seq=0, provenance="model")))
    intent_b = ledger.append(MetricEventRecord(
        name="dispatch_intent",
        value=_dispatch_intent_payload(
            turn=1, ref="unit:11:sentry", call_seq=None, snapshot_seq=0,
            candidates_seq=0, provenance="model")))
    ledger.close()
    # Both intents alone: unresolved (the forensic crash-window case).
    assert _unresolved_dispatch_intents(_records(path)) == [intent_a,
                                                             intent_b]
    # A cancellation settles intent_b; a correlated result settles
    # intent_a — each by its OWN seq.
    ledger = TranscriptLedger.attach(path)
    ledger.append(MetricEventRecord(
        name="mutation_dispatch_cancelled",
        value=_dispatch_cancelled_payload(
            intent_seq=intent_b, reason="ACTING deadline reached", turn=1,
            ref="unit:11:sentry", call_seq=None, snapshot_seq=0,
            candidates_seq=0, provenance="model")))
    ledger.append(ActionResultRecord(
        turn=1, ref=GOTO_REF, domain="unit", verb="goto_e",
        outcome="executed", provenance_class="model", readback="verified",
        detail=_dispatch_result_detail(intent_a, "sent")))
    ledger.close()
    assert _unresolved_dispatch_intents(_records(path)) == []


def test_runner_acting_deadline_aborted_group_excluded_from_metrics(tmp_path):
    """W3-2/DEF-2 acceptance (8): the deadline aborted_group records are
    excluded from the attempt/illegal-action accounting and never count
    as model participation — they are pipeline-side aborts, not
    attempts (metrics._is_attempt gates on the outcome)."""
    result, runner, session, records = _run_mid_batch_deadline_stop(tmp_path)
    aborted = [a for a in _actions(records) if a.outcome == "aborted_group"]
    assert len(aborted) == 1 and aborted[0].provenance_class == "model"
    # The two executed model actions are the attempts; the deadline
    # aborted_group is not.
    assert result.metrics.illegal_attempts_total == 2
    # Participation: one turn with opportunities, one with a credited
    # model execution — the aborted_group never credits.
    assert result.metrics.participation.numerator == 1
    assert result.metrics.participation.denominator == 1
    # The direct pin over the stopped transcript.
    report = compute_metrics(records)
    attempts = [a for a in _actions(records)
                if a.provenance_class == "model"
                and a.outcome in ("executed", "rejected")]
    assert report.illegal_attempts_total == len(attempts) == 2


def test_runner_acting_deadline_stop_degraded_between_groups(tmp_path):
    """W3-2 acceptance (2a): the degraded route stops between groups —
    the first defaulted group's send straddles the deadline and
    completes (D13); the next group's re-entry into _execute_planned
    stops at its loop-top guard: its action gets the deadline
    aborted_group record and was never sent."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _DeadlineStepSession(worlds, clock, step_delays_s=[190.0])
    client = FakeModelClient([
        error_response("timeout 1"),
        error_response("timeout 2"),  # the repair also fails -> degraded
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    records = _records(runner.ledger.path)
    actions = _actions(records)
    defaults = [a for a in actions if a.provenance_class == "scripted_default"]
    executed = [a for a in defaults if a.outcome == "executed"]
    aborted = [a for a in defaults if a.outcome == "aborted_group"]
    # The first group (the research pick) straddled and completed; the
    # second group (the city production pick) was never dispatched.
    assert [a.domain for a in executed] == ["tech"]
    assert [a.domain for a in aborted] == ["city"]
    assert aborted[0].readback == "none"
    assert aborted[0].detail == "not dispatched: ACTING deadline reached"
    assert len([c for c in session.calls if c[0] is not None]) == 1
    stops = _stop_events(records)
    assert len(stops) == 1
    payload = json.loads(stops[0].value)
    assert payload["protocol"] == "partial-batch/v1"
    assert payload["turn"] == 1
    assert payload["reason"] == "acting_deadline"
    assert payload["route"] == "degraded"
    assert payload["completed_result_seqs"] == [executed[0].seq]
    assert payload["unstarted_refs"] == []
    assert payload["unresolved_intent_seqs"] == []


def test_runner_acting_deadline_stop_degraded_unstarted_groups(tmp_path):
    """W3-2 acceptance (2b): the degraded route's unstarted groups —
    with the deadline already reached, the FIRST group's loop-top guard
    stops before any send: its action gets the aborted_group record and
    the later groups' refs ride in the summary's unstarted_refs."""
    worlds = [turn_world(1), turn_world(2)]
    clock = FakeClock()
    session = _DeadlineStepSession(worlds, clock, step_delays_s=[])
    runner = make_runner(tmp_path, client=FakeModelClient([]),
                         session=session, max_turns=1, clock=clock)
    adapter = runner._load_adapter()
    runner._session = session  # direct-call tests: run() never ran
    # The deadline is already reached at the degraded route's entry.
    runner._acting_deadline_at = clock() - 1.0
    snapshot = _fixture_snapshot(session)
    table = session.build_candidate_table()
    # Premise: the degraded default resolves both mandatory choices
    # (research completed + city production empty) — two groups.
    from lechonk.comparator import defaults as defaults_mod
    log = defaults_mod.decide(snapshot, table.offered())
    assert [d.domain for d in log.decisions] == ["tech", "city"]

    with pytest.raises(StateDeadlineExceeded):
        runner._execute_degraded(
            adapter, turn=1, table=table, snapshot=snapshot,
            snapshot_seq=0, candidates_seq=0, why="test: degraded")
    # No send at all: the guard fired before the first group's dispatch.
    assert session.calls == []
    records = _records(runner.ledger.path)
    aborted = [a for a in _actions(records) if a.outcome == "aborted_group"]
    assert [a.domain for a in aborted] == ["tech"]
    assert aborted[0].detail == "not dispatched: ACTING deadline reached"
    stops = _stop_events(records)
    assert len(stops) == 1
    payload = json.loads(stops[0].value)
    assert payload["route"] == "degraded"
    assert payload["completed_result_seqs"] == []
    # The later (unstarted) group's ref rides in the summary.
    city_decision = [d for d in log.decisions if d.domain == "city"][0]
    assert payload["unstarted_refs"] == [city_decision.candidate_id]
    assert payload["unresolved_intent_seqs"] == []


def test_runner_acting_deadline_stop_skips_standing_order(tmp_path):
    """W3-2 acceptance (3): the standing-order re-issue is skipped by
    the guard — turn 1 stores two standing orders (its sends consume no
    fake time); turn 2's first continuation straddles the deadline and
    completes (D13), the second continuation's guard stops the loop:
    its re-issue is never dispatched and the summary reports it as a
    remaining standing actor."""
    worlds = [
        turn_world(1),
        turn_world(2, unit_overrides=[(10, {"activity": 0}),
                                      (11, {"activity": 0})]),
        turn_world(3),
    ]
    clock = FakeClock()
    session = _DeadlineStepSession(
        worlds, clock, step_delays_s=[0.0, 0.0, 190.0])
    client = FakeModelClient([
        ok_response(["unit:10:sentry", "unit:11:sentry"],
                    turn_id="t1", with_memory=False),
        ok_response([], turn_id="t2", with_memory=False),  # responsive empty
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    records = _records(runner.ledger.path)
    actions = _actions(records)
    continuations = [a for a in actions
                     if a.provenance_class == "continued_model_order"]
    # Only unit 10's continuation was dispatched (it straddled the
    # deadline and completed); unit 11's re-issue was skipped by the
    # guard — its send never happened.
    assert [a.ref for a in continuations] == ["unit:10:sentry"]
    assert continuations[0].outcome == "executed"
    assert continuations[0].readback == "verified"
    assert continuations[0].turn == 2
    sends = [c for c in session.calls if c[0] is not None]
    assert len(sends) == 3  # 2 turn-1 model orders + 1 continuation
    assert len([c for c in sends if c[0] == ("unit", 11, "sentry")]) == 1
    # No deadline aborted_group records: standing orders are reported
    # via the summary's remaining_standing_actors, not per-action
    # records.
    assert not [a for a in actions if a.outcome == "aborted_group"]
    stops = _stop_events(records)
    assert len(stops) == 1
    payload = json.loads(stops[0].value)
    assert payload["route"] == "standing"
    assert payload["turn"] == 2
    assert payload["reason"] == "acting_deadline"
    assert payload["protocol"] == "partial-batch/v1"
    assert payload["completed_result_seqs"] == [continuations[0].seq]
    assert payload["unstarted_refs"] == []
    assert payload["remaining_standing_actors"] == ["11"]
    assert payload["unresolved_intent_seqs"] == []


def test_runner_acting_deadline_stop_blocks_verified_resend(tmp_path):
    """W3-2 acceptance (4): the verified resend is never dispatched past
    the deadline — the ACTING deadline passes during the §3 line 35
    reconciliation (the reconnect read); W3-1's pre-send check at the
    resend site blocks the resubmission (verified: no loop-level guard
    needed — the resend site's own pre-send check is the authority), and
    the run truncates with the deadline reason."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _DeadlineRefreshSession(worlds, clock,
                                      disconnect_on=GOTO_ACTION,
                                      refresh_delay_s=200.0)
    client = FakeModelClient([
        ok_response([GOTO_REF], turn_id="t1", with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    records = _records(runner.ledger.path)
    # The original dispatch was sent exactly once; the resubmission was
    # never dispatched (blocked at the resend site's FIRST pre-send
    # check — no resend intent was even written, no cancellation).
    assert len([c for c in session.calls if c[0] == GOTO_ACTION]) == 1
    intents = _intent_events(records)
    assert len(intents) == 1  # only the original dispatch's intent
    cancels = [r for r in records
               if isinstance(r, MetricEventRecord)
               and r.name == "mutation_dispatch_cancelled"]
    assert cancels == []
    # The uncertain attempt is ledgered and correlated to its intent
    # (the W3-1 contract: the send's fate is durably recorded).
    uncertain = [a for a in _actions(records) if a.ref == GOTO_REF]
    assert len(uncertain) == 1
    assert uncertain[0].readback == "inconclusive"
    envelope = json.loads(uncertain[0].detail)
    assert envelope["protocol"] == "dispatch-result/v1"
    assert envelope["intent_seq"] == intents[0].seq
    # The stop here is W3-1's pre-send authority inside _execute_one
    # (the final authority), not a loop guard: the loop-level
    # partial-batch accounting is the loop guards' contract, and this
    # single-action batch leaves nothing unrecorded.
    assert _stop_events(records) == []
    assert not [a for a in _actions(records) if a.outcome == "aborted_group"]
    assert _unresolved_dispatch_intents(records) == []


def test_runner_acting_deadline_in_flight_send_keeps_full_budget(tmp_path):
    """W3-2/D13 acceptance (5): an in-flight send that started before
    the deadline keeps its FULL 30 s mutation deadline (never bounded by
    the remaining ACTING time) and completes with its result; the NEXT
    loop iteration stops. The documented ACTING overshoot (up to one
    mutation deadline) is expected."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _DeadlineStepSession(worlds, clock, step_delays_s=[190.0])
    client = FakeModelClient([
        ok_response([GOTO_REF, "unit:11:sentry"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    # The straddling send was dispatched with the FULL mutation
    # deadline (30 s), not the remaining ACTING time (D13: no
    # min(remaining) bounding — explicitly not adopted).
    assert [c[1] for c in session.calls] == [30.0]
    records = _records(runner.ledger.path)
    actions = _actions(records)
    executed = [a for a in actions if a.outcome == "executed"]
    # The straddling send completed with its result...
    assert [a.ref for a in executed] == [GOTO_REF]
    assert executed[0].readback == "verified"
    # ...and the next iteration stopped: the second action was never
    # dispatched and carries the deadline aborted_group record.
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    assert [a.ref for a in aborted] == ["unit:11:sentry"]
    assert aborted[0].detail == "not dispatched: ACTING deadline reached"
    stops = _stop_events(records)
    assert len(stops) == 1
    payload = json.loads(stops[0].value)
    assert payload["route"] == "planned"
    assert payload["completed_result_seqs"] == [executed[0].seq]
    assert payload["unresolved_intent_seqs"] == []


def test_runner_acting_deadline_game_over_wins(tmp_path):
    """W3-2/DEF-1 acceptance (6): an ALREADY-ESTABLISHED game-over wins
    over a newly-noticed deadline expiry — the batch's first action ends
    the game (terminated) while its send also carries the ACTING
    deadline past: the outcome flows through the game-over path
    (VICTORY), never a deadline truncation, and no acting_deadline_stop
    is emitted."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _GameOverSlowStepSession(
        worlds, clock, terminate_on=GOTO_ACTION, delay_s=200.0)
    client = FakeModelClient([
        ok_response([GOTO_REF, "unit:11:sentry"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    # The game-over outcome, NOT a deadline truncation.
    assert result.outcome == "VICTORY"
    assert "game over mid-batch" in result.reason
    records = _records(runner.ledger.path)
    # No deadline stop was emitted: the established termination won.
    assert _stop_events(records) == []
    actions = _actions(records)
    # The remaining planned action was aborted by the GAME-OVER path
    # (the W2-4 mid-batch handling), not by the deadline.
    aborted = [a for a in actions if a.outcome == "aborted_group"]
    assert [a.ref for a in aborted] == ["unit:11:sentry"]
    assert aborted[0].detail == "game ended mid-batch"


def test_runner_acting_deadline_guard_fires_at_exact_deadline(tmp_path):
    """W3-2 (the single >= convention): exact equality with the ACTING
    deadline means NO send — the loop-top guard fires when
    clock() >= _acting_deadline_at (the first send consumes exactly the
    180 s budget; the second action is stopped, never dispatched)."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = _DeadlineStepSession(worlds, clock, step_delays_s=[180.0])
    client = FakeModelClient([
        ok_response([GOTO_REF, "unit:11:sentry"], turn_id="t1",
                    with_memory=False),
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    # The first send completed (in-flight, D13); at exactly the
    # deadline the guard stops the second action.
    assert len([c for c in session.calls if c[0] is not None]) == 1
    records = _records(runner.ledger.path)
    aborted = [a for a in _actions(records) if a.outcome == "aborted_group"]
    assert [a.ref for a in aborted] == ["unit:11:sentry"]
    stops = _stop_events(records)
    assert len(stops) == 1
    assert json.loads(stops[0].value)["route"] == "planned"


def test_runner_acting_deadline_phase_check_fires_at_exact_deadline(tmp_path):
    """W3-2 (the single >= convention at the phase check): a model call
    consuming EXACTLY the 180 s ACTING budget trips the post-model-phase
    deadline check (clock() - acting_entered >= acting_deadline_s) —
    the run truncates with the deadline reason."""
    worlds = [turn_world(1), turn_world(2), turn_world(3)]
    clock = FakeClock()
    session = FakeSession(worlds)
    client = FakeModelClient([
        advancing_response(
            lambda: ok_response([], turn_id="t1", with_memory=False),
            clock, seconds=180.0),  # exactly the ACTING deadline
    ])
    runner = make_runner(tmp_path, client=client, session=session,
                         max_turns=2, clock=clock)
    result = runner.run()
    assert result.outcome == "TRUNCATED"
    assert "ACTING exceeded its deadline of 180.0s" in result.reason
    # Nothing was dispatched: the phase check fired before execution.
    assert session.calls == []
