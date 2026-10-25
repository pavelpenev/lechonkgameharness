# SPDX-License-Identifier: MIT
"""Metric worked fixtures (WP2 acceptance, design-plan-freeciv-v1.1.md §4
line 54): every §3 metric has a worked fixture with a HAND-COMPUTED
expected value.

The fixtures are data: tests/data/wp2_metric_fixtures.json holds the
transcript record sequences AND the hand-computed expectations (with the
arithmetic spelled out in each fixture's "hand_computed" field). This test
appends each fixture through the REAL TranscriptLedger (which assigns
seq/ts/checksum and validates against the frozen union), computes the
metrics over the resulting records, and asserts the hand-computed values.

Record order = seq order. A record may carry an "alias" (stripped before
validation); any of origin_seq/call_seq/snapshot_seq/candidates_seq may be
"@<alias>" to reference another record's ledger seq.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from lechonk.core.ledger import TranscriptLedger
from lechonk.core.metrics import compute_metrics
from lechonk.core.schemas import TranscriptRecord

FIXTURES_PATH = Path(__file__).parent / "data" / "wp2_metric_fixtures.json"
_SEQ_FIELDS = ("origin_seq", "call_seq", "snapshot_seq", "candidates_seq")
_RECORD_ADAPTER = TypeAdapter(TranscriptRecord)


def _load_fixtures() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))["fixtures"]


def build_records(fixture: dict, tmp_path: Path) -> list:
    """Append a fixture's records through a real TranscriptLedger and
    return the validated records (with ledger-assigned seq/ts/checksum)."""
    path = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(path)
    # First pass: assign aliases to seqs (append order = seq order).
    alias_to_seq: dict[str, int] = {}
    prepared: list[dict] = []
    for index, record in enumerate(fixture["records"]):
        record = dict(record)
        alias = record.pop("alias", None)
        if alias is not None:
            alias_to_seq[alias] = index
        prepared.append(record)
    # Second pass: resolve "@alias" seq references, then append.
    for record in prepared:
        for field in _SEQ_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.startswith("@"):
                record[field] = alias_to_seq[value[1:]]
        ledger.append(_RECORD_ADAPTER.validate_python(record))
    ledger.close()
    return ledger.records()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return _load_fixtures()


def _assert_share(actual, expected: dict, label: str) -> None:
    assert actual.numerator == expected["numerator"], (
        f"{label}: numerator {actual.numerator} != {expected['numerator']}")
    assert actual.denominator == expected["denominator"], (
        f"{label}: denominator {actual.denominator} != "
        f"{expected['denominator']}")
    assert actual.status == expected["status"], (
        f"{label}: status {actual.status!r} != {expected['status']!r}")
    if expected["value"] is None:
        assert actual.value is None, f"{label}: value {actual.value} != None"
    else:
        assert actual.value == pytest.approx(expected["value"]), (
            f"{label}: value {actual.value} != {expected['value']}")


# ---------------------------------------------------------------------------
# participation gate + per-domain model-control shares
# ---------------------------------------------------------------------------


def test_fixture_participation(fixtures, tmp_path):
    """Hand-computed (see the fixture's hand_computed field):
    opportunity turns = 9, model-participation turns = 8 -> 8/9 = 0.888...
    >= 0.8 -> gate PASS. research 6/7 = 0.857..., production 5/5 = 1.0,
    expansion 1/1 -> denominator < 5 -> 'insufficient sample'."""
    records = build_records(fixtures["participation"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["participation"]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is exp["participation_gate_pass"]
    for domain, expected in exp["domain_model_shares"].items():
        _assert_share(report.domain_model_shares[domain], expected,
                      f"domain_model_share:{domain}")


# ---------------------------------------------------------------------------
# commitment follow-through (both shares, censoring, dedup)
# ---------------------------------------------------------------------------


def test_fixture_commitment_followthrough(fixtures, tmp_path):
    """Hand-computed: 10 emitted, 1 deduped, 1 censored-at-horizon ->
    denominator 8. fulfilled_any 6/8 = 0.75 (c5 missed its deadline;
    c12 is a production commitment no produce action realized).
    fulfilled_model 5/8 = 0.625 (c7 realized by a scripted_default is
    never credited; c11 realized via a continued_model_order linked to its
    model origin IS credited)."""
    records = build_records(fixtures["commitment_followthrough"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["commitment_followthrough"]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    assert report.commitment_total == exp["commitment_total"]
    assert report.commitment_deduped == exp["commitment_deduped"]
    assert report.commitment_censored == exp["commitment_censored"]
    _assert_share(report.commitment_followthrough_any,
                  exp["commitment_followthrough_any"],
                  "commitment_followthrough_any")
    _assert_share(report.commitment_followthrough_model,
                  exp["commitment_followthrough_model"],
                  "commitment_followthrough_model")


def test_continued_model_order_linkage_is_resolved(fixtures, tmp_path):
    """The continuation's origin_seq must resolve to the model action that
    originated the order (the fulfilled_model credit depends on it)."""
    records = build_records(fixtures["commitment_followthrough"], tmp_path)
    continuations = [r for r in records
                     if getattr(r, "provenance_class", None)
                     == "continued_model_order"]
    assert len(continuations) == 1
    by_seq = {r.seq: r for r in records}
    origin = by_seq[continuations[0].origin_seq]
    assert origin.provenance_class == "model"
    assert origin.outcome == "executed"
    assert origin.target == "Settlers"  # matches the c11 commitment


# ---------------------------------------------------------------------------
# reason-but-don't-act (window, censoring, dedup, defensive response)
# ---------------------------------------------------------------------------


def test_fixture_reason_but_dont_act(fixtures, tmp_path):
    """Hand-computed: 9 mentions, 1 deduped, 2 censored at run boundaries
    -> denominator 6. acted 4/6 (entity targeting, tile/8-neighbors, and a
    move strictly toward the threat tile all count; two mentions with
    nothing near them do not) -> share = 1 - 4/6 = 2/6 = 0.333..."""
    records = build_records(fixtures["reason_but_dont_act"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["reason_but_dont_act"]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    assert report.reason_mentions_total == exp["reason_mentions_total"]
    assert report.reason_mentions_deduped == exp["reason_mentions_deduped"]
    assert report.reason_mentions_censored == exp["reason_mentions_censored"]
    _assert_share(report.reason_but_dont_act, exp["reason_but_dont_act"],
                  "reason_but_dont_act")


# ---------------------------------------------------------------------------
# illegal-action taxonomy (over ALL model-emitted attempts incl. repairs)
# ---------------------------------------------------------------------------


def test_fixture_illegal_action_taxonomy(fixtures, tmp_path):
    """Hand-computed: 10 model-emitted attempts (the turn-10 rejection is
    the repair output, counted). schema_error 2/10 = 0.2, state_fog_error
    2/10 = 0.2, illegal_in_rule 2/10 = 0.2. aborted_group / scripted_default
    / engine_automatic / housekeeping records are NOT attempts."""
    records = build_records(fixtures["illegal_action_taxonomy"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["illegal_action_taxonomy"]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    assert report.illegal_attempts_total == exp["illegal_attempts_total"]
    for cls, expected in exp["illegal_action_shares"].items():
        _assert_share(report.illegal_action_shares[cls], expected,
                      f"illegal_action_share:{cls}")


# ---------------------------------------------------------------------------
# minimum-denominator rule + zero-opportunity turns
# ---------------------------------------------------------------------------


def test_fixture_minimum_denominator(fixtures, tmp_path):
    """Hand-computed: participation 1/3 -> 'insufficient sample' (not a
    number), gate indeterminate; no commitments/mentions -> 'n/a';
    1 illegal attempt -> 'insufficient sample'."""
    records = build_records(fixtures["minimum_denominator"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["minimum_denominator"]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is None
    _assert_share(report.commitment_followthrough_any,
                  exp["commitment_followthrough_any"], "followthrough_any")
    _assert_share(report.commitment_followthrough_model,
                  exp["commitment_followthrough_model"], "followthrough_model")
    _assert_share(report.reason_but_dont_act, exp["reason_but_dont_act"],
                  "reason_but_dont_act")
    assert report.illegal_attempts_total == exp["illegal_attempts_total"]
    for cls, expected in exp["illegal_action_shares"].items():
        _assert_share(report.illegal_action_shares[cls], expected,
                      f"illegal_action_share:{cls}")


def test_fixture_zero_opportunity(fixtures, tmp_path):
    """A turn with no offered consequential choice never enters the
    participation denominator: 0 opportunity turns -> 'n/a'."""
    records = build_records(fixtures["zero_opportunity"], tmp_path)
    report = compute_metrics(records)
    exp = fixtures["zero_opportunity"]["expectation"]
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is None


# ---------------------------------------------------------------------------
# report plumbing: metric_events + to_dict
# ---------------------------------------------------------------------------


def test_metric_events_cover_every_metric(fixtures, tmp_path):
    """The runner emits one metric_event per metric; every §3 metric name
    must be present in the event stream."""
    records = build_records(fixtures["participation"], tmp_path)
    report = compute_metrics(records)
    events = dict(report.metric_events())
    for name in ("participation", "participation_gate_pass",
                 "domain_model_share:research",
                 "domain_model_share:production",
                 "domain_model_share:expansion",
                 "commitment_followthrough_any",
                 "commitment_followthrough_model",
                 "commitment_censored", "commitment_deduped",
                 "reason_but_dont_act",
                 "reason_mentions_censored", "reason_mentions_deduped",
                 "illegal_action_share:schema_error",
                 "illegal_action_share:state_fog_error",
                 "illegal_action_share:illegal_in_rule",
                 "illegal_attempts_total"):
        assert name in events, f"missing metric_event {name}"
    assert events["participation"] == pytest.approx(8 / 9)
    assert events["participation_gate_pass"] is True
    # to_dict is JSON-serializable (the artifact/report path).
    assert json.loads(json.dumps(report.to_dict()))["run_turns"] == [1, 10]


# ---------------------------------------------------------------------------
# commitment_board (WP3 — the replay viewer's commitments board, §7.6)
# ---------------------------------------------------------------------------


def _board_records():
    """A small record sequence: turns 1..3; a model action fulfilling a
    research commitment, a scripted_default action fulfilling an
    expansion commitment (never credited as model follow-through), a
    missed commitment, a censored (open) commitment, and a duplicate
    commitment emission (de-duplicated)."""
    from lechonk.core.schemas import (
        ActionResultRecord,
        CommitmentRecord,
        TurnStartRecord,
    )
    return [
        TurnStartRecord(turn=1),
        TurnStartRecord(turn=2),
        TurnStartRecord(turn=3),
        CommitmentRecord(
            turn=1,
            commitments=[
                {"domain": "research", "target": "Alphabet",
                 "deadline_turn": 3},
                {"domain": "expansion", "target": "tile:1,1",
                 "deadline_turn": 3},
                {"domain": "military", "target": "unit:20",
                 "deadline_turn": 3},
                {"domain": "other", "target": "future",
                 "deadline_turn": 99},
            ],
            threats=[], call_seq=None),
        # a duplicate emission of the research commitment (de-duped)
        CommitmentRecord(
            turn=2,
            commitments=[{"domain": "research", "target": "Alphabet",
                          "deadline_turn": 3}],
            threats=[], call_seq=None),
        # the model fulfills the research commitment (turn 2 <= 3)
        ActionResultRecord(
            turn=2, ref="tech:player:research_tech_alphabet_0",
            domain="tech", verb="research_tech_alphabet_0",
            target="Alphabet", outcome="executed",
            provenance_class="model", readback="verified", detail=""),
        # the scripted default fulfills the expansion commitment
        ActionResultRecord(
            turn=2, ref="unit:10:build_city", domain="unit",
            verb="build_city", target="tile:1,1", outcome="executed",
            provenance_class="scripted_default", readback="verified",
            detail=""),
        # the military commitment is never fulfilled -> missed
    ]


def test_commitment_board_statuses():
    from lechonk.core.metrics import commitment_board
    board = commitment_board(_board_records())
    by_key = {(c["domain"], c["target"]): c for c in board}
    # de-duplicated: 4 unique commitments, not 5
    assert len(board) == 4
    research = by_key[("research", "Alphabet")]
    assert research["status"] == "fulfilled_model"
    assert research["deadline_turn"] == 3 and research["emitted_turn"] == 1
    assert research["fulfilled_by"] == [
        "tech:player:research_tech_alphabet_0"]
    assert research["fulfilled_model_by"] == [
        "tech:player:research_tech_alphabet_0"]
    expansion = by_key[("expansion", "tile:1,1")]
    # realized by a scripted default: fulfilled by ANY policy, never
    # credited as model follow-through
    assert expansion["status"] == "fulfilled_other"
    assert expansion["fulfilled_model_by"] == []
    assert expansion["fulfilled_by"] == ["unit:10:build_city"]
    military = by_key[("military", "unit:20")]
    assert military["status"] == "missed"
    assert military["fulfilled_by"] == []
    future = by_key[("other", "future")]
    # deadline 99 is outside the run [1, 3]: censored, not failed
    assert future["status"] == "open"


def test_commitment_board_consistent_with_compute_metrics():
    """The board uses the SAME frozen matching semantics as
    compute_metrics (no parallel implementation): the aggregate
    numerators equal the board's status counts."""
    from lechonk.core.metrics import commitment_board, compute_metrics
    records = _board_records()
    board = commitment_board(records)
    report = compute_metrics(records)
    in_run = [c for c in board if c["status"] != "open"]
    fulfilled_any = [c for c in in_run
                     if c["status"] in ("fulfilled_model",
                                        "fulfilled_other")]
    fulfilled_model = [c for c in in_run
                       if c["status"] == "fulfilled_model"]
    assert len(in_run) == 3  # the censored commitment is excluded
    assert report.commitment_censored == 1
    assert report.commitment_deduped == 1
    assert report.commitment_followthrough_any.numerator == len(fulfilled_any)
    assert report.commitment_followthrough_any.denominator == len(in_run)
    assert (report.commitment_followthrough_model.numerator
            == len(fulfilled_model))


def test_commitment_board_continued_model_order_credit(tmp_path):
    """fulfilled_model: a continued_model_order whose ORIGIN action is
    model-class and itself matches the commitment credits the model."""
    from lechonk.core.ledger import TranscriptLedger
    from lechonk.core.metrics import commitment_board
    from lechonk.core.schemas import (
        ActionResultRecord,
        CommitmentRecord,
        TurnStartRecord,
    )
    path = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(TurnStartRecord(turn=2))
    ledger.append(CommitmentRecord(
        turn=1,
        commitments=[{"domain": "military", "target": "unit:20",
                      "deadline_turn": 2}],
        threats=[], call_seq=None))
    origin_seq = ledger.append(ActionResultRecord(
        turn=1, ref="unit:10:attack_n", domain="unit", verb="attack_n",
        target="unit:20", affected="unit:20", actor_tile=(2, 1),
        outcome="executed", provenance_class="model", readback="verified",
        detail="model attack order"))
    ledger.append(ActionResultRecord(
        turn=2, ref="unit:10:attack_n", domain="unit", verb="attack_n",
        target="unit:20", affected="unit:20", actor_tile=(2, 1),
        outcome="executed", provenance_class="continued_model_order",
        origin_seq=origin_seq, readback="verified", detail="continued"))
    ledger.close()
    from lechonk.core.ledger import load_transcript
    board = commitment_board(load_transcript(path).records)
    assert len(board) == 1
    assert board[0]["status"] == "fulfilled_model"
    # both the origin action and its continued_model_order re-issue
    # fulfill the commitment (model-attributed); the ref is de-duplicated
    assert board[0]["fulfilled_model_by"] == ["unit:10:attack_n"]
    assert board[0]["fulfilled_model_by_count"] == 2
    assert board[0]["fulfilled_by_count"] == 2


# ---------------------------------------------------------------------------
# FW-10: typed commitment targets + the additive verb field
# (dated addendum artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/)
# ---------------------------------------------------------------------------


def _fw10_action(*, turn, domain, verb, target, ref=None):
    from lechonk.core.schemas import ActionResultRecord
    return ActionResultRecord(
        turn=turn, ref=ref or f"{domain}:actor:{verb}", domain=domain,
        verb=verb, target=target, outcome="executed",
        provenance_class="model", readback="verified", detail="")


def _fw10_board(commitment, action):
    """One commitment (turn 1, deadline 2) + one executed action (turn 2)
    over turns [1, 2]; returns the commitment board entry."""
    from lechonk.core.metrics import commitment_board
    from lechonk.core.schemas import CommitmentRecord, TurnStartRecord
    records = [
        TurnStartRecord(turn=1),
        TurnStartRecord(turn=2),
        CommitmentRecord(turn=1, commitments=[commitment], threats=[],
                         call_seq=None),
        action,
    ]
    board = commitment_board(records)
    assert len(board) == 1
    return board[0]


def test_fw10_legacy_commitment_payload_loads_and_matches(tmp_path):
    """FW-10 acceptance: a legacy Commitment payload (the old field shape,
    no target_kind/verb) loads through the real ledger + frozen union and
    matches by the documented legacy rule (kind "any", exact normalized
    equality)."""
    from lechonk.core.ledger import TranscriptLedger, load_transcript
    from lechonk.core.metrics import commitment_board
    from lechonk.core.schemas import (
        ActionResultRecord,
        CommitmentRecord,
        TurnStartRecord,
    )
    path = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(TurnStartRecord(turn=2))
    # the pre-addendum field shape, exactly
    ledger.append(CommitmentRecord(
        turn=1,
        commitments=[{"domain": "research", "target": "Pottery",
                      "deadline_turn": 2}],
        threats=[], call_seq=None))
    ledger.append(ActionResultRecord(
        turn=2, ref="tech:player:research_tech_pottery_1", domain="tech",
        verb="research_tech_pottery_1", target="Pottery", outcome="executed",
        provenance_class="model", readback="verified", detail=""))
    ledger.close()
    records = load_transcript(path).records
    commitment = [r for r in records
                  if isinstance(r, CommitmentRecord)][0].commitments[0]
    # legacy defaults: kind "any", verb None
    assert commitment.target_kind == "any"
    assert commitment.verb is None
    # matches by the legacy rule: exact normalized equality
    report = compute_metrics(records)
    assert report.commitment_followthrough_any.numerator == 1
    assert report.commitment_followthrough_any.denominator == 1
    assert report.commitment_followthrough_model.numerator == 1
    board = commitment_board(records)
    assert board[0]["status"] == "fulfilled_model"


def test_fw10_legacy_free_text_target_matches_exact_only():
    """FW-10 legacy rule: a free-text target (kind "any") matches by exact
    normalized equality — the old bidirectional substring match is gone in
    both directions, while case/whitespace normalization is retained."""
    from lechonk.core.schemas import Commitment
    action = _fw10_action(turn=2, domain="tech",
                          verb="research_tech_pottery_1", target="Pottery")
    # exact (normalized) -> fulfilled
    entry = _fw10_board(
        Commitment(domain="research", target=" pottery ", deadline_turn=2),
        action)
    assert entry["status"] == "fulfilled_model"
    # substring (commitment shorter than the action target) -> missed
    entry = _fw10_board(
        Commitment(domain="research", target="Pott", deadline_turn=2),
        action)
    assert entry["status"] == "missed"
    # substring (commitment longer than the action target) -> missed
    entry = _fw10_board(
        Commitment(domain="research", target="Pottery glaze",
                   deadline_turn=2),
        action)
    assert entry["status"] == "missed"


def test_fw10_wildcard_target_matches_any_action():
    """The "*" wildcard is retained and matches any action target, on the
    legacy kind and on typed kinds alike."""
    from lechonk.core.schemas import Commitment
    action = _fw10_action(turn=2, domain="tech",
                          verb="research_tech_pottery_1", target="Pottery")
    for kind in ("any", "tile", "unit"):
        entry = _fw10_board(
            Commitment(domain="other", target_kind=kind, target="*",
                       deadline_turn=2),
            action)
        assert entry["status"] == "fulfilled_model", kind


def test_fw10_typed_target_kinds_are_kind_aware():
    """Typed targets match by exact equality AND the action-side descriptor
    family must agree: name kinds bind to the corresponding action domain,
    descriptor kinds to the typed descriptor form."""
    from lechonk.core.schemas import Commitment
    produce = _fw10_action(turn=2, domain="city", verb="produce_warriors",
                           target="Warriors",
                           ref="city:3:produce_warriors")
    research = _fw10_action(turn=2, domain="tech",
                            verb="research_tech_warriors_0",
                            target="Warriors",
                            ref="tech:player:research_tech_warriors_0")
    # kind "tech" binds to research actions, not produce actions
    entry = _fw10_board(
        Commitment(domain="other", target_kind="tech", target="Warriors",
                   deadline_turn=2), produce)
    assert entry["status"] == "missed"
    entry = _fw10_board(
        Commitment(domain="other", target_kind="tech", target="Warriors",
                   deadline_turn=2), research)
    assert entry["status"] == "fulfilled_model"
    # kind "production" binds to produce_* actions
    entry = _fw10_board(
        Commitment(domain="production", target_kind="production",
                   target="Warriors", deadline_turn=2), produce)
    assert entry["status"] == "fulfilled_model"
    # descriptor kinds match the typed descriptor forms
    fortify = _fw10_action(turn=2, domain="unit", verb="fortify",
                           target="unit:7", ref="unit:7:fortify")
    entry = _fw10_board(
        Commitment(domain="military", target_kind="unit", target="unit:7",
                   verb="defend", deadline_turn=2), fortify)
    assert entry["status"] == "fulfilled_model"
    build = _fw10_action(turn=2, domain="unit", verb="build_city",
                         target="tile:2,1", ref="unit:7:build_city")
    entry = _fw10_board(
        Commitment(domain="expansion", target_kind="tile",
                   target="tile:2,1", verb="build", deadline_turn=2), build)
    assert entry["status"] == "fulfilled_model"
    negotiate = _fw10_action(turn=2, domain="dipl", verb="stop_negotiation",
                             target="player:2", ref="dipl:2:stop_negotiation")
    entry = _fw10_board(
        Commitment(domain="diplomacy", target_kind="player",
                   target="player:2", verb="negotiate", deadline_turn=2),
        negotiate)
    assert entry["status"] == "fulfilled_model"
    buy = _fw10_action(turn=2, domain="city", verb="city_buy_production",
                       target="city:3", ref="city:3:city_buy_production")
    # verb unset: any action — FW-11 enforces a set verb, and 'build' is
    # not compatible with city_buy_production (dated addendum §5)
    entry = _fw10_board(
        Commitment(domain="other", target_kind="city", target="city:3",
                   deadline_turn=2), buy)
    assert entry["status"] == "fulfilled_model"
    # exact string equality alone is NOT enough: the family must agree
    entry = _fw10_board(
        Commitment(domain="military", target_kind="tile", target="unit:7",
                   deadline_turn=2), fortify)
    assert entry["status"] == "missed"


# ---------------------------------------------------------------------------
# FW-11: verb compatibility, emission-turn lower bound, in-run dedup
# preference (dated addendum
# artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/ §5)
# ---------------------------------------------------------------------------


def _assert_fw11_commitment_fixture(fixtures, tmp_path, name):
    """Build a fixture's records through the real TranscriptLedger and
    assert its hand-computed commitment expectations: run turns, the
    dedup/censor accounting, and both follow-through shares (the full
    arithmetic is spelled out in the fixture's hand_computed field)."""
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    assert report.commitment_total == exp["commitment_total"]
    assert report.commitment_deduped == exp["commitment_deduped"]
    assert report.commitment_censored == exp["commitment_censored"]
    _assert_share(report.commitment_followthrough_any,
                  exp["commitment_followthrough_any"],
                  "commitment_followthrough_any")
    _assert_share(report.commitment_followthrough_model,
                  exp["commitment_followthrough_model"],
                  "commitment_followthrough_model")
    return records


def test_fw11_fixture_verb_mismatch_does_not_fulfill(fixtures, tmp_path):
    """Hand-computed (see the fixture): two military commitments with verb
    'attack'; the attack_n action fulfills unit:20 (verb class attack_*),
    the goto_n action on unit:21 matches domain+target but its verb is
    outside the 'attack' class -> only 1/2 fulfilled; denominator 2 < 5
    -> 'insufficient sample' (not a number, §3 line 42)."""
    _assert_fw11_commitment_fixture(
        fixtures, tmp_path, "verb-mismatch-does-not-fulfill")


def test_fw11_fixture_pre_emission_action_does_not_fulfill(fixtures,
                                                          tmp_path):
    """Hand-computed (see the fixture): the Pottery commitment is emitted
    on turn 2 (deadline 3); its only matching action is the turn-1
    research, which predates the emission -> the emission-turn lower
    bound (emitted_turn <= action.turn <= deadline) excludes it. The
    Alphabet commitment is fulfilled by the turn-3 research (2 <= 3 <= 3)
    -> 1/2 fulfilled; denominator 2 < 5 -> 'insufficient sample'."""
    _assert_fw11_commitment_fixture(
        fixtures, tmp_path, "pre-emission-action-does-not-fulfill")


def test_fw11_fixture_duplicate_prefers_in_run(fixtures, tmp_path):
    """Hand-computed (see the fixture): two emissions share the key
    (Pottery, research) - a stale memory-carried one (turn 1, deadline 20,
    outside the run [1, 9]) and the in-run one (turn 2, deadline 4). The
    dedup preference keeps the in-run emission -> total 2, deduped 1,
    censored 0, denominator 1, fulfilled 1/1 by the turn-3 research
    (2 <= 3 <= 4); denominator 1 < 5 -> 'insufficient sample'."""
    records = _assert_fw11_commitment_fixture(
        fixtures, tmp_path, "duplicate-prefers-in-run")
    # the board uses the same dedup preference: the kept emission is the
    # in-run one (emitted turn 2, deadline 4), not the stale turn-1 one
    # (first-wins would have kept it and censored the commitment away)
    from lechonk.core.metrics import commitment_board
    board = commitment_board(records)
    assert len(board) == 1
    assert board[0]["emitted_turn"] == 2
    assert board[0]["deadline_turn"] == 4
    assert board[0]["status"] == "fulfilled_model"
    assert board[0]["fulfilled_by"] == [
        "tech:cur_player:research_tech_pottery_1"]


# ---------------------------------------------------------------------------
# FW-12: readback-gated credit, expansion opportunity domain, gate
# domain-majority conjunct
# ---------------------------------------------------------------------------


def test_fw12_build_city_candidate_is_an_expansion_opportunity():
    """FW-12: metric_domain maps a build_city candidate to "expansion", and
    "expansion" is in OPPORTUNITY_DOMAINS — a build_city-only turn is a
    turn-with-opportunities (housekeeping still excluded)."""
    from lechonk.core.metrics import OPPORTUNITY_DOMAINS, opportunity_domains
    from lechonk.core.schemas import Candidate
    build = Candidate(id="unit:10:build_city", domain="unit",
                      actor_id="unit:10", verb="build_city")
    keep = Candidate(id="unit:11:cancel_order", domain="unit",
                     actor_id="unit:11", verb="cancel_order")
    assert "expansion" in OPPORTUNITY_DOMAINS
    assert opportunity_domains([build]) == ["expansion"]
    assert opportunity_domains([build, keep]) == ["expansion"]


def test_fw12_fixture_contradicted_does_not_credit(fixtures, tmp_path):
    """Hand-computed (see the fixture): the turn-3 CONTRADICTED research
    (readback 'contradicted', rejection_class illegal_in_rule) does not
    fulfill the Pottery commitment and does not count as acted against the
    unit:20 threat; the turn-4 inconclusive scripted_default research does
    not fulfill Currency; the turn-5 engine_automatic research (readback
    'none' by construction) fulfills Writing by any policy."""
    from lechonk.core.metrics import commitment_board
    name = "fw12-contradicted-does-not-credit"
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    assert report.commitment_total == exp["commitment_total"]
    assert report.commitment_deduped == exp["commitment_deduped"]
    assert report.commitment_censored == exp["commitment_censored"]
    _assert_share(report.commitment_followthrough_any,
                  exp["commitment_followthrough_any"],
                  "commitment_followthrough_any")
    _assert_share(report.commitment_followthrough_model,
                  exp["commitment_followthrough_model"],
                  "commitment_followthrough_model")
    assert report.reason_mentions_total == exp["reason_mentions_total"]
    assert report.reason_mentions_deduped == exp["reason_mentions_deduped"]
    assert report.reason_mentions_censored == exp["reason_mentions_censored"]
    # acted-count 0 (numerator 1 = the single mention): the contradicted
    # attack on unit:20 is readback-gated out
    _assert_share(report.reason_but_dont_act, exp["reason_but_dont_act"],
                  "reason_but_dont_act")
    # commitment_board uses the SAME readback-gated semantics (no parallel
    # implementation)
    statuses = {(e["domain"], e["target"]): e["status"]
                for e in commitment_board(records)}
    assert statuses == {
        ("research", "Pottery"): "missed",      # contradicted -> no credit
        ("research", "Currency"): "missed",     # inconclusive -> no credit
        ("research", "Writing"): "fulfilled_other",  # engine_automatic
    }


def test_fw12_fixture_gate_fails_on_domain_share(fixtures, tmp_path):
    """Hand-computed (see the fixture): participation 5/6 = 0.833... >= 0.8,
    but the research domain share 2/6 = 0.333... is numeric and <= 0.5 ->
    the gate FAILS even though the participation share alone would pass."""
    name = "fw12-gate-fails-domain-share"
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is exp["participation_gate_pass"]
    for domain, expected in exp["domain_model_shares"].items():
        _assert_share(report.domain_model_shares[domain], expected,
                      f"domain_model_share:{domain}")


def test_fw12_fixture_gate_passes_with_one_domain_na(fixtures, tmp_path):
    """Hand-computed (see the fixture): participation 5/6 >= 0.8, research
    and production numeric and > 0.5, expansion n/a (zero executed
    decisions) — the n/a domain does not block -> the gate PASSES."""
    name = "fw12-gate-passes-one-domain-na"
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is exp["participation_gate_pass"]
    for domain, expected in exp["domain_model_shares"].items():
        _assert_share(report.domain_model_shares[domain], expected,
                      f"domain_model_share:{domain}")


def test_fw12_fixture_gate_passes_all_domains_na(fixtures, tmp_path):
    """Hand-computed (see the fixture): participation 5/6 >= 0.8 and all
    three SHARE_DOMAINS n/a (no executed decisions in any of them) -> the
    gate PASSES on the participation conjunct alone."""
    name = "fw12-gate-all-domains-na"
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation_gate_pass is exp["participation_gate_pass"]
    for domain, expected in exp["domain_model_shares"].items():
        _assert_share(report.domain_model_shares[domain], expected,
                      f"domain_model_share:{domain}")


def test_fw12_fixture_build_city_expansion_opportunity(fixtures, tmp_path):
    """Hand-computed (see the fixture): build_city-only turns (opportunity
    domains ['expansion'], as the runner now derives) are turns-with-
    opportunities — all 6 enter the participation denominator — and the
    model's build_city decisions give expansion 5/5 = 1.0 -> gate PASS."""
    name = "fw12-build-city-expansion-opportunity"
    records = build_records(fixtures[name], tmp_path)
    report = compute_metrics(records)
    exp = fixtures[name]["expectation"]
    assert report.run_turns == tuple(exp["run_turns"])
    _assert_share(report.participation, exp["participation"], "participation")
    assert report.participation.denominator == 6  # every turn counts
    assert report.participation_gate_pass is exp["participation_gate_pass"]
    for domain, expected in exp["domain_model_shares"].items():
        _assert_share(report.domain_model_shares[domain], expected,
                      f"domain_model_share:{domain}")


# ---------------------------------------------------------------------------
# D1 (W3-4): the participation numerator is gated on readback credit
# ---------------------------------------------------------------------------


def _d1_records():
    """A 6-turn run pinning the D1 participation gate end to end:
    turn 1: opportunity + an executed/VERIFIED model research -> counts;
    turn 2: opportunity + an executed/CONTRADICTED model research
    (rejection_class illegal_in_rule — the rules engine proved
    non-execution) and an executed/INCONCLUSIVE one -> neither credits;
    turn 3: opportunity + a rejected model attempt -> never counted;
    turn 4: opportunity + a model housekeeping-verb decision (cancel_order)
    and an engine_automatic resolution -> neither is model participation;
    turn 5: opportunity + an executed/verified model research -> counts;
    turn 6: an empty candidate table (no opportunity domains) + an
    executed/verified model research -> excluded (zero-opportunity turns
    never enter the numerator).
    Hand-computed: opportunity turns {1,2,3,4,5} = 5; model-participation
    turns {1,5} = 2 -> participation 2/5 = 0.4 < 0.8 -> gate FAIL."""
    from lechonk.core.schemas import (
        ActionResultRecord,
        CandidatesRefRecord,
        TurnStartRecord,
    )

    def opp(turn):
        return CandidatesRefRecord(
            turn=turn, table_hash="sha256:" + "01" * 32, candidate_count=10,
            by_domain={"tech": 10}, opportunity_domains=["research"])

    def action(turn, ref, domain, verb, target, *, outcome="executed",
               provenance_class="model", readback="verified",
               rejection_class=None):
        return ActionResultRecord(
            turn=turn, ref=ref, domain=domain, verb=verb, target=target,
            outcome=outcome, provenance_class=provenance_class,
            readback=readback, rejection_class=rejection_class, detail="")

    return [
        TurnStartRecord(turn=1),
        opp(1),
        action(1, "tech:player:research_tech_alphabet_0", "tech",
               "research_tech_alphabet_0", "Alphabet"),
        TurnStartRecord(turn=2),
        opp(2),
        # CONTRADICTED: emitted "executed", the read-back proved
        # non-execution (D1: provably non-executed -> no credit).
        action(2, "tech:player:research_tech_pottery_1", "tech",
               "research_tech_pottery_1", "Pottery", readback="contradicted",
               rejection_class="illegal_in_rule"),
        # INCONCLUSIVE: the read-back is non-probative -> no credit either.
        action(2, "tech:player:research_tech_bronze_working_5", "tech",
               "research_tech_bronze_working_5", "Bronze working",
               readback="inconclusive"),
        TurnStartRecord(turn=3),
        opp(3),
        # A rejected attempt is never model participation (pin).
        action(3, "tech:player:research_tech_currency_3", "tech",
               "research_tech_currency_3", "Currency", outcome="rejected",
               readback="none", rejection_class="illegal_in_rule"),
        TurnStartRecord(turn=4),
        opp(4),
        # A model housekeeping-verb decision: never participation.
        action(4, "unit:10:cancel_order", "unit", "cancel_order", "unit:10"),
        # An engine-automatic resolution: never model participation.
        action(4, "tech:player:research_tech_auto_4", "tech",
               "research_tech_auto", "Auto",
               provenance_class="engine_automatic", readback="none"),
        TurnStartRecord(turn=5),
        opp(5),
        action(5, "tech:player:research_tech_writing_5", "tech",
               "research_tech_writing_5", "Writing"),
        TurnStartRecord(turn=6),
        # An empty candidate table: no opportunity domains -> the turn is
        # not a turn-with-opportunities even though a model action executed.
        CandidatesRefRecord(
            turn=6, table_hash="sha256:" + "01" * 32, candidate_count=0,
            by_domain={}, opportunity_domains=[]),
        action(6, "tech:player:research_tech_masonry_6", "tech",
               "research_tech_masonry_6", "Masonry"),
    ]


def test_d1_participation_numerator_is_readback_gated():
    """D1: _is_model_participation gains the _credits_execution conjunct —
    executed+verified model actions count; executed+CONTRADICTED
    (illegal_in_rule) and executed+inconclusive ones do not; rejected
    attempts never counted; the housekeeping-verb and engine-automatic
    exclusions are unchanged; and the aggregate intersects with
    opportunity turns, so a zero-opportunity turn never counts."""
    from lechonk.core.metrics import _is_model_participation, compute_metrics
    from lechonk.core.schemas import ActionResultRecord
    records = _d1_records()
    by_turn = {}
    for rec in records:
        if isinstance(rec, ActionResultRecord):
            by_turn.setdefault(rec.turn, []).append(rec)
    # (1) executed + verified model action counts.
    assert _is_model_participation(by_turn[1][0])
    assert _is_model_participation(by_turn[5][0])
    # (2) executed + CONTRADICTED (illegal_in_rule) does NOT count; an
    # executed + inconclusive read-back does not count either.
    assert not _is_model_participation(by_turn[2][0])
    assert not _is_model_participation(by_turn[2][1])
    # (3) a rejected attempt is never counted (pin).
    assert not _is_model_participation(by_turn[3][0])
    # (4) housekeeping/engine-auto exclusion unchanged: the model's
    # housekeeping-verb decision and the engine_automatic resolution are
    # not model participation.
    assert not _is_model_participation(by_turn[4][0])
    assert not _is_model_participation(by_turn[4][1])
    # (5) the aggregate: opportunity turns {1,2,3,4,5} = 5, participating
    # {1,5} = 2 — the contradicted/inconclusive/rejected/housekeeping turns
    # and the zero-opportunity turn 6 (verified model action) never enter
    # the numerator.
    report = compute_metrics(records)
    assert report.participation.numerator == 2
    assert report.participation.denominator == 5
    assert report.participation.value == pytest.approx(2 / 5)
    assert report.participation.status == "ok"
    assert report.participation_gate_pass is False


def test_d1_contradicted_execution_alone_drops_the_numerator():
    """The D1 discontinuity in isolation: the SAME run with the turn-2
    read-back flipped from 'verified' to 'contradicted'
    (rejection_class illegal_in_rule) drops the participation numerator
    from 3 to 2 — the record still says outcome "executed" either way,
    so the gate is read-back-gated, not outcome-gated; recorded evidence
    is never recomputed."""
    from lechonk.core.metrics import compute_metrics
    from lechonk.core.schemas import ActionResultRecord

    def run_with(readback):
        records = _d1_records()
        for rec in records:
            if (isinstance(rec, ActionResultRecord) and rec.turn == 2
                    and rec.verb == "research_tech_pottery_1"):
                rec.readback = readback
                rec.rejection_class = ("illegal_in_rule"
                                       if readback == "contradicted" else None)
        return compute_metrics(records)

    gated = run_with("contradicted")
    assert (gated.participation.numerator, gated.participation.denominator) \
        == (2, 5)
    # Flip the same record's read-back back to verified: turn 2 counts
    # again — the numerator is read-back-gated, not outcome-gated (the
    # record still says outcome "executed" either way).
    ungated = run_with("verified")
    assert (ungated.participation.numerator,
            ungated.participation.denominator) == (3, 5)


# ---------------------------------------------------------------------------
# D2 (W3-5): domain shares are gated on readback credit
# ---------------------------------------------------------------------------


def _d2_action(turn, verb, *, outcome="executed", provenance_class="model",
               readback="verified", rejection_class=None):
    from lechonk.core.schemas import ActionResultRecord
    return ActionResultRecord(
        turn=turn, ref=f"city:3:{verb}", domain="city", verb=verb,
        target="Warriors", outcome=outcome, provenance_class=provenance_class,
        readback=readback, rejection_class=rejection_class, detail="")


def _d2_shares(records):
    from lechonk.core.metrics import compute_metrics
    from lechonk.core.schemas import TurnStartRecord
    report = compute_metrics([TurnStartRecord(turn=1), *records])
    return report.domain_model_shares


def test_d2_contradicted_execution_excluded_from_both_sides():
    """D2: a CONTRADICTED production execution (readback 'contradicted',
    rejection_class illegal_in_rule — the rules engine proved
    non-execution) enters NEITHER the share numerator NOR the
    denominator: a domain containing only that record is n/a (0/0),
    not 0/1. Flipping the same record's read-back back to 'verified'
    puts it into both sides (1/1) — the gate is read-back-gated, not
    outcome-gated; recorded evidence is never recomputed."""
    from lechonk.core.metrics import STATUS_INSUFFICIENT, STATUS_NA
    gated = _d2_shares([
        _d2_action(1, "produce_warriors", readback="contradicted",
                   rejection_class="illegal_in_rule"),
    ])
    assert (gated["production"].numerator,
            gated["production"].denominator) == (0, 0)
    assert gated["production"].status == STATUS_NA
    assert gated["production"].value is None
    # Flip the same record's read-back to verified: it enters both the
    # numerator and the denominator (the record still says outcome
    # "executed" either way).
    credited = _d2_shares([
        _d2_action(1, "produce_warriors"),
    ])
    assert (credited["production"].numerator,
            credited["production"].denominator) == (1, 1)
    assert credited["production"].status == STATUS_INSUFFICIENT


def test_d2_engine_automatic_remains_in_denominator_only():
    """D2: an engine_automatic record with readback 'none' (by
    construction) stays in the denominator — it is creditable but
    non-model, so it counts against the share (0/1, 'insufficient
    sample', not n/a) and never enters the numerator."""
    from lechonk.core.metrics import STATUS_INSUFFICIENT
    shares = _d2_shares([
        _d2_action(1, "produce_warriors", provenance_class="engine_automatic",
                   readback="none"),
    ])
    assert (shares["production"].numerator,
            shares["production"].denominator) == (0, 1)
    assert shares["production"].status == STATUS_INSUFFICIENT
    assert shares["production"].value is None


def test_d2_share_statuses_unchanged_na_and_insufficient():
    """D2 does not change the _share status boundaries: a domain with no
    creditable executions at all is n/a (denominator 0); 1-4 creditable
    executions stay 'insufficient sample' (not a number)."""
    from lechonk.core.metrics import STATUS_INSUFFICIENT, STATUS_NA
    shares = _d2_shares([])
    assert shares["production"].status == STATUS_NA
    assert (shares["production"].numerator,
            shares["production"].denominator) == (0, 0)
    shares = _d2_shares([_d2_action(1, f"produce_warriors_{i}")
                         for i in range(4)])
    assert (shares["production"].numerator,
            shares["production"].denominator) == (4, 4)
    assert shares["production"].status == STATUS_INSUFFICIENT
    assert shares["production"].value is None


def test_d2_mixed_fixture_hand_computed_shares():
    """Hand-computed mixed fixture (production domain, all turn 1): four
    verified model productions + one CONTRADICTED model production
    (excluded from both sides) + one engine_automatic production
    (denominator only) + one REJECTED model attempt (never executed, so
    never in either side). Share = 4/5 = 0.8, 'ok' — under the ungated
    rule it would have been 4/6 = 0.666...; research/expansion have no
    executions and stay n/a."""
    from lechonk.core.metrics import STATUS_OK, STATUS_NA
    records = [
        # 4 verified model productions: numerator + denominator.
        *[_d2_action(1, f"produce_warriors_{i}") for i in range(4)],
        # CONTRADICTED: excluded from BOTH sides (D2).
        _d2_action(1, "produce_warriors_broken", readback="contradicted",
                   rejection_class="illegal_in_rule"),
        # engine_automatic: denominator only (creditable, non-model).
        _d2_action(1, "produce_warriors_auto",
                   provenance_class="engine_automatic", readback="none"),
        # REJECTED: never executed, never in either side (pin).
        _d2_action(1, "produce_warriors_rejected", outcome="rejected",
                   readback="none", rejection_class="illegal_in_rule"),
    ]
    shares = _d2_shares(records)
    assert (shares["production"].numerator,
            shares["production"].denominator) == (4, 5)
    assert shares["production"].status == STATUS_OK
    assert shares["production"].value == pytest.approx(4 / 5)
    for domain in ("research", "expansion"):
        assert (shares[domain].numerator, shares[domain].denominator) == (0, 0)
        assert shares[domain].status == STATUS_NA
