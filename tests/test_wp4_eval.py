# SPDX-License-Identifier: MIT
"""Tests for scripts/wp4_eval.py — the WP4 evaluation driver's NEW logic
(design-plan-freeciv-v1.1.md §4 line 58 + §7.7), offline only.

Covers: the §7.7 divergence computation (alignment, per-dimension diffs,
L1 summary, turns missing from one arm), the per-turn wall-clock/API/
action analysis over a synthetic ledger transcript (repair attempts =
retries, wall-clock derivation), the progress-curve extraction
(population sum, known tiles with an honest None fallback), the CLI
parsing of the cut-ladder shapes (--pairs/--horizon/--create-turns/
--seed/--run-id) and the per-pair seed derivation, early-terminal pair
completion reporting (both arms present, TRUNCATED separated from
strategic losses), and the honest-pilot no-ranking-language guard.
Proven components (paired.py, metrics.py, ledger, runner) are NOT
re-tested here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lechonk.core.ledger import TranscriptLedger
from lechonk.core.schemas import (
    ActionResultRecord,
    CandidatesRefRecord,
    ModelCallRecord,
    OutcomeRecord,
    SnapshotRefRecord,
    TurnStartRecord,
)

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "wp4_eval", REPO / "scripts" / "wp4_eval.py")
wp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp4)


# ---------------------------------------------------------------------------
# the §7.7 divergence computation
# ---------------------------------------------------------------------------

def _summary(turn, score, cities, units, tiles=None):
    return {"turn": turn, "score": score, "cities_own": cities,
            "units_own": units, "tiles_known": tiles}


def test_divergence_per_turn_diffs_and_l1():
    c = [_summary(5, 10, 1, 5, 400), _summary(6, 12, 2, 7, 420)]
    m = [_summary(5, 10, 1, 5, 400), _summary(6, 15, 1, 6, 400)]
    d = wp4.divergence_curve(c, m)
    assert d["dimensions"] == ["score", "cities_own", "units_own",
                               "tiles_known"]
    assert d["summary"]["turns_compared"] == 2
    e5, e6 = d["per_turn"]
    assert e5["turn"] == 5
    assert all(e5[k] == 0 for k in ("score_diff", "cities_own_diff",
                                    "units_own_diff", "tiles_known_diff"))
    assert e5["l1"] == 0
    assert e6["score_diff"] == 3 and e6["cities_own_diff"] == 1
    assert e6["units_own_diff"] == 1 and e6["tiles_known_diff"] == 20
    assert e6["l1"] == 25
    assert d["summary"]["mean_l1"] == 12.5
    assert d["summary"]["max_l1"] == 25
    assert d["summary"]["mean_score_diff"] == 1.5
    # descriptive framing, never counterfactual
    assert "counterfactual" in d["note"]


def test_divergence_missing_dimension_is_never_a_number():
    c = [_summary(5, 10, 1, 5, 400)]
    m = [_summary(5, 10, 1, 5, None)]  # model arm: no tile counts recorded
    d = wp4.divergence_curve(c, m)
    e = d["per_turn"][0]
    assert e["tiles_known_diff"] is None
    assert e["l1"] is None  # a partial dimension never fabricates an L1
    assert d["summary"]["mean_l1"] is None


def test_divergence_only_in_one_arm_is_reported_not_compared():
    c = [_summary(5, 10, 1, 5, 400), _summary(6, 11, 1, 5, 400)]
    m = [_summary(5, 10, 1, 5, 400), _summary(7, 12, 1, 5, 400)]
    d = wp4.divergence_curve(c, m)
    assert d["summary"]["turns_compared"] == 1
    assert d["summary"]["turns_only_in_comparator"] == [6]
    assert d["summary"]["turns_only_in_model"] == [7]
    assert len(d["per_turn"]) == 1


# ---------------------------------------------------------------------------
# the per-turn analysis over a real ledger transcript
# ---------------------------------------------------------------------------

def _write_transcript(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(transcript)
    ledger.append(TurnStartRecord(turn=5))
    ledger.append(SnapshotRefRecord(turn=5, snapshot_hash="h5",
                                    token_count=1200))
    ledger.append(CandidatesRefRecord(turn=5, table_hash="t5",
                                      candidate_count=9,
                                      by_domain={"unit": 9},
                                      opportunity_domains=["unit_action"]))
    ledger.append(ModelCallRecord(call_id="r/t5/call1", attempt=1,
                                  prompt_hash="p1", latency_ms=100,
                                  tokens_in=1000, tokens_out=200,
                                  cost_usd=0.01, fingerprint="f"))
    ledger.append(ModelCallRecord(call_id="r/t5/call2", attempt=2,
                                  prompt_hash="p2", latency_ms=50,
                                  tokens_in=800, tokens_out=100,
                                  cost_usd=0.008, fingerprint="f"))
    ledger.append(ActionResultRecord(turn=5, ref="unit:1:move_nw",
                                     domain="unit", verb="move_nw",
                                     outcome="executed",
                                     provenance_class="model",
                                     readback="verified"))
    ledger.append(TurnStartRecord(turn=6))
    ledger.append(ModelCallRecord(call_id="r/t6/call1", attempt=1,
                                  prompt_hash="p3", latency_ms=70,
                                  tokens_in=900, tokens_out=150,
                                  cost_usd=0.009, fingerprint="f"))
    ledger.append(OutcomeRecord(outcome="TRUNCATED",
                                reason="infrastructure failure: x"))
    ledger.close()
    return transcript


def test_per_turn_analysis_wall_clock_api_and_actions(tmp_path):
    from lechonk.core.ledger import load_transcript

    load = load_transcript(_write_transcript(tmp_path))
    rows = wp4.per_turn_analysis(load.records)
    assert [r["turn"] for r in rows] == [5, 6]
    # both rows' wall-clock closed (turn 5 by turn 6's start, turn 6 by
    # the outcome record) — never None when a following record exists
    assert all(r["wall_clock_s"] is not None for r in rows)
    # the model_call records carry no turn field: they are assigned to
    # the turn whose turn_start precedes them
    t5, t6 = rows
    assert t5["api_calls"] == 2 and t6["api_calls"] == 1
    assert t5["api_repair_attempts"] == 1 and t6["api_repair_attempts"] == 0
    assert t5["api_tokens_in"] == 1800 and t5["api_tokens_out"] == 300
    assert t5["api_cost_usd"] == round(0.01 + 0.008, 6)
    assert t5["actions_executed"] == 1
    assert t5["provenance"] == {"model": 1}
    assert t5["readbacks"] == {"verified": 1}
    assert t5["snapshot_tokens"] == 1200
    assert t5["opportunity_domains"] == ["unit_action"]
    # turn 6 has no closing turn_start; the outcome record closes it
    assert t6["wall_clock_s"] is not None


def test_per_turn_analysis_rejection_classes(tmp_path):
    transcript = tmp_path / "t.jsonl"
    ledger = TranscriptLedger(transcript)
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(ActionResultRecord(turn=1, ref="unit:1:attack_e",
                                     domain="unit", verb="attack_e",
                                     outcome="rejected",
                                     rejection_class="state_fog_error"))
    ledger.append(ActionResultRecord(turn=1, ref="bogus raw ref",
                                     domain="unknown", verb="",
                                     outcome="rejected",
                                     rejection_class="schema_error"))
    ledger.close()
    from lechonk.core.ledger import load_transcript

    load = load_transcript(transcript)
    rows = wp4.per_turn_analysis(load.records)
    assert rows[0]["actions_rejected"] == 2
    assert rows[0]["rejections_by_class"] == {"state_fog_error": 1,
                                              "schema_error": 1}


# ---------------------------------------------------------------------------
# progress curves + the divergence inputs
# ---------------------------------------------------------------------------

def test_progress_curves_population_and_known_tiles():
    summaries = [
        {"turn": 5, "score": 1, "gold": 54, "cities_own": 1, "units_own": 5,
         "techs_researched": 1, "research_target": "Alphabet",
         "research_turns_left": 24, "tiles_known": 410,
         "cities": [{"id": 1, "name": "A", "size": 1,
                     "production": "Warriors"}]},
        {"turn": 6, "score": 2, "gold": 55, "cities_own": 2, "units_own": 6,
         "techs_researched": 1, "research_target": "Pottery",
         "research_turns_left": 9, "tiles_known": 430,
         "cities": [{"id": 1, "name": "A", "size": 2,
                     "production": "Warriors"},
                    {"id": 2, "name": "B", "size": 1,
                     "production": "Warriors"}]},
    ]
    p = wp4.progress_curves(summaries)
    assert p["turns"] == [5, 6]
    assert p["population"] == [1, 3]  # the sum of city sizes
    assert p["tiles_known"] == [410, 430]
    assert p["score"] == [1, 2]
    # a summary without cities still yields a (0) population, not a crash
    p2 = wp4.progress_curves([{"turn": 1, "cities": None}])
    assert p2["population"] == [0]
    # a missing tiles_known stays None (honest gap)
    p3 = wp4.progress_curves([{"turn": 1}])
    assert p3["tiles_known"] == [None]


def test_progress_curves_unsorted_input_is_sorted():
    p = wp4.progress_curves([{"turn": 7, "score": 2},
                             {"turn": 6, "score": 1}])
    assert p["turns"] == [6, 7] and p["score"] == [1, 2]


# ---------------------------------------------------------------------------
# CLI: the cut-ladder shapes + the per-pair seed derivation
# ---------------------------------------------------------------------------

def test_cli_cut_ladder_shapes(monkeypatch):
    captured = {}

    def fake_orchestrate(pairs, horizon, create_turns, seed, run_id,
                         spectate):
        captured.update(pairs=pairs, horizon=horizon,
                        create_turns=create_turns, seed=seed, run_id=run_id,
                        spectate=spectate)
        return 0

    monkeypatch.setattr(wp4, "cmd_orchestrate", fake_orchestrate)
    # the WP4 target shape
    assert wp4.main([]) == 0
    assert captured == {"pairs": 2, "horizon": 50, "create_turns": 4,
                        "seed": 20261104, "run_id": "wp4-eval",
                        "spectate": False}
    # the ladder shapes are reachable through the same parameters
    assert wp4.main(["--pairs", "1"]) == 0
    assert captured["pairs"] == 1 and captured["horizon"] == 50
    assert wp4.main(["--pairs", "2", "--horizon", "25"]) == 0
    assert captured["pairs"] == 2 and captured["horizon"] == 25
    assert wp4.main(["--spectate"]) == 0
    assert captured["spectate"] is True


def test_cli_phase_subcommands(monkeypatch):
    seen = {}

    def fake_create(run_id, pair_index, seed, create_turns, horizon):
        seen["create"] = (run_id, pair_index, seed, create_turns, horizon)
        return 0

    def fake_arm(run_id, pair_index, policy, spectate):
        seen["arm"] = (run_id, pair_index, policy, spectate)
        return 0

    monkeypatch.setattr(wp4, "cmd_create", fake_create)
    monkeypatch.setattr(wp4, "cmd_arm", fake_arm)
    assert wp4.main(["--run-id", "wp4-eval", "--seed", "7",
                     "--horizon", "25", "--create-turns", "3",
                     "create", "--pair-index", "2"]) == 0
    assert seen["create"] == ("wp4-eval", 2, 7, 3, 25)
    assert wp4.main(["--run-id", "wp4-eval", "arm", "--policy", "model",
                     "--pair-index", "1"]) == 0
    assert seen["arm"] == ("wp4-eval", 1, "model", False)


def test_pair_seed_and_ids():
    assert wp4.pair_seed(20261104, 1) == 20261104
    assert wp4.pair_seed(20261104, 2) == 20261105  # distinct fresh games
    assert wp4.pair_id_for("wp4-eval", 1) == "wp4-eval-p1"
    assert wp4.pair_dir_for("wp4-eval", 2).endswith("wp4-eval/p2")


# ---------------------------------------------------------------------------
# early-terminal pair completion + crash/strategic separation
# ---------------------------------------------------------------------------

def _arm_manifest_stub(arm, outcome, reason, turns_done, turns_planned,
                       early_report=None):
    early = (None if turns_done >= turns_planned
             else (early_report
                   or f"arm terminated early: outcome={outcome} "
                      f"reason={reason!r} turns_completed={turns_done}/"
                      f"{turns_planned}"))
    return {
        "arm": arm,
        "policy": f"policy-{arm}",
        "run_id": f"wp4-eval-p1/{arm}",
        "outcome": outcome,
        "reason": reason,
        "outcome_class": wp4.outcome_class(outcome),
        "survived": wp4.survived(outcome),
        "turns_completed": turns_done,
        "turns_planned": turns_planned,
        "early_terminal": early is not None,
        "early_terminal_report": early,
        "wall_clock_s": 10.0,
        "transcript": f"/tmp/{arm}/transcript.jsonl",
        "replay_bundle": f"/tmp/{arm}/replay",
        "model_fingerprint": None if arm == "comparator" else "sha256:fp",
        "post_run": {
            "metrics": {"participation": {"numerator": 3, "denominator": 3,
                                          "value": 1.0, "status": "ok"},
                        "illegal_action_shares": {
                            "schema_error": {"numerator": 0, "denominator": 5,
                                             "value": 0.0, "status": "ok"}}},
            "provenance_counts": {"model": 4} if arm == "model"
            else {"scripted_default": 6},
            "per_turn": [{"turn": 5, "wall_clock_s": 1.5,
                          "api_repair_attempts": 1 if arm == "model" else 0,
                          "api_tokens_in": 100, "api_tokens_out": 20,
                          "api_latency_ms": 90, "api_cost_usd": 0.01}],
            "repair_attempts": 1 if arm == "model" else 0,
            "api_totals": ({"calls": 2, "tokens_in": 100, "tokens_out": 20,
                            "cost_usd": 0.011, "latency_ms": 90}
                           if arm == "model" else None),
        },
    }


def test_pair_metrics_report_survives_an_early_terminal():
    """The driver must complete the pair honestly when the model arm ends
    early (DEFEATED before the horizon): both arms reported, the early
    terminal recorded, and DEFEATED stays a strategic loss — never an
    infrastructure failure."""
    c = _summary(5, 10, 1, 5, 400)
    m = _summary(5, 10, 1, 5, 400)
    div = wp4.divergence_curve([c], [m])
    arms = {
        "comparator": _arm_manifest_stub("comparator", "COMPLETED",
                                         "horizon reached", 2, 2),
        "model": _arm_manifest_stub("model", "DEFEATED",
                                    "my player is defeated", 1, 2),
    }
    report = wp4.pair_metrics_report(
        pair_id="wp4-eval-p1", run_id="wp4-eval", pair_index=1,
        parameters={"pairs": 2, "horizon": 2, "create_turns": 4,
                    "seed": 20261104, "seed_base": 20261104,
                    "username": "myagent", "aifill": 2, "ruleset": "classic"},
        checkpoint={"path": "/c/ckpt.sav.xz", "sha256": "sha256:ab",
                    "turn": 5},
        contract_sha256="sha256:cd", arm_manifests=arms,
        summaries_comparator=[c], summaries_model=[m],
        divergence=wp4.divergence_curve([c], [m]),
        drift={"drift_suspect": False, "reproducibility_claim": True},
        checks={"per_turn_summaries_recorded": {"pass": True, "detail": ""}},
        verdict="PASS")
    # the pair is completed honestly: BOTH arms reported, the early
    # terminal recorded — never dropped
    assert set(report["arms"]) == {"comparator", "model"}
    model = report["arms"]["model"]
    assert model["outcome"] == "DEFEATED"
    assert model["outcome_class"] == "strategic_terminal"
    assert model["early_terminal"] is True
    assert model["early_terminal_report"]
    assert model["retries"] == {"repair_attempts": 1}
    assert report["arms"]["comparator"]["outcome_class"] == \
        "horizon_completed"
    # DEFEATED stays a strategic loss — separated from infrastructure
    assert report["crash_vs_strategic"]["strategic_terminals"] == ["model"]
    assert report["crash_vs_strategic"]["infrastructure_truncations"] == []
    # the divergence curve is recorded per pair (§7.7)
    assert report["divergence"]["summary"]["turns_compared"] == 1
    # honesty metrics ride through from the arm manifests
    assert report["arms"]["model"]["honesty_metrics"]["participation"][
        "value"] == 1.0


def test_pair_metrics_report_truncated_is_infrastructure():
    arms = {
        "comparator": _arm_manifest_stub("comparator", "COMPLETED",
                                         "horizon reached", 2, 2),
        "model": _arm_manifest_stub("model", "TRUNCATED",
                                    "infrastructure failure: timeout",
                                    1, 2),
    }
    report = wp4.pair_metrics_report(
        pair_id="wp4-eval-p1", run_id="wp4-eval", pair_index=1,
        parameters={"pairs": 2, "horizon": 2, "create_turns": 4,
                    "seed": 20261104, "seed_base": 20261104,
                    "username": "myagent", "aifill": 2, "ruleset": "classic"},
        checkpoint={"path": "/c/ckpt.sav.xz", "sha256": "sha256:ab",
                    "turn": 5},
        contract_sha256="sha256:cd", arm_manifests=arms,
        summaries_comparator=[], summaries_model=[],
        divergence=wp4.divergence_curve([], []),
        drift={"drift_suspect": False, "reproducibility_claim": False},
        checks={}, verdict="FAIL")
    model = report["arms"]["model"]
    assert model["outcome_class"] == "infrastructure_truncation"
    assert model["survived"] is None  # unknown — never fabricated
    assert report["crash_vs_strategic"]["infrastructure_truncations"] == \
        ["model"]
    assert report["crash_vs_strategic"]["strategic_terminals"] == []


def test_pair_report_labels_degraded_dominant_on_gate_fail(tmp_path):
    """§3 line 38: a FAILED participation gate labels the pair
    degraded-dominant (reported in full, excluded from model-performance
    claims); the plan-mandated label must pass the honest-pilot guard,
    and the transport-timeout share is summarized (exit-review INFO 3)."""
    c = _summary(5, 10, 1, 5, 400)
    comparator = _arm_manifest_stub("comparator", "COMPLETED",
                                    "horizon reached", 2, 2)
    model = _arm_manifest_stub("model", "COMPLETED", "horizon reached", 2, 2)
    model["post_run"]["metrics"] = dict(
        model["post_run"]["metrics"],
        participation={"numerator": 32, "denominator": 50, "value": 0.64,
                       "status": "ok"},
        participation_gate_pass=False)
    api_log = tmp_path / "api_calls.jsonl"
    api_log.write_text(
        json.dumps({"attempt": 1, "turn": 5, "outcome": "transport_error",
                    "error": "transport failure: The read operation "
                             "timed out"}) + "\n"
        + json.dumps({"attempt": 2, "turn": 5, "outcome": "transport_error",
                      "error": "transport failure: The read operation "
                               "timed out"}) + "\n",
        encoding="utf-8")
    model["api_calls_log"] = str(api_log)
    arms = {"comparator": comparator, "model": model}
    report = wp4.pair_metrics_report(
        pair_id="wp4-eval-p1", run_id="wp4-eval", pair_index=1,
        parameters={"pairs": 2, "horizon": 2, "create_turns": 4,
                    "seed": 20261104, "seed_base": 20261104,
                    "username": "myagent", "aifill": 2, "ruleset": "classic"},
        checkpoint={"path": "/c/ckpt.sav.xz", "sha256": "sha256:ab",
                    "turn": 5},
        contract_sha256="sha256:cd", arm_manifests=arms,
        summaries_comparator=[c], summaries_model=[c],
        divergence=wp4.divergence_curve([c], [c]),
        drift={"drift_suspect": False, "reproducibility_claim": True},
        checks={"per_turn_summaries_recorded": {"pass": True, "detail": ""}},
        verdict="PASS")
    assert report["degraded_dominant"] == "degraded-dominant"
    assert report["degraded_dominant_consequence"] == \
        wp4.DEGRADED_DOMINANT_CONSEQUENCE
    tt = report["arms"]["model"]["transport_timeouts"]
    assert tt["timeout_calls_total"] == 2
    assert tt["initial_timeout_turns"] == [5]
    rendered = wp4.render_pair_report_md(report)
    assert "degraded-dominant" in rendered
    assert "transport read-timeouts" in rendered
    wp4.assert_no_ranking_language(rendered, "pair md")
    wp4.assert_no_ranking_language(json.dumps(report, default=str),
                                   "pair json")


def test_generated_reports_carry_no_ranking_language():
    """Acceptance: no model-ranking language in any generated artifact —
    the reports render clean and the guard rejects ranking language."""
    c = _summary(5, 10, 1, 5, 400)
    arms = {
        "comparator": _arm_manifest_stub("comparator", "COMPLETED",
                                         "horizon reached", 2, 2),
        "model": _arm_manifest_stub("model", "COMPLETED",
                                    "horizon reached", 2, 2),
    }
    report = wp4.pair_metrics_report(
        pair_id="wp4-eval-p1", run_id="wp4-eval", pair_index=1,
        parameters={"pairs": 2, "horizon": 2, "create_turns": 4,
                    "seed": 20261104, "seed_base": 20261104,
                    "username": "myagent", "aifill": 2, "ruleset": "classic"},
        checkpoint={"path": "/c/ckpt.sav.xz", "sha256": "sha256:ab",
                    "turn": 5},
        contract_sha256="sha256:cd", arm_manifests=arms,
        summaries_comparator=[c], summaries_model=[c],
        divergence=wp4.divergence_curve([c], [c]),
        drift={"drift_suspect": False, "reproducibility_claim": True},
        checks={"per_turn_summaries_recorded": {"pass": True, "detail": ""}},
        verdict="PASS")
    rendered = wp4.render_pair_report_md(report)
    dumped = json.dumps(report, default=str)
    # honest-pilot framing is present, ranking language is not
    assert "no model-ranking claims" in rendered
    wp4.assert_no_ranking_language(rendered, "pair md")
    wp4.assert_no_ranking_language(dumped, "pair json")
    with pytest.raises(RuntimeError, match="model-ranking language"):
        wp4.assert_no_ranking_language(
            "the model outperforms the comparator", "x")
    # the plan-mandated §3 line 38 gate-failure label is emittable (a
    # labeling rule, not a ranking claim). Fact: "dominant" (domin+ant)
    # never contained the blocked stem "dominat" (domin+at) — the label
    # was never blocked; the guard's (?<!degraded-) lookbehind on the
    # "dominat" stem GUARANTEES the label stays emittable even if the
    # blocked set is later tightened (e.g. adding "dominant").
    wp4.assert_no_ranking_language(
        "labeled degraded-dominant per §3 (reported in full, excluded "
        "from model-performance claims)", "degraded-dominant label")
    wp4.assert_no_ranking_language(
        "degraded-dominant: the pair is labeled degraded-dominant",
        "degraded-dominant label (twice)")
    # actual ranking language stays blocked (the "dominat-" stems)
    with pytest.raises(RuntimeError, match="model-ranking language"):
        wp4.assert_no_ranking_language(
            "the model dominates the comparator", "x")
    # metric vocabulary is not ranking language: the honesty metrics'
    # ±2-turn window note and the framing's own meta-phrase must pass
    wp4.assert_no_ranking_language(
        "1 - acted-rate over threat mentions whose +/2-turn window "
        "falls inside the run; no model-ranking claims", "metrics text")


def test_overall_report_render(tmp_path):
    p = {
        "pair_id": "wp4-eval-p1", "run_id": "wp4-eval", "pair_index": 1,
        "verdict": "PASS", "checkpoint_sha256": "sha256:ab",
        "degraded_dominant": False, "model_participation": 1.0,
        "drift": {"drift_suspect": False, "reproducibility_claim": True},
        "divergence": wp4.divergence_curve([_summary(5, 10, 1, 5, 400)],
                                           [_summary(5, 10, 1, 5, 400)]),
        "arms": {
            "comparator": _arm_manifest_stub("comparator", "COMPLETED",
                                             "horizon reached", 2, 2),
            "model": _arm_manifest_stub("model", "COMPLETED",
                                        "horizon reached", 2, 2),
        },
    }
    overall = {
        "run_id": "wp4-eval",
        "generated_by": "scripts/wp4_eval.py", "generated_at": "t",
        "pilot_framing": wp4.PILOT_FRAMING,
        "parameters": {"pairs": 2, "horizon": 50, "create_turns": 4,
                       "seed": 20261104, "username": "myagent",
                       "aifill": 2, "ruleset": "classic",
                       "harness_revision": "sha256:ef"},
        "pairs": [p],
        "totals": {"pairs_run": 1, "verdicts": {"wp4-eval-p1": "PASS"},
                   "infrastructure_truncations": [],
                   "strategic_terminals": [],
                   "horizon_completions": ["wp4-eval-p1/comparator",
                                           "wp4-eval-p1/model"],
                   "api_calls": 3, "tokens_in": 100, "tokens_out": 20,
                   "cost_usd": 0.03, "wall_clock_s": 120.0},
        "cut_ladder": "applied between pairs",
        "verdict": "PASS",
    }
    rendered = wp4.render_overall_md(overall)
    dumped = json.dumps(overall, default=str)
    assert "no model-ranking claims" in rendered
    wp4.assert_no_ranking_language(rendered, "overall md")
    wp4.assert_no_ranking_language(dumped, "overall json")
    assert "OVERALL VERDICT: **PASS**" in rendered
    # the §3 line 38 labeling rule renders per pair: gate PASS → not
    # labeled; gate FAIL → the degraded-dominant label + the §3 consequence
    assert "participation gate: PASS (1.0" in rendered
    degraded_pair = dict(p, degraded_dominant=True, model_participation=0.64)
    degraded_overall = dict(overall, pairs=[degraded_pair])
    degraded_md = wp4.render_overall_md(degraded_overall)
    assert "degraded-dominant" in degraded_md
    wp4.assert_no_ranking_language(degraded_md, "degraded overall md")


# ---------------------------------------------------------------------------
# the recorder's known-tile enrichment
# ---------------------------------------------------------------------------

class _InnerStub:
    provenance_class = "model"

    def complete_turn(self, prompt):
        from lechonk.core.model_client import ModelResponse
        return ModelResponse(call_id=prompt.call_id, prompt_hash="ph",
                             latency_ms=1, tokens_in=1, tokens_out=1,
                             cost_usd=0.0, fingerprint="f")

    def complete_repair(self, prompt):
        raise AssertionError("not used here")


def _prompt_with_tiles(call_id):
    from lechonk.core.model_client import TurnPrompt

    snapshot = json.dumps({
        "turn": 5,
        "player": {"score": 3, "gold": 30},
        "cities": [], "units": [],
        "map_view": {"tile_counts": {"visible": 40, "last_known": 10,
                                     "unexplored": 950}},
    })
    return TurnPrompt(call_id=call_id, turn_id="r/t5", memory="m",
                      snapshot=snapshot, delta="d", candidates="c")


def test_recorder_enriches_known_tiles(tmp_path):
    turns_path = tmp_path / "turns.jsonl"
    rec = wp4.TurnSummaryRecorder(_InnerStub(), turns_path)
    resp = rec.complete_turn(_prompt_with_tiles("r/t5/call1"))
    assert resp.batch is None  # the stub's response (no batch) passes through
    rec.close()
    rows = [json.loads(l) for l in
            turns_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["turn"] == 5 and rows[0]["score"] == 3
    assert rows[0]["tiles_known"] == 50
    assert rows[0]["tiles_visible"] == 40
    assert rows[0]["tiles_last_known"] == 10
    assert rec.summaries() == rows


def test_recorder_garbage_snapshot_yields_none_tiles(tmp_path):
    from lechonk.core.model_client import TurnPrompt

    turns_path = tmp_path / "turns.jsonl"
    rec = wp4.TurnSummaryRecorder(_InnerStub(), turns_path)
    prompt = TurnPrompt(call_id="r/t5/call1", turn_id="r/t5", memory="m",
                         snapshot="not json", delta="d", candidates="c")
    rec.complete_turn(prompt)
    rec.close()
    # an unparseable snapshot yields no summary row at all — the recorder
    # never fabricates a turn summary (the divergence curve then reports
    # the turn as missing from the arm, not as a fabricated zero)
    rows = [json.loads(l) for l in
            turns_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows == []
    assert rec.summaries() == []


# ---------------------------------------------------------------------------
# the outcome taxonomy: crashes stay separated from strategic losses
# ---------------------------------------------------------------------------

def test_outcome_taxonomy_separation():
    """TRUNCATED is infrastructure; DEFEATED/VICTORY are strategic; the
    pair completes and reports either way — never silently dropped."""
    assert wp4.outcome_class("TRUNCATED") == "infrastructure_truncation"
    assert wp4.outcome_class("DEFEATED") == "strategic_terminal"
    assert wp4.outcome_class("VICTORY") == "strategic_terminal"
    assert wp4.outcome_class("COMPLETED") == "horizon_completed"
    assert wp4.survived("COMPLETED") is True
    assert wp4.survived("VICTORY") is True
    assert wp4.survived("DEFEATED") is False
    assert wp4.survived("TRUNCATED") is None


# ---------------------------------------------------------------------------
# W1-6 (a): sampling manifest truth — the shared actual-sampling helper
# ---------------------------------------------------------------------------

def test_actual_client_sampling_is_the_client_truth():
    """The helper returns the sampling the MistralClient ACTUALLY runs
    (decision 4, 2026-10-24 amendment): temperature 1.0, reasoning_effort
    "high" — never the fabricated greedy dict, never a top_p key."""
    from lechonk.model import mistral_client as mc
    from lechonk.model.sampling import actual_client_sampling

    s = actual_client_sampling()
    assert s["temperature"] == mc.DEFAULT_TEMPERATURE == 1.0
    assert s["reasoning_effort"] == mc.DEFAULT_REASONING_EFFORT == "high"
    assert actual_client_sampling(reasoning_effort="none")["reasoning_effort"] == "none"
    assert s["max_tokens"] == mc.DEFAULT_MAX_TOKENS
    assert s["response_format"] == "json_schema strict (WP2 spike decision)"
    assert "top_p" not in s       # the client never sends it — never recorded
    assert "random_seed" not in s  # the client sends none — nothing fabricated


def test_all_four_scripts_persist_the_actual_sampling():
    """All four run drivers call the shared helper and none keeps the
    fabricated sampling dict (source-level check: importing the WP2
    drivers would parse argv and create run dirs)."""
    for name in ("wp4_eval.py", "wp3_paired_smoke.py", "wp2_model_run.py",
                 "wp2_fault_run.py"):
        src = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "actual_client_sampling()" in src, name
        assert '"top_p"' not in src, name
        assert '"temperature": 0.0' not in src, name


# ---------------------------------------------------------------------------
# W1-6 (b): probe usage counted exactly once in api_totals
# ---------------------------------------------------------------------------

class _ProbeLogStub:
    """A CallLog stand-in with the to_dict() keys the record builder
    reads (per-request logs carry a real call_id; the AGGREGATE
    FingerprintProbe carries none — that difference is the fixture)."""

    def __init__(self, call_id, tokens_in, tokens_out, latency_ms,
                 cost_usd):
        self._d = {
            "call_id": call_id, "prompt_hash": "p", "kind": "probe",
            "model_requested": "mistral-large-4",
            "model_responded": "mistral-large-4",
            "enforcement": "json_schema", "temperature": 1.0,
            "latency_ms": latency_ms, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "tokens_total": tokens_in + tokens_out,
            "cost_usd": cost_usd, "finish_reason": "stop",
            "response_id": "r", "fingerprint": "sha256:x",
            "transport_retries": 0, "error": None,
            "ts": "2026-10-25T00:00:00+00:00",
        }

    def to_dict(self):
        return dict(self._d)


def test_probe_requests_logged_exactly_once(tmp_path):
    """Hand-computed fixture: probe1 = 2 requests (100+200 in, 10+20 out),
    re-probe = 1 request (50 in, 5 out), turn calls = 2 requests. The
    probe AGGREGATE (tokens_in=300, tokens_out=30 — the sum of probe1's
    requests) is never written, so api_totals sums each request exactly
    once. Under the pre-W1-6 accounting the aggregate was written next to
    its own constituents and tokens_in summed to 2250 — double-counted."""
    probe1_logs = [_ProbeLogStub("probe-1", 100, 10, 11, 0.001),
                   _ProbeLogStub("probe-2", 200, 20, 22, 0.002)]
    reprobe_logs = [_ProbeLogStub("reprobe-1", 50, 5, 12, 0.0005)]
    turn_calls = [
        {"turn": 5, "attempt": 1, "tokens_in": 1000, "tokens_out": 200,
         "latency_ms": 100, "cost_usd": 0.01},
        {"turn": 6, "attempt": 1, "tokens_in": 900, "tokens_out": 150,
         "latency_ms": 70, "cost_usd": 0.009},
    ]
    records = wp4.probe_request_log_records(probe1_logs, reprobe_logs)
    # one line per HTTP request — never the aggregate
    assert [r["call_id"] for r in records] == \
        ["probe-1", "probe-2", "reprobe-1"]
    assert [r["kind"] for r in records] == \
        ["probe", "probe", "probe_reprobe"]
    assert sum(r["tokens_in"] for r in records) == 350  # 100+200+50, not
    # +300 (the aggregate) — the aggregate never enters the log
    totals = wp4.api_totals_from_call_log(records + turn_calls)
    assert totals == {
        "calls": 5,
        "tokens_in": 100 + 200 + 50 + 1000 + 900,      # 2250 without the
        "tokens_out": 10 + 20 + 5 + 200 + 150,         # aggregate line
        "cost_usd": round(0.001 + 0.002 + 0.0005 + 0.01 + 0.009, 6),
        "latency_ms": 11 + 22 + 12 + 100 + 70,
    }


def test_probe_request_records_never_carry_the_aggregate():
    """The aggregate FingerprintProbe has no call_id/enforcement of its
    own; every emitted record must carry a real call_id from an
    underlying request (the pre-W1-6 code fell back to the literal
    'probe' call_id for the aggregate record)."""
    logs = [_ProbeLogStub(f"probe-{i}", 100, 10, 10, 0.001) for i in range(3)]
    records = wp4.probe_request_log_records(logs, [])
    assert len(records) == 3
    assert all(r["call_id"] != "probe" for r in records)
    assert all(r["enforcement"] is not None for r in records)


# ---------------------------------------------------------------------------
# W1-6 (c): the phase subprocess is bounded (timeout -> recorded failure)
# ---------------------------------------------------------------------------

def test_phase_timeout_derivation_from_config():
    """Documented derivation: overhead + turns x per-turn bound."""
    assert wp4.phase_timeout_s(0) == wp4.PHASE_OVERHEAD_S
    assert wp4.phase_timeout_s(4) == (wp4.PHASE_OVERHEAD_S
                                      + 4 * wp4.PHASE_PER_TURN_BOUND_S)
    # the default WP4 shape: create 4 turns, arms 50 turns
    assert wp4.phase_timeout_s(4) == 900 + 4 * 1200
    assert wp4.phase_timeout_s(50) == 900 + 50 * 1200


def test_run_subprocess_timeout_maps_to_failure_rc(tmp_path, monkeypatch):
    """An expired phase subprocess is killed and REPORTED (rc 124, the
    timeout recorded in the phase log) — never an unbounded wait, never
    a raised exception."""
    import subprocess as sp

    def fake_run(cmd, **kwargs):
        raise sp.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(wp4.subprocess, "run", fake_run)
    log_lines = []
    log_path = tmp_path / "phase.log"
    rc = wp4.run_subprocess(lambda m: log_lines.append(m),
                            ["create", "--pair-index", "1"],
                            str(log_path), timeout_s=5)
    assert rc == 124
    assert any("PHASE TIMEOUT" in m for m in log_lines)
    assert "[phase timeout] killed after 5s" in log_path.read_text(
        encoding="utf-8")


def test_run_phase_timeout_is_a_recorded_failure_not_a_crash(
        tmp_path, monkeypatch):
    """run_phase treats the timeout rc like any failure: ONE bounded
    retry, then the rc returns to the caller's existing recorded-failure
    handling."""
    seen_timeouts = []
    monkeypatch.setattr(
        wp4, "run_subprocess",
        lambda log, args, lp, timeout_s:
            seen_timeouts.append(timeout_s) or 124)
    monkeypatch.setattr(wp4.time, "sleep", lambda s: None)
    rc = wp4.run_phase(lambda m: None, ["create"],
                       str(tmp_path / "x.log"), timeout_s=7)
    assert rc == 124
    assert seen_timeouts == [7, 7]  # the retry carries the same bound
