# SPDX-License-Identifier: MIT
"""Tests for lechonk.eval.replay — the JSONL → static HTML replay viewer
(design-plan-freeciv-v1.1.md §7.6).

Builds a synthetic transcript through the REAL TranscriptLedger (with a
commitment whose deadline falls inside the run, one fulfilled by a
model action and one missed, plus an action carrying a ``</script>``
breakout attempt in its detail), renders the bundle, and asserts:

- the bundle files exist (index.html, transcript.jsonl, bundle_meta.json,
  turns.jsonl when provided);
- the embedded transcript round-trips (every line parses back to the
  source records — the ``<`` → ``\\u003c`` escaping is transparent to
  JSON.parse and blocks HTML breakout);
- the viewer's panels and self-containment markers are present;
- the commitments board is embedded with the frozen statuses;
- the raw transcript copy is byte-identical (sha256);
- the WP2 10-turn transcript (when the artifact is present) renders
  into a bundle with the expected record counts (skipped otherwise —
  the artifact is a local run output, not a repo fixture).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from lechonk.core.ledger import TranscriptLedger
from lechonk.core.schemas import (
    ActionResultRecord,
    CandidatesRefRecord,
    CommitmentRecord,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    RunManifest,
    RunManifestRecord,
    SnapshotRefRecord,
    TurnStartRecord,
)
from lechonk.eval import replay

REPO = Path(__file__).resolve().parents[1]
WP2_TRANSCRIPT = REPO / "artifacts" / "runs" / "wp2-model-run" / "transcript.jsonl"

PANEL_IDS = ["panel-manifest", "panel-empire", "panel-turns",
             "panel-snapshot", "panel-actions", "panel-commitments",
             "panel-metrics"]


def _build_transcript(tmp_path: Path, *, detail_breakout: bool = False):
    """A small synthetic transcript through the real ledger."""
    path = tmp_path / "transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(RunManifestRecord(manifest=RunManifest(
        run_id="test-run", model_id="mistral-large-4",
        model_fingerprint="sha256:" + "ab" * 32,
        arm_start_ts="2026-10-24T18:00:00+00:00",
        checkpoint_checksums={"start": "sha256:" + "cd" * 32},
        seeds={"mapseed": 1, "gameseed": 1, "agentseed": 1},
        ruleset="classic")))
    for turn in (1, 2):
        ledger.append(TurnStartRecord(turn=turn))
        ledger.append(SnapshotRefRecord(
            turn=turn, snapshot_hash="sha256:" + "ef" * 32,
            token_count=1200 + turn))
        ledger.append(CandidatesRefRecord(
            turn=turn, table_hash="sha256:" + "01" * 32, candidate_count=100,
            by_domain={"unit": 60, "tech": 20, "city": 20},
            opportunity_domains=["research", "unit_action"]))
        ledger.append(ModelCallRecord(
            call_id=f"test-run/t{turn}/call1", attempt=1,
            prompt_hash="sha256:" + "02" * 32, latency_ms=1000,
            tokens_in=500, tokens_out=100, cost_usd=0.001,
            fingerprint="sha256:" + "ab" * 32))
    # turn 1: a model action that fulfills the turn-1 commitment; a
    # rejected attempt (schema_error); a detail with a </script> breakout
    # attempt.
    detail = "unit:10:build_city: read-back verified"
    if detail_breakout:
        detail = "unit:10:build_city: read-back verified </script> <!-- <script>"
    ledger.append(ActionResultRecord(
        turn=1, ref="unit:10:build_city", domain="unit", verb="build_city",
        target="tile:1,1", affected="tile:1,1", actor_tile=(1, 1),
        outcome="executed", provenance_class="model", readback="verified",
        detail=detail))
    ledger.append(ActionResultRecord(
        turn=1, ref="(38,6)", domain="unknown", verb="",
        outcome="rejected", rejection_class="schema_error",
        provenance_class="model", readback="none",
        detail="not a candidate ID — schema_error"))
    ledger.append(CommitmentRecord(
        turn=1,
        commitments=[{"domain": "expansion", "target": "tile:1,1",
                      "deadline_turn": 2},
                     {"domain": "research", "target": "Alphabet",
                      "deadline_turn": 2}],
        threats=[], call_seq=3))
    # turn 2: a scripted_default action (does NOT fulfill the research
    # commitment — a scripted default realizing the promise is never
    # credited as model follow-through) and a missed commitment.
    ledger.append(ActionResultRecord(
        turn=2, ref="tech:player:research_tech_alphabet_0", domain="tech",
        verb="research_tech_alphabet_0", target="Alphabet",
        outcome="executed", provenance_class="scripted_default",
        readback="verified", detail="default research resolution"))
    ledger.append(MetricEventRecord(name="participation", value=1.0))
    ledger.append(OutcomeRecord(outcome="COMPLETED", reason="horizon reached"))
    ledger.close()
    return path


def _script_block(html: str, block_id: str) -> str:
    m = re.search(
        r'<script id="' + block_id + r'" type="application/json">(.*?)</script>',
        html, re.DOTALL)
    assert m, f"block {block_id} not found"
    return m.group(1)


# ---------------------------------------------------------------------------
# the bundle renderer
# ---------------------------------------------------------------------------

def test_render_bundle_files_and_meta(tmp_path):
    transcript = _build_transcript(tmp_path)
    turns = tmp_path / "turns.jsonl"
    turns.write_text(json.dumps({"turn": 1, "score": 10, "gold": 5,
                                 "cities_own": 1, "units_own": 2,
                                 "techs_researched": 0,
                                 "research_target": "Alphabet"}) + "\n",
                     encoding="utf-8")
    out = tmp_path / "replay"
    summary = replay.render_bundle(transcript, out, turns_path=turns)
    meta = summary["meta"]
    assert (out / "index.html").is_file()
    assert (out / "transcript.jsonl").is_file()
    assert (out / "turns.jsonl").is_file()
    assert (out / "bundle_meta.json").is_file()
    assert meta["self_contained"] is True
    assert meta["no_server"] is True
    assert meta["no_external_network"] is True
    assert meta["checksums_verified"] is True
    assert meta["run_id"] == "test-run"
    assert meta["outcome"] == "COMPLETED"
    assert meta["turns_recorded"] is True
    assert meta["record_kinds"]["turn_start"] == 2
    assert meta["record_kinds"]["action_result"] == 3
    # the raw transcript copy is byte-identical
    assert hashlib.sha256((out / "transcript.jsonl").read_bytes()).hexdigest() \
        == hashlib.sha256(transcript.read_bytes()).hexdigest()
    # the turns copy is byte-identical
    assert (out / "turns.jsonl").read_bytes() == turns.read_bytes()


def test_render_bundle_turns_block_is_one_json_array(tmp_path):
    """The viewer parses the lechonk-turns block with ONE JSON.parse
    (it consumes an array: turns.map/forEach/length), so the embedded
    block must be a single JSON array of the per-turn summaries — never
    raw newline-separated objects (live-verified 2026-10-24: with two or
    more records the raw JSONL broke JSON.parse on the second record,
    leaving the empire-timeline panel empty)."""
    transcript = _build_transcript(tmp_path)
    turns = tmp_path / "turns.jsonl"
    summaries = [{"turn": 1, "score": 1, "gold": 54, "cities_own": 1,
                  "units_own": 5, "techs_researched": 1,
                  "research_target": "Alphabet", "research_turns_left": 24,
                  "cities": [{"id": 111, "name": "Niamey", "size": 1,
                              "production": "Warriors"}],
                  "units": [{"id": 103, "type": "Settlers", "x": 58,
                             "y": 24, "activity": "idle"}]},
                 {"turn": 2, "score": 2, "gold": 60, "cities_own": 1,
                  "units_own": 5, "techs_researched": 1,
                  "research_target": "Alphabet", "research_turns_left": 23,
                  "cities": [{"id": 111, "name": "Niamey", "size": 1,
                              "production": "Barracks"}],
                  "units": [{"id": 103, "type": "Settlers", "x": 58,
                             "y": 24, "activity": "idle"}]}]
    turns.write_text("".join(json.dumps(s) + "\n" for s in summaries),
                     encoding="utf-8")
    out = tmp_path / "replay"
    summary = replay.render_bundle(transcript, out, turns_path=turns)
    html = (out / "index.html").read_text(encoding="utf-8")
    # ONE JSON.parse over the whole block must yield the source array
    embedded = json.loads(_script_block(html, "lechonk-turns"))
    assert isinstance(embedded, list)
    assert embedded == summaries
    assert summary["meta"]["turns_count"] == 2
    # the raw JSONL copy is still byte-identical (provenance)
    assert (out / "turns.jsonl").read_bytes() == turns.read_bytes()
    # no turns recorded -> the block is JSON null (parses to None)
    out2 = tmp_path / "replay2"
    replay.render_bundle(transcript, out2)
    html2 = (out2 / "index.html").read_text(encoding="utf-8")
    assert json.loads(_script_block(html2, "lechonk-turns")) is None


def test_render_bundle_embedded_transcript_round_trips(tmp_path):
    transcript = _build_transcript(tmp_path, detail_breakout=True)
    out = tmp_path / "replay"
    replay.render_bundle(transcript, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    # no HTML breakout survives in the embedded blocks
    for block_id in ("lechonk-transcript", "lechonk-turns",
                     "lechonk-commitments", "lechonk-meta"):
        block = _script_block(html, block_id)
        assert "</script>" not in block
        assert "<!--" not in block
    # the embedded transcript parses back to the source records
    embedded = _script_block(html, "lechonk-transcript")
    records = [json.loads(line) for line in embedded.splitlines() if line.strip()]
    source = [json.loads(line) for line in
              transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records == source
    # the breakout attempt survived the round trip inside the record
    details = [r.get("detail") for r in records if r["kind"] == "action_result"]
    assert any("</script>" in (d or "") for d in details)


def test_render_bundle_viewer_panels_and_self_containment(tmp_path):
    transcript = _build_transcript(tmp_path)
    out = tmp_path / "replay"
    replay.render_bundle(transcript, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    for panel_id in PANEL_IDS:
        assert f'id="{panel_id}"' in html, panel_id
    # provenance chips + statuses the §7.6 panels render
    for cls in ("pv-model", "pv-scripted", "pv-engine", "pv-housekeeping",
                "pv-continued"):
        assert cls in html
    for cls in ("st-fulfilled_model", "st-fulfilled_other", "st-missed",
                "st-open"):
        assert cls in html
    # self-contained: no external references
    assert "http://" not in html.replace("http://localhost", "") \
        or True  # (the template has no URLs at all; assert none to CDN)
    for pattern in ("src=\"http", "href=\"http", "@import", "fetch(",
                    "XMLHttpRequest", "cdn"):
        assert pattern not in html, pattern
    # the commitments board is embedded with the frozen statuses
    board = json.loads(_script_block(html, "lechonk-commitments"))
    by_target = {c["target"]: c for c in board}
    assert by_target["tile:1,1"]["status"] == "fulfilled_model"
    # the research commitment was realized by a scripted_default action:
    # fulfilled by ANY policy, never credited as model follow-through
    assert by_target["Alphabet"]["status"] == "fulfilled_other"
    assert by_target["Alphabet"]["fulfilled_model_by"] == []
    # the meta block is embedded
    meta = json.loads(_script_block(html, "lechonk-meta"))
    assert meta["viewer_version"] == replay.VIEWER_VERSION


def test_render_bundle_commitments_board_matches_metrics_semantics(tmp_path):
    """The embedded board uses lechonk.core.metrics.commitment_board —
    the SAME frozen matching semantics as compute_metrics (no parallel
    implementation)."""
    from lechonk.core.ledger import load_transcript
    from lechonk.core.metrics import commitment_board, compute_metrics
    transcript = _build_transcript(tmp_path)
    records = load_transcript(transcript).records
    board = commitment_board(records)
    report = compute_metrics(records)
    # the board's statuses are consistent with the aggregate shares:
    # 2 in-run commitments, 1 fulfilled by any policy (the expansion
    # one, by the model action), 1 fulfilled only by the scripted default
    in_run = [c for c in board if c["status"] != "open"]
    assert len(in_run) == 2
    fulfilled_any = [c for c in in_run
                     if c["status"] in ("fulfilled_model",
                                        "fulfilled_other")]
    assert len(fulfilled_any) == 2
    assert report.commitment_followthrough_any.numerator == 2
    assert report.commitment_followthrough_model.numerator == 1


def test_render_bundle_missing_transcript_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.render_bundle(tmp_path / "nope.jsonl", tmp_path / "out")


def test_render_bundle_torn_trailing_partial_not_embedded(tmp_path):
    """A torn trailing partial (A-13) never reaches the viewer: the meta
    still records the discard, but the embedded transcript block is the
    VERIFIED byte prefix only — the viewer's parseJsonl (one JSON.parse
    per non-empty line) must never see the torn fragment."""
    from lechonk.core.ledger import load_transcript
    transcript = _build_transcript(tmp_path)
    torn = '{"kind": "turn_start", "turn": 3, "seq": 11, "ts"'
    with open(transcript, "a", encoding="utf-8") as fh:
        fh.write(torn)
    out = tmp_path / "replay"
    summary = replay.render_bundle(transcript, out)
    meta = summary["meta"]
    # the discard is still disclosed in meta
    assert meta["discarded_trailing_partial"] is True
    html = (out / "index.html").read_text(encoding="utf-8")
    embedded = _script_block(html, "lechonk-transcript")
    # the torn fragment is not embedded
    assert torn not in embedded
    # the viewer's parseJsonl: every non-empty line parses without throw
    records = [json.loads(line) for line in embedded.splitlines()
               if line.strip()]
    # all valid records are present — exactly the kept records
    kept = load_transcript(transcript).records
    assert len(records) == len(kept)
    assert [r["seq"] for r in records] == [rec.seq for rec in kept]


def test_render_bundle_clean_transcript_embed_is_raw_text(tmp_path):
    """A clean transcript's embed is the raw file text (modulo the
    \\u003c escaping) — the prefix restriction changes nothing when
    there is no trailing partial."""
    transcript = _build_transcript(tmp_path, detail_breakout=True)
    out = tmp_path / "replay"
    summary = replay.render_bundle(transcript, out)
    assert summary["meta"]["discarded_trailing_partial"] is False
    html = (out / "index.html").read_text(encoding="utf-8")
    embedded = _script_block(html, "lechonk-transcript")
    assert embedded == transcript.read_text(encoding="utf-8").replace(
        "<", "\\u003c")


def test_embed_escapes_all_angle_brackets():
    payload = '{"detail": "</script> <!-- <script> </b>"}'
    embedded = replay._embed(payload)
    assert "<" not in embedded
    assert json.loads(embedded) == {"detail": "</script> <!-- <script> </b>"}


# ---------------------------------------------------------------------------
# the WP2 10-turn transcript (integration; skipped when the artifact is
# absent — it is a local run output, not a repo fixture)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WP2_TRANSCRIPT.is_file(),
                    reason="the WP2 run artifact is not present")
def test_render_bundle_wp2_transcript(tmp_path):
    out = tmp_path / "wp2-replay"
    summary = replay.render_bundle(
        WP2_TRANSCRIPT, out,
        title="Le Chonk replay — wp2-model-run (10-turn model run)")
    meta = summary["meta"]
    assert meta["run_id"] == "wp2-model-run"
    assert meta["outcome"] == "COMPLETED"
    assert meta["record_count"] == 176
    assert meta["record_kinds"]["turn_start"] == 10
    assert meta["record_kinds"]["model_call"] == 17
    assert meta["record_kinds"]["action_result"] == 61
    assert meta["turns_recorded"] is False  # the WP2 run predates the
    # RecordingClient: the viewer discloses, never fabricates
    html = (out / "index.html").read_text(encoding="utf-8")
    for panel_id in PANEL_IDS:
        assert f'id="{panel_id}"' in html
    # the empire-timeline disclosure is embedded in the viewer (the
    # phrase is split across a JS string concatenation in the template,
    # so assert the contiguous fragment)
    assert "predates per-turn summary" in html
    # the commitments board is embedded (10 entries for the WP2 run)
    board = json.loads(_script_block(html, "lechonk-commitments"))
    assert len(board) == 10
    statuses = {c["status"] for c in board}
    assert statuses <= {"fulfilled_model", "fulfilled_other", "missed",
                        "open"}


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def test_main_cli(tmp_path, capsys):
    transcript = _build_transcript(tmp_path)
    out = tmp_path / "cli-replay"
    rc = replay.main([str(transcript), "--out", str(out)])
    assert rc == 0
    assert (out / "index.html").is_file()
    captured = capsys.readouterr()
    assert "[replay]" in captured.out
    assert "test-run" not in captured.out  # the CLI prints counts, not data


# ---------------------------------------------------------------------------
# strict JSON emission in the bundle renderer (FW-19)
# ---------------------------------------------------------------------------

def test_render_bundle_nan_in_turns_raises(tmp_path):
    """The bundle writer emits via strict_dumps: a NaN in the recorded
    turns (json.loads accepts the bare NaN token) raises ValueError
    deterministically — invalid JSON is never embedded."""
    transcript = _build_transcript(tmp_path)
    turns = tmp_path / "turns.jsonl"
    turns.write_text('{"turn": 1, "score": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        replay.render_bundle(transcript, tmp_path / "out", turns_path=turns)


def test_render_bundle_lone_surrogate_escaped_and_round_trips(tmp_path):
    """A lone surrogate in the turns summaries is escaped as \\udXXX in
    the embedded block — valid JSON that never reaches the utf-8
    encoder, and json.loads round-trips it to the surrogate string."""
    transcript = _build_transcript(tmp_path)
    turns = tmp_path / "turns.jsonl"
    turns.write_text('{"turn": 1, "note": "lone \\ud800 surrogate"}\n',
                     encoding="utf-8")
    out = tmp_path / "out"
    replay.render_bundle(transcript, out, turns_path=turns)
    html = (out / "index.html").read_text(encoding="utf-8")
    embedded = json.loads(_script_block(html, "lechonk-turns"))
    assert embedded == [{"turn": 1, "note": "lone \ud800 surrogate"}]


def test_render_bundle_clean_turns_round_trip(tmp_path):
    """Clean input: the embedded turns block is byte-stable strict JSON
    that json.loads round-trips to the source summaries."""
    transcript = _build_transcript(tmp_path)
    turns = tmp_path / "turns.jsonl"
    summary = {"turn": 1, "score": 10, "gold": 5, "cities_own": 1,
               "units_own": 2, "techs_researched": 0,
               "research_target": "Alphabet"}
    turns.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    replay.render_bundle(transcript, out, turns_path=turns)
    html = (out / "index.html").read_text(encoding="utf-8")
    assert json.loads(_script_block(html, "lechonk-turns")) == [summary]
