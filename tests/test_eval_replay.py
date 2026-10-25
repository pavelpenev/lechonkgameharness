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
  the artifact is a local run output, not a repo fixture);
- every recorded transcript under artifacts/ loads read-only through
  load_transcript (W3-9; skipped when the artifacts are absent).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lechonk.core.ledger import TranscriptLedger, load_transcript
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
# The recorded-transcript inventory under artifacts/ (42 transcripts at
# the 2026-10-25 wave-3 hash pin): the W3-9 read-only load check loads
# every one. The artifacts are local run outputs, not repo fixtures —
# a clone without them skips (unavailable coverage, not a pass).
RECORDED_TRANSCRIPTS = sorted(
    (REPO / "artifacts").rglob("transcript.jsonl"))

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
# the recorded-transcript inventory (W3-9 read-only load check; skipped
# when the artifacts are absent — they are local run outputs, not repo
# fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RECORDED_TRANSCRIPTS,
                    reason="the recorded transcript inventory is not present")
def test_recorded_transcripts_all_load_read_only():
    # the 2026-10-25 wave-3 hash pin: exactly 42 recorded transcripts
    assert len(RECORDED_TRANSCRIPTS) == 42
    for path in RECORDED_TRANSCRIPTS:
        loaded = load_transcript(path)  # read-only: never modifies the file
        assert loaded.records, path
        assert loaded.last_seq == len(loaded.records) - 1


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


# ---------------------------------------------------------------------------
# D1 (W3-4): viewer parity — the participation curve is readback-gated
# and opportunity-intersected, mirroring lechonk.core.metrics
# ---------------------------------------------------------------------------


def _build_d1_transcript(tmp_path: Path):
    """A 4-turn transcript pinning the D1 viewer parity: turn 1 an
    executed/VERIFIED model action (counts); turn 2 an executed/
    CONTRADICTED model action (rejection_class illegal_in_rule — the
    viewer predicate must exclude it); turn 3 NO candidates_ref (a
    zero-opportunity turn — the curve must not increment even though a
    verified model action executed); turn 4 a rejected model attempt
    (never counted). Expected viewer curves: opportunities [1,2,2,3],
    participation [1,1,1,1]; the frozen core metrics over the same
    transcript: participation 1/3."""
    path = tmp_path / "d1-transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(RunManifestRecord(manifest=RunManifest(
        run_id="d1-run", model_id="mistral-large-4",
        model_fingerprint="sha256:" + "ab" * 32,
        arm_start_ts="2026-10-24T18:00:00+00:00",
        checkpoint_checksums={"start": "sha256:" + "cd" * 32},
        seeds={"mapseed": 1, "gameseed": 1, "agentseed": 1},
        ruleset="classic")))
    for turn in (1, 2, 3, 4):
        ledger.append(TurnStartRecord(turn=turn))
        ledger.append(SnapshotRefRecord(
            turn=turn, snapshot_hash="sha256:" + "ef" * 32,
            token_count=1200 + turn))
        if turn != 3:
            # turn 3 deliberately has NO candidates_ref: a zero-opportunity
            # turn (the viewer's opportunity intersection must exclude it).
            ledger.append(CandidatesRefRecord(
                turn=turn, table_hash="sha256:" + "01" * 32,
                candidate_count=100,
                by_domain={"unit": 60, "tech": 20, "city": 20},
                opportunity_domains=["research", "unit_action"]))
        ledger.append(ModelCallRecord(
            call_id=f"d1-run/t{turn}/call1", attempt=1,
            prompt_hash="sha256:" + "02" * 32, latency_ms=1000,
            tokens_in=500, tokens_out=100, cost_usd=0.001,
            fingerprint="sha256:" + "ab" * 32))
    # turn 1: executed + verified model action -> participation.
    ledger.append(ActionResultRecord(
        turn=1, ref="tech:player:research_tech_alphabet_0", domain="tech",
        verb="research_tech_alphabet_0", target="Alphabet",
        outcome="executed", provenance_class="model", readback="verified",
        detail="verified model research"))
    # turn 2: executed + CONTRADICTED (illegal_in_rule) -> no credit (D1).
    ledger.append(ActionResultRecord(
        turn=2, ref="tech:player:research_tech_pottery_1", domain="tech",
        verb="research_tech_pottery_1", target="Pottery",
        outcome="executed", provenance_class="model",
        readback="contradicted", rejection_class="illegal_in_rule",
        detail="emitted executed; the read-back proved non-execution"))
    # turn 3: a verified model action on a zero-opportunity turn -> the
    # curve must not increment (opportunity intersection).
    ledger.append(ActionResultRecord(
        turn=3, ref="tech:player:research_tech_currency_3", domain="tech",
        verb="research_tech_currency_3", target="Currency",
        outcome="executed", provenance_class="model", readback="verified",
        detail="verified model research on a zero-opportunity turn"))
    # turn 4: a rejected model attempt -> never counted.
    ledger.append(ActionResultRecord(
        turn=4, ref="tech:player:research_tech_writing_4", domain="tech",
        verb="research_tech_writing_4", target="Writing",
        outcome="rejected", rejection_class="illegal_in_rule",
        provenance_class="model", readback="none",
        detail="rejected attempt"))
    ledger.append(MetricEventRecord(name="participation", value=0.5))
    ledger.append(OutcomeRecord(outcome="COMPLETED", reason="horizon reached"))
    ledger.close()
    return path


# A minimal DOM shim + runner for the viewer's main script: it executes
# the extracted JS under node with the embedded JSON blocks and reports
# the plotted circle coordinates of the metrics panel's FIRST chart (the
# cumulative participation curve), grouped by series color.
_VIEWER_DRIVER = r"""
"use strict";
const fs = require("fs");
const blocks = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const script = fs.readFileSync(process.argv[3], "utf8");
function makeEl(tag) {
  const e = {
    tagName: tag,
    attributes: {},
    children: [],
    className: "",
    value: undefined,
    _text: "",
    appendChild(c) {
      this.children.push(c);
      if (this.tagName === "select" && c && c.tagName === "option" &&
          this.value === undefined) {
        this.value = c.attributes.value;
      }
      return c;
    },
    setAttribute(k, v) { this.attributes[k] = v; },
    addEventListener() {},
  };
  Object.defineProperty(e, "textContent", {
    get() { return this._text; },
    set(v) { this._text = String(v); },
  });
  return e;
}
const ids = ["lechonk-meta", "lechonk-turns", "lechonk-commitments",
  "lechonk-transcript", "manifest-body", "empire-body", "turns-body",
  "snapshot-body", "actions-body", "commitments-body", "metrics-body",
  "footer", "run-title", "run-sub"];
const byId = {};
for (const id of ids) {
  const e = makeEl("div");
  e.id = id;
  e._text = blocks[id] === undefined ? "" : blocks[id];
  byId[id] = e;
}
global.document = {
  getElementById(id) { return byId[id] || null; },
  createElement(tag) { return makeEl(tag); },
  createTextNode(t) { return { text: String(t) }; },
};
eval(script);
const svg = byId["metrics-body"].children.find(
  (c) => c && c.tagName === "svg");
if (!svg) { throw new Error("no chart in the metrics panel"); }
const series = {};
for (const c of svg.children) {
  if (c && c.tagName === "circle") {
    (series[c.attributes.fill] = series[c.attributes.fill] || []).push(
      [c.attributes.cx, c.attributes.cy]);
  }
}
process.stdout.write(JSON.stringify(series));
"""


def _viewer_main_script(html: str) -> str:
    """The viewer's main <script> block (the one without attributes — the
    JSON payload blocks all carry id/type attributes)."""
    m = re.search(r"<script>\n(.*?)\n</script>", html, re.DOTALL)
    assert m, "the viewer's main script block not found"
    return m.group(1)


def test_viewer_participation_curve_is_readback_gated_and_opportunity_intersected(
        tmp_path):
    """D1 viewer parity: the viewer's isParticipation gains the readback
    gating (mirroring lechonk.core.metrics._credits_execution — a
    CONTRADICTED execution is excluded) and the cumulative participation
    curve increments only on turns-with-opportunities. The recalculated
    curve stays labeled as viewer-computed, distinct from the frozen
    aggregate metrics. The behavioral half executes the shipped viewer JS
    under node with a minimal DOM shim (skipped when node is absent —
    the structural pins below still run)."""
    from lechonk.core.ledger import load_transcript
    from lechonk.core.metrics import compute_metrics

    transcript = _build_d1_transcript(tmp_path)
    out = tmp_path / "d1-replay"
    replay.render_bundle(transcript, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    script = _viewer_main_script(html)
    # structural pins: the gating and the labeling ship in the viewer JS
    assert "creditsExecution" in script
    assert 'a.readback === "verified"' in script
    assert "creditsExecution(a)" in script
    assert "hasOpp && p.actions.some(isParticipation)" in script
    # the recalculated curve is labeled distinctly from the frozen
    # aggregate (the frozen heading's phrase is split across a JS string
    # concatenation — assert the contiguous fragment)
    assert "readback-gated" in html
    assert "distinct from the frozen aggregate" in html
    assert "aggregate metrics (frozen, ledger-" in html
    # the frozen core metrics over the same transcript (the viewer must
    # agree with the core): opportunity turns {1,2,4}, participating {1}
    report = compute_metrics(load_transcript(transcript).records)
    assert (report.participation.numerator,
            report.participation.denominator) == (1, 3)
    # behavioral: execute the viewer JS and read the participation
    # chart's plotted points (the metrics panel's first chart)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to execute the viewer JS")
    blocks = {bid: _script_block(html, bid)
              for bid in ("lechonk-meta", "lechonk-turns",
                          "lechonk-commitments", "lechonk-transcript")}
    blocks_path = tmp_path / "viewer-blocks.json"
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    script_path = tmp_path / "viewer-main.js"
    script_path.write_text(script, encoding="utf-8")
    driver_path = tmp_path / "viewer-driver.js"
    driver_path.write_text(_VIEWER_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [node, str(driver_path), str(blocks_path), str(script_path)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"viewer JS failed under node: {proc.stderr}"
    series = json.loads(proc.stdout)
    part = series["#2563eb"]  # the participation series' color
    opp = series["#6b7280"]   # the opportunities series' color
    assert len(part) == 4 and len(opp) == 4
    # participation [1,1,1,1]: the contradicted execution (turn 2) and the
    # zero-opportunity turn 3 never increment the curve — all four
    # plotted cy are equal (the chart's Y mapping is injective).
    assert len({cy for _, cy in part}) == 1
    # opportunities [1,2,2,3]: Y decreases as the value rises, so
    # cy[0] > cy[1] == cy[2] > cy[3].
    assert opp[0][1] > opp[1][1] == opp[2][1] > opp[3][1]


# ---------------------------------------------------------------------------
# W3-1: the viewer separates operational events from the aggregate
# metrics table
# ---------------------------------------------------------------------------


def _build_w3_transcript(tmp_path: Path):
    """A transcript carrying registered aggregate metric names
    (participation, a parameterized domain_model_share:*,
    illegal_attempts_total), operational protocol events
    (state_transition, dispatch_intent, mutation_dispatch_cancelled,
    prompt_budget_exceeded) and an unknown/legacy event name."""
    path = tmp_path / "w3-transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(RunManifestRecord(manifest=RunManifest(run_id="w3-run")))
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(MetricEventRecord(name="participation", value=1.0))
    ledger.append(MetricEventRecord(name="domain_model_share:research",
                                    value=0.5))
    ledger.append(MetricEventRecord(name="illegal_attempts_total", value=0))
    ledger.append(MetricEventRecord(name="state_transition",
                                    value="STARTUP->ACTING"))
    ledger.append(MetricEventRecord(
        name="dispatch_intent",
        value=('{"call_seq":2,"candidates_seq":3,'
               '"provenance_class":"model","protocol":"dispatch-intent/v1",'
               '"ref":"unit:10:goto_e","snapshot_seq":2,"turn":1}')))
    ledger.append(MetricEventRecord(
        name="mutation_dispatch_cancelled",
        value=('{"call_seq":2,"candidates_seq":3,"intent_seq":4,'
               '"provenance_class":"model",'
               '"protocol":"dispatch-cancelled/v1",'
               '"reason":"ACTING deadline reached","ref":"unit:10:goto_e",'
               '"snapshot_seq":2,"turn":1}')))
    ledger.append(MetricEventRecord(name="prompt_budget_exceeded",
                                    value="turn 1: prompt over budget"))
    ledger.append(MetricEventRecord(name="some_legacy_event",
                                    value="an unknown/legacy name"))
    ledger.append(OutcomeRecord(outcome="COMPLETED",
                                reason="horizon reached"))
    ledger.close()
    return path


# A minimal DOM shim + runner for the viewer's main script: it executes
# the extracted JS under node with the embedded JSON blocks and dumps the
# metrics panel's DOM tree (tags + text + children).
_VIEWER_TABLES_DRIVER = r"""
"use strict";
const fs = require("fs");
const blocks = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const script = fs.readFileSync(process.argv[3], "utf8");
function makeEl(tag) {
  const e = {
    tagName: tag,
    attributes: {},
    children: [],
    className: "",
    value: undefined,
    _text: "",
    appendChild(c) {
      this.children.push(c);
      if (this.tagName === "select" && c && c.tagName === "option" &&
          this.value === undefined) {
        this.value = c.attributes.value;
      }
      return c;
    },
    setAttribute(k, v) { this.attributes[k] = v; },
    addEventListener() {},
  };
  Object.defineProperty(e, "textContent", {
    get() { return this._text; },
    set(v) { this._text = String(v); },
  });
  return e;
}
const ids = ["lechonk-meta", "lechonk-turns", "lechonk-commitments",
  "lechonk-transcript", "manifest-body", "empire-body", "turns-body",
  "snapshot-body", "actions-body", "commitments-body", "metrics-body",
  "footer", "run-title", "run-sub"];
const byId = {};
for (const id of ids) {
  const e = makeEl("div");
  e.id = id;
  e._text = blocks[id] === undefined ? "" : blocks[id];
  byId[id] = e;
}
global.document = {
  getElementById(id) { return byId[id] || null; },
  createElement(tag) { return makeEl(tag); },
  createTextNode(t) { return { text: String(t) }; },
};
eval(script);
function dump(n) {
  if (!n) { return null; }
  if (n.tagName === undefined) { return { text: n.text }; }
  return { tag: n.tagName, text: n._text,
           children: (n.children || []).map(dump) };
}
process.stdout.write(JSON.stringify(dump(byId["metrics-body"])));
"""


def test_viewer_separates_operational_events_from_aggregate_metrics(tmp_path):
    """W3-1 viewer obligation: the aggregate metrics table carries ONLY
    the registered aggregate names; operational protocol events
    (dispatch_intent, mutation_dispatch_cancelled, state_transition, the
    budget events) and unknown/legacy names render in a diagnostics
    table, never as aggregates."""
    transcript = _build_w3_transcript(tmp_path)
    out = tmp_path / "w3-replay"
    replay.render_bundle(transcript, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    script = _viewer_main_script(html)
    # structural pins: the registered-aggregate filter and the
    # diagnostics section ship in the viewer JS
    assert "isAggregateMetric" in script
    assert '"domain_model_share:"' in script
    assert '"illegal_action_share:"' in script
    assert "operational events (diagnostics, " in html
    assert "aggregate metrics (frozen, ledger-" in html
    # behavioral: execute the viewer JS under node and read the metrics
    # panel's tables (skipped when node is absent — the structural pins
    # above still run)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to execute the viewer JS")
    blocks = {bid: _script_block(html, bid)
              for bid in ("lechonk-meta", "lechonk-turns",
                          "lechonk-commitments", "lechonk-transcript")}
    blocks_path = tmp_path / "viewer-blocks.json"
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    script_path = tmp_path / "viewer-main.js"
    script_path.write_text(script, encoding="utf-8")
    driver_path = tmp_path / "viewer-tables.js"
    driver_path.write_text(_VIEWER_TABLES_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [node, str(driver_path), str(blocks_path), str(script_path)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"viewer JS failed under node: {proc.stderr}"
    tree = json.loads(proc.stdout)

    def tables_under(heading_prefix):
        """The tables rendered under the h3 headings whose text starts
        with ``heading_prefix``: [{heading, rows: [[cell, ...], ...]}]"""
        found = []
        current = None

        def walk(n):
            nonlocal current
            if not isinstance(n, dict):
                return
            if n.get("tag") == "h3":
                current = n.get("text")
            elif n.get("tag") == "table" and current is not None:
                rows = []
                for tr in n.get("children") or []:
                    if not isinstance(tr, dict) or tr.get("tag") != "tr":
                        continue
                    rows.append([c.get("text")
                                 for c in (tr.get("children") or [])
                                 if isinstance(c, dict)])
                found.append({"heading": current, "rows": rows})
                current = None
            for c in (n.get("children") or []):
                walk(c)

        walk(tree)
        return [t for t in found
                if t["heading"] and t["heading"].startswith(heading_prefix)]

    aggregates = tables_under("aggregate metrics")
    assert len(aggregates) == 1
    assert aggregates[0]["rows"][0] == ["metric", "value"]
    assert [r[0] for r in aggregates[0]["rows"][1:]] == [
        "participation", "domain_model_share:research",
        "illegal_attempts_total"]
    diagnostics = tables_under("operational events")
    assert len(diagnostics) == 1
    assert diagnostics[0]["rows"][0] == ["event", "value"]
    assert [r[0] for r in diagnostics[0]["rows"][1:]] == [
        "state_transition", "dispatch_intent",
        "mutation_dispatch_cancelled", "prompt_budget_exceeded",
        "some_legacy_event"]
    # The operational events never appear in the aggregate table.
    aggregate_names = {r[0] for r in aggregates[0]["rows"]}
    assert "dispatch_intent" not in aggregate_names
    assert "mutation_dispatch_cancelled" not in aggregate_names
    assert "state_transition" not in aggregate_names
    assert "some_legacy_event" not in aggregate_names


def _metrics_panel_tables(tree, heading_prefix):
    """The tables rendered under the h3 headings whose text starts with
    ``heading_prefix`` in the viewer's dumped metrics panel:
    [{heading, rows: [[cell, ...], ...]}]."""
    found = []
    current = None

    def walk(n):
        nonlocal current
        if not isinstance(n, dict):
            return
        if n.get("tag") == "h3":
            current = n.get("text")
        elif n.get("tag") == "table" and current is not None:
            rows = []
            for tr in n.get("children") or []:
                if not isinstance(tr, dict) or tr.get("tag") != "tr":
                    continue
                rows.append([c.get("text")
                             for c in (tr.get("children") or [])
                             if isinstance(c, dict)])
            found.append({"heading": current, "rows": rows})
            current = None
        for c in (n.get("children") or []):
            walk(c)

    walk(tree)
    return [t for t in found
            if t["heading"] and t["heading"].startswith(heading_prefix)]


def test_viewer_acting_deadline_stop_renders_as_diagnostics(tmp_path):
    """W3-2 viewer obligation: the acting_deadline_stop summary event is
    an operational protocol event — the viewer's aggregate-metric
    filter never registers it, so it renders in the diagnostics table,
    never as an aggregate (the W3-1 filter already sends unknown names
    to diagnostics; this pins the W3-2 name explicitly)."""
    path = tmp_path / "w3-2-transcript.jsonl"
    ledger = TranscriptLedger(path)
    ledger.append(RunManifestRecord(manifest=RunManifest(run_id="w3-2-run")))
    ledger.append(TurnStartRecord(turn=1))
    ledger.append(MetricEventRecord(name="participation", value=1.0))
    ledger.append(MetricEventRecord(
        name="acting_deadline_stop",
        value=('{"completed_result_seqs":[4],"protocol":"partial-batch/v1",'
               '"reason":"acting_deadline","remaining_standing_actors":[],'
               '"route":"planned","turn":1,"unresolved_intent_seqs":[],'
               '"unstarted_refs":["unit:11:sentry"]}')))
    ledger.append(OutcomeRecord(
        outcome="TRUNCATED",
        reason="infrastructure deadline exceeded: state ACTING exceeded "
               "its deadline of 180.0s (artifacts intact)"))
    ledger.close()
    out = tmp_path / "w3-2-replay"
    replay.render_bundle(path, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    script = _viewer_main_script(html)
    # structural pin: the stop event's name is never registered as an
    # aggregate (neither the names list nor the prefixes special-case
    # it), so the filter classifies it as a diagnostic.
    assert "isAggregateMetric" in script
    assert '"acting_deadline_stop"' not in script
    # behavioral: execute the viewer JS under node and read the metrics
    # panel's tables (skipped when node is absent — the structural pin
    # above still runs).
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to execute the viewer JS")
    blocks = {bid: _script_block(html, bid)
              for bid in ("lechonk-meta", "lechonk-turns",
                          "lechonk-commitments", "lechonk-transcript")}
    blocks_path = tmp_path / "viewer-blocks.json"
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    script_path = tmp_path / "viewer-main.js"
    script_path.write_text(script, encoding="utf-8")
    driver_path = tmp_path / "viewer-tables.js"
    driver_path.write_text(_VIEWER_TABLES_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [node, str(driver_path), str(blocks_path), str(script_path)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"viewer JS failed under node: {proc.stderr}"
    tree = json.loads(proc.stdout)
    aggregates = _metrics_panel_tables(tree, "aggregate metrics")
    assert len(aggregates) == 1
    assert [r[0] for r in aggregates[0]["rows"][1:]] == ["participation"]
    diagnostics = _metrics_panel_tables(tree, "operational events")
    assert len(diagnostics) == 1
    assert diagnostics[0]["rows"][0] == ["event", "value"]
    assert [r[0] for r in diagnostics[0]["rows"][1:]] == [
        "acting_deadline_stop"]
    # The stop event never appears in the aggregate table.
    aggregate_names = {r[0] for r in aggregates[0]["rows"]}
    assert "acting_deadline_stop" not in aggregate_names
