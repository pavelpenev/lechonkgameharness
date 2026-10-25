# SPDX-License-Identifier: MIT
"""lechonk.eval.replay — JSONL transcript → static HTML replay viewer (MIT).

design-plan-freeciv-v1.1.md §7.6: "Single self-contained HTML + JS
loading a JSONL transcript (bundled per run): per-turn timeline (score
curve, cities, units, gold), snapshot summary panel, model batch with
provenance chips (model/scripted-default/engine-automatic/housekeeping,
color-coded), commitments board (open/fulfilled/missed with deadline
turns), metrics timeline (participation, follow-through,
reason-but-don't-act, illegal-action shares), and the run manifest.
Published artifacts let judges without the game verify method."

The bundle (a directory, no server, no external network):

- ``index.html``      — the self-contained viewer. The transcript JSONL,
  the optional per-turn summaries (``turns.jsonl``, recorded by
  ``lechonk.eval.paired.RecordingClient``), the commitments board
  (computed by ``lechonk.core.metrics.commitment_board`` with the SAME
  frozen matching semantics as the metrics) and the bundle metadata are
  EMBEDDED as ``application/json`` script blocks — opening the file
  from disk (``file://``) works: no fetch, no server, no CDN.
- ``transcript.jsonl`` — the raw checksummed transcript (provenance /
  tooling copy; the viewer's data is the embedded copy).
- ``turns.jsonl``      — the raw per-turn summaries, when recorded.
- ``bundle_meta.json`` — what was rendered, the source paths + sha256,
  the record counts, and the ledger's checksum-verification result.

The transcript's per-record checksums are verified by
``lechonk.core.ledger.load_transcript`` at RENDER time (the viewer
itself does not re-implement the canonical-JSON checksum); the bundle
meta records the verification result. The per-turn empire timeline
(score/cities/units/gold) renders from ``turns.jsonl`` when the run
recorded it; older transcripts (e.g. the WP2 10-turn run, which predates
the RecordingClient) get an explicit disclosure, never a fabricated
curve.

Licensing: MIT — this module NEVER imports civrealm (decision 7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lechonk.core.jsonutil import strict_dumps
from lechonk.core.ledger import load_transcript, valid_prefix_bytes
from lechonk.core.metrics import commitment_board
from lechonk.core.schemas import (
    ActionResultRecord,
    CandidatesRefRecord,
    CommitmentRecord,
    MetricEventRecord,
    ModelCallRecord,
    OutcomeRecord,
    RunManifestRecord,
    SnapshotRefRecord,
    TurnStartRecord,
)

VIEWER_VERSION = "lechonk-replay-v1"


# ---------------------------------------------------------------------------
# the viewer template (single self-contained HTML + JS, no external refs)
# ---------------------------------------------------------------------------

_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }
header.top { background: #17202a; color: #fff; padding: 14px 22px; }
header.top h1 { margin: 0 0 4px 0; font-size: 18px; }
header.top .sub { font-size: 12.5px; color: #b8c2cf; }
main { padding: 18px 22px 60px 22px; max-width: 1180px; margin: 0 auto; }
section.panel { background: #fff; border: 1px solid #dde2e8; border-radius: 8px;
  padding: 14px 18px; margin: 16px 0; box-shadow: 0 1px 2px
  rgba(16,24,40,.04); }
section.panel > h2 { margin: 0 0 10px 0; font-size: 15px; }
section.panel > h3 { margin: 16px 0 6px 0; font-size: 13px; color: #475467; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eef0f3;
  vertical-align: top; }
th { color: #475467; font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11.5px; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600; color: #fff; white-space: nowrap; }
.pv-model { background: #2563eb; }
.pv-continued { background: #0d9488; }
.pv-scripted { background: #d97706; }
.pv-engine { background: #6b7280; }
.pv-housekeeping { background: #64748b; }
.out-executed { background: #16a34a; }
.out-rejected { background: #dc2626; }
.out-aborted { background: #ea580c; }
.rb-verified { background: #16a34a; }
.rb-contradicted { background: #dc2626; }
.rb-inconclusive { background: #d97706; }
.rb-none { background: #9ca3af; }
.oc-COMPLETED { background: #16a34a; }
.oc-DEFEATED { background: #dc2626; }
.oc-VICTORY { background: #16a34a; }
.oc-TRUNCATED { background: #d97706; }
.st-fulfilled_model { background: #16a34a; }
.st-fulfilled_other { background: #0d9488; }
.st-missed { background: #dc2626; }
.st-open { background: #9ca3af; }
.disclosure { background: #fef3c7; border: 1px solid #fcd34d; color: #78350f;
  padding: 8px 12px; border-radius: 6px; font-size: 12.5px; margin: 8px 0; }
.note { color: #667085; font-size: 12px; }
select { font-size: 12.5px; padding: 3px 6px; }
svg.chart { width: 100%; height: 180px; background: #fbfcfe;
  border: 1px solid #eef0f3; border-radius: 6px; }
.legend { font-size: 11.5px; color: #475467; margin: 4px 0 8px 0; }
.legend span { display: inline-block; margin-right: 14px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 5px; vertical-align: -1px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; }
.card { border: 1px solid #e4e7ec; border-radius: 8px; padding: 10px 12px;
  min-width: 260px; max-width: 420px; flex: 1 1 300px; background: #fbfcfe; }
.card h4 { margin: 0 0 6px 0; font-size: 12.5px; }
footer { color: #98a2b3; font-size: 11.5px; text-align: center;
  padding: 18px 0 8px 0; }
a { color: #2563eb; }
</style>
</head>
<body>
<header class="top">
  <h1 id="run-title">Le Chonk replay</h1>
  <div class="sub" id="run-sub"></div>
</header>
<main>
  <section class="panel" id="panel-manifest">
    <h2>Run manifest (decision 8)</h2>
    <div id="manifest-body">loading…</div>
  </section>
  <section class="panel" id="panel-empire">
    <h2>Empire timeline (per turn)</h2>
    <div id="empire-body">loading…</div>
  </section>
  <section class="panel" id="panel-turns">
    <h2>Turn timeline</h2>
    <div id="turns-body">loading…</div>
  </section>
  <section class="panel" id="panel-snapshot">
    <h2>Snapshot summary</h2>
    <div id="snapshot-body">loading…</div>
  </section>
  <section class="panel" id="panel-actions">
    <h2>Model batch — actions with provenance</h2>
    <div id="actions-body">loading…</div>
  </section>
  <section class="panel" id="panel-commitments">
    <h2>Commitments board</h2>
    <div id="commitments-body">loading…</div>
  </section>
  <section class="panel" id="panel-metrics">
    <h2>Metrics timeline</h2>
    <div id="metrics-body">loading…</div>
  </section>
</main>
<footer id="footer"></footer>
<script id="lechonk-transcript" type="application/json">__TRANSCRIPT_JSONL__</script>
<script id="lechonk-turns" type="application/json">__TURNS_JSONL__</script>
<script id="lechonk-commitments" type="application/json">__COMMITMENTS_JSON__</script>
<script id="lechonk-meta" type="application/json">__META_JSON__</script>
<script>
(function () {
"use strict";
function $(id) { return document.getElementById(id); }
function el(tag, attrs, kids) {
  var n = document.createElement(tag);
  if (attrs) { for (var k in attrs) {
    if (k === "class") { n.className = attrs[k]; }
    else if (k === "text") { n.textContent = attrs[k]; }
    else { n.setAttribute(k, attrs[k]); } } }
  (kids || []).forEach(function (c) {
    n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
  return n;
}
function parseBlock(id) {
  var t = $(id).textContent.trim();
  if (!t || t === "null") { return null; }
  return JSON.parse(t);
}
function parseJsonl(text) {
  return text.split("\\n").filter(function (l) { return l.trim(); })
    .map(function (l) { return JSON.parse(l); });
}
function trunc(s, n) {
  s = String(s === null || s === undefined ? "" : s);
  return s.length > n ? s.slice(0, n) + "…" : s;
}
function esc(s) { return String(s).replace(/[&<>]/g, function (c) {
  return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }

var meta = parseBlock("lechonk-meta") || {};
var commitments = parseBlock("lechonk-commitments") || [];
var turns = parseBlock("lechonk-turns");
var records = parseJsonl($("lechonk-transcript").textContent);

var byKind = {};
records.forEach(function (r) { (byKind[r.kind] = byKind[r.kind] || []).push(r); });
var manifestRec = (byKind.run_manifest || [])[0];
var manifest = manifestRec ? manifestRec.manifest : {};
var outcomeRec = (byKind.outcome || [])[0] || null;
var turnStarts = (byKind.turn_start || []).map(function (r) { return r.turn; });
var turnsRange = turnStarts.length
  ? Math.min.apply(null, turnStarts) + "–" + Math.max.apply(null, turnStarts)
  : "—";

var MODEL_CLASSES = { model: true, continued_model_order: true };
var HOUSEKEEPING_VERBS = { cancel_order: true, keep_activity: true };

function pvClass(p) {
  if (p === "model") { return "pv-model"; }
  if (p === "continued_model_order") { return "pv-continued"; }
  if (p === "scripted_default") { return "pv-scripted"; }
  if (p === "engine_automatic") { return "pv-engine"; }
  if (p === "housekeeping") { return "pv-housekeeping"; }
  return "pv-housekeeping";
}
function outClass(o) {
  if (o === "executed") { return "out-executed"; }
  if (o === "rejected") { return "out-rejected"; }
  return "out-aborted";
}
function rbClass(r) {
  if (r === "verified") { return "rb-verified"; }
  if (r === "contradicted") { return "rb-contradicted"; }
  if (r === "inconclusive") { return "rb-inconclusive"; }
  return "rb-none";
}
function chip(text, cls) { return el("span", { class: "chip " + cls, text: text }); }

// ---- per-turn aggregation -------------------------------------------------
var perTurn = {};
function slot(turn) {
  if (!perTurn[turn]) {
    perTurn[turn] = { turn: turn, snapshot: null, candidates: null,
      actions: [], calls: [], commitments: [] }; }
  return perTurn[turn];
}
turnStarts.forEach(function (t) { slot(t); });
// model_call records carry no turn field (the frozen schema): their turn
// is the turn_start with the largest seq <= the call's seq (the runner
// emits the call inside its turn, after that turn's turn_start).
var turnStartSeqs = (byKind.turn_start || []).map(function (r) {
  return [r.seq, r.turn]; }).sort(function (a, b) { return a[0] - b[0]; });
function turnForSeq(seq) {
  var turn = null;
  for (var i = 0; i < turnStartSeqs.length; i++) {
    if (turnStartSeqs[i][0] <= seq) { turn = turnStartSeqs[i][1]; }
    else { break; }
  }
  return turn;
}
records.forEach(function (r) {
  if (r.kind === "snapshot_ref") { slot(r.turn).snapshot = r; }
  else if (r.kind === "candidates_ref") { slot(r.turn).candidates = r; }
  else if (r.kind === "action_result") { slot(r.turn).actions.push(r); }
  else if (r.kind === "model_call") {
    var ct = turnForSeq(r.seq);
    if (ct !== null) { slot(ct).calls.push(r); }
  }
  else if (r.kind === "commitment") { slot(r.turn).commitments.push(r); }
});
var turnList = Object.keys(perTurn).map(Number).sort(function (a, b) {
  return a - b; });
function isParticipation(a) {
  return a.outcome === "executed" && MODEL_CLASSES[a.provenance_class] &&
    !HOUSEKEEPING_VERBS[a.verb];
}
function turnActions(t, outcome) {
  return perTurn[t].actions.filter(function (a) {
    return !outcome || a.outcome === outcome; });
}

// ---- tiny SVG line chart ---------------------------------------------------
function lineChart(seriesList, opts) {
  opts = opts || {};
  var W = 1000, H = 180, P = { l: 46, r: 12, t: 10, b: 24 };
  var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H,
    preserveAspectRatio: "none" });
  var all = [];
  seriesList.forEach(function (s) {
    s.points.forEach(function (p) { all.push(p[1]); }); });
  if (!all.length) { return svg; }
  var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  if (lo === hi) { lo -= 1; hi += 1; }
  var n = Math.max.apply(null, seriesList.map(function (s) {
    return s.points.length; }));
  function X(i) { return P.l + (W - P.l - P.r) * (n <= 1 ? 0 : i / (n - 1)); }
  function Y(v) { return P.t + (H - P.t - P.b) * (1 - (v - lo) / (hi - lo)); }
  // gridlines + y labels
  [0, 0.5, 1].forEach(function (f) {
    var y = P.t + (H - P.t - P.b) * f;
    svg.appendChild(el("line", { x1: P.l, y1: y, x2: W - P.r, y2: y,
      stroke: "#eef0f3", "stroke-width": 1 }));
    var val = hi - f * (hi - lo);
    svg.appendChild(el("text", { x: P.l - 6, y: y + 4, "text-anchor": "end",
      "font-size": 10, fill: "#98a2b3",
      text: opts.fmt ? opts.fmt(val) : String(Math.round(val * 100) / 100) }));
  });
  seriesList.forEach(function (s) {
    if (!s.points.length) { return; }
    var d = s.points.map(function (p, i) {
      return (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(p[1]).toFixed(1); })
      .join(" ");
    svg.appendChild(el("path", { d: d, fill: "none", stroke: s.color,
      "stroke-width": 2 }));
    s.points.forEach(function (p, i) {
      svg.appendChild(el("circle", { cx: X(i), cy: Y(p[1]), r: 2.6,
        fill: s.color })); });
  });
  // x labels (first/last turn)
  if (n) {
    svg.appendChild(el("text", { x: P.l, y: H - 6, "font-size": 10,
      fill: "#98a2b3", text: String(seriesList[0].points[0][0]) }));
    svg.appendChild(el("text", { x: W - P.r, y: H - 6, "text-anchor": "end",
      "font-size": 10, fill: "#98a2b3",
      text: String(seriesList[0].points[n - 1][0]) }));
  }
  return svg;
}
function legend(seriesList) {
  var d = el("div", { class: "legend" });
  seriesList.forEach(function (s) {
    d.appendChild(el("span", {}, [
      el("span", { class: "sw", style: "display:inline-block;width:10px;" +
        "height:10px;border-radius:2px;margin-right:5px;background:" +
        s.color }), s.label]));
  });
  return d;
}

// ---- header ----------------------------------------------------------------
(function renderHeader() {
  var oc = outcomeRec ? outcomeRec.outcome : "?";
  $("run-title").textContent = "Le Chonk replay — " +
    (manifest.run_id || "(unknown run)");
  $("run-sub").textContent =
    "outcome " + oc + " · turns " + turnsRange + " · " +
    records.length + " transcript records · model " +
    (manifest.model_id || "n/a") + " · self-contained static bundle " +
    "(no server, no external network)";
  if (outcomeRec && outcomeRec.outcome !== "COMPLETED") {
    $("run-sub").appendChild(el("div", { class: "note",
      text: "outcome reason: " + (outcomeRec.reason || "(none)") }));
  }
})();

// ---- manifest panel --------------------------------------------------------
(function renderManifest() {
  var body = $("manifest-body");
  var rows = [
    ["run_id", manifest.run_id],
    ["outcome", outcomeRec ? outcomeRec.outcome : null],
    ["image_digest", manifest.image_digest],
    ["capstring", manifest.capstring],
    ["vendored_base_sha", manifest.vendored_base_sha],
    ["patch_head_sha", manifest.patch_head_sha],
    ["harness_revision", manifest.harness_revision],
    ["seeds", manifest.seeds ? JSON.stringify(manifest.seeds) : null],
    ["ruleset", manifest.ruleset],
    ["model_id", manifest.model_id],
    ["sampling", manifest.sampling ? JSON.stringify(manifest.sampling) : null],
    ["prompt_hashes", manifest.prompt_hashes
      ? Object.keys(manifest.prompt_hashes).length + " templates" : null],
    ["model_fingerprint", manifest.model_fingerprint],
    ["arm_start_ts", manifest.arm_start_ts],
    ["checkpoint_checksums", manifest.checkpoint_checksums
      ? JSON.stringify(manifest.checkpoint_checksums) : null],
    ["config_after_load", manifest.config_after_load
      ? JSON.stringify(manifest.config_after_load) : null],
    ["server_settings", manifest.server_settings
      ? JSON.stringify(manifest.server_settings) : null],
  ];
  // forward-compat extras (paired runs record these)
  ["paired_pair_id", "paired_arm", "paired_contract_sha256",
   "fingerprint_evidence", "enforcement_decision",
   "action_batch_schema_hash"].forEach(function (k) {
    if (manifest[k] !== undefined && manifest[k] !== null) {
      rows.push([k, typeof manifest[k] === "object"
        ? JSON.stringify(manifest[k]) : manifest[k]]); }
  });
  var t = el("table");
  t.appendChild(el("tr", {}, [el("th", { text: "field" }),
    el("th", { text: "value" })]));
  rows.forEach(function (r) {
    if (r[1] === null || r[1] === undefined) { return; }
    var v = String(r[1]);
    t.appendChild(el("tr", {}, [
      el("td", { class: "mono", text: r[0] }),
      el("td", { class: "mono", title: v, text: trunc(v, 160) })]));
  });
  body.appendChild(t);
  body.appendChild(el("div", { class: "note", text:
    "per-record checksums verified at render time by " +
    "lechonk.core.ledger.load_transcript: " +
    (meta.checksums_verified ? "all " + meta.record_count +
     " records valid" : "see bundle_meta.json") +
    (meta.discarded_trailing_partial ? " (a trailing partial record was " +
     "discarded on load)" : "") + "." }));
})();

// ---- empire timeline -------------------------------------------------------
(function renderEmpire() {
  var body = $("empire-body");
  if (!turns || !turns.length) {
    body.appendChild(el("div", { class: "disclosure", text:
      "The empire timeline (score curve, cities, units, gold) is not " +
      "recorded in this bundle: the transcript predates per-turn summary " +
      "persistence (lechonk.eval.paired.RecordingClient records " +
      "turns.jsonl for new runs, e.g. the WP3 paired-smoke arms). The " +
      "turn timeline below is computed from the transcript itself." }));
    return;
  }
  var series = [
    { key: "score", label: "score", color: "#2563eb" },
    { key: "cities_own", label: "cities", color: "#16a34a" },
    { key: "units_own", label: "units", color: "#d97706" },
    { key: "gold", label: "gold", color: "#7c3aed" },
  ];
  series.forEach(function (s) {
    body.appendChild(el("h3", { text: s.label }));
    var pts = turns.map(function (t) { return [t.turn, t[s.key] || 0]; });
    body.appendChild(legend([{ label: s.label, color: s.color }]));
    body.appendChild(lineChart([{ label: s.label, color: s.color,
      points: pts }]));
  });
  var t = el("table");
  t.appendChild(el("tr", {}, [el("th", { text: "turn" }),
    el("th", { class: "num", text: "score" }),
    el("th", { class: "num", text: "cities" }),
    el("th", { class: "num", text: "units" }),
    el("th", { class: "num", text: "gold" }),
    el("th", { text: "research" }),
    el("th", { class: "num", text: "techs" })]));
  turns.forEach(function (s) {
    t.appendChild(el("tr", {}, [
      el("td", { class: "num", text: String(s.turn) }),
      el("td", { class: "num", text: String(s.score) }),
      el("td", { class: "num", text: String(s.cities_own) }),
      el("td", { class: "num", text: String(s.units_own) }),
      el("td", { class: "num", text: String(s.gold) }),
      el("td", { class: "mono", text: trunc(s.research_target || "—", 24) }),
      el("td", { class: "num", text: String(s.techs_researched) })]));
  });
  body.appendChild(el("h3", { text: "per-turn table" }));
  body.appendChild(t);
})();

// ---- turn timeline ---------------------------------------------------------
(function renderTurns() {
  var body = $("turns-body");
  var s1 = { label: "snapshot tokens", color: "#2563eb", points: [] };
  var s2 = { label: "candidates", color: "#0d9488", points: [] };
  var s3 = { label: "actions executed", color: "#16a34a", points: [] };
  var s4 = { label: "actions rejected", color: "#dc2626", points: [] };
  var s5 = { label: "api tokens in", color: "#7c3aed", points: [] };
  turnList.forEach(function (t) {
    var p = perTurn[t];
    s1.points.push([t, p.snapshot ? p.snapshot.token_count : 0]);
    s2.points.push([t, p.candidates ? p.candidates.candidate_count : 0]);
    s3.points.push([t, turnActions(t, "executed").length]);
    s4.points.push([t, turnActions(t, "rejected").length]);
    s5.points.push([t, p.calls.reduce(function (a, c) {
      return a + (c.tokens_in || 0); }, 0)]);
  });
  body.appendChild(legend([s1, s2, s3, s4, s5]));
  body.appendChild(lineChart([s1, s2, s3, s4, s5]));
  var t = el("table");
  t.appendChild(el("tr", {}, [
    el("th", { text: "turn" }), el("th", { class: "num", text: "snap tok" }),
    el("th", { class: "num", text: "cands" }),
    el("th", { class: "num", text: "exec" }),
    el("th", { class: "num", text: "rej" }),
    el("th", { class: "num", text: "abort" }),
    el("th", { class: "num", text: "calls" }),
    el("th", { class: "num", text: "tok in" }),
    el("th", { class: "num", text: "tok out" }),
    el("th", { class: "num", text: "cost $" }),
    el("th", { text: "opportunities" })]));
  turnList.forEach(function (turn) {
    var p = perTurn[turn];
    var calls = p.calls;
    t.appendChild(el("tr", {}, [
      el("td", { class: "num", text: String(turn) }),
      el("td", { class: "num", text: p.snapshot
        ? String(p.snapshot.token_count) : "—" }),
      el("td", { class: "num", text: p.candidates
        ? String(p.candidates.candidate_count) : "—" }),
      el("td", { class: "num", text: String(turnActions(turn, "executed").length) }),
      el("td", { class: "num", text: String(turnActions(turn, "rejected").length) }),
      el("td", { class: "num", text: String(turnActions(turn, "aborted_group").length) }),
      el("td", { class: "num", text: String(calls.length) }),
      el("td", { class: "num", text: String(calls.reduce(function (a, c) {
        return a + (c.tokens_in || 0); }, 0)) }),
      el("td", { class: "num", text: String(calls.reduce(function (a, c) {
        return a + (c.tokens_out || 0); }, 0)) }),
      el("td", { class: "num", text: calls.reduce(function (a, c) {
        return a + (c.cost_usd || 0); }, 0).toFixed(6) }),
      el("td", { class: "mono", text: p.candidates &&
        p.candidates.opportunity_domains
        ? p.candidates.opportunity_domains.join(",") : "—" })]));
  });
  body.appendChild(t);
})();

// ---- snapshot summary panel ------------------------------------------------
(function renderSnapshot() {
  var body = $("snapshot-body");
  if (!turnList.length) { body.textContent = "no turns"; return; }
  var sel = el("select", { id: "snap-turn" });
  turnList.forEach(function (t) {
    sel.appendChild(el("option", { value: String(t), text: "turn " + t })); });
  body.appendChild(el("div", {}, [document.createTextNode("turn: "), sel]));
  var detail = el("div", { id: "snap-detail" });
  body.appendChild(detail);
  function renderDetail() {
    var turn = Number(sel.value);
    var p = perTurn[turn];
    detail.textContent = "";
    if (p.snapshot) {
      detail.appendChild(el("h3", { text: "snapshot_ref" }));
      detail.appendChild(el("div", { class: "mono", text:
        "turn " + p.snapshot.turn + " · token_count " +
        p.snapshot.token_count + " · snapshot_hash " +
        trunc(p.snapshot.snapshot_hash, 24) + "…" }));
    }
    if (p.candidates) {
      detail.appendChild(el("h3", { text: "candidates_ref (the offered table)" }));
      detail.appendChild(el("div", { class: "mono", text:
        "turn " + p.candidates.turn + " · candidate_count " +
        p.candidates.candidate_count + " · table_hash " +
        trunc(p.candidates.table_hash, 24) + "…" }));
      var byDom = p.candidates.by_domain || {};
      var t = el("table");
      t.appendChild(el("tr", {}, [el("th", { text: "domain" }),
        el("th", { class: "num", text: "offered" })]));
      Object.keys(byDom).sort().forEach(function (d) {
        t.appendChild(el("tr", {}, [el("td", { class: "mono", text: d }),
          el("td", { class: "num", text: String(byDom[d]) })]));
      });
      detail.appendChild(t);
      detail.appendChild(el("div", { class: "note", text:
        "opportunity domains (≥1 consequential choice): " +
        (p.candidates.opportunity_domains || []).join(", ") }));
    }
    if (p.calls.length) {
      detail.appendChild(el("h3", { text: "model calls" }));
      p.calls.forEach(function (c) {
        detail.appendChild(el("div", { class: "mono", text:
          c.call_id + " · attempt " + c.attempt + " · " + c.latency_ms +
          " ms · in " + c.tokens_in + " / out " + c.tokens_out +
          " · $" + (c.cost_usd || 0).toFixed(6) + " · fingerprint " +
          trunc(c.fingerprint, 20) + (c.error ? " · ERROR: " + c.error
          : "") }));
      });
    }
    if (!p.snapshot && !p.candidates && !p.calls.length) {
      detail.appendChild(el("div", { class: "note",
        text: "no records for this turn" }));
    }
  }
  sel.addEventListener("change", renderDetail);
  renderDetail();
})();

// ---- actions panel ---------------------------------------------------------
(function renderActions() {
  var body = $("actions-body");
  if (!turnList.length) { body.textContent = "no turns"; return; }
  var sel = el("select", { id: "act-turn" });
  turnList.forEach(function (t) {
    sel.appendChild(el("option", { value: String(t), text: "turn " + t })); });
  body.appendChild(el("div", {}, [document.createTextNode("turn: "), sel]));
  var detail = el("div", { id: "act-detail" });
  body.appendChild(detail);
  function renderDetail() {
    var turn = Number(sel.value);
    var p = perTurn[turn];
    detail.textContent = "";
    var acts = p.actions;
    if (!acts.length) {
      detail.appendChild(el("div", { class: "note",
        text: "no action_result records this turn" }));
      return;
    }
    acts.forEach(function (a) {
      var card = el("div", { class: "card" });
      var head = el("div", {});
      head.appendChild(chip(a.provenance_class, pvClass(a.provenance_class)));
      head.appendChild(document.createTextNode(" "));
      head.appendChild(chip(a.outcome, outClass(a.outcome)));
      head.appendChild(document.createTextNode(" "));
      head.appendChild(chip("read-back " + a.readback,
        rbClass(a.readback)));
      if (a.provenance_class === "continued_model_order" &&
          a.origin_seq !== null && a.origin_seq !== undefined) {
        head.appendChild(document.createTextNode(" "));
        head.appendChild(chip("↳ origin seq #" + a.origin_seq,
          "pv-continued"));
      }
      card.appendChild(head);
      var t = el("table");
      [["ref", a.ref], ["domain", a.domain], ["verb", a.verb],
       ["target", a.target], ["affected", a.affected],
       ["actor_tile", a.actor_tile ? JSON.stringify(a.actor_tile) : null],
       ["rejection_class", a.rejection_class],
       ["reason", a.reason || null],
       ["detail", a.detail]].forEach(function (r) {
        if (r[1] === null || r[1] === undefined || r[1] === "") { return; }
        t.appendChild(el("tr", {}, [
          el("td", { class: "mono", text: r[0] }),
          el("td", { class: "mono", title: String(r[1]),
            text: trunc(String(r[1]), 220) })]));
      });
      card.appendChild(t);
      detail.appendChild(card);
    });
  }
  sel.addEventListener("change", renderDetail);
  renderDetail();
})();

// ---- commitments board -----------------------------------------------------
(function renderCommitments() {
  var body = $("commitments-body");
  if (!commitments.length) {
    body.appendChild(el("div", { class: "note",
      text: "no commitments recorded in this transcript" }));
  }
  var counts = {};
  commitments.forEach(function (c) {
    counts[c.status] = (counts[c.status] || 0) + 1; });
  var summary = el("div", { class: "legend" });
  ["fulfilled_model", "fulfilled_other", "missed", "open"].forEach(
    function (s) {
      if (!counts[s]) { return; }
      summary.appendChild(el("span", {}, [
        chip(s.replace("_", " ") + " ×" + counts[s],
          "st-" + s)]));
      summary.appendChild(document.createTextNode(" "));
    });
  body.appendChild(summary);
  var cards = el("div", { class: "cards" });
  commitments.forEach(function (c) {
    var card = el("div", { class: "card" });
    card.appendChild(el("h4", {}, [
      chip(c.status.replace("_", " "), "st-" + c.status),
      document.createTextNode(" " + c.domain)]));
    var t = el("table");
    [["target", c.target], ["deadline turn", c.deadline_turn],
     ["emitted turn", c.emitted_turn],
     ["fulfilled by (any)", (c.fulfilled_by || []).join(", ") || null],
     ["fulfilled by (model)", (c.fulfilled_model_by || []).join(", ")
       || null]].forEach(function (r) {
      if (r[1] === null || r[1] === undefined || r[1] === "") { return; }
      t.appendChild(el("tr", {}, [
        el("td", { class: "mono", text: r[0] }),
        el("td", { class: "mono", title: String(r[1]),
          text: trunc(String(r[1]), 200) })]));
    });
    card.appendChild(t);
    cards.appendChild(card);
  });
  body.appendChild(cards);
  body.appendChild(el("div", { class: "note", text:
    "statuses computed by lechonk.core.metrics.commitment_board with the " +
    "SAME frozen matching semantics as the run's metrics (no parallel " +
    "implementation); 'open' = deadline outside the run (censored, not " +
    "failed)." }));
})();

// ---- metrics timeline ------------------------------------------------------
(function renderMetrics() {
  var body = $("metrics-body");
  // cumulative participation curve (viewer-computed; the aggregate is the
  // frozen metric_event below)
  var opp = [], part = [];
  var cumOpp = 0, cumPart = 0;
  turnList.forEach(function (t) {
    var p = perTurn[t];
    if (p.candidates && (p.candidates.opportunity_domains || []).length) {
      cumOpp += 1; }
    if (p.actions.some(isParticipation)) { cumPart += 1; }
    opp.push([t, cumOpp]);
    part.push([t, cumPart]);
  });
  body.appendChild(el("h3", { text:
    "cumulative model participation (viewer-computed from the ledger)" }));
  body.appendChild(legend([
    { label: "turns with opportunities", color: "#6b7280" },
    { label: "turns with model participation", color: "#2563eb" }]));
  body.appendChild(lineChart([
    { label: "opportunities", color: "#6b7280", points: opp },
    { label: "participation", color: "#2563eb", points: part }]));
  // the frozen aggregate metric events
  var events = (byKind.metric_event || []).filter(function (e) {
    return e.name !== "state_transition"; });
  if (events.length) {
    body.appendChild(el("h3", { text: "aggregate metrics (frozen, ledger-" +
      "computed)" }));
    var t = el("table");
    t.appendChild(el("tr", {}, [el("th", { text: "metric" }),
      el("th", { text: "value" })]));
    events.forEach(function (e) {
      t.appendChild(el("tr", {}, [
        el("td", { class: "mono", text: e.name }),
        el("td", { class: "mono", text: String(e.value) })]));
    });
    body.appendChild(t);
  }
  // provenance mix per turn
  body.appendChild(el("h3", { text: "provenance mix (executed actions)" }));
  var classes = ["model", "continued_model_order", "scripted_default",
    "engine_automatic", "housekeeping"];
  var colors = { model: "#2563eb", continued_model_order: "#0d9488",
    scripted_default: "#d97706", engine_automatic: "#6b7280",
    housekeeping: "#64748b" };
  var seriesList = classes.map(function (c) {
    return { label: c, color: colors[c],
      points: turnList.map(function (t) {
        return [t, turnActions(t, "executed").filter(function (a) {
          return a.provenance_class === c; }).length]; }) }; });
  body.appendChild(legend(seriesList));
  body.appendChild(lineChart(seriesList));
})();

// ---- footer ----------------------------------------------------------------
$("footer").textContent =
  "lechonk replay viewer " + (meta.viewer_version || "") +
  " · generated " + (meta.generated_at || "") +
  " · source transcript sha256 " + trunc(meta.transcript_sha256 || "", 16) +
  "… · self-contained: no server, no external network, no scripts beyond " +
  "this file";
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# the bundle renderer
# ---------------------------------------------------------------------------

def _embed(text: str) -> str:
    """Embed JSON text in a ``<script type="application/json">`` block.
    Every ``<`` becomes ``\\u003c``: ``<`` is not a JSON structural
    character, so it can only appear inside string values, where
    ``\\u003c`` is a valid escape that decodes back to ``<`` — and no HTML
    breakout sequence (``</script>``, ``<!--``, ``<script``) can survive
    in the script data."""
    return text.replace("<", "\\u003c")


def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_bundle(transcript_path: str | Path, out_dir: str | Path, *,
                  turns_path: Optional[str | Path] = None,
                  title: Optional[str] = None) -> dict[str, Any]:
    """Render one transcript into a static replay bundle under
    ``out_dir`` (created). Returns the bundle summary (also persisted as
    ``bundle_meta.json``)."""
    transcript = Path(transcript_path)
    if not transcript.is_file():
        raise FileNotFoundError(f"transcript not found: {transcript}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The ledger verifies every per-record checksum on load.
    load = load_transcript(transcript)
    records = load.records
    board = commitment_board(records)
    # The embedded transcript block is parsed by the viewer with
    # parseJsonl (one JSON.parse per non-empty line), so only the
    # VERIFIED byte prefix is embedded — never the raw discarded
    # trailing partial (a torn JSON fragment would break JSON.parse;
    # the discard is still recorded in meta). For a clean transcript
    # the prefix is the whole file (byte-identical embed).
    raw_text = (transcript.read_bytes()
                [:valid_prefix_bytes(str(transcript))]
                .decode("utf-8"))

    # The raw copies (provenance / tooling).
    shutil.copyfile(transcript, out / "transcript.jsonl")
    turns_raw = None
    turns_summaries = None
    if turns_path is not None:
        turns_src = Path(turns_path)
        if turns_src.is_file():
            turns_raw = turns_src.read_text(encoding="utf-8")
            shutil.copyfile(turns_src, out / "turns.jsonl")
            # The viewer's lechonk-turns block is parsed with ONE
            # JSON.parse (it consumes an array: turns.map/forEach), so
            # the JSONL is re-embedded as a single JSON array — never as
            # raw newline-separated objects (live-verified 2026-10-24:
            # the raw JSONL broke JSON.parse on the second record).
            turns_summaries = [json.loads(line)
                               for line in turns_raw.splitlines()
                               if line.strip()]

    kinds: dict[str, int] = {}
    for rec in records:
        kinds[rec.kind] = kinds.get(rec.kind, 0) + 1
    manifest_rec = next((r for r in records
                         if isinstance(r, RunManifestRecord)), None)
    outcome_rec = next((r for r in records
                        if isinstance(r, OutcomeRecord)), None)
    manifest = manifest_rec.manifest if manifest_rec else None

    import time as _time
    meta = {
        "viewer_version": VIEWER_VERSION,
        "generated_by": "lechonk.eval.replay.render_bundle (WP3 package A; "
                        "design-plan-freeciv-v1.1.md §7.6)",
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "self_contained": True,
        "no_server": True,
        "no_external_network": True,
        "transcript_source": str(transcript),
        "transcript_sha256": _sha256_file(transcript),
        "record_count": len(records),
        "record_kinds": kinds,
        "checksums_verified": True,  # load_transcript raised otherwise
        "discarded_trailing_partial": load.discarded_trailing_partial,
        "turns_source": str(turns_path) if turns_path else None,
        "turns_recorded": turns_raw is not None,
        "turns_count": (len(turns_summaries)
                        if turns_summaries is not None else 0),
        "commitments_board_size": len(board),
        "run_id": manifest.run_id if manifest else None,
        "outcome": outcome_rec.outcome if outcome_rec else None,
    }
    html_title = title or (
        f"Le Chonk replay — {manifest.run_id if manifest else transcript.name}")
    html = (_VIEWER_TEMPLATE
            .replace("__TITLE__", _html_escape(html_title))
            .replace("__TRANSCRIPT_JSONL__", _embed(raw_text))
            .replace("__TURNS_JSONL__",
                     _embed(strict_dumps(turns_summaries))
                     if turns_summaries is not None else "null")
            .replace("__COMMITMENTS_JSON__", _embed(strict_dumps(board)))
            .replace("__META_JSON__", _embed(strict_dumps(meta))))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "bundle_meta.json").write_text(
        strict_dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    return {"bundle_dir": str(out), "meta": meta,
            "index_html": str(out / "index.html")}


# ---------------------------------------------------------------------------
# CLI: python -m lechonk.eval.replay <transcript> [--turns ...] [--out ...]
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a lechonk JSONL transcript into a static "
                    "replay bundle (self-contained HTML, no server).")
    parser.add_argument("transcript", help="path to transcript.jsonl")
    parser.add_argument("--turns", default=None,
                        help="optional path to turns.jsonl (the per-turn "
                             "summaries recorded by RecordingClient)")
    parser.add_argument("--out", default=None,
                        help="bundle output directory (default: "
                             "<transcript dir>/replay)")
    parser.add_argument("--title", default=None, help="viewer title")
    args = parser.parse_args(argv)
    transcript = Path(args.transcript).resolve()
    out = Path(args.out).resolve() if args.out else \
        transcript.parent / "replay"
    summary = render_bundle(transcript, out, turns_path=args.turns,
                            title=args.title)
    meta = summary["meta"]
    print(f"[replay] rendered {meta['record_count']} records "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(meta['record_kinds'].items()))})")
    print(f"[replay] bundle: {summary['bundle_dir']}")
    print(f"[replay] index:  {summary['index_html']}")
    print(f"[replay] turns recorded: {meta['turns_recorded']} · "
          f"commitments board: {meta['commitments_board_size']} · "
          f"outcome: {meta['outcome']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
