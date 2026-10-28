#!/usr/bin/env node
// Screenshot-first, zero-perturbation presenter for a running Freeciv game.
// It consumes the gateway's manifest and append-only event stream. It never
// sends game commands and does not record continuous video.
// SCRIPT-METADATA:
// name: capture_presenter
// description: Attach a Freeciv observer, render bounded model trace UI, and capture event-triggered PNGs
// tags: freeciv,capture,playwright,screenshot
// language: javascript

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { performance } = require('perf_hooks');

const DEFAULTS = Object.freeze({
  manifestWaitMs: 120000,
  eventPollMs: 250,
  settleMs: 350,
  maxRenderedEvents: 100,
  maxRenderedChars: 18000,
  queueLimit: 3,
  screenshotTimeoutMs: 15000,
  viewportWidth: 1920,
  viewportHeight: 1080,
});

function usage() {
  console.log('Usage: node scripts/capture_presenter.js --capture-dir PATH [--headless] [--renderer 2d|webgl] [--tileset amplio2|isotrident] [--allow-experimental-tileset]');
  console.log('       node scripts/capture_presenter.js --self-test');
}

function parseArgs(argv) {
  const out = { headless: false, reattach: false, selfTest: false, allowExperimentalTileset: false, captureDir: null, renderer: '2d', tileset: 'amplio2', ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--headless') out.headless = true;
    else if (a === '--allow-experimental-tileset') out.allowExperimentalTileset = true;
    else if (a === '--reattach') out.reattach = true;
    else if (a === '--self-test') out.selfTest = true;
    else if (a === '--help' || a === '-h') out.help = true;
    else if (a === '--capture-dir') out.captureDir = argv[++i];
    else if (a === '--poll-ms') out.eventPollMs = Number(argv[++i]);
    else if (a === '--manifest-wait-ms') out.manifestWaitMs = Number(argv[++i]);
    else if (a === '--queue-limit') out.queueLimit = Number(argv[++i]);
    else if (a === '--screenshot-timeout-ms') out.screenshotTimeoutMs = Number(argv[++i]);
    else if (a === '--renderer') out.renderer = String(argv[++i] || '').toLowerCase();
    else if (a === '--tileset') out.tileset = String(argv[++i] || '').toLowerCase();
    else throw new Error(`unknown argument: ${a}`);
  }
  if (!Number.isFinite(out.eventPollMs) || out.eventPollMs < 20) throw new Error('--poll-ms must be >= 20');
  if (!Number.isFinite(out.queueLimit) || out.queueLimit < 1) throw new Error('--queue-limit must be >= 1');
  if (!['2d', 'webgl'].includes(out.renderer)) throw new Error('--renderer must be 2d or webgl');
  if (!['amplio2', 'isotrident'].includes(out.tileset)) throw new Error('--tileset must be amplio2 or isotrident');
  if (out.tileset === 'isotrident' && !out.allowExperimentalTileset) throw new Error('--tileset isotrident is experimental; pass --allow-experimental-tileset after accepting unverified visual quality');
  if (out.tileset !== 'amplio2' && out.renderer !== '2d') throw new Error('isotrident is supported only with --renderer 2d');
  return out;
}

function nowIso() { return new Date().toISOString(); }
function safeString(v, max = 4000) {
  if (v == null) return '';
  let s;
  try { s = typeof v === 'string' ? v : JSON.stringify(v); } catch (_) { s = String(v); }
  return String(s).slice(0, max);
}
function nonempty(v) { return typeof v === 'string' && v.trim().length > 0; }
function stableId(value) { return crypto.createHash('sha256').update(String(value)).digest('hex').slice(0, 16); }
function observerUsername(captureId) { return `capobs${stableId(captureId)}`; }
function validFreecivUsername(name) { return /^[A-Za-z][A-Za-z0-9]*$/.test(name) && name.length <= 25; }
function eventKind(e) {
  const outer = String(e && e.kind || 'unknown'); const p = eventPayload(e);
  // The gateway wraps runner events as kind=progress with the semantic event
  // in payload.event. Preserve the semantic kind for rendering/triggers.
  if (/^(progress|event|runner_event)$/i.test(outer)) {
    if (typeof p.event === 'string' && p.event) return p.event;
    if (p.event && typeof p.event === 'object' && p.event.kind) return String(p.event.kind);
    if (typeof p.event_kind === 'string' && p.event_kind) return p.event_kind;
  }
  return outer;
}
function eventPayload(e) { return (e && e.payload && typeof e.payload === 'object') ? e.payload : {}; }

function validateManifest(m) {
  if (!m || typeof m !== 'object') throw new Error('manifest is not a JSON object');
  for (const k of ['run_id', 'capture_id', 'expected_username']) {
    if (!nonempty(m[k])) throw new Error(`manifest missing non-empty ${k}`);
  }
  const port = Number(m.session_port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('manifest session_port is invalid');
  if (!nonempty(String(m.schema_version || ''))) throw new Error('manifest missing schema_version');
  if (m.session_ready !== true) throw new Error('manifest session_ready is not true');
  return { ...m, session_port: port };
}
function manifestEnded(m) { return Boolean(m && (m.finished === true || /^(finished|complete|completed|ended|failed)$/i.test(String(m.status || m.state || '')))); }

function displayForEvent(e) {
  const p = eventPayload(e);
  const pieces = [];
  const raw = p.raw_delta && typeof p.raw_delta === 'object' ? p.raw_delta : {};
  for (const key of ['reasoning', 'answer', 'text', 'delta']) {
    if (raw[key] != null && safeString(raw[key])) pieces.push(`${key}: ${safeString(raw[key], 2200)}`);
  }
  for (const key of ['reasoning', 'thought', 'analysis', 'answer', 'text', 'delta', 'message', 'error']) {
    if (p[key] != null && safeString(p[key])) pieces.push(`${key}: ${safeString(p[key], 2200)}`);
  }
  if (p.actions != null) pieces.push(`actions: ${safeString(p.actions, 2600)}`);
  if (!pieces.length && e && e.message) pieces.push(safeString(e.message, 2200));
  return pieces.join('\n');
}

function sourceSeq(e) {
  const n = Number(e && e.capture_seq);
  return Number.isInteger(n) ? n : null;
}
function validateEventIdentity(e, manifest) {
  if (!e || e.run_id !== manifest.run_id || e.capture_id !== manifest.capture_id) return `identity mismatch (run_id=${e && e.run_id}, capture_id=${e && e.capture_id})`;
  if (!nonempty(String(e.schema_version || ''))) return 'missing schema_version';
  return null;
}
function hasExactRosterMatch(probe, expected) { return Boolean(probe && probe.expectedUsernameMatches === 1 && probe.expectedInRosterUsername && expected); }
function socketEvidenceReady(e) { return Boolean(e && e.matching_sockets > 0 && e.open_matching > 0 && e.received_frames > 0); }
function screenshotOptions(tempPath) { return { path: tempPath, type: 'png', fullPage: false }; }
function presenterBaselineEvent() { return { schema_version: 'capture-presenter/v1', kind: 'presenter_attached', payload: {}, capture_seq: null, event_id: `presenter-attached-${Date.now()}` }; }
function mergeStreamChunk(old, _previousChunk, chunk) { return String(old) + String(chunk); }
function boundedAppend(old, chunk, max = 12000) { return `${old || ''}${chunk || ''}`.slice(-max); }
function streamPart(e, kind) {
  const p = eventPayload(e); const raw = p.raw_delta && typeof p.raw_delta === 'object' ? p.raw_delta : {};
  for (const [key, lane] of [['reasoning', 'reasoning'], ['answer', 'answer'], ['tool', 'tools'], ['tools', 'tools']]) {
    if (raw[key] != null) return { lane, text: safeString(raw[key], 4000) };
    if (p[key] != null) return { lane, text: safeString(p[key], 4000) };
  }
  if (raw.delta != null || p.delta != null) return { lane: /reason|think|analysis/i.test(kind) ? 'reasoning' : 'answer', text: safeString(raw.delta ?? p.delta, 4000) };
  return null;
}
function streamIdentity(e) {
  const p = eventPayload(e); const raw = p.raw_delta && typeof p.raw_delta === 'object' ? p.raw_delta : {};
  return { physical: p.physical_call_id ?? raw.physical_call_id ?? p.call_id ?? p.request_id ?? p.generation_id ?? p.model_call_id ?? null, logical: p.logical_call_id ?? raw.logical_call_id ?? null, response: p.response_id ?? raw.response_id ?? null };
}
function sameStreamIdentity(a, b) {
  if (!a || !b) return false;
  const fields = ['physical', 'logical', 'response']; let shared = false;
  for (const field of fields) {
    const av = a[field] == null ? null : String(a[field]); const bv = b[field] == null ? null : String(b[field]);
    if (av && bv) { if (av !== bv) return false; shared = true; }
  }
  return shared;
}
function actionDisplay(p, kind) {
  const ref = p.ref || p.target_ref || null; const verb = p.verb || p.action_name || p.command || p.tool || p.tool_name || null;
  const name = ref && verb ? `${safeString(ref, 300)}:${safeString(verb, 300)}` : (verb || (ref ? safeString(ref, 500) : null));
  const outcome = p.outcome || p.status || p.result || null; const readback = p.readback || null;
  if (!name && !outcome && !readback) return null;
  return `${name || (/engine|automatic/i.test(String(kind)) ? 'engine automatic' : kind)}${outcome ? ` [${safeString(outcome, 200)}]` : ''}${readback ? ` {${safeString(readback, 300)}}` : ''}`;
}

// Runner action records use compact descriptors such as "tile:12,8". Keep
// parsing deliberately conservative: a bad descriptor should leave the
// observer on its current overview rather than guessing at a location.
function focusTileFromEvent(e) {
  const p = eventPayload(e);
  const candidates = [p.affected, p.target, p.actor_tile, p.tile, p.location];
  for (const value of candidates) {
    if (Array.isArray(value) && value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]))) {
      return { x: Number(value[0]), y: Number(value[1]), source: Array.isArray(p.affected) && value === p.affected ? 'affected' : 'structured' };
    }
    const m = String(value ?? '').match(/(?:^|:)\s*(-?\d+)\s*,\s*(-?\d+)\s*$/);
    if (m) return { x: Number(m[1]), y: Number(m[2]), source: value === p.affected ? 'affected' : value === p.target ? 'target' : 'descriptor' };
  }
  return null;
}

function shouldScreenshot(e) {
  const k = eventKind(e).toLowerCase();
  return /^(turn[_-]?start|turn[_-]?started|turn[_-]?decision|action[_-]?result|game[_-]?outcome|run[_-]?(finished|outcome)|session[_-]?finished|capture[_-]?finished)$/.test(k)
    || /turn[_-]?start|turn[_-]?decision|action[_-]?result|game[_-]?outcome|capture[_-]?finished/.test(k);
}

function screenshotPriority(e) {
  const k = eventKind(e).toLowerCase();
  if (isTerminal(e)) return 100;
  if (/turn[_-]?start|turn[_-]?decision/.test(k)) return 40;
  const p = eventPayload(e);
  const text = [p.domain, p.verb, p.action, p.action_name, p.command, p.ref, p.target, p.affected]
    .map(v => safeString(v, 500)).join(' ').toLowerCase();
  if (/attack|combat|fight|conquer|found[_ ]?city|build[_ ]?city|research|tech|diplom|treaty|wonder|advance|revolt|capture/.test(text)) return 80;
  if (/goto|move|explore|wait|fortify|sentry|produce_auto|automatic|production|unit:.*:(go|move)/.test(text)) return 10;
  return k.includes('action') ? 25 : 20;
}

function isRoutineScreenshot(e) { return screenshotPriority(e) <= 10; }

function isTerminal(e) {
  return /^(game[_-]?outcome|run[_-]?(finished|outcome)|session[_-]?finished|capture[_-]?finished)$/.test(eventKind(e).toLowerCase());
}

function writeJsonLine(stream, obj) {
  stream.write(`${JSON.stringify(obj)}\n`);
}
function startupDiagnostic(dir, message, details = {}) {
  if (!dir) return;
  try { fs.mkdirSync(dir, { recursive: true }); fs.appendFileSync(path.join(dir, 'presenter-startup.log'), `${JSON.stringify({ at: nowIso(), message, ...details })}\n`, { mode: 0o600 }); } catch (_) {}
}

async function waitForManifest(dir, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const file = path.join(dir, 'manifest.json');
  while (Date.now() < deadline) {
    try { return validateManifest(JSON.parse(await fsp.readFile(file, 'utf8'))); }
    catch (e) {
      if (e.code !== 'ENOENT' && !/Unexpected end|JSON/.test(String(e.message))) {
        // A malformed manifest may be in the middle of an atomic write; keep
        // polling briefly, then return the useful validation error.
      }
    }
    await new Promise(r => setTimeout(r, 150));
  }
  throw new Error(`manifest.json not found or valid within ${timeoutMs}ms (${file})`);
}

async function probeObserver(page, expectedUsername) {
  return page.evaluate((expected) => {
    const val = (name) => { try { return window[name]; } catch (_) { return undefined; } };
    const c = val('client');
    const conn = c && c.conn;
    const sockets = conn ? [conn.websocket, conn.webSocket, conn.ws, conn.socket, conn.sock].filter(Boolean) : [];
    const wsReady = sockets.some(s => s.readyState === 1);
    const roster = val('players');
    const entries = roster && typeof roster === 'object'
      ? Object.values(roster).map(p => ({ username: p && p.username, name: p && p.name })).filter(p => p.username || p.name) : [];
    const names = entries.map(p => p.name).filter(Boolean);
    const usernames = entries.map(p => p.username).filter(Boolean);
    const body = document.body ? document.body.innerText : '';
    const playing = conn ? (conn.playing == null ? null : conn.playing) : undefined;
    const observing = val('observing');
    const observer = observing === true || (conn && (conn.observer === true || conn.playing == null));
    return {
      observer: observer === true && Boolean(conn),
      connected: Boolean(conn),
      websocketReady: wsReady,
      connEstablished: Boolean(conn && conn.established === true),
      globalWebSocketReady: Boolean(val('ws') && val('ws').readyState === 1),
      observerEvidence: { observing: observing === true, connObserver: conn ? conn.observer : undefined },
      playing,
      playingEvidence: conn ? true : false,
      roster: entries,
      usernames,
      expectedInRoster: usernames.includes(expected),
      expectedInRosterUsername: usernames.includes(expected),
      expectedUsernameMatches: usernames.filter(username => username === expected).length,
      expectedInDisplayName: names.includes(expected),
      expectedInPageText: false,
      canvasCount: document.querySelectorAll('canvas').length,
      gameInfo: val('game_info') ? { turn: val('game_info').turn, year: val('game_info').year } : null,
      tileset: { name: val('tileset_name') || null, width: Number(val('normal_tile_width')) || null, height: Number(val('normal_tile_height')) || null, imageCount: Number(val('tileset_image_count')) || null, resources: performance.getEntriesByType('resource').map(x => x.name).filter(x => /tileset.*(?:amplio2|isotrident)/i.test(x)).slice(-8) },
    };
  }, expectedUsername);
}

async function injectSidebar(page, manifest, limits) {
  await page.evaluate(({ manifest: m, maxEvents, maxChars }) => {
    if (document.getElementById('lechonk-capture-sidebar')) return;
    const style = document.createElement('style');
    style.id = 'lechonk-capture-style';
    style.textContent = `#lechonk-capture-sidebar{position:fixed;z-index:2147483647;right:0;top:0;width:390px;height:100vh;box-sizing:border-box;background:rgba(10,14,22,.94);color:#e8edf6;font:13px/1.35 system-ui,sans-serif;padding:12px;box-shadow:-3px 0 12px #000;display:flex;flex-direction:column;gap:8px}#lechonk-capture-sidebar h2{font-size:16px;margin:0}#lechonk-capture-sidebar .muted{color:#9aa7ba;font-size:11px}#lechonk-capture-status{color:#9fe6ae}#lechonk-capture-gap{color:#ffc76b;min-height:16px}#lechonk-capture-reasoning{overflow:auto;white-space:pre-wrap;word-break:break-word;flex:1;min-height:100px;background:#151d2b;padding:8px;border-radius:4px}#lechonk-capture-actions{max-height:220px;overflow:auto;white-space:pre-wrap;background:#111722;padding:8px;border-radius:4px}#lechonk-capture-sidebar button{color:#e8edf6;background:#27364e;border:1px solid #5a739c;border-radius:3px;padding:4px 8px;cursor:pointer}`;
    document.head.appendChild(style);
    const aside = document.createElement('aside'); aside.id = 'lechonk-capture-sidebar';
    aside.innerHTML = '<h2>Model capture</h2><div class="muted" id="lechonk-capture-id"></div><div class="muted">bounded live tail; full trace: capture-events.jsonl</div><div id="lechonk-capture-status">waiting for events</div><div id="lechonk-capture-gap"></div><button type="button" id="lechonk-capture-follow">follow: on</button><div id="lechonk-capture-reasoning"></div><div id="lechonk-capture-actions"></div>';
    document.body.appendChild(aside);
    document.getElementById('lechonk-capture-id').textContent = `run ${m.run_id} · capture ${m.capture_id}`;
    let follow = true; let entries = []; let actions = []; const streams = Object.create(null);
    const render = () => {
      const r = document.getElementById('lechonk-capture-reasoning');
      const a = document.getElementById('lechonk-capture-actions');
      r.textContent = entries.map(x => `[${x.label}]\n${x.text}`).join('\n\n').slice(-maxChars);
      a.textContent = actions.join('\n').slice(-6000);
      if (follow) { r.scrollTop = r.scrollHeight; a.scrollTop = a.scrollHeight; }
    };
    document.getElementById('lechonk-capture-follow').addEventListener('click', (ev) => { follow = !follow; ev.currentTarget.textContent = `follow: ${follow ? 'on' : 'off'}`; if (follow) render(); });
    window.__lechonkCaptureUpdate = (item) => {
      if (!item) return;
      if (item.gap) document.getElementById('lechonk-capture-gap').textContent = `LIVE GAP: ${item.gap}`;
      if (item.status) document.getElementById('lechonk-capture-status').textContent = item.status;
      if (item.streams) { entries = Object.entries(item.streams).map(([label, text]) => ({ label, text: String(text).slice(-maxChars) })); actions = Array.isArray(item.actions) ? item.actions.slice(-maxEvents) : actions; }
      if (item.stream && item.chunk) {
        const lane = String(item.stream); const chunk = String(item.chunk).slice(0, 4000);
        streams[lane] = (streams[lane] || '') + chunk;
        const found = entries.find(x => x.label === lane); if (found) found.text = streams[lane]; else entries.push({ label: lane, text: streams[lane] });
        if (entries.length > maxEvents) entries = entries.slice(-maxEvents);
      } else if (item.text && item.semantic) { entries.push({ label: item.kind || 'event', text: String(item.text).slice(0, 4000) }); if (entries.length > maxEvents) entries = entries.slice(-maxEvents); }
      if (item.action) { actions.push(String(item.action).slice(0, 1200)); if (actions.length > maxEvents) actions = actions.slice(-maxEvents); }
      render();
    };
  }, { manifest, maxEvents: limits.maxRenderedEvents, maxChars: limits.maxRenderedChars });
}

async function sha256File(file) {
  return new Promise((resolve, reject) => { const h = crypto.createHash('sha256'); const s = fs.createReadStream(file); s.on('data', d => h.update(d)); s.on('error', reject); s.on('end', () => resolve(h.digest('hex'))); });
}

async function qualifyTileset(page, opts) {
  const runtime = await page.evaluate(() => ({
    name: window.tileset_name || null,
    width: Number(window.normal_tile_width) || null,
    height: Number(window.normal_tile_height) || null,
    imageCount: Number(window.tileset_image_count) || null,
    resources: performance.getEntriesByType('resource').map(x => x.name).filter(x => /tileset.*(?:amplio2|isotrident)/i.test(x)),
  }));
  const expected = opts.tileset === 'isotrident' ? { name: 'isotrident', width: 64, height: 32, imageCount: 2 } : { name: 'amplio2', width: 96, height: 48, imageCount: 3 };
  const loadedSheets = runtime.resources.filter(x => x.includes(`freeciv-web-tileset-${expected.name}-`)).length;
  const assetLoadCheckPassed = runtime.name === expected.name && runtime.width === expected.width && runtime.height === expected.height && runtime.imageCount === expected.imageCount && loadedSheets >= expected.imageCount;
  const result = { requested: expected, runtime, loadedSheets, asset_load_check_passed: assetLoadCheckPassed, visual_qualification: opts.tileset === 'isotrident' ? 'experimental_unverified' : 'bundled_default' };
  if (!assetLoadCheckPassed) throw new Error(`tileset asset load check failed: ${JSON.stringify(result)}`);
  return result;
}

class Presenter {
  constructor(opts, manifest, page, browser, context) {
    this.opts = opts; this.manifest = manifest; this.page = page; this.browser = browser; this.context = context;
    this.dir = opts.captureDir; this.eventsFile = path.join(this.dir, 'capture-events.jsonl'); this.imagesDir = path.join(this.dir, 'images');
    this.ackDir = path.join(this.dir, 'checkpoint-acks');
    this.imagesLog = path.join(this.dir, 'images.jsonl'); this.presenterLog = path.join(this.dir, 'presenter-events.jsonl');
    this.offset = 0; this.pending = Buffer.alloc(0); this.fileIdentity = null; this.replaying = true; this.replayLimit = 0; this.fileLock = null; this.lastSeq = 0; this.newestRelevantSeq = null; this.previousStreamEvent = null; this.seenEventIds = new Set(); this.uiStreams = Object.create(null); this.uiActions = []; this.uiPending = null; this.uiTimer = null; this.uiTask = null; this.uiUpdates = 0; this.pageOpTail = Promise.resolve(); this.writeTail = Promise.resolve(); this.screenshotBroken = false; this.queue = []; this.active = false; this.polling = false; this.pollTask = null; this.drainTask = null; this.stopping = false; this.terminalSeen = false; this.localImageSeq = 0; this.timer = null; this.errors = [];
  }
  withPageGate(fn) { const next = this.pageOpTail.then(fn, fn); this.pageOpTail = next.catch(() => {}); return next; }
  record(kind, payload = {}) { const line = { schema_version: 'capture-presenter/v1', at: nowIso(), run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, kind, payload }; try { writeJsonLine(this.presenterStream, line); } catch (_) {} }
  async start() {
    await fsp.mkdir(this.imagesDir, { recursive: true });
    await fsp.mkdir(this.ackDir, { recursive: true, mode: 0o700 });
    try { this.fileLock = await fsp.open(path.join(this.dir, '.presenter.lock'), 'wx', 0o600); } catch (_) { throw new Error(`another presenter appears active (${path.join(this.dir, '.presenter.lock')}); use a new capture directory`); }
    try {
      try { this.replayLimit = (await fsp.stat(this.eventsFile)).size; } catch (_) { this.replayLimit = null; }
      this.presenterStream = fs.createWriteStream(this.presenterLog, { flags: 'a', mode: 0o600 });
      this.imagesStream = fs.createWriteStream(this.imagesLog, { flags: 'a', mode: 0o600 });
      this.record('presenter_started', { event_file: this.eventsFile, surface: 'composite_viewport_fallback', continuous_video: false, binding_mode: this.manifest.binding_mode || 'prototype_binding', nonce_evidence: this.manifest.nonce_evidence || 'not_provided' });
      await injectSidebar(this.page, this.manifest, this.opts);
      await this.poll();
      if (!this.stopping) { const baseline = presenterBaselineEvent(); this.record('presenter_attached_baseline_requested', { trigger: baseline.kind, temporal_relation: 'unknown' }); this.enqueue(baseline); }
      this.timer = setInterval(() => { this.pollTask = this.poll().catch(e => this.fail(e)); }, this.opts.eventPollMs);
    } catch (e) {
      if (this.timer) clearInterval(this.timer); if (this.presenterStream) this.presenterStream.destroy(); if (this.imagesStream) this.imagesStream.destroy();
      try { await this.fileLock.close(); } catch (_) {} try { await fsp.unlink(path.join(this.dir, '.presenter.lock')); } catch (_) {}
      try { await this.context.close(); } catch (_) {} try { await this.browser.close(); } catch (_) {}
      throw e;
    }
  }
  async poll() {
    if (this.stopping || this.polling) return;
    this.polling = true;
    let handle;
    try {
      const stat = await fsp.stat(this.eventsFile);
      if (this.replaying && this.replayLimit === null) this.replayLimit = stat.size;
      const identity = `${stat.dev}:${stat.ino}`;
      if (this.fileIdentity && identity !== this.fileIdentity) { this.record('live_gap', { reason: 'event_file_rotated', previous_identity: this.fileIdentity, new_identity: identity }); this.offset = 0; this.pending = Buffer.alloc(0); }
      this.fileIdentity = identity;
      if (stat.size < this.offset) { this.record('live_gap', { reason: 'event_file_shrank', previous_offset: this.offset, new_offset: stat.size }); this.offset = 0; this.pending = Buffer.alloc(0); }
      handle = await fsp.open(this.eventsFile, 'r');
      const readEnd = this.replaying ? Math.min(stat.size, this.replayLimit) : stat.size;
      const available = Math.max(0, readEnd - this.offset); const buf = Buffer.alloc(available);
      if (available) { const read = await handle.read(buf, 0, available, this.offset); this.offset += read.bytesRead; }
      const chunk = Buffer.concat([this.pending, buf]);
      let start = 0;
      for (let i = 0; i < chunk.length; i += 1) if (chunk[i] === 10) {
        const raw = chunk.subarray(start, i).toString('utf8').trim(); start = i + 1; if (!raw) continue;
        let e; try { e = JSON.parse(raw); } catch (_) { this.record('live_gap', { reason: 'invalid_event_json' }); continue; }
        await this.consume(e);
      }
      this.pending = chunk.subarray(start);
      if (this.replaying && this.offset >= this.replayLimit) {
        if (this.pending.length) { this.record('live_gap', { reason: 'reattach_partial_line_discarded', bytes: this.pending.length }); this.pending = Buffer.alloc(0); }
        this.replaying = false; this.record('reattach_watermark_reached', { replay_limit_bytes: this.replayLimit, next_capture_seq: this.lastSeq + 1 }); this.scheduleUi(true);
      }
    } catch (e) { if (e.code !== 'ENOENT') this.record('capture_events_read_error', { error: e.message }); }
    finally { if (handle) await handle.close(); this.polling = false; }
  }
  async consume(e) {
    const identityError = validateEventIdentity(e, this.manifest);
    if (identityError) { this.fail(new Error(identityError)); return; }
    const seq = sourceSeq(e); if (seq == null) { this.record('live_gap', { reason: 'missing_or_invalid_capture_seq', event_kind: eventKind(e) }); }
    const eventId = e.event_id || e.id || (seq == null ? null : `seq:${seq}`); if (eventId && this.seenEventIds.has(String(eventId))) { this.record('duplicate_event', { event_id: eventId, capture_seq: seq }); return; } if (eventId) this.seenEventIds.add(String(eventId));
    if (seq != null) { if (seq <= this.lastSeq) { this.record('duplicate_event', { capture_seq: seq, event_id: eventId }); return; } if (this.lastSeq && seq > this.lastSeq + 1) { const gap = `${this.lastSeq + 1}-${seq - 1} (source sequence)`; this.record('live_gap', { gap, previous_seq: this.lastSeq, next_seq: seq }); await this.withPageGate(() => this.page.evaluate(g => window.__lechonkCaptureUpdate && window.__lechonkCaptureUpdate({ gap }), gap)); } this.lastSeq = seq; }
    const text = displayForEvent(e); const p = eventPayload(e); const k = eventKind(e);
    const part = streamPart(e, k); let stream = part && part.lane; let chunk = part && part.text;
    const prev = this.previousStreamEvent; const currentIdentity = streamIdentity(e);
    const mirror = prev && prev.kind && /request[_-]?delta/i.test(prev.kind) && /model[_-]?output/i.test(k) && sameStreamIdentity(prev.identity, currentIdentity) && prev.lane === stream && prev.text === chunk;
    if (mirror) { this.record('model_delta_mirror_suppressed', { physical_call_id: currentIdentity.physical, response_id: currentIdentity.response, lane: stream, source_seq: seq }); stream = null; chunk = null; this.previousStreamEvent = null; }
    else if (part) this.previousStreamEvent = { kind: k, identity: currentIdentity, lane: stream, text: chunk };
    else this.previousStreamEvent = null;
    let action = null;
    if (/^(turn[_-]?decision|action[_-]?result|game[_-]?outcome|run[_-]?(finished|outcome))$/i.test(k)) {
      action = actionDisplay(p, k);
    }
    const status = `${k}${seq == null ? '' : ` · source #${seq}`}`;
    if (!this.replaying) {
      if (stream && chunk) this.uiStreams[stream] = boundedAppend(this.uiStreams[stream], chunk);
      if (action) { this.uiActions.push(action); if (this.uiActions.length > this.opts.maxRenderedEvents) this.uiActions = this.uiActions.slice(-this.opts.maxRenderedEvents); }
      this.uiPending = { seq, kind: k, text: mirror ? '' : text, action, status, streams: { ...this.uiStreams }, actions: this.uiActions.slice() };
      this.scheduleUi(false);
    }
    if (!this.replaying && shouldScreenshot(e)) {
      this.newestRelevantSeq = seq == null ? this.newestRelevantSeq : seq;
      this.enqueue(e);
    }
    if (!this.replaying && isTerminal(e)) { this.terminalSeen = true; this.record('terminal_event_seen', { source_seq: seq, kind: k }); setTimeout(() => this.stop(), Math.max(800, this.opts.settleMs + 500)); }
  }
  scheduleUi(snapshot) {
    if (snapshot) this.uiPending = { status: 'reattach watermark reached', streams: { ...this.uiStreams }, actions: this.uiActions.slice(), snapshot: true };
    if (this.uiTimer) return;
    this.uiTimer = setTimeout(() => { this.uiTimer = null; this.uiTask = (async () => { const item = this.uiPending; this.uiPending = null; if (!item) return; this.uiUpdates += 1; try { await this.withPageGate(() => this.page.evaluate((x) => window.__lechonkCaptureUpdate && window.__lechonkCaptureUpdate(x), item)); } catch (e) { this.record('ui_update_error', { error: e.message }); } })().finally(() => { this.uiTask = null; }); }, snapshot ? 0 : 200);
  }
  enqueue(e) {
    const trigger = eventKind(e); const seq = sourceSeq(e);
    const checkpoint = Boolean(eventPayload(e).capture_checkpoint) && seq != null;
    if (this.screenshotBroken || this.stopping) { this.record('screenshot_request_failed', { status: 'failed_screenshot_circuit_breaker', trigger, source_seq: seq }); this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_screenshot_circuit_breaker', trigger, source_event: { capture_seq: seq, event_id: e.event_id || e.id || null } }); if (checkpoint) this.writeCheckpointAck(e, 'failed', 'screenshot_circuit_breaker').catch(() => {}); return; }
    const priority = screenshotPriority(e); const routine = isRoutineScreenshot(e);
    // Routine movement/production frames are replaceable. Milestones and the
    // outcome keep their own queue slots so a burst cannot hide the story.
    if (routine && !checkpoint) {
      const old = this.queue.find(x => isRoutineScreenshot(x));
      if (old) {
        this.queue = this.queue.filter(x => x !== old);
        this.record('screenshot_coalesced', { dropped_seq: sourceSeq(old), dropped_kind: eventKind(old), replacement_seq: seq, status: 'coalesced_routine' });
        this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'coalesced_routine', trigger: eventKind(old), source_event: { capture_seq: sourceSeq(old), event_id: old.event_id || old.id || null }, note: 'Routine screenshot replaced by a newer routine event.' });
      }
    }
    // Coalesce repeated triggers for one source sequence while retaining the
    // latest event. This keeps capture nonblocking and bounded.
    if (this.queue.some(x => sourceSeq(x) === seq && seq != null)) {
      if (checkpoint) this.writeCheckpointAck(e, 'failed', 'duplicate_checkpoint').catch(() => {});
      return;
    }
    if (isTerminal(e)) {
      const routineItems = this.queue.filter(x => isRoutineScreenshot(x) && !eventPayload(x).capture_checkpoint);
      this.queue = this.queue.filter(x => !(isRoutineScreenshot(x) && !eventPayload(x).capture_checkpoint));
      for (const dropped of routineItems) {
        const drop = { dropped_seq: sourceSeq(dropped), dropped_kind: eventKind(dropped), replacement_seq: seq, status: 'coalesced_for_terminal' };
        this.record('screenshot_coalesced', drop);
        this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'coalesced_for_terminal', trigger: eventKind(dropped), source_event: { capture_seq: sourceSeq(dropped), event_id: dropped.event_id || dropped.id || null }, note: 'Routine screenshot removed to capture terminal outcome promptly.' });
      }
    }
    if (this.queue.length >= this.opts.queueLimit) {
      const victimIndex = this.queue.findIndex(x => !eventPayload(x).capture_checkpoint && screenshotPriority(x) < priority);
      if (checkpoint && victimIndex < 0) {
        const rejected = { dropped_seq: seq, dropped_kind: trigger, status: 'failed_checkpoint_queue_full', incoming_priority: priority };
        this.record('screenshot_coalesced', rejected);
        this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_checkpoint_queue_full', trigger, source_event: { capture_seq: seq, event_id: e.event_id || e.id || null } });
        this.writeCheckpointAck(e, 'failed', 'queue_full').catch(() => {});
        return;
      }
      const index = victimIndex;
      if (index < 0 && priority <= Math.min(...this.queue.map(screenshotPriority))) {
        const rejected = { dropped_seq: seq, dropped_kind: trigger, status: 'failed_coalesced_incoming', incoming_priority: priority };
        this.record('screenshot_coalesced', rejected);
        this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_coalesced_incoming', trigger, source_event: { capture_seq: seq, event_id: e.event_id || e.id || null }, note: 'Incoming screenshot rejected because the bounded queue contains only equal or higher-priority events.' });
        if (checkpoint) this.writeCheckpointAck(e, 'failed', 'queue_full').catch(() => {});
        return;
      }
      const dropIndex = index >= 0 ? index : 0; const dropped = this.queue.splice(dropIndex, 1)[0];
      if (eventPayload(dropped).capture_checkpoint) {
        this.writeCheckpointAck(dropped, 'failed', 'queue_evicted').catch(() => {});
      }
      const drop = { dropped_seq: sourceSeq(dropped), dropped_kind: eventKind(dropped), replacement_seq: seq, status: 'failed_coalesced', dropped_priority: screenshotPriority(dropped), replacement_priority: priority };
      this.record('screenshot_coalesced', drop); this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_coalesced', trigger: eventKind(dropped), source_event: { capture_seq: sourceSeq(dropped), event_id: dropped.event_id || dropped.id || null }, note: 'Bounded queue dropped a lower-priority screenshot before PNG capture.' });
    }
    if (isTerminal(e)) this.queue.unshift(e); else this.queue.push(e);
    this.drainScreenshots();
  }
  async drainScreenshots() {
    if (this.drainTask) return this.drainTask; this.drainTask = (async () => {
    if (this.active || this.stopping && !this.queue.length) return; this.active = true;
    while (this.queue.length) { const e = this.queue.shift(); await new Promise(r => setTimeout(r, this.opts.settleMs)); await this.capture(e); }
    this.active = false;
    if (this.stopping) await this.closeStreams();
    })(); try { await this.drainTask; } finally { this.drainTask = null; }
  }
  async focusCamera(e) {
    const requested = focusTileFromEvent(e);
    if (!requested) return { mode: 'overview_manual', requested: null, result: 'no_structured_tile' };
    try {
      const result = await this.withPageGate(() => this.page.evaluate(async ({ x, y }) => {
        const fn = typeof window.map_pos_to_tile === 'function' ? window.map_pos_to_tile : null;
        const center = typeof window.center_tile_mapcanvas === 'function' ? window.center_tile_mapcanvas : null;
        if (!fn || !center) return { result: 'unavailable', has_map_pos_to_tile: Boolean(fn), has_center_tile_mapcanvas: Boolean(center) };
        let tile;
        try { tile = fn(x, y); } catch (error) { return { result: 'map_pos_to_tile_error', error: String(error && error.message || error) }; }
        if (!tile) return { result: 'tile_unresolved' };
        try {
          center(tile);
          if (typeof window.update_map_canvas_full === 'function') window.update_map_canvas_full();
          // Allow the map canvas to consume the center request. This confirms
          // the command was accepted, but cannot prove what pixels were drawn.
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          return { result: 'requested_unverified', tile: { x, y } };
        } catch (error) { return { result: 'center_error', error: String(error && error.message || error), tile: { x, y } }; }
      }, requested));
      const normalized = result && typeof result === 'object' ? result : { result: 'unknown' };
      return { mode: normalized.result === 'requested_unverified' ? 'event_tile_unverified' : 'overview_manual', source: requested.source, requested, ...normalized };
    } catch (error) {
      return { mode: 'overview_manual', requested, result: 'evaluate_error', error: error.message };
    }
  }
  async capture(e) {
    const requestedAt = nowIso(); const requestMono = performance.now(); const seq = sourceSeq(e); const local = ++this.localImageSeq; const stamp = `${Date.now()}-${String(local).padStart(4, '0')}`; const imageId = `${this.manifest.capture_id}-${stamp}`; const finalPath = path.join(this.imagesDir, `${imageId}.png`); const tempPath = `${finalPath}.part`;
    try {
      const camera = await this.focusCamera(e);
      this.record('camera_result', { source_seq: seq, trigger: eventKind(e), ...camera });
      await this.withPageGate(() => this.page.screenshot({ ...screenshotOptions(tempPath), timeout: this.opts.screenshotTimeoutMs }));
      const fd = await fsp.open(tempPath, 'r+'); await fd.sync(); await fd.close(); await fsp.rename(tempPath, finalPath);
      const dirFd = await fsp.open(this.imagesDir, 'r'); try { await dirFd.sync(); } finally { await dirFd.close(); }
      const stat = await fsp.stat(finalPath); const hash = await sha256File(finalPath);
      const completionSeq = this.lastSeq || seq; const late = seq != null && completionSeq != null && completionSeq > seq;
  const meta = { schema_version: 'capture-image/v1', image_id: imageId, file: path.relative(this.dir, finalPath), sha256: hash, bytes: stat.size, dimensions: { width: this.opts.viewportWidth, height: this.opts.viewportHeight }, run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, source_event: { capture_seq: seq, kind: eventKind(e), event_id: e.event_id || e.id || null }, requested_capture_seq: seq, completion_capture_seq: completionSeq, newest_relevant_seq: this.newestRelevantSeq, overtaken_by_newer_event: late, watermark: e.watermark ?? eventPayload(e).watermark ?? null, trigger: eventKind(e), requested_at: requestedAt, completed_at: nowIso(), start_monotonic_ms: requestMono, end_monotonic_ms: performance.now(), clock_domain_id: `presenter-${process.pid}`, turn: eventPayload(e).turn ?? eventPayload(e).turn_number ?? null, status: eventPayload(e).status || eventPayload(e).result || 'unknown', fog: eventPayload(e).fog ?? 'unknown', temporal_relation: late ? 'late_after_intervening_events' : 'unknown', observer_freshness: 'unknown', readback: 'unknown', visual_evidence: 'unavailable', screenshot_surface: 'composite_viewport_fallback', camera_mode: camera.mode, camera: camera, visual_settings: { renderer: this.opts.renderer, tileset: this.opts.tileset, qualification: this.opts.tilesetQualification || null, asset_hashes: this.opts.tilesetAssetHashes || null }, note: 'Sidebar is presentation-only; PNG is not proof of game state transition.' };
      await this.appendImageMeta(meta);
      this.record('screenshot_completed', meta);
      if (eventPayload(e).capture_checkpoint) await this.writeCheckpointAck(e, 'captured', null, imageId);
    } catch (err) { try { await fsp.unlink(tempPath); } catch (_) {} this.record('screenshot_error', { source_seq: seq, kind: eventKind(e), error: err.message }); this.errors.push(err); if (eventPayload(e).capture_checkpoint) await this.writeCheckpointAck(e, 'failed', err.message); if (/timeout|timed out/i.test(err.message)) { this.screenshotBroken = true; await this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_screenshot_timeout', trigger: eventKind(e), source_event: { capture_seq: seq, event_id: e.event_id || e.id || null } }); if (this.timer) clearInterval(this.timer); if (this.uiTimer) { clearTimeout(this.uiTimer); this.uiTimer = null; } while (this.queue.length) { const dropped = this.queue.shift(); await this.appendImageMeta({ schema_version: 'capture-image/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, status: 'failed_screenshot_circuit_breaker', trigger: eventKind(dropped), source_event: { capture_seq: sourceSeq(dropped), event_id: dropped.event_id || dropped.id || null } }); if (eventPayload(dropped).capture_checkpoint) await this.writeCheckpointAck(dropped, 'failed', 'screenshot_circuit_breaker'); } this.stopping = true; } }
  }
  async writeCheckpointAck(e, status, reason = null, imageId = null) {
    const seq = sourceSeq(e); if (seq == null || !eventPayload(e).capture_checkpoint) return;
    const ack = { schema_version: 'capture-checkpoint-ack/v1', run_id: this.manifest.run_id, capture_id: this.manifest.capture_id, capture_seq: seq, status, image_id: imageId, reason, at: nowIso(), completion_evidence: status === 'captured' ? 'png_renamed_and_images_log_write_callback' : null, filesystem_durability: 'not_fsync_verified' };
    const target = path.join(this.ackDir, `${seq}.json`); const temp = `${target}.${process.pid}.${Date.now()}.tmp`;
    await fsp.writeFile(temp, `${JSON.stringify(ack)}\n`, { mode: 0o600 });
    try { await fsp.link(temp, target); } catch (err) { if (err.code !== 'EEXIST') throw err; } finally { try { await fsp.unlink(temp); } catch (_) {} }
  }
  async appendImageMeta(meta) { const task = this.writeTail.then(() => new Promise((resolve, reject) => { const line = `${JSON.stringify(meta)}\n`; this.imagesStream.write(line, err => err ? reject(err) : resolve()); })); this.writeTail = task.catch(() => {}); return task; }
  fail(e) { this.errors.push(e); this.record('presenter_error', { error: e.message }); console.error(`[capture-presenter] ${e.message}`); this.stop(); }
  async stop() {
    if (this.stopping) return; this.stopping = true; if (this.timer) clearInterval(this.timer);
    this.record('presenter_stopping', { terminal_seen: this.terminalSeen, last_source_seq: this.lastSeq, queued: this.queue.length });
    if (this.pollTask) await this.pollTask;
    if (this.uiTimer) { clearTimeout(this.uiTimer); this.uiTimer = null; }
    if (this.uiTask) await this.uiTask;
    await this.pageOpTail;
    await this.writeTail;
    if (this.queue.length || this.active) await this.drainScreenshots(); else await this.closeStreams();
  }
  async closeStreams() {
    if (this.closed) return; this.closed = true; this.record('presenter_finished', { errors: this.errors.length, terminal_seen: this.terminalSeen, last_source_seq: this.lastSeq, completeness: this.terminalSeen ? 'terminal_event_observed' : 'stopped_before_terminal' });
    await this.writeTail;
    await new Promise(r => this.presenterStream.end(r)); await new Promise(r => this.imagesStream.end(r));
    try { await this.fileLock.close(); } catch (_) {} try { await fsp.unlink(path.join(this.dir, '.presenter.lock')); } catch (_) {}
    try { await this.context.close(); } catch (_) {} try { await this.browser.close(); } catch (_) {}
  }
}

async function run(opts) {
  if (!opts.captureDir) throw new Error('--capture-dir is required');
  await fsp.mkdir(opts.captureDir, { recursive: true });
  const manifest = await waitForManifest(opts.captureDir, opts.manifestWaitMs);
  if (manifestEnded(manifest)) throw new Error(`manifest is already finished (${manifest.status || manifest.state || 'finished'}); refusing stale attach`);
  if (!opts.headless && !process.env.DISPLAY && !process.env.WAYLAND_DISPLAY) throw new Error('headed presenter requires DISPLAY or WAYLAND_DISPLAY; pass --headless for smoke capture');
  let chromium; try { ({ chromium } = require('playwright')); } catch (_) { throw new Error('Playwright is required (npm install)'); }
  const browser = await chromium.launch({ headless: opts.headless }); const context = await browser.newContext({ viewport: { width: opts.viewportWidth, height: opts.viewportHeight }, deviceScaleFactor: 1 });
  if (opts.tileset === 'isotrident') {
    const tilesetDir = path.resolve(__dirname, 'tilesets');
    const config = await fsp.readFile(path.join(tilesetDir, 'tileset_config_isotrident.js'));
    const spec = await fsp.readFile(path.join(tilesetDir, 'tileset_spec_isotrident.js'));
    const routePath = suffix => url => { try { return String(url.pathname || url).endsWith(suffix); } catch (_) { return false; } };
    await context.route(routePath('/javascript/2dcanvas/tileset_config_amplio2.js'), route => route.fulfill({ status: 200, contentType: 'application/javascript', body: config }));
    await context.route(routePath('/javascript/2dcanvas/tileset_spec_amplio2.js'), route => route.fulfill({ status: 200, contentType: 'application/javascript', body: spec }));
  }
  const page = await context.newPage();
  const expectedProxyPath = `/civsocket/${1000 + manifest.session_port}`;
  const wsEvidence = { expected_path: expectedProxyPath, sockets: 0, matching_sockets: 0, received_frames: 0, sent_frames: 0, open_matching: 0, closed_matching: 0, errors: 0, urls: [] };
  page.on('websocket', ws => {
    const socketUrl = typeof ws.url === 'function' ? ws.url() : '';
    const matching = (() => { try { return new URL(socketUrl).pathname === expectedProxyPath; } catch (_) { return false; } })();
    wsEvidence.sockets += 1; if (matching) wsEvidence.matching_sockets += 1; if (wsEvidence.urls.length < 4) wsEvidence.urls.push({ path: (() => { try { return new URL(socketUrl).pathname; } catch (_) { return 'invalid'; } })(), matching });
    if (matching) wsEvidence.open_matching += 1;
    ws.on('framereceived', () => { if (matching) wsEvidence.received_frames += 1; });
    ws.on('framesent', () => { if (matching) wsEvidence.sent_frames += 1; });
    ws.on('close', () => { if (matching) { wsEvidence.closed_matching += 1; wsEvidence.open_matching = Math.max(0, wsEvidence.open_matching - 1); } });
    ws.on('socketerror', () => { if (matching) wsEvidence.errors += 1; });
  });
  const observerName = observerUsername(manifest.capture_id);
  if (!validFreecivUsername(observerName)) throw new Error(`generated observer username violates Freeciv validator: ${observerName}`);
  const url = `http://localhost:8080/webclient/?action=observe&civserverport=${manifest.session_port}&civserverhost=unknown&multi=true&type=multiplayer${opts.renderer === 'webgl' ? '&renderer=webgl' : ''}`;
  console.log(`[capture-presenter] attaching run=${manifest.run_id} capture=${manifest.capture_id} port=${manifest.session_port} expected=${manifest.expected_username}`);
  await page.goto(url, { waitUntil: 'load', timeout: 30000 }); await page.waitForTimeout(1500);
  const input = page.locator('#username_req'); if (await input.count()) await input.fill(observerName);
  const button = page.getByRole('button', { name: /observe game/i }).first(); if (!(await button.count())) throw new Error('Observe Game button not found; refusing to attach'); await button.click();
  let probe = null; const deadline = Date.now() + 30000;
  while (Date.now() < deadline) { await page.waitForTimeout(500); probe = await probeObserver(page, manifest.expected_username); if (probe.observer === true && probe.playing === null && hasExactRosterMatch(probe, manifest.expected_username)) break; }
  const portWitness = new URL(page.url()).searchParams.get('civserverport');
  const nonceWitness = new URL(page.url()).searchParams.get('session_nonce');
  const socketReady = socketEvidenceReady(wsEvidence);
  if (!probe || probe.observer !== true || probe.connected !== true || !socketReady || probe.playing !== null || !probe.gameInfo || !Array.isArray(probe.roster) || !probe.roster.length || !hasExactRosterMatch(probe, manifest.expected_username) || String(portWitness) !== String(manifest.session_port) || (manifest.session_nonce && nonceWitness !== manifest.session_nonce)) { startupDiagnostic(opts.captureDir, 'observer binding not provable', { phase: 'binding', probe, websocket: wsEvidence, page_url: page.url(), portWitness, nonceWitness }); await context.close(); await browser.close(); throw new Error(`observer binding not provable; expected active websocket ${expectedProxyPath} with frames, connected observer=true, playing=null, nonempty game_info/roster, exactly one roster username ${manifest.expected_username}, URL port ${manifest.session_port}, nonce=${manifest.session_nonce ? 'matching' : 'not-provided'}; evidence=${JSON.stringify({ probe, websocket: wsEvidence, page_url: page.url(), portWitness, nonceWitness })}`); }
  const tilesetQualification = await qualifyTileset(page, opts);
  opts.tilesetQualification = tilesetQualification;
  if (opts.tileset === 'isotrident') {
    opts.tilesetAssetHashes = {
      config: await sha256File(path.resolve(__dirname, 'tilesets/tileset_config_isotrident.js')),
      spec: await sha256File(path.resolve(__dirname, 'tilesets/tileset_spec_isotrident.js')),
    };
  }
  manifest.binding_mode = manifest.session_nonce ? 'verified_nonce_binding' : 'prototype_binding'; manifest.nonce_evidence = manifest.session_nonce ? 'url_query_match' : 'not_provided';
  console.log(`[capture-presenter] bound observer=true playing=null roster=${manifest.expected_username} port=${portWitness} nonce=${manifest.nonce_evidence} renderer=${opts.renderer} tileset=${opts.tileset}`);
  const presenter = new Presenter(opts, manifest, page, browser, context); await presenter.start();
  const stop = () => presenter.stop(); process.once('SIGINT', stop); process.once('SIGTERM', stop);
  await new Promise(resolve => { const poll = setInterval(() => { if (presenter.closed) { clearInterval(poll); resolve(); } }, 100); });
}

async function selfTest() {
  const assert = (x, msg) => { if (!x) throw new Error(msg); };
  const m = validateManifest({ schema_version: 'gateway/v1', session_ready: true, run_id: 'r', capture_id: 'c', session_port: 6300, expected_username: 'p' }); assert(m.session_port === 6300, 'manifest');
  assert(manifestEnded({ ...m, status: 'finished' }), 'ended manifest'); assert(!manifestEnded({ ...m, status: 'ready' }), 'ready manifest');
  assert(stableId('abc').length === 16, 'stable id'); assert(validFreecivUsername(observerUsername('abc')), 'observer username validator'); assert(!validFreecivUsername('capture-observer-abc'), 'invalid observer username rejected'); assert(sourceSeq({ capture_seq: 4 }) === 4, 'source sequence'); assert(sourceSeq({ capture_seq: 'x' }) === null, 'invalid sequence');
  assert(shouldScreenshot({ kind: 'turn_start' }), 'turn trigger'); assert(shouldScreenshot({ kind: 'action_result' }), 'action trigger'); assert(shouldScreenshot({ kind: 'capture_finished' }), 'capture terminal trigger'); assert(!shouldScreenshot({ kind: 'model_delta' }), 'non-trigger');
  assert(screenshotPriority({ kind: 'game_outcome' }) > screenshotPriority({ kind: 'action_result', payload: { verb: 'goto' } }), 'terminal priority');
  assert(screenshotPriority({ kind: 'action_result', payload: { verb: 'build_city', domain: 'unit' } }) > screenshotPriority({ kind: 'action_result', payload: { verb: 'goto' } }), 'milestone priority');
  assert(isRoutineScreenshot({ kind: 'action_result', payload: { verb: 'goto' } }), 'routine classification');
  assert(displayForEvent({ payload: { raw_delta: { reasoning: 'safe', answer: 'ok' } } }).includes('reasoning: safe'), 'nested raw delta display');
  assert(validateEventIdentity({ run_id: 'r', capture_id: 'c', schema_version: 'gateway/v1' }, m) === null, 'event identity');
  assert(validateEventIdentity({ run_id: 'wrong', capture_id: 'c', schema_version: 'gateway/v1' }, m), 'capture mismatch');
  assert(hasExactRosterMatch({ expectedInRosterUsername: true, expectedUsernameMatches: 1 }, 'p'), 'unique roster username');
  assert(!hasExactRosterMatch({ expectedInRosterUsername: true, expectedUsernameMatches: 2 }, 'p'), 'duplicate roster rejected');
  assert(!hasExactRosterMatch({ expectedInRosterUsername: false, expectedUsernameMatches: 0, expectedInDisplayName: true }, 'p'), 'display name is not username');
  assert(displayForEvent({ payload: { raw_delta: { reasoning: '<b>escaped</b>' } } }).includes('<b>escaped</b>'), 'safe text retained for textContent');
  assert(socketEvidenceReady({ matching_sockets: 1, open_matching: 1, received_frames: 1 }), 'active matching websocket');
  assert(!socketEvidenceReady({ matching_sockets: 1, open_matching: 1, received_frames: 0 }), 'no frame is not ready');
  assert(!socketEvidenceReady({ matching_sockets: 1, open_matching: 0, received_frames: 2 }), 'closed socket is not ready');
  assert(screenshotOptions('/tmp/image.png.part').type === 'png', 'explicit screenshot MIME');
  assert(presenterBaselineEvent().kind === 'presenter_attached', 'current-state baseline trigger');
  assert(mergeStreamChunk('', '', 'Hel') === 'Hel' && mergeStreamChunk('Hel', 'Hel', 'Hel') === 'HelHel' && mergeStreamChunk('Hel', 'Hel', 'lo') === 'Hello', 'repeated provider chunks retained');
  let stress = ''; for (let i = 0; i < 100000; i += 1) stress = boundedAppend(stress, 'x'); assert(stress.length === 12000, 'bounded reducer memory');
  assert(streamPart({ payload: { reasoning: 'direct' } }, 'model_output').text === 'direct', 'direct model output normalized');
  assert(streamPart({ payload: { raw_delta: { reasoning: 'raw' } } }, 'request_delta').text === 'raw', 'raw request delta normalized');
  assert(sameStreamIdentity(streamIdentity({ payload: { response_id: 'resp-1', physical_call_id: 'pc-1', logical_call_id: 'lc-1' } }), streamIdentity({ payload: { response_id: 'resp-1', physical_call_id: 'pc-1', logical_call_id: 'lc-1' } })), 'physical/logical mirror identity');
  assert(sameStreamIdentity(streamIdentity({ payload: { response_id: 'resp-1' } }), streamIdentity({ payload: { response_id: 'resp-1' } })), 'response-only mirror identity');
  assert(!sameStreamIdentity(streamIdentity({ payload: { response_id: 'resp-1', physical_call_id: 'pc-1' } }), streamIdentity({ payload: { response_id: 'resp-1', physical_call_id: 'pc-2' } })), 'contradictory physical calls retained');
  assert(!sameStreamIdentity(streamIdentity({ payload: {} }), streamIdentity({ payload: { physical_call_id: 'pc-1' } })), 'missing identity retained');
  assert(actionDisplay({ ref: 'unit:102', verb: 'build_city', outcome: 'executed', readback: 'verified' }, 'action_result') === 'unit:102:build_city [executed] {verified}', 'structured action display');
  assert(actionDisplay({ outcome: 'executed' }, 'engine_progress') === 'engine automatic [executed]', 'automatic action display');
  assert(JSON.stringify(focusTileFromEvent({ payload: { affected: 'tile:12,8' } })) === JSON.stringify({ x: 12, y: 8, source: 'affected' }), 'affected tile focus');
  assert(focusTileFromEvent({ payload: { target: 'city:London' } }) === null, 'unresolved focus stays overview');
  assert(parseArgs(['--capture-dir', '/tmp/c', '--reattach']).reattach === true, 'reattach option');
  assert(parseArgs(['--capture-dir', '/tmp/c', '--renderer', 'webgl']).renderer === 'webgl', 'webgl renderer option');
  assert(parseArgs(['--capture-dir', '/tmp/c', '--tileset', 'amplio2']).tileset === 'amplio2', 'amplio2 tileset option');
  let gatedTileset = false; try { parseArgs(['--capture-dir', '/tmp/c', '--tileset', 'isotrident']); } catch (_) { gatedTileset = true; } assert(gatedTileset, 'isotrident tileset gate');
  assert(parseArgs(['--capture-dir', '/tmp/c', '--tileset', 'isotrident', '--allow-experimental-tileset']).tileset === 'isotrident', 'isotrident experimental option');
  let gate = Promise.resolve(); let active = 0; let maxActive = 0;
  const gated = (fn) => { const next = gate.then(async () => { active += 1; maxActive = Math.max(maxActive, active); try { return await fn(); } finally { active -= 1; } }); gate = next.catch(() => {}); return next; };
  await Promise.all([1, 2, 3].map(i => gated(() => new Promise(resolve => setTimeout(() => resolve(i), 1))))); assert(maxActive === 1, 'page operation gate');
  // Exercise the real Presenter gate and timeout circuit with injected fakes.
  const fakeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'capture-presenter-test-')); const fakeImages = path.join(fakeDir, 'images'); fs.mkdirSync(fakeImages);
  const written = []; const fakeStream = { write(line, cb) { written.push(String(line)); if (cb) cb(); return true; }, end(cb) { if (cb) cb(); }, destroy() {} };
  const fakePage = { async screenshot() { throw new Error('Timeout 5ms exceeded'); }, async evaluate() {} };
  const fakeOpts = { captureDir: fakeDir, settleMs: 0, queueLimit: 3, screenshotTimeoutMs: 5, viewportWidth: 100, viewportHeight: 100, eventPollMs: 100, maxRenderedEvents: 10, maxRenderedChars: 1000 };
  const fakeManifest = { run_id: 'r', capture_id: 'c', expected_username: 'p', session_port: 1 };
  const fp = new Presenter(fakeOpts, fakeManifest, fakePage, { async close() {} }, { async close() {} }); fp.imagesDir = fakeImages; fp.ackDir = path.join(fakeDir, 'checkpoint-acks'); await fsp.mkdir(fp.ackDir); fp.imagesStream = fakeStream; fp.presenterStream = fakeStream; fp.fileLock = { async close() {} };
  let fpActive = 0; let fpMax = 0; const op1 = fp.withPageGate(async () => { fpActive += 1; fpMax = Math.max(fpMax, fpActive); await new Promise(r => setTimeout(r, 2)); fpActive -= 1; }); const op2 = fp.withPageGate(async () => { fpActive += 1; fpMax = Math.max(fpMax, fpActive); fpActive -= 1; }); await Promise.all([op1, op2]); assert(fpMax === 1, 'real Presenter gate');
  fp.queue.push({ kind: 'action_result', capture_seq: 2, payload: { capture_checkpoint: true } }); fp.timer = setInterval(() => {}, 1000); await fp.capture({ kind: 'action_result', capture_seq: 1, payload: { capture_checkpoint: true } }); await fp.writeTail; assert(fp.screenshotBroken && fp.queue.length === 0 && fp.timer._destroyed, 'real timeout circuit cleanup'); assert(written.some(line => line.includes('failed_screenshot_timeout')) && written.some(line => line.includes('failed_screenshot_circuit_breaker')), 'timeout failure metadata'); const ack1 = JSON.parse(await fsp.readFile(path.join(fp.ackDir, '1.json'), 'utf8')); const ack2 = JSON.parse(await fsp.readFile(path.join(fp.ackDir, '2.json'), 'utf8')); assert(ack1.status === 'failed' && ack2.status === 'failed', 'checkpoint failure acknowledgements'); fs.rmSync(fakeDir, { recursive: true, force: true });
  assert(eventKind({ kind: 'progress', payload: { event: 'turn_decision' } }) === 'turn_decision', 'gateway progress unwrap');
  console.log('capture_presenter self-test: PASS');
}

if (require.main === module) (async () => { let opts; try { opts = parseArgs(process.argv.slice(2)); if (opts.help) return usage(); if (opts.selfTest) return selfTest(); await run(opts); } catch (e) { if (opts && opts.captureDir) startupDiagnostic(opts.captureDir, e.message, { phase: 'startup' }); console.error(`[capture-presenter] fatal: ${e.message}`); process.exitCode = 1; } })();

module.exports = { parseArgs, validateManifest, manifestEnded, displayForEvent, shouldScreenshot, screenshotPriority, isRoutineScreenshot, sourceSeq, stableId, observerUsername, validFreecivUsername, hasExactRosterMatch, socketEvidenceReady, screenshotOptions, presenterBaselineEvent, mergeStreamChunk, boundedAppend, streamPart, streamIdentity, sameStreamIdentity, actionDisplay, focusTileFromEvent };
