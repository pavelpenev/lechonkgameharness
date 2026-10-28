# Project Plan: Le Chonk Freeciv Harness

## Append-only conversation history and compaction — 2026-10-27

**Type:** architecture / feature
**Status:** implementation complete; long-game validation pending
**Design:** [docs/conversation-history-design.md](docs/conversation-history-design.md)

### Objective

Make interactive play retain the exact Mistral conversation, including thinking
chunks and execution outcomes, so common prefixes can be cached as in a coding
harness. Keep a versioned state summary for durable strategic context and compact
at a deterministic 500,000 estimated-token trigger. Keep evaluation prompts and
request contracts unchanged.

### Decisions

- `conversation_v1` replaces `hybrid_v1` for interactive runs; `legacy` remains
  the library/evaluation default.
- Conversation messages are append-only and transactional: commit only after
  action execution, waiting, terminal classification, and memory advancement.
- Exact assistant content, including nested reasoning chunks, is replayed. A
  failed or partial turn does not create a phantom assistant message.
- The next user message contains actual prior execution results plus the fresh
  authoritative fog-gated snapshot, delta, and candidate table.
- Compaction is only between completed turns at 500,000 estimated tokens. It
  creates a versioned model-authored state summary and retains an exact recent
  50,000–100,000-token tail in a new cache epoch; failure is fail-closed.
- A stable per-run/epoch cache key and cached-token usage are recorded when the
  transport supports them.
- Recovery binds a verified Freeciv save, conversation snapshot, and committed
  strategy memory into a new child run; it is not an in-place restart.

### Dependency-ordered implementation

1. [x] Conversation store: exact messages/chunks, transactional staging,
   checksummed persistence, state file, token accounting, and cache epochs.
2. [x] Transport/client: complete and streaming cache key, cached-token
   telemetry, exact assistant replay, repair transaction, and compaction call.
3. [x] Runner lifecycle: append actual execution report only after completed
   turns; abort safely on transport/truncation/action failure.
4. [x] Interactive wiring: select `conversation_v1`, pass state/conversation
   artifact paths, preserve `legacy` evaluation, and expose checked `--resume`.
5. [x] Verification: focused unit tests, legacy compatibility, 300-turn bounded
   simulation, compaction failure/torn-record cases, then a live multi-turn run.

### Verification gates

- [x] Stable prefix and exact thinking-chunk replay across turns.
- [x] Repairs append messages in order without duplicate or phantom responses;
  dropped retry flags were fixed and covered by regression tests.
- [x] Current fog-gated state overrides historical text and old refs cannot run.
- [x] Conversation remains below the trigger; compaction leaves paired messages
  and a bounded exact tail.
- [x] Compaction/state writes are checksummed and atomic; incompatible records
  fail closed.
- [x] Cache key is stable within an epoch and cached-token usage is visible in
  synthetic/live transport characterization.
- [x] Legacy prompt bytes, schemas, and evaluation fixtures remain unchanged.
- [x] Synthetic two-turn reasoning/cache replay and forced compaction succeeded.
- [x] A live Freeciv two-turn run used conversation caching and exact history;
  it reached the bounded controller guard, which has since been fixed offline.
- [x] Live parent/child characterization staged a verified save, restored the
  conversation and player/turn identity, and completed a child model turn.
- [x] Fresh and resumed live launches applied the configured ping timeout and
  verified the typed server-setting readback; the confirmed value/status are in
  runtime metadata (1260 seconds with the default 1200-second acting deadline).
- [ ] Longer live characterization confirms sustained latency. There is no
  continuous heartbeat while model generation blocks the worker, and no live
  call longer than the configured timeout has yet been tested.

### Validation status

The implementation has passed the focused conversation, transport, runner,
compaction, journal, and compatibility tests. A synthetic two-turn run confirmed
full thinking-chunk replay, cache-prefix reuse, and forced compaction. Token
threshold calibration is currently based on the repository's conservative
character estimator and synthetic measurements; it still needs a long-run live
measurement against provider usage.

The live Freeciv characterization completed two model turns and used cached
conversation history. Fresh and resumed launches also verified the typed
Freeciv `pingtimeout` readback and persisted the effective value/status in
runtime metadata. A fresh parent/child run then exported a verified
start-of-turn save, staged it into a distinct child run, restored conversation
and committed memory, verified player/turn identity, and completed one child
turn. Longer sustained-latency validation remains pending; the worker has no
continuous heartbeat during model generation, so this behavior is not yet
proven beyond the configured timeout.

### Risks and boundaries

- Full reasoning history consumes context quickly despite cached input.
- The 500,000 value is an application trigger, not a provider-guaranteed limit.
- Model-authored summaries can omit facts; recent exact tail and uncertainty
  labels mitigate but do not eliminate that risk.
- Provider cache hits and TTL are not guaranteed; measure `cached_tokens`.
- Recovery is fail-closed on checksum, identity, configuration, and path mismatch;
  only a verified child continuation may load a parent checkpoint.
- The configured Freeciv ping timeout is typed, verified by server-setting
  readback, and persisted in runtime metadata. There is no continuous heartbeat
  during model generation, so sustained calls beyond that liveness window remain
  a risk until live-tested.

### Superseded work

`hybrid_v1` and its bounded `context.jsonl` journal remain readable for old
artifacts but are no longer the interactive target. See the superseded design
at [docs/hybrid-history-design.md](docs/hybrid-history-design.md).

## Screenshot-first gameplay capture and competition edits — 2026-10-28

**Type:** architecture / feature
**Status:** drafted; design and plan independently reviewed with corrections; implementation not started
**Design:** [docs/gameplay-capture-design.md](docs/gameplay-capture-design.md)

### Objective

Make a Freeciv run observable and efficiently publishable without requiring
continuous video capture. The live view is a headed, non-playing observer page
showing the actual game plus an injected sidebar containing the full model
output, reasoning trace, action status, and readback. The capture pipeline keeps
immutable, provenance-rich map screenshots and trace artifacts, then produces
an editable still-image story and a deterministic competition highlight of at
most 120 seconds. The user can start the whole stack with one command, with an
  explicit camera-failure overview/manual fallback; observer identity or IPC
  failures remain no-ship.

### Scope and boundaries

- In scope: observer binding, structured event capture, live presentation,
  semantic screenshot capture, camera safety, durable artifacts, editable scene
  lists, still-image compositing, and deterministic FFmpeg export.
- Out of scope for this plan: optical-flow or invented motion, a continuous
  video recorder, optional microclips/audio before the still pipeline is
  validated, and claims of competition compliance beyond the known 120-second
  limit until the competition specification is supplied.
- The full raw trace remains a separate authoritative artifact. A virtualized
  sidebar may render a readable window, but it must not discard or rewrite the
  raw model output. The sidebar is deterministically regenerated from events;
  no captured sidebar bitmap is authoritative.
- Capture consumers are fail-open with respect to gameplay: a slow UI, disk
  writer, screenshot worker, or renderer must not extend the model/action
  deadline or block the runner. Missing or stale evidence is labelled, never
  silently converted into causality.

### Decisions

- Use a dynamic observer port and per-run nonce/IPC handshake. The orchestrator
  owns port allocation, IPC nonce, manifest binding, and Freeciv server
  verification; attach occurs only after all four checks succeed. A stale,
  wrong-run, wrong-port, wrong-nonce, playing observer, or handshake timeout is
  rejected fail-closed.
- Do not overload the existing model `--stream` flag. Add separate `--capture`
  and `--capture-dir` flags; capture mode requires `--stream`, which is also
  the phase-0 fixture source for model chunks and receipts.
- Keep immutable source events and separate delivery receipts. Every event has
  source identity, source sequence, capture sequence, clock information, and
  stable hashes; gaps, drops, and reconciliation state are explicit.
- Preserve exact action semantics: intent, dispatch outcome, readback result,
  repair, and visual evidence are separate fields. An action is not shown as
  executed merely because the model requested it or a frame changed.
- Treat screenshot ordering as a causal partial order. `capture_seq` is only a
  deterministic tie-breaker; unrelated domains are marked unordered or
  approximate unless a causal edge or watermark exists.
- Store original PNGs and append-only image records with atomic temp-write,
  fsync, rename, and directory fsync. Never overwrite an existing artifact;
  recover only a verified valid prefix after interruption.
- Keep the frozen transcript/action schemas unchanged. Capture envelopes refer
  to ledger sequence numbers; an `ActionResult` joins the `MetricEventRecord`
  intent only when the validated `dispatch-result/v1` detail envelope inside
  `ActionResultRecord.detail` has a matching `intent_seq`. Pre-delivery
  rejects have no intent, and repairs remain distinct events.
- The narrative is suggested by metadata but manually editable. The final EDL
  names every source image, trace range, event/provenance ID, and integer hold
  duration; no scene may invent motion or causality.

### Dependency-ordered implementation

0. **Capability spike and no-ship gate** (observer/runner owner)

   - [ ] Produce a standalone spike report (before any run manifest exists)
     proving orchestrator-owned port allocation is feasible, then hash that
     report into the later manifest. Bind the server-visible nonce to the
     parent/child IPC process ID, run/capture nonce, allocated port/session
     identity, checkpoint, expected turn, and manifest hash before attachment;
     fail closed on any mismatch or timeout. IPC alone is never trusted.
   - [ ] Add a disposable headed observer probe with connection generation and
     explicit non-playing validation. The current adapter's post-reset port
     selection is a feasibility risk; if it cannot be made deterministic and
     verified, the spike fails and capture does not proceed.
   - [ ] Keep the model `--stream` contract unchanged; add separate
     `--capture` and `--capture-dir` flags and require `--stream` in capture
     mode. The phase-0 streamed fixture is spike-local, presentation-only, and
     may use fake receipts; production `--capture` begins only after the
     gateway schema exists and must not treat the fixture as production data.
   - [ ] Verify camera pan/zoom safety, map-layer clipping, and representative
     move/founding/combat/diplomacy visibility without changing game state.
   - [ ] Measure observer freshness and demonstrate a representative event is
     visible in both the live page and a map-only clipped capture.
   - [ ] Instrument negative pan/zoom/reconnect/screenshot tests and prove they
     emit no game commands while the observer remains non-playing. If camera
     safety fails, permit only a labeled overview/manual path requiring human
     selection and approval. Treat clipping failure separately: use a labeled
     full-composite fallback, never unsafe attachment, and do not claim
     production-ready highlights.
   - [ ] Record capability results and alternatives in the spike report; an
     identity or IPC failure is no-ship, not a degraded capture. Only camera
     failure may use the labeled overview/manual fallback requiring human
     selection and approval; clipping has a separate labeled full-composite
     fallback.

1. **Schemas, session manifest, and event gateway** (runner/schema owner)

   - [ ] Define a separate, versioned capture envelope for `capture_id`,
     `source_instance`, `source_kind`, `source_seq`, and `capture_seq` allocated
     by one locked coordinator. Primary IDs are idempotent on duplicate ingest;
     immutable pre-bind identities are never rewritten. Add a separate later
     `run_binding` event, process-instance UUID, clock domain, source monotonic
     time, segment, lineage boundary, turn/phase, parent/causal IDs, and
     payload hashes; do not physically sort across clock domains.
   - [ ] Add a manifest binding configuration, validated `run_id`,
     server-visible nonce and session identity, observer generation, checkpoint
     evidence and IDs, artifact roots, schema versions, and
     clean/interrupted/finalized state. The separate `run_binding` event aliases
     these validated values without rewriting source event IDs.
   - [ ] Implement structured gateway inputs for model trace, runner/action
     ledger, Freeciv readback, observer probes, and screenshots; stdout remains
     diagnostic only. Preserve the transcript schema: intent is the existing
     `MetricEventRecord`, while result correlation is the canonical JSON in
     `ActionResultRecord.detail` for `dispatch-result/v1`. Use a validated
     parser; missing, malformed, or legacy records produce no join, and a
     pre-delivery rejection has no synthesized intent.
   - [ ] Separate lossy live UI and screenshot queues from the low-volume
     provenance journal and high-volume trace spool. Define bounded backpressure,
     fail-open behavior, gap markers, valid-prefix recovery, and reconciliation.
     Receipts use the canonical statuses `offered`, `enqueued`, `accepted`,
     `flushed`, `fsynced`, `dropped`, and `reconciled`; every legal transition
     is an append-only receipt. Trace records carry per-call chunk
     sequence, lane first/last chunk sequence, decoded UTF-8 payload
     hash/length, and durable spool valid-prefix offset/hash. Store provenance
     as separate decoded UTF-8 byte offsets and original escaped-JSONL byte
     offsets, each with checksum and encoding. Final trace states are exactly
     `complete`, `partial`, `gap-reconciled`, or `unknown`. Test hard kills and
     never claim losslessness after every failure mode.
   - [ ] Persist indexes atomically and make crash recovery idempotent. Preserve
     source artifacts and reject duplicate IDs or attempted overwrite.
   - [ ] At finalization, copy exact valid prefixes of transcript/API logs,
     model trace, spool, and journal into capture-owned immutable files (never
     hardlinks). Record source path, source size, copied bytes, hash, encoding,
     schema, and partial-tail status; EDL references these immutable copies.
   - [ ] Reference existing `ActionResult` outcome/readback enums
     (`executed`, `rejected`, `aborted_group`; `verified`, `contradicted`,
     `inconclusive`, `none`) and prove capture work does not consume gameplay
     latency budget.

2. **Live presenter and user setup** (presenter/UI owner)

   - [ ] Inject a readable, virtualized sidebar into the headed observer page,
     deterministically derived from a normalized-event watermark using pinned
     renderer, fonts, and layout: full-output access, tail/follow, search,
     copy/export, lane/status badges,
     and explicit `Thinking`, `Lookup`, `Executing`, `Readback`, and
     `Unknown/Unconfirmed` labels.
   - [ ] Apply text safety (bounded DOM insertion, escaping, truncation only in
     view, preserved raw bytes), stop/resume controls, and observer validation
     before accepting events. Pause affects display/capture only; it must never
     silently pause or alter the game.
   - [ ] Add one-command README/CLI setup that provisions the game, starts the
     runner, opens the bound observer, and creates a unique capture directory;
     provide only the camera-failure overview/manual fallback with a clear
     degraded label, and document the separate clipping fallback.
   - [ ] Make live gaps, stale observer state, backpressure, and paused capture
     visible without asserting action success.

3. **Semantic screenshot worker** (capture/camera owner)

   - [ ] Implement semantic triggers for founding, growth, research,
     diplomacy, combat, rejection, repair, production, game-over, operator,
     turn/decision start, and action results. Add bounded post-result bursts at
     0, 0.5, 1.5, and 3 seconds with coalescing; never use unbounded polling.
   - [ ] Add the read-only camera director only after the spike passes. Each
     request records target, map layer, camera state, observer generation,
     expected turn, freshness, expiry, and actual probe result.
   - [ ] Write immutable PNGs plus append-only `images.jsonl` records (not
     editorial scene indexes) containing image ID, trigger IDs, request and
     completion `capture_seq`, causal watermark, request/start/completion clock
     timestamps and clock-domain IDs, observer connection/generation/
     freshness/staleness, screenshot temporal relation, visual evidence,
     readback relation, camera and fog state, dimensions, and PNG SHA-256.
     Maintain valid-prefix
     counters/hash, fsync indexes, and orphan recovery; `scene_index` is added
     only by the editorial EDL.
   - [ ] Set `post_result_frame` from the temporal event relationship only.
     Observer freshness, readback, and visual evidence are orthogonal metadata
     and remain independently labelled; none may relabel the temporal relation
     or invent confirmation.
   - [ ] Exercise queue overflow, disk-full, cancellation, stale observer,
     interrupted writes, and orphan recovery. Run storage/CPU soak before a
     long game.

4. **Editable still story and highlight export** (editor/render owner)

   - [ ] Define a versioned scene-list/EDL with unique editorial `scene_index`,
     manual ordering, causal relation, image and trace references, exact UTF-8
     quote ranges, title/end-card metadata, integer hold frames, and frame
     quantization. Each scene carries or validates source image temporal
     relation, observer connection freshness, visual evidence, camera/fog state,
     spectator disclosure, and `needs_review`/uncertainty labels; the renderer
     must display those labels honestly. Keep narrative suggestions separate
     from editor decisions.
   - [ ] Decode and composite each immutable map PNG with the pinned sidebar
     renderer at its normalized event watermark; do not use a captured sidebar
     bitmap. Fail closed when required provenance or decoded pixels are missing.
     Produce both required deliverables: a reviewed story with user-chosen
     frame count and a separate competition highlight; the story is not
     constrained to 3,600 frames or 120 seconds. The suggested 10–20 minute
     story duration is a non-binding initial target.
   - [ ] Add a pinned 1920x1080/30 fps FFmpeg profile by default, with an
     approved 720p/30 fallback recorded in the manifest only after benchmark.
     For the highlight only, include transitions and title/end cards in a
     maximum 3,600 frames and actual duration no greater than 120.000 seconds.
     Audio remains out of scope; if later enabled, validate the longest decoded
     audio/video duration. Validate the selected profile's dimensions and
     decoded frame count/duration, with no unintended overwrites.

5. **Verification, smoke, and review** (verification owner)

   - [ ] Build synthetic end-to-end fixtures covering schema validation,
     causal ordering, receipts, cache/trace reconciliation, `images.jsonl`,
     EDL edits, and deterministic rendering.
   - [ ] Run the one-command launch smoke from a clean environment, verifying
     provisioning, dynamic binding, manifest/server checks, artifact
     permissions, and fallback behavior before the real preview smoke.
   - [ ] After phases 0–2, run a lightweight clean-environment gateway/
     presenter smoke without screenshots: verify attach, live sidebar,
     action/rejection labels, pause/resume display behavior, permissions, and
     zero extra provider requests or request-shape changes.
   - [ ] After phase 3, run the actual 3–5-turn real smoke (preview only, not a
     10–20 minute game) with screenshots and final `images.jsonl`/valid-prefix
     indexes. It must capture at least one observed battle or rejection through
     the screenshot pipeline, then receive human checks that unconfirmed
     visuals are not represented as proof. If neither event occurs in the first
     3–5 turns, extend the controlled smoke or run a separate scenario; do not
     mark the smoke passed by pretending the event occurred.
   - [ ] Fault-inject queue/full stream, cancellation, process crash, forked
     run, stale observer, partial trace, and contradictory readback; prove no
     fabricated action or causal label is emitted.
   - [ ] Later run full-game load/disk/CPU tests and human editorial review for
     battle readability, sidebar legibility, and storytelling. A human-reviewed
     story is required; 10–20 minutes is a non-binding initial target. The
     separate <=120-second highlight is also required before promotion. Final
     exports must pass decode/provenance checks; unknown competition
     requirements remain an explicit discovery item.

### Provisional measurable targets and gates

The following are benchmark hypotheses, not claims about current behavior:

- Normal synthetic event loss: zero missing journal events; injected overload
  must produce an explicit gap/drop receipt and reconciled valid prefix.
- Presenter freshness target: p95 event-to-sidebar display under 500 ms during
  a bounded smoke; screenshot completion target p95 under 3 seconds where the
  observer supports it. Record methodology, hardware, queue depth, and sample
  count rather than relying on wall-clock anecdotes.
- Capture overhead target: measure p95 capture enqueue latency and paired
  capture-on/off critical-path and per-turn wall time. Assert the acting
  deadline and API request shape are unchanged; record the benchmark method and
  block promotion on unexplained regressions rather than using an unmeasurable
  deadline-extension claim.
- Storage target: no artifact overwrite, atomic index validity after clean and
  interrupted shutdown, and a documented disk budget/alert threshold before
  long runs.
- Record image bytes/turn, trace growth/turn, queue depth, and capture-off vs
  capture-on provider request counts; the latter must be zero extra requests
  and the request shape must be unchanged.
- Export gate: both required deliverables decode at their selected manifest
  dimensions and frame counts, every displayed scene resolves to an immutable
  source/provenance record, and only the competition highlight is constrained
  to <=3,600 frames and <=120.000 seconds.

### Owner boundaries, risks, and sequencing

- The runner/schema owner owns the manifest, ledger, gateway, receipts, and
  action truth; the observer/presenter owner owns only display and probes; the
  capture owner owns PNG acquisition/camera requests; the editor/render owner
  owns EDL and FFmpeg; the verification owner owns fixtures, fault injection,
  soak measurements, and final gates. No consumer may mutate authoritative
  runner state.
- Capture artifacts are local by default with restrictive file permissions;
  the presenter visibly labels spectator exposure and does not silently retain
  or delete raw reasoning. Retention/deletion is an explicit user operation.
- Primary risks are unsupported observer/camera behavior, stale or incomplete
  traces, unbounded reasoning/storage growth, map clipping that hides events,
  and unknown competition rules. Mitigations are the spike/no-ship gate,
  explicit degraded labels and reconciliation, virtualized UI plus budgets,
  human visibility review, and a specification discovery task.
- Default sequencing is 0 → 1 → 2 → lightweight presenter smoke → 3 → actual
  3–5-turn screenshot smoke → 4 → full-game soak → human story sign-off →
  competition export, with promotion states `spike → prototype → smoke → soak
  → human story → competition`. A failed phase blocks its dependent phase
  except for documented fallback experiments; promotion requires both reviewed
  story and highlight, and competition promotion is blocked until rules are
  known.

### Review gates and current status

The gameplay-capture design and this implementation plan have each undergone
independent deep review; the corrections above are recorded as design/plan
review outcomes. This plan is drafted and intentionally unimplemented. No
implementation, verification, or competition compliance is asserted by this
section; future changes must preserve the dependency, schema, evidence, and
failure gates above.
