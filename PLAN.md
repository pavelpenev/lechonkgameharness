# Project Plan: Le Chonk Freeciv Harness

## Append-only conversation history and compaction — 2026-10-27

**Type:** architecture / feature
**Status:** implementation complete; superseded by the 2026-10-29 release evidence below
  (mechanically completed to absolute turn 100; strategic/control limitations remain)
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
- [Historical limitation] Longer live characterization was not part of the
  release evidence. There is no continuous heartbeat while model generation
  blocks the worker, and no live call longer than the configured timeout was
  established by this section.

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
turn. The older sustained-latency caveat remains a limitation: the worker has
no continuous heartbeat during model generation, so behavior beyond the
configured timeout is not claimed here.

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
**Status:** historical full-architecture plan; implementation not started for that
  scope (the later screenshot/highlight prototype is separately evidenced below)
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

## Comprehensive gameplay-control reliability — 2026-10-29

**Type:** architecture / correctness

**Status:** independently reviewed; planned, not implemented

**Design and evidence:** [control audit](docs/harness-control-audit-2026-10-29.md),
[control design](docs/harness-control-design.md)

This section supersedes the assumption that a structural `PASS` implies good
gameplay. The 51–100 run completed mechanically but exposed only unit actions;
its own participation gate was false. No current game is restarted by this
planning change. Full reasoning traces, prefix caching, 500,000-token
compaction, no local output cap, Le Chonk identity, and existing recovery
lineages remain constraints.

### Deadline-first release scope

The user supplied a rounded 15-hour window on 2026-10-29. Taking the observed
08:27:35 UTC clock as a planning anchor gives an approximate 23:27 UTC cutoff.
This is a planning estimate, not a promise that the game will finish on
schedule.

The first two hours (target 10:27 UTC) are the containment window: fix the
active-player technology readback and regression, run focused offline checks,
and run a 3–5-turn captured smoke (a 10-turn smoke is optional only if its
measured calls and capture synchronization fit the window). Use the existing
versioned `gameplay_v6`/`candidate_surface_v3` configuration; do not introduce a
new profile or resolver during the deadline cut. The smoke and its external
preflight gate must verify that all own settlers, cities, legal research, and
required production choices are reachable, that critical actors have visible
state, and that identity, capture synchronization, degraded-run reporting, and
recovery evidence are present. Run a synthetic mature-empire stress fixture as
well, because a fresh short game may not reproduce late-game starvation. Any
missing essential coverage fails the gate visibly.

If that smoke is healthy, freeze the exact launcher/runtime configuration and
start the main recording by T+2 hours: Le Chonk, two AI opponents, default
tiles, conversation caching, the 500,000-token compaction policy, no local
output-token cap, and matching manifest/capture metadata. The next nine hours
(approximately through 19:27 UTC) are for recorded gameplay, checkpoint health
audits, and preparing the concise story in parallel; completion of a full game
is not guaranteed. Measure p50/p95 turn time, control coverage, capture sync,
degraded status, and recovery evidence during the continuation. The prior
roughly 45-second mean is not a safe assumption after restoring strategic
controls. Set the latest restart cutoff from the measured distribution and
remaining-turn estimate.

At about T+11 hours (approximately 19:27 UTC), freeze the selected footage and
reserve four hours for the two-minute highlight, concise edited story,
rendering, review, and one retry. Target a final export by about T+14 hours
(approximately 22:27 UTC), leaving one hour for validation and delivery. A
partial game must be labelled as such; it must not be presented as a complete
victory or merged with footage from another run without clear provenance.

The full health-system rewrite, new catalog tools, the multi-round controller,
and broad capability expansion remain backlog work. If the preflight gate fails,
do not launch an untrustworthy deadline run; use the existing
`lechonk-42-story` highlight or a local fix/fallback with clear provenance.
Missing government, rates, transport, diplomacy, or victory controls remain
explicit readiness limits. No old checkpoint changes policy silently: use a
fresh run or an explicitly validated fork.

The narrowed deadline cut has advisor review approval as a prioritization plan;
that is not code or capability validation. The full analysis above remains the
longer-term plan.

### Plan risks and gates

Risks are catalog bloat and latency, read/mutation races, uncertain checkpoint
recovery, prompt/tool compatibility, and an unverified victory path. The top
level result is `NOT_READY` when a deterministic mechanical or control-access
gate fails; model outcome warnings remain distinct. Any automatic continuation
past a checkpoint requires the recorded gates and review. This plan adds no
implementation or claim of completion; each package needs its own verification
evidence before promotion.

### Dependency-ordered execution plan

**Status:** independent design and execution-plan reviews passed after corrections; implementation remains planned. The audit and design
now incorporate the confirmed active-player readback reproduction, the legacy
offered-count limitation, tool-call reasoning replay risk, optional discovery,
state-barrier rules, and explicit incomplete-phase semantics.

| Package | Owner / targets | Depends on | Deliverable and exit |
|---|---|---|---|
| P0 evidence, health, resolver (M) | runner, metrics, schema, CLI; `runner.py`, `schemas.py`, `wp2_model_run.py`, `execute.py`, launcher/resume resolver | baseline tests | Active-player verifier; raw/catalog/offered/omitted coverage; one resolver gives the same effective identity, horizon, and sampling across launcher/CLI/recovery. T01/T02 pass; old profiles readable. |
| P1 static containment (M) | candidate/prompt; admission and observation | P0 | Mature 100+ unit/20+ city fixture exposes settlers, cities, research, contact, and required production alternatives or fails visibly. Declared-scale smoke only. |
| P2 capability spikes (M) | adapter/session and model transport | P0 | Immutable state barrier; interrupted refresh returns `mutation_outcome_unknown`; Mistral direct/no-query/chained/malformed/timeout/strict-final/thinking-tools probes; mid-phase save result. |
| P3 catalog and queries (L) | adapter/catalog and recovery identity | P2 | Revisioned fog-safe dashboard, actor-indexed selectors, bounded pages, hashes, stale/owner rejection, complete supported-action enumeration. T03/T04 pass. |
| P4 rounds and durability (L) | runner, turn, conversation, recovery | P3, recovery spike | Per-round leases and fresh validation, one mutating action per actor initially, explicit finish, monotonic budgets, partial truth, WAL/commit marker, phase-boundary fork. T05–T09 pass. |
| P5a core/government/rates/diplomacy (L) | adapter/vendor integration | P3 | Settlement, production, research, combat, government, rates, selected diplomacy each have registry, legality, mapping, readback, stale-ref tests, and scripted real-adapter smoke. |
| P5b specialists/upgrades/trade (L) | adapter/vendor integration | P5a | Each selected capability has the same registry, fog, legality, mapping, readback, and test exit; unsupported remains labelled. |
| P5c naval/transport (L) | adapter/vendor integration | P5b | Naval and transport paths pass the same exact capability gate before island scenarios. |
| P5d selected victory (L) | adapter/vendor integration | P5c and chosen ruleset | Space launch or other enabled victory path has handler, available-action exposure, readback, and terminal test; no victory claim from source docs alone. |
| P6 validated rollout and UX (M) | CLI/docs/capture/manifest/health integration | P4, P5a–P5d, pre-promotion smoke fixture | Integrates one resolver and independent labels; default promotion remains gated by P7. Old lineages preserved; no resolver creation here. |
| P7 promotion validation (L) | verification, run operations, capture | P4, P5, P6 integration | Fresh 10-turn and mature-state runs, scripted adapter smoke, 2-AI scenario, 50→100→200/300 soak, chosen-ruleset terminal run. Report p50/p90/p95 per phase and per request, calls, output, cache, bytes/turn. Promotion is this package's exit. |

The new production profile omits `temperature`, `top_p`, and `max_tokens` so
provider defaults are used as requested; its default `reasoning_effort=high` is
an explicit trace policy. The current direct driver makes that setting
configurable via `--reasoning-effort`; existing historical profiles retain
their pinned request shape and are not rewritten. Actual field presence is
recorded.

### Acceptance test matrix

| ID | Fixture / fault | Expected result | Gate |
|---|---|---|---|
| T01 | Reproduce known unit-first starvation with 100+ units, 512-unit stress, 20+ cities | Baseline records the known failure honestly; P3 later requires supported critical actions reachable with no starvation or duplicate pages | P0/P3 |
| T02 | Opponent 1 first; active 2 researches 7 while opponent researches 3 | Readback verifies active player's 7 | P0 |
| T03 | Ordinary dirty snapshot race; separately interrupted potentially mutating refresh | Ordinary inconsistency retries within bound or fails before dispatch; interrupted mutating refresh returns `mutation_outcome_unknown` and stops | P2 |
| T04 | 64-entry/16KiB pages, unequal options, direct last actor | Complete enumeration once; oversized entry typed error | P3 |
| T05 | Mutation between lease actions, reconnect, phase, wrong owner | Relevant precondition change rejects affected remaining refs; unrelated actor change may proceed after fresh validation; reconnect/phase change invalidates entire lease; no rebinding | P4 |
| T06 | Direct, no-query, one-query, chained, malformed, timeout, repair, thinking+tools | Matching results, no orphan exchange, exact raw trace; unresolved replay blocks promotion | P2/P4 |
| T07 | More than 20 units in one phase | Multiple rounds, one explicit finish, monotonic budgets | P4 |
| T08 | Crash at prepare, action ack, commit, boundary save | Truthful partial record; only prior verified save forkable | P4 |
| T09 | 500k mid-phase, timeout/unknown anchor, compaction crash | Safe between-round compaction; exact tail; no output cap or staged exchange compaction | P4 |
| T10 | Government/rates/specialist/upgrade/treaty controls | Registry, legality, mapping, readback, tests; unsupported labelled | P5 |
| T11 | Launcher, CLI, resume, segment 50→100 | Same profile and horizons/sampling; incompatible lineage fails | P0/P6 |
| T12 | Capture smoke, disk-full, soak, terminal ruleset, same-turn multi-round IDs | Separate evidence/gameplay health; phase/round/request/tool-call/revision IDs join without overwrite; sidebar labels lookup/thinking/acting/current-round and does not advance game; timings per phase and per request; no readiness without victory path/editorial review | P7 |

Original PLAN sections above remain the historical baseline. The capture
prototype now produces verified screenshots and highlights in saved runs, but
the older full capture architecture gates are not all verified; this section
does not mark them complete. The offline baseline is 1,318 tests passed in
46.02 seconds, with no new live game started for this plan. Pre-existing source
and vendor changes remain preserved.

All API, model, ledger, capture, and sidebar events need distinct phase, round,
request, tool-call, and revision IDs. Joins must preserve honest intent,
readback, lookup, thinking, acting, and current-round statuses; same-turn
decisions must never overwrite one another. Timing reports include both own
phase and individual request/decision distributions, not an average that
mistakes API calls for game turns. New fixtures use honest synthetic or newly
instrumented isolated state; missing raw pre-admission data cannot be recreated
from legacy logs.

Validation starts with a fresh 10-turn game and isolated mature-state 10-turn
run, plus a scripted real-adapter settlement/production/research smoke kept
separate from autonomous gameplay. Then run 50 stop/audit → 100 resume →
200/300 soak, followed by a terminal run for the pinned ruleset. Before a
full-game claim, record enabled victory conditions and map requirements and
complete every required path. Do not change actual run or map settings silently.

### Review record

Accepted: optional discovery; proven tools→strict-final baseline; snapshot
barrier rather than a claimed Python lock; actor-indexed selectors; 20 as a
per-round bound; explicit `STOPPED_INCOMPLETE_PHASE`; exact traces/no local
output cap; old profiles and lineages; separate capture evidence quality.

Deferred/rejected: silent checkpoint policy switching; forced city cadence;
automatic finish; replay after uncertain mutation; documentation-only claims
for transport, trade, or spaceship; every action in one prompt; unprobed
tool/strict-schema behavior. Independent re-review remains required at P0,
P3/P4, and P7; no item here is implemented.

### Review record — 2026-10-29

The architecture advisor and independent plan reviewer both returned PASS after
corrections to the snapshot barrier, batch lease, replay probe, compaction
durability, and resolver dependencies. The 1,318-test/46.02-second baseline,
document diff, and local links were verified. No new live game or code change
was made in this review. User approval and live capability probes remain
pending; future phase gates still require their own evidence.

## Release evidence completion — 2026-10-29

The release cut completed the scoped, mechanically verifiable work: the
own-player research/readback fix; the critical-controls admission gate;
`--reasoning-effort` CLI support; resume and conversation compaction; the actual
same-game recording through absolute turn 100 across resumed segments; and the
118-second narrated demo. The latest segment,
`artifacts/runs/lechonk-none-recording-100-20261029/`, covers own phases 51–100
(50 phases) and passes 15 structural checks. Its participation gate is false
and 29 model actions were rejected for `state_fog_error`; this evidence is not
a victory or strategic benchmark.

The comprehensive gameplay-control architecture above remains explicitly
deferred/not implemented. Government, rates, transport, diplomacy, full
capability coverage, and a verified victory path remain backlog items; the
release does not claim that all gameplay issues are fixed. Earlier pending
language in this plan is historical where retained, and is superseded only for
the mechanically completed scope stated here.

The dated pre-review offline baseline was `.venv/bin/pytest -q`: 1,347 passed;
the final full verification is `.venv/bin/pytest -q`: 1,355 passed in 46.77 s.
The unchanged Node presenter self-test passed, and media-reuse validation passed
alongside the approved artifact hash checks. Ruff was unavailable and therefore
was not passed. The review fixes cover persisting incomplete usage anchors,
validating generated audio as actual PCM WAV, and quote-safe concat paths.
Generated test logs and obsolete previews were archived reversibly; the inventory is
`/home/pav/code/lechonkgameharness-archive/cleanup-20261029-final/inventory.json`.
Source capture assets were retained. Generated video/audio artifacts remain
local and ignored; the checkout contains the renderer, provenance-oriented
documentation, and this evidence summary, not generated binaries.
