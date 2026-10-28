# Gameplay Capture View and Editorial Recording Design (Draft)

Status: screenshot-first design draft. This is a design artifact, not an implementation or approval gate.

## Goal and recommendation

Use one verified, headed Playwright read-only visible presenter observer as the default live surface: the actual Freeciv map occupies most of the page and an injected sidebar, fed by a local event channel, shows turn/status, full streamed model lanes, intended actions, execution/readback results, and engine events. Capture retained map-only PNGs from the observer canvas/image surface (or explicitly choose a full-composite PNG); the live page is never a refreshed-still slideshow. An offline scene list combines map stills with normalized sidebar content for a compressed story and a separate competition highlight of at most 120 seconds. A latest-still fallback is labeled `not_live`.

Screenshot-first avoids demanding a continuous full-game recording, makes key decisions legible, and removes video-PTS synchronization as an MVP prerequisite. It does not prove chronology or causality by itself: every image needs event correlation, observer-state metadata, stale-age reporting, and honest pre/post labels. Stills do not capture motion; slideshow holds, hard cuts, and simple pans must never imply unseen continuous action. Optional 3–8 second video microclips are non-MVP and require explicit user choice plus a prototype.

## Scope and non-goals

In scope: one dynamically attached verified observer; live map plus sidebar with virtualized, searchable, selectable, followable verbatim reasoning/answer/tool lanes; non-blocking semantic screenshot worker and camera director; raw traces, transcript/API/ledger references, immutable PNGs, event journal, and manifest (traces are complete only after clean reconciliation, otherwise classified valid prefixes/partials); editable deterministic scene list; FFmpeg-generated story (initially 10–20 minutes) and independent <=120.000-second highlight.

Not in MVP: mandatory full-length video, browser screencast APIs, video markers/PTS calibration, optical flow or synthetic sprite movement, player commands, narration/music/codec assumptions, deterministic Freeciv replay, or automatic final-story selection.

## Existing constraints and topology

The runner has structured lifecycle boundaries, model traces, transcript/API records, and a separate checksummed ledger, but no unified live bus. Progress text is not authoritative and must not be parsed. Installed Playwright is 1.49.0. A new structured capture sink at runner/model/ledger boundaries and a runner `--stream` mode are prerequisites; without them, capture refuses to claim a live full-stream view. The orchestrator owns runner launch with a capture-ID nonce, unique assigned port, and expected manifest checkpoint/turn. A parent-child IPC handshake binds `capture_id`, nonce, child process ID, run ID, allocated port, server process/session identity, manifest hash/checkpoint, and expected turn. The server must expose the nonce/session identity in its observer-visible handshake; an IPC acknowledgment alone cannot prove that the browser reached the intended server. The orchestrator compares all values with its allocation and run manifest before attachment, then the nonplaying observer probe must agree. A spike must verify this protocol and its actual server-visible nonce; mismatch, timeout, unavailable identity, or unverifiable external attach fails closed. Existing replay is transcript/trace HTML, not a deterministic simulation.

Create `artifacts/captures/<capture_id>/` before `session_ready`. Attach one visible headed presenter observer only after the nonce-bound run identity, server/session, assigned port, checkpoint, read-only/non-playing mode, loaded map, turn/year/phase, and spectator fog disclosure are verified. The authoritative PNG is a clipped DOM game viewport excluding the sidebar; the normalized event watermark and sidebar are persisted separately and derived during output. If clipping cannot be proven, mark the image incomplete and use an explicitly labeled full-composite fallback. A second hidden recorder observer is optional only if independently proven useful, not required for screenshot MVP. If used, verify identity/fog/load parity and preserve one-observer fallback. The viewer is labeled `spectator observer`; it may see more than the model. No sidebar, camera command, or screenshot worker may dispatch a game order.

## Normalized event gateway

The runner-owned fanout consumes structured lifecycle, model-client, ledger, observer, and executor records. Every event has immutable primary ID `(capture_id, source_instance_id, source_kind, source_seq)` before a run is known. A single capture coordinator owns one locked incrementing `capture_seq`, assigned on receipt, and every event/image stores it plus its source causal sequence. Each source also has `process_instance_id`, `clock_domain_id`, `segment_id`, `source_monotonic_ns`, and lineage/boundary events. Runner/ledger/action-intent sequences are authoritative causal order; receipt order is only an observation. Different clock domains are never globally sorted or compared. Cross-source ties use source order, `capture_seq`, and explicit boundary/gap records, never a claim that receipt order establishes physical causality. A later immutable `run_binding` aliases a validated run ID and never rewrites event IDs. Duplicate IDs are idempotent.

Structured event map:

- `session_ready` is provisional until binding validation; `turn_start` is the semantic turn boundary.
- `ruleset_lookup_started/finished` are explicit; progress text is display-only.
- Physical model calls carry provider `call_id`, attempt, lane, and `chunk_seq`; completed traces preserve that call ID.
- `turn_decision` is model intent. Action intent is a durable ledger `MetricEventRecord` with `intent_seq`, never progress. A frozen `ActionResultRecord` joins an intent only when dispatch-result/v1 contains that intent; pre-dispatch rejection or abort has no intent join. It carries exactly one outcome (`executed`, `rejected`, or `aborted_group`) and one readback status (`verified`, `contradicted`, `inconclusive`, or `none`). Repair is a separate event; the UI may derive labels but cannot alter this record.
- `engine_event` covers automatic changes. `game_over` is authoritative game state; run outcome and capture lifecycle are separate.
- `capture_received`, `observer_ready`, `camera_command/result`, `screenshot_requested/completed/failed`, `live_gap`, `reconciled`, compaction, recovery, and errors have their own provenance. Screenshot records include request sequence, completion sequence, and the causal watermark observed at request/completion.

Delivery is a separate receipt state (`offered`, `enqueued`, `accepted`, `flushed`, `fsynced`, `dropped`, `reconciled`), not a mutation of immutable events. Four paths remain separate: semantic control events, a high-volume trace spool, a lossy live-UI stream, and a coalescing screenshot queue. The spool is fail-open, not a promise of losslessness: accepted/durable bytes and their fsync cadence are recorded, while a hard crash may lose data. A bounded live queue may drop telemetry, emits range-based `live_gap`, marks the manifest degraded, and never blocks gameplay. Finalization reconciles accepted/durable trace text against model completion; restored text may not preserve original timing or chunk boundaries.

## Screenshot strategy and causality

The worker coalesces routine turns and captures milestones: turn decisions, city founding/growth, research, diplomacy, combat, rejection/repair, production, game-over, and user-selected events. A `baseline_earlier_same_turn` image may be captured during thinking. An optional strong pre-state requires a `request_id`, `image_id`, `observer_connection_id` and generation, expected turn, PNG hash, runner receipt sequence/time, expiry, and matching `intent_seq`; the runner must receive the screenshot ACK before durable dispatch intent in the same clock domain. This is metadata, not a blocking dispatch fence: if the ACK is late, absent, expired, or mismatched, label baseline unavailable/earlier. Trigger timing alone is never evidence of a before image. The image stores its request/completion `capture_seq` and causal watermark; receipt order is not physical causality.

After `action_result`, request a bounded burst at approximately 0, 0.5, 1.5, and 3 seconds, probing observer turn/year/phase and camera. `observer_fresh` is true only when browser instrumentation provides a server-packet counter, connection generation, or proven heartbeat; unchanged DOM or turn/year is not freshness evidence. `post_result_frame` means only that it was captured after the result request; neither proves the action. `visual_change_observed` requires a domain-specific probe (for example, a confirmed city/unit/tile state); otherwise use `visual_consequence_unconfirmed`. Choose the first image whose probe confirms the transition; otherwise label pixels `unconfirmed` and do not present them as authoritative outcome. Delayed/disconnected observers produce stale/uncertain records, not fabricated pairs. Every timestamp carries `clock_domain_id`. Each image records actual capture start/end, trigger IDs, request/completion timestamps, observer-update confirmation, turn/year/phase, camera, fog, stale age, dimensions, and SHA-256. If freshness instrumentation is unavailable, freshness is `unknown`, distinct from visual consequence.

The bounded queue coalesces superseded routine requests but never silently overwrites evidence. Failed, empty, stale, or dropped requests receive image status and a gap record. Live UI displays latest verified image, capture age, and causality. A screenshot is evidence of what the observer rendered, not proof the engine accepted an action.

## Read-only camera director

Camera control is a pre-MVP spike, not an assumed capability. For relevant action/affected-tile coordinates from the ledger (never progress text), prove a read-only observer pan/zoom API and record `camera_command`/`camera_result`, world coordinates, and confirmation. The negative test must prove no game command was emitted, and reconnect/camera use must reverify observer identity/read-only state. If the spike fails, use operator/manual overview and label coverage missing; never claim an unverified camera moved.

## Exact trace and screenshot payload provenance

At finalization, copy exact valid prefixes of transcript, model trace, API, and capture-journal sources into capture-owned immutable files; do not hardlink, overwrite, or claim complete bytes after hard kill. Store source path, byte count, size, hash, encoding/schema version, and partial-tail status. Each completed physical model call emits a completion record containing first/last `chunk_seq` per lane, full decoded UTF-8 payload hash and length, and the durable spool valid-prefix byte offset/hash. Reconcile it against the source trace hash and emit exactly `complete`, `partial`, `gap-reconciled`, or `unknown`; missing chunks never become a lossless claim. A trace quote records source hash, lane, physical `call_id`, and UTF-8 decoded byte offsets plus original JSONL byte offsets; JSON escapes, decoded text, and newline delimiters are distinct. Editorial summaries are labeled summaries, never source quotes.

PNG files are immutable and atomic: temporary write, flush/fsync, rename, directory fsync, then append one record to append-only `images.jsonl`. Keep the manifest small as an atomic valid-prefix index; recover orphan PNGs by hash and record them rather than scanning/re-writing a quadratic manifest. Image IDs include capture/source/process/sequence lineage. Manifest transitions use temporary write, flush/fsync, atomic replace, and directory fsync. A hard crash can still lose final in-memory records; state is `interrupted`, `degraded`, or `incomplete`, never falsely `complete`.

## Manifest and image schema

The small manifest references append-only `images.jsonl`; it does not embed image records:

```json
{"schema":"capture-manifest-screenshot-v1","capture_id":"cap-...","run_id":null,
 "manifest_state":"recording","completeness":"incomplete",
 "clock_domains":[{"clock_domain_id":"clock-...","process_instance_id":"proc-..."}],
 "observer":{"url":"dynamic","readonly_verified":true,"spectator_label":true,"live_surface":"headed_playwright"},
 "images_jsonl":"images.jsonl","sources":{},"gaps":[],"scene_list":null}
```

An `images.jsonl` record is independently append-only and hashed:

```json
{"image_id":"img-...","path":"images/img-....png","sha256":"...",
 "trigger_event_ids":["..."],"request_capture_seq":0,"completion_capture_seq":0,
 "causal_watermark":{"source_kind":"ledger","source_seq":0},"capture_start_ns":0,"capture_end_ns":0,
 "observer_update_confirmed_ns":null,"clock_domain_id":"clock-...","stale_age_ms":0,
 "turn":12,"year":4000,"phase":"movement","camera":{"world":[4,8]},
 "temporal_relation":"post_result_frame","connection_freshness":"fresh",
 "visual_evidence":"consequence_unconfirmed","fog":"spectator_observer",
 "viewport":"map_only_clipped","status":"complete"}
```

Image records use independent fields: `temporal_relation` is `pre_ack_confirmed`, `baseline_earlier_same_turn`, `post_result_frame`, or `unknown`; `connection_freshness` is `fresh`, `stale`, or `unknown`; and `visual_evidence` is `change_observed`, `consequence_unconfirmed`, or `unavailable`. Turn/year/phase alone never proves an action. Overall completeness fails on gaps, unreconciled tails, queue loss, or crash loss.

## Live and offline outputs

The live presenter is the actual headed Playwright observer page with injected sidebar DOM and a local event channel, not a refreshed still. Sidebar chips include `thinking`, `lookup`, `intent`, `executing`, `rejected`, `engine`, `uncertain`, and `live gap`; controls provide search, selection, scroll/follow, and full-stream lanes. Full trace remains available when the UI shows a tail. A latest-still fallback is explicitly labeled `not_live` and cannot satisfy live acceptance.

The versioned deterministic scene list uses an integer output-frame timebase and contains unique integer `scene_index`, `scene_id`, image IDs, hold frame count, optional hard cut/transparent transition frame count, source event IDs, temporal/connection/visual evidence fields, camera/fog disclosure, exact trace quote ranges, and `needs_review`. Validation rejects duplicate or missing `scene_index` values. Scene order is a causal partial order: source/ledger order is authoritative only within its clock domain, while cross-domain order requires explicit causal edges, watermarks, or lineage boundaries. `capture_seq` and image ID provide deterministic non-semantic tie-breaks only; they must never be presented as physical chronology. Unordered or approximate scenes are visibly marked and cannot claim a chronological transition. Frame quantization is recorded; title/end-card frames count toward duration. No optical flow or invented motion is allowed.

FFmpeg renders both outputs from still-image scenes using the deterministic default profile 1920×1080, constant 30 fps, and integer 1/30-second frame timebase. Holds, hard cuts, transparent transitions, title frames, and end-card frames are exact integer counts; the highlight is capped at 3,600 frames including title/end frames. If a rendering-performance spike proves 1080p unsuitable, a documented 1280×720 fallback may be selected before capture. Validate decoded frame count and `ffprobe` metadata, and require both container duration and longest audio/video stream duration <=120.000 seconds. Audio, subtitles, and other media rules remain configurable. Pinned tools provide semantic, not bit-identical, reproducibility.

## Optional video, not MVP prerequisite

Only after screenshot acceptance may the user authorize 3–8 second microclips. A rolling recorder or continuous capture must be designed and tested first; Playwright cannot retrospectively start a recording on demand. Use ordinary `recordVideo` with capability detection only for future forward clips. No screencast API, full-game video, video segmentation, or PTS marker is required for the screenshot MVP. A microclip carries source segment, observer identity, start/end, and uncertainty; if timing is not calibrated it remains illustrative, never exact action evidence. Stills and event provenance remain primary.

## Prototype, tests, and acceptance

Prototype against existing logs first, labeling missing screenshot provenance. Before live acceptance, the structured `--stream` sinks, nonce-bound orchestrator handshake, and headed Playwright sidebar must exist. Then run a 3–5-turn live test including a battle or rejection and human review. Acceptance requires dynamic readiness, verified read-only observer, actual live map+sidebar (not still fallback), validation that map clipping captures all game layers, representative events visibly appear on screen, full streamed lanes, immutable hashed images and metadata, frozen ActionResult joins, no gameplay commands, explicit gaps/uncertainty, clean finalization, and both story and <=120-second decoded renders. If camera automation fails, manual selection plus human approval is mandatory before production-ready status. A screenshot-first run does not require full-length video.

Test readiness timeout, observer disconnect/delayed updates, camera API failure/manual fallback, pre-state unavailable, action-result burst timing, stale/empty/failed PNG, slow consumer/queue gaps, duplicate IDs, process restart/fork, disk error, partial JSONL tails, hard kill, manifest/orphan recovery, trace offsets, spectator disclosure, and game-over versus run outcome. Measure p95 event enqueue, live sidebar latency, screenshot request-to-completion, bounded queue/drop counters, PNG size/turn/stale-age, observer camera visibility, same-domain pre-ACK confirmation, recovery correctness, and long-run storage/throughput soak. Provisional hypotheses (to be verified by a spike, not claimed as passes) are p95 enqueue under 5 ms, no gameplay callback blocking, bounded screenshot latency under 3 seconds in a healthy observer, and explicit zero silent drops. Human review must verify battle scenes do not claim unconfirmed pixels as outcome. A 3–5-turn preview proves only the preview, not a 10–20-minute story or long-game capacity.

## Trade-offs and references

Still-first is legible and storage-efficient but lacks motion and may show stale state; short clips add drama but increase timing/storage risk. One verified presenter is the default; a second observer adds identity/fog/load failure modes. Sidebar versus bottom bar, story length, and competition media rules remain user choices.

Official references: [Playwright video API](https://playwright.dev/docs/api/class-video), [Playwright videos guide](https://playwright.dev/docs/videos), [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html), and [FFmpeg FAQ](https://ffmpeg.org/faq.html).
