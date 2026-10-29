# Harness control design — 2026-10-29

**Status:** independently reviewed proposal; implementation and live capability validation not started.

## Objective and preserved constraints

Make every supported legal game action and its fog-safe context discoverable,
regardless of empire size. The model chooses strategy; the harness must not
script expansion or silently substitute choices. Raw model output, including
thinking, remains immutable and observable. Active provider history is
append-only within a cache epoch; explicit compaction starts a new epoch.
There is no local model-output token cap. The 500,000-token compaction trigger
remains, using the trusted usage anchor and conservative suffix accounting.

Compaction may occur only after a completed, readback-verified round with no
pending tool/action exchange; phase boundaries are preferred. Game recovery
checkpoints remain phase-boundary-only until trustworthy mid-phase saves are
proven. These are different boundaries. Existing profiles and recovery
lineages retain their historical behavior.

## Architecture

```text
fog-safe observation + available actions + controller context
    -> consistent immutable state revision
    -> supported legal action catalog + capability/omission registry
    -> compact dashboard and optional indexed, paginated queries
    -> model-selected strict decision
    -> accepted execution lease -> fresh validation -> action/readback
    -> another same-phase round, explicit finish_phase, or recorded stop
```

The source is the player's controller, never omniscient saves or capture
observers. Every offered actor has current compact metadata or an explicit
inspection path. The dashboard records revision, supported legal counts,
summary-delivered counts, urgent actors omitted, and direct query selectors.
Zero available actions and unknown availability are different states.

Introduce a versioned production profile, provisionally
`gameplay_control_v1`, resolved identically by launcher, direct CLI, manifest,
and recovery. Old checkpoints cannot silently change profile. An experimental
migration is an explicit child fork with a new epoch, fresh authoritative
state, and validated lineage; it is not an exact continuation.

## Decision and phase rules

Discovery is optional when the supplied current state and refs suffice. The
only game mutations come from validated model decisions. An execution batch
contains at most 20 actions and initially at most one mutating action per actor;
dependent actions use another round. Explicit `finish_phase` is distinct from
an empty batch. A wait decision must explicitly state whether it ends the phase;
there is no implicit end on an empty menu or lack of consequential options.

Only model-requested phase completion or an actual terminal game event ends a
phase successfully. After every dispatch, check termination before another
action or finish. Budget exhaustion stops with `STOPPED_INCOMPLETE_PHASE`;
it does not send `phase_done`. A model may choose finish after seeing a complete
no-opportunity summary. Physical requests, discovery/execution rounds, schema
repairs, transport retries, and elapsed time have distinct budgets under one
monotonic phase deadline. Cleanup I/O cannot issue extra game commands.

## Control contracts

### Consistent state barrier

Each adapter worker creates one immutable revision tuple in a critical section
and snapshot barrier: fog-safe observation, available actions, unit flags,
action probabilities, phase ID, connection generation, and monotonic state
version. Queries read only that retained immutable snapshot. Collection records
version-before and version-after; queued packets are drained before the barrier.
If versions differ, collection retries within a fixed bound; if consistency is
still unproven after that bound, no revision is published and dispatch fails
closed. This is a dispatch/generation check, not a claim that a Python lock
stops server/network mutation.

Context-only collection remains read-only. A potentially mutating refresh that
is interrupted returns `mutation_outcome_unknown`; the runner issues no more
action, finish, checkpoint, reconnect, or same-lineage retry, and archives the
link to the previous verified checkpoint. Promotion requires a fault test that
successful refresh leaves activities unchanged. If the backend cannot provide
consistent collection, the tool profile remains blocked rather than using
stale observations as current.

### Operational reachability

A capability is supported only when its registry row, adapter mapping, fog-safe
state, verifier, and scenario tests are approved. The legal catalog contains
only currently legal instances of supported capabilities. Other states remain
distinct: upstream not implemented, not legal now, adapter filtered, summary
omitted, not delivered, or model not selected. Address known actors by ID and
indexed filters (domain, unit role, activity, region), rather than requiring
sequential global paging. A two-query path—locate actor, then fetch an exact
action selector by actor ID and stable action key—must work at any empire size.

Listing all choices may require `ceil(count/page_size)` pages; each page
discloses total and next cursor. The provisional page envelope is 64 entries
and 16 KiB bounded JSON including metadata. An oversized single entry produces
a typed error, never silently truncated fields. These are acceptance-envelope
values, not measured latency claims; P2 benchmarks set recorded budgets.

Static containment must identify all known settlers, all own cities, research,
contact, and complete required production alternatives at its declared scale.
It fails if the complete required table cannot fit; reserving one arbitrary
city option is not sufficient. The full catalog is stored once per revision by
content hash, with immutable delivered pages and delta indexes, and no automatic
raw deletion. A direct known-actor selector must not depend on late global-page
reachability.

### Tool/state machine

Discovery remains optional. With sufficient dashboard state, the model can make
a direct strict decision. Otherwise it may perform at most the profile's
configured number of read-only exchanges. Every assistant tool call receives
exactly one matching `role=tool` result, including errors; invalid or unknown
call IDs cannot commit an orphan exchange. Tools are then disabled and a strict
schema final is requested. Direct JSON in tool-enabled mode is locally schema
validated; server strictness is not assumed.

P2 must test direct, no-query, one-query, chained, malformed, timeout,
strict-final repair, and thinking-plus-tools paths. It must resolve whether the
provider/SDK can replay thinking losslessly in active history. Raw historical
reasoning is always retained. If provider-valid replay cannot preserve its
semantics, the capability is explicitly unresolved and tool mode remains
test-only; there is no quiet new drop. Physical requests, schema repair,
transport retry, and catalog errors have separate monotonic budgets.

### Batch leases and phase stop

Before dispatch, issued refs become a private execution lease containing exact
ordered payload hashes, actor/ownership, source and target preconditions,
phase, generation, and revision. Each leased action is freshly validated
against those fields; semantics are never rebound. External refs expire on
every revision change. An accepted private lease may survive a state revision
change only for remaining refs whose exact payload, ownership, actor,
source/target preconditions, and current legality still match. Phase or
connection-generation changes invalidate the entire lease. No ref is rebound.

The new profile initially permits at most one mutating action per actor per
batch; repeated same-actor dependencies go to the next model round. Legacy
behavior is unchanged. Terminal state is checked after every dispatch.
Mutation uncertainty stops the whole phase with no finish. No-ops are
diagnosed, not retried indefinitely. Only explicit model `finish_phase` or
actual game termination completes a phase. Budget exhaustion yields
`STOPPED_INCOMPLETE_PHASE`, preserves the last verified checkpoint, and never
claims a green completed turn. Cleanup I/O may continue under a separate
deadline but may not issue game commands.

### Durability and compaction

A completed-round WAL record is evidence, not itself a resumable game save.
Prepare exact messages, tool results, action readbacks, sequence numbers, and
hashes; commit success only after the complete exchange. Recovery manifests
name the exact game-save hash, conversation prefix sequence/hash/epoch, memory
sequence/hash, own phase, catalog, and connection identity. Ordered prepare →
fsync files → commit marker is required; cross-process atomic game/filesystem
writes are not promised.

Mid-phase failed lineages are excluded from child replay. A new fork starts from
the previous verified server state and never replays uncertain mutations. Old
profiles are never auto-migrated.

The 500k trigger remains. Prefer compaction at a phase boundary, but a new
round protocol may compact between completed rounds when readback is known and
no tool/action exchange is pending. The summary binds current phase, revision,
and completed-round progress; compaction does not advance turn or memory age.
Before the next request, trusted usage anchor and suffix are checked. Unsafe or
failed compaction stops incomplete without output caps or partial truncation.
Tests cover a mid-phase threshold crossing, crash during compaction, and no
staged-exchange compaction; boundary-only checkpointing still excludes
unrecoverable mid-phase compactions.

### Capability roadmap

Release the core land slice with complete visibility/readback for settlement,
movement/combat, workers, city production, research, and chosen diplomacy.
Add government, rates, and basic negotiation as tested vertical slices. Trade,
transport, specialists, upgrades, maritime actions, espionage, airlift, richer
treaties, and victory mechanics are later slices. The matrix must list upstream
handler, legal preconditions, observation fields, dispatch mapping, verifier,
tests, and known gaps. A land-only prototype cannot advertise a typical full
game or spaceship victory until required controls pass.

### Health and release gates

Report independent statuses: `mechanical_integrity`, `control_access`,
`model_outcomes`, and `capture_readiness`, each `pass`, `fail`, `unknown`, or
`not_assessed`. Hard gates cover checksums, horizons, stale refs, action
coverage/reachability, execution truth, and artifact integrity. Model metrics
are trends or warnings tied to opportunity availability; legitimate holds and
waits are not automatic failures. Rejections are classified as backend,
model-illegal, stale-state, or readback failure.

Capture readiness requires technical provenance and control gates. Editorial
strategic-transition quality is measured only in the selected highlight
window; ordinary turns need not contain a strategic transition. A technically
complete but strategically inaccessible run is labelled accordingly.

Catalog, delta-index, delivered-page, and raw-trace bytes per turn are measured.
Content deduplication and delta indexes may reduce storage, but raw records are
never automatically deleted. A disk-budget alert and explicit archive-failure
stop preserve dispatch truth; disk-full behavior is fault-tested.

The initial target is a fresh isolated game using the same classic ruleset,
Le Chonk, two AI opponents, normal default tiles, and a mainland core fixture.
Before full-game promotion, record actual map/ruleset/victory conditions and
complete every required control: space victory requires launch-path support;
island scenarios require naval transport. No run settings are silently changed.

### API compatibility and risks

Main risks are catalog latency/bloat, read/mutation races, uncertain mid-phase
recovery, prompt/tool compatibility, and incomplete capability coverage for the
intended victory. Mistral's documented function-calling, reasoning replay,
structured output, and prefix caching are compatible with this shape ([function
calling](https://docs.mistral.ai/studio/conversations/function-calling),
[reasoning](https://docs.mistral.ai/studio/conversations/reasoning),
[structured output](https://docs.mistral.ai/studio/conversations/structured-output),
[prompt caching](https://docs.mistral.ai/studio/conversations/advanced/prompt-caching),
and [API reference](https://docs.mistral.ai/api), consulted 2026-10-29), but
the deployed Large 4 fingerprint and installed SDK need a live capability probe.
The initial implementation should fail closed or remain test-only when a spike
fails; it should not quietly downgrade to legacy starvation.
