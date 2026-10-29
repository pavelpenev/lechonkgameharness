# Harness control audit — 2026-10-29

**Status:** evidence review snapshot; no implementation in this audit. The
deadline patch that followed resolved A05; findings below preserve the snapshot
and distinguish historical evidence from current code.

**Scope.** This audit covers the dirty `main` worktree at commit `393a4fe`,
the vendored CivRealm changes, and saved runs `lechonk-rehearsal-50-20261028`,
`lechonk-rehearsal-43-cont4-20261029`, and `lechonk-rehearsal-100-20261029`.
It does not start a game or claim a victory condition. The offline baseline was
`.venv/bin/python -m pytest -q`: 1,318 passed in 46.02 seconds. Ruff was
unavailable and no lint command is declared.

## Findings

### A01 — P0, observed: the model lost strategic actions

`CANDIDATE_TABLE_TOKEN_BUDGET` is 1,200 in `src/lechonk/core/schemas.py:53-57`.
`build_candidates()` emits domains in unit, city, tech, diplomacy order
(`src/lechonk/adapter/candidates.py:239-262`), and legacy admission keeps that
order (`src/lechonk/core/schemas.py:352-416`). The interactive launcher used
`gameplay_v2`, which maps to `legacy_v1` (`scripts/launch.sh`,
`scripts/wp2_model_run.py:360-364`; runner selection at `runner.py:799-817`).

The saved run offered 60 unit, 17 city, and 7 tech candidates at turn 20; by
turn 30 it offered 89 unit and 2 city candidates; turns 35–42 and 51–100 were
unit-only. In turns 51–100 the offered table had 89–92 unit candidates and
reported 201–379 omitted candidates. Settler 159 remained idle at `(17,44)`;
settler 171 appeared by turn 70 at the same location. Four own cities remained
four. The model explicitly reported missing settler, city, and tech refs and
did not invent them, as required by the exact-ref protocol.

The run does not persist the uncapped pre-admission table, so it cannot prove
that a particular `build_city` action was legal on a particular tile. It does
prove that no such ref reached the model. The causal chain is therefore:
unit-first construction → legacy cap → unreachable strategic controls →
default production and holds → a mechanically complete but strategically
stalled run.

### A02 — P1, source-proven: existing policies are containment, not a complete fix

`candidate_surface_v2` uses an 8,000-token domain/actor round-robin. v3 uses
12,000 tokens and reserves immediate research plus one rotating production
choice per city (`src/lechonk/core/candidate_surface_v2.py`,
`candidate_surface_v3.py:63-83`). Neither explicitly reserves a live settler's
movement/founding actions or every consequential actor. A larger static table
will fail again as the empire grows. The tests are synthetic and do not cover
100+ units, unequal city option counts, or a real settler.

### A03 — P1, source-proven/risk: state and action surfaces truncate independently

Research and production choices are capped before rendering
(`src/lechonk/adapter/obs_extract.py:342-353,488-525`). The final snapshot may
elide own idle/distant units, messages, the ASCII map, and region prose
(`obs_extract.py:985-1067`). Candidate construction is independent of this
snapshot. This can expose an action whose actor has no current state or leave a
settler without route geometry. The saved run proves candidate starvation, but
does not prove that actor elision caused that specific failure. Existing lookup
is an encyclopedia lookup, not a general live-state query.

### A04 — P1, source-proven: one model batch ends the phase

`ActionBatch` is capped at 20 actions (`src/lechonk/core/schemas.py:472-479`).
The runner executes one batch and then finishes the phase
(`src/lechonk/core/runner.py:2053-2101`). The client has one bounded lookup
exchange followed by a strict final response (`src/lechonk/model/turn_client.py:924-1002`).
This was not the direct cause of omitted settlers, but it makes a growing army
impossible to control comprehensively and couples discovery, repair, and phase
completion. A future controller needs multiple bounded rounds with explicit
finish.

### A05 — P1, confirmed historical bug: tech readback could choose the wrong player

`_verify_tech()` scans for the first player containing `tech_goal`,
`bulbs_researched`, or `embassy_txt` (`src/lechonk/adapter/execute.py:479-491`).
The vendored player state initializes those marker keys for every player
(`vendor/civrealm/src/civrealm/freeciv/players/player_state.py:58-63`), so key
presence does not identify the active player. A no-embassy reproduction with
opponent 1 first, active player 2, opponent researching 3, and active player
researching 7 returned `contradicted: researching is 3, expected 7`. The
verifier must use the known active player ID. This finding was reproduced in the
2026-10-29 audit snapshot and resolved in the deadline patch: `_verify_tech`
now selects `post_obs` by the session's `my_player_id` in
`src/lechonk/adapter/execute.py:480-513`, with regression coverage in
`tests/test_execute.py:419-464` (including the opponent-first reproduction).
The saved-run evidence above remains historical and is not silently rewritten.

### A06 — P1, known lifecycle risk: reads have an uncertain mutation window

The session documents that refresh/action-probability queries may cancel and
restore activities; a disconnect during the window can leave activities
cancelled (`src/lechonk/adapter/session.py:950-966`). Context collection itself
is intended read-only (`session.py:364-394`). Fault injection and explicit
uncertain-mutation recovery are missing from the acceptance story.

### A07 — P0, observed: the top-level PASS is misleading

`run_meta.json.verdict` is computed from structural checks only
(`scripts/wp2_model_run.py:2546-2634`). The saved run's
`metrics_report.json` says `participation_gate_pass: false`, research model
share `0/12`, and commitment follow-through `0/16`, while `run_meta.verdict`
is `PASS`. Metadata copies metrics after verdict computation
(`wp2_model_run.py:2712-2721`). `keep_activity` readback means the unit became
unactable for that phase (`execute.py:384-394`), not that it made progress.

Legacy records save the offered count and offered-by-domain, but omit raw total,
omitted-by-domain, omitted actor IDs, and the complete pre-admission catalog
because no admission object exists (`runner.py:1951-1968`; schema documentation
at `schemas.py:595-607`). Outcome metrics are useful signals, not authoritative
game facts. Health must
separate mechanical integrity, control accessibility, model outcomes, and
capture readiness.

### A08 — P1, intentional capability gap

The whitelist excludes government, rates, transport, trade, specialists,
upgrades, richer treaty construction, and other controls
(`src/lechonk/adapter/candidates.py:13-26`). Vendor handlers exist for some
of these (government/rates, specialists, upgrades, treaty actions), while
transport/trade and spaceship victory paths are only source-declared or
unverified in this harness. The capability matrix must distinguish each case;
a land prototype cannot claim a typical complete game or victory path.

### A09 — P1, configuration and metadata fragmentation

The launcher defaults to `gameplay_v2`/legacy while the direct runner defaults
to `legacy_eval`; v3–v6 are fragmented opt-in profiles. Recovery identity binds
profile and admission policy, so a resumed run cannot silently switch policy.
The resumed artifact mixes a local target of 50, engine `max_turns` 51, and an
absolute completion horizon of 100 (`scripts/wp2_model_run.py:2663,2699-2700`).
Metadata needs segment start/end and absolute horizon fields. The run comparison
tool's `state_wal_relationship: not_proven` is a diagnostic limitation, not
evidence of corrupted data.

### A10 — P1/P2, observed with uncertain causality: speed and context

The slow segment included 44,480 output tokens and a repetitive “hmm” loop
before a timeout. The fast segment averaged about 2,904 output tokens and
34.6 seconds per initial call. Both used explicit `temperature=1.0`,
`reasoning_effort=high`, omitted `top_p`, and omitted `max_tokens`; no local
output cap caused the change. Compaction preceded the shorter segment, and
the provider fingerprint changed, so causality is unproven. The existing
trusted usage anchor and 500k compaction transaction are already tested; the
missing test is the complete timeout → unknown usage → compaction → checkpoint
→ resume sequence.

### A11 — P1, source-proven: tool-call replay has a reasoning-content gap

The ordinary assistant commit preserves exact provider content, but the
tool-call path constructs a replay assistant message through
`turn_client.py:642-680`. That helper normalizes the assistant to answer text
while retaining raw traces separately. This is a potential loss of provider
thinking chunks in the active tool-call conversation; it is not evidence that
all raw reasoning is lost. The next Mistral capability probe must establish a
provider-valid, lossless replay representation. Until then, tool-enabled
production mode remains unresolved/test-only rather than silently discarding
reasoning.

### A12 — P1, source-proven: the 20-action limit is a phase-control bottleneck

The existing runner can issue at most one 20-action batch before phase finish.
Persistent orders make this less severe than “the army is impossible to
control,” but fresh actions for a large army cannot all be issued in that phase.
The fix must make 20 a per-round bound, preserve legacy behavior, and require
explicit model-selected phase finish.

### A13 — P1, source-proven: the existing lookup cannot recover omitted state

`turn_client.py:85-108` exposes `lookup_freeciv` as an encyclopedia lookup;
runner startup wiring (`runner.py:1102-1114`) closes over that reference data.
It cannot query the current map, actors, or legal actions. A proposed live
catalog therefore needs separate fog-safe adapter queries, not a larger
encyclopedia response.

## Capability matrix snapshot

This is a source inventory, not a claim that every path is live-tested.

| Capability | Evidence | Current classification |
|---|---|---|
| Government/rates | `vendor/civrealm/src/civrealm/freeciv/players/government.py:90-143` | handler-present, harness filtered, live-unverified |
| Specialists | `vendor/civrealm/tests/city/test_city_change_specialist.py:44` | source/test-present, adapter exposure unverified |
| Unit upgrade | `vendor/civrealm/src/civrealm/freeciv/units/unit_actions.py:1256` | handler-present, harness filtered, live-unverified |
| Treaty actions | `vendor/civrealm/src/civrealm/freeciv/diplomacy_actions.py:479-510` | handler-present for listed actions, broader treaty construction unverified |
| Trade/transport | `vendor/civrealm/docs/advanced_materials/fullgame_action.md:35-55,150-164` | documentation-declared, execution mapping unverified |
| Spaceship launch | `vendor/civrealm/src/civrealm/freeciv/units/spacerace.py:82-95` and `SpaceCtrl` options | source-declared, available-action/readback path uncertain |

The matrix must be updated with a legal precondition, observation field,
dispatch mapping, verifier, and test before a capability becomes supported for
a release scenario. Unsupported victory outcomes remain explicitly restricted.

## Strengths and limits

Fog-gated observations, current per-action revalidation, checksum/WAL recovery,
trusted cache anchors, exact reasoning traces, and lineage checks are valuable
foundations. This audit did not run a new game, recover the missing raw action
table, establish a terminal objective, or prove that a strategy is good. The
next design must preserve those strengths while making control reachability
observable and testable.
