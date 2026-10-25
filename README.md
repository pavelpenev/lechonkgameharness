# Le Chonk Plays Freeciv

A harness that lets `mistral-large-4` (Mistral Studio API) play Freeciv, built for the
"Le Chonk wants to play" community challenge (October 2026). The harness runs a full
Freeciv stack locally, drives it through a typed, fog-gated observation/action interface,
and records every policy decision in a provenance ledger with frozen honesty metrics.

> **Status (2026-10-25):** evaluation of record complete (`artifacts/runs/wp4-eval3/`).
> Everything in this README is accurate as built; the [Evaluation](#evaluation) section
> reports the recorded run numbers. No numbers on this page are provisional, and none
> are model-ranking claims — this is an honest pilot of one model under a disciplined
> harness.

## What this is — and what it is not (scoped positioning)

"LLM plays a 4X game" is not new: CivBench Season #001 (February 2026) ran eight
models through ~200-turn Freeciv-like games. This project does not claim novelty in
playing the game, and it makes **no model-ranking claims** — this is an honest pilot of
one model under a disciplined harness.

What is claimed:

1. **A model-vs-scripted provenance ledger plus honesty metrics with frozen operational
   definitions** — commitment follow-through (with model-attribution, so a scripted
   default realizing the model's promise is never credited to the model),
   reason-but-don't-act (entity-ID-based, no free-text parsing), and an illegal-action
   taxonomy (schema-error / state-fog-error / illegal-in-rule). In our survey
   (October 2026) no surveyed project ships this instrumented.
2. **Reproducibility discipline** — Docker image pinned by digest, seed control,
   exported-checkpoint paired evaluation under a predeclared paired contract, and
   replay artifacts (recorded-transcript playback; action re-execution is not claimed).
3. **Harness engineering lessons** — the upstream quirks we had to work around are
   documented below and in the [development story](docs/dev-story.md).

What is **not** claimed:

- **No model ranking.** One model, one harness, one machine.
- **No counterfactual pair claims.** The two arms of an evaluation pair start from an
  identical exported checkpoint, but classic-AI opponents are not proven deterministic
  given fixed seeds, so pair arms are **not counterfactual-identical** (see
  [Reproducibility](#reproducibility)). AI-state divergence is reported as a
  descriptive per-pair diagnostic only.
- No claim that CivRealm is our work: the OSS basis is pre-disclosed below and in the
  development story, as the challenge rules explicitly contemplate.

## Prerequisites and setup

Tested topology: everything runs **locally on one Linux amd64 machine** (no remote
hosts, no cloud). Machine used for development: 24 threads, 78 GB RAM.

| Requirement | Version / notes |
|---|---|
| Docker Engine | local daemon; tested with 29.7.2. If the session user lacks socket access: `sudo usermod -aG docker <user>` + re-login, or use `sg docker -c "..."` per command. |
| Image | `civrealm/freeciv-web`, pinned **by digest**: `sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a` (amd64, ~4.36 GB). Pulled and retagged locally as `freeciv/freeciv-web:latest`. Tags are not trustworthy pins (`latest` = `1.4.6`, pushed 2024-04-03, which predates `1.3.2`). |
| Python | 3.12 via [uv](https://docs.astral.sh/uv/) (uv-managed venv; CivRealm's pins predate Python 3.14, and 3.12 is the PR-#7-validated version). `uv venv`, then `uv pip install -e vendor/civrealm -e . pytest`. Ray is deliberately **not** installed (PR-#7-validated config; see below). |
| Node | playwright 1.49.0 (`npm install`; chromium headless shell at `~/.cache/ms-playwright`) — used only for the zero-perturbation observer and video capture. The harness itself never needs a browser. |
| Model access | `mistral-large-4` via the Mistral Studio API. Key in the `MISTRAL_API_KEY` environment variable (the loader reads the process env first, else parses **only** the `MISTRAL_API_KEY` line from a repo-root `.env` at runtime; the value is never printed, logged, or written to artifacts). |

### Provisioning

`scripts/provision.sh` is the scripted, idempotent provisioning recipe — it encodes the
full environment setup and is safe to re-run (existing resources are detected and
reused, never destroyed). Its first step is submodule-based vendoring: if
`vendor/civrealm` is absent or uninitialized it runs `git submodule update --init
--recursive` (cloning from the HTTPS URL recorded in `.gitmodules`), then verifies the
checked-out HEAD against the gitlink recorded in the index and that the known-good
SHAs (`dev` @ `1b144d6`, civgov-fixes head `36ae456`) are ancestors — failing loudly
on any mismatch. Then the uv Python 3.12 venv, the image pull **by digest**,
published-port collision checks, and a hardened first start with a retried pubstatus
health probe. Re-run end-to-end on the working host on 2026-10-25 after the submodule
rework (exit 0; gitlink guard verified, every other guard skipped/verified):
`artifacts/runs/wp5-freshenv/provision_run_submodule.log`.

Local Docker topology (single container, `freeciv-web`, host-persistent):

```
host 8080 -> container 80      webapp (the compose map; upstream's 8443 mapping is stale in this image build)
host 4002      -> container 4002
host 6000-6009 -> container 6000-6009   freeciv servers (publite2 pool; clients use the 6300 range)
host 7000-7009 -> container 7000-7009
host 8888      -> container 8080
host 1240      -> container 1240
```

All client traffic routes through the webapp on :8080
(`/civsocket/{1000+client_port}`). Stack boot takes ~30 s after container start;
`http://localhost:8080/pubstatus` is the health endpoint (the image's own health check
is fragile — the recipe wraps it with retries and fails loudly). Stack RSS is ~2.6 GiB.

**Operational caveat:** game servers quit ~20 s after the player client disconnects
(`--quitidle 20`) — no live game persists between sessions; restore from an exported
checkpoint (see [Reproducibility](#reproducibility)).

The vendored CivRealm code touches the network at *import time* (`port_utils` performs
an HTTP pubstatus fetch on import) and creates `logs/`, `recordings/`, `game_scores/`
in the CWD — the runner imports the adapter lazily and runs it from a dedicated working
directory.

## Architecture

Module map as built (plan §7.1):

```
repo/
  vendor/civrealm/            # GPLv3+ mirror: upstream dev @ 1b144d6 + PR #7 fork @ 36ae456
  src/lechonk/
    adapter/                   # GPL-3.0-or-later — the ONLY package that imports civrealm
      session.py               #   owns FreecivBaseEnv lifecycle: fc_args config before reset,
                               #   deadline-guarded RPC, guarded close, live-state copy-out
      obs_extract.py           #   CivRealm observation dict -> core.schemas.Snapshot (fog-gated,
                               #   8,000-token budget, visibility-safe truncation with disclosure)
      candidates.py            #   legal-candidate table per domain (unit gating; combat odds
                               #   from upstream action_prob)
      execute.py               #   candidate -> action tuple; per-action read-back verification
                               #   (VERIFIED / CONTRADICTED / INCONCLUSIVE); build_city special case
    core/                      # MIT — no civrealm imports, testable offline
      schemas.py               #   pydantic Snapshot / Delta / ActionBatch / CandidateTable /
                               #   TranscriptRecord union (frozen) / RunManifest
      ledger.py                #   provenance ledger: append-only JSONL, per-record checksum,
                               #   fsync at intent-before-write, torn-tail discard on restart
      metrics.py               #   honesty metrics from the ledger (frozen definitions, worked fixtures)
      runner.py                #   turn pipeline + state machine + supervision deadlines + stall
                               #   watchdog + batch group-abort + repair flow + degraded defaults
      memory.py                #   strategy notes (commitments, threat hypotheses)
      delta.py                 #   delta report (<=800 tokens)
      model_client.py          #   model client protocol
    model/                     # MIT
      mistral_client.py        #   Studio API client; json_schema structured output; <=2 API
                               #   attempts per turn; per-call logging (tokens, latency, cost)
      prompts/                 #   versioned templates (hashes in the run manifest)
    comparator/                # MIT
      scripted.py              #   comparator policy (published priority lists + hysteresis)
      defaults.py              #   degraded-default entry point (separate from the comparator)
    eval/                      # MIT
      paired.py                #   paired-run plumbing, predeclared contract, pair manifest
      checkpoint.py            #   checkpoint export/preserve (copy-after-completion, stability check)
      replay.py                #   JSONL -> static HTML replay bundle (no server)
  tests/                       # offline suite: schemas, metrics fixtures, fog fixtures,
                               # runner fault injection, lifecycle kill/teardown/disconnect
  artifacts/                   # transcripts, manifests, checkpoints, replay bundles (per run)
```

**Turn pipeline:** snapshot (<=8,000 tokens) + delta report (<=800) + strategy memory
(<=1,500) -> one Mistral call per turn (<=20 actions, <=2 total API attempts including at
most one repair) -> every action validated against enumerated legal candidates ->
sequential execution with per-action read-back verification -> finish phase. History is
compressed (last turn full, older turns one-liners); overflow is truncated
visibility-safely (idle/distant units elided first, never armed-in-contact ones) and the
snapshot header discloses what was omitted. The model may *request* the finish phase;
only the pipeline sends it.

**Turn state machine:** `ACTING -> FINISH_SENT -> WAITING -> NEXT_OWN_PHASE | GAME_OVER |
FAILED`, with per-operation and per-state supervision deadlines, a 10-minute run-level
stall watchdog, and a process-level watchdog (ordered shutdown -> SIGTERM -> SIGKILL).
Outcome taxonomy: `COMPLETED` (horizon) / `DEFEATED` / `VICTORY` / `TRUNCATED` (with
reason) — infrastructure failure is always separated from strategic loss.

**Provenance ledger** (frozen in the transcript schema): every policy-level decision is
recorded as `model` / `continued_model_order` (linked to its origin) / `scripted_default`
(with reason) / `engine_automatic` / `housekeeping`, with the snapshot ID, the offered
candidates, the choice, the call ID, the execution result and the verification evidence.
The participation gate (model participation on >=80% of turns-with-opportunities) is
enforced on every run: a failing gate labels the run `degraded-dominant` — the run is
still reported in full, but excluded from model-performance claims. The gate passes
iff participation >= 0.80 AND every numeric-status (ok) per-domain share is
majority-model (> 0.5). Small domains (n/a at zero decisions, insufficient sample at
1-4) are exempt by design and cannot block the gate; insufficient participation
leaves the gate unset rather than passed. Engine-automatic
resolutions of new research/production choices are counted, never hidden.

## OSS disclosure

This harness builds on other people's work, pre-disclosed:

- **CivRealm** (github.com/bigai-ai/civrealm, GPLv3+; default branch `dev`, dormant —
  last push 2024-09-11) — the Freeciv Gymnasium client environment (WebSocket/JSON
  protocol, descended from chris1869/freeciv-bot). Vendored as the git submodule
  `vendor/civrealm`, pinned to branch `lechonk-harness` of the pavelpenev/civrealm
  mirror @ `8eb6dba`: upstream `dev` @ `1b144d6` + the **PR #7 fork branch**
  (author Baby1991, branch `civgov-fixes`, head `36ae456`; 3 commits, 7 files,
  +75/−18) plus our ray-guard extension, committed on the mirror as `8eb6dba`
  ("Make FreecivParallelEnv/FreecivA3CEnv imports optional (ray intentionally
  absent)"). The vendor tree is never modified by us beyond that committed
  extension (quirks are worked around adapter-side).
- **PR #7** fixes we rely on, each classified in writing at the WP0 gate — see
  [`docs/gate-record.md`](docs/gate-record.md) (6 entries with git-diff/file:line
  evidence): the `research_name` wrong-lookup fix, sentry (`ActSEntry`)
  re-registration, per-unit `activity`/`ssa_controller` observation fields, the ray
  import made optional (extended by us — entry 1), and the `AcceptTreaty` pact-clause
  fix (classified not-needed-for-our-surface). ActExplore the PR attempted to fix
  remains **broken upstream** (the freeciv-web server build refuses
  `SSA_AUTOEXPLORE`), so auto-explore stays out of the action surface.
- **The `civrealm/freeciv-web` Docker image** — freeciv 3.3-dev server
  (capstring `+Freeciv.Web.Devel-3.3`), freeciv-web webapp (AGPLv3), freeciv-proxy.
  Pinned by digest (above).
- The feasibility evidence for the whole stack was, at planning time, the PR author's
  self-report ("full games of 150 turns" on Python 3.12 without ray) — a single,
  unmerged, zero-comment source. WP0 was its validation, and it is never cited as
  independent evidence.

### Upstream quirks we worked around (harness engineering lessons)

All worked around adapter-side; the vendor tree was never modified. Each of these was
found live and cost real debugging time — they are the "harness lessons" claim above:

- `import civrealm` performs an HTTP fetch at import time and (mainline) imports ray
  unconditionally — lazy import after the stack is up, plus a ray-guard extension.
- **Upstream hands out live mutable state** — turn-state dicts get rebound and map
  numpy arrays are mutated in place, so a held pre-step observation silently becomes
  post-step. Fixed with a recursive deep-copy at the session boundary.
- **Sentry read-back quirk:** the read path's activity-restore deliberately skips
  `ACTIVITY_SENTRY`, so a sentry is always undone within the same step — sentry
  read-back is INCONCLUSIVE-by-quirk (never CONTRADICTED), disclosed rather than hidden.
- **Fortify in-effect state:** the fortify read-back must accept `ACTIVITY_FORTIFYING`
  (the in-effect activity), not only the sent-packet echo.
- **Map arrays are x-major** (`status[x, y]`) — an extractor initially read them
  transposed; caught by live verification and pinned with a non-square regression test.
- **Tech costs are not exposed** through CivRealm's ruleset packets (verified live, all
  `None`) — see the [tech-cost degradation disclosure](#exclusions-and-defaults).
- **`available_actions` omits a city's CURRENT production target** — this silently
  caused the comparator's frozen production ladder to alternate Warriors -> Barracks
  every turn; resolved with the user-approved 2026-10-24 production-hysteresis
  amendment (below).
- Unbounded upstream waits everywhere: `CivController.close()` waits on a game-over
  message with no deadline, all `requests` HTTP calls have no timeout, `Ports.get()`
  spins forever — every one is wrapped with a hard deadline adapter-side.
- Upstream's own 30 s timers (`server_timeout`, `begin_turn_timeout`) truncate slow
  agents; raised to align with the supervision deadlines.
- `reset(seed=)` sets only mapseed/agentseed; the gameseed is sent only when
  `debug.gameseed` is present in `fc_args` — set explicitly for full determinism.
- `self_play=True` (upstream default) silently renames the user on connection
  congestion, breaking the save-name/username coupling — set to `False`; congestion
  raises loudly instead.
- Saves are named `{username}_T{turn}_{timestamp}` and `load_game()` validates the
  username prefix — paired arms run under the same pinned username as the checkpoint
  creator (sequential, so no connection conflict).
- CivRealm deletes per-turn autosaves via unauthenticated webapp HTTP endpoints — the
  reason we built an independent copy-after-completion checkpoint export path.
- The image's metaserver DB rows and pubstatus pages are hours-stale; in-page client
  state is the only authoritative live evidence.

## Licensing

Per the project's licensing posture (plan decision 7):

- **The repo as distributed is a combined work under GPL-3.0-or-later terms.**
  `pyproject.toml` declares `GPL-3.0-or-later`.
- **`lechonk.adapter` is GPL-3.0-or-later** (SPDX header in-file) — it is the only
  package that imports the vendored CivRealm.
- **Independent modules that never import CivRealm — `lechonk.core`, `lechonk.model`,
  `lechonk.comparator`, `lechonk.eval` — are MIT** (SPDX headers in-file,
  purity-test-enforced). The Mistral client, the provenance ledger, the metrics
  computation and the replay viewer stay MIT.
- **Vendored CivRealm carries its upstream notices** (`vendor/civrealm/LICENSE` and
  file headers). Its `pyproject.toml` metadata is self-contradictory — the license
  field reads `"GLP3.0"` (a typo) and the classifiers include a stray LGPLv3 entry;
  **the file headers govern: GPLv3+**. This characterization was re-verified against
  the vendored tree on 2026-10-24 before this section was written.
- **Freeciv mainline is GPLv2+**; the image's freeciv-web client is **AGPLv3** — AGPL
  §13 (source offer for network users) is not triggered because nothing is hosted for
  remote users; the stack runs locally.

## Exclusions and defaults

### Action-surface exclusions (v1) — each with its explicit default

| Excluded | Why | Explicit default |
|---|---|---|
| Exploration (auto-explore) | Broken upstream: PR #7 attempted the fix and left `ActExplore` unregistered; the freeciv-web server build refuses `SSA_AUTOEXPLORE` | No component ever requests auto-explore; units follow the comparator unit defaults (improve by terrain yield, else fortify/hold) |
| Autosettler / auto-worker | `ActAutoSettler` is made functional by PR #7 but handing units to the server-side agent would silently erode model participation | Workers are never handed to the server-side agent; they follow the comparator unit defaults |
| Naval, air, spies, caravans/trade routes | Out of the v1 surface | No naval/air/spy/caravan verbs are exposed; such units hold (fortify / keep activity) per the unit defaults |
| Advanced treaty construction | Diplomacy is read-status + decline/exit only in v1 | Decline/stop all negotiations; no proactive offers (also the comparator's diplomacy default) |
| Citizen optimization | Out of the v1 surface | Citizens keep the server-assigned tiles; no `city_work` reassignment by policy |
| Tax/luxury tuning (stretch) | Stretch item, not reached | Server defaults retained (`set_sci_lux_tax` not exposed) |
| Religion depth | Out of the v1 surface | No religion verbs exposed in the classic-ruleset surface used |

First-contact diplomacy note: in Freeciv, first contact defaults to **WAR** — disclosed,
not treated as a harness decision. Chat, government changes, sell verbs and
`ignore_diplomacy_request` (unregistered upstream; the decline path is
`stop_negotiation`) are likewise not exposed.

### Comparator and degraded-default tables (published verbatim)

The **comparator** is the active scripted policy (used for paired evaluation); the
**degraded defaults** are the minimal end-of-turn fallback used when the model is
unresponsive after the repair cap. Separate entry points, both logged; every defaulted
decision carries a reason.

- **Research priority list** (first affordable legal candidate wins): Alphabet →
  Pottery → Agriculture (irrigation prerequisite chain) → Bronze Working → Writing →
  Currency → Masonry → Wheel → trade-adjacent techs; tie-break: lowest cost, then list
  order.
- **Production priority list** (per city, first legal): defensive unit if city
  undefended → builder-type unit while improvements remain and builders < cities+1 →
  growth building (Granary-class) → science building → wealth fallback. Buy-production
  when gold > 2× buy_cost and shield stock < 25% complete.
- **AMENDMENT (2026-10-24, user-approved) — production hysteresis; this amends the
  frozen production priority list above.** Live evidence
  (`artifacts/runs/wp1-comparator/`, seed 20261026, 20-turn unattended run): upstream
  `available_actions` never lists a city's CURRENT production target as a produce
  candidate, so the frozen ladder alternated Warriors → Barracks → Warriors every turn
  (city 111, turns 2–20, `defensive_unit_undefended_city` / `fallback_production` each
  turn) and nothing completed — the comparator was faithful to the frozen wording; the
  wording was missing the current-target case. **Rule:** a production tier is SATISFIED
  if the city's current production target already belongs to that tier's output class,
  even when the target is absent from the candidate table; a satisfied top applicable
  tier does not switch. Hysteresis never suppresses a legitimate higher-priority
  switch: a current target satisfying only a lower tier does not protect it (an
  undefended city producing Barracks still switches to the defensive unit; a defended
  city producing a growth building still switches to a builder unit while the builder
  tier applies). **Rationale:** tier-class satisfaction — not exact-item equality
  against the candidate table — is what the ladder's intent ("defensive unit if city
  undefended") already means; the candidate-table omission is an upstream enumeration
  artifact, not a policy signal. The degraded-defaults entry point is unaffected (it
  preserves production anyway and only resolves EMPTY production, where no target
  exists to satisfy a tier). Implemented in `src/lechonk/comparator/scripted.py`
  (`production_pick`); re-verified live under `artifacts/runs/wp1-comparator-amended/`.
- **Unit defaults:** found city on the first legal candidate tile when a settler exists
  and no city is being founded this turn; else improve (road/irrigation/mine by terrain
  yield); else fortify. Military: defend the nearest own city; attack only adjacent
  visible targets with favorable odds (or without odds data if the probe fails,
  disclosed).
- **Diplomacy default:** decline/stop all negotiations; no proactive offers.
- **Degraded-defaults entry point** (distinct from comparator): preserve all current
  orders and production, finish phase; only NEW mandatory choices (research completed,
  city production empty) resolve via the priority lists, logged `scripted_default`.

**Tech-cost degradation disclosure:** tech costs are NOT exposed through CivRealm's
ruleset packets (verified live against the running stack, all `None`). The research
tie-break ("lowest cost") therefore **degrades to list order** — the published
priority list order is the effective tie-break. Disclosed rather than worked around.

## Fog disclosure

**Observation purity is a pipeline invariant** over the whole policy-facing surface:
observations, candidates, summaries, deltas, error messages, and strategy memory are
all visibility-gated. Unexplored / last-known / currently visible are distinguished end
to end (the map domain's status trichotomy: 0 = unknown, 1 = known-fogged, 2 = seen).
Privileged data (saves, scorelogs, admin output) never enters the policy pipeline; the
replay/evaluation views are separate. The illegal-action stream is treated as fog
telemetry (the state-error share is reported per run). This invariant is enforced by
offline tests, including world-pair fixtures: two worlds differing only in hidden
mutable state must produce equivalent policy inputs.

Upstream quirks with fog implications, disclosed:

- **Sentry read-back quirk:** upstream's read path cancels a unit's activity before
  querying action probabilities and never restores `ACTIVITY_SENTRY`, so a sentry is
  always undone within the same step. Sentry read-back is INCONCLUSIVE-by-quirk —
  the observation cannot distinguish sent-then-cancelled from never-sent.
- **Fortify in-effect state:** the read-back accepts `ACTIVITY_FORTIFYING` — a unit
  reported as "fortifying" is in effect fortified.
- **Vanished-enemy semantics:** a unit that leaves the visible area is last-known,
  **not destroyed** — the prompt-level fog disclaimer ("vanished ≠ destroyed; unknowns
  are real") and the extractor both preserve this distinction.
- **Cold-start policy:** after a checkpoint load, revealed-but-not-visible mutable
  fields (owners, terrain extras, unit contents) are **unknown until observed** —
  never carried stale from a previous session's knowledge.

## Reproducibility

- **Image pinned by digest** (not by tag — the image's tags are demonstrably
  untrustworthy); the digest is recorded in every run manifest, together with the
  embedded server commit, capstring, vendored base + patch head SHAs, harness revision,
  Python lockfile, all seeds, ruleset (`classic`, CivRealm default), server settings
  (`phasemode player`, `timeout 0`, `pingtimeout 720`, `autotoggle disabled`,
  `victories` SPACERACE|ALLIED, `aifill` pinned), model ID + sampling + prompt/schema
  hashes, effective post-load config, and checkpoint checksums.
- **Seeds:** `randomly_generate_seeds=False` with explicit `mapseed`/`gameseed`/
  `agentseed` set before reset.
- **Checkpoint export:** our own copy-after-completion path (independent of CivRealm's
  autosave deletion): with the server's begin-turn `/save` completed, the save file is
  copied out via `docker cp` with a stat → copy → re-stat → checksum-compare stability
  check; the exported file (sha256 recorded) is the paired-run artifact. Exports are
  logged in `artifacts/checkpoints/exports.jsonl`.
- **Paired evaluation:** both arms of a pair load the IDENTICAL exported checkpoint
  under a predeclared paired contract (equal observation format, candidate
  restrictions, action budgets, lifecycle handling — hashed and recorded in both arm
  manifests), run sequentially on an otherwise-idle machine.
- **Model fingerprint:** every arm records a per-model fingerprint (fixed
  capacity-revealing probe prompt, `fingerprint_probe_v2`; K=3 samples collected
  under the arm's sampling — temperature 1.0, reasoning_effort high, no top_p;
  the recorded fingerprint is the sorted, deduplicated set of the sample output
  hashes) and arm-start timestamps —
  `mistral-large-4` is a public preview whose weights can drift under the same ID;
  the fingerprint is a canary (serving nondeterminism can false-positive; an unchanged
  sample set does not prove unchanged weights). **Drift policy:** on a between-arm
  mismatch, re-probe once; a persistent mismatch marks the pair drift-suspect —
  excluded from reproducibility claims, both arms still reported individually.
- **Replay:** every run's append-only checksummed JSONL transcript renders into a
  self-contained static HTML bundle (per-turn timeline, provenance chips, commitments
  board, metrics timeline, manifest) — judges can verify method without the game.
  Render with `scripts/wp3_replay_render.py <run_dir>`.
- **Known limitation (honest):** the two arms of a pair are **not
  counterfactual-identical**. They start from the same checkpoint and share the
  scenario/map parameters, but the classic-AI opponents are not proven deterministic
  given fixed seeds, and the two policies' different actions diverge the game state.
  AI-state divergence over turns is therefore reported as a descriptive per-pair
  diagnostic (score, city count, units, known tiles) — not as a causal estimate.

### Sampling recorded in run manifests (old vs new semantics)

- **What the client actually runs (decision 4, 2026-10-24 amendment):** every
  model call — turn calls, repair attempts, and capability/fingerprint probes
  alike — runs at temperature 1.0 with `reasoning_effort="high"`; the client
  never sends `top_p` and never sets a random seed. This is the single source of
  truth (`lechonk.model.sampling.actual_client_sampling`), and it is what all
  four run drivers (`wp4_eval.py`, `wp3_paired_smoke.py`, `wp2_model_run.py`,
  `wp2_fault_run.py`) now persist in their manifests.
- **Old manifests (before this change) are NOT rewritten.** Manifests recorded
  before 2026-10-25 persist a fabricated sampling dict
  (`{"temperature": 0.0, ..., "top_p": 1.0, "random_seed": null, ...}`) that the
  client never ran — a greedy-sampling description left over from the pre-FW-20
  spike era. That evidence is never edited: when comparing runs across this
  boundary, treat any manifest `sampling.temperature == 0.0` (or a `top_p` key)
  as a pre-W1-6 record whose actual sampling was temperature 1.0,
  `reasoning_effort="high"`, no `top_p` — the same as every post-W1-6 manifest
  records explicitly.
- **API-call totals:** arm manifests' `post_run.api_totals` now count each HTTP
  request exactly once. Totals recorded before this change double-counted
  fingerprint-probe usage (the probe's aggregate record plus its per-request
  records were both summed); the probe's per-request records remain the log's
  unit of account, and the probe aggregate lives in
  `fingerprint_evidence.probe_1_details`.

## Evaluation

Evaluation of record: **`artifacts/runs/wp4-eval3/`** (2026-10-24, overall verdict
**PASS**). Two pairs × 50 turns, both arms of each pair loaded from identical exported
turn-5 checkpoints (sha256-verified, loaded-state equality checked, predeclared paired
contract hash identical in both arm manifests). All four arms **COMPLETED**
(`horizon_completed`, survived, 50/50 turns; no TRUNCATED arms, no strategic
terminals). Model-arm totals: **180 API calls, 1,744,241 tokens in / 1,055,018 out,
$2.45 total cost**, 10,388.9 s total wall-clock (~2.9 h). Both pairs PASS; per-pair
details are in `artifacts/runs/wp4-eval3/metrics_report.md` and the per-pair reports
it links. Honest-pilot framing throughout: **descriptive reporting only, no
model-ranking claims.**

- **Pairs run** — `wp4-eval3-p1` and `wp4-eval3-p2`, each comparator-arm-then-model-arm
  sequential from the identical checksummed checkpoint
  (`e09c96e2…8bd290` and `1542929c…9fe773`). Outcome taxonomy per arm: COMPLETED /
  horizon_completed, 50/50 turns, no early terminals. Model-arm fingerprints matched
  the recorded reference in both pairs (drift_suspect=False,
  reproducibility_claim=True; the fingerprint is a canary — an unchanged output does
  not prove unchanged weights). Full acceptance-check lists (12/12 PASS per pair) are
  in the per-pair reports.

- **Participation gate** — p2 **PASS** (0.94 ≥ 0.80; 47/50 turns-with-opportunities).
  p1 **FAIL** (0.64 < 0.80; 32/50). Per the plan's rule (§3 line 38), p1 is labeled
  **degraded-dominant** in its pair report and the overall report: reported in full,
  but **excluded from model-performance claims**. The main driver is measured, not
  conjectured: 21 of p1's 88 turn calls failed with transport read-timeouts (see
  limitations below).

- **Metrics tables** — the reports are the artifacts of record; this section does not
  duplicate them. `artifacts/runs/wp4-eval3/p1/metrics_report.md` and
  `.../p2/metrics_report.md` carry, per arm: the full per-turn progress curves
  (score/cities/population/units/research/known tiles, descriptive), the per-turn
  wall-clock/tokens/latency/cost table, and the honesty metrics. Headline shares as
  recorded:
  - *Commitment follow-through* — p1 model: 1/3 (insufficient sample, no claim);
    p2 model: 0/1 (insufficient sample, no claim). Model-attributed shares are
    identical (a scripted default realizing a promise is never credited to the model).
    Reason-but-don't-act: n/a (insufficient sample) in both pairs.
  - *Illegal-action shares* (over all model-emitted action attempts, minimum-denominator
    rule) — p1 model, n=193: illegal_in_rule 2.1%, schema_error 18.1%, state_fog_error
    2.6%. p2 model, n=262: illegal_in_rule 1.5%, schema_error 17.9%, state_fog_error
    1.5%.
  - *Provenance shares* (executed actions) — p1 model: model 0.943,
    continued_model_order 0.038, engine_automatic 0.019. p2 model: model 0.92,
    continued_model_order 0.067, engine_automatic 0.013. The comparator arms executed
    3 actions in 50 turns each (see the comparator fixed-point disclosure below).

- **Divergence curves** — the AI-state divergence-over-turns diagnostic per pair
  (score, city count, units, known tiles; 50 turns compared per pair): p1 mean L1
  55.64, max 129; p2 mean L1 71.7, max 135. **Descriptive only; the arms are not
  counterfactual-identical** (see the limitation above), so no causal or
  counterfactual claim is made from these curves. Full per-turn tables:
  `artifacts/runs/wp4-eval3/p1/divergence.json`, `.../p2/divergence.json`, and the
  per-pair reports.

- **Feasibility numbers** — measured over the evaluation model arms: per-turn
  wall-clock ~35–177 s in p1 and ~34–130 s in p2 (full per-turn tables in the pair
  reports), model-arm wall-clock 6,169.2 s (p1) and 4,186.5 s (p2) against 16–17 s for
  the scripted comparator arms, total API cost $2.45 for 180 calls. Token budgets held:
  max snapshot ~3.7k of the 8,000-token budget (see limitations for the
  under-informing watch).

Crashes/stalls are reported separately from strategic losses in all of the above; this
evaluation recorded none (no TRUNCATED arms, no strategic terminals).

### Limitations and honest disclosures

1. **p1 transport read-timeouts.** 21 of p1's 88 turn calls failed with
   `transport failure: The read operation timed out` (6 initial attempts on turns
   5, 6, 7, 8, 10, 11; 15 repair attempts on 15 further turns) — each such turn fell
   to the degraded default, and this is the main driver of p1's 0.64 participation
   below the 0.80 gate. Evidence: `artifacts/runs/wp4-eval3/p1/metrics_report.md` and
   the p1 model arm's `api_calls.jsonl`.
2. **Completion-token cap at scale.** The 8,192-token completion cap was hit in 35 of
   p1's 88 turn calls (40%) and 47 of p2's 90 (52%) (derived from the arms'
   `api_calls.jsonl`: calls with `finish_reason: length`). Repair recovery was
   imperfect: on 3 (p1) and 7 (p2) turns both the initial and the repair attempt were
   capped, so the turn fell to the degraded default. The WP2 brevity prompting
   (directional 0/3 in its 3-turn spot-check) did not eliminate capping at
   evaluation scale.
3. **The degraded-default resolution sub-path never fired live.** The
   scripted_default priority-list resolution path (the degraded default's
   action-choosing branch) never executed an action in the model arms — the eval3
   model-arm provenance counts contain no scripted_default (degraded turns executed
   nothing rather than a scripted choice). Tracked since WP2 as a watch item
   (WP2 open item (a), recorded in the project's handoff devlog).
4. **Snapshot under-informing.** The observation snapshot used ~3.1k (p1) / ~3.7k
   (p2) of its 8,000-token budget at turn 50 (derived from the arms'
   `api_calls.jsonl` `section_tokens.snapshot`) — still well under budget at
   evaluation scale, so the watch item from WP2 (snapshot ~1.5k of 8,000 at opening
   scale) remains open: the model may be under-informed relative to the budget.
5. **Measurement condition.** The evaluation ran sequentially on an idle machine
   (plan constraint); absolute wall-clock and latency figures are conditioned on
   that environment and the p1 read-timeout share shows the provider-side latency
   tail was nonetheless real.
6. **Comparator fixed point on checkpoint-loaded states.** The scripted comparator
   executes the published §7.4 policy each turn through the same runner pipeline as
   the model arm. On the evaluation of record's checkpoint-loaded states (a single
   small city with all units on the founding tile, research already on the priority
   target), the published policy reaches a fixed point after its first turn: the
   research target already is the priority pick, production is hysteresis-satisfied,
   and the settlers never move because the published civilian ladder only founds a
   city on the settler's CURRENT tile (never legal on an occupied city tile) and has
   no move-to-new-site provision. The comparator arm therefore shows 3 executed
   actions in 50 turns — verified turn-by-turn against a direct `scripted.decide`
   call on the reloaded checkpoint (turn 5: identical decisions; turns 6–7: zero
   decisions). This is a faithful consequence of the published policy on this state,
   not a harness defect; it makes the comparator arm a weak baseline on
   checkpoint-loaded small-empire states. **The paired comparison is therefore
   model-vs-that-baseline.** Evidence and the ready-made disclosure block:
   `artifacts/runs/wp4-comparator-diagnosis/README.md` (probe source, JSON results,
   stdout log alongside).
7. **Two truncated predecessor runs, preserved.** `artifacts/runs/wp4-eval/` (both
   model arms TRUNCATED at 0/50: `state ACTING exceeded its deadline of 180.0s`) and
   `artifacts/runs/wp4-eval2/` (TRUNCATED at 8/50 and 12/50, same recorded reason)
   are preserved unmodified as honest engineering history — infrastructure
   truncations reported as TRUNCATED, never scored as strategic losses (the
   development story's evaluation-day section gives the root causes and fixes). The
   cut ladder (plan §4 line 58) was **never invoked**: the run's `--pairs/--horizon`
   parameters reached the ladder shapes directly.

## Known seams

Agreed simplifications in the current implementation, documented rather than hidden:

- **Elided units and the candidate table.** The candidate table is the legal-action
  surface: every unit visible in the snapshot is offered, including units the
  observation budget would otherwise elide from summaries. Elision disclosure applies
  to the snapshot's unit list (`omitted.elided_unit_ids`), not to the candidate
  enumeration itself.
- **Turn identity is positional.** A turn is identified by its index within the run
  (`<run_id>/t<turn>`), not by a game-side turn number; re-mapping to in-game turn
  counters is left to the consumer.
- **Requests timeout guard is process-global.** The adapter's outbound HTTP timeout
  wrapping is applied process-wide and intentionally never restored (one harness
  environment per process; see the rationale in `src/lechonk/adapter/session.py`).

## Video and development story

- **Video:** [`artifacts/submission/video.mp4`](artifacts/submission/video.mp4) — 99.1 s,
  1280x720 H.264. Sourced from the zero-perturbation observer (Playwright chromium headless;
  observers never take a player slot) plus replay-viewer playback, assembled by
  `artifacts/submission/build/assemble.sh` with a per-source manifest in
  `artifacts/submission/build/video_manifest.json`. Capture scripts:
  `scripts/wp0_observe_attach.js` (observer attach flow), `scripts/wp0_spectate.js`.
- **Development story:** [`docs/dev-story.md`](docs/dev-story.md) — how the harness was
  built, what broke, and what it was based on (including the OSS disclosure above), now
  including the evaluation-day narrative (the two truncated predecessor runs, the
  comparator fixed-point diagnosis, and the gate outcomes).
