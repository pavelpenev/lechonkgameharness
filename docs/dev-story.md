# Development story — Le Chonk Plays Freeciv

*How this harness was built, what broke along the way, and what it was based on.*

**People and tools:** one developer, working with AI coding agents (Chartreux) as the
implementation workforce; every phase was independently reviewed before exit against
the normative plan. The [evaluation](#evaluation-day) section below records the
evaluation of record and the two truncated runs it took to get there, in the same
evidence-traceable style as the rest of this story.

**Provenance of this document:** drafted from the project's own records — the session
handoff devlog from the pre-open-source sessions (primary source), the normative design plan
(`design-plan-freeciv-v1.1.md`, its decision record and §7.4 amendment), the per-patch
gate record (`docs/gate-record.md`), and the preserved run artifacts under
`artifacts/runs/`. Every claim below traces to one of those; nothing is reconstructed
from memory alone.

## What it is based on (OSS disclosure, required by the rules)

The challenge rules ask submissions to state what OSS projects the solution was based
on. The honest answer: **CivRealm** (github.com/bigai-ai/civrealm, GPLv3+), the Freeciv
Gymnasium client environment, vendored at pinned SHAs (upstream `dev` @ `1b144d6` plus
the PR #7 fork branch `civgov-fixes` @ `36ae456`, mirrored fork-deletion-proof into
`vendor/` as the literal first provisioning step), and the **`civrealm/freeciv-web`
Docker image** (freeciv 3.3-dev server, freeciv-web webapp, freeciv-proxy), pinned by
digest. Everything above that layer — the adapter, fog gating, candidate enumeration,
read-back verification, the provenance ledger, the honesty metrics, the comparator, the
paired-run plumbing, the replay viewer — is ours. The PR #7 fixes we depend on are each
classified in writing in `docs/gate-record.md`. This disclosure was written into the
plan on day 0, before any of the work: the rules explicitly contemplate OSS-based
entries, and pre-disclosing the basis removes any ambiguity about what is ours.

## Day 0 — planning under verified rules

Before any code: the official challenge rules text was verified in-repo
(`rules.md`) — one entry per person, a video of at most 2 minutes (hard cap), judged on
complexity / performance / reproducibility. One rule check, one amendment (the video
cap forced WP5 to plan a 90–120 s video). The research register that shaped the design
was verified against primary sources in the same round: CivBench Season #001 (eight
models, ~200-turn games, "reason-but-don't-act" as the largest fault), the CivRealm
ecosystem gotchas (verified in source, with file:line locators), and the survey that
grounded the scoped-positioning decision: no surveyed project ships a model-vs-scripted
provenance ledger with frozen honesty metrics, and no Mistral model played Freeciv in
that field. The plan went through four review rounds before implementation was
approved, and it deliberately wrote down the failure paths in advance: the WP0 gate
(where a base switch was still permitted), the scenario-cut (the honest fallback
deliverable after the gate), and the WP4 evaluation cut ladder.

## WP0 — the spike day (provisioning, vendoring, first proof)

The day was mostly plumbing, and the plumbing fought back in instructive ways:

- **The image's own tags lie.** `latest` (= 1.4.6) was pushed *before* 1.3.2, so a tag
  pin would have been a coin flip. The image is pinned by digest, and the digest is in
  every run manifest.
- **The documented port map is stale.** Upstream's `init_server.py` advertises an 8443
  mapping that nothing in this image build listens on; the compose map (host 8080 →
  container 80) is the correct one. Found by probing, not by reading docs.
- **Upstream code touches the network at import time** (`port_utils` performs an HTTP
  pubstatus fetch when `civrealm` is imported) and imports ray unconditionally on
  mainline. PR #7 makes ray optional; we extended the guard (recorded in the gate
  record) so the whole `civrealm.envs` package imports cleanly with ray absent — which
  is now the validated configuration.
- **The fork-deletion-proof vendoring decision** was made exactly because PR #7 was an
  unmerged PR against a dormant upstream (last push 2024-09-11): if the fork vanished,
  the reproducibility story would have vanished with it. Mirroring both SHAs into
  `vendor/` first removed that risk before anything depended on the stack.
- **The observer trap:** clicking "Join/Observe" on the game details page *joins as a
  player* (`action=multi`). The zero-perturbation observe path is the app's own
  `action=observe` URL pattern; observers never take a player slot. Also: snap Firefox
  refuses a second instance, killing the Selenium/geckodriver route — Playwright
  chromium headless is the capture path. A 10 s clip was recorded and ffprobe-verified
  on day 1, so video footage was smoke-tested six days before it was needed.
- **The lost turn-11 game and the `--quitidle` lesson.** The spike built a real game,
  drove it to turn 11, and exported its checkpoint — and then the game *vanished*:
  freeciv servers in this image quit ~20 s after the player client disconnects
  (`--quitidle 20`), and the load test's own teardown disconnected. The game state on
  port 6300 was lost; publite2 respawned a fresh pregame server in its place. The
  lesson — no live game persists between sessions, only exported checkpoints do —
  shaped everything after: the checkpoint export path became the state-of-record
  mechanism, and the `--quitidle 20` fragility was carried forward into WP3's
  lifecycle design explicitly.
- **Exit:** all seven WP0 items passed, including the raw checkpoint
  export → server restart → load → identity-verification primitive (the one
  non-cuttable capability, proven on day 1), and the per-patch gate record was written
  with six PR #7 entries, each with git-diff/file:line evidence. The independent
  review's two warnings (provision.sh missing plan-mandated steps; the gate record
  missing three of six entries) were fixed and closed the same day.

## WP1 — the adapter, the fog, and the bug the live run caught

The adapter layer (GPLv3+, the only package importing CivRealm) and the offline-tested
core (MIT) were built with the fog invariant enforced from the start: 226 offline tests
including all fog fixtures, world-pair tests (two worlds differing only in hidden
mutable state must produce equivalent policy inputs), and a purity test that keeps
CivRealm imports out of the MIT packages.

Three findings from this phase are the heart of the engineering story:

- **The x-major map transposition bug.** Upstream's map state writes its arrays
  x-major (`status[x, y]`); our first extractor read them transposed. This was not
  caught by unit tests with square fixtures — it was caught by the **live comparator
  run**, where the extracted world visibly disagreed with the real one, and was pinned
  with a non-square regression test so it could never come back. (A square map hides a
  transpose; the regression test uses non-square fixtures for exactly that reason.)
- **Upstream hands out live mutable state.** Turn-state dicts get rebound and map numpy
  arrays are mutated *in place*, so an observation held before a step silently became
  the post-step observation — the worst kind of bug in a fog-gated system, because it
  corrupts exactly the history bookkeeping the fog rules depend on. Fixed with a
  recursive deep-copy at the session boundary (`adapter/session.py`).
- **The read-back quirks.** Upstream's read path cancels a unit's activity before
  querying action probabilities and its restore deliberately skips
  `ACTIVITY_SENTRY` — so a sentry is always undone within the same step, and sentry
  read-back is INCONCLUSIVE-by-quirk (never CONTRADICTED). The fortify read-back had to
  accept `ACTIVITY_FORTIFYING` (the in-effect state), not just the packet echo. Both
  are disclosed in the README rather than papered over.
- **The frozen-table 2-cycle, and the user-approved amendment.** The comparator's
  frozen production ladder — "defensive unit if city undefended → builder → growth →
  science → wealth" — alternated Warriors → Barracks → Warriors every single turn in
  the live 20-turn run and completed nothing. The comparator was *faithful to the
  frozen wording*; the wording was missing a case the upstream API hides: a city's
  CURRENT production target never appears in `available_actions`, so the ladder could
  never see that a tier was already satisfied. The fix was not a code hack but a plan
  amendment — production hysteresis (a tier is SATISFIED if the current target belongs
  to its output class; a satisfied top tier does not switch) — **approved by the user
  and dated 2026-10-24** in the plan's §7.4, implemented in `production_pick`, covered
  by five new tests, and re-verified live: the post-amendment run shows 1 produce
  switch in 20 turns versus 19 before, with production actually completing.
- **The tech-cost degradation** was verified live in this phase: tech costs are simply
  not exposed through CivRealm's ruleset packets (all `None`), so the research
  tie-break ("lowest cost") degrades to list order. Disclosed; carried into the README.
- **Exit:** comparator 20/20 turns unattended in 18.0 s with 0 whitelist violations;
  review verdict "exit with notes", both fixed same day.

## WP2 — the model loop, the ledger, and the 8,192-token wall

- **The enforcement spike was decided live, not on paper:** `mistral-large-4` responds
  on the exact pinned ID and supports `json_schema` structured output with the real
  ActionBatch schema — so structured output is the default enforcement, recorded in
  the run manifest. One API quirk cost a fix: greedy sampling (temperature 0) requires
  `top_p=1`.
- **The 8,192-token truncation finding.** The first live prompt version hit the
  completion-token cap on 8 of 17 calls — the model was being too verbose for its own
  good, and truncated output is unusable output. The fix was a brevity/output-discipline
  rule in the system and turn-prefix templates (v2, with v1 kept for history); a 3-turn
  spot-check showed 0/3 cap hits versus the 8/17 baseline (directional evidence only —
  the cap rate is a tracked watch item for the evaluation arms).
- **Key loading, done narrowly.** The model client loads `MISTRAL_API_KEY` from the
  process env first, else parses **only** the `MISTRAL_API_KEY` line from a repo-root
  `.env` at runtime — a user-approved narrow exception to the standing never-touch-
  `.env` rule. The value is never printed, logged, or written to artifacts, and no
  other `.env` content is ever read into any context.
- **The ledger and metrics** froze as pydantic schema, coverage-checked against every
  honesty metric's required inputs, each metric validated by a hand-computed worked
  fixture. The participation gate, the model-attribution rule (a scripted default
  realizing the model's promise is never credited to the model), and the
  minimum-denominator rule all live in the frozen schema, not in reporting code
  discretion.
- **10 live model-driven turns:** 50 verified / 0 inconclusive / 0 contradicted
  read-backs; attempts-per-turn within the ≤2 cap; participation 9/10 (gate PASS).
  Fault injection (timeout, malformed output, invalid refs ×2 → repair cap → degraded
  default) passed 16/16.
- **Exit feasibility decision:** mean 103.4 s/turn measured — neither cut-ladder
  trigger fired; the 2 pairs × 50 turns target fits one idle day with 1.5× contingency
  (~4.34 h of model-arm time). No ladder reduction.

## WP3 — lifecycle hardening and the paired smoke that took seven attempts

The lifecycle work delivered the unglamorous guarantees: the append-only checksummed
transcript survives SIGKILL mid-write (kill test, torn-tail discard on restart);
teardown disconnects produce bounded FAILED outcomes with artifacts intact; a
mid-mutation disconnect is settled only by full-state reconciliation — never blind
replay; the independent checkpoint export path (stat → copy → re-stat → checksum-compare)
is decoupled from CivRealm's autosave deletion; and the replay viewer renders every
transcript into a self-contained static HTML bundle.

**The paired smoke — seven root-caused attempts.** The paired-smoke primitive (fresh
deterministic game → checkpoint export at turn 5 → comparator arm and model arm, each 3
turns from the identical checkpoint under the predeclared paired contract) is the exact
scenario-cut machinery the evaluation depends on, so it was proven before evaluation
day. Its run log (`artifacts/runs/wp3-paired-smoke/run_stdout.log`) preserves seven
orchestrator attempts before the PASS. The failure modes, as preserved in the log
(the root-cause work itself happened in-session; the log records what failed, the
surviving retry shows it was resolved):

1. **Create phase: a step exceeded its 30 s deadline** — and the guarded close then
   *refused to close a failed session* (fail-safe behaving as designed), aborting the
   attempt.
2. **Create phase: `startup failed: TypeError: 'NoneType' object is not subscriptable`**
   — a startup-path crash surfacing from inside the upstream session setup.
3. **Create phase: the same startup failure again** — repeated before the root cause
   was found.
4. **Comparator arm: `load_state_matches_export` FAIL** — the arm's first-turn state
   did not canonically equal the export-time state. This is the paired contract's most
   important check (both arms must provably play from the identical checkpoint), and it
   caught a real export/load state mismatch rather than letting a silently-divergent
   pair through.
5. **Create phase: the startup failure recurred** once more.
6. **Comparator arm: `load_state_matches_export` FAIL again**, then the pair ended
   **`arms_sequential` FAIL** — the model arm's start window overlapped the comparator
   arm's end by a second; an off-by-one boundary in the sequential-arm check rather
   than actual concurrency.
7. **PASS.** All ten acceptance checks green: identical checkpoint recorded *and
   loaded* by both arms, arms sequential, identical predeclared contract hash in both
   manifests, early-terminal reporting armed, both transcripts valid, fingerprints
   recorded with the drift policy applied, and replay bundles rendered for both arms.

The point of preserving all seven: the machinery that makes evaluation claims
trustworthy (checkpoint identity, sequential execution, contract hashing) is exactly
the machinery that failed loudly and was fixed — rather than a machinery that would
have let a silently-divergent pair through to evaluation day.

## Evaluation day

Evaluation day (2026-10-24) took three runs. The machinery that WP3 had proven —
identical checksummed checkpoints, sequential arms, predeclared contract hashes, the
acceptance-check battery — worked from the first attempt; what failed was the
interface between the frozen turn deadline and the model's real latency tail. Each
run's artifacts are preserved unmodified under `artifacts/runs/`, including the two
truncated ones: infrastructure truncations are reported as TRUNCATED, never scored as
strategic losses.

**Run 1 — `artifacts/runs/wp4-eval/`: both model arms TRUNCATED at 0/50.** The
recorded reason in both arm manifests is exact: `infrastructure deadline exceeded:
state ACTING exceeded its deadline of 180.0s (artifacts intact)`. The diagnostic: the
180 s ACTING deadline had been frozen into the runner's supervision state machine back
when WP2 measured a mean of 103.4 s per live turn (the WP2-exit feasibility
decision, recorded in the handoff devlog), but evaluation-day turns kept landing in the ~175 s timeout band — initial
attempts consumed nearly the whole deadline, leaving no room for a budgeted repair
attempt inside the turn. The deadline is part of the frozen paired contract, so the
fix could not be "raise the number". The decision was to **keep the 180 s deadline**
and make repairs budget-aware instead: a repair attempt is only attempted when the
remaining turn deadline can actually fit it, rather than unconditionally burning the
second attempt against a clock that has already run out.

**Run 2 — `artifacts/runs/wp4-eval2/`: still truncated, deeper into the horizon.**
p1 reached 8/50 and p2 12/50 before the same recorded deadline reason ended the arms
(`eval_manifest.json` in the run directory). The cause was found mid-run: the
transport SDK's own internal retry was silently doubling a budgeted repair — one
logical repair attempt was consuming two turns' worth of the latency tail, blowing the
180 s deadline that the budget-aware gating had just accounted for. The doubling was
removed (one attempt is one attempt), the fix was made mid-run, and the truncated run
was preserved as-is rather than quietly discarded.

**Run 3 — `artifacts/runs/wp4-eval3/`: the evaluation of record.** The full 2 pairs ×
50 turns shape completed: all four arms COMPLETED (50/50, survived, no early
terminals), both pairs PASS, 180 model-arm API calls for a total of **$2.45**
(1,744,241 tokens in / 1,055,018 out; ~2.9 h wall-clock on the idle machine). Two
findings defined the day's honest bottom line:

- **The comparator looked broken, and wasn't.** Both comparator arms executed only 3
  actions in 50 turns — at first read, a wiring bug (the wrong `decide` entry point,
  say). The diagnosis (`artifacts/runs/wp4-comparator-diagnosis/`) proved otherwise
  with a live probe: loading the p1 checkpoint (sha256-verified) onto a fresh server
  and calling the published `scripted.decide` directly reproduces the arm
  bit-for-bit — turn 5's two decisions (`city:111:produce_barracks`,
  `unit:114:fortify`) are identical to what the arm executed, and turns 6–7 yield
  zero decisions on the same candidate tables. On checkpoint-loaded small-empire
  states the published policy reaches a fixed point after its first turn (research
  already on the priority target, production hysteresis-satisfied, and no
  move-the-settler provision in the civilian ladder). Not a wiring bug → no rerun, no
  behavior change; the paired comparison is model-vs-that-baseline, disclosed as a
  limitation in the README.
- **The participation gate split the pairs.** p2 PASS at 0.94; p1 FAIL at 0.64. The
  p1 report measures the driver rather than hand-waving it: 21 of 88 turn calls failed
  with transport read-timeouts, each such turn falling to the degraded default. Per
  the plan's rule (§3 line 38), p1 is labeled **degraded-dominant** — reported in
  full, excluded from model-performance claims. No model-ranking language anywhere:
  this is an honest pilot of one model, and the reports say so in their headers.

The cost of getting here, honestly: three runs for the price of one evaluation —
$0.55 spent on run 1 and $0.36 on run 2 bought the two fixes that made run 3
complete, and both truncated runs remain in the tree as evidence that the harness
reports infrastructure failure honestly (an infrastructure TRUNCATED with intact
artifacts is a correct outcome, per the plan's decision 8). The full per-pair
reports, per-turn tables, divergence curves, and acceptance-check lists live in
`artifacts/runs/wp4-eval3/`; the README's Evaluation section carries the headline
numbers and the limitations list.
