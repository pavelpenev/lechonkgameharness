# Per-patch gate classification record — WP0

Date: 2026-10-24. Required by the normative plan `design-plan-freeciv-v1.1.md`
(WP0 section, line 50: "Per-patch classification at the gate, in writing: each
PR #7 item (incl. the AcceptTreaty commit) is applied / not-needed-for-our-surface
/ failed"). The plan names no file location for the record; this file
(`docs/gate-record.md`) was created for it.

Vendored tree state at classification time:
`vendor/civrealm` git HEAD = `36ae456` (PR #7 fork branch `civgov-fixes`
@ `36ae456` over upstream `dev` @ `1b144d6`), verified via `git rev-parse HEAD`
and `git log`; the working tree was clean then. Post-classification, the
ray-guard extension (entry 1) was committed on the pavelpenev/civrealm
mirror as branch `lechonk-harness` @ `8eb6dba` ("Make
FreecivParallelEnv/FreecivA3CEnv imports optional (ray intentionally
absent)"), and `vendor/civrealm` is now registered as a git submodule of
this repository pinned at that commit.

---

## 1. Ray-import guard extension

- **File:** `vendor/civrealm/src/civrealm/envs/__init__.py`
- **Classification:** applied-by-us
- **Date:** 2026-10-24
- **Rationale:** the ray-guard patch is ours; it extends PR #7's partial
  ray-guard so `civrealm.envs` imports cleanly with ray absent.
- **Evidence:**
  - Upstream `dev` @ `1b144d6` imported `ParallelTensorEnv`
    unconditionally (`git diff 1b144d6 -- src/civrealm/envs/__init__.py`).
  - PR #7 commit `bb2cdf1` ("... make ray optional") wrapped only the
    `ParallelTensorEnv` import in try/except (`git show bb2cdf1 --
    src/civrealm/envs/__init__.py`); it still imported
    `freeciv_parallel_env` / `freeciv_a3c_env` unconditionally.
  - Our change, now committed on the pavelpenev/civrealm mirror as
    `8eb6dba` on branch `lechonk-harness` (`git show 8eb6dba --
    src/civrealm/envs/__init__.py`), wraps the `freeciv_parallel_env` /
    `freeciv_a3c_env` imports in try/except, setting
    `FreecivParallelEnv = None` and `FreecivA3CEnv = None` on ImportError —
    see the file at that commit, lines 12–18.
  - Both modules import `ray` at module level
    (`freeciv_parallel_env.py:19`, `freeciv_a3c_env.py:18`), so without the
    second guard `import civrealm.envs` fails when ray is absent.
  - Behavioral check, run 2026-10-24 with `.venv/bin/python` (ray not
    installed): `import civrealm.envs` succeeds; `ParallelTensorEnv`,
    `FreecivParallelEnv`, `FreecivA3CEnv` are all `None`;
    `FreecivBaseEnv` resolves to
    `civrealm.envs.freeciv_base_env.FreecivBaseEnv`.

## 2. AcceptTreaty commit

- **File (commit):** PR #7 fork commit `36ae456` "Accept a pact clause the
  counterpart proposed" — changes
  `src/civrealm/freeciv/players/diplomacy_actions.py` (AcceptTreaty
  validity check, ~lines 558–566) and adds
  `tests/dipl/test_accept_pact_from_counterpart.py`.
- **Classification:** not-needed-for-our-surface
- **Date:** 2026-10-24
- **Rationale:** the patch fixes the validity check for accepting a pact
  proposed by the counterpart, but our v1 action surface exposes diplomacy
  as read-status + decline/exit only — treaty acceptance is not a verb we
  use.
- **Evidence:**
  - Diff verified via `git show 36ae456 -- src/civrealm/freeciv/players/diplomacy_actions.py`:
    the fix relaxes `AcceptTreaty.is_action_valid` (pacts have no direction;
    `AddClause` lists them only with us as giver, so counterpart-proposed
    pacts previously read as never acceptable) and adds a diplomacy unit test.
  - Applied: the commit is the vendored HEAD (`36ae456`), so the fix is
    present in the vendored tree; the classification is about relevance to
    our surface, not presence.
  - Surface: plan `design-plan-freeciv-v1.1.md` action surface (§3, line 37)
    exposes diplomacy as "read status; decline/exit"; treaty acceptance is
    not in the v1 verb set (line 147 lists the explicitly-not-exposed set).
  - Plan line 7 pre-declared the same expected classification
    ("expected gate classification: not-needed-for-our-surface").

## 3. ActExplore

- **File:** `vendor/civrealm/src/civrealm/freeciv/units/unit_actions.py`
  (class `ActExplore` at line 868; registration list at lines 275–288)
- **Classification:** excluded
- **Date:** 2026-10-24
- **Rationale:** ActExplore is defined but left unregistered because the
  freeciv-web server build refuses `SSA_AUTOEXPLORE`; it stays out of the v1
  action surface per plan decision 1 and §3 exclusions.
- **Evidence:**
  - `unit_actions.py:283–284` (committed in PR #7 `bb2cdf1`, verified via
    `git show bb2cdf1 -- src/civrealm/freeciv/units/unit_actions.py`):
    "ActExplore: the freeciv-web server build refuses SSA_AUTOEXPLORE
    silently (unit_server_side_agent_set returns FALSE), so it stays
    unregistered" — `ActExplore` is commented out of the registration loop.
  - `unit_actions.py:868–883`: the class exists and would send
    `packet_unit_server_side_agent_set` with `agent=SSA_AUTOEXPLORE`
    (`fc_types.py:30`), but is never registered as an available action.
  - Plan `design-plan-freeciv-v1.1.md` line 7: PR #7 attempted the fix and
    left it unregistered; the server refuses `SSA_AUTOEXPLORE`
    (`unit_server_side_agent_set` returns FALSE); auto-explore stays out of
    the v1 action surface. §3 (line 37) lists exploration among the excluded
    verbs.
  - Handoff corroboration: the prior session's handoff devlog records the
    finding "ActExplore = excluded (server refuses SSA_AUTOEXPLORE)".
  - Spike coverage: grep for `AUTOEXPLORE|ActExplore` (case-insensitive)
    across `scripts/` and `artifacts/` returned no matches — no spike
    artifact exercises ActExplore, consistent with its exclusion.

## 4. research_name wrong-lookup fix

- **File:** `vendor/civrealm/src/civrealm/freeciv/players/player_state.py`
  (fix at lines 73–74)
- **Classification:** applied
- **Date:** 2026-10-24
- **Rationale:** PR #7 commit `bb2cdf1` corrects the `research_name` lookup
  from `governments` to `techs`; the fix is committed at the vendored HEAD,
  so the player observation's research name resolves from the tech table as
  the plan requires.
- **Evidence:**
  - Upstream `dev` @ `1b144d6` looked up `player_state['researching']` in
    `self.rule_ctrl.governments` and read `governments[...]['name']` —
    a tech id never matches a government entry, so `research_name` stayed
    `None` (verified via `git diff 1b144d6 HEAD --
    src/civrealm/freeciv/players/player_state.py`).
  - PR #7 commit `bb2cdf1` ("Fix research_name lookup, register
    autosettlers/sentry via server-side agent packet, make ray optional")
    changes both the guard and the lookup to `self.rule_ctrl.techs`;
    `git log -L 73,74:src/civrealm/freeciv/players/player_state.py
    1b144d6..HEAD` pins the change to that commit.
  - Working tree `player_state.py:73–74`:
    `if player_state['researching'] in self.rule_ctrl.techs:` /
    `player_state['research_name'] =
    self.rule_ctrl.techs[player_state['researching']]['name']`. The
    ray-guard extension is committed (commit `8eb6dba` on branch
    `lechonk-harness`) and the working tree carries no uncommitted
    `envs/__init__.py` modification, so the committed fix is what ships.
  - Plan line 137 records the same mainline break (`player_state.py:73–74`,
    looked up in governments) and gates our snapshot's research display on
    this fix being verified applied.

## 5. ActSEntry sentry re-registration

- **File:** `vendor/civrealm/src/civrealm/freeciv/units/unit_actions.py`
  (class `ActSEntry` at line 480; registration list at lines 278–287,
  `ActSEntry` active at line 282)
- **Classification:** applied
- **Date:** 2026-10-24
- **Rationale:** mainline left `ActSEntry` commented out of the registration
  loop; PR #7 commit `bb2cdf1` re-registers it, and `sentry` is in our v1
  verb set (plan line 143), so this entry records the plan's explicit
  "verify at the per-patch gate" check for the re-enable.
- **Evidence:**
  - Upstream `dev` @ `1b144d6`: `ActSEntry` appears only in the
    commented-out tail of the registration list (verified via `git diff
    1b144d6 HEAD -- src/civrealm/freeciv/units/unit_actions.py`: the
    removed line is `# ActTileInfo, ActActSel, ActSEntry, ActWait, ActNuke`).
  - PR #7 commit `bb2cdf1` moves `ActAutoSettler, ActSEntry,` into the
    active registration list (`unit_actions.py:282`) and adds
    `is_action_valid` to `ActSEntry` (`unit_actions.py:483–485`: a unit
    already in `ACTIVITY_SENTRY` cannot be sentried again).
  - `ActSEntry._action_packet` (`unit_actions.py:487–488`) sends
    `packet_unit_change_activity` with `ACTIVITY_SENTRY` / `EXTRA_NONE` —
    the activity packet, not `packet_unit_server_side_agent_set`, so the
    re-enabled verb does not depend on the server-side-agent path whose
    `SSA_AUTOEXPLORE` refusal excludes ActExplore (entry 3).
  - Gate verification (plan line 143): registration in the active list,
    the validity gate, and the packet construction were all confirmed at
    source level in the vendored tree at `36ae456`.
  - Live behavioral verification (WP1 package B smoke, 2026-10-24, fresh
    deterministic reset seed 20261025; evidence under
    `artifacts/runs/wp1-adapter/`): the verb is available live — 5 sentry
    candidates in the turn-1 candidate table, raw upstream
    `available_actions['unit'][uid]['sentry'] == True` for units
    101/103/104/105/106 — and a sentry order was sent to a live game
    (`unit:103:sentry` -> `('unit', 103, 'sentry')`); the step completed
    cleanly. Source-verified quirk (`unit_actions.py:183-226`): the read
    path's `cancel_orders_before_query_pro` cancels a unit's activity
    before querying action probabilities and `restore_activity` never
    restores `ACTIVITY_SENTRY`, so a sentry is always undone within the
    same step; the observation cannot distinguish sent-then-cancelled
    from never-sent, and the adapter's sentry read-back is therefore
    INCONCLUSIVE-by-quirk (never CONTRADICTED).

## 6. activity / ssa_controller observation fields

- **File:** `vendor/civrealm/src/civrealm/freeciv/units/unit_state.py`
  (fields at lines 62–63; matching observation-space entries lower in the
  file)
- **Classification:** applied
- **Date:** 2026-10-24
- **Rationale:** PR #7 commit `1effc2d` exposes per-unit `activity` and
  `ssa_controller` in the unit observation, letting a client tell an idle
  unit from a working or server-agent-controlled one; both fields are
  present and committed at the vendored HEAD.
- **Evidence:**
  - Upstream `dev` @ `1b144d6` lacked both fields (verified via `git diff
    1b144d6 HEAD -- src/civrealm/freeciv/units/unit_state.py`).
  - PR #7 commit `1effc2d` ("Expose unit activity and ssa_controller in the
    unit observation") adds `unit_state['activity'] =
    punit.get('activity', 0)` and `unit_state['ssa_controller'] =
    punit.get('ssa_controller', 0)` (`unit_state.py:62–63`) plus
    corresponding `gymnasium.spaces.Box` entries in the observation space
    declaration.
  - The same fields feed PR #7's action-validity gates elsewhere
    (`unit_actions.py:332`, `:341`, `:857`, `:873` all check
    `punit['ssa_controller']`), consistent with the re-registrations in
    entry 5.
  - Plan line 138 records the expectation: "PR #7 adds `activity` +
    `ssa_controller`" — matches the vendored tree.

---

## Unverified items

- The freeciv-web server's live refusal of `SSA_AUTOEXPLORE`
  (`unit_server_side_agent_set` returning FALSE) was **not re-verified
  against the running container** in this session — it would require sending
  an action to the live turn-11 game. The claim rests on the PR #7 committed
  code comment (`unit_actions.py:283–284`), the plan's earlier verification
  (line 7, WP0 line 50), and the prior session's HANDOFF record.
- The vendored `.git` history was used as the provenance record; the remote
  PR #7 was not fetched (network lookup deliberately not performed; the plan
  requires the fork to be mirrored fork-deletion-proof into `vendor/`, which
  is what was inspected).
- The plan specifies the gate record must exist "in writing" but gives no
  location or format for it; `docs/gate-record.md` was created for this
  purpose (first use of a `docs/` directory in this repo).
