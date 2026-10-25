# FW-10 schema-coverage addendum — typed commitment targets + additive verb

- **Date:** 2026-10-24. **Package:** FW-10 (plan §5, harness-fixes-v1, chain S
  position 1). **Status:** ADDITIVE AMENDMENT to the frozen WP2 schema —
  recorded here per the plan's schema-addition rule ("the only schema
  addition is the user-approved additive `Commitment.verb` plus typed
  targets, recorded in a NEW dated schema-coverage addendum").
- **Amends:** `artifacts/runs/wp2-schema-coverage/schema-metric-coverage.md`
  (the frozen coverage check, 2026-10-24). This addendum is a NEW dated
  artifact; nothing under `artifacts/runs/wp2-schema-coverage/` was
  rewritten or removed.
- **Code:** `src/lechonk/core/schemas.py` (`Commitment` + the
  `COMMITMENT_TARGET_KINDS` vocabulary + the freeze-docstring amendment
  note), `src/lechonk/core/metrics.py` (`_target_match` rewritten to
  kind-aware exact equality), `src/lechonk/model/prompts.py` +
  `src/lechonk/model/prompts/system_v3.txt` (system prompt grammar v3).

---

## 1. The additive change (exact)

`Commitment` (frozen WP2 field set: `domain`, `target`, `deadline_turn`)
gains exactly two fields, both with defaults; `extra="forbid"` is retained
(real fields with defaults are added — unknown fields are never accepted):

| field | type | default | meaning |
|---|---|---|---|
| `target_kind` | `Literal["tech","production","unit","city","player","tile","any"]` | `"any"` | the typed target family |
| `verb` | `Optional[str]` | `None` | the intended action verb |

Nothing is removed or renamed; no frozen record kind, field, or literal
vocabulary changed. Pinned by
`tests/test_schemas.py::test_commitment_fw10_additivity_only_named_fields_added`
and `tests/test_schema_freeze.py::test_fw10_commitment_amendment_is_additive_over_the_freeze`;
the WP2 freeze test (`tests/test_schema_freeze.py`) still passes.

## 2. Typed target grammar

The `target_kind` vocabulary and the canonical `target` forms — the
action-side descriptor forms `runner._describe` already emits
(`runner.py:1232-1277`, `:1198-1218`), reused unchanged:

| `target_kind` | canonical `target` form | example | realized by |
|---|---|---|---|
| `tech` | tech name | `"Pottery"` | tech research actions (`research_*`, `set_tech_goal_*`) |
| `production` | production name | `"Warriors"` | city `produce_*` actions |
| `unit` | `"unit:<id>"` | `"unit:7"` | actions ordering that unit |
| `city` | `"city:<id>"` | `"city:3"` | actions on that city |
| `player` | `"player:<id>"` | `"player:2"` | dipl actions toward that player |
| `tile` | `"tile:<x>,<y>"` | `"tile:12,34"` | tile-targeting actions (`goto_*`, `attack_*`, `build_city`, `city_work_*`) |
| `any` | free text (legacy) | `"Pottery"` | exact string equality only (legacy rule, §4) |

`"*"` as the `target` is the wildcard and matches any action target; it is
retained on every kind.

## 3. Matching semantics (kind-aware exact equality — no substring)

`metrics._target_match` is rewritten (this FW-10, not deferred to FW-11 —
the change was trivially separable: one predicate, one caller, and every
existing fixture/board test already matched exactly or by wildcard):

- the `"*"` wildcard matches any action target;
- otherwise the normalized (case-insensitive, whitespace-stripped)
  commitment target must **EQUAL** the normalized action target — the old
  bidirectional substring match is removed in both directions;
- typed kinds additionally require the action-side descriptor family to
  agree: `tile`/`unit`/`city`/`player` targets must be the typed
  descriptors (`"tile:x,y"` / `"unit:<id>"` / `"city:<id>"` /
  `"player:<id>"`); `tech` targets must be realized by a tech-domain
  action, `production` targets by a city `produce_*` action.

The commitment-domain gate (`_COMMITMENT_DOMAIN_MAP`) is unchanged.
**Verb compatibility is NOT yet enforced in the predicate** — FW-11 (chain
S position 2) implements it from the mapping in §5; the `verb` field is
stored from this package onward.

**Caveat (plan §7, B1 assumption):** re-running metrics over old recorded
ledgers can change previously published numbers — substring-only matches
vanish (a commitment `"Pott"` no longer fulfills an action targeting
`"Pottery"`).

## 4. Legacy fallback (pre-addendum records)

Free-text targets from pre-addendum records load with kind `"any"` (the
field default) and match by **exact string equality after normalization**.
No backfill migration of old ledger data. Old-schema instances still load
because the amendment is additive with defaults.

## 5. Verb grammar + commitment-verb → action-verb compatibility mapping (for user sign-off)

`verb: Optional[str] = None` — unset means any action (backward
compatible). Concrete vocabulary: **research, build, attack, move,
negotiate, defend**.

| commitment `verb` | compatible action verbs |
|---|---|
| `research` | `research_*`, `set_tech_goal_*` (tech domain) |
| `build` | `build_city`, `produce_*`, `build_road`, `irrigation`, `mine` |
| `attack` | `attack_*` |
| `move` | `goto_*` |
| `negotiate` | `stop_negotiation`, `cancel_treaty`, `accept_treaty` |
| `defend` | `fortify`, `sentry` |

Recorded here for user sign-off; enforcement lands in FW-11 (exact match
after normalization, or this verb-class mapping).

## 6. System prompt grammar v3

`SYSTEM_PROMPT = "system_v3"` (`src/lechonk/model/prompts.py`); the
template is `src/lechonk/model/prompts/system_v3.txt` (`system_v1` /
`system_v2` retained on disk for history, hashes pinned). v3 = v2 + the
typed commitment grammar (interface rule 3): the `target_kind` vocabulary,
the canonical target forms, the exact-match rule, and the verb vocabulary
with the compatibility mapping from §5. Prompt hash:
`sha256:e549096dd6b3dec92aa12fd71588930dd3d4687a59aabb9da0887afcad1dd46b`
(pinned in `tests/test_prompts.py` `RECORDED_HASHES`).

## 7. Verification

- Acceptance command (actual output recorded below):
  `uv run --project /home/pav/code/lechonkgameharness --no-sync python -m pytest tests/test_schemas.py tests/test_metrics.py tests/test_schema_freeze.py tests/test_prompts.py -q`
- The freeze test still passes (additive only).
- Explicit tests: a legacy `Commitment` payload (old field shape) loads
  and matches by the documented legacy rule
  (`tests/test_metrics.py::test_fw10_legacy_commitment_payload_loads_and_matches`,
  `tests/test_schemas.py::test_commitment_fw10_legacy_payload_loads_with_legacy_defaults`);
  an explicit additivity assertion — only the named fields added
  (`tests/test_schemas.py::test_commitment_fw10_additivity_only_named_fields_added`);
  the dated addendum artifact exists
  (`tests/test_schema_freeze.py::test_fw10_dated_schema_coverage_addendum_exists`).
- No existing artifact under `artifacts/runs/**` was modified; this
  addendum is a new dated directory, committed with `git add -f` in a
  separate evidence-only commit (FW-05 precedent: 20c99b9).

```
$ uv run --project /home/pav/code/lechonkgameharness --no-sync python -m pytest tests/test_schemas.py tests/test_metrics.py tests/test_schema_freeze.py tests/test_prompts.py -q
........................................................................ [ 84%]
.............                                                            [100%]
85 passed in 0.16s
```

Full-suite ripple check (same tree): `python -m pytest -q` → 528 passed,
1 failed — `tests/test_mistral_client.py::test_json_schema_path_sends_strict_structured_output_request`
(a PRE-EXISTING failure at HEAD 8f723d2, unrelated to FW-10: FW-06 split the
candidate-ID end anchor into `CANDIDATE_ID_REGEX` (`\Z`, Python `re`) and
`CANDIDATE_ID_FIELD_PATTERN` (`\z`, pydantic's Rust regex) and that test —
last touched in the initial commit 6939f3e — still asserts the JSON-schema
pattern equals `CANDIDATE_ID_REGEX`. No FW-10 file is involved: the diff
touches neither the regex constants, nor `mistral_client.py`, nor that
test.)
