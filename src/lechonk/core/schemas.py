# SPDX-License-Identifier: MIT
"""lechonk.core.schemas — typed policy-facing schemas (MIT).

Per design-plan-freeciv-v1.1.md §7.3. The TranscriptRecord union and the
RunManifest field set are FROZEN as of WP2 (2026-10-24): the record kinds are
run_manifest / turn_start / snapshot_ref / candidates_ref / model_call /
action_result / commitment / metric_event / outcome, each carrying
``seq``, ``ts``, ``checksum`` (assigned by core.ledger at append time), and
``outcome`` carries COMPLETED/DEFEATED/VICTORY/TRUNCATED + reason.

The freeze is coverage-checked: every §3 honesty-metric input (participation
gate incl. per-domain shares, commitment follow-through with censoring and
dedup, reason-but-don't-act with its ±2-turn window and defensive-response
definition, the illegal-action taxonomy, the minimum-denominator rule) is
enumerated against this schema in
artifacts/runs/wp2-schema-coverage/schema-metric-coverage.md — the gaps the
check found were fixed in this schema BEFORE the freeze and are disclosed
there (per-record turn on the ref/action records, structured action identity
``domain``/``verb``/``target``/``affected``/``actor_tile``, provenance linkage
``origin_seq``/``snapshot_seq``/``candidates_seq``/``call_seq``, verification
evidence ``readback``, ``scripted_default`` reason, the ``commitment`` record
carrying the batch's threat hypotheses, per-domain offered counts on
``candidates_ref``).

Additive amendment (2026-10-24, FW-10 — dated addendum:
artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/addendum.md):
``Commitment`` gains ``target_kind`` (Literal ``tech``/``production``/
``unit``/``city``/``player``/``tile``/``any``, default ``"any"``) and
``verb`` (Optional[str], default None) — the typed commitment targets and
the user-approved verb field. Additive only: pre-addendum payloads load
unchanged (legacy free-text targets land on kind ``"any"``); no frozen
record kind, field, or vocabulary is removed or renamed.

This module NEVER imports civrealm (test-enforced: tests/test_core_purity.py).

Token budgets (§3): Snapshot ≤ 8,000 tokens rendered, Delta ≤ 800,
strategy memory ≤ 1,500.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from lechonk.core.jsonutil import strict_dumps

# §3 numeric budgets (falsifiable commitments).
SNAPSHOT_TOKEN_BUDGET = 8000
DELTA_TOKEN_BUDGET = 800
MEMORY_TOKEN_BUDGET = 1500
ACTION_BATCH_MAX_ACTIONS = 20
# Candidate-table render cap (§3 line 32 overflow policy): the rendered
# table may not exceed this budget; armed-in-contact candidates are never
# dropped, the rest are elided lowest-priority-first.
CANDIDATE_TABLE_TOKEN_BUDGET = 1200


class CandidateTableCapExceeded(ValueError):
    """Armed-in-contact candidates alone exceed the candidate-table cap —
    an explicit misconfiguration (never a silent drop, FW-13)."""

# Candidate IDs are opaque tokens of the form ``domain:actor:verb``
# (e.g. ``unit:7:goto_n``, ``city:3:produce_settlers``,
# ``tech:player:research_alphabet``, ``dipl:2:stop_negotiation``).
# ActionBatch actions reference candidate IDs ONLY: a raw coordinate, a free-text
# move description, or any extra field is a schema violation (§7.3: "raw
# coordinates from the model are rejected as schema-error").
CANDIDATE_ID_REGEX = r"^(unit|city|tech|dipl):[A-Za-z0-9_]+:[a-z0-9_]+\Z"
# Pydantic v2 compiles Field patterns with the Rust regex crate, which
# rejects Python's \Z; its end anchor is \z.
CANDIDATE_ID_FIELD_PATTERN = r"^(unit|city|tech|dipl):[A-Za-z0-9_]+:[a-z0-9_]+\z"
CandidateRef = Field(pattern=CANDIDATE_ID_FIELD_PATTERN)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate for rendered JSON text.

    ~4 characters per token for JSON-ish text. The WP2 pipeline replaces this
    with the real tokenizer measurement; the budget checks must hold under both.
    """
    return max(1, -(-len(text) // 4))


# ---------------------------------------------------------------------------
# Snapshot (§7.3, ≤8,000 tokens rendered)
# ---------------------------------------------------------------------------


class ResearchView(BaseModel):
    """Own research display: current target + progress + turns-left.

    The name is computed from the tech domain (cross-checked), never trusted
    from the upstream ``research_name`` field alone (plan §7.2).
    """

    current_target: Optional[str] = None
    current_target_id: Optional[int] = None
    bulbs_researched: Optional[int] = None
    cost_total: Optional[int] = None
    turns_left: Optional[int] = None


class PlayerView(BaseModel):
    player_id: int
    name: Optional[str] = None
    nation: Optional[str] = None
    score: Optional[int] = None
    gold: Optional[int] = None
    tax_rate: Optional[int] = None
    science_rate: Optional[int] = None
    luxury_rate: Optional[int] = None
    government: Optional[str] = None
    is_alive: Optional[bool] = None
    techs_researched: Optional[int] = None
    research: ResearchView = Field(default_factory=ResearchView)


class ResearchCandidate(BaseModel):
    tech_id: int
    name: str
    cost: Optional[int] = None  # bulbs; None when the ruleset cost is unavailable
    note: str = ""  # e.g. "requires: X, Y"


class CityView(BaseModel):
    id: int
    name: str
    x: int
    y: int
    is_own: bool
    owner_id: Optional[int] = None  # None = unknown (fogged foreign city)
    owner_name: Optional[str] = None
    visibility: Literal["visible", "last_known"] = "visible"
    # Exact coordinates plus relative direction/distance to the capital
    # (SAGA lesson: exact coords AND relative statements, no inference).
    rel_to_capital: Optional[str] = None  # e.g. "2E 1S (dist 2)"; None if no capital
    size: Optional[int] = None  # None = unknown (fogged foreign city)
    food_surplus: Optional[int] = None
    shield_surplus: Optional[int] = None
    production: Optional[str] = None  # current production name; None if nothing/unknown
    production_turns_left: Optional[int] = None
    buy_cost: Optional[int] = None
    # WP1 package C (§7.4 buy rule: "shield stock < 25% complete"): the stored
    # shields and the fraction of the current production already paid for
    # (shield_stock / build cost, as upstream's prod_process reports it).
    # None when unknown or when nothing is in production.
    shield_stock: Optional[int] = None
    production_progress: Optional[float] = None
    # WP2 schema freeze (HANDOFF open item reconciled): the extractor lists
    # the city's buildable unit names ALPHABETICALLY, capped at the snapshot
    # token budget (obs_extract MAX_PRODUCTION_CANDIDATES) — NOT "top-N by
    # priority list" as §7.3's draft wording said; the priority ladder lives
    # in the comparator (§7.4), not in the observation. The cap is disclosed
    # via omitted.production_candidates_capped.
    production_candidates: list[str] = Field(default_factory=list)  # buildable names


class MoveCandidate(BaseModel):
    direction: str  # compass: N NE E SE S SW W NW
    terrain: Optional[str] = None  # destination terrain when known
    note: str = ""  # e.g. "visible", "last known", "unexplored"


class AttackCandidate(BaseModel):
    direction: str
    target_unit_id: Optional[int] = None
    target_type: Optional[str] = None
    odds: Optional[float] = None  # attacker win probability, when exposed


class UnitView(BaseModel):
    id: int
    owner_id: Optional[int] = None
    owner_name: Optional[str] = None
    is_own: bool
    type_name: str
    x: int
    y: int
    # Terrain name of the unit's own tile, fog-gated: None when the tile is
    # not currently visible (terrain is written only for seen tiles,
    # map_state.py). Own units' tiles are always visible to their owner.
    # (WP1 package C: the §7.4 improve-by-terrain-yield rule reads this.)
    terrain: Optional[str] = None
    hp: Optional[int] = None  # None = unknown (revealed-but-not-visible foreign unit)
    moves_left: Optional[int] = None  # None = unknown (foreign placeholder -1)
    activity: Optional[int] = None  # raw activity id (PR #7 field)
    activity_name: Optional[str] = None
    ssa_controller: Optional[int] = None
    veteran: Optional[bool] = None
    # Fog-honesty markers.
    visibility: Literal["visible", "not_visible"] = "visible"
    # True for armed own units in contact with a visible enemy: never elided
    # by the truncation policy (§3).
    in_contact: bool = False
    move_candidates: list[MoveCandidate] = Field(default_factory=list)
    attack_candidates: list[AttackCandidate] = Field(default_factory=list)


class Landmark(BaseModel):
    kind: str  # "own_city" | "foreign_city" | "enemy_units"
    label: str
    x: int
    y: int
    rel_to_capital: Optional[str] = None
    visibility: Literal["visible", "last_known"] = "visible"
    note: str = ""


class MapView(BaseModel):
    """Textual region summaries + named landmarks; exact coordinates included.

    The tile-status trichotomy is preserved end to end:
    0=unexplored / 1=last-known / 2=visible (plan §7.2, map_state.py:119-132).
    """

    width: int
    height: int
    tile_counts: dict[str, int] = Field(default_factory=dict)  # visible/last_known/unexplored
    grid: Optional[str] = None  # compact ASCII rendering when it fits the budget
    landmarks: list[Landmark] = Field(default_factory=list)
    region_notes: list[str] = Field(default_factory=list)


class DiplomacyView(BaseModel):
    counterpart_id: int
    counterpart_name: Optional[str] = None
    state: str  # DS_* name, e.g. "War", "No contact"
    meeting_active: bool = False
    clauses: list[str] = Field(default_factory=list)
    # Embassy-gated intel summary, exactly as upstream exposes it: fields the
    # client does not know render as "unknown" (never 0/-1 placeholders).
    intel: Optional[str] = None


class OmittedDisclosure(BaseModel):
    """Truncation disclosure per §3 line 32 — the snapshot header discloses
    what was omitted. Deterministic, visibility-safe truncation only."""

    summary: str = "nothing omitted"
    elided_unit_ids: list[int] = Field(default_factory=list)
    research_candidates_capped: bool = False
    production_candidates_capped: bool = False
    messages_elided: int = 0
    map_grid_omitted: bool = False


class Snapshot(BaseModel):
    turn: int
    player: PlayerView
    research_candidates: list[ResearchCandidate] = Field(default_factory=list)
    cities: list[CityView] = Field(default_factory=list)
    units: list[UnitView] = Field(default_factory=list)
    map_view: MapView
    diplomacy: list[DiplomacyView] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    omitted: OmittedDisclosure = Field(default_factory=OmittedDisclosure)

    def render(self) -> str:
        """Compact JSON rendering (the policy-facing surface). Strict
        canonical semantics: NaN/Infinity rejected, lone surrogates
        escaped (core.jsonutil)."""
        return strict_dumps(self.model_dump(mode="json"))

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())


# ---------------------------------------------------------------------------
# Delta (§7.3, ≤800 tokens) — frozen in WP2
# ---------------------------------------------------------------------------

DeltaEventKind = Literal[
    "unit_changed",
    "unit_lost_own",  # own unit lost; a vanished FOREIGN unit is never "lost"
    "city_growth",
    "production_complete",
    "research_progress",
    "research_complete",
    "new_contact",
    "meeting",
    "combat",
    "warning",
]


class DeltaEvent(BaseModel):
    kind: DeltaEventKind
    subject: str = ""
    detail: str = ""


class Delta(BaseModel):
    turn: int
    events: list[DeltaEvent] = Field(default_factory=list)

    def render(self) -> str:
        return strict_dumps(self.model_dump(mode="json"))

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())


# ---------------------------------------------------------------------------
# CandidateTable + ActionBatch (§7.3)
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """One enumerated legal candidate. ``id`` matches CANDIDATE_ID_REGEX."""

    model_config = ConfigDict(extra="forbid")

    id: str = CandidateRef
    domain: Literal["unit", "city", "tech", "dipl"]
    actor_id: str
    verb: str
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    # FW-13: armed-in-contact units' candidates are never elided by the
    # candidate-table cap (the runner marks them from the snapshot).
    in_contact: bool = False


def _render_line(c: Candidate) -> str:
    """One candidate-table line: ``id — summary`` (id alone unsummarized)."""
    return f"{c.id} — {c.summary}" if c.summary else c.id


def _omission_line(omitted: int) -> str:
    """The §3 line-32 omission disclosure appended to a capped table."""
    return f"candidates omitted: {omitted} (lowest priority)"


class CandidateTable(BaseModel):
    turn: int
    candidates: list[Candidate] = Field(default_factory=list)

    def by_id(self) -> dict[str, Candidate]:
        return {c.id: c for c in self.candidates}

    def offered(self) -> "CandidateTable":
        """The deterministically capped offered subset (§3 line 32 overflow
        policy): armed-in-contact candidates are never dropped; the rest
        are admitted in table order while they fit
        CANDIDATE_TABLE_TOKEN_BUDGET, reserving room for the omission
        disclosure line. Armed-in-contact candidates alone exceeding the
        cap is a misconfiguration, not a drop — CandidateTableCapExceeded."""
        protected_tokens = sum(
            estimate_tokens(_render_line(c))
            for c in self.candidates if c.in_contact)
        if protected_tokens > CANDIDATE_TABLE_TOKEN_BUDGET:
            raise CandidateTableCapExceeded(
                f"armed-in-contact candidates alone exceed the "
                f"candidate-table cap: {protected_tokens} > "
                f"{CANDIDATE_TABLE_TOKEN_BUDGET} tokens")
        # Reserve the worst-case disclosure line (the omitted count can
        # only shrink as candidates are dropped).
        reserve = estimate_tokens(_omission_line(len(self.candidates)))
        kept: list[Candidate] = []
        used = protected_tokens
        for cand in self.candidates:
            line_tokens = estimate_tokens(_render_line(cand))
            if cand.in_contact:
                # armed-in-contact candidates are never dropped (§3): they
                # are admitted ahead of the budget check.
                kept.append(cand)
                used += line_tokens
                continue
            if used + line_tokens + reserve > CANDIDATE_TABLE_TOKEN_BUDGET:
                continue  # elided: lowest priority, over the capped budget
            kept.append(cand)
            used += line_tokens
        return CandidateTable(turn=self.turn, candidates=kept)

    def render(self) -> str:
        """Compact policy-facing rendering: one ``id — summary`` line per
        offered candidate, capped by CANDIDATE_TABLE_TOKEN_BUDGET with the
        dropped count disclosed in a trailing omission line."""
        offered = self.offered()
        lines = [_render_line(c) for c in offered.candidates]
        omitted = len(self.candidates) - len(offered.candidates)
        if omitted:
            lines.append(_omission_line(omitted))
        return "\n".join(lines)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())


class ActionItem(BaseModel):
    """A single model action: a candidate-ID reference ONLY.

    ``extra="forbid"`` plus the ``ref`` pattern make raw coordinates and
    free-text move descriptions schema errors (§7.3).
    """

    model_config = ConfigDict(extra="forbid")

    ref: str = CandidateRef


class Commitment(BaseModel):
    """Structured commitment (§3 commitment follow-through).

    FW-10 additive amendment (2026-10-24): ``target_kind`` types the target
    and ``verb`` optionally states the intended action. Both carry defaults,
    so pre-addendum payloads (free-text ``target`` only) load unchanged and
    land on the legacy kind ``"any"`` — see the dated addendum
    (artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/).
    Matching is kind-aware exact equality (core.metrics._target_match).
    """

    model_config = ConfigDict(extra="forbid")

    domain: Literal["research", "production", "expansion", "military", "diplomacy", "other"]
    target: str
    deadline_turn: int
    # Typed target (§3 line 40): the action-side descriptor family the
    # target names — tech/production name, "unit:<id>", "city:<id>",
    # "player:<id>", "tile:x,y"; "any" = legacy free text (exact match
    # only); "*" as the target is the wildcard. Canonical forms are the
    # ones runner._describe emits on the action side.
    target_kind: Literal["tech", "production", "unit", "city", "player", "tile", "any"] = "any"
    # Optional intended-action verb (vocabulary: research, build, attack,
    # move, negotiate, defend — see the dated addendum for the
    # commitment-verb -> action-verb compatibility mapping).
    verb: Optional[str] = None


class MemoryUpdate(BaseModel):
    """Per-turn memory section carried by the ActionBatch output (§7.5)."""

    model_config = ConfigDict(extra="forbid")

    commitments: list[Commitment] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)  # entity IDs, no free text
    notes: str = ""


class ActionBatch(BaseModel):
    """Model output for one turn (≤20 actions)."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    actions: list[ActionItem] = Field(default_factory=list, max_length=ACTION_BATCH_MAX_ACTIONS)
    commitments: list[Commitment] = Field(default_factory=list)
    finish_requested: bool = False
    notes: str = ""
    memory_update: Optional[MemoryUpdate] = None


# ---------------------------------------------------------------------------
# TranscriptRecord union + RunManifest (§7.3) — FROZEN in WP2 (2026-10-24)
# ---------------------------------------------------------------------------

# §3 line 38 provenance classes (policy-choice level). Frozen.
ProvenanceClass = Literal[
    "model", "continued_model_order", "scripted_default", "engine_automatic",
    "housekeeping",
]
PROVENANCE_CLASSES: tuple[str, ...] = (
    "model", "continued_model_order", "scripted_default", "engine_automatic",
    "housekeeping",
)
# Provenance classes whose executed decisions count as model participation
# (§3 line 38: "≥1 executed model-class or model-originated
# continued_model_order decision").
MODEL_PARTICIPATION_CLASSES: tuple[str, ...] = ("model", "continued_model_order")

# §3 line 43 illegal-action taxonomy classes. Frozen.
RejectionClass = Literal["schema_error", "state_fog_error", "illegal_in_rule"]
REJECTION_CLASSES: tuple[str, ...] = (
    "schema_error", "state_fog_error", "illegal_in_rule",
)

# Per-action read-back verification evidence (adapter/execute.py statuses).
ReadbackStatus = Literal["verified", "contradicted", "inconclusive", "none"]
READBACK_STATUSES: tuple[str, ...] = (
    "verified", "contradicted", "inconclusive", "none",
)

# §3 line 34 outcome taxonomy. Frozen.
OutcomeKind = Literal["COMPLETED", "DEFEATED", "VICTORY", "TRUNCATED"]
OUTCOME_KINDS: tuple[str, ...] = ("COMPLETED", "DEFEATED", "VICTORY", "TRUNCATED")

# The frozen record-kind list (§7.3).
TRANSCRIPT_KINDS: tuple[str, ...] = (
    "run_manifest", "turn_start", "snapshot_ref", "candidates_ref",
    "model_call", "action_result", "commitment", "metric_event", "outcome",
)

# Candidate domains (§7.2) and the metric domains they roll up to
# (§3 lines 38-41; classification lives in core.metrics.metric_domain).
CANDIDATE_DOMAINS: tuple[str, ...] = ("unit", "city", "tech", "dipl")
METRIC_DOMAINS: tuple[str, ...] = (
    "research", "production", "expansion", "unit_action", "diplomacy",
    "city_other", "housekeeping", "other",
)
# Commitment domains (§3 line 40) — the commitment grammar.
COMMITMENT_DOMAINS: tuple[str, ...] = (
    "research", "production", "expansion", "military", "diplomacy", "other",
)
# Commitment target kinds (FW-10 typed targets, 2026-10-24): the action-side
# descriptor families produced by runner._describe (tech name, production
# name, "unit:<id>", "city:<id>", "player:<id>", "tile:x,y") plus "any" —
# legacy free-text targets, matched by exact equality only. The "*" target
# is the wildcard (any kind). See the dated addendum:
# artifacts/runs/wp2-schema-coverage/2026-10-24-verb-typed-targets/
COMMITMENT_TARGET_KINDS: tuple[str, ...] = (
    "tech", "production", "unit", "city", "player", "tile", "any",
)


class TranscriptRecordBase(BaseModel):
    """One checksummed transcript record (§3 line 34: one union type per
    JSONL line). ``seq``/``ts``/``checksum`` are assigned by core.ledger at
    append time — construct records without them and let the ledger fill
    them in."""

    model_config = ConfigDict(extra="forbid")

    seq: int = 0
    ts: str = ""  # ISO-8601, ledger-assigned
    checksum: str = ""  # per-record checksum over the canonical JSON of the
    # record (checksum field excluded), ledger-assigned


class RunManifestRecord(TranscriptRecordBase):
    kind: Literal["run_manifest"] = "run_manifest"
    manifest: "RunManifest"


class TurnStartRecord(TranscriptRecordBase):
    """The pipeline entered ACTING for ``turn`` (state machine §3 line 34)."""

    kind: Literal["turn_start"] = "turn_start"
    turn: int


class SnapshotRefRecord(TranscriptRecordBase):
    """Reference to the turn's fog-gated snapshot (hash + token count);
    the full snapshot is not duplicated into the transcript."""

    kind: Literal["snapshot_ref"] = "snapshot_ref"
    turn: int
    snapshot_hash: str
    token_count: int


class CandidatesRefRecord(TranscriptRecordBase):
    """Reference to the turn's offered candidate table (the §3 ledger's
    offered-candidates requirement): hash + count, plus the per-domain
    offered counts the participation gate reads (opportunity_domains lists
    the metric domains with ≥1 offered consequential choice this turn)."""

    kind: Literal["candidates_ref"] = "candidates_ref"
    turn: int
    table_hash: str
    candidate_count: int
    by_domain: dict[str, int] = Field(default_factory=dict)
    opportunity_domains: list[str] = Field(default_factory=list)


class ModelCallRecord(TranscriptRecordBase):
    """One model API call (≤2 per turn including at most one repair,
    §3 line 32 / decision 4). ``attempt`` is 1 (primary) or 2 (repair);
    ``error`` is set when the call failed at transport level (timeout etc.)
    — SDK/transport retries are transparent and never counted here."""

    kind: Literal["model_call"] = "model_call"
    call_id: str
    attempt: int = 1
    prompt_hash: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    fingerprint: str
    error: Optional[str] = None


class ActionResultRecord(TranscriptRecordBase):
    """One policy-choice-level provenance record (§3 line 38): the choice
    (``ref`` — a candidate ID, or the raw model-emitted string for
    schema-error rejections), its structured identity, the execution result
    and verification evidence, and the provenance linkage (snapshot ID,
    offered candidates, call ID, origin for continued_model_order, reason
    for scripted_default).

    Outcomes: ``executed`` (the mutation was sent; ``readback`` carries the
    verification evidence), ``rejected`` (never sent; ``rejection_class``
    carries the illegal-action taxonomy class), ``aborted_group`` (never
    sent; the entity's group was aborted by an earlier stale/illegal action,
    §3 line 33).
    """

    kind: Literal["action_result"] = "action_result"
    turn: int
    ref: str
    domain: str = ""  # candidate domain (unit/city/tech/dipl); "unknown" for
    # unparseable schema-error refs
    verb: str = ""
    # Structured target descriptor (commitment matching, §3 line 40):
    # tech name / production name / "tile:x,y" (founding or worked tile) /
    # "unit:<id>" (orders on a unit) / "city:<id>" / "player:<id>".
    target: Optional[str] = None
    # Structured affected descriptor (reason-but-don't-act matching,
    # §3 line 41): "unit:<id>" for attacks with a known target, else
    # "tile:x,y" for unit actions; equals target for non-unit actions.
    affected: Optional[str] = None
    # The acting entity's tile before the action (None when unknown).
    actor_tile: Optional[tuple[int, int]] = None
    outcome: Literal["executed", "rejected", "aborted_group"]
    rejection_class: Optional[RejectionClass] = None
    provenance_class: ProvenanceClass = "model"
    # continued_model_order: seq of the originating model decision.
    origin_seq: Optional[int] = None
    # scripted_default: the reason the default resolved the choice.
    reason: str = ""
    # Provenance linkage (§3 line 38): snapshot ID, offered candidates,
    # call ID.
    snapshot_seq: Optional[int] = None
    candidates_seq: Optional[int] = None
    call_seq: Optional[int] = None
    readback: ReadbackStatus = "none"
    detail: str = ""


class ThreatMention(BaseModel):
    """One structured threat hypothesis (§3 line 41): an opponent/barbarian
    entity referenced in the model's structured output fields — entity IDs
    only, the metric never parses free text. ``tile`` is the entity's tile
    when visible at mention time (None under fog)."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str  # e.g. "unit:31", "city:8", "player:2"
    turn: int
    tile: Optional[tuple[int, int]] = None


class CommitmentRecord(TranscriptRecordBase):
    """The model's structured memory statement for one batch (§7.5: the
    memory_update rides the ActionBatch — never a separate API call).
    Carries the batch's commitments (falsifiable: domain, target kind +
    target, optional verb, deadline turn) and threat hypotheses. The metric
    de-duplicates commitments per (target, domain) and threats per
    (entity, turn)."""

    kind: Literal["commitment"] = "commitment"
    turn: int
    commitments: list[Commitment] = Field(default_factory=list)
    threats: list[ThreatMention] = Field(default_factory=list)
    call_seq: Optional[int] = None


class MetricEventRecord(TranscriptRecordBase):
    kind: Literal["metric_event"] = "metric_event"
    name: str
    value: Union[int, float, str]


class OutcomeRecord(TranscriptRecordBase):
    """Run outcome (§3 line 34): COMPLETED (horizon) / DEFEATED / VICTORY /
    TRUNCATED (with reason) — infrastructure failure is separated from
    strategic loss in all reporting."""

    kind: Literal["outcome"] = "outcome"
    outcome: OutcomeKind
    reason: str = ""


TranscriptRecord = Union[
    RunManifestRecord,
    TurnStartRecord,
    SnapshotRefRecord,
    CandidatesRefRecord,
    ModelCallRecord,
    ActionResultRecord,
    CommitmentRecord,
    MetricEventRecord,
    OutcomeRecord,
]


class RunManifest(BaseModel):
    """Per-run manifest (decision 8). The FIELD SET is frozen as of WP2
    (2026-10-24): the decision-8 list below, no additions or removals without
    a plan amendment; ``extra="allow"`` is retained for forward-compat
    fields recorded by later packages. Optionality is retained because the
    manifest record is emitted at run start and WP3/eval owns populating
    every field; the freeze is of the field set and its semantics.

    ``model_fingerprint`` (decision 8, FW-21) persists the K=3 sample hash
    SET, not a single hash: the sorted, deduplicated set of
    ``sha256:<hex>`` sample output hashes joined by ``|`` (see
    ``lechonk.model.mistral_client.encode_fingerprint_set``). A legacy
    single-sample fingerprint (``sha256:<hex>``, the pre-FW-21 encoding)
    is exactly the one-element set and remains loadable — the documented
    fallback. The field set is unchanged; this note documents the string
    encoding only."""

    model_config = ConfigDict(extra="allow")

    run_id: Optional[str] = None
    image_digest: Optional[str] = None
    server_commit: Optional[str] = None
    capstring: Optional[str] = None
    vendored_base_sha: Optional[str] = None
    patch_head_sha: Optional[str] = None
    harness_revision: Optional[str] = None
    python_lockfile: Optional[str] = None
    seeds: dict[str, Optional[int]] = Field(default_factory=dict)  # mapseed/gameseed/agentseed
    ruleset: Optional[str] = None
    server_settings: dict[str, Any] = Field(default_factory=dict)
    model_id: Optional[str] = None
    sampling: dict[str, Any] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    config_after_load: dict[str, Any] = Field(default_factory=dict)
    checkpoint_checksums: dict[str, str] = Field(default_factory=dict)
    model_fingerprint: Optional[str] = None
    arm_start_ts: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Candidate-ID pattern as a compiled regex for non-pydantic validation sites
# (candidates.py / execute.py, later packages).
CANDIDATE_ID_RE = re.compile(CANDIDATE_ID_REGEX)


def is_candidate_id(value: Any) -> bool:
    return isinstance(value, str) and CANDIDATE_ID_RE.match(value) is not None
