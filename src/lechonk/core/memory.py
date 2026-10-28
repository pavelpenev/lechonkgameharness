# SPDX-License-Identifier: MIT
"""lechonk.core.memory — strategy memory (WP2).

§3 line 32: "strategy memory ≤ 1,500 (current plan, falsifiable commitments
with deadline turns, threat hypotheses)". §7.5: the memory update is NOT a
separate API call — it rides the per-turn ActionBatch's ``memory_update``
section; on a repair attempt the last valid memory persists.

Pre/post-advance separation: the prompt for turn N is rendered from the
memory as of the END of turn N-1 (the pre-advance view). Updates from turn
N's batch are STAGED during the turn and COMMITTED post-advance (after the
turn's actions resolve), so they first appear in turn N+1's prompt. This
keeps what was sent to the model frozen (``begin_turn`` caches the rendered
text) and never lets a mid-turn mutation change the prompt that was sent.

Budget: the rendered memory is ≤ MEMORY_TOKEN_BUDGET (1,500) tokens
(core.schemas.estimate_tokens); deterministic truncation when over budget —
the OLDEST threat hypotheses are elided first, then the notes are
shortened, then the OLDEST commitments are elided — and the rendered text
discloses the elision. Commitments whose deadline turn has passed are
dropped from the prompt-facing view at ``advance`` (the ledger keeps them
for the metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from lechonk.core.schemas import (
    MEMORY_TOKEN_BUDGET,
    Commitment,
    MemoryUpdate,
    estimate_tokens,
)


@dataclass
class _MemoryState:
    commitments: dict[tuple[str, str], Commitment] = field(default_factory=dict)
    # entity_id -> turn first seen (insertion order = recency order).
    threats: dict[str, int] = field(default_factory=dict)
    notes: str = ""


class StrategyMemory:
    """Commitments (domain, target, deadline_turn) + threat hypotheses
    (entity IDs), ≤1,500 tokens rendered, pre/post-advance separation."""

    def __init__(self, *, budget: int = MEMORY_TOKEN_BUDGET):
        self._budget = budget
        self._current = _MemoryState()
        self._staged: Optional[_MemoryState] = None
        self._frozen_text: Optional[str] = None  # the pre-advance view
        self._frozen_profile = "legacy"
        self._frozen_turn = 0

    # ------------------------------------------------------------------
    # pre/post-advance separation
    # ------------------------------------------------------------------

    def begin_turn(self, turn: int, *, profile: str = "legacy") -> None:
        """Freeze the pre-advance view: the memory rendered for this turn's
        prompt is the committed state (updates from the previous turn have
        been advanced); later staging cannot change it."""
        profile = _normalise_profile(profile)
        self._frozen_profile = profile
        self._frozen_turn = turn
        self._frozen_text = self._render(self._current, turn, profile=profile)

    def render_for_prompt(self, *, profile: str | None = None) -> str:
        """The ≤1,500-token memory text for the current turn's prompt (the
        frozen pre-advance view; call ``begin_turn`` first)."""
        if self._frozen_text is None:
            chosen = _normalise_profile(profile or "legacy")
            self._frozen_profile = chosen
            self._frozen_turn = 0
            self._frozen_text = self._render(self._current, turn=0,
                                             profile=chosen)
        elif profile is not None and _normalise_profile(profile) != self._frozen_profile:
            # A caller may request a different presentation (for example the
            # gameplay prompt) without changing the committed state.  Keep
            # the normal no-argument path frozen and deterministic.
            return self._render(self._current, turn=self._frozen_turn,
                                profile=_normalise_profile(profile))
        return self._frozen_text

    def apply(self, update: MemoryUpdate, *, turn: int) -> None:
        """Stage a memory update from the ActionBatch's memory_update
        section (never a separate API call). Staged, not yet visible to
        prompts."""
        staged = _MemoryState(
            commitments=dict(self._current.commitments),
            threats=dict(self._current.threats),
            notes=self._current.notes,
        )
        for c in update.commitments:
            # Upsert per (target, domain): a re-stated commitment updates
            # its deadline (the ledger's metric de-duplicates first-wins;
            # the prompt-facing memory keeps the model's latest statement).
            staged.commitments[(c.target, c.domain)] = c
        # Threats are the model's CURRENT hypothesis set (replace), entity
        # IDs only — no free text.
        staged.threats = {t: turn for t in update.threats}
        staged.notes = update.notes
        self._staged = staged

    def advance(self, turn: int) -> None:
        """Commit the staged update post-advance (after the turn's actions
        resolve): it becomes the pre-advance view for the next turn.
        Commitments whose deadline turn has passed (deadline ≤ turn) are
        dropped from the prompt-facing view — the ledger keeps them for
        the metrics."""
        if self._staged is not None:
            self._current = self._staged
            self._staged = None
        self._current.commitments = {
            key: c for key, c in self._current.commitments.items()
            if c.deadline_turn > turn
        }
        self._frozen_text = None

    # ------------------------------------------------------------------
    # accessors (ledger/metric-facing)
    # ------------------------------------------------------------------

    def commitments(self) -> list[Commitment]:
        return list(self._current.commitments.values())

    def threats(self) -> list[str]:
        return list(self._current.threats)

    # ------------------------------------------------------------------
    # durable committed state
    # ------------------------------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        """Return a versioned, JSON-safe snapshot of committed memory.

        Staged and frozen prompt state is deliberately excluded.  A resume
        operation must begin from the last completed turn, never from a
        partially applied model response.
        """
        return {
            "version": 1,
            "budget": self._budget,
            "commitments": [c.model_dump(mode="json")
                            for c in self._current.commitments.values()],
            "threats": list(self._current.threats.keys()),
            "notes": self._current.notes,
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any], *, budget: int | None = None) -> "StrategyMemory":
        """Restore a committed memory snapshot after strict validation.

        The input is treated as untrusted artifact data.  Unknown fields,
        malformed commitments, duplicate keys, invalid entity IDs, and
        oversized notes are rejected before constructing the object.
        """
        if not isinstance(payload, Mapping) or payload.get("version") != 1:
            raise ValueError("unsupported strategy-memory state version")
        allowed = {"version", "budget", "commitments", "threats", "notes"}
        if set(payload) != allowed:
            raise ValueError("invalid strategy-memory state fields")
        stored_budget = payload.get("budget")
        if not isinstance(stored_budget, int) or isinstance(stored_budget, bool) or stored_budget <= 0:
            raise ValueError("invalid strategy-memory budget")
        chosen_budget = stored_budget if budget is None else budget
        if not isinstance(chosen_budget, int) or isinstance(chosen_budget, bool) or chosen_budget <= 0:
            raise ValueError("invalid strategy-memory budget")
        commitments_raw = payload.get("commitments")
        threats_raw = payload.get("threats")
        notes = payload.get("notes")
        if not isinstance(commitments_raw, list) or not isinstance(threats_raw, list):
            raise ValueError("invalid strategy-memory collections")
        if not isinstance(notes, str) or len(notes) > 200_000:
            raise ValueError("invalid strategy-memory notes")
        commitments: dict[tuple[str, str], Commitment] = {}
        for raw in commitments_raw:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid strategy-memory commitment")
            try:
                commitment = Commitment.model_validate(dict(raw))
            except Exception as exc:
                raise ValueError("invalid strategy-memory commitment") from exc
            key = (commitment.target, commitment.domain)
            if key in commitments:
                raise ValueError("duplicate strategy-memory commitment")
            commitments[key] = commitment
        threats: dict[str, int] = {}
        for threat in threats_raw:
            if not isinstance(threat, str) or not threat or len(threat) > 512:
                raise ValueError("invalid strategy-memory threat")
            if threat in threats:
                raise ValueError("duplicate strategy-memory threat")
            threats[threat] = 0
        result = cls(budget=chosen_budget)
        result._current = _MemoryState(commitments=commitments,
                                        threats=threats, notes=notes)
        # Never restore an in-progress prompt or staged update.
        result._staged = None
        result._frozen_text = None
        result._frozen_profile = "legacy"
        result._frozen_turn = 0
        return result

    # ------------------------------------------------------------------
    # rendering + budget
    # ------------------------------------------------------------------

    def _render(self, state: _MemoryState, turn: int, *, profile: str = "legacy") -> str:
        profile = _normalise_profile(profile)
        commitments = sorted(
            state.commitments.values(),
            key=lambda c: (c.deadline_turn, c.domain, c.target))
        threats = list(state.threats)  # insertion order = recency
        notes = state.notes
        elided_threats = 0
        elided_commitments = 0
        elided_note_chars = 0
        # Deterministic truncation to the budget (the FULL rendered text,
        # headers and disclosure included): elide the OLDEST threat
        # hypotheses first, then shorten the notes, then elide the OLDEST
        # commitments.
        while True:
            text = self._render_text(commitments, threats, notes, turn,
                                     elided_threats, elided_commitments,
                                     elided_note_chars, profile=profile)
            if estimate_tokens(text) <= self._budget:
                return text
            if threats:
                threats.pop(0)
                elided_threats += 1
            elif notes:
                before = len(notes)
                notes = notes[: max(0, before - 64)].rstrip()
                elided_note_chars += before - len(notes)
            elif commitments:
                commitments.pop(0)
                elided_commitments += 1
            else:
                return text  # the bare header alone exceeds: nothing to elide

    def _render_text(self, commitments: list[Commitment], threats: list[str],
                     notes: str, turn: int, elided_threats: int,
                     elided_commitments: int,
                     elided_note_chars: int = 0,
                     *, profile: str = "legacy") -> str:
        profile = _normalise_profile(profile)
        lines = [f"# strategy memory (turn {turn})"]
        if profile == "gameplay":
            lines.append("## commitments (domain, target_kind, target, verb, deadline turn)")
        else:
            lines.append("## commitments (domain, target, deadline turn)")
        if commitments:
            for c in commitments:
                if profile == "gameplay":
                    verb = f" verb={c.verb}" if c.verb else ""
                    lines.append(
                        f"- domain={c.domain} target_kind={c.target_kind} "
                        f"target={c.target}{verb} deadline_turn={c.deadline_turn}"
                    )
                else:
                    lines.append(f"- {c.domain}: {c.target} "
                                 f"(by turn {c.deadline_turn})")
        else:
            lines.append("- none")
        lines.append("## threat hypotheses (entity IDs)")
        if threats:
            for t in threats:
                lines.append(f"- {t}")
        else:
            lines.append("- none")
        if notes:
            lines.append("## notes")
            lines.append(notes)
        if elided_threats or elided_commitments or elided_note_chars:
            parts = []
            if elided_threats:
                parts.append(f"{elided_threats} oldest threat hypotheses elided")
            if elided_commitments:
                parts.append(f"{elided_commitments} oldest commitments elided")
            if elided_note_chars:
                parts.append(f"{elided_note_chars} note characters elided")
            lines.append(f"(memory truncated: {', '.join(parts)} elided to "
                         f"fit the {self._budget}-token budget)")
        return "\n".join(lines)


def _normalise_profile(profile: str) -> str:
    """Return a supported renderer profile, rejecting silent data loss."""
    if profile == "gameplay_v1":
        profile = "gameplay"
    if profile not in {"legacy", "gameplay"}:
        raise ValueError(f"unknown strategy-memory render profile: {profile!r}")
    return profile
