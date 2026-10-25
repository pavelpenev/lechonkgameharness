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
from typing import Optional

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

    # ------------------------------------------------------------------
    # pre/post-advance separation
    # ------------------------------------------------------------------

    def begin_turn(self, turn: int) -> None:
        """Freeze the pre-advance view: the memory rendered for this turn's
        prompt is the committed state (updates from the previous turn have
        been advanced); later staging cannot change it."""
        self._frozen_text = self._render(self._current, turn)

    def render_for_prompt(self) -> str:
        """The ≤1,500-token memory text for the current turn's prompt (the
        frozen pre-advance view; call ``begin_turn`` first)."""
        if self._frozen_text is None:
            self._frozen_text = self._render(self._current, turn=0)
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
    # rendering + budget
    # ------------------------------------------------------------------

    def _render(self, state: _MemoryState, turn: int) -> str:
        commitments = sorted(
            state.commitments.values(),
            key=lambda c: (c.deadline_turn, c.domain, c.target))
        threats = list(state.threats)  # insertion order = recency
        notes = state.notes
        elided_threats = 0
        elided_commitments = 0
        # Deterministic truncation to the budget (the FULL rendered text,
        # headers and disclosure included): elide the OLDEST threat
        # hypotheses first, then shorten the notes, then elide the OLDEST
        # commitments.
        while True:
            text = self._render_text(commitments, threats, notes, turn,
                                     elided_threats, elided_commitments)
            if estimate_tokens(text) <= self._budget:
                return text
            if threats:
                threats.pop(0)
                elided_threats += 1
            elif notes:
                notes = notes[: max(0, len(notes) - 64)].rstrip()
            elif commitments:
                commitments.pop(0)
                elided_commitments += 1
            else:
                return text  # the bare header alone exceeds: nothing to elide

    def _render_text(self, commitments: list[Commitment], threats: list[str],
                     notes: str, turn: int, elided_threats: int,
                     elided_commitments: int) -> str:
        lines = [f"# strategy memory (turn {turn})"]
        lines.append("## commitments (domain, target, deadline turn)")
        if commitments:
            for c in commitments:
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
        if elided_threats or elided_commitments:
            parts = []
            if elided_threats:
                parts.append(f"{elided_threats} oldest threat hypotheses")
            if elided_commitments:
                parts.append(f"{elided_commitments} oldest commitments")
            lines.append(f"(memory truncated: {', '.join(parts)} elided to "
                         f"fit the {self._budget}-token budget)")
        return "\n".join(lines)
