# SPDX-License-Identifier: MIT
"""Bounded, fog-safe decision history for interactive model prompts.

History is evidence about completed turns.  It is deliberately separate from
the current snapshot and candidate table: historical intentions never become
authoritative game facts or executable candidate references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping

from lechonk.core.schemas import estimate_tokens


HISTORY_TOKEN_BUDGET = 4000
HISTORY_RECENT_FLOOR = 4
_MAX_NOTE_CHARS = 240
_MAX_FIELD_CHARS = 96
_MAX_PLANNED = 8
_MAX_RESULTS = 12
_MAX_STORED_EPISODES = 64
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _text(value: Any, limit: int = _MAX_FIELD_CHARS) -> str:
    if value is None:
        return ""
    text = _CONTROL_RE.sub(" ", str(value)).replace("\r", " ")
    text = " ".join(text.split())
    return text[:limit].rstrip()


def _descriptor(value: Any) -> dict[str, str]:
    """Return an intent descriptor without retaining an executable ref."""
    return {
        key: _text(_value(value, key), _MAX_FIELD_CHARS)
        for key in ("domain", "verb", "target", "affected")
        if _text(_value(value, key), _MAX_FIELD_CHARS)
    }


def _result(value: Any) -> dict[str, str]:
    keys = ("domain", "verb", "target", "affected", "outcome",
            "rejection_class", "provenance_class", "readback")
    return {
        key: _text(_value(value, key), _MAX_FIELD_CHARS)
        for key in keys
        if _text(_value(value, key), _MAX_FIELD_CHARS)
    }


@dataclass(frozen=True)
class DecisionEpisode:
    """A normalized, completed turn suitable for historical context."""

    turn: int
    notes: str = ""
    planned: tuple[dict[str, str], ...] = ()
    results: tuple[dict[str, str], ...] = ()
    degraded: bool = False
    repair_used: bool = False
    status: str = "completed"

    @classmethod
    def from_parts(
        cls,
        *,
        turn: int,
        notes: Any = "",
        planned: Any = (),
        results: Any = (),
        degraded: bool = False,
        repair_used: bool = False,
        status: Any = "completed",
    ) -> "DecisionEpisode":
        planned_values = tuple(
            _descriptor(item) for item in (planned or ())
            if _descriptor(item)
        )[:_MAX_PLANNED]
        result_values = tuple(
            _result(item) for item in (results or ()) if _result(item)
        )[:_MAX_RESULTS]
        return cls(
            turn=int(turn),
            notes=_text(notes, _MAX_NOTE_CHARS),
            planned=planned_values,
            results=result_values,
            degraded=bool(degraded),
            repair_used=bool(repair_used),
            status=_text(status, 32) or "completed",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "status": self.status,
            "degraded": self.degraded,
            "repair_used": self.repair_used,
            "intent": {
                "notes": self.notes,
                "planned": list(self.planned),
            },
            "execution": list(self.results),
        }


@dataclass
class DecisionHistory:
    """Deterministic bounded journal of completed decision episodes.

    ``add`` is the only mutation and must be called after a turn's execution
    has completed.  Rendering keeps the newest episodes first, but never
    includes raw prompts, snapshots, reasoning, candidate refs, or action
    payloads.  Older eviction is disclosed in the rendered section.
    """

    budget: int = HISTORY_TOKEN_BUDGET
    recent_floor: int = HISTORY_RECENT_FLOOR
    _episodes: list[DecisionEpisode] = field(default_factory=list)
    _evicted_total: int = 0

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("history budget must be positive")
        if self.recent_floor < 0:
            raise ValueError("recent floor must not be negative")

    @property
    def episodes(self) -> tuple[DecisionEpisode, ...]:
        return tuple(self._episodes)

    def add(self, episode: DecisionEpisode) -> None:
        if not isinstance(episode, DecisionEpisode):
            raise TypeError("history accepts DecisionEpisode values")
        if self._episodes and episode.turn <= self._episodes[-1].turn:
            raise ValueError("history turns must increase")
        self._episodes.append(episode)
        # Keep the in-memory journal bounded even before rendering.  The
        # durable ledger can retain the complete record separately.
        while len(self._episodes) > _MAX_STORED_EPISODES:
            self._episodes.pop(0)
            self._evicted_total += 1

    def _render(self, episodes: list[DecisionEpisode], evicted: int) -> str:
        body = {
            "historical_context": True,
            "authority": "CURRENT_SNAPSHOT_AND_CANDIDATES_OVERRIDE_HISTORY",
            "evicted_older_episodes": evicted,
            "episodes": [episode.as_dict() for episode in reversed(episodes)],
        }
        return "## RECENT VERIFIED HISTORY\n" + json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    def render(self) -> str:
        """Render a stable, JSON data section within the configured budget."""
        if not self._episodes:
            return self._render([], 0)
        selected = list(self._episodes)
        evicted = self._evicted_total
        floor = min(self.recent_floor, len(selected))
        while len(selected) > floor and estimate_tokens(
                self._render(selected, evicted)) > self.budget:
            # Keep the newest configured floor when the hard budget permits.
            selected.pop(0)
            evicted += 1
        rendered = self._render(selected, evicted)
        # A very small custom budget can be smaller than the recent floor.
        # Compact each retained episode's untrusted text/data first, then
        # evict oldest entries only if the hard bound still cannot be met.
        if estimate_tokens(rendered) > self.budget:
            selected = [DecisionEpisode.from_parts(
                turn=item.turn, notes="", planned=(), results=(),
                degraded=item.degraded, repair_used=item.repair_used,
                status=item.status,
            ) for item in selected]
            while len(selected) > 1 and estimate_tokens(
                    self._render(selected, evicted)) > self.budget:
                selected.pop(0)
                evicted += 1
            rendered = self._render(selected, evicted)
        return rendered
