# SPDX-License-Identifier: MIT
"""lechonk.comparator — the scripted comparator policy and the degraded-default
entry point (WP1 package C; design-plan-freeciv-v1.1.md §7.4 and §3 line 44).

MIT: this package never imports civrealm or lechonk.adapter (test-enforced in
tests/test_comparator_purity.py). It consumes core.schemas Snapshot +
CandidateTable and emits candidate-ID choices; all execution goes through the
adapter (the turn driver owns it).

Two separate entry points, both callable, both producing a decision log:
- ``scripted.decide`` — the comparator: the ACTIVE scripted policy (§7.4).
- ``defaults.decide`` — the degraded defaults: preserve all current orders
  and production, finish the phase; only NEW mandatory choices (research
  completed, city production empty) resolve via the same published priority
  lists, each logged with reason ``scripted_default``.
"""

from lechonk.comparator import defaults, scripted
from lechonk.comparator.scripted import (
    PROVENANCE_CLASS,
    Decision,
    DecisionLog,
    is_exposed_verb,
)

__all__ = [
    "PROVENANCE_CLASS",
    "Decision",
    "DecisionLog",
    "defaults",
    "is_exposed_verb",
    "scripted",
]
