# SPDX-License-Identifier: MIT
"""Strategy-memory tests (WP2): commitments + threat hypotheses, ≤1,500
tokens, pre/post-advance separation, updates riding the ActionBatch's
memory_update (never a separate API call)."""

from __future__ import annotations

from lechonk.core.memory import StrategyMemory
from lechonk.core.schemas import (
    MEMORY_TOKEN_BUDGET,
    Commitment,
    MemoryUpdate,
    estimate_tokens,
)


def _update(commitments=(), threats=(), notes=""):
    return MemoryUpdate(
        commitments=[Commitment(domain=d, target=t, deadline_turn=dl)
                     for d, t, dl in commitments],
        threats=list(threats), notes=notes)


def test_memory_renders_commitments_and_threats():
    memory = StrategyMemory()
    memory.begin_turn(3)
    assert "turn 3" in memory.render_for_prompt()
    memory.apply(_update(
        commitments=[("research", "Pottery", 8)],
        threats=["unit:20", "city:4"], notes="watch the north"), turn=3)
    memory.advance(3)
    memory.begin_turn(4)
    text = memory.render_for_prompt()
    assert "research: Pottery (by turn 8)" in text
    assert "unit:20" in text and "city:4" in text
    assert "watch the north" in text


def test_gameplay_profile_preserves_typed_commitment_fields():
    memory = StrategyMemory()
    memory.apply(MemoryUpdate(commitments=[
        Commitment(domain="research", target_kind="tech", target="Pottery",
                   verb="research", deadline_turn=8),
        Commitment(domain="military", target_kind="unit", target="*",
                   verb="defend", deadline_turn=12),
        Commitment(domain="expansion", target_kind="any", target="frontier",
                   deadline_turn=14),
    ]), turn=3)
    memory.advance(3)
    memory.begin_turn(4, profile="gameplay_v1")
    text = memory.render_for_prompt()
    assert "target_kind=tech target=Pottery verb=research" in text
    assert "target_kind=unit target=* verb=defend" in text
    assert "target_kind=any target=frontier" in text
    assert "deadline_turn=8" in text
    # The alternate spelling is accepted without changing the frozen view.
    assert memory.render_for_prompt(profile="gameplay") == text


def test_legacy_profile_does_not_change_for_typed_commitment():
    memory = StrategyMemory()
    memory.apply(MemoryUpdate(commitments=[
        Commitment(domain="research", target_kind="tech", target="Pottery",
                   verb="research", deadline_turn=8),
    ]), turn=3)
    memory.advance(3)
    memory.begin_turn(4)
    assert memory.render_for_prompt() == (
        "# strategy memory (turn 4)\n"
        "## commitments (domain, target, deadline turn)\n"
        "- research: Pottery (by turn 8)\n"
        "## threat hypotheses (entity IDs)\n"
        "- none"
    )


def test_gameplay_profile_budget_elision_accounts_for_typed_fields():
    memory = StrategyMemory(budget=100)
    memory.apply(MemoryUpdate(commitments=[
        Commitment(domain="research", target_kind="tech",
                   target=f"Technology-{i}-with-a-long-name", verb="research",
                   deadline_turn=50 + i)
        for i in range(20)
    ], threats=[f"unit:{i}" for i in range(20)], notes="n" * 500), turn=1)
    memory.advance(1)
    memory.begin_turn(2, profile="gameplay")
    text = memory.render_for_prompt()
    assert estimate_tokens(text) <= 100
    assert "memory truncated:" in text
    assert "oldest threat hypotheses" in text
    assert "oldest commitments elided" in text


def test_pre_post_advance_separation():
    """The prompt for turn N is the pre-advance view (end of turn N-1);
    a staged update is invisible until advance() commits it post-advance."""
    memory = StrategyMemory()
    memory.begin_turn(1)
    assert "Pottery" not in memory.render_for_prompt()
    memory.apply(_update(commitments=[("research", "Pottery", 5)]), turn=1)
    # Staged: still invisible in THIS turn's frozen prompt.
    assert "Pottery" not in memory.render_for_prompt()
    memory.advance(1)  # post-advance commit
    memory.begin_turn(2)
    assert "Pottery" in memory.render_for_prompt()


def test_begin_turn_freezes_the_rendered_text():
    """A mid-turn staging cannot change the prompt that was sent."""
    memory = StrategyMemory()
    memory.apply(_update(threats=["unit:20"]), turn=0)
    memory.advance(0)
    memory.begin_turn(1)
    sent = memory.render_for_prompt()
    memory.apply(_update(threats=["unit:99"]), turn=1)
    assert memory.render_for_prompt() == sent  # frozen
    memory.advance(1)
    memory.begin_turn(2)
    assert "unit:99" in memory.render_for_prompt()
    assert "unit:20" not in memory.render_for_prompt()  # threats replace


def test_expired_commitments_leave_the_prompt_view():
    memory = StrategyMemory()
    memory.apply(_update(commitments=[("research", "Pottery", 2)]), turn=1)
    memory.advance(1)
    memory.begin_turn(2)
    assert "Pottery" in memory.render_for_prompt()
    memory.advance(2)  # deadline 2 has passed at turn 2's advance
    memory.begin_turn(3)
    assert "Pottery" not in memory.render_for_prompt()


def test_budget_is_enforced_with_disclosure():
    # Documented deterministic order: the OLDEST threat hypotheses are
    # elided first, then the notes are shortened, then the OLDEST
    # commitments. With a large notes block the threats go entirely.
    memory = StrategyMemory(budget=100)  # small budget for the test
    memory.apply(_update(
        commitments=[("research", f"Tech{i}", 50 + i) for i in range(6)],
        threats=[f"unit:{i}" for i in range(30)],
        notes="n" * 500), turn=1)
    memory.advance(1)
    memory.begin_turn(2)
    text = memory.render_for_prompt()
    assert estimate_tokens(text) <= 100
    assert "memory truncated: 30 oldest threat hypotheses" in text
    assert "note characters" in text
    assert "unit:0" not in text and "unit:29" not in text

    # When the budget allows some threats, the NEWEST survive.
    memory = StrategyMemory(budget=100)
    memory.apply(_update(
        commitments=[("research", "Pottery", 9)],
        threats=[f"unit:{i}" for i in range(30)]), turn=1)
    memory.advance(1)
    memory.begin_turn(2)
    text = memory.render_for_prompt()
    assert estimate_tokens(text) <= 100
    assert "memory truncated:" in text
    assert "unit:0" not in text      # oldest elided
    assert "unit:29" in text        # newest kept
    assert "Pottery" in text        # commitments kept


def test_default_budget_is_1500_tokens():
    memory = StrategyMemory()
    assert memory._budget == MEMORY_TOKEN_BUDGET == 1500
    memory.apply(_update(
        commitments=[("research", "Pottery", 9)],
        threats=["unit:20"], notes="x" * 100), turn=1)
    memory.advance(1)
    memory.begin_turn(2)
    assert estimate_tokens(memory.render_for_prompt()) <= 1500


def test_committed_state_round_trips_without_staged_or_frozen_state():
    memory = StrategyMemory()
    memory.apply(_update(commitments=[("research", "Pottery", 9)],
                         threats=["unit:20"], notes="keep scouting"), turn=1)
    memory.advance(1)
    memory.begin_turn(2)
    memory.apply(_update(threats=["unit:99"], notes="unfinished"), turn=2)

    restored = StrategyMemory.from_state(memory.dump_state())
    assert [c.target for c in restored.commitments()] == ["Pottery"]
    assert restored.threats() == ["unit:20"]
    restored.begin_turn(3)
    assert "keep scouting" in restored.render_for_prompt()
    assert "unfinished" not in restored.render_for_prompt()


def test_memory_state_rejects_bad_or_malicious_payloads():
    memory = StrategyMemory()
    state = memory.dump_state()
    for bad in (
        {**state, "version": 99},
        {**state, "notes": "x" * 200_001},
        {**state, "threats": ["unit:1", "unit:1"]},
        {**state, "commitments": [{"domain": "bad", "target": "x", "deadline_turn": 1}]},
    ):
        try:
            StrategyMemory.from_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid state was accepted")
