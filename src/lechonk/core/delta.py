# SPDX-License-Identifier: MIT
"""lechonk.core.delta — the turn-over-turn delta report (WP2, §7.3 ≤800 tokens).

Pure function over core.schemas.Snapshot pairs — no civrealm, no adapter,
offline-testable. A vanished FOREIGN unit is never reported as lost (fog
honesty, §3 line 36): only own units (always visible to their owner) produce
unit_lost_own / unit_changed events. Combat events are deliberately not
inferred from unit disappearances (vanishing ≠ destruction); combat
evidence lives in the attack read-backs in the transcript.

The runner enforces the ≤800-token budget by deterministically dropping the
oldest events and appending a warning event (disclosure).
"""

from __future__ import annotations

from typing import Optional

from lechonk.core.schemas import DELTA_TOKEN_BUDGET, Delta, DeltaEvent, Snapshot, estimate_tokens


def compute_delta(prev: Optional[Snapshot], cur: Snapshot) -> Delta:
    """The delta report from ``prev`` to ``cur`` (None prev = first turn:
    an empty delta)."""
    if prev is None or prev.turn == cur.turn:
        return Delta(turn=cur.turn)
    events: list[DeltaEvent] = []

    prev_units = {u.id: u for u in prev.units if u.is_own}
    cur_units = {u.id: u for u in cur.units if u.is_own}
    # Elision honesty (§3, C6): a unit listed in a snapshot's
    # omitted.elided_unit_ids was dropped by the observation budget, not by
    # the game — it is NOT comparable across that snapshot's boundary, so it
    # produces neither a loss nor an appearance. Genuinely vanished own units
    # (absent from cur and not elided) still report as losses.
    cur_elided = set(cur.omitted.elided_unit_ids)
    prev_elided = set(prev.omitted.elided_unit_ids)
    for uid in sorted(set(prev_units) - set(cur_units) - cur_elided):
        events.append(DeltaEvent(
            kind="unit_lost_own", subject=f"unit:{uid}",
            detail=f"own unit {uid} ({prev_units[uid].type_name}) is gone"))
    for uid in sorted(set(cur_units) - set(prev_units) - prev_elided):
        u = cur_units[uid]
        events.append(DeltaEvent(
            kind="unit_changed", subject=f"unit:{uid}",
            detail=f"new own unit {uid} ({u.type_name}) at ({u.x},{u.y})"))
    for uid in sorted(set(prev_units) & set(cur_units)):
        a, b = prev_units[uid], cur_units[uid]
        if (a.x, a.y, a.hp, a.activity_name) != (b.x, b.y, b.hp, b.activity_name):
            events.append(DeltaEvent(
                kind="unit_changed", subject=f"unit:{uid}",
                detail=(f"unit {uid} ({a.type_name}): ({a.x},{a.y}) "
                        f"hp {a.hp} {a.activity_name or 'idle'} -> "
                        f"({b.x},{b.y}) hp {b.hp} {b.activity_name or 'idle'}")))

    prev_cities = {c.id: c for c in prev.cities if c.is_own}
    cur_cities = {c.id: c for c in cur.cities if c.is_own}
    for cid in sorted(set(cur_cities) - set(prev_cities)):
        c = cur_cities[cid]
        events.append(DeltaEvent(
            kind="city_growth", subject=f"city:{cid}",
            detail=f"new own city {c.name} at ({c.x},{c.y}) size {c.size}"))
    for cid in sorted(set(prev_cities) & set(cur_cities)):
        a, b = prev_cities[cid], cur_cities[cid]
        if a.size is not None and b.size is not None and b.size > a.size:
            events.append(DeltaEvent(
                kind="city_growth", subject=f"city:{cid}",
                detail=f"city {a.name} grew {a.size} -> {b.size}"))
        if a.production is not None and b.production is None:
            events.append(DeltaEvent(
                kind="production_complete", subject=f"city:{cid}",
                detail=f"city {a.name} completed {a.production}"))

    pr, cr = prev.player.research, cur.player.research
    if pr.current_target_id != cr.current_target_id:
        if cr.current_target_id is None:
            events.append(DeltaEvent(
                kind="research_complete", subject=pr.current_target or "tech",
                detail=f"research {pr.current_target} completed; "
                       f"no new target"))
        else:
            events.append(DeltaEvent(
                kind="research_progress", subject=cr.current_target or "tech",
                detail=(f"research target {pr.current_target} -> "
                        f"{cr.current_target} (bulbs {pr.bulbs_researched} "
                        f"-> {cr.bulbs_researched})")))
    elif (pr.bulbs_researched != cr.bulbs_researched
          or pr.turns_left != cr.turns_left):
        events.append(DeltaEvent(
            kind="research_progress", subject=cr.current_target or "tech",
            detail=(f"research {cr.current_target}: bulbs "
                    f"{pr.bulbs_researched} -> {cr.bulbs_researched}, "
                    f"{pr.turns_left} -> {cr.turns_left} turns left")))

    prev_dipl = {d.counterpart_id: d for d in prev.diplomacy}
    cur_dipl = {d.counterpart_id: d for d in cur.diplomacy}
    for pid in sorted(set(cur_dipl) - set(prev_dipl)):
        events.append(DeltaEvent(
            kind="new_contact", subject=f"player:{pid}",
            detail=f"new contact: {cur_dipl[pid].counterpart_name or pid} "
                   f"({cur_dipl[pid].state})"))
    for pid in sorted(set(prev_dipl) & set(cur_dipl)):
        a, b = prev_dipl[pid], cur_dipl[pid]
        if a.state != b.state:
            events.append(DeltaEvent(
                kind="meeting", subject=f"player:{pid}",
                detail=f"diplomatic state with {a.counterpart_name or pid}: "
                       f"{a.state} -> {b.state}"))
        if a.meeting_active != b.meeting_active and b.meeting_active:
            events.append(DeltaEvent(
                kind="meeting", subject=f"player:{pid}",
                detail=f"meeting opened with {a.counterpart_name or pid}"))

    if prev.omitted.summary != "nothing omitted" or \
            cur.omitted.summary != "nothing omitted":
        events.append(DeltaEvent(
            kind="warning", subject="snapshot",
            detail=f"snapshot truncation: {cur.omitted.summary}"))

    return Delta(turn=cur.turn, events=events)


def enforce_delta_budget(delta: Delta, *,
                         budget: int = DELTA_TOKEN_BUDGET) -> Delta:
    """Deterministically drop the oldest events until the delta fits the
    ≤800-token budget, disclosing the elision with a warning event.

    The warning event's own size is budgeted too: the drop loop runs until
    ``events + warning`` render within the budget, so the disclosure can
    never push the report back over its own budget (the prompt assembler
    enforces the same bound and would raise)."""
    if delta.estimated_tokens() <= budget:
        return delta
    events = list(delta.events)
    elided = 0
    while True:
        warning = DeltaEvent(
            kind="warning", subject="delta",
            detail=f"delta truncated: {elided} oldest events elided to fit "
                   f"the {budget}-token budget")
        if not events:
            events.append(warning)
            break
        if estimate_tokens(
                Delta(turn=delta.turn,
                      events=events + [warning]).render()) <= budget:
            events.append(warning)
            break
        events.pop(0)
        elided += 1
    return Delta(turn=delta.turn, events=events)
