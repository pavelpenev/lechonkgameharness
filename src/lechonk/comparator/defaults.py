# SPDX-License-Identifier: MIT
"""lechonk.comparator.defaults — the degraded-default entry point (§7.4 line 163).

DISTINCT from the comparator (plan §3 line 44): the degraded defaults are the
minimal end-of-turn fallback used when the model is unresponsive after the
repair cap. They PRESERVE all current orders and production and finish the
phase; only NEW mandatory choices resolve via the same published priority
lists (lechonk.comparator.scripted), and each such resolution is logged with
reason ``scripted_default``.

New mandatory choices (the only resolutions emitted):
- research completed (no active research target) -> research priority list;
- city production empty -> production priority list (per city).

Everything else is preserved, not acted on: no unit actions (a settler with a
legal found-city candidate is left alone), no production switches (a city
already producing keeps its target), no buys, no diplomacy, no proactive
offers of any kind. Both entry points are separate callables producing a
decision log; tests/test_comparator.py proves the behavioral difference.
"""

from __future__ import annotations

from lechonk.core.schemas import CandidateTable, Snapshot
from lechonk.comparator.scripted import (
    Decision,
    DecisionLog,
    _by_actor,
    own_cities,
    production_context,
    production_pick,
    research_pick,
)

REASON = "scripted_default"  # §7.4 line 163: logged with reason scripted_default


def decide(snapshot: Snapshot, table: CandidateTable) -> DecisionLog:
    """The degraded-default entry point: preserve everything, resolve only
    the new mandatory choices (research completed, city production empty) via
    the published priority lists, and finish the phase."""
    decisions: list[Decision] = []

    # Research completed: no active target -> resolve via the priority list.
    if snapshot.player.research.current_target_id is None:
        pick = research_pick(snapshot, table)
        if pick is not None:
            decisions.append(Decision(
                domain="tech", candidate_id=pick.id, verb=pick.verb,
                actor_id=pick.actor_id, reason=REASON, trigger="research_completed",
                detail=(f"research completed with no active target; resolved "
                        f"to {pick.payload.get('tech_name')} via the "
                        f"published priority list")))

    # City production empty -> resolve via the production ladder. Cities
    # with a current production target are preserved, not switched.
    ctx = production_context(snapshot)
    city_cands = _by_actor(table, "city")
    for city in own_cities(snapshot):
        if city.production is not None:
            continue  # preserve current production
        produce_cands = [c for c in city_cands.get(str(city.id), [])
                         if c.verb.startswith("produce_")]
        if not produce_cands:
            continue
        pick, tier = production_pick(city, produce_cands, ctx)
        if pick is not None:
            decisions.append(Decision(
                domain="city", candidate_id=pick.id, verb=pick.verb,
                actor_id=pick.actor_id, reason=REASON, trigger="production_empty",
                detail=(f"city {city.name} production empty; resolved to "
                        f"{pick.payload.get('production')} via the published "
                        f"production ladder ({tier})")))

    return DecisionLog(policy="degraded_default", turn=snapshot.turn,
                       decisions=tuple(decisions))
