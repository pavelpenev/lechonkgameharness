# SPDX-License-Identifier: MIT
"""FW-08 (C6): compute_delta elision honesty fixtures.

A unit listed in a snapshot's omitted.elided_unit_ids was dropped by the
observation budget, not by the game: it must produce neither a loss nor an
appearance. A genuinely fog-removed own unit (absent and not elided) still
reports as a loss.
"""

from lechonk.core.delta import compute_delta
from lechonk.core.schemas import MapView, PlayerView, Snapshot, UnitView


def _unit(uid: int, x: int = 5, y: int = 5) -> UnitView:
    return UnitView(id=uid, is_own=True, type_name="warriors", x=x, y=y)


def _snapshot(turn: int, units: list[UnitView],
              elided: list[int] | None = None) -> Snapshot:
    snap = Snapshot(
        turn=turn,
        player=PlayerView(player_id=0),
        units=units,
        map_view=MapView(width=10, height=10),
    )
    if elided:
        snap.omitted.elided_unit_ids = list(elided)
    return snap


def test_unit_elided_in_cur_not_reported_lost():
    prev = _snapshot(1, [_unit(7)])
    cur = _snapshot(2, [], elided=[7])
    delta = compute_delta(prev, cur)
    assert delta.events == []


def test_unit_elided_in_prev_not_reported_new():
    prev = _snapshot(1, [], elided=[9])
    cur = _snapshot(2, [_unit(9)])
    delta = compute_delta(prev, cur)
    assert delta.events == []


def test_fog_removed_unit_still_reported_lost():
    prev = _snapshot(1, [_unit(3), _unit(7)])
    cur = _snapshot(2, [_unit(3)])
    delta = compute_delta(prev, cur)
    lost = [e for e in delta.events if e.kind == "unit_lost_own"]
    assert [e.subject for e in lost] == ["unit:7"]


def test_elision_does_not_mask_genuine_loss():
    prev = _snapshot(1, [_unit(3), _unit(7)])
    cur = _snapshot(2, [], elided=[3])
    delta = compute_delta(prev, cur)
    lost = [e for e in delta.events if e.kind == "unit_lost_own"]
    assert [e.subject for e in lost] == ["unit:7"]
