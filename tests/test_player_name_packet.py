"""Pregame ruler-name selection stays separate from the login username."""

from types import SimpleNamespace

from civrealm.freeciv.civ_controller import CivController
from civrealm.freeciv.utils.fc_types import packet_nation_select_req


class _ControllerHarness:
    def __init__(self, *, ai: bool):
        self.username = "agentabcxyz"
        self.rule_ctrl = SimpleNamespace(
            nations={0: {"leader_name": ["Default Leader"], "style": 7}},
            get_nation_options=lambda: [0],
        )
        self.clstate = SimpleNamespace(player_num=lambda: 0)
        self.player_ctrl = SimpleNamespace(
            get_player=lambda _player_id: {
                "name": "Generated Username",
                "flags": [1 if ai else 0],
            }
        )
        self.requests = []
        self.ws_client = SimpleNamespace(send_request=self.requests.append)


def _choose(controller, monkeypatch, *, ai: bool):
    harness = _ControllerHarness(ai=ai)
    controller.__dict__.update(harness.__dict__)
    monkeypatch.setattr("civrealm.freeciv.civ_controller.random.randint", lambda *_: 0)
    controller.pregame_choose_nation(0)
    assert len(harness.requests) == 1
    return harness.requests[0], harness.username


def test_human_packet_uses_display_name_without_changing_username(monkeypatch):
    controller = CivController.__new__(CivController)
    # The caller (the harness runner) supplies this optional display setting.
    monkeypatch.setitem(__import__("civrealm.configs", fromlist=["fc_args"]).fc_args,
                        "player_name", "Le Chonk")

    packet, username = _choose(controller, monkeypatch, ai=False)

    assert packet["pid"] == packet_nation_select_req
    assert packet["name"] == "Le Chonk"
    assert username == "agentabcxyz"


def test_ai_packet_keeps_nation_leader_name(monkeypatch):
    controller = CivController.__new__(CivController)
    monkeypatch.setitem(__import__("civrealm.configs", fromlist=["fc_args"]).fc_args,
                        "player_name", "Le Chonk")

    packet, username = _choose(controller, monkeypatch, ai=True)

    assert packet["name"] == "Default Leader"
    assert username == "agentabcxyz"
