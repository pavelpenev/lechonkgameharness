import pytest

from lechonk.adapter.session import AdapterSession, SessionError
from lechonk.core.recovery import validate_player_name


def test_player_name_is_trimmed_and_utf8_bounded():
    assert validate_player_name("  Le Chonk  ") == "Le Chonk"
    with pytest.raises(ValueError):
        validate_player_name("é" * 24)
    with pytest.raises(ValueError):
        validate_player_name("Le\nChonk")


def _session(name="Le Chonk"):
    return AdapterSession(workdir=".", seed=1, username="myagent",
                          player_name=name)


def test_player_name_readback_requires_exact_username_and_name():
    session = _session()
    session._actual_player_roster = [{"username": "myagent", "name": "Le Chonk"}]
    session._verify_player_name_readback()
    assert session.player_name_verified is True


def test_player_name_readback_fails_closed_on_mismatch():
    session = _session()
    session._actual_player_roster = [{"username": "myagent", "name": "Other"}]
    with pytest.raises(SessionError, match="mismatch"):
        session._verify_player_name_readback()
