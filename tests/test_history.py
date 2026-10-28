from lechonk.core.history import DecisionEpisode, DecisionHistory
from lechonk.core.schemas import estimate_tokens


def episode(turn, notes="plan", results=()):
    return DecisionEpisode.from_parts(
        turn=turn,
        notes=notes,
        planned=[{
            "ref": "unit:9:move_north",
            "domain": "unit",
            "verb": "move_north",
            "target": "unit:9",
        }],
        results=results,
    )


def test_history_renders_intent_and_actual_outcome_without_candidate_refs():
    history = DecisionHistory()
    history.add(episode(1, results=[{
        "ref": "unit:9:move_north",
        "domain": "unit",
        "verb": "move_north",
        "target": "unit:9",
        "outcome": "rejected",
        "rejection_class": "stale_candidate",
        "readback": "none",
        "detail": "secret prompt and raw coordinates",
    }]))

    rendered = history.render()
    assert '"turn":1' in rendered
    assert "stale_candidate" in rendered
    assert "unit:9:move_north" not in rendered
    assert "secret prompt" not in rendered
    assert "CURRENT_SNAPSHOT_AND_CANDIDATES_OVERRIDE_HISTORY" in rendered


def test_history_sanitizes_and_caps_untrusted_notes():
    text = "hello\x00\x1b[31m" + ("x" * 1000)
    entry = episode(1, notes=text)
    assert "\x00" not in entry.notes
    assert "\x1b" not in entry.notes
    assert len(entry.notes) <= 240


def test_history_eviction_is_deterministic_and_disclosed():
    first = DecisionHistory(budget=350)
    second = DecisionHistory(budget=350)
    for turn in range(1, 25):
        value = episode(turn, notes=f"turn {turn} strategic note")
        first.add(value)
        second.add(value)
    rendered = first.render()
    assert rendered == second.render()
    assert '"evicted_older_episodes":' in rendered
    assert '"turn":24' in rendered
    assert estimate_tokens(rendered) <= 350


def test_history_keeps_recent_floor_when_budget_allows():
    history = DecisionHistory(budget=400)
    for turn in range(1, 10):
        history.add(episode(turn, notes="short"))
    rendered = history.render()
    for turn in range(6, 10):
        assert f'"turn":{turn}' in rendered
    assert '"turn":1' not in rendered
    assert estimate_tokens(rendered) <= 400


def test_history_stays_bounded_over_long_run():
    history = DecisionHistory()
    for turn in range(1, 301):
        history.add(episode(turn, notes="x" * 240, results=[{
            "domain": "unit", "verb": "move_north", "target": "unit:1",
            "outcome": "executed", "readback": "verified",
        }] * 12))
        assert estimate_tokens(history.render()) <= 4000
    assert len(history.episodes) <= 64
    assert '"turn":300' in history.render()


def test_history_requires_increasing_completed_turns():
    history = DecisionHistory()
    history.add(episode(4))
    try:
        history.add(episode(4))
    except ValueError as error:
        assert "increase" in str(error)
    else:
        raise AssertionError("duplicate turns must be rejected")
