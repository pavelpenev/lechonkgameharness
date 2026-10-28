from scripts.model_repetition import RepetitionDetector


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _drive(detector, clock, text="hmm "):
    for _ in range(16):
        detector.observe(reasoning=text * 4096)
        clock.advance(15)


def test_repetition_warning_is_metadata_only_and_chunk_boundary_independent():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="c1", turn=212, attempt=1, run_id="r")
    _drive(detector, clock, "hmm — ")
    assert events
    warning = next(event for event in events if "unique_block_ratio" in event)
    assert warning["event"] == "model_stream_repetition_warning"
    assert warning["action"] == "none; stream continues"
    assert warning["call_id"] == "c1"
    assert warning["unique_block_ratio"] <= 0.10
    assert "hmm" not in str(warning)


def test_repetition_detector_does_not_warn_for_diverse_prose():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="c2", turn=1, attempt=1, run_id="r")
    for i in range(16):
        detector.observe(reasoning=" ".join(
            f"paragraph_{i}_token_{j}" for j in range(3000)))
        clock.advance(15)
    assert not events


def test_repetition_detector_window_is_bounded_and_resets_per_call():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="old", turn=1, attempt=1, run_id="r")
    for _ in range(1000):
        detector.observe(answer="x" * 128)
        clock.advance(1)
    assert all(len(lane["blocks"]) <= detector.window_size
               for lane in detector._lanes.values())
    detector.reset(call_id="new", turn=2, attempt=1, run_id="r")
    assert detector.delta_count == 0
    assert detector.call_id == "new"
    assert all(not lane["blocks"] for lane in detector._lanes.values())


def test_casefolded_blocks_are_identical_under_arbitrary_chunk_splits():
    text = ("Straße — Δelta  " * 2000)
    whole = RepetitionDetector()
    split = RepetitionDetector()
    whole.observe(reasoning=text)
    for char in text:
        split.observe(reasoning=char)
    assert list(whole._lanes["reasoning"]["blocks"]) == list(
        split._lanes["reasoning"]["blocks"])
    assert whole._lanes["reasoning"]["pending"] == split._lanes[
        "reasoning"]["pending"]


def test_warning_sink_failure_cannot_break_stream_observation():
    clock = FakeClock()

    def broken_sink(_event):
        raise RuntimeError("sink unavailable")

    detector = RepetitionDetector(clock=clock, warning_sink=broken_sink)
    detector.reset(call_id="c3", turn=3, attempt=1, run_id="r")
    _drive(detector, clock, "hmm, hmm — ")
    detector.observe(reasoning="still streaming")


def test_low_character_entropy_signal_catches_variable_delimiters():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="entropy", turn=3, attempt=1, run_id="r")
    # Delimiters vary, but the alphabet remains deliberately small.
    pattern = "hmm — h|hmm ~ h;hmm / h,"
    for _ in range(16):
        detector.observe(reasoning=(pattern * 4096)[:4096])
        clock.advance(15)
    entropy_events = [event for event in events
                      if event.get("signal") == "low_character_entropy_v1"]
    assert entropy_events
    assert entropy_events[0]["entropy_bits"] <= 3.75
    assert entropy_events[0]["action"] == "none; stream continues"


def test_low_entropy_hysteresis_clears_on_diverse_stream():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="entropy2", turn=3, attempt=1, run_id="r")
    low = "hmm — h|hmm ~ h;"
    for _ in range(16):
        detector.observe(reasoning=(low * 4096)[:4096])
        clock.advance(15)
    first_count = len(events)
    diverse = " ".join(f"token_{i}" for i in range(2000))
    for _ in range(8):
        detector.observe(reasoning=diverse)
        clock.advance(15)
    assert len(events) == first_count
    for _ in range(2):
        detector.observe(reasoning=(low * 4096)[:4096])
        clock.advance(15)
    assert len(events) == first_count

def test_normal_prose_json_and_base64_do_not_trigger_entropy_signal():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="normal", turn=4, attempt=1, run_id="r")
    import base64
    prose = " ".join(f"sentence_{i}_contains_distinct_words" for i in range(3000))
    payload = base64.b64encode(bytes(range(256))).decode("ascii")
    for i in range(16):
        detector.observe(answer=(prose if i % 2 else payload * 40))
        clock.advance(15)
    assert not [event for event in events
                 if event.get("signal") == "low_character_entropy_v1"]


def test_entropy_evaluations_are_time_spaced_during_bursty_stream():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="burst", turn=5, attempt=1, run_id="r")
    fragment = "hmm|h"
    for _ in range(240):
        for offset in range(8):
            detector.observe(reasoning=(fragment * 40)[offset * 512:(offset + 1) * 512])
        clock.advance(1)
    assert any(event.get("signal") == "low_character_entropy_v1"
               for event in events)


def test_entropy_three_evaluations_at_exact_ten_second_intervals_trigger():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="ten-second", turn=7, attempt=1, run_id="r")
    low = "hmm|h"
    for _ in range(16):
        detector.observe(reasoning=(low * 4096)[:4096])
        clock.advance(10)
    assert any(event.get("signal") == "low_character_entropy_v1"
               for event in events)


def test_entropy_candidate_clears_when_stream_enters_hysteresis_band():
    clock = FakeClock()
    events = []
    detector = RepetitionDetector(clock=clock, warning_sink=events.append)
    detector.reset(call_id="band", turn=6, attempt=1, run_id="r")
    low = "hmm — h|hmm ~ h;"
    for _ in range(16):
        detector.observe(reasoning=(low * 4096)[:4096])
        clock.advance(15)
    first_count = len(events)
    diverse = " ".join(f"unique_token_{i}" for i in range(3000))
    for _ in range(16):
        detector.observe(reasoning=diverse)
        clock.advance(15)
    lane = detector._lanes["reasoning"]
    assert not lane["entropy_evaluations"]
    assert lane["entropy_low_since"] is None
    for _ in range(2):
        detector.observe(reasoning=(low * 4096)[:4096])
        clock.advance(15)
    assert len(events) == first_count
