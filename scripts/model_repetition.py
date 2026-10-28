"""Metadata-only detector for pathological repeated model streams.

The detector never changes or persists model output. It keeps a bounded rolling
window of fingerprints and a small transient block assembly buffer while a
single physical request is in flight. It emits a warning callback when that
request is unusually repetitive for a sustained period.
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
import time
from typing import Any, Callable


WarningSink = Callable[[dict[str, Any]], None]


class RepetitionDetector:
    """Bounded, warning-only repetition detector for one provider call."""

    block_size = 64
    window_size = 256

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 warning_sink: WarningSink | None = None) -> None:
        self._clock = clock
        self._warning_sink = warning_sink
        self.reset()

    def reset(self, *, call_id: str | None = None, turn: Any = None,
              attempt: Any = None, run_id: str | None = None) -> None:
        self.call_id = call_id
        self.turn = turn
        self.attempt = attempt
        self.run_id = run_id
        self.started_at = self._clock()
        self._lanes = {"reasoning": self._new_lane(), "answer": self._new_lane()}
        self.delta_count = 0
        self._warned = False
        self._last_warning_at: float | None = None
        self._entropy_warned = {"reasoning": False, "answer": False}
        self._entropy_last_warning_at = {"reasoning": None, "answer": None}

    @staticmethod
    def _new_lane() -> dict[str, Any]:
        return {"pending": "", "blocks": deque(maxlen=RepetitionDetector.window_size),
                "chars": 0, "evaluations": deque(maxlen=3),
                "low_since": None, "entropy_blocks": deque(maxlen=8),
                "entropy_pending": [0] * 256, "entropy_pending_chars": 0,
                "entropy_evaluations": deque(maxlen=3),
                "entropy_low_since": None, "entropy_last_evaluation_at": None}

    @staticmethod
    def _normalize(value: Any) -> str:
        # Per-chunk case-folding is stable under arbitrary stream chunking.
        # Whitespace collapse and cross-chunk Unicode normalization are
        # deliberately avoided because both can depend on provider framing.
        # Per-codepoint case folding and whitespace mapping are stable under
        # arbitrary provider chunk boundaries.
        return "".join(" " if char.isspace() else char.casefold()
                       for char in str(value))

    def observe(self, *, reasoning: Any = "", answer: Any = "") -> None:
        self.delta_count += 1
        for lane_name, value in (("reasoning", reasoning), ("answer", answer)):
            if value:
                self._observe_lane(lane_name, self._normalize(value))

    def _observe_lane(self, lane_name: str, text: str) -> None:
        lane = self._lanes[lane_name]
        lane["pending"] += text
        lane["chars"] += len(text)
        while len(lane["pending"]) >= self.block_size:
            block, lane["pending"] = lane["pending"][:self.block_size], lane["pending"][self.block_size:]
            lane["blocks"].append(hashlib.sha256(block.encode("utf-8")).digest())
        self._observe_entropy(lane_name, lane, text)
        self._maybe_warn(lane_name, lane)

    def _observe_entropy(self, lane_name: str, lane: dict[str, Any], text: str) -> None:
        for char in text:
            bucket = ord(char) % 256
            lane["entropy_pending"][bucket] += 1
            lane["entropy_pending_chars"] += 1
            if lane["entropy_pending_chars"] == 4096:
                lane["entropy_blocks"].append(tuple(lane["entropy_pending"]))
                lane["entropy_pending"] = [0] * 256
                lane["entropy_pending_chars"] = 0
                self._maybe_entropy_warn(lane_name, lane)

    def _maybe_entropy_warn(self, lane_name: str, lane: dict[str, Any]) -> None:
        blocks = lane["entropy_blocks"]
        if len(blocks) < 8:
            return
        now = self._clock()
        if now - self.started_at < 120:
            return
        histogram = [sum(block[index] for block in blocks)
                     for index in range(256)]
        total = sum(histogram)
        if total <= 0:
            return
        entropy = -sum((count / total) * math.log2(count / total)
                       for count in histogram if count)
        evaluations = lane["entropy_evaluations"]
        if entropy > 3.75:
            # A candidate must remain continuously low.  Values in the
            # hysteresis band are treated as recovery, so stale evaluations
            # cannot be carried across a high-entropy excursion.
            lane["entropy_low_since"] = None
            evaluations.clear()
            return
        last_evaluation = lane["entropy_last_evaluation_at"]
        if last_evaluation is not None and now - last_evaluation < 10:
            return
        lane["entropy_last_evaluation_at"] = now
        if lane["entropy_low_since"] is None:
            lane["entropy_low_since"] = now
        evaluations.append((now, entropy))
        if (len(evaluations) < 3 or lane["entropy_low_since"] is None or
                now - lane["entropy_low_since"] < 30):
            return
        last = self._entropy_last_warning_at[lane_name]
        if self._entropy_warned[lane_name] and (last is None or now - last < 300):
            return
        self._entropy_warned[lane_name] = True
        self._entropy_last_warning_at[lane_name] = now
        if self._warning_sink is None:
            return
        try:
            self._warning_sink({
                "event": "model_stream_repetition_warning",
                "run_id": self.run_id,
                "turn": self.turn,
                "call_id": self.call_id,
                "attempt": self.attempt,
                "lane": lane_name,
                "elapsed_s": max(0.0, now - self.started_at),
                "delta_count": self.delta_count,
                "char_count": lane["chars"],
                "window_size": len(blocks),
                "signal": "low_character_entropy_v1",
                "entropy_bits": entropy,
                "action": "none; stream continues",
            })
        except Exception:
            pass

    def _maybe_warn(self, lane_name: str, lane: dict[str, Any]) -> None:
        now = self._clock()
        if lane["chars"] < 64 * 1024 or now - self.started_at < 120:
            return
        blocks = lane["blocks"]
        if len(blocks) < 16:
            return
        unique_ratio = len(set(blocks)) / len(blocks)
        if unique_ratio > 0.10:
            lane["low_since"] = None
            return
        if lane["low_since"] is None:
            lane["low_since"] = now
        lane["evaluations"].append((now, unique_ratio))
        evals = lane["evaluations"]
        if len(evals) < 3 or evals[-1][0] - evals[0][0] < 30:
            return
        if self._warned and (self._last_warning_at is None or now - self._last_warning_at < 300):
            return
        self._warned = True
        self._last_warning_at = now
        if self._warning_sink is None:
            return
        try:
            self._warning_sink({
                "event": "model_stream_repetition_warning",
                "run_id": self.run_id,
                "turn": self.turn,
                "call_id": self.call_id,
                "attempt": self.attempt,
                "lane": lane_name,
                "elapsed_s": max(0.0, now - self.started_at),
                "delta_count": self.delta_count,
                "char_count": lane["chars"],
                "window_size": len(blocks),
                "unique_block_ratio": unique_ratio,
                "action": "none; stream continues",
            })
        except Exception:
            # Observability must never break the provider stream.
            pass
