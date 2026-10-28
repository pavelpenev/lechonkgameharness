from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_capture_story import render_story


def _story(tmp_path: Path, holds: list[float]) -> tuple[Path, Path]:
    capture = tmp_path / "capture"
    image_dir = capture / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "frame.png"
    Image.new("RGB", (1920, 1080), (20, 80, 120)).save(image)
    storyboard = tmp_path / "story.json"
    storyboard.write_text(json.dumps({"beats": [
        {"image": "images/frame.png", "turn": i + 1, "title": f"Beat {i + 1}",
         "caption": "A captured game frame.", "hold_seconds": hold}
        for i, hold in enumerate(holds)
    ]}), encoding="utf-8")
    return capture, storyboard


def test_story_rejects_path_escape_and_nonfinite_hold(tmp_path: Path) -> None:
    capture, storyboard = _story(tmp_path, [1.0])
    payload = json.loads(storyboard.read_text())
    payload["beats"][0]["image"] = "../outside.png"
    storyboard.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        render_story(storyboard, capture, tmp_path / "out")
    payload["beats"][0]["image"] = "images/frame.png"
    payload["beats"][0]["hold_seconds"] = "nan"
    storyboard.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hold_seconds"):
        render_story(storyboard, capture, tmp_path / "out2")


def test_seven_noninteger_beats_match_verified_video_metadata(tmp_path: Path) -> None:
    capture, storyboard = _story(tmp_path, [0.37 + i * 0.03 for i in range(7)])
    result = render_story(storyboard, capture, tmp_path / "out")
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert result["beat_count"] == 7
    assert provenance["actual_frame_count"] == sum(item["hold_frames"] for item in json.loads(
        (tmp_path / "out" / "storyboard.json").read_text())["beats"])
    assert abs(provenance["actual_duration_seconds"] - result["duration_seconds"]) < 0.001
