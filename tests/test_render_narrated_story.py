from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import pytest

from scripts.render_narrated_story import FPS, _video_frames, render, validate_story


def _story(tmp_path: Path, duration: float = 1.0) -> Path:
    from PIL import Image

    image = tmp_path / "frame.png"
    Image.new("RGB", (320, 180), (40, 80, 120)).save(image)
    path = tmp_path / "story.json"
    path.write_text(json.dumps({"scenes": [
        {"kind": "intro", "title": "LE CHONK", "duration": duration},
        {"kind": "gameplay", "image": str(image), "title": "A city", "duration": duration},
    ]}), encoding="utf-8")
    return path


def test_validate_supports_intro_and_absolute_multiroot_image(tmp_path):
    _, scenes, total = validate_story(_story(tmp_path, 1.5))
    assert [scene["kind"] for scene in scenes] == ["intro", "gameplay"]
    assert scenes[1]["image_path"].is_absolute()
    assert total == pytest.approx(3.0)


def test_validate_rejects_missing_gameplay_image(tmp_path):
    path = tmp_path / "story.json"
    path.write_text(json.dumps({"scenes": [{"kind": "gameplay", "duration": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="image"):
        validate_story(path)


def test_validate_enforces_two_minute_cap(tmp_path):
    path = _story(tmp_path, 61)
    with pytest.raises(ValueError, match="hard limit"):
        validate_story(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_render_with_apostrophe_output_publishes_exact_concat_artifacts(tmp_path):
    storyboard = _story(tmp_path, 1.0)
    output = tmp_path / "publisher's cut"
    result = render(storyboard, output)
    expected_frames = 2 * FPS
    assert result["scene_count"] == 2
    assert output.joinpath("story.mp4").is_file()
    assert output.joinpath("silent.mp4").is_file()
    assert output.joinpath("audio.wav").is_file()
    assert _video_frames(output / "story.mp4") == expected_frames
    assert _video_frames(output / "silent.mp4") == expected_frames
    with wave.open(str(output / "audio.wav"), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24000
        assert audio.getnframes() == expected_frames * 800
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema"] == "narrated-story-v1"
    assert provenance["video"] == "story.mp4"
    assert provenance["silent_video"] == "silent.mp4"
    assert len(provenance["scenes"]) == 2
