#!/usr/bin/env python3
"""Render a curated screenshot storyboard into a readable highlight video.

The storyboard is deliberately explicit: each beat names one captured PNG and
supplies editorial copy.  The renderer crops the baked-in live sidebar away,
keeps the map large, and adds a clean, readable panel without inventing game
motion or claims.  It is offline and never starts a game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import wrap
from typing import Any

FPS = 30
WIDTH, HEIGHT = 1920, 1080
MAP_WIDTH, PANEL_X = 1320, 1340
MAX_SECONDS = 120


def _probe_video(path: Path) -> tuple[float, int, int, int, float]:
    """Return encoded duration and frame count, failing closed on bad output."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=nb_frames,width,height,r_frame_rate",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))
    frames = 0
    width = height = 0
    fps = 0.0
    for stream in data.get("streams", []):
        try:
            frames = max(frames, int(stream.get("nb_frames") or 0))
        except (TypeError, ValueError):
            pass
        width = max(width, int(stream.get("width") or 0))
        height = max(height, int(stream.get("height") or 0))
        rate = str(stream.get("r_frame_rate") or "0/1").split("/", 1)
        if len(rate) == 2 and float(rate[1]):
            fps = max(fps, float(rate[0]) / float(rate[1]))
    if duration <= 0 or frames <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("ffprobe could not verify encoded duration, frames, dimensions, and fps")
    return duration, frames, width, height, fps


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _safe_image(capture_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("beat image must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = capture_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(capture_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"beat image escapes capture directory: {value}") from exc
    if not resolved.is_file():
        raise ValueError(f"beat image does not exist: {resolved}")
    return resolved


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _draw_wrapped(draw, text: str, xy: tuple[int, int], width: int, font, fill, line_gap: int = 8) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        if not line and draw.textlength(word, font=font) > width:
            chunk = ""
            for char in word:
                trial = chunk + char
                if chunk and draw.textlength(trial, font=font) > width:
                    lines.append(chunk)
                    chunk = char
                else:
                    chunk = trial
            line = chunk
            continue
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    step = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_gap
    for item in lines:
        draw.text((x, y), item, font=font, fill=fill)
        y += step
    return y


def render_beat(source: Path, target: Path, beat: dict[str, Any], index: int, total: int) -> None:
    from PIL import Image, ImageDraw, ImageFilter
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    # The capture viewport is 1920x1080 and the live sidebar occupies its
    # right edge. Crop that presentation surface before adding editorial copy.
    crop_right = min(1530, image.width)
    map_image = image.crop((0, 0, crop_right, image.height))
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 24))
    map_image.thumbnail((MAP_WIDTH - 28, HEIGHT - 28), Image.Resampling.LANCZOS)
    map_x = 14 + (MAP_WIDTH - 28 - map_image.width) // 2
    map_y = 14 + (HEIGHT - 28 - map_image.height) // 2
    canvas.paste(map_image, (map_x, map_y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((MAP_WIDTH, 0, WIDTH, HEIGHT), fill=(13, 22, 36))
    draw.line((MAP_WIDTH, 0, MAP_WIDTH, HEIGHT), fill=(73, 101, 133), width=3)

    title_font = _font(44, True)
    beat_font = _font(25, True)
    body_font = _font(31)
    small_font = _font(22)
    white, muted, accent = (242, 246, 250), (171, 190, 211), (100, 205, 180)
    pad = 42
    turn = _text(beat.get("turn"), "")
    marker = f"BEAT {index + 1}/{total}" + (f"  ·  TURN {turn}" if turn else "")
    draw.text((PANEL_X + pad, 42), marker, font=beat_font, fill=accent)
    y = 102
    y = _draw_wrapped(draw, _text(beat.get("title"), "Untitled moment"), (PANEL_X + pad, y),
                      WIDTH - PANEL_X - 2 * pad, title_font, white, 10) + 25
    subtitle = _text(beat.get("subtitle"), "")
    if subtitle:
        y = _draw_wrapped(draw, subtitle, (PANEL_X + pad, y), WIDTH - PANEL_X - 2 * pad,
                          beat_font, accent, 7) + 25
    caption = _text(beat.get("caption"), "")
    if caption:
        y = _draw_wrapped(draw, caption, (PANEL_X + pad, y), WIDTH - PANEL_X - 2 * pad,
                          body_font, white, 12) + 24
    note = "Editorial caption · captured game frame"
    draw.line((PANEL_X + pad, HEIGHT - 108, WIDTH - pad, HEIGHT - 108), fill=(55, 75, 98), width=2)
    draw.text((PANEL_X + pad, HEIGHT - 82), note, font=small_font, fill=muted)
    canvas.save(target, "PNG", optimize=True)


def render_story(storyboard_path: Path, capture_dir: Path, output_dir: Path) -> dict[str, Any]:
    story = json.loads(storyboard_path.read_text(encoding="utf-8"))
    beats = story.get("beats") if isinstance(story, dict) else None
    if not isinstance(beats, list) or not beats:
        raise ValueError("storyboard must contain a non-empty beats array")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    total_frames = 0
    normalized: list[dict[str, Any]] = []
    for beat in beats:
        if not isinstance(beat, dict):
            raise ValueError("each beat must be an object")
        requested_hold = float(beat.get("hold_seconds", 3.0))
        if not math.isfinite(requested_hold) or requested_hold <= 0 or requested_hold > MAX_SECONDS:
            raise ValueError("hold_seconds must be > 0 and <= 120")
        hold_frames = max(1, round(requested_hold * FPS))
        total_frames += hold_frames
        item = dict(beat)
        item["source_image"] = str(_safe_image(capture_dir, beat.get("image")))
        item["requested_hold_seconds"] = requested_hold
        item["hold_seconds"] = hold_frames / FPS
        item["hold_frames"] = hold_frames
        normalized.append(item)
    if total_frames > MAX_SECONDS * FPS:
        raise ValueError(f"storyboard duration {total_frames / FPS:.2f}s exceeds {MAX_SECONDS}s")
    frames_dir = Path(tempfile.mkdtemp(prefix="story-frames-", dir=str(staging)))
    try:
        concat_lines: list[str] = []
        for index, beat in enumerate(normalized):
            frame = frames_dir / f"beat-{index:03d}.png"
            render_beat(Path(beat["source_image"]), frame, beat, index, len(normalized))
            concat_lines.append(f"file '{frame.as_posix()}'")
            concat_lines.append(f"duration {beat['hold_frames'] / FPS:.6f}")
        concat_lines.append(f"file '{(frames_dir / f'beat-{len(normalized)-1:03d}.png').as_posix()}'")
        concat = frames_dir / "concat.txt"
        concat.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        video = staging / "story.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-vf", f"fps={FPS},format=yuv420p", "-frames:v", str(total_frames), "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-movflags", "+faststart", str(video),
        ], check=True)
        actual_duration, actual_frames, width, height, fps = _probe_video(video)
        if actual_frames != total_frames or (width, height) != (WIDTH, HEIGHT) or abs(fps - FPS) > 0.01:
            raise ValueError(f"encoded frame count {actual_frames} differs from storyboard {total_frames}")
        storyboard_copy = staging / "storyboard.json"
        storyboard_copy.write_text(json.dumps({**story, "beats": normalized}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        provenance = {
            "schema": "capture-story-v1", "source_capture_dir": str(capture_dir.resolve()),
            "storyboard": str(storyboard_path.resolve()), "video": str((output_dir / "story.mp4").resolve()),
            "requested_duration_seconds": total_frames / FPS,
            "actual_duration_seconds": actual_duration, "actual_frame_count": actual_frames,
            "beat_count": len(normalized),
            "beat_frame_ranges": [],
            "source_images": [{"path": x["source_image"], "sha256": hashlib.sha256(Path(x["source_image"]).read_bytes()).hexdigest()} for x in normalized],
        }
        cursor = 0
        for index, item in enumerate(normalized):
            end = cursor + int(item["hold_frames"])
            provenance["beat_frame_ranges"].append({
                "beat": index + 1, "title": _text(item.get("title")),
                "start_frame": cursor, "end_frame_exclusive": end,
                "midpoint_frame": cursor + max(0, int(item["hold_frames"]) // 2),
            })
            cursor = end
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        staging.replace(output_dir)
        return {"video": str(output_dir / "story.mp4"), "duration_seconds": actual_duration,
                "frame_count": actual_frames, "width": width, "height": height, "fps": fps,
                "beat_count": len(normalized), "output_dir": str(output_dir)}
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        if staging.exists() and not output_dir.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(render_story(args.storyboard.resolve(), args.capture_dir.resolve(), args.output_dir.resolve()), indent=2))
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"render_capture_story: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
