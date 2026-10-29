#!/usr/bin/env python3
# SCRIPT-METADATA:
# name: render_narrated_story
# description: Render a polished, evidence-linked Freeciv story with title cards and narration.
# tags: video,freeciv,editorial,ffmpeg
# language: python
"""Render a fancy screenshot-first gameplay story.

The input is an editorial JSON document with a ``scenes`` list.  This script
never edits capture roots: source images are copied only by reference into the
provenance document.  Each scene becomes an encoded video chapter with a
small Ken-Burns push on gameplay frames, then chapters and optional WAV
narration are concatenated.  A silent fallback is always written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

FPS = 30
WIDTH, HEIGHT = 1920, 1080
MAX_SECONDS = 120.0
NAVY = (11, 20, 36)
CREAM = (244, 239, 227)
MUTED = (157, 174, 198)
LIME = (212, 237, 100)
PANEL = (15, 29, 48)


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = ("Lato-Heavy.ttf", "Lato-Bold.ttf") if bold else ("Lato-Regular.ttf", "Lato-Medium.ttf")
    roots = (Path("/usr/share/fonts/truetype/lato"), Path("/usr/share/fonts/truetype/dejavu"))
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words = str(text or "").split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _draw_lines(draw, lines, xy, font, fill, gap=8):
    x, y = xy
    box = font.getbbox("Ag")
    step = box[3] - box[1] + gap
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += step
    return y


def _path(value: Any, label: str, must_exist: bool = True) -> Path | None:
    if value is None:
        if must_exist:
            raise ValueError(f"{label} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a path string")
    p = Path(value).expanduser().resolve()
    if must_exist and not p.is_file():
        raise ValueError(f"{label} does not exist: {p}")
    return p


def _duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    value = float(result.stdout.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid audio duration for {path}")
    return value


def _audio_frames(path: Path) -> int:
    import wave
    with wave.open(str(path), "rb") as stream:
        value = stream.getnframes()
    if value <= 0:
        raise ValueError(f"ffprobe could not verify audio frames for {path}")
    return value


def _video_frames(path: Path) -> int:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    value = int(result.stdout.strip())
    if value <= 0:
        raise ValueError(f"ffprobe could not verify video frames for {path}")
    return value


def validate_story(story_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    story = json.loads(story_path.read_text(encoding="utf-8"))
    scenes = story.get("scenes") if isinstance(story, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("storyboard must contain a non-empty scenes array")
    normalized = []
    total = 0.0
    for index, raw in enumerate(scenes, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"scene {index} must be an object")
        kind = str(raw.get("kind", "gameplay"))
        if kind not in {"intro", "gameplay", "outro"}:
            raise ValueError(f"scene {index} has invalid kind {kind!r}")
        image = _path(raw.get("image"), f"scene {index} image", must_exist=(kind == "gameplay"))
        crop = raw.get("crop")
        if crop is not None:
            if image is None or not isinstance(crop, list) or len(crop) != 4:
                raise ValueError(f"scene {index} crop requires [left,top,right,bottom] and an image")
            if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in crop):
                raise ValueError(f"scene {index} crop coordinates must be finite numbers")
            from PIL import Image
            with Image.open(image) as probe:
                left, top, right, bottom = (int(v) for v in crop)
                if left < 0 or top < 0 or right > probe.width or bottom > probe.height or right <= left or bottom <= top:
                    raise ValueError(f"scene {index} crop is outside image bounds")
        audio = _path(raw.get("audio"), f"scene {index} audio", must_exist=False)
        requested = float(raw.get("duration", 5.0))
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError(f"scene {index} duration must be positive")
        audio_seconds = _duration(audio) if audio else 0.0
        actual = max(requested, audio_seconds + 0.8 if audio else requested)
        actual = math.ceil(actual * FPS) / FPS
        item = dict(raw)
        item.update(index=index, kind=kind, image_path=image, audio_path=audio,
                   requested_duration=requested, audio_duration=audio_seconds, duration=actual)
        normalized.append(item)
        total += actual
    if total > MAX_SECONDS + 1e-6:
        raise ValueError(f"story duration {total:.2f}s exceeds hard limit {MAX_SECONDS:.0f}s")
    return story, normalized, total


def _frame(scene: dict[str, Any], source, frame_no: int, scene_frames: int, total_scenes: int):
    from PIL import Image, ImageDraw, ImageOps
    progress = frame_no / max(1, scene_frames - 1)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(canvas)
    if scene["kind"] == "gameplay":
        # ``source`` is loaded and cropped once per scene by ``render``.
        original = source
        zoom = 1.0 + 0.04 * progress
        target_w, target_h = 1240, 900
        fitted = ImageOps.fit(original, (round(target_w * zoom), round(target_h * zoom)), method=Image.Resampling.LANCZOS, centering=(0.5 + 0.02 * progress, 0.5))
        left = (fitted.width - target_w) // 2
        top = (fitted.height - target_h) // 2
        canvas.paste(fitted.crop((left, top, left + target_w, top + target_h)), (40, 72))
        draw.rectangle((40, 72, 1280, 972), outline=(65, 91, 121), width=3)
        entrance = min(1.0, progress / 0.4)
        panel_x = 1304 + int((1.0 - entrance) * 90)
        draw.rectangle((panel_x, 0, WIDTH, HEIGHT), fill=PANEL)
        draw.line((panel_x, 0, panel_x, HEIGHT), fill=LIME, width=3)
        draw.text((panel_x + 46, 48), str(scene.get("kicker", f"SCENE {scene['index']}" )).upper(), font=_font(24, True), fill=LIME)
        turn = scene.get("turn")
        if turn is not None:
            draw.text((panel_x + 46, 90), f"TURN {turn}", font=_font(64, True), fill=CREAM)
        y = 188 if turn is not None else 126
        y = _draw_lines(draw, _wrap(draw, scene.get("title", "Untitled moment"), _font(43, True), 480), (panel_x + 46, y), _font(43, True), CREAM, 8) + 28
        y = _draw_lines(draw, _wrap(draw, scene.get("caption", ""), _font(29), 480), (panel_x + 46, y), _font(29), (225, 231, 238), 11) + 22
        chips = scene.get("chips") or []
        for chip in chips[:4]:
            label = str(chip).upper()
            draw.rounded_rectangle((panel_x + 46, y, min(1790, panel_x + 46 + 55 + int(draw.textlength(label, font=_font(19, True)))), y + 34), radius=8, fill=(34, 54, 73), outline=(86, 113, 138), width=1)
            draw.text((panel_x + 61, y + 7), label, font=_font(19, True), fill=LIME)
            y += 47
        draw.text((panel_x + 46, 1016), f"{scene['index']:02d} / {total_scenes:02d}   ·   CAPTURED FRAME", font=_font(18), fill=MUTED)
    else:
        # Intro/outro title cards use a restrained grid and animated accent line.
        for x in range(0, WIDTH, 80):
            draw.line((x, 0, x, HEIGHT), fill=(16, 31, 51), width=1)
        for y in range(0, HEIGHT, 80):
            draw.line((0, y, WIDTH, y), fill=(16, 31, 51), width=1)
        draw.rectangle((120, 188, 1800, 192), fill=LIME)
        line_x = 120 + int(1680 * progress)
        draw.rectangle((120, 188, line_x, 192), fill=CREAM)
        title = str(scene.get("title", "LE CHONK"))
        draw.text((120, 270), title, font=_font(116, True), fill=CREAM)
        draw.text((126, 415), str(scene.get("caption", "ONE MODEL. A WORLD TO BUILD.")), font=_font(38), fill=LIME)
        draw.text((126, 510), str(scene.get("kicker", "FREECIV · TWO AI OPPONENTS")), font=_font(25, True), fill=MUTED)
    # Global editorial progress accent.
    draw.rectangle((40, 1038, 1880, 1042), fill=(38, 57, 78))
    draw.rectangle((40, 1038, 40 + int(1840 * ((scene["index"] - 1 + progress) / total_scenes)), 1042), fill=LIME)
    fade = max(0.0, (0.25 - progress) / 0.25, (progress - 0.75) / 0.25)
    if fade > 0:
        canvas = Image.blend(canvas, Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0)), min(1.0, fade))
    return canvas


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)


def render(storyboard: Path, output_dir: Path) -> dict[str, Any]:
    story, scenes, total = validate_story(storyboard)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        chapters = staging / "chapters"
        frames_root = staging / "frames"
        chapters.mkdir(); frames_root.mkdir()
        chapter_paths: list[Path] = []
        audio_paths: list[Path] = []
        timeline = []
        cursor = 0.0
        for scene in scenes:
            frames = round(scene["duration"] * FPS)
            scene_dir = frames_root / f"scene-{scene['index']:03d}"
            scene_dir.mkdir()
            source = None
            if scene["kind"] == "gameplay":
                from PIL import Image
                with Image.open(scene["image_path"]) as opened:
                    source = opened.convert("RGB")
                crop = scene.get("crop")
                if crop is not None:
                    source = source.crop(tuple(int(v) for v in crop))
                else:
                    source = source.crop((0, 0, min(1530, source.width), source.height))
            chapter = chapters / f"scene-{scene['index']:03d}.mp4"
            encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-frames:v", str(frames), "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(chapter)], stdin=subprocess.PIPE)
            assert encoder.stdin is not None
            try:
                for n in range(frames):
                    encoder.stdin.write(_frame(scene, source, n, frames, len(scenes)).tobytes())
                encoder.stdin.close()
                if encoder.wait(timeout=120) != 0:
                    raise RuntimeError(f"ffmpeg failed encoding scene {scene['index']}")
            except Exception:
                encoder.kill()
                encoder.wait()
                raise
            chapter_paths.append(chapter)
            audio = scene.get("audio_path")
            normalized_audio = staging / f"audio-{scene['index']:03d}.wav"
            expected_samples = frames * 800
            if audio:
                pre_audio = staging / f"pre-audio-{scene['index']:03d}.wav"
                _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", str(audio), "-af", "aresample=24000,adelay=300:all=1,loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(pre_audio)])
                _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", str(pre_audio), "-af", f"apad=whole_len={expected_samples},atrim=end_sample={expected_samples},asetpts=N/SR/TB", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized_audio)])
            else:
                _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-af", f"atrim=end_sample={expected_samples},asetpts=N/SR/TB", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized_audio)])
            if _audio_frames(normalized_audio) != expected_samples:
                raise ValueError(f"scene {scene['index']} audio length is not exactly {expected_samples} samples")
            audio_paths.append(normalized_audio)
            timeline.append({"index": scene["index"], "id": scene.get("id", f"scene-{scene['index']}"), "kind": scene["kind"], "start_seconds": cursor, "duration_seconds": scene["duration"], "image": str(scene["image_path"]) if scene["image_path"] else None, "audio": str(audio) if audio else None, "crop": scene.get("crop"), "title": scene.get("title"), "kicker": scene.get("kicker"), "caption": scene.get("caption"), "chips": scene.get("chips", []), "evidence": scene.get("evidence")})
            cursor += scene["duration"]
        # The concat demuxer reads paths relative to its manifest.  Generated
        # names contain no shell/concat metacharacters, so relative paths keep
        # apostrophes (and backslashes) in the staging directory harmless.
        video_list = staging / "video.txt"; video_list.write_text("\n".join(f"file 'chapters/{p.name}'" for p in chapter_paths) + "\n", encoding="utf-8")
        audio_list = staging / "audio.txt"; audio_list.write_text("\n".join(f"file '{p.name}'" for p in audio_paths) + "\n", encoding="utf-8")
        silent = staging / "silent.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-f", "concat", "-safe", "0", "-i", str(video_list), "-c", "copy", "-movflags", "+faststart", str(silent)])
        audio = staging / "audio.wav"
        _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c", "copy", str(audio)])
        final = staging / "story.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", str(silent), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(final)])
        expected_video_frames = sum(round(scene["duration"] * FPS) for scene in scenes)
        if _video_frames(final) != expected_video_frames or _video_frames(silent) != expected_video_frames:
            raise ValueError("encoded video frame count does not match quantized timeline")
        # ``silent`` already lives at the published fallback path.
        provenance = {"schema": "narrated-story-v1", "storyboard": str(storyboard.resolve()), "storyboard_sha256": _sha(storyboard), "video": "story.mp4", "silent_video": "silent.mp4", "audio": "audio.wav", "output_hashes": {"story_sha256": _sha(final), "silent_sha256": _sha(silent), "audio_sha256": _sha(audio)}, "duration_seconds": total, "fps": FPS, "width": WIDTH, "height": HEIGHT, "scenes": timeline, "storyboard_data": story, "source_hashes": [{"path": str(s["image_path"]), "sha256": _sha(s["image_path"])} for s in scenes if s["image_path"]], "audio_hashes": [{"path": str(s["audio_path"]), "sha256": _sha(s["audio_path"])} for s in scenes if s["audio_path"]], "visual": {"map_area": [40, 72, 1280, 972], "panel_area": [1304, 0, 1920, 1080], "palette": {"navy": NAVY, "cream": CREAM, "lime": LIME}}}
        (staging / "timeline.json").write_text(json.dumps({"story": story, "scenes": timeline}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.replace(output_dir)
        return {"output_dir": str(output_dir), "video": str(output_dir / "story.mp4"), "silent_video": str(output_dir / "silent.mp4"), "duration_seconds": total, "scene_count": len(scenes)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _, scenes, total = validate_story(args.storyboard)
        print(json.dumps({"status": "valid", "scene_count": len(scenes), "duration_seconds": total}))
    else:
        print(json.dumps(render(args.storyboard, args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
