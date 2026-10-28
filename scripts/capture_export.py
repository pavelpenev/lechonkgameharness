#!/usr/bin/env python3
"""Deterministic, screenshot-first export for a Freeciv capture.

This is intentionally an offline editor/render helper.  It never starts a
game and never interprets model stdout.  ``images.jsonl`` is the visual source
of truth; ``capture-events.jsonl`` is used only for short, explicitly-labelled
sidebar excerpts.  Missing or uncertain evidence is retained and labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FPS = 30
HIGHLIGHT_MAX_FRAMES = 120 * FPS
DEFAULT_HOLD_FRAMES = 90
DEFAULT_HIGHLIGHT_SCENE_LIMIT = 8
EDIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return records, [f"missing:{path.name}"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            records.append(value)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}:{line_no}:{exc}")
    return records, errors


def _first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _one_line(value: Any, limit: int = 240) -> str:
    text = " ".join(_text(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _event_id(record: dict[str, Any]) -> str:
    return str(_first(record, "event_id", "id", "eventId", default=""))


def _event_kind(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    outer = str(_first(record, "kind", "event_kind", "type", default="") or "")
    semantic = _first(payload, "event", "kind", "type", default=None) if isinstance(payload, dict) else None
    if isinstance(semantic, dict):
        semantic = _first(semantic, "event", "kind", "type")
    if semantic and (outer.lower() in {"progress", "model_output", "model_lifecycle", "trace", "event"}):
        return str(semantic)
    return outer or str(semantic or "unknown")


def _event_excerpt(record: dict[str, Any]) -> str:
    value = _first(record, "summary", "message", "detail", "text", "reason", "action")
    payload = record.get("payload")
    raw_delta = payload.get("raw_delta") if isinstance(payload, dict) else None
    if isinstance(raw_delta, dict):
        value = _first(raw_delta, "reasoning", "answer", "text", "message", "tool", "raw_text")
    if value is None and isinstance(payload, dict):
        value = _first(payload, "summary", "message", "detail", "text", "reason", "action",
                       "reasoning", "answer", "status", "result", "raw_delta")
    if isinstance(payload, dict):
        action_ref = _first(payload, "action_ref", "ref", "action_id")
        outcome = _first(payload, "outcome", "status", "result")
        if isinstance(payload.get("event"), dict):
            action_ref = action_ref or _first(payload["event"], "action_ref", "ref", "action_id")
            outcome = outcome or _first(payload["event"], "outcome", "status", "result")
        if action_ref is not None or outcome is not None:
            value = f"action={action_ref or 'unknown'} outcome={outcome or 'unknown'}; {value or ''}"
    if value is None:
        kind = _event_kind(record)
        value = kind
    return _one_line(value)


def _scene_salience(kinds: Iterable[str], excerpts: Iterable[str], signals: Iterable[str] = ()) -> tuple[int, str]:
    """Return a stable editorial score and reason from joined event metadata."""
    kind_values = [str(kind).lower() for kind in kinds]
    signal_values = [str(item).lower() for item in signals]
    text = " ".join(kind_values + [str(item) for item in excerpts] + signal_values)
    # ``outcome=executed`` is the normal result of every successful action;
    # it must not turn routine movement into a game-ending scene.  Terminal
    # scoring below is based on the semantic event kind only.
    automatic = any("provenance_class:engine_automatic" in item for item in signal_values)
    score = 1
    reason = "routine"
    for needles, value, label in (
        (("combat", "attack", "battle", "destroy", "capture"), 80, "combat"),
        (("diplom", "treaty", "war", "peace"), 65, "diplomacy"),
        (("city", "found", "settle", "build", "wonder"), 55, "city/build"),
        (("research", "technology", "discover", "science"), 45, "discovery"),
        (("action_result", "executed", "verified"), 20, "verified action"),
    ):
        if any(needle in text for needle in needles):
            if value > score:
                score, reason = value, label
    # Only semantic event kinds can declare terminal state. Free-form model
    # reasoning and ordinary action results may mention victory/defeat/outcome.
    if any(kind in {"game_over", "victory", "defeat", "outcome", "game_outcome", "conquest"}
           for kind in kind_values):
        score, reason = max(score, 100), "outcome"
    if automatic and reason != "outcome":
        # Automatic production/research is useful audit context, but should
        # not displace model decisions in an editorial highlight.
        score, reason = min(score, 5), "engine automatic"
    return score, reason


def _event_salience_signals(event: dict[str, Any]) -> list[str]:
    """Extract bounded structured action fields without exposing raw payloads."""
    signals: list[str] = []
    candidates: list[dict[str, Any]] = [event]
    payload = event.get("payload")
    if isinstance(payload, dict):
        candidates.append(payload)
        semantic = payload.get("event")
        if isinstance(semantic, dict):
            candidates.append(semantic)
    for candidate in candidates:
        for key in ("domain", "verb", "action", "target", "affected", "actor_tile", "action_ref"):
            value = candidate.get(key)
            if value is not None:
                signals.append(_one_line(value, 120))
        provenance = candidate.get("provenance_class")
        if provenance is not None:
            signals.append(f"provenance_class:{_one_line(provenance, 80)}")
    return signals


def _sequence(record: dict[str, Any]) -> tuple[int, int, str]:
    source = record.get("source_event") if isinstance(record.get("source_event"), dict) else {}
    raw = _first(source, "capture_seq", "source_seq", "seq")
    if raw is None:
        raw = _first(record, "capture_seq", "completion_capture_seq", "sequence", "seq")
    completion = _first(record, "completion_capture_seq", "capture_end_ns", "completed_at", default="")
    try:
        return (0, int(raw), str(completion))
    except (TypeError, ValueError):
        stamp = str(_first(record, "capture_end_ns", "timestamp", "time", "created_at", default=""))
        return (1, 0, stamp)


def _safe_edit_id(value: str) -> str:
    if not EDIT_ID_RE.fullmatch(value):
        raise ValueError("edit-id must contain only letters, digits, '.', '_' and '-' (max 96 chars)")
    return value


def _source_image(capture_dir: Path, record: dict[str, Any]) -> tuple[Path | None, str]:
    declared_status = str(_first(record, "status", default=""))
    explicit_failure = (declared_status.startswith("failed") or declared_status in {
        "error", "dropped", "cancelled", "canceled", "incomplete", "invalid"})
    if explicit_failure:
        # Failed/coalesced requests are retained as audit records, never
        # mistaken for a visual still with an absent filename.
        return None, declared_status
    raw = _first(record, "path", "image_path", "filename", "file")
    if not raw:
        return None, "missing_path"
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = capture_dir / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(capture_dir.resolve())
    except (OSError, ValueError):
        return None, "image_outside_capture_dir"
    if not resolved.is_file():
        return resolved, "missing_image"
    try:
        from PIL import Image
        with Image.open(resolved) as image:
            image.verify()
    except Exception:
        return resolved, "corrupt_image"
    # The capture status may describe semantic game evidence (for example
    # ``unknown``), not PNG write success. A safe, decodable file is still a
    # renderable visual; retain the original status in source_record/labels.
    return resolved, "complete"


def _event_index(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {eid: event for event in events if (eid := _event_id(event))}


def _labels(image: dict[str, Any]) -> list[str]:
    labels = ["SPECTATOR OBSERVER"]
    freshness = str(_first(image, "connection_freshness", "observer_freshness", default="unknown"))
    evidence = str(_first(image, "visual_evidence", default="unavailable"))
    temporal = str(_first(image, "temporal_relation", default="unknown"))
    if freshness != "fresh":
        labels.append(f"observer freshness: {freshness}")
    if evidence != "change_observed":
        labels.append(f"visual evidence: {evidence}")
    if temporal == "unknown":
        labels.append("temporal relation: unknown")
    elif temporal not in {"pre_ack_confirmed", "post_result_frame"}:
        labels.append(f"temporal relation: {temporal}")
    if str(_first(image, "fog", default="spectator_observer")) != "model_view":
        labels.append("spectator view may reveal more than the model")
    surface = str(_first(image, "screenshot_surface", "viewport", default="unknown"))
    if surface in {"composite_viewport_fallback", "full_page_fallback"}:
        labels.append(f"visual surface: {surface}; source already contains presentation sidebar")
    elif surface != "map_only_clipped":
        labels.append(f"visual surface: {surface}; map-only not proven")
    if _first(image, "needs_review", default=False) or _first(image, "status") not in (None, "complete"):
        labels.append("NEEDS REVIEW")
    if str(_first(image, "status", default="complete")) not in {"complete", "captured", "success"}:
        labels.append(f"capture gap: {_first(image, 'status')}")
    return labels


def build_scenes(capture_dir: Path, images: list[dict[str, Any]], events: list[dict[str, Any]],
                 *, hold_frames: int = DEFAULT_HOLD_FRAMES) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize evolving capture records into an editable, honest EDL."""
    event_by_id = _event_index(events)
    events_by_seq: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        raw_seq = _first(event, "capture_seq", "source_seq", "seq")
        try:
            events_by_seq.setdefault(int(raw_seq), []).append(event)
        except (TypeError, ValueError):
            pass
    warnings: list[str] = []
    scenes: list[dict[str, Any]] = []
    for index, image in enumerate(sorted(images, key=_sequence)):
        image_id = str(_first(image, "image_id", "id", default=f"image-{index:06d}"))
        path, status = _source_image(capture_dir, image)
        if status != "complete":
            warnings.append(f"{image_id}: skipped visual frame ({status})")
        trigger_ids = _first(image, "trigger_event_ids", "event_ids", "source_event_ids", default=[])
        if isinstance(trigger_ids, str):
            trigger_ids = [trigger_ids]
        trigger_ids = [str(item) for item in trigger_ids if item is not None]
        excerpts = [_event_excerpt(event_by_id[eid]) for eid in trigger_ids if eid in event_by_id]
        joined_events: list[dict[str, Any]] = []
        source_event = image.get("source_event")
        if isinstance(source_event, dict):
            source_event_id = _first(source_event, "event_id", "id")
            if source_event_id is not None and not trigger_ids:
                trigger_ids = [str(source_event_id)]
            source_seq = _first(source_event, "capture_seq", "source_seq", "seq")
            try:
                joined = events_by_seq.get(int(source_seq), [])
            except (TypeError, ValueError):
                joined = []
            joined_events.extend(joined)
            excerpts.extend(_event_excerpt(event) for event in joined)
            if not trigger_ids and source_seq is not None:
                trigger_ids = [f"capture-seq:{source_seq}"]
        if trigger_ids and not excerpts:
            excerpts.extend(_event_excerpt(event_by_id[eid]) for eid in trigger_ids if eid in event_by_id)
        lane_events = list(joined_events)
        lane_events.extend(event_by_id[eid] for eid in trigger_ids if eid in event_by_id and event_by_id[eid] not in lane_events)
        if not excerpts:
            # An absent event join is intentionally visible; do not guess a nearby action.
            excerpts = ["No joined structured event excerpt"]
            warnings.append(f"{image_id}: no joined event excerpt")
        event_kinds = [_event_kind(event) for event in lane_events]
        signals = [signal for event in lane_events for signal in _event_salience_signals(event)]
        salience, salience_reason = _scene_salience(event_kinds, excerpts, signals)
        canonical = json.dumps(image, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        scene_id = "scene-" + hashlib.sha256(canonical).hexdigest()[:16]
        scene_turn = _first(image, "turn")
        if scene_turn is None:
            for joined_event in lane_events:
                joined_payload = joined_event.get("payload")
                if isinstance(joined_payload, dict):
                    scene_turn = _first(joined_payload, "turn")
                    if scene_turn is not None:
                        break
                scene_turn = _first(joined_event, "turn")
                if scene_turn is not None:
                    break
        scenes.append({
            "scene_index": index,
            "scene_id": scene_id,
            "image_id": image_id,
            "image_path": str(path.relative_to(capture_dir)) if path and path.is_relative_to(capture_dir) else None,
            "image_status": status,
            "hold_frames": max(1, int(hold_frames)),
            "salience": salience,
            "salience_reason": salience_reason,
            "event_kinds": event_kinds,
            "source_event_ids": trigger_ids,
            "event_excerpts": excerpts,
            "event_lanes": [{"event_id": _event_id(event) or None,
                             "kind": _event_kind(event),
                             "capture_seq": _first(event, "capture_seq", "source_seq", "seq"),
                             "lane": ((event.get("payload") or {}).get("raw_delta") or {}).get("lane") if isinstance(event.get("payload"), dict) else None,
                             "trace_ref": _first(event, "trace_ref", "trace_id", "call_id", default=None),
                             "excerpt": _event_excerpt(event),
                             "raw_event": event} for event in lane_events],
            "labels": _labels(image),
            "turn": scene_turn,
            "year": _first(image, "year"),
            "phase": _first(image, "phase"),
            "temporal_relation": _first(image, "temporal_relation", default="unknown"),
            "connection_freshness": _first(image, "connection_freshness", "observer_freshness", default="unknown"),
            "visual_evidence": _first(image, "visual_evidence", default="unavailable"),
            "fog": _first(image, "fog", default="spectator_observer"),
            "ordering_basis": "capture sequence/timestamp tie-break; not physical causality",
            "needs_review": bool(_first(image, "needs_review", default=False) or status != "complete"),
            "renderable": status == "complete",
            "source_record": image,
        })
    if not scenes:
        warnings.append("no image records")
    return scenes, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_publish_root(capture_dir: Path) -> Path | None:
    """Return the shallow user-facing output root for a normal run layout.

    Temporary capture directories used by tests and isolated tooling do not
    match this layout and therefore are never written outside their fixture.
    """
    resolved = capture_dir.resolve()
    if resolved.name != "capture" or resolved.parent.parent.name != "runs":
        return None
    artifacts = resolved.parent.parent.parent
    if artifacts.name != "artifacts":
        return None
    return artifacts / "highlights"


def _publish_outputs(capture_dir: Path, out_dir: Path, edit_id: str,
                     *, publish_dir: Path | None = None) -> Path | None:
    """Copy completed deliverables to a shallow, collision-safe directory."""
    root = publish_dir.resolve() if publish_dir is not None else _default_publish_root(capture_dir)
    if root is None:
        return None
    run_id = capture_dir.parent.name
    def safe(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        return cleaned[:96] or "capture"
    base = root / f"{safe(run_id)}--{safe(edit_id)}"
    target = base
    suffix = 2
    root.mkdir(parents=True, exist_ok=True)
    # Reserve the name with an exclusive lock directory.  A plain exists()
    # check races when two exporters finish at the same time; the successful
    # mkdir is the ownership decision.  The final rename remains atomic.
    lock: Path | None = None
    first_candidate = True
    while lock is None:
        candidate = base if first_candidate else root / f"{base.name}--{suffix}"
        first_candidate = False
        candidate_lock = root / f".{candidate.name}.lock"
        try:
            candidate_lock.mkdir()
        except FileExistsError:
            if candidate == base:
                suffix = 2
            else:
                suffix += 1
            continue
        if candidate.exists():
            candidate_lock.rmdir()
            suffix += 1
            continue
        target, lock = candidate, candidate_lock
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=root))
    try:
        copied: list[str] = []
        for name in ("highlight.mp4", "story.mp4", "preview.png", "scene_list.json", "edl.json"):
            source = out_dir / name
            if source.is_file():
                shutil.copy2(source, temp / name)
                copied.append(name)
        provenance = {
            "schema": "capture-published-v1",
            "source_capture_dir": str(capture_dir),
            "source_edit_dir": str(out_dir),
            "run_id": run_id,
            "edit_id": edit_id,
            "files": copied,
        }
        _write_json(temp / "provenance.json", provenance)
        temp.rename(target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        if lock is not None:
            lock.rmdir()
    return target


def _compose(scene: dict[str, Any], capture_dir: Path, target: Path, *, width: int = 1920, height: int = 1080) -> bool:
    """Compose a still with a deterministic sidebar; preserve composite sources."""
    if scene["image_status"] != "complete" or not scene["image_path"]:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        source = Image.open(capture_dir / scene["image_path"]).convert("RGB")
        surface = str(_first(scene.get("source_record", {}), "screenshot_surface", "viewport", default="unknown"))
        if surface in {"composite_viewport_fallback", "full_page_fallback"}:
            # Presenter already injected the sidebar. Adding another one would
            # misrepresent the capture and obscure the map.
            canvas = Image.new("RGB", (width, height), (18, 22, 30))
            fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
            canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target, format="PNG", optimize=False)
            return True
        sidebar_w = 430
        canvas = Image.new("RGB", (width, height), (18, 22, 30))
        map_box = (0, 0, width - sidebar_w, height)
        source.thumbnail((map_box[2], map_box[3]), Image.Resampling.LANCZOS)
        canvas.paste(source, ((map_box[2] - source.width) // 2, (height - source.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((width - sidebar_w, 0, width, height), fill=(18, 22, 30))
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        regular = ImageFont.truetype(font_path, 24) if os.path.isfile(font_path) else ImageFont.load_default()
        bold = ImageFont.truetype(bold_path, 27) if os.path.isfile(bold_path) else regular
        y = 28
        draw.text((width - sidebar_w + 24, y), "SPECTATOR OBSERVER", font=bold, fill=(255, 210, 80)); y += 52
        meta = [f"scene {scene['scene_index']}", f"turn {scene['turn'] or 'unknown'}", f"phase {scene['phase'] or 'unknown'}"]
        for line in meta:
            draw.text((width - sidebar_w + 24, y), line, font=regular, fill=(235, 240, 248)); y += 34
        y += 14
        draw.line((width - sidebar_w + 24, y, width - 24, y), fill=(90, 100, 115), width=2); y += 20
        for label in scene["labels"]:
            draw.text((width - sidebar_w + 24, y), _one_line(label, 34), font=regular, fill=(255, 160, 120)); y += 31
        y += 16
        draw.text((width - sidebar_w + 24, y), "EVENT EXCERPT", font=bold, fill=(150, 210, 255)); y += 38
        for excerpt in scene["event_excerpts"][:3]:
            for line in _wrap(excerpt, 28):
                if y > height - 42:
                    break
                draw.text((width - sidebar_w + 24, y), line, font=regular, fill=(235, 240, 248)); y += 30
            y += 8
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="PNG", optimize=False)
        return True
    except Exception as exc:
        raise RuntimeError(f"could not compose {scene['scene_id']}: {exc}") from exc


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        if len(line) + len(word) + bool(line) <= width:
            line = (line + " " + word).strip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _validate_video(path: Path, expected_frames: int, fps: int = FPS, max_seconds: float | None = None) -> dict[str, Any]:
    probe = ["ffprobe", "-v", "error", "-count_frames",
             "-show_entries", "format=duration:stream=index,codec_type,width,height,r_frame_rate,nb_read_frames,duration",
             "-of", "json", str(path)]
    result = subprocess.run(probe, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    stream = video
    frames = int(stream.get("nb_read_frames") or 0)
    stream_durations = [float(item["duration"]) for item in streams if item.get("duration") is not None]
    container_duration = float((data.get("format") or {}).get("duration") or 0.0)
    duration = max([container_duration, *stream_durations, 0.0])
    if frames != expected_frames:
        raise RuntimeError(f"encoded frame count {frames} != expected {expected_frames}")
    if stream.get("width") != 1920 or stream.get("height") != 1080 or stream.get("r_frame_rate") != f"{fps}/1":
        raise RuntimeError(f"unexpected video profile: {stream}")
    if max_seconds is not None and duration > max_seconds + 1e-6:
        raise RuntimeError(f"video duration {duration:.6f}s exceeds {max_seconds:.3f}s")
    _run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    return {"frames": frames, "duration_seconds": duration, "width": 1920, "height": 1080, "fps": fps}


def _visual_fingerprint(capture_dir: Path, scene: dict[str, Any]) -> str | None:
    """Small perceptual-ish fingerprint used only to collapse identical stills."""
    if scene.get("image_status") != "complete" or not scene.get("image_path"):
        return None
    try:
        from PIL import Image
        with Image.open(capture_dir / scene["image_path"]) as image:
            image = image.convert("L").resize((16, 9), Image.Resampling.BILINEAR)
            # Quantized pixels are stable across harmless PNG metadata changes.
            return hashlib.sha1(bytes(pixel // 16 for pixel in image.tobytes())).hexdigest()
    except Exception:
        return None


def _select_highlight_scenes(scenes: list[dict[str, Any]], capture_dir: Path,
                             *, limit: int = DEFAULT_HIGHLIGHT_SCENE_LIMIT) -> list[dict[str, Any]]:
    usable = [scene for scene in scenes if scene.get("image_status") == "complete"]
    if not usable:
        return []
    # Collapse only adjacent identical visuals, retaining the more meaningful event.
    deduped: list[dict[str, Any]] = []
    fingerprints: dict[str, int] = {}
    for position, scene in enumerate(usable):
        fingerprint = _visual_fingerprint(capture_dir, scene)
        # Always retain the final frame as the editorial bookend, even when it
        # repeats the preceding still.
        if position != len(usable) - 1 and fingerprint and deduped and fingerprint == fingerprints.get("last"):
            if scene.get("salience", 1) > deduped[-1].get("salience", 1):
                deduped[-1] = scene
            continue
        deduped.append(scene)
        fingerprints["last"] = fingerprint
    # Keep meaningful chronology bookends. An unattached presenter frame is a
    # loading/observer view, not an establishing game scene, when later joined
    # visuals exist. Apply this before the scene-count early return so short
    # captures do not retain a useless blank opening.
    meaningful = [index for index, scene in enumerate(deduped)
                  if (scene.get("event_kinds")
                      and not set(scene.get("event_kinds", [])).issubset({"presenter_attached"}))
                  or (scene.get("source_event_ids")
                      and any(kind not in {"presenter_attached", "presenter-ready"}
                              for kind in scene.get("event_kinds", [])))]
    if len(deduped) <= limit:
        return ([deduped[index] for index in meaningful] if meaningful else deduped)
    # Prefer the first consequential visual as the opening beat; an early
    # turn-decision/map-loading still is often mostly unexplored black space.
    consequential = [index for index in meaningful if int(deduped[index].get("salience", 1)) >= 20]
    first_index = (consequential[0] if consequential else meaningful[0]) if meaningful else 0
    last_index = meaningful[-1] if meaningful else len(deduped) - 1
    keep = {first_index, last_index}

    # Spend slots across turns before ranking globally. This prevents a burst
    # of turn-1 action frames from crowding every later strategic decision out
    # of a multi-turn highlight.
    turn_candidates: dict[str, list[int]] = {}
    for index, scene in enumerate(deduped):
        turn = scene.get("turn")
        if turn is not None:
            turn_candidates.setdefault(str(turn), []).append(index)
    for indices in turn_candidates.values():
        best = max(indices, key=lambda index: (int(deduped[index].get("salience", 1)), -index))
        if len(keep) >= limit:
            break
        keep.add(best)

    ranked = sorted((index for index in range(len(deduped)) if index not in keep),
                    key=lambda index: (-int(deduped[index].get("salience", 1)), index))
    for index in ranked:
        if len(keep) >= limit:
            break
        keep.add(index)
    return [scene for index, scene in enumerate(deduped) if index in keep]


def _editorial_hold(scene: dict[str, Any]) -> int:
    base = max(1, int(scene["hold_frames"]))
    if base >= HIGHLIGHT_MAX_FRAMES:
        return base
    score = int(scene.get("salience", 1))
    if score >= 100:
        return max(base, 105)
    if score >= 20:
        return max(base, 75)
    return min(base, 60)


def render_video(scenes: list[dict[str, Any]], capture_dir: Path, out_dir: Path, name: str,
                 *, max_frames: int | None = None) -> dict[str, Any]:
    usable = _select_highlight_scenes(scenes, capture_dir)
    if not usable:
        raise RuntimeError("no decodable images available for rendering")
    selected: list[dict[str, Any]] = []
    remaining = max_frames
    for scene in usable:
        hold = _editorial_hold(scene)
        if remaining is not None:
            if remaining <= 0:
                break
            hold = min(hold, remaining)
        item = dict(scene)
        item["hold_frames"] = hold
        selected.append(item)
        if remaining is not None:
            remaining -= hold
    if not selected:
        raise RuntimeError("duration cap left no renderable scenes")
    total = sum(int(scene["hold_frames"]) for scene in selected)
    with tempfile.TemporaryDirectory(prefix="capture-export-", dir=out_dir) as temp:
        temp_dir = Path(temp)
        concat = temp_dir / "frames.txt"
        lines: list[str] = []
        for index, scene in enumerate(selected):
            still = temp_dir / f"scene-{index:06d}.png"
            _compose(scene, capture_dir, still)
            lines.append(f"file '{still.as_posix().replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'")
            lines.append(f"duration {scene['hold_frames'] / FPS:.9f}")
        # concat demuxer needs the last file twice to honour the final duration.
        lines.append(lines[-2])
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output = out_dir / name
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
              "-frames:v", str(total), "-vf", f"fps={FPS}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", str(output)])
    validated = _validate_video(output, total, max_seconds=(max_frames / FPS if max_frames else None))
    # Return the exact records used by the encoder, including editorial hold
    # durations. The EDL must describe the rendered timeline rather than the
    # source scene-list defaults.
    validated["selected_scene_ids"] = [scene["scene_id"] for scene in selected]
    validated["selected_scenes"] = selected
    validated["selected_scene_count"] = len(selected)
    validated["source_scene_count"] = len(scenes)
    return validated


def export(capture_dir: Path, *, edit_id: str, mode: str = "both", hold_frames: int = DEFAULT_HOLD_FRAMES,
           publish_dir: Path | None = None) -> dict[str, Any]:
    edit_id = _safe_edit_id(edit_id)
    capture_dir = capture_dir.resolve()
    out_dir = capture_dir / "derived" / edit_id
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing edit directory: {out_dir}")
    out_dir.mkdir(parents=True)
    images, image_errors = _jsonl(capture_dir / "images.jsonl")
    events, event_errors = _jsonl(capture_dir / "capture-events.jsonl")
    manifest_path = next((capture_dir / name for name in ("manifest.json", "capture-manifest.json")
                          if (capture_dir / name).is_file()), None)
    manifest: dict[str, Any] = {}
    manifest_warnings: list[str] = []
    if manifest_path is not None:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                manifest_warnings = [f"{manifest_path.name}: manifest is not an object"]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifest_warnings = [f"{manifest_path.name}: invalid manifest: {exc}"]
    else:
        manifest_warnings = ["manifest unavailable"]
    scenes, scene_warnings = build_scenes(capture_dir, images, events, hold_frames=hold_frames)
    successful_count = sum(scene["renderable"] for scene in scenes)
    skipped = [{"scene_index": scene["scene_index"], "image_id": scene["image_id"],
                "status": scene["image_status"]} for scene in scenes if not scene["renderable"]]
    manifest_complete = manifest.get("complete") is True and manifest.get("partial") is not True
    if manifest_complete:
        completeness = "complete"
    elif manifest.get("partial") is True or manifest.get("status") in {"partial", "interrupted", "error"}:
        completeness = "partial"
    else:
        completeness = "unknown"
    if completeness != "complete":
        for scene in scenes:
            scene["labels"].append(f"capture completeness: {completeness}; NEEDS REVIEW")
            scene["needs_review"] = True
    warnings = image_errors + event_errors + manifest_warnings + scene_warnings
    presenter_events, presenter_errors = _jsonl(capture_dir / "presenter-events.jsonl")
    presenter_finished = next((event for event in reversed(presenter_events)
                               if event.get("kind") == "presenter_finished"), None)
    presenter_integrity = {
        "finished": presenter_finished is not None,
        "terminal_seen": bool((presenter_finished or {}).get("payload", {}).get("terminal_seen")),
        "errors": (presenter_finished or {}).get("payload", {}).get("errors"),
        "warnings": presenter_errors,
    }
    scene_list = {
        "schema": "capture-scene-list-v1",
        "edit_id": edit_id,
        "source_capture_dir": str(capture_dir),
        "source_files": {name: {"path": str(capture_dir / name), "sha256": _sha256(capture_dir / name)}
                         for name in ("images.jsonl", "capture-events.jsonl") if (capture_dir / name).is_file()},
        "manifest": manifest,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "fps": FPS,
        "frame_timebase": "integer 1/30-second frames",
        "spectator_disclosure": "SPECTATOR OBSERVER may reveal more than the model; labels are retained.",
        "capture_completeness": completeness,
        "presenter_integrity": presenter_integrity,
        "no_invented_motion": True,
        "warnings": warnings,
        "diagnostics": {"image_records": len(scenes), "successful_visuals": successful_count,
                        "skipped_visuals": len(skipped), "skipped": skipped},
        "scenes": scenes,
    }
    _write_json(out_dir / "scene_list.json", scene_list)
    _write_json(out_dir / "edl.json", scene_list)
    result: dict[str, Any] = {"edit_id": edit_id, "output_dir": str(out_dir), "scene_count": len(scenes),
                              "successful_visuals": successful_count, "skipped_visuals": len(skipped),
                              "skipped": skipped, "capture_completeness": completeness, "warnings": warnings}
    if mode in {"preview", "both"}:
        preview = out_dir / "preview.png"
        result["preview"] = str(preview)
        # Preview the opening frame of the actual editorial selection, not a
        # presenter-attached/loading frame that may be retained in the audit
        # scene list but excluded from the highlight.
        preview_scenes = _select_highlight_scenes(scenes, capture_dir)
        for scene in preview_scenes or scenes:
            if _compose(scene, capture_dir, preview):
                break
    if mode in {"highlight", "both"}:
        result["highlight"] = {"path": str(out_dir / "highlight.mp4"),
                                **render_video(scenes, capture_dir, out_dir, "highlight.mp4",
                                              max_frames=HIGHLIGHT_MAX_FRAMES)}
        edl = dict(scene_list)
        selected_ids = set(result["highlight"].get("selected_scene_ids", []))
        edl["schema"] = "capture-edit-list-v1"
        edl["source_scene_count"] = len(scenes)
        edl["selected_scene_count"] = len(selected_ids)
        rendered_by_id = {scene["scene_id"]: scene for scene in result["highlight"].get("selected_scenes", [])}
        edl["scenes"] = [rendered_by_id[scene_id] for scene_id in result["highlight"].get("selected_scene_ids", [])
                          if scene_id in rendered_by_id]
        _write_json(out_dir / "edl.json", edl)
    if mode == "story":
        result["story"] = {"path": str(out_dir / "story.mp4"),
                            **render_video(scenes, capture_dir, out_dir, "story.mp4")}
    published = _publish_outputs(capture_dir, out_dir, edit_id, publish_dir=publish_dir)
    if published is not None:
        result["published_dir"] = str(published)
        result["published_files"] = sorted(path.name for path in published.iterdir() if path.is_file())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--edit-id", default=None)
    parser.add_argument("--mode", choices=("scene-list", "preview", "highlight", "story", "both"), default="both")
    parser.add_argument("--hold-frames", type=int, default=DEFAULT_HOLD_FRAMES)
    parser.add_argument("--publish-dir", type=Path, default=None,
                        help="shallow output root (defaults to artifacts/highlights for normal runs)")
    args = parser.parse_args(argv)
    if args.hold_frames < 1:
        parser.error("--hold-frames must be positive")
    edit_id = args.edit_id or "edit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        result = export(args.capture_dir, edit_id=edit_id, mode=args.mode, hold_frames=args.hold_frames,
                        publish_dir=args.publish_dir)
    except (FileExistsError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"capture export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
