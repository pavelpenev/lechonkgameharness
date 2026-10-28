"""Focused offline tests for the screenshot-first capture exporter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.capture_export import HIGHLIGHT_MAX_FRAMES, _scene_salience, _select_highlight_scenes, build_scenes, export


def _capture(tmp_path: Path, *, count: int = 3) -> Path:
    from PIL import Image

    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    images = []
    events = []
    for index in range(count):
        name = f"images/map-{index}.png"
        Image.new("RGB", (320, 180), (30 + index * 20, 50, 90)).save(root / name)
        event_id = f"event-{index}"
        images.append({
            "image_id": f"image-{index}", "path": name,
            "capture_seq": count - index, "trigger_event_ids": [event_id],
            "turn": index + 1, "temporal_relation": "unknown",
            "connection_freshness": "stale" if index == 1 else "fresh",
            "visual_evidence": "consequence_unconfirmed", "fog": "spectator_observer",
        })
        events.append({"event_id": event_id, "kind": "turn_decision",
                       "message": f"quote 'unicode ✓' <tag> {index}"})
    (root / "images.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in images) + "\n", encoding="utf-8")
    (root / "capture-events.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in events) + "\n", encoding="utf-8")
    return root


def test_scene_order_and_uncertainty_labels(tmp_path):
    root = _capture(tmp_path)
    images = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (root / "capture-events.jsonl").read_text().splitlines()]
    scenes, warnings = build_scenes(root, images, events, hold_frames=7)
    assert [scene["image_id"] for scene in scenes] == ["image-2", "image-1", "image-0"]
    assert all(scene["hold_frames"] == 7 for scene in scenes)
    assert all("SPECTATOR OBSERVER" in scene["labels"] for scene in scenes)
    assert any("temporal relation: unknown" in scene["labels"] for scene in scenes)
    assert not warnings


def test_presenter_source_event_and_nested_payload_are_joined(tmp_path):
    root = _capture(tmp_path, count=1)
    images = [{"image_id": "presenter-image", "file": "images/map-0.png",
               "source_event": {"capture_seq": 42, "kind": "action_result"},
               "screenshot_surface": "full_page_fallback"}]
    events = [{"capture_seq": 42, "kind": "action_result",
               "payload": {"message": "nested result ✓"}}]
    scenes, warnings = build_scenes(root, images, events)
    assert scenes[0]["source_event_ids"] == ["capture-seq:42"]
    assert scenes[0]["event_excerpts"] == ["nested result ✓"]
    assert "visual surface: full_page_fallback; source already contains presentation sidebar" in scenes[0]["labels"]
    assert not warnings


def test_gateway_progress_wrapper_uses_semantic_event_and_raw_delta(tmp_path):
    root = _capture(tmp_path, count=1)
    images = [{"image_id": "gateway-image", "file": "images/map-0.png",
               "source_event": {"capture_seq": 19, "event_id": "gw-19", "kind": "progress"},
               "screenshot_surface": "composite_viewport_fallback"}]
    events = [{"schema_version": "capture-gateway/v1", "capture_seq": 19,
               "kind": "progress", "event_id": "gw-19",
               "payload": {"event": "action_result", "action_ref": "unit:3:move",
                            "outcome": "executed",
                            "raw_delta": {"lane": "reasoning", "reasoning": "choose the eastern tile ✓"}}}]
    scenes, warnings = build_scenes(root, images, events)
    assert scenes[0]["event_lanes"][0]["kind"] == "action_result"
    assert scenes[0]["event_lanes"][0]["lane"] == "reasoning"
    assert "action=unit:3:move outcome=executed" in scenes[0]["event_excerpts"][0]
    assert "choose the eastern tile" in scenes[0]["event_excerpts"][0]
    assert not warnings


def test_missing_image_is_retained_but_not_rendered(tmp_path):
    root = _capture(tmp_path, count=1)
    record = json.loads((root / "images.jsonl").read_text())
    record["path"] = "images/not-there.png"
    (root / "images.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = export(root, edit_id="missing-image", mode="preview")
    scene = json.loads((root / "derived/missing-image/scene_list.json").read_text())["scenes"][0]
    assert scene["image_status"] == "missing_image"
    assert scene["needs_review"] is True
    assert not (root / "derived/missing-image/preview.png").exists()
    assert result["scene_count"] == 1


def test_manifest_partial_propagates_to_scene_and_result(tmp_path):
    root = _capture(tmp_path, count=1)
    (root / "manifest.json").write_text(json.dumps({"status": "finished", "complete": False, "partial": True}), encoding="utf-8")
    result = export(root, edit_id="partial-capture", mode="scene-list")
    data = json.loads((root / "derived/partial-capture/scene_list.json").read_text())
    assert data["capture_completeness"] == "partial"
    assert result["capture_completeness"] == "partial"
    assert data["scenes"][0]["needs_review"] is True
    assert any("capture completeness: partial" in label for label in data["scenes"][0]["labels"])


def test_failed_capture_records_are_audit_gaps_not_visuals(tmp_path):
    root = _capture(tmp_path, count=1)
    records = [
        {"image_id": "good", "file": "images/map-0.png", "capture_seq": 1, "status": "complete"},
        {"image_id": "coalesced", "capture_seq": 2, "status": "failed_coalesced"},
        {"image_id": "timeout", "file": "images/missing.png", "capture_seq": 3, "status": "failed_screenshot_timeout"},
        {"image_id": "missing", "capture_seq": 4},
    ]
    (root / "images.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    result = export(root, edit_id="capture-gaps", mode="preview")
    assert result["successful_visuals"] == 1
    assert result["skipped_visuals"] == 3
    statuses = {item["image_id"]: item["status"] for item in result["skipped"]}
    assert statuses == {"coalesced": "failed_coalesced", "timeout": "failed_screenshot_timeout", "missing": "missing_path"}
    data = json.loads((root / "derived/capture-gaps/scene_list.json").read_text())
    assert data["diagnostics"]["skipped_visuals"] == 3
    assert Path(result["preview"]).is_file()


def test_unknown_semantic_status_with_valid_png_remains_renderable(tmp_path):
    root = _capture(tmp_path, count=1)
    records = [{"image_id": f"unknown-{index}", "file": "images/map-0.png",
                "capture_seq": index, "status": "unknown"} for index in range(12)]
    records.extend([{"image_id": "failed", "status": "failed_coalesced", "capture_seq": 12},
                    {"image_id": "missing", "file": "images/nope.png", "status": "unknown", "capture_seq": 13}])
    (root / "images.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    result = export(root, edit_id="unknown-semantic-status", mode="preview")
    assert result["successful_visuals"] == 12
    assert result["skipped_visuals"] == 2
    assert {item["status"] for item in result["skipped"]} == {"failed_coalesced", "missing_image"}
    data = json.loads((root / "derived/unknown-semantic-status/scene_list.json").read_text())
    assert all(scene["renderable"] for scene in data["scenes"][:12])
    assert all(scene["needs_review"] for scene in data["scenes"][:12])


def test_unique_output_and_unicode_quote_safe_preview(tmp_path):
    root = _capture(tmp_path, count=1)
    result = export(root, edit_id="unicode-edit", mode="preview")
    assert Path(result["preview"]).is_file()
    scene_list = (root / "derived/unicode-edit/scene_list.json").read_text(encoding="utf-8")
    assert "unicode ✓" in scene_list
    with pytest.raises(FileExistsError):
        export(root, edit_id="unicode-edit", mode="scene-list")


def test_normal_run_publishes_shallow_user_facing_outputs(tmp_path):
    source = _capture(tmp_path / "fixture", count=1)
    root = tmp_path / "artifacts" / "runs" / "demo-run" / "capture"
    root.parent.mkdir(parents=True)
    shutil.copytree(source, root)
    result = export(root, edit_id="demo-edit", mode="preview")
    published = Path(result["published_dir"])
    assert published == tmp_path / "artifacts" / "highlights" / "demo-run--demo-edit"
    assert (published / "preview.png").is_file()
    assert (published / "scene_list.json").is_file()
    assert (published / "edl.json").is_file()
    provenance = json.loads((published / "provenance.json").read_text())
    assert provenance["source_capture_dir"] == str(root.resolve())
    assert (root / "derived/demo-edit/scene_list.json").is_file()


def test_explicit_publish_directory_avoids_overwrite(tmp_path):
    root = _capture(tmp_path, count=1)
    publish_root = tmp_path / "published"
    first = export(root, edit_id="same", mode="scene-list", publish_dir=publish_root)
    second = export(root, edit_id="same-2", mode="scene-list", publish_dir=publish_root)
    assert Path(first["published_dir"]).is_dir()
    assert Path(second["published_dir"]).is_dir()
    assert Path(first["published_dir"]) != Path(second["published_dir"])


def test_publication_reserves_names_when_lock_already_exists(tmp_path):
    root = _capture(tmp_path, count=1)
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    run_id = root.parent.name
    (publish_root / f".{run_id}--same.lock").mkdir()
    result = export(root, edit_id="same", mode="scene-list", publish_dir=publish_root)
    assert Path(result["published_dir"]).name == f"{run_id}--same--2"
    assert (publish_root / f".{run_id}--same.lock").exists()
    assert not (publish_root / f".{run_id}--same--2.lock").exists()


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg not installed")
def test_highlight_is_capped_and_decoded(tmp_path):
    root = _capture(tmp_path, count=3)
    result = export(root, edit_id="short-highlight", mode="highlight", hold_frames=HIGHLIGHT_MAX_FRAMES)
    assert result["highlight"]["frames"] == HIGHLIGHT_MAX_FRAMES
    assert result["highlight"]["duration_seconds"] <= 120.000001
    edl = json.loads((root / "derived/short-highlight/edl.json").read_text())
    assert sum(scene["hold_frames"] for scene in edl["scenes"]) == result["highlight"]["frames"]
    assert edl["selected_scene_count"] == result["highlight"]["selected_scene_count"]


def test_rejects_outside_image_without_command_execution(tmp_path):
    root = _capture(tmp_path, count=1)
    record = json.loads((root / "images.jsonl").read_text())
    record["path"] = "../../definitely-not-an-image;touch pwned.png"
    (root / "images.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = export(root, edit_id="safe-path", mode="scene-list")
    scene = json.loads((root / "derived/safe-path/scene_list.json").read_text())["scenes"][0]
    assert scene["image_status"] == "image_outside_capture_dir"
    assert not (root / "pwned.png").exists()
    assert result["scene_count"] == 1


def test_highlight_selection_prefers_consequential_events_and_preserves_bookends(tmp_path):
    root = _capture(tmp_path, count=10)
    images = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (root / "capture-events.jsonl").read_text().splitlines()]
    for index, event in enumerate(events):
        event["kind"] = "combat" if index == 6 else "turn_decision"
    scenes, warnings = build_scenes(root, images, events)
    selected = _select_highlight_scenes(scenes, root)
    assert not warnings
    assert len(selected) <= 8
    assert selected[0]["image_id"] == "image-9"
    assert selected[-1]["image_id"] == "image-0"
    assert any(scene["salience_reason"] == "combat" for scene in selected)


def test_highlight_selection_collapses_adjacent_duplicate_pngs(tmp_path):
    root = _capture(tmp_path, count=3)
    images = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    # Two records intentionally point at exactly the same visual.
    images[1]["path"] = images[0]["path"]
    events = [json.loads(line) for line in (root / "capture-events.jsonl").read_text().splitlines()]
    scenes, _ = build_scenes(root, images, events)
    selected = _select_highlight_scenes(scenes, root)
    assert len(selected) < 3


def test_structured_action_fields_drive_salience(tmp_path):
    root = _capture(tmp_path, count=2)
    images = [
        {"image_id": "attack", "file": "images/map-0.png", "source_event": {"capture_seq": 10}},
        {"image_id": "city", "file": "images/map-0.png", "source_event": {"capture_seq": 11}},
    ]
    events = [
        {"capture_seq": 10, "kind": "progress", "payload":
         {"event": "action_result", "domain": "military", "verb": "attack",
          "target": {"tile": [3, 4]}}},
        {"capture_seq": 11, "kind": "progress", "payload":
         {"event": "action_result", "domain": "city", "verb": "build_city",
          "affected": {"tile": [5, 6]}}},
    ]
    scenes, warnings = build_scenes(root, images, events)
    assert not warnings
    assert scenes[0]["salience_reason"] == "combat"
    assert scenes[0]["salience"] == 80
    assert scenes[1]["salience_reason"] == "city/build"


def test_executed_action_outcome_is_not_terminal_salience():
    score, reason = _scene_salience(
        ["action_result"], ["action=unit:3:move outcome=executed"], ["domain:unit", "verb:goto_e"]
    )
    assert score == 20
    assert reason == "verified action"


def test_free_form_victory_text_and_turn_end_are_not_terminal():
    score, reason = _scene_salience(
        ["action_result"], ["We need victory before defeat; outcome is uncertain"], ["verb:goto_e"]
    )
    assert score == 20
    assert reason == "verified action"
    score, reason = _scene_salience(["turn_end"], ["turn ended"], [])
    assert score == 1
    assert reason == "routine"


def test_engine_automatic_action_does_not_displace_model_action():
    score, reason = _scene_salience(
        ["action_result"], ["action=city:1:produce_auto outcome=executed"],
        ["domain:city", "verb:produce_auto", "provenance_class:engine_automatic"],
    )
    assert score <= 5
    assert reason == "engine automatic"


def test_highlight_selection_skips_unjoined_opening_and_covers_turns(tmp_path):
    root = _capture(tmp_path, count=10)
    images = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (root / "capture-events.jsonl").read_text().splitlines()]
    images[0]["trigger_event_ids"] = []
    images[0]["source_event"] = {"event_id": "presenter-attached", "kind": "presenter_attached"}
    events[0] = {"event_id": "presenter-attached", "kind": "presenter_attached"}
    for index, event in enumerate(events[1:], 1):
        event["kind"] = "progress"
        event["payload"] = {"event": "action_result", "turn": index, "verb": "goto_e"}
    scenes, warnings = build_scenes(root, images, events)
    selected = _select_highlight_scenes(scenes, root, limit=5)
    assert selected[0]["image_id"] != "image-0"
    assert len({str(scene["turn"]) for scene in selected}) >= 4


def test_short_highlight_also_drops_loading_opening(tmp_path):
    root = _capture(tmp_path, count=2)
    images = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (root / "capture-events.jsonl").read_text().splitlines()]
    images[0]["trigger_event_ids"] = ["presenter-attached"]
    events[0] = {"event_id": "presenter-attached", "kind": "presenter_attached"}
    scenes, _ = build_scenes(root, images, events)
    selected = _select_highlight_scenes(scenes, root, limit=8)
    assert [scene["image_id"] for scene in selected] == ["image-1"]
