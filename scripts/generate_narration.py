#!/usr/bin/env python3
# SCRIPT-METADATA:
# name: generate_narration
# description: Generate provenance-tracked Voxtral TTS WAV segments
# tags: audio,narration,mistral,post-production
# language: python
"""Generate speech segments for an editorial cut.

The script deliberately makes one provider request per missing segment and
does not retry paid requests.  Existing files are reused only when the exact
request (including text and settings) is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import wave
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_VOICE_ID = "voice_592"  # Marie (French), Voxtral TTS 3
DEFAULT_EXPRESSIVENESS = 1.0  # Studio Robust
DEFAULT_TIMEOUT_MS = 120_000
_SEGMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _load_key() -> None:
    from lechonk.model.mistral_client import load_api_key_from_env_file
    load_api_key_from_env_file()


def _request_fingerprint(segment: dict[str, Any], model: str, voice_id: str,
                         expressiveness: float) -> str:
    payload = {"id": segment["id"], "text": segment["text"],
               "model": model, "voice_id": voice_id,
               "expressiveness": expressiveness, "response_format": "wav"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _atomic_manifest_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_wav(path: Path) -> float:
    """Validate an actual, complete PCM WAV and return its duration."""
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frames = stream.getnframes()
            if (stream.getcomptype() != "NONE" or channels <= 0 or rate <= 0
                    or sample_width <= 0 or frames <= 0):
                raise ValueError("WAV must contain positive PCM audio parameters")
            decoded = len(stream.readframes(frames)) // (channels * sample_width)
            if decoded != frames:
                raise ValueError("WAV data is truncated")
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        raise ValueError(f"invalid PCM WAV: {path}") from exc
    return frames / rate


def _load_segments(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("segments JSON must be a non-empty list")
    result = []
    seen = set()
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("text"), str):
            raise ValueError("each segment must contain string id and text")
        if not _SEGMENT_ID.fullmatch(item["id"]):
            raise ValueError(f"invalid segment id: {item['id']!r}")
        if item["id"] in seen:
            raise ValueError(f"duplicate segment id: {item['id']!r}")
        if not item["text"].strip():
            raise ValueError(f"empty text for segment {item['id']!r}")
        seen.add(item["id"])
        result.append({"id": item["id"], "text": item["text"]})
    return result


def generate(segments_path: Path, output_dir: Path, *, model: str = DEFAULT_MODEL,
             voice_id: str = DEFAULT_VOICE_ID,
             expressiveness: float = DEFAULT_EXPRESSIVENESS) -> dict[str, Any]:
    if not 1.0 <= expressiveness <= 5.0:
        raise ValueError("expressiveness must be between 1.0 and 5.0")
    segments = _load_segments(segments_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {"segments": []}
    records = {item["id"]: item for item in old.get("segments", [])}
    # Complete all safety checks before the first paid request.  A stale or
    # corrupted existing artifact must never be silently overwritten.
    for segment in segments:
        fingerprint = _request_fingerprint(segment, model, voice_id, expressiveness)
        target = output_dir / f"{segment['id']}.wav"
        prior = records.get(segment["id"])
        if target.exists():
            if prior is None or prior.get("request_sha256") != fingerprint:
                raise RuntimeError(
                    f"existing audio {target} has no matching manifest request; "
                    "refusing to overwrite")
            recorded_sha = prior.get("sha256")
            if not isinstance(recorded_sha, str) \
                    or hashlib.sha256(target.read_bytes()).hexdigest() != recorded_sha:
                raise RuntimeError(f"existing audio {target} failed manifest hash check")
            _validate_wav(target)
    _load_key()
    from mistralai.client import Mistral
    from mistralai.client.utils.retries import BackoffStrategy, RetryConfig
    sdk = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=DEFAULT_TIMEOUT_MS,
                  retry_config=RetryConfig(
                      strategy="none",
                      backoff=BackoffStrategy(initial_interval=1, max_interval=1,
                                              exponent=1, max_elapsed_time=1),
                      retry_connection_errors=False))
    for segment in segments:
        fingerprint = _request_fingerprint(segment, model, voice_id, expressiveness)
        target = output_dir / f"{segment['id']}.wav"
        prior = records.get(segment["id"])
        if prior and prior.get("request_sha256") == fingerprint and target.is_file():
            continue
        response = sdk.audio.speech.complete(
            model=model, input=segment["text"], voice_id=voice_id,
            expressiveness=expressiveness, response_format="wav", stream=False,
            timeout_ms=DEFAULT_TIMEOUT_MS)
        encoded = getattr(response, "audio_data", None)
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError(f"TTS response for {segment['id']!r} had no audio_data")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RuntimeError(f"invalid base64 audio for {segment['id']!r}") from exc
        temporary = target.with_suffix(".wav.partial")
        try:
            temporary.write_bytes(audio)
            duration = _validate_wav(temporary)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        records[segment["id"]] = {
            "id": segment["id"], "text": segment["text"],
            "model": model, "voice_id": voice_id,
            "expressiveness": expressiveness, "response_format": "wav",
            "request_sha256": fingerprint, "sha256": hashlib.sha256(audio).hexdigest(),
            "duration_seconds": duration, "path": target.name,
        }
        _atomic_manifest_write(manifest_path, {"schema_version": "narration-v1",
                                                "segments": list(records.values())})
    manifest = {"schema_version": "narration-v1",
                "segments": [records[s["id"]] for s in segments]}
    _atomic_manifest_write(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--expressiveness", type=float, default=DEFAULT_EXPRESSIVENESS)
    args = parser.parse_args()
    manifest = generate(args.segments, args.output_dir, model=args.model,
                        voice_id=args.voice_id, expressiveness=args.expressiveness)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "segments": len(manifest["segments"])}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
