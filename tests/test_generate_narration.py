from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import types
import wave
from pathlib import Path

import pytest

import scripts.generate_narration as narration


def _segments(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _wav_bytes(frames: int = 24000) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * frames)
    return stream.getvalue()


def test_segment_ids_are_path_safe(tmp_path):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "../escape", "text": "hello"}])
    with pytest.raises(ValueError, match="invalid segment id"):
        narration._load_segments(source)


def test_existing_audio_mismatch_fails_before_key_load(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "new text"}])
    output = tmp_path / "out"
    output.mkdir()
    (output / "intro.wav").write_bytes(b"old")
    (output / "manifest.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    monkeypatch.setattr(narration, "_load_key", lambda: pytest.fail("key loaded"))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        narration.generate(source, output)


def test_existing_audio_hash_corruption_fails(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "hello"}])
    output = tmp_path / "out"
    output.mkdir()
    audio = output / "intro.wav"
    audio.write_bytes(b"original")
    fingerprint = narration._request_fingerprint({"id": "intro", "text": "hello"},
                                                   narration.DEFAULT_MODEL,
                                                   narration.DEFAULT_VOICE_ID, 1.0)
    manifest = {"segments": [{"id": "intro", "request_sha256": fingerprint,
                               "sha256": "not-the-file"}]}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(narration, "_load_key", lambda: pytest.fail("key loaded"))
    with pytest.raises(RuntimeError, match="hash check"):
        narration.generate(source, output)


def test_matching_manifest_invalid_wav_fails_before_key_load(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "hello"}])
    output = tmp_path / "out"
    output.mkdir()
    audio = output / "intro.wav"
    audio.write_bytes(b"ID3-not-a-wav")
    fingerprint = narration._request_fingerprint({"id": "intro", "text": "hello"},
                                                   narration.DEFAULT_MODEL,
                                                   narration.DEFAULT_VOICE_ID, 1.0)
    manifest = {"segments": [{"id": "intro", "request_sha256": fingerprint,
                               "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}]}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(narration, "_load_key", lambda: pytest.fail("key loaded"))
    with pytest.raises(ValueError, match="invalid PCM WAV"):
        narration.generate(source, output)


def test_invalid_wav_is_not_installed(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "hello"}])
    output = tmp_path / "out"
    fake_client = types.ModuleType("mistralai.client")
    fake_client.Mistral = lambda **kwargs: types.SimpleNamespace(
        audio=types.SimpleNamespace(speech=types.SimpleNamespace(
            complete=lambda **request: types.SimpleNamespace(
                audio_data=base64.b64encode(b"not a wav").decode()))))
    fake_package = types.ModuleType("mistralai")
    monkeypatch.setattr(narration, "_load_key", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "mistralai", fake_package)
    monkeypatch.setitem(sys.modules, "mistralai.client", fake_client)
    retry_module = types.ModuleType("mistralai.client.utils.retries")
    retry_module.RetryConfig = lambda **kwargs: kwargs
    retry_module.BackoffStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mistralai.client.utils", types.ModuleType("mistralai.client.utils"))
    monkeypatch.setitem(sys.modules, "mistralai.client.utils.retries", retry_module)
    with pytest.raises(Exception):
        narration.generate(source, output)
    assert not (output / "intro.wav").exists()
    assert not (output / "intro.wav.partial").exists()


def test_non_wav_payload_is_rejected_before_install(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "hello"}])
    output = tmp_path / "out"
    fake_client = types.ModuleType("mistralai.client")
    fake_client.Mistral = lambda **kwargs: types.SimpleNamespace(
        audio=types.SimpleNamespace(speech=types.SimpleNamespace(
            complete=lambda **request: types.SimpleNamespace(
                audio_data=base64.b64encode(b"ID3\x04\x00not-an-mp3-or-wav").decode()))))
    fake_package = types.ModuleType("mistralai")
    monkeypatch.setattr(narration, "_load_key", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "mistralai", fake_package)
    monkeypatch.setitem(sys.modules, "mistralai.client", fake_client)
    retry_module = types.ModuleType("mistralai.client.utils.retries")
    retry_module.RetryConfig = lambda **kwargs: kwargs
    retry_module.BackoffStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mistralai.client.utils", types.ModuleType("mistralai.client.utils"))
    monkeypatch.setitem(sys.modules, "mistralai.client.utils.retries", retry_module)
    with pytest.raises(ValueError, match="invalid PCM WAV"):
        narration.generate(source, output)
    assert not (output / "intro.wav").exists()
    assert not (output / "intro.wav.partial").exists()


def test_valid_wav_is_installed_and_reused_without_second_request(tmp_path, monkeypatch):
    source = tmp_path / "segments.json"
    _segments(source, [{"id": "intro", "text": "hello"}])
    output = tmp_path / "out"
    calls = []
    fake_client = types.ModuleType("mistralai.client")
    fake_client.Mistral = lambda **kwargs: types.SimpleNamespace(
        audio=types.SimpleNamespace(speech=types.SimpleNamespace(
            complete=lambda **request: (calls.append(request) or types.SimpleNamespace(
                audio_data=base64.b64encode(_wav_bytes()).decode())))))
    fake_package = types.ModuleType("mistralai")
    monkeypatch.setattr(narration, "_load_key", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "mistralai", fake_package)
    monkeypatch.setitem(sys.modules, "mistralai.client", fake_client)
    retry_module = types.ModuleType("mistralai.client.utils.retries")
    retry_module.RetryConfig = lambda **kwargs: kwargs
    retry_module.BackoffStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mistralai.client.utils", types.ModuleType("mistralai.client.utils"))
    monkeypatch.setitem(sys.modules, "mistralai.client.utils.retries", retry_module)

    first = narration.generate(source, output)
    second = narration.generate(source, output)
    assert len(calls) == 1
    assert first == second
    assert first["segments"][0]["duration_seconds"] == pytest.approx(1.0)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == second
