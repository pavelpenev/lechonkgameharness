# SPDX-License-Identifier: MIT
"""Offline tests for the WP5 no-secrets checker (no network, no .env access)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import no_secrets_check as nsc  # noqa: E402


def test_mistral_assignment_with_value_flagged():
    findings = nsc.scan_text("x", "MISTRAL_API_KEY=dummy0123456789abcdef")
    kinds = [f["kind"] for f in findings]
    assert "mistral_key_assignment" in kinds


def test_name_only_reference_not_flagged():
    findings = nsc.scan_text("x", "The MISTRAL_API_KEY env var holds the key.")
    assert not any(f["kind"] == "mistral_key_assignment" for f in findings)


def test_placeholder_value_not_flagged():
    findings = nsc.scan_text("x", "MISTRAL_API_KEY=<your-key-here>")
    assert not any(f["kind"] == "mistral_key_assignment" for f in findings)


def test_snippet_is_redacted():
    findings = nsc.scan_text("x", "MISTRAL_API_KEY=dummy0123456789abcdef")
    mistral = [f for f in findings if f["kind"] == "mistral_key_assignment"][0]
    assert "dummy0123456789abcdef" not in mistral["snippet"]
    assert "dumm" in mistral["snippet"]


def test_sha256_digest_classified_false_positive():
    digest = "4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a"
    findings = [f for f in nsc.scan_text("x", digest) if f["kind"] == "high_entropy_token"]
    assert findings and findings[0]["classification"] == "likely_false_positive"


def test_scan_repo_skips_env_and_bins(tmp_path):
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=shouldneverberead123456")
    (tmp_path / "blob.dat").write_bytes(b"\0\x01\x02")
    (tmp_path / "notes.txt").write_text("safe prose only")
    findings, scanned, binary_skipped, excluded, oversized = nsc.scan_repo(tmp_path)
    assert scanned == 1  # only notes.txt
    assert binary_skipped == 1
    assert sorted(excluded) == [".env"]
    assert not any("shouldneverberead" in (f.get("snippet") or "") for f in findings)


def test_placeholder_word_not_flagged():
    findings = nsc.scan_text("x", "API_KEY=changeme")
    assert not any(f["kind"] == "credential_shaped_assignment" for f in findings)
