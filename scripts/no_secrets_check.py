#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WP5 no-secrets check (design-plan-freeciv-v1.1.md §4, WP5).

Scans the repository for accidental secret leakage:

- Mistral key assignments: ``MISTRAL_API_KEY=`` / ``mistral_api_key:`` followed
  by a VALUE. Documented name-only references (no value) are not findings.
- Generic credential-shaped assignments (``*_KEY=`` / ``*_TOKEN=`` /
  ``*_SECRET=`` / ``*_PASSWORD=`` with a non-placeholder value).
- High-entropy tokens in text files (Mistral keys carry no known stable
  prefix, so detection relies on assignment patterns plus an entropy sweep).

Every finding is classified: checksum-shaped matches (image digests, sha256
checksums, npm integrity hashes) are reported as likely false positives rather
than suppressed; anything else is ``needs_review``. Snippets are redacted to a
prefix plus the token length — nothing secret-looking is ever reproduced in
full by this tool.

Excluded from scanning (never read): any file named ``.env`` (the Mistral key
is loaded from it at runtime through the narrow user-approved loader path;
this tool deliberately never opens it), ``.venv/``, ``node_modules/``, any
``.git/`` directory (including ``vendor/civrealm/.git`` internals),
``__pycache__/`` and compiled artifacts, and binary files (detected by a NUL
byte in the first block).

Usage:
  .venv/bin/python scripts/no_secrets_check.py [--repo REPO] [--out REPORT_JSON]

Default report path: artifacts/runs/wp5-prep/no_secrets_report.json

MIT: imports only the standard library (CivRealm-free).
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

EXCLUDED_DIR_NAMES = {".venv", "node_modules", ".git", "__pycache__", ".pytest_cache"}
EXCLUDED_FILE_NAMES = {".env"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".png", ".jpg", ".jpeg", ".webm",
                     ".sav", ".xz", ".gz", ".zip", ".whl", ".ico", ".zst",
                     ".zstd", ".bin", ".woff", ".woff2", ".ttf"}
MAX_SCAN_BYTES = 8 * 1024 * 1024

# MISTRAL_API_KEY= followed by a value (single- or double-quoted or bare).
_MISTRAL_ASSIGN = re.compile(
    r"(?im)\b(mistral[_\-]?api[_\-]?key)\b\s*[:=]\s*['\"]?([^\s'\"]+)['\"]?")
# Generic credential-shaped assignments in config-style text. Case-sensitive
# ALL_CAPS names only (the credential convention) — lowercase identifiers
# ending in "key(s)" (e.g. a dict named ``keys``) are not credentials.
_GENERIC_ASSIGN = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*)\s*=\s*['\"]?([^\s'\"]+)['\"]?")
# High-entropy candidate tokens. Deliberately narrow: no underscores, dots,
# slashes or hyphens (Python identifiers, module paths, URLs and slugs are not
# secrets), >= 32 chars, and the character-class filter below (letters AND
# digits) excludes prose.
_ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9+=]{32,}")
# Known checksum/integrity shapes (false positives by design).
_SHA256_HEX = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_NPM_INTEGRITY = re.compile(r"^sha512-[A-Za-z0-9+/]{80,}={0,2}$")
_BASE64_PADDED = re.compile(r"^[A-Za-z0-9+/]{40,}={1,2}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_PLACEHOLDER_MARKERS = ("<", ">", "...", "${", "%(", "{{", "{")
_PLACEHOLDER_WORDS = {"", "your_key_here", "your-api-key", "yourkey",
                      "changeme", "placeholder", "redacted", "todo", "none",
                      "null", "xxx", "example", "insert_key_here"}


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip("'\"")
    if v.lower() in _PLACEHOLDER_WORDS or v.isdigit():
        return True
    return any(m in v for m in _PLACEHOLDER_MARKERS)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


_HEX_DIGEST = re.compile(r"^[0-9a-fA-F]{32,}$")
_RECORDED_HASH_PREFIXES = ("sha256=", "sha256-", "sha512=", "sha512-",
                           "sha1=", "md5=", "container=", "local=")


def classify_token(token: str) -> tuple:
    """Return (classification, note) for a high-entropy token."""
    stripped = token
    for prefix in _RECORDED_HASH_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if _HEX_DIGEST.match(stripped) or _SHA256_HEX.match(stripped) \
            or _NPM_INTEGRITY.match(stripped) or _BASE64_PADDED.match(stripped):
        return "likely_false_positive", "hex digest / recorded hash / integrity digest"
    if _IMAGE_DIGEST.match(stripped):
        return "likely_false_positive", "Docker image digest"
    return "needs_review", "high-entropy token, unrecognized shape"


def _looks_like_secret_value(value: str) -> bool:
    """A real credential value carries digits, or is long and mixed-case.
    Prose words ('followed', 'chat') and numeric config values are not."""
    v = value.strip().strip("'\"").strip(",;")
    if len(v) < 8:
        return False
    if any(c.isdigit() for c in v):
        return True
    return len(v) >= 16 and any(c.isupper() for c in v) and any(c.islower() for c in v)


def _redact(value: str) -> str:
    v = value.strip()
    return v[:4] + "..." + f"(len={len(v)})" if len(v) > 4 else "(short value)"


def scan_text(rel_path: str, text: str):
    findings = []
    for m in _MISTRAL_ASSIGN.finditer(text):
        value = m.group(2)
        if _is_placeholder(value) or not _looks_like_secret_value(value):
            continue
        findings.append({
            "path": rel_path, "line": text.count("\n", 0, m.start()) + 1,
            "kind": "mistral_key_assignment",
            "classification": "needs_review",
            "snippet": m.group(1) + "=" + _redact(value),
            "note": "MISTRAL_API_KEY= followed by a value (not a documented name-only reference)",
        })
    for m in _GENERIC_ASSIGN.finditer(text):
        name, value = m.group(1), m.group(2)
        if name.upper().endswith("MISTRAL_API_KEY") or _is_placeholder(value) \
                or not _looks_like_secret_value(value):
            continue
        if re.fullmatch(r"[0-9a-f]{64}", value.lower()) or _NPM_INTEGRITY.match(value):
            continue
        findings.append({
            "path": rel_path, "line": text.count("\n", 0, m.start()) + 1,
            "kind": "credential_shaped_assignment",
            "classification": "needs_review",
            "snippet": name + "=" + _redact(value),
            "note": "credential-shaped assignment with a non-placeholder value",
        })
    for m in _ENTROPY_TOKEN.finditer(text):
        token = m.group(0)
        # Skip URL path segments (drive file IDs, commit SHAs inside links).
        if text[m.start() - 1:m.start()] == "/" and text[m.end():m.end() + 1] == "/":
            continue
        has_letter = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if not (has_letter and has_digit) or _entropy(token) < 3.5:
            continue
        classification, note = classify_token(token)
        findings.append({
            "path": rel_path, "line": text.count("\n", 0, m.start()) + 1,
            "kind": "high_entropy_token",
            "classification": classification,
            "snippet": token[:12] + f"...(len={len(token)})",
            "note": note,
        })
    return findings


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def scan_repo(repo: Path, skip_paths=()):
    skip = {Path(p).resolve() for p in skip_paths}
    findings = []
    scanned = 0
    binary_skipped = 0
    excluded = []
    oversized = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIR_NAMES)
        for fname in sorted(files):
            path = Path(root) / fname
            rel = str(path.relative_to(repo))
            if path.resolve() in skip:
                excluded.append(rel + " (report output)")
                continue
            if fname in EXCLUDED_FILE_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
                excluded.append(rel)
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_SCAN_BYTES:
                oversized.append({"path": rel, "size_bytes": size})
                continue
            if _is_binary(path):
                binary_skipped += 1
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            findings.extend(scan_text(rel, text))
    return findings, scanned, binary_skipped, excluded, oversized


def summarize(findings):
    summary = {
        "mistral_key_assignments": sum(1 for f in findings if f["kind"] == "mistral_key_assignment"),
        "credential_shaped_assignments": sum(1 for f in findings if f["kind"] == "credential_shaped_assignment"),
        "high_entropy_tokens": sum(1 for f in findings if f["kind"] == "high_entropy_token"),
        "likely_false_positives": sum(1 for f in findings if f["classification"] == "likely_false_positive"),
        "needs_review": sum(1 for f in findings if f["classification"] == "needs_review"),
    }
    summary["genuine_leak_candidates"] = summary["needs_review"]
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=str(REPO / "artifacts/runs/wp5-prep/no_secrets_report.json"))
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    findings, scanned, binary_skipped, excluded, oversized = scan_repo(
        repo, skip_paths=[args.out])
    report = {
        "tool": "scripts/no_secrets_check.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "exclusions": {
            "never_read": [".env (any depth)", ".venv/", "node_modules/", ".git/ (incl. vendor/civrealm/.git)"],
            "suffix_skips": sorted(EXCLUDED_SUFFIXES),
            "binary_detection": "NUL byte in first 8 KiB",
            "max_file_bytes": MAX_SCAN_BYTES,
        },
        "files_scanned": scanned,
        "files_skipped_binary": binary_skipped,
        "files_skipped_by_name_or_suffix": len(excluded),
        "files_skipped_oversized": oversized,
        "findings": findings,
        "summary": summarize(findings),
        "notes": [
            "Mistral API keys carry no known stable prefix; detection combines "
            "assignment-pattern matching with an entropy sweep, so a key that "
            "never appears in a KEY=... assignment shape is only caught by the "
            "entropy sweep (best-effort, disclosed here).",
            "Checksum-shaped matches (image digests, recorded sha256s, npm "
            "integrity hashes) are reported as likely false positives, not suppressed.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"scanned {scanned} text files "
          f"({binary_skipped} binary skipped, {len(excluded)} excluded by name/suffix, "
          f"{len(oversized)} oversized)")
    print(f"findings: {len(findings)} "
          f"({report['summary']['likely_false_positives']} likely false positives, "
          f"{report['summary']['needs_review']} need review)")
    for f in findings:
        print(f"  [{f['classification']}] {f['kind']} {f['path']}:{f['line']} {f['snippet']} — {f['note']}")
    print(f"report: {out}")
    return 0 if report["summary"]["needs_review"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
