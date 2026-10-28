# SPDX-License-Identifier: GPL-3.0-or-later
"""A small, deterministic index for Freeciv's generated wiki manual.

The manual is an extracted, static reference.  It is deliberately kept
separate from the live ruleset reference: text in this module is useful for
orientation, but never constitutes game state or legality.  MediaWiki and
HTML are treated as data; this parser does not evaluate templates, links, or
markup.
"""
from __future__ import annotations

import hashlib
import html
import hmac
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .classic_reference import PINNED_IMAGE_REF

SCHEMA = "freeciv_manual_reference_v1"
AUTHORITY = "static_manual_ruleset_reference_non_authoritative"
DEFAULT_MAX_TEXT_CHARS = 12000
DEFAULT_RENDER_MAX_CHARS = 8000
_WORDS = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"a", "an", "and", "are", "can", "do", "for", "how", "is", "of", "on", "the", "to", "what", "with"})
_CATEGORY_ALIASES = {
    "unit": "unit", "units": "unit", "building": "building", "buildings": "building",
    "improvement": "building", "improvements": "building", "wonder": "wonder", "wonders": "wonder",
    "tech": "tech", "techs": "tech", "technology": "tech", "technologies": "tech",
    "terrain": "terrain", "terrains": "terrain", "government": "government", "governments": "government",
    "command": "command", "commands": "command", "option": "option", "options": "option",
}
_HEADING = re.compile(r"^(?P<marks>={1,6})\s*(?P<text>.*?)\s*(?P=marks)\s*$")
_ROW_START = re.compile(r"^\s*<tr\b[^>]*>\s*<td\b[^>]*>\s*\[\[Image:[^]]+\]\]", re.I)
_CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.I | re.S)
_TAG = re.compile(r"<[^>]*>")
_LINK = re.compile(r"\[\[(?:[^]|]+\|)?([^]]+)\]\]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManualReferenceError(ValueError):
    """The extracted manual failed an integrity or safety check."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clean(value: str, limit: int) -> str:
    value = html.unescape(value.replace("\r\n", "\n").replace("\r", "\n"))
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    value = re.sub(r"\[\[Image:[^]]+\]\]", " ", value, flags=re.I)
    value = _LINK.sub(r"\1", value)
    value = value.replace("'''", "").replace("''", "")
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = _TAG.sub(" ", value)
    value = re.sub(r"\{\{.*?\}\}", " ", value, flags=re.S)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value).strip()
    if limit < 1:
        return ""
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


def _category(title: str, filename: str) -> str:
    low = (title + " " + filename).casefold()
    for name, category in (("unit", "unit"), ("tech", "tech"), ("terrain", "terrain"), ("building", "building"), ("wonder", "wonder"), ("government", "government"), ("command", "command"), ("option", "option")):
        if name in low:
            return category
    return "manual"


@dataclass(frozen=True)
class ManualEntry:
    key: str
    name: str
    category: str
    text: str = ""
    section: str = ""
    source_file: str = ""
    source_line: int = 0
    authority: str = AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "category": self.category,
                "text": self.text, "section": self.section, "source_file": self.source_file,
                "source_line": self.source_line, "authority": self.authority}


@dataclass(frozen=True)
class ManualReference:
    entries: tuple[ManualEntry, ...] = ()
    status: str = "available"
    source_path: str = ""
    image_ref: str = ""
    image_version: str = ""
    manual_version: str = ""
    ruleset: str = "classic"
    source_sha256: str = ""
    file_sha256: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = SCHEMA
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda e: (e.category, e.name.casefold(), e.key, e.source_file)))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        identity = _canonical({"status": self.status, "ruleset": self.ruleset, "entries": [e.to_dict() for e in ordered]})
        object.__setattr__(self, "_hash", hashlib.sha256(identity.encode()).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {"schema_version": self.schema_version, "authority": AUTHORITY, "status": self.status,
                  "source_path": self.source_path, "image_ref": self.image_ref,
                  "image_version": self.image_version, "ruleset": self.ruleset,
                  "manual_version": self.manual_version, "source_sha256": self.source_sha256,
                  "file_sha256": dict(self.file_sha256), "warnings": list(self.warnings),
                  "entries": [e.to_dict() for e in self.entries]}
        if include_hash:
            result["content_hash"] = self.content_hash
        return result


def _entry_from_heading(lines: list[str], start: int, end: int, title: str, section: str, source: str, category: str, limit: int) -> ManualEntry:
    name = _clean(title, limit) or f"section-{start}"
    body = _clean("\n".join(lines[start + 1:end]), limit)
    return ManualEntry(key=name.casefold().replace(" ", "_"), name=name, category=category, text=body, section=section, source_file=source, source_line=start + 1)


def parse_mediawiki(text: str, *, source_file: str = "", max_entries: int = 4096, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> tuple[ManualEntry, ...]:
    """Parse headings and generated table rows without executing markup."""
    lines = text.splitlines()
    title = ""
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        match = _HEADING.match(line)
        if not match:
            continue
        level = len(match.group("marks"))
        value = _clean(match.group("text"), max_text_chars)
        if level == 1:
            title = value
        elif level >= 3:
            headings.append((i, level, value))
    category = _category(title, source_file)
    entries: list[ManualEntry] = []
    for pos, (start, level, value) in enumerate(headings):
        end = len(lines)
        for nxt, nxt_level, _ in headings[pos + 1:]:
            if nxt_level <= level:
                end = nxt
                break
        entries.append(_entry_from_heading(lines, start, end, value, title, source_file, category, max_text_chars))
        if len(entries) >= max_entries:
            return tuple(entries)
    # Tables generated by the manual use one image-bearing first cell for each
    # top-level row.  Splitting on those row starts keeps nested stat tables
    # intact and avoids an HTML parser's error recovery changing the text.
    starts = [i for i, line in enumerate(lines) if _ROW_START.match(line)]
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        raw = "\n".join(lines[start:end])
        cells = [_clean(x, max_text_chars) for x in _CELL.findall(raw)]
        if len(cells) < 2:
            continue
        # Terrain rows put the display name beside the image in the first
        # cell; buildings/wonders put it in the second cell.  The latter
        # also contains the first numeric/stat column, so never use it when
        # the image cell already has visible text.
        name = cells[0].strip() or cells[1].strip()
        if not name or name.casefold() in {e.name.casefold() for e in entries}:
            continue
        entries.append(ManualEntry(key=name.casefold().replace(" ", "_"), name=name, category=category,
                                   text=_clean(raw, max_text_chars), section=title, source_file=source_file, source_line=start + 1))
        if len(entries) >= max_entries:
            break
    return tuple(entries)


def _manifest_files(manifest: dict[str, Any]) -> dict[str, str]:
    raw = manifest.get("manual_files") or manifest.get("files") or manifest.get("manual_file_sha256")
    result: dict[str, str] = {}
    if isinstance(raw, dict):
        for name, digest in raw.items():
            if isinstance(name, str) and isinstance(digest, str): result[name] = digest
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str): result[item["path"]] = item["sha256"]
    return result


def load_manual_reference(reference_dir: str | Path, *, max_files: int = 16, max_file_bytes: int = 512_000, max_total_bytes: int = 4_000_000, max_entries: int = 4096, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> ManualReference:
    root = Path(reference_dir)
    if not root.is_dir():
        return ManualReference(status="unavailable", source_path=str(root), warnings=("manual reference is absent",))
    files_root = root / "manual" if (root / "manual").is_dir() else root
    manifest_path = root / "manual_reference.json"
    if not manifest_path.is_file(): manifest_path = root / "reference.json"
    if not manifest_path.is_file():
        return ManualReference(status="unavailable", source_path=str(files_root), warnings=("manual manifest is absent",))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in {"freeciv-manual-reference-v1", SCHEMA}:
            raise ManualReferenceError("unsupported manual manifest schema")
        if manifest.get("ruleset") != "classic":
            raise ManualReferenceError("manual manifest is not for the classic ruleset")
        expected = _manifest_files(manifest)
        if not expected:
            raise ManualReferenceError("manual manifest has no per-file checksums")
        normalized: dict[str, str] = {}
        for name, digest in expected.items():
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts or candidate.parts[:1] == ("manual",):
                if candidate.parts[:1] != ("manual",) or len(candidate.parts) != 2:
                    raise ManualReferenceError("manual manifest contains an unsafe path")
                name = candidate.name
            normalized[name] = digest
        expected = normalized
        image_ref = manifest.get("image_ref")
        if not isinstance(image_ref, str) or not hmac.compare_digest(image_ref, PINNED_IMAGE_REF):
            raise ManualReferenceError("manual manifest image is not the pinned Freeciv image")
        image_id = manifest.get("image_id")
        if image_id is not None and image_id != image_ref.rsplit("@", 1)[-1]:
            raise ManualReferenceError("manual manifest image identity mismatch")
        files = sorted(p for p in files_root.iterdir() if p.is_file() and p.suffix.casefold() in {".mediawiki", ".wiki", ".html"})
        if not files or len(files) > max_files: raise ManualReferenceError("manual file count exceeds limit or is empty")
        actual_names = {p.name for p in files}
        if actual_names != set(expected): raise ManualReferenceError("manual manifest file list does not match extracted files")
        total = 0; entries: list[ManualEntry] = []; checks: list[tuple[str, str]] = []
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(files_root.resolve()): raise ManualReferenceError(f"unsafe manual path: {path}")
            size = path.stat().st_size
            total += size
            if size > max_file_bytes or total > max_total_bytes: raise ManualReferenceError("manual size exceeds limit")
            rel = path.relative_to(files_root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            recorded = expected.get(rel)
            if not recorded or not _SHA256.fullmatch(recorded) or not hmac.compare_digest(digest, recorded): raise ManualReferenceError(f"manual checksum verification failed: {rel}")
            checks.append((rel, digest)); entries.extend(parse_mediawiki(path.read_text(encoding="utf-8"), source_file=rel, max_entries=max_entries - len(entries), max_text_chars=max_text_chars))
            if len(entries) >= max_entries: raise ManualReferenceError("manual entry count exceeds limit")
        canonical_files = _canonical(dict(checks)).encode("utf-8")
        source_sha256 = hashlib.sha256(canonical_files).hexdigest()
        recorded_source = manifest.get("source_sha256")
        if recorded_source is not None and (not isinstance(recorded_source, str) or not hmac.compare_digest(recorded_source, source_sha256)):
            raise ManualReferenceError("manual source checksum verification failed")
        return ManualReference(tuple(entries), source_path=str(files_root), image_ref=str(manifest.get("image_ref", "")), image_version=str(manifest.get("image_version", "")), manual_version=str(manifest.get("manual_version", "")), ruleset="classic", source_sha256=source_sha256, file_sha256=tuple(checks))
    except ManualReferenceError: raise
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ManualReferenceError(f"manual reference unavailable: {type(exc).__name__}") from exc


def search_entries(reference: ManualReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10) -> tuple[dict[str, Any], ...]:
    if not isinstance(query, str) or not query.strip() or limit <= 0: return ()
    terms = tuple(x for x in _WORDS.findall(query.casefold()) if x not in _STOP)
    allowed = {_CATEGORY_ALIASES.get(str(x).casefold(), str(x).casefold()) for x in categories} if categories is not None else None
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for entry in reference.entries:
        if allowed is not None and _CATEGORY_ALIASES.get(entry.category.casefold(), entry.category.casefold()) not in allowed: continue
        hay = " ".join((entry.key, entry.name, entry.category, entry.text, entry.section)).casefold()
        if terms and not all(term in hay for term in terms): continue
        wanted = query.casefold().strip()
        names = {entry.key.casefold(), entry.name.casefold()}
        # Human queries commonly use the singular while the manual title is
        # plural ("warrior" -> "Warriors").  Treat only this simple suffix
        # as an exact-title alias so ranking stays deterministic and narrow.
        singular = {name[:-1] for name in names if name.endswith("s") and len(name) > 3}
        plural = {name + "s" for name in names if not name.endswith("s")}
        exact = wanted in names or wanted in singular or wanted in plural
        matches.append((0 if exact else 1, entry.name.casefold(), entry.to_dict()))
    matches.sort(key=lambda x: (x[0], x[1], x[2]["key"], x[2]["source_file"]))
    return tuple(item[2] for item in matches[:limit])


def render_search(reference: ManualReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10, max_chars: int = DEFAULT_RENDER_MAX_CHARS) -> dict[str, Any]:
    result = {"schema_version": reference.schema_version, "authority": AUTHORITY, "status": reference.status, "source_path": reference.source_path, "image_ref": reference.image_ref, "image_version": reference.image_version, "ruleset": reference.ruleset, "content_hash": reference.content_hash, "query": query, "entries": [], "omitted": 0}
    used = 0
    for item in search_entries(reference, query, categories=categories, limit=limit):
        size = len(_canonical(item)) + 1
        if used + size > max_chars: result["omitted"] += 1
        else: result["entries"].append(item); used += size
    return result


__all__ = ["AUTHORITY", "ManualEntry", "ManualReference", "ManualReferenceError", "load_manual_reference", "parse_mediawiki", "search_entries", "render_search"]
