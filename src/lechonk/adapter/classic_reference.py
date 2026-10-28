# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, non-authoritative index of extracted classic Freeciv rulesets.

Only display strings are read from ruleset files.  In particular, this module
never interprets requirements, effects, or any other ruleset expression.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "freeciv_classic_reference_v1"
AUTHORITY = "static_ruleset_source_non_authoritative"
# The reference is trusted only when it came from the image used by the
# harness.  A digest-shaped value in a user-edited manifest is not enough:
# provenance must identify this exact known image.
PINNED_IMAGE_REF = "civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a"
# Keep the complete assignment grammar here.  Ruleset files contain many
# non-display assignments between ``name`` and ``helptext``; treating those as
# continuations corrupts the preceding display value.
_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_DISPLAY_FIELDS = frozenset(("name", "rule_name", "helptext"))
_SECTION = re.compile(r"^\s*\[([^]]+)\]")
_STOP = {"a", "an", "and", "are", "can", "do", "for", "how", "is", "of", "on", "the", "to", "what", "with"}
_WORDS = re.compile(r"\w+", re.UNICODE)
_CATEGORIES = {"units": "unit", "buildings": "improvement", "improvements": "improvement", "wonders": "wonder", "techs": "tech", "technologies": "tech", "terrain": "terrain", "governments": "government", "extras": "extra", "nations": "nation", "specialists": "specialist"}
# Keep category spelling compatible with the live RulesetCtrl lookup.  The
# extra classic-only categories remain available for ruleset files that carry
# those sections.
_CATEGORY_ALIASES = {
    "tech": "tech", "techs": "tech", "technology": "tech", "technologies": "tech",
    "unit": "unit", "units": "unit",
    "building": "improvement", "buildings": "improvement",
    "improvement": "improvement", "improvements": "improvement",
    "wonder": "wonder", "wonders": "wonder",
    "terrain": "terrain", "terrains": "terrain",
    "government": "government", "governments": "government",
    "extra": "extra", "extras": "extra",
    "nation": "nation", "nations": "nation",
    "specialist": "specialist", "specialists": "specialist",
}


def _singular_plural_equal(left: str, right: str) -> bool:
    """Match only a regular trailing-``s`` singular/plural pair."""
    if left == right:
        return True
    if len(left) < 3 or len(right) < 4:
        return False
    if right.endswith("s") and not right.endswith("ss") and right[:-1] == left:
        return True
    if left.endswith("s") and not left.endswith("ss") and left[:-1] == right:
        return True
    return False


class ClassicReferenceError(ValueError):
    """The extracted reference failed its integrity or safety checks."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clean(value: str, limit: int) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value).strip()
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


def _quoted(value: str) -> list[str]:
    """Decode quoted fragments only; wrappers such as _() are ignored."""
    out: list[str] = []
    i = 0
    while i < len(value):
        if value[i] not in "\"'":
            i += 1
            continue
        quote, start = value[i], i
        i += 1
        escaped = False
        while i < len(value):
            char = value[i]
            if char == quote and not escaped:
                token = value[start:i + 1]
                try:
                    decoded = ast.literal_eval(token)
                except (SyntaxError, ValueError, UnicodeError):
                    decoded = None
                if isinstance(decoded, str):
                    out.append(decoded)
                i += 1
                break
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            i += 1
        else:
            break
    return out


def _has_unclosed_quote(value: str) -> bool:
    """Return whether *value* ends inside a quoted string.

    Freeciv uses C-style ``\\`` line continuations in long help strings.  We
    only consume following lines while this lexical state is open, rather than
    treating arbitrary non-assignment lines as part of the previous field.
    """
    quote = ""
    escaped = False
    for char in value:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
    return bool(quote)


def _display_value(field: str, raw: str, limit: int) -> str:
    value = _clean("".join(_quoted(raw)), limit)
    # Freeciv's translated display names may carry a disambiguation marker,
    # e.g. ``_("?unit:Warriors")``.  The marker is metadata, not user-facing
    # text.  Restrict stripping to name fields so help text remains verbatim.
    if field in {"name", "rule_name"}:
        value = re.sub(r"^\?[A-Za-z_][A-Za-z0-9_-]*:", "", value)
    return value


def _tree_hash(root: Path) -> str:
    parts: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ClassicReferenceError(f"symlink in classic reference: {path}")
        if path.is_file():
            rel = ("./" + path.relative_to(root).as_posix()).encode()
            digest = hashlib.sha256(path.read_bytes()).hexdigest().encode()
            # Match the extractor's `sha256sum | sha256sum`: each line is
            # ``<digest>  ./relative/path``.
            parts.append(digest + b"  " + rel + b"\n")
    return hashlib.sha256(b"".join(parts)).hexdigest()


@dataclass(frozen=True)
class ClassicEntry:
    key: str
    name: str
    rule_name: str = ""
    helptext: str = ""
    source_file: str = ""
    source_line: int = 0
    section: str = ""
    category: str = "ruleset"
    authority: str = AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "rule_name": self.rule_name,
                "helptext": self.helptext, "source_file": self.source_file,
                "source_line": self.source_line, "section": self.section,
                "category": self.category, "authority": self.authority}


@dataclass(frozen=True)
class ClassicReference:
    entries: tuple[ClassicEntry, ...] = ()
    status: str = "available"
    source_path: str = ""
    image_ref: str = ""
    image_version: str = ""
    tree_sha256: str = ""
    warnings: tuple[str, ...] = ()
    schema_version: str = SCHEMA
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda e: (e.name.casefold(), e.key, e.source_file)))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        object.__setattr__(self, "_hash", hashlib.sha256(_canonical([e.to_dict() for e in ordered]).encode()).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._hash

    @property
    def provenance(self) -> dict[str, str]:
        """Stable JSON-safe source provenance, separate from semantic hash."""
        return {
            "authority": AUTHORITY,
            "source_path": self.source_path,
            "image_ref": self.image_ref,
            "image_version": self.image_version,
            "tree_sha256": self.tree_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-native snapshot for session/prompt serializers."""
        return {
            "schema_version": self.schema_version,
            "authority": AUTHORITY,
            "status": self.status,
            "source_path": self.source_path,
            "image_ref": self.image_ref,
            "image_version": self.image_version,
            "tree_sha256": self.tree_sha256,
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _category(path: str) -> str:
    stem = Path(path).stem.casefold()
    return _CATEGORIES.get(stem, "ruleset")


def parse_classic_ruleset(text: str, *, source_file: str, max_text_chars: int = 8000, max_entries: int = 2048) -> tuple[ClassicEntry, ...]:
    entries: list[ClassicEntry] = []
    section = ""
    line_no = 0
    fields: dict[str, str] = {}
    starts: dict[str, int] = {}
    active: str | None = None

    def finish() -> None:
        nonlocal fields, starts, active
        if not fields or len(entries) >= max_entries:
            fields = {}
            starts = {}
            active = None
            return
        values = {key: _display_value(key, value, max_text_chars) for key, value in fields.items()}
        name = values.get("name") or values.get("rule_name") or section
        rule = values.get("rule_name", "")
        helptext = values.get("helptext", "")
        if name or helptext:
            entries.append(ClassicEntry(section, name, rule, helptext, source_file, starts.get("name", line_no), section, _category(source_file)))
        fields = {}
        starts = {}
        active = None

    for line_no, line in enumerate(text.splitlines(), 1):
        match = _SECTION.match(line)
        if match:
            finish()
            section = match.group(1).strip()
            continue
        match = _ASSIGN.match(line)
        if match:
            field_name, value = match.groups()
            # Every syntactically valid assignment is a lexical boundary.
            # Only the three display fields are retained in the entry.
            if field_name in _DISPLAY_FIELDS:
                fields[field_name] = value
                starts[field_name] = line_no
                active = field_name if _has_unclosed_quote(value) else None
            else:
                active = None
        elif active and line.strip() and not line.lstrip().startswith((";", "#")):
            fields[active] += "\n" + line
            if not _has_unclosed_quote(fields[active]):
                active = None
    finish()
    return tuple(entries)


def load_classic_reference(reference_dir: str | Path, *, max_files: int = 128, max_file_bytes: int = 1_048_576, max_total_bytes: int = 8_388_608, max_entries: int = 2048, max_text_chars: int = 8000) -> ClassicReference:
    root = Path(reference_dir)
    classic = root / "classic"
    if not root.is_dir() or not classic.is_dir():
        return ClassicReference(status="unavailable", source_path=str(classic), warnings=("classic ruleset reference is absent",))
    manifest_path = root / "reference.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_ref = manifest.get("image_ref")
        if not isinstance(image_ref, str) or not hmac.compare_digest(image_ref, PINNED_IMAGE_REF):
            raise ClassicReferenceError(
                "classic reference image is not the pinned Freeciv image: "
                f"expected {PINNED_IMAGE_REF}, got {image_ref!r}; "
                "re-run ./scripts/extract_freeciv_reference.sh with the "
                "pinned image (do not edit image_ref by hand)."
            )
        expected = manifest.get("classic_ruleset_tree_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ClassicReferenceError("reference.json has no valid classic tree checksum")
        files = sorted(p for p in classic.rglob("*") if p.is_file())
        if len(files) > max_files:
            raise ClassicReferenceError("classic ruleset file count exceeds limit")
        total = 0
        entries: list[ClassicEntry] = []
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(classic.resolve()):
                raise ClassicReferenceError(f"unsafe classic ruleset path: {path}")
            size = path.stat().st_size
            if size > max_file_bytes or (total := total + size) > max_total_bytes:
                raise ClassicReferenceError("classic ruleset size exceeds limit")
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(classic).as_posix()
            entries.extend(parse_classic_ruleset(text, source_file=rel, max_text_chars=max_text_chars, max_entries=max_entries - len(entries)))
            if len(entries) >= max_entries:
                raise ClassicReferenceError("classic ruleset entry count exceeds limit")
        actual = _tree_hash(classic)
        # Keep this fail-closed.  A stale manifest must never be treated as a
        # close enough match: the extracted files are an input boundary and
        # accepting them would make the provenance advertised to the model
        # false.  Constant-time comparison is inexpensive and avoids making
        # the integrity check depend on ordinary string comparison semantics.
        if not hmac.compare_digest(actual, expected):
            raise ClassicReferenceError(
                "classic ruleset tree checksum verification failed: "
                f"reference.json records {expected}, but the extracted files "
                f"hash to {actual}. The manifest is stale or was generated by "
                "an incompatible legacy hash algorithm; re-run "
                "./scripts/extract_freeciv_reference.sh to regenerate the "
                "reference (do not edit the checksum by hand)."
            )
        return ClassicReference(tuple(entries), source_path=str(classic), image_ref=image_ref, image_version=str(manifest.get("image_version", "")), tree_sha256=actual)
    except ClassicReferenceError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ClassicReferenceError(f"classic reference unavailable: {type(exc).__name__}") from exc


def search_entries(reference: ClassicReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10) -> tuple[dict[str, Any], ...]:
    if not isinstance(query, str) or not query.strip() or limit <= 0:
        return ()
    terms = tuple(x for x in _WORDS.findall(query.casefold()) if x not in _STOP)
    allowed = {
        _CATEGORY_ALIASES.get(str(x).casefold(), "")
        for x in categories
        if isinstance(x, str) and _CATEGORY_ALIASES.get(x.casefold())
    } if categories is not None else None
    matches: list[tuple[int, int, int, str, dict[str, Any]]] = []
    for entry in reference.entries:
        if allowed is not None and entry.category.casefold() not in allowed:
            continue
        title_tokens = tuple(_WORDS.findall(" ".join((entry.key, entry.name, entry.rule_name)).casefold()))
        help_tokens = set(_WORDS.findall(entry.helptext.casefold()))
        if not terms or not all(
            any(_singular_plural_equal(term, candidate) for candidate in title_tokens)
            or term in help_tokens
            for term in terms
        ):
            continue
        exact = query.casefold().strip() in {entry.key.casefold(), entry.name.casefold(), entry.rule_name.casefold()}
        title_exact = sum(term in title_tokens for term in terms)
        title_alias = sum(
            any(_singular_plural_equal(term, candidate) for candidate in title_tokens if term != candidate)
            for term in terms
        )
        matches.append((0 if exact else 1, -title_alias, -title_exact, entry.name.casefold(), entry.to_dict()))
    matches.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]["key"], x[4]["source_file"]))
    return tuple(item[4] for item in matches[:limit])


def render_search(reference: ClassicReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10, max_chars: int = 8000) -> dict[str, Any]:
    result = {"schema_version": reference.schema_version, "authority": AUTHORITY, "status": reference.status, "source_path": reference.source_path, "image_ref": reference.image_ref, "image_version": reference.image_version, "tree_sha256": reference.tree_sha256, "content_hash": reference.content_hash, "query": query, "entries": [], "omitted": 0}
    used = 0
    for item in search_entries(reference, query, categories=categories, limit=limit):
        size = len(_canonical(item)) + 1
        if used + size > max_chars:
            result["omitted"] += 1
        else:
            result["entries"].append(item)
            used += size
    return result


__all__ = ["AUTHORITY", "PINNED_IMAGE_REF", "ClassicEntry", "ClassicReference", "ClassicReferenceError", "load_classic_reference", "parse_classic_ruleset", "search_entries", "render_search"]
