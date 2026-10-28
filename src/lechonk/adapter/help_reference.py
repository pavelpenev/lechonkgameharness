# SPDX-License-Identifier: GPL-3.0-or-later
"""Safe, bounded index for Freeciv's static ``helpdata.txt``.

Freeciv's file is an INI-like source file, not a data-only format: values are
wrapped in C gettext macros and string literals.  This adapter intentionally
parses only the documented fields and string literals.  It never imports,
executes, evaluates, or expands the source, and it does not expose generated
tables (which require live ruleset state).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REFERENCE_SCHEMA_VERSION = "freeciv_help_reference_v1"
REFERENCE_SOURCE = "Freeciv data/helpdata.txt (static help nodes)"
DEFAULT_MAX_TEXT_CHARS = 16_000
DEFAULT_RENDER_MAX_CHARS = 8_000
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SECTION_RE = re.compile(r"^\s*\[([^]]+)\]\s*$")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
_KNOWN_FIELDS = {"name", "text", "generate", "categories"}
_INLINE_TABLE_RE = re.compile(r"(?m)^\$([A-Za-z][A-Za-z0-9_]*)\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset({"a", "an", "and", "are", "can", "do", "does", "for", "how", "is", "of", "on", "the", "to", "what", "where", "which", "who", "why", "with"})
_DOMAIN_MAP = {
    "units": "unit", "unit": "unit", "improvements": "improvement", "improvement": "improvement", "buildings": "improvement", "building": "improvement", "wonders": "wonder", "wonder": "wonder",
    "techs": "tech", "terrain": "terrain", "extras": "extra", "goods": "goods",
    "specialists": "specialist", "governments": "government", "ruleset": "ruleset",
    "tileset": "tileset", "nations": "nation",
}


def _clean(value: Any, limit: int) -> str:
    value = "" if value is None else str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub(" ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value).strip()
    if limit < 1:
        return ""
    if len(value) > limit:
        return value[: max(0, limit - 1)].rstrip() + "…"
    return value


def _literal_strings(value: str) -> list[str]:
    """Decode only quoted literals; malformed literals are omitted."""
    # Freeciv uses a backslash immediately before physical newlines.
    value = re.sub(r"\\\r?\n", "", value)
    result: list[str] = []
    for token in _STRING_RE.findall(value):
        try:
            decoded = ast.literal_eval(token)
        except (SyntaxError, ValueError, UnicodeError):
            continue
        if isinstance(decoded, str):
            result.append(decoded)
    return result


def _value_strings(value: str) -> list[str]:
    return _literal_strings(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HelpEntry:
    """One static help node, with dynamic/generated content explicitly marked."""

    key: str
    name: str
    text: str = ""
    generate: str = ""
    categories: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    generated_tables: tuple[str, ...] = ()
    omitted: bool = False
    source_line: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"key": self.key, "name": self.name}
        if self.text:
            result["text"] = self.text
        if self.generate:
            result["generate"] = self.generate
            result["generated_content_unavailable"] = True
        if self.generated_tables:
            result["generated_tables"] = list(self.generated_tables)
            result["generated_content_unavailable"] = True
        if self.categories:
            result["categories"] = list(self.categories)
        if self.domains:
            result["domains"] = list(self.domains)
        if self.omitted:
            result["omitted"] = True
        return result


@dataclass(frozen=True)
class HelpReference:
    """Immutable static help index and provenance/status metadata."""

    entries: tuple[HelpEntry, ...] = ()
    status: str = "available"
    source: str = REFERENCE_SOURCE
    source_path: str = ""
    source_sha256: str = ""
    warnings: tuple[str, ...] = ()
    schema_version: str = REFERENCE_SCHEMA_VERSION
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda e: (e.name.casefold(), e.key)))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        # The content identity is semantic index content, not its local path or
        # source byte formatting (those remain separately exposed as provenance).
        identity = _canonical({"status": self.status, "entries": [entry.to_dict() for entry in ordered]})
        object.__setattr__(self, "_hash", hashlib.sha256(identity.encode()).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "status": self.status,
            "warnings": list(self.warnings),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if include_hash:
            result["content_hash"] = self.content_hash
        return result

    def canonical_json(self) -> str:
        return _canonical(self.to_dict(include_hash=False))


def _section_blocks(text: str) -> Iterable[tuple[str, int, dict[str, str]]]:
    current: tuple[str, int, dict[str, str]] | None = None
    field_name: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        section_match = _SECTION_RE.match(line)
        if section_match:
            if current is not None:
                yield current
            current = (section_match.group(1).strip(), line_number, {})
            field_name = None
            continue
        if current is None:
            continue
        assignment = _ASSIGN_RE.match(line)
        if assignment:
            field_name = assignment.group(1)
            current[2][field_name] = assignment.group(2)
        elif field_name and line.strip() and not line.lstrip().startswith((";", "#")):
            current[2][field_name] += "\n" + line
    if current is not None:
        yield current


def parse_helpdata(
    text: str,
    *,
    source_path: str | None = None,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> HelpReference:
    """Parse supported static ``[help_*]`` nodes from extracted source text."""
    if not isinstance(text, str) or not text:
        return HelpReference(status="unavailable", source_path=source_path or "", warnings=("helpdata is absent",))
    raw = text.encode("utf-8", "replace")
    entries: list[HelpEntry] = []
    warnings: list[str] = []
    saw_section = False
    for section, line_number, fields in _section_blocks(text):
        if not section.startswith("help_"):
            continue
        saw_section = True
        unknown = sorted(set(fields) - _KNOWN_FIELDS)
        if unknown:
            warnings.append(f"ignored unsupported fields in {section}")
        names = _value_strings(fields.get("name", ""))
        name = _clean(names[0] if names else section.removeprefix("help_"), max_text_chars)
        name = name.removeprefix("?help:").strip()
        text_parts = _value_strings(fields.get("text", ""))
        text_value = _clean("\n\n".join(text_parts), max_text_chars)
        generated_tables = tuple(sorted(set(_INLINE_TABLE_RE.findall(text_value))))
        generate_values = _value_strings(fields.get("generate", ""))
        generate = _clean(generate_values[0] if generate_values else "", max_text_chars)
        category_values = _value_strings(fields.get("categories", ""))
        categories = tuple(sorted({_clean(item, 128) for item in category_values if _clean(item, 128)}))
        domain_source = generate.strip().casefold() or section.removeprefix("help_gen_").casefold()
        domains = tuple(sorted({_DOMAIN_MAP[term] for term in re.findall(r"[a-z]+", domain_source) if term in _DOMAIN_MAP}))
        entries.append(HelpEntry(section, name or section, text_value, generate, categories, domains, generated_tables, len(text_value) >= max_text_chars and bool(text_parts), line_number))
    status = "available" if saw_section else "unsupported"
    if not saw_section:
        warnings.append("no supported [help_*] sections found")
    return HelpReference(
        entries=tuple(entries),
        status=status,
        source_path=source_path or "",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        warnings=tuple(warnings),
    )


def load_helpdata(path: str | Path | None, *, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> HelpReference:
    """Load an extracted file; missing/unreadable files become explicit status."""
    if path is None:
        return parse_helpdata("", max_text_chars=max_text_chars)
    try:
        file_path = Path(path)
        manifest_path = file_path.with_name("reference.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_sha = manifest.get("helpdata_sha256")
            actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if not isinstance(expected_sha, str) or expected_sha != actual_sha:
                return HelpReference(
                    status="unavailable",
                    source_path=str(file_path),
                    warnings=("helpdata failed reference.json checksum verification",),
                )
        return parse_helpdata(file_path.read_text(encoding="utf-8"), source_path=str(file_path), max_text_chars=max_text_chars)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        return HelpReference(status="unavailable", source_path=str(path), warnings=(f"helpdata unavailable: {type(exc).__name__}",))


def read_entry(reference: HelpReference, name_or_key: str) -> dict[str, Any] | None:
    """Read one exact key or case-insensitive display name."""
    if not isinstance(name_or_key, str):
        return None
    folded = name_or_key.casefold()
    for entry in reference.entries:
        if entry.key == name_or_key or entry.name.casefold() == folded:
            return entry.to_dict()
    return None


def search_entries(reference: HelpReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10) -> tuple[dict[str, Any], ...]:
    """Search names, keys, and static text with exact-title precedence.

    Queries are matched as either a literal substring or a conjunction of
    meaningful word terms.  This lets questions such as ``What is combat?``
    find ``Combat`` without broadening results to unrelated words.
    """
    if not isinstance(query, str) or not query.strip() or limit <= 0:
        return ()
    query = _clean(query, 512).casefold()
    terms = tuple(term for term in _WORD_RE.findall(query) if term not in _STOP_WORDS)
    matches: list[tuple[int, int, str, dict[str, Any]]] = []
    if not terms:
        return ()
    allowed = {_DOMAIN_MAP.get(str(category).casefold(), str(category).casefold()) for category in categories} if categories is not None else None
    for entry in reference.entries:
        if allowed is not None and not allowed.intersection(entry.domains):
            continue
        record = entry.to_dict()
        haystack = " ".join((entry.key, entry.name, entry.text)).casefold()
        words = set(_WORD_RE.findall(haystack))
        literal_match = query in haystack
        term_match = all(term in words or term in haystack for term in terms)
        if literal_match or term_match:
            exact = entry.name.casefold() == query or entry.key.casefold() == query
            # Exact title/key, then literal phrase, then token conjunction;
            # longer titles sort after otherwise equivalent shorter titles.
            matches.append((0 if exact else 1 if literal_match else 2, -len(terms), entry.name.casefold(), record))
    matches.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in matches[:limit])


def render_search(reference: HelpReference, query: str, *, categories: Sequence[str] | None = None, limit: int = 10, max_chars: int = DEFAULT_RENDER_MAX_CHARS) -> dict[str, Any]:
    """Render bounded results with provenance and explicit omission count."""
    entries: list[dict[str, Any]] = []
    omitted = 0
    used = 0
    for entry in search_entries(reference, query, categories=categories, limit=limit):
        encoded = _canonical(entry)
        if used + len(encoded) + 1 > max_chars:
            omitted += 1
            continue
        entries.append(entry)
        used += len(encoded) + 1
    return {"schema_version": reference.schema_version, "source": reference.source, "source_path": reference.source_path, "source_sha256": reference.source_sha256, "content_hash": reference.content_hash, "status": reference.status, "query": query, "entries": entries, "omitted": omitted}


__all__ = ["HelpEntry", "HelpReference", "parse_helpdata", "load_helpdata", "read_entry", "search_entries", "render_search"]
