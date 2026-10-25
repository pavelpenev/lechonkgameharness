# SPDX-License-Identifier: MIT
"""lechonk.core.jsonutil — the single strict canonical-JSON helper (FW-09).

One canonical JSON semantics for transcript checksums (core.ledger) and
compact payload renderings (core.schemas Snapshot/Delta; later the
model-client hash paths):

- NaN and Infinity are REJECTED (allow_nan=False): json.dumps raises
  ValueError deterministically — bare NaN/Infinity tokens (invalid JSON)
  are never emitted.
- Lone surrogates (U+D800..U+DFFF) are ESCAPED as \\udXXX escape
  sequences: valid JSON with stable checksums, and a lone surrogate never
  reaches the utf-8 encoder (UnicodeEncodeError). Escaping each surrogate
  code unit individually also re-encodes a proper surrogate pair as the
  pair of escapes JSON uses for astral characters, so the escape form
  round-trips through json.loads.

For clean (non-NaN, surrogate-free) payloads the output is byte-identical
to the pre-FW-09 json.dumps forms, so recorded-artifact checksums still
verify.
"""

from __future__ import annotations

import json
import re
from typing import Any

_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def strict_dumps(obj: Any, *, sort_keys: bool = False) -> str:
    """The strict canonical-JSON rule (see the module docstring): reject
    NaN/Infinity, escape lone surrogates. ``sort_keys`` selects the
    key-ordered canonical form (checksums, transcript lines) vs the model's
    field order (compact payload renderings)."""
    text = json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str)
    if _SURROGATE_RE.search(text):
        text = _SURROGATE_RE.sub(
            lambda m: "\\u%04x" % ord(m.group()), text)
    return text


def canonical_json(payload: Any) -> str:
    """Canonical JSON for transcript lines and checksums: sorted keys,
    compact separators, UTF-8, non-ASCII preserved, strict NaN/surrogate
    semantics (see strict_dumps)."""
    return strict_dumps(payload, sort_keys=True)
