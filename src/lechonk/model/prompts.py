# SPDX-License-Identifier: MIT
"""lechonk.model.prompts — versioned prompt templates (§7.5) + loader/hash.

Templates are FILES under ``model/prompts/`` (versioned by a ``_vN`` filename
suffix), never inline strings; the run manifest records their hashes
(decision 8). The template files themselves carry no license header line —
they are prompt content sent to the model; the MIT licensing of this package
(including the templates) is declared here and in the package SPDX headers.

Deterministic prompt hash: ``sha256:<hex>`` over the UTF-8 file bytes.

Structure (§7.5):
- system prompt: role, interface rules (candidate IDs only + budgets +
  commitment grammar), fog disclaimer, action-surface cheat sheet;
- turn prompt: [stable prefix: rules + cheat sheet + output contract] +
  memory (≤1,500) + snapshot (≤8,000) + delta (≤800) + candidate table
  summary → output: ActionBatch — the stable prefix enables prompt caching;
- three repair prompt variants (schema-error / state-fog-error /
  illegal-in-rule), rendered with ``string.Template`` placeholders;
- the fixed fingerprint probe prompt (decision 8).

``memory_update`` is NOT a separate API call: the per-turn ActionBatch output
carries it (one request per turn total, decision 4).
"""

from __future__ import annotations

import hashlib
import string
from pathlib import Path

from lechonk.core.schemas import (
    DELTA_TOKEN_BUDGET,
    SNAPSHOT_TOKEN_BUDGET,
    estimate_tokens,
)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Token budgets (§3) — the prompt text annotates them; the assembler enforces
# them on the variable sections as defense-in-depth.
MEMORY_TOKEN_BUDGET = 1500

SYSTEM_PROMPT = "system_v3"  # FW-10: v3 documents the typed commitment
# grammar (target_kind + verb vocabulary); v1/v2 retained on disk for history
TURN_PREFIX = "turn_prefix_v2"
REPAIR_SCHEMA_ERROR = "repair_schema_error_v1"
REPAIR_STATE_FOG_ERROR = "repair_state_fog_error_v1"
REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_v1"
FINGERPRINT_PROBE = "fingerprint_probe_v1"

ALL_PROMPTS: tuple[str, ...] = (
    SYSTEM_PROMPT,
    TURN_PREFIX,
    REPAIR_SCHEMA_ERROR,
    REPAIR_STATE_FOG_ERROR,
    REPAIR_ILLEGAL_IN_RULE,
    FINGERPRINT_PROBE,
)

REPAIR_TEMPLATES: dict[str, str] = {
    "schema_error": REPAIR_SCHEMA_ERROR,
    "state_fog_error": REPAIR_STATE_FOG_ERROR,
    "illegal_in_rule": REPAIR_ILLEGAL_IN_RULE,
}


class PromptError(RuntimeError):
    """Base for prompt-template errors."""


class PromptNotFoundError(PromptError):
    """A named template file does not exist under model/prompts/."""


class PromptRenderError(PromptError):
    """A template could not be rendered (missing placeholder field)."""


class PromptBudgetExceeded(PromptError):
    """A variable section exceeds its §3 token budget."""


def prompt_path(name: str) -> Path:
    return PROMPT_DIR / f"{name}.txt"


def load_prompt(name: str) -> str:
    """Load a versioned template file (exact name, e.g. ``system_v2``)."""
    path = prompt_path(name)
    if not path.is_file():
        raise PromptNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    """Deterministic prompt hash: ``sha256:<hex>`` over the UTF-8 file bytes."""
    return "sha256:" + hashlib.sha256(
        prompt_path(name).read_bytes()).hexdigest()


def all_prompt_hashes() -> dict[str, str]:
    """Every template file present under model/prompts/ → its hash."""
    return {
        path.stem: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PROMPT_DIR.glob("*.txt"))
    }


def prompt_manifest_hashes() -> dict[str, str]:
    """The prompt-hash block for the run manifest (decision 8)."""
    hashes = all_prompt_hashes()
    missing = [n for n in ALL_PROMPTS if n not in hashes]
    if missing:
        raise PromptNotFoundError(f"declared templates missing: {missing}")
    return hashes


def render_template(name: str, **fields: str) -> str:
    """Render a template file with ``string.Template`` placeholders.

    Raises PromptRenderError when the template references a placeholder the
    caller did not supply (JSON braces in the templates are literal text and
    never treated as placeholders).
    """
    text = load_prompt(name)
    template = string.Template(text)
    missing = [i for i in template.get_identifiers() if i not in fields]
    if missing:
        raise PromptRenderError(
            f"template {name}: missing fields {sorted(missing)}")
    return template.substitute(fields)


def system_prompt() -> str:
    return load_prompt(SYSTEM_PROMPT)


def fingerprint_probe_prompt() -> str:
    """The fixed fingerprint probe prompt (decision 8)."""
    return load_prompt(FINGERPRINT_PROBE)


# ---------------------------------------------------------------------------
# Per-turn prompt assembly (§7.5)
# ---------------------------------------------------------------------------

_SECTIONS: tuple[tuple[str, int], ...] = (
    ("MEMORY", MEMORY_TOKEN_BUDGET),
    ("SNAPSHOT", SNAPSHOT_TOKEN_BUDGET),
    ("DELTA", DELTA_TOKEN_BUDGET),
    ("CANDIDATE TABLE", 0),  # the candidate table has no §3 token budget
)


def turn_prompt(
    *,
    memory: str,
    snapshot: str,
    delta: str,
    candidate_summary: str,
    enforce_budgets: bool = True,
) -> str:
    """The per-turn user message (§7.5): [stable prefix: rules + cheat sheet]
    + memory (≤1,500) + snapshot (≤8,000) + delta (≤800) + candidate table
    summary → output: ActionBatch. The stable prefix is byte-identical across
    turns, enabling prompt caching (chonk-doom lesson, plan §2).

    With ``enforce_budgets`` (default) the variable sections are checked
    against their §3 budgets with the deterministic token estimate — the
    extractor truncates upstream; this is defense-in-depth.
    """
    bodies = {
        "MEMORY": memory,
        "SNAPSHOT": snapshot,
        "DELTA": delta,
        "CANDIDATE TABLE": candidate_summary,
    }
    parts = [load_prompt(TURN_PREFIX)]
    for header, budget in _SECTIONS:
        body = bodies[header].strip()
        if enforce_budgets and budget and estimate_tokens(body) > budget:
            raise PromptBudgetExceeded(
                f"section {header} exceeds its budget: "
                f"{estimate_tokens(body)} > {budget} tokens")
        annotation = f" (≤ {budget:,} tokens)" if budget else ""
        parts.append(f"## {header}{annotation}\n{body}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Repair prompts (attempt 2, per error class — §7.5)
# ---------------------------------------------------------------------------


def repair_prompt(
    error_class: str,
    *,
    detail: str,
    failed_refs: str,
    refreshed_candidates: str,
) -> str:
    """Render the repair instruction for one illegal-action-taxonomy class."""
    template_name = REPAIR_TEMPLATES.get(error_class)
    if template_name is None:
        raise PromptError(
            f"unknown repair error class {error_class!r}; "
            f"expected one of {sorted(REPAIR_TEMPLATES)}")
    return render_template(
        template_name,
        error_detail=detail,
        failed_refs=failed_refs,
        refreshed_candidates=refreshed_candidates,
    )


def repair_messages(
    original_messages: list[dict[str, str]],
    raw_content: str,
    error_class: str,
    *,
    detail: str,
    failed_refs: str,
    refreshed_candidates: str,
) -> list[dict[str, str]]:
    """Repair-attempt messages (§7.5): the original conversation + the failed
    assistant output + the class-specific repair instruction. Carrying the
    original messages preserves context and prompt caching."""
    return list(original_messages) + [
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": repair_prompt(
                error_class,
                detail=detail,
                failed_refs=failed_refs,
                refreshed_candidates=refreshed_candidates,
            ),
        },
    ]
