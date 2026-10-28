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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from pathlib import Path

from lechonk.core.model_client import PromptBudgetExceeded
from lechonk.core.schemas import (
    CANDIDATE_TABLE_TOKEN_BUDGET,
    DELTA_TOKEN_BUDGET,
    MEMORY_TOKEN_BUDGET,
    SNAPSHOT_TOKEN_BUDGET,
    estimate_tokens,
)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptBundle:
    """Immutable prompt bytes selected by a recovery lineage.

    ``logical_profile`` remains the historical profile name used in cache and
    digest keys; ``bundle_id`` selects the frozen files without mutating the
    process-global profile registry.
    """
    bundle_id: str
    logical_profile: str
    directory: Path | None = None

    _FILES = {
        "system": "system.txt",
        "turn": "turn_prefix.txt",
        "schema_error": "repair_schema.txt",
        "state_fog_error": "repair_state_fog.txt",
        "illegal_in_rule": "repair_illegal_in_rule.txt",
    }

    def path_for(self, logical_name: str) -> Path:
        if self.directory is None:
            return prompt_path(logical_name)
        # Frozen bundle files use short stable names; map the logical profile
        # names to those files without importing or executing bundle content.
        if logical_name in self._FILES:
            return self.directory / self._FILES[logical_name]
        if logical_name.startswith("system_"):
            return self.directory / "system.txt"
        if logical_name.startswith("turn_prefix_"):
            return self.directory / "turn_prefix.txt"
        if logical_name.startswith("repair_schema_"):
            return self.directory / "repair_schema.txt"
        if logical_name.startswith("repair_state_fog_"):
            return self.directory / "repair_state_fog.txt"
        if logical_name.startswith("repair_illegal_in_rule_"):
            return self.directory / "repair_illegal_in_rule.txt"
        if logical_name == FINGERPRINT_PROBE:
            return prompt_path(logical_name)
        return self.directory / f"{logical_name}.txt"


@dataclass(frozen=True)
class PromptSnapshot:
    """Immutable bytes/text and hashes for one selected prompt bundle."""
    profile: str
    bundle_id: str | None
    texts: Mapping[str, str]
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "texts", MappingProxyType(dict(self.texts)))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))

    @property
    def manifest_hashes(self) -> dict[str, str]:
        spec = PROMPT_PROFILES[self.profile]
        names = [spec["system"], spec["turn"], *spec["repairs"].values(),
                 FINGERPRINT_PROBE]
        return {name: self.hashes[name] for name in names}


def capture_prompt_snapshot(profile: str = "legacy_eval", *,
                            bundle_id: str | None = None) -> PromptSnapshot:
    """Read the selected prompt files atomically enough for startup.

    Reading the complete byte map twice prevents capturing a half-written
    bundle.  The returned text and hashes are then used for the lifetime of a
    client, so later edits cannot alter a running conversation.
    """
    profile = normalize_prompt_profile(profile)
    if bundle_id is not None:
        bundle = prompt_bundle(bundle_id, profile=profile)
    spec = PROMPT_PROFILES[profile]
    names = [spec["system"], spec["turn"], HYBRID_TURN_PREFIX,
             *spec["repairs"].values(), FINGERPRINT_PROBE]
    def read_map() -> dict[str, bytes]:
        result = {}
        for name in names:
            # The hybrid prefix is a global compatibility template.  It is
            # intentionally shared across frozen gameplay bundles.
            path = (prompt_path(name) if name == HYBRID_TURN_PREFIX else
                    (bundle.path_for(name) if bundle_id is not None
                     else prompt_path(name)))
            try:
                result[name] = path.read_bytes()
            except FileNotFoundError as exc:
                raise PromptNotFoundError(f"prompt template not found: {path}") from exc
        return result
    first = read_map()
    second = read_map()
    if first != second:
        raise PromptConfigError("prompt bundle changed while being captured")
    texts = {name: data.decode("utf-8") for name, data in first.items()}
    hashes = {name: "sha256:" + hashlib.sha256(data).hexdigest()
              for name, data in first.items()}
    # The fingerprint probe is global but is part of recovery identity.
    return PromptSnapshot(profile, bundle_id, texts, hashes)


_BUNDLE_REGISTRY = {
    "gameplay_v2_original": PromptBundle(
        "gameplay_v2_original", "gameplay_v2",
        PROMPT_DIR / "gameplay_v2_original"),
    "gameplay_v2_turnid_fix": PromptBundle(
        "gameplay_v2_turnid_fix", "gameplay_v2",
        PROMPT_DIR / "gameplay_v2_turnid_fix"),
    "gameplay_v3": PromptBundle(
        "gameplay_v3", "gameplay_v3", PROMPT_DIR / "gameplay_v3"),
    "gameplay_v4": PromptBundle(
        "gameplay_v4", "gameplay_v4", PROMPT_DIR / "gameplay_v4"),
    "gameplay_v5": PromptBundle(
        "gameplay_v5", "gameplay_v5", PROMPT_DIR / "gameplay_v5"),
    "gameplay_v6": PromptBundle(
        "gameplay_v6", "gameplay_v6", PROMPT_DIR / "gameplay_v6"),
}


def prompt_bundle(bundle_id: str | None = None,
                  *, profile: str = "legacy_eval") -> PromptBundle:
    if bundle_id is None:
        return PromptBundle("active", normalize_prompt_profile(profile), None)
    try:
        bundle = _BUNDLE_REGISTRY[bundle_id]
    except KeyError as exc:
        raise PromptError(f"unknown prompt bundle {bundle_id!r}") from exc
    if bundle.logical_profile != normalize_prompt_profile(profile):
        raise PromptError(
            f"prompt bundle {bundle_id!r} does not match profile {profile!r}")
    for logical_name in ("system", "turn", "schema_error",
                         "state_fog_error", "illegal_in_rule"):
        if not bundle.path_for(logical_name).is_file():
            raise PromptNotFoundError(
                f"prompt bundle file not found: {bundle.path_for(logical_name)}")
    return bundle

# W1-7: the former single ASSEMBLED_PROMPT_TOKEN_BUDGET guarded TWO
# different sums — the initial assembled check sums ONLY the four variable
# section bodies, while the repair check
# (turn_client._check_repair_conversation_budget) sums the COMPLETE repair
# conversation (system prompt + full turn prompt + failed assistant output
# + repair instruction). Same value for both meant an initial prompt that
# passed could fail the repair check purely on the fixed text. The two
# checks now each name the budget they enforce; both keep the pre-split
# value, so no live behavior changes.

# The initial-assembled-check budget (§3): the fully assembled turn prompt's
# variable section bodies may not jointly exceed the sum of the per-section
# budgets — each section is capped individually by _SECTIONS, and the
# assembled check in turn_prompt guards the total:
#   MEMORY_TOKEN_BUDGET + SNAPSHOT_TOKEN_BUDGET + DELTA_TOKEN_BUDGET
#   + CANDIDATE_TABLE_TOKEN_BUDGET  (= 1,500 + 8,000 + 800 + 1,200 = 11,500).
ASSEMBLED_SECTION_BODY_TOKEN_BUDGET = (
    MEMORY_TOKEN_BUDGET + SNAPSHOT_TOKEN_BUDGET
    + DELTA_TOKEN_BUDGET + CANDIDATE_TABLE_TOKEN_BUDGET
)

# The repair-conversation budget (§3, FW-13): bounds the COMPLETE repair
# conversation — system prompt + full turn prompt (fixed prefix + the four
# section bodies) + the failed assistant output + the repair instruction —
# a strictly larger sum than the section-body budget above covers. Current
# value is the pre-split constant's value (11,500), so the repair check
# behaves exactly as before; the value is independent of the section sum
# and may diverge from it.
REPAIR_CONVERSATION_TOKEN_BUDGET = 11_500

# gameplay_v3 exposes the versioned candidate_surface_v2 admission policy.
# Its 8,000-token candidate surface needs a larger assembled-body and
# non-conversation repair allowance; the legacy values above remain frozen.
GAMEPLAY_V3_CANDIDATE_TABLE_TOKEN_BUDGET = 8_000
GAMEPLAY_V3_ASSEMBLED_SECTION_BODY_TOKEN_BUDGET = (
    MEMORY_TOKEN_BUDGET + SNAPSHOT_TOKEN_BUDGET + DELTA_TOKEN_BUDGET
    + GAMEPLAY_V3_CANDIDATE_TABLE_TOKEN_BUDGET)
# This covers the 18,300-token variable body, fixed prompt text, the bounded
# failed response excerpt, and the repair instruction with headroom. It is a
# prompt-construction guard, never an API output-token setting.
GAMEPLAY_V3_REPAIR_CONVERSATION_TOKEN_BUDGET = 24_000

# Hybrid history is deliberately a separate policy budget. The legacy
# section-body and repair budgets remain unchanged for evaluation prompts.
HISTORY_TOKEN_BUDGET = 4_000
HYBRID_REPAIR_CONVERSATION_TOKEN_BUDGET = (
    REPAIR_CONVERSATION_TOKEN_BUDGET + HISTORY_TOKEN_BUDGET)
GAMEPLAY_V3_HYBRID_REPAIR_CONVERSATION_TOKEN_BUDGET = (
    GAMEPLAY_V3_REPAIR_CONVERSATION_TOKEN_BUDGET + HISTORY_TOKEN_BUDGET)

# Token budgets (§3) — the prompt text annotates them; the assembler enforces
# them on the variable sections as defense-in-depth. MEMORY_TOKEN_BUDGET,
# SNAPSHOT_TOKEN_BUDGET, DELTA_TOKEN_BUDGET and CANDIDATE_TABLE_TOKEN_BUDGET
# are the single definitions in core.schemas — imported, never redefined.

SYSTEM_PROMPT = "system_v3"  # FW-10: v3 documents the typed commitment
# grammar (target_kind + verb vocabulary); v1/v2 retained on disk for history
TURN_PREFIX = "turn_prefix_v2"
HYBRID_TURN_PREFIX = "turn_prefix_hybrid_v1"
REPAIR_SCHEMA_ERROR = "repair_schema_error_v1"
REPAIR_STATE_FOG_ERROR = "repair_state_fog_error_v1"
REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_v1"
GAMEPLAY_SYSTEM_PROMPT = "system_gameplay_v1"
GAMEPLAY_TURN_PREFIX = "turn_prefix_gameplay_v1"
GAMEPLAY_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v1"
GAMEPLAY_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v1"
GAMEPLAY_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v1"
# gameplay_v2 is the opt-in gameplay profile with the bundled, read-only
# Freeciv encyclopedia exposed as an optional lookup tool.  Keep v1 frozen:
# its names and bytes are part of legacy run/replay identity.
GAMEPLAY_V2_SYSTEM_PROMPT = "system_gameplay_v2"
GAMEPLAY_V2_TURN_PREFIX = "turn_prefix_gameplay_v2"
GAMEPLAY_V2_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v2"
GAMEPLAY_V2_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v2"
GAMEPLAY_V2_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v2"
GAMEPLAY_V3_SYSTEM_PROMPT = "system_gameplay_v3"
GAMEPLAY_V3_TURN_PREFIX = "turn_prefix_gameplay_v3"
GAMEPLAY_V3_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v3"
GAMEPLAY_V3_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v3"
GAMEPLAY_V3_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v3"
GAMEPLAY_V4_SYSTEM_PROMPT = "system_gameplay_v4"
GAMEPLAY_V4_TURN_PREFIX = "turn_prefix_gameplay_v4"
GAMEPLAY_V4_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v4"
GAMEPLAY_V4_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v4"
GAMEPLAY_V4_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v4"
GAMEPLAY_V5_SYSTEM_PROMPT = "system_gameplay_v5"
GAMEPLAY_V5_TURN_PREFIX = "turn_prefix_gameplay_v5"
GAMEPLAY_V5_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v5"
GAMEPLAY_V5_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v5"
GAMEPLAY_V5_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v5"
GAMEPLAY_V6_SYSTEM_PROMPT = "system_gameplay_v6"
GAMEPLAY_V6_TURN_PREFIX = "turn_prefix_gameplay_v6"
GAMEPLAY_V6_REPAIR_SCHEMA_ERROR = "repair_schema_gameplay_v6"
GAMEPLAY_V6_REPAIR_STATE_FOG_ERROR = "repair_state_fog_gameplay_v6"
GAMEPLAY_V6_REPAIR_ILLEGAL_IN_RULE = "repair_illegal_in_rule_gameplay_v6"
# FW-21: v2 (capacity-revealing) is the live probe prompt — the fingerprint
# client (mistral_client.FINGERPRINT_PROBE_PROMPT) loads it directly; the v1
# verbatim-echo prompt is retained on disk for history
# The probe is a global immutable dependency of every profile, rather than a
# bundle-local file: keeping it outside gameplay_v3/v4/v5 bundles preserves the
# established v3 hashes and makes provider-capability identity comparable
# across paired gameplay experiments.
FINGERPRINT_PROBE = "fingerprint_probe_v2"

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

# Profiles are deliberately explicit.  The legacy profile is the default and
# its active file set is frozen so adding a new profile cannot silently alter
# manifests or recovery checks for old runs.
PROMPT_PROFILES: dict[str, dict[str, object]] = {
    "legacy_eval": {
        "system": SYSTEM_PROMPT,
        "turn": TURN_PREFIX,
        "repairs": dict(REPAIR_TEMPLATES),
    },
    "gameplay_v1": {
        "system": GAMEPLAY_SYSTEM_PROMPT,
        "turn": GAMEPLAY_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_REPAIR_ILLEGAL_IN_RULE,
        },
    },
    "gameplay_v2": {
        "system": GAMEPLAY_V2_SYSTEM_PROMPT,
        "turn": GAMEPLAY_V2_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_V2_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_V2_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_V2_REPAIR_ILLEGAL_IN_RULE,
        },
    },
    "gameplay_v3": {
        "system": GAMEPLAY_V3_SYSTEM_PROMPT,
        "turn": GAMEPLAY_V3_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_V3_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_V3_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_V3_REPAIR_ILLEGAL_IN_RULE,
        },
    },
    "gameplay_v4": {
        "system": GAMEPLAY_V4_SYSTEM_PROMPT,
        "turn": GAMEPLAY_V4_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_V4_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_V4_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_V4_REPAIR_ILLEGAL_IN_RULE,
        },
    },
    "gameplay_v5": {
        "system": GAMEPLAY_V5_SYSTEM_PROMPT,
        "turn": GAMEPLAY_V5_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_V5_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_V5_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_V5_REPAIR_ILLEGAL_IN_RULE,
        },
    },
    "gameplay_v6": {
        "system": GAMEPLAY_V6_SYSTEM_PROMPT,
        "turn": GAMEPLAY_V6_TURN_PREFIX,
        "repairs": {
            "schema_error": GAMEPLAY_V6_REPAIR_SCHEMA_ERROR,
            "state_fog_error": GAMEPLAY_V6_REPAIR_STATE_FOG_ERROR,
            "illegal_in_rule": GAMEPLAY_V6_REPAIR_ILLEGAL_IN_RULE,
        },
    },
}


class PromptError(RuntimeError):
    """Base for prompt-template errors."""


class PromptNotFoundError(PromptError):
    """A named template file does not exist under model/prompts/."""


class PromptRenderError(PromptError):
    """A template could not be rendered (missing placeholder fields)."""


class PromptConfigError(PromptError):
    """Startup configuration error: the fixed prompt text alone exceeds the
    repair-conversation budget, so no repair conversation could ever fit.
    Raised once at client initialization (fail fast, not a mid-game kill)."""


def validate_conversation_system_message(messages: list[dict[str, object]],
                                         *, profile: str,
                                         bundle_id: str | None = None,
                                         snapshot: PromptSnapshot | None = None) -> None:
    """Reject persisted conversation state from a different prompt bundle."""
    expected = system_prompt(profile, bundle_id=bundle_id, snapshot=snapshot)
    if (not messages or messages[0].get("role") != "system"
            or messages[0].get("content") != expected):
        raise PromptError(
            "conversation system prompt does not match selected prompt bundle")


# PromptBudgetExceeded is canonical in lechonk.core.model_client so the
# runner can catch it without importing the model package; the raise sites
# are the prompt assembler here, which re-exports the same class.


def prompt_path(name: str) -> Path:
    return PROMPT_DIR / f"{name}.txt"


def load_prompt(name: str, *, bundle_id: str | None = None,
                profile: str = "legacy_eval") -> str:
    """Load a versioned template file (exact name, e.g. ``system_v2``)."""
    path = (prompt_bundle(bundle_id, profile=profile).path_for(name)
            if bundle_id is not None else prompt_path(name))
    if not path.is_file():
        raise PromptNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def prompt_hash(name: str, *, bundle_id: str | None = None,
                profile: str = "legacy_eval") -> str:
    """Deterministic prompt hash: ``sha256:<hex>`` over the UTF-8 file bytes."""
    path = (prompt_bundle(bundle_id, profile=profile).path_for(name)
            if bundle_id is not None else prompt_path(name))
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def all_prompt_hashes() -> dict[str, str]:
    """Every template file present under model/prompts/ → its hash."""
    return {
        path.stem: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PROMPT_DIR.glob("*.txt"))
    }


def normalize_prompt_profile(profile: str = "legacy_eval") -> str:
    if profile not in PROMPT_PROFILES:
        raise PromptError(f"unknown prompt profile {profile!r}; expected one of "
                          f"{sorted(PROMPT_PROFILES)}")
    return profile


def prompt_manifest_hashes(profile: str = "legacy_eval", *,
                           bundle_id: str | None = None,
                           snapshot: PromptSnapshot | None = None) -> dict[str, str]:
    """The prompt-hash block for the run manifest (decision 8)."""
    profile = normalize_prompt_profile(profile)
    spec = PROMPT_PROFILES[profile]
    names = [spec["system"], spec["turn"], *spec["repairs"].values(),
             FINGERPRINT_PROBE]
    if snapshot is not None:
        if snapshot.profile != profile or snapshot.bundle_id != bundle_id:
            raise PromptError("prompt snapshot does not match selected bundle")
        hashes = {name: snapshot.hashes[name] for name in names}
    else:
        hashes = {name: prompt_hash(name, bundle_id=bundle_id, profile=profile)
                  for name in names}
    missing = [n for n in names if n not in hashes]
    if missing:
        raise PromptNotFoundError(f"declared templates missing: {missing}")
    return hashes


def active_prompt_hashes(profile: str = "legacy_eval", *,
                         bundle_id: str | None = None,
                         snapshot: PromptSnapshot | None = None) -> dict[str, str]:
    """Alias naming the profile-scoped hash API used by interactive runs."""
    return prompt_manifest_hashes(profile, bundle_id=bundle_id, snapshot=snapshot)


def render_template(name: str, *, bundle_id: str | None = None,
                    profile: str = "legacy_eval",
                    snapshot: PromptSnapshot | None = None,
                    **fields: str) -> str:
    """Render a template file with ``string.Template`` placeholders.

    Raises PromptRenderError when the template references a placeholder the
    caller did not supply (JSON braces in the templates are literal text and
    never treated as placeholders).
    """
    if snapshot is not None:
        if snapshot.profile != normalize_prompt_profile(profile) or snapshot.bundle_id != bundle_id:
            raise PromptError("prompt snapshot does not match selected bundle")
        text = snapshot.texts.get(name)
        if text is None:
            raise PromptNotFoundError(f"prompt template not captured: {name}")
    else:
        text = load_prompt(name, bundle_id=bundle_id, profile=profile)
    template = string.Template(text)
    missing = [i for i in template.get_identifiers() if i not in fields]
    if missing:
        raise PromptRenderError(
            f"template {name}: missing fields {sorted(missing)}")
    return template.substitute(fields)


def system_prompt(profile: str = "legacy_eval", *,
                  bundle_id: str | None = None,
                  snapshot: PromptSnapshot | None = None) -> str:
    spec = PROMPT_PROFILES[normalize_prompt_profile(profile)]
    if snapshot is not None:
        if snapshot.profile != normalize_prompt_profile(profile) or snapshot.bundle_id != bundle_id:
            raise PromptError("prompt snapshot does not match selected bundle")
        return snapshot.texts[spec["system"]]
    return load_prompt(spec["system"], bundle_id=bundle_id, profile=profile)


def check_fixed_prompt_budget(profile: str = "legacy_eval", *,
                              bundle_id: str | None = None,
                              snapshot: PromptSnapshot | None = None) -> int:
    """FW-13 startup invariant: the fixed (turn-invariant) prompt text — the
    system prompt plus the stable turn prefix — must fit the repair-
    conversation budget on its own. The fixed text appears in every
    conversation, and the repair-conversation budget is the only check whose
    sum includes it (the initial assembled check sums only the four variable
    section bodies) — so if the fixed text alone exceeds that budget, no
    repair conversation could ever fit. Meant to be computed once at client
    initialization (fail fast at startup, not as a mid-game kill). Returns
    the fixed-text token cost.

    The check is a cheap invariant: the fixed templates are trivially small
    today, and this pins that against future template growth.
    """
    profile = normalize_prompt_profile(profile)
    spec = PROMPT_PROFILES[profile]
    cost = (estimate_tokens(system_prompt(profile, bundle_id=bundle_id,
                                          snapshot=snapshot))
            + estimate_tokens(load_prompt(HYBRID_TURN_PREFIX)
                              if snapshot is None else snapshot.texts[HYBRID_TURN_PREFIX])
            + estimate_tokens(load_prompt(spec["turn"], bundle_id=bundle_id,
                                          profile=profile)
                              if snapshot is None else snapshot.texts[spec["turn"]]))
    if cost > repair_conversation_budget(profile):
        raise PromptConfigError(
            f"fixed prompt text exceeds the repair-conversation budget: "
            f"{cost} > {repair_conversation_budget(profile)} tokens")
    return cost


def candidate_table_budget(profile: str = "legacy_eval") -> int:
    """Return the profile's model-facing candidate-table budget."""
    if normalize_prompt_profile(profile) == "gameplay_v6":
        return 12_000
    return (GAMEPLAY_V3_CANDIDATE_TABLE_TOKEN_BUDGET
            if normalize_prompt_profile(profile) in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
            else CANDIDATE_TABLE_TOKEN_BUDGET)


def assembled_section_body_budget(profile: str = "legacy_eval",
                                  *, hybrid: bool = False) -> int:
    """Return the profile-aware variable-section budget."""
    profile = normalize_prompt_profile(profile)
    base = ((MEMORY_TOKEN_BUDGET + SNAPSHOT_TOKEN_BUDGET + DELTA_TOKEN_BUDGET + 12000)
            if profile == "gameplay_v6" else GAMEPLAY_V3_ASSEMBLED_SECTION_BODY_TOKEN_BUDGET
            if profile in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
            else ASSEMBLED_SECTION_BODY_TOKEN_BUDGET)
    return base + (HISTORY_TOKEN_BUDGET if hybrid else 0)


def repair_conversation_budget(profile: str = "legacy_eval",
                               *, hybrid: bool = False,
                               context_policy: str | None = None) -> int:
    """Return the complete repair-conversation construction budget."""
    profile = normalize_prompt_profile(profile)
    if (profile in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"} and context_policy is not None
            and context_policy != "conversation_v1"):
        raise PromptConfigError(
            f"{profile} requires context_policy=conversation_v1; "
            "its expanded repair budget is not valid for non-conversation "
            "requests")
    base = (GAMEPLAY_V3_REPAIR_CONVERSATION_TOKEN_BUDGET
            if profile in {"gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"}
            else REPAIR_CONVERSATION_TOKEN_BUDGET)
    if hybrid:
        base += HISTORY_TOKEN_BUDGET
    return base


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
    ("CANDIDATE TABLE", CANDIDATE_TABLE_TOKEN_BUDGET),
)


def turn_prompt(
    *,
    memory: str,
    snapshot: str,
    delta: str,
    candidate_summary: str,
    history: str = "",
    enforce_budgets: bool = True,
    profile: str = "legacy_eval",
    bundle_id: str | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
) -> str:
    """The per-turn user message (§7.5): [stable prefix: rules + cheat sheet]
    + memory (≤1,500) + snapshot (≤8,000) + delta (≤800) + candidate table
    summary → output: ActionBatch. The stable prefix is byte-identical across
    turns, enabling prompt caching (chonk-doom lesson, plan §2).

    With ``enforce_budgets`` (default) the variable sections are checked
    against their §3 budgets with the deterministic token estimate — the
    extractor truncates upstream; this is defense-in-depth.
    """
    spec = PROMPT_PROFILES[normalize_prompt_profile(profile)]
    if prompt_snapshot is not None and (
            prompt_snapshot.profile != normalize_prompt_profile(profile)
            or prompt_snapshot.bundle_id != bundle_id):
        raise PromptError("prompt snapshot does not match selected bundle")
    bodies = {
        "MEMORY": memory,
        "HISTORY": history,
        "SNAPSHOT": snapshot,
        "DELTA": delta,
        "CANDIDATE TABLE": candidate_summary,
    }
    hybrid = bool(history.strip())
    # Hybrid history remains a legacy compatibility feature.  Gameplay v1
    # uses the conversation profile's stable prefix directly.
    turn_text = (prompt_snapshot.texts[spec["turn"]]
                 if prompt_snapshot is not None else
                 load_prompt(spec["turn"], bundle_id=bundle_id,
                             profile=profile))
    hybrid_text = (prompt_snapshot.texts[HYBRID_TURN_PREFIX]
                   if prompt_snapshot is not None else load_prompt(HYBRID_TURN_PREFIX))
    parts = ([hybrid_text, turn_text]
             if hybrid else [turn_text])
    candidate_budget = candidate_table_budget(profile)
    sections = (("MEMORY", MEMORY_TOKEN_BUDGET),
                ("HISTORY", HISTORY_TOKEN_BUDGET),
                ("SNAPSHOT", SNAPSHOT_TOKEN_BUDGET),
                ("DELTA", DELTA_TOKEN_BUDGET),
                ("CANDIDATE TABLE", candidate_budget)) \
        if hybrid else (("MEMORY", MEMORY_TOKEN_BUDGET),
                        ("SNAPSHOT", SNAPSHOT_TOKEN_BUDGET),
                        ("DELTA", DELTA_TOKEN_BUDGET),
                        ("CANDIDATE TABLE", candidate_budget))
    for header, budget in sections:
        body = bodies[header].strip()
        if enforce_budgets and budget and estimate_tokens(body) > budget:
            raise PromptBudgetExceeded(
                f"section {header} exceeds its budget: "
                f"{estimate_tokens(body)} > {budget} tokens")
        annotation = f" (≤ {budget:,} tokens)" if budget else ""
        parts.append(f"## {header}{annotation}\n{body}")
    assembled = "\n\n".join(parts)
    if enforce_budgets:
        # Initial assembled check: the sum of the four variable section
        # BODIES ONLY — the fixed prefix text is deliberately not part of
        # this sum; the repair-conversation budget bounds it elsewhere.
        total = sum(estimate_tokens(bodies[h].strip()) for h, _ in sections)
        body_budget = assembled_section_body_budget(profile, hybrid=hybrid)
        if total > body_budget:
            raise PromptBudgetExceeded(
                f"assembled turn prompt exceeds the section-body budget: "
                f"{total} > {body_budget} tokens")
    return assembled


# ---------------------------------------------------------------------------
# Repair prompts (attempt 2, per error class — §7.5)
# ---------------------------------------------------------------------------


def repair_prompt(
    error_class: str,
    *,
    detail: str,
    failed_refs: str,
    refreshed_candidates: str,
    profile: str = "legacy_eval",
    bundle_id: str | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
) -> str:
    """Render the repair instruction for one illegal-action-taxonomy class."""
    spec = PROMPT_PROFILES[normalize_prompt_profile(profile)]
    template_name = spec["repairs"].get(error_class)
    if template_name is None:
        raise PromptError(
            f"unknown repair error class {error_class!r}; "
            f"expected one of {sorted(REPAIR_TEMPLATES)}")
    return render_template(
        template_name,
        error_detail=detail,
        failed_refs=failed_refs,
        refreshed_candidates=refreshed_candidates,
        bundle_id=bundle_id,
        profile=profile,
        snapshot=prompt_snapshot,
    )


def repair_messages(
    original_messages: list[dict[str, str]],
    raw_content: str,
    error_class: str,
    *,
    detail: str,
    failed_refs: str,
    refreshed_candidates: str,
    profile: str = "legacy_eval",
    bundle_id: str | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
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
                profile=profile,
                bundle_id=bundle_id,
                prompt_snapshot=prompt_snapshot,
            ),
        },
    ]
