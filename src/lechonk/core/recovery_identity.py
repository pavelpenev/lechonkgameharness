# SPDX-License-Identifier: MIT
"""Pure, finite recovery-identity construction.

Recovery files must be resumed with the exact prompt and game lineage that
created them.  This module deliberately has no filesystem or process side
effects: callers validate a manifest and its bound files first, then use the
finite specs below to resolve its digest.  There is no caller supplied digest
override and no generic legacy fallback.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from lechonk.core.jsonutil import canonical_json
from lechonk.core.recovery import RecoveryError


_HISTORICAL_FIELDS = (
    "context_policy", "prompt_profile", "model_id", "prompt_hashes",
    "ruleset", "seed", "action_schema", "classic_reference_present",
    "classic_reference_schema_version", "classic_ruleset_tree_sha256",
    "manual_reference_present", "manual_reference_schema_version",
    "manual_reference_hash", "manual_reference_source_sha256",
)
_SCENARIO_FIELDS = ("aifill", "configured_ai_opponents")
_V3_FIELDS = ("prompt_bundle_id", "candidate_surface_policy",
              "candidate_surface_cap", "candidate_admission_version",
              "compaction_schema", "prompt_budget_schema")
_ALL_FIELDS = frozenset((*_HISTORICAL_FIELDS, *_SCENARIO_FIELDS, *_V3_FIELDS))
_ABSENT = object()
_BUILTIN_NAMES = frozenset({"legacy_pre_scenario_v1", "legacy_scenario_v2",
                            "interactive-config-v3"})
_ORIGINAL_PROMPT_HASHES = MappingProxyType({
    "fingerprint_probe_v2": "sha256:29c72ee59512c80521b3d55975e3dd82e3ea624a4e186655649b968ef78035cb",
    "repair_illegal_in_rule_gameplay_v2": "sha256:ece8374c1eb624d4fa21794ca13962005851bd4ac426c2304b3c2e67375a5f0a",
    "system_gameplay_v2": "sha256:dde0063c27abce2405bf92676cf96bc297449a5cd7f90770541849a97482775b",
    "turn_prefix_gameplay_v2": "sha256:701a8e229e4d31783c043d1732da6e2a02bb7ca16f3a0ae09cf4cb213138b59d",
    "repair_schema_gameplay_v2": "sha256:a314a9fc3b65fbe2f994720f856bd59ff0457546a79d3dd9cf8c0f84d9b3d4fe",
    "repair_state_fog_gameplay_v2": "sha256:1616affee2dec1e5357c6bdf89c022f472fa680cd64ce4ac588bb8ba37760b40",
})
_CORRECTED_PROMPT_HASHES = MappingProxyType({
    **_ORIGINAL_PROMPT_HASHES,
    "system_gameplay_v2": "sha256:b8d7473bea7389bdb48484a8edcab24b42ca5436a8fe4af0bc8e96a7db0f87be",
    "turn_prefix_gameplay_v2": "sha256:3b2d4d50d799fdac09cb00b3cc1a3d21bb9736ab110131c23c8fe7f956741aa6",
    "repair_schema_gameplay_v2": "sha256:1649819b77a1154b0573bc059d5d8eefc40f8110b2bba7e721a5ba5230d2beba",
})
_BUNDLE_PROFILES = MappingProxyType({
    "gameplay_v2_original": "gameplay_v2",
    "gameplay_v2_turnid_fix": "gameplay_v2",
    "gameplay_v3": "gameplay_v3",
    "gameplay_v4": "gameplay_v4",
    "gameplay_v5": "gameplay_v5",
    "gameplay_v6": "gameplay_v6",
})


def _validate_identity(values: Mapping[str, Any]) -> None:
    required_strings = ("context_policy", "prompt_profile", "model_id",
                        "ruleset", "action_schema")
    for key in required_strings:
        if not isinstance(values.get(key), str) or not values[key]:
            raise RecoveryError(f"invalid recovery identity field: {key}")
    hashes = values.get("prompt_hashes")
    if not isinstance(hashes, Mapping) or not hashes or any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in hashes.items()):
        raise RecoveryError("invalid recovery identity prompt hashes")
    for key in ("classic_reference_present", "manual_reference_present"):
        if type(values.get(key)) is not bool:
            raise RecoveryError(f"invalid recovery identity field: {key}")
    for key in ("seed", "aifill", "configured_ai_opponents"):
        if type(values.get(key)) is not int:
            raise RecoveryError(f"invalid recovery identity field: {key}")
    optional_strings = ("classic_reference_schema_version",
                        "classic_ruleset_tree_sha256",
                        "manual_reference_schema_version", "manual_reference_hash",
                        "manual_reference_source_sha256", "prompt_bundle_id",
                        "candidate_surface_policy", "candidate_admission_version",
                        "compaction_schema", "prompt_budget_schema")
    for key in optional_strings:
        if values.get(key) is not None and not isinstance(values[key], str):
            raise RecoveryError(f"invalid recovery identity field: {key}")
    if values.get("candidate_surface_cap") is not None and type(values["candidate_surface_cap"]) is not int:
        raise RecoveryError("invalid recovery identity field: candidate_surface_cap")


def _validate_spec(spec: DigestSpec) -> None:
    if not isinstance(spec.name, str) or not spec.name or not isinstance(spec.schema_id, str) or not spec.schema_id:
        raise RecoveryError("invalid recovery digest spec")
    if not isinstance(spec.include_scenario, bool) or not isinstance(spec.include_v3, bool):
        raise RecoveryError("invalid recovery digest spec flags")
    unknown = set(spec.fixed) - _ALL_FIELDS
    if unknown:
        raise RecoveryError(f"unknown fixed recovery identity fields: {sorted(unknown)}")
    if spec.expected_prompt_hashes is not None:
        if not isinstance(spec.expected_prompt_hashes, Mapping) or any(
                not isinstance(k, str) or not isinstance(v, str) or not v
                for k, v in spec.expected_prompt_hashes.items()):
            raise RecoveryError("invalid expected prompt hashes")
    if spec.expected_prompt_bundle_id is not None and not isinstance(spec.expected_prompt_bundle_id, str):
        raise RecoveryError("invalid expected prompt bundle")
    if spec.name == "legacy_pre_scenario_v1":
        if (spec.schema_id != "legacy-pre-scenario-v1" or spec.include_scenario or
                spec.include_v3 or spec.expected_prompt_bundle_id != "gameplay_v2_original" or
                dict(spec.fixed) != {"prompt_profile": "gameplay_v2", "aifill": 2,
                                     "configured_ai_opponents": 1}):
            raise RecoveryError("forged legacy pre-scenario digest spec")
        if dict(spec.expected_prompt_hashes or {}) != _ORIGINAL_PROMPT_HASHES:
            raise RecoveryError("untrusted legacy pre-scenario prompt hashes")
    elif spec.name == "legacy_scenario_v2":
        if (spec.schema_id != "legacy-scenario-v2" or not spec.include_scenario or
                spec.include_v3 or spec.expected_prompt_bundle_id != "gameplay_v2_turnid_fix" or
                dict(spec.fixed) != {"prompt_profile": "gameplay_v2"}):
            raise RecoveryError("forged legacy scenario digest spec")
        if dict(spec.expected_prompt_hashes or {}) != _CORRECTED_PROMPT_HASHES:
            raise RecoveryError("untrusted legacy scenario prompt hashes")
    elif spec.name == "interactive-config-v3":
        if (spec.schema_id != "interactive-config-v3" or not spec.include_scenario or
                not spec.include_v3 or spec.fixed or spec.expected_prompt_hashes is not None or
                spec.expected_prompt_bundle_id is not None):
            raise RecoveryError("forged interactive digest spec")
    else:
        raise RecoveryError("recovery digest spec is not in the built-in registry")


def _validate_specs(specs: Sequence[DigestSpec]) -> None:
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        raise RecoveryError("invalid recovery digest registry")
    for spec in specs:
        _validate_spec(spec)
        if spec.name not in _BUILTIN_NAMES:
            raise RecoveryError("recovery digest spec is not in the built-in registry")


@dataclass(frozen=True)
class IdentityInput:
    """All values that may participate in a recovery identity."""

    context_policy: str
    prompt_profile: str
    model_id: str
    prompt_hashes: Mapping[str, str]
    ruleset: str
    seed: int
    action_schema: str
    classic_reference_present: bool
    classic_reference_schema_version: str | None
    classic_ruleset_tree_sha256: str | None
    manual_reference_present: bool
    manual_reference_schema_version: str | None
    manual_reference_hash: str | None
    manual_reference_source_sha256: str | None
    aifill: int
    configured_ai_opponents: int
    prompt_bundle_id: str | None = None
    candidate_surface_policy: str = "legacy_v1"
    candidate_surface_cap: int | None = None
    candidate_admission_version: str | None = None
    compaction_schema: str | None = None
    prompt_budget_schema: str | None = None

    def as_dict(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in _ALL_FIELDS}
        _validate_identity(values)
        return values


@dataclass(frozen=True)
class DigestSpec:
    """A named, finite digest schema and its lineage constraints."""

    name: str
    schema_id: str
    include_scenario: bool
    include_v3: bool = False
    expected_prompt_hashes: Mapping[str, str] | None = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    expected_prompt_bundle_id: str | None = None


@dataclass(frozen=True)
class LineageMatch:
    spec: DigestSpec
    digest: str


def canonical_payload(spec: DigestSpec, identity: IdentityInput) -> dict[str, Any]:
    """Build the exact payload for a named digest schema."""
    _validate_spec(spec)
    values = identity.as_dict()
    if spec.name == "interactive-config-v3":
        required_v3 = ("prompt_bundle_id", "compaction_schema", "prompt_budget_schema")
        policy = values["candidate_surface_policy"]
        candidate_policy_invalid = (
            policy == "candidate_surface_v2" and
            (not values["candidate_admission_version"] or
             type(values["candidate_surface_cap"]) is not int or
             values["candidate_surface_cap"] <= 0))
        if (any(not values[key] for key in required_v3) or
                candidate_policy_invalid or
                policy not in {"legacy_v1", "candidate_surface_v2", "candidate_surface_v3"}):
            raise RecoveryError("incomplete interactive-config-v3 identity")
        bundle = values["prompt_bundle_id"]
        if bundle not in _BUNDLE_PROFILES or values["prompt_profile"] != _BUNDLE_PROFILES[bundle]:
            raise RecoveryError("prompt bundle/profile mismatch")
        coherent = {
            ("gameplay_v3", "gameplay_v3", "candidate_surface_v2", 8000,
             "candidate_surface_v2-r1", "conversation-compaction-v1",
             "gameplay-v3-budget-v1"),
            ("gameplay_v4", "gameplay_v4", "candidate_surface_v2", 8000,
             "candidate_surface_v2-r1", "conversation-compaction-v1",
             "gameplay-v4-budget-v1"),
            ("gameplay_v5", "gameplay_v5", "candidate_surface_v2", 8000,
             "candidate_surface_v2-r1", "conversation-compaction-v1",
             "gameplay-v5-budget-v1"),
            ("gameplay_v6", "gameplay_v6", "candidate_surface_v3", 12000,
             "candidate_surface_v3-r1", "conversation-compaction-v1",
             "gameplay-v6-budget-v1"),
            ("gameplay_v2_original", "gameplay_v2", "legacy_v1", None, None,
             "conversation-compaction-v1", "legacy-budget-v1"),
            ("gameplay_v2_turnid_fix", "gameplay_v2", "legacy_v1", None, None,
             "conversation-compaction-v1", "legacy-budget-v1"),
        }
        identity_tuple = (values["prompt_bundle_id"], values["prompt_profile"],
                          values["candidate_surface_policy"], values["candidate_surface_cap"],
                          values["candidate_admission_version"], values["compaction_schema"],
                          values["prompt_budget_schema"])
        if identity_tuple not in coherent:
            raise RecoveryError("incoherent policy/compaction/prompt-budget identity")
    if spec.expected_prompt_hashes is not None and dict(values["prompt_hashes"]) != dict(spec.expected_prompt_hashes):
        raise RecoveryError(f"prompt hashes do not match lineage {spec.name}")
    if spec.expected_prompt_bundle_id is not None and values["prompt_bundle_id"] != spec.expected_prompt_bundle_id:
        raise RecoveryError(f"prompt bundle does not match lineage {spec.name}")
    for key, expected in spec.fixed.items():
        if values[key] != expected:
            raise RecoveryError(f"{key} does not match lineage {spec.name}")
    payload = {key: values[key] for key in _HISTORICAL_FIELDS}
    if spec.include_scenario:
        payload.update({key: values[key] for key in _SCENARIO_FIELDS})
    if spec.include_v3:
        payload.update({key: values[key] for key in (
            "prompt_bundle_id", "candidate_surface_policy",
            "candidate_surface_cap", "candidate_admission_version",
            "compaction_schema", "prompt_budget_schema")})
    return payload


def digest_for(spec: DigestSpec, identity: IdentityInput) -> str:
    """Return the canonical SHA-256 digest for ``spec`` and ``identity``."""
    try:
        payload = canonical_payload(spec, identity)
        encoded = canonical_json(payload).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
    except RecoveryError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RecoveryError("invalid recovery identity") from exc


def resolve_lineage(checkpoint: Mapping[str, Any] | Any,
                    runtime: IdentityInput,
                    specs: Sequence[DigestSpec]) -> LineageMatch:
    """Resolve exactly one finite lineage, or fail closed.

    Old manifests have no schema marker and are checked only against the
    caller-provided finite legacy specs.  Explicit unknown schemas are never
    silently downgraded to a legacy candidate.
    """
    _validate_identity(runtime.as_dict())
    _validate_specs(specs)
    if isinstance(checkpoint, Mapping):
        marker = checkpoint["config_digest_schema"] if "config_digest_schema" in checkpoint else _ABSENT
        config_digest = checkpoint.get("config_digest", _ABSENT)
    else:
        marker = getattr(checkpoint, "config_digest_schema", _ABSENT)
        config_digest = getattr(checkpoint, "config_digest", _ABSENT)
    if marker is None:
        raise RecoveryError("explicit null recovery digest schema")
    if marker is not _ABSENT and not isinstance(marker, str):
        raise RecoveryError("invalid recovery digest schema")
    if config_digest is _ABSENT:
        config_digest = None
    if not isinstance(config_digest, str):
        raise RecoveryError("checkpoint has no config digest")
    candidates = [spec for spec in specs if (marker is _ABSENT or spec.schema_id == marker)]
    if marker is not _ABSENT and not candidates:
        raise RecoveryError(f"unknown recovery digest schema: {marker}")
    matches: list[LineageMatch] = []
    for spec in candidates:
        try:
            digest = digest_for(spec, runtime)
        except RecoveryError:
            continue
        if digest == config_digest:
            matches.append(LineageMatch(spec, digest))
    if len(matches) != 1:
        if not matches:
            raise RecoveryError("checkpoint does not match a known recovery lineage")
        raise RecoveryError("checkpoint matches multiple recovery lineages")
    return matches[0]


def _resolve_lineage_for_tests(checkpoint: Mapping[str, Any], runtime: IdentityInput,
                               specs: Sequence[DigestSpec]) -> LineageMatch:
    """Test-only resolver allowing synthetic specs to exercise ambiguity."""
    return _resolve(checkpoint, runtime, specs)


def _resolve(checkpoint: Mapping[str, Any], runtime: IdentityInput,
             specs: Sequence[DigestSpec]) -> LineageMatch:
    """Internal implementation; public resolution validates the closed registry."""
    marker = checkpoint["config_digest_schema"] if "config_digest_schema" in checkpoint else _ABSENT
    if marker is None:
        raise RecoveryError("explicit null recovery digest schema")
    config_digest = checkpoint.get("config_digest", _ABSENT)
    if not isinstance(config_digest, str):
        raise RecoveryError("checkpoint has no config digest")
    candidates = [s for s in specs if marker is _ABSENT or s.schema_id == marker]
    if marker is not _ABSENT and not candidates:
        raise RecoveryError(f"unknown recovery digest schema: {marker}")
    matches = []
    for spec in candidates:
        try:
            digest = digest_for(spec, runtime)
        except RecoveryError:
            continue
        if digest == config_digest:
            matches.append(LineageMatch(spec, digest))
    if len(matches) != 1:
        raise RecoveryError("checkpoint matches multiple recovery lineages" if len(matches) > 1 else "checkpoint does not match a known recovery lineage")
    return matches[0]


def legacy_specs(*, original_prompt_hashes: Mapping[str, str],
                 corrected_prompt_hashes: Mapping[str, str],
                 original_bundle_id: str = "gameplay_v2_original",
                 corrected_bundle_id: str = "gameplay_v2_turnid_fix") -> tuple[DigestSpec, DigestSpec]:
    """Return the only two supported unversioned historical lineages."""
    try:
        original_map = dict(original_prompt_hashes)
        corrected_map = dict(corrected_prompt_hashes)
    except (TypeError, ValueError) as exc:
        raise RecoveryError("invalid legacy lineage prompt hashes") from exc
    if (original_bundle_id != "gameplay_v2_original" or
            corrected_bundle_id != "gameplay_v2_turnid_fix" or
            original_map != dict(_ORIGINAL_PROMPT_HASHES) or
            corrected_map != dict(_CORRECTED_PROMPT_HASHES)):
        raise RecoveryError("legacy lineage provenance is not the frozen registry")
    return (
        DigestSpec("legacy_pre_scenario_v1", "legacy-pre-scenario-v1",
                   include_scenario=False, expected_prompt_hashes=original_prompt_hashes,
                   expected_prompt_bundle_id=original_bundle_id,
                   fixed={"prompt_profile": "gameplay_v2", "aifill": 2,
                          "configured_ai_opponents": 1}),
        DigestSpec("legacy_scenario_v2", "legacy-scenario-v2",
                   include_scenario=True, expected_prompt_hashes=corrected_prompt_hashes,
                   expected_prompt_bundle_id=corrected_bundle_id,
                   fixed={"prompt_profile": "gameplay_v2"}),
    )


def interactive_v3_spec() -> DigestSpec:
    """The explicit fresh-run schema for immutable prompt/policy identity."""
    return DigestSpec("interactive-config-v3", "interactive-config-v3",
                      include_scenario=True, include_v3=True)


def interactive_v4_spec() -> DigestSpec:
    """The v4 profile uses the same explicit interactive schema registry."""
    return interactive_v3_spec()
