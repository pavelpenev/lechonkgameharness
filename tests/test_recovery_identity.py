# SPDX-License-Identifier: MIT
from dataclasses import replace

import pytest

from lechonk.core.recovery import RecoveryError
from lechonk.core.recovery_identity import (
    DigestSpec,
    IdentityInput,
    digest_for,
    interactive_v3_spec,
    legacy_specs,
    resolve_lineage,
    _resolve_lineage_for_tests,
)


ORIGINAL = {
    "fingerprint_probe_v2": "sha256:29c72ee59512c80521b3d55975e3dd82e3ea624a4e186655649b968ef78035cb",
    "repair_illegal_in_rule_gameplay_v2": "sha256:ece8374c1eb624d4fa21794ca13962005851bd4ac426c2304b3c2e67375a5f0a",
    "system_gameplay_v2": "sha256:dde0063c27abce2405bf92676cf96bc297449a5cd7f90770541849a97482775b",
    "turn_prefix_gameplay_v2": "sha256:701a8e229e4d31783c043d1732da6e2a02bb7ca16f3a0ae09cf4cb213138b59d",
    "repair_schema_gameplay_v2": "sha256:a314a9fc3b65fbe2f994720f856bd59ff0457546a79d3dd9cf8c0f84d9b3d4fe",
    "repair_state_fog_gameplay_v2": "sha256:1616affee2dec1e5357c6bdf89c022f472fa680cd64ce4ac588bb8ba37760b40",
}
CORRECTED = {
    "fingerprint_probe_v2": "sha256:29c72ee59512c80521b3d55975e3dd82e3ea624a4e186655649b968ef78035cb",
    "repair_illegal_in_rule_gameplay_v2": "sha256:ece8374c1eb624d4fa21794ca13962005851bd4ac426c2304b3c2e67375a5f0a",
    "system_gameplay_v2": "sha256:b8d7473bea7389bdb48484a8edcab24b42ca5436a8fe4af0bc8e96a7db0f87be",
    "turn_prefix_gameplay_v2": "sha256:3b2d4d50d799fdac09cb00b3cc1a3d21bb9736ab110131c23c8fe7f956741aa6",
    "repair_schema_gameplay_v2": "sha256:1649819b77a1154b0573bc059d5d8eefc40f8110b2bba7e721a5ba5230d2beba",
    "repair_state_fog_gameplay_v2": "sha256:1616affee2dec1e5357c6bdf89c022f472fa680cd64ce4ac588bb8ba37760b40",
}


def identity(**changes):
    values = dict(
        context_policy="conversation_v1", prompt_profile="gameplay_v2",
        model_id="mistral-large-4", prompt_hashes=ORIGINAL,
        ruleset="classic", seed=20261028, action_schema="ActionBatch-v1",
        classic_reference_present=True,
        classic_reference_schema_version="freeciv-reference-v1",
        classic_ruleset_tree_sha256="406610064a5c8b3d1f95ddb039635e30c9d44b0e204b6347a4060c0cad643d3f",
        manual_reference_present=True,
        manual_reference_schema_version="freeciv_manual_reference_v1",
        manual_reference_hash="e93703e51d01c1387f4c2e36ed562c38b58657612a47d8bf08918b5bfaad3ff5",
        manual_reference_source_sha256="eecae0ccea2fca07d50b4572a22ff8ed9f669cdcfddbc0539601d26902500c3e",
        aifill=2, configured_ai_opponents=1, prompt_bundle_id="gameplay_v2_original",
    )
    values.update(changes)
    return IdentityInput(**values)


def test_historical_digest_reproduces_checkpoint():
    spec, _ = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    # This is the exact pre-scenario payload recorded by the first checkpoint.
    assert digest_for(spec, identity()) == "sha256:0ebe33b64ec90cb013770ee8d926c96888ff7beb18ef438c6816bb2122e3c9eb"


def test_legacy_resolver_rejects_changed_values_and_unknown_schema():
    specs = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    checkpoint = {"config_digest": digest_for(specs[0], identity())}
    assert resolve_lineage(checkpoint, identity(), specs).spec.name == "legacy_pre_scenario_v1"
    with pytest.raises(RecoveryError):
        resolve_lineage(checkpoint, replace(identity(), seed=99), specs)
    with pytest.raises(RecoveryError):
        resolve_lineage({**checkpoint, "config_digest_schema": "future-v9"}, identity(), specs)


def test_scenario_schema_binds_opponents_and_prompt_hashes():
    specs = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    active = identity(prompt_hashes=CORRECTED, prompt_bundle_id="gameplay_v2_turnid_fix",
                      aifill=3, configured_ai_opponents=2)
    checkpoint = {"config_digest": digest_for(specs[1], active)}
    assert resolve_lineage(checkpoint, active, specs).spec.name == "legacy_scenario_v2"
    with pytest.raises(RecoveryError):
        resolve_lineage(checkpoint, replace(active, aifill=2, configured_ai_opponents=1), specs)


def test_corrected_scenario_digest_matches_three_player_lineage():
    specs = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    active = identity(prompt_hashes=CORRECTED, prompt_bundle_id="gameplay_v2_turnid_fix",
                      seed=20261029, aifill=3, configured_ai_opponents=2)
    digest = digest_for(specs[1], active)
    assert resolve_lineage({"config_digest": digest}, active, specs).spec.name == "legacy_scenario_v2"
    with pytest.raises(RecoveryError):
        resolve_lineage({"config_digest": digest}, replace(active, aifill=4, configured_ai_opponents=3), specs)


def test_v3_binds_policy_and_budget_identity():
    spec = interactive_v3_spec()
    current = identity(prompt_profile="gameplay_v3", prompt_bundle_id="gameplay_v3", prompt_hashes=CORRECTED,
                       aifill=3, configured_ai_opponents=2,
                       candidate_surface_policy="candidate_surface_v2",
                       candidate_surface_cap=8000,
                       candidate_admission_version="candidate_surface_v2-r1",
                       compaction_schema="conversation-compaction-v1",
                       prompt_budget_schema="gameplay-v3-budget-v1")
    digest = digest_for(spec, current)
    assert resolve_lineage({"config_digest_schema": spec.schema_id,
                            "config_digest": digest}, current, [spec]).spec == spec
    with pytest.raises(RecoveryError):
        resolve_lineage({"config_digest_schema": spec.schema_id,
                         "config_digest": digest},
                        replace(current, candidate_surface_cap=1200), [spec])


def test_v4_has_distinct_bundle_and_budget_lineage():
    spec = interactive_v3_spec()
    current = identity(
        prompt_profile="gameplay_v4", prompt_bundle_id="gameplay_v4",
        prompt_hashes={**CORRECTED, "system_gameplay_v4": "sha256:v4"},
        candidate_surface_policy="candidate_surface_v2",
        candidate_surface_cap=8000,
        candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v4-budget-v1")
    digest = digest_for(spec, current)
    assert resolve_lineage({"config_digest_schema": spec.schema_id,
                            "config_digest": digest}, current, [spec]).spec == spec
    with pytest.raises(RecoveryError, match="mismatch"):
        digest_for(spec, replace(current, prompt_profile="gameplay_v3"))


def test_v5_has_distinct_bundle_and_budget_lineage():
    spec = interactive_v3_spec()
    current = identity(
        prompt_profile="gameplay_v5", prompt_bundle_id="gameplay_v5",
        prompt_hashes={**CORRECTED, "system_gameplay_v5": "sha256:v5"},
        candidate_surface_policy="candidate_surface_v2",
        candidate_surface_cap=8000,
        candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v5-budget-v1")
    digest = digest_for(spec, current)
    assert resolve_lineage({"config_digest_schema": spec.schema_id,
                            "config_digest": digest}, current, [spec]).spec == spec
    with pytest.raises(RecoveryError, match="mismatch"):
        digest_for(spec, replace(current, prompt_profile="gameplay_v4"))


def test_v6_binds_candidate_surface_v3_12000_and_rejects_v4_mix():
    spec = interactive_v3_spec()
    current = identity(
        prompt_profile="gameplay_v6", prompt_bundle_id="gameplay_v6",
        prompt_hashes={**CORRECTED, "system_gameplay_v6": "sha256:v6"},
        candidate_surface_policy="candidate_surface_v3", candidate_surface_cap=12000,
        candidate_admission_version="candidate_surface_v3-r1",
        compaction_schema="conversation-compaction-v1",
        prompt_budget_schema="gameplay-v6-budget-v1")
    digest = digest_for(spec, current)
    assert resolve_lineage({"config_digest_schema": spec.schema_id,
                            "config_digest": digest}, current, [spec]).spec == spec
    with pytest.raises(RecoveryError, match="incoherent"):
        digest_for(spec, replace(current, prompt_profile="gameplay_v4", prompt_bundle_id="gameplay_v4"))


def test_ambiguous_finite_specs_fail_closed():
    base = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)[0]
    same = base
    digest = digest_for(base, identity())
    with pytest.raises(RecoveryError, match="multiple"):
        _resolve_lineage_for_tests({"config_digest": digest}, identity(), [base, same])


def test_public_resolver_rejects_synthetic_specs_and_explicit_null_schema():
    base = DigestSpec("synthetic", "synthetic", include_scenario=False)
    with pytest.raises(RecoveryError, match="built-in"):
        resolve_lineage({"config_digest": digest_for(base, identity())}, identity(), [base])
    specs = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    with pytest.raises(RecoveryError, match="null"):
        resolve_lineage({"config_digest": digest_for(specs[0], identity()),
                         "config_digest_schema": None}, identity(), specs)


def test_malformed_runtime_and_unknown_fixed_field_fail_as_recovery_errors():
    spec, _ = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    with pytest.raises(RecoveryError):
        digest_for(spec, replace(identity(), seed="bad"))
    with pytest.raises(RecoveryError, match="unknown fixed"):
        digest_for(DigestSpec("bad", "bad", False, fixed={"not_a_field": 1}), identity())


def test_builtin_names_cannot_be_forged_and_v3_requires_complete_identity():
    specs = legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED)
    forged = replace(specs[0], schema_id="legacy-pre-scenario-v1", include_scenario=True)
    with pytest.raises(RecoveryError, match="forged"):
        resolve_lineage({"config_digest": digest_for(forged, identity())}, identity(), [forged])
    incomplete = identity(prompt_profile="gameplay_v3", prompt_bundle_id="gameplay_v3",
                          prompt_hashes=CORRECTED, candidate_surface_policy=None,
                          candidate_surface_cap=None, candidate_admission_version=None,
                          compaction_schema=None, prompt_budget_schema=None)
    with pytest.raises(RecoveryError, match="incomplete"):
        digest_for(interactive_v3_spec(), incomplete)


def test_legacy_registry_rejects_forged_hashes_or_bundle():
    with pytest.raises(RecoveryError, match="frozen registry"):
        legacy_specs(original_prompt_hashes={**ORIGINAL, "system_gameplay_v2": "sha256:forged"},
                     corrected_prompt_hashes=CORRECTED)
    with pytest.raises(RecoveryError, match="frozen registry"):
        legacy_specs(original_prompt_hashes=ORIGINAL, corrected_prompt_hashes=CORRECTED,
                     corrected_bundle_id="arbitrary-bundle")


def test_v3_rejects_incoherent_policy_and_bundle_profile():
    spec = interactive_v3_spec()
    base = identity(prompt_profile="gameplay_v3", prompt_bundle_id="gameplay_v3",
                    prompt_hashes=CORRECTED, candidate_surface_policy="candidate_surface_v2",
                    candidate_surface_cap=8000, candidate_admission_version="candidate_surface_v2-r1",
                    compaction_schema="conversation-compaction-v1", prompt_budget_schema="gameplay-v3-budget-v1")
    with pytest.raises(RecoveryError, match="policy"):
        digest_for(spec, replace(base, candidate_surface_cap=1200))
    with pytest.raises(RecoveryError, match="mismatch"):
        digest_for(spec, replace(base, prompt_profile="gameplay_v2"))
    with pytest.raises(RecoveryError, match="mismatch"):
        digest_for(spec, replace(base, prompt_bundle_id="future-gameplay"))


def test_legacy_specs_normalizes_malformed_arguments_to_recovery_error():
    with pytest.raises(RecoveryError, match="invalid legacy"):
        legacy_specs(original_prompt_hashes=None, corrected_prompt_hashes=CORRECTED)


def test_v3_rejects_cross_version_policy_and_budget_combinations():
    spec = interactive_v3_spec()
    historical_bundle_with_v3_policy = identity(
        prompt_profile="gameplay_v2", prompt_bundle_id="gameplay_v2_turnid_fix",
        prompt_hashes=CORRECTED, candidate_surface_policy="candidate_surface_v2",
        candidate_surface_cap=8000, candidate_admission_version="candidate_surface_v2-r1",
        compaction_schema="conversation-compaction-v1", prompt_budget_schema="gameplay-v3-budget-v1")
    with pytest.raises(RecoveryError, match="incoherent"):
        digest_for(spec, historical_bundle_with_v3_policy)
    v3_with_legacy_budget = replace(historical_bundle_with_v3_policy,
                                    prompt_profile="gameplay_v3", prompt_bundle_id="gameplay_v3",
                                    candidate_surface_policy="legacy_v1", candidate_surface_cap=None,
                                    candidate_admission_version=None)
    with pytest.raises(RecoveryError, match="incoherent"):
        digest_for(spec, v3_with_legacy_budget)
