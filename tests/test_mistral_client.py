# SPDX-License-Identifier: MIT
"""Mistral client tests — offline mock-transport verification of the WP2
package-A surface (design-plan-freeciv-v1.1.md decision 4 + decision 8):

- the exact model ID is pinned (no silent substitution);
- the API key comes from the ``MISTRAL_API_KEY`` process variable; when
  absent, the narrow loader (``load_api_key_from_env_file``) loads it from
  the repo-root ``.env`` — parsing ONLY that variable's line (optional
  quotes, other variables and comments ignored), setting it into the
  process environment, never logging or serializing it — and raises the
  clear startup error when neither source provides a key;
- the per-turn attempts contract: ≤2 total API attempts including at most
  one repair; SDK/transport retries inside one logical attempt are
  transparent and never counted;
- both enforcement paths (json_schema structured outputs vs
  schema-validated parsing) behind one interface;
- per-call logging fields (tokens, latency, cost, fingerprint data);
- the no-secret assertion: the key never appears in any log line or
  serialized artifact;
- the per-arm model fingerprint probe (fixed prompt, temperature 0, output
  hash) and the P10 probe methods under a mock transport;
- the model package stays civrealm-free and MIT-marked (plan §7.1).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lechonk.core.schemas import (
    ACTION_BATCH_MAX_ACTIONS,
    CANDIDATE_ID_FIELD_PATTERN,
    ActionBatch,
)
from lechonk.model import mistral_client as mc
from lechonk.model import prompts
from lechonk.model.mistral_client import (
    MODEL_ID,
    EnforcementMode,
    MistralClient,
    MistralKeyError,
    ModelIdError,
    AttemptBudgetExhausted,
    RepairRequest,
    SchemaError,
    TransportError,
    TurnAttempts,
)

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "src" / "lechonk" / "model"

CANARY_KEY = "sk-CANARY-9f3a91c2-do-not-leak"

VALID_BATCH = json.dumps({
    "turn_id": "run1/t1",
    "actions": [{"ref": "unit:7:goto_n"}],
    "commitments": [{"domain": "expansion", "target": "north",
                     "deadline_turn": 5}],
    "finish_requested": False,
    "notes": "move north",
    "memory_update": {"commitments": [], "threats": ["unit:12"],
                      "notes": "watch the east"},
})


# ---------------------------------------------------------------------------
# Mock transport (no network, no SDK import)
# ---------------------------------------------------------------------------


@dataclass
class MockTransport:
    """Scripted transport: a queue of outcomes (RawCompletion or Exception)."""

    outcomes: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, *, messages, model, temperature, max_tokens,
                 response_format, random_seed, timeout_ms=None,
                 disable_transport_retries=False):
        self.calls.append({
            "messages": list(messages), "model": model,
            "temperature": temperature, "max_tokens": max_tokens,
            "response_format": response_format, "random_seed": random_seed,
            "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def raw_completion(content, *, model=MODEL_ID, response_id="resp-1",
                   finish_reason="stop", prompt_tokens=100,
                   completion_tokens=50, transport_retries=0):
    return mc.RawCompletion(
        content=content, model=model, response_id=response_id, created=0,
        finish_reason=finish_reason, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        transport_retries=transport_retries)


def make_client(transport, *, key=CANARY_KEY, **kwargs):
    return MistralClient(api_key=key, transport=transport, **kwargs)


def turn_messages():
    return [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": prompts.turn_prompt(
            memory="mem", snapshot="snap", delta="delta",
            candidate_summary="unit:7:goto_n — move north")},
    ]


# ---------------------------------------------------------------------------
# Model ID pinning (decision 4: no silent substitution)
# ---------------------------------------------------------------------------


def test_model_id_is_pinned_exact():
    assert MODEL_ID == "mistral-large-4"


@pytest.mark.parametrize("other", [
    "mistral-large-latest", "mistral-large", "mistral-medium",
    "codestral-latest", "pixtral-large-latest",
])
def test_model_substitution_rejected(other):
    with pytest.raises(ModelIdError) as excinfo:
        MistralClient(api_key=CANARY_KEY, model=other,
                      transport=MockTransport())
    assert "silent substitution" in str(excinfo.value)
    assert MODEL_ID in str(excinfo.value)


# ---------------------------------------------------------------------------
# API key handling: process environment only, clear startup error
# ---------------------------------------------------------------------------


def test_absent_key_is_clear_startup_error(monkeypatch):
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(mc, "default_dotenv_path", lambda: None)
    with pytest.raises(MistralKeyError) as excinfo:
        MistralClient(transport=MockTransport())
    msg = str(excinfo.value)
    assert mc.API_KEY_ENV_VAR in msg
    assert "environment" in msg
    # the error never contains a key value
    assert CANARY_KEY not in msg


def test_key_read_from_process_environment(monkeypatch):
    monkeypatch.setenv(mc.API_KEY_ENV_VAR, CANARY_KEY)
    client = MistralClient(transport=MockTransport())
    assert client.model == MODEL_ID


def test_no_key_anywhere_is_clear_startup_error(monkeypatch):
    """With the variable absent AND no resolvable .env (default dotenv path
    monkeypatched away), construction fails with the clear startup error
    naming the variable, never any key value."""
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(mc, "default_dotenv_path", lambda: None)
    with pytest.raises(MistralKeyError) as excinfo:
        MistralClient(transport=MockTransport())
    msg = str(excinfo.value)
    assert mc.API_KEY_ENV_VAR in msg
    assert CANARY_KEY not in msg


def test_model_sources_no_third_party_dotenv():
    """Source scan: the .env support is the package's own narrow parser —
    no third-party dotenv import or loader call anywhere under src/lechonk/
    model."""
    pattern = re.compile(
        r"(\bimport\s+dotenv\b|\bfrom\s+dotenv\b|\bload_dotenv\s*\()")
    offenders = []
    for path in sorted(MODEL_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path))
    assert not offenders, f"model files using a dotenv library: {offenders}"


def test_key_error_message_never_contains_a_key_value(monkeypatch):
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(mc, "default_dotenv_path", lambda: None)
    with pytest.raises(MistralKeyError) as excinfo:
        MistralClient(transport=MockTransport())
    assert CANARY_KEY not in str(excinfo.value)


# ---------------------------------------------------------------------------
# .env loader (narrow runtime loader for the repo-root .env)
# ---------------------------------------------------------------------------

# A test canary only — never a real key.
LOADER_KEY = "sk-canary-loader-4f8e21b7-test-only"


def _write_dotenv(path, key_line):
    path.write_text(
        "# comment at top\n"
        "OTHER_VAR=should-be-ignored\n"
        f"{key_line}\n"
        "ANOTHER_VAR=also-ignored\n",
        encoding="utf-8",
    )
    return path


def test_loader_loads_unquoted_key(tmp_path, monkeypatch, caplog):
    env_file = _write_dotenv(
        tmp_path / ".env", f"MISTRAL_API_KEY={LOADER_KEY}")
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    with caplog.at_level("DEBUG"):
        assert mc.load_api_key_from_env_file(env_file) is None
    assert os.environ.get(mc.API_KEY_ENV_VAR) == LOADER_KEY
    assert LOADER_KEY not in caplog.text


def test_loader_loads_quoted_keys(tmp_path, monkeypatch):
    variants = [
        f'MISTRAL_API_KEY="{LOADER_KEY}"',
        f"MISTRAL_API_KEY='{LOADER_KEY}'",
        f'  MISTRAL_API_KEY =  "{LOADER_KEY}"  ',
    ]
    for index, key_line in enumerate(variants):
        env_file = tmp_path / f"dotenv{index}"
        _write_dotenv(env_file, key_line)
        monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
        mc.load_api_key_from_env_file(env_file)
        assert os.environ.get(mc.API_KEY_ENV_VAR) == LOADER_KEY


def test_loader_ignores_other_variables_and_comments(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# MISTRAL_API_KEY=commented-out-key\n"
        "MISTRAL_API_KEY_WRONG=nope\n"
        "PREFIX_MISTRAL_API_KEY=nope\n"
        f"MISTRAL_API_KEY={LOADER_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    mc.load_api_key_from_env_file(env_file)
    assert os.environ.get(mc.API_KEY_ENV_VAR) == LOADER_KEY


def test_loader_env_var_precedence(tmp_path, monkeypatch):
    env_file = _write_dotenv(tmp_path / ".env", f"MISTRAL_API_KEY={LOADER_KEY}")
    monkeypatch.setenv(mc.API_KEY_ENV_VAR, "env-precedence-key")
    mc.load_api_key_from_env_file(env_file)
    assert os.environ.get(mc.API_KEY_ENV_VAR) == "env-precedence-key"


def test_loader_short_circuits_without_reading_file(tmp_path, monkeypatch):
    """With the env var already set, the loader returns without touching the
    file (even a missing one) and without changing the value."""
    monkeypatch.setenv(mc.API_KEY_ENV_VAR, "env-precedence-key")
    mc.load_api_key_from_env_file(tmp_path / "does-not-exist.env")
    assert os.environ.get(mc.API_KEY_ENV_VAR) == "env-precedence-key"


def test_loader_missing_file_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    with pytest.raises(MistralKeyError) as excinfo:
        mc.load_api_key_from_env_file(tmp_path / "absent.env")
    msg = str(excinfo.value)
    assert mc.API_KEY_ENV_VAR in msg
    assert LOADER_KEY not in msg


def test_env_file_without_key_line_raises(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# only comments and other variables here\nOTHER_VAR=not-the-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    with pytest.raises(MistralKeyError) as excinfo:
        mc.load_api_key_from_env_file(env_file)
    msg = str(excinfo.value)
    assert mc.API_KEY_ENV_VAR in msg
    assert "not-the-key" not in msg


def test_loader_exposes_nothing_key_shaped(tmp_path, monkeypatch, caplog):
    """No-exposure assertion: the loader returns None, logs nothing
    key-shaped, and its error path carries no key material."""
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    env_file = _write_dotenv(tmp_path / ".env", f"MISTRAL_API_KEY={LOADER_KEY}")
    with caplog.at_level("DEBUG"):
        assert mc.load_api_key_from_env_file(env_file) is None
    blob = caplog.text
    assert re.search(r"sk-[A-Za-z0-9_-]{8,}", blob) is None
    assert LOADER_KEY not in blob


def test_client_acquires_key_from_dotenv_fallback(
        tmp_path, monkeypatch, caplog):
    """End to end: no env var, default dotenv path monkeypatched to a temp
    .env — the client constructs and the key reaches only the environ."""
    env_file = _write_dotenv(tmp_path / ".env", f"MISTRAL_API_KEY={LOADER_KEY}")
    monkeypatch.delenv(mc.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(mc, "default_dotenv_path", lambda: env_file)
    with caplog.at_level("DEBUG"):
        client = MistralClient(transport=MockTransport())
    assert client.model == MODEL_ID
    assert os.environ.get(mc.API_KEY_ENV_VAR) == LOADER_KEY
    assert re.search(r"sk-[A-Za-z0-9_-]{8,}", caplog.text) is None


# ---------------------------------------------------------------------------
# Attempts contract (decision 4): ≤2 attempts/turn, ≤1 repair
# ---------------------------------------------------------------------------


def test_attempt_cap_two_total_one_repair():
    transport = MockTransport(outcomes=[raw_completion("garbage"),
                                        raw_completion("garbage")])
    client = make_client(transport)
    with pytest.raises(AttemptBudgetExhausted):
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    assert len(transport.calls) == 2  # never a third API attempt


def test_turn_attempts_counter_caps():
    attempts = TurnAttempts()
    assert attempts.begin_attempt(is_repair=False) == 1
    assert attempts.begin_attempt(is_repair=True) == 2
    with pytest.raises(AttemptBudgetExhausted):
        attempts.begin_attempt(is_repair=False)  # third attempt forbidden
    with pytest.raises(AttemptBudgetExhausted):
        attempts.begin_attempt(is_repair=True)  # second repair forbidden


def test_repair_attempt_counted_and_succeeds():
    transport = MockTransport(outcomes=[raw_completion("not json"),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    result = client.action_batch(
        messages=turn_messages(), prompt_hash="sha256:test",
        attempts=TurnAttempts(),
        repair_candidates_for=lambda rr: "FULL TABLE: unit:7:goto_n")
    assert result.batch.turn_id == "run1/t1"
    assert result.attempts_used == 2
    assert result.repairs_used == 1
    assert len(result.call_logs) == 2
    assert result.call_logs[1].kind == "repair"
    assert result.call_logs[1].attempt == 2
    # the repair conversation carries the original messages + the failed
    # assistant output + the class-specific repair instruction
    repair_call = transport.calls[1]
    assert len(repair_call["messages"]) == len(turn_messages()) + 2
    assert repair_call["messages"][-2]["role"] == "assistant"
    assert repair_call["messages"][-2]["content"] == "not json"
    assert repair_call["messages"][-1]["role"] == "user"
    assert "SCHEMA ERROR" in repair_call["messages"][-1]["content"]
    assert "FULL TABLE: unit:7:goto_n" in repair_call["messages"][-1]["content"]


class _RetryableNetworkError(Exception):
    """A retryable transport-level error (connection reset)."""


@dataclass
class InternallyRetryingTransport:
    """Simulates SDK-internal transparent retry/backoff: the first network
    call fails with a retryable error and the retry INSIDE the same
    complete() invocation succeeds. The client sees exactly ONE logical
    attempt (decision 4: SDK/transport retries never count against the
    per-turn cap). With ``disable_transport_retries=True`` (the budgeted
    repair attempts, WP4) the retryable error propagates immediately —
    exactly one network request."""

    calls: list = field(default_factory=list)
    network_calls: int = 0

    def complete(self, *, messages, model, temperature, max_tokens,
                 response_format, random_seed, timeout_ms=None,
                 disable_transport_retries=False):
        self.calls.append({
            "messages": list(messages),
            "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
        })
        while True:
            self.network_calls += 1
            try:
                if self.network_calls == 1:
                    raise _RetryableNetworkError("connection reset")
                return raw_completion(
                    VALID_BATCH, transport_retries=self.network_calls - 1)
            except _RetryableNetworkError:
                if disable_transport_retries:
                    raise  # single request attempt: no internal retry
                continue  # transparent retry inside the logical attempt


def test_transport_retries_inside_one_attempt_never_counted():
    transport = InternallyRetryingTransport()
    client = make_client(transport)
    attempts = TurnAttempts()
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=attempts)
    # the internal retry consumed NO turn budget: one logical attempt, no repair
    assert len(transport.calls) == 1
    assert transport.network_calls == 2  # retried inside the attempt
    assert attempts.attempts == 1
    assert attempts.repairs == 0
    assert result.attempts_used == 1
    assert result.repairs_used == 0
    # ...but the retry is recorded on the call log (logged, not counted)
    assert result.call_logs[0].transport_retries == 1
    assert result.call_logs[0].error is None


def test_transport_hard_failure_consumes_attempt_then_repair_reasks():
    transport = MockTransport(
        outcomes=[TransportError("500 after SDK retries"),
                  raw_completion(VALID_BATCH)])
    client = make_client(transport)
    messages = turn_messages()
    result = client.action_batch(messages=messages,
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.batch.turn_id == "run1/t1"
    assert result.attempts_used == 2
    assert result.repairs_used == 1
    assert len(transport.calls) == 2
    # the failed attempt is logged with its error
    assert result.call_logs[0].error is not None
    assert "500" in result.call_logs[0].error
    # the repair re-asks the SAME turn (no model output was produced)
    assert transport.calls[1]["messages"] == messages
    assert result.call_logs[1].kind == "repair"


def test_both_attempts_transport_fail_exhausts_budget():
    transport = MockTransport(outcomes=[TransportError("down"),
                                        TransportError("down again")])
    client = make_client(transport)
    with pytest.raises(AttemptBudgetExhausted) as excinfo:
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    assert "transport" in str(excinfo.value)
    assert len(transport.calls) == 2
    assert len(client.call_logs) == 2  # both failed attempts logged


def test_exhausted_attempts_object_rejects_before_any_call():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    attempts = TurnAttempts(attempts=2, repairs=1)
    with pytest.raises(AttemptBudgetExhausted):
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test", attempts=attempts)
    assert transport.calls == []  # no API call was made


def test_complete_once_passes_per_call_timeout_to_the_transport():
    """WP4 budget-aware repair gating: ``complete_once`` threads the
    optional per-call request timeout override into the transport; the
    default (None) leaves the configured timeout untouched."""
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:x",
                         attempt=1, kind="initial")
    assert transport.calls[0]["timeout_ms"] is None
    # Attempt 1 keeps the transport's transparent retries (default False).
    assert transport.calls[0]["disable_transport_retries"] is False

    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:x",
                         attempt=2, kind="repair", timeout_ms=75_000)
    assert transport.calls[0]["timeout_ms"] == 75_000
    # The attempt is logged like every other call.
    assert client.call_logs[-1].attempt == 2
    assert client.call_logs[-1].kind == "repair"
    assert client.call_logs[-1].error is None


def test_complete_once_threads_disable_transport_retries_to_the_transport():
    """WP4 gating, the derivation chain complete_once -> transport: the
    retry-disabled flag reaches the transport exactly as passed — True on a
    budgeted repair (exactly one request attempt), False by default."""
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:x",
                         attempt=2, kind="repair", timeout_ms=55_000,
                         disable_transport_retries=True)
    assert transport.calls[0]["disable_transport_retries"] is True
    assert transport.calls[0]["timeout_ms"] == 55_000


def test_repair_with_retries_disabled_makes_one_request_and_surfaces_failure():
    """WP4 gating, end-to-end through the client: a repair attempt with
    ``disable_transport_retries=True`` makes EXACTLY ONE network request on a
    retryable transport error (the SDK's RetryConfig strategy="none"
    semantics) and surfaces the failure in the attempt outcome — the repair's
    TOTAL duration is bounded by its budget-derived timeout."""
    transport = InternallyRetryingTransport()
    client = make_client(transport)
    outcome = client.complete_once(messages=turn_messages(),
                                   prompt_hash="sha256:x",
                                   attempt=2, kind="repair", timeout_ms=55_000,
                                   disable_transport_retries=True)
    assert transport.network_calls == 1  # NO internal transport retry
    assert outcome.error is not None
    assert "connection reset" in outcome.error
    assert outcome.batch is None and outcome.raw is None
    # The failed repair attempt is logged like every other call.
    assert client.call_logs[-1].kind == "repair"
    assert client.call_logs[-1].transport_retries == 0

    # Attempt 1 (retries NOT disabled) still retries transparently: one
    # logical attempt, two network calls, the retry only logged.
    transport = InternallyRetryingTransport()
    client = make_client(transport)
    outcome = client.complete_once(messages=turn_messages(),
                                   prompt_hash="sha256:x",
                                   attempt=1, kind="initial")
    assert transport.network_calls == 2
    assert outcome.error is None and outcome.batch is not None
    assert outcome.raw.transport_retries == 1
    assert client.call_logs[-1].transport_retries == 1


# ---------------------------------------------------------------------------
# Enforcement paths (decision 4: both behind one interface)
# ---------------------------------------------------------------------------


def test_json_schema_path_sends_strict_structured_output_request():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport, enforcement="json_schema")
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    response_format = transport.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "action_batch"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema_definition"]
    assert schema == ActionBatch.model_json_schema()
    # the ref field is constrained to candidate IDs
    ref_pattern = schema["$defs"]["ActionItem"]["properties"]["ref"]["pattern"]
    assert ref_pattern == CANDIDATE_ID_FIELD_PATTERN


def test_validated_parse_path_requests_json_object():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)  # default = provisional validated_parse
    assert client.enforcement is EnforcementMode.VALIDATED_PARSE
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    assert transport.calls[0]["response_format"] == {"type": "json_object"}


def test_validated_parse_extracts_fenced_json():
    fenced = "Here is my turn:\n```json\n" + VALID_BATCH + "\n```\nDone."
    transport = MockTransport(outcomes=[raw_completion(fenced)])
    client = make_client(transport)
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.batch.turn_id == "run1/t1"
    assert result.attempts_used == 1


def test_validated_parse_extracts_json_from_prose():
    prose = "Sure! {" + VALID_BATCH[1:-1] + "} hope that helps"
    transport = MockTransport(outcomes=[raw_completion(prose)])
    client = make_client(transport)
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.batch.turn_id == "run1/t1"


def test_raw_coordinate_ref_is_schema_error_then_repair():
    bad = json.dumps({"turn_id": "run1/t1",
                      "actions": [{"ref": "(12, 34)"}]})
    transport = MockTransport(outcomes=[raw_completion(bad),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    result = client.action_batch(
        messages=turn_messages(), prompt_hash="sha256:test",
        attempts=TurnAttempts(),
        repair_candidates_for=lambda rr: "FULL TABLE: unit:7:goto_n")
    assert result.batch.actions[0].ref == "unit:7:goto_n"
    assert result.repairs_used == 1
    repair_user = transport.calls[1]["messages"][-1]["content"]
    assert "SCHEMA ERROR" in repair_user
    assert "candidate IDs" in repair_user


def test_both_enforcement_paths_identical_semantics():
    """Both paths produce identical transcript semantics (decision 4)."""
    batches = []
    for enforcement in ("json_schema", "validated_parse"):
        transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
        client = make_client(transport, enforcement=enforcement)
        result = client.action_batch(messages=turn_messages(),
                                     prompt_hash="sha256:test",
                                     attempts=TurnAttempts())
        batches.append(result.batch)
    assert batches[0] == batches[1]


def test_parse_action_batch_rejects_empty_and_non_json():
    with pytest.raises(SchemaError):
        mc.parse_action_batch("")
    with pytest.raises(SchemaError):
        mc.parse_action_batch("   ")
    with pytest.raises(SchemaError):
        mc.parse_action_batch("I move my unit north. No JSON here.")


def test_parse_action_batch_rejects_extra_fields():
    with pytest.raises(SchemaError) as excinfo:
        mc.parse_action_batch(json.dumps({
            "turn_id": "t", "actions": [], "move_unit": {"id": 7}}))
    assert excinfo.value.detail


def test_action_batch_max_actions_enforced():
    too_many = json.dumps({
        "turn_id": "t",
        "actions": [{"ref": "unit:7:goto_n"}] * (ACTION_BATCH_MAX_ACTIONS + 1),
    })
    with pytest.raises(SchemaError):
        mc.parse_action_batch(too_many)


# ---------------------------------------------------------------------------
# Repair prompts per error class (§7.5, §3 taxonomy)
# ---------------------------------------------------------------------------


def _ref_validator(error_class, failed_refs, detail):
    """A runner-style ref validator: fails only while the batch still
    references one of the failed refs (the refreshed table no longer
    contains them, so a conforming repair output passes)."""
    failed = set(failed_refs)

    def validate(batch):
        if any(item.ref in failed for item in batch.actions):
            return RepairRequest(error_class=error_class, detail=detail,
                                 failed_refs=failed_refs)
        return None
    return validate


REPAIRED_BATCH = json.dumps({
    "turn_id": "run1/t1",
    "actions": [{"ref": "unit:7:goto_s"}],  # the refreshed candidate
    "commitments": [],
    "finish_requested": True,
    "notes": "repaired",
})

ILLEGAL_BATCH = json.dumps({
    "turn_id": "run1/t1",
    "actions": [{"ref": "city:3:produce_settlers"}],  # became illegal
    "commitments": [],
    "finish_requested": False,
    "notes": "produce settlers",
})


def test_state_fog_error_repair_prompt_with_refreshed_candidates():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH),
                                        raw_completion(REPAIRED_BATCH)])
    client = make_client(transport)
    result = client.action_batch(
        messages=turn_messages(), prompt_hash="sha256:test",
        attempts=TurnAttempts(),
        ref_validator=_ref_validator(
            "state_fog_error", ["unit:7:goto_n"],
            "ref unit:7:goto_n is not in the current candidate table"),
        repair_candidates_for=lambda rr: (
            "REFRESHED: unit:7:goto_s (the unit can still move south)"))
    assert result.repairs_used == 1
    assert result.batch.actions[0].ref == "unit:7:goto_s"
    repair_user = transport.calls[1]["messages"][-1]["content"]
    assert "STATE-FOG ERROR" in repair_user
    assert "unit:7:goto_n" in repair_user
    assert "REFRESHED: unit:7:goto_s" in repair_user


def test_illegal_in_rule_repair_prompt():
    transport = MockTransport(outcomes=[raw_completion(ILLEGAL_BATCH),
                                        raw_completion(REPAIRED_BATCH)])
    client = make_client(transport)
    result = client.action_batch(
        messages=turn_messages(), prompt_hash="sha256:test",
        attempts=TurnAttempts(),
        ref_validator=_ref_validator(
            "illegal_in_rule", ["city:3:produce_settlers"],
            "the city already changed production"),
        repair_candidates_for=lambda rr: "REFRESHED: city:3:produce_warriors")
    assert result.repairs_used == 1
    repair_user = transport.calls[1]["messages"][-1]["content"]
    assert "ILLEGAL-IN-RULE" in repair_user
    assert "city:3:produce_settlers" in repair_user
    assert "REFRESHED: city:3:produce_warriors" in repair_user


def test_ref_validator_failure_on_repair_exhausts_budget():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    with pytest.raises(AttemptBudgetExhausted) as excinfo:
        client.action_batch(
            messages=turn_messages(), prompt_hash="sha256:test",
            attempts=TurnAttempts(),
            ref_validator=_ref_validator("state_fog_error", ["unit:7:goto_n"],
                                         "still unknown"))
    assert "state_fog_error" in str(excinfo.value)
    assert len(transport.calls) == 2


def test_repair_schema_failure_on_repair_exhausts_budget():
    transport = MockTransport(outcomes=[raw_completion("garbage"),
                                        raw_completion("still garbage")])
    client = make_client(transport)
    with pytest.raises(AttemptBudgetExhausted) as excinfo:
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    assert "schema_error" in str(excinfo.value)
    assert len(transport.calls) == 2


# ---------------------------------------------------------------------------
# Per-call logging (decision 4: tokens, latency, cost; decision 8: fingerprint)
# ---------------------------------------------------------------------------


def test_per_call_logging_fields():
    transport = MockTransport(outcomes=[raw_completion(
        VALID_BATCH, prompt_tokens=1000, completion_tokens=200,
        response_id="resp-xyz", finish_reason="stop")])
    client = make_client(transport)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    entry = client.call_logs[0].to_dict()
    assert entry["tokens_in"] == 1000
    assert entry["tokens_out"] == 200
    assert entry["tokens_total"] == 1200
    assert isinstance(entry["latency_ms"], int) and entry["latency_ms"] >= 0
    # cost from the provisional pricing table ($0.50/$1.50 per MTok)
    assert entry["cost_usd"] == round((1000 * 0.50 + 200 * 1.50) / 1e6, 6)
    assert entry["model_requested"] == MODEL_ID
    assert entry["model_responded"] == MODEL_ID
    assert entry["prompt_hash"] == "sha256:test"
    assert entry["enforcement"] == "validated_parse"
    assert entry["fingerprint"] == "unprobed"  # before the arm probe
    assert entry["finish_reason"] == "stop"
    assert entry["response_id"] == "resp-xyz"
    assert entry["error"] is None
    assert entry["kind"] == "initial"
    assert entry["attempt"] == 1


def test_call_log_to_dict_has_no_secret_fields():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    entry = client.call_logs[0].to_dict()
    blob = json.dumps(entry)
    assert CANARY_KEY not in blob
    assert "api_key" not in entry
    assert "key" not in {k.lower() for k in entry}


def test_no_secret_in_any_log_line_or_sink(monkeypatch):
    """The canary key never appears in any serialized call log, nor in the
    lines a log_sink receives (decision 4: never logged)."""
    monkeypatch.setenv(mc.API_KEY_ENV_VAR, CANARY_KEY)
    lines = []

    def sink(record):
        lines.append(json.dumps(record.to_dict(), sort_keys=True))

    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH),
                                        raw_completion("garbage"),
                                        raw_completion(VALID_BATCH),
                                        TransportError("boom"),
                                        TransportError("boom again")])
    client = MistralClient(transport=transport, log_sink=sink)
    # a successful turn (1 call)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    # a repaired turn (2 calls)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    # a transport-failing turn (2 calls, both fail)
    with pytest.raises(AttemptBudgetExhausted):
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    blob = "\n".join(lines) + json.dumps(
        [l.to_dict() for l in client.call_logs])
    assert CANARY_KEY not in blob
    assert len(transport.calls) == 5
    assert len(lines) == len(client.call_logs) == 5


def test_fingerprint_travels_with_call_logs_after_probe():
    transport = MockTransport(outcomes=[
        raw_completion("LECHONK-FINGERPRINT-7f3a91c2"),
        raw_completion(VALID_BATCH),
    ])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    assert client.call_logs[-1].fingerprint == probe.fingerprint
    assert client.call_logs[-1].fingerprint != "unprobed"


# ---------------------------------------------------------------------------
# Fingerprint probe (decision 8) + P10 probes (mock transport)
# ---------------------------------------------------------------------------


def test_fingerprint_probe_deterministic_and_fixed():
    content = "LECHONK-FINGERPRINT-7f3a91c2"
    transport = MockTransport(outcomes=[raw_completion(content)])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    expected = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert probe.fingerprint == expected
    assert probe.output_hash == expected.removeprefix("sha256:")
    # deterministic sampling: temperature 0, no response_format
    assert transport.calls[0]["temperature"] == 0.0
    assert transport.calls[0]["response_format"] is None
    # the fixed probe prompt (versioned file) with its recorded hash
    assert probe.prompt_hash == prompts.prompt_hash(prompts.FINGERPRINT_PROBE)
    assert probe.model_requested == MODEL_ID
    assert probe.model_responded == MODEL_ID
    # cached on the client and carried by later call logs
    assert client.arm_fingerprint == expected
    # probe calls are logged too (decision 4: every call is logged)
    assert client.call_logs[0].kind == "probe"
    assert client.call_logs[0].attempt == 0


def test_fingerprint_probe_transport_error_propagates():
    transport = MockTransport(outcomes=[TransportError("no route")])
    client = make_client(transport)
    with pytest.raises(TransportError):
        client.probe_fingerprint()
    assert client.arm_fingerprint is None


def test_probe_model_id_mock_success():
    transport = MockTransport(outcomes=[raw_completion("OK")])
    client = make_client(transport)
    probe = client.probe_model_id()
    assert probe.responded is True
    assert probe.model_requested == MODEL_ID
    assert probe.model_responded == MODEL_ID
    assert probe.error is None
    assert client.call_logs[0].kind == "probe"


def test_probe_model_id_mock_transport_failure_captured():
    transport = MockTransport(outcomes=[TransportError("404 model not found")])
    client = make_client(transport)
    probe = client.probe_model_id()
    assert probe.responded is False
    assert probe.model_responded is None
    assert "404" in probe.error


def test_probe_json_schema_support_mock_supported():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    probe = client.probe_json_schema_support(probe_prompt="return a batch")
    assert probe.responded is True
    assert probe.json_schema_supported is True
    assert probe.response_format_type == "json_schema"
    assert probe.strict is True
    assert probe.schema_hash == mc.action_batch_schema_hash()
    assert probe.response_shape["valid_action_batch"] is True
    assert probe.response_shape["top_level_keys"] == sorted(
        json.loads(VALID_BATCH).keys())
    # the request carried the real ActionBatch schema, strict
    sent = transport.calls[0]["response_format"]
    assert sent == {
        "type": "json_schema",
        "json_schema": {
            "name": "action_batch",
            "description": sent["json_schema"]["description"],
            "schema_definition": mc.action_batch_json_schema(),
            "strict": True,
        },
    }
    assert transport.calls[0]["temperature"] == 0.0


def test_probe_json_schema_support_mock_unsupported():
    transport = MockTransport(
        outcomes=[raw_completion(json.dumps({"unexpected": True}))])
    client = make_client(transport)
    probe = client.probe_json_schema_support(probe_prompt="return a batch")
    assert probe.responded is True
    assert probe.json_schema_supported is False
    assert probe.response_shape["valid_action_batch"] is False
    assert "validation_error" in probe.response_shape


def test_probe_json_schema_support_mock_transport_failure():
    transport = MockTransport(outcomes=[TransportError("503")])
    client = make_client(transport)
    probe = client.probe_json_schema_support(probe_prompt="return a batch")
    assert probe.responded is False
    assert probe.json_schema_supported is False
    assert "503" in probe.error


# ---------------------------------------------------------------------------
# Schema hash + spike decision record (decision 4/8, run manifest)
# ---------------------------------------------------------------------------


def test_action_batch_schema_hash_stable_and_derived():
    schema = ActionBatch.model_json_schema()
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    expected = "sha256:" + hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    assert mc.action_batch_schema_hash() == expected
    assert mc.action_batch_schema_hash() == mc.action_batch_schema_hash()
    assert mc.action_batch_json_schema() == schema


# ---------------------------------------------------------------------------
# strict JSON emission in the schema hash (FW-19)
# ---------------------------------------------------------------------------

def test_action_batch_schema_hash_nan_raises(monkeypatch):
    """The schema hash canonicalizes via strict_dumps: a NaN anywhere in
    the payload raises ValueError deterministically — a bare NaN token
    can never enter the hashed canonical form. (The lru_cache wrapper is
    bypassed: earlier tests may already have warmed it; the schema dict
    is deep-copied: mutating the cached dict would leak into later
    tests.)"""
    schema = copy.deepcopy(mc.action_batch_json_schema())
    schema["properties"]["__probe__"] = float("nan")
    monkeypatch.setattr(mc, "action_batch_json_schema", lambda: schema)
    monkeypatch.setattr(mc, "action_batch_schema_hash",
                        mc.action_batch_schema_hash.__wrapped__)
    with pytest.raises(ValueError):
        mc.action_batch_schema_hash()


def test_action_batch_schema_hash_lone_surrogate_escaped(monkeypatch):
    """A lone surrogate in the schema is escaped as \\udXXX in the
    canonical form — the utf-8 encode inside the hash never sees a raw
    surrogate, and the hash stays deterministic (deep copy: the cached
    schema dict is shared across tests)."""
    schema = copy.deepcopy(mc.action_batch_json_schema())
    schema["description"] = "lone \ud800 surrogate"
    monkeypatch.setattr(mc, "action_batch_json_schema", lambda: schema)
    monkeypatch.setattr(mc, "action_batch_schema_hash",
                        mc.action_batch_schema_hash.__wrapped__)
    first = mc.action_batch_schema_hash()
    assert first.startswith("sha256:")
    assert mc.action_batch_schema_hash() == first


def test_spike_decision_provisional_while_blocked():
    decision = mc.spike_decision()
    assert decision["probe"] == "P10"
    assert decision["model_id"] == MODEL_ID
    assert decision["paths_implemented"] == ["json_schema",
                                             "validated_parse"]
    assert decision["decision"] == "validated_parse"
    assert decision["decision_status"] == "provisional_live_probe_blocked"
    assert decision["json_schema_supported_live"] is None
    assert decision["evidence_artifact"] == mc.SPIKE_EVIDENCE_ARTIFACT


@pytest.mark.parametrize("supported,expected", [
    (True, "json_schema"),
    (False, "validated_parse"),
])
def test_spike_decision_with_live_evidence(supported, expected):
    decision = mc.spike_decision(live_probe_done=True,
                                 json_schema_supported=supported)
    assert decision["decision"] == expected
    assert decision["decision_status"] == "decided_live"
    assert decision["json_schema_supported_live"] is supported


# ---------------------------------------------------------------------------
# Package purity (plan §7.1: model is MIT and civrealm-free)
# ---------------------------------------------------------------------------


def test_importing_model_package_never_imports_civrealm():
    probe = (
        "import sys\n"
        "import lechonk.model\n"
        "import lechonk.model.mistral_client\n"
        "import lechonk.model.prompts\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m == 'civrealm' or m.startswith('civrealm.'))\n"
        "print('CIVREALM_MODULES:', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(REPO))
    assert proc.returncode == 0, (
        f"lechonk.model transitively imported civrealm:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "CIVREALM_MODULES: []" in proc.stdout


def test_model_sources_never_import_civrealm():
    import_re = re.compile(r"^\s*(import|from)\s+civrealm\b", re.MULTILINE)
    offenders = []
    for path in sorted(MODEL_DIR.rglob("*.py")):
        if import_re.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert not offenders, f"model files importing civrealm: {offenders}"


def test_model_package_declares_mit_spdx():
    for path in sorted(MODEL_DIR.rglob("*.py")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "# SPDX-License-Identifier: MIT", (
            f"{path} lost its MIT SPDX header")
