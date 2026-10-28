# SPDX-License-Identifier: MIT
"""Offline tests for the ModelClient-protocol adapter over the Mistral
client (src/lechonk/model/turn_client.py) and the single-attempt seam
(``MistralClient.complete_once``) — the WP2 package C integration seam.

A mock transport drives the real client (no network, no SDK import): the
message construction (§7.5), the three outcome mappings (valid batch /
schema error / transport error), the repair-prompt construction, the per-call
API logging (decision 4: tokens, latency, cost + the rendered section sizes
for the §3 budget table), and the no-secret assertion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import lechonk.model.turn_client as turn_client_module
import lechonk.model.conversation as conversation_module

from lechonk.core.model_client import (
    REPAIR_TIMEOUT_MARGIN_S,
    RepairPrompt,
    TurnPrompt,
)
from lechonk.core.schemas import estimate_tokens
from lechonk.model import mistral_client as mc
from lechonk.model import prompts
from lechonk.model.prompts import (
    REPAIR_CONVERSATION_TOKEN_BUDGET,
    PromptBudgetExceeded,
    repair_conversation_budget,
)
from lechonk.model.turn_client import (
    MistralTurnClient,
    COMPACTION_CONTEXT_WINDOW_TOKENS,
    COMPACTION_OUTPUT_HEADROOM_TOKENS,
    COMPACTION_INPUT_GUARD_TOKENS,
    _check_repair_conversation_budget,
    _prompt_hash,
    _raw_attempts_from,
    _turn_number,
)
from lechonk.model.conversation import ConversationConfig, ConversationManager, ConversationError

CANARY = "sk-CANARY-9f3a91c2-do-not-leak"


def test_v3_repair_budget_allows_expanded_candidate_surface():
    messages = [{"role": "user", "content": "x" * 48_000}]
    with pytest.raises(PromptBudgetExceeded):
        _check_repair_conversation_budget(
            messages, repair_conversation_budget("gameplay_v2",
                                                 context_policy="legacy"))
    _check_repair_conversation_budget(
        messages, repair_conversation_budget("gameplay_v3",
                                             context_policy="conversation_v1"))

VALID_BATCH = json.dumps({
    "turn_id": "run/t1",
    "actions": [{"ref": "unit:1:goto_n"}],
    "commitments": [],
    "finish_requested": True,
    "notes": "move north",
})

MALFORMED_BATCH = json.dumps({
    "turn_id": "run/t1",
    "actions": [{"ref": "move unit 10 one tile north"}],  # raw free text
    "commitments": [],
    "finish_requested": True,
})


@dataclass
class MockTransport:
    outcomes: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, *, messages, model, temperature, reasoning_effort,
                 max_tokens, response_format, random_seed,
                 timeout_ms=None, disable_transport_retries=False,
                 prompt_cache_key=None, tools=None, tool_choice=None,
                 parallel_tool_calls=None):
        self.calls.append({
            "messages": messages, "model": model,
            "temperature": temperature, "max_tokens": max_tokens,
            "response_format": response_format, "random_seed": random_seed,
            "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
            "prompt_cache_key": prompt_cache_key,
            "tools": tools, "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def raw(content, **kw):
    return mc.RawCompletion(
        content=content, model=mc.MODEL_ID,
        response_id=kw.get("response_id", "resp-mock"), created=0,
        finish_reason=kw.get("finish_reason", "stop"),
        prompt_tokens=kw.get("prompt_tokens", 1000),
        completion_tokens=kw.get("completion_tokens", 200),
        total_tokens=(kw.get("prompt_tokens", 1000)
                      + kw.get("completion_tokens", 200)),
        transport_retries=kw.get("transport_retries", 0),
        reasoning_text=kw.get("reasoning_text", ""),
        answer_text=kw.get("answer_text", content),
        content_chunks=kw.get("content_chunks"),
        tool_calls=kw.get("tool_calls"),
        assistant_message=kw.get("assistant_message"))


def make_client(outcomes, *, enforcement="json_schema"):
    transport = MockTransport(outcomes=outcomes)
    client = mc.MistralClient(api_key=CANARY, transport=transport,
                              enforcement=enforcement)
    return client, transport


def test_compaction_request_has_no_local_output_cap_and_records_early_failure():
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    seen = {}

    def complete(**kwargs):
        seen.update(kwargs)
        return raw('{"version":1,"summary":"ok","commitments":[],"uncertainties":[],"facts":[]}')

    client._complete = complete
    result = adapter._compact_with_model([{"role": "user", "content": "short"}])
    assert result["version"] == 1
    assert "max_tokens" not in seen

    huge = "x" * 4_100_000
    with pytest.raises(ConversationError, match="context headroom"):
        adapter._compact_with_model([{"role": "user", "content": huge}])
    assert adapter.call_records[-1]["outcome"] == "error"
    assert adapter.call_records[-1]["error"]


def test_capacity_compaction_fails_closed_without_mutating_history():
    capacity = mc.ProviderCapacityError("backend capacity", retry_after="9")
    client, _ = make_client([capacity])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    manager = ConversationManager(system_message={"role": "system", "content": "s"})
    manager.messages.append({"role": "user", "content": "history"})
    before = list(manager.messages)
    adapter._conversation_manager = manager

    with pytest.raises(mc.ProviderCapacityError):
        adapter._compact_with_model(manager.messages)

    assert manager.messages == before
    assert adapter.call_records[-1]["error_kind"] == "provider_capacity"
    assert adapter.call_records[-1]["http_status"] == 429
    assert adapter.call_records[-1]["provider_code"] == "3505"


def test_compaction_carries_triggering_turn_and_logical_call_id():
    events = []
    client, transport = make_client([raw(
        '{"version":1,"summary":"ok","commitments":[],'
        '"uncertainties":[],"facts":[]}' )])
    client.call_event_callback = events.append
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    adapter._active_turn_number = 17
    adapter._active_logical_call_id = "run/t17"
    assert adapter._compact_with_model([{"role": "user", "content": "history"}])
    record = adapter.call_records[-1]
    assert record["turn"] == 17
    assert record["logical_call_id"] == "run/t17"
    assert record["call_id"] == client.call_logs[-1].call_id
    started = next(event for event in events
                   if event["event"] == "model_request_started")
    assert started["physical_kind"] == "conversation_compaction"
    assert started["turn"] == 17
    assert started["logical_call_id"] == "run/t17"


def test_compaction_headroom_constants_leave_summary_margin():
    assert COMPACTION_CONTEXT_WINDOW_TOKENS > COMPACTION_OUTPUT_HEADROOM_TOKENS


def test_compaction_headroom_boundary_is_checked_before_transport(monkeypatch):
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    client._complete = lambda **kwargs: raw(
        '{"version":1,"summary":"ok","commitments":[],"uncertainties":[],"facts":[]}')
    # The compaction request contains the historical message plus its final
    # instruction.  Make that pair land exactly on the available boundary.
    monkeypatch.setattr(turn_client_module, "estimate_tokens", lambda _: 463_904)
    adapter._compact_with_model([{"role": "user", "content": "history"}])
    monkeypatch.setattr(turn_client_module, "estimate_tokens", lambda _: 463_905)
    with pytest.raises(ConversationError, match="context headroom"):
        adapter._compact_with_model([{"role": "user", "content": "history"}])


def test_compaction_verified_provider_prefix_wires_transport_and_guard(monkeypatch):
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    manager = ConversationManager(system_message={"role": "system", "content": "s"})
    manager.messages.extend({"role": "user", "content": str(i)} for i in range(143))
    def shaped(messages):
        count = len(list(messages))
        return {144: 1_113_200, 137: 1_053_001, 1: 77}.get(count, count)
    monkeypatch.setattr(conversation_module, "_message_tokens", shaped)
    manager.record_usage(
        500_595, estimated_tokens=1_113_200, message_count=144,
        request_hash="sha256:" + hashlib.sha256(
            conversation_module.canonical_json(manager.messages[:144]).encode("utf-8")
        ).hexdigest())
    adapter._conversation_manager = manager
    seen = {}
    client._complete = lambda **kwargs: (seen.update(kwargs) or raw(
        '{"version":1,"summary":"ok","commitments":[],"uncertainties":[],"facts":[]}'))
    adapter._compact_with_model(manager.messages[:137])
    assert seen["messages"][-1]["role"] == "user"
    assert "max_tokens" not in seen
    record = adapter.call_records[-1]
    assert record["outcome"] == "ok"
    assert record["preflight"]["method"] == "provider_prefix"
    assert record["preflight"]["guard"] == COMPACTION_INPUT_GUARD_TOKENS


def test_compaction_deadline_and_invalid_json_are_logged_as_errors():
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                clock=lambda: 10.0)
    adapter._turn_deadline = 10.5
    with pytest.raises(TimeoutError, match="budget exhausted"):
        adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert adapter.call_records[-1]["outcome"] == "error"
    assert "budget exhausted" in adapter.call_records[-1]["error"]

    adapter._turn_deadline = None
    client._complete = lambda **kwargs: raw("not-json")
    with pytest.raises(ConversationError, match="invalid JSON"):
        adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert adapter.call_records[-1]["outcome"] == "error"
    assert adapter.call_records[-1]["error"] == "compaction returned invalid JSON"


def test_compaction_typed_commitment_semantic_error_is_logged(monkeypatch):
    client, _ = make_client([])
    events = []
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                output_callback=events.append)
    client._complete = lambda **kwargs: raw(json.dumps({
        "version": 1, "summary": "ok",
        "commitments": [{"domain": "research", "target": "Writing",
                          "target_kind": "tech", "deadline_turn": 1,
                          "unexpected": True}],
        "uncertainties": [], "facts": []}))
    with pytest.raises(ConversationError, match="invalid typed commitment"):
        adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert adapter.call_records[-1]["outcome"] == "error"
    assert events[-1]["reasoning"] == ""
    assert "unexpected" in events[-1]["answer"]


def test_compaction_semantic_repair_is_one_bounded_second_attempt(monkeypatch):
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    responses = iter([raw("not-json", reasoning_text="first-thought"),
                      raw('{"version":1,"summary":"fixed","commitments":[],'
                           '"uncertainties":[],"facts":[]}')])
    seen = []
    client._complete = lambda **kwargs: (seen.append(kwargs) or next(responses))
    result = adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert result["summary"] == "fixed"
    assert [record["kind"] for record in adapter.call_records] == [
        "conversation_compaction", "conversation_compaction_repair"]
    assert seen[1]["attempt"] == 2 and seen[1]["disable_transport_retries"] is True
    assert "not-json" in seen[1]["messages"][-2]["content"]
    assert seen[1]["messages"][:2] == seen[0]["messages"]
    assert "max_tokens" not in seen[0] and "max_tokens" not in seen[1]
    assert adapter.call_records[0]["raw_attempts"][0]["answer"] == "not-json"
    assert adapter.call_records[1]["raw_attempts"][0]["answer"]


def test_compaction_semantic_repair_failure_stays_atomic_and_bounded(monkeypatch):
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    client._complete = lambda **kwargs: raw("still-invalid")
    with pytest.raises(ConversationError, match="invalid JSON"):
        adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert len(adapter.call_records) == 2
    assert adapter.call_records[1]["kind"] == "conversation_compaction_repair"


def test_compaction_transport_errors_prefer_partial_physical_call_id():
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    wanted = object()
    client.call_logs.extend([object(), wanted])
    client._call_log_for = lambda call_id: wanted if call_id == "physical-2" else None
    exc = RuntimeError("transport")
    exc.partial = type("Partial", (), {"call_id": "physical-2"})()
    assert adapter._compaction_error_call_log(exc, 0) is wanted


def test_compaction_repair_result_uses_exact_physical_call_log():
    client, _ = make_client([])
    adapter = MistralTurnClient(client, context_policy="conversation_v1")
    repaired = raw('{"version":1,"summary":"fixed","commitments":[],"uncertainties":[],"facts":[]}')
    repaired.call_id = "physical-2"
    log = SimpleNamespace(call_id="physical-2", prompt_hash="p",
                          model_requested="m", model_responded="m",
                          enforcement=None, temperature=None, latency_ms=1,
                          tokens_in=1, tokens_out=1, tokens_total=2,
                          cached_tokens=0, cost_usd=0, finish_reason="stop",
                          response_id="r", fingerprint=None, transport_retries=0,
                          error=None, ts=0)
    client._call_log_for = lambda call_id: log if call_id == "physical-2" else None
    responses = iter([raw("bad"), repaired])
    client._complete = lambda **kwargs: next(responses)
    adapter._compact_with_model([{"role": "user", "content": "history"}])
    assert adapter.call_records[-1]["call_id"] == "physical-2"
    assert adapter.call_records[-1]["kind"] == "conversation_compaction_repair"


def turn_prompt(**kw):
    defaults = dict(
        call_id="run/t1/call1", turn_id="run/t1",
        memory="memory text", snapshot="snapshot text",
        delta="delta text", candidates="unit:1:goto_n — move north",
        budget_s=100.0)
    defaults.update(kw)
    return TurnPrompt(**defaults)


# ---------------------------------------------------------------------------
# complete_once (the single-attempt seam)
# ---------------------------------------------------------------------------


def test_complete_once_ok_schema_error_and_transport_error():
    messages = [{"role": "user", "content": "hi"}]

    client, _ = make_client([raw(VALID_BATCH)])
    ok = client.complete_once(messages=messages, prompt_hash="sha256:x",
                              attempt=1, kind="initial")
    assert ok.error is None and ok.malformed is None
    assert ok.batch is not None and ok.batch.turn_id == "run/t1"
    assert ok.raw_content == VALID_BATCH
    assert len(client.call_logs) == 1

    client, _ = make_client([raw(MALFORMED_BATCH)])
    bad = client.complete_once(messages=messages, prompt_hash="sha256:x",
                               attempt=1, kind="initial")
    assert bad.error is None and bad.batch is None
    assert bad.malformed is not None and "ref" in bad.malformed
    assert bad.raw_content == MALFORMED_BATCH

    client, _ = make_client([mc.TransportError("request timed out")])
    err = client.complete_once(messages=messages, prompt_hash="sha256:x",
                               attempt=1, kind="initial")
    assert err.error is not None and "timed out" in err.error
    assert err.batch is None and err.raw is None
    # The failed attempt is logged too (decision 4: every call is logged).
    assert client.call_logs[-1].error is not None
    assert client.call_logs[-1].tokens_in is None


# ---------------------------------------------------------------------------
# complete_turn: message construction + the outcome mappings
# ---------------------------------------------------------------------------


def test_complete_turn_builds_the_versioned_turn_prompt():
    client, transport = make_client([raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    prompt = turn_prompt()
    resp = adapter.complete_turn(prompt)

    assert resp.error is None and resp.malformed is None
    assert resp.batch is not None and resp.batch.turn_id == "run/t1"
    assert resp.call_id == "run/t1/call1"
    assert resp.tokens_in == 1000 and resp.tokens_out == 200
    assert resp.latency_ms >= 0
    assert resp.fingerprint == "unprobed"

    # The messages: system = the versioned system prompt; user = the turn
    # prompt with the four §7.5 sections in order.
    messages = transport.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompts.system_prompt()
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    for header in ("## MEMORY", "## SNAPSHOT", "## DELTA",
                   "## CANDIDATE TABLE"):
        assert header in user
    assert user.index("## MEMORY") < user.index("## SNAPSHOT") \
        < user.index("## DELTA") < user.index("## CANDIDATE TABLE")
    assert "snapshot text" in user and "delta text" in user
    assert "unit:1:goto_n — move north" in user
    # json_schema enforcement per the WP2 spike decision.
    assert transport.calls[0]["response_format"]["type"] == "json_schema"
    assert transport.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_complete_turn_maps_malformed_and_transport_error():
    client, _ = make_client([raw(MALFORMED_BATCH)])
    adapter = MistralTurnClient(client)
    resp = adapter.complete_turn(turn_prompt())
    assert resp.batch is None
    assert resp.malformed is not None
    # The raw emitted ref is extracted for the schema-error accounting.
    assert resp.raw_attempts == ["move unit 10 one tile north"]

    client, _ = make_client([mc.TransportError("request timed out")])
    adapter = MistralTurnClient(client)
    resp = adapter.complete_turn(turn_prompt())
    assert resp.error is not None and "timed out" in resp.error
    assert resp.batch is None and resp.malformed is None
    assert resp.raw_attempts == []


def test_output_callback_receives_complete_attempt_and_isolation():
    client, _ = make_client([raw(VALID_BATCH)])
    events = []
    adapter = MistralTurnClient(client, output_callback=events.append)
    adapter.complete_turn(turn_prompt())
    assert events[0]["event"] == "model_output"
    assert events[0]["answer"] == VALID_BATCH
    assert events[0]["reasoning"] == ""

    client, _ = make_client([raw(MALFORMED_BATCH)])
    def broken(_event):
        raise RuntimeError("display failed")
    adapter = MistralTurnClient(client, output_callback=broken)
    response = adapter.complete_turn(turn_prompt())
    assert response.malformed is not None


# ---------------------------------------------------------------------------
# complete_repair: the class-specific repair prompt
# ---------------------------------------------------------------------------


def test_complete_repair_appends_the_failed_output_and_repair_prompt():
    client, transport = make_client([raw(MALFORMED_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt())
    repair = RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1",
        memory="memory text", snapshot="snapshot text", delta="delta text",
        candidates="unit:1:goto_n — move north", budget_s=50.0,
        error_class="schema_error", failed_refs=["move unit 10 one tile north"],
        refreshed_candidates="unit:1:goto_n — move north (full table "
                             "re-attached)")
    resp = adapter.complete_repair(repair)

    assert resp.batch is not None
    assert resp.call_id == "run/t1/call2"
    # The repair conversation: original + failed assistant output + the
    # class-specific repair instruction.
    messages = transport.calls[1]["messages"]
    assert len(messages) == 4
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == MALFORMED_BATCH
    assert messages[3]["role"] == "user"
    assert "SCHEMA ERROR" in messages[3]["content"]
    # The schema-error variant re-attaches the full table and carries the
    # malformed detail (it has no $failed_refs placeholder by design: the
    # whole batch is re-emitted).
    assert "unit:1:goto_n — move north (full table re-attached)" \
        in messages[3]["content"]
    assert "violated the ActionBatch schema" in messages[3]["content"]
    # The original conversation is preserved (prompt caching, §7.5).
    assert messages[0]["content"] == prompts.system_prompt()


def test_complete_repair_unresponsive_reasks_the_same_prompt():
    """The unresponsive class (transport-level failure): the repair is a
    bounded re-ask of the SAME turn prompt — MistralClient.action_batch's
    own transport-error behavior (the single repair slot re-asks the turn)."""
    client, transport = make_client(
        [mc.TransportError("request timed out"), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt())
    repair = RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1",
        memory="m", snapshot="s", delta="d", candidates="c",
        error_class="unresponsive", failed_refs=[],
        refreshed_candidates="c")
    resp = adapter.complete_repair(repair)
    assert resp.batch is not None
    messages = transport.calls[1]["messages"]
    original = transport.calls[0]["messages"]
    assert messages == original  # the same turn prompt, re-asked
    assert len(messages) == 2  # no assistant placeholder, no repair suffix


def test_complete_repair_state_fog_uses_the_fog_template():
    client, transport = make_client([raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    # Attempt 1 returns a batch with an invalid ref: the RUNNER classifies it
    # (state_fog_error) and issues the repair; the adapter only renders.
    adapter.complete_turn(turn_prompt())
    repair = RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1",
        memory="m", snapshot="s", delta="d", candidates="c",
        error_class="state_fog_error",
        failed_refs=["unit:424242:goto_n"], refreshed_candidates="c")
    adapter.complete_repair(repair)
    user = transport.calls[1]["messages"][3]["content"]
    assert "STATE-FOG ERROR" in user
    assert "unit:424242:goto_n" in user
    assert "well-formed candidate IDs" in user


def test_complete_repair_before_complete_turn_raises():
    client, _ = make_client([raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    with pytest.raises(RuntimeError, match="before complete_turn"):
        adapter.complete_repair(RepairPrompt(
            call_id="run/t1/call2", turn_id="run/t1", memory="m",
            snapshot="s", delta="d", candidates="c",
            error_class="unresponsive"))


# ---------------------------------------------------------------------------
# budget-aware repair gating: the per-call hard request timeout (WP4)
# ---------------------------------------------------------------------------


def test_repair_attempt_derives_per_call_timeout_from_budget():
    """WP4 gating: the repair attempt's per-call hard request timeout is the
    remaining budget minus the margin; the initial attempt keeps the client's
    configured default (timeout_ms None). The repair's transport-level
    internal retry is DISABLED (the repair's TOTAL duration must stay within
    the budget-derived timeout); attempt 1 keeps transparent retries."""
    client, transport = make_client([raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt(budget_s=100.0))
    adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1",
        memory="memory text", snapshot="snapshot text", delta="delta text",
        candidates="unit:1:goto_n — move north", budget_s=50.0,
        error_class="unresponsive"))
    assert transport.calls[0]["timeout_ms"] is None
    assert transport.calls[0]["disable_transport_retries"] is False
    assert transport.calls[1]["timeout_ms"] == int(
        (50.0 - REPAIR_TIMEOUT_MARGIN_S) * 1000)
    assert transport.calls[1]["disable_transport_retries"] is True


def test_repair_without_budget_keeps_the_default_timeout():
    client, transport = make_client([raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt(budget_s=None))
    adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1", memory="m", snapshot="s",
        delta="d", candidates="c", budget_s=None,
        error_class="unresponsive"))
    assert transport.calls[1]["timeout_ms"] is None
    # No budget to derive a timeout from, but the repair is still a single
    # request attempt: the harness's own attempt structure is the retry
    # policy on the repair path.
    assert transport.calls[1]["disable_transport_retries"] is True


def test_repair_attempt_makes_exactly_one_transport_request():
    """WP4 gating, retry-disabled semantics: a budgeted repair that hits a
    transport-level retryable error makes EXACTLY ONE network request (no
    internal transparent retry) and surfaces the failure on this attempt —
    the observed attempt duration is bounded by the budget-derived timeout.
    The initial attempt still retries transparently inside the transport."""
    from lechonk.model.mistral_client import TransportError

    class RetryHonoringTransport:
        def __init__(self):
            self.calls = []
            self.network_calls = 0

        def complete(self, *, messages, model, temperature, reasoning_effort,
                     max_tokens, response_format, random_seed, timeout_ms=None,
                     disable_transport_retries=False, _retry=False):
            if not _retry:
                self.calls.append({
                    "timeout_ms": timeout_ms,
                    "disable_transport_retries": disable_transport_retries,
                })
            self.network_calls += 1
            if disable_transport_retries:
                # budgeted repair: single request — the retryable failure
                # propagates immediately (no internal retry)
                raise TransportError(
                    "Mistral API transport failure: read timed out")
            # attempt 1: transport-level retry is transparent
            if self.network_calls == 1:
                return self.complete(
                    messages=messages, model=model, temperature=temperature,
                    reasoning_effort=reasoning_effort, max_tokens=max_tokens,
                    response_format=response_format,
                    random_seed=random_seed, timeout_ms=timeout_ms,
                    disable_transport_retries=disable_transport_retries,
                    _retry=True)
            return raw(VALID_BATCH)

    transport = RetryHonoringTransport()
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    adapter = MistralTurnClient(client)
    # Attempt 1: the transport-internal retry succeeds — one logical attempt.
    resp1 = adapter.complete_turn(turn_prompt(budget_s=180.0))
    assert resp1.batch is not None
    assert transport.calls[0]["timeout_ms"] is None
    assert transport.calls[0]["disable_transport_retries"] is False
    # The repair: one network request, the failure surfaces on this attempt.
    resp2 = adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1", memory="m", snapshot="s",
        delta="d", candidates="c", budget_s=60.0,
        error_class="unresponsive"))
    assert resp2.batch is None and resp2.error is not None
    assert "timed out" in resp2.error
    assert transport.calls[1]["timeout_ms"] == int(
        (60.0 - REPAIR_TIMEOUT_MARGIN_S) * 1000)
    assert transport.calls[1]["disable_transport_retries"] is True
    assert client.call_logs[-1].kind == "repair"
    assert client.call_logs[-1].transport_retries == 0
    assert client.call_logs[-1].error is not None


# ---------------------------------------------------------------------------
# per-call API logging (decision 4) + the §3 budget-table inputs
# ---------------------------------------------------------------------------


def test_per_call_api_log_records_and_section_tokens():
    records = []
    client, _ = make_client([raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client, api_log=records.append)
    prompt = turn_prompt()
    adapter.complete_turn(prompt)
    adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1",
        memory=prompt.memory, snapshot=prompt.snapshot,
        delta=prompt.delta, candidates=prompt.candidates,
        error_class="unresponsive"))

    assert len(records) == 2
    first, second = records
    # Decision 4: tokens, latency, cost per call — the API-reported counts
    # are the ground truth.
    for record in (first, second):
        assert record["tokens_in"] == 1000
        assert record["tokens_out"] == 200
        assert record["tokens_total"] == 1200
        assert isinstance(record["latency_ms"], int)
        assert record["cost_usd"] is None
        assert record["cost_status"] == "unknown"
        assert record["model_requested"] == mc.MODEL_ID
        assert record["model_responded"] == mc.MODEL_ID
        assert record["enforcement"] == "json_schema"
        assert record["fingerprint"] == "unprobed"
        assert record["error"] is None
        assert record["prompt_hash"].startswith("sha256:")
    assert first["attempt"] == 1 and first["kind"] == "initial"
    assert first["outcome"] == "ok"
    assert second["attempt"] == 2 and second["kind"] == "repair"
    assert first["turn"] == 1 and second["turn"] == 1
    assert first["call_id"] == "run/t1/call1"
    # The rendered section sizes for the §3 budget table.
    assert first["section_tokens"] == {
        "memory": estimate_tokens("memory text"),
        "snapshot": estimate_tokens("snapshot text"),
        "delta": estimate_tokens("delta text"),
        "candidates": estimate_tokens("unit:1:goto_n — move north"),
    }
    # The records are JSON-serializable (the run persists them as JSONL).
    json.dumps(records)


def test_outcome_and_error_fields_in_the_api_log():
    records = []
    client, _ = make_client(
        [raw(MALFORMED_BATCH), mc.TransportError("request timed out")])
    adapter = MistralTurnClient(client, api_log=records.append)
    adapter.complete_turn(turn_prompt())
    adapter.complete_turn(turn_prompt())
    assert records[0]["outcome"] == "schema_error"
    assert records[0]["raw_attempts"] == ["move unit 10 one tile north"]
    assert records[1]["outcome"] == "transport_error"
    assert records[1]["error"] is not None
    assert records[1]["tokens_in"] is None


def test_capacity_exception_is_promoted_to_turn_api_log_without_repair():
    records = []
    client, _ = make_client([mc.ProviderCapacityError("capacity")])
    adapter = MistralTurnClient(client, api_log=records.append)

    with pytest.raises(mc.ProviderCapacityError):
        adapter.complete_turn(turn_prompt())

    assert len(records) == 1
    record = records[0]
    assert record["outcome"] == "provider_capacity"
    assert record["error_kind"] == "provider_capacity"
    assert record["http_status"] == 429
    assert record["provider_type"] == "backend_out_of_capacity"
    assert record["provider_code"] == "3505"
    assert record["retry_after"] is None


def test_no_secret_in_any_serialized_record():
    records = []
    client, _ = make_client(
        [raw(MALFORMED_BATCH), raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client, api_log=records.append)
    adapter.complete_turn(turn_prompt())
    adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call2", turn_id="run/t1", memory="m", snapshot="s",
        delta="d", candidates="c", error_class="schema_error",
        failed_refs=["bad"]))
    blob = json.dumps(records) + json.dumps(
        [l.to_dict() for l in client.call_logs])
    assert CANARY not in blob


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_turn_number_parsing():
    assert _turn_number("wp2-model-run/t7") == 7
    assert _turn_number("run/t1") == 1
    assert _turn_number("no-turn-here") is None
    assert _turn_number("") is None


def test_raw_attempts_extraction():
    # A fenced JSON block inside prose.
    fenced = ('Sure! Here is my move:\n```json\n{"turn_id": "t", '
              '"actions": [{"ref": "unit:1:goto_n"}, {"ref": "(3,4)"}]}'
              '\n```\nthanks')
    assert _raw_attempts_from(fenced) == ["unit:1:goto_n", "(3,4)"]
    # Prose-wrapped JSON (no fences).
    wrapped = 'blah {"actions": [{"ref": "city:2:produce_warriors"}]} blah'
    assert _raw_attempts_from(wrapped) == ["city:2:produce_warriors"]
    # A JSON object without an actions list.
    assert _raw_attempts_from('{"turn_id": "t"}') == []
    # No JSON at all.
    assert _raw_attempts_from("I move my unit north") == []
    # Non-dict actions items stringify.
    assert _raw_attempts_from('{"actions": ["unit:1:goto_n"]}') == \
        ["unit:1:goto_n"]


def test_prompt_hash_clean_ids_unaffected():
    # C5: the regex end-anchor fix does not disturb the hash path for clean ids.
    messages = [{"role": "user", "content": "unit:7:goto_n"}]
    expected = "sha256:" + hashlib.sha256(json.dumps(
        messages, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    assert _prompt_hash(messages) == expected
    # Clean turn ids still parse; a newline-terminated id no longer does.
    assert _turn_number("run/t7") == 7
    assert _turn_number("run/t7\n") is None


# ---------------------------------------------------------------------------
# FW-18: the strict canonical-JSON rule in the prompt-hash path
# ---------------------------------------------------------------------------

def test_prompt_hash_nan_raises_deterministically():
    # C9b: NaN/Infinity are rejected (allow_nan=False) — a deterministic
    # ValueError, never a bare NaN/Infinity token inside the hashed text.
    with pytest.raises(ValueError):
        _prompt_hash([{"role": "user", "content": float("nan")}])
    with pytest.raises(ValueError):
        _prompt_hash([{"role": "user", "content": float("inf")}])
    with pytest.raises(ValueError):
        _prompt_hash([{"role": "user", "content": float("-inf")}])


def test_prompt_hash_lone_surrogate_escaped_not_unicode_encode_error():
    # C9b: a lone surrogate never reaches the utf-8 encoder — the strict
    # helper escapes it as \udXXX in the canonical form (valid JSON, stable
    # hash); pre-FW-18 this raised UnicodeEncodeError.
    messages = [{"role": "user", "content": "a\ud800b"}]
    prompt_hash = _prompt_hash(messages)  # must not raise
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).replace("\ud800", "\\ud800")
    expected = "sha256:" + hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    assert prompt_hash == expected


def test_prompt_hash_clean_payload_unchanged_vs_pre_fw18_json_dumps():
    # Regression: for clean payloads the hash is byte-identical to the
    # pre-FW-18 json.dumps construction (same sort_keys/separators), so
    # recorded prompt hashes still verify.
    messages = [
        {"role": "system", "content": "system prompt café — ü"},
        {"role": "user", "content": "snapshot unit:7:goto_n"},
    ]
    expected = "sha256:" + hashlib.sha256(json.dumps(
        messages, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    assert _prompt_hash(messages) == expected


# ---------------------------------------------------------------------------
# FW-13 budget sub-items: the repair conversation inside the budget check +
# the fixed-text startup invariant
# ---------------------------------------------------------------------------


def test_repair_conversation_over_budget_raises_prompt_budget_exceeded():
    """FW-13: the repair is inside the budget check. A complete repair
    conversation (prior messages + failed output + repair instruction +
    re-attached candidate table) over the assembled-prompt budget raises
    PromptBudgetExceeded BEFORE any transport call — the runner's dedicated
    bounded fail-stop handles it, never a silent oversize."""
    client, transport = make_client([raw(MALFORMED_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt())
    oversized_table = "x" * (prompts.REPAIR_CONVERSATION_TOKEN_BUDGET * 4)
    with pytest.raises(PromptBudgetExceeded):
        adapter.complete_repair(RepairPrompt(
            call_id="run/t1/call2", turn_id="run/t1",
            memory="memory text", snapshot="snapshot text", delta="delta text",
            candidates="unit:1:goto_n — move north",
            error_class="schema_error", failed_refs=["unit:1:goto_n"],
            refreshed_candidates=oversized_table))
    # The oversize repair never reached the transport.
    assert len(transport.calls) == 1


def test_client_initialization_checks_fixed_prompt_budget(monkeypatch):
    """FW-13 startup invariant: client construction fails fast with an
    explicit configuration error when the fixed prompt text (system + turn
    prefix) alone exceeds the assembled-prompt budget — at startup, not as
    a mid-game kill. Normal construction (the trivially small templates)
    is unaffected."""
    monkeypatch.setattr(prompts, "REPAIR_CONVERSATION_TOKEN_BUDGET", 1)
    client, _ = make_client([raw(VALID_BATCH)])
    with pytest.raises(prompts.PromptConfigError):
        MistralTurnClient(client)
    # With the real (trivially small) templates, construction succeeds.
    monkeypatch.undo()
    client, _ = make_client([raw(VALID_BATCH)])
    MistralTurnClient(client)


# ---------------------------------------------------------------------------
# W1-7: the two-budget reality — the initial check sums ONLY the four
# section bodies; the repair check sums the COMPLETE conversation (fixed
# text included). Same pre-split value, two different sums.
# ---------------------------------------------------------------------------


def test_initial_fits_section_budget_but_full_conversation_exceeds_repair_budget():
    """W1-7 divergence case, at the natural (equal) budget values: the four
    section bodies sit just under their per-section caps, so their joint sum
    fits the section-body budget and the initial turn assembles and sends —
    while the COMPLETE repair conversation (system prompt + full turn prompt
    + failed output + repair instruction, fixed text included) exceeds the
    repair-conversation budget and fails closed before any transport call.
    Before the split this divergence was invisible: one constant guarded two
    different sums."""
    assert (prompts.ASSEMBLED_SECTION_BODY_TOKEN_BUDGET
            == prompts.REPAIR_CONVERSATION_TOKEN_BUDGET)
    client, transport = make_client([raw(MALFORMED_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.complete_turn(turn_prompt(
        memory="m" * (prompts.MEMORY_TOKEN_BUDGET * 4 - 4),
        snapshot="s" * (prompts.SNAPSHOT_TOKEN_BUDGET * 4 - 4),
        delta="d" * (prompts.DELTA_TOKEN_BUDGET * 4 - 4),
        candidates="c" * (prompts.CANDIDATE_TABLE_TOKEN_BUDGET * 4 - 4),
    ))
    # The initial check passed (the section-body sum fits the section-body
    # budget) and the turn reached the transport.
    assert len(transport.calls) == 1
    # The same turn's repair conversation — the bodies PLUS the fixed prompt
    # text and the repair exchange — exceeds the repair-conversation budget.
    with pytest.raises(PromptBudgetExceeded):
        adapter.complete_repair(RepairPrompt(
            call_id="run/t1/call2", turn_id="run/t1",
            memory="memory text", snapshot="snapshot text", delta="delta text",
            candidates="unit:1:goto_n — move north",
            error_class="schema_error", failed_refs=["unit:1:goto_n"],
            refreshed_candidates="unit:1:goto_n — move north"))
    # The oversize repair never reached the transport.
    assert len(transport.calls) == 1


def test_inverse_divergence_sections_over_initial_but_repair_within_budget(
        monkeypatch):
    """W1-7, the inverse divergence. With the current equal values it cannot
    occur naturally (the repair conversation strictly contains the section
    bodies), so the test sets the two budgets apart: a section-body sum over
    the section-body budget fails the initial check, while the same content
    in a complete repair conversation fits the enlarged repair-conversation
    budget — each check behaves per its own budget."""
    monkeypatch.setattr(prompts, "ASSEMBLED_SECTION_BODY_TOKEN_BUDGET", 10)
    monkeypatch.setattr(prompts, "REPAIR_CONVERSATION_TOKEN_BUDGET",
                        10 * REPAIR_CONVERSATION_TOKEN_BUDGET)
    bodies = dict(memory="m" * 400, snapshot="s" * 400,
                  delta="d" * 400, candidate_summary="c" * 400)  # ~400 tokens
    # Initial check: each section fits its own cap, but the joint sum
    # exceeds the (tightened) section-body budget -> raises.
    with pytest.raises(PromptBudgetExceeded):
        prompts.turn_prompt(**bodies)
    # The same content as a complete repair conversation: the repair check
    # sums the full conversation against the (enlarged) repair budget —
    # within it, so no raise.
    turn = prompts.turn_prompt(**bodies, enforce_budgets=False)
    messages = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": turn},
        {"role": "assistant", "content": "{bad output}"},
        {"role": "user", "content": prompts.repair_prompt(
            "schema_error", detail="no JSON object", failed_refs="(none)",
            refreshed_candidates="TABLE")},
    ]
    _check_repair_conversation_budget(messages)  # must not raise


def test_conversation_replays_exact_assistant_chunks_and_execution_report():
    first = raw(VALID_BATCH)
    first.content_chunks = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]},
        {"type": "text", "text": VALID_BATCH},
    ]
    second = raw(VALID_BATCH)
    transport = MockTransport(outcomes=[first, second])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    response = adapter.complete_turn(turn_prompt())
    assert response.batch is not None
    adapter.commit_turn({"turn": 1, "execution": [{"outcome": "verified"}]})
    adapter.complete_turn(turn_prompt(turn_id="run/t2", call_id="run/t2/call1"))
    sent = transport.calls[1]["messages"]
    assert sent[0] == transport.calls[0]["messages"][0]
    assert sent[1] == transport.calls[0]["messages"][1]
    assert sent[2]["role"] == "assistant"
    assert sent[2]["content"] == first.content_chunks
    assert sent[3]["role"] == "user"
    assert sent[3]["content"].startswith("[EXECUTION REPORT]")


def test_resumed_manager_cache_generation_is_used_before_first_request():
    """A restored post-compaction epoch must select its key immediately."""
    first = raw(VALID_BATCH)
    transport = MockTransport(outcomes=[first])
    client = mc.MistralClient(api_key=CANARY, transport=transport,
                              prompt_cache_key="run:epoch-0")
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()},
        prompt_cache_key="run:epoch-0:epoch-2")
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)

    adapter.complete_turn(turn_prompt())

    assert transport.calls[0]["prompt_cache_key"] == "run:epoch-0:epoch-2"
    assert client.prompt_cache_key == "run:epoch-0:epoch-2"


def test_conversation_preserves_fallback_execution_without_assistant_message():
    """A transport-only turn still leaves an honest engine report for the next turn."""
    client, transport = make_client([mc.TransportError("offline"), raw(VALID_BATCH)])
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    response = adapter.complete_turn(turn_prompt())
    assert response.batch is None
    adapter.commit_turn({"turn": 1, "execution": [{"outcome": "fallback"}]})
    adapter.complete_turn(turn_prompt(turn_id="run/t2", call_id="run/t2/call1"))
    sent = transport.calls[1]["messages"]
    assert not any(message.get("role") == "assistant" for message in sent[2:])
    assert any(("[EXECUTION REPORT]" in str(message.get("content"))
                or "[ENGINE-ONLY REPORT]" in str(message.get("content")))
               for message in sent if message.get("role") == "user")


def test_compaction_and_replay_share_one_monotonic_turn_budget():
    """Compaction must leave only its remaining deadline for the replay call."""
    now = [0.0]
    transport = MockTransport(outcomes=[raw(VALID_BATCH), raw(VALID_BATCH)])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    adapter = MistralTurnClient(client, clock=lambda: now[0])
    adapter.complete_turn(turn_prompt(budget_s=20.0))
    adapter._turn_deadline = 20.0
    adapter._compaction_used = True
    now[0] = 8.0
    adapter._attempt(adapter._turn_messages, turn_prompt(budget_s=20.0),
                     attempt=1, kind="initial")
    assert transport.calls[1]["timeout_ms"] == 7_000


def test_conversation_repair_uses_calibrated_provider_prompt_tokens():
    transport = MockTransport(outcomes=[raw(MALFORMED_BATCH)])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    adapter.complete_turn(turn_prompt())
    # The calibrated repair envelope is now the documented 1M context with
    # 64k headroom; this remains an oversize request near that boundary.
    manager.record_usage(935_000,
                         request_hash=turn_client_module._prompt_hash(
                             adapter._turn_messages))
    base = turn_prompt()
    with pytest.raises(PromptBudgetExceeded, match="calibrated"):
        adapter.complete_repair(RepairPrompt(
            call_id=base.call_id, turn_id=base.turn_id, memory=base.memory,
            snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
            budget_s=base.budget_s, error_class="schema_error",
            refreshed_candidates="candidate table"))


def test_conversation_repair_uses_provider_completion_without_overestimating_suffix():
    transport = MockTransport(outcomes=[raw(MALFORMED_BATCH), raw(VALID_BATCH)])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    base = turn_prompt()
    adapter.complete_turn(base)
    manager.record_usage(311_027, estimated_tokens=1_000,
                         message_count=len(manager.messages),
                         completion_tokens=40_317)
    repaired = adapter.complete_repair(RepairPrompt(
        call_id=base.call_id, turn_id=base.turn_id, memory=base.memory,
        snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
        budget_s=base.budget_s, error_class="schema_error",
        refreshed_candidates="candidate table"))
    assert repaired.batch is not None
    assert len(transport.calls) == 2


def test_repair_calibration_hash_mismatch_falls_back_to_raw_estimate():
    transport = MockTransport(outcomes=[raw(MALFORMED_BATCH), raw(VALID_BATCH)])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    base = turn_prompt()
    adapter.complete_turn(base)
    manager.record_usage(935_000, request_hash="sha256:stale")
    repaired = adapter.complete_repair(RepairPrompt(
        call_id=base.call_id, turn_id=base.turn_id, memory=base.memory,
        snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
        budget_s=base.budget_s, error_class="schema_error",
        refreshed_candidates="candidate table"))
    assert repaired.batch is not None
    assert len(transport.calls) == 2


def test_conversation_repair_commits_one_failed_assistant_then_repair_pair():
    bad = raw(MALFORMED_BATCH)
    bad.content_chunks = [{"type": "text", "text": MALFORMED_BATCH}]
    good = raw(VALID_BATCH)
    transport = MockTransport(outcomes=[bad, good])
    client = mc.MistralClient(api_key=CANARY, transport=transport)
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                conversation_manager=manager)
    base = turn_prompt()
    first = adapter.complete_turn(base)
    assert first.batch is None
    repaired = adapter.complete_repair(RepairPrompt(
        call_id=base.call_id, turn_id=base.turn_id, memory=base.memory,
        snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
        budget_s=base.budget_s, error_class="schema_error",
        refreshed_candidates=base.candidates))
    assert repaired.batch is not None
    adapter.commit_turn({"turn": 1, "status": "completed"})
    roles = [message["role"] for message in manager.messages[1:]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert manager.messages[2]["content"] == bad.content_chunks


# ---------------------------------------------------------------------------
# Optional model-initiated ruleset lookup protocol
# ---------------------------------------------------------------------------

def _tool_raw(*, query="warrior", call_id="call-1"):
    calls = [{"id": call_id, "type": "function", "function": {
        "name": "lookup_freeciv",
        "arguments": json.dumps({"query": query, "limit": 3}),
    }}]
    assistant = {"role": "assistant", "content": None, "tool_calls": calls}
    return raw("", reasoning_text="Need the unit rules first", content_chunks=[],
                tool_calls=calls, assistant_message=assistant)


def test_lookup_direct_response_keeps_single_legacy_call():
    client, transport = make_client([raw(VALID_BATCH)])
    adapter = MistralTurnClient(
        client, ruleset_lookup=lambda request: {"never": "called"},
        enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt())
    assert response.batch is not None
    assert len(transport.calls) == 1
    assert transport.calls[0]["tools"] is not None
    assert transport.calls[0]["tool_choice"] == "auto"
    assert transport.calls[0]["parallel_tool_calls"] is False
    # The adapter omits response_format on the tool request; transport owns
    # translating that tool-mode request to an unstructured API call.


def test_set_ruleset_lookup_only_enables_interactive_gameplay_profile():
    client, transport = make_client([raw(VALID_BATCH), raw(VALID_BATCH)])
    adapter = MistralTurnClient(client)
    adapter.set_ruleset_lookup(lambda request: {"entries": []})
    adapter.complete_turn(turn_prompt())
    assert transport.calls[0]["tools"] is None

    client, transport = make_client([raw(VALID_BATCH)])
    adapter = MistralTurnClient(client, context_policy="conversation_v1",
                                prompt_profile="gameplay_v2")
    adapter.set_ruleset_lookup(lambda request: {"entries": []})
    adapter.complete_turn(turn_prompt())
    assert transport.calls[0]["tools"][0]["function"]["name"] == "lookup_freeciv"


def test_lookup_appends_matching_tool_message_and_strict_final_request():
    client, transport = make_client([_tool_raw(), raw(VALID_BATCH)])
    looked_up = []
    adapter = MistralTurnClient(
        client, ruleset_lookup=lambda request: looked_up.append(request) or {
            "entries": [{"name": "Warrior", "attack": 1}]},
        enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt())
    assert response.batch is not None
    assert looked_up == [{"ok": True, "query": "warrior", "categories": [],
                          "limit": 3}]
    assert len(transport.calls) == 2
    first, final = transport.calls
    assert first["tools"][0]["function"]["name"] == "lookup_freeciv"
    assert final["tools"] is None
    assert final["response_format"]["type"] == "json_schema"
    assistant_tool = next(message for message in final["messages"]
                          if message.get("tool_calls"))
    tool_result = next(message for message in final["messages"]
                       if message.get("role") == "tool")
    assert assistant_tool["tool_calls"][0]["id"] == "call-1"
    assert tool_result["tool_call_id"] == "call-1"
    assert '"attack":1' in tool_result["content"]
    assert "LOOKUP COMPLETE" in tool_result["content"]


def test_lookup_tool_final_normalizes_streamed_assistant_chunks_to_api_valid_content():
    """Provider thinking/text chunks must not be replayed as assistant content."""
    calls = [{"id": "chunk-call", "type": "function", "function": {
        "name": "lookup_freeciv",
        "arguments": json.dumps({"query": "warrior", "limit": 3}),
    }}]
    first = raw("", reasoning_text="think", content_chunks=[
        {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]},
        {"type": "text", "text": ""},
    ], tool_calls=calls, assistant_message={
        "role": "assistant", "content": [
            {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]},
            {"type": "text"},
        ], "tool_calls": calls,
    })
    client, transport = make_client([first, raw(VALID_BATCH)])
    adapter = MistralTurnClient(
        client, ruleset_lookup=lambda request: {"entries": [{"name": "Warrior"}]},
        enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt())
    assert response.batch is not None
    final = transport.calls[1]
    assistant = next(message for message in final["messages"]
                     if message.get("tool_calls"))
    assert isinstance(assistant["content"], str)
    assert assistant["content"] == ""
    assert assistant["tool_calls"] == calls
    assert first.content_chunks[0]["type"] == "thinking"
    assert first.reasoning_text == "think"


def test_lookup_invalid_arguments_are_returned_to_model_without_hallucinating():
    bad_call = [{"id": "bad", "type": "function", "function": {
        "name": "lookup_freeciv", "arguments": "not json"}}]
    client, transport = make_client([
        raw("", tool_calls=bad_call,
            assistant_message={"role": "assistant", "content": None,
                               "tool_calls": bad_call}),
        raw(VALID_BATCH),
    ])
    invoked = []
    adapter = MistralTurnClient(
        client, ruleset_lookup=lambda request: invoked.append(request),
        enable_ruleset_lookup=True)
    assert adapter.complete_turn(turn_prompt()).batch is not None
    assert invoked == []
    tool_result = next(message["content"] for message in transport.calls[1]["messages"]
                       if message.get("role") == "tool")
    assert "invalid lookup arguments" in tool_result
    assert "entries" not in tool_result


def test_lookup_topics_are_capped_and_deadline_prevents_final_request():
    calls = []
    tool_calls = [{"id": str(i), "type": "function", "function": {
        "name": "lookup_freeciv", "arguments": json.dumps({"query": str(i)})}}
        for i in range(16)]
    client, transport = make_client([raw("", tool_calls=tool_calls,
        assistant_message={"role": "assistant", "content": None,
                           "tool_calls": tool_calls})])
    now = [0.0]
    def lookup(request):
        calls.append(request)
        now[0] = 10.0
        return {"ok": True}
    adapter = MistralTurnClient(client, clock=lambda: now[0],
                                ruleset_lookup=lookup,
                                enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt(budget_s=10.0))
    assert response.error is not None
    assert len(calls) == 1
    assert len(transport.calls) == 1


def test_lookup_excess_tool_calls_get_matching_cap_errors():
    tool_calls = [{"id": str(i), "type": "function", "function": {
        "name": "lookup_freeciv", "arguments": json.dumps({"query": str(i)})}}
        for i in range(6)]
    client, transport = make_client([
        raw("", tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": None,
                               "tool_calls": tool_calls}),
        raw(VALID_BATCH),
    ])
    invoked = []
    adapter = MistralTurnClient(
        client, ruleset_lookup=lambda request: invoked.append(request) or {
            "entries": []}, enable_ruleset_lookup=True)
    assert adapter.complete_turn(turn_prompt()).batch is not None
    assert len(invoked) == 3
    tool_messages = [message for message in transport.calls[1]["messages"]
                     if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [str(i)
                                                                        for i in range(6)]
    assert all("lookup limit reached" in message["content"]
               for message in tool_messages[3:])


def test_lookup_more_than_safe_tool_messages_fails_closed_without_orphans():
    tool_calls = [{"id": str(i), "type": "function", "function": {
        "name": "lookup_freeciv", "arguments": json.dumps({"query": str(i)})}}
        for i in range(17)]
    client, transport = make_client([raw("", tool_calls=tool_calls,
        assistant_message={"role": "assistant", "content": None,
                           "tool_calls": tool_calls})])
    adapter = MistralTurnClient(client, ruleset_lookup=lambda request: {},
                                enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt())
    assert response.error is not None
    assert "safe tool-call limit" in response.error
    assert len(transport.calls) == 1


def test_lookup_missing_tool_call_id_fails_closed():
    calls = [{"type": "function", "function": {
        "name": "lookup_freeciv", "arguments": '{"query":"road"}'}}]
    client, transport = make_client([raw("", tool_calls=calls,
        assistant_message={"role": "assistant", "content": None,
                           "tool_calls": calls})])
    adapter = MistralTurnClient(client, ruleset_lookup=lambda request: {},
                                enable_ruleset_lookup=True)
    response = adapter.complete_turn(turn_prompt())
    assert response.error is not None
    assert "valid id" in response.error
    assert len(transport.calls) == 1


def test_lookup_unknown_category_is_actionable_and_does_not_widen_search():
    calls = [{"id": "cat", "type": "function", "function": {
        "name": "lookup_freeciv", "arguments": json.dumps({
            "query": "road", "categories": ["not-a-category"]})}}]
    client, transport = make_client([
        raw("", tool_calls=calls,
            assistant_message={"role": "assistant", "content": None,
                               "tool_calls": calls}),
        raw(VALID_BATCH),
    ])
    invoked = []
    adapter = MistralTurnClient(client,
                                ruleset_lookup=lambda request: invoked.append(request),
                                enable_ruleset_lookup=True)
    assert adapter.complete_turn(turn_prompt()).batch is not None
    assert invoked == []
    tool = next(message for message in transport.calls[1]["messages"]
                if message.get("role") == "tool")
    assert "unknown alias" in tool["content"]


def test_lookup_conversation_commits_assistant_tool_and_tool_result_exactly():
    client, transport = make_client([_tool_raw(), raw(VALID_BATCH)])
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(
        client, context_policy="conversation_v1", conversation_manager=manager,
        ruleset_lookup=lambda request: {"entries": [request["query"]]},
        enable_ruleset_lookup=True)
    adapter.complete_turn(turn_prompt())
    adapter.commit_turn({"turn": 1, "status": "completed"})
    roles = [message["role"] for message in manager.messages[1:]]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert manager.messages[2]["tool_calls"][0]["id"] == "call-1"
    assert manager.messages[3]["tool_call_id"] == "call-1"


def test_lookup_tool_final_repair_preserves_lookup_exchange_in_request_and_commit():
    client, transport = make_client([_tool_raw(), raw(MALFORMED_BATCH),
                                     raw(VALID_BATCH)])
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(
        client, context_policy="conversation_v1", conversation_manager=manager,
        ruleset_lookup=lambda request: {"entries": [request["query"]]},
        enable_ruleset_lookup=True)
    base = turn_prompt()
    first = adapter.complete_turn(base)
    assert first.malformed is not None
    repaired = adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call3", turn_id="run/t1", memory=base.memory,
        snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
        budget_s=50.0, error_class="schema_error",
        failed_refs=["move unit 10 one tile north"],
        refreshed_candidates=base.candidates))
    assert repaired.batch is not None
    repair_messages = transport.calls[2]["messages"]
    assistant_index = next(i for i, message in enumerate(repair_messages)
                           if message.get("tool_calls"))
    assert repair_messages[assistant_index]["tool_calls"][0]["id"] == "call-1"
    assert repair_messages[assistant_index + 1]["role"] == "tool"
    assert repair_messages[assistant_index + 2]["role"] == "assistant"
    failed_index = next(i for i, message in enumerate(repair_messages)
                        if message.get("content") == MALFORMED_BATCH)
    assert repair_messages[failed_index]["role"] == "assistant"
    assert repair_messages[failed_index + 1]["role"] == "user"
    assert repair_messages[-1]["role"] == "user"
    adapter.commit_turn({"turn": 1, "status": "completed"})
    roles = [message["role"] for message in manager.messages[1:]]
    assert roles == ["user", "assistant", "tool", "assistant", "user",
                     "assistant", "user"]


def test_failed_lookup_repair_aborts_incomplete_tool_transaction_cleanly():
    client, transport = make_client([_tool_raw(), raw(MALFORMED_BATCH),
                                     mc.TransportError("repair timeout")])
    manager = ConversationManager(
        system_message={"role": "system", "content": prompts.system_prompt()})
    adapter = MistralTurnClient(
        client, context_policy="conversation_v1", conversation_manager=manager,
        ruleset_lookup=lambda request: {"entries": []},
        enable_ruleset_lookup=True)
    base = turn_prompt()
    adapter.complete_turn(base)
    failed = adapter.complete_repair(RepairPrompt(
        call_id="run/t1/call3", turn_id=base.turn_id, memory=base.memory,
        snapshot=base.snapshot, delta=base.delta, candidates=base.candidates,
        budget_s=50.0, error_class="schema_error"))
    assert failed.error is not None
    # Must not raise ConversationError or commit an orphaned role=tool.
    adapter.commit_turn({"turn": 1, "status": "degraded"})
    assert manager.messages[-1]["role"] == "user"
    assert "degraded" in manager.messages[-1]["content"]
