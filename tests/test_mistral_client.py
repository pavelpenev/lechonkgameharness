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
- the per-arm model fingerprint probe (fixed prompt v2, the arm sampling,
  the K=3 sample hash set) and the P10 probe methods under a mock transport;
- the action_batch repair contract (W1-5): both repair attempts are
  budgeted — disable_transport_retries=True plus the explicit timeout_ms
  (the documented ACTION_BATCH_REPAIR_TIMEOUT_MS, caller-overridable);
- the fingerprint probe's fail-closed sample validation (W1-5, DEF-4):
  fewer than K=3 usable samples, or a non-string sample, raises
  FingerprintProbeError (no silent partial fingerprint, no AttributeError);
- the arm sampling (FW-20): every call path — turn calls and probes alike —
  carries temperature 1.0 and reasoning_effort "high" and never sends top_p;
- the model package stays civrealm-free and MIT-marked (plan §7.1).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import typing
import types
import threading
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
    ProviderCapacityError,
    TurnAttempts,
)


def test_provider_capacity_classification_requires_exact_structured_fields():
    class Response:
        status_code = 429
        headers = {"Retry-After": "12"}

        def json(self):
            return {"error": {"type": "backend_out_of_capacity", "code": "3505"}}

    err = types.SimpleNamespace(raw_response=Response())
    classified = mc._capacity_error_from_sdk(err)
    assert isinstance(classified, ProviderCapacityError)
    assert classified.http_status == 429
    assert classified.provider_code == "3505"
    assert classified.retry_after == "12"

    generic = types.SimpleNamespace(
        raw_response=types.SimpleNamespace(
            status_code=429,
            json=lambda: {"error": {"type": "rate_limit", "code": "999"}},
        ))
    assert mc._capacity_error_from_sdk(generic) is None

    class RenderedSDKError:
        raw_response = None
        status_code = None

        def __str__(self):
            return ('Status 429. Body: {"type":"backend_out_of_capacity",'
                    '"code":"3505"}')

    rendered = RenderedSDKError()
    # SDK errors in some versions expose only their rendered body.
    assert isinstance(mc._capacity_error_from_sdk(rendered), ProviderCapacityError)


def test_provider_capacity_error_bypasses_action_batch_repair():
    capacity = ProviderCapacityError("capacity")
    transport = MockTransport(outcomes=[capacity])
    client = make_client(transport)
    with pytest.raises(ProviderCapacityError):
        client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    assert len(transport.calls) == 1
    assert len(client.call_logs) == 1
    assert client.call_logs[0].error_kind == "provider_capacity"

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


def test_reasoning_chunks_are_separated_from_final_answer():
    class Text:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class Thinking:
        type = "thinking"

        def __init__(self, text):
            self.thinking = [Text(text)]

    assert mc._message_content_parts([
        Thinking("first "),
        {"type": "thinking", "thinking": [{"text": "second"}]},
        Text('{"actions": []}'),
    ]) == ("first second", '{"actions": []}')
    assert mc._message_content_text("plain") == "plain"


def test_sdk_tool_call_request_preserves_assistant_message(monkeypatch):
    """The opt-in first call omits schema output and retains tool-call JSON."""
    errors = types.ModuleType("mistralai.client.errors")
    errors.NoResponseError = type("NoResponseError", (Exception,), {})
    errors.ResponseValidationError = type("ResponseValidationError", (Exception,), {})
    errors.SDKError = type("SDKError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "mistralai.client.errors", errors)
    seen = {}
    message = types.SimpleNamespace(
        content=[{"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]}],
        tool_calls=[{"id": "call-1", "type": "function", "function": {
            "name": "lookup_freeciv", "arguments": '{"topics":["combat"]}'}}],
    )
    class Chat:
        def complete(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")],
                usage=types.SimpleNamespace(prompt_tokens=4, completion_tokens=3,
                                            total_tokens=7), model=MODEL_ID,
                id="resp-tool", created=1)
    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = types.SimpleNamespace(chat=Chat())
    raw = transport.complete(messages=[{"role": "user", "content": "x"}],
                              model=MODEL_ID, temperature=1.0,
                              reasoning_effort="high", max_tokens=None,
                              response_format=None, random_seed=None,
                              tools=[{"type": "function", "function": {"name": "lookup_freeciv"}}])
    assert seen["tool_choice"] == "auto"
    assert seen["parallel_tool_calls"] is False
    assert "response_format" not in seen
    assert raw.tool_calls[0]["function"]["arguments"] == '{"topics":["combat"]}'
    assert raw.assistant_message["tool_calls"] == raw.tool_calls
    assert raw.reasoning_text == "plan"


def test_sdk_stream_aggregates_fragmented_tool_calls(monkeypatch):
    errors = types.ModuleType("mistralai.client.errors")
    errors.NoResponseError = type("NoResponseError", (Exception,), {})
    errors.ResponseValidationError = type("ResponseValidationError", (Exception,), {})
    errors.SDKError = type("SDKError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "mistralai.client.errors", errors)
    seen = {}
    events = [
        types.SimpleNamespace(data=types.SimpleNamespace(
            id="s1", model=MODEL_ID, created=1,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                content=[{"type": "thinking", "signature": "sig-1",
                           "thinking": [{"type": "text", "text": "plan"}]},
                         {"type": "reference", "id": "ref-1"},
                         {"type": "text"}],
                tool_calls=[{"index": 0, "id": "call-1",
                    "type": "function", "function": {"name": "lookup_", "arguments": '{"topics":'}}]),
                finish_reason=None)], usage=None)),
        types.SimpleNamespace(data=types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                content=None, tool_calls=[{"index": 0, "function": {
                    "name": "freeciv", "arguments": '["combat"]}'}}]),
                finish_reason="tool_calls")], usage=types.SimpleNamespace(
                    prompt_tokens=2, completion_tokens=4, total_tokens=6))),
    ]
    class Events:
        def __aiter__(self):
            self.i = 0
            return self
        async def __anext__(self):
            if self.i == len(events):
                raise StopAsyncIteration
            item = events[self.i]; self.i += 1; return item
        async def aclose(self):
            pass
    class Chat:
        async def stream_async(self, **kwargs):
            seen.update(kwargs)
            return Events()
    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = types.SimpleNamespace(chat=Chat())
    raw = transport.stream_complete(messages=[{"role": "user", "content": "x"}],
                                     model=MODEL_ID, temperature=1.0,
                                     reasoning_effort="high", max_tokens=None,
                                     response_format=None, random_seed=None,
                                     tools=[{"type": "function", "function": {"name": "lookup_freeciv"}}])
    assert seen["tool_choice"] == "auto"
    assert raw.tool_calls[0]["id"] == "call-1"
    assert raw.tool_calls[0]["function"] == {
        "name": "lookup_freeciv", "arguments": '{"topics":["combat"]}'
    }
    assert raw.assistant_message["tool_calls"] == raw.tool_calls
    assert {"type": "reference", "id": "ref-1"} in raw.content_chunks
    assert {"type": "text"} in raw.content_chunks
    assert {"type": "text", "text": ""} not in raw.assistant_message["content"]


def test_replayable_content_chunks_filters_empty_stream_fragments():
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "why"}]},
        {"type": "thinking", "thinking": [{"type": "text", "text": ""}]},
        "",
        {"type": "text", "text": "{"},
        {"type": "text", "text": "\"actions\": []}"},
    ]
    assert mc._replayable_content_chunks(content) == [
        {"type": "thinking", "thinking": [{"type": "text", "text": "why"}]},
        {"type": "text", "text": "{\"actions\": []}"},
    ]


def test_replayable_content_chunks_preserves_reference_and_signature_fields():
    content = [
        {"type": "thinking", "signature": "sig-1", "thinking": [
            {"type": "text", "text": "plan"},
            {"type": "reference", "reference_ids": ["ref-1"]},
        ]},
        {"type": "reference", "reference_ids": ["ref-2"]},
        {"type": "tool_reference", "tool": "web_search", "title": "Rules"},
        {"type": "text"},
    ]
    assert mc._replayable_content_chunks(content) == content[:3]


def test_replayable_content_chunks_canonicalizes_reasoning_alias():
    content = [{"type": "reasoning", "signature": "sig-2",
                "thinking": [{"type": "text", "text": "plan"}]}]
    assert mc._replayable_content_chunks(content) == [
        {"type": "thinking", "signature": "sig-2",
         "thinking": [{"type": "text", "text": "plan"}]},
    ]


def test_replayable_content_chunks_drops_unknown_and_malformed_references():
    content = [
        {"type": "unknown", "value": "drop"},
        {"type": "reference", "id": "missing-reference-ids"},
        {"type": "tool_reference", "tool": "web_search"},
        {"type": "text", "text": "valid"},
    ]
    assert mc._replayable_content_chunks(content) == [
        {"type": "text", "text": "valid"},
    ]


# ---------------------------------------------------------------------------
# Mock transport (no network, no SDK import)
# ---------------------------------------------------------------------------


@dataclass
class MockTransport:
    """Scripted transport: a queue of outcomes (RawCompletion or Exception)."""

    outcomes: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, *, messages, model, temperature, reasoning_effort,
                 max_tokens, response_format, random_seed, timeout_ms=None,
                 disable_transport_retries=False, **extra):
        # extra captures any unexpected sampling kwarg (e.g. a forbidden
        # top_p emission would surface here, never silently dropped)
        self.calls.append({
            "messages": list(messages), "model": model,
            "temperature": temperature, "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "response_format": response_format, "random_seed": random_seed,
            "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
            **extra,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class StreamMockTransport(MockTransport):
    deltas: list = field(default_factory=list)

    def stream_complete(self, *, messages, model, temperature,
                        reasoning_effort, max_tokens, response_format,
                        random_seed, timeout_ms=None,
                        absolute_timeout_ms=None,
                        disable_transport_retries=False, on_delta=None):
        self.calls.append({"stream": True, "timeout_ms": timeout_ms,
                           "absolute_timeout_ms": absolute_timeout_ms})
        for delta in self.deltas:
            if on_delta is not None:
                on_delta(delta)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def raw_completion(content, *, model=MODEL_ID, response_id="resp-1",
                   finish_reason="stop", prompt_tokens=100,
                   completion_tokens=50, transport_retries=0,
                   sample_contents=None):
    return mc.RawCompletion(
        content=content, model=model, response_id=response_id, created=0,
        finish_reason=finish_reason, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(prompt_tokens + completion_tokens
                      if prompt_tokens is not None and completion_tokens is not None
                      else None),
        transport_retries=transport_retries,
        sample_contents=sample_contents)


@dataclass
class MultiSampleMockTransport:
    """Scripted transport DECLARING the multi-sample ``n`` parameter (the
    SDK path's shape): one request returns K sample contents."""

    outcomes: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, *, messages, model, temperature, reasoning_effort,
                 max_tokens, response_format, random_seed, n=None,
                 timeout_ms=None, disable_transport_retries=False):
        self.calls.append({
            "messages": list(messages), "model": model,
            "temperature": temperature, "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "response_format": response_format, "random_seed": random_seed,
            "n": n, "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(transport, *, key=CANARY_KEY, **kwargs):
    return MistralClient(api_key=key, transport=transport, **kwargs)


def test_opt_in_stream_routes_deltas_and_keeps_final_parser_input():
    deltas = [{"reasoning": "thinking ", "answer": "", "content": []},
              {"reasoning": "", "answer": VALID_BATCH,
               "content": [{"type": "text", "text": VALID_BATCH}]}]
    transport = StreamMockTransport(
        outcomes=[mc.RawCompletion(
            content=VALID_BATCH, model=MODEL_ID, response_id="stream-1",
            created=0, finish_reason="stop", prompt_tokens=1,
            completion_tokens=2, total_tokens=3,
            reasoning_text="thinking ", answer_text=VALID_BATCH,
            content_chunks=[{"type": "thinking", "thinking": [{"text": "thinking "}]},
                            {"type": "text", "text": VALID_BATCH}])],
        deltas=deltas)
    seen = []
    client = make_client(transport, stream=True,
                         stream_callback=seen.append)
    result = client.complete_once(messages=turn_messages(),
                                  prompt_hash="sha256:test", attempt=1,
                                  kind="initial")
    assert result.batch is not None
    assert result.raw_content == VALID_BATCH
    assert seen == deltas
    assert transport.calls == [{"stream": True, "timeout_ms": 120000,
                                "absolute_timeout_ms": None}]


def test_prompt_cache_key_is_opt_in_and_reaches_transport():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport, prompt_cache_key="run-epoch-1")
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:test",
                         attempt=1, kind="initial")
    assert transport.calls[0]["prompt_cache_key"] == "run-epoch-1"


def test_cached_tokens_are_logged_without_exposing_cache_key():
    raw = raw_completion(VALID_BATCH)
    raw.cached_tokens = 64
    transport = MockTransport(outcomes=[raw])
    client = make_client(transport, prompt_cache_key="run-epoch-1")
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:test",
                         attempt=1, kind="initial")
    entry = client.call_logs[0].to_dict()
    assert entry["cached_tokens"] == 64
    assert "prompt_cache_key" not in entry


def test_cached_input_tokens_use_discounted_price_and_are_bounded():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH,
                                                        prompt_tokens=1000,
                                                        completion_tokens=200)])
    raw = transport.outcomes[0]
    raw.cached_tokens = 800
    client = make_client(transport)
    client.complete_once(messages=turn_messages(), prompt_hash="sha256:test",
                         attempt=1, kind="initial")
    entry = client.call_logs[0].to_dict()
    expected = ((200 * 0.50) + (800 * 0.50 * 0.10) + (200 * 1.50)) / 1e6
    assert entry["cost_usd"] == round(expected, 6)

    # Provider metadata above the prompt count is clamped, never negative.
    raw2 = raw_completion(VALID_BATCH, prompt_tokens=1000, completion_tokens=0)
    raw2.cached_tokens = 5000
    transport2 = MockTransport(outcomes=[raw2])
    client2 = make_client(transport2)
    client2.complete_once(messages=turn_messages(), prompt_hash="sha256:test",
                          attempt=1, kind="initial")
    assert client2.call_logs[0].cost_usd == round(1000 * 0.50 * 0.10 / 1e6, 6)


def test_transport_error_with_partial_completion_keeps_usage_accounting():
    partial = raw_completion(VALID_BATCH, prompt_tokens=900,
                             completion_tokens=120)
    partial.cached_tokens = 640
    error = TransportError("read timed out")
    error.partial = partial
    transport = MockTransport(outcomes=[error])
    client = make_client(transport)
    outcome = client.complete_once(messages=turn_messages(), prompt_hash="sha256:test",
                                   attempt=1, kind="initial")
    assert outcome.error == "read timed out"
    entry = client.call_logs[0]
    assert entry.error == "read timed out"
    assert entry.tokens_in == 900
    assert entry.tokens_out == 120
    assert entry.cached_tokens == 640
    assert entry.cost_usd is not None


def test_partial_completion_without_provider_usage_remains_unknown():
    """A truncated provider stream must not be reported as zero usage."""
    partial = raw_completion(VALID_BATCH, prompt_tokens=None,
                             completion_tokens=None)
    assert partial.total_tokens is None
    client = make_client(MockTransport([partial]))
    outcome = client.complete_once(messages=[{"role": "user", "content": "hi"}],
                                   prompt_hash="sha256:unknown", attempt=1,
                                   kind="initial")
    assert outcome.error is None
    log = client.call_logs[-1]
    assert log.tokens_in is None
    assert log.tokens_out is None
    assert log.tokens_total is None
    assert log.cost_usd is None
    assert log.usage_status == "unavailable"


def test_stream_usage_accepts_dict_shaped_usage():
    class Events:
        def __init__(self):
            self.items = iter([{"data": {"id": "r", "model": MODEL_ID,
                "created": 0, "choices": [{"delta": {"content":
                    [{"type": "text", "text": VALID_BATCH}]},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                           "total_tokens": 14,
                           "prompt_tokens_details": {"cached_tokens": 8}}}}])
        def __aiter__(self): return self
        async def __anext__(self):
            try: return next(self.items)
            except StopIteration: raise StopAsyncIteration
        async def aclose(self): return None

    class SDK:
        class Chat:
            async def stream_async(self, **kwargs): return Events()
        chat = Chat()

    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = SDK()
    transport._sdk_no_retries = transport._sdk
    raw = transport.stream_complete(
        messages=turn_messages(), model=MODEL_ID, temperature=1.0,
        reasoning_effort="none", max_tokens=100, response_format=None,
        random_seed=None, timeout_ms=1000)
    assert raw.prompt_tokens == 10
    assert raw.completion_tokens == 4
    assert raw.cached_tokens == 8


def test_stream_timeout_partial_dict_usage_is_retained():
    import asyncio

    class Events:
        def __init__(self):
            self.items = iter([{
                "data": {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 31,
                        "completion_tokens": 7,
                        "total_tokens": 38,
                        "prompt_tokens_details": {"cached_tokens": 16},
                    },
                }
            }])

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.items:
                self.items = None
                return {"data": {"choices": [], "usage": {
                    "prompt_tokens": 31,
                    "completion_tokens": 7,
                    "total_tokens": 38,
                    "prompt_tokens_details": {"cached_tokens": 16},
                }}}
            await asyncio.sleep(0.05)
            return {"data": {"choices": []}}

        async def aclose(self):
            return None

    class SDK:
        class Chat:
            async def stream_async(self, **kwargs):
                return Events()
        chat = Chat()

    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = SDK()
    transport._sdk_no_retries = transport._sdk
    with pytest.raises(mc.TransportError, match="timed out") as raised:
        transport.stream_complete(
            messages=[], model=MODEL_ID, temperature=1.0,
            reasoning_effort="high", max_tokens=10, response_format=None,
            random_seed=None, timeout_ms=5)
    partial = raised.value.partial
    assert partial.prompt_tokens == 31
    assert partial.completion_tokens == 7
    assert partial.total_tokens == 38
    assert partial.cached_tokens == 16


def test_sdk_stream_collector_supports_two_sequential_calls():
    class Events:
        def __init__(self, items):
            self.items = iter(items)
        def __aiter__(self):
            return self
        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration
        async def aclose(self):
            return None

    class Chat:
        def __init__(self):
            self.calls = 0
        async def stream_async(self, **kwargs):
            self.calls += 1
            return Events([
                {"data": {"id": f"s{self.calls}", "model": MODEL_ID,
                          "choices": [{"delta": {"content": [
                              {"type": "thinking", "thinking": [
                                  {"type": "text", "text": "why"}]}]},
                                        "finish_reason": None}]}},
                {"data": {"id": f"s{self.calls}", "model": MODEL_ID,
                          "usage": {"prompt_tokens": 1,
                                    "completion_tokens": 2,
                                    "total_tokens": 3},
                          "choices": [{"delta": {"content": [
                              {"type": "text", "text": VALID_BATCH}]},
                                        "finish_reason": "stop"}]}}])

    class SDK:
        def __init__(self):
            self.chat = Chat()

    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = SDK()
    transport._sdk_no_retries = transport._sdk
    seen = []
    for _ in range(2):
        raw = transport.stream_complete(
            messages=[], model=MODEL_ID, temperature=1.0,
            reasoning_effort="high", max_tokens=100,
            response_format=None, random_seed=None, on_delta=seen.append)
        assert raw.finish_reason == "stop"
        assert raw.content == VALID_BATCH
        assert raw.reasoning_text == "why"
    assert len(seen) == 4


def test_sdk_stream_timeout_emits_partial_and_rejects_missing_finish():
    import asyncio

    class Events:
        def __aiter__(self): return self
        async def __anext__(self):
            await asyncio.sleep(0.05)
            return {"data": {"choices": []}}
        async def aclose(self): return None

    class SDK:
        class Chat:
            async def stream_async(self, **kwargs): return Events()
        chat = Chat()

    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = SDK()
    transport._sdk_no_retries = transport._sdk
    seen = []
    with pytest.raises(mc.TransportError, match="timed out"):
        transport.stream_complete(
            messages=[], model=MODEL_ID, temperature=1.0,
            reasoning_effort="high", max_tokens=10, response_format=None,
            random_seed=None, timeout_ms=5, on_delta=seen.append)
    assert seen and seen[-1]["complete"] is False

    class EofSDK:
        class Chat:
            async def stream_async(self, **kwargs):
                class Eof:
                    def __aiter__(self): return self
                    async def __anext__(self): raise StopAsyncIteration
                    async def aclose(self): return None
                return Eof()
        chat = Chat()
    transport._sdk = EofSDK()
    with pytest.raises(mc.TransportError, match="finish reason"):
        transport.stream_complete(
            messages=[], model=MODEL_ID, temperature=1.0,
            reasoning_effort="high", max_tokens=10, response_format=None,
            random_seed=None)


def test_sdk_stream_absolute_timeout_cancels_continuous_stream_and_keeps_partial():
    """A stream that continually emits bytes still obeys the outer deadline."""
    import asyncio

    class Events:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0.001)
            return {"data": {"id": "loop", "model": MODEL_ID,
                              "choices": [{"delta": {"content": [
                                  {"type": "thinking", "thinking": [
                                      {"type": "text", "text": "hmm "}]}]},
                                            "finish_reason": None}]}}

        async def aclose(self):
            self.closed = True

    class Chat:
        def __init__(self):
            self.events = Events()

        async def stream_async(self, **kwargs):
            return self.events

    class SDK:
        def __init__(self):
            self.chat = Chat()

    sdk = SDK()
    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = sdk
    transport._sdk_no_retries = sdk
    seen = []
    with pytest.raises(mc.TransportError, match="timed out") as raised:
        transport.stream_complete(
            messages=[], model=MODEL_ID, temperature=1.0,
            reasoning_effort="high", max_tokens=None, response_format=None,
            random_seed=None, absolute_timeout_ms=20, on_delta=seen.append)
    assert seen and seen[-1].get("partial") is True
    partial = raised.value.partial
    assert partial is not None
    assert partial.reasoning_text.startswith("hmm ")
    assert sdk.chat.events.closed is True


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

    def complete(self, *, messages, model, temperature, reasoning_effort,
                 max_tokens, response_format, random_seed, timeout_ms=None,
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


def test_complete_once_explicit_none_disables_schema_for_tool_discovery():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    outcome = client.complete_once(
        messages=turn_messages(), prompt_hash="sha256:tool",
        attempt=1, kind="initial", response_format=None,
        tools=[{"type": "function", "function": {"name": "lookup_freeciv"}}],
        tool_choice="auto", parallel_tool_calls=False)
    assert outcome.batch is not None
    call = transport.calls[0]
    assert call["response_format"] is None
    assert call["tools"][0]["function"]["name"] == "lookup_freeciv"
    assert call["tool_choice"] == "auto"
    assert call["parallel_tool_calls"] is False


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
# action_batch repair contract (W1-5: budgeted repair attempts)
# ---------------------------------------------------------------------------


def test_action_batch_repair_timeout_default_is_documented_constant():
    """W1-5: action_batch has no runner budget input, so the repair bound
    is the documented module constant (overridable per call), defaulting to
    the per-logical-attempt wall-clock bound."""
    assert mc.ACTION_BATCH_REPAIR_TIMEOUT_MS == mc.DEFAULT_TIMEOUT_MS
    assert mc.ACTION_BATCH_REPAIR_TIMEOUT_MS > 0


def test_action_batch_transport_failure_repair_is_bounded():
    """W1-5 repair contract, transport-failure path: the repair attempt
    passes disable_transport_retries=True and the explicit timeout_ms (its
    TOTAL duration stays within the bound); the initial attempt keeps the
    transport's transparent retries and no per-call timeout override."""
    transport = MockTransport(outcomes=[TransportError("500 after retries"),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.repairs_used == 1
    assert transport.calls[0]["disable_transport_retries"] is False
    assert transport.calls[0]["timeout_ms"] is None
    assert transport.calls[1]["disable_transport_retries"] is True
    assert transport.calls[1]["timeout_ms"] == mc.ACTION_BATCH_REPAIR_TIMEOUT_MS


def test_action_batch_validation_failure_repair_is_bounded():
    """W1-5 repair contract, validation-failure path: the same budgeted
    kwargs on the repair attempt that re-asks after a schema/ref failure."""
    transport = MockTransport(outcomes=[raw_completion("garbage"),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.repairs_used == 1
    assert transport.calls[0]["disable_transport_retries"] is False
    assert transport.calls[0]["timeout_ms"] is None
    assert transport.calls[1]["disable_transport_retries"] is True
    assert transport.calls[1]["timeout_ms"] == mc.ACTION_BATCH_REPAIR_TIMEOUT_MS


def test_action_batch_repair_timeout_is_caller_overridable():
    """W1-5: the repair timeout is overridable per call and reaches the
    transport exactly as passed."""
    transport = MockTransport(outcomes=[raw_completion("garbage"),
                                        raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts(), timeout_ms=55_000)
    assert transport.calls[1]["timeout_ms"] == 55_000
    assert transport.calls[1]["disable_transport_retries"] is True


def test_action_batch_both_transport_fail_repair_still_bounded():
    """W1-5: on the both-attempts-fail path the (failed) repair attempt was
    still dispatched with the budgeted kwargs."""
    transport = MockTransport(outcomes=[TransportError("down"),
                                        TransportError("down again")])
    client = make_client(transport)
    with pytest.raises(AttemptBudgetExhausted):
        client.action_batch(messages=turn_messages(),
                            prompt_hash="sha256:test",
                            attempts=TurnAttempts())
    assert transport.calls[1]["disable_transport_retries"] is True
    assert transport.calls[1]["timeout_ms"] == mc.ACTION_BATCH_REPAIR_TIMEOUT_MS


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
    # Cache-adjusted cost is unknown when the provider omits cache details.
    assert entry["cost_usd"] is None
    assert entry["cost_status"] == "unknown"
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
        raw_completion("probe-sample-1"),
        raw_completion("probe-sample-2"),
        raw_completion("probe-sample-3"),
        raw_completion(VALID_BATCH),
    ])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    assert client.call_logs[-1].fingerprint == probe.fingerprint
    assert client.call_logs[-1].fingerprint != "unprobed"


# ---------------------------------------------------------------------------
# Fingerprint probe (decision 8, FW-21: K=3 multi-sample, prompt v2) + P10
# ---------------------------------------------------------------------------


def test_fingerprint_probe_k3_samples_sorted_hash_set():
    """FW-21: the probe collects K=3 samples under the arm sampling and
    records the sorted, deduplicated hash set; output budgeting remains
    provider-managed."""
    contents = ["alpha-output", "beta-output", "gamma-output"]
    transport = MockTransport(outcomes=[raw_completion(c) for c in contents])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    expected_hashes = sorted(
        hashlib.sha256(c.encode("utf-8")).hexdigest() for c in contents)
    assert mc.FINGERPRINT_PROBE_SAMPLES == 3
    assert mc.FINGERPRINT_PROBE_MAX_TOKENS is None
    assert probe.sample_attempts == 3
    assert probe.sample_mode == mc.FINGERPRINT_MODE_SEQUENTIAL  # MockTransport declares no n
    assert probe.output_hashes == expected_hashes
    assert probe.fingerprint == "|".join(
        f"sha256:{h}" for h in expected_hashes)
    assert probe.max_tokens is None
    # arm sampling on every probe request (FW-20), with no local output cap
    assert len(transport.calls) == 3
    for call in transport.calls:
        assert call["temperature"] == 1.0
        assert call["reasoning_effort"] == "high"
        assert "top_p" not in call
        assert call["response_format"] is None
        assert call["max_tokens"] is None
    # the fixed probe prompt v2 (versioned file) with its recorded hash
    assert mc.FINGERPRINT_PROBE_PROMPT == "fingerprint_probe_v2"
    assert probe.prompt_hash == prompts.prompt_hash("fingerprint_probe_v2")
    assert (transport.calls[0]["messages"][-1]["content"]
            == prompts.load_prompt("fingerprint_probe_v2"))
    assert probe.model_requested == MODEL_ID
    assert probe.model_responded == MODEL_ID
    # cached on the client and carried by later call logs
    assert client.arm_fingerprint == probe.fingerprint
    # probe calls are logged too (decision 4: every call is logged)
    assert [log.kind for log in client.call_logs] == ["probe"] * 3
    assert all(log.attempt == 0 for log in client.call_logs)


def test_fingerprint_probe_records_attempts_vs_unique_hashes():
    """FW-21: K=3 attempts may yield fewer than three unique outputs —
    the probe records attempts vs unique hashes (the deduplicated set)."""
    transport = MockTransport(outcomes=[
        raw_completion("same"), raw_completion("same"),
        raw_completion("other")])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    assert probe.sample_attempts == 3
    expected = sorted({hashlib.sha256(c.encode("utf-8")).hexdigest()
                       for c in ("same", "other")})
    assert len(probe.output_hashes) == 2
    assert probe.output_hashes == expected
    assert probe.fingerprint == "|".join(f"sha256:{h}" for h in expected)
    assert probe.to_dict()["sample_attempts"] == 3
    assert len(probe.to_dict()["output_hashes"]) == 2


def test_fingerprint_probe_single_request_n3_mode():
    """FW-21: when the transport supports multi-sample requests (the SDK
    path: Chat.complete takes n — offline-verified), the K=3 samples come
    from ONE request with n=3, and the mode is recorded."""
    raw = raw_completion("first-choice",
                         sample_contents=["s1", "s2", "s3"])
    transport = MultiSampleMockTransport(outcomes=[raw])
    client = make_client(transport)
    probe = client.probe_fingerprint()
    assert probe.sample_mode == mc.FINGERPRINT_MODE_SINGLE_REQUEST
    assert len(transport.calls) == 1
    assert transport.calls[0]["n"] == 3
    assert transport.calls[0]["max_tokens"] is None
    assert transport.calls[0]["temperature"] == 1.0
    assert transport.calls[0]["reasoning_effort"] == "high"
    assert "top_p" not in transport.calls[0]
    assert probe.sample_attempts == 3
    expected = sorted(hashlib.sha256(s.encode("utf-8")).hexdigest()
                      for s in ("s1", "s2", "s3"))
    assert probe.output_hashes == expected
    assert probe.fingerprint == "|".join(f"sha256:{h}" for h in expected)
    assert len(client.call_logs) == 1
    assert client.call_logs[0].kind == "probe"


def test_fingerprint_set_encoding_round_trip_and_legacy_scalar():
    """FW-21: the persisted encoding is the sorted, deduplicated set of
    'sha256:<hex>' hashes joined by '|'; a legacy scalar fingerprint is
    exactly the one-element encoding (the documented fallback)."""
    encoded = mc.encode_fingerprint_set(["bb", "aa", "bb"])
    assert encoded == "sha256:aa|sha256:bb"
    assert mc.decode_fingerprint_set(encoded) == ["sha256:aa", "sha256:bb"]
    legacy = "sha256:" + "cd" * 32
    assert mc.decode_fingerprint_set(legacy) == [legacy]
    assert mc.decode_fingerprint_set(None) == []
    assert mc.decode_fingerprint_set("") == []


def test_fingerprint_probe_v2_prompt_file_present_and_v1_retained():
    """FW-21: the v2 probe prompt exists on disk and is the one the probe
    uses; v1 (the degenerate verbatim-echo prompt) is retained on disk."""
    v2 = prompts.prompt_path("fingerprint_probe_v2")
    v1 = prompts.prompt_path("fingerprint_probe_v1")
    assert v2.is_file() and v1.is_file()
    text = v2.read_text(encoding="utf-8")
    for label in ("MODEL:", "LAB:", "PRODUCT:", "CAPITAL:", "SEQ:"):
        assert label in text
    # v2 is not the degenerate v1 verbatim echo
    assert "LECHONK-FINGERPRINT-7f3a91c2" not in text


def test_fingerprint_probe_transport_error_propagates():
    transport = MockTransport(outcomes=[TransportError("no route")])
    client = make_client(transport)
    with pytest.raises(TransportError):
        client.probe_fingerprint()
    assert client.arm_fingerprint is None


@pytest.mark.parametrize("sample_contents", [
    ["only-one"],
    ["one", "two"],
])
def test_fingerprint_probe_fails_closed_on_missing_samples(sample_contents):
    """W1-5 / DEF-4 fail-closed: fewer than FINGERPRINT_PROBE_SAMPLES
    usable samples (a contract-violating transport returning fewer choices
    than the requested n=3) FAILS the probe with a recorded
    FingerprintProbeError — never a silent partial fingerprint."""
    raw = raw_completion("content", sample_contents=list(sample_contents))
    transport = MultiSampleMockTransport(outcomes=[raw])
    client = make_client(transport)
    with pytest.raises(mc.FingerprintProbeError):
        client.probe_fingerprint()
    assert client.arm_fingerprint is None


def test_fingerprint_probe_fails_closed_on_missing_multi_sample_fallback():
    """W1-5: a single-request outcome with no per-choice contents falls
    back to ONE sample — below K=3, so the probe fails closed."""
    transport = MultiSampleMockTransport(
        outcomes=[raw_completion("single-content", sample_contents=[])])
    client = make_client(transport)
    with pytest.raises(mc.FingerprintProbeError):
        client.probe_fingerprint()
    assert client.arm_fingerprint is None


def test_fingerprint_probe_fails_closed_on_none_sample_sequential():
    """W1-5: a None sample content (contract-violating transport) must not
    crash the probe with AttributeError — the probe fails closed instead."""
    transport = MockTransport(outcomes=[raw_completion(None),
                                        raw_completion("s2"),
                                        raw_completion("s3")])
    client = make_client(transport)
    with pytest.raises(mc.FingerprintProbeError):
        client.probe_fingerprint()
    assert client.arm_fingerprint is None


def test_fingerprint_probe_fails_closed_on_none_sample_multi_request():
    """W1-5: a None choice content inside a multi-sample response also
    fails the probe closed (no AttributeError escape)."""
    raw = raw_completion("content", sample_contents=[None, "s2", "s3"])
    transport = MultiSampleMockTransport(outcomes=[raw])
    client = make_client(transport)
    with pytest.raises(mc.FingerprintProbeError):
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
    assert transport.calls[0]["max_tokens"] is None
    assert "top_p" not in transport.calls[0]


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
    assert transport.calls[0]["temperature"] == 1.0
    assert transport.calls[0]["reasoning_effort"] == "high"
    assert transport.calls[0]["max_tokens"] is None
    assert "top_p" not in transport.calls[0]


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
# Arm sampling (FW-20: temperature 1.0, reasoning_effort high, never top_p)
# ---------------------------------------------------------------------------


def test_default_arm_sampling_settings():
    """FW-20: the constructor defaults are the arm sampling — temperature
    1.0 (previously 0.0) and reasoning_effort "high"."""
    client = make_client(MockTransport())
    assert client.temperature == 1.0
    assert client.reasoning_effort == "high"
    assert mc.DEFAULT_TEMPERATURE == 1.0
    assert mc.DEFAULT_REASONING_EFFORT == "high"


def test_default_max_tokens_is_provider_managed():
    """Ordinary turns do not impose an application output-token cap."""
    client = make_client(MockTransport())
    assert client.max_tokens is None
    assert mc.DEFAULT_MAX_TOKENS is None


def test_explicit_max_tokens_remains_available():
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport, max_tokens=1024)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    assert transport.calls[0]["max_tokens"] == 1024


def test_sdk_request_omits_max_tokens_when_provider_managed():
    """The ordinary SDK request leaves output budgeting to the provider."""
    from types import SimpleNamespace

    seen = {}
    class Chat:
        def complete(self, **kwargs):
            seen.update(kwargs)
            message = SimpleNamespace(content=VALID_BATCH)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2,
                                      total_tokens=3),
                model=MODEL_ID, id="resp", created=0)
    transport = object.__new__(mc.MistralSDKTransport)
    transport._sdk = SimpleNamespace(chat=Chat())
    transport._sdk_no_retries = transport._sdk
    transport.complete(messages=[], model=MODEL_ID, temperature=1.0,
                       reasoning_effort="high", max_tokens=None,
                       response_format=None, random_seed=None)
    assert "max_tokens" not in seen


def test_every_call_path_carries_arm_sampling():
    """ALL call paths — the turn path (initial AND repair attempts) and
    every probe path (fingerprint, model-id, json-schema) — carry the arm
    sampling: temperature 1.0, reasoning_effort "high", and NEVER a top_p
    key (via a mock transport; FW-20 removed every temperature=0 override
    and the greedy top_p=1.0 dict)."""
    transport = MockTransport(outcomes=[
        raw_completion("garbage"),  # turn attempt 1: schema error
        raw_completion(VALID_BATCH),  # turn repair attempt
        raw_completion("probe-sample-1"),  # fingerprint probe (K=3,
        raw_completion("probe-sample-2"),  # one request per sample:
        raw_completion("probe-sample-3"),  # sequential mode)
        raw_completion("OK"),  # model-id probe
        raw_completion(VALID_BATCH),  # json-schema probe
    ])
    client = make_client(transport)
    result = client.action_batch(messages=turn_messages(),
                                 prompt_hash="sha256:test",
                                 attempts=TurnAttempts())
    assert result.batch.turn_id == "run1/t1"
    client.probe_fingerprint()
    client.probe_model_id()
    client.probe_json_schema_support(probe_prompt="return a batch")
    assert len(transport.calls) == 7
    assert [log.kind for log in client.call_logs] == [
        "initial", "repair", "probe", "probe", "probe", "probe", "probe"]
    for call in transport.calls:
        assert call["temperature"] == 1.0
        assert call["reasoning_effort"] == "high"
        assert "top_p" not in call


def test_call_log_records_reasoning_effort():
    """CallLog records reasoning_effort (additive field, default "high",
    freeze-safe)."""
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport)
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    log = client.call_logs[0]
    assert log.reasoning_effort == "high"
    assert log.to_dict()["reasoning_effort"] == "high"
    assert log.temperature == 1.0


def test_reasoning_effort_configurable_per_client():
    """A client-level reasoning_effort override reaches the transport and
    the call log (tests may override via mock transports)."""
    transport = MockTransport(outcomes=[raw_completion(VALID_BATCH)])
    client = make_client(transport, reasoning_effort="low")
    client.action_batch(messages=turn_messages(), prompt_hash="sha256:test",
                        attempts=TurnAttempts())
    assert transport.calls[0]["reasoning_effort"] == "low"
    assert client.call_logs[0].reasoning_effort == "low"


def test_installed_sdk_chat_complete_accepts_reasoning_effort():
    """FW-20 offline SDK verification: the installed mistralai package's
    Chat.complete accepts reasoning_effort as an explicit parameter and
    "high" is a valid ReasoningEffort value. Skipped gracefully when the
    package is absent."""
    chat = pytest.importorskip("mistralai.client.chat")
    effort = pytest.importorskip("mistralai.client.models.reasoningeffort")
    sig = inspect.signature(chat.Chat.complete)
    assert "reasoning_effort" in sig.parameters
    # ReasoningEffort is Union[Literal[...], UnrecognizedStr]: unwrap the
    # Literal member before checking the vocabulary
    literals = [
        arg for arg in typing.get_args(effort.ReasoningEffort)
        if typing.get_origin(arg) is typing.Literal
    ]
    assert literals, "ReasoningEffort lost its Literal vocabulary"
    assert all("high" in typing.get_args(lit) for lit in literals)


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


def test_physical_events_correlate_exact_call_log_and_preserve_transport_shape():
    class Transport:
        def __init__(self):
            self.calls = []
        def complete(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return mc.RawCompletion(
                content=VALID_BATCH, model=MODEL_ID, response_id="r",
                created=0, finish_reason="stop", prompt_tokens=2,
                completion_tokens=3, total_tokens=5)

    first_t, second_t = Transport(), Transport()
    first = make_client(first_t)
    events = []
    observed = MistralClient(api_key=CANARY_KEY, transport=second_t,
                             call_event_callback=events.append)
    first._complete(messages=[{"role": "user", "content": "x"}], attempt=1,
                    kind="initial", prompt_hash="p")
    observed._complete(messages=[{"role": "user", "content": "x"}], attempt=1,
                        kind="initial", prompt_hash="p")
    assert first_t.calls == second_t.calls
    assert [e["event"] for e in events] == [
        "model_request_started", "model_request_finished"]
    assert events[0]["call_id"] == observed.call_logs[0].call_id
    assert events[-1]["call_id"] == observed.call_logs[0].call_id


def test_call_lifecycle_finishes_when_log_sink_fails_or_transport_is_cancelled():
    events = []
    def bad_sink(_record):
        raise OSError("durable sink unavailable")
    client = MistralClient(api_key=CANARY_KEY,
                           transport=MockTransport(outcomes=[raw_completion(VALID_BATCH)]),
                           log_sink=bad_sink, call_event_callback=events.append)
    with pytest.raises(OSError, match="durable sink"):
        client._complete(messages=[{"role": "user", "content": "x"}],
                         attempt=1, kind="initial", prompt_hash="p")
    assert [e["event"] for e in events] == [
        "model_request_started", "model_request_finished"]

    class CancelTransport(MockTransport):
        def complete(self, **kwargs):
            raise KeyboardInterrupt()
    events.clear()
    cancelled = MistralClient(api_key=CANARY_KEY,
                              transport=CancelTransport(),
                              call_event_callback=events.append)
    with pytest.raises(KeyboardInterrupt):
        cancelled._complete(messages=[{"role": "user", "content": "x"}],
                            attempt=1, kind="initial", prompt_hash="p")
    assert [e["event"] for e in events] == [
        "model_request_started", "model_request_finished"]


def test_concurrent_physical_calls_get_unique_ids_and_logs():
    class Transport:
        def complete(self, **kwargs):
            return mc.RawCompletion(content=VALID_BATCH, model=MODEL_ID,
                                    response_id="r", created=0,
                                    finish_reason="stop", prompt_tokens=1,
                                    completion_tokens=1, total_tokens=2)
    client = MistralClient(api_key=CANARY_KEY, transport=Transport())
    errors = []
    def run():
        try:
            client._complete(messages=[{"role": "user", "content": "x"}],
                             attempt=1, kind="initial", prompt_hash="p")
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=run) for _ in range(12)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors
    assert len({record.call_id for record in client.call_logs}) == 12
