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

import pytest

from lechonk.core.model_client import (
    REPAIR_TIMEOUT_MARGIN_S,
    RepairPrompt,
    TurnPrompt,
)
from lechonk.core.schemas import estimate_tokens
from lechonk.model import mistral_client as mc
from lechonk.model import prompts
from lechonk.model.prompts import (
    ASSEMBLED_PROMPT_TOKEN_BUDGET,
    PromptBudgetExceeded,
)
from lechonk.model.turn_client import (
    MistralTurnClient,
    _prompt_hash,
    _raw_attempts_from,
    _turn_number,
)

CANARY = "sk-CANARY-9f3a91c2-do-not-leak"

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
                 timeout_ms=None, disable_transport_retries=False):
        self.calls.append({
            "messages": messages, "model": model,
            "temperature": temperature, "max_tokens": max_tokens,
            "response_format": response_format, "random_seed": random_seed,
            "timeout_ms": timeout_ms,
            "disable_transport_retries": disable_transport_retries,
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
        transport_retries=kw.get("transport_retries", 0))


def make_client(outcomes, *, enforcement="json_schema"):
    transport = MockTransport(outcomes=outcomes)
    client = mc.MistralClient(api_key=CANARY, transport=transport,
                              enforcement=enforcement)
    return client, transport


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
    assert client.call_logs[-1].tokens_in == 0


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
        assert record["cost_usd"] == round(
            (1000 * 0.50 + 200 * 1.50) / 1_000_000, 6)
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
    assert records[1]["tokens_in"] == 0


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
    oversized_table = "x" * (prompts.ASSEMBLED_PROMPT_TOKEN_BUDGET * 4)
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
    monkeypatch.setattr(prompts, "ASSEMBLED_PROMPT_TOKEN_BUDGET", 1)
    client, _ = make_client([raw(VALID_BATCH)])
    with pytest.raises(prompts.PromptConfigError):
        MistralTurnClient(client)
    # With the real (trivially small) templates, construction succeeds.
    monkeypatch.undo()
    client, _ = make_client([raw(VALID_BATCH)])
    MistralTurnClient(client)
