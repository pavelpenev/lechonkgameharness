# SPDX-License-Identifier: MIT
"""lechonk.model.turn_client — the ModelClient-protocol adapter over the
Mistral Studio client (WP2 package C integration seam).

``core.runner`` consumes the core-defined ``ModelClient`` protocol
(core/model_client.py: ``complete_turn`` / ``complete_repair`` — ONE logical
API attempt per protocol call; the RUNNER owns the ≤2-attempt cap and the
repair flow, decision 4). ``MistralClient`` (package A) owns transport, the
output enforcement (``json_schema`` per the WP2 spike decision, recorded in
the run manifest), per-call logging and the arm fingerprint. This adapter is
the seam between the two:

- ``complete_turn`` renders the versioned system + turn prompts (§7.5:
  [stable prefix] + memory + snapshot + delta + candidate table) and performs
  ONE logical API attempt, mapping the outcome to a ``ModelResponse``;
- ``complete_repair`` renders the class-specific repair prompt (§7.5)
  appended to the original conversation + the failed assistant output, and
  performs the single repair attempt;
- per-call API logging (decision 4): every protocol call appends one record
  with tokens, latency, cost, fingerprint and error — plus the rendered
  section sizes for the §3 token/budget table. The API key never appears
  (the client's ``CallLog`` is redaction-safe by construction; this module
  adds no prompt text — only sizes and hashes).

TOKEN MEASUREMENT (WP2 acceptance): the API-reported prompt/completion token
counts per call are the ground truth; the rendered memory/snapshot/delta/
candidate section sizes (``core.schemas.estimate_tokens``) are recorded
alongside them so the per-turn numbers can be compared against the §3
budgets (memory ≤ 1,500, snapshot ≤ 8,000, delta ≤ 800).

This module never imports civrealm (plan §7.1).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Optional

from lechonk.core.model_client import (
    REPAIR_TIMEOUT_MARGIN_S,
    ModelResponse,
    RepairPrompt,
    TurnPrompt,
)
from lechonk.core.schemas import estimate_tokens
from lechonk.model import prompts as prompt_templates
from lechonk.model.mistral_client import MistralClient

_TURN_ID_RE = re.compile(r"/t(\d+)$")


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    """Deterministic prompt hash: sha256 over the canonical JSON of the
    message list (the ModelCallRecord carries it; never the prompt text)."""
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _turn_number(turn_id: str) -> Optional[int]:
    """The turn number from the runner's turn id (``<run>/t<turn>``)."""
    m = _TURN_ID_RE.search(turn_id or "")
    return int(m.group(1)) if m else None


def _raw_attempts_from(content: str) -> list[str]:
    """Best-effort extraction of the model-emitted action refs from a
    malformed output (the ledger records one rejected attempt per emitted
    ref; an empty list lets the runner substitute ``<malformed-batch>``)."""
    text = (content or "").strip()
    candidates = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(match.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        actions = value.get("actions")
        if not isinstance(actions, list):
            return []
        refs = []
        for item in actions:
            if isinstance(item, dict) and "ref" in item:
                refs.append(str(item["ref"]))
            else:
                refs.append(str(item))
        return refs
    return []


class MistralTurnClient:
    """The core ``ModelClient`` protocol over a ``MistralClient``.

    One protocol call = one logical API attempt. The repair prompt carries
    the original conversation + the failed assistant output + the
    class-specific repair instruction (§7.5), preserving prompt caching.
    """

    def __init__(self, client: MistralClient, *,
                 api_log: Optional[Callable[[dict[str, Any]], None]] = None):
        self.client = client
        self._api_log = api_log
        self._turn_messages: Optional[list[dict[str, str]]] = None
        self._failed_content: str = ""
        self._failed_malformed: Optional[str] = None
        # One record per protocol call (decision 4 per-call API logging).
        self.call_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # the ModelClient protocol
    # ------------------------------------------------------------------

    def complete_turn(self, prompt: TurnPrompt) -> ModelResponse:
        messages = [
            {"role": "system",
             "content": prompt_templates.system_prompt()},
            {"role": "user",
             "content": prompt_templates.turn_prompt(
                 memory=prompt.memory, snapshot=prompt.snapshot,
                 delta=prompt.delta, candidate_summary=prompt.candidates)},
        ]
        self._turn_messages = messages
        self._failed_content = ""
        self._failed_malformed = None
        return self._attempt(messages, prompt, attempt=1, kind="initial")

    def complete_repair(self, prompt: RepairPrompt) -> ModelResponse:
        if self._turn_messages is None:
            raise RuntimeError(
                "complete_repair called before complete_turn (the runner "
                "calls them sequentially within one turn)")
        if prompt.error_class not in prompt_templates.REPAIR_TEMPLATES:
            # The unresponsive class (transport-level failure, no parseable
            # output): the repair is a bounded re-ask of the SAME turn
            # prompt — exactly MistralClient.action_batch's transport-error
            # behavior (the single repair slot re-asks the same turn).
            return self._attempt(list(self._turn_messages), prompt,
                                 attempt=2, kind="repair")
        detail = self._repair_detail(prompt)
        repair_text = prompt_templates.repair_prompt(
            prompt.error_class,
            detail=detail,
            failed_refs=", ".join(prompt.failed_refs) or "(none)",
            refreshed_candidates=prompt.refreshed_candidates,
        )
        failed_content = self._failed_content or (
            "(no model output — the previous attempt failed at transport "
            "level)")
        messages = list(self._turn_messages) + [
            {"role": "assistant", "content": failed_content},
            {"role": "user", "content": repair_text},
        ]
        return self._attempt(messages, prompt, attempt=2, kind="repair")

    # ------------------------------------------------------------------
    # one logical attempt + the per-call API log record
    # ------------------------------------------------------------------

    def _repair_detail(self, prompt: RepairPrompt) -> str:
        if prompt.error_class == "schema_error":
            return (f"The output violated the ActionBatch schema: "
                    f"{self._failed_malformed or 'malformed output'}")
        if prompt.error_class == "state_fog_error":
            return ("The refs above are well-formed candidate IDs but were "
                    "not in this turn's offered candidate table (the model "
                    "acted on state it could not see).")
        if prompt.error_class == "illegal_in_rule":
            return ("The refs above were offered this turn but are no longer "
                    "legal on revalidation (stale after a mid-batch "
                    "mutation).")
        return ("The previous attempt produced no usable output; re-emit "
                "the ActionBatch.")

    def _attempt(self, messages: list[dict[str, str]], prompt: TurnPrompt,
                 *, attempt: int, kind: str) -> ModelResponse:
        prompt_hash = _prompt_hash(messages)
        # Budget-aware repair gating (WP4): on the repair attempt the
        # prompt's budget_s (the runner's remaining ACTING budget) becomes a
        # per-call hard request timeout (remaining minus the margin), so a
        # timed-out repair surfaces as a transport failure on THIS attempt
        # and the turn never exceeds its deadline by construction. The
        # repair's transport-level internal retry is DISABLED: the harness's
        # attempt structure (attempt 1 + one repair) is the retry policy, and
        # a transport retry inside the budgeted repair would silently double
        # its time (observed in wp4-eval2 turn 13: ~55 s budget-derived
        # timeout ran ~111 s, truncating the turn on the 180 s deadline).
        # The initial attempt keeps the client's configured default timeout
        # and the transport's transparent retries.
        timeout_ms: Optional[int] = None
        is_repair = kind == "repair"
        if is_repair and prompt.budget_s is not None:
            timeout_ms = int(max(0.0, prompt.budget_s
                                 - REPAIR_TIMEOUT_MARGIN_S) * 1000)
        outcome = self.client.complete_once(
            messages=messages, prompt_hash=prompt_hash, attempt=attempt,
            kind=kind, timeout_ms=timeout_ms,
            disable_transport_retries=is_repair)
        # The client logs every logical attempt; correlate this call's log.
        call_log = self.client.call_logs[-1]
        if outcome.raw_content:
            self._failed_content = outcome.raw_content
        if outcome.malformed is not None:
            self._failed_malformed = outcome.malformed

        if outcome.error is not None:
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                error=outcome.error, latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd or 0.0,
                fingerprint=call_log.fingerprint)
            outcome_kind = "transport_error"
            raw_attempts: list[str] = []
        elif outcome.malformed is not None or outcome.batch is None:
            raw_attempts = _raw_attempts_from(outcome.raw_content)
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                raw_attempts=raw_attempts,
                malformed=outcome.malformed or "no batch",
                latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd or 0.0,
                fingerprint=call_log.fingerprint)
            outcome_kind = "schema_error"
        else:
            raw_attempts = []
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                batch=outcome.batch, latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd or 0.0,
                fingerprint=call_log.fingerprint)
            outcome_kind = "ok"

        record = {
            "call_id": prompt.call_id,
            "turn": _turn_number(prompt.turn_id),
            "attempt": attempt,
            "kind": kind,
            "prompt_hash": prompt_hash,
            "model_requested": call_log.model_requested,
            "model_responded": call_log.model_responded,
            "enforcement": call_log.enforcement,
            "temperature": call_log.temperature,
            "latency_ms": call_log.latency_ms,
            # TOKEN MEASUREMENT: the API-reported counts are the ground truth.
            "tokens_in": call_log.tokens_in,
            "tokens_out": call_log.tokens_out,
            "tokens_total": call_log.tokens_total,
            "cost_usd": call_log.cost_usd,
            "finish_reason": call_log.finish_reason,
            "response_id": call_log.response_id,
            "fingerprint": call_log.fingerprint,
            "transport_retries": call_log.transport_retries,
            "error": call_log.error,
            "outcome": outcome_kind,
            "raw_attempts": raw_attempts,
            # The rendered section sizes (estimate_tokens) for the §3 budget
            # comparison: memory ≤ 1,500, snapshot ≤ 8,000, delta ≤ 800.
            "section_tokens": {
                "memory": estimate_tokens(prompt.memory),
                "snapshot": estimate_tokens(prompt.snapshot),
                "delta": estimate_tokens(prompt.delta),
                "candidates": estimate_tokens(prompt.candidates),
            },
            "ts": call_log.ts,
        }
        self.call_records.append(record)
        if self._api_log is not None:
            self._api_log(record)
        return response
