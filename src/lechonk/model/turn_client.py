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
import inspect
import json
import re
import time
from typing import Any, Callable, Mapping, Optional

from lechonk.core.model_client import (
    REPAIR_TIMEOUT_MARGIN_S,
    PromptBudgetExceeded,
    ModelResponse,
    RepairPrompt,
    RulesetLookup,
    TurnPrompt,
)
from lechonk.core.jsonutil import strict_dumps
from lechonk.core.schemas import estimate_tokens
from lechonk.model import prompts as prompt_templates
from lechonk.model.prompts import (
    REPAIR_CONVERSATION_TOKEN_BUDGET,
)
from lechonk.model.mistral_client import MistralClient, ProviderCapacityError
from lechonk.model.conversation import (
    ConversationConfig, ConversationManager, CompactionRequired,
    ConversationError, _validate_compaction_summary,
)

_TURN_ID_RE = re.compile(r"/t(\d+)\Z")
CONVERSATION_REPAIR_TOKEN_BUDGET = 520_000
# Large 4 is documented with a one-million-token context window.  Keep a
# conservative margin for the generated summary and provider framing; this is
# a request preflight, not an output-token cap for normal turns.
COMPACTION_CONTEXT_WINDOW_TOKENS = 1_000_000
COMPACTION_OUTPUT_HEADROOM_TOKENS = 64_000
# Request-framing/calibration guard only; this is not a model output cap.
COMPACTION_INPUT_GUARD_TOKENS = 8_192
_USE_CLIENT_DEFAULT = object()

LOOKUP_MAX_TOPICS = 3
LOOKUP_MAX_TOOL_MESSAGES = 16
LOOKUP_MAX_RESULTS = 8
LOOKUP_MAX_OUTPUT_CHARS = 8_000
LOOKUP_MAX_QUERY_CHARS = 512
LOOKUP_CATEGORY_ALIASES = (
    "tech", "techs", "technology", "technologies", "unit", "units",
    "building", "buildings", "improvement", "improvements", "terrain",
    "terrains", "government", "governments", "extra", "extras",
)

FREECIV_LOOKUP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_freeciv",
        "description": (
            "Read-only lookup in the bundled Freeciv ruleset encyclopedia. "
            "Use it only for game rules, units, buildings, technologies, "
            "terrain, governments, or advances; it cannot inspect or change "
            "the current game."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": LOOKUP_MAX_QUERY_CHARS},
                "categories": {
                    "type": "array", "items": {"type": "string",
                                                   "enum": list(LOOKUP_CATEGORY_ALIASES)},
                    "maxItems": 8,
                },
                "limit": {"type": "integer", "minimum": 1,
                          "maximum": LOOKUP_MAX_RESULTS},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _prompt_hash(messages: list[dict[str, Any]]) -> str:
    """Deterministic prompt hash: sha256 over the canonical JSON of the
    message list (the ModelCallRecord carries it; never the prompt text)."""
    canonical = strict_dumps(messages, sort_keys=True)
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


def _check_repair_conversation_budget(
        messages: list[dict[str, Any]], budget: int = None) -> None:
    """FW-13: the repair is inside the budget check. The COMPLETE repair
    conversation — prior messages (system prompt + turn prompt), the failed
    assistant output, the repair instruction, and the re-attached candidate
    table — must fit the repair-conversation budget (W1-7: a different sum
    from the initial check's section-body budget — this one includes the
    fixed text). An oversize repair raises PromptBudgetExceeded so the
    runner's dedicated bounded fail-stop handles it, never a silent
    oversize."""
    total = sum(estimate_tokens(
        m["content"] if isinstance(m.get("content"), str)
        else strict_dumps(m.get("content"), sort_keys=True))
        for m in messages)
    limit = (REPAIR_CONVERSATION_TOKEN_BUDGET if budget is None else budget)
    if total > limit:
        raise PromptBudgetExceeded(
            f"complete repair conversation exceeds the repair-conversation "
            f"budget: {total} > {limit} tokens")


def _repair_excerpt(content: str, *, max_chars: int = 12_000) -> str:
    """Keep repair input bounded while preserving both response edges."""
    if len(content) <= max_chars:
        return content
    head = max_chars // 2
    tail = max_chars - head
    return (content[:head] + "\n\n[model output elided for repair]\n\n"
            + content[-tail:])


class MistralTurnClient:
    """The core ``ModelClient`` protocol over a ``MistralClient``.

    One protocol call = one logical API attempt. The repair prompt carries
    the original conversation + the failed assistant output + the
    class-specific repair instruction (§7.5), preserving prompt caching.
    """

    def __init__(self, client: MistralClient, *,
                 api_log: Optional[Callable[[dict[str, Any]], None]] = None,
                 output_callback: Optional[Callable[[dict[str, Any]], None]] = None,
                 context_policy: str = "legacy",
                 prompt_profile: str = "legacy_eval",
                 prompt_bundle_id: str | None = None,
                 prompt_snapshot: Optional[prompt_templates.PromptSnapshot] = None,
                 conversation_manager: Optional[ConversationManager] = None,
                 clock: Callable[[], float] = time.monotonic,
                 ruleset_lookup: Optional[RulesetLookup] = None,
                 enable_ruleset_lookup: bool = False):
        # FW-13 startup invariant: the fixed prompt text (system prompt +
        # stable turn prefix) must fit the assembled-prompt budget on its
        # own, or no turn could ever be assembled. Computed once, here at
        # initialization — fail fast at startup, not as a mid-game kill on
        # some later turn.
        self.prompt_profile = prompt_templates.normalize_prompt_profile(
            prompt_profile)
        self.prompt_bundle_id = prompt_bundle_id
        if prompt_bundle_id is not None:
            prompt_templates.prompt_bundle(prompt_bundle_id,
                                           profile=self.prompt_profile)
        self.prompt_snapshot = (prompt_snapshot or
                                prompt_templates.capture_prompt_snapshot(
                                    self.prompt_profile,
                                    bundle_id=prompt_bundle_id))
        if (self.prompt_snapshot.profile != self.prompt_profile or
                self.prompt_snapshot.bundle_id != prompt_bundle_id):
            raise ValueError("prompt_snapshot does not match selected prompt bundle")
        prompt_templates.check_fixed_prompt_budget(
            self.prompt_profile, bundle_id=prompt_bundle_id,
            snapshot=self.prompt_snapshot)
        self.client = client
        self._api_log = api_log
        self._output_callback = output_callback
        if context_policy not in {"legacy", "conversation_v1"}:
            raise ValueError("context_policy must be legacy or conversation_v1")
        if (self.prompt_profile != "legacy_eval"
                and context_policy != "conversation_v1"):
            raise ValueError(
                "non-legacy prompt profiles require context_policy="
                "'conversation_v1'")
        self.context_policy = context_policy
        self._conversation_manager = conversation_manager
        self._conversation_messages: list[dict[str, Any]] = []
        self._conversation_turn_messages: Optional[list[dict[str, Any]]] = None
        self._conversation_staged: list[dict[str, Any]] = []
        self._conversation_attempt_message: Optional[dict[str, Any]] = None
        self._turn_messages: Optional[list[dict[str, str]]] = None
        self._failed_content: str = ""
        self._failed_malformed: Optional[str] = None
        self._hybrid_active = False
        self._active_budget_s: Optional[float] = None
        self._turn_deadline: Optional[float] = None
        self._clock = clock
        self._ruleset_lookup = ruleset_lookup
        self._enable_ruleset_lookup = bool(enable_ruleset_lookup and
                                           ruleset_lookup is not None)
        self._last_tool_calls: list[dict[str, Any]] = []
        self._lookup_exchange: list[dict[str, Any]] = []
        self._compaction_used = False
        self._active_logical_call_id: Optional[str] = None
        self._active_turn_number: Optional[int] = None
        # One record per protocol call (decision 4 per-call API logging).
        self.call_records: list[dict[str, Any]] = []

    @property
    def conversation_enabled(self) -> bool:
        return self.context_policy == "conversation_v1"

    @property
    def prompt_hashes(self) -> dict[str, str]:
        """Copy of the immutable prompt identity used by this client."""
        return self.prompt_snapshot.manifest_hashes

    def set_ruleset_lookup(self, lookup: Optional[RulesetLookup]) -> None:
        """Install the startup-scoped encyclopedia lookup, if eligible.

        The runner invokes this after the adapter session has captured the
        immutable ruleset reference.  Tool use is intentionally restricted to
        the interactive gameplay profile; legacy and evaluation clients keep
        their original request shape even if a callable is accidentally
        supplied.
        """
        eligible = (self.context_policy == "conversation_v1" and
                    self.prompt_profile in {"gameplay_v2", "gameplay_v3", "gameplay_v4", "gameplay_v5", "gameplay_v6"})
        if lookup is not None and not callable(lookup):
            raise TypeError("ruleset lookup must be callable or None")
        self._ruleset_lookup = lookup if eligible else None
        self._enable_ruleset_lookup = bool(eligible and lookup is not None)

    def _ensure_conversation_manager(self) -> ConversationManager:
        if self._conversation_manager is None:
            self._conversation_manager = ConversationManager(
                system_message={"role": "system",
                                "content": prompt_templates.system_prompt(
                                    self.prompt_profile,
                                    bundle_id=self.prompt_bundle_id,
                                    snapshot=self.prompt_snapshot)},
                config=ConversationConfig(),
                prompt_cache_key=getattr(self.client, "prompt_cache_key", None),
            )
        return self._conversation_manager

    def _compact_with_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask Mistral for a bounded structured summary before compaction.

        This is deliberately outside the ActionBatch protocol.  If the model
        cannot return a valid object, compaction fails closed and the caller
        keeps the complete conversation rather than silently dropping history.
        """
        # Replay the original ordered messages so this compaction request has
        # the same cacheable prefix as the conversation itself.  The final
        # instruction is the only new suffix; putting the whole conversation
        # inside one user string would destroy role boundaries and cache reuse.
        prompt = (
            "Summarize the preceding historical game conversation for future "
            "play. Return JSON with keys version, summary, commitments, "
            "uncertainties, and facts; version must be 1. Commitments may be "
            "legacy strings or strict typed Commitment objects; historical_note "
            "is optional. Keep it concise. "
            "Mark observations historical; the "
            "next authoritative snapshot and candidate table override it.")
        instruction = {"role": "user", "content": prompt}
        compaction_messages = list(messages) + [instruction]
        manager = self._conversation_manager
        if manager is not None:
            preflight = manager.estimate_prefixed_request(
                messages, [instruction], guard_tokens=COMPACTION_INPUT_GUARD_TOKENS)
        else:
            raw_estimate = sum(estimate_tokens(strict_dumps(message, sort_keys=True))
                               for message in compaction_messages)
            preflight = {"effective": raw_estimate + COMPACTION_INPUT_GUARD_TOKENS,
                         "raw": raw_estimate, "method": "raw",
                         "guard": COMPACTION_INPUT_GUARD_TOKENS,
                         "calibration": {}}
        # This is an input/framing estimate plus reserved headroom, not a
        # hard guarantee about an uncapped model completion length.
        estimated_input = preflight["effective"]
        if estimated_input + COMPACTION_OUTPUT_HEADROOM_TOKENS > COMPACTION_CONTEXT_WINDOW_TOKENS:
            error = ("compaction request exceeds context headroom: "
                     f"estimated_input={estimated_input}, raw={preflight['raw']}, "
                     f"method={preflight['method']}, guard={preflight['guard']}, "
                     f"headroom={COMPACTION_OUTPUT_HEADROOM_TOKENS}, "
                     f"context={COMPACTION_CONTEXT_WINDOW_TOKENS}")
            self._record_compaction_call(error=error, preflight=preflight)
            raise ConversationError(error)
        timeout_ms = None
        if self._turn_deadline is not None:
            remaining = self._turn_deadline - self._clock()
            if remaining <= REPAIR_TIMEOUT_MARGIN_S:
                error = "turn budget exhausted before compaction"
                self._record_compaction_call(error=error, preflight=preflight)
                raise TimeoutError(error)
            timeout_ms = int((remaining - REPAIR_TIMEOUT_MARGIN_S) * 1000)
        call_log_count = len(self.client.call_logs)
        try:
            raw = self.client._complete(  # transport-only, no ActionBatch parser
                messages=compaction_messages,
                attempt=1, kind="conversation_compaction",
                prompt_hash=_prompt_hash(compaction_messages),
                response_format={"type": "json_object"},
                disable_transport_retries=True,
                timeout_ms=timeout_ms,
                prompt_cache_key=(getattr(self._conversation_manager,
                                          "prompt_cache_key", None)
                                  or getattr(self.client, "prompt_cache_key", None)),
                call_event_context={
                    "logical_call_id": self._active_logical_call_id,
                    "turn": self._active_turn_number,
                    "attempt": 1,
                    "physical_kind": "conversation_compaction",
                })
        except Exception as exc:
            call_log = self._compaction_error_call_log(exc, call_log_count)
            self._record_compaction_call(error=str(exc), call_log=call_log,
                                         preflight=preflight, error_exc=exc)
            raise
        text = raw.answer_text or raw.content or ""
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as exc:
            lookup = getattr(self.client, "_call_log_for", None)
            call_log = (lookup(getattr(raw, "call_id", None))
                        if lookup is not None else None)
            if call_log is None and len(self.client.call_logs) > call_log_count:
                call_log = self.client.call_logs[-1]
            self._record_compaction_call(
                raw=raw, error="compaction returned invalid JSON", call_log=call_log,
                preflight=preflight)
            return self._repair_compaction(compaction_messages, messages, raw,
                                           "compaction returned invalid JSON")
        if not isinstance(value, dict):
            lookup = getattr(self.client, "_call_log_for", None)
            call_log = (lookup(getattr(raw, "call_id", None))
                        if lookup is not None else None)
            if call_log is None and len(self.client.call_logs) > call_log_count:
                call_log = self.client.call_logs[-1]
            self._record_compaction_call(
                raw=raw, error="compaction returned a non-object", call_log=call_log,
                preflight=preflight)
            return self._repair_compaction(compaction_messages, messages, raw,
                                           "compaction returned a non-object")
        try:
            value = _validate_compaction_summary(value)
        except ConversationError as exc:
            lookup = getattr(self.client, "_call_log_for", None)
            call_log = (lookup(getattr(raw, "call_id", None))
                        if lookup is not None else None)
            if call_log is None and len(self.client.call_logs) > call_log_count:
                call_log = self.client.call_logs[-1]
            self._record_compaction_call(
                raw=raw, error=f"compaction summary validation failed: {exc}",
                call_log=call_log, preflight=preflight)
            return self._repair_compaction(
                compaction_messages, messages, raw,
                f"compaction summary validation failed: {exc}")
        lookup = getattr(self.client, "_call_log_for", None)
        call_log = (lookup(getattr(raw, "call_id", None))
                    if lookup is not None else None)
        if call_log is None and len(self.client.call_logs) > call_log_count:
            call_log = self.client.call_logs[-1]
        self._record_compaction_call(raw=raw, call_log=call_log,
                                     preflight=preflight)
        return value

    def _repair_compaction(self, original_messages: list[dict[str, Any]],
                           prefix_messages: list[dict[str, Any]], raw: Any,
                           error: str) -> dict[str, Any]:
        """Make one bounded semantic repair attempt; never repair transport errors."""
        raw_text = getattr(raw, "answer_text", None) or getattr(raw, "content", "") or ""
        raw_text = raw_text or "[empty model output]"
        repair = ("The previous compaction output was invalid: " + error +
                  ". Return only a JSON object with exactly these keys: "
                  "version (integer 1), summary (nonempty string), commitments "
                  "(legacy strings or strict Commitment objects), uncertainties "
                  "(strings), facts (strings), and optional historical_note "
                  "(string). Do not invent facts.")
        suffix = [*original_messages[len(prefix_messages):],
                  {"role": "assistant", "content": raw_text[:12_000]},
                  {"role": "user", "content": repair}]
        repair_messages = list(prefix_messages) + suffix
        manager = self._conversation_manager
        if manager is not None:
            preflight = manager.estimate_prefixed_request(
                prefix_messages, suffix, guard_tokens=COMPACTION_INPUT_GUARD_TOKENS)
        else:
            raw_estimate = sum(estimate_tokens(strict_dumps(message, sort_keys=True))
                               for message in repair_messages)
            preflight = {"effective": raw_estimate + COMPACTION_INPUT_GUARD_TOKENS,
                         "raw": raw_estimate, "method": "raw",
                         "guard": COMPACTION_INPUT_GUARD_TOKENS, "calibration": {}}
        if preflight["effective"] + COMPACTION_OUTPUT_HEADROOM_TOKENS > COMPACTION_CONTEXT_WINDOW_TOKENS:
            bounded = "compaction repair exceeds context headroom"
            self._record_compaction_call(error=bounded, preflight=preflight,
                                         attempt=2, kind="conversation_compaction_repair")
            raise ConversationError(bounded)
        if self._turn_deadline is not None:
            remaining = self._turn_deadline - self._clock()
            if remaining <= REPAIR_TIMEOUT_MARGIN_S:
                bounded = "turn budget exhausted before compaction repair"
                self._record_compaction_call(error=bounded, preflight=preflight,
                                             attempt=2, kind="conversation_compaction_repair")
                raise TimeoutError(bounded)
            timeout_ms = int((remaining - REPAIR_TIMEOUT_MARGIN_S) * 1000)
        else:
            timeout_ms = None
        count = len(self.client.call_logs)
        try:
            repaired = self.client._complete(
                messages=repair_messages, attempt=2,
                kind="conversation_compaction_repair",
                prompt_hash=_prompt_hash(repair_messages),
                response_format={"type": "json_object"},
                disable_transport_retries=True, timeout_ms=timeout_ms,
                prompt_cache_key=(getattr(manager, "prompt_cache_key", None)
                                  or getattr(self.client, "prompt_cache_key", None)),
                call_event_context={"logical_call_id": self._active_logical_call_id,
                                    "turn": self._active_turn_number, "attempt": 2,
                                    "physical_kind": "conversation_compaction_repair"})
        except Exception as exc:
            call_log = self._compaction_error_call_log(exc, count)
            self._record_compaction_call(error=str(exc), call_log=call_log,
                                         preflight=preflight, attempt=2,
                                         kind="conversation_compaction_repair",
                                         error_exc=exc)
            raise
        text = repaired.answer_text or repaired.content or ""
        repair_lookup = getattr(self.client, "_call_log_for", None)
        repair_call_log = (repair_lookup(getattr(repaired, "call_id", None))
                           if repair_lookup is not None else None)
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ConversationError("compaction repair returned a non-object")
            value = _validate_compaction_summary(value)
        except (ValueError, TypeError) as exc:
            message = "compaction returned invalid JSON"
            self._record_compaction_call(raw=repaired, error=message,
                call_log=repair_call_log,
                preflight=preflight, attempt=2, kind="conversation_compaction_repair")
            raise ConversationError(message) from exc
        except ConversationError as exc:
            self._record_compaction_call(raw=repaired,
                error=f"compaction repair validation failed: {exc}",
                call_log=repair_call_log,
                preflight=preflight, attempt=2, kind="conversation_compaction_repair")
            raise ConversationError(str(exc)) from exc
        self._record_compaction_call(raw=repaired, preflight=preflight,
                                     call_log=repair_call_log,
                                     attempt=2, kind="conversation_compaction_repair")
        return value

    def _compaction_error_call_log(self, exc: BaseException, before_count: int) -> Any:
        """Associate transport failures by physical partial call ID first."""
        partial = getattr(exc, "partial", None)
        call_id = getattr(partial, "call_id", None)
        lookup = getattr(self.client, "_call_log_for", None)
        if call_id is not None and lookup is not None:
            call_log = lookup(call_id)
            if call_log is not None:
                return call_log
        if len(self.client.call_logs) > before_count:
            return self.client.call_logs[-1]
        return None

    def _record_compaction_call(self, *, raw: Any = None,
                                error: Optional[str] = None,
                                call_log: Any = None,
                                preflight: Optional[dict[str, Any]] = None,
                                attempt: int = 1,
                                kind: str = "conversation_compaction",
                                error_exc: Optional[BaseException] = None) -> None:
        """Expose compaction calls through the same durable API log sinks."""
        log = call_log
        if log is None:
            # Preflight and transport failures can happen before the client
            # has a CallLog.  Keep them observable through the same sinks.
            record = {
                "call_id": None, "logical_call_id": self._active_logical_call_id,
                "turn": self._active_turn_number, "attempt": attempt,
                "kind": kind, "prompt_hash": None,
                "model_requested": getattr(self.client, "model", None),
                "model_responded": None, "enforcement": None,
                "temperature": None, "latency_ms": None,
                "tokens_in": None, "tokens_out": None,
                "tokens_total": None, "cached_tokens": None,
                "cost_usd": None, "finish_reason": None,
                "response_id": None, "fingerprint": None,
                "transport_retries": None, "error": error,
                "error_kind": (getattr(call_log, "error_kind", None)
                               if call_log is not None else
                               ("provider_capacity" if error_exc is not None and
                                error_exc.__class__.__name__ == "ProviderCapacityError"
                                else None)),
                "http_status": getattr(error_exc, "http_status", None),
                "provider_type": getattr(error_exc, "provider_type", None),
                "provider_code": getattr(error_exc, "provider_code", None),
                "retry_after": getattr(error_exc, "retry_after", None),
                "usage_status": "unavailable",
                "cost_status": "unknown",
                "outcome": "error" if error else "ok",
                "raw_attempts": [self._compaction_raw_trace(raw)] if raw else [],
                "section_tokens": {}, "preflight": preflight, "ts": time.time(),
            }
            self.call_records.append(record)
            if self._api_log is not None:
                self._api_log(record)
            if self._output_callback is not None:
                try:
                    self._output_callback({
                        "event": "model_output", "turn": self._active_turn_number,
                        "logical_call_id": self._active_logical_call_id,
                        "attempt": attempt,
                        "kind": kind,
                        "reasoning": getattr(raw, "reasoning_text", "") if raw else "",
                        "answer": (getattr(raw, "answer_text", None) or
                                   getattr(raw, "content", "")) if raw else "",
                        "content_chunks": getattr(raw, "content_chunks", None) if raw else None,
                        "error": error,
                    })
                except Exception:
                    pass
            return
        record = {
            "call_id": log.call_id, "logical_call_id": self._active_logical_call_id,
            "turn": self._active_turn_number, "attempt": attempt,
            "kind": kind, "prompt_hash": log.prompt_hash,
            "model_requested": log.model_requested,
            "model_responded": log.model_responded,
            "enforcement": log.enforcement, "temperature": log.temperature,
            "latency_ms": log.latency_ms, "tokens_in": log.tokens_in,
            "tokens_out": log.tokens_out, "tokens_total": log.tokens_total,
            "cached_tokens": log.cached_tokens, "cost_usd": log.cost_usd,
            "finish_reason": log.finish_reason, "response_id": log.response_id,
            "fingerprint": log.fingerprint, "transport_retries": log.transport_retries,
            "error": error or log.error, "outcome": "error" if error else "ok",
            "error_kind": getattr(log, "error_kind", None),
            "http_status": getattr(log, "http_status", None),
            "provider_type": getattr(log, "provider_type", None),
            "provider_code": getattr(log, "provider_code", None),
            "retry_after": getattr(log, "retry_after", None),
            "usage_status": getattr(log, "usage_status", None),
            "cost_status": getattr(log, "cost_status", None),
            "raw_attempts": ([self._compaction_raw_trace(raw)] if raw else []),
            "section_tokens": {}, "preflight": preflight,
            "ts": log.ts,
        }
        self.call_records.append(record)
        if self._api_log is not None:
            self._api_log(record)
        if self._output_callback is not None:
            event = {
                "event": "model_output", "turn": self._active_turn_number,
                "logical_call_id": self._active_logical_call_id, "attempt": attempt,
                "kind": kind,
                "reasoning": getattr(raw, "reasoning_text", "") if raw else "",
                "answer": (getattr(raw, "answer_text", None) or
                           getattr(raw, "content", "")) if raw else "",
                "content_chunks": getattr(raw, "content_chunks", None) if raw else None,
                "error": error or log.error,
            }
            try:
                self._output_callback(event)
            except Exception:
                pass

    @staticmethod
    def _compaction_raw_trace(raw: Any) -> dict[str, Any]:
        """Durable, redaction-free raw completion fields for compaction audits."""
        return {
            "call_id": getattr(raw, "call_id", None),
            "reasoning": getattr(raw, "reasoning_text", "") or "",
            "answer": getattr(raw, "answer_text", None) or getattr(raw, "content", "") or "",
            "content_chunks": getattr(raw, "content_chunks", None),
            "tool_calls": getattr(raw, "tool_calls", None),
        }

    @staticmethod
    def _assistant_message(outcome: Any) -> Optional[dict[str, Any]]:
        """Return the exact JSON-compatible assistant content when present."""
        raw = getattr(outcome, "raw", None)
        if raw is None:
            return None
        # Tool-call continuations have a stricter wire shape than the local
        # trace representation.  Streaming Mistral responses may expose
        # thinking/text transition chunks (including empty text chunks) in
        # ``assistant_message.content``.  Those chunks are useful for
        # observability, but are not valid assistant content for the next
        # request.  Preserve the calls and use normalized API-valid chunks;
        # reasoning and raw chunks remain available on RawCompletion/traces.
        tool_calls = getattr(raw, "tool_calls", None)
        exact = getattr(raw, "assistant_message", None)
        if tool_calls is None and isinstance(exact, Mapping):
            tool_calls = exact.get("tool_calls")
        if tool_calls:
            # Mistral's tool-call continuation endpoint accepts a string
            # assistant content reliably, while typed reasoning chunks can
            # still be rejected by provider validation.  Keep full reasoning
            # and raw chunks in RawCompletion/model_traces, but use only the
            # normalized answer string for this protocol transition.
            content = getattr(raw, "answer_text", None)
            if not isinstance(content, str):
                content = getattr(raw, "content", "")
            if not isinstance(content, str):
                content = ""
            try:
                calls = json.loads(json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                return None
            return {"role": "assistant", "content": content,
                    "tool_calls": calls}
        if exact is not None:
            try:
                return json.loads(json.dumps(exact, ensure_ascii=False))
            except (TypeError, ValueError):
                pass
        chunks = getattr(raw, "content_chunks", None)
        if chunks:
            # Copy through JSON to prevent later mutation of SDK-owned data.
            try:
                content: Any = json.loads(json.dumps(chunks, ensure_ascii=False))
            except (TypeError, ValueError):
                content = getattr(raw, "answer_text", None) or getattr(raw, "content", "")
        else:
            content = getattr(raw, "answer_text", None) or getattr(raw, "content", "")
        if content in (None, "", []):
            return None
        message: dict[str, Any] = {"role": "assistant", "content": content}
        return message

    def _conversation_base(self) -> list[dict[str, Any]]:
        return list(self._conversation_messages)

    def commit_turn(self, execution_report: str | dict[str, Any] | None = None) -> None:
        """Commit the completed turn after execution and finish resolution.

        The runner calls this only after action results are known.  A staged
        assistant response is therefore never allowed to imply successful
        execution in a later request before the turn has actually completed.
        """
        if not self.conversation_enabled:
            return
        if self._conversation_turn_messages is None or not self._conversation_staged:
            if (execution_report is not None and self._conversation_manager is not None
                    and hasattr(self._conversation_manager, "append_execution_report")):
                report_obj = (execution_report if isinstance(execution_report, dict)
                              else {"text": str(execution_report)})
                # The manager still has the staged user prompt from the
                # failed model attempt. Discard that incomplete transaction
                # before recording the engine-only fallback report.
                abort_manager = getattr(self._conversation_manager, "abort_turn", None)
                if callable(abort_manager):
                    abort_manager()
                self._conversation_manager.append_execution_report(report_obj)
                self._conversation_messages = self._conversation_manager.request_messages()
            self.abort_turn()
            return
        if self._conversation_manager is not None:
            report_obj = None
            if execution_report is not None:
                report_obj = (execution_report if isinstance(execution_report, dict)
                              else {"text": str(execution_report)})
            # A failed tool-final repair leaves only assistant(tool_calls) and
            # role=tool messages staged.  Never hand that incomplete protocol
            # transaction to ConversationManager (which correctly rejects an
            # orphaned tool exchange); abort it and retain only the engine's
            # authoritative execution report.
            if (self._conversation_staged and
                    self._conversation_staged[-1].get("role") != "assistant"):
                abort_manager = getattr(self._conversation_manager,
                                        "abort_turn", None)
                if callable(abort_manager):
                    abort_manager()
                if report_obj is not None:
                    self._conversation_manager.append_execution_report(report_obj)
                    self._conversation_messages = self._conversation_manager.request_messages()
                self.abort_turn()
                return
            self._conversation_manager.commit_exchange(
                list(self._conversation_turn_messages) +
                list(self._conversation_staged), execution_report=report_obj)
            self._conversation_messages = self._conversation_manager.request_messages()
        else:
            self._conversation_messages = (
                list(self._conversation_turn_messages) +
                list(self._conversation_staged))
            if execution_report is not None:
                if isinstance(execution_report, dict):
                    text = strict_dumps(execution_report, sort_keys=True)
                else:
                    text = str(execution_report)
                self._conversation_messages.append({
                    "role": "user",
                    "content": (
                        "HISTORICAL EXECUTION REPORT (authoritative only for "
                        "what happened in the prior turn; current state below "
                        "overrides it):\n" + text),
                })
        self._conversation_turn_messages = None
        self._conversation_staged = []
        self._conversation_attempt_message = None

    def abort_turn(self) -> None:
        """Discard the current user/assistant transaction after a partial turn."""
        self._conversation_turn_messages = None
        self._conversation_staged = []
        self._conversation_attempt_message = None

    def _emit_output(self, *, prompt: TurnPrompt, attempt: int, kind: str,
                     outcome: Any) -> None:
        """Publish the complete returned reasoning and answer best-effort."""
        if self._output_callback is None:
            return
        raw = outcome.raw
        event: dict[str, Any] = {
            "event": "model_output",
            "turn": _turn_number(prompt.turn_id),
            "attempt": attempt,
            "kind": kind,
            "reasoning": raw.reasoning_text if raw is not None else "",
            "answer": (raw.answer_text if raw is not None and
                        raw.answer_text is not None else outcome.raw_content),
            "content_chunks": (raw.content_chunks if raw is not None else None),
            "tool_calls": (getattr(raw, "tool_calls", None)
                           if raw is not None else None),
            "error": outcome.error,
        }
        try:
            self._output_callback(event)
        except Exception:
            # A display/trace sink must never affect game execution.
            return

    # ------------------------------------------------------------------
    # the ModelClient protocol
    # ------------------------------------------------------------------

    def complete_turn(self, prompt: TurnPrompt) -> ModelResponse:
        self._active_logical_call_id = prompt.call_id
        self._active_turn_number = _turn_number(prompt.turn_id)
        self._active_budget_s = prompt.budget_s
        self._turn_deadline = (self._clock() + prompt.budget_s
                               if prompt.budget_s is not None else None)
        self._compaction_used = False
        current = {"role": "user", "content": prompt_templates.turn_prompt(
            memory=prompt.memory, snapshot=prompt.snapshot,
            delta=prompt.delta, candidate_summary=prompt.candidates,
            history=prompt.history, profile=self.prompt_profile,
            bundle_id=self.prompt_bundle_id,
            prompt_snapshot=self.prompt_snapshot)}
        if self.conversation_enabled:
            if self._conversation_turn_messages is not None:
                raise RuntimeError("complete_turn called before prior turn committed")
            manager = self._ensure_conversation_manager()
            # A resumed ConversationManager may carry a post-compaction cache
            # generation.  Synchronize the transport before the first normal
            # request; waiting until a later compaction would send the wrong
            # cache key for the restored prefix.
            manager_cache_key = getattr(manager, "prompt_cache_key", None)
            if manager_cache_key and hasattr(self.client, "prompt_cache_key"):
                self.client.prompt_cache_key = manager_cache_key
            try:
                messages = manager.begin_turn(
                    current, projected_tokens=0)
            except CompactionRequired:
                # The candidate user message is already counted by
                # begin_turn; force compaction of the existing prefix before
                # retrying the same message after a threshold crossing.
                manager.compact(self._compact_with_model, force=True)
                self._compaction_used = True
                if hasattr(self.client, "prompt_cache_key"):
                    self.client.prompt_cache_key = manager.prompt_cache_key
                if self._turn_deadline is not None and (
                        self._turn_deadline - self._clock()
                        <= REPAIR_TIMEOUT_MARGIN_S):
                    # Do not start the new request after compaction if its
                    # deadline has expired; the runner can safely apply its
                    # degraded fallback without a phantom assistant turn.
                    self._turn_messages = []
                    return ModelResponse(
                        call_id=prompt.call_id, prompt_hash=_prompt_hash([current]),
                        error="turn budget exhausted after conversation compaction")
                messages = manager.begin_turn(
                    current, projected_tokens=0)
            self._conversation_messages = manager.request_messages()
            self._conversation_turn_messages = list(messages)
            self._conversation_staged = []
            self._conversation_attempt_message = None
        else:
            messages = [
                {"role": "system",
                 "content": prompt_templates.system_prompt(
                     self.prompt_profile,
                     bundle_id=self.prompt_bundle_id,
                     snapshot=self.prompt_snapshot)}, current]
        self._hybrid_active = bool(prompt.history.strip())
        self._turn_messages = messages
        self._failed_content = ""
        self._failed_malformed = None
        self._lookup_exchange = []
        first = self._attempt(messages, prompt, attempt=1, kind="initial",
                              tools=([FREECIV_LOOKUP_TOOL]
                                     if self._enable_ruleset_lookup else None),
                              response_format=(None if self._enable_ruleset_lookup
                                               else _USE_CLIENT_DEFAULT))
        if self._enable_ruleset_lookup and self._last_tool_calls:
            return self._finish_lookup_turn(prompt, messages, first)
        return first

    def _lookup_request(self, call: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and bound one provider-generated lookup request."""
        function = call.get("function") if isinstance(call, Mapping) else None
        args = function.get("arguments") if isinstance(function, Mapping) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                args = None
        if not isinstance(args, Mapping):
            return {"ok": False, "error": "invalid lookup arguments"}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "query must be a non-empty string"}
        query = query.strip()[:LOOKUP_MAX_QUERY_CHARS]
        categories = args.get("categories", [])
        if categories is None:
            categories = []
        if not isinstance(categories, list):
            return {"ok": False, "error": "categories must be an array of known category aliases"}
        if any(not isinstance(x, str) or x.casefold() not in LOOKUP_CATEGORY_ALIASES
               for x in categories[:8]):
            return {"ok": False, "error": (
                "categories contain an unknown alias; use tech, unit, "
                "building, terrain, government, or extra")}
        categories = [x.casefold() for x in categories[:8]]
        limit = args.get("limit", LOOKUP_MAX_RESULTS)
        try:
            limit = max(1, min(LOOKUP_MAX_RESULTS, int(limit)))
        except (TypeError, ValueError):
            limit = LOOKUP_MAX_RESULTS
        return {"ok": True, "query": query, "categories": categories,
                "limit": limit}

    def _call_ruleset_lookup(self, request: Mapping[str, Any]) -> Any:
        """Invoke the injected read-only lookup without coupling to its adapter."""
        assert self._ruleset_lookup is not None
        # The public contract is one mapping.  For small test doubles and
        # existing adapters that expose keyword arguments, support that shape
        # too, without swallowing exceptions raised by the actual lookup.
        fn = self._ruleset_lookup
        try:
            sig = inspect.signature(fn)
            names = set(sig.parameters)
        except (TypeError, ValueError):
            names = set()
        if {"query", "categories", "limit"}.issubset(names):
            return fn(query=request["query"], categories=request["categories"],
                      limit=request["limit"])
        return fn(request)

    def _finish_lookup_turn(self, prompt: TurnPrompt,
                            base_messages: list[dict[str, Any]],
                            first: ModelResponse) -> ModelResponse:
        """Execute model-requested read-only lookups, then request strict JSON."""
        assistant = self._conversation_attempt_message
        if assistant is None:
            # In legacy mode the same exact provider message is still needed
            # for the next cacheable prefix.
            assistant = self._assistant_message(getattr(self, "_last_outcome", None))
        if assistant is None:
            return ModelResponse(call_id=prompt.call_id,
                                 prompt_hash=_prompt_hash(base_messages),
                                 error="lookup requested without assistant message")
        if len(self._last_tool_calls) > LOOKUP_MAX_TOOL_MESSAGES:
            return ModelResponse(
                call_id=prompt.call_id, prompt_hash=_prompt_hash(base_messages),
                error=(f"lookup request exceeded safe tool-call limit "
                       f"({LOOKUP_MAX_TOOL_MESSAGES})"))
        if any(not isinstance(call, Mapping) or
               not isinstance(call.get("id"), str) or not call.get("id")
               for call in self._last_tool_calls):
            return ModelResponse(
                call_id=prompt.call_id, prompt_hash=_prompt_hash(base_messages),
                error="lookup request contained a tool call without a valid id")
        tool_messages: list[dict[str, Any]] = [assistant]
        # Return a role=tool result for every bounded requested call ID.  We
        # execute at most three real lookups; excess calls receive an explicit
        # cap error so the provider never sees orphaned tool calls.
        for index, call in enumerate(self._last_tool_calls[:LOOKUP_MAX_TOOL_MESSAGES]):
            call_id = str(call.get("id") or f"lookup-{index + 1}")
            if index >= LOOKUP_MAX_TOPICS:
                result = {"ok": False, "error":
                          f"lookup limit reached ({LOOKUP_MAX_TOPICS})"}
                req = {"ok": False}
            else:
                req = self._lookup_request(call)
            if index >= LOOKUP_MAX_TOPICS:
                pass
            elif not req.get("ok"):
                result: Any = req
            elif self._turn_deadline is not None and self._clock() >= self._turn_deadline:
                result = {"ok": False, "error": "turn deadline exhausted before lookup"}
            else:
                try:
                    result = self._call_ruleset_lookup(req)
                except Exception as exc:
                    result = {"ok": False, "error": f"lookup failed: {exc}"}
            try:
                encoded = json.dumps(result, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"))
            except (TypeError, ValueError):
                encoded = json.dumps({"ok": False, "error": "lookup returned non-JSON data"})
            if len(encoded) > LOOKUP_MAX_OUTPUT_CHARS:
                encoded = encoded[:LOOKUP_MAX_OUTPUT_CHARS] + "\n[lookup output truncated]"
            tool_messages.append({"role": "tool", "tool_call_id": call_id,
                                  "name": "lookup_freeciv", "content": encoded})
        # A tool-capable model may otherwise continue emitting a function
        # shaped object when structured-output enforcement is permissive
        # (notably validated_parse).  Keep the transition instruction inside
        # the final role=tool content: this preserves the exact assistant ->
        # tool -> assistant protocol, and makes the request prefix byte-for-
        # byte identical to the committed conversation for cache reuse.
        transition = (
            "\n\nLOOKUP COMPLETE. Tools are unavailable now. Return ONLY a "
            "valid ActionBatch JSON object for turn_id "
            f"{prompt.turn_id!r}, using only the candidate IDs in the "
            "current turn prompt. Do not return lookup arguments or prose.")
        if tool_messages and tool_messages[-1].get("role") == "tool":
            tool_messages[-1]["content"] += transition
        self._lookup_exchange = tool_messages
        final_messages = list(base_messages) + tool_messages
        self._turn_messages = final_messages
        return self._attempt(final_messages, prompt, attempt=2, kind="tool_final",
                             response_format=_USE_CLIENT_DEFAULT)

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
            profile=self.prompt_profile,
            bundle_id=self.prompt_bundle_id,
            prompt_snapshot=self.prompt_snapshot,
        )
        failed_content = self._failed_content or (
            "(no model output — the previous attempt failed at transport "
            "level)")
        if self._hybrid_active:
            failed_content = _repair_excerpt(failed_content)
        failed_assistant = self._conversation_attempt_message or {
            "role": "assistant", "content": failed_content}
        messages = list(self._turn_messages) + [
            failed_assistant,
            {"role": "user", "content": repair_text},
        ]
        # FW-13: the repair is inside the budget check. Enforce the
        # repair-conversation budget (W1-7) over the COMPLETE repair
        # conversation — prior messages, the failed assistant output, the
        # repair instruction, and the re-attached candidate table — so an
        # oversized repair surfaces as the dedicated bounded fail-stop
        # (prompt_budget_exceeded), never a silent oversize or a generic
        # infrastructure truncation. The unresponsive re-ask above sends
        # the exact turn prompt the initial check already bounded; this
        # check guards the branch that adds content.
        repair_budget = (
            COMPACTION_CONTEXT_WINDOW_TOKENS - COMPACTION_OUTPUT_HEADROOM_TOKENS
            if self.conversation_enabled else
            prompt_templates.repair_conversation_budget(
                self.prompt_profile, hybrid=self._hybrid_active,
                context_policy=self.context_policy))
        calibration_exact = (
            self._conversation_manager is not None and
            self._conversation_manager.observed_request_hash ==
            _prompt_hash(self._turn_messages))
        if (self.conversation_enabled and self._conversation_manager is not None
                and self._conversation_manager.observed_prompt_tokens > 0
                and calibration_exact):
            # The provider's prompt count is the calibrated base.  Estimate
            # only the newly appended failed-assistant/repair suffix, avoiding
            # a second character estimate over the huge replayed prefix.
            base = self._conversation_manager.observed_prompt_tokens
            suffix = messages[len(self._turn_messages):]
            suffix_tokens = sum(estimate_tokens(
                m["content"] if isinstance(m.get("content"), str)
                else strict_dumps(m.get("content"), sort_keys=True))
                for m in suffix)
            completion_tokens = self._conversation_manager.observed_completion_tokens
            # The failed assistant is the provider completion already counted
            # by the calibrated request.  Reuse that exact measurement when
            # available; otherwise retain the conservative raw suffix estimate.
            if completion_tokens is not None and suffix:
                failed_estimate = estimate_tokens(
                    suffix[0]["content"] if isinstance(suffix[0].get("content"), str)
                    else strict_dumps(suffix[0].get("content"), sort_keys=True))
                suffix_tokens = max(0, suffix_tokens - failed_estimate)
                effective = (base + completion_tokens + suffix_tokens
                             + COMPACTION_INPUT_GUARD_TOKENS)
                method = "provider_prompt_completion"
            else:
                effective = base + suffix_tokens + COMPACTION_INPUT_GUARD_TOKENS
                method = "raw_suffix"
            if effective + COMPACTION_OUTPUT_HEADROOM_TOKENS > COMPACTION_CONTEXT_WINDOW_TOKENS:
                raise PromptBudgetExceeded(
                    f"complete repair conversation exceeds the calibrated "
                    f"repair budget: {effective} > "
                    f"{COMPACTION_CONTEXT_WINDOW_TOKENS - COMPACTION_OUTPUT_HEADROOM_TOKENS} "
                    f"(method={method})")
        else:
            _check_repair_conversation_budget(messages, repair_budget)
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

    def _attempt(self, messages: list[dict[str, Any]], prompt: TurnPrompt,
                 *, attempt: int, kind: str,
                 tools: Optional[list[dict[str, Any]]] = None,
                 response_format: Any = _USE_CLIENT_DEFAULT) -> ModelResponse:
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
        if self._turn_deadline is not None:
            remaining = self._turn_deadline - self._clock()
            if prompt.budget_s is not None:
                remaining = min(remaining, prompt.budget_s)
            if remaining <= REPAIR_TIMEOUT_MARGIN_S:
                return ModelResponse(
                    call_id=prompt.call_id, prompt_hash=prompt_hash,
                    error="turn budget exhausted before model request")
            if (is_repair or getattr(self.client, "stream", False) or
                    self._compaction_used or
                    (self._enable_ruleset_lookup and
                     kind in {"initial", "tool_final"})):
                timeout_ms = int((remaining - REPAIR_TIMEOUT_MARGIN_S) * 1000)
        kwargs: dict[str, Any] = {
            "messages": messages, "prompt_hash": prompt_hash,
            "attempt": attempt, "kind": kind, "timeout_ms": timeout_ms,
            "disable_transport_retries": is_repair,
            "call_event_context": {
                "logical_call_id": prompt.call_id,
                "turn": _turn_number(prompt.turn_id),
                "attempt": attempt,
                "physical_kind": kind,
            },
        }
        if tools is not None:
            kwargs.update({"tools": tools, "tool_choice": "auto",
                           "parallel_tool_calls": False,
                           # Explicitly clear the client's default
                           # ActionBatch schema for tool discovery.  The
                           # transport omits this field from the provider
                           # request when tools are present.
                           "response_format": None})
        elif response_format is not _USE_CLIENT_DEFAULT:
            kwargs["response_format"] = response_format
        try:
            outcome = self.client.complete_once(**kwargs)
        except ProviderCapacityError as exc:
            # complete_once deliberately re-raises capacity failures so the
            # runner can fail closed.  Preserve a turn-level durable record
            # here because the exception bypasses the normal ModelResponse
            # record builder; the low-level physical CallLog is already
            # present and is the source of usage/identity fields.
            partial = getattr(exc, "partial", None)
            call_id = getattr(partial, "call_id", None)
            lookup = getattr(self.client, "_call_log_for", None)
            call_log = lookup(call_id) if lookup is not None and call_id else None
            if call_log is None and getattr(self.client, "call_logs", None):
                call_log = self.client.call_logs[-1]
            if call_log is not None:
                record = {
                    "call_id": prompt.call_id,
                    "turn": _turn_number(prompt.turn_id),
                    "attempt": attempt, "kind": kind,
                    "prompt_hash": prompt_hash,
                    "model_requested": call_log.model_requested,
                    "model_responded": call_log.model_responded,
                    "enforcement": call_log.enforcement,
                    "temperature": call_log.temperature,
                    "latency_ms": call_log.latency_ms,
                    "tokens_in": call_log.tokens_in,
                    "tokens_out": call_log.tokens_out,
                    "tokens_total": call_log.tokens_total,
                    "cached_tokens": call_log.cached_tokens,
                    "cost_usd": call_log.cost_usd,
                    "finish_reason": call_log.finish_reason,
                    "response_id": call_log.response_id,
                    "fingerprint": call_log.fingerprint,
                    "transport_retries": call_log.transport_retries,
                    "error": str(exc), "error_kind": "provider_capacity",
                    "http_status": exc.http_status,
                    "provider_type": exc.provider_type,
                    "provider_code": exc.provider_code,
                    "retry_after": exc.retry_after,
                    "usage_status": getattr(call_log, "usage_status", None),
                    "cost_status": getattr(call_log, "cost_status", None),
                    "outcome": "provider_capacity",
                    "raw_attempts": _raw_attempts_from(
                        getattr(partial, "answer_text", None) or
                        getattr(partial, "content", "")) if partial else [],
                    "section_tokens": {
                        "memory": estimate_tokens(prompt.memory),
                        "snapshot": estimate_tokens(prompt.snapshot),
                        "delta": estimate_tokens(prompt.delta),
                        "candidates": estimate_tokens(prompt.candidates),
                        **({"history": estimate_tokens(prompt.history)}
                           if prompt.history else {}),
                    },
                    "ts": call_log.ts,
                }
                self.call_records.append(record)
                if self._api_log is not None:
                    self._api_log(record)
            raise
        self._last_outcome = outcome
        self._last_tool_calls = list(getattr(outcome.raw, "tool_calls", None) or [])
        self._emit_output(prompt=prompt, attempt=attempt, kind=kind,
                          outcome=outcome)
        # The client logs every logical attempt; correlate this call's log.
        lookup = getattr(self.client, "_call_log_for", None)
        call_log = (lookup(outcome.call_id)
                    if lookup is not None and
                    getattr(outcome, "call_id", None) is not None else None)
        if call_log is None:
            # Compatibility with older ModelClient doubles that predate
            # physical call correlation.
            call_log = self.client.call_logs[-1]
        if self.conversation_enabled and self._conversation_manager is not None:
            estimated_request_tokens = sum(
                estimate_tokens(strict_dumps(message)) for message in messages)
            self._conversation_manager.record_usage(
                call_log.tokens_in, estimated_request_tokens,
                message_count=len(messages),
                completion_tokens=call_log.tokens_out,
                request_hash=prompt_hash)
        if outcome.raw_content:
            self._failed_content = outcome.raw_content
        if outcome.malformed is not None:
            self._failed_malformed = outcome.malformed

        if self._last_tool_calls and outcome.error is None:
            # Tool requests are a valid intermediate model response, not a
            # schema failure.  The caller executes the bounded local tool and
            # performs the strict final ActionBatch request.
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd,
                usage_status=getattr(call_log, "usage_status", None),
                cost_status=getattr(call_log, "cost_status", None),
                fingerprint=call_log.fingerprint)
            outcome_kind = "tool_call"
            raw_attempts = []
        elif outcome.error is not None:
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                error=outcome.error, latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd,
                usage_status=getattr(call_log, "usage_status", None),
                cost_status=getattr(call_log, "cost_status", None),
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
                cost_usd=call_log.cost_usd,
                usage_status=getattr(call_log, "usage_status", None),
                cost_status=getattr(call_log, "cost_status", None),
                fingerprint=call_log.fingerprint)
            outcome_kind = "schema_error"
        else:
            raw_attempts = []
            response = ModelResponse(
                call_id=prompt.call_id, prompt_hash=prompt_hash,
                batch=outcome.batch, latency_ms=call_log.latency_ms,
                tokens_in=call_log.tokens_in, tokens_out=call_log.tokens_out,
                cost_usd=call_log.cost_usd,
                usage_status=getattr(call_log, "usage_status", None),
                cost_status=getattr(call_log, "cost_status", None),
                fingerprint=call_log.fingerprint)
            outcome_kind = "ok"

        # Conversation mode is transactional.  A malformed initial response
        # is retained only as part of a successful repair exchange; transport
        # failures never create a phantom assistant message.
        if self.conversation_enabled:
            assistant = self._assistant_message(outcome)
            if kind == "initial":
                self._conversation_attempt_message = assistant
                self._conversation_staged = [assistant] if (
                    assistant is not None and outcome_kind == "ok") else []
            elif kind == "tool_final":
                self._conversation_attempt_message = assistant
                if outcome_kind == "ok" and assistant is not None:
                    self._conversation_staged = list(self._lookup_exchange) + [assistant]
                else:
                    self._conversation_staged = list(self._lookup_exchange)
            elif kind == "repair" and outcome_kind == "ok" and assistant is not None:
                repair_tail = messages[len(self._turn_messages):]
                # In a lookup turn, ``_turn_messages`` already contains the
                # assistant tool request and role=tool results.  They are
                # therefore not part of repair_tail, but must be retained in
                # the committed exchange and in the exact history prefix.
                # For ordinary turns lookup_exchange is empty and this is
                # equivalent to the legacy behavior.
                self._conversation_staged = (
                    list(self._lookup_exchange) + repair_tail + [assistant])

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
            "cached_tokens": call_log.cached_tokens,
            "cost_usd": call_log.cost_usd,
            "finish_reason": call_log.finish_reason,
            "response_id": call_log.response_id,
            "fingerprint": call_log.fingerprint,
            "transport_retries": call_log.transport_retries,
            "error": call_log.error,
            "error_kind": getattr(call_log, "error_kind", None),
            "http_status": getattr(call_log, "http_status", None),
            "provider_type": getattr(call_log, "provider_type", None),
            "provider_code": getattr(call_log, "provider_code", None),
            "retry_after": getattr(call_log, "retry_after", None),
            "usage_status": getattr(call_log, "usage_status", None),
            "cost_status": getattr(call_log, "cost_status", None),
            "outcome": outcome_kind,
            "raw_attempts": raw_attempts,
            # The rendered section sizes (estimate_tokens) for the §3 budget
            # comparison: memory ≤ 1,500, snapshot ≤ 8,000, delta ≤ 800.
            "section_tokens": {
                "memory": estimate_tokens(prompt.memory),
                "snapshot": estimate_tokens(prompt.snapshot),
                "delta": estimate_tokens(prompt.delta),
                "candidates": estimate_tokens(prompt.candidates),
                **({"history": estimate_tokens(prompt.history)}
                   if prompt.history else {}),
            },
            "ts": call_log.ts,
        }
        self.call_records.append(record)
        if self._api_log is not None:
            self._api_log(record)
        return response
