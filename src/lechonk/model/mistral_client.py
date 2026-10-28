# SPDX-License-Identifier: MIT
"""lechonk.model.mistral_client — Mistral Studio API client (MIT).

design-plan-freeciv-v1.1.md decision 4: the exact model ID ``mistral-large-4``
(real, v26.10; ``mistral-large-latest`` does NOT point at it — no silent
substitution, the pin is enforced at construction). The API key comes from
the ``MISTRAL_API_KEY`` process-environment variable; when absent at runtime,
``load_api_key_from_env_file`` loads it from the repository-root ``.env``
(parsing ONLY that variable's line — comments and every other line are
ignored) into the process environment. The key value is never logged,
printed, or serialized; when neither source provides the key, construction
fails with a clear ``MistralKeyError`` that names the variable, never a value.

One request budget per turn (normative): ≤2 total API attempts per turn
including at most one repair; SDK/transport-level retries are transparent
(logged, never counted against the cap). The attempts-counter contract is
``TurnAttempts``; ``MistralClient.action_batch`` enforces it end to end.

Per-call logging (decision 4): tokens, latency, cost, and model fingerprint
data (decision 8) — see ``CallLog``. Call logs never contain the API key or
prompt text (only the prompt hash).

Output enforcement (decision 4, WP2 spike / probe P10): BOTH paths are
implemented behind one interface —
  (a) ``json_schema``: Mistral structured outputs with the real
      ``core.schemas.ActionBatch`` JSON schema (strict mode);
  (b) ``validated_parse``: schema-validated parsing of the model text
      against ``core.schemas.ActionBatch``.
Both paths produce identical transcript semantics. The spike decides the
default and the decision is written to the run manifest — see
``spike_decision``. At implementation time the live probe P10 was BLOCKED
(no key available in the environment or the repo ``.env``), so the
provisional default is ``validated_parse`` (fail-safe: it does not depend on
an unverified platform capability); ``scripts/wp2_spike.py`` re-decides with
live evidence once a key is available.

Arm sampling (decision 4, 2026-10-24 amendment): EVERY call — turn calls
(including repairs) and capability probes alike — runs at the arm sampling:
temperature 1.0 with ``reasoning_effort="high"``; the client NEVER sends
``top_p`` (the previous greedy temperature-0 + ``top_p=1.0`` sampling is
forbidden).

Per-arm model fingerprint (decision 8, FW-21): ``probe_fingerprint`` sends a
FIXED probe prompt (v2, capacity-revealing) under the arm sampling and
collects K=3 samples — one request with ``n=3`` when the SDK path supports
it, else three sequential requests (the mode is recorded). The persisted
fingerprint is the sorted, deduplicated SET of the sample output hashes
(see ``encode_fingerprint_set``); it travels with every per-call log and
goes into the run manifest. The probe is fail-closed (DEF-4): fewer than
FINGERPRINT_PROBE_SAMPLES usable samples raises ``FingerprintProbeError``
(the caller aborts the evaluation with a recorded infrastructure failure).

This module never imports civrealm (plan §7.1). The official ``mistralai``
SDK is imported lazily by ``MistralSDKTransport`` only — offline tests drive
the client through a mock transport and never import the SDK.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

from pydantic import ValidationError

from lechonk.core.jsonutil import strict_dumps
from lechonk.core.schemas import ActionBatch
from lechonk.model import prompts as prompt_templates

# ---------------------------------------------------------------------------
# Pinned constants (decision 4)
# ---------------------------------------------------------------------------

MODEL_ID = "mistral-large-4"
API_KEY_ENV_VAR = "MISTRAL_API_KEY"

# One request budget per turn (normative): ≤2 total API attempts per turn
# including at most one repair.
MAX_ATTEMPTS_PER_TURN = 2
MAX_REPAIRS_PER_TURN = 1

# Per-logical-attempt wall-clock bound (the §3 ACTING deadline is 180 s; one
# attempt must fit well inside it, leaving room for the repair attempt).
DEFAULT_TIMEOUT_MS = 120_000
# Ordinary turn and diagnostic probe requests leave the completion budget to
# the provider/model.
DEFAULT_MAX_TOKENS: Optional[int] = None
# Internal sentinel: unlike None, this explicitly omits the provider field.
UNSET_MAX_TOKENS = object()
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REASONING_EFFORT = "high"

# The action_batch repair attempts are BUDGETED (core/model_client.py
# repair contract): each repair passes disable_transport_retries=True and
# this explicit timeout_ms, so the repair's TOTAL duration stays within the
# bound. action_batch has no runner budget input (spike path), so the bound
# is this documented module constant, overridable per call.
ACTION_BATCH_REPAIR_TIMEOUT_MS = DEFAULT_TIMEOUT_MS

# Per-arm fingerprint probe (decision 8, FW-21): K samples collected under
# the arm sampling. The probe leaves output budgeting to the provider/model,
# just like ordinary turns.
FINGERPRINT_PROBE_SAMPLES = 3
FINGERPRINT_PROBE_MAX_TOKENS: Optional[int] = None
# The probe prompt is v2 (capacity-revealing). The v1 verbatim-echo prompt
# is retained on disk: any compliant model satisfies it identically, which
# degenerates the K-sample canary (deep-review finding).
FINGERPRINT_PROBE_PROMPT = "fingerprint_probe_v2"
# Sample-collection modes (recorded on the probe).
FINGERPRINT_MODE_SINGLE_REQUEST = "single_request_n3"
FINGERPRINT_MODE_SEQUENTIAL = "sequential_x3"
# The persisted fingerprint encoding (decision 8 amendment, FW-21): the
# sorted, deduplicated set of "sha256:<hex>" sample hashes joined by "|".
# A legacy single-sample fingerprint ("sha256:<hex>") is exactly the
# one-element encoding — old references remain loadable.
FINGERPRINT_HASH_SEPARATOR = "|"

# Studio list prices, USD per million tokens. PROVISIONAL for
# ``mistral-large-4``: carried from the published Mistral Large list price
# (web check 2026-10-24: $0.50/MTok input, $1.50/MTok output); verify against
# the console before evaluation runs. Configurable per client (``pricing``).
DEFAULT_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    MODEL_ID: {"input": 0.50, "output": 1.50},
}
# Mistral's prompt-cache input price is documented as 10% of standard input.
CACHED_INPUT_PRICE_FRACTION = 0.10


class EnforcementMode(str, Enum):
    """The two output-enforcement paths (decision 4)."""

    JSON_SCHEMA = "json_schema"
    VALIDATED_PARSE = "validated_parse"


# Provisional default enforcement while the live spike (P10) is blocked.
# validated_parse is the fail-safe default: it does not depend on an
# unverified platform capability. scripts/wp2_spike.py re-decides with live
# evidence and the decision is written to the run manifest.
PROVISIONAL_DEFAULT_ENFORCEMENT = EnforcementMode.VALIDATED_PARSE


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MistralClientError(RuntimeError):
    """Base for mistral_client errors."""


class MistralKeyError(MistralClientError):
    """Startup error: no API key was found — neither the ``MISTRAL_API_KEY``
    process-environment variable nor a ``MISTRAL_API_KEY`` line in the
    repository-root ``.env`` (via the narrow loader). The message names the
    missing variable only; it never contains any key value."""


class ModelIdError(MistralClientError):
    """The pinned model ID was substituted — forbidden (decision 4)."""


class AttemptBudgetExhausted(MistralClientError):
    """The per-turn API attempt budget (≤2 attempts, ≤1 repair) is spent with
    no valid ActionBatch. The runner falls back to the degraded defaults."""


class TransportError(MistralClientError):
    """A logical API attempt failed at transport level (after the SDK's own
    transparent retries). The attempt is logged and counts against the
    per-turn budget."""


class ProviderCapacityError(TransportError):
    """Exact Mistral backend-capacity exhaustion (HTTP 429/code 3505)."""

    def __init__(self, message: str, *, http_status: int = 429,
                 provider_type: str = "backend_out_of_capacity",
                 provider_code: str = "3505",
                 retry_after: Optional[str] = None):
        super().__init__(message)
        self.http_status = http_status
        self.provider_type = provider_type
        self.provider_code = provider_code
        self.retry_after = retry_after


def _capacity_error_from_sdk(exc: BaseException) -> Optional[ProviderCapacityError]:
    """Classify only the structured Mistral backend-capacity response."""
    response = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status is None:
        rendered = str(exc)
        if ('"raw_status_code":429' in rendered or "status=429" in rendered
                or "Status 429" in rendered):
            status = 429
    if status != 429:
        return None
    body: Any = None
    for obj in (response, exc):
        if isinstance(obj, Mapping):
            body = obj
            break
        for name in ("json", "body", "response_json", "data"):
            value = getattr(obj, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, Mapping):
                body = value
                break
        if isinstance(body, Mapping):
            break
    # Some mistralai releases expose only a rendered ``Body: {...}`` string
    # on SDKError.  Decode that body defensively; never classify from a mere
    # status or a message substring.
    if not isinstance(body, Mapping):
        rendered = str(exc)
        marker = rendered.find("Body:")
        if marker >= 0:
            candidate = rendered[marker + len("Body:"):].strip()
            try:
                body, _ = json.JSONDecoder().raw_decode(candidate)
            except (TypeError, ValueError):
                body = None
    if not isinstance(body, Mapping):
        return None
    detail = body.get("error") if isinstance(body.get("error"), Mapping) else body
    provider_type = detail.get("type") if isinstance(detail, Mapping) else None
    provider_code = detail.get("code") if isinstance(detail, Mapping) else None
    if provider_type != "backend_out_of_capacity" or str(provider_code) != "3505":
        return None
    retry_after = None
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    return ProviderCapacityError(
        f"Mistral provider capacity exhausted (status=429, type={provider_type}, code={provider_code})",
        retry_after=str(retry_after) if retry_after is not None else None,
    )


class FingerprintProbeError(MistralClientError):
    """The fingerprint probe could not obtain a valid sample set (DEF-4,
    fail-closed): fewer than FINGERPRINT_PROBE_SAMPLES usable samples
    arrived, or a sample was not usable content. The caller must abort the
    evaluation with a recorded infrastructure failure — never a
    metric-bearing comparison against an invalid fingerprint."""


class SchemaError(MistralClientError):
    """Output enforcement failed (illegal-action taxonomy class
    ``schema_error``): the model output is not a valid ActionBatch."""

    def __init__(self, detail: str, *, raw_excerpt: Optional[str] = None):
        super().__init__(f"model output is not a valid ActionBatch: {detail}")
        self.detail = detail
        self.raw_excerpt = (raw_excerpt or "")[:200]


# ---------------------------------------------------------------------------
# API key acquisition (decision 4): env var first, then the repo-root .env
# ---------------------------------------------------------------------------

DOTENV_FILENAME = ".env"


def parse_api_key_line(text: str) -> Optional[str]:
    """Parse ONLY the ``MISTRAL_API_KEY`` line out of .env-style text.

    Supports optional surrounding single/double quotes and whitespace.
    Comments and every other variable line are ignored. Returns ``None``
    when no ``MISTRAL_API_KEY`` line is present (caller decides).
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() != API_KEY_ENV_VAR:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value
    return None


def default_dotenv_path() -> Optional[Path]:
    """Locate the repository-root ``.env`` (existence only — never its
    contents): the ``src``-layout repository root of this source tree when
    identifiable, else the first ``.env`` found walking up from the working
    directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "lechonk").is_dir():
            return parent / DOTENV_FILENAME
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / DOTENV_FILENAME).is_file():
            return parent / DOTENV_FILENAME
    return None


def load_api_key_from_env_file(env_file: Optional[Path] = None) -> None:
    """Load ``MISTRAL_API_KEY`` into the process environment (idempotent).

    Precedence: the variable already present in the process environment
    wins. Otherwise the repository-root ``.env`` (``env_file``, or
    ``default_dotenv_path()``) is parsed for its ``MISTRAL_API_KEY`` line
    ONLY and the value is set into ``os.environ``. Raises ``MistralKeyError``
    when neither source provides the key. The value is never logged,
    printed, or serialized anywhere on this path.
    """
    if os.environ.get(API_KEY_ENV_VAR):
        return
    path = Path(env_file) if env_file is not None else default_dotenv_path()
    if path is not None and path.is_file():
        value = parse_api_key_line(path.read_text(encoding="utf-8"))
        if value:
            os.environ[API_KEY_ENV_VAR] = value
            return
    raise MistralKeyError(
        f"{API_KEY_ENV_VAR} is not set in the process environment and no "
        f"{DOTENV_FILENAME} file with {API_KEY_ENV_VAR} was found. Set the "
        f"variable or the {DOTENV_FILENAME} entry to start a run.")


# ---------------------------------------------------------------------------
# Transport abstraction (offline-testable; the SDK lives behind it)
# ---------------------------------------------------------------------------

Message = dict[str, Any]  # content may be text or reasoning chunk objects

_KEEP = object()  # sentinel: use the client's configured response_format


@dataclass
class RawCompletion:
    """One logical API attempt's raw result, normalized by the transport."""

    content: str
    model: str
    response_id: str
    created: int
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    transport_retries: int = 0
    """Transparent retries performed INSIDE this logical attempt (SDK-level
    retries are not individually observable; the SDK debug logger records
    them). Never counted against the per-turn cap (decision 4)."""
    sample_contents: Optional[list[str]] = None
    """The per-choice output texts when the request asked for n>1 samples
    (ONE multi-sample request, decision 8 fingerprint probe); None for a
    single-sample request (``content`` carries the one output)."""
    reasoning_text: str = ""
    """Complete visible reasoning/thinking text, when returned by the API."""
    answer_text: Optional[str] = None
    """Final answer text; defaults to ``content`` for older transports."""
    content_chunks: Optional[list[Any]] = None
    """Lossless JSON-friendly representation of the returned content chunks."""
    cached_tokens: Optional[int] = None
    """Prompt tokens served from the provider cache, when reported."""
    tool_calls: Optional[list[dict[str, Any]]] = None
    """Lossless normalized assistant tool calls, if the model requested any."""
    assistant_message: Optional[Message] = None
    """SDK-valid assistant message suitable for appending to the next request."""
    call_id: Optional[str] = None


StreamCallback = Callable[[dict[str, Any]], None]
CallEventCallback = Callable[[dict[str, Any]], None]


class ChatTransport(Protocol):
    """Transport contract: one logical chat completion. Implementations must
    perform their own transparent retry/backoff by default (the SDK transport
    configures ``RetryConfig``) — retries inside one call never count against
    the per-turn attempt cap.

    ``timeout_ms``: optional per-call hard request timeout override
    (milliseconds). None = the transport's configured default. It bounds the
    WHOLE single request attempt — connect and read included — so a hung
    request surfaces as a timeout failure within ``timeout_ms``. The
    budget-aware repair gating (WP4) uses it to bound the repair attempt by
    the remaining ACTING budget.

    ``disable_transport_retries``: when True the implementation MUST NOT
    retry at transport level — exactly one request attempt is made, so the
    call's observed duration is bounded by ``timeout_ms`` (plus small
    overhead). Budget-aware repair gating (WP4) sets it on REPAIR attempts:
    the harness-level attempt structure (attempt 1 + one repair) is the
    harness's retry policy, and a transport retry inside a budgeted repair
    would silently exceed the repair's budget. The initial attempt keeps the
    transport's transparent retries.

    ``reasoning_effort``: the reasoning-effort setting passed through to the
    API (decision 4 amendment: the arm runs ``"high"``). The transport NEVER
    sends ``top_p``.

    ``n``: optional multi-sample request — K samples in ONE request (the
    decision-8 fingerprint probe). The transport returns a ``RawCompletion``
    whose ``sample_contents`` carries the per-choice texts; None (the
    default) is an ordinary single-sample request."""

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float,
        reasoning_effort: str,
        max_tokens: Optional[int],
        response_format: Optional[dict[str, Any]],
        random_seed: Optional[int],
        n: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
        prompt_cache_key: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> RawCompletion: ...


def _usage_value(usage: Any, name: str, default: Any = 0) -> Any:
    """Read SDK or dict-shaped usage fields without assuming one SDK version."""
    value = getattr(usage, name, default)
    if isinstance(usage, dict):
        value = usage.get(name, default)
    return value


def _cached_prompt_tokens(usage: Any) -> Optional[int]:
    details = _usage_value(usage, "prompt_tokens_details", None)
    value = _usage_value(details, "cached_tokens", None) if details is not None else None
    return int(value) if isinstance(value, (int, float)) else None


def _chunk_value(chunk: Any, name: str, default: Any = None) -> Any:
    value = getattr(chunk, name, default)
    if value is default and isinstance(chunk, dict):
        value = chunk.get(name, default)
    return value


def _chunk_to_jsonable(value: Any) -> Any:
    """Serialize SDK/Pydantic chunks without dropping unknown fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_chunk_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _chunk_to_jsonable(item)
                for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _chunk_to_jsonable(model_dump(mode="json", by_alias=True))
        except TypeError:
            return _chunk_to_jsonable(model_dump())
    return {name: _chunk_to_jsonable(getattr(value, name))
            for name in ("type", "text", "thinking")
            if hasattr(value, name)} or str(value)


def _tool_calls_jsonable(value: Any) -> list[dict[str, Any]]:
    """Normalize SDK tool calls while retaining every provider field."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [item for item in (_chunk_to_jsonable(v) for v in items)
            if isinstance(item, dict)]


def _assistant_message(content: Any, tool_calls: Optional[list[dict[str, Any]]]) -> Message:
    """Construct the SDK-valid assistant message for tool-call replay."""
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _merge_tool_call_delta(aggregated: list[dict[str, Any]], delta: Any) -> None:
    """Merge one fragmented streaming tool-call delta by its index."""
    for call in _tool_calls_jsonable(delta):
        index = call.get("index")
        if not isinstance(index, int):
            index = len(aggregated)
        while len(aggregated) <= index:
            aggregated.append({"index": len(aggregated)})
        target = aggregated[index]
        for key, value in call.items():
            if key in {"index", "function"}:
                continue
            if value is not None:
                target[key] = value
        function = call.get("function")
        if isinstance(function, dict):
            dest = target.setdefault("function", {})
            for key, value in function.items():
                if value is None:
                    continue
                if key == "arguments" and isinstance(value, str):
                    dest[key] = str(dest.get(key, "")) + value
                elif key == "name" and key in dest and isinstance(dest[key], str):
                    # Providers occasionally split names too; concatenate only
                    # when the fragment is not a repeated complete value.
                    if value != dest[key]:
                        dest[key] += value
                else:
                    dest[key] = value


def _message_content_parts(content: Any) -> tuple[str, str]:
    """Return ``(reasoning_text, answer_text)`` from SDK message content."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return "", content
    if not isinstance(content, list):
        return "", str(content)
    reasoning: list[str] = []
    answer: list[str] = []

    def visit(chunk: Any, *, thinking: bool = False) -> None:
        kind = _chunk_value(chunk, "type", "")
        nested = _chunk_value(chunk, "thinking", None)
        is_thinking = thinking or kind in {"thinking", "reasoning"}
        if nested is not None:
            for item in (nested if isinstance(nested, list) else [nested]):
                visit(item, thinking=True)
            return
        text = _chunk_value(chunk, "text", None)
        if isinstance(text, str):
            (reasoning if is_thinking else answer).append(text)

    for chunk in content:
        visit(chunk)
    return "".join(reasoning), "".join(answer)


def _message_content_text(content: Any) -> str:
    """Normalize message content to final answer text for parsing."""
    _reasoning, answer = _message_content_parts(content)
    return answer


def _replayable_content_chunks(content: Any) -> list[dict[str, Any]]:
    """Normalize streamed/SDK content into valid replayable typed chunks.

    Streaming emits empty transition fragments and sometimes plain strings;
    replaying those verbatim as an assistant message is rejected by the API.
    Preserve valid provider dictionaries (including reasoning signatures and
    reference chunks) field-for-field.  Only empty text fragments and typed
    chunks whose required nested content is malformed are removed.
    """
    items = content if isinstance(content, list) else [content]
    result: list[dict[str, Any]] = []

    def append_text(text: Any) -> None:
        if not isinstance(text, str) or not text:
            return
        if (result and result[-1].get("type") == "text"
                and set(result[-1]) == {"type", "text"}):
            result[-1]["text"] += text
        else:
            result.append({"type": "text", "text": text})

    for item in items:
        obj = _chunk_to_jsonable(item)
        if isinstance(obj, str):
            append_text(obj)
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "text":
            append_text(obj.get("text"))
            continue
        if not isinstance(kind, str) or not kind:
            continue
        if kind in {"thinking", "reasoning"}:
            nested = obj.get("thinking", [])
            nested_items = nested if isinstance(nested, list) else [nested]
            valid_nested = []
            for child in nested_items:
                child_obj = _chunk_to_jsonable(child)
                if not isinstance(child_obj, dict):
                    continue
                child_type = child_obj.get("type")
                if child_type == "text":
                    if isinstance(child_obj.get("text"), str) and child_obj["text"]:
                        valid_nested.append(child_obj)
                elif child_type == "reference":
                    ids = child_obj.get("reference_ids")
                    if isinstance(ids, list) and all(isinstance(i, (str, int))
                                                     for i in ids):
                        valid_nested.append(child_obj)
                elif child_type == "tool_reference":
                    if (isinstance(child_obj.get("tool"), str)
                            and isinstance(child_obj.get("title"), str)):
                        valid_nested.append(child_obj)
            if not valid_nested:
                continue
            copied = dict(obj)
            # The SDK wire schema names this block ``thinking``; some
            # providers expose the equivalent as ``reasoning``.
            copied["type"] = "thinking"
            copied["thinking"] = valid_nested
            result.append(copied)
            continue
        if kind == "reference":
            ids = obj.get("reference_ids")
            if isinstance(ids, list) and all(isinstance(i, (str, int)) for i in ids):
                result.append(obj)
        elif kind == "tool_reference":
            if (isinstance(obj.get("tool"), str)
                    and isinstance(obj.get("title"), str)):
                result.append(obj)
    return result


class MistralSDKTransport:
    """Live transport over the official ``mistralai`` SDK (imported lazily).

    SDK retry/backoff is configured at construction (``RetryConfig`` backoff
    strategy + connection-error retries) and is TRANSPARENT: retries inside
    one logical attempt never touch the per-turn attempts counter (decision
    4). A second SDK client with ``RetryConfig(strategy="none")`` backs
    ``complete(..., disable_transport_retries=True)`` — the budgeted repair
    attempts (WP4): exactly one request attempt, bounded by the per-call
    ``timeout_ms``. The SDK debug logger (``lechonk.model.mistral_sdk``)
    records retry events; it never receives the API key.
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        sdk_logger_name: str = "lechonk.model.mistral_sdk",
    ):
        import logging

        from mistralai.client import Mistral
        from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

        debug_logger = logging.getLogger(sdk_logger_name)
        self._api_key = api_key
        self._sdk = Mistral(
            api_key=api_key,
            timeout_ms=timeout_ms,
            retry_config=RetryConfig(
                strategy="backoff",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=8_000,
                    exponent=2.0,
                    max_elapsed_time=60_000,
                ),
                retry_connection_errors=True,
            ),
            debug_logger=debug_logger,
        )
        # No-retry client for budgeted repair attempts (WP4): strategy "none"
        # makes the SDK's retry wrapper execute exactly one request, so the
        # per-call timeout_ms bounds the attempt's TOTAL duration.
        self._sdk_no_retries = Mistral(
            api_key=api_key,
            timeout_ms=timeout_ms,
            retry_config=RetryConfig(
                strategy="none",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=8_000,
                    exponent=2.0,
                    max_elapsed_time=60_000,
                ),
                retry_connection_errors=False,
            ),
            debug_logger=debug_logger,
        )

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float,
        reasoning_effort: str,
        max_tokens: Optional[int],
        response_format: Optional[dict[str, Any]],
        random_seed: Optional[int],
        n: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
        prompt_cache_key: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> RawCompletion:
        from mistralai.client.errors import (
            NoResponseError,
            ResponseValidationError,
            SDKError,
        )

        sdk = self._sdk_no_retries if disable_transport_retries else self._sdk
        # Multi-sample request kwargs (decision 8 fingerprint probe): n is
        # passed ONLY when requested — offline-verified against mistralai
        # 3.4.0 (Chat.complete takes n); the response then carries one
        # choice per sample.
        multi_sample: dict[str, Any] = {} if n is None else {"n": n}
        try:
            # Arm sampling (decision 4 amendment): reasoning_effort is passed
            # through as an explicit Chat.complete parameter (offline-verified
            # against mistralai 3.4.0); top_p is NEVER sent (omitting it is
            # safe — the SDK leaves it unset).
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "random_seed": random_seed,
                "timeout_ms": timeout_ms,
                **multi_sample,
            }
            # Tool discovery must be a plain chat request; some API versions
            # reject response_format alongside tools, so omit the field (not
            # JSON null) only for this opt-in shape. Legacy calls retain it.
            if not (tools is not None and response_format is None):
                request_kwargs["response_format"] = response_format
            # None means provider/model-managed output budget.  Omit the
            # field entirely; sending JSON null is not equivalent for every
            # SDK/API version.
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            if prompt_cache_key is not None:
                request_kwargs["prompt_cache_key"] = prompt_cache_key
            if tools is not None:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = ("auto" if tool_choice is None
                                                   else tool_choice)
                request_kwargs["parallel_tool_calls"] = (False if parallel_tool_calls is None
                                                           else parallel_tool_calls)
            response = sdk.chat.complete(
                **request_kwargs,
            )
        except (NoResponseError, ResponseValidationError) as exc:
            raise TransportError(
                f"Mistral API transport failure: {exc}") from exc
        except SDKError as exc:
            status = getattr(getattr(exc, "raw_response", None),
                             "status_code", None)
            capacity = _capacity_error_from_sdk(exc)
            if capacity is not None:
                raise capacity from exc
            raise TransportError(
                f"Mistral API error (status={status}): {exc}") from exc
        choices = list(response.choices or [])
        choice = choices[0] if choices else None
        usage = response.usage
        reasoning_text, answer_text = _message_content_parts(
            choice.message.content if choice is not None else None)
        tool_calls = _tool_calls_jsonable(
            getattr(choice.message, "tool_calls", None) if choice is not None else None)
        assistant = (_assistant_message(choice.message.content, tool_calls)
                     if choice is not None else None)
        return RawCompletion(
            content=answer_text,
            model=response.model,
            response_id=response.id,
            created=response.created,
            finish_reason=choice.finish_reason if choice is not None else None,
            prompt_tokens=_usage_value(usage, "prompt_tokens", None) if usage is not None else None,
            completion_tokens=_usage_value(usage, "completion_tokens", None) if usage is not None else None,
            total_tokens=_usage_value(usage, "total_tokens", None) if usage is not None else None,
            sample_contents=(
                [_message_content_text(c.message.content) for c in choices]
                if n is not None else None),
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            content_chunks=(
                _replayable_content_chunks(choice.message.content)
                if choice is not None else None),
            cached_tokens=_cached_prompt_tokens(usage),
            tool_calls=tool_calls or None,
            assistant_message=assistant,
        )

    def stream_complete(
        self,
        *, messages: list[Message], model: str, temperature: float,
        reasoning_effort: str, max_tokens: Optional[int],
        response_format: Optional[dict[str, Any]], random_seed: Optional[int],
        timeout_ms: Optional[int] = None,
        absolute_timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
        on_delta: Optional[StreamCallback] = None,
        prompt_cache_key: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> RawCompletion:
        """Collect a streaming response and report each received delta.

        The SDK timeout is a connect/read inactivity timeout. The caller still
        owns the absolute turn deadline; this method intentionally does not
        claim to enforce a total generation deadline while blocked in the SDK.
        """
        from mistralai.client.errors import (
            NoResponseError, ResponseValidationError, SDKError,
        )
        sdk = self._sdk_no_retries if disable_transport_retries else self._sdk
        # EventStreamAsync is bound to the loop that consumes it. The
        # synchronous harness creates one short-lived loop per call, so use a
        # fresh SDK object for every streamed request.
        if hasattr(self, "_api_key"):
            from mistralai.client import Mistral
            from mistralai.client.utils.retries import BackoffStrategy, RetryConfig
            sdk = Mistral(
                api_key=self._api_key, timeout_ms=timeout_ms,
                retry_config=RetryConfig(
                    strategy="none" if disable_transport_retries else "backoff",
                    backoff=BackoffStrategy(initial_interval=500,
                                             max_interval=8_000,
                                             exponent=2.0,
                                             max_elapsed_time=60_000),
                    retry_connection_errors=not disable_transport_retries,
                ))
        import asyncio

        # Keep provider chunks losslessly for traces/diagnostics.  A separate
        # sanitized view is constructed only for assistant-message replay.
        raw_chunks: list[Any] = []
        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        response_id = ""
        response_model = model
        created = 0
        finish_reason: Optional[str] = None
        usage: Any = None
        tool_calls: list[dict[str, Any]] = []
        def partial_completion() -> RawCompletion:
            return RawCompletion(
                content="".join(answer_parts), model=response_model,
                response_id=response_id, created=created,
                finish_reason=finish_reason,
                prompt_tokens=_usage_value(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=_usage_value(usage, "completion_tokens", None) if usage else None,
                total_tokens=_usage_value(usage, "total_tokens", None) if usage else None,
                reasoning_text="".join(reasoning_parts),
                answer_text="".join(answer_parts), content_chunks=raw_chunks,
                cached_tokens=_cached_prompt_tokens(usage),
                tool_calls=tool_calls or None,
                assistant_message=_assistant_message(
                    _replayable_content_chunks(raw_chunks), tool_calls or None))
        def raise_partial(message: str) -> None:
            error = TransportError(message)
            error.partial = partial_completion()
            raise error
        async def collect() -> None:
            nonlocal response_id, response_model, created, usage, finish_reason
            events = None
            try:
                request_kwargs: dict[str, Any] = {
                    "model": model, "messages": messages,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "random_seed": random_seed,
                    "timeout_ms": timeout_ms,
                }
                if not (tools is not None and response_format is None):
                    request_kwargs["response_format"] = response_format
                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens
                if prompt_cache_key is not None:
                    request_kwargs["prompt_cache_key"] = prompt_cache_key
                if tools is not None:
                    request_kwargs["tools"] = tools
                    request_kwargs["tool_choice"] = ("auto" if tool_choice is None
                                                       else tool_choice)
                    request_kwargs["parallel_tool_calls"] = (False if parallel_tool_calls is None
                                                               else parallel_tool_calls)
                events = await sdk.chat.stream_async(**request_kwargs)
                iterator = events.__aiter__()
                while True:
                    try:
                        event = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=(timeout_ms / 1000 if timeout_ms else None),
                        )
                    except StopAsyncIteration:
                        break
                    data = _chunk_value(event, "data", event)
                    response_id = _chunk_value(data, "id", response_id) or response_id
                    response_model = _chunk_value(data, "model", response_model) or response_model
                    created = _chunk_value(data, "created", created) or created
                    usage = _chunk_value(data, "usage", usage) or usage
                    choices = _chunk_value(data, "choices", []) or []
                    choice = choices[0] if choices else None
                    if choice is None:
                        continue
                    finish_reason = _chunk_value(choice, "finish_reason", finish_reason) or finish_reason
                    delta = _chunk_value(choice, "delta", None)
                    delta_tools = _chunk_value(delta, "tool_calls", None)
                    if delta_tools is not None:
                        _merge_tool_call_delta(tool_calls, delta_tools)
                    content = _chunk_value(delta, "content", None)
                    if content is None:
                        if on_delta is not None and delta_tools is not None:
                            on_delta({"complete": False, "reasoning": "", "answer": "",
                                      "content": [], "tool_calls": _tool_calls_jsonable(delta_tools),
                                      "response_id": response_id,
                                      "finish_reason": finish_reason})
                        continue
                    json_content = (_chunk_to_jsonable(content)
                                    if not isinstance(content, list)
                                    else [_chunk_to_jsonable(item) for item in content])
                    if isinstance(content, list):
                        raw_chunks.extend(_chunk_to_jsonable(item) for item in content)
                    else:
                        raw_chunks.append(_chunk_to_jsonable(content))
                    reasoning_delta, answer_delta = _message_content_parts(content)
                    reasoning_parts.append(reasoning_delta)
                    answer_parts.append(answer_delta)
                    if on_delta is not None:
                        try:
                            on_delta({
                                "complete": False,
                                "reasoning": reasoning_delta,
                                "answer": answer_delta,
                                "content": json_content,
                                "tool_calls": (_tool_calls_jsonable(delta_tools)
                                               if delta_tools is not None else []),
                                "response_id": response_id,
                                "finish_reason": finish_reason,
                            })
                        except Exception:
                            pass
            finally:
                if events is not None:
                    close = getattr(events, "aclose", None)
                    if close is not None:
                        await close()
                async_client = getattr(
                    getattr(sdk, "sdk_configuration", None),
                    "async_client", None)
                close_client = getattr(async_client, "aclose", None)
                if close_client is not None:
                    await close_client()

        try:
            coro = collect()
            if absolute_timeout_ms is not None:
                asyncio.run(asyncio.wait_for(coro, absolute_timeout_ms / 1000))
            else:
                asyncio.run(coro)
        except (NoResponseError, ResponseValidationError) as exc:
            error = TransportError(f"Mistral API transport failure: {exc}")
            error.partial = partial_completion()
            raise error from exc
        except SDKError as exc:
            status = getattr(getattr(exc, "raw_response", None), "status_code", None)
            capacity = _capacity_error_from_sdk(exc)
            if capacity is not None:
                capacity.partial = partial_completion()
                raise capacity from exc
            error = TransportError(f"Mistral API error (status={status}): {exc}")
            error.partial = partial_completion()
            raise error from exc
        except asyncio.TimeoutError as exc:
            if on_delta is not None:
                try:
                    on_delta({"complete": False, "partial": True,
                              "error": "stream timeout"})
                except Exception:
                    pass
            raise_partial("Mistral streaming request timed out")
        except Exception as exc:
            if on_delta is not None:
                try:
                    on_delta({"complete": False, "partial": True,
                              "error": str(exc)})
                except Exception:
                    pass
            error = TransportError(f"Mistral streaming failure: {exc}")
            error.partial = partial_completion()
            raise error from exc
        if finish_reason is None:
            raise_partial("Mistral stream ended without finish reason")
        prompt_tokens = _usage_value(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = _usage_value(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = _usage_value(usage, "total_tokens", None) if usage is not None else None
        return RawCompletion(
            content="".join(answer_parts), model=response_model,
            response_id=response_id, created=created, finish_reason=finish_reason,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total_tokens, reasoning_text="".join(reasoning_parts),
            answer_text="".join(answer_parts), content_chunks=raw_chunks,
            cached_tokens=_cached_prompt_tokens(usage),
            tool_calls=tool_calls or None,
            assistant_message=_assistant_message(
                _replayable_content_chunks(raw_chunks), tool_calls or None),
        )


def _sdk_chat_complete_accepts_n() -> bool:
    """Whether the installed mistralai SDK's ``Chat.complete`` takes the
    multi-sample ``n`` parameter (offline-verified against mistralai 3.4.0,
    ``chat.py``). Re-checked lazily so an older SDK makes the fingerprint
    probe fall back to sequential sampling instead of failing."""
    try:
        from mistralai.client.chat import Chat
    except Exception:
        return False
    return "n" in inspect.signature(Chat.complete).parameters


# ---------------------------------------------------------------------------
# Attempts contract (decision 4)
# ---------------------------------------------------------------------------


@dataclass
class TurnAttempts:
    """Per-turn API-attempt accounting (decision 4): ≤2 total API attempts per
    turn including at most one repair. The runner creates one per turn; the
    client enforces the caps on every logical attempt. SDK/transport retries
    inside one logical attempt are transparent and never counted."""

    attempts: int = 0
    repairs: int = 0

    def begin_attempt(self, *, is_repair: bool) -> int:
        """Account one logical API attempt; returns its 1-based number.

        Raises AttemptBudgetExhausted when the per-turn cap (2 attempts, 1
        repair) is already spent.
        """
        if self.attempts >= MAX_ATTEMPTS_PER_TURN:
            raise AttemptBudgetExhausted(
                f"per-turn API attempt cap reached: {self.attempts}/"
                f"{MAX_ATTEMPTS_PER_TURN} attempts used, {self.repairs}/"
                f"{MAX_REPAIRS_PER_TURN} repairs used")
        if is_repair and self.repairs >= MAX_REPAIRS_PER_TURN:
            raise AttemptBudgetExhausted(
                f"per-turn repair cap reached: {self.repairs}/"
                f"{MAX_REPAIRS_PER_TURN} repairs used")
        self.attempts += 1
        if is_repair:
            self.repairs += 1
        return self.attempts


# ---------------------------------------------------------------------------
# Per-call logging (decision 4: tokens, latency, cost; decision 8: fingerprint)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# The persisted fingerprint encoding (decision 8, FW-21)
# ---------------------------------------------------------------------------


def encode_fingerprint_set(output_hashes: Iterable[str]) -> str:
    """The persisted fingerprint encoding: the sorted, deduplicated set of
    sample output hashes as ``"sha256:<hex>"`` elements joined by ``|``.

    A legacy single-sample fingerprint (``"sha256:<hex>"``, the pre-FW-21
    encoding) is exactly the one-element encoding — old references remain
    loadable (the documented fallback)."""
    unique = sorted({f"sha256:{h}" for h in output_hashes})
    return FINGERPRINT_HASH_SEPARATOR.join(unique)


def decode_fingerprint_set(fingerprint: Optional[str]) -> list[str]:
    """Decode a persisted fingerprint into its sorted, deduplicated hash
    set. A legacy scalar fingerprint decodes to a one-element list (the
    documented fallback); None/blank decodes to the empty list."""
    if not fingerprint:
        return []
    return sorted({part for part in fingerprint.split(FINGERPRINT_HASH_SEPARATOR)
                   if part})


@dataclass
class CallLog:
    """Per-call log record (decision 4 + decision 8). NEVER contains the API
    key or prompt text — only the prompt hash. ``to_dict`` is redaction-safe
    by construction."""

    call_id: str
    attempt: int  # 1-based logical attempt within the turn; 0 = probe call
    kind: str  # "initial" | "repair" | "probe"
    prompt_hash: str
    model_requested: str
    model_responded: Optional[str]
    enforcement: str
    temperature: float
    latency_ms: int
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    tokens_total: Optional[int]
    cost_usd: Optional[float]
    finish_reason: Optional[str]
    response_id: Optional[str]
    fingerprint: str  # arm model fingerprint (the K=3 sample hash SET
    # encoding, or "unprobed")
    transport_retries: int
    error: Optional[str]
    ts: str
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    cached_tokens: Optional[int] = None
    error_kind: Optional[str] = None
    http_status: Optional[int] = None
    provider_type: Optional[str] = None
    provider_code: Optional[str] = None
    retry_after: Optional[str] = None
    usage_status: str = "unavailable"
    cost_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "attempt": self.attempt,
            "kind": self.kind,
            "prompt_hash": self.prompt_hash,
            "model_requested": self.model_requested,
            "model_responded": self.model_responded,
            "enforcement": self.enforcement,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "cached_tokens": self.cached_tokens,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "cost_usd": self.cost_usd,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "fingerprint": self.fingerprint,
            "transport_retries": self.transport_retries,
            "error": self.error,
            "error_kind": self.error_kind,
            "http_status": self.http_status,
            "provider_type": self.provider_type,
            "provider_code": self.provider_code,
            "retry_after": self.retry_after,
            "usage_status": self.usage_status,
            "cost_status": self.cost_status,
            "ts": self.ts,
        }


# ---------------------------------------------------------------------------
# Turn-loop data types
# ---------------------------------------------------------------------------


@dataclass
class RepairRequest:
    """A repair request for the second (final) API attempt of a turn.

    ``error_class`` is one of the illegal-action taxonomy classes
    (``schema_error`` / ``state_fog_error`` / ``illegal_in_rule``). The runner
    classifies ref-validation failures through this type; the client renders
    the class-specific repair prompt (§7.5).
    """

    error_class: str
    detail: str
    failed_refs: list[str] = field(default_factory=list)


@dataclass
class ActionBatchResult:
    """A validated ActionBatch plus its turn accounting."""

    batch: ActionBatch
    attempts_used: int
    repairs_used: int
    call_logs: list[CallLog]
    prompt_hash: str


@dataclass
class AttemptOutcome:
    """One LOGICAL API attempt's outcome — the single-attempt seam used by
    the core-runner adapter (``lechonk.model.turn_client``).

    ``MistralClient.action_batch`` drives one whole turn (attempt 1 + the one
    repair, owning the ≤2-attempt cap); ``MistralClient.complete_once``
    performs exactly ONE logical attempt, leaving the cap and the repair flow
    to the caller (the WP2 runner owns both, core/runner.py)."""

    raw: Optional[RawCompletion]
    error: Optional[str]  # transport-level failure (timeout, API error)
    batch: Optional[ActionBatch]  # the schema-validated batch, or None
    malformed: Optional[str]  # the schema-error detail when the output
    # violated the ActionBatch schema (repair class ``schema_error``)
    raw_content: str  # the model's raw output text (feeds the repair prompt)
    call_id: Optional[str] = None  # exact physical CallLog identity


@dataclass
class FingerprintProbe:
    """Per-arm model fingerprint (decision 8, FW-21): the FIXED probe prompt
    v2 (capacity-revealing; versioned file, hash recorded) under the arm's
    own sampling (temperature 1.0, reasoning_effort high, no top_p),
    collecting K=3 samples. ``fingerprint`` is the PERSISTED encoding: the
    sorted, deduplicated set of the sample output hashes (see
    ``encode_fingerprint_set``). ``output_hashes`` records the same set as
    bare hex hashes; ``sample_attempts`` vs ``len(output_hashes)`` records
    attempts vs unique outputs (K=3 attempts may yield fewer than three
    unique outputs). ``sample_mode`` records how the samples were collected
    and ``max_tokens`` the output budget requested by the harness (``None``
    means provider-managed)."""

    fingerprint: str  # the persisted set encoding (encode_fingerprint_set)
    output_hashes: list[str]  # sorted, deduplicated bare hex sample hashes
    sample_attempts: int  # K: the number of samples collected
    sample_mode: str  # single_request_n3 | sequential_x3
    max_tokens: Optional[int]  # None: provider-managed output budget
    model_requested: str
    model_responded: str
    prompt_hash: str
    temperature: float
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    tokens_total: Optional[int]
    latency_ms: int
    cost_usd: Optional[float]
    response_id: str
    finish_reason: Optional[str]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "output_hashes": list(self.output_hashes),
            "sample_attempts": self.sample_attempts,
            "sample_mode": self.sample_mode,
            "max_tokens": self.max_tokens,
            "model_requested": self.model_requested,
            "model_responded": self.model_responded,
            "prompt_hash": self.prompt_hash,
            "temperature": self.temperature,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "response_id": self.response_id,
            "finish_reason": self.finish_reason,
            "ts": self.ts,
        }


@dataclass
class ModelIdProbe:
    """P10a evidence: the exact model ID responds (no silent substitution)."""

    model_requested: str
    model_responded: Optional[str]
    responded: bool
    response_id: Optional[str]
    finish_reason: Optional[str]
    latency_ms: int
    error: Optional[str]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_requested": self.model_requested,
            "model_responded": self.model_responded,
            "responded": self.responded,
            "response_id": self.response_id,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "ts": self.ts,
        }


@dataclass
class JsonSchemaProbe:
    """P10b evidence: ``json_schema`` structured-output support on the exact
    model ID, tested with the real ActionBatch schema (candidate-ID
    constrained refs)."""

    model_requested: str
    model_responded: Optional[str]
    schema_hash: str
    response_format_type: str
    strict: bool
    responded: bool
    json_schema_supported: bool
    response_shape: dict[str, Any]
    latency_ms: int
    error: Optional[str]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_requested": self.model_requested,
            "model_responded": self.model_responded,
            "schema_hash": self.schema_hash,
            "response_format_type": self.response_format_type,
            "strict": self.strict,
            "responded": self.responded,
            "json_schema_supported": self.json_schema_supported,
            "response_shape": self.response_shape,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "ts": self.ts,
        }


# ---------------------------------------------------------------------------
# ActionBatch JSON schema (single source of truth: the pydantic model)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def action_batch_json_schema() -> dict[str, Any]:
    """The JSON schema for ``core.schemas.ActionBatch`` — derived from the
    pydantic model, never hand-maintained."""
    return ActionBatch.model_json_schema()


@lru_cache(maxsize=1)
def action_batch_schema_hash() -> str:
    """``sha256:<hex>`` over the canonical JSON of the ActionBatch schema —
    the schema hash recorded by the WP2 spike artifact (decision 8)."""
    canonical = strict_dumps(action_batch_json_schema(), sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Schema-validated parsing (enforcement path b)
# ---------------------------------------------------------------------------


def _json_object_candidates(text: str):
    """Yield candidate JSON-object texts: the raw text, fenced blocks, and
    the brace-balanced span (model outputs wrap JSON in prose/fences)."""
    yield text
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        yield match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    for candidate in _json_object_candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _summarize_validation_error(exc: ValidationError) -> str:
    """Bounded, deterministic summary of a pydantic ValidationError."""
    parts = []
    for error in exc.errors()[:3]:
        loc = ".".join(str(p) for p in error.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {error.get('msg', '')}")
    detail = "; ".join(parts) if parts else str(exc)
    return detail[:300]


def parse_action_batch(content: str) -> ActionBatch:
    """Enforce the ActionBatch schema on raw model output (enforcement path
    b — schema-validated parsing; path a relies on the platform but the same
    validation is the final gate either way). Raises SchemaError (taxonomy
    class ``schema_error``)."""
    text = (content or "").strip()
    if not text:
        raise SchemaError("empty model output")
    value = _extract_json_object(text)
    if value is None:
        raise SchemaError("no JSON object found in model output",
                          raw_excerpt=text)
    try:
        return ActionBatch.model_validate(value)
    except ValidationError as exc:
        raise SchemaError(_summarize_validation_error(exc),
                          raw_excerpt=text) from exc


# ---------------------------------------------------------------------------
# Spike decision record (decision 4 — written to the run manifest, decision 8)
# ---------------------------------------------------------------------------

SPIKE_EVIDENCE_ARTIFACT = "artifacts/runs/wp2-spike/spike_report.json"


def spike_decision(
    *,
    live_probe_done: bool = False,
    json_schema_supported: Optional[bool] = None,
    evidence_artifact: Optional[str] = None,
) -> dict[str, Any]:
    """The WP2 enforcement-spike decision record (decision 4), shaped for the
    run manifest (decision 8). With live P10 evidence the decision follows the
    probe; while the live probe is blocked the provisional default is
    ``validated_parse`` and the status says so."""
    if live_probe_done and json_schema_supported is not None:
        decision = (EnforcementMode.JSON_SCHEMA.value
                    if json_schema_supported
                    else EnforcementMode.VALIDATED_PARSE.value)
        status = "decided_live"
        rationale = (
            "P10 live probe on the exact model ID: json_schema structured "
            f"outputs {'supported' if json_schema_supported else 'NOT supported'}; "
            f"decision = {decision}.")
    else:
        decision = PROVISIONAL_DEFAULT_ENFORCEMENT.value
        status = "provisional_live_probe_blocked"
        rationale = (
            "P10 live probe blocked (MISTRAL_API_KEY not in the process "
            "environment at decision time); provisional default = "
            "validated_parse (fail-safe: no unverified platform dependency). "
            "Both paths implemented; re-decide with live evidence.")
    return {
        "probe": "P10",
        "model_id": MODEL_ID,
        "paths_implemented": [EnforcementMode.JSON_SCHEMA.value,
                              EnforcementMode.VALIDATED_PARSE.value],
        "decision": decision,
        "decision_status": status,
        "json_schema_supported_live": json_schema_supported,
        "evidence_artifact": evidence_artifact or SPIKE_EVIDENCE_ARTIFACT,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

RefValidator = Callable[[ActionBatch], Optional[RepairRequest]]
RepairCandidatesFor = Callable[[RepairRequest], str]


class MistralClient:
    """Studio API client for the exact model ID ``mistral-large-4``.

    Construction reads the API key from the process environment
    (``MISTRAL_API_KEY``); when absent there, the repository-root ``.env`` is
    consulted (narrow loader — only that one line is parsed); an absent key
    is a clear startup error. The key is never logged, printed, or
    serialized.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = MODEL_ID,
        enforcement: Optional[str | EnforcementMode] = None,
        transport: Optional[ChatTransport] = None,
        pricing: Optional[dict[str, dict[str, float]]] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        random_seed: Optional[int] = None,
        log_sink: Optional[Callable[[CallLog], None]] = None,
        stream: bool = False,
        stream_callback: Optional[StreamCallback] = None,
        call_event_callback: Optional[CallEventCallback] = None,
        prompt_cache_key: Optional[str] = None,
    ):
        if model != MODEL_ID:
            raise ModelIdError(
                f"model id {model!r} is not the pinned {MODEL_ID!r}; silent "
                "substitution is forbidden (decision 4)")
        if api_key is None:
            load_api_key_from_env_file()
            api_key = os.environ[API_KEY_ENV_VAR]
        if not api_key:
            raise MistralKeyError(
                f"{API_KEY_ENV_VAR} is empty. Export the variable (or set it "
                "in the repository-root .env) with a non-empty value.")
        self._api_key = api_key  # never logged, never serialized
        self.model = model
        self.enforcement = EnforcementMode(
            enforcement if enforcement is not None
            else PROVISIONAL_DEFAULT_ENFORCEMENT)
        self.pricing = pricing if pricing is not None else DEFAULT_PRICING_USD_PER_MTOK
        self.max_tokens = max_tokens
        self.timeout_ms = timeout_ms
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.random_seed = random_seed
        self._transport: ChatTransport = (
            transport if transport is not None
            else MistralSDKTransport(api_key, timeout_ms=timeout_ms))
        self.call_logs: list[CallLog] = []
        self._log_sink = log_sink
        self.stream = stream
        self.stream_callback = stream_callback
        self.call_event_callback = call_event_callback
        self.prompt_cache_key = prompt_cache_key
        self.arm_fingerprint: Optional[str] = None
        self._call_counter = 0
        self._call_lock = threading.Lock()

    # -- logging ------------------------------------------------------------

    def _next_call_id(self) -> str:
        with self._call_lock:
            self._call_counter += 1
            return f"call-{self._call_counter:04d}"

    def _emit_log(self, record: CallLog) -> None:
        with self._call_lock:
            self.call_logs.append(record)
        if self._log_sink is not None:
            self._log_sink(record)

    def _call_log_for(self, call_id: Optional[str]) -> Optional[CallLog]:
        if call_id is None:
            return None
        with self._call_lock:
            for record in reversed(self.call_logs):
                if record.call_id == call_id:
                    return record
        return None

    def _cost_usd(
        self, tokens_in: Optional[int], tokens_out: Optional[int],
        cached_tokens: Optional[int] = None,
    ) -> Optional[float]:
        price = self.pricing.get(self.model)
        if (not price or tokens_in is None or tokens_out is None
                or cached_tokens is None):
            return None
        # Usage metadata is provider-owned; clamp malformed values rather
        # than allowing a negative or over-counted bill estimate.
        cached = max(0, min(tokens_in, cached_tokens or 0))
        uncached = tokens_in - cached
        input_cost = (uncached * price["input"]
                      + cached * price["input"] * CACHED_INPUT_PRICE_FRACTION)
        cost = (input_cost
                + tokens_out * price["output"]) / 1_000_000
        return round(cost, 6)

    def _make_call_log(
        self,
        *,
        call_id: str,
        attempt: int,
        kind: str,
        prompt_hash: str,
        latency_ms: int,
        raw: Optional[RawCompletion],
        error: Optional[str],
        reasoning_effort: str,
        error_exc: Optional[BaseException] = None,
    ) -> CallLog:
        return CallLog(
            call_id=call_id,
            attempt=attempt,
            kind=kind,
            prompt_hash=prompt_hash,
            model_requested=self.model,
            model_responded=raw.model if raw is not None else None,
            enforcement=self.enforcement.value,
            temperature=self.temperature,
            latency_ms=latency_ms,
            tokens_in=raw.prompt_tokens if raw is not None else None,
            tokens_out=raw.completion_tokens if raw is not None else None,
            tokens_total=raw.total_tokens if raw is not None else None,
            cost_usd=(self._cost_usd(
                          raw.prompt_tokens, raw.completion_tokens,
                          raw.cached_tokens)
                      if raw is not None else None),
            finish_reason=raw.finish_reason if raw is not None else None,
            response_id=raw.response_id if raw is not None else None,
            fingerprint=self.arm_fingerprint or "unprobed",
            transport_retries=raw.transport_retries if raw is not None else 0,
            error=error,
            ts=_utc_now_iso(),
            reasoning_effort=reasoning_effort,
            cached_tokens=raw.cached_tokens if raw is not None else None,
            error_kind=("provider_capacity" if isinstance(error_exc, ProviderCapacityError)
                        else None),
            http_status=(error_exc.http_status
                         if isinstance(error_exc, ProviderCapacityError) else None),
            provider_type=(error_exc.provider_type
                           if isinstance(error_exc, ProviderCapacityError) else None),
            provider_code=(error_exc.provider_code
                           if isinstance(error_exc, ProviderCapacityError) else None),
            retry_after=(error_exc.retry_after
                         if isinstance(error_exc, ProviderCapacityError) else None),
            usage_status=(
                "reported" if raw is not None and all(
                    value is not None for value in (
                        raw.prompt_tokens, raw.completion_tokens,
                        raw.total_tokens)) else (
                    "partial" if raw is not None and any(
                        value is not None for value in (
                            raw.prompt_tokens, raw.completion_tokens,
                            raw.total_tokens)) else "unavailable")),
            cost_status=("reported" if raw is not None and
                         raw.prompt_tokens is not None and
                         raw.completion_tokens is not None and
                         raw.cached_tokens is not None else "unknown"),
        )

    # -- one logical attempt -------------------------------------------------

    def _response_format_payload(self) -> Optional[dict[str, Any]]:
        """The Mistral ``response_format`` payload for the configured
        enforcement path (None = free text, used by the probes)."""
        if self.enforcement is EnforcementMode.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_batch",
                    "description": (
                        "One turn of play as a batch of candidate-ID actions "
                        "(lechonk ActionBatch)."),
                    "schema_definition": action_batch_json_schema(),
                    "strict": True,
                },
            }
        if self.enforcement is EnforcementMode.VALIDATED_PARSE:
            return {"type": "json_object"}
        return None

    def _complete(
        self,
        *,
        messages: list[Message],
        attempt: int,
        kind: str,
        prompt_hash: str,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] | object = None,
        response_format: Any = _KEEP,
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
        n: Optional[int] = None,
        prompt_cache_key: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        call_event_context: Optional[Mapping[str, Any]] = None,
    ) -> RawCompletion:
        """One LOGICAL API attempt: calls the transport (SDK/transport
        retries inside it are transparent and never counted), logs the call.
        Raises TransportError on hard transport failure (the attempt is still
        logged). ``timeout_ms`` overrides the per-call request timeout
        (budget-aware repair gating, WP4); None = the configured default.
        ``disable_transport_retries`` forces exactly one request attempt in
        the transport — used on budgeted repair attempts so their TOTAL
        duration stays within ``timeout_ms``; None/False keeps the transport's
        transparent retries (attempt 1 behavior, unchanged).
        ``reasoning_effort``: None = the configured arm effort; the transport
        NEVER sends ``top_p``. ``n``: None = an ordinary single-sample
        request; an integer asks for that many samples in ONE request (the
        decision-8 fingerprint probe; the transport must support it — see
        ``_fingerprint_sample_mode``)."""
        t0 = time.perf_counter()
        # Allocate the physical request identity before any provider work.
        # The same immutable ID is used by lifecycle events and CallLog.
        call_id = self._next_call_id()
        event_context = dict(call_event_context or {})
        event_context.update({"call_id": call_id})
        self._emit_call_event({"event": "model_request_started", **event_context})

        def emit_delta(delta: dict[str, Any]) -> None:
            self._emit_call_event({
                "event": "model_request_delta", **event_context,
                "delta": delta,
            })
            # Preserve the historical raw stream_callback contract exactly.
            if self.stream_callback is not None:
                try:
                    self.stream_callback(delta)
                except Exception:
                    pass

        def emit_finished(*, error: Optional[str] = None,
                          raw: Optional[RawCompletion] = None) -> None:
            self._emit_call_event({
                "event": "model_request_finished", **event_context,
                "error": error,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "response_id": raw.response_id if raw is not None else None,
                "tokens_in": raw.prompt_tokens if raw is not None else None,
                "tokens_out": raw.completion_tokens if raw is not None else None,
            })

        def record_log(record: CallLog, *, error: Optional[str] = None,
                       raw: Optional[RawCompletion] = None) -> None:
            # A durable sink may fail, but the lifecycle event must still
            # close the physical request so observers cannot strand a pulse.
            try:
                self._emit_log(record)
            finally:
                emit_finished(error=error, raw=raw)
        effective_temperature = (self.temperature if temperature is None
                                 else temperature)
        effective_reasoning_effort = (self.reasoning_effort
                                      if reasoning_effort is None
                                      else reasoning_effort)
        effective_max_tokens = (None if (max_tokens is UNSET_MAX_TOKENS or
                                        kind.startswith("conversation_compaction")) else
                                self.max_tokens if max_tokens is None else max_tokens)
        effective_response_format = (self._response_format_payload()
                                     if response_format is _KEEP
                                     else response_format)
        # n is passed ONLY when a multi-sample request is made: every
        # ordinary call keeps the exact pre-FW-21 transport call shape.
        multi_sample: dict[str, Any] = {} if n is None else {"n": n}
        effective_cache_key = (self.prompt_cache_key
                               if prompt_cache_key is None else prompt_cache_key)
        cache_kwargs = ({} if effective_cache_key is None else
                        {"prompt_cache_key": effective_cache_key})
        tool_kwargs = ({} if tools is None else {
            "tools": tools, "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        })
        try:
            if self.stream and n is None and hasattr(self._transport, "stream_complete"):
                raw = self._transport.stream_complete(
                    messages=messages, model=self.model,
                    temperature=effective_temperature,
                    reasoning_effort=effective_reasoning_effort,
                    max_tokens=effective_max_tokens,
                    response_format=effective_response_format,
                    random_seed=self.random_seed, timeout_ms=self.timeout_ms,
                    absolute_timeout_ms=timeout_ms,
                    disable_transport_retries=disable_transport_retries,
                    on_delta=emit_delta,
                    **tool_kwargs,
                    **cache_kwargs,
                )
            else:
                raw = self._transport.complete(
                    messages=messages, model=self.model,
                    temperature=effective_temperature,
                    reasoning_effort=effective_reasoning_effort,
                    max_tokens=effective_max_tokens,
                    response_format=effective_response_format,
                    random_seed=self.random_seed, timeout_ms=timeout_ms,
                    disable_transport_retries=disable_transport_retries,
                    **multi_sample,
                    **tool_kwargs,
                    **cache_kwargs,
                )
        except TransportError as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            # Streaming may have produced a partial completion before the
            # connection failed.  Preserve its provider usage in accounting;
            # the error remains attached to the call record.
            partial = getattr(exc, "partial", None)
            if partial is not None:
                partial.call_id = call_id
            record_log(self._make_call_log(
                call_id=call_id,
                attempt=attempt, kind=kind, prompt_hash=prompt_hash,
                latency_ms=latency_ms, raw=partial, error=str(exc),
                reasoning_effort=effective_reasoning_effort, error_exc=exc),
                error=str(exc), raw=partial)
            raise
        except Exception as exc:  # unexpected transport bug — same accounting
            latency_ms = int((time.perf_counter() - t0) * 1000)
            wrapped = TransportError(f"transport failure: {exc}")
            partial = getattr(wrapped, "partial", None)
            if partial is not None:
                partial.call_id = call_id
            record_log(self._make_call_log(
                call_id=call_id,
                attempt=attempt, kind=kind, prompt_hash=prompt_hash,
                latency_ms=latency_ms, raw=partial, error=str(wrapped),
                reasoning_effort=effective_reasoning_effort, error_exc=wrapped),
                error=str(wrapped), raw=partial)
            raise wrapped from exc
        except BaseException:
            # Cancellation/interrupts must not strand a request heartbeat.
            emit_finished(error="cancelled")
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw.call_id = call_id
        record_log(self._make_call_log(
            call_id=call_id,
            attempt=attempt, kind=kind, prompt_hash=prompt_hash,
            latency_ms=latency_ms, raw=raw, error=None,
            reasoning_effort=effective_reasoning_effort), raw=raw)
        return raw

    def _emit_call_event(self, event: dict[str, Any]) -> None:
        """Best-effort local observability; never changes provider traffic."""
        callback = self.call_event_callback
        if callback is None:
            return
        try:
            callback(dict(event))
        except Exception:
            return

    # -- output enforcement ---------------------------------------------------

    def parse_action_batch(self, content: str) -> ActionBatch:
        """Enforce the ActionBatch schema on raw model output. Both
        enforcement paths produce identical transcript semantics (decision
        4): path (a) ``json_schema`` relies on the platform, path (b)
        ``validated_parse`` parses the text — either way this validation is
        the final gate."""
        return parse_action_batch(content)

    def _parse_and_validate(
        self,
        raw: RawCompletion,
        ref_validator: Optional[RefValidator],
    ) -> tuple[Optional[ActionBatch], Optional[RepairRequest]]:
        """Returns ``(batch, None)`` on success or ``(None, repair_request)``."""
        try:
            batch = parse_action_batch(raw.content)
        except SchemaError as exc:
            return None, RepairRequest(
                error_class="schema_error", detail=exc.detail)
        if ref_validator is not None:
            repair_request = ref_validator(batch)
            if repair_request is not None:
                return None, repair_request
        return batch, None

    # -- the per-turn model call (one request budget, decision 4) --------------

    def action_batch(
        self,
        *,
        messages: list[Message],
        prompt_hash: str,
        attempts: TurnAttempts,
        ref_validator: Optional[RefValidator] = None,
        repair_candidates_for: Optional[RepairCandidatesFor] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_ms: int = ACTION_BATCH_REPAIR_TIMEOUT_MS,
    ) -> ActionBatchResult:
        """One turn's model call under the one-request budget (decision 4):
        attempt 1, then at most one repair attempt (schema-error, or a
        ref-validation failure classified by the runner as ``state_fog_error``
        / ``illegal_in_rule``). ≤2 total API attempts per turn; SDK/transport
        retries inside an attempt are transparent and never counted.

        - ``messages``: the full message list for attempt 1 (system + user).
        - ``ref_validator``: optional ``ActionBatch -> Optional[RepairRequest]``
          run BEFORE any execution; the runner classifies unknown refs as
          ``state_fog_error`` and known-but-illegal refs as
          ``illegal_in_rule``.
        - ``repair_candidates_for``: optional ``RepairRequest -> str`` rendering
          the candidate text attached to the repair prompt (the runner renders
          the full table for schema errors, the refreshed candidates for the
          failed refs otherwise).
        - ``timeout_ms``: the explicit per-call timeout bounding each REPAIR
          attempt; both repair paths also pass
          ``disable_transport_retries=True`` (the core/model_client.py repair
          contract: a repair attempt's TOTAL duration stays within the
          timeout). Defaults to the documented module constant
          ``ACTION_BATCH_REPAIR_TIMEOUT_MS``, overridable per call. The
          initial attempt is unaffected (transparent transport retries).

        Raises AttemptBudgetExhausted when the budget is spent with no valid
        ActionBatch (the runner then applies the degraded defaults).
        """
        log_start = len(self.call_logs)

        def finish(batch: ActionBatch) -> ActionBatchResult:
            return ActionBatchResult(
                batch=batch,
                attempts_used=attempts.attempts,
                repairs_used=attempts.repairs,
                call_logs=self.call_logs[log_start:],
                prompt_hash=prompt_hash,
            )

        # ---- attempt 1 (initial)
        attempt_no = attempts.begin_attempt(is_repair=False)
        try:
            raw = self._complete(
                messages=messages, attempt=attempt_no, kind="initial",
                prompt_hash=prompt_hash, temperature=temperature,
                max_tokens=max_tokens)
        except ProviderCapacityError:
            raise
        except TransportError as exc:
            # No model output: the single repair slot re-asks the same turn.
            # BUDGETED repair (core/model_client.py repair contract): one
            # transport request, TOTAL duration within timeout_ms.
            attempt_no = attempts.begin_attempt(is_repair=True)
            try:
                raw = self._complete(
                    messages=messages, attempt=attempt_no, kind="repair",
                    prompt_hash=prompt_hash, temperature=temperature,
                    max_tokens=max_tokens, timeout_ms=timeout_ms,
                    disable_transport_retries=True)
            except ProviderCapacityError:
                raise
            except TransportError as exc2:
                raise AttemptBudgetExhausted(
                    f"both API attempts failed at transport level "
                    f"(attempt 1: {exc}; attempt 2: {exc2}); per-turn cap "
                    f"({MAX_ATTEMPTS_PER_TURN}) reached") from exc2
            batch, failure = self._parse_and_validate(raw, ref_validator)
            if failure is not None:
                raise AttemptBudgetExhausted(
                    f"repair attempt (2/{MAX_ATTEMPTS_PER_TURN}) failed: "
                    f"{failure.error_class}: {failure.detail}; no attempts "
                    "remain") from None
            return finish(batch)

        batch, failure = self._parse_and_validate(raw, ref_validator)
        if failure is None:
            return finish(batch)

        # ---- attempt 2 (the one repair)
        repair_messages = prompt_templates.repair_messages(
            messages,
            raw.content,
            failure.error_class,
            detail=failure.detail,
            failed_refs=", ".join(failure.failed_refs) or "(none)",
            refreshed_candidates=(
                repair_candidates_for(failure)
                if repair_candidates_for is not None
                else "(the candidate table from the turn prompt above is "
                     "unchanged and still applies)"),
        )
        attempt_no = attempts.begin_attempt(is_repair=True)
        try:
            # BUDGETED repair (core/model_client.py repair contract): one
            # transport request, TOTAL duration within timeout_ms.
            raw2 = self._complete(
                messages=repair_messages, attempt=attempt_no, kind="repair",
                prompt_hash=prompt_hash, temperature=temperature,
                max_tokens=max_tokens, timeout_ms=timeout_ms,
                disable_transport_retries=True)
        except ProviderCapacityError:
            raise
        except TransportError as exc:
            raise AttemptBudgetExhausted(
                f"repair attempt (2/{MAX_ATTEMPTS_PER_TURN}) failed at "
                f"transport level: {exc}; no attempts remain") from exc
        batch2, failure2 = self._parse_and_validate(raw2, ref_validator)
        if failure2 is not None:
            raise AttemptBudgetExhausted(
                f"repair attempt (2/{MAX_ATTEMPTS_PER_TURN}) failed: "
                f"{failure2.error_class}: {failure2.detail}; no attempts "
                "remain") from None
        return finish(batch2)

    # -- single-attempt seam (the ModelClient adapter, model/turn_client) --

    def complete_once(
        self,
        *,
        messages: list[Message],
        prompt_hash: str,
        attempt: int,
        kind: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Any = _KEEP,
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        call_event_context: Optional[Mapping[str, Any]] = None,
    ) -> AttemptOutcome:
        """ONE logical API attempt (see ``AttemptOutcome``). Never raises
        for transport or schema failures — both are reported in the outcome
        (the caller's repair/degraded flow decides what happens next).
        ``timeout_ms`` overrides the per-call request timeout (budget-aware
        repair gating, WP4); None = the configured default — it bounds the
        WHOLE single request (connect and read included).
        ``disable_transport_retries``: REPAIR attempts MUST pass True — the
        attempt then makes exactly one request and its TOTAL duration is
        bounded by ``timeout_ms``; the initial attempt keeps the transport's
        transparent retries.

        ``action_batch`` drives one whole turn (attempt 1 + the one repair,
        owning the ≤2-attempt cap); ``complete_once`` performs exactly ONE
        logical attempt, leaving the cap and the repair flow to the caller
        (the WP2 runner owns both, core/runner.py). SDK/transport retries
        inside the attempt are transparent and never counted (decision 4).
        The attempt is logged like every other call."""
        try:
            raw = self._complete(
                messages=messages, attempt=attempt, kind=kind,
                prompt_hash=prompt_hash, temperature=temperature,
                max_tokens=max_tokens, timeout_ms=timeout_ms,
                response_format=response_format,
                disable_transport_retries=disable_transport_retries,
                tools=tools, tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                call_event_context=call_event_context)
        except ProviderCapacityError:
            raise
        except TransportError as exc:
            partial = getattr(exc, "partial", None)
            return AttemptOutcome(raw=partial, error=str(exc), batch=None,
                                  malformed=None, raw_content="",
                                  call_id=getattr(partial, "call_id", None))
        try:
            batch = parse_action_batch(raw.content)
        except SchemaError as exc:
            return AttemptOutcome(raw=raw, error=None, batch=None,
                                  malformed=exc.detail,
                                  raw_content=raw.content, call_id=raw.call_id)
        return AttemptOutcome(raw=raw, error=None, batch=batch, malformed=None,
                              raw_content=raw.content, call_id=raw.call_id)

    # -- per-arm model fingerprint (decision 8) --------------------------------

    def _fingerprint_sample_mode(self) -> str:
        """How the K=3 probe samples are collected: ONE request with n=3
        when the transport supports multi-sample requests (the SDK path —
        ``Chat.complete`` takes ``n``, offline-verified against mistralai
        3.4.0), else three sequential requests. The mode is recorded on the
        probe."""
        try:
            params = inspect.signature(self._transport.complete).parameters
        except (TypeError, ValueError):
            return FINGERPRINT_MODE_SEQUENTIAL
        if "n" not in params:
            return FINGERPRINT_MODE_SEQUENTIAL
        if (isinstance(self._transport, MistralSDKTransport)
                and not _sdk_chat_complete_accepts_n()):
            return FINGERPRINT_MODE_SEQUENTIAL
        return FINGERPRINT_MODE_SINGLE_REQUEST

    def probe_fingerprint(self) -> FingerprintProbe:
        """Per-arm model fingerprint (decision 8, FW-21): the FIXED probe
        prompt v2 (capacity-revealing; versioned file, hash recorded) under
        the arm's sampling (temperature 1.0, reasoning_effort high, no
        top_p — decision 4 amendment: every call runs at the arm sampling),
        collecting K=3 samples — one request with n=3 when the SDK path
        supports it, else three sequential requests (the mode is recorded).
        The probe leaves max_tokens unset so the provider can use its full
        available output budget; this is recorded as ``None``. The
        persisted fingerprint is the sorted, deduplicated SET of the sample
        output hashes (``encode_fingerprint_set``). Cached on the client
        (``arm_fingerprint``) and carried by every subsequent per-call log.
        K logical probe requests (not turn calls).

        Fail-closed (DEF-4): when fewer than FINGERPRINT_PROBE_SAMPLES
        usable samples arrive, or a sample is not usable content (e.g.
        None from a contract-violating transport), the probe raises
        ``FingerprintProbeError`` — the caller must abort the evaluation
        with a recorded infrastructure failure, never run a metric-bearing
        comparison against an invalid fingerprint."""
        prompt_text = prompt_templates.load_prompt(FINGERPRINT_PROBE_PROMPT)
        prompt_hash = prompt_templates.prompt_hash(FINGERPRINT_PROBE_PROMPT)
        messages: list[Message] = [
            {"role": "system",
             "content": "You are answering a fixed model-fingerprint probe. "
                        "Follow the template exactly."},
            {"role": "user", "content": prompt_text},
        ]
        mode = self._fingerprint_sample_mode()
        logs_before = len(self.call_logs)
        if mode == FINGERPRINT_MODE_SINGLE_REQUEST:
            raw = self._complete(
                messages=messages, attempt=0, kind="probe",
                prompt_hash=prompt_hash,
                call_event_context={"physical_kind": "probe"},
                response_format=None, n=FINGERPRINT_PROBE_SAMPLES)
            raws = [raw]
            samples = list(raw.sample_contents or [raw.content])
        else:
            raws = []
            samples = []
            for _ in range(FINGERPRINT_PROBE_SAMPLES):
                raw = self._complete(
                    messages=messages, attempt=0, kind="probe",
                    prompt_hash=prompt_hash,
                    call_event_context={"physical_kind": "probe"},
                    response_format=None)
                raws.append(raw)
                samples.append(raw.content)
        # DEF-4, fail-closed: a probe that cannot obtain the required
        # sample set is a recorded infrastructure failure, never a
        # metric-bearing comparison with an invalid fingerprint. The
        # content guard also keeps a contract-violating transport (None
        # sample) from crashing the probe with AttributeError.
        if len(samples) != FINGERPRINT_PROBE_SAMPLES:
            raise FingerprintProbeError(
                f"fingerprint probe failed: expected "
                f"{FINGERPRINT_PROBE_SAMPLES} samples, got {len(samples)} "
                f"(mode {mode}); the caller must abort the evaluation with "
                "a recorded infrastructure failure")
        invalid = [i for i, sample in enumerate(samples)
                   if not isinstance(sample, str)]
        if invalid:
            raise FingerprintProbeError(
                f"fingerprint probe failed: sample(s) {invalid} are not "
                f"usable content (types: "
                f"{[type(samples[i]).__name__ for i in invalid]}); the "
                "caller must abort the evaluation with a recorded "
                "infrastructure failure")
        probe_logs = self.call_logs[logs_before:]
        output_hashes = sorted({
            hashlib.sha256(sample.encode("utf-8")).hexdigest()
            for sample in samples})
        tokens_in_values = [r.prompt_tokens for r in raws]
        tokens_out_values = [r.completion_tokens for r in raws]
        total_values = [r.total_tokens for r in raws]
        tokens_in = (sum(tokens_in_values) if all(v is not None for v in tokens_in_values)
                     else None)
        tokens_out = (sum(tokens_out_values) if all(v is not None for v in tokens_out_values)
                      else None)
        tokens_total = (sum(total_values) if all(v is not None for v in total_values)
                        else None)
        probe = FingerprintProbe(
            fingerprint=encode_fingerprint_set(output_hashes),
            output_hashes=output_hashes,
            sample_attempts=len(samples),
            sample_mode=mode,
            max_tokens=FINGERPRINT_PROBE_MAX_TOKENS,
            model_requested=self.model,
            model_responded=raws[-1].model,
            prompt_hash=prompt_hash,
            temperature=self.temperature,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            latency_ms=sum(log.latency_ms for log in probe_logs),
            cost_usd=self._cost_usd(tokens_in, tokens_out),
            response_id=raws[-1].response_id,
            finish_reason=raws[-1].finish_reason,
            ts=_utc_now_iso(),
        )
        self.arm_fingerprint = probe.fingerprint
        return probe

    # -- live probes (P10) ------------------------------------------------------

    def probe_model_id(self) -> ModelIdProbe:
        """P10a: confirm the EXACT model ID responds (no silent
        substitution). A transport failure is captured as evidence
        (``responded=False``), not raised. Logged as a probe call."""
        messages: list[Message] = [
            {"role": "user", "content": "Reply with exactly: OK"},
        ]
        try:
            raw = self._complete(
                messages=messages, attempt=0, kind="probe",
                prompt_hash="p10a-model-id",
                call_event_context={"physical_kind": "probe"},
                response_format=None)
        except TransportError as exc:
            return ModelIdProbe(
                model_requested=self.model, model_responded=None,
                responded=False, response_id=None, finish_reason=None,
                latency_ms=self.call_logs[-1].latency_ms,
                error=str(exc), ts=_utc_now_iso())
        return ModelIdProbe(
            model_requested=self.model, model_responded=raw.model,
            responded=True, response_id=raw.response_id,
            finish_reason=raw.finish_reason,
            latency_ms=self.call_logs[-1].latency_ms,
            error=None, ts=_utc_now_iso())

    def probe_json_schema_support(self, *, probe_prompt: str) -> JsonSchemaProbe:
        """P10b: test ``json_schema`` structured-output support on the exact
        model ID — a minimal ActionBatch-shaped request carrying the REAL
        ActionBatch JSON schema (candidate-ID-constrained ``ref`` fields),
        strict mode, the arm sampling (decision 4 amendment). Evidence
        records the response SHAPE (no raw model output beyond what
        evidences the decision). Logged as a probe call."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "action_batch",
                "description": (
                    "One turn of play as a batch of candidate-ID actions "
                    "(lechonk ActionBatch)."),
                "schema_definition": action_batch_json_schema(),
                "strict": True,
            },
        }
        messages: list[Message] = [
            {"role": "user", "content": probe_prompt},
        ]
        try:
            raw = self._complete(
                messages=messages, attempt=0, kind="probe",
                prompt_hash="p10b-json-schema",
                call_event_context={"physical_kind": "probe"},
                response_format=response_format)
        except TransportError as exc:
            return JsonSchemaProbe(
                model_requested=self.model, model_responded=None,
                schema_hash=action_batch_schema_hash(),
                response_format_type="json_schema", strict=True,
                responded=False, json_schema_supported=False,
                response_shape={},
                latency_ms=self.call_logs[-1].latency_ms,
                error=str(exc), ts=_utc_now_iso())
        latency_ms = self.call_logs[-1].latency_ms
        value = _extract_json_object(raw.content)
        supported = False
        shape: dict[str, Any] = {
            "content_is_json_object": value is not None,
            "top_level_keys": sorted(value.keys()) if value is not None else [],
            "finish_reason": raw.finish_reason,
            "model_responded": raw.model,
        }
        if value is not None:
            try:
                ActionBatch.model_validate(value)
                supported = True
                shape["valid_action_batch"] = True
            except ValidationError as exc:
                shape["valid_action_batch"] = False
                shape["validation_error"] = _summarize_validation_error(exc)
        return JsonSchemaProbe(
            model_requested=self.model, model_responded=raw.model,
            schema_hash=action_batch_schema_hash(),
            response_format_type="json_schema", strict=True,
            responded=True, json_schema_supported=supported,
            response_shape=shape, latency_ms=latency_ms, error=None,
            ts=_utc_now_iso())
