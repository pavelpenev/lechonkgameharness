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

Per-arm model fingerprint (decision 8): ``probe_fingerprint`` sends a FIXED
probe prompt under the arm sampling and records the output hash; the
fingerprint travels with every per-call log and goes into the run manifest.

This module never imports civrealm (plan §7.1). The official ``mistralai``
SDK is imported lazily by ``MistralSDKTransport`` only — offline tests drive
the client through a mock transport and never import the SDK.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol

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
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REASONING_EFFORT = "high"

# Studio list prices, USD per million tokens. PROVISIONAL for
# ``mistral-large-4``: carried from the published Mistral Large list price
# (web check 2026-10-24: $0.50/MTok input, $1.50/MTok output); verify against
# the console before evaluation runs. Configurable per client (``pricing``).
DEFAULT_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    MODEL_ID: {"input": 0.50, "output": 1.50},
}


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

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}

_KEEP = object()  # sentinel: use the client's configured response_format


@dataclass
class RawCompletion:
    """One logical API attempt's raw result, normalized by the transport."""

    content: str
    model: str
    response_id: str
    created: int
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    transport_retries: int = 0
    """Transparent retries performed INSIDE this logical attempt (SDK-level
    retries are not individually observable; the SDK debug logger records
    them). Never counted against the per-turn cap (decision 4)."""


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
    sends ``top_p``."""

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float,
        reasoning_effort: str,
        max_tokens: int,
        response_format: Optional[dict[str, Any]],
        random_seed: Optional[int],
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
    ) -> RawCompletion: ...


def _message_content_text(content: Any) -> str:
    """Normalize an SDK message content (str | list of chunks | None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


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
        max_tokens: int,
        response_format: Optional[dict[str, Any]],
        random_seed: Optional[int],
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
    ) -> RawCompletion:
        from mistralai.client.errors import (
            NoResponseError,
            ResponseValidationError,
            SDKError,
        )

        sdk = self._sdk_no_retries if disable_transport_retries else self._sdk
        try:
            # Arm sampling (decision 4 amendment): reasoning_effort is passed
            # through as an explicit Chat.complete parameter (offline-verified
            # against mistralai 3.4.0); top_p is NEVER sent (omitting it is
            # safe — the SDK leaves it unset).
            response = sdk.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                response_format=response_format,
                random_seed=random_seed,
                timeout_ms=timeout_ms,
            )
        except (NoResponseError, ResponseValidationError) as exc:
            raise TransportError(
                f"Mistral API transport failure: {exc}") from exc
        except SDKError as exc:
            status = getattr(getattr(exc, "raw_response", None),
                             "status_code", None)
            raise TransportError(
                f"Mistral API error (status={status}): {exc}") from exc
        choices = list(response.choices or [])
        choice = choices[0] if choices else None
        usage = response.usage
        return RawCompletion(
            content=_message_content_text(choice.message.content)
            if choice is not None else "",
            model=response.model,
            response_id=response.id,
            created=response.created,
            finish_reason=choice.finish_reason if choice is not None else None,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


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
    tokens_in: int
    tokens_out: int
    tokens_total: int
    cost_usd: Optional[float]
    finish_reason: Optional[str]
    response_id: Optional[str]
    fingerprint: str  # arm model fingerprint (probe hash) or "unprobed"
    transport_retries: int
    error: Optional[str]
    ts: str
    reasoning_effort: str = DEFAULT_REASONING_EFFORT

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


@dataclass
class FingerprintProbe:
    """Per-arm model fingerprint (decision 8): fixed probe prompt, the arm's
    own sampling (temperature 1.0, reasoning_effort high, no top_p),
    recorded output hash."""

    fingerprint: str  # "sha256:<hex>" of the probe output
    output_hash: str
    model_requested: str
    model_responded: str
    prompt_hash: str
    temperature: float
    tokens_in: int
    tokens_out: int
    tokens_total: int
    latency_ms: int
    cost_usd: Optional[float]
    response_id: str
    finish_reason: Optional[str]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "output_hash": self.output_hash,
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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        random_seed: Optional[int] = None,
        log_sink: Optional[Callable[[CallLog], None]] = None,
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
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.random_seed = random_seed
        self._transport: ChatTransport = (
            transport if transport is not None
            else MistralSDKTransport(api_key, timeout_ms=timeout_ms))
        self.call_logs: list[CallLog] = []
        self._log_sink = log_sink
        self.arm_fingerprint: Optional[str] = None
        self._call_counter = 0

    # -- logging ------------------------------------------------------------

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"call-{self._call_counter:04d}"

    def _emit_log(self, record: CallLog) -> None:
        self.call_logs.append(record)
        if self._log_sink is not None:
            self._log_sink(record)

    def _cost_usd(self, tokens_in: int, tokens_out: int) -> Optional[float]:
        price = self.pricing.get(self.model)
        if not price:
            return None
        cost = (tokens_in * price["input"]
                + tokens_out * price["output"]) / 1_000_000
        return round(cost, 6)

    def _make_call_log(
        self,
        *,
        attempt: int,
        kind: str,
        prompt_hash: str,
        latency_ms: int,
        raw: Optional[RawCompletion],
        error: Optional[str],
        reasoning_effort: str,
    ) -> CallLog:
        return CallLog(
            call_id=self._next_call_id(),
            attempt=attempt,
            kind=kind,
            prompt_hash=prompt_hash,
            model_requested=self.model,
            model_responded=raw.model if raw is not None else None,
            enforcement=self.enforcement.value,
            temperature=self.temperature,
            latency_ms=latency_ms,
            tokens_in=raw.prompt_tokens if raw is not None else 0,
            tokens_out=raw.completion_tokens if raw is not None else 0,
            tokens_total=raw.total_tokens if raw is not None else 0,
            cost_usd=(self._cost_usd(raw.prompt_tokens, raw.completion_tokens)
                      if raw is not None else None),
            finish_reason=raw.finish_reason if raw is not None else None,
            response_id=raw.response_id if raw is not None else None,
            fingerprint=self.arm_fingerprint or "unprobed",
            transport_retries=raw.transport_retries if raw is not None else 0,
            error=error,
            ts=_utc_now_iso(),
            reasoning_effort=reasoning_effort,
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
        max_tokens: Optional[int] = None,
        response_format: Any = _KEEP,
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
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
        NEVER sends ``top_p``."""
        t0 = time.perf_counter()
        effective_temperature = (self.temperature if temperature is None
                                 else temperature)
        effective_reasoning_effort = (self.reasoning_effort
                                      if reasoning_effort is None
                                      else reasoning_effort)
        effective_max_tokens = (self.max_tokens if max_tokens is None
                                else max_tokens)
        effective_response_format = (self._response_format_payload()
                                     if response_format is _KEEP
                                     else response_format)
        try:
            raw = self._transport.complete(
                messages=messages,
                model=self.model,
                temperature=effective_temperature,
                reasoning_effort=effective_reasoning_effort,
                max_tokens=effective_max_tokens,
                response_format=effective_response_format,
                random_seed=self.random_seed,
                timeout_ms=timeout_ms,
                disable_transport_retries=disable_transport_retries,
            )
        except TransportError as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._emit_log(self._make_call_log(
                attempt=attempt, kind=kind, prompt_hash=prompt_hash,
                latency_ms=latency_ms, raw=None, error=str(exc),
                reasoning_effort=effective_reasoning_effort))
            raise
        except Exception as exc:  # unexpected transport bug — same accounting
            latency_ms = int((time.perf_counter() - t0) * 1000)
            wrapped = TransportError(f"transport failure: {exc}")
            self._emit_log(self._make_call_log(
                attempt=attempt, kind=kind, prompt_hash=prompt_hash,
                latency_ms=latency_ms, raw=None, error=str(wrapped),
                reasoning_effort=effective_reasoning_effort))
            raise wrapped from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._emit_log(self._make_call_log(
            attempt=attempt, kind=kind, prompt_hash=prompt_hash,
            latency_ms=latency_ms, raw=raw, error=None,
            reasoning_effort=effective_reasoning_effort))
        return raw

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
        except TransportError as exc:
            # No model output: the single repair slot re-asks the same turn.
            attempt_no = attempts.begin_attempt(is_repair=True)
            try:
                raw = self._complete(
                    messages=messages, attempt=attempt_no, kind="repair",
                    prompt_hash=prompt_hash, temperature=temperature,
                    max_tokens=max_tokens)
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
            raw2 = self._complete(
                messages=repair_messages, attempt=attempt_no, kind="repair",
                prompt_hash=prompt_hash, temperature=temperature,
                max_tokens=max_tokens)
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
        timeout_ms: Optional[int] = None,
        disable_transport_retries: bool = False,
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
                disable_transport_retries=disable_transport_retries)
        except TransportError as exc:
            return AttemptOutcome(raw=None, error=str(exc), batch=None,
                                  malformed=None, raw_content="")
        try:
            batch = parse_action_batch(raw.content)
        except SchemaError as exc:
            return AttemptOutcome(raw=raw, error=None, batch=None,
                                  malformed=exc.detail,
                                  raw_content=raw.content)
        return AttemptOutcome(raw=raw, error=None, batch=batch, malformed=None,
                              raw_content=raw.content)

    # -- per-arm model fingerprint (decision 8) --------------------------------

    def probe_fingerprint(self) -> FingerprintProbe:
        """Per-arm model fingerprint: a FIXED probe prompt (versioned file,
        hash recorded) under the arm's sampling (temperature 1.0,
        reasoning_effort high, no top_p — decision 4 amendment: every call
        runs at the arm sampling); the recorded output hash is the
        fingerprint. Cached on the client (``arm_fingerprint``) and carried
        by every subsequent per-call log. One logical probe attempt (not a
        turn call)."""
        prompt_text = prompt_templates.fingerprint_probe_prompt()
        messages: list[Message] = [
            {"role": "system",
             "content": "You are a deterministic fingerprint probe. Follow "
                        "the instruction exactly."},
            {"role": "user", "content": prompt_text},
        ]
        raw = self._complete(
            messages=messages, attempt=0, kind="probe",
            prompt_hash=prompt_templates.prompt_hash(
                prompt_templates.FINGERPRINT_PROBE),
            max_tokens=256, response_format=None)
        output_hash = hashlib.sha256(
            raw.content.encode("utf-8")).hexdigest()
        probe = FingerprintProbe(
            fingerprint=f"sha256:{output_hash}",
            output_hash=output_hash,
            model_requested=self.model,
            model_responded=raw.model,
            prompt_hash=prompt_templates.prompt_hash(
                prompt_templates.FINGERPRINT_PROBE),
            temperature=self.temperature,
            tokens_in=raw.prompt_tokens,
            tokens_out=raw.completion_tokens,
            tokens_total=raw.total_tokens,
            latency_ms=self.call_logs[-1].latency_ms,
            cost_usd=self._cost_usd(raw.prompt_tokens, raw.completion_tokens),
            response_id=raw.response_id,
            finish_reason=raw.finish_reason,
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
                prompt_hash="p10a-model-id", max_tokens=16,
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
                max_tokens=1024, response_format=response_format)
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
