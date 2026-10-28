# SPDX-License-Identifier: MIT
"""lechonk.core.model_client — the core-defined model-client interface (WP2).

Core stays model-agnostic and offline-testable: the runner consumes THIS
interface, never a concrete client. Package A (src/lechonk/model/) owns the
real Mistral client and the versioned prompt templates; tests drive the
runner with a fake client implementing this protocol.

One request budget per turn (decision 4): ≤2 total API attempts including
at most one repair. The runner enforces the cap; the client is called at
most twice per turn (``complete_turn`` then at most once
``complete_repair``). SDK/transport-level retries are transparent to this
interface (logged by the client, never counted against the cap).

Output enforcement (json_schema vs schema-validated parsing) is package A's
WP2 spike decision; both paths produce the SAME semantics here:
``ModelResponse.batch`` is a schema-validated ActionBatch or None, with
``malformed`` set when the output violated the batch schema (the repair
class is then schema_error) and ``error`` set when the call failed at
transport level (timeout etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from lechonk.core.schemas import ActionBatch

# Repair classes (the per-class repair semantics, §7.5): the three
# illegal-action taxonomy classes plus "unresponsive" (transport-level
# failure with no parseable output — the repair is a bounded re-ask).
REPAIR_CLASS_SCHEMA = "schema_error"
REPAIR_CLASS_STATE_FOG = "state_fog_error"
REPAIR_CLASS_ILLEGAL = "illegal_in_rule"
REPAIR_CLASS_UNRESPONSIVE = "unresponsive"
REPAIR_CLASSES: tuple[str, ...] = (
    REPAIR_CLASS_SCHEMA, REPAIR_CLASS_STATE_FOG, REPAIR_CLASS_ILLEGAL,
    REPAIR_CLASS_UNRESPONSIVE,
)

# Budget-aware repair gating (WP4): the repair attempt's per-call hard
# request timeout is the remaining ACTING budget minus this margin, so a
# timed-out repair still leaves the margin for the degraded default and the
# turn's deadline check — the turn never exceeds the ACTING deadline by
# construction.
REPAIR_TIMEOUT_MARGIN_S = 5.0


class PromptBudgetExceeded(RuntimeError):
    """An assembled prompt exceeded its §3 token budget.

    Canonical home is core so the runner can catch it without importing the
    model package; the model-side assembler (lechonk.model.prompts)
    re-exports this name. The runner maps it to the dedicated bounded
    fail-stop — the ``prompt_budget_exceeded`` metric event, never a generic
    infrastructure truncation (FW-13)."""


@dataclass
class TurnPrompt:
    """One turn's prompt payload (the runner renders the sections; the
    client owns the templates, §7.5)."""

    call_id: str
    turn_id: str
    memory: str
    snapshot: str
    delta: str
    candidates: str
    # Optional bounded historical decision journal.  Empty preserves the
    # legacy prompt byte-for-byte; hybrid interactive runs populate it.
    history: str = ""
    # The runner's remaining ACTING-phase budget (seconds); the client uses
    # it for its own call timeout — on the repair attempt it is the per-call
    # hard request timeout's base (budget minus REPAIR_TIMEOUT_MARGIN_S).
    # Advisory for fakes.
    budget_s: Optional[float] = None


@dataclass
class RepairPrompt(TurnPrompt):
    """The repair prompt (attempt 2, per error class, §7.5): the failed
    refs and the refreshed candidate list are attached."""

    error_class: str = REPAIR_CLASS_UNRESPONSIVE  # one of REPAIR_CLASSES
    failed_refs: list[str] = field(default_factory=list)
    refreshed_candidates: str = ""


# A deliberately small, model-facing extension point.  The callable receives
# one bounded request object and returns JSON-serialisable, read-only reference
# material.  Keeping this type in core avoids importing the Freeciv adapter in
# the model protocol.
RulesetLookup = Callable[[Mapping[str, Any]], Any]


@dataclass
class ModelResponse:
    """One model call's result + per-call logging (decision 4: every call
    is logged — tokens, latency, cost)."""

    call_id: str
    prompt_hash: str
    batch: Optional[ActionBatch] = None
    # The model-emitted action refs as raw strings, for schema-error
    # accounting when the output violated the batch schema (the ledger
    # records one rejected attempt per emitted ref).
    raw_attempts: list[str] = field(default_factory=list)
    # Set when the output violated the ActionBatch schema (repair class
    # schema_error).
    malformed: Optional[str] = None
    # Set when the call failed at transport level (timeout, API error).
    error: Optional[str] = None
    latency_ms: int = 0
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    usage_status: Optional[str] = None
    cost_status: Optional[str] = None
    fingerprint: str = ""


@runtime_checkable
class ModelClient(Protocol):
    """The model-client interface the runner consumes (core-defined so core
    stays model-agnostic; package A provides the Mistral implementation).

    A client MAY declare a ``provenance_class`` attribute (one of the frozen
    ``PROVENANCE_CLASSES``; the default when absent is ``"model"``): the
    runner labels the policy-choice-level ledger records with it. WP3
    paired runs drive a scripted comparator arm through the same pipeline;
    its decisions are ``scripted_default``, never ``model``.

    Timeout semantics (budget-aware repair gating, WP4 — normative): a
    per-call timeout derived from ``TurnPrompt.budget_s`` (the runner's
    remaining ACTING budget) must bound the WHOLE single request — connect
    and read included — so a hung request surfaces as a transport-level
    failure within the timeout. On ``complete_repair`` the client MUST
    disable any SDK/transport-level internal retry: the harness-level
    attempt structure (attempt 1 + one repair) is the harness's retry
    policy, and a transport retry inside the budgeted repair would exceed
    the repair's budget — the repair attempt's TOTAL duration must stay
    within the budget-derived timeout (plus small overhead). The initial
    ``complete_turn`` attempt keeps transparent transport retries.
    """

    def complete_turn(self, prompt: TurnPrompt) -> ModelResponse:
        """The primary per-turn call (attempt 1)."""
        ...

    def complete_repair(self, prompt: RepairPrompt) -> ModelResponse:
        """The repair call (attempt 2, at most once per turn). Its TOTAL
        duration — any transport-level retry included — must be bounded by
        the budget-derived per-call timeout (see the class docstring)."""
        ...
