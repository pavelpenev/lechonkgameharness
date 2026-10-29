# SPDX-License-Identifier: MIT
"""Transactional, cache-friendly conversation history for interactive runs.

The manager deliberately owns only message history.  A caller supplies a
compactor when the projected context crosses the configured threshold; no
messages are silently discarded.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from lechonk.core.jsonutil import canonical_json
from lechonk.core.schemas import Commitment, estimate_tokens


class ConversationError(RuntimeError):
    pass


class ConversationPersistenceError(ConversationError, OSError):
    """The WAL is durable but the state snapshot could not be published."""


class CompactionRequired(ConversationError):
    """Raised when a new turn would cross the configured context limit."""


@dataclass(frozen=True)
class ConversationConfig:
    trigger_tokens: int = 500_000
    tail_tokens: int = 100_000
    policy_version: str = "conversation_v1"

    def __post_init__(self) -> None:
        if self.trigger_tokens <= 0 or self.tail_tokens <= 0:
            raise ValueError("conversation token limits must be positive")
        if self.tail_tokens >= self.trigger_tokens:
            raise ValueError("tail_tokens must be below trigger_tokens")


def _copy_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # JSON round-trip gives callers isolation and guarantees SDK-safe values.
    try:
        return json.loads(canonical_json(list(messages)))
    except (TypeError, ValueError) as exc:
        raise ConversationError(f"messages are not JSON-compatible: {exc}") from exc


def _message_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    return sum(estimate_tokens(canonical_json(message)) for message in messages)


def _safe_message_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    """Upper-bound unmeasured JSON suffixes by UTF-8 bytes plus framing."""
    encoded = [canonical_json(message).encode("utf-8") for message in messages]
    return sum(len(value) + 4 for value in encoded)


def _message_hash(message: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(message).encode("utf-8")).hexdigest()


def _message_hash_payload(messages: Iterable[Mapping[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(list(messages)).encode("utf-8")).hexdigest()


def _valid_hash(value: Any) -> bool:
    return (isinstance(value, str) and value.startswith("sha256:")
            and len(value) == len("sha256:") + 64
            and all(ch in "0123456789abcdef" for ch in value.removeprefix("sha256:")))


def _valid_anchor_schema(anchor: Any) -> bool:
    if not isinstance(anchor, dict):
        return False
    if (isinstance(anchor.get("epoch"), bool) or not isinstance(anchor.get("epoch"), int)
            or anchor["epoch"] < 0):
        return False
    for field in ("prompt_tokens", "estimated_tokens", "message_count"):
        value = anchor.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if field != "message_count" and value <= 0:
            return False
        if field == "message_count" and value < 0:
            return False
    completion = anchor.get("completion_tokens")
    if completion is not None and (isinstance(completion, bool) or not isinstance(completion, int)
                                   or completion < 0):
        return False
    return (_valid_hash(anchor.get("request_hash"))
            and (anchor.get("assistant_hash") is None or _valid_hash(anchor.get("assistant_hash"))))


def _fsync_parent_directory(path: str | Path) -> None:
    """Persist the directory entry containing ``path``.

    File fsync alone does not make an atomic replacement (or a newly-created
    journal) survive a crash: the parent directory entry must be synced too.
    Errors are intentionally allowed to propagate so callers do not report a
    state transition as durable when it is not.
    """
    directory = Path(path).parent.resolve(strict=True)
    dir_fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _validate_compaction_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the model-authored checkpoint before replacing history.

    Compaction is destructive, so a syntactically valid JSON object is not
    sufficient.  Keep this contract deliberately small and JSON-native.
    """
    required = ("summary", "commitments", "uncertainties", "facts")
    if (not isinstance(summary.get("version"), int) or
            isinstance(summary.get("version"), bool) or
            summary.get("version") != 1):
        raise ConversationError("compaction summary version must be 1")
    if any(key not in summary for key in required):
        raise ConversationError(
            "compaction summary must contain summary, commitments, "
            "uncertainties, and facts")
    allowed = {"version", "summary", "commitments", "uncertainties", "facts",
               "historical_note"}
    if not set(summary) <= allowed:
        raise ConversationError("compaction summary contains unknown fields")
    text = summary["summary"]
    if not isinstance(text, str) or not text.strip():
        raise ConversationError("compaction summary text must be a nonempty string")
    normalized: dict[str, Any] = {"version": 1, "summary": text.strip()}
    if "historical_note" in summary:
        historical_note = summary["historical_note"]
        if not isinstance(historical_note, str) or not historical_note.strip():
            raise ConversationError(
                "compaction historical_note must be a nonempty string")
        normalized["historical_note"] = historical_note.strip()
    for key in required[1:]:
        value = summary[key]
        if not isinstance(value, list) or len(value) > 128:
            raise ConversationError(f"compaction {key} must be a bounded list")
        items: list[Any] = []
        for index, item in enumerate(value):
            if key == "commitments" and isinstance(item, Mapping):
                try:
                    commitment = Commitment.model_validate(item, strict=True)
                except Exception as exc:
                    raise ConversationError(
                        "compaction commitments contains an invalid typed commitment") from exc
                encoded = commitment.model_dump(mode="json", exclude_none=False)
                items.append(encoded)
            elif isinstance(item, str) and item.strip():
                items.append(item.strip())
            else:
                item_type = type(item).__name__
                raise ConversationError(
                    f"compaction {key}[{index}] invalid item type {item_type}")
        normalized[key] = items
    if estimate_tokens(canonical_json(normalized)) > 8_000:
        raise ConversationError("compaction summary exceeds its token budget")
    return normalized


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    if not messages or messages[0].get("role") != "system":
        raise ConversationError("conversation must begin with a system message")
    # Tool use is deliberately validated as a single protocol transaction:
    # assistant(tool_calls) -> matching tool results -> assistant(final).
    # Keeping this constraint here (rather than in the API adapter) prevents
    # an orphan result from ever entering state, the journal, or a compacted
    # prefix.
    pending: list[str] | None = None
    awaiting_final = False
    for index, message in enumerate(messages[1:], 1):
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant", "tool"):
            raise ConversationError(f"invalid conversation role at {index}")
        if role == "assistant":
            calls = message.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list) or not calls:
                    raise ConversationError("assistant tool_calls must be a non-empty list")
                if pending is not None or awaiting_final:
                    raise ConversationError("assistant tool calls interrupt a pending tool transaction")
                ids: list[str] = []
                for call in calls:
                    if not isinstance(call, Mapping) or not isinstance(call.get("id"), str) or not call["id"].strip():
                        raise ConversationError("assistant tool call has no valid id")
                    if call["id"] in ids:
                        raise ConversationError("assistant tool call ids must be unique")
                    ids.append(call["id"])
                pending = ids
                awaiting_final = False
            elif awaiting_final:
                # This is the required final assistant response after all
                # tool results. Its thinking/content is preserved verbatim.
                awaiting_final = False
            elif pending is not None:
                raise ConversationError("assistant final arrived before all tool results")
            continue
        if role == "tool":
            if pending is None or awaiting_final:
                raise ConversationError("orphan tool result")
            tool_id = message.get("tool_call_id")
            if not isinstance(tool_id, str) or tool_id not in pending:
                raise ConversationError("tool result has no matching tool_call_id")
            pending.remove(tool_id)
            if not pending:
                pending = None
                awaiting_final = True
            continue
        if pending is not None or awaiting_final:
            raise ConversationError("user message interrupts a tool transaction")
        special = isinstance(content, str) and (
            content.startswith("[EXECUTION REPORT]") or
            content.startswith("[ENGINE-ONLY REPORT]") or
            content.startswith("[CONVERSATION COMPACTION]"))
        if role == "user" and special:
            if content.startswith("[ENGINE-ONLY REPORT]"):
                # Degraded turns have no assistant response; preserve this
                # explicit engine event as a valid standalone group.
                continue
            if index == 1 or messages[index - 1].get("role") != "assistant":
                # compaction may directly follow system, while an execution
                # report must follow an assistant response.
                if not content.startswith("[CONVERSATION COMPACTION]") or index != 1:
                    raise ConversationError("invalid report/checkpoint ordering")
    if pending is not None or awaiting_final:
        raise ConversationError("conversation ends with an incomplete tool transaction")


class ConversationManager:
    """Append-only conversation with turn transactions and explicit compaction.

    ``begin_turn`` stages a user message.  ``stage_assistant`` may be called
    for the complete response (including thinking chunks), then ``commit_turn``
    publishes both.  ``abort_turn`` drops staged data after a partial failure.
    """

    def __init__(self, *, system_message: Mapping[str, Any],
                 config: ConversationConfig | None = None,
                 state_path: str | Path | None = None,
                 journal_path: str | Path | None = None,
                 prompt_cache_key: str | None = None):
        self.config = config or ConversationConfig()
        self.messages: list[dict[str, Any]] = _copy_messages([system_message])
        self._staged: list[dict[str, Any]] | None = None
        self._staged_assistant: dict[str, Any] | None = None
        self.epoch = 0
        self.state_path = Path(state_path) if state_path else None
        self.journal_path = Path(journal_path) if journal_path else None
        self.prompt_cache_key = prompt_cache_key
        self._observed_prompt_tokens: int = 0
        self._observed_estimated_tokens: int = 0
        self._observed_request_message_count: int | None = None
        self._observed_completion_tokens: int | None = None
        self._observed_request_hash: str | None = None
        # Last provider measurement that is known to describe a committed
        # assistant response.  Unlike the latest-call calibration this is
        # deliberately retained when a later call has unknown usage.
        self._trusted_usage_anchor: dict[str, Any] | None = None
        self._pending_usage_anchor: dict[str, Any] | None = None
        self._journal_next_seq = 0
        self._journal_records_applied = 0
        self._transaction_before: dict[str, Any] | None = None
        self._poisoned = False
        if self.journal_path and self.journal_path.exists():
            existing = self.load_journal(self.journal_path)
            self._journal_next_seq = len(existing)
            self._journal_records_applied = len(existing)

    @property
    def estimated_tokens(self) -> int:
        return max(_message_tokens(self.messages), self._observed_prompt_tokens)

    def record_usage(self, prompt_tokens: int | None,
                     estimated_tokens: int | None = None,
                     *, message_count: int | None = None,
                     completion_tokens: int | None = None,
                     request_hash: str | None = None) -> None:
        """Calibrate the character estimator with provider prompt usage."""
        # A pending anchor belongs only to the current measurement.  Clear it
        # before any branch so missing or unusable usage cannot later be
        # promoted by an unrelated commit.
        self._pending_usage_anchor = None
        if prompt_tokens is None:
            # A missing current measurement must never leave stale calibration
            # eligible for a later repair request.
            self._observed_prompt_tokens = 0
            self._observed_estimated_tokens = 0
            self._observed_request_message_count = None
            self._observed_completion_tokens = None
            self._observed_request_hash = None
        elif prompt_tokens > 0:
            # Keep the latest complete request size: each turn has a
            # different current-state suffix, so a historical maximum would
            # overstate the next repair's base conversation.
            self._observed_prompt_tokens = int(prompt_tokens)
            # A missing local estimate must not leave a prior estimate
            # attached to this newer provider measurement.
            self._observed_estimated_tokens = (
                int(estimated_tokens) if estimated_tokens is not None and estimated_tokens > 0
                else 0)
            self._observed_request_message_count = (
                int(message_count) if message_count is not None and message_count >= 0
                else None)
            self._observed_completion_tokens = (
                int(completion_tokens) if completion_tokens is not None and completion_tokens >= 0
                else None)
            self._observed_request_hash = request_hash
            # Promote only a complete, hashable request measurement.  The
            # assistant correspondence is attached by commit_exchange/
            # commit_turn once the response is durably committed.
            if (_valid_hash(request_hash)
                    and estimated_tokens is not None and estimated_tokens > 0
                    and isinstance(message_count, int) and not isinstance(message_count, bool)
                    and message_count >= 0):
                self._pending_usage_anchor = {
                    "epoch": self.epoch,
                    "prompt_tokens": int(prompt_tokens),
                    "estimated_tokens": int(estimated_tokens or 0),
                    "message_count": int(message_count),
                    "completion_tokens": (int(completion_tokens)
                                           if completion_tokens is not None and completion_tokens >= 0
                                           else None),
                    "request_hash": request_hash,
                    "assistant_hash": None,
                }
        else:
            # Zero is not a usable prompt measurement; invalidate any older
            # calibration just like a missing provider usage object.
            self._observed_prompt_tokens = 0
            self._observed_estimated_tokens = 0
            self._observed_request_message_count = None
            self._observed_completion_tokens = None
            self._observed_request_hash = None
            self._pending_usage_anchor = None

    @property
    def observed_prompt_tokens(self) -> int:
        return self._observed_prompt_tokens

    @property
    def observed_estimated_tokens(self) -> int:
        return self._observed_estimated_tokens

    @property
    def observed_completion_tokens(self) -> int | None:
        """Completion tokens reported for the latest calibrated request."""
        return self._observed_completion_tokens

    @property
    def observed_request_hash(self) -> str | None:
        return self._observed_request_hash

    def estimate_prefixed_request(self, prefix_messages: Iterable[Mapping[str, Any]],
                                   appended_messages: Iterable[Mapping[str, Any]],
                                   *, guard_tokens: int = 0) -> dict[str, Any]:
        """Estimate a request while conservatively exploiting a known prefix.

        Provider usage is trusted only when it describes an exact persisted
        prefix and the local estimate of that same request agrees.  Otherwise
        the full raw estimate is used; no ratio scaling is attempted.
        """
        prefix = _copy_messages(prefix_messages)
        appended = _copy_messages(appended_messages)
        raw = _message_tokens([*prefix, *appended])
        calibration = {
            "observed_prompt_tokens": self._observed_prompt_tokens,
            "observed_estimated_tokens": self._observed_estimated_tokens,
            "observed_request_message_count": self._observed_request_message_count,
        }
        count = self._observed_request_message_count
        trusted = self._trusted_usage_anchor
        trusted_valid = self._trusted_anchor_valid()
        # Unknown usage invalidates only the latest calibration.  A trusted
        # anchor is usable only after its assistant response was committed.
        if (not self._observed_prompt_tokens or not self._observed_request_hash) and trusted_valid:
            count = int(trusted["message_count"])
        provider_tokens = self._observed_prompt_tokens if self._observed_prompt_tokens > 0 else (
            int(trusted["prompt_tokens"]) if trusted_valid else 0)
        provider_estimate = self._observed_estimated_tokens if self._observed_prompt_tokens > 0 else (
            int(trusted["estimated_tokens"]) if trusted_valid else 0)
        provider_hash = self._observed_request_hash if self._observed_prompt_tokens > 0 else (
            trusted["request_hash"] if trusted_valid else None)
        trusted_used = self._observed_prompt_tokens <= 0 and trusted_valid
        valid = (provider_tokens > 0 and
                 isinstance(count, int) and not isinstance(count, bool) and
                 0 <= len(prefix) <= count <= len(self.messages) and
                 prefix == self.messages[:len(prefix)] and
                 provider_estimate > 0 and
                 _message_tokens(self.messages[:count]) == provider_estimate and
                 isinstance(provider_hash, str) and hashlib.sha256(
                     canonical_json(self.messages[:count]).encode("utf-8")).hexdigest()
                 == provider_hash.removeprefix("sha256:"))
        if valid:
            suffix_estimate = (_safe_message_tokens(appended) if trusted_used
                               else _message_tokens(appended))
            effective = provider_tokens + suffix_estimate + max(0, int(guard_tokens))
            method = "provider_prefix"
        # A compaction prefix can include the assistant completion and its
        # locally appended execution report, even though the calibrated
        # provider request ended before those messages.  Reuse the measured
        # provider prefix only when the request identity is exact and the
        # extension has the expected committed assistant response.  This
        # avoids the large character-estimator overestimate while preserving
        # the full cacheable history.
        elif (provider_tokens > 0 and
              (self._observed_completion_tokens if self._observed_prompt_tokens > 0 else trusted.get("completion_tokens") if trusted_valid else None) is not None and
              isinstance(count, int) and not isinstance(count, bool) and
              0 <= count < len(prefix) <= len(self.messages) and
              prefix == self.messages[:len(prefix)] and
              provider_estimate > 0 and
              _message_tokens(self.messages[:count]) == provider_estimate and
              provider_hash and
              hashlib.sha256(canonical_json(self.messages[:count]).encode("utf-8")).hexdigest()
              == provider_hash.removeprefix("sha256:") and
              self.messages[count].get("role") == "assistant" and
              (not trusted_valid or trusted.get("assistant_hash") in (None, _message_hash(self.messages[count])))):
            suffix_tokens = (_safe_message_tokens(prefix[count + 1:]) if trusted_used
                             else _message_tokens(prefix[count + 1:]))
            appended_tokens = (_safe_message_tokens(appended) if trusted_used
                               else _message_tokens(appended))
            completion = self._observed_completion_tokens if self._observed_prompt_tokens > 0 else trusted["completion_tokens"]
            effective = (provider_tokens + completion + suffix_tokens +
                         appended_tokens + max(0, int(guard_tokens)))
            method = "provider_extended_prefix"
        else:
            effective = raw + max(0, int(guard_tokens))
            method = "raw"
        return {"effective": effective, "raw": raw,
                "method": method, "guard": max(0, int(guard_tokens)),
                "calibration": calibration}

    def _trusted_anchor_valid(self) -> bool:
        anchor = self._trusted_usage_anchor
        if not isinstance(anchor, dict) or anchor.get("epoch") != self.epoch:
            return False
        count = anchor.get("message_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count >= len(self.messages):
            return False
        if not _valid_hash(anchor.get("request_hash")) or not _valid_hash(anchor.get("assistant_hash")):
            return False
        if hashlib.sha256(canonical_json(self.messages[:count]).encode("utf-8")).hexdigest() != anchor["request_hash"].removeprefix("sha256:"):
            return False
        return _message_hash(self.messages[count]) == anchor["assistant_hash"]

    def _latest_calibration_valid(self) -> bool:
        count = self._observed_request_message_count
        if (self._observed_prompt_tokens <= 0 or self._observed_estimated_tokens <= 0
                or (count is not None and (not isinstance(count, int) or isinstance(count, bool)))
                or (count is not None and (count < 0 or count > len(self.messages)))
                ):
            return False
        # Pre-anchor snapshots did not persist a request hash.  Preserve
        # their conservative aggregate calibration; hash-bearing records are
        # validated strictly below.
        if self._observed_request_hash is None:
            return True
        if count is None:
            return False
        if not _valid_hash(self._observed_request_hash):
            return False
        return (_message_tokens(self.messages[:count]) == self._observed_estimated_tokens
                and _message_hash_payload(self.messages[:count]) == self._observed_request_hash)

    def import_trusted_usage_anchor(
            self, evidence: Mapping[str, Any], *,
            prior_messages: Iterable[Mapping[str, Any]]) -> None:
        """Import verified usage from a previously committed request.

        This is intentionally fail-closed for recovery from legacy snapshots:
        callers must provide the exact prior request prefix and the assistant
        response that followed it.  No provider measurement is installed if
        any epoch, hash, token, or correspondence check fails.
        """
        if not isinstance(evidence, Mapping):
            raise ConversationError("trusted usage evidence must be an object")
        try:
            epoch = evidence["epoch"]
            prompt_tokens = evidence["prompt_tokens"]
            estimated_tokens = evidence["estimated_tokens"]
            count = evidence["message_count"]
            request_hash = evidence["request_hash"]
            assistant_hash = evidence["assistant_hash"]
        except KeyError as exc:
            raise ConversationError(f"trusted usage evidence missing {exc.args[0]}") from exc
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != self.epoch):
            raise ConversationError("trusted usage evidence epoch mismatch")
        if (isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0):
            raise ConversationError("trusted usage evidence prompt_tokens must be positive")
        if (isinstance(estimated_tokens, bool) or not isinstance(estimated_tokens, int)
                or estimated_tokens <= 0):
            raise ConversationError("trusted usage evidence estimated_tokens must be positive")
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 1 or count >= len(self.messages)):
            raise ConversationError("trusted usage evidence message_count is invalid")
        if not _valid_hash(request_hash):
            raise ConversationError("trusted usage evidence request_hash is invalid")
        if not _valid_hash(assistant_hash):
            raise ConversationError("trusted usage evidence assistant_hash is invalid")
        prior = _copy_messages(prior_messages)
        if prior != self.messages[:count]:
            raise ConversationError("trusted usage evidence prefix does not match current history")
        if _message_tokens(prior) != estimated_tokens:
            raise ConversationError("trusted usage evidence local estimate mismatch")
        actual_request_hash = _message_hash_payload(prior)
        if actual_request_hash != request_hash:
            raise ConversationError("trusted usage evidence request hash mismatch")
        assistant = self.messages[count]
        if assistant.get("role") != "assistant" or _message_hash(assistant) != assistant_hash:
            raise ConversationError("trusted usage evidence assistant correspondence mismatch")
        self._trusted_usage_anchor = {
            "epoch": epoch,
            "prompt_tokens": prompt_tokens,
            "estimated_tokens": estimated_tokens,
            "message_count": count,
            "completion_tokens": (evidence.get("completion_tokens")
                                   if isinstance(evidence.get("completion_tokens"), int)
                                   and not isinstance(evidence.get("completion_tokens"), bool)
                                   and evidence.get("completion_tokens") >= 0 else None),
            "request_hash": request_hash,
            "assistant_hash": assistant_hash,
        }

    @property
    def staged(self) -> bool:
        return self._staged is not None

    def request_messages(self) -> list[dict[str, Any]]:
        """Return a detached copy suitable for an SDK request."""
        return _copy_messages(self.messages)

    def begin_turn(self, user_message: Mapping[str, Any], *, projected_tokens: int = 0) -> list[dict[str, Any]]:
        if self.staged:
            raise ConversationError("a conversation turn is already staged")
        current = _copy_messages([user_message])[0]
        candidate = self.messages + [current]
        # Provider prompt usage describes the committed prefix from the last
        # request. Add only the newly appended user message here; counting the
        # whole candidate again would double-count the prefix and hide the
        # true compaction boundary.
        estimated_current = _message_tokens(self.messages)
        if self._latest_calibration_valid():
            # Reconcile provider count for the last request with committed
            # messages added after that request (assistant/report).  The API's
            # completion_tokens is authoritative for the assistant response;
            # estimate only an execution report or other locally appended
            # suffix, avoiding a chars/4 estimate of long reasoning output.
            suffix = (self.messages[self._observed_request_message_count:]
                      if self._observed_request_message_count is not None
                      and len(self.messages) >= self._observed_request_message_count
                      else [])
            if suffix and self._observed_completion_tokens is not None:
                suffix_tokens = _message_tokens(suffix)
                if suffix[0].get("role") == "assistant":
                    suffix_tokens -= _message_tokens([suffix[0]])
                    suffix_tokens += self._observed_completion_tokens
                base_tokens = self._observed_prompt_tokens + max(0, suffix_tokens)
            elif self._observed_request_message_count is None:
                # Backward-compatible calibration for callers that only have
                # aggregate estimates and cannot identify the response suffix.
                base_tokens = self._observed_prompt_tokens + max(
                    0, estimated_current - self._observed_estimated_tokens)
            else:
                base_tokens = self._observed_prompt_tokens
        else:
            base_tokens = max(estimated_current, self._observed_prompt_tokens)
        additional_tokens = _message_tokens([current]) + max(0, projected_tokens)
        if base_tokens + additional_tokens >= self.config.trigger_tokens:
            raise CompactionRequired(
                f"conversation projected context exceeds {self.config.trigger_tokens} tokens")
        self._staged = candidate
        return _copy_messages(candidate)

    def stage_assistant(self, content: Any) -> None:
        if not self.staged:
            raise ConversationError("stage_assistant requires begin_turn")
        message = {"role": "assistant", "content": content}
        self._staged_assistant = _copy_messages([message])[0]

    def commit_turn(self, *, execution_report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._staged is None or self._staged_assistant is None:
            raise ConversationError("cannot commit incomplete conversation turn")
        old_len = len(self.messages)
        candidate = self._staged + [self._staged_assistant]
        if execution_report is not None:
            candidate.append(_copy_messages([{"role": "user", "content":
                "[EXECUTION REPORT]\n" + canonical_json(dict(execution_report))}])[0])
        _validate_messages(candidate)
        self._begin_transaction()
        self.messages = candidate
        self._refresh_trusted_anchor()
        self._staged = None
        self._staged_assistant = None
        seq = self._append_journal(self.messages[old_len:])
        self._persist_committed(seq)
        return self.request_messages()

    def append_execution_report(self, execution_report: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Record engine-only progress when no assistant response was usable.

        A degraded fallback turn must remain visible to the next model call,
        but it must not invent an assistant message that Mistral never
        returned.  The report is a normal, labeled user message and is
        durably journaled as its own incomplete-turn record.
        """
        if self.staged:
            raise ConversationError("cannot append an execution report mid-turn")
        # A provider measurement from a failed/uncommitted response must not
        # later be associated with an unrelated assistant at the same index.
        before = len(self.messages)
        self._begin_transaction()
        self._pending_usage_anchor = None
        self._append_state_message(execution_report, engine_only=True)
        seq = self._append_journal(self.messages[before:], kind="execution_report")
        self._persist_committed(seq)
        return self.request_messages()

    def commit_exchange(self, messages: Iterable[Mapping[str, Any]], *,
                        execution_report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Commit a completed exchange, including a repair sequence.

        ``messages`` must contain the staged user message and any complete
        assistant/repair messages.  This keeps the manager transactional
        while preserving the exact assistant thinking chunks required by the
        provider for subsequent cache-friendly requests.
        """
        if self._staged is None:
            raise ConversationError("commit_exchange requires begin_turn")
        committed = _copy_messages(messages)
        if committed[:len(self._staged)] != self._staged:
            raise ConversationError("exchange does not match staged turn")
        if len(committed) <= len(self._staged):
            raise ConversationError("cannot commit exchange without assistant response")
        if committed[-1].get("role") != "assistant":
            raise ConversationError("exchange must end with assistant response")
        old_len = len(self.messages)
        if execution_report is not None:
            committed.append(_copy_messages([{"role": "user", "content":
                "[EXECUTION REPORT]\n" + canonical_json(dict(execution_report))}])[0])
        _validate_messages(committed)
        self._begin_transaction()
        self.messages = committed
        self._refresh_trusted_anchor()
        self._staged = None
        self._staged_assistant = None
        seq = self._append_journal(self.messages[old_len:])
        self._persist_committed(seq)
        return self.request_messages()

    def abort_turn(self) -> None:
        self._staged = None
        self._staged_assistant = None
        self._pending_usage_anchor = None

    def compact(self, compactor: Callable[[list[dict[str, Any]]], Mapping[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
        """Compact only at a transaction boundary; failure leaves state intact."""
        if self.staged:
            raise ConversationError("cannot compact during a turn")
        if self.estimated_tokens < self.config.trigger_tokens and not force:
            return self.request_messages()
        tail_groups: list[list[dict[str, Any]]] = []
        # # Select the exact retained tail before summarizing.  The compactor
        # must never see messages that remain verbatim after the summary.
        used = _message_tokens([self.messages[0]])
        groups = self._turn_groups()
        for group in reversed(groups):
            cost = _message_tokens(group)
            if used + cost > self.config.tail_tokens:
                break
            tail_groups.insert(0, group)
            used += cost
        tail = [message for group in tail_groups for message in group]
        evicted = [self.messages[0], *self.messages[1:len(self.messages) - len(tail)]]
        summary = compactor(_copy_messages(evicted))
        if not isinstance(summary, Mapping):
            raise ConversationError("compactor must return a JSON object")
        summary = _validate_compaction_summary(summary)
        summary_payload = {"type": "conversation_compaction",
                           "policy_version": self.config.policy_version,
                           "epoch": self.epoch + 1,
                           "summary": dict(summary)}
        summary_message = _copy_messages([{"role": "user", "content":
                           "[CONVERSATION COMPACTION]\n" + canonical_json(summary_payload)}])[0]
        self._begin_transaction()
        self.messages = [self.messages[0], summary_message, *tail]
        self.epoch += 1
        # The retained history is a new cache generation; any prior provider
        # anchor describes the old epoch and must not survive rotation.
        self._trusted_usage_anchor = None
        self._pending_usage_anchor = None
        self._observed_prompt_tokens = _message_tokens(self.messages)
        self._observed_estimated_tokens = self._observed_prompt_tokens
        self._observed_request_message_count = None
        self._observed_completion_tokens = None
        self._observed_request_hash = None
        if self.prompt_cache_key:
            # A compacted prefix is a new cache generation.  Keep the key
            # stable within an epoch, but never mix cache entries whose
            # retained message prefix differs.
            self.prompt_cache_key = f"{self.prompt_cache_key}:epoch-{self.epoch}"
        seq = self._append_journal([summary_message, *tail], kind="conversation_compaction")
        self._persist_committed(seq)
        return self.request_messages()

    def _turn_groups(self) -> list[list[dict[str, Any]]]:
        """Split history into complete user/assistant turn groups.

        An execution report is a user message attached to the preceding
        assistant response.  Compaction never retains half of such a group.
        """
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        pending: list[str] | None = None
        awaiting_final = False
        def completed(group: list[dict[str, Any]]) -> bool:
            """Whether a group has a complete assistant-side outcome."""
            if len(group) < 2 or group[0].get("role") != "user" or \
                    group[1].get("role") != "assistant":
                return False
            last = group[-1]
            if last.get("role") == "assistant":
                return True
            return (last.get("role") == "user" and
                    isinstance(last.get("content"), str) and
                    last["content"].startswith("[EXECUTION REPORT]"))
        start = 2 if len(self.messages) > 1 and isinstance(
            self.messages[1].get("content"), str) and self.messages[1]["content"].startswith(
                "[CONVERSATION COMPACTION]") else 1
        for message in self.messages[start:]:
            role = message.get("role")
            content = message.get("content")
            report = isinstance(content, str) and content.startswith("[EXECUTION REPORT]")
            engine_only = isinstance(content, str) and content.startswith("[ENGINE-ONLY REPORT]")
            if role == "user" and engine_only:
                if current:
                    if completed(current) and pending is None and not awaiting_final:
                        groups.append(current)
                    else:
                        raise ConversationError("conversation contains incomplete turn group")
                    current = []
                groups.append([message])
                continue
            if role == "user" and not report:
                if pending is not None or awaiting_final:
                    raise ConversationError("conversation contains incomplete tool transaction")
                if current:
                    if current[-1].get("role") == "assistant":
                        # A second user message immediately after an
                        # assistant is the repair request for this turn.
                        current.append(message)
                        continue
                    if not completed(current):
                        raise ConversationError("conversation contains incomplete turn group")
                    groups.append(current)
                current = [message]
            elif role == "assistant" and current:
                calls = message.get("tool_calls")
                if calls is not None:
                    if not isinstance(calls, list) or not calls or pending is not None or awaiting_final:
                        raise ConversationError("conversation contains invalid tool calls")
                    ids = [call.get("id") for call in calls
                           if isinstance(call, Mapping) and isinstance(call.get("id"), str)]
                    if len(ids) != len(calls) or len(set(ids)) != len(ids):
                        raise ConversationError("conversation contains invalid tool call ids")
                    pending = ids
                elif awaiting_final:
                    awaiting_final = False
                elif pending is not None:
                    raise ConversationError("conversation contains incomplete tool transaction")
                current.append(message)
            elif role == "tool" and current:
                tool_id = message.get("tool_call_id")
                if pending is None or not isinstance(tool_id, str) or tool_id not in pending:
                    raise ConversationError("conversation contains orphan tool result")
                pending.remove(tool_id)
                if not pending:
                    pending = None
                    awaiting_final = True
                current.append(message)
            elif role == "user" and report and current and current[-1].get("role") == "assistant":
                if pending is not None or awaiting_final:
                    raise ConversationError("conversation contains incomplete tool transaction")
                current.append(message)
            else:
                raise ConversationError("conversation contains invalid role ordering")
        if current:
            if pending is not None or awaiting_final:
                raise ConversationError("conversation contains incomplete tool transaction")
            if not completed(current):
                raise ConversationError("conversation contains incomplete turn group")
            groups.append(current)
        return groups

    def _append_state_message(self, report: Mapping[str, Any], *, engine_only: bool = False) -> None:
        prefix = "[ENGINE-ONLY REPORT]" if engine_only else "[EXECUTION REPORT]"
        safe = _copy_messages([{"role": "user", "content":
                                prefix + "\n" + canonical_json(dict(report))}])[0]
        self.messages.append(safe)

    def _begin_transaction(self) -> None:
        if self._poisoned:
            raise ConversationPersistenceError(
                "conversation persistence is poisoned; reload and reconcile the WAL")
        if self._transaction_before is not None:
            raise ConversationError("conversation transaction already active")
        self._transaction_before = {
            "messages": _copy_messages(self.messages), "epoch": self.epoch,
            "prompt_cache_key": self.prompt_cache_key,
            "observed_prompt_tokens": self._observed_prompt_tokens,
            "observed_estimated_tokens": self._observed_estimated_tokens,
            "observed_request_message_count": self._observed_request_message_count,
            "observed_completion_tokens": self._observed_completion_tokens,
            "observed_request_hash": self._observed_request_hash,
            "trusted_usage_anchor": (dict(self._trusted_usage_anchor)
                                      if self._trusted_usage_anchor is not None else None),
            "pending_usage_anchor": (dict(self._pending_usage_anchor)
                                      if self._pending_usage_anchor is not None else None),
            "journal_next_seq": self._journal_next_seq,
            "journal_records_applied": self._journal_records_applied,
            "staged": _copy_messages(self._staged) if self._staged is not None else None,
            "staged_assistant": (_copy_messages([self._staged_assistant])[0]
                                 if self._staged_assistant is not None else None),
        }

    def _rollback_transaction(self) -> None:
        before = self._transaction_before
        if before is None:
            return
        self.messages = before["messages"]
        self.epoch = before["epoch"]
        self.prompt_cache_key = before["prompt_cache_key"]
        for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                      "observed_request_message_count", "observed_completion_tokens",
                      "observed_request_hash"):
            setattr(self, "_" + field, before[field])
        self._trusted_usage_anchor = before["trusted_usage_anchor"]
        self._pending_usage_anchor = before["pending_usage_anchor"]
        self._journal_next_seq = before["journal_next_seq"]
        self._journal_records_applied = before["journal_records_applied"]
        self._staged = before["staged"]
        self._staged_assistant = before["staged_assistant"]
        self._transaction_before = None

    def _finish_transaction(self) -> None:
        self._transaction_before = None

    def _refresh_trusted_anchor(self) -> None:
        anchor = self._pending_usage_anchor
        if not isinstance(anchor, dict):
            return
        count = anchor.get("message_count")
        if (anchor.get("epoch") == self.epoch and isinstance(count, int)
                and 0 <= count < len(self.messages)
                and self.messages[count].get("role") == "assistant"):
            anchor["assistant_hash"] = _message_hash(self.messages[count])
            self._trusted_usage_anchor = anchor
            self._pending_usage_anchor = None

    def _append_journal(self, new_messages: list[dict[str, Any]], *, kind: str = "conversation_turn") -> int:
        if self.journal_path is None:
            return self._journal_records_applied
        write_started = False
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_exists = self.journal_path.exists()
            seq = self._journal_next_seq
            payload = {"kind": kind, "seq": seq,
                       "epoch": self.epoch, "messages": new_messages,
                       "prompt_cache_key": self.prompt_cache_key,
                       "observed_prompt_tokens": self._observed_prompt_tokens,
                       "observed_estimated_tokens": self._observed_estimated_tokens,
                       "observed_request_message_count": self._observed_request_message_count,
                       "observed_completion_tokens": self._observed_completion_tokens,
                       "observed_request_hash": self._observed_request_hash,
                       "trusted_usage_anchor": self._trusted_usage_anchor}
            payload["checksum"] = hashlib.sha256(
                canonical_json({k: v for k, v in payload.items() if k != "checksum"}).encode("utf-8")
            ).hexdigest()
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                write_started = True
                fh.write(canonical_json(payload) + "\n")
                fh.flush(); os.fsync(fh.fileno())
            if not journal_exists:
                _fsync_parent_directory(self.journal_path)
        except Exception as exc:
            if write_started:
                # The bytes may be durable even when fsync returned an error;
                # never retry and risk a duplicate sequence in this process.
                self._poisoned = True
                self._transaction_before = None
                raise ConversationPersistenceError(
                    "journal write durability is ambiguous; reload and reconcile the WAL"
                ) from exc
            self._rollback_transaction()
            raise
        self._journal_next_seq += 1
        self._journal_records_applied = self._journal_next_seq
        return self._journal_next_seq

    @staticmethod
    def load_journal(path: str | Path) -> list[dict[str, Any]]:
        """Load completed transaction snapshots, rejecting any corruption."""
        file_path = Path(path)
        if not file_path.exists():
            return []
        records: list[dict[str, Any]] = []
        expected_seq = 0
        previous_epoch = 0
        raw_lines = file_path.read_bytes().split(b"\n")
        for index, raw_line in enumerate(raw_lines, 1):
            is_unterminated_tail = index == len(raw_lines) and raw_line
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                checksum = record.pop("checksum")
                expected = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
                if checksum != expected or record.get("kind") not in {
                        "conversation_turn", "conversation_compaction",
                        "execution_report", "conversation_base"}:
                    raise ValueError("checksum or record kind mismatch")
                if not isinstance(record.get("messages"), list):
                    raise ValueError("messages missing")
                if record.get("seq") != expected_seq:
                    raise ValueError("journal sequence is not contiguous")
                if record.get("kind") == "conversation_base" and expected_seq != 0:
                    raise ValueError("conversation base must be sequence zero")
                if int(record.get("epoch", 0)) < previous_epoch:
                    raise ValueError("journal epoch regressed")
                expected_seq += 1
                previous_epoch = int(record.get("epoch", 0))
                records.append(record)
            except Exception as exc:
                if is_unterminated_tail:
                    raise ConversationError(
                        f"unterminated trailing conversation journal record {index}; "
                        "refusing to append after an ambiguous WAL tail")
                raise ConversationError(f"invalid conversation journal record {index}") from exc
        return records

    @classmethod
    def reconstruct(cls, system_message: Mapping[str, Any], records: list[Mapping[str, Any]], *,
                    config: ConversationConfig | None = None) -> "ConversationManager":
        """Rebuild a conversation from transaction journal records."""
        obj = cls(system_message=system_message, config=config)
        expected_seq = 0
        previous_epoch = 0
        for record in records:
            if record.get("seq") != expected_seq:
                raise ConversationError("journal sequence is not contiguous")
            epoch = int(record.get("epoch", 0))
            if epoch < previous_epoch:
                raise ConversationError("journal epoch regressed")
            expected_seq += 1
            previous_epoch = epoch
            messages = _copy_messages(record.get("messages", []))
            if record.get("kind") == "conversation_base":
                if expected_seq != 1:
                    raise ConversationError("conversation base must be sequence zero")
                obj.messages = messages
            elif record.get("kind") == "conversation_compaction":
                obj.messages = [obj.messages[0], *messages]
                obj.epoch = int(record.get("epoch", obj.epoch))
            else:
                obj.messages.extend(messages)
            _validate_messages(obj.messages)
            obj.epoch = int(record.get("epoch", obj.epoch))
            obj.prompt_cache_key = record.get("prompt_cache_key", obj.prompt_cache_key)
            for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                          "observed_request_message_count", "observed_completion_tokens",
                          "observed_request_hash"):
                if field in record:
                    setattr(obj, "_" + field, record[field])
            if "trusted_usage_anchor" in record:
                obj._trusted_usage_anchor = record["trusted_usage_anchor"]
            obj._journal_records_applied += 1
        if obj._trusted_usage_anchor is not None and not _valid_anchor_schema(obj._trusted_usage_anchor):
            raise ConversationError("invalid trusted usage anchor in journal")
        return obj
    def _persist(self, *, journal_records_applied: int | None = None) -> None:
        if self.state_path is None:
            return
        self._persist_to(self.state_path, journal_records_applied=(
            self._journal_records_applied if journal_records_applied is None
            else journal_records_applied))

    def _state_payload(self, *, journal_records_applied: int) -> dict[str, Any]:
        payload = {"version": 1, "policy_version": self.config.policy_version,
                   "epoch": self.epoch, "messages": self.messages,
                   "prompt_cache_key": self.prompt_cache_key,
                   "journal_records_applied": journal_records_applied,
                   # Preserve provider calibration across a continuation.
                   # These values are advisory and are validated below; the
                   # message stream remains the source of truth when absent
                   # in older state files.
                   "observed_prompt_tokens": self._observed_prompt_tokens,
                   "observed_estimated_tokens": self._observed_estimated_tokens,
                   "observed_request_message_count": self._observed_request_message_count,
                   "observed_completion_tokens": self._observed_completion_tokens,
                   "observed_request_hash": self._observed_request_hash,
                   "trusted_usage_anchor": self._trusted_usage_anchor}
        payload["checksum"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return payload

    def _persist_to(self, path: str | Path, *, journal_records_applied: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._state_payload(journal_records_applied=journal_records_applied)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(payload) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_parent_directory(path)

    def export_standalone(self, path: str | Path) -> None:
        """Write a checksum-verified snapshot independent of any WAL."""
        if self._transaction_before is not None or self._poisoned:
            raise ConversationPersistenceError("cannot export an active or poisoned conversation")
        self._persist_to(path, journal_records_applied=0)

    def fork_persistence(self, new_state: str | Path, new_journal: str | Path) -> None:
        """Create a durable child lineage with a self-contained base record."""
        if self._transaction_before is not None or self.staged:
            raise ConversationError("cannot fork a staged conversation")
        if self._poisoned:
            raise ConversationPersistenceError("cannot fork a poisoned conversation")
        state_path, journal_path = Path(new_state), Path(new_journal)
        if journal_path.name != "conversation.jsonl" or journal_path != state_path.with_name("conversation.jsonl"):
            raise ConversationError("fork journal must be the conversation.jsonl sibling of child state")
        # A fork is a new lineage, never an overwrite operation.  Check all
        # destinations before creating either artifact so a failed fork
        # cannot damage an existing child or parent-owned files.
        if state_path.exists() or journal_path.exists():
            raise ConversationError("fork destinations already exist")
        payload = {"kind": "conversation_base", "seq": 0,
                   "epoch": self.epoch, "messages": _copy_messages(self.messages),
                   "prompt_cache_key": self.prompt_cache_key,
                   "observed_prompt_tokens": self._observed_prompt_tokens,
                   "observed_estimated_tokens": self._observed_estimated_tokens,
                   "observed_request_message_count": self._observed_request_message_count,
                   "observed_completion_tokens": self._observed_completion_tokens,
                   "observed_request_hash": self._observed_request_hash,
                   "trusted_usage_anchor": self._trusted_usage_anchor}
        payload["checksum"] = hashlib.sha256(
            canonical_json({k: v for k, v in payload.items() if k != "checksum"}).encode("utf-8")
        ).hexdigest()
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{journal_path.name}.fork-",
                                         suffix=".tmp", dir=journal_path.parent)
        tmp_journal = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(canonical_json(payload) + "\n")
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp_journal, journal_path)
            _fsync_parent_directory(journal_path)
        except Exception:
            try:
                tmp_journal.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        try:
            self._persist_to(state_path, journal_records_applied=1)
        except Exception as exc:
            raise ConversationPersistenceError(
                "fork journal is durable but child state publication failed; "
                "orphan child lineage requires manual reconciliation") from exc
        self.state_path, self.journal_path = state_path, journal_path
        self._journal_next_seq = self._journal_records_applied = 1

    def _persist_committed(self, journal_records_applied: int) -> None:
        """Publish a WAL-backed transition or raise a distinct fatal error."""
        try:
            self._persist(journal_records_applied=journal_records_applied)
        except Exception as exc:
            self._finish_transaction()
            self._poisoned = True
            raise ConversationPersistenceError(
                "conversation WAL committed but state snapshot publication failed"
            ) from exc
        self._finish_transaction()

    @classmethod
    def load(cls, path: str | Path, *, config: ConversationConfig | None = None) -> "ConversationManager":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("messages"), list):
            raise ConversationError("invalid conversation state")
        checksum = data.pop("checksum", None)
        if not isinstance(checksum, str) or checksum != hashlib.sha256(
                canonical_json(data).encode("utf-8")).hexdigest():
            raise ConversationError("conversation state checksum mismatch")
        requested = config.policy_version if config else data.get("policy_version")
        if data.get("policy_version") != requested:
            raise ConversationError("conversation state policy mismatch")
        messages = _copy_messages(data["messages"])
        _validate_messages(messages)
        journal_path = Path(path).with_name("conversation.jsonl")
        existing_records = cls.load_journal(journal_path) if journal_path.exists() else []
        applied_marker = data.get("journal_records_applied")
        if applied_marker is None:
            # Older snapshots may predate the WAL marker.  Migrate only when
            # the complete journal reconstructs exactly the stored snapshot;
            # otherwise refuse to guess which side of a failed write won.
            if existing_records:
                reconstructed = cls.reconstruct(messages[0], existing_records,
                                                 config=config)
                if (reconstructed.messages != messages or
                        reconstructed.epoch != int(data.get("epoch", 0)) or
                        reconstructed.prompt_cache_key != data.get("prompt_cache_key") or
                        any(getattr(reconstructed, "_" + field) != data.get(field, {
                                "observed_prompt_tokens": 0,
                                "observed_estimated_tokens": 0,
                                "observed_request_message_count": None,
                                "observed_completion_tokens": None,
                                "observed_request_hash": None,
                            }[field]) for field in
                            ("observed_prompt_tokens", "observed_estimated_tokens",
                             "observed_request_message_count", "observed_completion_tokens",
                             "observed_request_hash"))):
                    raise ConversationError("legacy conversation snapshot does not match journal")
                applied_marker = len(existing_records)
            else:
                applied_marker = 0
        if (not isinstance(applied_marker, int) or isinstance(applied_marker, bool)
                or applied_marker < 0 or applied_marker > len(existing_records)):
            raise ConversationError("invalid journal application marker")
        obj = cls(system_message=messages[0], config=config,
                  state_path=path, journal_path=journal_path if journal_path.exists() else None,
                  prompt_cache_key=data.get("prompt_cache_key"))
        obj.messages = messages
        obj.epoch = int(data.get("epoch", 0))
        for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                      "observed_request_message_count", "observed_completion_tokens"):
            value = data.get(field)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ConversationError(f"invalid conversation calibration: {field}")
        request_hash = data.get("observed_request_hash")
        if request_hash is not None and not _valid_hash(request_hash):
            raise ConversationError("invalid conversation calibration: observed_request_hash")
        obj._observed_prompt_tokens = int(data.get("observed_prompt_tokens", 0) or 0)
        obj._observed_estimated_tokens = int(data.get("observed_estimated_tokens", 0) or 0)
        obj._observed_request_message_count = data.get("observed_request_message_count")
        obj._observed_completion_tokens = data.get("observed_completion_tokens")
        obj._observed_request_hash = data.get("observed_request_hash")
        anchor = data.get("trusted_usage_anchor")
        if anchor is not None and not _valid_anchor_schema(anchor):
            raise ConversationError("invalid trusted usage anchor")
        obj._trusted_usage_anchor = anchor
        obj._journal_next_seq = len(existing_records)
        obj._journal_records_applied = applied_marker
        if applied_marker:
            prefix = cls.reconstruct(messages[0], existing_records[:applied_marker], config=config)
            if (prefix.messages != obj.messages or prefix.epoch != obj.epoch or
                    prefix.prompt_cache_key != obj.prompt_cache_key or
                    any(getattr(prefix, "_" + field) != getattr(obj, "_" + field)
                        for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                                       "observed_request_message_count", "observed_completion_tokens",
                                       "observed_request_hash"))):
                raise ConversationError("conversation snapshot does not match applied journal prefix")
            if prefix._trusted_usage_anchor != obj._trusted_usage_anchor:
                raise ConversationError("conversation snapshot trusted anchor mismatch")
        for record in existing_records[applied_marker:]:
            record_messages = _copy_messages(record.get("messages", []))
            if record.get("kind") == "conversation_base":
                if obj._journal_records_applied != 0:
                    raise ConversationError("conversation base must be the first journal record")
                obj.messages = record_messages
            elif record.get("kind") == "conversation_compaction":
                obj.messages = [obj.messages[0], *record_messages]
            else:
                obj.messages.extend(record_messages)
            _validate_messages(obj.messages)
            obj.epoch = int(record.get("epoch", obj.epoch))
            obj.prompt_cache_key = record.get("prompt_cache_key", obj.prompt_cache_key)
            for field in ("observed_prompt_tokens", "observed_estimated_tokens",
                          "observed_request_message_count", "observed_completion_tokens",
                          "observed_request_hash"):
                if field in record:
                    setattr(obj, "_" + field, record[field])
            if "trusted_usage_anchor" in record:
                obj._trusted_usage_anchor = record["trusted_usage_anchor"]
            obj._journal_records_applied += 1
        if obj._trusted_usage_anchor is not None and not _valid_anchor_schema(obj._trusted_usage_anchor):
            raise ConversationError("invalid trusted usage anchor in journal")
        return obj
