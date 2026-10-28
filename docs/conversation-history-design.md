# Append-only Conversation History and Compaction

**Status:** implemented; long-game validation pending
**Version:** conversation-v1
**Scope:** interactive Mistral runs. Legacy evaluation callers keep their existing two-message prompt contract.

## Objective

Make interactive play behave like a coding harness: preserve the exact ordered
conversation, including Mistral reasoning chunks, so every turn can benefit from
the cached common prefix and retain decisions that are not visible in the latest
map. Keep a separate, small state file for durable strategic facts. Compact the
conversation at a deterministic 500,000 estimated-token trigger so it cannot grow
without bound.

## Core rules

- The system prompt and immutable protocol prefix are created once per cache
  epoch and never rewritten within that epoch.
- Each completed turn appends the exact user message and exact assistant response
  content, including nested `thinking` and final text chunks. A repair appends its
  failed assistant response, repair user message, and repaired assistant response.
- A response is staged until action execution, waiting, terminal classification,
  and memory advancement finish. Transport failure, truncation, or an aborted
  turn appends no assistant message and cannot claim planned actions ran.
- The next user message contains the prior turn's actual execution report and a
  new authoritative fog-gated snapshot, delta, and candidate table.
- Current state and legal candidates override historical conversation text. The
  state file is a summary, not a source of current visibility.
- `legacy` remains the default library/evaluation policy. The interactive
  launcher selects `conversation_v1`; `hybrid_v1` is superseded and retained
  only for reading older artifacts.

## Message and state storage

Conversation state is local and append-only. Each committed transaction records
the turn, ordered messages, response/chunk hash, prompt-policy version, cache
epoch, and transcript sequence. Records are checksummed and fsynced before the
next model request. A torn final line may be discarded; a complete checksum
failure or policy mismatch fails closed.

The state file is versioned and atomic. It contains the model-authored strategic
summary, unresolved commitments, uncertainties, compaction epoch, and source
message hash. It never replaces the current observation. The interactive runner
also binds a verified start-of-turn Freeciv save, immutable conversation snapshot,
and committed strategy memory into a checksummed recovery manifest. `--resume`
stages those artifacts into a new child run and verifies configuration, username,
player, and turn identity before model work; parent artifacts remain immutable.

## Compaction policy

Compaction occurs only between completed turns, never during a repair or while
actions are unresolved. Before appending the next user turn, estimate the full
conversation plus the pending prompt. When it reaches 500,000 tokens, issue one
bounded structured compaction request over the existing conversation. The
compactor produces a versioned summary/state record; it labels historical claims
and uncertainties and preserves commitments and execution outcomes.

After successful compaction, start a new cache epoch with the stable
system/protocol prefix, the bounded state-summary message and source hash, an
exact recent tail of approximately 50,000–100,000 tokens ending at a completed
turn, and the next authoritative current-state user message.

The old conversation is retained as an artifact. Compaction never silently
truncates, splits a thinking chunk, or leaves an assistant message without its
paired user message. If the compactor fails after its bounded retry, the run
fails closed with durable diagnostics rather than sending a request likely to
exceed context.

The 500,000-token value is an application trigger, not a Mistral model limit.
Token estimates are conservative and leave headroom for the next response,
repair, and estimator error. Provider-reported prompt usage is recorded when
available.

## Caching and observability

Each run/epoch uses a non-secret stable `prompt_cache_key`. The key changes only
after compaction. The transport sends it for complete and streaming calls when
supported, and records `usage.prompt_tokens_details.cached_tokens` alongside
ordinary input/output usage. Cache hits are provider-managed and are measured,
not assumed. Cached input reduces repeated input cost and work; it does not
increase context capacity.

## Compatibility and safety

The strict `ActionBatch` schema and candidate-only execution protocol do not
change. Raw prompts, API keys, and unbounded malformed responses are not copied
into the state file. Full traces remain audit artifacts. Historical model text is
data, not instructions; current fog-gated sections are authoritative.

Legacy prompts and request shapes remain byte-compatible for evaluation. New
conversation artifacts carry explicit policy, schema, epoch, token-trigger, and
source-hash metadata so replay can reject incompatible state.

## Implementation phases

1. Add a transactional conversation store with exact message/chunk preservation,
   token accounting, cache epochs, checksummed persistence, and state-file I/O.
2. Add conversation-aware turn-client and transport support, including repairs,
   cache key, cached-token accounting, and model-authored compaction.
3. Integrate runner commit/abort at the completed-turn boundary and pass actual
   execution reports before the next current-state prompt.
4. Make `conversation_v1` the interactive policy and retain `legacy` for
   evaluation; document artifacts and checked child continuation.
5. Test exact prefix reuse, reasoning replay, repair transactions, compaction
   trigger/failure, fog/current-state precedence, torn records, and bounded
   long-run growth; then run a short live multi-turn characterization.

## Risks and open decisions

- Full reasoning history grows quickly even when cached; the 500,000 trigger and
  50,000–100,000-token tail must be measured against actual context usage.
- A model-authored summary can omit a strategically important fact. Preserve the
  recent exact tail and state uncertainties explicitly; summary generation is not
  lossless.
- A recovery manifest is valid only with its matching verified save, conversation
  snapshot, committed memory, configuration digest, and distinct child run.
- The configured Freeciv ping timeout is applied during setup and verified by a
  typed server-setting readback. The confirmed value/status are persisted in
  runtime metadata. There is no continuous heartbeat while model generation
  blocks the worker, and no sustained live call beyond the configured timeout
  has been tested, so that liveness risk remains open.
- Cache TTL and hit guarantees are provider-managed; cached-token telemetry is
  required before tuning the history policy.

## Implementation and validation record

The conversation store, exact thinking-chunk replay, transactional turn
commit/abort, cache-key and cached-token instrumentation, state-file persistence,
compaction path, and checked child-run recovery are implemented. Retry handling
no longer drops caller flags. The tracked CivRealm patch applies the configured
ping timeout during fresh and loaded setup paths, and startup verifies the typed
server-setting readback. Fresh and resumed live launches passed that check at
the expected effective value of 1260 seconds for the default 1200-second acting
deadline. The launch fast path also validates the pinned image, vendor
gitlink/head, and tracked patch before reuse.
The legacy evaluation policy remains isolated from `conversation_v1`.

Synthetic validation completed two turns with reasoning replay and cache-prefix
reuse, then forced compaction and verified the new epoch, bounded exact tail, and
state record. The 500,000-token trigger is partly calibrated using the project's
conservative character-based estimate; a provider-token long-run measurement is
still pending.

A live Freeciv characterization completed two model turns using conversation
history and caching. A fresh parent/child recovery run then loaded the verified
save and completed one child model turn with matching player/turn identity. A
longer sustained-latency game remains useful follow-up validation: there is no
continuous heartbeat during model generation, and no call longer than the
configured timeout has yet been tested.
