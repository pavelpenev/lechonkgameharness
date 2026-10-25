# SPDX-License-Identifier: MIT
"""lechonk.model — the model package (MIT, plan §7.1).

- ``mistral_client.py`` — Mistral Studio API client for the exact model ID
  ``mistral-large-4`` (decision 4): per-turn attempt budget (≤2 total API
  attempts including at most one repair; SDK/transport retries transparent),
  per-call logging (tokens, latency, cost, fingerprint data), both output
  enforcement paths (``json_schema`` vs schema-validated parsing — the WP2
  spike decides the default), and the per-arm model fingerprint probe
  (decision 8).
- ``prompts.py`` + ``prompts/`` — versioned prompt template FILES (§7.5) with
  a deterministic hash function; the run manifest records their hashes.

This package never imports civrealm (MIT; the adapter is the only civrealm
importer, plan §7.1).
"""
