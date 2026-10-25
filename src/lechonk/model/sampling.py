# SPDX-License-Identifier: MIT
"""The ACTUAL arm sampling persisted in run manifests (decision 4).

W1-6 (sampling manifest truth): the evaluation drivers used to persist a
fabricated sampling dict — ``{"temperature": 0.0, ..., "top_p": 1.0,
"random_seed": None, ...}`` — that the client never ran. The client's real
arm sampling (decision 4, 2026-10-24 amendment) is temperature 1.0 with
``reasoning_effort="high"``; the client NEVER sends ``top_p`` and never
sets a random seed. ``actual_client_sampling`` returns that actual
sampling, shared by all four drivers, so every manifest records what
really ran.
"""

from lechonk.model.mistral_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
)


def actual_client_sampling() -> dict:
    """The sampling the MistralClient ACTUALLY runs on every call (turn
    calls, repairs, and capability probes alike — decision 4, 2026-10-24
    amendment): temperature 1.0, reasoning_effort "high", the default
    output budget, and the strict json_schema response_format.
    ``top_p`` is never sent by the client and is never recorded here;
    ``random_seed`` is absent (the client sends none)."""
    return {
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "response_format": "json_schema strict (WP2 spike decision)",
    }
