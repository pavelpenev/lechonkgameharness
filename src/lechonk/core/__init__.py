# SPDX-License-Identifier: MIT
"""lechonk.core — MIT, CivRealm-free, testable offline.

This package must NEVER import civrealm (enforced by tests/test_core_purity.py).
The runner (lechonk.core.runner) is the composition root and is imported
directly; it pulls in lechonk.comparator (also CivRealm-free) but never the
adapter at module level (the adapter import is lazy, only after the Docker
stack is up).
"""

from lechonk.core import delta, ledger, memory, metrics, model_client, schemas

__all__ = ["delta", "ledger", "memory", "metrics", "model_client", "schemas"]
