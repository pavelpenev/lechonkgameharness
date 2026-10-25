# SPDX-License-Identifier: MIT
"""lechonk.eval — WP3 evaluation plumbing (MIT; CivRealm-free).

design-plan-freeciv-v1.1.md §7.1 package layout: ``eval/`` owns the
paired-run plumbing (paired.py), the JSONL → static-HTML replay viewer
(replay.py), and the checkpoint export/preserve path (checkpoint.py).

MIT (decision 7): this package NEVER imports civrealm — not at module
level and not transitively. The docker-cp checkpoint orchestration talks
to the Docker daemon over subprocess only; the paired-run plumbing drives
arms as subprocesses and consumes the frozen core schemas. The only
adapter touchpoint is a LAZY, function-level import of
``lechonk.adapter.obs_extract`` (itself CivRealm-free) inside
``paired.ScriptedComparatorClient.complete_turn`` — the same lazy-import
discipline the runner uses (§7.1: the adapter import happens only after
the Docker stack is up). Test-enforced in tests/test_eval_paired.py.
"""
