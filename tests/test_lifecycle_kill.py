# SPDX-License-Identifier: MIT
"""WP3 lifecycle tests: the watchdog-kill test (plan §4 line 56, §3
line 34) — the append-only transcript survives SIGKILL exercised
MID-WRITE (a torn tail), not merely between two complete records.

The test drives scripts/wp3_kill_test.py's run_kill_scenario(): a child
process runs the REAL ledger writer (TranscriptLedger) appending large
identical checksummed records with the §3 action-boundary fsync; the
parent SIGKILLs the child the moment the file size is mid-stride (a
record write in progress). The test asserts the full §3 line 34
contract: every complete record survives and checksum-verifies, the
trailing partial record is discarded on reload, and the ledger resumes
cleanly (seq continuity). Evidence is persisted under
artifacts/runs/wp3-kill/.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "wp3_kill_test", REPO / "scripts" / "wp3_kill_test.py")
wp3_kill_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp3_kill_test)


def test_sigkill_mid_write_survival_and_torn_tail_discard():
    """The mid-write SIGKILL scenario end to end (see module docstring)."""
    outdir = REPO / "artifacts" / "runs" / "wp3-kill"
    evidence = wp3_kill_test.run_kill_scenario(outdir)

    checks = evidence["checks"]
    failed = {name: entry for name, entry in checks.items()
              if not entry["pass"]}
    assert not failed, f"kill-scenario checks failed: {failed}"
    # The WP3 acceptance: the kill landed INSIDE a write (a torn tail),
    # not between two complete records.
    assert evidence["kill"]["torn_bytes_at_kill"] > 0
    assert evidence["kill"]["returncode"] == -9  # SIGKILL
    # Every complete record survived; the torn tail was discarded.
    assert evidence["survivors"]["count"] >= 1
    assert evidence["torn_tail"]["bytes"] > 0
    # The ledger resumed cleanly (seq continuity).
    assert checks["attach_resumes_seq"]["pass"]
    assert checks["resumed_append_seq_continues"]["pass"]
    assert checks["post_recovery_transcript_clean"]["pass"]
    # Evidence persisted.
    assert (outdir / "kill_evidence.json").exists()
    assert (outdir / "kill_evidence.md").exists()
    assert (outdir / "transcript.jsonl").exists()
    assert (outdir / "child_stdout.log").exists()
