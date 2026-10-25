# SPDX-License-Identifier: MIT
"""WP3 package B deliverable 1: the watchdog-kill test (plan §4 line 56,
§3 line 34) — the append-only transcript survives SIGKILL, exercised
MID-WRITE (a torn tail), not merely between two complete records.

Design: a child process runs the REAL ledger writer
(lechonk.core.ledger.TranscriptLedger) appending large identical
checksummed records, each with the §3 action-boundary fsync
(durable=True). The parent polls the transcript's file size: a record
write in progress shows a size that is NOT a multiple of the record
stride (the kernel grows the file progressively during a large buffered
write — verified empirically: 2 MiB growth steps for a 64 MiB record).
The moment the size is mid-stride, the parent SIGKILLs the child: the
kill lands INSIDE the record write, leaving a torn tail at EOF.

Proven afterwards (§3 line 34):

- every COMPLETE record survives and its checksum verifies
  (load_transcript re-validates every line against the frozen union);
- the trailing partial record (the torn bytes at EOF) is DISCARDED on
  reload — never silently truncated mid-file (corruption before the
  trailing record would raise LedgerCorruptionError);
- the ledger resumes cleanly: TranscriptLedger.attach truncates the torn
  tail out of the FILE and the next append continues at last valid
  seq + 1 (seq continuity).

Run standalone (persists evidence under a per-run unique subdirectory
of artifacts/runs/wp3-kill/ — the recurring root is never rewritten,
so prior evidence is never deleted or overwritten):
    .venv/bin/python scripts/wp3_kill_test.py
The pytest suite drives the same scenario via run_kill_scenario()
(tests/test_lifecycle_kill.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUN = "wp3-kill"
RUNDIR = REPO / "artifacts" / "runs" / RUN


def _default_outdir() -> Path:
    """A per-run unique subdirectory under the recurring wp3-kill root
    (timestamp-pid): a manual rerun never rewrites prior evidence; the
    recurring root itself stays frozen."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return RUNDIR / f"{stamp}-{os.getpid()}"

# 8 MiB records: each write takes long enough (~10-30 ms of progressive
# growth) for the parent's size polling to catch it mid-stride, while
# keeping the persisted transcript artifact modest.
PAYLOAD_BYTES = 8 * 1024 * 1024
RECORDS = 5  # appended after the first (stride-establishing) record


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# the child: the real ledger writer
# ---------------------------------------------------------------------------


def child_main(path: str, count: int, payload_bytes: int) -> int:
    """--child mode: append large identical checksummed records through
    the real TranscriptLedger with the §3 action-boundary fsync."""
    from lechonk.core.ledger import TranscriptLedger
    from lechonk.core.schemas import MetricEventRecord

    ledger = TranscriptLedger(path)
    record = MetricEventRecord(name="kill_test_payload",
                               value="A" * payload_bytes)
    ledger.append(record, durable=True)  # record 0: establishes the stride
    with open(path, "rb") as fh:
        stride = len(fh.readline())
    print(f"READY stride={stride} count={count}", flush=True)
    for _ in range(count):
        ledger.append(record, durable=True)
    ledger.close()
    print("DONE", flush=True)
    return 0


# ---------------------------------------------------------------------------
# the parent: SIGKILL mid-write + verification
# ---------------------------------------------------------------------------


def _await_ready(log_path: Path, proc: subprocess.Popen, timeout_s: float
                 ) -> int:
    """Wait for the child's READY line; return the record stride."""
    deadline = time.monotonic() + timeout_s
    pattern = re.compile(rb"READY stride=(\d+) count=(\d+)")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"child exited before READY (rc={proc.returncode}); "
                f"log: {log_path.read_bytes()[-2000:]!r}")
        match = pattern.search(log_path.read_bytes())
        if match:
            return int(match.group(1))
        time.sleep(0.005)
    raise TimeoutError(f"no READY line within {timeout_s}s")


def _kill_mid_write(proc: subprocess.Popen, path: Path, stride: int,
                    timeout_s: float) -> dict:
    """Poll the transcript size; SIGKILL the child the moment a record
    write is IN PROGRESS (size not a multiple of the stride) — the kill
    lands inside the write, leaving a torn tail at EOF."""
    deadline = time.monotonic() + timeout_s
    samples = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"child exited before the mid-write kill (rc="
                f"{proc.returncode}) — the kill window was missed")
        size = path.stat().st_size
        samples += 1
        if size % stride != 0:
            # A record write is in progress: SIGKILL now. The bytes after
            # the last complete record are a torn record prefix.
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
            return {
                "killed": True,
                "signal": "SIGKILL",
                "returncode": proc.returncode,
                "size_at_kill": size,
                "stride": stride,
                "complete_records_at_kill": size // stride,
                "torn_bytes_at_kill": size % stride,
                "poll_samples": samples,
            }
        time.sleep(0.0002)
    raise TimeoutError(f"no mid-write window within {timeout_s}s")


def run_kill_scenario(outdir, *, python: str = sys.executable,
                      records: int = RECORDS,
                      payload_bytes: int = PAYLOAD_BYTES,
                      timeout_s: float = 180.0) -> dict:
    """Run the mid-write SIGKILL scenario and verify the §3 line 34
    durability contract. Returns the evidence dict (checks included);
    persists the evidence under ``outdir``."""
    from lechonk.core.ledger import TranscriptLedger, load_transcript
    from lechonk.core.schemas import OutcomeRecord

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    transcript = outdir / "transcript.jsonl"
    child_log = outdir / "child_stdout.log"
    for path in (transcript, child_log):
        if path.exists():
            path.unlink()

    checks: dict = {}

    def check(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        log(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    t0 = time.monotonic()
    with open(child_log, "wb") as log_fh:
        proc = subprocess.Popen(
            [python, str(Path(__file__).resolve()), "--child",
             str(transcript), str(records), str(payload_bytes)],
            stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(REPO))
        try:
            stride = _await_ready(child_log, proc, timeout_s)
            log(f"child READY: stride={stride} records={records} "
                f"payload_bytes={payload_bytes}")
            kill = _kill_mid_write(proc, transcript, stride, timeout_s)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    log(f"SIGKILL landed: {json.dumps(kill, sort_keys=True)}")

    # --- the kill landed INSIDE a write (the WP3 acceptance) -----------
    check("kill_landed_mid_write", kill["torn_bytes_at_kill"] > 0,
          f"SIGKILL landed inside a record write: "
          f"{kill['torn_bytes_at_kill']} torn bytes at EOF (file size "
          f"{kill['size_at_kill']}, stride {stride}) — not between two "
          f"complete records")
    check("killed_by_sigkill", kill["returncode"] == -signal.SIGKILL,
          f"child returncode={kill['returncode']} "
          f"(-SIGKILL={-signal.SIGKILL})")

    # --- every complete record survives, checksums verify --------------
    raw = transcript.read_bytes()
    torn_bytes = len(raw) % stride
    check("torn_tail_present_at_eof", torn_bytes > 0,
          f"file size {len(raw)} = {len(raw) // stride} complete records "
          f"+ {torn_bytes} torn bytes; last byte is not a newline: "
          f"{raw[-1:] != b'\\n'}")
    load = load_transcript(transcript)
    check("torn_tail_discarded_on_reload", load.discarded_trailing_partial,
          "load_transcript discarded the trailing partial record")
    check("discarded_bytes_match_torn_tail",
          load.discarded_bytes == torn_bytes,
          f"discarded_bytes={load.discarded_bytes} == torn tail "
          f"{torn_bytes}")
    seqs = [r.seq for r in load.records]
    check("complete_records_survive",
          len(seqs) == len(raw) // stride,
          f"{len(seqs)} complete records survived the mid-write kill "
          f"(seqs {seqs[0]}..{seqs[-1]})")
    check("seq_contiguous", seqs == list(range(len(seqs))),
          f"seq continuity over the survivors: {seqs}")
    check("checksums_verified",
          all(r.checksum for r in load.records),
          "every surviving record re-validated against the frozen union "
          "and checksum-verified by load_transcript (a checksum failure "
          "would have raised, not passed)")
    # The torn tail is a PREFIX of a record line (torn mid-write, not
    # garbage): it must start the canonical record shape and contain no
    # newline.
    tail = raw[-torn_bytes:]
    check("torn_tail_is_record_prefix",
          tail.startswith(b'{"checksum":"') and b"\n" not in tail,
          f"the {torn_bytes} torn bytes start a checksummed record line "
          f"and contain no newline (first 32 bytes: {tail[:32]!r})")

    # --- the ledger resumes cleanly (seq continuity) --------------------
    recovered = TranscriptLedger.attach(transcript)
    check("attach_resumes_seq", recovered.next_seq == len(seqs),
          f"attach resumes at last valid seq + 1: next_seq="
          f"{recovered.next_seq} (survivors={len(seqs)})")
    check("attach_truncated_torn_tail_from_file",
          transcript.read_bytes() == raw[:len(raw) - torn_bytes],
          "the torn tail was truncated out of the FILE (the next append "
          "lands right after the last valid record)")
    resumed_seq = recovered.append(
        OutcomeRecord(outcome="COMPLETED",
                      reason="resumed after the mid-write SIGKILL "
                             "(wp3-kill)"),
        durable=True)
    recovered.close()
    check("resumed_append_seq_continues", resumed_seq == len(seqs),
          f"the resumed append took seq {resumed_seq} "
          f"(last valid + 1)")
    load2 = load_transcript(transcript)
    check("post_recovery_transcript_clean",
          not load2.discarded_trailing_partial
          and [r.seq for r in load2.records] == list(range(len(seqs) + 1)),
          f"the recovered transcript loads clean: {len(load2.records)} "
          f"records, seq 0..{len(seqs)}, no torn tail")
    check("post_recovery_ends_with_newline",
          transcript.read_bytes().endswith(b"\n"),
          "the recovered transcript is flushable (newline-terminated)")

    wall = time.monotonic() - t0
    evidence = {
        "run": RUN,
        "scenario": "real TranscriptLedger writer SIGKILLed mid-record-"
                    "write (torn tail at EOF), then reload + attach resume",
        "child": {"python": python, "records": records,
                  "payload_bytes": payload_bytes, "stride": stride},
        "kill": kill,
        "torn_tail": {
            "bytes": torn_bytes,
            "sha256": hashlib.sha256(tail).hexdigest(),
            "first_64_bytes_hex": tail[:64].hex(),
        },
        "survivors": {"count": len(seqs), "first_seq": seqs[0],
                      "last_seq": seqs[-1]},
        "wall_clock_s": round(wall, 2),
        "checks": checks,
        "verdict": "PASS" if all(c["pass"] for c in checks.values())
                   else "FAIL",
        "artifacts": {
            "transcript": str(transcript),
            "child_stdout": str(child_log),
            "evidence_json": str(outdir / "kill_evidence.json"),
            "evidence_md": str(outdir / "kill_evidence.md"),
        },
    }
    with open(outdir / "kill_evidence.json", "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    # A run_meta.json for the P11 audit (this run has no game session:
    # the kill test audits the ledger writer only — identity/close are
    # n/a there).
    with open(outdir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "run": RUN,
            "outcome": {"outcome": "COMPLETED",
                        "reason": "resumed after the mid-write SIGKILL "
                                  "(wp3-kill)",
                        "turns_completed": 0,
                        "close_status": None},
            "wall_clock_s": round(wall, 2),
            "injection": {"scenario": evidence["scenario"],
                          "kill": kill},
            "transcript": {"path": str(transcript),
                           "records": len(seqs) + 1,
                           "discarded_trailing_partial": False,
                           "last_seq": len(seqs)},
            "checks": checks,
            "verdict": evidence["verdict"],
            "note": "no game session in this run (the kill test audits "
                    "the ledger writer only); the resumed append after "
                    "attach is the transcript's outcome record",
        }, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    lines = [
        "# WP3 kill test — mid-write SIGKILL survival evidence",
        "",
        f"- scenario: the real `TranscriptLedger` writer appends "
        f"{records + 1} identical {payload_bytes}-byte checksummed "
        f"records (stride {stride} bytes), each with the §3 "
        f"action-boundary fsync (`durable=True`); the parent SIGKILLs "
        f"the child the moment the file size is mid-stride (a record "
        f"write in progress)",
        f"- kill: landed inside a record write — {torn_bytes} torn bytes "
        f"at EOF (file size {len(raw)} = {len(raw) // stride} complete "
        f"records + torn tail); child returncode {kill['returncode']}",
        f"- survivors: {len(seqs)} complete records (seq "
        f"{seqs[0]}..{seqs[-1]}), every one re-validated against the "
        f"frozen union and checksum-verified on reload",
        f"- torn tail: {torn_bytes} bytes, sha256 "
        f"`{hashlib.sha256(tail).hexdigest()}`, first 64 bytes "
        f"`{tail[:64].hex()}` — a record-line prefix with no newline "
        f"(torn mid-write, not garbage)",
        f"- recovery: `TranscriptLedger.attach` truncated the torn tail "
        f"out of the FILE and resumed at seq {len(seqs)}; the resumed "
        f"append is newline-terminated and the recovered transcript "
        f"loads clean",
        f"- wall clock: {wall:.2f} s",
        "",
        "## Checks",
        "",
    ]
    for name, entry in checks.items():
        lines.append(f"- {'PASS' if entry['pass'] else 'FAIL'} — "
                     f"`{name}`: {entry['detail']}")
    with open(outdir / "kill_evidence.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote kill_evidence.json/.md under {outdir}")
    log(f"VERDICT: {evidence['verdict']} "
        f"({sum(1 for c in checks.values() if c['pass'])}/{len(checks)} "
        f"checks)")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true",
                        help="child mode: run the real ledger writer")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("count", nargs="?", type=int, default=None)
    parser.add_argument("payload_bytes", nargs="?", type=int, default=None)
    parser.add_argument("--outdir", default=None,
                        help="evidence directory (default: a per-run "
                             "unique subdir of artifacts/runs/wp3-kill/)")
    args = parser.parse_args()
    if args.child:
        return child_main(args.path, args.count or RECORDS,
                          args.payload_bytes or PAYLOAD_BYTES)
    outdir = args.outdir if args.outdir is not None else _default_outdir()
    evidence = run_kill_scenario(outdir)
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
