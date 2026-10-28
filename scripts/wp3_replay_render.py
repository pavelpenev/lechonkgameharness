# SPDX-License-Identifier: MIT
"""WP3 package A — render a run's JSONL transcript into a static replay
bundle (design-plan-freeciv-v1.1.md §7.6).

Usage:
  .venv/bin/python scripts/wp3_replay_render.py <run_dir>
  .venv/bin/python scripts/wp3_replay_render.py <run_dir> --transcript NAME
  .venv/bin/python scripts/wp3_replay_render.py <run_dir> --turns NAME
  .venv/bin/python scripts/wp3_replay_render.py <run_dir> --traces NAME
  .venv/bin/python scripts/wp3_replay_render.py <run_dir> --out DIR

Default: renders <run_dir>/transcript.jsonl (+ turns.jsonl when present)
into <run_dir>/replay/ — a self-contained HTML bundle (no server, no
external network; the transcript, the per-turn summaries and the
commitments board are embedded in the HTML).

MIT: this script imports only lechonk.eval.replay (CivRealm-free).
"""

import argparse
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lechonk.eval.replay import render_bundle  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="WP3 — render a run's transcript into a static replay "
                    "bundle (self-contained HTML, no server)")
    parser.add_argument("run_dir", help="the run directory under "
                                        "artifacts/runs/")
    parser.add_argument("--transcript", default="transcript.jsonl",
                        help="transcript file name inside the run dir "
                             "(default: transcript.jsonl)")
    parser.add_argument("--turns", default="turns.jsonl",
                        help="per-turn summaries file name inside the run "
                             "dir (default: turns.jsonl; used only when "
                             "present)")
    parser.add_argument("--out", default="replay",
                        help="bundle directory name inside the run dir "
                             "(default: replay)")
    parser.add_argument("--title", default=None, help="viewer title")
    parser.add_argument("--traces", default=None,
                        help="model_traces.jsonl name inside run dir; "
                             "embed full model output when supplied")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (Path(REPO) / "artifacts" / "runs" / run_dir).resolve()
    transcript = run_dir / args.transcript
    turns = run_dir / args.turns
    out = run_dir / args.out
    traces = run_dir / args.traces if args.traces else None
    if args.traces and not traces.is_file():
        print(f"[wp3-replay] FATAL: model traces not found: {traces}")
        return 2
    if not transcript.is_file():
        print(f"[wp3-replay] FATAL: transcript not found: {transcript}")
        return 2
    summary = render_bundle(
        transcript, out,
        turns_path=turns if turns.is_file() else None,
        title=args.title,
        traces_path=traces if traces and traces.is_file() else None)
    meta = summary["meta"]
    kinds = ", ".join(f"{k}:{v}" for k, v in sorted(meta["record_kinds"].items()))
    print(f"[wp3-replay] run dir:  {run_dir}")
    print(f"[wp3-replay] records:  {meta['record_count']} ({kinds})")
    print(f"[wp3-replay] bundle:   {summary['bundle_dir']}")
    print(f"[wp3-replay] index:    {summary['index_html']}")
    print(f"[wp3-replay] turns recorded: {meta['turns_recorded']} · "
          f"commitments board: {meta['commitments_board_size']} · "
          f"outcome: {meta['outcome']}")
    print(f"[wp3-replay] self-contained: {meta['self_contained']} · "
          f"no server: {meta['no_server']} · "
          f"no external network: {meta['no_external_network']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
