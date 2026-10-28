#!/usr/bin/env python3
"""Offline, descriptive comparison of checksummed game-run artifacts.

Only ``transcript.jsonl``, ``api_calls.jsonl`` and checksum-verified,
standalone conversation-state files are read.  Conversation state is mined
from user-visible snapshot objects only; assistant messages, traces, and
reasoning are never copied into the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lechonk.core.jsonutil import canonical_json
from lechonk.eval.paired import turn_summary_from_snapshot_json


def _checksum(obj: Mapping[str, Any], field: str = "checksum") -> str:
    payload = {k: v for k, v in obj.items() if k != field}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _known(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _sum_known(values: Iterable[Any]) -> Any:
    vals = [v for v in values if _known(v)]
    return sum(vals) if vals else None


def _read_jsonl(path: Path, *, require_checksum: bool = False) -> tuple[list[dict], dict]:
    records: list[dict] = []
    status = {"present": path.exists(), "valid": True, "checksum_invalid": 0,
              "checksum_unavailable": 0, "parse_invalid": 0, "seq_invalid": False}
    if not path.exists():
        status["valid"] = False
        return records, status
    expected_seq = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
        except Exception:
            status["parse_invalid"] += 1
            status["valid"] = False
            continue
        checksum = obj.get("checksum")
        if checksum is None:
            status["checksum_unavailable"] += 1
            if require_checksum:
                status["valid"] = False
                continue
        elif checksum != _checksum(obj):
            status["checksum_invalid"] += 1
            status["valid"] = False
            continue
        seq = obj.get("seq")
        if seq is not None:
            if seq != expected_seq:
                status["seq_invalid"] = True
                status["valid"] = False
            expected_seq = (seq + 1) if isinstance(seq, int) else expected_seq
        records.append(obj)
    return records, status


def _snapshot_objects(value: Any) -> Iterable[dict]:
    """Yield only snapshot-shaped objects from user-visible JSON values."""
    if isinstance(value, dict):
        if (isinstance(value.get("player"), dict) and
                isinstance(value.get("cities"), list) and
                isinstance(value.get("units"), list)):
            yield value
        for child in value.values():
            yield from _snapshot_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _snapshot_objects(child)
    elif isinstance(value, str) and len(value) <= 2_000_000:
        marker = value.find("## SNAPSHOT")
        if marker >= 0:
            section = re.split(r"\n##\s+", value[marker + len("## SNAPSHOT"):], maxsplit=1)[0]
            section = section.replace("```json", "").replace("```", "").strip()
            try:
                start = section.find("{")
                if start >= 0:
                    parsed, _ = json.JSONDecoder().raw_decode(section[start:])
                    yield from _snapshot_objects(parsed)
            except Exception:
                pass
            return
        try:
            parsed = json.loads(value)
        except Exception:
            return
        if isinstance(parsed, (dict, list)):
            yield from _snapshot_objects(parsed)


def _user_content(message: Mapping[str, Any]) -> Iterable[Any]:
    if message.get("role") != "user":
        return ()
    content = message.get("content")
    if isinstance(content, list):
        return (part.get("text") for part in content if isinstance(part, dict))
    return (content,)


def _read_snapshots(run: Path) -> tuple[dict[int, dict], dict]:
    found: dict[int, dict] = {}
    status = {"files": 0, "valid_files": 0, "invalid_files": 0,
              "checksum_invalid": 0, "duplicate_turns": [], "conflicts": [],
              "state_available": False, "current_state_available": False,
              "source": None, "epochs": []}
    preferred_snapshot_found = False
    # The active state is authoritative; older cumulative checkpoints are
    # useful only when the active state is absent.
    preferred = [run / "state.json", run / "conversation-state.json"]
    recovery_path = run / "recovery.json"
    if recovery_path.exists():
        try:
            recovery_obj = json.loads(recovery_path.read_text(encoding="utf-8"))
            bound = recovery_obj.get("conversation_state_path")
            if isinstance(bound, str):
                bound_path = Path(bound)
                if not bound_path.is_absolute():
                    bound_path = run / bound_path
                if (bound_path.resolve().is_relative_to(run.resolve()) and
                        bound_path.is_file() and not bound_path.is_symlink()):
                    preferred.insert(0, bound_path)
        except Exception:
            pass
    preferred = list(dict.fromkeys(preferred))
    def safe_file(path: Path) -> bool:
        try:
            return (path.exists() and path.is_file() and not path.is_symlink() and
                    path.resolve().is_relative_to(run.resolve()))
        except OSError:
            return False
    unsafe = [str(p) for p in preferred if p.exists() and not safe_file(p)]
    candidates = [p for p in preferred if safe_file(p)]
    for p in sorted((set(run.glob("checkpoints/conversation-state*.json")) |
                     set(run.glob("conversation-state*.json"))) - set(candidates)):
        if safe_file(p):
            candidates.append(p)
        elif p.exists():
            unsafe.append(str(p))
    status["unsafe_paths"] = unsafe
    for path in candidates:
        status["files"] += 1
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict) or obj.get("checksum") != _checksum(obj):
                raise ValueError("checksum mismatch")
            status["valid_files"] += 1
            if obj.get("epoch") is not None:
                status["epochs"].append(obj.get("epoch"))
            if path in preferred:
                status["state_available"] = True
                status["source"] = str(path)
        except Exception:
            status["invalid_files"] += 1
            status["checksum_invalid"] += 1
            continue
        # Never inspect assistant content.  Only traverse user message values.
        for message in obj.get("messages", []):
            if not isinstance(message, dict):
                continue
            for raw in _user_content(message):
                for snapshot in _snapshot_objects(raw):
                    summary = turn_summary_from_snapshot_json(json.dumps(snapshot))
                    if summary is not None and _known(summary.get("turn")):
                        snapshot_turn = int(summary["turn"])
                        candidate = {
                            k: summary.get(k) for k in (
                                "score", "gold", "techs_researched", "research_target",
                                "research_turns_left", "cities_own", "units_own", "cities")
                        }
                        candidate["production"] = [
                            c.get("production") for c in summary.get("cities", [])]
                        candidate["compaction_epoch"] = obj.get("epoch")
                        if snapshot_turn in found:
                            # Epoch changes are compaction metadata, not a
                            # game-state conflict.
                            old_state = {k: v for k, v in found[snapshot_turn].items()
                                         if k != "compaction_epoch"}
                            new_state = {k: v for k, v in candidate.items()
                                         if k != "compaction_epoch"}
                            if old_state == new_state:
                                continue
                            status["conflicts"].append(snapshot_turn)
                            status.setdefault("conflict_sources", []).append(str(path))
                            # Preferred/current state was processed first.
                            continue
                        found[snapshot_turn] = candidate
                        if path in preferred:
                            preferred_snapshot_found = True
    status["state_available"] = bool(found)
    status["current_state_available"] = preferred_snapshot_found
    return found, status


def _turns_for(run: Path, transcript: list[dict], api: list[dict], snapshots: dict[int, dict],
               lo: int | None, hi: int | None) -> dict[int, dict]:
    turns: dict[int, dict] = {}
    def row(turn: int) -> dict:
        return turns.setdefault(turn, {"turn": turn, "state": {}, "actions": {},
            "provider": {}, "compaction_epoch": None})
    for rec in transcript:
        turn = rec.get("turn")
        if not isinstance(turn, int):
            continue
        r = row(turn)
        kind = rec.get("kind")
        if kind == "action_result":
            key = rec.get("outcome")
            if key in ("executed", "rejected", "aborted_group"):
                r["actions"][key] = r["actions"].get(key, 0) + 1
            if key == "rejected":
                cls = rec.get("rejection_class") or "unknown"
                r["actions"]["rejections_by_class"] = r["actions"].get("rejections_by_class", {})
                r["actions"]["rejections_by_class"][cls] = r["actions"]["rejections_by_class"].get(cls, 0) + 1
            if rec.get("degraded") is True:
                r["actions"]["degraded"] = r["actions"].get("degraded", 0) + 1
        elif kind == "metric_event" and rec.get("name") in {"compaction", "compaction_epoch"}:
            r["compaction_epoch"] = rec.get("value")
    # API logs are intentionally handled as a separate allow-listed source.
    # They commonly have no turn_start transcript record for compaction calls.
    for rec in api:
        turn = rec.get("turn")
        if not isinstance(turn, int):
            continue
        r = row(turn)
        p = r["provider"]
        p["calls"] = p.get("calls", 0) + 1
        p.setdefault("input_tokens", []).append(rec.get("tokens_in"))
        p.setdefault("output_tokens", []).append(rec.get("tokens_out"))
        p.setdefault("cached_tokens", []).append(_cached(rec))
        p.setdefault("latency_ms", []).append(rec.get("latency_ms"))
        if rec.get("kind") == "repair":
            r["actions"]["repairs"] = r["actions"].get("repairs", 0) + 1
        epoch = rec.get("compaction_epoch", rec.get("epoch"))
        if epoch is not None:
            r["compaction_epoch"] = epoch
        if rec.get("degraded") is True or rec.get("outcome") == "degraded":
            r["actions"]["degraded"] = r["actions"].get("degraded", 0) + 1
    if not api:
        # Legacy artifacts can have model-call records only.  Use those as a
        # fallback; when api_calls.jsonl exists it is the authoritative source
        # and prevents double counting.
        for rec in transcript:
            if rec.get("kind") != "model_call" or not isinstance(rec.get("turn"), int):
                continue
            r = row(rec["turn"]); p = r["provider"]
            p["calls"] = p.get("calls", 0) + 1
            for key, value in (("input_tokens", rec.get("tokens_in")),
                                ("output_tokens", rec.get("tokens_out")),
                                ("cached_tokens", _cached(rec)),
                                ("latency_ms", rec.get("latency_ms"))):
                p.setdefault(key, []).append(value)
            if rec.get("kind") == "repair":
                r["actions"]["repairs"] = r["actions"].get("repairs", 0) + 1
    for turn, summary in snapshots.items():
        if lo is None or turn >= lo:
            if hi is None or turn <= hi:
                row(turn)["state"] = summary
                row(turn)["compaction_epoch"] = summary.get("compaction_epoch")
    selected = {}
    for turn, r in turns.items():
        if (lo is not None and turn < lo) or (hi is not None and turn > hi):
            continue
        p = r["provider"]
        for key, vals in list(p.items()):
            if isinstance(vals, list):
                p[key] = _sum_known(vals)
        r["state_available"] = bool(r["state"])
        selected[turn] = r
    return selected


def _cached(rec: Mapping[str, Any]) -> Any:
    for key in ("cached_tokens", "tokens_cached", "prompt_cached_tokens"):
        if key in rec:
            return rec[key]
    details = rec.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return details.get("cached_tokens")
    return None


def _manifest(transcript: list[dict]) -> dict:
    for rec in transcript:
        if rec.get("kind") == "run_manifest" and isinstance(rec.get("manifest"), dict):
            return rec["manifest"]
    return {}


_SETTING_KEYS = {"config_digest", "prompt_bundle_id", "prompt_profile", "context_policy",
                 "aifill", "configured_ai_opponents", "seed", "ruleset", "ruleset_name",
                 "candidate_surface_policy", "compaction_schema",
                 "conversation_compaction_limit_tokens", "conversation_tail_target_tokens",
                 "model", "model_requested", "temperature", "top_p", "max_tokens",
                 "max_output_tokens", "api_timeout_s", "acting_deadline_s",
                 "victories", "timeout", "server_timeout", "pingtimeout", "self_play",
                 "gameseed", "mapseed", "agentseed", "seeds"}


def _settings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: manifest[k] for k in _SETTING_KEYS if k in manifest}
    for container in (manifest.get("config_after_load"), manifest.get("server_settings"),
                      manifest.get("config"), manifest.get("fc_args_overrides")):
        if isinstance(container, Mapping):
            out.update({k: container[k] for k in _SETTING_KEYS if k in container})
    return out


def _recovery_status(run: Path) -> dict[str, Any]:
    path = run / "recovery.json"
    result: dict[str, Any] = {"present": path.exists(), "valid": None,
                              "error": None, "epoch": None, "parent_run_id": None,
                              "wal_present": None, "wal_valid": None,
                              "wal_source": str(run / "conversation.jsonl"),
                              "state_wal_relationship": "not_proven"}
    if not path.exists():
        return result
    if path.is_symlink() or not path.is_file():
        result.update({"valid": False, "error": "recovery manifest is not a regular file"})
        return result
    try:
        from lechonk.core.recovery import load_checkpoint
        from lechonk.model.conversation import ConversationManager
        cp = load_checkpoint(path, artifact_root=run, verify_files=True)
        # Runtime artifacts keep the active WAL at the run root.  The
        # standalone checkpoint state in checkpoints/ is marker-0 evidence,
        # not the runtime WAL's sibling.
        wal_path = run / "conversation.jsonl"
        if wal_path.is_symlink() or (wal_path.exists() and not wal_path.is_file()):
            raise RuntimeError("conversation WAL is not a regular file")
        result["wal_present"] = wal_path.exists()
        if not wal_path.exists():
            raise RuntimeError("conversation WAL is absent")
        ConversationManager.load_journal(wal_path)
        result["wal_valid"] = True
        state = ConversationManager.load(cp.conversation_state_path)
        result.update({"valid": True, "epoch": state.epoch,
                       "parent_run_id": cp.parent_run_id, "turn": cp.turn})
    except Exception as exc:
        if result.get("wal_present") and result.get("wal_valid") is None:
            result["wal_valid"] = False
        result.update({"valid": False, "error": type(exc).__name__ + ": " + str(exc)})
    return result


def _run(run: Path, lo: int | None, hi: int | None) -> dict:
    transcript, ts = _read_jsonl(run / "transcript.jsonl")
    api, api_status = _read_jsonl(run / "api_calls.jsonl")
    snapshots, snap_status = _read_snapshots(run)
    manifest = _manifest(transcript)
    turns = _turns_for(run, transcript, api, snapshots, lo, hi)
    outcome = next((r for r in transcript if r.get("kind") == "outcome"), None)
    epochs = {rec.get("compaction_epoch", rec.get("epoch")) for rec in api
              if rec.get("turn") is None and
              rec.get("compaction_epoch", rec.get("epoch")) is not None}
    epochs.update(t["compaction_epoch"] for t in turns.values()
                  if t.get("compaction_epoch") is not None)
    epochs.update(snap_status.get("epochs", []))
    epochs = sorted(epochs)
    action_totals: dict[str, int] = {}
    for turn in turns.values():
        for key in ("executed", "rejected", "aborted_group", "repairs", "degraded"):
            if isinstance(turn["actions"].get(key), int):
                action_totals[key] = action_totals.get(key, 0) + turn["actions"][key]
    final_state = turns[max(turns)]["state"] if turns else None
    if not snap_status.get("current_state_available"):
        final_state = None
    provider_totals = {}
    for key in ("calls", "input_tokens", "output_tokens", "cached_tokens", "latency_ms"):
        provider_totals[key] = _sum_known(t["provider"].get(key) for t in turns.values())
    recovery = _recovery_status(run)
    fully_verified = bool(ts["valid"] and api_status["valid"] and
                         api_status["checksum_unavailable"] == 0 and
                         recovery["present"] and recovery["valid"] is True and
                         recovery.get("wal_valid") is True and
                         recovery.get("state_wal_relationship") == "proven")
    return {"run": str(run), "integrity": {"transcript": ts, "api_calls": api_status,
            "snapshots": snap_status, "recovery": recovery,
            "fully_verified": fully_verified,
            "verification_note": ("all allowed checksums and the state/WAL binding are verified"
                                   if fully_verified else
                                   "not fully verified: state/WAL binding is not proven or an allowed checksum is unavailable")},
        "settings": {"run_id": manifest.get("run_id"),
                     "parent_run_id": manifest.get("parent_run_id"), **_settings(manifest)},
            "aggregate": {"state_available": bool(snap_status.get("state_available")),
                       "final_state": final_state, "actions": action_totals,
                       "provider": provider_totals},
        "compaction_epochs": epochs,
        "outcome": ({"outcome": outcome.get("outcome"), "reason_available": bool(outcome.get("reason"))}
                    if outcome else None),
        "turns": [turns[k] for k in sorted(turns)]}


def compare_runs(paths: Iterable[str | Path], *, turn_start: int | None = None,
                 turn_end: int | None = None) -> dict:
    runs = [_run(Path(p).resolve(), turn_start, turn_end) for p in paths]
    signatures = [json.dumps({k: v for k, v in r["settings"].items()
                              if k not in {"run_id", "parent_run_id"}}, sort_keys=True)
                  for r in runs]
    flags: list[str] = []
    if len(set(signatures)) > 1:
        flags.append("different profile/settings are present")
    parents = [r["settings"].get("parent_run_id") for r in runs]
    parents += [r["integrity"]["recovery"].get("parent_run_id") for r in runs]
    if any(parents):
        flags.append("one or more runs declare a fork/parent lineage")
    turn_sets = [set(t["turn"] for t in r["turns"]) for r in runs]
    if turn_sets and len(set(map(lambda s: tuple(sorted(s)), turn_sets))) > 1:
        flags.append("turn windows are missing or do not overlap exactly")
    for r in runs:
        if r["integrity"]["api_calls"].get("checksum_unavailable"):
            flags.append(f"{Path(r['run']).name}: API records lack checksums")
        if r["integrity"]["recovery"].get("valid") is False:
            flags.append(f"{Path(r['run']).name}: recovery/checkpoint integrity unavailable")
        if r["integrity"]["snapshots"].get("duplicate_turns"):
            flags.append(f"{Path(r['run']).name}: multiple checkpoint snapshots overlap")
        if r["integrity"]["snapshots"].get("conflicts"):
            turns = ",".join(str(t) for t in sorted(set(r["integrity"]["snapshots"]["conflicts"])))
            flags.append(f"{Path(r['run']).name}: conflicting checkpoint snapshots at turns {turns}")
        if r["integrity"]["snapshots"].get("unsafe_paths"):
            flags.append(f"{Path(r['run']).name}: snapshot paths outside artifact root or symlinked")
        observed = [t["turn"] for t in r["turns"]]
        available = [t["turn"] for t in r["turns"] if t.get("state_available")]
        if observed:
            expected = set(range(min(observed), max(observed) + 1))
            missing = sorted(expected - set(available))
            missing += sorted(set(observed) - set(available) - expected)
            if missing:
                flags.append(f"{Path(r['run']).name}: missing state turns {','.join(map(str, missing))}")
        if r["integrity"]["recovery"].get("wal_valid") is False:
            flags.append(f"{Path(r['run']).name}: conversation WAL invalid or absent")
        if any(t["actions"].get("repairs") for t in r["turns"]):
            flags.append(f"{Path(r['run']).name}: repaired turns present")
        if any(t["actions"].get("degraded") for t in r["turns"]):
            flags.append(f"{Path(r['run']).name}: degraded turns present")
    return {"description": "Descriptive run comparison; no causal or ranking claim is made.",
            "turn_range": {"start": turn_start, "end": turn_end},
            "flags": sorted(set(flags)), "runs": runs}


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Game-run comparison", "", report["description"], ""]
    if report["flags"]:
        lines += ["Flags: " + "; ".join(report["flags"]), ""]
    lines += ["| run | turn | score | gold | techs | cities | units | input | output | cached | latency ms | repairs |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        name = Path(run["run"]).name
        for t in run["turns"]:
            s, p = t["state"], t["provider"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, t["turn"], *(s.get(k, "?") for k in ("score", "gold", "techs_researched", "cities_own", "units_own")),
                p.get("input_tokens", "?"), p.get("output_tokens", "?"), p.get("cached_tokens", "?"),
                p.get("latency_ms", "?"), t["actions"].get("repairs", "?")))
    return "\n".join(lines) + "\n"


def _atomic(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="two or more artifact directories")
    p.add_argument("--turn-start", type=int); p.add_argument("--turn-end", type=int)
    p.add_argument("--json-out", type=Path); p.add_argument("--markdown-out", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if len(a.runs) < 2:
        p.error("at least two run directories are required")
    report = compare_runs(a.runs, turn_start=a.turn_start, turn_end=a.turn_end)
    if a.json_out: _atomic(a.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n", a.overwrite)
    if a.markdown_out: _atomic(a.markdown_out, markdown(report), a.overwrite)
    if not a.json_out and not a.markdown_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
