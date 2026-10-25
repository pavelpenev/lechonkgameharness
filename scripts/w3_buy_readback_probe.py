#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""W3-8 live buy-readback probe (harness-fixes-v2 plan section 4 wave 3,
ruling D8: authorized; correction conditional on a conclusive result).

Answers D8a on the LIVE local Freeciv server: on a successful
city_buy_production, does the city's shield_stock INCREASE (the harness
verifier's pinned direction, adapter/execute.py) or DECREASE (the hypothesis
that a real buy resets the stock on completion)? The pinned verifier
(tests/test_execute.py::test_buy_production_readback) asserts VERIFIED iff
post_stock > pre_stock; a conclusive post < pre activates the pre-approved
D8b correction (verifier + pinned test + README discontinuity, one commit).

Isolation rules (the stricter sol procedure — this probe NEVER touches an
existing game):
- the probe starts a FRESH, isolated, disposable single-player game via
  AdapterSession (one env per process; the freeciv-web backend spawns a new
  game server per client connection). It never attaches to, joins, or
  mutates any existing live game, and it never drives the browser
  webclient (the HANDOFF join-URL trailing-slash convention applies only
  to browser joins, which this probe never performs).
- the player name is UNIQUE per run (UTC timestamp + pid), so the
  save/username coupling (section 7.1) can never collide with any existing
  game's player or saves.
- every run writes a NEW, never-used evidence directory
  artifacts/runs/w3-buy-readback/<UTC-timestamp>-<pid>/ (created with
  exist_ok=False: a retry can never append over an earlier attempt). The
  evidence dir stays untracked (artifacts/ is gitignored).
- the probe is fully scripted: NO model call and NO .env access (the
  client key loader is never involved).
- exactly ONE city_buy_production is executed per run, via the normal
  bounded step (execute_candidate) — no manual packets, no retries.

Procedure:
(a) start a fresh isolated game via AdapterSession;
(b) play scripted (found a city, hold a tangible production — a unit or a
    building, never Coinage) until an own city offers city_buy_production
    with a positive buy_cost, sufficient gold, and the city not founded
    this turn (the offered candidate already encodes the vendor's
    is_action_valid constraints, city_actions.py:299-330);
(c) capture the frozen pre-state;
(d) execute exactly ONE city_buy_production via execute_candidate;
(e) capture the immediate post-state BEFORE finishing the turn;
(f) corroborate success by treasury expenditure / purchase-state change /
    a corresponding server update — stock movement alone never counts;
(g) write pre.json, post.json, operation.json, verdict.md and sha256
    checksums into the evidence dir;
(h) print the observation triple (pre_stock, post_stock, corroborated?).

Decision table (D8b):
- post > pre AND corroborated: the existing verifier direction is
  CONFIRMED — record the evidence, no code change, metric-bearing runs
  unblock.
- post < pre AND corroborated: the D8b correction ACTIVATES (pre-approved).
- post == pre / unobservable / ambiguous success / wrong identity / no
  buyable city in a bounded session: INCONCLUSIVE — record a
  FAILED/INCOMPLETE attempt honestly; do not change the verifier; do not
  unblock metric-bearing runs.

Usage:
  .venv/bin/python scripts/w3_buy_readback_probe.py [--seed N]
                                                     [--max-turns N]

Exit codes: 0 conclusive observation recorded (either direction),
1 unexpected error, 2 stack down, 3 inconclusive / failed attempt.

GPL-3.0-or-later (decision 7): this script imports the adapter package
(the only CivRealm importer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# --- run constants -----------------------------------------------------------
RUN_ROOT = os.path.join(REPO, "artifacts", "runs", "w3-buy-readback")
PUBSTATUS = "http://localhost:8080/pubstatus"
SEED = 20261025            # distinct from the WP0/1/2/3 seeds
AIFILL = 2
MAX_TURNS = 40             # bounded session
WALL_BUDGET_S = 1500.0     # bounded session (~25 min)
STACK_WAIT_S = 150.0       # pubstatus retry budget (stack boot ~30 s)
FINISH_DEADLINE_S = 120.0  # supplementary post-turn capture only
STOCK_ZERO_RELAX_TURN = 8  # accept shield_stock == 0 from this turn on
# IG_CONVERT genus in the classic ruleset is exactly Coinage
# (capitalization): a "production" that converts shields to gold, which the
# buy validity excludes (city_actions.py:312-314).
EXCLUDED_PRODUCTION = frozenset({"Coinage"})
PREFERRED_UNITS = ("Warriors", "Militia")

EXIT_CONCLUSIVE = 0
EXIT_ERROR = 1
EXIT_STACK_DOWN = 2
EXIT_INCONCLUSIVE = 3


# --- small utilities ---------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """Recursively convert an observation value to JSON-safe data (numpy
    arrays/bit-vectors -> lists via tolist())."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bytes, bool, int, float, type(None))):
        return value
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            return str(value)
    return str(value)


def write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True,
                  ensure_ascii=False, default=str)
        fh.write("\n")
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLogger:
    """Print to stdout AND persist to the run log (HANDOFF: persist run
    stdout/logs as artifacts under artifacts/runs/)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def wait_for_stack(log, deadline_s: float = STACK_WAIT_S) -> bool:
    """Bounded pubstatus retries (the stack boots ~30 s after a container
    restart). Uses urllib: this environment denylists the curl binary."""
    deadline = time.time() + deadline_s
    while True:
        try:
            with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
                if r.status == 200:
                    log("pubstatus 200 — stack up")
                    return True
        except Exception as e:
            if time.time() > deadline:
                log(f"FATAL: pubstatus unreachable: {e!r}")
                return False
            time.sleep(2)


# --- observation readers (plain dicts, same shapes as execute.py) ------------

def _city_dict(obs: Mapping[str, Any], cid: int) -> Optional[Mapping[str, Any]]:
    for k, v in (obs.get("city") or {}).items():
        try:
            if int(k) == cid and isinstance(v, Mapping):
                return v
        except (TypeError, ValueError):
            continue
    return None


def _my_player(obs: Mapping[str, Any],
               player_id: int) -> Optional[Mapping[str, Any]]:
    for k, v in (obs.get("player") or {}).items():
        if not isinstance(v, Mapping):
            continue
        if v.get("player_id") == player_id:
            return v
        try:
            if int(k) == player_id:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _own_units(obs: Mapping[str, Any],
               player_id: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for k, v in (obs.get("unit") or {}).items():
        if not isinstance(v, Mapping) or v.get("owner") != player_id:
            continue
        try:
            uid = int(k)
        except (TypeError, ValueError):
            continue
        out[uid] = {"type": v.get("type_rule_name"),
                    "x": v.get("x"), "y": v.get("y")}
    return out


def _city_actions(info: Mapping[str, Any], cid: int) -> Any:
    acts = ((info or {}).get("available_actions") or {}).get("city") or {}
    for key in (cid, str(cid)):
        if key in acts:
            return acts[key]
    return None


def capture_state(session, *, label: str, cid: int,
                  candidate: Any = None) -> dict[str, Any]:
    """Freeze the fields the decision needs (plus the raw city/player dicts
    for auditability) from the session's cached observation."""
    obs = session.obs
    info = session.info
    turn = int((info or {}).get("turn", 0) or 0)
    pid = int(session.player_id)
    city = _city_dict(obs, cid)
    me = _my_player(obs, pid)
    # session.production_name_map() is name -> (kind, value); invert it for
    # the (kind, value) -> name lookup the city observation carries.
    kv_to_name = {kv: name
                  for name, kv in session.production_name_map().items()}
    prod_name = None
    if city is not None:
        pk, pv = city.get("production_kind"), city.get("production_value")
        if isinstance(pk, int) and isinstance(pv, int) and pk != 0:
            prod_name = kv_to_name.get((pk, pv)) or f"kind{pk}#{pv}"
    city_summary = {
        "id": cid,
        "owner": (city or {}).get("owner"),
        "name": (city or {}).get("name"),
        "size": (city or {}).get("size"),
        "production_kind": (city or {}).get("production_kind"),
        "production_value": (city or {}).get("production_value"),
        "production_name": prod_name,
        "shield_stock": (city or {}).get("shield_stock"),
        "buy_cost": (city or {}).get("buy_cost"),
        "prod_process": (city or {}).get("prod_process"),
        "turns_to_prod_complete": (city or {}).get("turns_to_prod_complete"),
        "surplus_shield": (city or {}).get("surplus_shield"),
        "state": (city or {}).get("state"),
        "improvements": (city or {}).get("improvements"),
    }
    state: dict[str, Any] = {
        "label": label,
        "captured_utc": utcnow(),
        "turn": turn,
        "player_id": pid,
        "username": session.username,
        "city_id": cid,
        "city": city,
        "city_summary": city_summary,
        "treasury_gold": (me or {}).get("gold"),
        "player": me,
        "own_units": _own_units(obs, pid),
        "available_actions_city": _city_actions(info, cid),
    }
    if candidate is not None:
        state["offered_buy_candidate"] = {
            "id": candidate.id,
            "domain": candidate.domain,
            "actor_id": candidate.actor_id,
            "verb": candidate.verb,
            "payload": candidate.payload,
        }
    return state


# --- corroboration + decision ------------------------------------------------

def corroborate(pre: Mapping[str, Any], post: Mapping[str, Any],
                *, buy_cost: int) -> dict[str, Any]:
    """Corroboration of a SUCCESSFUL buy — stock movement alone never
    counts (the stock direction is the very thing under test)."""
    pre_gold = pre.get("treasury_gold")
    post_gold = post.get("treasury_gold")
    gold_delta = (pre_gold - post_gold
                  if isinstance(pre_gold, int) and isinstance(post_gold, int)
                  else None)
    pre_bc = pre["city_summary"].get("buy_cost")
    post_bc = post["city_summary"].get("buy_cost")
    pre_prod = (pre["city_summary"].get("production_kind"),
                pre["city_summary"].get("production_value"))
    post_prod = (post["city_summary"].get("production_kind"),
                 post["city_summary"].get("production_value"))
    new_units = sorted(set(post.get("own_units") or {})
                       - set(pre.get("own_units") or {}))
    signals = {
        # treasury expenditure: the gold drop equals the buy cost exactly
        "treasury_expenditure_gold_delta": gold_delta,
        "treasury_expenditure": (gold_delta == buy_cost
                                 and isinstance(gold_delta, int)),
        # purchase-state change: the city can no longer buy (buy_cost -> 0)
        "purchase_state_change_buy_cost_cleared": (
            isinstance(pre_bc, int) and pre_bc > 0
            and isinstance(post_bc, int) and post_bc == 0),
        # a corresponding server update within the immediate post-state
        "server_update_production_target_changed": pre_prod != post_prod,
        "server_update_new_own_units": new_units,
        "server_update": (pre_prod != post_prod) or bool(new_units),
        # recorded but NEVER sufficient on its own
        "stock_moved": (pre["city_summary"].get("shield_stock")
                        != post["city_summary"].get("shield_stock")),
    }
    corroborated = bool(
        signals["treasury_expenditure"]
        or signals["purchase_state_change_buy_cost_cleared"]
        or signals["server_update"])
    signals["corroborated"] = corroborated
    return signals


def decide_branch(pre: Mapping[str, Any], post: Mapping[str, Any],
                  signals: Mapping[str, Any]) -> tuple[str, str]:
    """The D8b decision table. Returns (branch, explanation)."""
    pid = pre.get("player_id")
    pre_city = pre.get("city")
    post_city = post.get("city")
    identity_ok = bool(
        pre_city is not None and post_city is not None
        and pre["city_summary"].get("owner") == pid
        and post["city_summary"].get("owner") == pid
        and pre.get("city_id") == post.get("city_id"))
    if not identity_ok:
        return ("INCONCLUSIVE",
                "wrong identity: the post-buy city is missing or not owned "
                "by this player")
    pre_stock = pre["city_summary"].get("shield_stock")
    post_stock = post["city_summary"].get("shield_stock")
    if not isinstance(pre_stock, int) or not isinstance(post_stock, int):
        return ("INCONCLUSIVE",
                "shield_stock not observable in the pre/post observation")
    if not signals.get("corroborated"):
        return ("INCONCLUSIVE",
                "ambiguous success: no corroboration of a successful buy "
                "(treasury expenditure / purchase-state change / server "
                "update all absent)")
    if post_stock > pre_stock:
        return ("CONFIRMED",
                "post_stock > pre_stock with corroboration: the pinned "
                "verifier direction (VERIFIED iff post > pre) matches the "
                "live server; no code change")
    if post_stock < pre_stock:
        return ("CORRECTION",
                "post_stock < pre_stock with corroboration: the D8b "
                "conditional correction ACTIVATES (verifier corrected to "
                "the observed direction)")
    return ("INCONCLUSIVE",
            "post_stock == pre_stock: the stock direction is unobservable "
            "for this buy")


# --- the scripted driver (no model call) --------------------------------------

def _kind_of(name: Optional[str], prod_map: Mapping[str, Any]) -> Optional[int]:
    if not name:
        return None
    kv = prod_map.get(name)
    return kv[0] if kv else None


def _tangible(name: Optional[str], prod_map: Mapping[str, Any],
              vut_utype: int, vut_improvement: int) -> bool:
    return (name is not None and name not in EXCLUDED_PRODUCTION
            and _kind_of(name, prod_map) in (vut_utype, vut_improvement))


def _pick_production_target(city, table, prod_map: Mapping[str, Any],
                            vut_utype: int, vut_improvement: int) -> Optional[str]:
    """A tangible production name for the city (units preferred, never
    Coinage), or None when nothing tangible is offered."""
    names: list[str] = []
    for cand in table.candidates:
        if (cand.domain == "city" and str(cand.actor_id) == str(city.id)
                and cand.verb.startswith("produce_")):
            name = cand.payload.get("production")
            if isinstance(name, str) and name:
                names.append(name)
    tangible = [n for n in names
                if _tangible(n, prod_map, vut_utype, vut_improvement)]
    units = [n for n in tangible
             if _kind_of(n, prod_map) == vut_utype]
    for pref in PREFERRED_UNITS:
        if pref in units:
            return pref
    if units:
        return sorted(units)[0]
    if tangible:
        return sorted(tangible)[0]
    return None


def _produce_candidate(table, cid: int, name: str):
    for cand in table.candidates:
        if (cand.domain == "city" and str(cand.actor_id) == str(cid)
                and cand.verb.startswith("produce_")
                and cand.payload.get("production") == name):
            return cand
    return None


def _buy_candidate(table, cid: int):
    for cand in table.candidates:
        if (cand.domain == "city" and cand.verb == "city_buy_production"
                and str(cand.actor_id) == str(cid)):
            return cand
    return None


def play_turn(session, log, exe, extract_snapshot, ExtractOutcome,
              vut_utype: int, vut_improvement: int):
    """Play one scripted turn. Returns a dict:
    {"kind": "terminal", "reason": ...} | {"kind": "continue"} |
    {"kind": "measure", "city": CityView, "candidate": Candidate,
     "raw_city": dict, "table": CandidateTable}."""
    session.refresh()
    table = session.build_candidate_table()
    extraction = extract_snapshot(
        session.obs, session.info,
        my_player_id=session.player_id,
        ruleset=session.ruleset,
        action_probs=session.action_probs)
    if extraction.outcome is not ExtractOutcome.OK or extraction.snapshot is None:
        return {"kind": "terminal",
                "reason": (f"extraction classified "
                           f"{extraction.outcome.value}: {extraction.reason}")}
    snap = extraction.snapshot
    own_cities = [c for c in snap.cities if c.is_own]
    gold = snap.player.gold
    prod_map = session.production_name_map()
    log(f"  turn {snap.turn}: gold={gold} own_cities="
        f"{[(c.id, c.name, c.production, c.shield_stock, c.buy_cost) for c in own_cities]}")

    if not own_cities:
        # Found a city with a settler when possible; otherwise nudge the
        # first own unit (bounded exploration until a city can be founded).
        build = [c for c in table.candidates
                 if c.domain == "unit" and c.verb == "build_city"]
        acted = None
        if build:
            acted = build[0]
        else:
            goto = [c for c in table.candidates
                    if c.domain == "unit" and c.verb.startswith("goto_")]
            if goto:
                acted = goto[0]
        if acted is not None:
            result = exe.execute_candidate(session, acted)
            log(f"  executed {acted.id} (readback {result.readback.status})")
            if result.terminated or result.truncated:
                return {"kind": "terminal",
                        "reason": (f"game ended at {acted.id}: terminated="
                                   f"{result.terminated} truncated="
                                   f"{result.truncated}")}
        return {"kind": "continue"}

    # Ensure the first own city holds a tangible production target
    # (never Coinage); at most one production switch per turn.
    city = own_cities[0]
    if not _tangible(city.production, prod_map, vut_utype, vut_improvement):
        target = _pick_production_target(city, table, prod_map,
                                         vut_utype, vut_improvement)
        cand = _produce_candidate(table, city.id, target) if target else None
        if cand is not None:
            result = exe.execute_candidate(session, cand)
            log(f"  set production: {cand.id} -> {target} "
                f"(readback {result.readback.status})")
            if result.terminated or result.truncated:
                return {"kind": "terminal",
                        "reason": (f"game ended at {cand.id}: terminated="
                                   f"{result.terminated} truncated="
                                   f"{result.truncated}")}
            # The buy check below must run against the FRESH table.
            table = session.build_candidate_table()
            extraction = extract_snapshot(
                session.obs, session.info,
                my_player_id=session.player_id,
                ruleset=session.ruleset,
                action_probs=session.action_probs)
            if extraction.outcome is not ExtractOutcome.OK \
                    or extraction.snapshot is None:
                return {"kind": "terminal",
                        "reason": (f"extraction classified "
                                   f"{extraction.outcome.value}: "
                                   f"{extraction.reason}")}
            snap = extraction.snapshot
            own_cities = [c for c in snap.cities if c.is_own]
            gold = snap.player.gold

    # Buy check: an own city offering city_buy_production (the offered
    # candidate already encodes the vendor's is_action_valid constraints:
    # positive buy_cost, sufficient gold, not founded this turn, not
    # Coinage, not did_buy) with a tangible production and an observable
    # stock direction.
    for city in own_cities:
        buy = _buy_candidate(table, city.id)
        if buy is None:
            continue
        raw = _city_dict(session.obs, city.id)
        buy_cost = raw.get("buy_cost") if raw else None
        stock = raw.get("shield_stock") if raw else None
        affordable = (isinstance(buy_cost, int) and buy_cost > 0
                      and isinstance(gold, int) and gold >= buy_cost)
        tangible = _tangible(city.production, prod_map,
                             vut_utype, vut_improvement)
        observable = (isinstance(stock, int)
                      and (stock >= 1
                           or (stock == 0
                               and snap.turn >= STOCK_ZERO_RELAX_TURN)))
        if affordable and tangible and observable:
            return {"kind": "measure", "city": city, "candidate": buy,
                    "raw_city": raw, "table": table}
    return {"kind": "continue"}


# --- verdict.md ---------------------------------------------------------------

def write_verdict(path: str, ctx: Mapping[str, Any]) -> None:
    lines = []
    add = lines.append
    add("# W3-8 live buy-readback probe — verdict")
    add("")
    add(f"- Run directory: `{ctx['run_dir_name']}`")
    add(f"- Started (UTC): {ctx['started_utc']}")
    add(f"- Finished (UTC): {ctx['finished_utc']}")
    add(f"- Probe script: `scripts/w3_buy_readback_probe.py` "
        f"(sha256 {ctx['script_sha256'][:12]}…)")
    add("")
    add("## Game identity (fresh, isolated, disposable)")
    add(f"- username (unique per run): `{ctx['username']}`")
    add(f"- player_id: {ctx.get('player_id')}, client port: "
        f"{ctx.get('port')}, seed: {ctx['seed']}, aifill: {ctx['aifill']}")
    add("- one env per process; the freeciv-web backend spawned a new game "
        "server for this connection; no existing game was attached to, "
        "joined, or mutated")
    add("")
    add("## Session summary")
    add(f"- Turns played: {ctx['turns_played']} (the buy happened on turn "
        f"{ctx.get('buy_turn')})")
    add(f"- City: id={ctx.get('city_id')} name={ctx.get('city_name')!r} "
        f"owner={ctx.get('city_owner')} size={ctx.get('city_size')}")
    add(f"- Production at buy: {ctx.get('production_name')!r} "
        f"(kind={ctx.get('production_kind')}, value={ctx.get('production_value')})")
    add(f"- Offered buy candidate: `{ctx.get('candidate_id')}` "
        f"payload={ctx.get('candidate_payload')}")
    add(f"- buy_cost: {ctx.get('buy_cost')}; treasury before: "
        f"{ctx.get('pre_gold')} gold")
    add("")
    add("## Observation triple (the decision input)")
    add(f"- **pre_stock = {ctx.get('pre_stock')}**")
    add(f"- **post_stock = {ctx.get('post_stock')}** (immediate post-buy "
        "capture, BEFORE finishing the turn)")
    add(f"- **corroborated = {ctx.get('corroborated')}**")
    sig = ctx.get("signals") or {}
    add(f"  - treasury expenditure: gold {ctx.get('pre_gold')} -> "
        f"{ctx.get('post_gold')} (delta {sig.get('treasury_expenditure_gold_delta')} "
        f"vs buy_cost {ctx.get('buy_cost')}): "
        f"{sig.get('treasury_expenditure')}")
    add(f"  - purchase-state change (buy_cost "
        f"{ctx.get('pre_buy_cost')} -> {ctx.get('post_buy_cost')}): "
        f"{sig.get('purchase_state_change_buy_cost_cleared')}")
    add(f"  - server update (production target changed: "
        f"{sig.get('server_update_production_target_changed')}; new own "
        f"units: {sig.get('server_update_new_own_units')}): "
        f"{sig.get('server_update')}")
    add(f"  - stock moved: {sig.get('stock_moved')} (recorded, NOT "
        "sufficient on its own)")
    if ctx.get("supplementary") is not None:
        sup = ctx["supplementary"]
        add("")
        add("## Supplementary post-turn observation (after finishing the "
            "phase; NOT the decision input)")
        if isinstance(sup, Mapping) and sup.get("error"):
            add(f"- finish-phase capture failed: {sup['error']}")
        else:
            add(f"- turn after finish: {sup.get('finish_turn_after')}; "
                f"shield_stock={sup.get('city_summary', {}).get('shield_stock')}; "
                f"production={sup.get('city_summary', {}).get('production_name')!r}; "
                f"own units={sorted(sup.get('own_units') or {})}")
    add("")
    add("## Raw verifier result (the CURRENT pinned verifier, unmodified)")
    add(f"- status: `{ctx.get('verifier_status')}`")
    add(f"- detail: {ctx.get('verifier_detail')!r}")
    add("")
    add("## Decision-table branch (D8b)")
    add(f"- **{ctx['branch']}**")
    add(f"- {ctx['branch_explanation']}")
    add("")
    add("## Honest statement")
    add(ctx["honest_statement"])
    add("")
    add("## Evidence files")
    for name, digest in (ctx.get("checksums") or {}).items():
        add(f"- `{name}` — sha256 `{digest}`")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--wall-budget-s", type=float, default=WALL_BUDGET_S)
    args = ap.parse_args(argv)

    started_utc = utcnow()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pid = os.getpid()
    run_dir_name = f"{stamp}-{pid}"
    run_dir = os.path.join(RUN_ROOT, run_dir_name)
    # A NEW, never-used directory per run: exist_ok=False, so a retry can
    # never append over an earlier attempt.
    os.makedirs(run_dir, exist_ok=False)
    workdir = os.path.join(run_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    log = RunLogger(os.path.join(run_dir, "run_stdout.log"))
    log(f"[w3-buy-readback] run dir {run_dir_name} (fresh, never-used)")

    # Unique player name per run: the save/username coupling (section 7.1)
    # can never collide with any existing game's player or saves.
    username = (f"w3bp{datetime.now(timezone.utc).strftime('%d%H%M%S')}"
                f"{pid % 100:02d}")
    log(f"[w3-buy-readback] unique username for this run: {username!r}")

    verdict_ctx: dict[str, Any] = {
        "run_dir_name": run_dir_name,
        "started_utc": started_utc,
        "username": username,
        "seed": args.seed,
        "aifill": AIFILL,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
    }
    inconclusive_reason: Optional[str] = None
    probe_error: Optional[str] = None
    branch: Optional[str] = None
    branch_explanation: Optional[str] = None

    if not wait_for_stack(log):
        verdict_ctx.update({
            "finished_utc": utcnow(),
            "turns_played": 0,
            "branch": "INCONCLUSIVE",
            "branch_explanation": ("the local Freeciv stack did not answer "
                                   "pubstatus within the bounded wait; the "
                                   "probe did not start a game"),
            "honest_statement": ("FAILED/INCOMPLETE attempt: the stack was "
                                 "down, so no live observation exists. The "
                                 "verifier was NOT changed and "
                                 "metric-bearing runs stay blocked."),
            "checksums": {},
        })
        write_verdict(os.path.join(run_dir, "verdict.md"), verdict_ctx)
        log("W3-8 DECISION BRANCH: INCONCLUSIVE (stack down)")
        log.close()
        return EXIT_STACK_DOWN

    # The adapter imports lazily (civrealm touches the network at import
    # time); only now that the stack is up.
    from lechonk.adapter import execute as exe
    from lechonk.adapter.obs_extract import (
        VUT_IMPROVEMENT,
        VUT_UTYPE,
        ExtractOutcome,
        extract_snapshot,
    )
    from lechonk.adapter.session import AdapterSession

    session = None
    close_status = "not_attempted"
    measurement: Optional[dict[str, Any]] = None
    turns_played = 0
    try:
        session = AdapterSession(workdir=workdir, seed=args.seed,
                                 username=username, aifill=AIFILL)
        t0 = time.time()
        session.start()
        log(f"reset ok in {time.time() - t0:.1f}s turn={session.turn} "
            f"player_id={session.player_id} username={session.username} "
            f"port={session.port}")
        if session.username != username:
            raise RuntimeError(
                f"username was renamed ({session.username!r} != "
                f"{username!r}); self_play=False must raise on congestion, "
                f"never silently rename")

        deadline = time.time() + args.wall_budget_s
        while turns_played < args.max_turns and time.time() < deadline:
            outcome = play_turn(session, log, exe, extract_snapshot,
                                ExtractOutcome, VUT_UTYPE, VUT_IMPROVEMENT)
            if outcome["kind"] == "terminal":
                inconclusive_reason = (
                    f"the game ended before a buyable city was reached: "
                    f"{outcome['reason']}")
                log(f"terminal: {outcome['reason']}")
                break
            if outcome["kind"] == "measure":
                measurement = outcome
                break
            exe.finish_phase(session)
            turns_played += 1
            log(f"  finished phase -> turn {session.turn}")
        else:
            if measurement is None and inconclusive_reason is None:
                inconclusive_reason = (
                    f"no own city offered a buyable city_buy_production "
                    f"within the bounded session (max_turns="
                    f"{args.max_turns}, wall budget {args.wall_budget_s}s)")

        if measurement is not None:
            city = measurement["city"]
            buy = measurement["candidate"]
            cid = int(city.id)
            pre = capture_state(session, label="pre_buy", cid=cid,
                                candidate=buy)
            t0 = time.monotonic()
            # Exactly ONE city_buy_production, via the normal bounded step.
            result = exe.execute_candidate(session, buy)
            step_elapsed = time.monotonic() - t0
            # Immediate post-state, BEFORE finishing the turn.
            post = capture_state(session, label="post_buy_immediate", cid=cid)
            log(f"BUY {buy.id} on city {cid} (turn {pre['turn']}): "
                f"readback={result.readback.status} "
                f"elapsed={step_elapsed:.2f}s terminated="
                f"{result.terminated} truncated={result.truncated}")
            buy_cost = pre["city_summary"]["buy_cost"]
            signals = corroborate(pre, post, buy_cost=buy_cost)
            branch, branch_explanation = decide_branch(pre, post, signals)
            pre_stock = pre["city_summary"]["shield_stock"]
            post_stock = post["city_summary"]["shield_stock"]
            log(f"OBSERVATION TRIPLE: pre_stock={pre_stock} "
                f"post_stock={post_stock} corroborated="
                f"{signals['corroborated']}")
            log(f"signals: {json.dumps(_jsonable(signals), sort_keys=True)}")
            log(f"DECISION BRANCH: {branch} — {branch_explanation}")

            # Supplementary post-turn capture (after finishing the phase;
            # NOT the decision input — documents the server-side update).
            supplementary: Any = None
            try:
                finish = exe.finish_phase(session,
                                          deadline=FINISH_DEADLINE_S)
                supplementary = capture_state(
                    session, label="post_turn_supplementary", cid=cid)
                supplementary["finish_turn_after"] = int(
                    (finish.info or {}).get("turn", 0) or 0)
            except Exception as exc:
                supplementary = {"error": repr(exc)}
                log(f"supplementary post-turn capture failed: {exc!r}")

            operation = {
                "probe": "w3-buy-readback (D8a live observation)",
                "run_dir": run_dir_name,
                "started_utc": started_utc,
                "captured_utc": utcnow(),
                "game_identity": {
                    "username": session.username,
                    "player_id": session.player_id,
                    "port": session.port,
                    "seed": args.seed,
                    "aifill": AIFILL,
                    "fresh_isolated_game": True,
                    "existing_games_touched": False,
                },
                "turns_played": turns_played,
                "buy_turn": pre["turn"],
                "city": pre["city_summary"],
                "offered_buy_candidate": pre.get("offered_buy_candidate"),
                "step": {
                    "action": list(result.action),
                    "elapsed_s": round(step_elapsed, 3),
                    "status": result.status,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "turn_before": result.turn_before,
                    "turn_after": result.turn_after,
                    "elapsed_s_result": round(result.elapsed_s, 3),
                },
                "raw_verifier_result": {
                    "status": result.readback.status,
                    "detail": result.readback.detail,
                    "evidence": result.readback.evidence,
                    "execution_detail": result.detail,
                },
                "observation_triple": {
                    "pre_stock": pre_stock,
                    "post_stock": post_stock,
                    "corroborated": signals["corroborated"],
                },
                "corroboration_signals": signals,
                "supplementary_post_turn": supplementary,
                "decision_branch": branch,
                "decision_explanation": branch_explanation,
            }
            write_json(os.path.join(run_dir, "pre.json"), pre)
            write_json(os.path.join(run_dir, "post.json"), post)
            write_json(os.path.join(run_dir, "operation.json"), operation)
            close_status = session.close()
            log(f"guarded close status: {close_status}")

            verdict_ctx.update({
                "finished_utc": utcnow(),
                "player_id": session.player_id,
                "port": session.port,
                "turns_played": turns_played,
                "buy_turn": pre["turn"],
                "city_id": cid,
                "city_name": pre["city_summary"]["name"],
                "city_owner": pre["city_summary"]["owner"],
                "city_size": pre["city_summary"]["size"],
                "production_name": pre["city_summary"]["production_name"],
                "production_kind": pre["city_summary"]["production_kind"],
                "production_value": pre["city_summary"]["production_value"],
                "candidate_id": buy.id,
                "candidate_payload": buy.payload,
                "buy_cost": buy_cost,
                "pre_gold": pre["treasury_gold"],
                "post_gold": post["treasury_gold"],
                "pre_buy_cost": pre["city_summary"]["buy_cost"],
                "post_buy_cost": post["city_summary"]["buy_cost"],
                "pre_stock": pre_stock,
                "post_stock": post_stock,
                "corroborated": signals["corroborated"],
                "signals": signals,
                "supplementary": supplementary
                if isinstance(supplementary, Mapping) else None,
                "verifier_status": result.readback.status,
                "verifier_detail": result.readback.detail,
                "branch": branch,
                "branch_explanation": branch_explanation,
            })
            if branch == "CONFIRMED":
                verdict_ctx["honest_statement"] = (
                    "CONCLUSIVE: a successful live buy was corroborated "
                    "(treasury expenditure / purchase-state change / server "
                    "update) and the observed shield_stock direction is "
                    "post > pre, matching the pinned verifier. The finding "
                    "is closed; NO code change; metric-bearing runs "
                    "unblock.")
            elif branch == "CORRECTION":
                verdict_ctx["honest_statement"] = (
                    "CONCLUSIVE: a successful live buy was corroborated and "
                    "the observed shield_stock direction is post < pre, "
                    "contradicting the pinned verifier. The pre-approved D8b "
                    "correction activates: the verifier, its pinned test, "
                    "and a README discontinuity note land together with "
                    "this probe script in one commit.")
            else:
                verdict_ctx["honest_statement"] = (
                    "INCONCLUSIVE: the observation does not conclusively "
                    "answer the stock direction. The verifier was NOT "
                    "changed and metric-bearing runs stay blocked.")
    except Exception as exc:
        import traceback
        log(f"FATAL: {exc!r}")
        log(traceback.format_exc())
        probe_error = f"probe error: {exc!r}"
        inconclusive_reason = inconclusive_reason or probe_error
    finally:
        if session is not None and not session.closed:
            try:
                close_status = session.close()
                log(f"guarded close status: {close_status}")
            except Exception as exc:
                log(f"close failed: {exc!r}")

    # Evidence finalization: verdict.md + sha256 checksums.
    if branch is None:
        branch = "INCONCLUSIVE"
        branch_explanation = (inconclusive_reason or
                              "the probe ended without a measurement")
        verdict_ctx["honest_statement"] = (
            "FAILED/INCOMPLETE attempt: " + branch_explanation +
            ". The verifier was NOT changed and metric-bearing runs stay "
            "blocked.")
    verdict_ctx.setdefault("finished_utc", utcnow())
    verdict_ctx.setdefault("turns_played", turns_played)
    verdict_ctx["branch"] = branch
    verdict_ctx["branch_explanation"] = branch_explanation

    if measurement is not None:
        pre_path = os.path.join(run_dir, "pre.json")
        post_path = os.path.join(run_dir, "post.json")
        op_path = os.path.join(run_dir, "operation.json")
        verdict_path = os.path.join(run_dir, "verdict.md")
        verdict_ctx["checksums"] = {
            "pre.json": sha256_file(pre_path),
            "post.json": sha256_file(post_path),
            "operation.json": sha256_file(op_path),
        }
        write_verdict(verdict_path, verdict_ctx)
        checksums = dict(verdict_ctx["checksums"])
        checksums["verdict.md"] = sha256_file(verdict_path)
        write_json(os.path.join(run_dir, "checksums.json"), checksums)
        log(f"evidence written: {run_dir}")
        log(f"checksums: {json.dumps(checksums, sort_keys=True)}")
    else:
        # FAILED/INCOMPLETE attempt: record the honest verdict (and any
        # partial evidence) without claiming an observation.
        write_verdict(os.path.join(run_dir, "verdict.md"), verdict_ctx)
        log(f"evidence written (FAILED/INCOMPLETE attempt): {run_dir}")

    if measurement is not None:
        log(f"W3-8 OBSERVATION TRIPLE: pre_stock="
            f"{verdict_ctx.get('pre_stock')} post_stock="
            f"{verdict_ctx.get('post_stock')} corroborated="
            f"{verdict_ctx.get('corroborated')}")
    log(f"W3-8 DECISION BRANCH: {branch}")
    log(f"W3-8 EVIDENCE DIR: {run_dir}")
    log.close()

    if branch in ("CONFIRMED", "CORRECTION"):
        return EXIT_CONCLUSIVE
    if probe_error is not None:
        return EXIT_ERROR
    if "stack was down" in str(verdict_ctx.get("branch_explanation", "")):
        return EXIT_STACK_DOWN
    return EXIT_INCONCLUSIVE


if __name__ == "__main__":
    sys.exit(main())
