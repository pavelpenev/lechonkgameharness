# SPDX-License-Identifier: GPL-3.0-or-later
"""WP1 package A live smoke: one fresh FreecivBaseEnv reset -> extractor.

Behavioral evidence for the observation extractor on REAL data (offline fog
fixtures live in tests/). Per design-plan-freeciv-v1.1.md §7.1:

- deterministic seeds: randomly_generate_seeds=False, explicit
  mapseed/gameseed/agentseed (reset(seed=) only sets mapseed+agentseed, so
  gameseed is set in fc_args directly), self_play=False (upstream default
  True silently renames the user on congestion), raised server_timeout /
  begin_turn_timeout (30 s defaults truncate slow runs);
- a FRESH reset spawns its own new game server; the turn-11 checkpoint and any
  running game are not touched; the fresh server quits idle (~20 s,
  --quitidle 20) after the client disconnects — expected and fine;
- env.close() is deliberately NOT called: CivController.close() waits on a
  game-over message in an unbounded loop (plan §7.1 line 99,
  civ_controller.py:578-583). The process exits; the server quits idle.
- run from a dedicated working directory: upstream creates logs/,
  recordings/, game_scores/ in CWD.

Outputs (persisted under artifacts/runs/wp1-extract/):
- smoke_stdout.log   — this run's stdout (via tee at the shell)
- snapshot_turn<N>.json — the extracted fog-gated Snapshot, rendered
- run_meta.json      — seeds, ports, fog-consistency check results, verdict
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, "artifacts", "runs", "wp1-extract")
WORKDIR = os.path.join(RUNDIR, "workdir")  # dedicated CWD for upstream side effects
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)  # upstream writes logs/, recordings/, game_scores/ into CWD

SEED = 20261024  # distinct from the WP0 spike seed (42)
PUBSTATUS = "http://localhost:8080/pubstatus"

print(f"[wp1-smoke] repo={REPO}")
print(f"[wp1-smoke] cwd={os.getcwd()} (dedicated workdir)")

# --- wait for the stack (hardened startup: bounded pubstatus retries) ------
deadline = time.time() + 120
while True:
    try:
        with urllib.request.urlopen(PUBSTATUS, timeout=10) as r:
            if r.status == 200:
                print("[wp1-smoke] pubstatus 200 — stack up")
                break
    except Exception as e:
        if time.time() > deadline:
            print(f"[wp1-smoke] FATAL: pubstatus unreachable: {e!r}")
            sys.exit(2)
        time.sleep(2)

# --- import civrealm only now (import touches the network; plan §7.1) -------
from civrealm.configs import fc_args  # noqa: E402
from civrealm.envs import FreecivBaseEnv  # noqa: E402

from lechonk.adapter.obs_extract import (  # noqa: E402
    ExtractOutcome,
    RulesetNames,
    extract_snapshot,
)
from lechonk.core.schemas import SNAPSHOT_TOKEN_BUDGET  # noqa: E402

# --- deterministic fc_args (plan §7.1 adapter operational notes) -------------
fc_args["self_play"] = False
fc_args["aifill"] = 2  # pinned scenario (WP0 spike configuration)
fc_args["debug.randomly_generate_seeds"] = False
fc_args["debug.mapseed"] = SEED
fc_args["debug.gameseed"] = SEED
fc_args["debug.agentseed"] = SEED
fc_args["server_timeout"] = 300
fc_args["begin_turn_timeout"] = 900

env = FreecivBaseEnv()
t0 = time.time()
obs, info = env.reset(seed=SEED)
print(f"[wp1-smoke] reset ok in {time.time() - t0:.1f}s turn={info['turn']} "
      f"obs_domains={sorted(obs.keys())}")

pid = env.get_playerid()
username = env.get_username()
port = env.get_port()
print(f"[wp1-smoke] playerid={pid} username={username} client_port={port}")

# --- ruleset display tables (duck-typed from the ruleset controller) --------
rule_ctrl = env.civ_controller.rule_ctrl
n_impr = int(rule_ctrl.ruleset_control["num_impr_types"])
ruleset = RulesetNames(
    terrains={tid: t["name"] for tid, t in rule_ctrl.terrains.items()},
    unit_types={uid: u["name"] for uid, u in rule_ctrl.unit_types.items()},
    improvements={iid: i["name"] for iid, i in rule_ctrl.improvements.items()},
    nations={nid: (n.get("name") or n.get("adjective") or f"nation#{nid}")
            for nid, n in rule_ctrl.nations.items()},
    tech_costs={tid: (t.get("cost") if isinstance(t.get("cost"), int) else None)
                for tid, t in rule_ctrl.techs.items()},
    # city_state.py adds num_impr_types to production_value for VUT_UTYPE (6);
    # VUT_IMPROVEMENT (3) values are improvement ids directly.
    city_production_names=(
        {(6, uid + n_impr): u["name"]
         for uid, u in rule_ctrl.unit_types.items()}
        | {(3, iid): i["name"]
           for iid, i in rule_ctrl.improvements.items()}
    ),
)
print(f"[wp1-smoke] ruleset tables: {len(ruleset.terrains)} terrains, "
      f"{len(ruleset.unit_types)} unit types, {len(ruleset.improvements)} "
      f"improvements, {len(ruleset.tech_costs)} techs, "
      f"{len(ruleset.city_production_names)} production names")

# --- combat odds (plan §7.2: unit_data[uid].action_prob[dir] = {action: prob})
action_probs = {}
unit_data = getattr(env.civ_controller.unit_ctrl.prop_actions, "unit_data", {})
for uid, focus in unit_data.items():
    ap = getattr(focus, "action_prob", None)
    if ap:
        action_probs[int(uid)] = {int(d): dict(probs) for d, probs in ap.items()}
print(f"[wp1-smoke] action_probs for {len(action_probs)} unit(s) "
      f"(combat odds exposed: {bool(action_probs)})")

# --- extract ----------------------------------------------------------------
t1 = time.time()
extraction = extract_snapshot(
    obs, info,
    terminated=False, truncated=False,
    my_player_id=pid,
    ruleset=ruleset,
    action_probs=action_probs or None,
)
print(f"[wp1-smoke] extract_snapshot in {time.time() - t1:.2f}s "
      f"outcome={extraction.outcome.value} reason={extraction.reason!r}")
assert extraction.outcome is ExtractOutcome.OK, extraction
snap = extraction.snapshot
assert snap is not None

# --- fog-consistency assertions on REAL data --------------------------------
status_grid = obs["map"]["status"]
if hasattr(status_grid, "tolist"):
    status_grid = status_grid.tolist()
height = len(status_grid)
width = len(status_grid[0]) if height else 0


def tile_status(x, y):
    if 0 <= y < height and 0 <= x < width:
        return int(status_grid[y][x])
    return 0


obs_units = obs.get("unit", {})
obs_cities = obs.get("city", {})
obs_players = obs.get("player", {})
obs_techs = obs.get("tech", {})

checks = {}


def check(name, cond, detail=""):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"fog-consistency check failed: {name} {detail}"


check("outcome_ok", extraction.outcome is ExtractOutcome.OK)
check("snapshot_present", snap is not None)
check("player_id_matches", snap.player.player_id == pid,
      f"snapshot={snap.player.player_id} env={pid}")

# map trichotomy on real data
counts = snap.map_view.tile_counts
check("tile_counts_sum", counts["visible"] + counts["last_known"]
      + counts["unexplored"] == width * height,
      f"{counts} vs {width}x{height}")
raw_statuses = {int(s) for row in status_grid for s in row}
check("status_trichotomy_values", raw_statuses <= {0, 1, 2},
      f"raw statuses={sorted(raw_statuses)}")

# every foreign unit's visibility marker matches the raw tile status
foreign = [u for u in snap.units if not u.is_own]
check("foreign_units_visibility_matches_map",
      all((u.visibility == "visible") == (tile_status(u.x, u.y) == 2)
          for u in foreign),
      f"{len(foreign)} foreign unit(s)")

# every own unit in the observation is in the snapshot (own units never elided
# at this scale; also verifies no -1 placeholder leak for foreign units)
own_ids_obs = {uid for uid, u in obs_units.items() if u.get("owner") == pid}
own_ids_snap = {u.id for u in snap.units if u.is_own}
check("own_units_all_present", own_ids_obs == own_ids_snap,
      f"obs={len(own_ids_obs)} snap={len(own_ids_snap)}")
check("foreign_units_no_placeholder_leak",
      all(u.moves_left is None for u in foreign),
      "foreign moves_left must be unknown (None), never -1")
rendered = snap.render()
check("no_raw_minus_one_placeholders",
      '"moves_left":-1' not in rendered and '"size":-1' not in rendered)

# foreign cities: mutable fields gated on tile visibility
bad_cities = []
for c in snap.cities:
    if c.is_own:
        continue
    st = tile_status(c.x, c.y)
    if st == 2:
        if c.visibility != "visible" or c.size is None or c.owner_id is None:
            bad_cities.append((c.id, "visible-but-unknown", c.size))
    else:
        if c.visibility != "last_known" or c.size is not None \
                or c.owner_id is not None:
            bad_cities.append((c.id, "fogged-but-stale", c.size))
check("foreign_cities_fog_gated", not bad_cities, f"bad={bad_cities}")

# research display cross-checked against the tech domain (plan §7.2)
rt = snap.player.research
if rt.current_target_id is not None:
    tech_entry = obs_techs.get(rt.current_target_id, {})
    check("research_target_crosscheck",
          bool(tech_entry.get("is_researching"))
          and tech_entry.get("name") == rt.current_target,
          f"target={rt.current_target!r} tech_domain={tech_entry.get('name')!r}")
else:
    check("research_target_crosscheck", True, "no active research target")

# research candidates are exactly the PREREQS_KNOWN techs
cand_ids = {c.tech_id for c in snap.research_candidates}
expected_ids = {tid for tid, t in obs_techs.items()
                if t.get("inv_state") == 1}
check("research_candidates_match_tech_domain", cand_ids == expected_ids,
      f"candidates={sorted(cand_ids)} expected={sorted(expected_ids)}")

# token budget
check("token_budget", snap.estimated_tokens() <= SNAPSHOT_TOKEN_BUDGET,
      f"{snap.estimated_tokens()} <= {SNAPSHOT_TOKEN_BUDGET}")

# embassy-gated intel respected: the snapshot must never surface intel that
# upstream's gating withholds (player_state.py:98-130). Hearsay leaks
# government/gold/research only when > 0; tax/science/luxury and the tech
# list only with an embassy (detectable via filled tech_<id> invention
# fields). Zeroed packet fields for opponents are placeholders, not knowledge.
_TECH_FIELD = re.compile(r"^tech_\d+$")
NO_INTEL = "no embassy intel: gold/research/government unknown"
intel_bad = []
for d in snap.diplomacy:
    p = obs_players.get(d.counterpart_id, {})
    intel = d.intel or ""
    has_embassy = any(_TECH_FIELD.match(k) and v is not None
                      for k, v in p.items())
    if not intel.startswith(("hearsay intel: ", "embassy intel: ")):
        if intel != NO_INTEL:
            intel_bad.append((d.counterpart_id, f"malformed intel {intel!r}"))
        continue
    kind, _, claims = intel.partition(": ")
    if kind == "embassy" and not has_embassy:
        intel_bad.append((d.counterpart_id, "embassy intel without embassy"))
    if "tax" in claims and not has_embassy:
        intel_bad.append((d.counterpart_id, "tax claimed without embassy"))
    m = re.search(r"gold (-?\d+)", claims)
    if m:
        claimed = int(m.group(1))
        backed = (isinstance(p.get("gold"), int) and p["gold"] == claimed
                  and (has_embassy or claimed > 0))
        if not backed:
            intel_bad.append((d.counterpart_id,
                              f"gold claim {claimed} not backed by upstream"))
    m = re.search(r"government (\S+)", claims)
    if m and not (isinstance(p.get("government"), int)
                  and (has_embassy or p["government"] > 0)):
        intel_bad.append((d.counterpart_id, "government claim not backed"))
    if "researching" in claims and not (isinstance(p.get("researching"), int)
                                        and (has_embassy
                                             or p["researching"] > 0)):
        intel_bad.append((d.counterpart_id, "research claim not backed"))
check("embassy_intel_respected", not intel_bad,
      f"{len(snap.diplomacy)} counterpart(s), bad={intel_bad}")

# --- persist artifacts -------------------------------------------------------
snap_path = os.path.join(RUNDIR, f"snapshot_turn{snap.turn}.json")
with open(snap_path, "w", encoding="utf-8") as f:
    f.write(rendered)
    f.write("\n")
print(f"[wp1-smoke] snapshot written: {snap_path}")

try:
    vendored_head = subprocess.run(
        ["git", "-C", os.path.join(REPO, "vendor", "civrealm"),
         "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception:
    vendored_head = None

meta = {
    "run": "wp1-extract-smoke",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": SEED,
    "fc_args_overrides": {
        "self_play": False, "aifill": 2,
        "debug.randomly_generate_seeds": False,
        "debug.mapseed": SEED, "debug.gameseed": SEED,
        "debug.agentseed": SEED,
        "server_timeout": 300, "begin_turn_timeout": 900,
    },
    "client_port": port,
    "player_id": pid,
    "username": username,
    "vendored_civrealm_head": vendored_head,
    "outcome": extraction.outcome.value,
    "reason": extraction.reason,
    "turn": snap.turn,
    "map_size": [width, height],
    "tile_counts": counts,
    "player": {"name": snap.player.name, "gold": snap.player.gold,
               "research_target": rt.current_target},
    "research_candidates": [c.name for c in snap.research_candidates],
    "cities": [{"id": c.id, "name": c.name, "own": c.is_own,
                "visibility": c.visibility} for c in snap.cities],
    "units": [{"id": u.id, "type": u.type_name, "own": u.is_own,
               "visibility": u.visibility} for u in snap.units],
    "diplomacy": [{"counterpart": d.counterpart_name, "state": d.state,
                   "intel": d.intel} for d in snap.diplomacy],
    "snapshot_tokens_estimated": snap.estimated_tokens(),
    "token_budget": SNAPSHOT_TOKEN_BUDGET,
    "omitted": snap.omitted.model_dump(),
    "action_probs_exposed": bool(action_probs),
    "fog_checks": checks,
    "env_close": "skipped deliberately — CivController.close() contains an "
                 "unbounded game-over wait loop (plan §7.1 line 99); the "
                 "fresh game server quits idle (~20 s, --quitidle 20)",
    "snapshot_file": snap_path,
    "verdict": "PASS",
}
meta_path = os.path.join(RUNDIR, "run_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
print(f"[wp1-smoke] meta written: {meta_path}")

print("[wp1-smoke] snapshot summary:")
print(f"  turn={snap.turn} player={snap.player.name} gold={snap.player.gold} "
      f"research={rt.current_target!r} turns_left={rt.turns_left}")
print(f"  research_candidates={[c.name for c in snap.research_candidates]}")
print(f"  cities={[(c.name, c.visibility) for c in snap.cities]}")
print(f"  units={[(u.id, u.type_name, u.visibility) for u in snap.units]}")
print(f"  map={width}x{height} counts={counts} grid={'yes' if snap.map_view.grid else 'no'}")
print(f"  diplomacy={[(d.counterpart_name, d.state) for d in snap.diplomacy]}")
print(f"  tokens={snap.estimated_tokens()}/{SNAPSHOT_TOKEN_BUDGET} "
      f"omitted={snap.omitted.summary!r}")
print(f"[wp1-smoke] messages={len(snap.messages)}")
print("[wp1-smoke] VERDICT: PASS — fog-consistent Snapshot on real data")
# env.close() intentionally not called (unbounded game-over wait loop).
sys.exit(0)
