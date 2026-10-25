"""WP0-6 helper: load the T11 checkpoint and HOLD the myagent connection.

The plain wp0_load_test.py exits after verification, dropping the connection,
which lets the publite2 server quit (--quitidle 20) and the live game is lost
(that is how the turn-11 game disappeared at ~10:26 after wp0-5). This variant
keeps the connection open while the Playwright observer attaches, so the
running turn-11 game stays alive for the zero-perturbation observation.

Run: .venv/bin/python scripts/wp0_load_hold.py [hold_seconds]
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, 'artifacts', 'runs', 'wp0-load')
os.makedirs(RUNDIR, exist_ok=True)
os.chdir(RUNDIR)

from civrealm.configs import fc_args
from civrealm.envs import FreecivBaseEnv

SAVE = 'myagent_T11_2026-10-24-10_24'
EXPECT_TURN = 11
HOLD_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 600

fc_args['self_play'] = False
fc_args['debug.load_game'] = SAVE
fc_args['server_timeout'] = 300
fc_args['begin_turn_timeout'] = 900

env = FreecivBaseEnv()
t0 = time.time()
obs, info = env.reset()
pid = env.get_playerid()
username = env.get_username()
port = env.get_port()
turn = info['turn']
own = {cid: c for cid, c in obs.get('city', {}).items() if c.get('owner') == pid}
units = {uid: u for uid, u in obs.get('unit', {}).items() if u.get('owner') == pid}
print(f'[load] ok in {time.time() - t0:.1f}s', flush=True)
print(f'[identity] username={username} playerid={pid} turn={turn} port={port}', flush=True)
print(f'[state] gold={obs["player"][pid].get("gold")} cities={len(own)} '
      f'units={len(units)} researching={obs["player"][pid].get("researching")}', flush=True)

checks = {
    'username': username == 'myagent',
    'turn': turn == EXPECT_TURN,
    'player_attached': pid is not None and obs.get('player', {}).get(pid) is not None,
    'state_restored': len(own) == 2 and len(units) == 5,
}
print('[checks]', checks, flush=True)
print('[verdict]', 'PASS' if all(checks.values()) else 'FAIL', flush=True)

if all(checks.values()):
    print(f'[hold] holding myagent connection on civserver port {port} '
          f'for {HOLD_SECONDS}s (Ctrl-C / kill to release)', flush=True)
    deadline = time.time() + HOLD_SECONDS
    while time.time() < deadline:
        time.sleep(15)
        print(f'[hold] alive t+{int(deadline - time.time())}s remaining, '
              f'turn={info.get("turn", "?")}', flush=True)
print('[hold] released', flush=True)
