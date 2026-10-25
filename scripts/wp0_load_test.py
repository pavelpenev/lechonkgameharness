"""WP0 P8 test: load an exported checkpoint after a stack restart and verify identity.

Acceptance: exported save -> docker restart -> load via debug.load_game ->
identity verification (username, turn number, player attachment, state fields).
"""
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, 'artifacts', 'runs', 'wp0-load')
os.makedirs(RUNDIR, exist_ok=True)
os.chdir(RUNDIR)

from civrealm.configs import fc_args
from civrealm.envs import FreecivBaseEnv

SAVE = 'myagent_T11_2026-10-24-10_24'
EXPECT_TURN = 11

fc_args['self_play'] = False
fc_args['debug.load_game'] = SAVE
fc_args['server_timeout'] = 300
fc_args['begin_turn_timeout'] = 900

env = FreecivBaseEnv()
t0 = time.time()
obs, info = env.reset()
pid = env.get_playerid()
username = env.get_username()
turn = info['turn']
own = {cid: c for cid, c in obs.get('city', {}).items() if c.get('owner') == pid}
units = {uid: u for uid, u in obs.get('unit', {}).items() if u.get('owner') == pid}
print(f'[load] ok in {time.time() - t0:.1f}s')
print(f'[identity] username={username} playerid={pid} turn={turn}')
print(f'[state] gold={obs["player"][pid].get("gold")} cities={len(own)} '
      f'units={len(units)} researching={obs["player"][pid].get("researching")}')

checks = {
    'username': username == 'myagent',
    'turn': turn == EXPECT_TURN,
    'player_attached': pid is not None and obs.get('player', {}).get(pid) is not None,
    'state_restored': len(own) == 2 and len(units) == 5,
}
print('[checks]', checks)
print('[verdict]', 'PASS' if all(checks.values()) else 'FAIL')
