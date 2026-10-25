"""WP0 spike: scripted headless game against the vendored CivRealm stack.

Acceptance (design-plan-freeciv-v1.1.md WP0): headless reset; scripted
settle/move/set-research/set-production; 10 phase advances; per-step
read-back evidence printed. Run only while the freeciv-web container is up.
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, 'artifacts', 'runs', 'wp0-spike')
os.makedirs(RUNDIR, exist_ok=True)
os.chdir(RUNDIR)

from civrealm.configs import fc_args
from civrealm.envs import FreecivBaseEnv

SEED = 42

fc_args['self_play'] = False
fc_args['aifill'] = 2
fc_args['debug.randomly_generate_seeds'] = False
fc_args['debug.mapseed'] = SEED
fc_args['debug.gameseed'] = SEED
fc_args['debug.agentseed'] = SEED
fc_args['server_timeout'] = 300
fc_args['begin_turn_timeout'] = 900

env = FreecivBaseEnv()
t0 = time.time()
obs, info = env.reset(seed=SEED)
print(f'[reset] ok in {time.time() - t0:.1f}s turn={info["turn"]} '
      f'playerid={env.get_playerid()} obs_domains={sorted(obs.keys())}')

pid = env.get_playerid()


def pick(ctrl, pred):
    for actor_id, action_dict in info.get('available_actions', {}).get(ctrl, {}).items():
        for name, valid in action_dict.items():
            if valid and pred(name):
                return (ctrl, actor_id, name)
    return None


def step(action, label):
    global obs, info
    t = time.time()
    obs, reward, term, trunc, info = env.step(action)
    print(f'  [{label}] {action[2] if action else "None"} '
          f'-> {time.time() - t:.1f}s term={term} trunc={trunc}')
    return term or trunc


def own_cities():
    return {cid: c for cid, c in obs.get('city', {}).items() if c.get('owner') == pid}


def own_units():
    return {uid: u for uid, u in obs.get('unit', {}).items() if u.get('owner') == pid}


evidence = {'research': False, 'production': False, 'move': False, 'settle': False}

for i in range(10):
    turn = info['turn']
    acts = info.get('available_actions', {})
    n_valid = {c: sum(sum(1 for v in d.values() if v) for d in acts.get(c, {}).values())
               for c in acts}
    print(f'[turn {turn}] gold={obs["player"][pid].get("gold")} '
          f'cities={len(own_cities())} units={len(own_units())} valid={n_valid}')

    # research: first available research_tech action
    a = pick('tech', lambda n: n.startswith('research_tech'))
    if a:
        before = obs['player'][pid].get('researching')
        if step(a, 'research'):
            break
        after = obs['player'][pid].get('researching')
        print(f'    read-back: researching {before} -> {after}')
        if after != before:
            evidence['research'] = True

    # production: first produce_* action on an own city
    a = pick('city', lambda n: n.startswith('produce_'))
    if a:
        city = obs['city'].get(a[1], {})
        before = (city.get('production_kind'), city.get('production_value'))
        if step(a, 'production'):
            break
        city = obs['city'].get(a[1], {})
        after = (city.get('production_kind'), city.get('production_value'))
        print(f'    read-back: city {a[1]} production {before} -> {after}')
        if before != after:
            evidence['production'] = True

    # settle: build_city if legal
    a = pick('unit', lambda n: n == 'build_city')
    if a:
        if step(a, 'settle'):
            break
        evidence['settle'] = len(own_cities()) > 0

    # move: first goto_* action on an own unit
    a = pick('unit', lambda n: n.startswith('goto_'))
    if a:
        unit = obs['unit'].get(a[1], {})
        before = (unit.get('x'), unit.get('y'))
        if step(a, 'move'):
            break
        unit = obs['unit'].get(a[1], {})
        after = (unit.get('x'), unit.get('y'))
        print(f'    read-back: unit {a[1]} pos {before} -> {after}')
        if before != after:
            evidence['move'] = True

    # end the phase/turn
    if step(None, 'end-turn'):
        break
    print(f'[advance] now turn={info["turn"]}')

print('[evidence]', evidence)
print('[verdict]',
      'PASS' if evidence['research'] and evidence['production'] and evidence['move']
      else 'INCOMPLETE (see evidence above)')
sys.exit(0)
