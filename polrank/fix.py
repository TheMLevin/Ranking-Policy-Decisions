import json
import time

import numpy as np
from tqdm import tqdm

from elements.envs import run_env_with, Get_stats
from elements.policies import MixedPol, RandomRankingPol, TripPol
from utils.timing import sec_to_str


def fix(logger, score_types=None):
    """ Set up interpolation, with score types either
    from logger or overridden by provided ones """
    config = logger.config
    score_types = score_types if score_types is not None else config['score_types']
    update_logs = logger.update_logs
    return fix_states(
        config['env'],
        config['pol'],
        config['pol_d'],
        config['cond'],
        logger.data['scores'][0],
        logger.data['interpol'][0],
        score_types,
        config['n_inc'],
        config['n_test'],
        update_logs,
    )

def fix_states(env, pol, pol_d, cond, rankings, interpol, score_types, n_inc, n_test, update_logs):
    """ Given two policies, interpolates between them
    and saves outcomes for each interpolation."""
    start = time.time()

    bounds = []
    for st in [x for x in score_types if x != 'rand']:
        ys = interpol[st][1]
        score = (max(ys) - min(ys)) * .8 + min(ys)
        i = np.where(np.array(ys) >= score)[0][0]
        bounds.append((int(interpol[st][0][i]), st))
        # bounds.append((1 - interpol[st][4][i], int(interpol[st][0][i]), st))

    # _, bound, st = min(bounds)
    bound, st = min(bounds)

    state_ranking = [s for s, sc in rankings[st]]

    results = {}
    '''for i in range(2):
        muts = []
        rws = []
        print("\nBeginning fix for ranking type:", i)
        # This goes through whole ranking, and then does complete policy
        for r in tqdm(range(bound)):
            not_mut = state_ranking[:r] + state_ranking[r+1:bound if i else len(state_ranking)]
            mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)

            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test, rand=st=='rand')

            # Results for complete policy are kept under index '-1', at end of list
            rws.append([state_ranking[r], np.mean(tot_rs)])
            muts.append(mut_props)
            # print("Done interpolation with {}/{} mutations".format(i, len(state_ranking)),
            #     end='\r' if i < len(state_ranking) else '\n')
        results[f'fix_{i}'] = sorted(rws, key=lambda x: x[1])
        results[f'fix_{i}'] += list(zip(state_ranking[bound:], [results[f'fix_{i}'][-1][1]] * (len(state_ranking) - bound)))
        print(muts)'''

    for i in range(1, 3):
        muts = []
        rws = []
        print("\nBeginning trip of size:", i)
        # This goes through whole ranking, and then does complete policy
        for r in tqdm(range(bound)):
            not_mut = state_ranking[:r] + state_ranking[r+1:len(state_ranking)]
            mpol = TripPol(pol, pol_d, not_mut, i, abst=env.abst)

            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test, rand=st=='rand')

            # Results for complete policy are kept under index '-1', at end of list
            rws.append([state_ranking[r], np.mean(tot_rs)])
            muts.append(mut_props)
            # print("Done interpolation with {}/{} mutations".format(i, len(state_ranking)),
            #     end='\r' if i < len(state_ranking) else '\n')
        results[f'trip_{i+2}'] = sorted(rws, key=lambda x: x[1])
        results[f'trip_{i+2}'] += list(zip(state_ranking[bound:], [results[f'trip_{i+2}'][-1][1]] * (len(state_ranking) - bound)))
        print(muts)

    end = time.time()
    log = {
        'fix_time': sec_to_str(end - start)
    }
    update_logs(log)

    return results

def test_pol(env, pol, cond, n_test, rand=False):
    """ Test a polciy n_test times, keeping track
    of outcomes """
    tot_rs, passes, not_muts, mut_props = [], [], [], []
    stats = Get_stats(cond)

    for _ in range(n_test):
        if rand:
            pol.shuffle_rank()
        run_env_with(env, pol, stats)
        tot_r, pss, mut_n, steps = stats.get_stats(reset=True)
        tot_rs.append(tot_r)
        passes.append(pss)
        mut_props.append(mut_n/steps)
        not_muts.append(steps-mut_n)

    return tot_rs, passes, mut_n, not_muts
