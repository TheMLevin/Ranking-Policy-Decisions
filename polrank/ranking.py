import time

import numpy as np
from tqdm import tqdm

from elements.envs import run_env_with, Get_stats
from elements.policies import MixedPol, RandomRankingPol
from utils.timing import sec_to_str

INTERPOL_COLS = [
    "Number of Active States in Policy",
    "Reward Average",
    "Reward Variance",
    "Fraction Passing Executions",
    "Fraction Actions Taken Inactively",
    "Number of Active Actions Taken",
]

def rank(logger):
    """ Set up interpolation, with score types either
    from logger or overridden by provided ones """
    config = logger.config
    update_logs = logger.update_logs
    return rank_groups(
        config['env'],
        config['pol'],
        config['pol_d'],
        config['cond'],
        logger.data['groups'][0],
        logger.data['ranks'][0],
        logger.data['counts2'][0][2],
        config['n_inc'],
        config['n_test'],
        update_logs,
    )

def rank_groups(env, pol, pol_d, cond, groups, ranks, all_states, n_inc, n_test, update_logs):
    """ Given two policies, interpolates between them
    and saves outcomes for each interpolation. In last step does
    completely unmutated policy, and puts it at n_inc past the last index """
    start = time.time()

    if ranks is None:
        results = []
        print(f"\nBeginning ranking of {len(groups)} groups:")
        for group in tqdm(groups):
            mpol = MixedPol(pol, pol_d, group, abst=env.abst)
            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
            results.append((group, np.mean(tot_rs)))
        results = sorted(results, key=lambda x: x[1], reverse=True)
    else:
        results = ranks

    ranked, _ = list(zip(*results))
    ranked += tuple([all_states])
    inds, avgs, vrs, chks, mut_ps, n_muts = [], [], [], [], [], []
    print(f"\nBeginning cumulative group interpolation")
    for i in tqdm(range(len(ranked))):
        not_mut = set(sum(ranked[:i+1], []))
        mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)
        tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
        inds.append(len(not_mut))
        avgs.append(np.mean(tot_rs))
        vrs.append(np.var(tot_rs))
        chks.append(np.mean(passes))
        mut_ps.append(np.mean(mut_props))
        n_muts.append(np.mean(not_muts))

    mpol = MixedPol(pol, pol_d, 'all', abst=env.abst)
    tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
    inds.append(-1)
    avgs.append(np.mean(tot_rs))
    vrs.append(np.var(tot_rs))
    chks.append(np.mean(passes))
    mut_ps.append(np.mean(mut_props))
    n_muts.append(np.mean(not_muts))

    end = time.time()

    log = {
        'rank_and_interpol_time': sec_to_str(end - start)
    }
    update_logs(log)

    return results, {'group': (inds, avgs, vrs, chks, mut_ps, n_muts)}

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

    return tot_rs, passes, mut_props, not_muts
