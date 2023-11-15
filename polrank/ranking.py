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

GROUPSS = ['group-','group+','group+-']

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
        len(logger.data['counts'][0]),
        config['n_inc'],
        config['n_test'],
        update_logs,
        logger.data['ranks'][0]
    )

def rank_groups(env, pol, pol_d, cond, groupss, ranks, all_states, space, n_inc, n_test, update_logs, old):
    """ Given two policies, interpolates between them
    and saves outcomes for each interpolation. In last step does
    completely unmutated policy, and puts it at n_inc past the last index """
    start = time.time()


    results = {}
    rankings = {}
    for name, groups in zip(GROUPSS, groupss):
        result = []
        print(f"\nBeginning ranking of {name}:")
        for group in tqdm(groups):
            not_mut = [all_states[i] for i in group]
            mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)
            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
            result.append((not_mut, np.mean(tot_rs)))
        results[name] = sorted(result, key=lambda x: x[1], reverse=True)

        ranked, _ = list(zip(*result))
        ranked = ([],) + ranked + (all_states,)
        inds, avgs, vrs, chks, mut_ps, n_muts = [], [], [], [], [], []
        print(f"\nBeginning cumulative group interpolation of {name}")
        for i in tqdm(range(len(ranked))):
            not_mut = set(sum(ranked[:i+1], []))
            if inds and inds[-1] == len(not_mut):
                continue
            mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)
            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
            inds.append(min(len(not_mut), space))
            avgs.append(np.mean(tot_rs))
            vrs.append(np.var(tot_rs))
            chks.append(np.mean(passes))
            mut_ps.append(np.mean(mut_props))
            n_muts.append(np.mean(not_muts))

        inds.append(-1)
        avgs.append(np.mean(tot_rs))
        vrs.append(np.var(tot_rs))
        chks.append(np.mean(passes))
        mut_ps.append(np.mean(mut_props))
        n_muts.append(np.mean(not_muts))
        rankings[name] = (inds, avgs, vrs, chks, mut_ps, n_muts)

        # result = []
        # print(f"\nBeginning reverse ranking of {name}:")
        # for group in tqdm(groups):
        #     not_mut = [elem for i, elem in enumerate(all_states) if i not in group]
        #     mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)
        #     tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
        #     result.append(([all_states[i] for i in group], np.mean(tot_rs)))
        # results[f'{name}_causal'] = sorted(result, key=lambda x: x[1])
        #
        # '''print(old.keys())
        # results = old
        # result = old[f'{name}_causal']'''
        #
        # ranked, _ = list(zip(*result))
        # ranked = (all_states,) + ranked[::-1] + ([],)
        # print([len(x) for x in ranked])
        # inds, avgs, vrs, chks, mut_ps, n_muts = [], [], [], [], [], []
        # print(f"\nBeginning reverse cumulative group interpolation of {name}")
        # for i in tqdm(range(len(ranked))):
        #     not_mut = set(all_states) - set(sum(ranked[i:], []))
        #     if inds and inds[-1] == len(not_mut):
        #         continue
        #     mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)
        #     tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test)
        #     inds.append(max(len(not_mut), space))
        #     avgs.append(np.mean(tot_rs))
        #     vrs.append(np.var(tot_rs))
        #     chks.append(np.mean(passes))
        #     mut_ps.append(np.mean(mut_props))
        #     n_muts.append(np.mean(not_muts))
        # inds.append(-1)
        # avgs.append(np.mean(tot_rs))
        # vrs.append(np.var(tot_rs))
        # chks.append(np.mean(passes))
        # mut_ps.append(np.mean(mut_props))
        # n_muts.append(np.mean(not_muts))
        # rankings[f'{name}_causal'] = (inds, avgs, vrs, chks, mut_ps, n_muts)

    end = time.time()

    log = {
        'rank_and_interpol_time': sec_to_str(end - start)
    }
    update_logs(log)

    return results, rankings

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
