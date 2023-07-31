import random
import time

import numpy as np

from utils.timing import sec_to_str

def count(logger, n_runs=None):
    """ Get counts for states, either as many as
    logger requests, or n_runs if overriding """
    config = logger.config
    n_runs = n_runs if n_runs is not None else config['n_runs']
    update_logs = logger.update_logs
    update_cond = logger.update_cond
    auto_cond = config['cond_name'][:5] == 'score' and config['cond_name'][5:] == '_auto'
    return get_counts(
        n_runs,
        config['env'],
        config['pol'],
        config['pol_d'],
        config['mut_prob'],
        config['cond'],
        update_logs,
        update_cond,
        auto_cond)

def get_counts(n_runs, env, pol, pol_d, mut_prob, cond, update_logs, update_cond, auto_cond=False):
    """ Get successes and failure counts for each state """
    all_start = time.time()

    mut_probs = sorted([mut_prob, 1 - mut_prob])

    counts = [[],[],set()]
    succs = 0
    stepss = []
    for group in range(2):
        tot_rews = []
        print(f"\nBeginning counting {group}")
        print("Done with 0/{} counting runs".format(n_runs), end='\r')
        for i in range(n_runs):
            start = time.time()

            mut_states, norm_states, succ, tot_rew, steps = run_env_with_muts(env, pol, pol_d, mut_probs[group], cond)
            succs += 1 if succ else 0
            tot_rews.append(tot_rew)
            stepss.append(steps)
            update_counts(counts, mut_states, norm_states, succ, tot_rew, group, auto_cond)

            end = time.time()
            est_time_left = sec_to_str(((end - all_start) / (i+1)) * (n_runs - i+1))

            # (general) passes, abstract states, total time < estimated time left | (episode) reward, steps, time
            print("Done with {}/{} counting runs, p:{} as:{} tt:{}<{} | r:{} s:{} t:{}".format(
                i+1, n_runs, succs, len(counts[group]), sec_to_str(end-all_start), est_time_left, tot_rew, steps, sec_to_str(end-start)))
        all_end = time.time()

        if auto_cond:
            thresh = np.mean(tot_rews)
            counts = flex_to_counts(counts, thresh)
            succs += sum(rew >= thresh for rew in tot_rews)
            if not group:
                cond_name = f'score{thresh}'

    if auto_cond:
        update_cond(cond_name)

    '''logs = {
        'counting_abs_states': len(counts),
        # 'counting_rews': tot_rews,
        'counting_rews_mean': np.mean(tot_rews),
        # 'counting_steps': stepss,
        'counting_steps_mean': np.mean(stepss),
        'counting_succs': "{}/{}".format(succs, n_runs),
        'counting_time': sec_to_str(all_end - all_start),
        'counting_auto_cond?': auto_cond,
    }
    update_logs(logs)'''

    counts[2] = list(counts[2])

    return counts

def run_env_with_muts(env, pol, pol_d, mut_prob, cond):
    """Run environment, making mutations to pol_d according to
    mut_prob, and return mutations, visited states, and condition"""
    mut_states = set()
    norm_states = set()
    s, _ = env.reset()
    ss = env.abst(s)
    state_seq, action_seq, rew_seq = [s], [], []
    steps = 0
    done = False
    while not done:
        if ss in mut_states:
            a = pol_d(state_seq, action_seq, rew_seq)
        elif ss in norm_states:
            a = pol(state_seq, action_seq, rew_seq)
        elif random.random() > mut_prob:
            a = pol(state_seq, action_seq, rew_seq)
            norm_states.add(ss)
        else:
            a = pol_d(state_seq, action_seq, rew_seq)
            mut_states.add(ss)
        s, r, done, *_ = env.step(a)
        ss = env.abst(s)

        state_seq.append(s)
        action_seq.append(a)
        rew_seq.append(r)
        steps += 1

    succ = cond(state_seq, action_seq, rew_seq)

    return mut_states, norm_states, succ, sum(rew_seq), steps

def update_counts(counts, mut_states, norm_states, succ, tot_rew, group, auto_cond):
    if auto_cond or succ == group:
        counts[group].append((tuple(norm_states if group else mut_states), tot_rew))
    counts[2] |= mut_states
    counts[2] |= norm_states

def flex_to_counts(flex_counts, thresh):
    counts = [[],[], []]
    for group, runs in enumerate(flex_counts[:2]):
        for states, rew in runs:
            if (rew >= thresh) == group:
                counts[group].append((states, rew))
    counts[2] = flex_counts[2]
    return counts
