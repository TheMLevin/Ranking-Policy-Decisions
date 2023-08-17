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

    counts = []
    all_states = {}
    succs = 0
    stepss = []
    for group in range(2):
        count = []
        tot_rews = []
        print(f"\nBeginning counting {group}")
        print("Done with 0/{} counting runs".format(n_runs), end='\r')
        for i in range(n_runs):
            start = time.time()

            mut_states, norm_states, succ, tot_rew, steps = run_env_with_muts(env, pol, pol_d, mut_probs[group], cond)
            succs += 1 if succ else 0
            tot_rews.append(tot_rew)
            stepss.append(steps)
            update_counts(count, all_states, mut_states, norm_states, succ, tot_rew, group, auto_cond)

            end = time.time()
            est_time_left = sec_to_str(((end - all_start) / (i+1)) * (n_runs - i+1))

            # (general) passes, abstract states, total time < estimated time left | (episode) reward, steps, time
            print("Done with {}/{} counting runs in {}, p:{} as:{} tt:{}<{} | r:{} s:{} t:{}".format(
                i+1, n_runs, group, succs, len(all_states), sec_to_str(end-all_start), est_time_left, tot_rew, steps, sec_to_str(end-start)))
        all_end = time.time()

        if auto_cond:
            thresh = np.median(tot_rews)
            count = flex_to_counts(count, thresh, group)
            succs += sum(rew >= thresh for rew in tot_rews)
            if not group:
                cond_name = f'score{thresh}'

        counts.append(count)

    '''if auto_cond:
        update_cond(cond_name)'''

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

    counts.append(list(all_states.keys()))

    return counts

def run_env_with_muts(env, pol, pol_d, mut_prob, cond):
    """Run environment, making mutations to pol_d according to
    mut_prob, and return mutations, visited states, and condition"""
    mut_states = set()
    norm_states = set()
    succs = []
    rew_seqs = []
    stepss = []
    for n in range(5):
        print(f"Run {n+1}/5")
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

        succs.append(succ)
        rew_seqs.append(sum(rew_seq))
        stepss.append(steps)


    return mut_states, norm_states, int(np.mean(succs) + .5), np.mean(rew_seqs), np.mean(stepss)

def update_counts(counts, all_states, mut_states, norm_states, succ, tot_rew, group, auto_cond):
    for state in mut_states | norm_states:
        all_states[state] = 0
    if auto_cond or succ == group:
        counts.append((tuple(list(all_states.keys()).index(state) for state in (norm_states if group else mut_states)), tot_rew))

def flex_to_counts(flex_counts, thresh, tp):
    counts = []
    for states, rew in flex_counts:
        if (rew >= thresh) == tp:
            counts.append((states, rew))
    return counts
