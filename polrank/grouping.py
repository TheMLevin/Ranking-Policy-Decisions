import random
from functools import reduce
from scipy.sparse.linalg import svds

import numpy as np


def group(logger):
    config = logger.config
    # N = config['num_groups']
    # group_size = config['group_size']
    k = config['num_sigma']
    group_sizes = [int(x) for x in config['group_sizes'].split(',')]
    counts = logger.data['counts2'][0]
    print([len(x) for x in counts])

    Ms = [[],[],[]]
    all_states = counts[2]
    for tp in range(2):
        clusters, scores = list(zip(*counts[tp]))
        maxs = max(scores)
        mins = min(scores)
        for cluster, score in counts[tp]:
            v = (((score - mins) / (maxs - mins)) ** 2 - (0 if tp else 1)) * np.array([x in cluster for x in range(len(all_states))])
            Ms[tp].append(v)
            Ms[2].append(v)

    groups = []
    for M in Ms:
        M = np.array(M)
        M /= np.emath.logn(5, sum(M != 0) / 2 + 5)
        pca, *_ = svds(M.T, k)
        for composite in pca.T:
            composite = abs(composite)
            ids = np.argsort(composite)
            for size in group_sizes:
                groups.append([int(x) for x in ids[-size:]])
    '''for tp in range(2):
        clusters, scores = list(zip(*logger.data['counts2'][0][tp]))
        maxs = max(scores)
        ws = [(score if tp else 2 * maxs - score) for score in scores]
        for i in range(N):
            curr = random.choices(clusters, weights=ws, k=group_size)
            groups.append(reduce(lambda a, b: a & b, [set(cluster) for cluster in curr]))
    groups = [g for g in groups if g]
    while len(groups) > N:
        groups = sorted(groups, key=lambda x: (len(x), random.random()))
        groups = [groups[0] | groups[1]] + groups[2:]'''

    return groups
