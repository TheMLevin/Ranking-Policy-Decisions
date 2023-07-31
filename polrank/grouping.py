import random
from functools import reduce
import scipy

import numpy as np


def group(logger, pca=False):
    config = logger.config
    N = config['num_groups']
    group_size = config['group_size']

    groups = []
    if pca:
        group_sizes = [int(x) for x in config['group_sizes'].split(',')]

    for tp in range(2):
        clusters, scores = list(zip(*logger.data['counts2'][0][tp]))
        maxs = max(scores)
        ws = [(score if tp else 2 * maxs - score) for score in scores]
        for i in range(N):
            curr = random.choices(clusters, weights=ws, k=group_size)
            groups.append(reduce(lambda a, b: a & b, [set(cluster) for cluster in curr]))
    groups = [g for g in groups if g]
    while len(groups) > N:
        groups = sorted(groups, key=lambda x: (len(x), random.random()))
        groups = [groups[0] | groups[1]] + groups[2:]

    return [list(g) for g in groups]
