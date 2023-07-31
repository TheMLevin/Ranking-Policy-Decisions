import random
import json

import numpy as np
import torch

from counting2 import count
from grouping import group
from ranking import rank
import utils.cli as cli
from utils.logging import Logger
from visualisation.graphing import draw_interpol_results
from visualisation.histograms import score_histogram
from visualisation.see_env import run_and_save

def main():
    args = cli.parse_args()

    if not args.no_det:
        np.random.seed(args.env_seed)
        random.seed(args.env_seed)
        torch.manual_seed(args.env_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ### Logger setup ###
    logger = Logger(args.fileloc, args.load_loc)
    logger.init_config(args)
    logger.load_results()

    counts_do = {
        'redo': args.redo_all or not logger.is_done('counts2')
    }

    make_groups_do = {
        'redo': counts_do['redo'] or not logger.is_done('groups')
    }

    rank_groups_do = {
        'redo': True or make_groups_do['redo'] or not logger.is_done('ranks')
    }

    if counts_do['redo']:
        counts = count(logger)
        logger.update_counts(counts, combine=False, ind=2)
        logger.dump_results()
        logger.dump_config()

    if make_groups_do['redo']:
        groups = group(logger)
        logger.update_groups(groups)
        logger.dump_results()
        logger.dump_config()

    if rank_groups_do['redo']:
        ranks, interpol = rank(logger)
        logger.update_ranks(ranks)
        logger.update_interpolation(interpol)
        logger.dump_results()
        logger.dump_config()

    draw_interpol_results(logger, ['group'], 0, [1], x_fracs=True, y_fracs=True, smooth=False,
                          x_name='States Restored (%)', y_names=['Original Reward (%)'], combine_sbfl=False)
    draw_interpol_results(logger, ['group'], 4, [1], y_fracs=True,
                          trans_x=lambda x: 1 - x, x_name="Policy's Action Taken (% of Steps)",
                          y_names=['Original Reward (%)'], smooth=False, combine_sbfl=False)


if __name__ == '__main__':
    main()