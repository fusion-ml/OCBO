"""
Class of strategies that pick based on improvement on previously seen strats.
"""

from argparse import Namespace
import numpy as np

from dragonfly.utils.option_handler import get_option_specs
from strategies.multi_opt import MultiOpt

from util.misc_util import ei_acq, build_gp_posterior

class MEI(MultiOpt):

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        for gp in self.gps:
            build_gp_posterior(gp)
        query_ctx, query_pt = self._find_best_eval()
        return query_ctx, query_pt

    @staticmethod
    def get_opt_method_name():
        """Return the name of the method."""
        return 'mei'

    def _find_best_eval(self):
        """Given a particular gp, decide best point to query.
        Returns: Point index of best task/context and the point.
        """
        best_idx, best_pt = None, None
        most_gain = float('-inf')
        for idx in range(len(self.fcns)):
            gp = self.gps[idx]
            domain = self.domains[idx]
            curr_max_val = self.curr_best[self.f_names[idx]][-1][0]
            pt, val = ei_acq(gp, domain, self.options.max_opt_evals,
                             curr_max_val)
            gain = val
            if gain > most_gain:
                most_gain = gain
                best_idx = idx
                best_pt = pt
        return best_idx, best_pt
