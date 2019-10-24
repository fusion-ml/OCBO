"""
Class of strategies that pick based on improvement to mean.
"""

from argparse import Namespace
import numpy as np

from dragonfly.utils.option_handler import get_option_specs
from strategies.multi_opt import MultiOpt
from scipy.stats import norm as normal_distro

from util.misc_util import thomp_acq, ei_acq, sub_ei_acq, ttei_acq, \
                            expected_improvement_for_norm_diff, ucb_acq, \
                            build_gp_posterior


class MTS(MultiOpt):

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        for gp in self.gps:
            build_gp_posterior(gp)
        # Find the best mean values for each gp.
        best_f, best_pt, best_gain = None, None, float('-inf')
        queries = self._get_queried_pts()
        for f_idx, f_name in enumerate(self.f_names):
            gp = self.gps[f_idx]
            f_qs = queries[f_name]
            # Assemble points to draw sample from.
            low, high = zip(*self.domains[f_idx])
            rand_pts = np.random.uniform(low, high,
                                         (self.options.max_opt_evals, len(low)))
            samp_pts = np.vstack([f_qs, rand_pts])
            samp_vals = gp.draw_sample(samp_pts=samp_pts).ravel()
            max_prev = np.max(samp_vals[:len(f_qs)])
            best_new_idx = np.argmax(samp_vals[len(f_qs):]) + len(f_qs)
            gain = samp_vals[best_new_idx] - max_prev
            if gain > best_gain:
                best_f = f_idx
                best_pt = samp_pts[best_new_idx]
                best_gain = gain
        return best_f, best_pt

    def _get_queried_pts(self):
        """Returns dict mapping f_name -> list of points."""
        queries = {}
        for f_name in self.f_names:
            queries[f_name] = np.asarray([qi.pt
                for qi in self.query_history[f_name]])
        return queries

    @staticmethod
    def get_opt_method_name():
        return 'mts'
