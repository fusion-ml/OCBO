"""
Improve optimization strategies for joint GPs.
"""
from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from strategies.joint_opt import JointOpt
from util.misc_util import sample_grid, build_gp_posterior

class JointMEI(JointOpt):


    def decide_next_query(self):
        """Given a particular gp, decide best point to query.
        Returns: Function index and point to query next.
        """
        build_gp_posterior(self.gps[0])
        return self._find_best_query()

    @staticmethod
    def get_opt_method_name():
        """Get the name of the method."""
        return 'joint-mei'

    def _find_best_query(self):
        """Draw a joint sample over the entire space, use this to select
        the context and the action.
        Returns: Index of best context/task and the best point.
        """
        gp = self.gps[0]
        max_ei_values = []
        max_ei_points = []
        for idx in range(len(self.fcns)):
            f_idx = idx
            curr_best = self.curr_best[self.f_names[f_idx]][-1][0]
            ei_pts = sample_grid([self.f_locs[f_idx]], self.domains[0],
                                 self.options.max_opt_evals)
            mu, sigma = gp.eval(ei_pts, include_covar=True)
            sigma = np.sqrt(sigma.diagonal().ravel())
            norm_diff = (mu - curr_best) / sigma
            eis = sigma * (norm_diff * normal_distro.cdf(norm_diff) \
                  + normal_distro.pdf(norm_diff))
            ei_val = np.max(eis)
            pt_idx = np.argmax(eis)
            ei_pt = ei_pts[pt_idx, len(self.f_locs[0]):]
            max_ei_values.append(ei_val)
            max_ei_points.append(ei_pt)
        query_idx = np.argmax(max_ei_values)
        best_pt = max_ei_points[query_idx]

        return query_idx, best_pt
