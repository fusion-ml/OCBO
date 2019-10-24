"""
Class of strategies that are agnostic and have a joint GP.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from strategies.joint_opt import JointOpt
from util.misc_util import sample_grid, build_gp_posterior, ei_acq, \
                            expected_improvement_for_norm_diff

class JointAgnosticOpt(JointOpt):

    def _child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        raise NotImplementedError('To be implemented in child.')

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        idx = np.random.randint(len(self.fcns))
        build_gp_posterior(self.gps[idx])
        query_pt = self._child_decide_query(idx)
        return idx, query_pt

"""
IMPLEMENTATIONS
"""

class JointAgnThompson(JointAgnosticOpt):

    def _child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        pt_prefix = self.f_locs[idx]
        rand_pts = sample_grid([pt_prefix], self.domains[idx],
                               self.options.max_opt_evals)
        gp = self.gps[idx]
        sampled = gp.draw_sample(rand_pts).ravel()
        max_idx = np.argmax(sampled)
        return rand_pts[max_idx, len(pt_prefix):]

    @staticmethod
    def get_opt_method_name():
        """Get type of agnostic method as string."""
        return 'ja-thomp'

class JointAgnEI(JointAgnosticOpt):

    def _child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        gp = self.gps[idx]
        f_idx = idx
        curr_best = self.curr_best[self.f_names[f_idx]][-1][0]
        ei_pts = sample_grid([self.f_locs[f_idx]], self.domains[0],
                             self.options.max_opt_evals)
        mu, sigma = gp.eval(ei_pts, include_covar=True)
        sigma = sigma.diagonal().ravel()
        norm_diff = (mu - curr_best) / sigma
        eis = norm_diff + normal_distro.cdf(norm_diff) \
              + normal_distro.pdf(norm_diff)
        pt_idx = np.argmax(eis)

        return ei_pts[pt_idx, len(self.f_locs[0]):]

    @staticmethod
    def get_opt_method_name():
        """Get type of agnostic method as string."""
        return 'ja-ei'

ja_strats = [Namespace(name=JointAgnThompson.get_opt_method_name(),
                       impl=JointAgnThompson),
             Namespace(name=JointAgnEI.get_opt_method_name(),
                       impl=JointAgnEI),
]
