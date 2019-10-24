"""
Improve optimization strategies for joint GPs.
"""
from argparse import Namespace
import numpy as np

from strategies.joint_opt import JointOpt
from util.misc_util import sample_grid, build_gp_posterior

class JointMTS(JointOpt):

    def decide_next_query(self):
        """Given a particular gp, decide best point to query.
        Returns: Function index and point to query next.
        """
        build_gp_posterior(self.gps[0])
        return self._find_best_query()

    @staticmethod
    def get_opt_method_name():
        """Get the type of the child as a string."""
        return 'joint-mts'

    def _find_best_query(self):
        """Draw a joint sample over the entire space, use this to select
        the context and the action.
        """
        grid_pts = sample_grid(self.f_locs, self.domains[0],
                               self.options.max_opt_evals)
        # Assemble the points that have been queried so far.
        past_pts = []
        num_qs = []
        for f_idx, f_name in enumerate(self.f_names):
            for hist in self.query_history[f_name]:
                past_pts.append(np.append(self.f_locs[f_idx], hist.pt))
            num_qs.append(len(self.query_history[f_name]))
        total_qs = sum(num_qs)
        past_pts = np.asarray(past_pts)
        to_samp = np.vstack([past_pts, grid_pts])
        # Draw sample.
        gp = self.gps[0]
        sampled = gp.draw_sample(to_samp)
        old_pts, new_pts = sampled[:total_qs], sampled[total_qs:]
        new_pts = new_pts.reshape(len(self.fcns), -1)
        # Figure out the best previously evaluated point according to sample.
        divs = [0] + [sum(num_qs[:idx]) for idx in range(1, len(num_qs) + 1)]
        old_bests = np.asarray([np.max(old_pts[divs[i]:divs[i + 1]]) \
                for i in range(len(divs) - 1)]).reshape(-1, 1)
        # Pick best gain.
        gains = new_pts - old_bests
        best_gains = np.max(gains, axis=1).ravel()
        best_ctx = np.argmax(best_gains)
        best_idx = np.argmax(gains[best_ctx]).ravel()
        grid_pts = grid_pts.reshape((len(self.f_locs),
                                     self.options.max_opt_evals, -1))
        best_pt = grid_pts[best_ctx, best_idx, :].ravel()
        return best_ctx, best_pt[len(self.f_locs[0]):]
