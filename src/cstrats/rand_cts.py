"""
Random strategy in continuous context setting.
"""

import numpy as np

from cstrats.cts_opt import ContinuousOpt

class RandOpt(ContinuousOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'rand'

    def _determine_next_query(self):
        """Pick the next query uniformly at random."""
        ctx = self._get_ctx_candidates(1)[0]
        lows, highs = zip(*self.act_domain)
        act = np.random.uniform(lows, highs)
        return np.append(ctx, act)

    def _make_new_gp_fitter(self):
        """Override this method since we do not pick based on GPs."""
        return

    def _draw_next_gp(self):
        """Override this method since we do not pick based on GPs."""
        return
