"""
Randomly select context then optimize.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from cstrats.cts_opt import ContinuousOpt
from dragonfly.utils.option_handler import get_option_specs
from util.misc_util import sample_grid, uniform_draw, knowledge_gradient

agn_args = [\
        get_option_specs('agn_evals', False, 100,
            'Number of evaluations for each context to determine max.'),
]

class AgnosticOpt(ContinuousOpt):

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.agn_evals = options.agn_evals

    def _determine_next_query(self):
        # Get the contexts to test out.
        ctx = self._get_ctx_candidates(1)[0]
        pt = self._get_best_action(ctx)
        return pt

    def _get_best_action(self, ctx):
        """Get the improvement for the context.
        Args:
            ctx: ndarray characterizing the context.
        Returns: Best point.
        """
        raise NotImplementedError('Abstract Method')

class AgnEI(AgnosticOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'ei'

    def _get_best_action(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.agn_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.max(means)
        variances = covmat.diagonal().ravel()
        norm_diff = (means - best_post) / variances
        eis = norm_diff + normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff)
        ei_pt = act_set[np.argmax(eis)]
        return ei_pt

class AgnTS(AgnosticOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'ts'

    def _get_best_action(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.agn_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        sample = self.gp.draw_sample(means=means, covar=covmat).ravel()
        best_pt = act_set[np.argmax(sample)]
        return best_pt

agn_strats = [Namespace(impl=AgnEI, name=AgnEI.get_strat_name()),
              Namespace(impl=AgnTS, name=AgnTS.get_strat_name())]
