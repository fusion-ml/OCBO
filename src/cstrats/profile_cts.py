"""
Thompson Sampling strategies for continuous context.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from cstrats.cts_opt import ContinuousOpt
from dragonfly.utils.option_handler import get_option_specs
from util.misc_util import sample_grid, uniform_draw, knowledge_gradient

prof_args = [\
        get_option_specs('num_profiles', False, 50,
            'Number of contexts to consider picking from.'),
        get_option_specs('profile_evals', False, 100,
            'Number of evaluations for each context to determine max.'),
]

class ProfileOpt(ContinuousOpt):

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.num_profiles = options.num_profiles
        self.profile_evals = options.profile_evals

    def _determine_next_query(self):
        # Get the contexts to test out.
        ctxs = self._get_ctx_candidates(self.num_profiles)
        # For each context...
        best_pt, best_imp = None, float('-inf')
        for ctx in ctxs:
            # Find the best context and give its improvement.
            pt, imp = self._get_ctx_improvement(ctx)
            if imp > best_imp:
                best_pt, best_imp = pt, imp
        # Return the best context and action.
        return best_pt

    def _get_ctx_improvement(self, ctx):
        """Get the improvement for the context.
        Args:
            ctx: ndarray characterizing the context.
        Returns: Best action and the improvement it provides.
        """
        raise NotImplementedError('Abstract Method')

class ProfileEI(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'pei'

    def _get_ctx_improvement(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.min([np.max(means), np.max(self.y_data)])
        stds = np.sqrt(covmat.diagonal().ravel())
        norm_diff = (means - best_post) / stds
        eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff))
        ei_val = np.max(eis)
        ei_pt = act_set[np.argmax(eis)]
        return ei_pt, ei_val

class CMTSPM(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'cmts-pm'

    def _get_ctx_improvement(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.min([np.max(means), np.max(self.y_data)])
        sample = self.gp.draw_sample(means=means, covar=covmat).ravel()
        gain = np.max(sample) - best_post
        best_pt = act_set[np.argmax(sample)]
        return best_pt, gain

class ContinuousMultiTaskTS(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'cmts'

    def _get_ctx_improvement(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.argmax(means)
        sample = self.gp.draw_sample(means=means, covar=covmat).ravel()
        gain = np.max(sample) - sample[best_post]
        best_pt = act_set[np.argmax(sample)]
        return best_pt, gain

prof_strats = [Namespace(impl=ProfileEI, name=ProfileEI.get_strat_name()),
               Namespace(impl=CMTSPM, name=CMTSPM.get_strat_name()),
               Namespace(impl=ContinuousMultiTaskTS,
                         name=ContinuousMultiTaskTS.get_strat_name()),
]
