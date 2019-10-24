"""
Improve optimization correlated over chunks of joint GP.
"""
from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from dragonfly.utils.option_handler import get_option_specs

from strategies.joint_opt import JointOpt
from util.misc_util import sample_grid, build_gp_posterior, knowledge_gradient,\
        draw_all_related

corr_args = [
    get_option_specs('run_ei_after', False, float('inf'),
       'How long to until switch to EI.'),
    get_option_specs('num_candidates', False, 100,
        'How many candidate points to consider for evaluation.'),
    get_option_specs('num_eval_pts', False, 100,
        'How many points to use for evaluation.'),
]

class CorrelatedOpt(JointOpt):

    def __init__(self, f_infos, options, pre_tuned_gps=None, rn_info=None):
        """Constructor:
        Args:
            fcns: List of Namespaces containing function info.
            f_locs: Location of functions in Euclidean space.
            options: Namespace, needs tune_every for how often GP HPs should
                be tuned. Never tunes if 0.
            pre_tuned_gps: List of pre-tuned GPs corresponding to functions.
            rn_info: Tuple where the first element is the grid of points used
                to evaluate risk neutral policies represented as a List of
                ndarray of points for each function. The second is list of
                optimal values for the grid.
                If None and risk_neutral mode is turned on then a new grid
                will be generated.
        """
        self.round_robin_idx = 0
        super(CorrelatedOpt, self).__init__(f_infos, options, pre_tuned_gps,
                                            rn_info)

    def _child_decide_query(self):
        """Given a particular gp, decide best point to query.
        Returns: Function index and point to query next.
        """
        raise NotImplementedError('To be implemented in child.')

    def _get_child_type(self):
        """Get the type of the child as a string."""
        raise NotImplementedError('To be implemented in child.')

    def decide_next_query(self):
        """Given a particular gp, decide best point to query.
        Returns: Function index and point to query next.
        """
        build_gp_posterior(self.gps[0])
        if self.t < self.options.run_ei_after:
            return self._child_decide_query()
        else:
            return self._play_ei()

    def _play_ei(self):
        """Play EI in round robin fashion."""
        gp = self.gps[0]
        f_idx = self.round_robin_idx
        self.round_robin_idx = (self.round_robin_idx + 1) % len(self.fcns)
        curr_best = self.curr_best[self.f_names[f_idx]][-1][0]
        ei_pts = sample_grid([self.f_locs[f_idx]], self.domains[0],
                             self.options.max_opt_evals)
        mu, sigma = gp.eval(ei_pts, include_covar=True)
        sigma = np.sqrt(sigma.diagonal().ravel())
        norm_diff = (mu - curr_best) / sigma
        eis = sigma * (norm_diff * normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff))
        pt_idx = np.argmax(eis)
        return f_idx, ei_pts[pt_idx, len(self.f_locs[0]):]

"""
IMPLEMENTATIONS
"""

class REVI(CorrelatedOpt):

    def _child_decide_query(self):
        """Draw a joint sample over the entire space, use this to select
        the context and the action.
        """
        # Calculate current mean surface.
        gp = self.gps[0]
        noise = gp.get_estimated_noise()
        # Get the candidate and judgement points.
        judge_pts = sample_grid(self.f_locs, self.domains[0],
                               self.options.num_eval_pts)
        lows, highs = zip(*self.domains[0])
        num_cands = len(self.f_locs) * self.options.num_candidates
        cand_pts = sample_grid(self.f_locs, self.domains[0],
                               self.options.num_candidates)
        # Get the mean values of these points.
        conjoined = np.vstack([cand_pts, judge_pts])
        means, _ = gp.eval(conjoined, include_covar=False)
        judge_means = means[num_cands:].reshape(len(self.fcns),
                                                self.options.num_eval_pts)
        interactions = gp.get_pt_relations(cand_pts, judge_pts)
        cand_means, covar_mat = gp.eval(cand_pts, include_covar=True)
        cand_vars = covar_mat.diagonal()
        f_idx, best_cidx, best_val = None, None, float('-inf')
        for c_idx in range(len(cand_pts)):
            interaction = interactions[c_idx].reshape(len(self.fcns),
                                                      self.options.num_eval_pts)
            var = cand_vars[c_idx]
            improvement = 0
            # Judge the effect of the candidate point.
            for ctx_idx in range(len(self.fcns)):
                means = judge_means[ctx_idx]
                sigmas = interaction[ctx_idx] / np.sqrt(noise + var)
                improvement += knowledge_gradient(means, sigmas)
            if improvement > best_val:
                best_cidx, best_val = c_idx, improvement
                f_idx = int(np.floor(c_idx / self.options.num_candidates))
        best_pt = cand_pts[best_cidx, len(self.f_locs[0]):]
        return f_idx, best_pt

    def _get_ei(self, ctx_pts, ctx_means, covs, samp_pt, samp_mean, samp_var,
                noise):
        """Get the expected improvement for a context.
        Args:
            kern: The kernel function.
            data_pts: ndarray of the points in data set.
            data_means: Vector of means for the data.
            samp_pt: Vector for the point at which new evaluation is seen.
            samp_val: Value for the new sample.
            samp_mean: The mean for the sample.
            noise: Noise in the system.
        """
        scaling = np.sqrt(samp_var + noise)
        adj_vars = covs /scaling
        expected_increase = knowledge_gradient(ctx_means.ravel(), adj_vars)
        return expected_increase

    @staticmethod
    def get_opt_method_name():
        """Get the type of the child as a string."""
        return 'revi'

corr_strats = [Namespace(name='revi', impl=REVI)]
