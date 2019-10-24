"""
Strategies that try to maximize the posterior mean function.
"""

from argparse import Namespace
import numpy as np

from cstrats.cts_opt import ContinuousOpt
from dragonfly.utils.option_handler import get_option_specs
from util.misc_util import sample_grid, uniform_draw, knowledge_gradient

pm_args = [\
        get_option_specs('judge_act_size', False, 50,
            'Number of points to use to judge candidates.'),
        get_option_specs('judge_ctx_size', False, 50,
            'Number of points to use to judge candidates.'),
        get_option_specs('judge_ctx_thresh', False, None,
            'Threshold for how related judgement set should be in context.'),
        get_option_specs('judge_act_thresh', False, None,
            'Threshold for how related judgement set should be in context.'),
]

class PosteriorMaximization(ContinuousOpt):

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.judge_act_size = options.judge_act_size
        self.judge_ctx_size = options.judge_ctx_size
        self.judge_ctx_thresh = options.judge_ctx_thresh
        if self.judge_ctx_thresh is not None:
            self.judge_ctx_thresh = float(self.judge_ctx_thresh)
        self.judge_act_thresh = options.judge_act_thresh
        if self.judge_act_thresh is not None:
            self.judge_act_thresh = float(self.judge_act_thresh)

    def _make_judgement_set(self, cand_pt):
        """Make judgement set. Restruct to have some delta threshold in the
        context and action space. Distance between points uses l-infty norm.
        Args:
            cand_pt: ndarray of specific candidate point to make the set for.
        Returns: ndarray of points.
        """
        # Find context threshold.
        if self.judge_ctx_thresh is None:
            c_thresh = self.ctx_domain
        else:
            c_thresh = self._find_ctx_thresh(cand_pt)
        # Draw contexts based on context threshold constraint.
        ctxs = uniform_draw(c_thresh, self.judge_ctx_size)
        # Find Action threshold.
        if self.judge_act_thresh is None:
            a_thresh = self.act_domain
        else:
            a_thresh = self._find_act_thresh(cand_pt)
        ctxs = [list(cx) for cx in ctxs]
        return sample_grid(ctxs, a_thresh, self.judge_act_size)

    def _find_ctx_thresh(self, cand_pt):
        # Draw points at random with same action.
        cand_ctx, cand_act = np.split(cand_pt, [self.ctx_dim])
        check_ctxs = uniform_draw(self.ctx_domain, self.judge_ctx_size)
        dist_pairs = []
        for cc in check_ctxs:
            linf = np.max(np.abs(cc - cand_ctx))
            dist_pairs.append((linf, cc))
        dist_pairs.sort()
        dist_pairs = dist_pairs[::-1]
        check_ctxs = np.asarray([dp[1] for dp in dist_pairs])
        check_ctxs = check_ctxs.reshape(self.judge_ctx_size, self.ctx_dim)
        repeated_act = np.tile(cand_act, self.judge_ctx_size)
        repeated_act = repeated_act.reshape(self.judge_ctx_size,
                                            self.act_dim)
        check_pts = np.hstack([check_ctxs, repeated_act])
        check_pts = check_pts.reshape(self.judge_ctx_size, self.dim)
        # Find posterior covariances between cand_pt and drawn.
        all_pts = np.vstack([cand_pt, check_pts])
        _, covmat = self.gp.eval(all_pts, include_covar=True)
        covs = covmat[0]
        # Find furthest point that satisfies threshold.
        dist_idx = 1
        while dist_idx < self.judge_ctx_size \
                and covs[dist_idx] > self.judge_ctx_thresh:
            dist_idx += 1
        max_ldist = dist_pairs[dist_idx - 1][0]
        new_domain = []
        for dim, dim_domain in enumerate(self.ctx_domain):
            lower = np.max([cand_ctx[dim] - max_ldist, dim_domain[0]])
            upper = np.min([cand_ctx[dim] + max_ldist, dim_domain[1]])
            new_domain.append([lower, upper])
        return new_domain

    def _find_act_thresh(self, cand_pt):
        # Draw points at random with same action.
        cand_ctx, cand_act = np.split(cand_pt, [self.ctx_dim])
        check_acts = uniform_draw(self.act_domain, self.judge_act_size)
        dist_pairs = []
        for ca in check_acts:
            linf = np.max(np.abs(ca - cand_ctx))
            dist_pairs.append((linf, ca))
        dist_pairs.sort()
        dist_pairs = dist_pairs[::-1]
        check_acts = np.asarray([dp[1] for dp in dist_pairs])
        check_acts = check_acts.reshape(self.judge_act_size, self.act_dim)
        repeated_ctx = np.tile(cand_ctx, self.judge_act_size)
        repeated_ctx = repeated_ctx.reshape(self.judge_act_size,
                                            self.ctx_dim)
        check_pts = np.hstack([repeated_ctx, check_acts])
        check_pts = check_pts.reshape(self.judge_act_size, self.dim)
        # Find posterior covariances between cand_pt and drawn.
        all_pts = np.vstack([cand_pt, check_pts])
        _, covmat = self.gp.eval(all_pts, uncert_form='covar')
        covs = covmat[0]
        # Find furthest point that satisfies threshold.
        dist_idx = 1
        while dist_idx < self.judge_ctx_size \
                and covs[dist_idx] > self.judge_ctx_thresh:
            dist_idx += 1
        max_ldist = dist_pairs[dist_idx - 1][0]
        new_domain = []
        for dim, dim_domain in enumerate(self.act_domain):
            lower = np.max([cand_act[dim] - max_ldist, dim_domain[0]])
            upper = np.min([cand_act[dim] + max_ldist, dim_domain[1]])
            new_domain.append([lower, upper])
        return new_domain

class REVI(PosteriorMaximization):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'revi'

    def _determine_next_query(self):
        """Pick the next query uniformly at random."""
        if self.judge_act_thresh is None and self.judge_ctx_thresh is None:
            return self._set_eval_next_query()
        return self._diff_eval_next_query()

    def _set_eval_next_query(self):
        # Form candidate and judgement sets.
        cands = self._make_candidate_set()
        noise = self.gp.get_estimated_noise()
        judge = self._make_judgement_set(cands[0])
        conjoined = np.vstack([cands, judge])
        means, _ = self.gp.eval(conjoined, include_covar=False)
        judge_means = means[self.cand_size:].reshape(self.judge_ctx_size,
                                                     self.judge_act_size)
        interactions = self.gp.get_pt_relations(cands, judge)
        cand_vars = self.gp.eval(cands, include_covar=True)[1].diagonal()
        best_pt, best_val = None, float('-inf')
        for c_idx in range(self.cand_size):
            interaction = interactions[c_idx].reshape(self.judge_ctx_size,
                                                      self.judge_act_size)
            var = cand_vars[c_idx]
            improvement = 0
            # Judge the effect of the candidate point.
            for ctx_idx in range(self.judge_ctx_size):
                means = judge_means[ctx_idx]
                sigmas = interaction[ctx_idx] / np.sqrt(noise + var)
                improvement += knowledge_gradient(means, sigmas)
            if improvement > best_val:
                best_pt, best_val = cands[c_idx], improvement
        # Return the best candidate point.
        return best_pt

    def _diff_eval_next_query(self):
        # Form candidate and judgement sets.
        cands = self._make_candidate_set()
        noise = self.gp.get_estimated_noise()
        best_pt, best_val = None, float('-inf')
        for c_idx in range(self.cand_size):
            cand_pt = cands[c_idx]
            judge = self._make_judgement_set(cands[c_idx])
            # Find mean and covar mat for combined cand-judgement set.
            conjoined = np.vstack([cand_pt, judge])
            means, covar = self.gp.eval(conjoined, include_covar=False)
            judge_means = means[1:].reshape(self.judge_ctx_size,
                                            self.judge_act_size)
            interaction = self.gp.get_pt_relations([cand_pt], judge)
            interaction = interaction.reshape(self.judge_ctx_size,
                                              self.judge_act_size)
            var = float(self.gp.eval([cand_pt], include_covar=True)[1])
            improvement = 0
            # Judge the effect of the candidate point.
            for ctx_idx in range(self.judge_ctx_size):
                means = judge_means[ctx_idx]
                sigmas = interaction[ctx_idx] / np.sqrt(noise + var)
                improvement += knowledge_gradient(means, sigmas)
            if improvement > best_val:
                best_pt, best_val = cand_pt, improvement
        # Return the best candidate point.
        return best_pt

pm_strats = [Namespace(impl=REVI, name=REVI.get_strat_name())]
