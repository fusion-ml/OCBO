"""
Optimization for continuous contexts.
"""

from __future__ import division

from argparse import Namespace
from copy import deepcopy
import numpy as np

from dragonfly.gp.euclidean_gp import EuclideanGPFitter, euclidean_gp_args
from dragonfly.utils.option_handler import get_option_specs, load_options
from gp.gp_util import get_gp_fitter, get_tuned_gp, get_best_dragonfly_prior
from util.misc_util import uniform_draw

cts_opt_args = [\
        get_option_specs('tuning_methods', False, 'ml',
            ('Number for tuning hyperparameters, other possibilities are '
             'post_sampling and post_mean or any combination seperated'
             'by hyphens e.g. ml-post_sampling')),
        get_option_specs('tune_every', False, 1,
            'How often to tune hyperparams.'),
        get_option_specs('kernel_type', False, 'se',
            'The type of kernel to use in the GP.'),
        get_option_specs('use_additive_gp', False, False,
            'Whether to use an additive GP.'),
        get_option_specs('add_max_group_size', False, 6,
            'The maximum number of groups in the additive grouping.'),
        get_option_specs('hp_samples', False, 3,
            'Number of hyperparameter samples to draw every time GP is tuned.'),
        get_option_specs('eval_ctx_fidel', False, 100,
            'Number of contexts to use when doing evaluation.'),
        get_option_specs('eval_act_fidel', False, 100,
            'Number of actions per context to use when doing evaluation.'),
        get_option_specs('score_every', False, 1,
            ('How often to evaluate current score. For real experiments this'
             'should be set to None since scoring relies on many function'
             'evaluations')),
        get_option_specs('cand_size', False, 100,
            'Number of candidate points to consider picking.'),
        get_option_specs('gp_engine', False, 'dragonfly',
            'Engine to use for GP.'),
]

class ContinuousOpt(object):

    def __init__(self, function, domain, ctx_dim, options, eval_set=None,
                 ctx_constraints=None, is_synthetic=True):
        """Constructor.
        Args:
            function: The function to optimize.
            domain: The domain of the function as a list of lists.
                [[low1, high1],...]
            ctx_dim: The dimension devoted for context space. These are the
                first N dimensions given.
            options: Namespace of options.
            eval_set: A predetermined set to calculate score over.
            ctx_constraints: List of context points in which we have the
                ability to make evals on. If None, it is assumed to be
                unconstrained.
            is_synthetic: Whether we are working in the synthetic environment,
                if so then we can evaluate the function without worrying about
                cost.
        """
        self.function = function
        self.domain = domain
        self.ctx_domain = domain[:ctx_dim]
        self.act_domain = domain[ctx_dim:]
        self.dim = len(domain)
        self.ctx_dim = ctx_dim
        self.act_dim = len(self.act_domain)
        self.is_synthetic = is_synthetic
        if eval_set is None:
            self.eval_ctx_fidel = options.eval_ctx_fidel
            self.eval_act_fidel = options.eval_act_fidel
            self.eval_pts, self.eval_grid = self._get_eval_set()
            self._find_best_score()
        elif is_synthetic:
            self.set_eval_set(eval_set[0], eval_set[1])
        self.ctx_constraints = ctx_constraints
        # Options.
        self.options = options
        self.eval_ctx_fidel = self.options.eval_ctx_fidel
        self.eval_act_fidel = self.options.eval_act_fidel
        self.score_every = self.options.score_every
        self.tune_every = self.options.tune_every
        self.gp_engine = self.options.gp_engine
        self.cand_size = options.cand_size
        # GP Options
        self.gp_options = load_options(euclidean_gp_args, cmd_line=False)
        self.gp_options.kernel_type = options.kernel_type
        self.gp_options.hp_tune_criterion = options.tuning_methods
        self.gp_options.hp_samples = self.options.hp_samples
        self.gp_options.dim = self.dim
        self.gp_options.act_dim = self.act_dim
        self.gp_options.use_additive_gp = options.use_additive_gp
        self.gp_options.add_max_group_size = options.add_max_group_size
        self.eval_options = deepcopy(self.gp_options)
        self.eval_options.hp_tune_criterion = 'ml'
        # Set up for any special child functions.
        self._child_set_up(function, domain, ctx_dim, options)
        # Run specific attributes.
        self.set_clean()

    def set_clean(self):
        """Clean up and prepare for a new optimization."""
        self.gp = None
        self.gp_fitter = None
        self.x_data = []
        self.x_init = []
        self.y_data = []
        self.y_init = []
        self.query_history = []
        self.score_history = []
        self.hp_history = []
        self.t = 0
        self.pre_loaded_fit = False

    def optimize(self, max_capital, init_pts, pre_tune=False,
                 hp_tune_samps=None):
        """Do the whole optimization loop.
        Args:
            max_capital: The maximum amount of capital to use excluding inits.
            init_pts: List of init point queries.
            pre_tune: Whether to fix the GP during optimization tuning only
                on the initial points provided.
            hp_tune_samps: If this is set to an integer, hyperparameters
                will first be tuned on this number of samples, then fixed.
        """
        # Get initial points and load.
        init_rewards = [self.function(pt) for pt in init_pts]
        if hp_tune_samps is not None:
            self.pre_tune_gp(hp_tune_samps, init_pts, init_rewards)
        elif pre_tune:
            self.pre_load_points(init_pts, init_rewards)
        else:
            self.load_init_pts(init_pts, init_rewards)
        # While there is capital...
        for _ in range(max_capital):
            # Make suggestion of next eval and make function call.
            pt = self.suggest_next_eval()
            reward = self.function(pt)
            # Receive feedback.
            self.receive_feedback(pt, reward)
        # Return histories.
        return self.get_histories()

    def suggest_next_eval(self):
        """Suggest next evaluation to make.
        Returns: Evaluation point as a list.
        """
        self._draw_next_gp()
        return self._determine_next_query()

    def receive_feedback(self, eval_pt, reward):
        """Receive feedback from an evaluation.
        Args:
            eval_pt: Point of evaluation as a list.
            reward: Reward received at point.
        """
        self.x_data.append(eval_pt)
        self.y_data.append(reward)
        if self.gp is not None:
            self.gp.add_data_single(eval_pt, reward)
        self.t += 1
        self._update_history(eval_pt, reward)
        if not self.pre_loaded_fit and self.t % self.tune_every == 0:
            self._make_new_gp_fitter()

    def load_init_pts(self, eval_pts, rewards):
        """Give initial eval_pts and rewards. Tune after receiving.
        Args:
            eval_pts: List of lists representing points.
            rewards: List of rewards.
        """
        self.x_data += eval_pts
        self.y_data += rewards
        self.x_init = eval_pts
        self.y_init = rewards
        self._make_new_gp_fitter()

    def pre_load_points(self, eval_pts, rewards):
        """Give points to fit the GP, use this GP for the entire time.
        Args:
            eval_pts: List of lists representing points.
            rewards: List of rewards.
        """
        self.x_data += eval_pts
        self.y_data += rewards
        self.x_init = eval_pts
        self.y_init = rewards
        self.pre_loaded_fit = True
        self.gp = get_tuned_gp(self.gp_engine, eval_pts, rewards,
                               kernel_type=self.gp_options.kernel_type)

    def pre_tune_gp(self, hp_tune_samps, eval_pts, rewards):
        """Pre tune a GP's hyperparameters that will be fixed throughout.
        Args:
            hp_tune_samps: Number of samples to use for tuning hps.
            eval_pts: x_data to add to the gp.
            rewards: The rewards of the initial points.
        """
        if self.gp_engine is not 'dragonfly':
            raise NotImplementedError('Pretuning HPs only for dragonfly.')
        self.x_data += eval_pts
        self.y_data += rewards
        self.x_init = eval_pts
        self.y_init = rewards
        self.pre_loaded_fit = True
        self.gp = get_best_dragonfly_prior(self.function, self.domain,
                kernel_type=self.gp_options.kernel_type,
                num_samples=hp_tune_samps)
        for init_idx in range(len(rewards)):
            self.gp.add_data_single(eval_pts[init_idx], rewards[init_idx])

    def get_histories(self):
        """Return Namespace object containing history information."""
        return Namespace(query_history=self.query_history,
                score_history=self.score_history,
                hp_history=self.hp_history,
                x_init=self.x_init,
                y_init=self.y_init)

    def get_eval_set(self):
        """Get the set of points used for evaluation."""
        return self.eval_pts, self.eval_grid

    def set_eval_set(self, eval_pts, eval_grid):
        """Set the points used for evaluation."""
        self.eval_pts = eval_pts
        self.eval_grid = eval_grid
        self.eval_ctx_fidel, self.eval_act_fidel = eval_grid.shape[:2]
        self._find_best_score()

    def set_gp(self, gp):
        """Set the GP to be some GP that will not change through the algo."""
        self.gp = gp
        self.pre_loaded_fit = True

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        raise NotImplementedError('Abstract Method')

    def _determine_next_query(self):
        """Abstract method. Return point that should be queried next."""
        raise NotImplementedError('Abstract method')

    def _child_set_up(self, function, domain, ctx_dim, options):
        pass

    def _make_candidate_set(self):
        ctxs = self._get_ctx_candidates(self.cand_size)
        ctxs = ctxs.reshape(self.cand_size, self.ctx_dim)
        acts = uniform_draw(self.act_domain, self.cand_size)
        return np.hstack([ctxs, acts])

    def _get_ctx_candidates(self, num_candidates):
        """Get possible contexts that can be used for suggested eval points."""
        if self.ctx_constraints is None:
            lows, highs = zip(*self.ctx_domain)
            return np.random.uniform(lows, highs, (num_candidates, len(lows)))
        else:
            rand_idxs = np.random.randint(0, len(self.ctx_constraints),
                                          num_candidates)
            cands = [self.ctx_constraints[idx] for idx in rand_idxs]
            return np.asarray(cands)

    def _update_history(self, pt, reward):
        """Update history based on current data seen."""
        q_info = Namespace(t=self.t, pt=pt, reward=reward)
        self.query_history.append(q_info)
        if self.is_synthetic \
                and self.score_every is not None \
                and self.t % self.score_every == 0:
            score = self._get_current_score()
            regret = self.best_score - score
            s_info = Namespace(t=self.t, score=score, regret=regret)
            self.score_history.append(s_info)
        if self.gp is not None:
            hp_info = Namespace(t=self.t, hps=self.gp.get_kernel_hps())
            self.hp_history.append(hp_info)

    def _draw_next_gp(self):
        """Draw the next GP."""
        if self.pre_loaded_fit:
            return self.gp
        assert self.gp_fitter is not None
        self.gp = self.gp_fitter.get_next_gp()

    def _make_new_gp_fitter(self):
        """Make gp fitter based on observed data."""
        self.gp_fitter = get_gp_fitter(self.gp_engine, self.x_data, self.y_data,
                                       self.gp_options)
        self.gp_fitter.fit_gp()

    def _get_current_score(self, return_max_pts=False, eval_gp=None,
                           e_pts=None, e_grid=None):
        """Get current score using the posterior mean."""
        if eval_gp is None:
            if self.pre_loaded_fit:
                eval_gp = self.gp
            else:
                eval_fitter = get_gp_fitter(self.gp_engine, self.x_data,
                                            self.y_data, self.eval_options)
                eval_fitter.fit_gp()
                eval_gp = eval_fitter.get_next_gp()
        if e_pts is None or e_grid is None:
            e_pts = self.eval_pts
            e_grid = self.eval_grid
        # Get points at which posterior is highest.
        mu = eval_gp.eval(e_pts)[0].reshape(e_grid.shape[:2])
        max_idxs = np.argmax(mu, axis=1)
        max_pts = e_grid[np.arange(e_grid.shape[0]), max_idxs, :]
        # Find average reward over max_pts.
        score = 0
        for max_pt in max_pts:
            score += self.function(max_pt)
        avg_score = score / e_grid.shape[0]
        if return_max_pts:
            return avg_score, max_pts
        else:
            return avg_score

    def _find_best_score(self):
        evals = []
        for pt in self.eval_pts:
            evals.append(self.function(pt))
        evals = np.asarray(evals).reshape(self.eval_grid.shape[:2])
        self.best_score = np.mean(np.max(evals, axis=1))

    def _get_eval_set(self, ctx_fidel=None, act_fidel=None):
        """Get grid used to evaluate score.
        Returns set of points as both list of points and a grid format
        i.e. dim = (num_ctxs, num_acts, point).
        """
        if ctx_fidel is None:
            ctx_fidel = self.eval_ctx_fidel
        if act_fidel is None:
            act_fidel = self.eval_act_fidel
        rand_ctxs = uniform_draw(self.ctx_domain, ctx_fidel)
        rand_ctxs = np.repeat(rand_ctxs, act_fidel, axis=0)
        rand_acts = uniform_draw(self.act_domain, act_fidel)
        rand_acts = np.tile(rand_acts.ravel(), ctx_fidel)\
                      .reshape(ctx_fidel * act_fidel,
                               self.act_dim)
        eval_pts = np.hstack([rand_ctxs, rand_acts])
        eval_grid = eval_pts.reshape(ctx_fidel, act_fidel,
                                     self.ctx_dim + self.act_dim)
        return eval_pts, eval_grid
