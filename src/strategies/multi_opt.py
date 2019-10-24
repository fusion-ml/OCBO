"""
Classes to do multiple acquisitions.
"""

from __future__ import division

from argparse import Namespace
from collections import deque
from copy import deepcopy
from dragonfly.gp.euclidean_gp import EuclideanGPFitter, euclidean_gp_args
from dragonfly.utils.option_handler import get_option_specs, load_options
from util.misc_util import build_gp_posterior, sample_grid
import numpy as np

from gp.gp_util import get_gp_fitter
from util.misc_util import get_tuned_gp

multi_opt_args = [\
        get_option_specs('tuning_methods', False, 'post_sampling',
            ('Number for tuning hyperparameters, other possibilities are '
             'post_sampling and post_mean or any combination seperated'
             'by hyphens e.g. ml-post_sampling')),
        get_option_specs('hp_samples', False, 3,
            'Number of hyperparameter samples to draw every time GP is tuned.'),
        get_option_specs('kernel_type', False, 'se',
            'The type of kernel to use in the GP.'),
        get_option_specs('risk_neutral', False, False,
            'Whether to include risk neutral evaluations'),
        get_option_specs('rn_eval_size', False, 10000,
            ('Number of evaluations on posterior mean for each of the '
             'when using the risk neutral strategy.')),
        get_option_specs('gp_engine', False, 'dragonfly',
            'Engine to use for GP.'),
]

class MultiOpt(object):

    def __init__(self, f_infos, options, pre_tuned_gps=None, rn_info=None):
        """Constructor:
        Args:
            fcns: List of Namespaces containing function info.
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
        self.fcns = [fi.function for fi in f_infos]
        self.domains = [fi.domain for fi in f_infos]
        self.dims = [len(d) for d in self.domains]
        self.f_names = [fi.name for fi in f_infos]
        self.pre_tuned_gps = pre_tuned_gps
        self.tune_every = options.tune_every
        self.options = options
        self.risk_neutral = options.risk_neutral
        if rn_info is not None:
            self.rn_grids, self.rn_bests = rn_info
        elif self.risk_neutral:
            self.rn_grids, self.rn_bests = self._create_rn_grids()
        else:
            self.rn_grids, self.rn_bests = None, None

        self.gp_options = load_options(euclidean_gp_args, cmd_line=False)
        self.gp_options.kernel_type = options.kernel_type
        self.gp_options.hp_tune_criterion = options.tuning_methods
        self.gp_options.hp_tune_criterion = '-'.join(options.tuning_methods.split(','))
        self.gp_options.hp_samples = options.hp_samples
        self.hp_samples = options.hp_samples

        # Run specific attributes.
        self.gps = None
        self.gp_fitters = None
        self.sampled_gps = None
        self.query_history = None
        self.curr_best = None
        self.total_best = None
        self.choice_timeline = None
        self.allocation_count = None
        self.rn_choice = None
        self.rn_total = None
        self.t = 0
        self.set_clean()

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        raise NotImplementedError('Child implementation missing.')

    @staticmethod
    def get_opt_method_name():
        """Return the name of the method."""
        raise NotImplementedError('Child implementation missing.')

    def get_method_name_with_tuning(self):
        """Name of the method with the tuning method prepended."""
        return '-'.join([self.gp_options.hp_tune_criterion,
                         self.get_opt_method_name()])

    def optimize(self, capital, init_capital=5, init_pts=None):
        """Optimize given capital, init_capital.
        Args:
            capital: Amount of capital for optimization.
            init_capital: The amount of random queries to be given to EACH GP.
            init_pts: Instead of querying points at random, set points.
                Should be given as a list of lists.
        Returns: Query history and optimal policy.
        """
        # Perform initial queries.
        if init_pts is not None:
            self._handle_init_pts(init_pts)
        else:
            for idx in range(len(self.fcns)):
                self._make_random_queries(idx, init_capital)
        if self.pre_tuned_gps is None:
            self._update_models()
        # Spend capital.
        for _ in range(capital):
            self.t += 1
            if self.pre_tuned_gps is None:
                self._draw_next_gps()
            fcn_idx, pt = self.decide_next_query()
            val = self.fcns[fcn_idx](pt)
            self._add_point_to_gp(fcn_idx, pt, val)
            self._update_history(fcn_idx, pt, val)
        # Return query history and best points found.
        return self.get_histories()

    def prep_for_optimization(self, init_evals):
        """Prepare for optimization.
        Args:
            init_evals: List of Namespace objects where each namespace
                includes, pt, val, f_name (function name).
        """
        # Add to history.
        for q_info in init_evals:
            idx = self.f_names.index(q_info.f_name)
            pt = q_info.pt
            val = q_info.val
            q_info.init_pt = True
            q_info.t = 0
            self._update_history(idx, pt, val, init_pt=True, q_info=q_info)
        self._update_models()

    def draw_and_suggest_query(self):
        """Draws next gps and uses them to decide next query.
        Returns: Index of the function and the evaluation point.
        """
        if self.pre_tuned_gps is None:
            self._draw_next_gps()
        return self.decide_next_query()

    def receive_feedback(self, q_info):
        """Receive feedback from an experiment.
        Args:
            q_info: Namespace continaing pt, val, f_name.
        """
        self.t += 1
        idx = self.f_names.index(q_info.f_name)
        pt = q_info.pt
        val = q_info.val
        q_info.t = self.t
        self._update_history(idx, pt, val, q_info=q_info)

    def get_histories(self):
        """Return histories as Namespace object."""
        return Namespace(query_history=self.query_history,
                         curr_best=self.curr_best,
                         total_best=self.total_best,
                         choice_timeline=self.choice_timeline,
                         rn_choice=self.rn_choice,
                         rn_total=self.rn_total)

    def set_clean(self):
        """Erase query history and gp information."""
        if self.pre_tuned_gps is not None:
            self.gps = [deepcopy(gp) for gp in self.pre_tuned_gps]
        else:
            self.gps = [None for _ in self.f_names]
        self.gp_fitters = [None for _ in self.f_names]
        self.sampled_gps = [deque() for _ in self.f_names]
        self.query_history = {fn: [] for fn in self.f_names}
        self.curr_best = {fn: [] for fn in self.f_names}
        self.rn_choice = {fn: [] for fn in self.f_names}
        self.total_best = []
        self.rn_total = []
        self.choice_timeline = []
        self.allocation_count = [0 for _ in self.f_names]
        self.t = 0

    def _add_point_to_gp(self, f_idx, pt, val):
        """Adds point to specified GP."""
        self.gps[f_idx].add_data_single(pt, val)

    def _handle_init_pts(self, init_pts):
        """Handle inition pts in form of list of list of points"""
        for idx, pts in enumerate(init_pts):
            for pt in pts:
                val = self.fcns[idx](pt)
                if self.gps[idx] is not None:
                    self._add_point_to_gp(idx, pt, val)
                self._update_history(idx, pt, val, init_pt=True)

    def _make_random_queries(self, fcn_idx, queries):
        """Make a random query for the particular function."""
        for _ in range(queries):
            low, high = zip(*self.domains[fcn_idx])
            pt = np.random.uniform(low, high)
            y = self.fcns[fcn_idx](pt)
            if self.gps[fcn_idx] is not None:
                self._add_point_to_gp(fcn_idx, pt, y)
            self._update_history(fcn_idx, pt, y, init_pt=True)

    def _draw_next_gps(self):
        """Draw the next GPs for the next round of decision making. This refers
        to drawing new hyperparameters from the posterior.
        """
        # We should only ever be calling this method if we have posterior over
        # the hyperparameters and therefore have gp_fitters.
        assert None not in self.gp_fitters
        self.gps = []
        if 'post_sampling' in self.gp_options.hp_tune_criterion \
                and self.hp_samples > 1:
            for f_idx in range(len(self.fcns)):
                # See if we have already sampled this GP.
                cached_gps = self.sampled_gps[f_idx]
                if len(cached_gps) == self.hp_samples:
                    next_gp = cached_gps.pop()
                else:
                    next_gp = self.gp_fitters[f_idx].get_next_gp()
                cached_gps.append(next_gp)
                self.gps.append(next_gp)
        else:
            self.gps = [gpf.get_next_gp() for gpf in self.gp_fitters]

    def _update_models(self):
        """Update the GPFitter or tune GP model depending on method."""
        for idx in range(len(self.fcns)):
            self._update_model(idx)

    def _rn_update(self, fcn_idx):
        """Update quantities for risk neutral evaluation."""
        f_name = self.f_names[fcn_idx]
        rn_best = self.rn_choice[f_name]
        build_gp_posterior(self.gps[fcn_idx])
        mean_pts = self.rn_grids[fcn_idx]
        mean_vals = self.gps[fcn_idx].eval(mean_pts)[0].ravel()
        rn_pt = mean_pts[np.argmax(mean_vals)]
        rn_val = self.fcns[fcn_idx](rn_pt)
        rn_best.append((rn_val, rn_pt))

    def _create_rn_grids(self):
        """Create grid of points for risk neutral evaluation.
        Returns: List of ndarray of points for each function.
        """
        rn_grids = []
        rn_bests = []
        for f_idx in range(len(self.f_names)):
            lows, highs = zip(*self.domains[f_idx])
            curr_grid = np.random.uniform(lows, highs,
                            (self.options.rn_eval_size, len(lows)))
            best_val = float('-inf')
            for pt in curr_grid:
                pt_eval = self.fcns[f_idx](pt)
                if pt_eval > best_val:
                    best_val = pt_eval
            rn_grids.append(curr_grid)
            rn_bests.append(best_val)
        return rn_grids, rn_bests

    def _update_model(self, idx):
        """Update the GP fitter for a single index."""
        # Assemble x and y data.
        x_data, y_data = [], []
        for hist in self.query_history[self.f_names[idx]]:
            x_data.append(hist.pt)
            y_data.append(hist.val)
        gpf = get_gp_fitter(self.options.gp_engine, x_data, y_data,
                            self.gp_options)
        gpf.fit_gp()
        self.gp_fitters[idx] = gpf
        self.sampled_gps[idx] = deque()

    def _update_history(self, fcn_idx, pt, val, init_pt=False, q_info=None):
        self.allocation_count[fcn_idx] += 1
        if q_info is None:
            q_info = Namespace(pt=pt, val=val, init_pt=init_pt, time=self.t)
        f_name = self.f_names[fcn_idx]
        self.query_history[f_name].append(q_info)
        if self.risk_neutral:
            if not init_pt:
                self._rn_update(fcn_idx)
            else:
                rn_best = self.rn_choice[f_name]
                rn_best.append((val, pt))
        bests = self.curr_best[f_name]
        if len(bests) == 0 or bests[-1][0] < val:
            bests.append((val, pt))
        else:
            bests.append(bests[-1])
        if self.t > 0:
            accum = 0
            for f_bests in self.curr_best.values():
                accum += f_bests[-1][0]
            self.total_best.append(accum)
            self.choice_timeline.append(fcn_idx)
        if self.t > 0 and self.risk_neutral:
            accum = 0
            for rn_bests in self.rn_choice.values():
                accum += rn_bests[-1][0]
            self.rn_total.append(accum)
        if self.tune_every > 0 and self.t > 0:
            num_qs = len(self.query_history[f_name])
            if num_qs % self.tune_every == 0:
                self._update_model(fcn_idx)
