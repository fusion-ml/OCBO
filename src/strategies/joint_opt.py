"""
Multi-Opt strategy that uses a joint GP instead of many GPs.
"""
from __future__ import division

from argparse import Namespace
from copy import deepcopy
from dragonfly.gp.euclidean_gp import euclidean_gp_args
from dragonfly.utils.option_handler import get_option_specs, load_options
from util.misc_util import build_gp_posterior, sample_grid
import numpy as np

from gp.gp_util import get_gp_fitter
from strategies.multi_opt import MultiOpt

class JointOpt(MultiOpt):

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
        self.f_locs = []
        for f_inf in f_infos:
            if not hasattr(f_inf, 'f_loc'):
                raise ValueError('Function %s does not have a location'
                                 % f_inf.name)
            self.f_locs.append(f_inf.f_loc)
        super(JointOpt, self).__init__(f_infos, options, pre_tuned_gps,
                                       rn_info)
        act_dim = len(self.f_locs[0])
        self.gp_options.dim = self.dims[0] + act_dim
        self.gp_options.act_dim = act_dim

    def _add_point_to_gp(self, f_idx, pt, val):
        """Adds point to specified GP."""
        pt = np.append(self.f_locs[f_idx], pt)
        self.gps[f_idx].add_data_single(pt, val)

    def set_clean(self):
        """Erase query history and gp information."""
        if self.pre_tuned_gps is not None:
            self.gps = [gp for gp in self.pre_tuned_gps]
        else:
            self.gps = [None for _ in self.f_names]
        self.gp_fitters = [None for _ in self.f_names]
        self.query_history = {fn: [] for fn in self.f_names}
        self.curr_best = {fn: [] for fn in self.f_names}
        self.rn_choice = {fn: [] for fn in self.f_names}
        self.total_best = []
        self.rn_total = []
        self.choice_timeline = []
        self.allocation_count = [0 for _ in self.f_names]
        self.t = 0
        self.next_explore_idx = 0
        self.num_explores = 0

    def _rn_update(self, fcn_idx):
        f_name = self.f_names[fcn_idx]
        rn_best = self.rn_choice[f_name]
        rn_best = self.rn_choice[f_name]
        build_gp_posterior(self.gps[fcn_idx])
        mean_pts = self.rn_grids[fcn_idx]
        mean_vals = self.gps[fcn_idx].eval(mean_pts)[0].ravel()
        rn_pt = mean_pts[np.argmax(mean_vals), len(self.f_locs[0]):]
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
            curr_grid = sample_grid([self.f_locs[f_idx]], self.domains[0],
                                    self.options.max_opt_evals)
            best_val = float('-inf')
            for pt in curr_grid:
                pt_eval = self.fcns[f_idx](pt[len(self.f_locs[0]):])
                if pt_eval > best_val:
                    best_val = pt_eval
            rn_grids.append(curr_grid)
            rn_bests.append(best_val)
        return rn_grids, rn_bests

    def _draw_next_gps(self):
        """Draw the next GPs for the next round of decision making. This refers
        to drawing new hyperparameters from the posterior.
        """
        # We should only ever be calling this method if we have posterior over
        # the hyperparameters and therefore have gp_fitters.
        assert None not in self.gp_fitters
        next_gp = self.gp_fitters[0].get_next_gp()
        if 'post_sampling' in self.gp_options.hp_tune_criterion \
                and self.tune_every > 1:
            next_gp.build_posterior()
        self.gps = [next_gp for _ in self.fcns]

    def _update_models(self):
        """Update the GPFitter or tune GP model depending on method."""
        x_data, y_data = [], []
        for f_idx, f_name in enumerate(self.f_names):
            for hist in self.query_history[f_name]:
                x_data.append(np.append(self.f_locs[f_idx], hist.pt))
                y_data.append(hist.val)
        gpf = get_gp_fitter(self.options.gp_engine, x_data, y_data,
                            self.gp_options)
        gpf.fit_gp()
        self.gp_fitters = [gpf]

    def _update_model(self, idx):
        self._update_models()
