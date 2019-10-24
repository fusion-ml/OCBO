"""
GP class that interfaces with several different GP libraries.
"""
from copy import deepcopy

from dragonfly.gp.euclidean_gp import EuclideanGPFitter
from dragonfly.utils.general_utils import solve_lower_triangular

class GPWrapper(object):

    def __init__(self, gp_core, options):
        """Constructor."""
        self.gp_core = gp_core
        self.options = options

    def eval(self, pts, include_covar=False):
        """Get posterior mean and possibly covariance mat."""
        raise NotImplementedError('Abstract Method')

    def draw_sample(self, samp_pts=None, means=None, covar=None):
        """Draw a single sample from posterior at points."""
        raise NotImplementedError('Abstract Method')

    def add_data_single(self, pt, val):
        """Add a single data point and observation to the GP."""
        raise NotImplementedError('Abstract Method')

    def build_posterior(self):
        """Build the GP posterior."""
        raise NotImplementedError('Abstract Method')

    def get_estimated_noise(self):
        """Return the estimated noise."""
        raise NotImplementedError('Abstract Method.')

    def get_kernel_hps(self):
        """Return the hps of the kernel."""
        raise NotImplementedError('Abstract Method.')

    def get_pt_relations(self, pt_set1, pt_set2):
        """Return the covariances between points in first set with second."""
        raise NotImplementedError('Abstract Method.')

class GPFitterWrapper(object):

    def __init__(self, x_data, y_data, options):
        """Constructor."""
        self.x_data = x_data
        self.y_data = y_data
        self.options = options
        self._init_gpf()

    def fit_gp(self):
        """Fit the GP according to options."""
        raise NotImplementedError('Abstract Method')

    def get_next_gp(self):
        """Get the next GP from previously fitted.
        Returns: GPWrapper object.
        """
        raise NotImplementedError('Abstract Method')

    def _init_gpf(self):
        """Initialize the GP fitter."""
        raise NotImplementedError('Abstract Method')

class DragonflyGP(GPWrapper):

    def eval(self, pts, include_covar=False):
        """Get posterior mean and possibly covariance mat."""
        if include_covar:
            return self.gp_core.eval(pts, uncert_form='covar')
        else:
            return self.gp_core.eval(pts)

    def draw_sample(self, samp_pts=None, means=None, covar=None):
        """Draw a single sample from posterior at points."""
        return self.gp_core.draw_samples(1, X_test=samp_pts, mean_vals=means,
                                         covar=covar).ravel()

    def add_data_single(self, pt, val):
        """Add a single data point and observation to the GP."""
        self.gp_core.add_data_single(pt, val)

    def build_posterior(self):
        """Build the GP posterior."""
        if self.gp_core.alpha is None:
            self.gp_core.build_posterior()

    def get_estimated_noise(self):
        """Return the estimated noise."""
        return self.gp_core.noise_var

    def get_kernel_hps(self):
        """Return the hps of the kernel."""
        if self.options.use_additive_gp:
            top_scale = self.gp_core.kernel.hyperparams['scale']
            all_hps = {'top_level_scale': top_scale,
                       'groupings': self.gp_core.kernel.groupings,
                       'scales': [],
                       'lengthscales': []}
            for kern in self.gp_core.kernel.kernel_list:
                hps = kern.hyperparams
                all_hps['scales'].append(hps['scale'])
                all_hps['lengthscales'].append(hps['dim_bandwidths'])
            return all_hps
        hps = deepcopy(self.gp_core.kernel.hyperparams)
        hps['lengthscale'] = hps['dim_bandwidths']
        hps.pop('dim_bandwidths', None)
        return hps

    def get_pt_relations(self, pt_set1, pt_set2):
        """Return the covariances between points in first set with second."""
        prior_covs = self.gp_core.kernel(pt_set1, pt_set2)
        inters1 = self.gp_core.kernel(pt_set1, self.gp_core.X)
        V1 = solve_lower_triangular(self.gp_core.L, inters1.T)
        inters2 = self.gp_core.kernel(pt_set2, self.gp_core.X)
        V2 = solve_lower_triangular(self.gp_core.L, inters2.T)
        return prior_covs - V1.T.dot(V2)

class DragonflyGPFitter(GPFitterWrapper):

    def fit_gp(self):
        """Fit the GP according to options."""
        self.gpf_core.fit_gp_for_gp_bandit(num_samples=self.options.hp_samples)

    def get_next_gp(self):
        """Get the next GP from previously fitted.
        Returns: GPWrapper object.
        """
        next_gp = self.gpf_core.get_next_gp()[2]
        next_gp.build_posterior()
        return DragonflyGP(next_gp, self.options)

    def _init_gpf(self):
        """Initialize the GP fitter."""
        self.gpf_core = EuclideanGPFitter(self.x_data, self.y_data,
                                          self.options)
