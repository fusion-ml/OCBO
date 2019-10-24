"""
Interface for the gpytorch library.
NOTE: Need python 3.6 or above to run this!
"""
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, \
        SpectralMixtureKernel

from dragonfly.gp.euclidean_gp import EuclideanGPFitter
from gp.gp_interface import GPWrapper, GPFitterWrapper

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, x_data, y_data, kernel, likelihood):
        """Constructor.
        Args:
            x_data: X data as torch tensor.
            y_data: Y data as torch tensor.
            kernel: Kernel object to use.
            likelihood: Likelihood object.
        """
        super(ExactGPModel, self).__init__(x_data, y_data, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class PytorchGP(GPWrapper):

    def eval(self, pts, include_covar=False):
        """Get posterior mean and possibly covariance mat."""
        tensor_pts = torch.from_numpy(pts).float()
        with torch.no_grad():
            pred = self.gp_core(tensor_pts)
        mean = pred.mean.numpy()
        if not include_covar:
            return mean, None
        covar = pred.covariance_matrix.numpy()
        return mean, covar

    def draw_sample(self, samp_pts=None, means=None, covar=None):
        """Draw a single sample from posterior at points."""
        if samp_pts is not None:
            tensor_pts = torch.from_numpy(samp_pts).float()
            with torch.no_grad():
                pred = self.gp_core(tensor_pts)
            return pred.sample().numpy()
        elif means is not None and covar is not None:
            return np.random.multivariate_normal(means, covar)
        else:
            raise ValueError('No arguments provided.')

    def add_data_single(self, pt, val):
        """Add a single data point and observation to the GP."""
        # It doesn't look like GPytorch has a way to add on data,
        # so we just have to create a new object.
        old_x = self.gp_core.train_inputs[0]
        old_y = self.gp_core.train_targets
        tensor_pt = torch.from_numpy(pt).reshape(1, len(pt))
        new_x = torch.cat((old_x, tensor_pt)).float()
        new_y = torch.cat((old_y, torch.tensor([val]).float())).float()
        self.gp_core = ExactGPModel(new_x, new_y, self.gp_core.covar_module,
                                    self.gp_core.likelihood)

    def build_posterior(self):
        """Build the GP posterior."""
        # It looks like GPytorch builds posterior every time eval is made.
        pass

    def get_estimated_noise(self):
        """Return the estimated noise."""
        return self.gp_core.likelihood.noise.item()

    def get_kernel_hps(self):
        """Return the hps of the kernel."""
        hp_dict = {}
        kern = self.gp_core.covar_module
        hp_dict['lengthscale'] = kern.base_kernel.lengthscale.flatten()
        hp_dict['scale'] = kern.outputscale
        return hp_dict

class PytorchGPFitter(GPFitterWrapper):

    def fit_gp(self):
        """Fit the GP according to options."""
        # Put things into training mode.
        self.gpf_core.float()
        self.likelihood.train()
        # Now use Adam by default.
        optimizer = torch.optim.Adam([{'params': self.gpf_core.parameters()}],
                                     lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.gpf_core)
        # TODO: Allow length of training to be an option.
        for _ in range(500):
            optimizer.zero_grad()
            output = self.gpf_core(self.tensor_x)
            loss = -mll(output, self.tensor_y)
            loss.backward()
            optimizer.step()

    def get_next_gp(self):
        """Get the next GP from previously fitted.
        Returns: GPWrapper object.
        """
        # Put model and likelihood into eval mode.
        self.gpf_core.eval()
        self.likelihood.eval()
        return PytorchGP(self.gpf_core, self.options)

    def _init_gpf(self):
        """Initialize the GP fitter."""
        if isinstance(self.x_data, list):
            self.tensor_x = torch.tensor(self.x_data).float()
        else:
            self.tensor_x = torch.from_numpy(self.x_data).float()
        if isinstance(self.y_data, list):
            self.tensor_y = torch.tensor(self.y_data).float()
        else:
            self.tensor_y = torch.from_numpy(self.y_data).float()
        act_dim, dim = None, self.tensor_x.shape[1]
        option_keys = vars(self.options)
        if 'dim' in option_keys and 'act_dim' in option_keys:
            act_dim, dim = self.options.act_dim, self.options.dim
        kernel = _get_kernel_type(self.options.kernel_type, self.tensor_x,
                                  self.tensor_y, dim, act_dim)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gpf_core = ExactGPModel(self.tensor_x, self.tensor_y, kernel,
                                     self.likelihood)

def _get_kernel_type(kernel_type, x_data, y_data, total_dim=None, act_dim=None):
    if kernel_type == 'se':
        return ScaleKernel(RBFKernel(ard_num_dims=total_dim))
    elif kernel_type == 'sm':
        # TODO: Find the best way to adjust the num_mixtures over problems.
        kern = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=total_dim)
        kern.initialize_from_data(x_data, y_data)
        return kern
    if total_dim is not None and act_dim is not None:
        ctx_dim = total_dim - act_dim
        ctx_dims = torch.arange(1, ctx_dim + 1)
        act_dims = torch.arange(ctx_dim + 1, total_dim + 1)
        if kernel_type == 'linsum':
            return ScaleKernel(
                    LinearKernel(active_dims=ctx_dims, ard_num_dims=ctx_dim)
                    + RBFKernel(active_dims=act_dims, ard_num_dims=act_dim)
            )
        # TODO: For some reason cannot find PolynomialKernel.
        # elif kernel_type == 'quadsum':
        #     return ScaleKernel(PolynomialKernel(2, active_dims=ctx_dims)
        #             + RBFKernel(active_dims=act_dim))
    else:
        return ScaleKernel(RBFKernel())
    raise ValueError('Invalid kernel type: %s' % kernel_type)
