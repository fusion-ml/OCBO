"""
Utility for GP Interface.
"""
import numpy as np

from dragonfly.gp.euclidean_gp import EuclideanGP, EuclideanGPFitter,\
        euclidean_gp_args
from dragonfly.utils.option_handler import load_options
from gp.gp_interface import DragonflyGPFitter, DragonflyGP

# Check if running python 3, if so import gpython
import sys
if sys.version_info[0] == 3:
    from gp.gpytorch_interface import PytorchGP, PytorchGPFitter

def get_gp_fitter(gp_engine, x_data, y_data, options):
    """Get GP Fitter wrapper.
    Args:
        gp_engine: The GP implementation to use.
        x_data: The list of x_data seen so far.
        y_data: The list of y_data seen so far.
        options: Options Namespace object.
    """
    if gp_engine.lower() == 'dragonfly':
        return DragonflyGPFitter(x_data, y_data, options)
    elif gp_engine.lower() == 'gpytorch':
        # Import here since will only work for python 3.6>=.
        return PytorchGPFitter(x_data, y_data, options)
    else:
        raise ValueError('GP Engine %s not found.' % gp_engine)

def gp_regression(gp_engine, x_data, y_data, options=None):
    """Do GP regression and return GP object.
    Args:
        gp_engine: The GP implementation to use.
        x_data: The list of x_data seen so far.
        y_data: The list of y_data seen so far.
        options: Options Namespace object.
    """
    if options is None:
        options = load_options(euclidean_gp_args, cmd_line=False)
        options.hp_tune_criterion = 'ml'
        options.kernel_type = 'se'
        options.hp_samples = 0
    gpf = get_gp_fitter(gp_engine, x_data, y_data, options)
    gpf.fit_gp()
    return gpf.get_next_gp()

def get_best_dragonfly_prior(f, domain, kernel_type='se', num_samples=100):
    """
    Get best empirical prior for a function.
    Args:
        f: The function to model.
        domain: Domain of function as a list of lists. [[dim1_bds], ...]
        kernel_type: Name of the kernel to use (see dragonfly for names).
        num_samples: Number of samples to fit kernel hps on.
    Returns: GP object.
    """
    x_data, y_data = [], []
    low_b, high_b = zip(*domain)
    for _ in range(num_samples):
        pt = np.random.uniform(low_b, high_b)
        y_data.append(f(pt))
        x_data.append(pt)
    tuned_gp, options = get_tuned_gp('dragonfly', x_data, y_data, kernel_type)
    tuned_kernel = tuned_gp.gp_core.kernel
    tuned_mean = tuned_gp.gp_core.mean_func
    tuned_noise = tuned_gp.gp_core.noise_var
    empty_core = EuclideanGP([], [], tuned_kernel, tuned_mean, tuned_noise,
                             build_posterior=False)
    return DragonflyGP(empty_core, options)

def get_best_dragonfly_joint_prior(f_infos, kernel_type='se', samps_per=100):
    """
    Get best empirical prior for a function.
    Args:
        f_infos: Namespace instances for the slices of the function.
        kernel_type: Name of the kernel to use (see dragonfly for names).
        samps_per: Number of samples to take per slice.
    Returns: GP object.
    """
    x_data, y_data = None, []
    act_low, act_high = zip(*f_infos[0].domain)
    act_dim = len(act_low)
    for f_info in f_infos:
        prefixes = np.tile(f_info.f_loc, samps_per).reshape(samps_per, -1)
        rand_acts = np.random.uniform(act_low, act_high, (samps_per, act_dim))
        y_data += [f_info.function(a) for a in rand_acts]
        f_pts = np.hstack([prefixes, rand_acts])
        if x_data is None:
            x_data = f_pts
        else:
            x_data = np.vstack([x_data, f_pts])
    tuned_gp, options = get_tuned_gp('dragonfly', x_data, y_data, kernel_type)
    tuned_kernel = tuned_gp.gp_core.kernel
    tuned_mean = tuned_gp.gp_core.mean_func
    tuned_noise = tuned_gp.gp_core.noise_var
    empty_core = EuclideanGP([], [], tuned_kernel, tuned_mean, tuned_noise,
                             build_posterior=False)
    return DragonflyGP(empty_core, options)

def get_tuned_gp(gp_engine, x_data, y_data, kernel_type='se'):
    """
    Get a tuned gp for the data seen.
    Args:
        x_data: Data as ndarray.
        y_data: List of y_values seen.
        kernel: Type of kernel to use (see dragonfly names).
    Returns: Tuned GP as well as the options used.
    """
    options = load_options(euclidean_gp_args, cmd_line=False)
    options.kernel_type = kernel_type
    options.hp_tune_criterion = 'ml'
    options.hp_samples = 0
    gp_fitter = get_gp_fitter(gp_engine, x_data, y_data, options)
    gp_fitter.fit_gp()
    return gp_fitter.get_next_gp(), options
