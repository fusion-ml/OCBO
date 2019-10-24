"""
Utility function having to do with GP.
"""

from __future__ import division

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro
from copy import copy, deepcopy
import itertools

from dragonfly.exd.exd_utils import maximise_with_method
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.gp.euclidean_gp import EuclideanGP, EuclideanGPFitter,\
        euclidean_gp_args
from dragonfly.opt.gpb_acquisitions import asy_ucb, asy_ei, asy_ttei,\
        get_gp_sampler_for_parallel_strategy, maximise_acquisition,\
        _get_ucb_beta_th, _get_gp_eval_for_parallel_strategy,\
        _get_gp_ucb_dim, _expected_improvement_for_norm_diff
from dragonfly.utils.option_handler import load_options

from synth import synth_functions

def uniform_draw(domain, num_samps):
    """Draw from uniform distribution in the given domain.
    Args:
        domain: List of lists [[low1, high1],...]
        num_samps: Number of samples.
    """
    lows, highs = zip(*domain)
    return np.random.uniform(lows, highs, (num_samps, len(lows)))

def get_best_prior(f, domain, kernel_type='se', num_samples=100):
    """
    DEPRECATED!
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
    tuned_gp = get_tuned_gp(x_data, y_data, kernel_type)
    return EuclideanGP([], [], tuned_gp.kernel, tuned_gp.mean_func,
                       tuned_gp.noise_var, build_posterior=False)

def get_tuned_gp(x_data, y_data, kernel_type='se'):
    """
    DEPRECATED: USE gp_regression in gp_util.py instead!
    Get a tuned gp for the data seen.
    Args:
        x_data: Data as ndarray.
        y_data: List of y_values seen.
        kernel: Type of kernel to use (see dragonfly names).
    """
    options = load_options(euclidean_gp_args, cmd_line=False)
    options.kernel_type = kernel_type
    _, tuned_gp, _ = EuclideanGPFitter(x_data, y_data, options).fit_gp()
    return tuned_gp

def bests_to_rewards(bests):
    """Convert best rewards seen so far to rewards.
    Args:
        bests: List of best values seen so far.
    Returns: Transformed list of rewards.
    """
    return [bests[0]] + [bests[i] - bests[i - 1] for i in range(1, len(bests))]

def assemble_functions(names=None):
    """Get function information from names.
    Args:
        names: String of function names as X,X,X... If None, get all functions.
    Returns: List of namespaces in same order as name.
    """
    if names is None:
        return []
    names = names.split(',')
    to_return = []
    for name_idx, name in enumerate(names):
        found = False
        for f_info in synth_functions:
            if f_info.name.lower() == name.lower():
                f_copy = deepcopy(f_info)
                f_copy.name = f_info.name + ('_%d' % name_idx)
                to_return.append(f_copy)
                found = True
                break
        if not found:
            raise ValueError('Invalid function: %s' % name)
    return to_return

def expected_improvement_for_norm_diff(norm_diff):
  """ The expected improvement. """
  return norm_diff * normal_distro.cdf(norm_diff) + normal_distro.pdf(norm_diff)

def sample_grid(f_locs, function_domain, num_pts):
    """Given location of functions, randomly create a grid of points.
    Args:
        f_locs: The location of the functions as a list of ndarrays.
        function_domain: The domain of the functions in question represented
            as a list of lists [[dim1_low, dim1_high], ...]
        num_pts: Number of points to sample fo each of the functions.
    """
    lows, highs = zip(*function_domain)
    num_funcs = len(f_locs)
    f_locs = np.tile(np.asarray(f_locs), num_pts).reshape(-1, len(f_locs[0]))
    f_pts = np.random.uniform(lows, highs, (num_funcs * num_pts, len(lows)))
    return np.hstack([f_locs, f_pts])

def draw_all_related(ref_pt, kernel, f_locs, function_domain, num_pts,
                     ctx_thresh, action_thresh, action_increment=0.05):
    loc_idxs = []
    for f_idx, f_loc in enumerate(f_locs):
        if loc_is_related(ref_pt, f_loc, kernel, ctx_thresh):
            loc_idxs.append(f_idx)
    if not loc_idxs:
        loc_idxs = [f_locs.index(ref_pt[:len(f_locs[0])])]
    capital_per = int(num_pts / len(loc_idxs))
    all_pts = []
    for loc_idx in loc_idxs:
        all_pts.append(sample_related(ref_pt, kernel, f_locs[loc_idx],
                                      function_domain, num_pts, ctx_thresh,
                                      action_thresh,
                                      action_increment=action_increment))
    return loc_idxs, np.vstack(all_pts)

def sample_related(ref_pt, kernel, f_loc, function_domain, num_pts,
                   ctx_thresh, action_thresh, action_increment=0.05):
    """Sample related points in a context. Assume dimension 1 action space.
    Args:
        f_locs: The location of the functions as a list of ndarrays.
        function_domain: The domain of the functions in question represented
            as a list of lists [[dim1_low, dim1_high], ...]
        num_pts: Number of points to sample fo each of the functions.
    """
    if len(function_domain) > 1:
        raise NotImplementedError('Only implemented for one dim action')
    # Find the lower and upper bound of action to look at.
    lower = np.append(f_loc, ref_pt[len(f_loc):])
    while lower[-1] > function_domain[0][0] \
            and kernel([lower], [ref_pt]) > action_thresh:
        lower[-1] -= action_increment
    upper = np.append(f_loc, ref_pt[len(f_loc):])
    while upper[-1] < function_domain[0][1] \
            and kernel([upper], [ref_pt]) > action_thresh:
        upper[-1] += action_increment
    low = max(lower[-1], function_domain[0][0])
    high = min(upper[-1], function_domain[0][1])
    if low == high:
        low = max(low - action_increment, function_domain[0][0])
        high = min(high + action_increment, function_domain[0][1])
    # Draw Randomly.
    act_pts = np.random.uniform(low, high, num_pts).reshape(-1, 1)
    locations = np.repeat(np.asarray(f_loc), num_pts).reshape(-1, 1)
    return np.hstack([locations, act_pts])

def loc_is_related(ref_pt, f_loc, kernel, ctx_thresh):
    # Judge if f_loc is close enough.
    ctx_pt = np.append(f_loc, ref_pt[len(f_loc):])
    relatedness = float(kernel([ctx_pt], [ref_pt]))
    return relatedness >= ctx_thresh

def build_gp_posterior(gp):
    """Build the posterior for the gp if it hasn't been built yet."""
    gp.build_posterior()

def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in itertools.combinations_with_replacement(masks, n):
        yield sum(c)

def knowledge_gradient(a, b):
    """Where a and b are vectors calculate...
    E[max{a_1 + b_1 Z,..., a_n + b_n Z}]
    where Z is a normal random variable.
    """
    # Step 1: Sort (a_i, b_i) by values of b in increasing order.
    joined = list(zip(b, a))
    joined.sort()
    b, a = zip(*joined)
    # Step 2: Find vertices of the epigraph.
    a -= np.max(a)
    idxs = [0, 1]
    zs = [np.float('-inf'), (a[0] - a[1]) / (b[1] - b[0])]
    for i in range(2, len(a)):
        j = idxs[-1]
        z = (a[i] - a[j]) / (b[j] - b[i])
        while z < zs[-1]:
            idxs = idxs[:-1]
            zs = zs[:-1]
            j = idxs[-1]
            z = (a[i] - a[j]) / (b[j] - b[i])
        else:
            idxs.append(i)
            zs.append(z)
    zs.append(float('inf'))
    result = 0
    cdfs = normal_distro.cdf(zs)
    pdfs = normal_distro.pdf(zs)
    for i, idx in enumerate(idxs):
        cdf_dif = cdfs[i + 1] - cdfs[i]
        pdf_dif = pdfs[i] - pdfs[i+1]
        result += a[idx] * cdf_dif + b[idx] * pdf_dif
    return result

def get_eval_set(ctx_fidel, act_fidel, ctx_domain, act_domain):
    """Get grid used to evaluate score.
    Returns set of points as both list of points and a grid format
    i.e. dim = (num_ctxs, num_acts, point).
    """
    ctx_dim = len(ctx_domain)
    act_dim = len(act_domain)
    rand_ctxs = uniform_draw(ctx_domain, ctx_fidel)
    rand_ctxs = np.repeat(rand_ctxs, act_fidel, axis=0)
    rand_acts = uniform_draw(act_domain, act_fidel)
    rand_acts = np.tile(rand_acts.ravel(), ctx_fidel)\
                  .reshape(ctx_fidel * act_fidel,
                           act_dim)
    eval_pts = np.hstack([rand_ctxs, rand_acts])
    eval_grid = eval_pts.reshape(ctx_fidel, act_fidel,
                                 ctx_dim + act_dim)
    return eval_pts, eval_grid
"""
====================ACQUISITIONS=========================
"""

def ucb_acq(gp, domain, max_evals, t):
    """Do UCB acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
    """
    if not isinstance(gp, DragonflyGP):
        raise NotImplementedError('UCB not supported for non-DragonflyGP.')
    gp = gp.gp_core
    euc_domain = EuclideanDomain(domain)
    anc_data = _form_basic_direct_anc_data(euc_domain, max_evals)
    anc_data.t = t
    beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
    gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
    def _ucb_acq(x):
        """ Computes the GP-UCB acquisition. """
        mu, sigma = gp_eval(x)
        return mu + beta_th * sigma
    acquisition = lambda x: _ucb_acq(x.reshape(1, -1))
    opt_val, opt_pt = maximise_with_method(anc_data.acq_opt_method, acquisition,
                                           euc_domain, max_evals)
    return opt_pt, opt_val

def ei_acq(gp, domain, max_evals, curr_max_val, rand_opt=True):
    """Do TTEI acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
        curr_max_val: The current highest value seen so far.
    """
    if rand_opt or not isinstance(gp, DragonflyGP):
        return _rand_ei_acq(gp, domain, max_evals, curr_max_val)
    gp = gp.gp_core
    euc_domain = EuclideanDomain(domain)
    anc_data = _form_basic_direct_anc_data(euc_domain, max_evals)
    anc_data.curr_max_val = curr_max_val
    gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
    def _ei_acq(x):
        mu, sigma = gp_eval(x)
        norm_diff = (mu - anc_data.curr_max_val) / sigma
        return sigma * _expected_improvement_for_norm_diff(norm_diff)
    acquisition = lambda x: _ei_acq(x.reshape((1, -1)))
    max_val, max_pt = maximise_with_method(anc_data.acq_opt_method, acquisition,
                                           euc_domain, anc_data.max_evals,
                                           vectorised=True)
    return max_pt, max_val

def ttei_acq(gp, domain, max_evals, curr_max_val):
    """Do TTEI acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
        curr_max_val: The current highest value seen so far.
    """
    if not isinstance(gp, DragonflyGP):
        raise NotImplementedError('UCB not supported for non-DragonflyGP.')
    gp = gp.gp_core
    euc_domain = EuclideanDomain(domain)
    anc_data = _form_basic_direct_anc_data(euc_domain, max_evals)
    anc_data.curr_max_val = curr_max_val
    return asy_ttei(gp, anc_data)

def sub_ei_acq(gp, domain, max_evals, curr_max_val):
    if not isinstance(gp, DragonflyGP):
        raise NotImplementedError('UCB not supported for non-DragonflyGP.')
    gp = gp.gp_core
    euc_domain = EuclideanDomain(domain)
    anc_data = _form_basic_direct_anc_data(euc_domain, max_evals)
    anc_data.curr_max_val = curr_max_val

    #gpb_acquisitions.asy_ttei's second case
    max_acq_opt_evals = anc_data.max_evals
    anc_data = copy(anc_data)
    anc_data.max_evals = max_acq_opt_evals//2
    ei_argmax = asy_ei(gp, anc_data)

    # Now return the second argmax
    #gpb_acquisitions._ttei
    ref_point = copy(ei_argmax)

    ref_mean, ref_std = gp.eval([ref_point], 'std')
    ref_mean = float(ref_mean)
    ref_std = float(ref_std)
    def _tt_ei_acq(x):
        """ Acquisition for TTEI. """
        mu, sigma = gp.eval(x, 'std')
        comb_std = np.sqrt(ref_std ** 2 + sigma ** 2)
        norm_diff = (mu - ref_mean) / comb_std
        return comb_std * expected_improvement_for_norm_diff(norm_diff)
    return maximise_acquisition(_tt_ei_acq, anc_data), ref_mean, ref_std

def thomp_acq(gp, domain, max_evals, t, rand_opt=True):
    """Do TS acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
    Returns: The point at which the max was found and the value of the max.
    """
    if rand_opt or not isinstance(gp, DragonflyGP):
        return _rand_ts_acq(gp, domain, max_evals)
    gp = gp.gp_core
    euc_domain = EuclideanDomain(domain)
    anc_data = _form_basic_direct_anc_data(euc_domain, max_evals)
    anc_data.t = t
    anc_data.acq_opt_method = 'rand'
    gp_sample = get_gp_sampler_for_parallel_strategy(gp, anc_data)
    max_val, max_pt = maximise_with_method(anc_data.acq_opt_method, gp_sample,
                                           euc_domain, anc_data.max_evals,
                                           vectorised=True)
    return max_pt, max_val

def _rand_ts_acq(gp, domain, max_evals):
    """Do random TS acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
    Returns: The point at which the max was found and the value of the max.
    """
    pts = uniform_draw(domain, max_evals)
    sample = gp.draw_sample(pts)
    max_idx = np.argmax(sample)
    return pts[max_idx], sample[max_idx]

def _rand_ei_acq(gp, domain, max_evals, curr_max_val):
    """Do random EI acquisition.
    Args:
        gp: The GP to use.
        domain: In the form of [[dim1], [dim2], ...]
        max_evals: Max number of evaluations in opt method.
        t: Current time.
    Returns: The point at which the max was found and the value of the max.
    """
    pts = uniform_draw(domain, max_evals)
    means, covmat = gp.eval(pts, include_covar=True)
    stds = np.sqrt(covmat.diagonal().ravel())
    norm_diff = (means - curr_max_val) / stds
    eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
            + normal_distro.pdf(norm_diff))
    ei_val = np.max(eis)
    ei_pt = pts[np.argmax(eis)]
    return ei_pt, ei_val

def _form_basic_direct_anc_data(euc_domain, max_evals):
    """Form anc_data for dragonfly acquisition methods.
    Args:
        euc_domain: Wrapped domain object.
        max_evals: Maximum amount of evaluations to do when optimizing.
        t: Current time.
    """
    return Namespace(acq_opt_method='direct',
                     handle_parallel=None,
                     is_mf=False,
                     domain=euc_domain,
                     max_evals=max_evals)

