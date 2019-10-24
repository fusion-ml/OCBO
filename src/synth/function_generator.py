"""
Methods for getting random functions to optimize over.
"""

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mvn

from dragonfly.utils.option_handler import get_option_specs

rand_fs_args = [\
        get_option_specs('rand_masses', False, 0,
            'Number of random mass functions to have.'),
        get_option_specs('easy_masses', False, 0,
            'Number of easy random mass functions to have.'),
        get_option_specs('med_masses', False, 0,
            'Number of medium random mass functions to have.'),
        get_option_specs('hard_masses', False, 0,
            'Number of hard random mass functions to have.'),
        get_option_specs('max_masses', False, 10,
            'Maximum number of masses a function can have.'),
        get_option_specs('min_bandwidth', False, 0.25,
            'Minimum bandwidth a mass can have.'),
        get_option_specs('max_bandwidth', False, 0.5,
            'Maximum bandwidth a mass can have.'),
        get_option_specs('rand_mass_scale', False, 1,
            'Maximum height/depth a mass can have.'),
        get_option_specs('global_mass_scale', False, None,
            'Scale to have globally across masses'),
        get_option_specs('massf_dim', False, 2,
            'Dimension of the mass functions.'),
]

def get_random_mass_functions(num_fs, max_masses, min_bandwidth, max_bandwidth,
                              max_scale, domain):
    """Create random function to optimize that is sum of random point masses.
    Args:
        num_fs: Number of functions to return.
        max_masses: The maximum number of masses a function can have.
        max_bandwidth: The maximum bandwidth in any dimension.
        max_scale: The maximum magnitude of scale for any mass.
        domain: Domain for the functions as [[dim1_low, dim1_high],...]
    Returns: List of Namespaces with function info. Note that the max_val entry
        will be None in this.
    """
    f_infos = []
    for idx in range(num_fs):
        num_masses = np.random.randint(max_masses + 1)
        f_info = create_mass_func(num_masses, [0.1, max_bandwidth],
                                  [0, max_scale], domain)
        f_info.name = 'rand_mass_%d' % idx
        f_infos.append(f_info)
    return f_infos

def get_hard_mass_functions(num_fs, domain, scaling=None):
    """Create a list of hard mass functions.
    Returns: List of Namespaces with function info. Note that the max_val entry
        will be None in this.
    """
    f_infos = []
    scaling = scaling if scaling is not None else 10
    for idx in range(num_fs):
        num_masses = np.random.randint(5, 8)
        f_info = create_mass_func(num_masses, [0.25, 0.4],
                                  [0.5 * scaling, scaling], domain)
        f_info.name = 'hard_mass_%d' % idx
        f_infos.append(f_info)
    return f_infos

def get_med_mass_functions(num_fs, domain, scaling=None):
    """Create a list of hard mass functions.
    Returns: List of Namespaces with function info. Note that the max_val entry
        will be None in this.
    """
    f_infos = []
    scaling = scaling if scaling is not None else 5
    for idx in range(num_fs):
        num_masses = np.random.randint(3, 5)
        f_info = create_mass_func(num_masses, [0.4, 0.6],
                                  [0.25 * scaling, scaling], domain)
        f_info.name = 'med_mass_%d' % idx
        f_infos.append(f_info)
    return f_infos

def get_easy_mass_functions(num_fs, domain, scaling=None):
    """Create a list of easy mass functions.
    Returns: List of Namespaces with function info. Note that the max_val entry
        will be None in this.
    """
    f_infos = []
    scaling = scaling if scaling is not None else 1
    for idx in range(num_fs):
        num_masses = np.random.randint(0, 3)
        f_info = create_mass_func(num_masses, [0.7, 0.9], [0, scaling],
                                  domain)
        f_info.name = 'easy_mass_%d' % idx
        f_infos.append(f_info)
    return f_infos

def create_mass_func(num_masses, band_bounds, scale_bounds, domain):
    """Create a mass function.
    Args:
        num_masses: Number of masses in the function.
        band_bounds: Range of the bandwidths to generate as [lower, upper].
        scale_bounds: Range of the scales to generate as [lower, upper].
        domain: Domain of the function as [[dim1_low, dim1_high],...]
    Returns: Namespace detailing function.
    """
    lows, highs = zip(*domain)
    all_centers = np.random.uniform(lows, highs, (num_masses, len(lows)))
    all_bands = np.random.uniform(band_bounds[0], band_bounds[1],
                                  (num_masses, len(domain)))
    all_scales = np.random.uniform(scale_bounds[0], scale_bounds[1], num_masses)
    all_scales *= (np.random.binomial(1, 0.5, num_masses) - 0.5) * 2
    def create_mass(centers, bands, scale):
        return lambda x: scale * np.prod([np.exp(-1 * (x[i] - centers[i]) ** 2\
                / bands[i]) for i in range(len(domain))])
    masses = [create_mass(all_centers[idx], all_bands[idx], all_scales[idx])
              for idx in range(num_masses)]
    f = lambda x: np.sum([m(x) for m in masses])
    return Namespace(function=f, domain=domain, max_val=None,
                     center=all_centers, bands=all_bands, scales=all_scales)

def assemble_named_add_functions():
    """Assemble all of the named additive functions."""
    twentyfour = get_additive_function(24, 6)
    twentyfour.name = 'add24'
    twelve = get_additive_function(12, 3)
    twelve.name = 'add12'
    return [twentyfour, twelve]

def get_additive_function(num_dims, num_dims_per_group):
    """Get a function composed of three mode functions.
    Args:
        num_dims: Number of total dimensions for the function.
        num_dims_per_group: Number of dimensions each group should have.
            For now these needs to go into num_dims perfectly.
    Returns: Namespace containing the function.
    """
    sub_func = _get_three_mode(num_dims_per_group)
    shuffled_idxs = np.arange(num_dims)
    np.random.shuffle(shuffled_idxs)
    groups = num_dims / num_dims_per_group
    def add_func(x):
        x = x.ravel()
        shuffled = x[shuffled_idxs].ravel()
        total = 0
        for group in range(groups):
            start = group * num_dims_per_group
            end = (group + 1) * num_dims_per_group
            total += sub_func(shuffled[start:end])
        return total
    domain = [[-1, 1] for _ in range(num_dims)]
    return Namespace(function=add_func, domain=domain, max_val=None)

def _get_three_mode(num_d):
    lower_bound = -700
    variances = 0.01 * num_d ** 0.1 * np.ones(num_d)
    centers = np.asarray([[0.62, -0.38], [0.18, 0.58], [-0.58, -0.56]])
    centers = np.hstack([centers] + [np.asarray([[-0.66, -0.19, 0.62]]).T
                                     for _ in range(num_d - 2)])
    probs = [0.1, 0.8, 0.1]
    def to_return(x):
        total = 0
        for idx in range(3):
            total += probs[idx] * mvn.pdf(x, centers[idx], variances)
        total = np.log(total)
        return max([lower_bound, total])
    return to_return

def _plot_random_masses():
    max_masses = 10
    max_bandwidth = 0.5
    max_scale = 0.5
    domain = [[0, 1], [0, 1]]
    f_infos = get_random_mass_functions(9, max_masses, max_bandwidth, max_scale,
                                        domain)
    plot_idx = 1
    for f_info in f_infos:
        grid = np.zeros((100, 100))
        for x_idx, x in enumerate(np.linspace(0, 1, 100)):
            for y_idx, y in enumerate(np.linspace(0, 1, 100)):
                grid[x_idx, y_idx] = f_info.function([x, y])
        plt.subplot(3, 3, plot_idx)
        plt.imshow(grid)
        plot_idx += 1
    plt.show()

# Assemble add functions.
add_funcs = assemble_named_add_functions()

if __name__ == '__main__':
    _plot_random_masses()
