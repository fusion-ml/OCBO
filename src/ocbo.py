"""
Main file for running DOCBO simulations.
"""

import os
import pickle as pkl
from copy import deepcopy

from dragonfly.utils.option_handler import load_options, get_option_specs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gp.gp_util import get_best_dragonfly_joint_prior
from strategies import strategies, strat_args
from strategies.multi_opt import multi_opt_args
from synth.function_generator import rand_fs_args, get_random_mass_functions,\
        get_hard_mass_functions, get_easy_mass_functions, get_med_mass_functions
from synth.continuous_2d import assemble_chopped_1d_functions,\
        get_chopped_h42, get_chopped_h22, get_chopped_h31
from util.misc_util import get_best_prior, assemble_functions, partitions
from util.plotting_util import plot_summary

docbo_args = [\
        get_option_specs('run_id', False, None, 'Unique ID for the run.'),
        get_option_specs('options', False, None, 'Path to options file.'),
        get_option_specs('trials', False, 10, 'Number of trials to run.'),
        get_option_specs('max_capital', False, 50, 'Max number of capital.'),
        get_option_specs('init_capital', False, 5,
            'Number of initial capital to have per opt method.'),
        get_option_specs('methods', False, None,
            'Methods to test given as X,X,X-... All run if none provided.'),
        get_option_specs('functions', False, None,
            'Functions to test over given as X,X,X.... All if none provided.'),
        get_option_specs('cts_f', False, None,
            'Name of the contiuous function to chop up and use.'),
        get_option_specs('chop_h4_2', False, False,
            'Whether to chop up Hartmann 6 and add.'),
        get_option_specs('chop_h2_2', False, False,
            'Whether to chop up Hartmann 4 in to 2-2'),
        get_option_specs('chop_h3_1', False, False,
            'Whether to chop up Hartmann 4 in to 3-1'),
        get_option_specs('num_chops', False, 5,
            'Number of chops of continuous function to use.'),
        get_option_specs('tune_every', False, 1, 'How often to tune HPs.'),
        get_option_specs('pretune_for_joint', False, False,
            'Whether to pretune GP parameters for joint function.'),
        get_option_specs('tunings', False, None,
            ('List of tuning methods seperated by commas corresponding to each '
             'specified method. e.g. if there are two different methods could '
             'have ml,ml-post_sampling.')),
        get_option_specs('write_dir', False, None,
            'Path to directory to write results. Does not write if None.'),
        get_option_specs('show_fig', False, False, 'Show plot or not.'),
        get_option_specs('write_fig', False, False, 'Whether to write.'),
        get_option_specs('description', False, None,
            'String giving summary of the run.'),
        get_option_specs('max_opt_evals', False, 100,
            'Number of evaluations to make for optimization methods.'),
        get_option_specs('debug', False, False, 'Whether to start in debug.'),
        get_option_specs('fixed_prob_acq', False, None,
            'Acq to use for fixed prob'),
        get_option_specs('set_seed', False, True,
            'Whether to fix the random seed.'),
]

def run():
    """Runs all the simulations."""
    # Load in options for running the simulation.
    options = load_options(docbo_args + strat_args + rand_fs_args,
                           cmd_line=True)
    if options.debug:
        import pudb; pudb.set_trace()
    if options.set_seed:
        np.random.seed(100826730)
    if options.run_id is None:
        raise ValueError('run_id not specified.')
    rand_state = np.random.get_state()
    function_info = assemble_functions(options.functions)
    function_info += _assemble_random_functions(options)
    if options.cts_f is not None:
        function_info += assemble_chopped_1d_functions(options.cts_f,
                                                       options.num_chops)
    if options.chop_h4_2:
        function_info += get_chopped_h42()
    if options.chop_h2_2:
        function_info += get_chopped_h22()
    if options.chop_h3_1:
        function_info += get_chopped_h31()
    if len(function_info) == 0:
        raise ValueError('No functions specified')
    # Set up GPs and create optimizer classes.
    gps = _get_function_gps(function_info, options)
    methods = _assemble_methods(gps, function_info, options)
    if options.write_dir is not None:
        created_dir = _set_up_write_dir(options)
        _write_to_created(created_dir, 'run_info.pkl', options)
        _write_to_created(created_dir, 'rand_state.pkl', rand_state)
        _write_function_info(function_info, created_dir)
    else:
        created_dir = None
    # Run simulations.
    all_results = []
    for t_id in tqdm(range(options.trials)):
        results = _run_single_opt(methods, function_info, options)
        if options.write_dir is not None:
            _write_to_created(created_dir, 'trial_%d.pkl' % t_id, results)
        if options.show_fig or options.write_fig:
            all_results.append(results)
    # Write options for
    if options.show_fig or options.write_fig:
        if options.write_fig:
            if options.write_dir is None:
                raise ValueError('Write dir must be specified to write figure.')
            fig_write_path = '%s.png' % options.run_id if options.write_fig \
                             else None
            fig_write_path = os.path.join(created_dir, fig_write_path)
        else:
            fig_write_path = None
        plot_summary(function_info, all_results, options.show_fig,
                     fig_write_path)
    # If we were doing joint version of Branin visualize how points were
    # selected on the surface.
    if options.show_fig and options.cts_f == 'branin':
        _plot_joint_branin(all_results[0], function_info, created_dir)


def _run_single_opt(methods, function_info, options):
    """Do optimization for each method."""
    # Get initial points that should be used for all methods.
    init_pts = []
    for wrap in function_info:
        low, high = zip(*wrap.domain)
        init_pts.append([np.random.uniform(low, high)
                         for _ in range(options.init_capital)])
    results = {}
    for m_idx, method in enumerate(methods):
        method.set_clean()
        histories = method.optimize(options.max_capital, init_pts=init_pts)
        if options.tunings is not None:
            results[method.get_method_name_with_tuning()] = histories
        else:
            results[method.get_opt_method_name()] = histories
    return results

def _get_function_gps(function_info, options):
    """Get gps corresponding to each GP."""
    if options.pretune_for_joint:
        options.tune_every = 0
        pre_gp = get_best_dragonfly_joint_prior(function_info)
        return [pre_gp for _ in function_info]
    if options.tune_every > 0:
        return None
    gps = []
    for wrap in function_info:
        gps.append(get_best_prior(wrap.function, wrap.domain,
                                  num_samples=(len(wrap.domain) * 100)))
    return gps

def _assemble_methods(gps, function_info, options):
    """Assemble the methods used to optimize."""
    methods = []
    if options.methods is None:
        raise ValueError('Methods must be specified.')
    else:
        method_names = options.methods.split(',')
        tunings = None
        rn_info = None
        if options.tunings is not None:
            tunings = options.tunings.split(',')
            if len(tunings) != len(method_names):
                raise ValueError(('Number of tuning methods must match number '
                                  'of methods.'))
        for m_idx, m_name in enumerate(method_names):
            found = False
            for strat in strategies:
                if m_name.lower() == strat.name.lower():
                    if tunings is not None:
                        options.tuning_methods = tunings[m_idx]
                    methods.append(strat.impl(function_info, options,
                                              pre_tuned_gps=gps,
                                              rn_info=rn_info))
                    rn_info = (methods[-1].rn_grids, methods[-1].rn_bests)
                    found = True
                    break
            if not found:
                raise ValueError('Invalid method: %s' % m_name)
        # If we are using ran rn_grid then we do have a maximum for the
        # function. Set that here.
        if rn_info[1] is not None:
            for f_idx, max_val in enumerate(rn_info[1]):
                function_info[f_idx].max_val = max_val
    return methods

def _set_up_write_dir(options):
    """Create directory for this run_id and return path to directory."""
    create_dir = os.path.join(options.write_dir, options.run_id)
    if os.path.isdir(create_dir):
        raise ValueError('Run ID already exists: %s', options.run_id)
    else:
        os.makedirs(create_dir)
        return create_dir

def _write_to_created(created_dir, name, to_dump):
    """Write info to file.
    Args:
        created_dir: Path of created directory.
        name: Name of the file.
        to_dump: Information to dump to file.
    """
    write_path = os.path.join(created_dir, name)
    with open(write_path, 'wb') as f:
        pkl.dump(to_dump, f, protocol=2)

def _assemble_random_functions(options):
    """Assembles random functions as specified by options."""
    rand_infos = []
    domain = [[0, 1] for _ in range(options.massf_dim)]
    rand_infos += get_random_mass_functions(options.rand_masses,
            options.max_masses, options.min_bandwidth,
            options.max_bandwidth, options.rand_mass_scale, domain)
    scaling = options.global_mass_scale
    scaling = scaling if scaling is None else int(scaling)
    rand_infos += get_hard_mass_functions(options.hard_masses, domain, scaling)
    rand_infos += get_med_mass_functions(options.med_masses, domain, scaling)
    rand_infos += get_easy_mass_functions(options.easy_masses, domain, scaling)
    return rand_infos

def _write_function_info(f_infos, write_dir):
    """Write function infos to a file."""
    # Remove the function part of the f_infos since this cannot be saved.
    f_clone = [deepcopy(f_info) for f_info in f_infos]
    for f_info in f_clone:
        f_info.function = None
    _write_to_created(write_dir, 'functions.pkl', f_clone)

def _plot_joint_branin(result, f_infos, created_dir=None):
    """Visualize how joint discrete branin behaved."""
    branin_info = assemble_functions('branin')[0]
    domain = branin_info.domain
    # Make the surface.
    ctx_grid = np.linspace(domain[0][0], domain[0][1], 100)
    act_grid = np.linspace(domain[1][0], domain[1][1], 100)
    surf_grid = np.ndarray((100, 100))
    for ci, ctx in enumerate(ctx_grid):
        for ai, act in enumerate(act_grid):
            surf_grid[-1 * (ai + 1), ci] = branin_info.function([ctx, act])
    # For each of the methods...
    first_to_plot = True
    for method, res in result.iteritems():
        plt.imshow(surf_grid, cmap='jet', extent=(domain[0][0], domain[0][1],
                                                  domain[1][0], domain[1][1]))
        # Plot the points
        query_x, query_y = [], []
        rn_x, rn_y = [], []
        for finf in f_infos:
            # Get queries.
            f_queries = [q_info.pt for q_info in res.query_history[finf.name]]
            query_x += [finf.f_loc] * len(f_queries)
            query_y += f_queries
            # Get risk neutral estimates.
            f_rn = [choice[1] for choice in res.rn_choice[finf.name]][-1:]
            rn_x += [finf.f_loc] * len(f_rn)
            rn_y += f_rn
            plt.axvline(finf.f_loc, color='black', ls='--', alpha=0.25)
        plt.scatter(query_x, query_y, label='Queried', s=20)
        plt.scatter(rn_x, rn_y, marker='*', color='y', s=150, label='Estimates')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if first_to_plot:
            plt.legend(bbox_to_anchor=[1, 1])
            first_to_plot = False
        meth_name = method.upper()
        meth_name = meth_name.replace('JOINT-', '')
        plt.title(meth_name)
        if created_dir is not None:
            plt.savefig('%s/joint_branin_viz_%s.pdf' % (created_dir, method),
                        format='pdf')
        plt.show()
        plt.close()

if __name__ == '__main__':
    run()
