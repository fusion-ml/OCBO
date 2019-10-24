"""
Main file for running synthetic continuous experiments.
"""

import os
import pickle as pkl
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cstrats import cstrats, copts
from dragonfly.utils.option_handler import load_options, get_option_specs
from util.misc_util import assemble_functions, uniform_draw
from util.plotting_util import _plot_perf, _get_cmap

ocbo_args = [\
        get_option_specs('run_id', False, None, 'Unique ID for the run.'),
        get_option_specs('options', False, None, 'Path to options file.'),
        get_option_specs('trials', False, 3, 'Number of trials to run.'),
        get_option_specs('max_capital', False, 50, 'Max number of capital.'),
        get_option_specs('init_capital', False, 5,
            'Number of initial capital to have per opt method.'),
        get_option_specs('methods', False, None,
            'Methods to test given as X,X,X-... All run if none provided.'),
        get_option_specs('function', False, None,
            'Functions to test over given as X,X,X.... All if none provided.'),
        get_option_specs('act_dim', False, 1,
            'Size of the action domain.'),
        get_option_specs('ctx_constraints', False, None,
            ('Number of contexts to use to constrain evaluation picking.'
             'Contexts are picked uniformly at random.')),
        get_option_specs('hp_tune_samps', False, None,
            'Number of samples to use to tune HPs which will then be fixed.'),
        get_option_specs('write_dir', False, None,
            'Path to directory to write results. Does not write if None.'),
        get_option_specs('show_fig', False, False, 'Show plot or not.'),
        get_option_specs('write_fig', False, False, 'Whether to write.'),
        get_option_specs('description', False, None,
            'String giving summary of the run.'),
        get_option_specs('debug', False, False, 'Whether to start in debug.'),
        get_option_specs('set_seed', False, True,
            'Whether to fix the random seed.'),
]

def run():
    """Runs all the simulations."""
    # Load in options for running the simulation.
    options = load_options(ocbo_args + copts, cmd_line=True)
    if options.debug:
        import pudb; pudb.set_trace()
    if options.set_seed:
        np.random.seed(100826730)
    if options.run_id is None:
        raise ValueError('run_id not specified.')
    # Get information and write to files.
    rand_state = np.random.get_state()
    f_info = assemble_functions(options.function)[0]
    methods = _assemble_methods(f_info, options)
    if options.write_dir is not None:
        created_dir = _set_up_write_dir(options)
        _write_to_created(created_dir, 'run_info.pkl', options)
        _write_to_created(created_dir, 'rand_state.pkl', rand_state)
        eval_set = (methods[0].eval_pts, methods[0].eval_grid)
        _write_to_created(created_dir, 'eval_set.pkl', eval_set)
        pure_f_info = deepcopy(f_info)
        pure_f_info.function = None
        _write_to_created(created_dir, 'function_info.pkl', pure_f_info)
    # Assemble the methods that will be tested.
    # Run simulations.
    all_results = []
    for t_id in tqdm(range(options.trials)):
        results = _run_single_opt(methods, f_info, options)
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
        _plot_results(all_results, options.show_fig, fig_write_path)
        plt.close()
        if len(f_info.domain) == 2:
            curve_write_dir = None if not options.write_fig else created_dir
            _visualize_policy(methods, options.show_fig, curve_write_dir)

def _run_single_opt(methods, f_info, options):
    """Do optimization for each method."""
    # Get initial points that should be used for all methods.
    init_pts = list(uniform_draw(f_info.domain, options.init_capital))
    hp_tune_samps = options.hp_tune_samps if options.hp_tune_samps is None \
            else int(options.hp_tune_samps)
    results = {}
    for m_idx, method in enumerate(methods):
        method.set_clean()
        start_time = time.time()
        histories = method.optimize(options.max_capital, init_pts=init_pts,
                                    hp_tune_samps=hp_tune_samps)
        end_time = time.time()
        histories.time_elapsed = end_time - start_time
        results[method.get_strat_name()] = histories
    return results

def _assemble_methods(function_info, options):
    """Assemble the methods used to optimize."""
    methods = []
    if options.methods is None:
        raise ValueError('Methods must be specified.')
    else:
        function = function_info.function
        domain = function_info.domain
        ctx_dim = len(domain) - options.act_dim
        method_names = options.methods.split(',')
        if options.ctx_constraints is not None:
            num_slices = int(options.ctx_constraints)
            slices = uniform_draw(domain[:ctx_dim], num_slices)
        else:
            slices = None
        eval_set = None
        for m_idx, m_name in enumerate(method_names):
            found = False
            for strat in cstrats:
                if m_name.lower() == strat.name.lower():
                    methods.append(strat.impl(function, domain, ctx_dim,
                                              options, eval_set=eval_set,
                                              ctx_constraints=slices))
                    eval_set = methods[-1].get_eval_set()
                    found = True
                    break
            if not found:
                raise ValueError('Invalid method: %s' % m_name)
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
    with open(write_path, 'w') as f:
        pkl.dump(to_dump, f)

def _plot_results(all_results, show_fig, fig_write_path=None):
    """Plot the resulting performance."""
    methods = all_results[0].keys()
    values = len(list(all_results[0].values())[0].score_history)
    regret_curves = {}
    ts = None
    for method in methods:
        regret_curves[method] = np.zeros((len(all_results), values))
        for res_idx, res in enumerate(all_results):
            regret_curves[method][res_idx] = np.asarray([sc.regret
                for sc in res[method].score_history])
            if ts is None:
                ts = [sc.t for sc in res[method].score_history]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    cmap = _get_cmap(list(methods))
    _plot_perf(regret_curves, ax, cmap, ylab='Regret', ts=ts, log_plot=True)
    if fig_write_path is not None:
        plt.savefig(fig_write_path)
    if show_fig:
        plt.show()

def _visualize_policy(methods, show_fig=False, write_dir=None):
    if not show_fig and write_dir is None:
        return
    # Get the optimal curve.
    function = methods[0].function
    ctx_domain = methods[0].ctx_domain[0]
    act_domain = methods[0].act_domain[0]
    eval_pts = methods[0].eval_pts
    eval_grid = methods[0].eval_grid
    evals = []
    for pt in eval_pts:
        evals.append(function(pt))
    evals = np.asarray(evals).reshape(eval_grid.shape[:2])
    best_idxs = np.argmax(evals, axis=1)
    best_curve = eval_grid[np.arange(eval_grid.shape[0]), best_idxs]
    best_ctxs = [bc[0] for bc in best_curve]
    best_acts = [bc[1] for bc in best_curve]
    pairs = list(zip(best_ctxs, best_acts))
    pairs.sort()
    best_ctxs, best_acts = zip(*pairs)
    # Make the surface.
    ctx_grid = np.linspace(ctx_domain[0], ctx_domain[1], 100)
    act_grid = np.linspace(act_domain[0], act_domain[1], 100)
    surf_grid = np.ndarray((100, 100))
    for ci, ctx in enumerate(ctx_grid):
        for ai, act in enumerate(act_grid):
            surf_grid[-1 * (ai + 1), ci] = function([ctx, act])
    # For each of the methods...
    first_to_plot = True
    for method in methods:
        plt.figure(figsize=(7, 4))
        plt.imshow(surf_grid, cmap='jet', extent=(ctx_domain[0],
                                                  ctx_domain[1],
                                                  act_domain[0],
                                                  act_domain[1]))
        plt.plot(best_ctxs, best_acts, 'k--', label='Optimal Policy')
        # Find and plot the best found curve.
        est_curve = method._get_current_score(return_max_pts=True)[1]
        est_ctxs = [bc[0] for bc in est_curve]
        est_acts = [bc[1] for bc in est_curve]
        pairs = list(zip(est_ctxs, est_acts))
        pairs.sort()
        est_ctxs, est_acts = list(zip(*pairs))
        plt.plot(est_ctxs, est_acts, label='Estimated Policy')
        # Plot the queried points.
        cquery, aquery = list(zip(*method.x_data))
        plt.scatter(cquery, aquery, label='Evaluations')
        strat_name = method.get_strat_name()
        if strat_name == 'revi':
            strat_name = 'REVI'
        if strat_name == 'pts':
            strat_name = 'CMTS-PM'
        plt.title(strat_name)
        if write_dir is not None:
            write_path = os.path.join(write_dir, 'curve_%s.png' \
                    % method.get_strat_name())
            plt.savefig(write_path)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if first_to_plot:
            plt.legend(bbox_to_anchor=[1.6, 1])
            first_to_plot = False
        if show_fig:
            plt.show()
        plt.close()

if __name__ == '__main__':
    run()
