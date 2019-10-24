"""
Main file for making a summary plot of run that has already been made.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

from dragonfly.utils.option_handler import load_options, get_option_specs
from util.misc_util import assemble_functions
from util.plotting_util import plot_total, plot_props, _get_cmap, _plot_perf
from util.post_util import load_results

plot_args = [\
        get_option_specs('run_ids', False, None,
            'List of unique run_ids as comma seperated values.'),
        get_option_specs('write_dir', False, 'data',
            ('Write dir to look for data in.')),
        get_option_specs('method_prepend', False, False,
            'Whether to prepend in front of method name to split up runs.'),
        get_option_specs('show_fig', False, True, 'Whether to show the fig'),
        get_option_specs('write_path', False, None, 'Path to write fig.'),
        get_option_specs('use_legend', False, True,
            'Whether to show the legend.'),
]

ORDERING = ['revi', 'cmts-pm', 'pei', 'cmts', 'rand']

def run():
    plot_opts = load_options(plot_args, cmd_line=True)
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 19}
    matplotlib.rc('font', **font)
    if plot_opts.run_ids is None or plot_opts.write_dir is None:
        raise ValueError('Need to specify run_id and write_dir to read from.')
    ids = plot_opts.run_ids.split(',')
    cmap = None
    all_results = []
    for run_idx, run_id in enumerate(ids):
        read_dir = os.path.join(plot_opts.write_dir, run_id)
        if not os.path.isdir(read_dir):
            raise ValueError('No directory %s found.' % read_dir)
        options, results, f_infos = load_results(read_dir)
        if plot_opts.method_prepend:
            results = _prepend_to_method_name(results, run_idx)
        all_results += results
    _plot_results(all_results, plot_opts.show_fig, plot_opts)

def _prepend_to_method_name(results, to_append):
    modded = []
    to_append = str(to_append)
    for res in results:
        new_dict = {}
        for k, v in res.iteritems():
            new_dict['%s-%s' % (to_append, k)] = v
        modded.append(new_dict)
    return modded

def _plot_results(all_results, show_fig, plot_opts, fig_write_path=None):
    """Plot the resulting performance."""
    values = len(list(all_results[0].values())[0].score_history)
    regret_curves = {}
    ts = None
    for method in ORDERING:
        result_idxs = []
        for ar_idx, ar in enumerate(all_results):
            if method in ar:
                result_idxs.append(ar_idx)
        regret_curves[method] = np.zeros((len(result_idxs), values))
        for reg_idx, res_idx in enumerate(result_idxs):
            res = all_results[res_idx]
            regret_curves[method][reg_idx] = np.asarray([sc.regret
                for sc in res[method].score_history])
            if ts is None:
                ts = [sc.t for sc in res[method].score_history]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    cmap = _get_cmap(ORDERING)
    _plot_perf(regret_curves, ax, cmap, ylab='Regret', ts=ts, log_plot=True,
               add_legend=plot_opts.use_legend)
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
    for method in methods:
        plt.imshow(surf_grid, cmap='jet', extent=(ctx_domain[0], ctx_domain[1],
                                                  act_domain[0], act_domain[1]))
        plt.plot(best_ctxs, best_acts, 'k--')
        # Find and plot the best found curve.
        est_curve = method._get_current_score(return_max_pts=True)[1]
        est_ctxs = [bc[0] for bc in est_curve]
        est_acts = [bc[1] for bc in est_curve]
        pairs = list(zip(est_ctxs, est_acts))
        pairs.sort()
        est_ctxs, est_acts = list(zip(*pairs))
        plt.plot(est_ctxs, est_acts)
        # Plot the queried points.
        cquery, aquery = list(zip(*method.x_data))
        plt.scatter(cquery, aquery)
        plt.title(method.get_strat_name())
        if write_dir is not None:
            write_path = os.path.join(write_dir, 'curve_%s.png' \
                    % method.get_strat_name())
            plt.savefig(write_path)
        if show_fig:
            plt.show()
        plt.close()

if __name__ == '__main__':
    run()

