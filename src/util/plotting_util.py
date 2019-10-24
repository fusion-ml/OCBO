"""
Functions for plotting various quantities.
"""

from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import sem

from gp.gp_util import gp_regression
from util.post_util import get_total_best_mats, best_vals_to_regrets,\
        get_query_proportions, get_best_for_function, get_choice_timelines

def plot_summary(f_infos, results, show_figure=False, write_path=None):
    """Plot a bunch of information all at once."""
    cmap = _get_cmap(list(results[0].keys()))
    cant_regret = np.any([f_info.max_val is None for f_info in f_infos])
    if len(f_infos) > 7 or len(results[0]) > 7:
        # If there are too many function then just plot total regret.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 12))
        best_mats = get_total_best_mats(results)
        if cant_regret:
            _plot_perf(best_mats, ax, cmap, ylab='Performance')
            ax.set_title('Total Performance over %d Functions' % len(f_infos))
        else:
            regrets = best_vals_to_regrets(f_infos, best_mats)
            _plot_perf(regrets, ax, cmap)
            ax.set_title('Total Simple Regret over %d Functions' % len(f_infos))
        _write_or_show(show_figure, write_path)
        return
    additional_plots = 3 if len(f_infos) == 2 else 2
    num_rows = int(np.ceil((len(f_infos) + additional_plots) / 3))
    num_cols = 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12))
    axs = axs.ravel()
    # Plot total regret.
    best_mats = get_total_best_mats(results)
    if cant_regret:
        _plot_perf(best_mats, axs[0], cmap, ylab='Performance')
        axs[0].set_title('Total Performance over %d Functions' % len(f_infos))
    else:
        regrets = best_vals_to_regrets(f_infos, best_mats)
        _plot_perf(regrets, axs[0], cmap)
        axs[0].set_title('Total Simple Regret over %d Functions' % len(f_infos))
    # Plot proportions.
    f_names = [fi.name for fi in f_infos]
    proportions = get_query_proportions(results, f_names)
    _plot_proportions(f_names, proportions, axs[1], cmap)
    axs[1].set_title('Query Proportions')
    # Plot proportions trend if only two functions.
    if len(f_names) == 2:
        choice_timelines = get_choice_timelines(results)
        _plot_choice_trend(choice_timelines, axs[2])
        axs[2].set_title('Percent that picked %s' % f_names[1])
    # Plot regrets for each function.
    plot_idx = additional_plots
    for f_info in f_infos:
        f_best = get_best_for_function(f_info.name, results)
        if cant_regret:
            _plot_perf(f_best, axs[plot_idx], cmap, ylab='Performance')
        else:
            f_regrets = best_vals_to_regrets([f_info], f_best)
            _plot_perf(f_regrets, axs[plot_idx], cmap)
        axs[plot_idx].set_title(f_info.name)
        plot_idx += 1
    _write_or_show(show_figure, write_path)

def plot_total(f_infos, results, show_figure=False, write_path=None, ax=None,
               add_legend=True, cmap=None, use_tex=False, ordering=None,
               risk_neutral=False, log_plot=False, psuedo_regret=False):
    """Plot total regret.
    Args:
        f_infos: List of Namespace for function info used.
        results: Dict method -> List of results.
        show_figure: Whether to show the figure.
        write_path: Path to write the figure at.
        ax: pyplot ax object to use for plotting.
        add_legend: Whether to add a legend to the plot.
        cmap: Predefined color map to use.
        psuedo_regret: Plot regret instead by making maximum a bit larger
            than the max ever found.
    """
    if ordering is None:
        ordering = results[0].keys()
    if cmap is None:
        cmap = _get_cmap(ordering[::-1])
    best_mats = get_total_best_mats(results, use_risk_neutral=risk_neutral)
    cant_regret = np.any([f_info.max_val is None for f_info in f_infos])
    if cant_regret and psuedo_regret:
        global_best = max([np.mean(bm[:, -1]) for bm in best_mats.values()])
        global_best += 0.001
        for fi in f_infos:
            fi.max_val = global_best / len(f_infos)
        cant_regret = False
    to_plot = best_mats
    if use_tex:
        ylab = r'$\log \sum_{x \in X} f(x, a^*_t(x))$'
    else:
        ylab = 'Total Reward'
    if not cant_regret:
        to_plot = best_vals_to_regrets(f_infos, best_mats)
        if use_tex:
            ylab = r'$\log \sum_{x \in X} \max_{a \in A} f(x, a) - f(x, a^*_t(x))$'
        else:
            ylab = 'Total Simple Regret'
    if ax is None:
        plt.figure(figsize=(18,12))
        ax = plt.subplot(111)
    _plot_perf(to_plot, ax, cmap, ylab=ylab, add_legend=add_legend,
               ordering=ordering, log_plot=log_plot)
    _write_or_show(show_figure, write_path)

def plot_props(f_infos, results, show_figure=False, write_path=None, ax=None,
               add_legend=True, ordering=None, cmap=None):
    """Given results, plot proportions of how often each function as called.
    Args:
        f_infos: List of Namespace for function info used.
        results: Dict method -> List of results.
        show_figure: Whether to show the figure.
        write_path: Path to write the figure at.
        ax: pyplot ax object to use for plotting.
        add_legend: Whether to add a legend.
    """
    if ordering is None:
        ordering = results[0].keys()
    if cmap is None:
        cmap = _get_cmap(ordering[::-1])
    f_names = [fi.name for fi in f_infos]
    proportions = get_query_proportions(results, f_names)
    if ax is None:
        plt.figure(figsize=(18, 12))
        ax = plt.subplot(111)
    _plot_proportions(f_names, proportions, ax, cmap, add_legend=add_legend,
                      ordering=ordering)
    _write_or_show(show_figure, write_path)

def visualize_mean_surface(domain, x_data, y_data, ax=None,
                           gp_engine='dragonfly', gp_options=None):
    """Given two-D data, fit a GP to it and make a 3D plot.
    Args:
        domain: List of lists representing x domain.
        x_data: The x data as an ndarray.
        y_data: The y data as an ndarray.
        ax: The axis object, makes a new one if None.
    """
    # Get the surface points for the plot.
    dim1_pts = np.linspace(domain[0][0], domain[0][1], 50)
    dim2_pts = np.linspace(domain[1][0], domain[1][1], 50)
    dim1_pts, dim2_pts = np.meshgrid(dim1_pts, dim2_pts)
    surf_pts = np.hstack([dim1_pts.reshape(-1, 1), dim2_pts.reshape(-1, 1)])
    # Do the gp regression.
    gp = gp_regression(gp_engine, x_data, y_data, options=gp_options)
    surf_vals = gp.eval(surf_pts)[0].reshape(dim1_pts.shape)
    # Plot the surface and the data points.
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    ax.plot_surface(dim1_pts, dim2_pts, surf_vals, cmap=cm.coolwarm)
    ax.scatter(x_data[:,0], x_data[:,1], y_data, s=50)

def _plot_perf(regrets, ax, cmap, ylab='Simple Regret',
               add_legend=True, ordering=None, ts=None, log_plot=False):
    """Plot the total simple regrets.
    Args:
        regrets: dict method -> ndarray, each row is a run's regret.
        ax: pyplot ax object.
        cmap: dict function name -> color.
        add_legend: Whether to add a legend.
    """
    if ordering is None:
        ordering = regrets.keys()
    if log_plot:
        if not (np.asarray([(reg > 0).all() for reg in regrets.values()]).all()):
            global_min = min([np.min(reg) for reg in regrets.values()])
            for reg in regrets.values():
                reg += (0.01 - global_min)
        ax.set_yscale('log', nonposy='clip')
    max_val, min_val = float('-inf'), float('inf')
    for method in ordering:
        if method not in regrets:
            continue
        reg = regrets[method]
        if ts is None:
            plot_ts = range(reg.shape[1])
        else:
            plot_ts = ts
        avg = np.mean(reg, axis=0)
        max_avg, min_avg = max(avg), min(avg)
        max_val = max_val if max_val > max_avg else max_avg
        min_val = min_val if min_val < min_avg else min_avg
        err = sem(reg, axis=0)
        ax.plot(plot_ts, avg, c=cmap[method], label=method.upper(),
                linewidth=2)
        ax.fill_between(plot_ts, avg - err, avg + err, color=cmap[method],
                        alpha=0.4)
    ax.set_xlabel('t')
    ax.set_ylabel(ylab)
    if add_legend:
        leg = ax.legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

def _plot_proportions(f_names, proportions, ax, cmap, add_legend=True,
                      ordering=None):
    """Plot histogram of proportions for functions.
    Args:
        f_names: Name of the functions.
        proportions: dict method -> ndarray, each row is a run's proportion.
        ax: pyplot ax object.
        cmap: dict function name -> color.
        add_legend: Whether to add a legend.
    """
    if ordering is None:
        ordering = proportions.keys()
    idx = 0
    num_avgs = len(f_names)
    width = 1 / (num_avgs + 1)
    for method in ordering:
        if method not in proportions:
            continue
        props = proportions[method]
        avgs = np.mean(props, axis=0)
        errs = sem(props, axis=0)
        xs = 2 * np.arange(num_avgs) + width * idx
        ax.bar(xs, avgs, width, yerr=errs, label=method.upper(), align='edge',
               color=cmap[method])
        idx += 1
    ax.set_xticks(2 * np.arange(num_avgs) + width * num_avgs / 2)
    tick_names = []
    for fn in f_names:
        if 'constant' in fn.lower():
            tick_names.append('Constant')
        else:
            to_add = fn.replace('_', '')
            to_add = to_add.replace('0', '')
            to_add = to_add.capitalize()
            tick_names.append(to_add)
    ax.set_xticklabels(tick_names)
    ax.set_ylabel('Proportion')
    if add_legend:
        ax.legend()

def _plot_choice_trend(choice_timelines, ax):
    """Plot the percent of runs that picked context over time.
    (choice must be binary)
    Args:
        choice_timeline: dict method -> ndarray of choices.
        ax: pyplot ax object.
    """
    for method, choices in (choice_timelines.items()):
        percent = np.mean(choices, axis=0)
        ax.plot(range(len(percent)), percent, label=method)
    ax.set_xlabel('t')
    ax.set_ylabel('Percent')
    ax.legend()

def _get_cmap(m_names):
    """Get a color map mapping a method name to color."""
    cmap = {}
    cm_type = cm.get_cmap('Accent')
    for idx, colr in enumerate(cm_type(np.linspace(0, 1, len(m_names)))):
        cmap[m_names[idx]] = colr
    return cmap

def _write_or_show(show_figure, write_path):
    """Write figure or show figure or both."""
    if write_path is not None:
        plt.savefig(write_path, dpi=100)
    if show_figure:
        plt.show()
