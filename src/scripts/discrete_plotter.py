"""
Main file for making a summary plot of run that has already been made.
"""
import os
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from dragonfly.utils.option_handler import load_options, get_option_specs
from util.misc_util import assemble_functions
from util.plotting_util import plot_total, plot_props, _get_cmap
from util.post_util import load_results

plot_args = [\
        get_option_specs('run_ids', False, None,
            'List of unique run_ids as comma seperated values.'),
        get_option_specs('write_dir', False, None,
            ('Write dir to look for data in.')),
        get_option_specs('show_fig', False, False, 'Whether to show the fig'),
        get_option_specs('write_path', False, None, 'Path to write fig.'),
        get_option_specs('plot_props', False, None,
            ('Which run_ids to plot proportions for as well. Write as comma '
             'separated list e.g. if run_ids=id1,id2,id3 doing 1,3 will do '
             'plots for the first and third.')),
        get_option_specs('only_props', False, None,
            ('Which run_ids to plot proportions for as well. Write as comma '
             'separated list e.g. if run_ids=id1,id2,id3 doing 1,3 will do '
             'plots for the first and third.')),
        get_option_specs('legend_on_first', False, True,
            'Whether to only add legend on the first plot.'),
        get_option_specs('use_legend', False, True,
            'Whether to add legends to any plots.'),
        get_option_specs('titles', False, None,
            'Titles as X_X,X,X_X_X where underscores replaced with spaces'),
        get_option_specs('risk_neutral', False, False,
            'Plot as risk neutral strat instead.'),
        get_option_specs('log_plot', False, False,
            'Whether to force log scale on y axis.'),
        get_option_specs('force_reward', False, False,
            'Whether to force plotting reward instead of regret.'),
        get_option_specs('force_regret', False, True,
            'Whether to force converting to regret instead of reward.'),
        get_option_specs('exclude_rand', False, False,
            'Whether to exclude random from the plots.'),
        get_option_specs('xlim', False, None,
            'X limit on the plot.'),
]

ORDERING = None
ORDERING = ['mts', 'mei', 'ei', 'ts', 'rand', 'revi']

def run():
    plot_opts = load_options(plot_args, cmd_line=True)
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 19}
    matplotlib.rc('font', **font)
    if plot_opts.run_ids is None or plot_opts.write_dir is None:
        raise ValueError('Need to specify run_id and write_dir to read from.')
    ids = plot_opts.run_ids.split(',')
    if plot_opts.plot_props is None:
        should_prop = [False for _ in range(len(ids))]
    else:
        should_prop = [False for _ in range(len(ids))]
        for idx in plot_opts.plot_props.split(','):
            should_prop[int(idx) - 1] = True
    if len(ids) > 1 or sum(should_prop) > 0:
        num_to_plot = len(ids) + sum(should_prop)
        fig, axs = plt.subplots(nrows=1, ncols=num_to_plot, figsize=(34, 8))
        plt.subplots_adjust(wspace=0.3)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 10))
        axs = [ax]
    if plot_opts.titles is not None:
        titles = plot_opts.titles.split(',')
        if len(titles) != len(axs):
            raise ValueError('Number of titles must match number of plots')
    else:
        titles = None
    plot_idx = 0
    cmap = None
    cmap_labs = ['ei', 'ts', 'revi', 'mei', 'mts', 'rand']
    cmap = _get_cmap(cmap_labs)
    cmap[cmap_labs[2]] = [0.8, 0.65, 0.45, 1]
    for run_idx, run_id in enumerate(ids):
        read_dir = os.path.join(plot_opts.write_dir, run_id)
        if not os.path.isdir(read_dir):
            raise ValueError('No directory %s found.' % read_dir)
        options, loaded_results, f_infos = load_results(read_dir)
        results = []
        for res in loaded_results:
            new_res = {}
            for k, v in res.iteritems():
                if k == 'agn-ei' or k == 'ja-ei':
                    new_res['ei'] = v
                elif k == 'mts' or k == 'joint-mts':
                    new_res['mts'] = v
                elif k == 'mei' or k == 'joint-mei':
                    new_res['mei'] = v
                elif k == 'Random':
                    new_res['rand'] = v
                elif k == 'agn-thomp' or k == 'ja-thomp':
                    new_res['ts'] = v
                elif (k == 'rand' or k == 'joint-rand') \
                        and not plot_opts.exclude_rand:
                    new_res['rand'] = v
                else:
                    new_res[k] = v
            results.append(new_res)
        add_legend = plot_opts.use_legend
        # add_legend = (not plot_opts.legend_on_first or plot_idx == 0) and plot_opts.use_legend
        if plot_opts.force_reward:
            for fi in f_infos:
                fi.max_val = None
        if plot_opts.xlim is not None:
            axs[plot_idx].set_xlim(0, int(plot_opts.xlim))
        if plot_opts.only_props:
            plot_props(f_infos, results, ax=axs[plot_idx],
                       add_legend=add_legend, ordering=ORDERING, cmap=cmap)
        else:
            plot_total(f_infos, results, ax=axs[plot_idx], add_legend=add_legend,
                       ordering=ORDERING, cmap=cmap, log_plot=plot_opts.log_plot,
                       risk_neutral=plot_opts.risk_neutral,
                       psuedo_regret=plot_opts.force_regret)
        if titles is not None:
            axs[plot_idx].set_title(titles[plot_idx].replace('_', ' '))
        plot_idx += 1
        add_legend = (not plot_opts.legend_on_first or plot_idx == 0) and plot_opts.use_legend
        if should_prop[run_idx]:
            plot_props(f_infos, results, ax=axs[plot_idx],
                       add_legend=add_legend, ordering=ORDERING, cmap=cmap)
            if titles is not None:
                axs[plot_idx].set_title(titles[plot_idx].replace('_', ' '))
            plot_idx += 1
    if plot_opts.write_path is not None:
        plt.savefig(plot_opts.write_path)
    plt.show()

if __name__ == '__main__':
    # import pudb; pudb.set_trace()
    run()
