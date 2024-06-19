'''Adapted from original plot/plot_progress.py to plot mean and std over multiple runs

How to run:
```python3 plot_progress.py --data_dir [dirs of data to plot] --[no-]baseline```

args:
    data-dir - source of data to be analysed
    baseline/no-baseline - whether to plot baselines or not
'''

import argparse
import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import re
import math
import gzip
import matplotlib.gridspec as gridspec
from PIL import Image

import os
from collections import defaultdict
from matplotlib import pyplot as plt

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [6, 8]
}
rcParams.update(params)

def customize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis='y', length=0)

    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(('outward', 5))
        # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)


def plot_progress(args):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(18, 20))
    ax3[0].grid(axis='y', color="0.9", linestyle='--', linewidth=1)

    k = 0
    for fold in args.data_dir:
        k+=1
        idx_file = [os.path.join(fold, name)
                    for _, _, files in os.walk(fold)
                    for name in files
                    if 'meta-loop' in name
                ]

        min_len = None
        all_evals, all_centroids, all_mean_fits, all_max_fits, all_mean_succs, all_mean_hors, all_max_succs, all_max_hors = [], [], [], [], [], [], [], []
        for fil in idx_file:
            with open(fil) as i:
                evals, centroids, mean_fits, max_fits, mean_succs, mean_hors, max_succs, max_hors = [], [], [], [], [], [], [], []
                for line in i:
                    if re.match(r"^\d+.*$",line):
                        line = line.replace('[','')
                        line = line.replace(']','')
                        vals = line.split()
                        evals.append(int(vals[0]))
                        centroids.append(int(vals[1])/1000)
                        max_fits.append(float(vals[2]))
                        max_succs.append(float(vals[3]))
                        max_hors.append(-float(vals[4]))
                        mean_fits.append(float(vals[5]))
                        mean_succs.append(float(vals[6]))
                        mean_hors.append(-float(vals[7]))
                
                min_len = len(evals) if min_len is None else min(min_len, len(evals))
                all_evals.append(evals)
                all_centroids.append(centroids)
                all_max_fits.append(max_fits)
                all_max_succs.append(max_succs)
                all_max_hors.append(max_hors)
                all_mean_fits.append(mean_fits)
                all_mean_succs.append(mean_succs)
                all_mean_hors.append(mean_hors)

        all_evals = np.array([evals[:min_len] for evals in all_evals])
        all_centroids = np.array([centroids[:min_len] for centroids in all_centroids])
        all_max_fits = np.array([fits[:min_len] for fits in all_max_fits])
        all_max_succs = np.array([succs[:min_len] for succs in all_max_succs])
        all_max_hors = np.array([hors[:min_len] for hors in all_max_hors])  
        all_mean_fits = np.array([fits[:min_len] for fits in all_mean_fits])
        all_mean_succs = np.array([succs[:min_len] for succs in all_mean_succs])
        all_mean_hors = np.array([hors[:min_len] for hors in all_mean_hors])  

        all_centroids = np.array(all_centroids)
        means = np.mean(all_centroids, axis=0)
        stds = np.std(all_centroids, axis=0)
        selector_name = fold.split('/')[-2]
        ax1[1].plot(all_evals[0], means, linestyle='-',  label=f'{selector_name} mean')
        ax1[1].fill_between(all_evals[0], means + stds, means - stds, alpha=0.2, label=f'{selector_name} std')

        axes_names = [ax2, ax3]
        for i, metric in enumerate([all_mean_fits, all_mean_succs, all_mean_hors, all_max_fits, all_max_succs, all_max_hors]):
            metric = np.array(metric)
            means = np.mean(metric, axis=0)
            stds = np.std(metric, axis=0)

            axes_names[int(i/3)][int(i%3)].plot(all_evals[0], means, linestyle='-', label=f'{selector_name} mean')
            axes_names[int(i/3)][int(i%3)].fill_between(all_evals[0], means + stds, means - stds, alpha=0.2, label=f'{selector_name} std')

        if args.baseline:
            if i%3 == 0:
                axes_names[int(i/3)][int(i%3)].axhline(y = 31.29, color = 'r', linestyle = '--', label='full baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = 75.51, color = 'b', linestyle = '--', label='better baseline')
                axes_names[int(i/3)][int(i%3)].axhline(y = 29.6, color = 'g', linestyle = '-.', label='worse baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = 25.68, color = 'b', linestyle = '-.', label='okay-worse baseline')
                if i<3: 
                    axes_names[int(i/3)][int(i%3)].set_yticks(np.sort([31.29, 75.51, 25.68, np.min(metric)]))
                else:
                    axes_names[int(i/3)][int(i%3)].set_yticks(np.sort([31.29, 75.51, 25.68, np.max(metric), np.min(metric)]))            
            
            elif i%3 == 1:
                # and i<3:
                axes_names[int(i/3)][int(i%3)].axhline(y = .64, color = 'r', linestyle = '--', label='full baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = .91, color = 'b', linestyle = '--', label='better baseline')
                axes_names[int(i/3)][int(i%3)].axhline(y = .30, color = 'g', linestyle = '-.', label='worse baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = .61, color = 'b', linestyle = '-.', label='okay-worse baseline')
                axes_names[int(i/3)][int(i%3)].set_yticks(np.sort([.64, .91, .30, .61, np.max(metric)]))
            
            else:
                axes_names[int(i/3)][int(i%3)].axhline(y = 130, color = 'r', linestyle = '--', label='full baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = 77, color = 'b', linestyle = '--', label='better baseline')
                axes_names[int(i/3)][int(i%3)].axhline(y = 149, color = 'g', linestyle = '-.', label='worse baseline') 
                axes_names[int(i/3)][int(i%3)].axhline(y = 140, color = 'b', linestyle = '-.', label='okay-worse baseline')
                axes_names[int(i/3)][int(i%3)].set_yticks(np.sort([130, 77, 149, 140,  np.min(metric)]))

    # img = np.asarray(Image.open('./image.png'))
    # ax1[0].imshow(img, interpolation='nearest', extent=[ -2, 2, -3, 3])
    ax1[0].set_axis_off()
    ax1[2].set_axis_off()
    ax1[1].set_title('Coverage', fontsize=20)
    customize_axis(ax1[1])
    ax2[0].set_title('Mean fitness', fontsize=20)
    customize_axis(ax2[0])
    ax3[0].set_title('Max fitness', fontsize=20)
    customize_axis(ax3[0])
    ax2[1].set_title('Mean success', fontsize=20)
    customize_axis(ax2[1])
    ax3[1].set_title('Max success', fontsize=20)
    customize_axis(ax3[1])
    ax2[2].set_title('Mean horizon', fontsize=20)
    customize_axis(ax2[2])
    ax3[2].set_title('Min horizon', fontsize=20)
    customize_axis(ax3[2])

    # legend = ax1[1].legend(bbox_to_anchor=(0.1, 0.1, 1., 1.102), ncol=(1), fontsize=20)
    # frame = legend.get_frame()
    # frame.set_facecolor('1.0')
    # frame.set_edgecolor('1.0')

    fig.tight_layout()
    fig.subplots_adjust(wspace=.35, hspace=.45)
    fig.savefig('progress.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', nargs='+')
    parser.add_argument('--baseline', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    
    plot_progress(args)
