import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import re
import math
import gzip
import matplotlib.gridspec as gridspec

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
    #ax.get_yaxis().tick_left()

    # offset the spines
    for spine in ax.spines.values():
     spine.set_position(('outward', 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)

fig = figure(frameon=False) # no frame


#plt.box(False)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))


ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax3.grid(axis='y', color="0.9", linestyle='--', linewidth=1)

k = 0
for fil in sys.argv[1:]:
    with open(fil) as i:
        evals, centroids, mean_fits, max_fits = [], [], [], []
        for line in i:
            if re.match(r"^\d+.*$",line):
                line = line.replace('[','')
                line = line.replace(']','')
                vals = line.split()
                evals.append(int(vals[0]))
                centroids.append(int(vals[1])/1000)
                max_fits.append(float(vals[2]))
                mean_fits.append(float(vals[5]))

    n = fil.split('/')[-1]
    n = n.replace('_cpu_meta-loop.dat','')[2:]
    ax1.plot(evals, centroids, '-', linewidth=2, color=colors[k], label=n)
    ax2.plot(evals, mean_fits, '-', linewidth=2, color=colors[k], label=n)
    ax3.plot(evals, max_fits, '-', linewidth=2, color=colors[k], label=n)
    k += 1

ax1.set_title('Coverage')
customize_axis(ax1)
ax2.set_title('Mean fitness')
customize_axis(ax2)
ax3.set_title('Max fitness')
customize_axis(ax3)

legend = ax1.legend(loc=4)#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('1.0')

fig.tight_layout()
fig.savefig('progress.pdf')
fig.savefig('progress.png')
