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

# fig = figure(frameon=False) # no frame


#plt.box(False)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))

fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(18, 18))
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)
ax3[0].grid(axis='y', color="0.9", linestyle='--', linewidth=1)

k = 0
for fil in sys.argv[1:]:
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

    n = fil.split('/')[-1]
    n = n.replace('_cpu_meta-loop.dat','')[2:]
    ax1[1].plot(evals, centroids, '-', linewidth=2, color=colors[k], label=n)
    ax2[0].plot(evals, mean_fits, '-', linewidth=2, color=colors[k], label=n)
    ax3[0].plot(evals, max_fits, '-', linewidth=2, color=colors[k], label=n)
    ax2[1].plot(evals, mean_succs, '-', linewidth=2, color=colors[k], label=n)
    ax3[1].plot(evals, max_succs, '-', linewidth=2, color=colors[k], label=n)
    ax2[2].plot(evals, mean_hors, '-', linewidth=2, color=colors[k], label=n)
    ax3[2].plot(evals, max_hors, '-', linewidth=2, color=colors[k], label=n)
    k += 1

img = np.asarray(Image.open('./image.png'))
ax1[0].imshow(img)
ax1[0].set_axis_off()
ax1[2].set_axis_off()
ax1[1].set_title('Coverage')
customize_axis(ax1[1])
ax2[0].set_title('Mean fitness')
customize_axis(ax2[0])
ax3[0].set_title('Max fitness')
customize_axis(ax3[0])
ax2[1].set_title('Mean success')
customize_axis(ax2[1])
ax3[1].set_title('Max success')
customize_axis(ax3[1])
ax2[2].set_title('Mean horizon')
customize_axis(ax2[2])
ax3[2].set_title('Min horizon')
customize_axis(ax3[2])

# legend = ax1.legend(loc='upper center',  # Adjust location for aesthetics (optional)
#                     bbox_to_anchor=(0.5, 1.02))  # Anchor at center of top (x=0.5, y=1.02)
# legend = ax1[0].legend(bbox_to_anchor=(0.1, 1.1, 1., .102), ncol=(1))
# legend = ax1[1].legend(bbox_to_anchor=(0.1, 0.1, 1., 1.102), ncol=(1), fontsize=20)
# frame = legend.get_frame()
# frame.set_facecolor('0.9')
# frame.set_edgecolor('1.0')

fig.tight_layout()
fig.savefig('progress.pdf')
fig.savefig('progress.png')
