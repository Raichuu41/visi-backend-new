import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


standard_colors = plt.cm.tab10.colors


def loopindex(index, maxval):
    return index if index < maxval else index % maxval


def scatter(points, label, ax=None, legend=False, colors=None, **plotkwargs):
    if ax is None:
        _, ax = plt.subplots(1)

    labelset = np.unique(label)
    if colors is None:
        global standard_colors
        colors = standard_colors

    for i, lbl in enumerate(labelset):
        mask = label == lbl
        ax.scatter(points[mask, 0], points[mask, 1], c=colors[loopindex(i, len(colors))], label=lbl, **plotkwargs)

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 0.97))

    return ax


def save_fig(figure, outpath):
    legends = [ax.legend_ for ax in figure.axes if ax.legend_ is not None]
    plt.savefig(outpath, bbox_extra_artists=legends, bbox_inches='tight')
