import json
import os
from argparse import ArgumentParser
from hashlib import md5

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb as rgb

parser = ArgumentParser(description="Plots lablel distributions over some subsets.")
parser.add_argument('--input', type=str, nargs='+')
args = parser.parse_args()

for inp in args.input:

    d = json.load(open(inp, 'r'))['nodes']
    fname = os.path.basename(inp)
    print "Processing {}...".format(fname)
    col = (int(md5(inp).hexdigest(), 16) % 100) / 100.0

    labels = map(lambda x:x['label'], sorted(d.values(), key=lambda x: x['idx']))
    labels = np.array(labels)
    lut, labels = np.unique(labels, return_inverse=True)

    bins = np.bincount(labels).astype(float)
    bins = bins / bins.sum()
    plt.clf()
    plt.plot(bins, marker='o', label='all', c=rgb((col, 0, 0)))
    num_bins = len(bins)

    step = (len(labels)/800 + 1) * 100

    parts = range(1000, len(labels), step)
    for i, mx in enumerate(parts):
        bins = np.bincount(labels[:mx]).astype(float)
        bins = bins / bins.sum()
        bins.resize(num_bins)
        sat = 1 - float(i)/(len(parts))
        plt.plot(bins, marker='o', label=str(mx), c=rgb((col, sat, sat)))

    plt.plot([1.0/num_bins]*num_bins, c=rgb((col + 0.5 if col < 0.5 else col - 0.5, 0.5, 0.5)), zorder=-1)
    plt.title(fname)
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.savefig(os.path.join("./distr_imgs/", fname + ".png"))