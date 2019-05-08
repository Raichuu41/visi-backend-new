import os
import pandas as pd
import numpy as np
import deepdish as dd
from collections import Counter

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from export_plots import init_plot

info_name = '../pretraining/wikiart_datasets/info_elgammal_subset_train.hdf5'


def abbreviate(string, N=1):
    strings = string.split(' ')
    strings = [s.split('-') for s in strings]
    strings = np.concatenate(strings)
    shorts = []
    for s in strings[:-1]:
        if len(s) > N:
            s = s[:N] + '.'
        shorts.append(s)
    s = strings[-1]
    shorts.append(s)
    return ' '.join(shorts)


merge_df = None
for split in ['_train', '_val', '_test']:
    df = dd.io.load(info_name.replace('_train', split))['df']
    if merge_df is None:
        merge_df = df
    else:
        merge_df = merge_df.append(df)

N_abbreviate = 1
for task in ['artist_name', 'genre', 'style']:
    count = Counter(merge_df[task].dropna().values)
    count = sorted(count.items(), key=lambda x: x[0])
    fig, ax = init_plot()
    w,h = fig.get_size_inches()
    fig.set_size_inches((2*w, h*0.5))
    ax.bar(x=range(len(count)), height=map(lambda x: x[1], count), width=0.9,
           tick_label=map(lambda x: abbreviate(x[0], N_abbreviate), count))
    ax.tick_params(axis='x', rotation=90, labelsize='xx-small')
    plt.savefig('/export/home/kschwarz/Documents/Masters/Thesis/Plots/{}.png'.format('Dataset_histogram_{}'.
                                                                                     format(task.split('_')[0])),
                bbox_inches='tight', dpi=1000)