import os
import sys
import h5py
import deepdish as dd
import numpy as np

sys.path.append('/export/home/kschwarz/Documents/Masters/FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


embedding_file = 'MapNetCode/runs/mapping/embeddings.hdf5'
embeddings = dd.io.load(embedding_file)
embeddings = {int(k.split('_')[-1]): v for k, v in embeddings.items()}

fig, ax = plt.subplots(1, figsize=(10, 10))
plt.show()
for epoch, embedding in embeddings.items():
    ax.clear()
    ax.scatter(embedding[:, 0], embedding[:, 1], c='black')
    plt.pause(0.5)
