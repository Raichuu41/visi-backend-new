import os
import sys
import numpy as np
import deepdish as dd
import torch

from model import MapNet
from initialization import initialize_embedder, compute_embedding

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


feature_file = 'features/ShapeDataset_NarrowNet128_MobileNetV2_test.hdf5'  # ''features/VGG_info_artist_49_multilabel_val.hdf5'#'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
weight_file = 'runs/embedder/models/ShapeDataset_MapNet_embedder_09-12-13-07.pth.tar'  # 'runs/embedder/MapNet_embedder_09-11-15-45.pth.tar'#08-03-12-48_TSNENet_large_model_model_best.pth.tar'
info_file = '/export/home/kschwarz/Documents/Data/Geometric_Shapes/labels.hdf5'


def evaluate_embedder(feature_file, info_file, weight_file=None, experiment_id=None):
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
    feature = dd.io.load(feature_file)['features']
    net = MapNet(feature_dim=feature.shape[1], output_dim=2)
    initialize_embedder(net.embedder, weight_file=weight_file, feature=feature, experiment_id=experiment_id)
    embedding = compute_embedding(net.embedder, feature)

    shape_dataset = feature_file.split('/')[-1].startswith('ShapeDataset')

    if shape_dataset:
        outdir = 'ShapeDataset'

        df = dd.io.load(info_file)['df']
        split = feature_file.split('_')[-1].replace('.hdf5', '')
        if split == 'train':
            split = 0
        elif split == 'val':
            split = 1
        else:       # test
            split = 2
        df = df[df['split'] == split]
        df.index = range(len(df))
        classes = ['shape', 'n_shapes', 'color_shape', 'color_background']
    else:
        outdir = 'Wikiart'
        raise NotImplementedError('evaluate embedder only implemented for shape dataset')

    for c in classes:
        label = df[c]
        plt.figure()
        for color_idx, l in enumerate(np.sort(np.unique(label))):
            plt.scatter(embedding[label==l, 0], embedding[label==l, 1], c=plt.cm.tab20(color_idx), label=l)
        lgd = plt.legend(bbox_to_anchor=(1.3, 1))
        plt.title(c)
        plt.savefig('evaluation/' + outdir + '/initial_embedding_{}.jpg'.format(c),
                    bbox_extra_artists=(lgd,), bbox_inches='tight')