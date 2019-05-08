import deepdish as dd
import numpy as np
import pandas as pd
import sys
import time
import os
import torch
from sklearn.cluster import k_means

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


if not os.getcwd().endswith('/MapNetCode'):
    os.chdir('/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode')


feature_file_baseline = 'features/MobileNetV2_info_artist_49_multilabel_test.hdf5'
feature_file_test = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/features/cycle_001_feature.hdf5'
feature_file_val = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_val.hdf5'
info_file_test = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
info_file_val = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_val.hdf5'
weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/models/cycle_001_models.pth.tar'


def load_data(feature_file, info_file, split=0):
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')
    if not os.path.isfile(info_file):
        raise RuntimeError('Info file not found.')

    df = dd.io.load(info_file)['df']

    data = dd.io.load(feature_file)
    try:
        names, features = data['image_id'], data['feature']
    except KeyError:
        try:
            names, features = data['image_names'], data['features']
        except KeyError:
            names, features = data['image_name'], data['features']

    is_shape_dataset = 'ShapeDataset' in feature_file
    is_office_dataset = 'OfficeDataset' in feature_file
    is_bam_dataset = 'BAMDataset' in feature_file
    if is_shape_dataset:
        outdir = 'ShapeDataset'
        category = ['shape', 'n_shapes', 'color_shape', 'color_background']

        df = df[df['split'] == split]
        df.index = range(len(df))
    elif is_office_dataset:
        outdir = 'OfficeDataset'
        category = ['genre', 'style']
    elif is_bam_dataset:
        outdir = 'BAMDataset'
        category = ['content', 'emotion', 'media']
    else:
        outdir = 'Wikiart'
        category = ['artist_name', 'genre', 'style', 'technique', 'century']

    if not (names == df['image_id']).all():
        raise RuntimeError('Image names in info file and feature file do not match.')

    outdict = {'image_names': names,
               'features': features,
               'labels': {c: df[c] for c in category}}

    return outdict, outdir


def get_pure_clusters(vectors, label, n_min=1, random_state=123):
    pure_clusters = {}
    print('Start recursive clustering...')
    start = time.time()

    def recursive_clustering(vectors, label, n_min, indices):
        _, cluster_labels, _ = k_means(vectors, n_clusters=2, random_state=random_state)
        is_pure = [len(np.unique(label[cluster_labels == c])) == 1 for c in np.unique(cluster_labels)]
        for c in [0, 1]:
            mask = cluster_labels == c
            if is_pure[c] and mask.sum() >= n_min:
                if len(pure_clusters) == 0:
                    key = 0
                else:
                    key = np.max(pure_clusters.keys()) + 1

                pure_clusters[key] = indices[np.where(mask)[0]]
            elif mask.sum() <= n_min:
                continue
            else:
                recursive_clustering(vectors[mask], label[mask], n_min, indices[mask])

    recursive_clustering(vectors, label, indices=np.arange(0, len(label)), n_min=n_min)
    stop = time.time()
    print('Finished recursive clustering after {:.0f}min {:.0f}s.'.format((stop-start) / 60, (stop-start) % 60))
    cluster = pure_clusters.values()
    label = [label[idcs[0]] for idcs in cluster]

    return cluster, label


def get_exemplars(vectors, label, n_min):
    N = len(label)
    pure_cluster, exemplar_label = get_pure_clusters(vectors, label, n_min=n_min)
    exemplars = [np.mean(vectors[cluster_idcs], axis=0) for cluster_idcs in pure_cluster]

    # samples for which no exemplars have been found
    idcs_all = np.arange(0, N)
    have_exemplar = np.isin(idcs_all, np.concatenate(pure_cluster))
    rest_idcs = idcs_all[have_exemplar.__invert__()]
    rest_label = label[have_exemplar.__invert__()]

    print('Replace {:.1f}% of the points with their exemplars.'.format(have_exemplar.sum() * 100. / N))

    return exemplars, exemplar_label, rest_idcs, rest_label


def get_tensor_exemplars(vectors, n_cluster, random_state=123):
    print('Cluster feature space...')
    start = time.time()
    cluster_center, cluster_labels, _ = k_means(vectors, n_clusters=n_cluster, random_state=random_state)
    stop = time.time()
    print('Finished clustering after {:.0f}min {:.0f}s.'.format((stop - start) / 60, (stop - start) % 60))

    clusters = [np.where(cluster_labels == c)[0] for c in np.unique(cluster_labels)]
    radii = [np.linalg.norm(vectors[samples] - center, axis=1).max() for samples, center in zip(clusters, cluster_center)]
    feature_norm = np.linalg.norm(vectors, axis=1).mean()
    exemplars = [np.mean(vectors[cluster_idcs], axis=0) for cluster_idcs in clusters]

    return exemplars


if __name__ == '__main__':
    data_dict, outdir = load_data(feature_file_baseline, info_file_test, split=0)
    vectors = data_dict['features']
    label = (data_dict['labels']['genre'] == 'genre ').values.astype(int)
    n_min = 10
    exemplars, exemplar_labels, rest_idcs, rest_label = get_exemplars(vectors, label, n_min)







