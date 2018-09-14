import deepdish as dd
import numpy as np
import pandas as pd
import sys
import time
import os
import torch
from sklearn.cluster import k_means
import sklearn.metrics as metrics
from faiss_master import faiss
from collections import Counter
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


feature_file = 'features/ShapeDataset_NarrowNet128_MobileNetV2_test.hdf5'  # ''features/VGG_info_artist_49_multilabel_val.hdf5'#'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
info_file = '/export/home/kschwarz/Documents/Data/Geometric_Shapes/labels.hdf5'


def load_data(feature_file, info_file):
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')
    if not os.path.isfile(info_file):
        raise RuntimeError('Info file not found.')

    df = dd.io.load(info_file)['df']

    data = dd.io.load(feature_file)
    names, features = data['image_names'], data['features']

    is_shape_dataset = feature_file.split('/')[-1].startswith('ShapeDataset')
    if is_shape_dataset:
        outdir = 'ShapeDataset'
        category = ['shape', 'n_shapes', 'color_shape', 'color_background']

        split = feature_file.split('_')[-1].replace('.hdf5', '')
        if split == 'train':
            split = 0
        elif split == 'val':
            split = 1
        else:       # test
            split = 2
        df = df[df['split'] == split]
        df.index = range(len(df))

    else:
        outdir = 'Wikiart'
        category = ['artist_name', 'genre', 'style', 'technique', 'century']

    if not (names == df['image_id']).all():
        raise RuntimeError('Image names in info file and feature file do not match.')

    outdict = {'image_names': names,
               'features': features,
               'labels': {c: df[c] for c in category}}

    return outdict


def predict_cluster(data_dict, category, cluster_per_label=1, random_state=123):
    """Assign the main label of each cluster as predicted label."""
    gt = data_dict['labels'][category]
    n_cluster = len(np.unique(gt)) * cluster_per_label

    _, cluster_labels, _ = k_means(data_dict['features'], n_clusters=n_cluster, random_state=random_state)
    prediction = -1 * np.ones(gt.shape, dtype=gt.dtype)
    for cl in range(n_cluster):
        mask = cluster_labels == cl
        counter = Counter(gt[mask])
        main_label = counter.keys()[np.argmax(counter.values())]
        prediction[mask] = main_label

    assert np.all(prediction != -1), 'Error in clustering, not all samples received prediction.'

    return prediction, gt.values


def k_nn_accuracy(feature, gt_label, k=(1,2,4,8), average=None):
    if average not in [None, 'micro', 'macro']:
        raise NotImplementedError('average has to be None, "micro" or "macro".')

    gt_label = np.array(gt_label)

    N, d = feature.shape
    # search neighbors for each sample
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(feature.astype(np.float32))  # add vectors to the index
    _, neighbors = index.search(feature.astype(np.float32), max(k)+1)

    # predicted labels are labels of first entry (sample itself)
    prediction = gt_label[neighbors[:, 0]].reshape(-1, 1).repeat(max(k), axis=1)
    gt = np.stack([gt_label[neighbors[:, i]] for i in np.arange(1, max(k) + 1)], axis=1)

    is_correct = prediction == gt

    eval_dict = dict.fromkeys(k)
    for _k in k:
        if average == 'micro':
            acc = is_correct[:, :_k].sum() * 1.0 / is_correct[:, :_k].size
        else:
            labelset = np.unique(gt_label)
            acc = dict.fromkeys(labelset)
            for l in labelset:
                mask = gt_label == l
                acc[l] = is_correct[mask, :_k].sum() * 1.0 / is_correct[mask, :_k].size
            if average == 'macro':
                acc = np.mean(acc.values())
        eval_dict[_k] = acc
    return eval_dict


def evaluate(feature_file, info_file, data_dict, category=None, experiment_id=None):
    data_dict = load_data(feature_file, info_file)
    if category is None:        # evaluate for all categories
        category = data_dict['labels'].keys()
    for c in category:
        y_pred, y_true = predict_cluster(data_dict, category=c)
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        neighbor_acc = k_nn_accuracy(data_dict['features'], data_dict['labels'][c], average=None)
