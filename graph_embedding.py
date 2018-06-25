# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append('/export/home/kschwarz/Documents/Masters/FullPipeline/')
from sklearn.cluster import k_means
from time import time
from collections import Counter
import warnings
import os
import deepdish as dd
sys.path.append('/export/home/kschwarz/Documents/Masters/SmallNets/')
import h5py
sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod
from faiss_master import faiss


# DEAL WITH RANDOMNESS
seed = 123
np.random.seed(123)


# LOAD DATA
info_file = '../wikiart/datasets/info_artist_49_multilabel_test.hdf5'
# info_file = '../wikiart/datasets/info_artist_49_style_test.hdf5'
# feature_file = '../SmallNets/output/MobileNetV2_info_artist_49_style_train_small.hdf5'
feature_file = '../SmallNets/output/MobileNetV2_info_artist_49_multilabel_test_multilabel.hdf5'
# feature_file = '../SmallNets/output/MobileNetV2_info_artist_49_test.hdf5'
# feature_file = '../SmallNets/output/Inception3_info_artist_49_test.hdf5'
# feature_file = '../SmallNets/output/vgg16_artsiom_info_artist_49_test.hdf5'
# feature_file = '../SmallNets/output/MobileNetV2_info_artist_49_style_test_ap_labels_4.hdf5'
info_dict = dd.io.load(info_file)['df']

image_names = info_dict['image_id'].values.astype(str)
features = dd.io.load(feature_file)['features']
assert (image_names == dd.io.load(feature_file)['image_names']).all(), 'Features do not match image names in info file.'
# normalize features
features = (features - np.mean(features, axis=1, keepdims=True)) / np.linalg.norm(features, axis=1, keepdims=True)
categories = ['artist_name', 'style', 'genre', 'technique', 'century']
labels = np.stack(info_dict[k].values for k in categories if k in info_dict.keys()).transpose()


# SETTINGS
embedding_func = snack_embed_mod        # function to use for low dimensional projection / embedding
kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 1.0,
          'perplexity': 30, 'theta': 0.5, 'no_dims': 2, 'early_exaggeration': 1}         # kwargs for embedding_func

if __name__ == '__main__':              # dummy main to initialise global variables
    print('Global variables for embedding computation set.')


def initial_embedding(features, embedding_func, **kwargs):
    print('Compute embedding...')
    tic = time()
    embedding = embedding_func(np.stack(features).astype(np.double),
                               triplets=np.zeros((1, 3), dtype=np.long),     # dummy values
                               position_constraints=np.zeros((1, 3)),        # dummy values
                               **kwargs).transpose()
    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    return embedding


def construct_nodes(names, positions, labels=None):
    if labels is None:
        labels = [None] * len(names)
    nodes = pd.DataFrame({'name': names, 'x': positions[0], 'y': positions[1], 'labels': list(labels)})
    return nodes.to_dict(orient='index')


def df_to_dict(dataframe):
    return dataframe.to_dict(orient='index')


def dict_to_df(dictionary):
    return pd.DataFrame.from_dict(dictionary, orient='index')


def compute_graph(current_graph=[]):
    global categories
    if len(current_graph) == 0:
        global image_names, features, labels, embedding_func, kwargs
        positions = initial_embedding(features, embedding_func, **kwargs)
        nodes = construct_nodes(image_names, positions, labels)
        return nodes, categories

    else:
        node_df = dict_to_df(current_graph)
        nodes = construct_nodes(node_df['name'], (node_df['x'], node_df['y']), node_df['labels'])
        return nodes, categories
