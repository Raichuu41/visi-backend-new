# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
import deepdish as dd
import pandas as pd
import argparse
sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod

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


def get_globals():
    globals = argparse.Namespace()
    globals.image_names = image_names
    globals.features = features
    globals.categories = categories
    globals.labels = labels
    globals.embedding_func = embedding_func
    globals.embedding_func_kwargs = kwargs
    globals.embedding = None
    return globals


def df_to_dict(dataframe):
    dct = dataframe.to_dict(orient='index')
    # replace any values in arrays by lists
    array_keys = [k for k, v in dct[0].items() if isinstance(v, np.ndarray)]
    for d in dct:
        for k in array_keys:
            dct[d][k] = list(dct[d][k])
    return dct


def dict_to_df(dictionary):
    return pd.DataFrame.from_dict(dictionary, orient='index')