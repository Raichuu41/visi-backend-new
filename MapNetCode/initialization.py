import os
import numpy as np
import pandas as pd
import deepdish as dd
import torch

from communication import make_nodes
from train import initialize_embedder, compute_embedding
from model import MapNet


info_file = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
feature_file = 'MapNetCode/features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
weight_file = 'MapNetCode/runs/embedder/08-03-12-48_TSNENet_large_model_model_best.pth.tar'

# if not os.getcwd().endswith('/MapNetCode'):
#     os.chdir(os.path.join(os.getcwd(), 'MapNetCode'))


def load_feature(feature_file):
    """Load features from feature file."""
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')

    data = dd.io.load(feature_file)
    return data['image_name'], data['features']


def initialize():   #(info_file, feature_file, weight_file=None):
    data = dd.io.load(info_file)['df']
    id = data['image_id']
    categories = ['artist_name', 'style', 'genre', 'technique', 'century']
    label = np.stack([data[k] for k in data.keys() if k in categories], axis=1)

    ft_id, feature = load_feature(feature_file)

    if not (ft_id == id).all():
        raise ValueError('Image IDs in feature file do not match IDs in info file.')
    del ft_id

    # initialize the network
    net = MapNet(feature_dim=feature.shape[1], output_dim=2)
    initialize_embedder(net.embedder, weight_file)
    embedding = compute_embedding(net.embedder, feature)

    data = {
        'name': id,
        'label': label,
        'position': embedding,
        'feature': feature,
        'categories': categories
    }

    print('Prepare network for training...')
    # add a tiny bit of noise, so each parameter in mapping contributes     # TODO: CHECK IF THIS IS ACTUALLY HELPING
    for name, param in net.mapping.named_parameters():
        if name.endswith('weight'):
            param.data.copy_(param.data + torch.rand(param.shape).type_as(param.data) * 1e-8)
    print('Done.')

    return net, data
