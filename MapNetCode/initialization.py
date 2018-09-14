import os
import numpy as np
import pandas as pd
import deepdish as dd
import torch

from communication import make_nodes
from train import initialize_embedder, compute_embedding
from model import MapNet


info_file = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
feature_file = 'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
weight_file = 'runs/embedder/08-03-12-48_TSNENet_large_model_model_best.pth.tar'

info_file_shape = '/export/home/kschwarz/Documents/Data/Geometric_Shapes/labels.hdf5'
feature_file_shape = 'features/ShapeDataset_NarrowNet128_MobileNetV2_test.hdf5'
weight_file_shape = 'runs/embedder/models/ShapeDataset_MapNet_embedder_09-12-13-07.pth.tar'

# if not os.getcwd().endswith('/MapNetCode'):
#     os.chdir(os.path.join(os.getcwd(), 'MapNetCode'))


def load_feature(feature_file):
    """Load features from feature file."""
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')

    data = dd.io.load(feature_file)
    return data['image_names'], data['features']


def initialize(shape_dataset=False, **kwargs):   #(info_file, feature_file, weight_file=None):
    print(os.getcwd())
    if 'experiment_id' not in kwargs.keys():
        kwargs['experiment_id'] = None
    if shape_dataset:
        if kwargs['experiment_id'] is None:
            kwargs['experiment_id'] = 'ShapeDataset'
        else:
            kwargs['experiment_id'] = 'ShapeDataset_' + kwargs['experiment_id']

        data = dd.io.load(info_file_shape)['df']
        split = feature_file_shape.split('_')[-1].replace('.hdf5', '')
        print('Shape Dataset Split: {}'.format(split))
        if split == 'train':
            split = 0
        elif split == 'val':
            split = 1
        else:       # test
            split = 2
        data = data.loc[data['split'] == split]
        data.index = range(len(data))

        id = data['image_id']
        categories = ['shape', 'n_shapes', 'color_shape', 'color_background', 'group']
        ft_id, feature = load_feature(feature_file_shape)

    else:
        data = dd.io.load(info_file)['df']
        id = data['image_id']
        categories = ['artist_name', 'style', 'genre', 'technique', 'century']
        ft_id, feature = load_feature(feature_file)

    label = np.stack([data[k] if k in categories else [None] * len(id) for k in data.keys()], axis=1)

    if not (ft_id == id).all():
        raise ValueError('Image IDs in feature file do not match IDs in info file.')
    del ft_id

    # initialize the network
    net = MapNet(feature_dim=feature.shape[1], output_dim=2)
    if shape_dataset:
        initialize_embedder(net.embedder, weight_file_shape, **kwargs)
    else:
        initialize_embedder(net.embedder, weight_file, **kwargs)
    embedding = compute_embedding(net.embedder, feature)

    data = {
        'name': id,
        'label': label,
        'position': embedding,
        'feature': feature,
        'categories': categories,
        'experiment_id': kwargs['experiment_id']
    }

    print('Prepare network for training...')
    # add a tiny bit of noise, so each parameter in mapping contributes     # TODO: CHECK IF THIS IS ACTUALLY HELPING
    for name, param in net.mapping.named_parameters():
        if name.endswith('weight'):
            param.data.copy_(param.data + torch.rand(param.shape).type_as(param.data) * 1e-8)
    print('Done.')

    return net, data
