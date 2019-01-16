import os
import torch
import deepdish as dd
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from umap import UMAP

from model import load_featurenet
from dataset import load_ImageDataset
from helpers import get_imgid


thisdir = os.path.dirname(os.path.realpath(__file__))


# impath = '/export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224/test'
# normalization_file = os.path.join(thisdir, 'dataset_info/Wikiart_Elgammal_train_mean_std.pkl')
# batchsize = 32
# verbose = True
# outfilename = 'Wikiart_Elgammal_test'
# feature_dim = 512


def load_data(impath, normalization_file=None, verbose=False):
    if verbose:
        print('Load normalized resized Tensor Dataset.')
    return load_ImageDataset(impath, normalization_file=normalization_file, extensions=('.jpg', '.png'), size=224,
                             verbose=verbose)


def extract_features(impath, normalization_file=None, batchsize=64, use_gpu=True, verbose=False,
                     outdir=os.path.join(thisdir, 'features'), outfilename=None, feature_dim=None):
    """Extract initial features from images using ImageNet pretrained VGG16_bn."""
    use_gpu = use_gpu and torch.cuda.is_available()

    model = load_featurenet()
    if use_gpu:
        model = model.cuda()
    dataset = load_data(impath, normalization_file=normalization_file, verbose=verbose)

    loader_kwargs = {'num_workers': 4} if use_gpu else {}
    dataloader = DataLoader(dataset, batchsize, shuffle=False, drop_last=False, **loader_kwargs)

    features = []
    for i, data in enumerate(dataloader):
        if verbose:
            print('{}/{}'.format(i+1, len(dataloader)))
        input = data.cuda() if use_gpu else data
        output = model(input)
        features.append(output.data.cpu())

    features = torch.cat(features)
    features = features.numpy()

    pca = None
    if feature_dim is not None and feature_dim < features.shape[1]:
        if verbose:
            print('Reduce feature dimension from {} to {} using PCA.'.format(features.shape[1], feature_dim))
        pca = PCA(n_components=feature_dim)
        features = pca.fit_transform(features)

    img_id = map(get_imgid, dataset.filenames)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if outfilename is None:
        outfilename = dataset.impath.split('/')[-1]        # use name of image folder
    outfilename = outfilename.split('.')[0]         # in case ending is provided

    if feature_dim is not None:
        outfilename = outfilename + '_{}'.format(feature_dim)

    out_dict = {'image_id': img_id, 'features': features}
    dd.io.save(os.path.join(outdir, outfilename + '.hdf5'), out_dict)

    if pca is not None:
        with open(os.path.join(outdir, outfilename + '_PCA.pkl'), 'w') as f:
            pickle.dump(pca, f)

    if verbose:
        print('Saved features to {}'.format(os.path.join(outdir, outfilename + '.hdf5')))


def load_features(filename, dir=os.path.join(thisdir, 'features')):
    return dd.io.load(os.path.join(dir, filename))


def apply_pca(features, filename, dir=os.path.join(thisdir, 'features')):
    filename = filename.replace('.pkl', '')     # in case extension is provided
    with open(os.path.join(dir, filename + '.pkl'), 'r') as f:
        pca = pickle.load(f)

    return pca.transform(features)


def get_projection(features, projection_dim=2):
    projector = UMAP(n_neighbors=30, n_components=projection_dim, min_dist=0.1)
    return projector.fit_transform(features)


def get_data(dataset_name, impath=None, normalization_file=None, info_file=None, feature_dim=None, verbose=False):
    feature_dir = os.path.join(thisdir, 'features')
    projection_dir = os.path.join(thisdir, 'initial_projections')
    outfilename = '{}.hdf5'.format(dataset_name) if feature_dim is None \
        else '{}_{}.hdf5'.format(dataset_name, feature_dim)

    if not os.path.isfile(os.path.join(feature_dir, outfilename)):
        if impath is None:
            raise RuntimeError('Impath must be given if feature file of dataset does not exist.'
                               'Feature file has to be saved under: {}'
                               .format(os.path.join(feature_dir, dataset_name + '.hdf5')))

        if normalization_file is None:          # try to find normalization file
            normalization_file = os.path.join(os.path.join(thisdir, 'dataset_info', dataset_name + '_mean_std.pkl'))
            if not os.path.isfile(normalization_file):
                normalization_file = None

        print('Extract initial features of the images at {}.'.format(impath))
        extract_features(impath, normalization_file=normalization_file, batchsize=32,
                         use_gpu=torch.cuda.is_available(), outfilename=outfilename, outdir=feature_dir,
                         verbose=verbose)
    feature_data = load_features(outfilename, dir=feature_dir)

    # compute or load initial projection
    projection_file = os.path.join(projection_dir, outfilename)
    if not os.path.isfile(projection_file):
        print('Compute initial projection...')
        feature_data['projection'] = get_projection(feature_data['features'], projection_dim=2)
        print('Done.')
        if not os.path.isdir(projection_dir):
            os.makedirs(projection_dir)
        dd.io.save(projection_file, {'projection': feature_data['projection']})
    else:
        feature_data['projection'] = dd.io.load(projection_file)['projection']

    if info_file is not None:
        info = dd.io.load(info_file)
        if 'df' in info.keys():            # sometimes this format is used
            print('Search data info under "df" in info file.')
            info = info['df']
        if not 'image_id' in info.keys():
            raise RuntimeError('Info file needs to have image_id column.')
        info.set_index('image_id', inplace=True)
        if not sorted(info.index) == sorted(feature_data['image_id']):
            raise RuntimeWarning('Image IDs in info file and feature file cannot be matched completely.')
        # copy the relevant information
        data_info = pd.DataFrame(data=None, index=feature_data['image_id'], columns=info.columns)
        data_info.loc[info.index] = info.values
    else:
        data_info = None

    return feature_data, data_info