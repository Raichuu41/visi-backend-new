import os
import numpy as np
import matplotlib
import torch
import torchvision
import deepdish as dd
import pandas as pd
import pickle
from functools import partial
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from umap import UMAP
from copy import copy

from model import load_featurenet, mapnet_1, mapnet_2, mapnet_3, mapnet_4
from dataset import load_ImageDataset, ImageDataset
from helpers import get_imgid
from dataset import compute_mean_std
from aux import load_weights
from itertools import product


thisdir = os.path.dirname(os.path.realpath(__file__))


class Initializer(object):
    def __init__(self, dataset_name, impath=None, info_file=None,
                 feature_dim=None, outdir=thisdir,
                 data_extensions=('.jpg', '.png'),
                 N_multi_features=10,
                 verbose=False):
        self.dataset_name = dataset_name
        self.impath = impath
        self.info_file = info_file
        self.info = self.get_info(self.info_file)
        self.feature_dim = feature_dim
        self.data_extensions = data_extensions
        self.verbose = verbose

        self._dataset = None
        self._pca = None

        self.data_dir = os.path.join(outdir, 'dataset_info')
        self.feature_dir = os.path.join(outdir, 'features')
        self.projection_dir = os.path.join(outdir, 'initial_projections')

        self.normalization_file = os.path.join(self.data_dir, '{}_mean_std.pkl'.format(self.dataset_name))
        self.feature_file = os.path.join(self.feature_dir,
                                         '{}.h5'.format(self.dataset_name) if self.feature_dim is None else
                                         '{}_{}.h5'.format(self.dataset_name, self.feature_dim)
                                         )
        self.N_multi_features = N_multi_features
        self.multi_feature_file = self.feature_file.replace('.h5', '_multi.h5')
        self.projection_file = os.path.join(self.projection_dir, '{}.h5'.format(self.dataset_name))

    @staticmethod
    def get_info(info_file):
        if info_file is None or not os.path.isfile(info_file):
            return None
        try:
            info = dd.io.load(info_file)['df']
        except KeyError:
            info = dd.io.load(info_file)

        if 'image_id' in info.keys():
            info = info.set_index('image_id')
        return info

    def make_dataset(self, normalize=True, imsize=(224, 224), transform_list=None):
        if self.impath is None or not os.path.isdir(self.impath):
            raise AttributeError('Initializer needs valid impath to create dataset.')
        if transform_list is not None and not isinstance(transform_list, list):
            raise AttributeError('"transform_list" needs to be a list.')

        image_ids = None if self.info is None else self.info.index.values

        # standard image transformations
        transforms = [torchvision.transforms.ToTensor()]

        if normalize:
            if self.normalization_file is None or not os.path.isfile(self.normalization_file):
                raise AttributeError('Normalization file not found. '
                                     'Set normalize=False or make normalization file first '
                                     'using "make_normalization_file".')
            with open(self.normalization_file, 'r') as f:
                data_dict = pickle.load(f)
                mean = torch.FloatTensor(data_dict['mean'])
                std = torch.FloatTensor(data_dict['std'])
            transforms.append(torchvision.transforms.Normalize(mean=mean, std=std))
        if imsize is not None:
            transforms.insert(0, torchvision.transforms.Resize(imsize))

        if transform_list is not None:
            transforms = transform_list + transforms

        if self._dataset is not None:
            self._dataset.transform = torchvision.transforms.Compose(transforms)
        else:
            self._dataset = ImageDataset(self.impath, image_ids=image_ids, extensions=self.data_extensions,
                                         transform=torchvision.transforms.Compose(transforms))
        return self._dataset

    def make_normalization_file(self, imsize=(224, 224)):
        if self.impath is None or not os.path.isdir(self.impath):
            raise AttributeError('Initializer needs valid impath to compute normalization file.')

        image_ids = None if self.info is None else self.info.index.values
        self._dataset = ImageDataset(self.impath, image_ids=image_ids, extensions=self.data_extensions,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(imsize),
                                         torchvision.transforms.ToTensor()
                                     ]))
        if self.verbose:
            print('Compute normalization file...')
        compute_mean_std(self._dataset, filename=self.dataset_name, outdir=self.data_dir, verbose=self.verbose)
        if self.verbose:
            print('Done.')

    def set_normalization_file(self, normalization_file):
        if not os.path.isfile(normalization_file):
            raise IOError('Normalization file not found.')
        self.normalization_file = normalization_file
        if self._dataset is not None:
            for t in self._dataset.transform.transforms:
                if isinstance(t, torchvision.transforms.Normalize):
                    with open(self.normalization_file, 'r') as f:
                        data_dict = pickle.load(f)
                        mean = torch.FloatTensor(data_dict['mean'])
                        std = torch.FloatTensor(data_dict['std'])
                    t.mean = mean
                    t.std = std

    @staticmethod
    def get_features(dataset, batchsize=64, use_gpu=True, weightfile=None, verbose=False):
        use_gpu = use_gpu and torch.cuda.is_available()

        model = load_featurenet()       # VGG16_bn pretrained
        if weightfile is not None:
            pretrained_dict = load_weights(weightfile, model.state_dict())
            model.load_state_dict(pretrained_dict)

        if use_gpu:
            model = model.cuda()

        loader_kwargs = {'num_workers': 4} if use_gpu else {}
        dataloader = DataLoader(dataset, batchsize, shuffle=False, drop_last=False, **loader_kwargs)

        features = []
        for i, data in enumerate(dataloader):
            if verbose:
                print('{}/{}'.format(i + 1, len(dataloader)))
            input = data.cuda() if use_gpu else data
            output = model(input)
            features.append(output.data.cpu())

        features = torch.cat(features)
        return features.numpy()

    def make_feature_file(self, pca=None, **get_feature_kwargs):
        if self._dataset is None:
            raise RuntimeError('Dataset was not initialized. Please call "make_dataset" before starting.')
        if not any([isinstance(t, torchvision.transforms.Normalize) for t in self._dataset.transform.transforms]):
            raise RuntimeWarning('Dataset is not being normalized. Did you forget calling "make_dataset"?')
        save_pca = False

        features = self.get_features(dataset=self._dataset, verbose=self.verbose, **get_feature_kwargs)
        if self.feature_dim is not None and self.feature_dim < features.shape[1]:
            if self.verbose:
                print('Reduce feature dimension from {} to {} using PCA.'.format(features.shape[1], self.feature_dim))
            if pca is None:
                save_pca = True
                pca = PCA(n_components=self.feature_dim)
                features = pca.fit_transform(features)
            else:
                if not pca.n_components == self.feature_dim:
                    raise RuntimeError('n_components of PCA does not match feature dimension.')
                features = pca.transform(features)

        if not os.path.isdir(self.feature_dir):
            os.makedirs(self.feature_dir)

        img_id = map(lambda x: x.split('.')[0], self._dataset.filenames)

        out_dict = {'image_id': img_id, 'features': features, 'precomputed_pca': not save_pca}
        dd.io.save(self.feature_file, out_dict)

        if save_pca:
            with open(self.feature_file.replace('.h5', '_PCA.pkl'), 'w') as f:
                pickle.dump(pca, f)

        if self.verbose:
            print('Saved features to {}'.format(self.feature_file))
            if save_pca:
                print('Saved PCA to {}'.format(self.feature_file.replace('.h5', '_PCA.pkl')))

    def make_raw_feature_file(self, **get_feature_kwargs):
        if self._dataset is None:
            raise RuntimeError('Dataset was not initialized. Please call "make_dataset" before starting.')
        if not any([isinstance(t, torchvision.transforms.Normalize) for t in self._dataset.transform.transforms]):
            raise RuntimeWarning('Dataset is not being normalized. Did you forget calling "make_dataset"?')

        features = self.get_features(dataset=self._dataset, verbose=self.verbose, **get_feature_kwargs)

        if not os.path.isdir(self.feature_dir):
            os.makedirs(self.feature_dir)

        img_id = map(lambda x: x.split('.')[0], self._dataset.filenames)

        out_dict = {'raw': {'image_id': img_id, 'features': features}}

        if os.path.isfile(self.feature_file):
            data_dict = dd.io.load(self.feature_file)
            data_dict.update(out_dict)
            dd.io.save(self.feature_file, data_dict)
            if self.verbose:
                print('Appended features to {}'.format(self.feature_file))
        else:
            dd.io.save(self.feature_file, out_dict)
            if self.verbose:
                print('Saved features to {}'.format(self.feature_file))

    def make_multi_feature_file(self, pca=None, **get_feature_kwargs):
        if pca is not None:           # compute normal features and only apply multi pca to them
            ft_filename = self.feature_file
            self.feature_file = self.multi_feature_file
            self.make_feature_file(batchsize=16, pca=pca)     # make feature file - save it under multi features
            self.feature_file = ft_filename     # reset the name of the feature file

        else:
            if self._dataset is None:
                raise RuntimeError('Dataset was not initialized. Please call "make_dataset" before starting.')
            if not any([isinstance(t, torchvision.transforms.Normalize) for t in self._dataset.transform.transforms]):
                raise RuntimeWarning('Dataset is not being normalized. Did you forget calling "make_dataset"?')
            save_pca = False

            transform_orig = self._dataset.transform
            self._dataset.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomResizedCrop(
                            224, scale=(0.3, 1.0), ratio=(0.8, 1.2)),
                        torchvision.transforms.RandomRotation((-45, 45))
                    ], p=0.8)
                ] + transform_orig.transforms
            )

            features = []
            for i in range(self.N_multi_features):
                if self.verbose and (i % 10 == 0):
                    print('{}/{}'.format(i, self.N_multi_features))
                features.append(self.get_features(dataset=self._dataset, verbose=self.verbose, **get_feature_kwargs))

            self._dataset.transform = transform_orig

            if self.feature_dim is not None and self.feature_dim < features[0].shape[1]:
                if self.verbose:
                    print('Reduce feature dimension from {} to {} using PCA.'.format(features[0].shape[1], self.feature_dim))
                if pca is None:
                    save_pca = True
                    pca = PCA(n_components=self.feature_dim)
                    pca.fit(np.concatenate(features[:min(len(features), 10)]))           # fit some features jointly
                else:
                    if not pca.n_components == self.feature_dim:
                        raise RuntimeError('n_components of PCA does not match feature dimension.')
                features = [pca.transform(fts) for fts in features]

            features = np.stack(features)

            if not os.path.isdir(self.feature_dir):
                os.makedirs(self.feature_dir)

            img_id = map(get_imgid, self._dataset.filenames)

            out_dict = {'image_id': img_id, 'features': features, 'precomputed_pca': not save_pca}
            dd.io.save(self.multi_feature_file, out_dict)

            if save_pca:
                with open(self.multi_feature_file.replace('.h5', '_PCA.pkl'), 'w') as f:
                    pickle.dump(pca, f)

            if self.verbose:
                print('Saved features to {}'.format(self.multi_feature_file))
                if save_pca:
                    print('Saved PCA to {}'.format(self.multi_feature_file.replace('.h5', '_PCA.pkl')))

    def set_feature_file(self, feature_file):
        if not os.path.isfile(feature_file):
            raise IOError('Feature file not found.')
        self.feature_file = feature_file
        data = dd.io.load(feature_file)
        feature_dim = data['features'].shape[1]
        image_id = data['image_id']

        if self.feature_dim is None or self.feature_dim != feature_dim:
            raise RuntimeWarning('Change feature dimension from {} to {}'.format(self.feature_dim, feature_dim))
        self.feature_dim = feature_dim

        if self.info is not None:
            if not np.all(np.isin(image_id, self.info.index.values)):
                raise RuntimeWarning('Info file not up to date - '
                                     'Some entries in feature file are missing in info file.')
            if len(self.info) != len(image_id):
                raise RuntimeWarning('Image IDs in info file and feature file do not match.')

        if os.path.isfile(self.projection_file):
            p_data = dd.io.load(self.projection_file)
            if not np.all(image_id == p_data['image_id']):
                if sorted(image_id) == sorted(p_data['image_id']):
                    raise RuntimeWarning('Order in feature file does not match order in projection file.'
                                         'Re-order projection file.')
                    mapped_idcs = map(lambda x: np.where(image_id == x)[0][0], p_data['image_id'])
                    p_data['projection'] = p_data['projection'][np.argsort(mapped_idcs)]
                    p_data['image_id'] = image_id
                    dd.io.save(self.projection_file, p_data)
                else:
                    raise RuntimeWarning('Projection file does not match feature file.')

    @staticmethod
    def get_projection(features, projection_dim=2, verbose=False, random_state=123):
        projector = UMAP(n_neighbors=30, n_components=projection_dim, min_dist=0.1, random_state=random_state,
                         verbose=verbose)
        return projector.fit_transform(features)

    def make_projection_file(self, **get_projection_kwargs):
        if not os.path.isfile(self.feature_file):
            raise RuntimeError('Feature file not found. Please call "make_feature_file" before starting or use the'
                               '"get_projection" method.')
        data = dd.io.load(self.feature_file)
        projection = self.get_projection(data['features'], verbose=self.verbose, **get_projection_kwargs)
        out_dict = {'image_id': data['image_id'], 'projection': projection}
        dd.io.save(self.projection_file, out_dict)

        if self.verbose:
            print('Saved projection to {}'.format(self.projection_file))

    def set_projection_file(self, projection_file):
        if not os.path.isfile(projection_file):
            raise IOError('Projection file not found.')
        self.projection_file = projection_file
        data = dd.io.load(projection_file)
        image_id = data['image_id']

        if self.info is not None:
            if not np.all(np.isin(image_id, self.info.index.values)):
                raise RuntimeWarning('Info file not up to date - '
                                     'Some entries in projection file are missing in info file.')
            if len(self.info) != len(image_id):
                raise RuntimeWarning('Image IDs in info file and projection file do not match.')

        if os.path.isfile(self.feature_file):
            f_data = dd.io.load(self.feature_file)
            if not np.all(image_id == f_data['image_id']):
                if sorted(image_id) == sorted(f_data['image_id']):
                    raise RuntimeWarning('Order in projection file does not match order in feature file.'
                                         'Re-order feature file.')
                    mapped_idcs = map(lambda x: np.where(image_id == x)[0][0], f_data['image_id'])
                    f_data['features'] = f_data['features'][np.argsort(mapped_idcs)]
                    f_data['image_id'] = image_id
                    dd.io.save(self.feature_file, f_data)
                else:
                    raise RuntimeWarning('Feature file does not match projection file.')

    def get_data_dict(self, normalize_features=True):
        data_dict = dict.fromkeys(['image_id', 'features', 'projection', 'info'])

        if os.path.isfile(self.feature_file):
            data = dd.io.load(self.feature_file)
            data_dict['image_id'] = data['image_id']
            data_dict['features'] = data['features']
            if 'raw' in data.keys():
                if np.any(data['image_id'] != data['raw']['image_id']):
                    if sorted(data['image_id']) == sorted(data['raw']['image_id']):
                        sort_idx = map(lambda x: data['raw']['image_id'].index(x), data['image_id'])
                        data['raw']['features'] = data['raw']['features'][sort_idx]
                        data['raw']['image_id'] = list(np.array(data['raw']['image_id'])[sort_idx])
                        data_dict['features_raw'] = data['raw']['features']
                    else:
                        raise RuntimeError('Image IDs in feature file and raw feature file do not match.')
                else:
                    data_dict['features_raw'] = data['raw']['features']
            if normalize_features:
                data_dict['features'] /= np.linalg.norm(data_dict['features'], axis=1, keepdims=True)
                if 'features_raw' in data_dict.keys():
                    data_dict['features_raw'] /= np.linalg.norm(data_dict['features_raw'], axis=1, keepdims=True)

        if os.path.isfile(self.multi_feature_file):
            data = dd.io.load(self.multi_feature_file)
            if data_dict['image_id'] is not None:       # check if indexing is correct
                if np.any(data_dict['image_id'] != data['image_id']):
                    raise RuntimeError('Image IDs in feature file and projection file do not match.')
            else:
                data_dict['image_id'] = data['image_id']

            data_dict['multi_features'] = data['features']
            if normalize_features:
                if data_dict['multi_features'].ndim == 3:
                    data_dict['multi_features'] /= np.linalg.norm(data_dict['multi_features'], axis=2, keepdims=True)
                else:
                    data_dict['multi_features'] /= np.linalg.norm(data_dict['multi_features'], axis=1, keepdims=True)

        if os.path.isfile(self.projection_file):
            data = dd.io.load(self.projection_file)
            if data_dict['image_id'] is not None:       # check if indexing is correct
                if np.any(data_dict['image_id'] != data['image_id']):
                    raise RuntimeError('Image IDs in feature file and projection file do not match.')
            else:
                data_dict['image_id'] = data['image_id']
            data_dict['projection'] = data['projection']

        if self.info is not None:
            if data_dict['image_id'] is None:
                data_dict['image_id'] = self.info.index.values
            if not np.all(np.isin(data_dict['image_id'], self.info.index.values)):
                raise RuntimeWarning('Info file not up to date - '
                                     'Some entries in feature file / projection file are missing in info file.')
            if len(self.info) != len(data_dict['image_id']):
                raise RuntimeWarning('Image IDs in info file and feature file / projection file do not match.')
            elif not sorted(self.info.index.values) == sorted(data_dict['image_id']):
                raise RuntimeWarning('Image IDs in info file and feature file / projection file '
                                     'cannot be matched completely.')
            data_dict['info'] = self.info.loc[data_dict['image_id']]

        return data_dict

    def initialize(self, dataset=True, features=True, projection=True, multi_features=False, raw_features=False,
                   is_test=False):
        if dataset:
            if not os.path.isfile(self.normalization_file):
                if is_test:
                    normalization_file = self.normalization_file.replace('_test', '_train')
                    if not os.path.isfile(normalization_file):
                        raise IOError('Normalization file of training set not found.')
                    self.set_normalization_file(normalization_file)
                else:
                    self.make_normalization_file()
            self.make_dataset(normalize=True, imsize=(224, 224))

        if features:
            if not os.path.isfile(self.feature_file):
                if is_test:
                    pca_file = self.feature_file.replace('.h5', '_PCA.pkl').replace('_test', '_train')
                    if not os.path.isfile(pca_file):
                        raise IOError('PCA file of training set not found.')
                    with open(pca_file, 'r') as f:
                        pca = pickle.load(f)
                    self.make_feature_file(batchsize=16, pca=pca)
                else:
                    self.make_feature_file(batchsize=16)

        if projection:
            if not os.path.isfile(self.projection_file):
                self.make_projection_file()

        if multi_features:
            if not os.path.isfile(self.multi_feature_file):
                if is_test:
                    pca_file = self.multi_feature_file.replace('.h5', '_PCA.pkl').replace('_test', '_train')
                    if not os.path.isfile(pca_file):
                        raise IOError('PCA file of training set not found.')
                    with open(pca_file, 'r') as f:
                        pca = pickle.load(f)
                    self.make_multi_feature_file(batchsize=16, pca=pca)
                else:
                    self.make_multi_feature_file(batchsize=16)

        if raw_features:
            exists = os.path.isfile(self.feature_file)
            if exists:
                data_dict = dd.io.load(self.feature_file)
                exists = 'raw' in data_dict.keys()
            if not exists:
                self.make_raw_feature_file(batchsize=16)


if __name__ == '__main__':
    # labelnames = ['artist', 'genre', 'style']
    # splits = ['train', 'test']
    # dataset_dir = './dataset_info'
    # impath = '/export/home/kschwarz/Documents/Data/Wikiart_Elgammal'
    #
    # dataset_names = ['Wikiart_Elgammal_EQ_{}_{}'.format(l, s) for l, s in product(labelnames, splits)]

    labelnames = ['vectors']
    splits = ['train', 'test']
    dataset_dir = './dataset_info'
    impath = '/net/hci-storage02/groupfolders/compvis/datasets/Animals_with_Attributes2/single_folder_images2'

    dataset_names = ['AwA2_{}_{}'.format(l, s) for l, s in product(labelnames, splits)]

    # labelnames = ['label']
    # splits = ['test']
    # dataset_dir = './dataset_info'
    # impath = '/export/home/kschwarz/Documents/Data/STL/single_folder_images_test'

    # dataset_names = ['STL_{}_{}'.format(l, s) for l, s in product(labelnames, splits)]

    for i, dataset_name in enumerate(dataset_names):
        print('{}/{}'.format(i+1, len(dataset_names)))
        info_file = os.path.join(dataset_dir, 'info_{}.h5'.format(dataset_name))
        init = Initializer(dataset_name, impath=impath, info_file=info_file, verbose=True, feature_dim=512,
                           N_multi_features=100)
        init.initialize(multi_features=False, raw_features=True, is_test=dataset_name.endswith('_test'))
