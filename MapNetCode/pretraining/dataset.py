import os
import deepdish as dd
import pandas as pd
import numpy as np
import pickle
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, ToTensor

from PIL import Image


class Wikiart(data.Dataset):
    def __init__(self, path_to_info_file, path_to_images, transform=None, classes=None):
        self.impath = path_to_images
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.df = self._get_df(path_to_info_file)
        self.labels_to_ints = self._get_labels_to_ints()

    def _get_df(self, path_to_info_dict):
        assert os.path.isfile(path_to_info_dict), 'Info dict not found at {}'.format(path_to_info_dict)
        print('Load info file...')
        info_dict = dd.io.load(path_to_info_dict)['df']
        print('Done.')
        if not 'image_id' in info_dict.keys():
            info_dict['image_id'] = info_dict.index
        for c in self.classes:
            assert c in info_dict.keys(), \
                'Class {} does not exist in info dict. Select from {}'.format(c, info_dict.keys())
        classes = list(set(self.classes + ['image_id']))
        # # do not allow missing labels
        # dfs = [info_dict[c] for c in classes]
        # masks = np.array([df.isnull().values for df in dfs]).__invert__()
        # idcs = np.where(np.prod(masks, axis=0))[0]
        # dfs = {df.name: df[idcs] for df in dfs}
        # return pd.DataFrame(dfs)

        # do not allow samples with no labels
        mask = info_dict[self.classes].isnull().values.__invert__()
        valid = np.any(mask, axis=1)
        df = info_dict[valid]
        df.index = range(len(df))

        return df[classes]

    def _get_labels_to_ints(self):
        labels_to_ints = {}
        for c in self.classes:
            df = self.df[c].dropna()
            labels = set(df.values)
            if None in labels:
                labels.remove(None)
            labels_to_ints[c] = {l: i for i, l in enumerate(sorted(labels))}
        return labels_to_ints

    def convert_ints_to_labels(self, int_labels):
        labels = []
        for il, c in zip(int_labels, self.classes):
            il_list = self.labels_to_ints[c].values()
            idx = il_list.index(il)
            labels.append(self.labels_to_ints[c].keys()[idx])
        return labels

    def __len__(self):
        return len(self.df['image_id'])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.impath, self.df['image_id'][index] + '.jpg')).convert(mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        sample_labels = [self.df[c][index] for c in self.classes]
        int_labels = []
        for c, l in zip(self.classes, sample_labels):
            lbl = -1 if l not in self.labels_to_ints[c].keys() else self.labels_to_ints[c][l]
            int_labels.append(lbl)
        return img, int_labels


def compute_mean_std(path_to_info_file, impath='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images'):
    dset = Wikiart(path_to_info_file, path_to_images=impath, classes=['image_id'],
                   transform=Compose([Resize((224, 224)), ToTensor()]))
    batch_size = 1000
    loader = data.DataLoader(dset, batch_size=batch_size, num_workers=8)

    mean = []
    std = []
    for i, (img, _) in enumerate(loader):
        print('{}/{}'.format(i+1, len(loader)))
        img = img.view(img.shape[0], 3, -1)
        m = [im.mean(dim=1) for im in img]
        s = [im.std(dim=1) for im in img]
        mean.append(torch.stack(m).mean(dim=0))
        std.append(torch.stack(s).mean(dim=0))

    mean = torch.stack(mean).mean(dim=0)
    std = torch.stack(std).mean(dim=0)

    if not os.path.isdir('wikiart_datasets'):
        os.makedirs('wikiart_datasets')

    # fname = os.path.join('wikiart_datasets', path_to_info_file.split('/')[-1].split('.')[0])
    fname = path_to_info_file.replace('.hdf5', '')
    with open(fname + '_mean_std.pkl', 'w') as f:
        pickle.dump({'mean': list(mean.numpy()), 'std': list(std.numpy())}, f)

    print('saved \n\tstat file: {}'.format(fname + '_mean_std.pkl'))


# path_to_info_file = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_train.hdf5'
# impath = '/export/home/asanakoy/workspace/wikiart/images'
# compute_mean_std(path_to_info_file, impath)