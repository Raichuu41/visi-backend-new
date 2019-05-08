import os
import numpy as np
import deepdish as dd
import shutil
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from collections import Counter
from sklearn.model_selection import train_test_split


from helpers import ImageDataset


def compute_mean_std(dataset, filename=None, outdir='./dataset_info', verbose=False):
    batch_size = 100
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    mean = []
    std = []
    for i, img in enumerate(loader):
        if isinstance(img, tuple):          # in case of dataset of shape (data, target)
            img = img[0]
        if verbose:
            print('{}/{}'.format(i+1, len(loader)))
        img = img.view(img.shape[0], 3, -1)
        m = [im.mean(dim=1) for im in img]
        s = [im.std(dim=1) for im in img]
        mean.append(torch.stack(m).mean(dim=0))
        std.append(torch.stack(s).mean(dim=0))

    mean = torch.stack(mean).mean(dim=0)
    std = torch.stack(std).mean(dim=0)

    if filename is None:
        try:
            filename = dataset.impath.split('/')[-1]        # use name of image folder
        except AttributeError:
            filename = raw_input('Please enter name of Dataset to save mean_std file.')
    filename = os.path.join(outdir, '{}_mean_std.pkl'.format(filename))

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(filename, 'w') as f:
        pickle.dump({'mean': list(mean.numpy()), 'std': list(std.numpy())}, f)

    if verbose:
        print('saved \n\tstat file: {}'.format(filename))


def load_ImageDataset(impath, normalization_file=None, outdir_normalization_file='./dataset_info',
                      extensions=('.jpg', '.png'), size=None, verbose=False):
    """Returns normalized Tensor Dataset. If 'size' is specified the data is also resized."""
    dataset = ImageDataset(impath, extensions=extensions, transform=transforms.ToTensor())
    if normalization_file is None:
        normalization_file = '{}_mean_std.pkl'.format(dataset.impath.split('/')[-1])    # use name of image folder

    if not os.path.isfile(normalization_file):
        compute_mean_std(dataset, normalization_file.replace('_mean_std.pkl', ''), outdir=outdir_normalization_file,
                         verbose=verbose)
        normalization_file = os.path.join(outdir_normalization_file, normalization_file)
        print('Saved {}'.format(normalization_file))

    with open(normalization_file, 'r') as f:
        data_dict = pickle.load(f)
        mean = torch.FloatTensor(data_dict['mean'])
        std = torch.FloatTensor(data_dict['std'])

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if size is not None:
        transform_list.insert(0, transforms.Resize((size, size)))

    return ImageDataset(impath, extensions=extensions, transform=transforms.Compose(transform_list))


def make_separate_ImgFolder(info_file, impath, outdir, copy=True, extension='.jpg'):
    """Transfers images listed in info file to outdir. If copy is True images are copied instead of cutted."""
    # info_file = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_val.hdf5'
    # impath = '/export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224'
    # outdir = '/export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224/val'
    # copy = False
    # extension = '.jpg'

    img_ids = dd.io.load(info_file)['df']['image_id'].values

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    def get_filepath(img_id):
        return os.path.join(impath, img_id + extension)

    src = map(get_filepath, img_ids)

    if copy:
        mapfunc = lambda x: shutil.copy2(x, dst=outdir)
    else:
        mapfunc = lambda x: shutil.move(x, dst=outdir)

    map(mapfunc, src)


def make_wikiart_dataset(label, equally_distributed=False, min_per_label=None, frac_test=0.2,
                         outdir='./dataset_info', verbose=False):

    def filter_df_equally_distributed(info_df, label_col):
        labels = info_df[label_col].values
        labellist = np.unique(labels)
        count_labels = Counter(labels)
        n_per_label = min(count_labels.values())
        label_candidates = [np.random.permutation(np.where(labels == l)[0]) for l in labellist]
        sample_idcs = np.concatenate([np.random.choice(c, size=n_per_label, replace=False) for c in label_candidates])
        return info_df.iloc[sample_idcs]

    info_files = ['/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/' \
                  'wikiart_datasets/info_elgammal_subset_{}.hdf5'
                      .format(split) for split in ['train', 'test', 'val']]
    if equally_distributed:
        outfile_prefix = 'info_Wikiart_Elgammal_EQ_{}'.format(label.split('_')[0])
    else:
        outfile_prefix = 'info_Wikiart_Elgammal_{}'.format(label.split('_')[0])

    cols = ['image_id', label]
    df = None
    for info_file in info_files:
        df_ = dd.io.load(info_file)['df'][cols]
        df_ = df_.set_index('image_id')
        df_ = df_.dropna()
        if df is None:
            df = df_
        else:
            df.append(df_)

    if min_per_label is not None:
        label_count = Counter(df[label].values)
        outsorted = [k for k, v in label_count.items() if v < min_per_label]
        mask = np.isin(df[label].values, outsorted).__invert__()
        df = df[mask]
        if verbose and len(outsorted) > 0:
            print('Reduce labels from {} to {} due to "min_per_label".\nOut sorted labels: {}'
                  .format(len(label_count), len(label_count) - len(outsorted), outsorted))

    if equally_distributed:
        df = filter_df_equally_distributed(df, label_col=label)
        labelset = list(set(df[label].values))
        n_per_label = len(np.where(df[label] == labelset[0])[0])
        n_test = int(frac_test * n_per_label)
        if verbose:
            print('Take subset using {} samples per label.'.format(n_per_label))
        idcs_test = np.concatenate([
            np.random.choice(
                np.where(df[label] == l)[0], size=n_test, replace=False
            )
            for l in labelset])
        test_df = df.iloc[idcs_test]
        train_df = df.iloc[np.setdiff1d(range(len(df)), idcs_test)]
    else:
        train_df, test_df = train_test_split(df, test_size=frac_test)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    train_df.to_hdf(os.path.join(outdir, outfile_prefix + '_train.h5'), key='df')
    test_df.to_hdf(os.path.join(outdir, outfile_prefix + '_test.h5'), key='df')

    if verbose:
        print('Saved: \n\t{}\n\t{}'.format(os.path.join(outdir, outfile_prefix + '_train.h5'),
                                           os.path.join(outdir, outfile_prefix + '_test.h5')))


if __name__ == '__main__':
    make_wikiart_dataset(label='artist_name', equally_distributed=True, min_per_label=300, verbose=True)


