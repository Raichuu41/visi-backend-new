import os
import deepdish as dd
import shutil
import torch
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
    filename = '{}_mean_std.pkl'.format(filename)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(filename, 'w') as f:
        pickle.dump({'mean': list(mean.numpy()), 'std': list(std.numpy())}, f)

    if verbose:
        print('saved \n\tstat file: {}'.format(filename))


def load_ImageDataset(impath, normalization_file=None, extensions=('.jpg', '.png'), size=None, verbose=False):
    """Returns normalized Tensor Dataset. If 'size' is specified the data is also resized."""
    dataset = ImageDataset(impath, extensions=extensions, transform=transforms.ToTensor())
    if normalization_file is None:
        normalization_file = '{}_mean_std.pkl'.format(dataset.impath.split('/')[-1])    # use name of image folder

    if not os.path.isfile(normalization_file):
        outdir = './dataset_info'
        compute_mean_std(dataset, normalization_file.replace('_mean_std.pkl', ''), outdir=outdir, verbose=verbose)
        normalization_file = os.path.join(outdir, normalization_file)

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

