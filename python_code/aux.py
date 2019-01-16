import os
import torch
import shutil
import h5py
from tensorboardX import SummaryWriter
from PIL import Image
from math import ceil
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib as mpl
import random
import time


class TBPlotter(object):
    """Log values in tensorboard interface."""
    def __init__(self, logdir='runs/tensorboard'):
        super(TBPlotter, self).__init__()
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.logdir = logdir
        self.train_writer = SummaryWriter(os.path.join(logdir, 'train'))
        self.test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    def print_logdir(self):
        print('\033[95mtensorboard:\ntensorboard --logdir={}\033[0m'
              .format(os.path.join(os.getcwd(), self.logdir)))

    def write(self, name, data, epoch, test=False):
        if test:
            writer = self.test_writer
        else:
            writer = self.train_writer
        writer.add_scalar(name, data, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_config(args, exp_name, defaults=None, extras=None):
    if not isinstance(args, dict):
        configs = vars(args)
    else:
        configs = args.copy()
    to_file = {}
    if defaults is not None:
        for k, v in configs.items():
            if v != defaults[k]:
                to_file[k] = v
    else:
        to_file = configs.copy()
    if extras is not None:
        if not isinstance(extras, dict):
            print('"extras" in >>write_config<< should be dictionary. Now use key "extras" instead and '
                  'append the values.')
            extras = {'extras': extras}
        to_file.update(extras)

    if len(to_file):
        with open(exp_name + '_config.txt', 'wb') as f:
            for k, v in to_file.items():
                line = k + '\t\t\t\t\t\t' + str(v) + '\n'
                f.write(line)


def save_checkpoint(state, is_best, name, directory='runs/'):
    """Saves checkpoint to disk"""
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = name + '_checkpoint' + '.pth.tar'
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, os.path.join(directory, filename))
    if is_best:
        shutil.copyfile(os.path.join(directory, filename), os.path.join(directory, name + '_model_best.pth.tar'))


def multidim_colormap():
    cmaps = [plt.cm.gist_rainbow, plt.cm.cubehelix, plt.cm.terrain]
    cmaplist = []
    for cmap in cmaps:
        cmaplist.extend([cmap(i) for i in np.arange(0, cmap.N, dtype=int)])
    random.seed(123)
    random.shuffle(cmaplist)
    return mpl.colors.ListedColormap(cmaplist)


def plot_embedding_2d(embedding, labels, ax=None, colors=None, title=None):
    plt.ion()
    if colors is None:
        if len(np.unique(labels)) <= 10:
            colors = ['magenta', 'cyan', 'lime', 'indigo', 'y',
                  'lightseagreen', 'dodgerblue', 'coral', 'orange', 'mediumpurple']
        else:
            cmap = multidim_colormap()
            colors = [cmap(i) for i in np.linspace(0, cmap.N, len(np.unique(labels)), dtype=int)]

    label_to_int = {l: i for i, l in enumerate(sorted(set(labels)))}

    if ax is None:
        fig, ax = plt.subplots(1)
    if title:
        ax.set_title(title)
    for label in set(labels):
        idx = np.where(np.array(labels) == label)[0]
        x, y = embedding[idx].transpose()
        ax.scatter(x, y, c=colors[label_to_int[label]], label=str(label))
    n_col = int(np.ceil(len(np.unique(labels)) * 1.0 / 25))
    ax.legend(loc='center right', bbox_to_anchor=(1, 0.5), ncol=n_col)


def print_h5file(file):
    def printname(name):
        print(name)
    with h5py.File(file, 'r') as h5file:
        h5file.visit(printname)


def imgrid(images, ncols=3, fig=None, gs=None):
    N = len(images)
    ncols = int(min(ncols, N))
    nrows = int(ceil(float(N) / ncols))

    maxsize = max([max(img.size) for img in images])

    if fig is None:
        fig = plt.figure(figsize=(2*ncols, 2*nrows))
        fig.tight_layout()
    if gs is None:
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0, hspace=0)

    axes = []
    for ii in range(nrows):
        for jj in range(ncols):
            k = ii * ncols + jj
            if k == N:
                break
            ax = fig.add_subplot(gs[ii, jj])
            img = Image.new(mode='RGB', size=(maxsize, maxsize))
            img.paste(images[k])
            ax.imshow(img)
            ax.set_axis_off()
            axes.append(ax)
    return fig, axes


def plot_labelbox(ax, gt, predicted):
    c = 'g' if gt == predicted else 'r'
    ext = ax.get_images()[0].get_extent()
    rect = patches.Rectangle((ext[0], ext[3]), ext[1], ext[2], linewidth=4, edgecolor=c, facecolor='none')
    ax.add_patch(rect)


def make_video(dir, prefix, name):
    os.chdir(dir)
    if not name.endswith('.mp4'):
        name = name + '.mp4'
    command = 'ffmpeg -framerate 25 -i ' + prefix + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + name
    os.system(command)


def load_weights(weightfile, state_dict_model, prefix_file='', prefix_model=''):
    print("=> loading weights from '{}'".format(weightfile))
    try:
        pretrained_dict = torch.load(weightfile, map_location=lambda storage, loc: storage)['state_dict']
    except KeyError:
        pretrained_dict = torch.load(weightfile, map_location=lambda storage, loc: storage)
    # replace prefix in weightfile in case there is one when copying weights
    pretrained_dict = {k.replace(prefix_file, prefix_model): v for k, v in pretrained_dict.items()
                       # in case of multilabel weight file
                       if (k.replace(prefix_file, prefix_model) in state_dict_model.keys()
                           and v.shape == state_dict_model[k.replace(prefix_file, prefix_model)].shape)}  # number of classes might have changed
    # check which weights will be transferred
    for k in set(state_dict_model.keys() + pretrained_dict.keys()):
        if k in state_dict_model.keys() and k not in pretrained_dict.keys():
            print('\tWeights for "{}" were not found in weight file.'.format(k))
        elif k in pretrained_dict.keys() and k not in state_dict_model.keys():
            print('\tWeights for "{}" are not part of the used model.'.format(k))
        elif state_dict_model[k].shape != pretrained_dict[k].shape:
            print('\tShapes of "{}" are different in model ({}) and weight file ({}).'.
                  format(k, state_dict_model[k].shape, pretrained_dict[k].shape))
        else:  # everything is good
            pass

    state_dict_model.update(pretrained_dict)
    return state_dict_model


def scale_to_range(x, a=0, b=1):
    return (b-a) * (x - x.min()) / (x.max() - x.min()) + a