import os
import pandas as pd
import torch
import shutil
import h5py
# from tensorboardX import SummaryWriter
from PIL import Image
from math import ceil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import random
import stat
import time


class TBPlotter(object):
    """Log values in tensorboard interface."""
    def __init__(self, logdir='runs/tensorboard', overwrite=True):
        super(TBPlotter, self).__init__()
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        train_dir = os.path.join(logdir, 'train')
        test_dir = os.path.join(logdir, 'test')
        if overwrite:
            for rundir in [train_dir, test_dir]:
                shutil.rmtree(os.path.join(rundir), ignore_errors=True)
        self.logdir = logdir
        self.train_writer = SummaryWriter(train_dir)
        self.test_writer = SummaryWriter(test_dir)

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


class SHWriter(object):
    def __init__(self, prefix, verbose=False):
        self.prefix = prefix
        self.args = {}
        self.verbose = verbose

    def set_args(self, **kwargs):
        self.args = kwargs

    def add_args(self, **kwargs):
        self.args.update(**kwargs)

    @staticmethod
    def make_executable(filename):
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

    def write_sh(self, outfile, mode='w'):
        argstring = []
        for k, v in self.args.items():
            if not isinstance(v, str):
                if hasattr(v, '__iter__'):
                    v = ' '.join(str(vv) for vv in v)
                elif isinstance(v, bool):
                    if not v:
                        continue
                    v = ''
            argstring.append('--{} {}'.format(str(k), str(v)))
        argstring = ' '.join(argstring)

        with open(outfile, mode) as f:
            f.write('{} {}\n'.format(self.prefix, argstring))

        self.make_executable(outfile)
        if self.verbose:
            print('Saved {}.'.format(outfile))


class TexPlotter(object):
    def __init__(self, preamble_file=None, figsize=None):
        if preamble_file is not None:
            self.set_preamble(preamble_file)

        self.figsize = (3, 3.1756) if figsize is None else figsize          # height, width

    @staticmethod
    def load_preamble(preamble_file, commands_file=None):
        with open(preamble_file, 'r') as f:
            file_data = f.readlines()

        for l in file_data:
            l.replace('utf8', 'utf8x')  # Some bug Johann told be about

        if commands_file is not None:
            with open(commands_file, 'r') as f:
                file_data += f.readlines()

        return file_data

    def set_preamble(self, preamble_file='/export/home/kschwarz/Documents/Masters/Thesis/mpl_preamble.tex'):
        preamble = self.load_preamble(preamble_file)

        latex_custom_preamble = {
            "font.family": "serif",  # use serif/main font for text elements
            # "text.usetex": True,          # use inline math for ticks
            "text.usetex": False,  # use inline math for ticks
            "pgf.rcfonts": False,
            "text.latex.preamble": preamble,
            "pgf.preamble": preamble,
            "pgf.texsystem": "pdflatex",
            'hatch.linewidth': 10.,
            'hatch.color': 'w',
            'legend.fontsize': "small",
            'legend.handletextpad': 0.1
        }

        mpl.rcParams.update(latex_custom_preamble)

        return preamble

    @staticmethod
    def get_doc_lengths(path_to_log_file):
        # Fill with default values
        lengths_backup = {'columnwidth': 3.1756,
                          'linewidth': 3.1756,
                          'textwidth': 6.48955}
        try:
            with open(path_to_log_file, 'r') as log:
                lines = log.readlines()

                lengths = {}
                for l in lines:
                    if '___' in l:
                        name, length = l.split('=')
                        name = name.strip('_').lower()
                        length = length[:-3]
                        lengths[name] = float(length)

        except Exception:
            print('Could not load thesis.log file')
            lengths = lengths_backup

        if lengths == {}:
            print('Lengths was empty. Did you include "print_lengths.tex"?')
            ### PRINT_LENGTHS.tex
            # \usepackage {layouts}
            # \usepackage {xprintlen}
            # \usepackage {xparse}
            # \ExplSyntaxOn
            # % https: // tex.stackexchange.com / a / 123283 / 5764
            # \DeclareExpandableDocumentCommand { \printlengthas} {m m}
            #     { \dim_to_decimal_in_unit:nn {# 1} { 1 #2 } #2 }
            # \ExplSyntaxOff
            #
            # \message{___TEXTWIDTH___ =\printlengthas{\textwidth}{ in}}
            # \typeout{___LINEWIDTH___ =\printlengthas{\linewidth}{ in}}
            # \typeout{___COLUMNWIDTH___ =\printlengthas{\columnwidth}{ in}}

            lengths = lengths_backup

        return lengths

    def render(self, figure):
        figure.set_size_inches((self.figsize[1], self.figsize[0]))          # set width, height

    def save(self, figure, outfilename, **save_args):
        self.render(figure)
        bbox_extra_artists = np.concatenate([(ax.title, ax.legend_) for ax in figure.axes])
        bbox_extra_artists = bbox_extra_artists[bbox_extra_artists != None]
        plt.savefig(outfilename, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight', **save_args)


class TableWriter(object):
    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def write_bold(string):
        return r'\textbf{' + string + '}'

    def write_latex_table(self, outfile, mode='w', is_best=None, **latex_kwargs):
        if is_best is not None:
            str_vals = self.df.values.astype('|S64').flatten()
            is_best = is_best.flatten().astype(bool)
            str_vals[is_best] = map(self.write_bold, str_vals[is_best])
            df = pd.DataFrame(index=self.df.index, columns=self.df.columns, data=str_vals.reshape(self.df.values.shape))
        else:
            df = self.df

        with open(outfile, mode) as f:
            f.write(df.to_latex(**latex_kwargs))


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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def divide_for_iteration(number, n_iter):
    n_per_iter = [int(number / n_iter)] * n_iter
    for i in range(number % n_iter):
        n_per_iter[i] += 1
    return n_per_iter


def find_n_false_in_a_row(n_false, bool_list):
    """Returns start index of first n_false false entries."""
    counter = 0
    for idx, corr in enumerate(bool_list):
        if not corr:
            counter += 1
        else:
            counter = 0
        if counter == n_false:
            break
    if counter != n_false:
        return -1
    return idx + 1 - n_false


def find_n_false_total(n_false, bool_list):
    """Returns index of n_false-th false element."""
    counter = 0
    for idx, corr in enumerate(bool_list):
        if not corr:
            counter += 1
        if counter == n_false:
            break
    if counter != n_false:
        return -1
    return idx

"""
def find_free_filename(filename):
    ext = ""
    if "." in filename:
        idx = filename.rfind(".")
        ext = filename[idx:]
        filename = filename[:idx]
    for i in range(1000):
        testname = "{}_{:03d}{}".format(filename, i, ext)
        if not os.path.isfile(testname):
            return testname
    raise RuntimeError('No free filename found.')
"""


def find_free_filename(filename, counter=0):
    if counter > 999:
        raise RuntimeError('No free filename found.')
    if '.' in filename:  # filename has file extension
        testname = filename[::-1]   # invert filename to get extension point only
        testname = testname.replace('.', '_{:03d}.'.format(counter)[::-1], 1)
        testname = testname[::-1]   # re-invert filename
    else:
        testname = '{}_{:03d}'.format(filename, counter)
    if not os.path.isfile(testname):
        return testname
    return find_free_filename(filename, counter+1)
