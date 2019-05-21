import os
import argparse
import numpy as np
import deepdish as dd
import torch
import torch.nn.functional as F

from aux import write_config
from initialization import Initializer
from train import train_mapnet
from model import MapNet, mapnet_1, mapnet_2, mapnet_3, mapnet_4


def get_int_labels(dataset_name, labels, outdir='./dataset_info'):
    outfilename = 'classes_' + dataset_name.replace('_train', '.txt').replace('_test', '.txt')
    if os.path.isfile(os.path.join(outdir, outfilename)):
        label_to_int = {}
        with open(os.path.join(outdir, outfilename), 'rb') as f:
            for line in f:
                l, i = line.strip('\n').split('\t\t\t')
                label_to_int[l] = int(i)
                label_to_int['NOT___{}'.format(l)] = -int(i)
        label_to_int['None'] = 0

    else:
        pos_labels = [lbl for lbl in np.unique(labels) if (lbl != 'None' and not lbl.startswith('NOT___'))]
        label_to_int = {l: i+1 for i, l in enumerate(pos_labels)}

        with open(os.path.join(outdir, outfilename), 'wb') as f:
            for k, v in label_to_int.items():
                f.write('{}\t\t\t{}\n'.format(k, v))

        label_to_int.update({'NOT___{}'.format(l): -i - 1 for i, l in enumerate(pos_labels)})
        label_to_int['None'] = 0

    return np.array(map(lambda x: label_to_int[x], labels))


def filename_to_tuple(filename):
    numbers = filename.replace('n_per_label_', '').replace('n_corr_per_label_', '').split('_')[:2]
    try:
        numbers = tuple([int(n) for n in numbers])
    except ValueError:
        numbers = tuple([float(n.replace('-', '.')) for n in numbers])
    return numbers


def run_from_filename(filename):
    return int(filename.split('_')[-1].split('.')[0])


parser = argparse.ArgumentParser(description='Train using generated label file.')

# general configurations
parser.add_argument('--heuristics', nargs='+', default=('none', 'area', 'svm'), type=str,
                    help='Used heuristics used to generate labels.')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Allow print statements in code.')
parser.add_argument('--use_test', default=False, action='store_true',
                    help='Use testset to evaluate training on the fly.')
parser.add_argument('--use_multi_features', default=False, action='store_true',
                    help='Use feature variants for training.')
parser.add_argument('--use_pretrained', default=False, action='store_true',
                    help='Use ImageNet pretrained layers in MapNet')

# dataset configurations
parser.add_argument('--dataset_name', type=str,
                    help='Name of the used dataset.')
parser.add_argument('--dataset_dir', default='./dataset_info', type=str,
                    help='Directory of dataset info file.')
parser.add_argument('--label_file', type=str,
                    help='Path to label_file.')
parser.add_argument('--outdir', type=str, default=None,
                    help='General output directory.')

parser.add_argument('--feature_dim', default=None, type=int,
                    help='Dimensionality of the extracted features (reduced with PCA).')
parser.add_argument('--projection_dim', default=2, type=int,
                    help='Dimensionality of the computed projection.')
parser.add_argument('--n_layers', default=1, type=int,
                    help='Number of mapping layers when using pretrained MapNet.')

# training configurations
parser.add_argument('--weight_unlabeled', default=0.3, type=float,
                    help='Weight of samples without generated labels.')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate.')
parser.add_argument('--batch_size', default=2000, type=int,
                    help='Batch size.')
parser.add_argument('--batch_frac_labeled', default=0.7, type=float,
                    help='Fraction of labeled samples used per batch.')
parser.add_argument('--max_epochs', default=100, type=int,
                    help='Maximum number of epochs in training.')

args = parser.parse_args()

args.experiment_id = '{}_{}'.format(*filename_to_tuple(args.label_file.split('/')[-1])) + \
                     '_{:03d}'.format(run_from_filename(args.label_file))

# output configurations
if args.outdir is None:
    args.outdir = './automated_runs'
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

args.outdir_models = os.path.join(args.outdir, 'models', args.dataset_name)
if not os.path.isdir(args.outdir_models):
    os.makedirs(args.outdir_models)

args.logdir = os.path.join(args.outdir_models, '.log')
if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

# write config file:
args.config_dir = os.path.join(args.outdir_models, '.configs')
if not os.path.isdir(args.config_dir):
    os.makedirs(args.config_dir)
write_config(args, exp_name=os.path.join(args.config_dir, '{}'
                                         .format(args.dataset_name)))
if args.verbose:
    print('Saved {}'.format(os.path.join(args.config_dir, '{}.txt'
                                         .format(args.dataset_name))))


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    init = Initializer(args.dataset_name, impath=None, info_file=None, verbose=True,
                       feature_dim=args.feature_dim)
    # initialize files needed for training, like precomputed features
    init.initialize(dataset=False, projection=False,
                    multi_features=args.use_multi_features,
                    raw_features=args.use_pretrained,
                    is_test=args.dataset_name.endswith('_test'))
    data_dict = init.get_data_dict(normalize_features=not args.use_pretrained)
    # ^directly contains features and projections
    
    if args.use_multi_features:
        features = data_dict['multi_features']
    elif args.use_pretrained:
        features = data_dict['features_raw']
    else:
        features = data_dict['features']

    features_test, labels_test, int_labels_test = None, None, None
    if args.use_test:
        testset_name = args.dataset_name.replace('_train', '_test')
        info_file_test = os.path.join(args.dataset_dir, 'info_{}.h5'.format(testset_name))
        if not os.path.isfile(info_file_test):
            raise RuntimeError('Please provide test labels in info dataframe under:\n\t{}'.format(testset_name))
        init_test = Initializer(testset_name,
                                impath=None, info_file=info_file_test, verbose=True,
                                feature_dim=args.feature_dim)
        data_dict_test = init_test.get_data_dict(normalize_features=True)
        features_test = data_dict_test['features'] if not args.use_multi_features else data_dict_test['multi_features']
        labels_test = data_dict_test['info'].values.flatten().astype(str)

    label_dict = dd.io.load(args.label_file)

    if not (data_dict['image_id'] == label_dict['image_id']).all():
        raise RuntimeError('Image IDs in label file do not match Image IDs in feature file.')
    if not args.feature_dim == data_dict['features'].shape[1]:
        raise RuntimeError('Feature dimension and shape of features does not match.')

    for heuristic in args.heuristics:
        # saving options
        outpath = os.path.join(args.outdir_models, '{}_{}.pth.tar'.format(args.experiment_id, heuristic))
        logdir = os.path.join(args.logdir, args.experiment_id, heuristic)
        if args.use_multi_features:
            outpath = outpath.replace('.pth.tar', '_multi.pth.tar')
            logdir += '_multi'
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        # load labels and weights
        labels = label_dict['labels_{}'.format(heuristic)].astype(str)
        int_labels = get_int_labels(args.dataset_name, labels)

        if labels_test is not None:
            int_labels_test = get_int_labels(args.dataset_name, labels_test)        # use train dataset name to ensure same conversion

        weights = label_dict['weights_{}'.format(heuristic)]
        weights[np.isnan(weights)] = args.weight_unlabeled

        # train the model
        if not args.use_pretrained:
            model = MapNet(feature_dim=args.feature_dim, output_dim=args.projection_dim)
        else:
            if args.n_layers == 1:
                model = mapnet_1(pretrained=True)
            elif args.n_layers == 2:
                model = mapnet_2(pretrained=True)
            elif args.n_layers == 3:
                model = mapnet_3(pretrained=True)
            elif args.n_layers == 4:
                model = mapnet_4(pretrained=True)
            else:
                raise AttributeError('Numbers of layers not implemented.')

            # fix the reduction layer weights
            for param in model.mapping[0].parameters():
                param.requires_grad = False

            model.__delattr__('featurenet')         # delete unneccessary parameters

        if use_gpu:
            model = model.cuda()

        train_mapnet(model, features, int_labels,
                     test_features=features_test, test_labels=int_labels_test,
                     lr=args.lr, batch_size=args.batch_size, batch_frac_labeled=args.batch_frac_labeled,
                     outpath=outpath, log_dir=logdir, max_epochs=args.max_epochs,
                     use_gpu=use_gpu, verbose=True)
        # keep track of trained models
        with open(os.path.join(args.logdir, '.trained.txt'), 'a+') as f:
            f.write('{}\n'.format(args.label_file))
