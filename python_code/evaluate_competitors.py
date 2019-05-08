import os
import argparse
import numpy as np
import deepdish as dd
import torch

from initialization import Initializer
from loss import TripletSelector, select_semihard, select_random
from evaluation import RetrievalEvaluator, average_evaluation_df
import snack
from umap import UMAP
import warnings
from helpers import IndexDataset, PartiallyLabeledBatchSampler, BalancedBatchSampler


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


def get_semihard_batch_triplets(data, labels, n, weights=None, batch_size=10, batch_frac_labeled=0.7,
                                N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15):
    class TripletCreator(object):
        def __init__(self, triplet_selector,
                     data, data_labels, data_weights=None,
                     N_labels_per_minibatch=2, N_samples_per_label=5, N_pure_negatives=5):
            self.triplet_selector = triplet_selector
            self.data = data
            self.data_labels = data_labels
            self.data_weights = torch.ones(len(data_labels)) if data_weights is None else data_weights

            self.N_labels_per_minibatch = N_labels_per_minibatch
            self.N_samples_per_label = N_samples_per_label
            self.N_pure_negatives = N_pure_negatives

        def forward(self, indices):
            features = torch.tensor(self.data[indices])
            labels = torch.tensor(self.data_labels[indices])
            weights = torch.tensor(self.data_weights[indices], device=features.device)

            # compute triplet loss for minibatches
            minibatches = list(BalancedBatchSampler(labels.cpu().numpy(),
                                                    n_labels=self.N_labels_per_minibatch,
                                                    n_samples=self.N_samples_per_label,
                                                    n_pure_negatives=self.N_pure_negatives))

            triplets = []
            if len(minibatches) > 0:
                for mb in minibatches:  # compute minibatch wise triplet loss
                    _triplets, _, _, _ = triplet_selector.get_triplets(feature=features[mb],
                                                                       label=labels[mb],
                                                                       weights=weights[mb])
                    if _triplets is not None:
                        triplets.extend(np.array(mb)[_triplets.numpy()])

            else:
                warnings.warn('Did not find any positives in batch. No triplets formed.')

            return np.array(indices)[triplets]


    N_unlabeled_total = batch_size
    classweights = {l: 1. / 3 if l < 0 else 2. / 3 for l in set(labels) if
                    l != 0}  # 2/3 positive labeled, 1/3 class specific negatives
    batchsampler = PartiallyLabeledBatchSampler(labels, frac_labeled=batch_frac_labeled, batch_size=batch_size,
                                                N_unlabeled_total=N_unlabeled_total, classweights=classweights)
    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    triplet_creator = TripletCreator(triplet_selector=triplet_selector, data=data, data_labels=labels,
                                     data_weights=weights,
                                     N_labels_per_minibatch=N_labels_per_minibatch,
                                     N_samples_per_label=N_samples_per_label,
                                     N_pure_negatives=N_pure_negatives)
    triplets = []
    while len(triplets) < n:
        print('{}/{}'.format(len(triplets), n))
        for batch_idcs in batchsampler:
            triplets.extend(triplet_creator.forward(batch_idcs))

    triplets = np.stack(triplets)
    idcs = np.random.choice(range(len(triplets)), size=n, replace=False)

    return triplets[idcs]


parser = argparse.ArgumentParser(description='Train using generated label file.')

# general configurations
parser.add_argument('--heuristics', nargs='+', default=('none', 'area', 'svm'), type=str,
                    help='Used heuristics used to generate labels.')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Allow print statements in code.')

# dataset configurations
parser.add_argument('--dataset_name', type=str,
                    help='Name of the used dataset.')
parser.add_argument('--dataset_dir', default='./dataset_info', type=str,
                    help='Directory of dataset info file.')
parser.add_argument('--path_to_files', type=str, help='Path to (all) label files. '
                                                      'Expect files in subfolder named "dataset_name".')

parser.add_argument('--feature_dim', default=None, type=int,
                    help='Dimensionality of the extracted features (reduced with PCA).')
parser.add_argument('--projection_dim', default=2, type=int,
                    help='Dimensionality of the computed projection.')

# training configurations
parser.add_argument('--weight_unlabeled', default=0.3, type=float,
                    help='Weight of samples without generated labels.')
parser.add_argument('--batch_size', default=2000, type=int,
                    help='Batch size.')
parser.add_argument('--batch_frac_labeled', default=0.7, type=float,
                    help='Fraction of labeled samples used per batch.')

args = parser.parse_args()

# output configurations
args.outdir = './automated_runs/competitors'
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

args.outdir_triplets = os.path.join(args.outdir, '.triplets')
if not os.path.isdir(args.outdir_triplets):
    os.makedirs(args.outdir_triplets)

args.outdir_projections = os.path.join(args.outdir, '.projections')
if not os.path.isdir(args.outdir_projections):
    os.makedirs(args.outdir_projections)

args.outdir_eval = os.path.join(args.outdir, 'evaluation')
if not os.path.isdir(args.outdir_eval):
    os.makedirs(args.outdir_eval)


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    info = os.path.join(args.dataset_dir, 'info_{}.h5'.format(args.dataset_name))
    init = Initializer(args.dataset_name, impath=None, info_file=info, verbose=True,
                       feature_dim=args.feature_dim)
    init.initialize(dataset=False, projection=False,
                    multi_features=False,
                    is_test=args.dataset_name.endswith('_test'))
    data_dict = init.get_data_dict(normalize_features=True)

    features = data_dict['features']

    label_files = [f for f in os.listdir(os.path.join(args.path_to_files, args.dataset_name)) if f.endswith('.h5')]

    for i, label_file in enumerate(label_files):
        print('{}/{}'.format(i, len(label_files)))
        experiment_id = '{}_{}'.format(*filename_to_tuple(label_file.split('/')[-1])) + \
                        '_{:03d}'.format(run_from_filename(label_file))
        label_dict = dd.io.load(label_file)

        if not (data_dict['image_id'] == label_dict['image_id']).all():
            raise RuntimeError('Image IDs in label file do not match Image IDs in feature file.')
        if not args.feature_dim == data_dict['features'].shape[1]:
            raise RuntimeError('Feature dimension and shape of features does not match.')

        # saving options
        outpath_triplets = os.path.join(args.outdir_triplets, args.dataset_name,
                                   '{}.h5'.format(args.experiment_id))
        if not os.path.isdir(outpath_triplets):
            os.makedirs(os.path.dirname(outpath_triplets))

        outpath_projections = os.path.join(args.outdir_projections, args.dataset_name,
                                        '{}.h5'.format(args.experiment_id))
        if not os.path.isdir(outpath_projections):
            os.makedirs(os.path.dirname(outpath_projections))

        for heuristic in args.heuristics:
            # load labels and weights
            labels = label_dict['labels_{}'.format(heuristic)].astype(str)
            int_labels = get_int_labels(args.dataset_name, labels)

            weights = label_dict['weights_{}'.format(heuristic)]
            weights[np.isnan(weights)] = args.weight_unlabeled

            # create triplets
            if args.verbose:
                print('Compute triplets...')
            triplets = get_semihard_batch_triplets(data=features, labels=int_labels,
                                                   n=10000, weights=weights,
                                                   batch_size=args.batch_size, batch_frac_labeled=args.batch_frac_labeled,
                                                   N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15)

            outdict = {heuristic: triplets}
            if os.path.isfile(outpath_triplets):
                outdict.update(dd.io.load(outpath_triplets))
            dd.io.save(outpath_triplets, outdict)

            if args.verbose:
                print('Done.\nSaved {}.'.format(outpath_triplets))


            # compute the projections with the competitive methods

            outdict = {heuristic: {'SNaCK': None, 'UMAP': None}}

            # SNaCK
            # perform a grid search for contrib_cost_tsne, contrib_cost_triplets using 1000 samples
            n_tot = len(features)
            n_gs = min(2000, n_tot)
            n_triplets_gs = int((n_gs * 1.0 / n_tot) * 10000)
            samples_gs = np.random.choice(range(n_tot), size=n_gs, replace=False)
            triplets_gs = get_semihard_batch_triplets(data=features[samples_gs], labels=int_labels[samples_gs],
                                                      n=n_triplets_gs, weights=weights[samples_gs],
                                                      batch_size=args.batch_size,
                                                      batch_frac_labeled=args.batch_frac_labeled,
                                                      N_labels_per_minibatch=10, N_samples_per_label=5,
                                                      N_pure_negatives=15)
            gt_labels_gs = data_dict['info'].values.flatten()[samples_gs]
            params = [{"contrib_cost_tsne": 1000.0, "contrib_cost_triplets": 0.0},
                      {"contrib_cost_tsne": 750.0, "contrib_cost_triplets": 0.025},
                      {"contrib_cost_tsne": 500.0, "contrib_cost_triplets": 0.05},
                      {"contrib_cost_tsne": 250.0, "contrib_cost_triplets": 0.075},
                      {"contrib_cost_tsne": 0.0, "contrib_cost_triplets": 0.1}, ]

            recalls = []
            for param in params:
                projection = snack.snack_embed(
                    X_np=features[samples_gs].astype('float'),
                    triplets=triplets_gs,
                    theta=0.5, verbose=args.verbose,
                    **param
                )
                # evaluate retrieval to find best parameters
                evaluator = RetrievalEvaluator(data=projection, labels=gt_labels_gs)
                evaluator.retrieve()
                recalls.append(evaluator.compute_recall_at_p(p=0.6)[0])

            best_params = params[np.argmax(recalls)]

            projection = snack.snack_embed(
                features.astype('float'),
                triplets,
                theta=0.5, verbose=args.verbose,
                **best_params
            )
            outdict[heuristic]['SNaCK'] = projection

            # UMAP with labels
            target = int_labels.copy()
            target[target < 1] = -1
            transformer = UMAP(n_neighbors=30, n_components=2, verbose=args.verbose)

            transformer.fit(features, target)
            projection = transformer.embedding_

            outdict[heuristic]['UMAP'] = projection
            if os.path.isfile(outpath_projections):
                outdict.update(dd.io.load(outpath_projections))
            dd.io.save(outpath_projections, outdict)


            # evaluation
            outpath_eval = os.path.join(args.outdir_eval, '{}.h5'.format(args.dataset_name))
            gt_labels = data_dict['info'].values.flatten()
            p = np.arange(0.6, 1, 0.05)

            for key in ['SNaCK', 'UMAP']:
                projection = outdict[heuristic][key]

                evaluator = RetrievalEvaluator(data=projection, labels=gt_labels)
                evaluator.retrieve()
                evaluator.compute_recall_at_p(p)
                evaluator.export_results(outfilename=outpath_eval, index=args.experiment_id, mode='a',
                                         key=os.path.join(heuristic, key))

    # save averaged df
    data_dict = dd.io.load(outpath_eval)
    heuristics = ['none', 'area', 'svm']
    eval_cols = data_dict['none']['SNaCK'].columns
    names = ['SNaCK', 'UMAP']
    mode = 'w'
    for heuristic in heuristics:
        for name in names:
            df = data_dict[heuristic][name]
            df = average_evaluation_df(df)

            df.to_hdf(outpath_eval.replace('.h5', '_avg.h5'),
                      key=os.path.join(key, heuristic, name), mode=mode)
