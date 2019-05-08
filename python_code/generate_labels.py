import os
import argparse
import numpy as np
import deepdish as dd
from collections import Counter

from aux import write_config
from initialization import Initializer
from heuristic import simulate_user


parser = argparse.ArgumentParser(description='Simulate user labeling.')

# general configurations
parser.add_argument('--n_runs', default=1, type=int,
                    help='Number of runs to generate labels for '
                         '(produce statistics due to some randomness in labelling).')
parser.add_argument('--heuristics', nargs='+', default=('none', 'area', 'svm'), type=str,
                    help='Heuristics used to generate labels.')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Allow print statements in code.')

# dataset configurations
parser.add_argument('--dataset_name', type=str,
                    help='Name of the used dataset.')
parser.add_argument('--dataset_dir', default='./dataset_info', type=str,
                    help='Directory of dataset info file.')
parser.add_argument('--impath', default='/export/home/kschwarz/Documents/Data/Wikiart_Elgammal', type=str,
                    help='Path to images of dataset.')

parser.add_argument('--feature_dim', default=None, type=int,
                    help='Dimensionality of the extracted features (reduced with PCA).')

# heuristics configurations
parser.add_argument('--fac_corrected', default=None, type=int,
                    help='Factor of corrected to labeled samples.')
parser.add_argument('--n_selected_per_label', nargs='+', default=(5, 10, 20, 30, 50, 100, 300), type=float,
                    help='Range of values for labeled samples per label. '
                         'Integers denote absolute values, floats denote '
                         'labeled fraction of total number of samples per label.')
parser.add_argument('--n_min_clustering', default=5, type=int,
                    help='Minimal number of samples used to form a cluster.')
parser.add_argument('--n_min_sampling', default=3, type=int,
                    help='Minimal number of labeled samples per drawn cluster.')
# SVM heuristic configurations
parser.add_argument('--n_svm_iter', default=1, type=int,
                    help='Number of iterations in which to train the SVM in the SVM heuristic.')
parser.add_argument('--n_random_negatives_svm', default=100, type=int,
                    help='Number of random negatives used to train the SVM in the SVM heuristic.')
parser.add_argument('--weight_svm_random_negatives', default=0.1, type=float,
                    help='Weight of random negatives used to train the SVM in the SVM heuristic.')
parser.add_argument('--n_wrong_threshold', default=5, type=int,
                    help='Number of wrong predictions in a row to set threshold value of the SVM predictions '
                         'in the SVM heuristic.')

parser.add_argument('--weight_predictions', default=0.7, type=float,
                    help='Weight assigned to the predictions from the heuristic.')

args = parser.parse_args()

args.info_file = os.path.join(args.dataset_dir, 'info_{}.h5'.format(args.dataset_name))
if not os.path.isfile(args.info_file):
    raise RuntimeError('Please provide labels in info dataframe under:\n\t{}'.format(args.info_file))

# output configurations
args.outdir = './automated_runs'
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

args.outdir_projections = os.path.join(args.outdir, '.projections')
if not os.path.isdir(args.outdir_projections):
    os.makedirs(args.outdir_projections)

args.outdir_label_generations = os.path.join(args.outdir, 'generated_labels')
if not os.path.isdir(args.outdir_label_generations):
    os.makedirs(args.outdir_label_generations)

# write config file:
write_config(args, exp_name=os.path.join(args.outdir, '{}_fac_corrected_{}'
                                         .format(args.dataset_name, args.fac_corrected)))
if args.verbose:
    print('Saved {}'.format(os.path.join(args.outdir, '{}_fac_corrected_{}_config.txt'
                                         .format(args.dataset_name, args.fac_corrected))))

if __name__ == '__main__':
    init = Initializer(args.dataset_name, impath=args.impath, info_file=args.info_file, verbose=True,
                       feature_dim=args.feature_dim)
    init.initialize(projection=False, is_test=args.dataset_name.endswith('_test'))

    data_dict = init.get_data_dict(normalize_features=True)

    # initialize cliques
    cliques = None
    num_samples_per_clique = 10
    if 'clique_svm' in args.heuristics:
        clique_dir = './cliques'
        if not os.path.isdir(clique_dir):
            os.makedirs(clique_dir)
        clique_file = os.path.join(clique_dir, args.dataset_name + '_{}.h5'.format(num_samples_per_clique))
        if not os.path.isfile(clique_file):
            from cliquecnn_clustering.make_cliques import make_cliques
            cliques = make_cliques(features=data_dict['features'], num_samples_per_clique=num_samples_per_clique)
            clique_dict = {'image_id': data_dict['image_id'], 'cliques': cliques}
            dd.io.save(clique_file, clique_dict)
        else:
            clique_dict = dd.io.load(clique_file)
            if not np.all(clique_dict['image_id'] == data_dict['image_id']):
                raise RuntimeError('Image IDs in clique file do not match IDs in data_dict.')
            cliques = clique_dict['cliques']

    outdir = os.path.join(args.outdir_label_generations, args.dataset_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # make int labels starting counting from 1
    labelname = data_dict['info'].columns[0]
    labels = data_dict['info'][labelname].values.astype(str)
    label_to_int = {l: i + 1 for i, l in enumerate(np.unique(labels))}
    labels = map(lambda x: label_to_int[x], labels)

    # define inverse mapping from int to label
    int_label_to_label = {}
    negative_prefix = 'NOT___'
    for k, v in label_to_int.items():
        int_label_to_label[v] = k
        int_label_to_label[-v] = negative_prefix + k
    int_label_to_label[None] = None

    # # evaluate cliques
    # clique_accs = []
    # for c in cliques:
    #     count = Counter(np.array(labels)[c])
    #     main_lbl, count = sorted(count.items(), key=lambda x: x[1], reverse=True)[0]
    #     clique_accs.append(count * 1.0 / len(c))
    # print(np.mean(clique_accs))

    for n_selected_per_label in args.n_selected_per_label:
        # check if n_per_label is absolute value or fraction
        if n_selected_per_label != int(n_selected_per_label):       # fraction
            frac_n_per_label = n_selected_per_label
            frac_n_corr_per_label = args.fac_corrected * n_selected_per_label
            label_count = Counter(labels)
            n_selected_per_label = {k: int(n_selected_per_label * v) for k, v in label_count.items()}
            n_corrected_per_label = {k: int(args.fac_corrected * v) for k, v in n_selected_per_label.items()}
        else:
            n_selected_per_label = int(n_selected_per_label)
            n_corrected_per_label = int(args.fac_corrected * n_selected_per_label)
        for run in range(args.n_runs):
            outfilename = os.path.join(args.outdir_projections, args.dataset_name + '_{:03d}.h5'.format(run))
            if not os.path.isfile(outfilename):                 # compute a random initial projection for this run
                projection = init.get_projection(features=data_dict['features'], random_state=np.random.randint(10000))
                # save projection for reproduction purposes
                out_dict = {'image_id': data_dict['image_id'], 'projection': projection}
                dd.io.save(outfilename, out_dict)
            else:
                in_dict = dd.io.load(outfilename)
                if not np.all(in_dict['image_id'] == data_dict['image_id']):
                    in_dict['image_id'] = map(lambda x: x.split('/')[1], in_dict['image_id'])
                    dd.io.save(outfilename, in_dict)
                    in_dict = dd.io.load(outfilename)
                    if not np.all(in_dict['image_id'] == data_dict['image_id']):
                        raise RuntimeError('Image IDs in projection file do not match Image IDs in data dict.')
                projection = in_dict['projection']

            if isinstance(n_selected_per_label, dict):      # fraction
                outfilename = 'n_per_label_{}_n_corr_per_label_{}_{:03d}.h5'\
                    .format(frac_n_per_label, frac_n_corr_per_label, run)
            else:
                outfilename = 'n_per_label_{}_n_corr_per_label_{}_{:03d}.h5' \
                        .format(n_selected_per_label, n_corrected_per_label, run)

            if 'clique_svm' in args.heuristics:
                outfilename = os.path.join(outdir, 'weighted_clustersampling/larger_clusters', 'clique_svm_{}'.format(num_samples_per_clique), outfilename)
                if not os.path.isdir(os.path.dirname(outfilename)):
                    os.makedirs(os.path.dirname(outfilename))
            else:
                outfilename = os.path.join(outdir, outfilename)
            if os.path.isfile(outfilename):
                print('Warning: Generated label file already exists. - Skip this iteration.')
                continue

            pred_labels, pred_weights = simulate_user(features=data_dict['features'],
                                                      projections=projection, labels=labels,
                                                      heuristics=args.heuristics,
                                                      cliques=cliques,
                                                      min_per_cluster_clustering=args.n_min_clustering,
                                                      min_per_cluster_sampling=args.n_min_sampling,
                                                      n_selected_per_label=n_selected_per_label,
                                                      n_corrected_per_label=n_corrected_per_label,
                                                      n_svm_iter=args.n_svm_iter,
                                                      n_random_negatives_svm=args.n_random_negatives_svm,
                                                      n_wrong_threshold=args.n_wrong_threshold,
                                                      weights_svm_random_negatives=args.weight_svm_random_negatives,
                                                      weight_predictions=args.weight_predictions,
                                                      plot=False, verbose=args.verbose)

            # save generated labels
            out_dict = {'image_id': np.array(data_dict['image_id'], dtype=str),
                        'gt_labels': data_dict['info'][labelname].values.astype(str)}

            for h, pl, pw in zip(args.heuristics, pred_labels, pred_weights):
                pl_ = np.array(map(lambda x: int_label_to_label[x], pl), dtype=str)
                pw_ = np.array(pw, dtype=np.float32)
                out_dict.update({'labels_{}'.format(h): pl_, 'weights_{}'.format(h): pw_})

            dd.io.save(outfilename, out_dict)



