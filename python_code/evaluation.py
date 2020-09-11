import matplotlib
matplotlib.use('Agg') #use MPL without DISPLAY
import argparse
import numpy as np
import pandas as pd
import deepdish as dd
import os
import torch
import matplotlib.pyplot as plt
# import faiss
from functools import partial
from collections import Counter
from sklearn.metrics import precision_recall_curve, f1_score, auc, precision_score, recall_score, average_precision_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from itertools import product
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F


from initialization import Initializer
from model import MapNet, mapnet#_1, mapnet_2, mapnet_3, mapnet_4
# from aux import TexPlotter, TableWriter, load_weights
from train import evaluate_model

MACHINE_EPS = np.finfo(float).eps

parser = argparse.ArgumentParser(description='Evaluate trained models.')

# general configurations
parser.add_argument('--device', default=0, type=int, help='CUDA device to run evals on')

parser.add_argument('--labels', default=False, action='store_true',
                    help='Evaluate generated labels.')
parser.add_argument('--models', default=False, action='store_true',
                    help='Evaluate trained models.')
parser.add_argument('--pretraining', default=False, action='store_true',
                    help='Evaluate ImageNet pretraining.')
parser.add_argument('--n_layers', default=1, type=int, help='Number of mapping layers in pretrained MapNet.')
parser.add_argument('--pretrained_models', default=False, action='store_true',
                    help='Load pretrained mapnet structure.')

parser.add_argument('--plot', default=False, action='store_true',
                    help='Plot evaluation and save as .png .')

parser.add_argument('--dataset_name', type=str, help='Name of dataset.')
parser.add_argument('--path_to_files', type=str, help='Path to (all) label files or weight files'
                                                      'Expect files in subfolder named "dataset_name".')

parser.add_argument('--outdir', default=None, type=str, help='Output directory.')
parser.add_argument('--overwrite', default=False, action='store_true',
                    help='Overwrite existing output file.')

args = parser.parse_args()


class Evaluator(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.df = pd.DataFrame(columns=[], index=[0])

    def start_new_run(self):
        """Add a new row to the dataframe."""
        self.df.index = self.df.index + 1
        self.df.loc[0] = None
        self.df = self.df.sort_index()

    def export_results(self, outfilename, index=0, mode='w', key='df', overwrite=True):
        result_df = self.df.dropna('columns').mean().to_frame().transpose()  # in case of several runs
        result_df.index = [index]

        def path_in_dict(path, dictionary):
            d = dictionary.copy()
            for k in path.split('/'):
                try:
                    d = d[k]
                except KeyError:
                    return False
            return True

        if mode.startswith('a') and os.path.isfile(outfilename):
            filedata = dd.io.load(outfilename)
            if path_in_dict(key, filedata):
                existing_df = filedata.copy()
                for k in key.split('/'):            # get the correct dataframe from file
                    existing_df = existing_df[k]

                # create new dataframes with combined columns
                if not isinstance(existing_df.columns, pd.MultiIndex):
                    missing_levels = len(result_df.columns.levels) - 1
                    empty_levels = [''] * missing_levels
                    existing_df.columns = pd.MultiIndex.from_product([existing_df.columns, empty_levels])

                # combine columns of data frames
                combined_columns = existing_df.columns.join(result_df.columns, how='outer')
                existing_df = pd.DataFrame(index=existing_df.index, columns=combined_columns).combine_first(existing_df)
                result_df = pd.DataFrame(index=result_df.index, columns=combined_columns).combine_first(result_df)

                if overwrite and index in existing_df.index:
                    # fill NaNs in result_df with existing values
                    result_df = result_df.combine_first(existing_df)

                    # overwrite values
                    existing_df.loc[index] = result_df.loc[index]

                    result_df = existing_df
                else:
                    result_df = existing_df.append(result_df, sort=False)
        result_df.to_hdf(outfilename, key, mode=mode)


class ClusteringEvaluator(Evaluator):
    def __init__(self, data, labels, verbose=False):
        super(ClusteringEvaluator, self).__init__(verbose=verbose)
        self.data = data
        self.gt_labels = labels
        self.clustering = None
        self.pred_labels = None
        self.df = pd.DataFrame(columns=['F1', 'NMI', 'MAP'], index=[0])

    def make_kmeans_clustering(self, n_cluster):
        # kmeans = faiss.Kmeans(d=self.data.shape[1], k=n_cluster, verbose=self.verbose)
        # kmeans.train(self.data.astype(np.float32))
        # _, c_labels = kmeans.index.search(self.data.astype(np.float32), 1)
        # c_labels = c_labels.flatten()

        # Do not use faiss as it allows not statistic
        kmeans = KMeans(n_clusters=n_cluster)
        c_labels = kmeans.fit_predict(self.data)
        self.clustering = {cl: np.where(c_labels == cl)[0] for cl in np.unique(c_labels)}

    def predict_labels(self):
        if self.clustering is None:
            raise RuntimeError('No clustering predicted yet. Please call "make_kmeans_clustering" before starting.')
        self.pred_labels = np.zeros_like(self.gt_labels)
        # assign each cluster the majority gt label
        for c_idcs in self.clustering.values():
            gt_lbls = self.gt_labels[c_idcs]
            label_counter = Counter(gt_lbls)
            pred_lbl = label_counter.keys()[np.argmax(label_counter.values())]
            self.pred_labels[c_idcs] = pred_lbl

    def compute_nmi(self):
        if self.pred_labels is None:
            raise RuntimeError('No labels predicted yet. Please call "predict_labels" before starting.')
        nmi = normalized_mutual_info_score(labels_true=self.gt_labels, labels_pred=self.pred_labels)
        self.df.at[0, 'NMI'] = nmi
        return nmi

    def compute_f1_score(self, average='micro'):
        if self.pred_labels is None:
            raise RuntimeError('No labels predicted yet. Please call "predict_labels" before starting.')
        f1 = f1_score(y_true=self.gt_labels, y_pred=self.pred_labels, average=average)
        self.df.at[0, 'F1'] = f1
        return f1

    def compute_map_score(self):
        if self.clustering is None:
            raise RuntimeError('No clustering predicted yet. Please call "make_kmeans_clustering" before starting.')

        ap_scores = []
        for c_idcs in self.clustering.values():
            for sample_idx in c_idcs:
                retrieved = np.setdiff1d(c_idcs, sample_idx)
                y_true = self.gt_labels[retrieved] == self.gt_labels[sample_idx]
                y_score = np.ones_like(retrieved)                              # all samples in cluster are positives

                ap_scores.append(average_precision_score(y_true=y_true, y_score=y_score))       # nan if no positives are retrieved

        ap_scores = np.where(np.isfinite(ap_scores), ap_scores, 0)      # replace nans
        map_score = np.mean(ap_scores)
        self.df.at[0, 'MAP'] = map_score
        return map_score


class RetrievalEvaluator(Evaluator):
    def __init__(self, data, labels, verbose=False):
        super(RetrievalEvaluator, self).__init__(verbose=verbose)
        self.data = data
        self.gt_labels = labels
        self.nn = None
        self.df = pd.DataFrame(columns=pd.MultiIndex.from_product([['precision', 'recall'], ['']]), index=[0])

    def retrieve(self, k=None):
        if self.verbose:
            print('Retrieve neighbors...')
        if k is None:
            k = self.data.shape[0] - 1      # maximum number of neighbors
        index = faiss.IndexFlatL2(self.data.shape[1])
        index.add(self.data.astype(np.float32))
        self.nn = index.search(self.data.astype(np.float32), k=k+1)
        if self.verbose:
            print('Done.')

    def compute_recall_at_k(self, k):
        assert k < self.nn[1].shape[1], 'k cannot be larger than number of retrieved neighbors'
        gt_labels = self.gt_labels[self.nn[1][:, :k+1]]
        binary_gt = gt_labels[:, 1:] == gt_labels[:, 0].reshape(len(gt_labels), 1)

        occur_dict = Counter(self.gt_labels)
        recalls = binary_gt.sum(axis=1) * 1.0 / np.array(map(lambda x: occur_dict[x], gt_labels[:, 0]))
        return recalls.mean()

    def compute_precision_at_k(self, k):
        assert k < self.nn[1].shape[1], 'k cannot be larger than number of retrieved neighbors'
        gt_labels = self.gt_labels[self.nn[1][:, :k+1]]
        binary_gt = gt_labels[:, 1:] == gt_labels[:, 0].reshape(len(gt_labels), 1)

        precisions = binary_gt.sum(axis=1) * 1.0 / k
        return precisions.mean()

    @staticmethod
    def interpolate_prc(prc):
        # use p_interp = max(p_r'), r' >= r
        precision, recall, _ = prc
        p_interp = np.maximum.accumulate(precision)

        return p_interp, recall, prc[2]

    def compute_recall_at_p(self, p):
        if self.nn is None:
            raise RuntimeError('No retrievals performed yet. Please call "retrieve" before starting.')

        # scores according to neighbor position
        scores = np.arange(len(self.data) - 1, 0, -1).reshape(1, -1).repeat(len(self.data), axis=0)

        gt_labels = self.gt_labels[self.nn[1]]
        binary_gt = gt_labels[:, 1:] == gt_labels[:, 0].reshape(len(gt_labels), 1)

        # compute precision recall curve for each sample
        pr_curves = map(lambda y_true, probas_pred: precision_recall_curve(y_true, probas_pred),
                        binary_gt, scores)

        pr_curves = map(self.interpolate_prc, pr_curves)

        # compute 11-point interpolated average recall
        def interpolate_recall(prc, p=np.arange(0.0, 1.1, 0.1)):
            precision, recall, _ = prc
            r_interp = np.interp(x=p, xp=precision, fp=recall)
            return r_interp

        r_interp = np.stack(map(partial(interpolate_recall, p=p), pr_curves))
        recall = np.mean(r_interp, axis=0)

        if not hasattr(p, '__iter__'):
            p = [p]
            recall = [recall]
        index = pd.MultiIndex.from_product([['recall'], ['p={}'.format(pp) for pp in p]])
        index = self.df.columns.join(index, how='outer')

        # make new df and copy values of old df
        self.df = pd.DataFrame(columns=index, index=self.df.index).combine_first(self.df)
        for rec, pp in zip(recall, p):
            self.df.loc[0, ('recall', 'p={}'.format(pp))] = rec

        return recall


def filename_to_tuple(filename):
    numbers = filename.replace('n_per_label_', '').replace('n_corr_per_label_', '').split('_')[:2]
    try:
        numbers = tuple([int(n) for n in numbers])
    except ValueError:
        numbers = tuple([float(n.replace('-', '.')) for n in numbers])
    return numbers


def evaluate_pretraining(n_layers=1):
    outdir = './pretraining/evaluation'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # valset has 50 images per class
    impath = '/net/hci-storage02/groupfolders/compvis/bbrattol/ImageNet'
    valdir = os.path.join(impath, 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    labels = np.array(map(lambda x: x[1], val_dataset.imgs))
    ids = np.array(map(lambda x: x[0].split('/')[-1].replace('.JPEG', ''), val_dataset.imgs))        # 0 is reserved as negative label

    feature_file = './features/ImageNet_label_test.h5'
    feature_key = 'features_mapped_{}_layers'.format(n_layers)
    if not os.path.isfile(feature_file) or feature_key not in dd.io.load(feature_file).keys():
        if n_layers in {0,1,2,3,4}:
            model = mapnet(n_layers, pretrained=True, new_pretrain=True)
        else:
            raise AttributeError('Number of layers has to be [1,2,3,4].')

        if os.path.isfile(feature_file):
            data_dict = dd.io.load(feature_file)
            if not np.all(data_dict['image_id'] == ids):
                raise RuntimeError('IDs in feature file and dataset do not match.')
        else:
            data_dict = {}

        model = model.cuda()
        model.eval()

        def model_fn(x):
            fts_raw = model.featurenet(x)
            fts_mapped = model.mapping(F.relu(fts_raw))

            return fts_raw, fts_mapped

        fts = evaluate_model(model_fn=model_fn, data=val_dataset, batch_size=16, verbose=True)
        data_dict.update({'image_id': ids, 'features_raw': fts[0].numpy(), feature_key: fts[1].numpy()})

        dd.io.save(feature_file, data_dict)

    else:
        data_dict = dd.io.load(feature_file)

    outfile = os.path.join(outdir, 'precision_recall_features.h5')
    outdict = {}
    if not os.path.isfile(outfile):
        for fts, key in zip([data_dict['features_raw'], data_dict[feature_key]],
                            ['features_raw', feature_key]):
            # normalize features
            fts = fts / np.linalg.norm(fts, axis=1, keepdims=True)
            evaluator = RetrievalEvaluator(data=fts, labels=labels, verbose=True)
            k = 50            # valset has 50 images per class
            evaluator.retrieve(k)
            # compute precision and recall
            precision = evaluator.compute_precision_at_k(k)
            recall = evaluator.compute_recall_at_k(k)
            outdict[key] = {'precision': precision, 'recall': recall}
        dd.io.save(outfile, outdict)

    elif not feature_key in dd.io.load(outfile).keys():
        fts = data_dict[feature_key]
        fts = fts / np.linalg.norm(fts, axis=1, keepdims=True)
        key = feature_key
        evaluator = RetrievalEvaluator(data=fts, labels=labels, verbose=True)
        k = 50            # valset has 50 images per class
        evaluator.retrieve(k)
        # compute precision and recall
        precision = evaluator.compute_precision_at_k(k)
        recall = evaluator.compute_recall_at_k(k)
        outdict[key] = {'precision': precision, 'recall': recall}
        dd.io.save(outfile, outdict)
    else:
        print('Evaluation exists - Do not evaluate.')


def evaluate_labels(data_dir, outdir='./evaluation', outfilename=None, average='micro',
                    plot=False, overwrite=False):
    if not os.path.isdir(data_dir):
        raise AttributeError('data_dir not found.')

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if outfilename is None:
        outfilename = '{}.h5'.format(data_dir.split('/')[-1])

    methods = ['clique_svm', 'svm']
    # methods = ['none', 'area', 'svm']
    config_cols = ['n_per_label', 'n_corr_per_label']
    eval_cols = ['F1']

    if os.path.isfile(os.path.join(outdir, outfilename)) and not overwrite:
        print('Load {}.'.format(os.path.join(outdir, outfilename)))
        outdf_avg = dd.io.load(os.path.join(outdir, outfilename))['df']

    else:
        label_files = [f for f in os.listdir(data_dir) if (not f.startswith('.')
                                                           and os.path.isfile(os.path.join(data_dir, f)))]

        outdf = pd.DataFrame(columns=pd.MultiIndex.from_product([eval_cols, methods]), index=label_files)
        for col in config_cols:
            outdf[col] = None

        for f in label_files:
            outdf.loc[f, config_cols] = filename_to_tuple(f)
            data = dd.io.load(os.path.join(data_dir, f))
            gt_labels = np.array(data['gt_labels'], dtype=str)

            # # plot labeled and generated samples
            # projection = dd.io.load('/export/home/kschwarz/Documents/Masters/WebInterface/python_code/automated_runs/.projections/STL_label_train_000.h5')['projection']
            # lbl_none = data['labels_none']
            # lbl_clique_svm = data['labels_clique_svm']
            # wght_clique_svm = data['weights_clique_svm']
            # for lbl in set(gt_labels):
            #     fig, ax = plt.subplots(1)
            #     idcs = np.where(gt_labels == lbl)[0]
            #     ax.scatter(projection[:, 0], projection[:, 1], c='grey', alpha=0.3)
            #     # class members
            #     ax.scatter(projection[idcs, 0], projection[idcs, 1], facecolors='none', edgecolors='b')
            #
            #     # training
            #     idcs = np.where(lbl_none == lbl)[0]
            #     ax.scatter(projection[idcs, 0], projection[idcs, 1], facecolors='b', edgecolors='none')
            #
            #     pred_idcs = np.where(
            #         (lbl_clique_svm == lbl) & (wght_clique_svm == 0.7)
            #     )[0]
            #     # true positives
            #     idcs = pred_idcs[np.where(gt_labels[pred_idcs] == lbl)[0]]
            #     ax.scatter(projection[idcs, 0], projection[idcs, 1], facecolors='g', edgecolors='none')
            #
            #     # false positives
            #     idcs = pred_idcs[np.where(gt_labels[pred_idcs] != lbl)[0]]
            #     ax.scatter(projection[idcs, 0], projection[idcs, 1], facecolors='r', edgecolors='none')

            for method in methods:
                mkey = 'labels_{}'.format(method)
                if mkey not in data.keys():
                    print('Did not find method {} in file {}.'.format(method, f))
                    continue
                pred_labels = np.array(data[mkey], dtype=str)
                #FIXME: WHICH IS CORRECT???
                # outdf.loc[f, ('F1', method)] = f1_score(y_true=gt_labels, y_pred=pred_labels,
                #                                         average=average)

                f1s = []
                for lbl in set(gt_labels):
                    y_pred = (pred_labels == lbl).astype(int)
                    y_true = (gt_labels == lbl).astype(int)
                    w = sum(y_true) * 1.0 / len(y_pred)            # compute classweighted mean
                    f1 = f1_score(y_true=y_true, y_pred=y_pred)
                    f1s.append(w * f1)
                outdf.loc[f, ('F1', method)] = np.sum(f1s)

        config_ids = map(lambda x: str(filename_to_tuple(x)), label_files)
        outdf_avg = pd.DataFrame(index=np.unique(config_ids),
                                 columns=pd.MultiIndex.from_product([eval_cols, methods, ('mean', 'std')]))
        for col in config_cols:
            outdf_avg[col] = None
        for conf in np.unique(config_ids):
            idcs = np.where(np.array(config_ids) == conf)[0]
            outdf_avg.loc[conf, config_cols] = outdf.iloc[idcs[0]][config_cols].values
            outdf_avg.loc[conf, (eval_cols, methods, 'mean')] = outdf.iloc[idcs][eval_cols].mean().values
            outdf_avg.loc[conf, (eval_cols, methods, 'std')] = outdf.iloc[idcs][eval_cols].std().values

        outdf_avg.to_hdf(os.path.join(outdir, outfilename), 'df')

    if plot:
        plotter = TexPlotter()
        plotter.figsize = (3.5, plotter.get_doc_lengths('')['textwidth'])
        cmap = plt.cm.tab10

        def strtuple_to_ints(strtuple):
            try:
                vals = [int(s) for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
            except ValueError:
                vals = [float(s.replace('-', '.'))
                        for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
            return vals

        for ec in eval_cols:
            fig, ax = plt.subplots(1)
            ax.set_title(ec)
            x = np.array(map(lambda x: np.sum(strtuple_to_ints(x)), outdf_avg.index.values))
            sort_idx = np.argsort(x)
            for i, method in enumerate(methods):
                ax.errorbar(x[sort_idx], y=outdf_avg[(ec, method, 'mean')].values[sort_idx],
                            yerr=outdf_avg[(ec, method, 'std')].values[sort_idx],
                            label=method, c=cmap(i), marker='.', markersize=10,
                                elinewidth=0.5, capsize=3, capthick=0.5)
            plt.legend()
            plotter.render(fig)
            plt.show(block=False)
            plotter.save(fig, outfilename=os.path.join(outdir, outfilename.replace('.h5', '_{}.png'.format(ec))), dpi=200)
            plt.close()


def evaluate_labels_quantitative(data_dir, outdir='./evaluation', outfilename=None, average='micro',
                                 plot=False, overwrite=False):
    if not os.path.isdir(data_dir):
        raise AttributeError('data_dir not found.')

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if outfilename is None:
        outfilename = '{}.h5'.format(data_dir.split('/')[-1])

    methods = ['clique_svm', 'svm']
    # methods = ['none', 'area', 'svm']
    config_cols = ['n_per_label', 'n_corr_per_label']
    eval_cols = ['prec_train_data', 'prec_test_data', 'rec_train_data', 'rec_test_data']

    def evaluate():
        label_files = [f for f in os.listdir(data_dir) if (not f.startswith('.')
                                                           and os.path.isfile(os.path.join(data_dir, f)))]

        outdf = pd.DataFrame(columns=pd.MultiIndex.from_product([eval_cols, methods]), index=label_files)
        for col in config_cols:
            outdf[col] = None

        for f in label_files:
            outdf.loc[f, config_cols] = filename_to_tuple(f)
            data = dd.io.load(os.path.join(data_dir, f))
            gt_labels = np.array(data['gt_labels'], dtype=str)
            for method in methods:
                mkey = 'labels_{}'.format(method)
                if mkey not in data.keys():
                    print('Did not find method {} in file {}.'.format(method, f))
                    continue
                pred_labels = np.array(data[mkey], dtype=str)
                pred_weights = np.array(data['weights_{}'.format(method)])

                idcs_labeled = np.where((pred_weights == 1) &
                                        map(lambda x: not x.startswith('NOT___'), pred_labels))[0]
                idcs_pred = np.where(pred_weights == 0.7)[0]

                for name, idcs in zip(['train_data', 'test_data'], [idcs_labeled, idcs_pred]):
                    if len(idcs) > 0:
                        recalls = []
                        precisions = []
                        for lbl in set(gt_labels):
                            lbl_idcs = np.where(pred_labels[idcs] == lbl)[0]
                            class_weight = sum(gt_labels == lbl) *1.0 / len(gt_labels)
                            pred = np.zeros(len(pred_labels))
                            pred[idcs[lbl_idcs]] = 1
                            gt = gt_labels == lbl
                            recalls.append(recall_score(gt, pred) * class_weight) # class weighted
                            precisions.append(precision_score(gt, pred) * class_weight)  # class weighted
                        recall = np.sum(recalls)
                        precision = np.sum(precisions)
                    else:
                        recall = float('nan')
                        precision = float('nan')

                    outdf.loc[f, ('prec_{}'.format(name), method)] = precision
                    outdf.loc[f, ('rec_{}'.format(name), method)] = recall

        config_ids = map(lambda x: str(filename_to_tuple(x)), label_files)
        outdf_avg = pd.DataFrame(index=np.unique(config_ids),
                                 columns=pd.MultiIndex.from_product([eval_cols, methods, ('mean', 'std')]))
        for col in config_cols:
            outdf_avg[col] = None
        for conf in np.unique(config_ids):
            idcs = np.where(np.array(config_ids) == conf)[0]
            outdf_avg.loc[conf, config_cols] = outdf.iloc[idcs[0]][config_cols].values
            outdf_avg.loc[conf, (eval_cols, methods, 'mean')] = outdf.iloc[idcs][eval_cols].mean().values
            outdf_avg.loc[conf, (eval_cols, methods, 'std')] = outdf.iloc[idcs][eval_cols].std().values

        return outdf_avg

    if os.path.isfile(os.path.join(outdir, outfilename)) and not overwrite:
        print('Load {}.'.format(os.path.join(outdir, outfilename)))
        outdf_avg = dd.io.load(os.path.join(outdir, outfilename))['df']
        if not all([c in outdf_avg.columns.levels[0] for c in eval_cols]):
            df = evaluate()
            outdf_avg = outdf_avg.combine_first(df)
            outdf_avg.to_hdf(os.path.join(outdir, outfilename), 'df')
    else:
        outdf_avg = evaluate()
        outdf_avg.to_hdf(os.path.join(outdir, outfilename), 'df')

    if plot:
        plotter = TexPlotter()
        plotter.figsize = (3.5, plotter.get_doc_lengths('')['textwidth'])
        cmap = plt.cm.tab10

        def strtuple_to_ints(strtuple):
            try:
                vals = [int(s) for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
            except ValueError:
                vals = [float(s.replace('-', '.'))
                        for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
            return vals

        for ec in eval_cols:
            fig, ax = plt.subplots(1)
            ax.set_title(ec)
            x = np.array(map(lambda x: np.sum(strtuple_to_ints(x)), outdf_avg.index.values))
            sort_idx = np.argsort(x)
            for i, method in enumerate(methods):
                ax.errorbar(x[sort_idx], y=outdf_avg[(ec, method, 'mean')].values[sort_idx],
                            yerr=outdf_avg[(ec, method, 'std')].values[sort_idx],
                            label=method, c=cmap(i), marker='.', markersize=10,
                                elinewidth=0.5, capsize=3, capthick=0.5)
            plt.legend()
            plotter.render(fig)
            plt.show(block=False)
            plotter.save(fig, outfilename=os.path.join(outdir, outfilename.replace('.h5', '_{}.png'.format(ec))), dpi=200)
            plt.close()


def heuristic_from_weightfile(weightfilename):
    if 'clique_svm' in weightfilename.split('/')[-1]:
        return 'clique_svm'
    return weightfilename.split('_')[-1].split('.')[0]


def make_baseline(dataset_name, dataset_dir='./dataset_info', outdir='./evaluation',
                  feature_dim=None, mode='w', verbose=False):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    info_file = os.path.join(dataset_dir, 'info_{}.h5'.format(dataset_name))
    init = Initializer(dataset_name, info_file=info_file, feature_dim=feature_dim, verbose=verbose)
    init.initialize(dataset=False, features=True, projection=True, is_test=dataset_name.endswith('test'))
    data_dict = init.get_data_dict(normalize_features=True)

    # evaluation
    gt_labels = data_dict['info'].values.flatten()
    n_cluster = len(np.unique(gt_labels))

    for name, data in zip(['features', 'projection'], [data_dict['features'], data_dict['projection']]):
        key = os.path.join('baseline', name)
        evaluator = RetrievalEvaluator(data=data, labels=gt_labels)
        evaluator.retrieve()
        p = np.arange(0.6, 1, 0.05)
        evaluator.compute_recall_at_p(p=p)

        # evaluator = ClusteringEvaluator(data=data, labels=gt_labels)
        # evaluator.make_kmeans_clustering(n_cluster=n_cluster)
        # evaluator.predict_labels()
        # evaluator.compute_nmi()
        # evaluator.compute_f1_score(average='micro')
        evaluator.export_results(outfilename=os.path.join(outdir, dataset_name + '.h5'),
                                 index='baseline',
                                 mode=mode, key=key)
        mode = 'a'


def evaluate_training(dataset_name, weight_file, dataset_dir='./dataset_info', outdir='./evaluation',
                      n_stat_runs=1, pretraining=False, n_layers=1,
                      feature_dim=None, projection_dim=2,
                      batch_size=64, mode='w', verbose=False):
    use_gpu = torch.cuda.is_available()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    info_file = os.path.join(dataset_dir, 'info_{}.h5'.format(dataset_name))
    init = Initializer(dataset_name, info_file=info_file, feature_dim=feature_dim, verbose=verbose)
    data_dict = init.get_data_dict(normalize_features=not pretraining)

    if pretraining:
        if args.n_layers in {0,1,2,3,4}:
            model = mapnet(args.n_layers, pretrained=False)
    else:
        model = MapNet(feature_dim=data_dict['features'].shape[1], output_dim=projection_dim)
    if use_gpu:
        model = model.cuda()
    if hasattr(model, "featurenet"):
        delattr(model, "featurenet")

    # load weights
    best_weights = load_weights(weight_file, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()

    input_fts = data_dict['features'] if not pretraining else data_dict['features_raw']
    mapped_features = evaluate_model(model.mapping, input_fts,
                                     batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features = mapped_features.numpy()
    # normalize the features
    mapped_features /= np.linalg.norm(mapped_features, axis=1, keepdims=True)

    projection = evaluate_model(model.embedder, mapped_features,
                                batch_size=batch_size, use_gpu=use_gpu, verbose=verbose).numpy()

    # evaluation
    gt_labels = data_dict['info'].values.flatten()
    n_cluster = len(np.unique(gt_labels))

    int_gt = map(lambda x: {l: i for i, l in enumerate(np.unique(gt_labels))}[x], gt_labels)
    plt.scatter(projection[:, 0], projection[:, 1], c=int_gt, cmap=plt.cm.tab20)

    for name, data in zip(['features', 'projection'], [mapped_features, projection]):
        heuristic = heuristic_from_weightfile(weight_file)
        if heuristic == 'test':
            heuristic = os.path.join('test', weight_file.split('_')[-2])
        key = os.path.join(heuristic, name)
        evaluator = RetrievalEvaluator(data=data, labels=gt_labels)
        evaluator.retrieve()
        p = np.arange(0.6, 1, 0.05)
        evaluator.compute_recall_at_p(p=p)
        evaluator = ClusteringEvaluator(data=data, labels=gt_labels)
        for n in range(n_stat_runs):
            if n > 0:
                evaluator.start_new_run()
            evaluator.make_kmeans_clustering(n_cluster=n_cluster)
            evaluator.predict_labels()
            evaluator.compute_nmi()
            evaluator.compute_f1_score(average='micro')
        evaluator.export_results(outfilename=os.path.join(outdir, dataset_name + '.h5'),
                                 index=weight_file.split('/')[-1].split('.pth.tar')[0],
                                 mode=mode, key=key)
        mode = 'a'


def evaluate_training_split_train_test_retrieval\
                (dataset_name, weight_file, dataset_dir='./dataset_info', outdir='./evaluation',
                      n_stat_runs=1,
                      feature_dim=None, projection_dim=2,
                      batch_size=64, mode='w', verbose=False):
    use_gpu = torch.cuda.is_available()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    info_file = os.path.join(dataset_dir, 'info_{}.h5'.format(dataset_name))
    init = Initializer(dataset_name, info_file=info_file, feature_dim=feature_dim, verbose=verbose)
    data_dict = init.get_data_dict(normalize_features=True)

    label_file_dir = os.path.join('./automated_runs/generated_labels', dataset_name, 'weighted_clustersampling/clique_svm_10')
    label_file = 'n_per_label_{}_n_corr_per_label_{}_{}.h5'.format(*weight_file.split('/')[-1].split('_')[:3])
    label_dict = dd.io.load(os.path.join(label_file_dir, label_file))

    if not np.all(label_dict['image_id'] == data_dict['image_id']):
        raise RuntimeError('Image IDs in label file do not match Image IDs in info/feature file.')

    method = weight_file.split('/')[-1].split('_')[3].replace('.pth.tar', '')
    if method == 'clique':
        method = 'clique_svm'
    label_file_labels = label_dict['labels_{}'.format(method)]
    label_file_weights = label_dict['weights_{}'.format(method)]
    idcs_labeled_train = np.where((label_file_weights == 1) &
                                map(lambda x: not x.startswith('NOT___'), label_file_labels))[0]

    n_labeled = float(weight_file.split('/')[-1].split('_')[0])
    if n_labeled < 1:       # frac
        pass
    else:
        assert len(idcs_labeled_train) ==\
               len(np.unique(data_dict['info'].values)) * int(n_labeled),\
            'Number of positive labeled samples in label file do not match weightfilename.'
    # idcs_labeled_pred = np.where(label_file_weights == 0.7)[0]
    # idcs_labeled_test = np.setdiff1d(range(len(label_file_labels)), np.union1d(idcs_labeled_train, idcs_labeled_pred))
    idcs_labeled_test = np.setdiff1d(range(len(label_file_labels)), idcs_labeled_train)

    model = MapNet(feature_dim=data_dict['features'].shape[1], output_dim=projection_dim)
    if use_gpu:
        model = model.cuda()

    # load weights
    best_weights = load_weights(weight_file, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    mapped_features = evaluate_model(model.mapping, data_dict['features'],
                                     batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features = mapped_features.numpy()
    # normalize the features
    mapped_features /= np.linalg.norm(mapped_features, axis=1, keepdims=True)

    projection = evaluate_model(model.embedder, mapped_features,
                                batch_size=batch_size, use_gpu=use_gpu, verbose=verbose).numpy()

    # evaluation
    gt_labels = data_dict['info'].values.flatten()
    n_cluster = len(np.unique(gt_labels))

    for name, data in zip(['features', 'projection'], [mapped_features, projection]):
        heuristic = heuristic_from_weightfile(weight_file)
        if heuristic == 'test':
            heuristic = os.path.join('test', weight_file.split('_')[-2])
        for idcs, split in zip([idcs_labeled_train, idcs_labeled_test], ['train', 'test']):
            if len(idcs) == 0:
                continue
            key = os.path.join(heuristic, name, split)
            evaluator = RetrievalEvaluator(data=data[idcs], labels=gt_labels[idcs])
            evaluator.retrieve()
            p = np.arange(0.6, 1, 0.05)
            evaluator.compute_recall_at_p(p=p)
            # evaluator = ClusteringEvaluator(data=data, labels=gt_labels)
            # for n in range(n_stat_runs):
            #     if n > 0:
            #         evaluator.start_new_run()
            #     evaluator.make_kmeans_clustering(n_cluster=n_cluster)
            #     evaluator.predict_labels()
            #     evaluator.compute_nmi()
            #     evaluator.compute_f1_score(average='micro')
            evaluator.export_results(outfilename=os.path.join(outdir, dataset_name + '_splitstt.h5'),
                                     index=weight_file.split('/')[-1].split('.pth.tar')[0],
                                     mode=mode, key=key)
            mode = 'a'


def average_evaluation_df(df):
    index_info = map(lambda x: '_'.join(x.split('_')[:2]), df.index.values)
    cols = pd.MultiIndex.from_tuples([p[0] + (p[1], ) for p in product(df.columns.tolist(), ['mean', 'std'])])
    avg_df = pd.DataFrame(index=np.unique(index_info), columns=cols)

    for key in np.unique(index_info):
        valid = [idx.startswith(key) for idx in df.index.values]
        for col in df.columns.tolist():
            avg_df.at[key, col + ('mean',)] = df[valid][col].mean()
            avg_df.at[key, col + ('std',)] = df[valid][col].std()

    return avg_df


def plot_model_evaluation(dataset_name, outdir='./automated_runs/models/evaluation'):
    outfile = os.path.join(outdir, dataset_name + '_avg.h5')
    data_dict = dd.io.load(outfile)

    plotter = TexPlotter()
    plotter.figsize = (3.5, plotter.get_doc_lengths('')['textwidth'])
    cmap = plt.cm.tab10

    for data_dict_, key in zip([data_dict, ], ('', )):
        heuristics = ['svm', 'clique_svm']#['none', 'area', 'svm']#, 'clique_svm'] #
        try:
            baseline = data_dict['baseline']
        except KeyError:
            baseline = None
        for name in ['features', 'projection']:
            df = data_dict_[heuristics[0]][name]
            eval_cols = set(pd.MultiIndex(levels=df.columns.levels[:-1],
                                      labels=df.columns.labels[:-1],
                                      names=df.columns.names[:-1]).tolist())             # last level is mean, std

            for ec in eval_cols:
                fig, ax = plt.subplots(1)
                ax.grid(linewidth=0.3)
                ax.set_title(' '.join(ec) if not 'recall' in ec else '@'.join(ec))
                for i, heuristic in enumerate(heuristics):
                    if heuristic not in data_dict.keys():
                        continue
                    df = data_dict_[heuristic][name][ec]
                    x = np.array(map(lambda x: filename_to_tuple(x)[0], df.index.values))
                    sort_idx = np.argsort(x)

                    x = x[sort_idx][:-1]
                    y = df['mean'].values[sort_idx][:-1]
                    y_err = df['std'].values[sort_idx][:-1]

                    if 'recall' in ec and x.dtype == int:      # convert number to labeled samples
                        info_file = os.path.join('./dataset_info', 'info_{}.h5'.format(dataset_name))
                        info = dd.io.load(info_file)['df']
                        n_images = len(info)
                        n_classes = len(np.unique(info.values.flatten()))

                        y = y * n_images / n_classes
                        y_err = y_err * n_images / n_classes

                    ax.errorbar(x=x, y=y,
                                yerr=y_err,
                                label=heuristic, c=cmap(i), marker='.', markersize=10,
                                elinewidth=0.5, capsize=3, capthick=0.5)

                ax.set_xlabel('# labeled per class')
                if 'recall' in ec and x.dtype == int:
                    ax.set_ylabel('# retrieved per class')
                else:
                    ax.set_ylabel(' '.join(ec))

                # plot baseline
                if baseline is not None and ec in baseline[name].keys():
                    y = baseline[name][ec].values
                    xlim = ax.get_xlim()
                    if 'recall' in ec and not args.dataset_name.endswith('test'):
                        ax.plot(xlim, xlim, marker='', linestyle='--', c='grey')
                    else:
                        ax.plot(xlim, np.repeat(y, 2), marker='', linestyle='--', c='grey')

                plt.legend()
                plotter.render(fig)
                plt.show(block=False)
                if key == 'test':
                    outfilename = outfile.replace('_avg.h5', '_{}_test_{}.png'.format(name, '-'.join(ec)))
                else:
                    outfilename = outfile.replace('_avg.h5', '_{}_{}.png'.format(name, '-'.join(ec)))
                plotter.save(fig, outfilename=outfilename, dpi=200)
                plt.close()


def plot_model_evaluation_splits(dataset_name, outdir='./automated_runs/models/evaluation'):
    outfile = os.path.join(outdir, dataset_name + '_splitstt_avg.h5')
    data_dict = dd.io.load(outfile)

    plotter = TexPlotter()
    plotter.figsize = (3.5, plotter.get_doc_lengths('')['textwidth'])
    cmap = plt.cm.tab10

    heuristics = ['none', 'area', 'svm', 'clique_svm']
    splits = ['train', 'pred', 'test']
    for data_dict_, key in zip([data_dict, ], ('', )):
        for name in ['features', 'projection']:
            df = data_dict_[heuristics[0]][name][splits[0]]
            eval_cols = set(pd.MultiIndex(levels=df.columns.levels[:-1],
                                      labels=df.columns.labels[:-1],
                                      names=df.columns.names[:-1]).tolist())             # last level is mean, std
            for ec in eval_cols:
                fig, ax = plt.subplots(1)
                ax.grid(linewidth=0.3)
                ax.set_title(' '.join(ec) if not 'recall' in ec else '@'.join(ec))
                for i, heuristic in enumerate(heuristics):
                    # for split, linestyle in zip(splits, ['-', 'dotted', '--']):
                    for split, linestyle in zip(['test'], ['-']):
                        if split not in data_dict[heuristic][name].keys():
                            continue
                        df = data_dict_[heuristic][name][split][ec]
                        x = np.array(map(lambda x: filename_to_tuple(x)[0], df.index.values))
                        sort_idx = np.argsort(x)

                        x = x[sort_idx]
                        y = df['mean'].values[sort_idx]
                        y_err = df['std'].values[sort_idx]

                        if 'recall' in ec and x.dtype == int:      # convert number to labeled samples
                            info_file = os.path.join('./dataset_info', 'info_{}.h5'.format(dataset_name))
                            info = dd.io.load(info_file)['df']
                            n_images = len(info)
                            n_classes = len(np.unique(info.values.flatten()))

                            y = y * n_images / n_classes
                            y_err = y_err * n_images / n_classes

                        ax.errorbar(x=x, y=y,
                                    yerr=y_err, linestyle=linestyle,
                                    label='{} {}'.format(heuristic, split), c=cmap(i), marker='.', markersize=10,
                                    elinewidth=0.5, capsize=3, capthick=0.5)

                if 'recall' in ec and x.dtype == int:
                    ax.set_ylabel('# retrieved per class')
                    ax.set_xlabel('# labeled per class')
                else:
                    ax.set_xlabel('frac labeled per class')
                    ax.set_ylabel(' '.join(ec))

                # plot baseline
                # y = baseline[name][ec].values
                # xlim = ax.get_xlim()
                # if 'recall' in ec:
                #     ax.plot(xlim, xlim, marker='', linestyle='--', c='grey')
                # else:
                #     ax.plot(xlim, np.repeat(y, 2), marker='', linestyle='--', c='grey')

                plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
                plotter.render(fig)
                plt.show(block=False)
                if key == 'test':
                    outfilename = outfile.replace('_avg.h5', '_{}_test_{}.png'.format(name, '-'.join(ec)))
                else:
                    outfilename = outfile.replace('_avg.h5', '_{}_{}.png'.format(name, '-'.join(ec)))
                plotter.save(fig, outfilename=outfilename, dpi=200)
                plt.close()


def write_table_model_evaluation(dataset_names):
    outdir = './automated_runs/models/evaluation'
    outfile = os.path.join(outdir, 'evaluation_table.tex')

    evalfile = os.path.join(outdir, dataset_names[0] + '_avg.h5')
    data_dict = dd.io.load(evalfile)

    # format dict into big dataframe
    methods = ['none', 'area', 'svm']
    names = ['features', 'projection']
    eval_cols = ['F1']
    avg = ['mean']
    ids = np.array(map(lambda x: sum([int(xx) for xx in x.split('_')]), data_dict['none']['features'].index.values))
    order = np.argsort(ids)
    columns = pd.MultiIndex.from_product([[dn.replace('_', ' ') for dn in dataset_names], methods, eval_cols, avg])
    index = pd.MultiIndex.from_product([ids[order], names])

    df = pd.DataFrame(index=index, columns=columns)
    is_best_df = pd.DataFrame(index=index, columns=columns, data=False)

    for dataset_name in dataset_names:
        dn = dataset_name.replace('_', ' ')
        evalfile = os.path.join(outdir, dataset_name + '_avg.h5')
        data_dict = dd.io.load(evalfile)

        # fill dataframe
        idx = pd.IndexSlice
        for method in methods:
            for name in names:
                values = np.round(data_dict[method][name].loc[:, idx[eval_cols, avg]].values.astype(float), decimals=2)
                df.loc[idx[:, name], idx[dn, method, :, :]] = values[order]

        for name in names:
            for eval_col in eval_cols:
                values = np.round(df.loc[idx[:, name], idx[dn, :, eval_col, 'mean']].values.astype(float), decimals=2)
                col_idcs_best = np.argmax(values, axis=1)
                is_best = np.zeros(values.shape, dtype=bool)
                for row, col in enumerate(col_idcs_best):
                    is_best[row, col] = True
                is_best_df.loc[idx[:, name], idx[dn, :, eval_col, 'mean']] = is_best

    # remove rows and columns with single level
    if isinstance(df.index, pd.MultiIndex):
        levels = [level for level in df.index.levels if len(level) > 1]
        df.index = pd.MultiIndex.from_product(levels)
    if isinstance(df.columns, pd.MultiIndex):
        levels = [level for level in df.columns.levels if len(level) > 1]
        df.columns = pd.MultiIndex.from_product(levels)

    writer = TableWriter(df=df)
    writer.write_latex_table(outfile=outfile, mode='w', is_best=is_best_df.values, escape=False)


if __name__ == '__main__':
    print("Working on device {}".format(args.device))
    with torch.cuda.device(args.device):
        if args.pretraining:
            evaluate_pretraining(n_layers=args.n_layers)
            exit()

        if args.outdir is None:
            args.outdir = os.path.join(args.path_to_files, 'evaluation')
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
        try:
            data_dir = os.path.join(args.path_to_files, args.dataset_name.replace('_test', '_train'))
            if not os.path.isdir(data_dir):
                raise ReferenceError()
        except ReferenceError:
            print('Use direct path to files as data directoy.')
            data_dir = args.path_to_files
        if data_dir.endswith('_test'):          # evaluation on testset with trained models
            data_dir = data_dir.replace('_test', '_train')

        if args.labels:
            evaluate_labels(data_dir=data_dir, outdir=args.outdir, plot=args.plot, overwrite=args.overwrite)
            evaluate_labels_quantitative(data_dir=data_dir, outdir=args.outdir, plot=args.plot, overwrite=args.overwrite)

        if args.models:
            weight_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pth.tar')]
            feature_dim = 512
            # if not os.path.isfile(os.path.join(args.outdir, args.dataset_name + '.h5')):
            #     make_baseline(args.dataset_name, outdir=args.outdir, feature_dim=feature_dim, mode='a')
            for weight_file in weight_files:
                if not args.overwrite:
                    heuristic = heuristic_from_weightfile(weight_file)
                    try:
                        if heuristic == 'test':
                            heuristic = weight_file.split('_')[-2]
                            if heuristic == 'clique':
                                heuristic = 'clique_svm'
                            outdf = dd.io.load(os.path.join(args.outdir, args.dataset_name + '.h5'))['test'][heuristic]['features']
                        else:
                            outdf = dd.io.load(os.path.join(args.outdir, args.dataset_name + '.h5'))[heuristic]['features']
                        index = weight_file.split('/')[-1].split('.pth.tar')[0]
                        if index in outdf.index:
                            print('Evaluation exists - skip {}.'.format(index))
                            continue
                    except (KeyError, IOError):
                        pass
                try:
                    torch.load(weight_file)
                except (RuntimeError, KeyError) as e:
                    print('Error in weight file [{}:{}] - skip {}.'.format(type(e), e, weight_file))
                    continue
                evaluate_training(args.dataset_name, weight_file, outdir=args.outdir,
                                pretraining=args.pretrained_models, n_layers=args.n_layers,
                                n_stat_runs=3, feature_dim=feature_dim, mode='a')

            # for weight_file in weight_files:
            #     if not args.overwrite and os.path.isfile(os.path.join(args.outdir, args.dataset_name + '_splitstt.h5')):
            #         heuristic = heuristic_from_weightfile(weight_file)
            #         try:
            #             if heuristic == 'test':
            #                 heuristic = weight_file.split('_')[-2]
            #                 if heuristic == 'clique':
            #                     heuristic = 'clique_svm'
            #                 outdf = dd.io.load(os.path.join(args.outdir, args.dataset_name + '_splitstt.h5'))['test'][heuristic]['features']
            #             else:
            #                 outdf = dd.io.load(os.path.join(args.outdir, args.dataset_name + '_splitstt.h5'))[heuristic]['features']
            #             index = weight_file.split('/')[-1].split('.pth.tar')[0]
            #             if index in outdf['train'].index:
            #                 print('Evaluation exists - skip: {}.'.format(index))
            #                 continue
            #         except KeyError:
            #             pass
            #     evaluate_training_split_train_test_retrieval(args.dataset_name, weight_file, outdir=args.outdir,
            #                                                  n_stat_runs=3, feature_dim=feature_dim, mode='a')

            # save averaged df
            def save_averaged_splits():
                outfile = os.path.join(args.outdir, args.dataset_name + '_splitstt.h5')
                data_dict = dd.io.load(outfile)
                heuristics = ['none', 'area', 'svm', 'clique_svm']
                names = ['features', 'projection']
                splits = ['train', 'pred', 'test']
                mode = 'w'
                # copy baseline
                for name in names:
                    try:
                        df = data_dict['baseline'][name]
                        df.to_hdf(outfile.replace('.h5', '_avg.h5'), key=os.path.join('baseline', name), mode=mode)
                        mode = 'a'
                    except KeyError:
                        pass

                for data_dict_, key in zip([data_dict, ], ('', )):
                    for heuristic in heuristics:
                        for name in names:
                            for split in splits:
                                if split not in data_dict_[heuristic][name].keys():
                                    continue
                                df = data_dict_[heuristic][name][split]
                                valid_idcs = np.where(map(lambda x: not x.endswith('_test'), df.index.values))[0]
                                df = df.iloc[valid_idcs]
                                df = average_evaluation_df(df)

                                df.to_hdf(outfile.replace('.h5', '_avg.h5'), key=os.path.join(key, heuristic, name, split), mode=mode)
                                mode = 'a'
                return outfile

            def save_averaged():
                outfile = os.path.join(args.outdir, args.dataset_name + '.h5')
                data_dict = dd.io.load(outfile)
                heuristics = ['none', 'area', 'svm', 'clique_svm']

                names = ['features', 'projection']
                mode = 'w'
                # copy baseline
                for name in names:
                    try:
                        df = data_dict['baseline'][name]
                        df.to_hdf(outfile.replace('.h5', '_avg.h5'), key=os.path.join('baseline', name), mode=mode)
                        mode = 'a'
                    except KeyError:
                        pass

                for data_dict_, key in zip([data_dict, ], ('', )):
                    for heuristic in heuristics:
                        if heuristic not in data_dict.keys():
                            continue
                        for name in names:
                            df = data_dict_[heuristic][name]
                            df = average_evaluation_df(df)

                            valid_idcs = np.where(map(lambda x: not x.endswith('_test'), df.index.values))[0]
                            df = df.iloc[valid_idcs]

                            df.to_hdf(outfile.replace('.h5', '_avg.h5'), key=os.path.join(key, heuristic, name),
                                    mode=mode)
                            mode = 'a'
                return outfile

            outfile = save_averaged()
            plot_model_evaluation(args.dataset_name, outdir=args.outdir)

            # outfile = save_averaged_splits()
            # plot_model_evaluation_splits(args.dataset_name, outdir=args.outdir)

            if args.plot:
                data_dict = dd.io.load(outfile.replace('.h5', '_avg.h5'))

                plotter = TexPlotter()
                plotter.figsize = (3.5, plotter.get_doc_lengths('')['textwidth'])
                cmap = plt.cm.tab10

                def strtuple_to_ints(strtuple):
                    try:
                        vals = [int(s) for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
                    except ValueError:
                        vals = [float(s.replace('-', '.'))
                                for s in strtuple.replace('(', '').replace(')', '').strip().split(',')]
                    return vals

                for data_dict_, key in zip([data_dict, data_dict['test']], ('', 'test')):
                    heuristics = ['none', 'area', 'svm']
                    baseline = data_dict['baseline']
                    for name in ['features', 'projection']:
                        df = data_dict_[heuristics[0]][name]
                        eval_cols = df.columns.levels[0].values
                        base = baseline[name]

                        # get averages
                        for ec in eval_cols:
                            fig, ax = plt.subplots(1)
                            ax.set_title(ec)

                            for i, heuristic in enumerate(heuristics):
                                df = data_dict_[heuristic][name][ec]

                                x = np.array(map(lambda x: np.sum(filename_to_tuple(x)), df.index.values))
                                sort_idx = np.argsort(x)

                                ax.errorbar(x[sort_idx], y=df['mean'].values[sort_idx],
                                            yerr=df['std'].values[sort_idx],
                                            label=heuristic, c=cmap(i), marker='.', markersize=10,
                                            elinewidth=0.5, capsize=3, capthick=0.5)

                            # plot baseline
                            y = baseline[name][ec].values
                            x = ax.get_xlim()
                            ax.plot(x, np.repeat(y, 2), marker='', linestyle='--', c='grey')

                            plt.legend()
                            plotter.render(fig)
                            plt.show(block=False)
                            if key == 'test':
                                outfilename = outfile.replace('.h5', '_{}_test_{}.png'.format(name, ec))
                            else:
                                outfilename = outfile.replace('.h5', '_{}_{}.png'.format(name, ec))
                            plotter.save(fig, outfilename=outfilename, dpi=200)
                            plt.close()
