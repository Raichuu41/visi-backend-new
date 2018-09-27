import deepdish as dd
import numpy as np
import pandas as pd
import sys
import time
import os
import torch
from sklearn.cluster import k_means
import sklearn.metrics as metrics
from faiss_master import faiss
from collections import Counter
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('./MapNetCode')
from model import MapNet
from train import get_feature, mutual_k_nearest_neighbors, listed_k_nearest_neighbors, svm_k_nearest_neighbors, score_k_nearest_neighbors

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


if not os.getcwd().endswith('/MapNetCode'):
    os.chdir('/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode')


# feature_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/ShapeDataset_MapNet/features/cycle_004_feature.hdf5'
# info_file = '/export/home/kschwarz/Documents/Data/Geometric_Shapes/labels.hdf5'
# weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/ShapeDataset_MapNet/models/cycle_004_models.pth.tar'
# feature_file_shape = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/features/ShapeDataset_NarrowNet128_MobileNetV2_val.hdf5'
# feature_file_baseline = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_test.hdf5'
feature_file_baseline = 'features/MobileNetV2_info_artist_49_multilabel_test.hdf5'
feature_file_test = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/features/cycle_001_feature.hdf5'
feature_file_val = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_val.hdf5'
info_file_test = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
info_file_val = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_val.hdf5'
weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/models/cycle_001_models.pth.tar'
# feature_file_baseline = 'features/OfficeDataset_NarrowNet128_MobileNetV2_info_test.hdf5'
# feature_file_test = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/OfficeDataset_MapNet/features/cycle_001_feature.hdf5'
# feature_file_val = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_val.hdf5'
# info_file_test = '/export/home/kschwarz/Documents/Data/OfficeHomeDataset_10072016/info_test.hdf5'
# info_file_val = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_val.hdf5'
# weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/OfficeDatasetless_L2_MapNet/features/cycle_001_feature.hdf5'




def get_names_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names


def load_data(feature_file, info_file, split=0):
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')
    if not os.path.isfile(info_file):
        raise RuntimeError('Info file not found.')

    df = dd.io.load(info_file)['df']

    data = dd.io.load(feature_file)
    try:
        names, features = data['image_id'], data['feature']
    except KeyError:
        try:
            names, features = data['image_names'], data['features']
        except KeyError:
            names, features = data['image_name'], data['features']

    is_shape_dataset = 'ShapeDataset' in feature_file
    is_office_dataset = 'OfficeDataset' in feature_file
    is_bam_dataset = 'BAMDataset' in feature_file
    if is_shape_dataset:
        outdir = 'ShapeDataset'
        category = ['shape', 'n_shapes', 'color_shape', 'color_background']

        df = df[df['split'] == split]
        df.index = range(len(df))
    elif is_office_dataset:
        outdir = 'OfficeDataset'
        category = ['genre', 'style']
    elif is_bam_dataset:
        outdir = 'BAMDataset'
        category = ['content', 'emotion', 'media']
    else:
        outdir = 'Wikiart'
        category = ['artist_name', 'genre', 'style', 'technique', 'century']

    if not (names == df['image_id']).all():
        raise RuntimeError('Image names in info file and feature file do not match.')

    outdict = {'image_names': names,
               'features': features,
               'labels': {c: df[c] for c in category}}

    return outdict, outdir


def predict_cluster(feature, label, cluster_per_label=1, random_state=123):
    """Assign the main label of each cluster as predicted label."""
    gt = label.copy()
    n_cluster = len(np.unique(gt)) * cluster_per_label

    _, cluster_labels, _ = k_means(feature, n_clusters=n_cluster, random_state=random_state)
    prediction = np.full(gt.shape, None, dtype=gt.dtype)
    count_cluster = dict.fromkeys(np.unique(gt), 0)
    for cl in range(n_cluster):
        mask = cluster_labels == cl
        counter = Counter(gt[mask])
        main_label = counter.keys()[np.argmax(counter.values())]
        count_cluster[main_label] += 1
        prediction[mask] = main_label

    assert np.all(prediction != None), 'Error in clustering, not all samples received prediction.'

    return prediction, gt, count_cluster


def k_nn_accuracy(feature, gt_label, k=(1,2,4,8), average=None):
    if average not in [None, 'micro', 'macro']:
        raise NotImplementedError('average has to be None, "micro" or "macro".')

    gt_label = np.array(gt_label)

    N, d = feature.shape
    # search neighbors for each sample
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(feature.astype(np.float32))  # add vectors to the index
    _, neighbors = index.search(feature.astype(np.float32), max(k)+1)

    # predicted labels are labels of first entry (sample itself)
    prediction = gt_label[neighbors[:, 0]].reshape(-1, 1).repeat(max(k), axis=1)
    gt = np.stack([gt_label[neighbors[:, i]] for i in np.arange(1, max(k) + 1)], axis=1)

    is_correct = prediction == gt

    eval_dict = dict.fromkeys(k)
    for _k in k:
        if average == 'micro':
            acc = is_correct[:, :_k].sum() * 1.0 / is_correct[:, :_k].size
        else:
            labelset = np.unique(gt_label)
            acc = dict.fromkeys(labelset)
            for l in labelset:
                mask = gt_label == l
                acc[l] = is_correct[mask, :_k].sum() * 1.0 / is_correct[mask, :_k].size
            if average == 'macro':
                acc = np.mean(acc.values())
        eval_dict[_k] = acc
    return eval_dict


def evaluate(feature, label, cluster_per_label=1):
    y_pred, y_true, count_cluster = predict_cluster(feature, label, cluster_per_label=cluster_per_label)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    prec_micro, rec_micro, f1_micro, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
    prec_macro, rec_macro, f1_macro, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    neighbor_acc = k_nn_accuracy(feature, label, average=None)
    neighbor_acc_micro = k_nn_accuracy(feature, label, average='micro')
    neighbor_acc_macro = k_nn_accuracy(feature, label, average='macro')

    columns = np.append(np.unique(label), np.array(['micro', 'macro']))
    rows = ['precision', 'recall', 'f1', 'nmi', 'n_cluster'] + \
           ['neighbor_acc_at_{}'.format(_k) for _k in np.sort(neighbor_acc.keys())]

    df = pd.DataFrame(columns=columns, index=rows)

    df.loc['precision'] = np.append(prec, np.array([prec_micro, prec_macro]))
    df.loc['recall'] = np.append(rec, np.array([rec_micro, rec_macro]))
    df.loc['f1'] = np.append(f1, np.array([f1_micro, f1_macro]))
    df.loc['nmi', 'micro'] = nmi
    df.loc['n_cluster'] = count_cluster

    for _k in neighbor_acc.keys():
        for lbl, val in neighbor_acc[_k].items():
            df.loc['neighbor_acc_at_{}'.format(_k), lbl] = val
        df.loc['neighbor_acc_at_{}'.format(_k), 'micro'] = neighbor_acc_micro[_k]
        df.loc['neighbor_acc_at_{}'.format(_k), 'macro'] = neighbor_acc_macro[_k]

    return df


def run_evaluation(info_file, feature_file=None, split=0, weight_file=None, categories=None, cluster_per_label=1):
    data_dict, outdir = load_data(feature_file, info_file, split=split)
    feature = data_dict['features']
    if weight_file is not None and 'mapping/' in weight_file:
        outpath, filename = weight_file.split('mapping/')[1].split('/models/')
        outpath = os.path.join('evaluation', outdir, outpath)
        filename = filename.replace('models.pth.tar', 'evaluation_val.xlsx')
    elif 'mapping/' in feature_file:
        outpath, filename = feature_file.split('mapping/')[1].split('/features/')
        outpath = os.path.join('evaluation', outdir, outpath)
        filename = filename.replace('feature.hdf5', 'evaluation_train.xlsx')
    else:
        outpath = os.path.join('evaluation', outdir, 'unknown')
        filename = 'evaluation.xlsx'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if weight_file is not None:
        model = MapNet(feature_dim=feature.shape[1], output_dim=2)
        states = torch.load(weight_file, map_location='cpu')
        state_dict = states[max(states.keys())]['state_dict']
        model.load_state_dict(state_dict)

        feature = get_feature(model.mapping, feature)

    if categories is None:      # evaluate on all categories
        categories = data_dict['labels'].keys()

    writer = pd.ExcelWriter(os.path.join(outpath, filename))
    for c in categories:
        label = data_dict['labels'][c].values
        df = evaluate(feature, label, cluster_per_label)
        df.to_excel(writer, c)
    writer.save()
    print('Saved evaluation to {}'.format(os.path.join(outpath, filename)))


def epoch_wise_evaluation(info_file, feature_file, weight_file, split=0,
                          use_cuda=True, categories=None, plot_results=False, plot_labels=None):
    use_cuda = use_cuda and torch.cuda.is_available()
    # load label and initial features
    data_dict, outdir = load_data(feature_file, info_file, split=split)
    feature = data_dict['features']
    if categories is None:         # evaluate on all categories
        categories = data_dict['labels'].keys()

    # iterate over all models and compute features for each epoch
    net = MapNet(feature.shape[1], output_dim=2)

    models = torch.load(weight_file, map_location='cpu')
    dfs = dict.fromkeys(categories)                         # for each category dict containing dataframes of all epochs
    for c in dfs.keys():
        dfs[c] = dict.fromkeys(models.keys())
    for epoch in models.keys():
        if epoch % 5 == 0:
            print('\n{}/{}'.format(epoch, max(models.keys())))
        net.load_state_dict(models[epoch]['state_dict'])
        fts = get_feature(net.mapping, feature)

        for c in categories:
            label = data_dict['labels'][c]
            df = evaluate(fts, label)
            dfs[c][epoch] = df

    outpath, filename = weight_file.split('mapping/')[1].split('/models/')
    outpath = os.path.join('evaluation', outdir, outpath)
    filename = filename.replace('models.pth.tar', 'epoch_wise_evaluation.xlsx')

    writer = pd.ExcelWriter(os.path.join(outpath, filename))
    for c in categories:
        dfs[c] = pd.concat(dfs[c])
        dfs[c].to_excel(writer, c)
    writer.save()
    print('saved {}'.format(os.path.join(outpath, filename)))

    if plot_results:
        print('save plots to {}'.format(os.path.join(outpath, 'plots')))
        plot_dir = os.path.join(outpath, 'plots')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        for i, c in enumerate(categories):
            epochs, criteria = dfs[c].index.levels
            x = np.stack([e for e in epochs])
            if plot_labels is None:
                data_labels = dfs[c].keys()
            else:
                data_labels = plot_labels[i]
            for crit in criteria:
                plot_name = filename.replace('.xlsx', '_' + crit + '.png')
                named_colors = get_names_colors()
                fig, ax = plt.subplots(1)
                for lbl in data_labels:
                    y = np.stack([dfs[c].loc[e].loc[crit, lbl] for e in epochs])
                    color = lbl if lbl in named_colors else np.random.choice(named_colors)
                    named_colors.remove(color)
                    ax.plot(x, y, c=color, label=lbl)

                lgd = ax.legend(bbox_to_anchor=(1.01,1))
                ax.set_title(crit)
                plt.savefig(os.path.join(plot_dir, plot_name), bbox_extra_artists=(lgd,), bbox_inches='tight')


def evaluate_neighbor_methods(info_file, feature_file, k=[1, 10, 100], n_group=20):
    data_dict, outdir = load_data(feature_file, info_file, split=0)
    outpath = os.path.join('evaluation', outdir, 'unknown')
    filename = 'neighbor_method_analysis.xlsx'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    feature = data_dict['features']
    categories = data_dict['labels'].keys()
    eval_dict = dict.fromkeys(categories)
    for c in categories:
        eval_dict[c] = {'clique': None, 'svm': None, 'listed': None, 'score': None}
        label = data_dict['labels'][c]
        labellist = np.unique(label)
        for neighbor_fn, name in zip([mutual_k_nearest_neighbors, svm_k_nearest_neighbors,
                                      listed_k_nearest_neighbors, score_k_nearest_neighbors],
            ['clique', 'svm', 'listed', 'score']):
            print('METHOD: {}'.format(name))
            eval_dict[c][name] = {}
            for _k in k:
                print('{} neighbors...'.format(_k))
                accs = []
                for lbl in labellist:
                    acc = 0
                    n_avg = 3
                    N = min(len(np.where(label == lbl)[0]), n_group)
                    for i in range(n_avg):              # average over 3 groups of n_group sample
                        sample_indices = np.random.choice(np.where(label == lbl)[0], N, replace=False)
                        neighbors = neighbor_fn(feature, sample_indices, _k)
                        acc += np.isin(neighbors, np.where(label == lbl)[0]).sum() * 1.0 / _k
                    acc = acc / n_avg
                    accs.append(acc)
                eval_dict[c][name][_k] = np.mean(accs)

    writer = pd.ExcelWriter(os.path.join(outpath, filename))
    avg_vals = None
    for c in categories:
        df = None
        for name in eval_dict[c].keys():
            if df is None:
                df = pd.DataFrame.from_dict(eval_dict[c][name], orient='index')
                df.columns = [name]
            else:
                df[name] = pd.DataFrame.from_dict(eval_dict[c][name], orient='index')
        df.to_excel(writer, c)
        if avg_vals is None:
            avg_vals = df.values
        else:
            avg_vals += df.values
    avg_vals = avg_vals / len(categories)
    df = pd.DataFrame(avg_vals, index=k, columns=df.columns)
    df.to_excel(writer, 'average')
    writer.save()
    print('Saved {}'.format(os.path.join(outpath, filename)))

    return eval_dict


if __name__ == '__main__':
    categories = ['genre']
    # baseline train
    run_evaluation(info_file_test, feature_file_baseline, weight_file=None, categories=categories, cluster_per_label=2)
    # evaluation on train
    run_evaluation(info_file_test, feature_file_test, weight_file=None, categories=categories, cluster_per_label=2, split=2)
    # baseline val
    # run_evaluation(info_file_val, feature_file_val, weight_file=None, categories=categories, cluster_per_label=2, split=2)
    # evaluation on val
    run_evaluation(info_file_val, feature_file_val, weight_file=weight_file, categories=categories, cluster_per_label=2,
                   split=1)
    epoch_wise_evaluation(info_file_test, feature_file=feature_file_baseline, weight_file=weight_file, categories=categories,
                          split=1, plot_results=True, plot_labels=None)
    # epoch_wise_evaluation(info_file_val, feature_file=feature_file_val, weight_file=weight_file, categories=categories,
    #                       split=1, plot_results=True, plot_labels=None)

    # eval_dict = evaluate_neighbor_methods(info_file_test, feature_file_baseline, k=[1, 5, 10, 50, 100])
