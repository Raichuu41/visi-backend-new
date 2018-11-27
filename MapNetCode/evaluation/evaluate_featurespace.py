import deepdish as dd
import numpy as np
import pandas as pd
import sys
import time
import os
import torch
import sklearn.metrics as metrics
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from snack import snack_embed
from scipy import interpolate
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from eval_utils import kmeans, knn
from models import MapNet, mobilenet_v2, make_featurenet
os.chdir('..')
sys.path.append('.')
from train import get_feature, mutual_k_nearest_neighbors, listed_k_nearest_neighbors, svm_k_nearest_neighbors, score_k_nearest_neighbors
os.chdir('evaluation/')

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


# feature_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/ShapeDataset_MapNet/features/cycle_004_feature.hdf5'
# info_file = '/export/home/kschwarz/Documents/Data/Geometric_Shapes/labels.hdf5'
# weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/ShapeDataset_MapNet/models/cycle_004_models.pth.tar'
# feature_file_shape = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/features/ShapeDataset_NarrowNet128_MobileNetV2_val.hdf5'
# feature_file_baseline = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_test.hdf5'


# feature_file_baseline = 'features/MobileNetV2_info_elgammal_subset_val.hdf5'
# feature_file_test = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/features/cycle_001_feature.hdf5'
# feature_file_val = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_val.hdf5'
# feature_file_val = '../features/MobileNetV2_info_elgammal_subset_val_imagenet_512.hdf5'
# info_file_test = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
# info_file_val = '../pretraining/wikiart_datasets/info_elgammal_subset_val.hdf5'
# weight_file = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/evaluation/runs/models/artist_N9999_Ngen0010_Ncorr0000_None_0000_MobileNetV2_model_best.pth.tar'


# feature_file_baseline = 'features/OfficeDataset_NarrowNet128_MobileNetV2_info_test.hdf5'
# feature_file_test = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/OfficeDataset_MapNet/features/cycle_001_feature.hdf5'
# feature_file_val = 'features/NarrowNet128_MobileNetV2_info_artist_49_multilabel_val.hdf5'
# info_file_test = '/export/home/kschwarz/Documents/Data/OfficeHomeDataset_10072016/info_test.hdf5'
# info_file_val = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_val.hdf5'
# weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/OfficeDatasetless_L2_MapNet/features/cycle_001_feature.hdf5'
#
# feature_file_baseline = 'features/BAMDataset_NarrowNet128_MobileNetV2_info_test.hdf5'
# info_file_test = '/export/home/kschwarz/Documents/Data/BAM/info_test.hdf5'
# info_file_val = '/export/home/kschwarz/Documents/Data/BAM/info_val.hdf5'
# feature_file_val = 'features/BAMDataset_NarrowNet128_MobileNetV2_info_val.hdf5'
#
# feature_file_test = 'net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/BAMDatasetTEST_MapNet/features/cycle_001_feature.hdf5'
# weight_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/BAMDatasetTEST_MapNet/models/cycle_001_models.pth.tar'
# impath = '/export/home/kschwarz/Documents/Data/BAM'


def get_names_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names


def load_data(feature_file, info_file, split=0, category=None):
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
        if category is None:
            category = ['shape', 'n_shapes', 'color_shape', 'color_background']

        df = df[df['split'] == split]
        df.index = range(len(df))
    elif is_office_dataset:
        outdir = 'OfficeDataset'
        if category is None:
            category = ['genre', 'style']
    elif is_bam_dataset:
        outdir = 'BAMDataset'
        if category is None:
            category = ['content', 'emotion', 'media']
    else:
        outdir = 'Wikiart'
        if category is None:
            category = ['artist_name', 'genre', 'style', 'technique', 'century']

    if not (names == df['image_id']).all():
        raise RuntimeError('Image names in info file and feature file do not match.')

    labels = df[category[0]]
    outdict = {'image_names': names[labels.dropna().index],
               'features': features[labels.dropna().index],
               'labels': {c: df[c].dropna() for c in category}}

    return outdict, outdir


def predict_cluster(feature, label, cluster_per_label=1, random_state=123, use_gpu=False):
    """Assign the main label of each cluster as predicted label."""
    gt = label.copy()
    n_cluster = len(np.unique(gt)) * cluster_per_label
    km = kmeans(feature, n_centroids=n_cluster, gpu=use_gpu)
    cluster_center = km.cluster_centers_
    cluster_labels = km.labels_
    prediction = np.full(gt.shape, None, dtype=gt.dtype)
    count_cluster = dict.fromkeys(np.unique(gt), 0)
    for cl, center in zip(range(n_cluster), cluster_center):
        mask = cluster_labels == cl
        if mask.sum() == 0:
            continue
        counter = Counter(gt[mask])
        main_label = counter.keys()[np.argmax(counter.values())]
        count_cluster[main_label] += 1
        prediction[mask] = main_label

    assert np.all(prediction != None), 'Error in clustering, not all samples received prediction.'

    return prediction, gt, count_cluster


def k_nn_accuracy(feature, gt_label, k=(1,2,4,8), average=None, use_gpu=False):
    if average not in [None, 'micro', 'macro']:
        raise NotImplementedError('average has to be None, "micro" or "macro".')

    gt_label = np.array(gt_label)

    N, d = feature.shape
    # search neighbors for each sample
    _, neighbors, _ = knn(feature.astype(np.float32), max(k)+1, gpu=use_gpu)

    # predicted labels are labels of first entry (sample itself)
    prediction = gt_label[neighbors[:, 0]].reshape(-1, 1).repeat(max(k), axis=1)
    gt = np.stack([gt_label[neighbors[:, i]] for i in np.arange(1, max(k) + 1)], axis=1)

    is_correct = prediction == gt

    eval_dict_micro = dict.fromkeys(k)
    eval_dict_macro = dict.fromkeys(k)
    eval_dict_none = dict.fromkeys(k)
    for _k in k:
        eval_dict_micro[_k] = is_correct[:, :_k].sum() * 1.0 / is_correct[:, :_k].size

        labelset = np.unique(gt_label)
        acc = dict.fromkeys(labelset)
        for l in labelset:
            mask = gt_label == l
            acc[l] = is_correct[mask, :_k].sum() * 1.0 / is_correct[mask, :_k].size
        eval_dict_none[_k] = acc
        eval_dict_macro[_k] = np.mean(acc.values())

    return {'micro': eval_dict_micro, 'macro': eval_dict_macro, 'none': eval_dict_none}


def evaluate(feature, label, cluster_per_label=1, k=(1,2,4,8), use_gpu=False):
    valid = label != 'None'
    feature = feature[valid]
    label = label[valid]
    y_pred, y_true, count_cluster = predict_cluster(feature, label, cluster_per_label=cluster_per_label, use_gpu=use_gpu)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    prec_micro, rec_micro, f1_micro, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
    prec_macro, rec_macro, f1_macro, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    neighbor_accs = k_nn_accuracy(feature, label, k=k, average=None, use_gpu=use_gpu)
    neighbor_acc = neighbor_accs['none']
    neighbor_acc_micro = neighbor_accs['micro']
    neighbor_acc_macro = neighbor_accs['macro']

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


def run_evaluation(info_file, feature_file=None, split=0, weight_file=None, categories=None, cluster_per_label=1,
                   feature_dim=512):
    data_dict, outdir = load_data(feature_file, info_file, split=split, category=categories)
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
        if 'MapNet' in weight_file.split('/')[-1]:
            model = MapNet(feature_dim=feature_dim)
        elif 'MobileNetV2' in weight_file.split('/')[-1]:
            model, feature_dim_model = make_featurenet(mobilenet_v2(pretrained=True))
            if feature_dim_model != feature_dim:
                model = torch.nn.Sequential(model, torch.nn.ReLU(inplace=True),
                                            torch.nn.Linear(in_features=feature_dim_model, out_features=feature_dim))

        states = torch.load(weight_file, map_location='cpu')
        try:
            state_dict = states[max(states.keys())]['state_dict']
        except IndexError:
            state_dict = states
        model.load_state_dict(state_dict)
        try:
            feature = get_feature(model.features, feature)
        except AttributeError:
            feature = get_feature(model, feature)

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


def vis_ft_space_movement(fts_old, fts_new, binary_label=None):
    embedding_old = snack_embed(fts_old.astype(np.double),
                                contrib_cost_tsne=1,
                                triplets=np.ones((1,3), dtype=long),
                                contrib_cost_triplets=0,
                                theta=0.5)
    embedding_new = snack_embed(fts_new.astype(np.double),
                                contrib_cost_tsne=1,
                                triplets=np.ones((1,3), dtype=long),
                                contrib_cost_triplets=0,
                                theta=0.5)

    movement = np.linalg.norm(fts_new - fts_old, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    ax[0].scatter(embedding_old[:, 0], embedding_old[:, 1], cmap=plt.cm.hot, c=movement)
    ax[0].set_title('old')
    ax[1].scatter(embedding_new[:, 0], embedding_new[:, 1], cmap=plt.cm.hot, c=movement)
    ax[1].set_title('new')

    if binary_label is not None:
        ax[0].scatter(embedding_old[binary_label, 0], embedding_old[binary_label, 1], facecolors='none', edgecolors='b')
        ax[1].scatter(embedding_new[binary_label, 0], embedding_new[binary_label, 1], facecolors='none', edgecolors='b')

    plt.show(block=False)


def vis_mapping_continuity(feature_file, info_file, weight_file, category, label):
    data_dict, outdir = load_data(feature_file, info_file, split=0)
    fts_old = torch.from_numpy(data_dict['features'])
    positives = (data_dict['labels'][category] == label).values.astype(np.uint8)
    feature_std = torch.std(fts_old[positives], dim=0)
    model_old = MapNet(feature_dim=fts_old.shape[1], output_dim=2)
    model_new = MapNet(feature_dim=fts_old.shape[1], output_dim=2)
    states = torch.load(weight_file, map_location='cpu')

    batch_size = 8
    dataloader = torch.utils.data.DataLoader(fts_old, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    N_noise_per_sample = 15
    frac_std = 2.5


    model_old.load_state_dict(states[min(states.keys())]['state_dict'])
    model_new.load_state_dict(states[max(states.keys())]['state_dict'])
    if torch.cuda.is_available():
        model_old = model_old.cuda()
        model_new = model_new.cuda()

    model_old.eval()
    model_new.eval()
    embedding_old = []
    embedding_new = []
    for batch_idx, fts in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print('{}/{}'.format((batch_idx + 1) * fts.shape[0], len(dataloader.dataset)))
        if torch.cuda.is_available():
            fts = torch.autograd.Variable(fts.cuda())
        else:
            fts = torch.autograd.Variable(fts)

        noise = torch.randn_like(fts.repeat(N_noise_per_sample, 1)) * frac_std * feature_std.type_as(fts)         # scale by dimensionwise std of features
        noise_fts = fts.repeat(N_noise_per_sample, 1) + noise

        emb_old = model_old.embedder(fts).data.cpu().reshape(1, -1, 2)
        emb_old_noise = model_old.embedder(noise_fts).data.cpu().reshape(N_noise_per_sample, -1,2)
        embedding_old.append(torch.cat([emb_old, emb_old_noise]).transpose(1, 0))

        emb_new = model_new(fts).data.cpu().reshape(1, -1, 2)
        emb_new_noise = model_new(noise_fts).data.cpu().reshape(N_noise_per_sample, -1,2)
        embedding_new.append(torch.cat([emb_new, emb_new_noise]).transpose(1, 0))

    embedding_old = torch.cat(embedding_old).numpy()
    embedding_old_flat = embedding_old.reshape(-1, 2)
    embedding_new = torch.cat(embedding_new).numpy()
    embedding_new_flat = embedding_new.reshape(-1, 2)

    labels_flat = positives.reshape(-1, 1).repeat(N_noise_per_sample + 1, axis=1).reshape(-1)

    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    # take out 10 points and plot them with their noise points
    plot_points = np.random.choice(np.where(positives)[0], 10, replace=False)

    for axes, emb, emb_orig in zip(ax, [embedding_old_flat, embedding_new_flat], [embedding_old, embedding_new]):
        # interpolate labels in embedding
        x = np.linspace(emb[:, 0].min(), emb[:, 0].max(), 500)
        y = np.linspace(emb[:, 1].min(), emb[:, 1].max(), 500)
        grid_x, grid_y = np.meshgrid(x, y)
        interpolated = interpolate.griddata(emb, labels_flat, (grid_x, grid_y), method='linear')

        axes.imshow(np.tanh(interpolated), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
        # take out 10 points and plot them with their noise points
        for c, p in zip(plt.cm.tab10.colors, plot_points):
            axes.scatter(emb_orig[p, 0, 0], emb_orig[p, 0, 1], c=c, s=30, edgecolor='black')
            axes.scatter(emb_orig[p, 1:, 0], emb_orig[p, 1:, 1], c=c, s=10)

    plt.show(block=False)


def classification_acc_svm(train_data, train_lbl, val_data, val_lbl, penalty_svm=10., avg='micro'):
    clf = SVC(kernel='rbf', C=penalty_svm, gamma='auto', probability=False)
    print('Train SVM...')
    clf.fit(train_data, train_lbl)
    print('Done.')

    # evaluate SVM
    print('Predict...')
    prediction = clf.predict(val_data)
    print('Done.')

    # evaluate prediction
    if avg == 'macro':
        accs = []
        for lbl in np.unique(val_lbl):
            lbl_idcs = np.where(val_lbl == lbl)[0]
            accs.append((prediction[lbl_idcs] == val_lbl[lbl_idcs]).sum() * 1.0 / len(lbl_idcs))
        acc = np.mean(accs)
    else:
        acc = (prediction == val_lbl).sum() * 1.0 / len(val_lbl)
    return acc


def evaluate_feature_classification(info_file_train, feature_file_train,
                                    info_file_val, feature_file_val,
                                    task, feature_dim):
    train_dict, _ = load_data(feature_file=feature_file_train, info_file=info_file_train, category=[task])
    val_dict, _ = load_data(feature_file=feature_file_val, info_file=info_file_val, category=[task])

    # filter task-non-labeled
    for data_dict in [train_dict, val_dict]:
        valid = data_dict['labels'][task].dropna().index
        data_dict['features'] = data_dict['features'][valid]
        data_dict['labels'] = data_dict['labels'][task][valid].values

    if train_dict['features'].shape[1] != feature_dim:
        print('Reduce features from {} to {} dimensions with PCA'.format(train_dict['features'].shape[1], feature_dim))
        pca = PCA(n_components=feature_dim)
        train_dict['features'] = pca.fit_transform(train_dict['features'])
        val_dict['features'] = pca.transform(val_dict['features'])

    print(classification_acc_svm(train_data=train_dict['features'], train_lbl=train_dict['labels'],
                                 val_data=val_dict['features'], val_lbl=val_dict['labels']))


def GTE(features, triplets):
    dp = np.linalg.norm(features[triplets[:, 0]] - features[triplets[:, 1]], axis=1)
    dn = np.linalg.norm(features[triplets[:, 0]] - features[triplets[:, 2]], axis=1)

    gte = (dn < dp).sum() * 1.0 / len(triplets)
    return gte


if __name__ == '__main__':
    categories = ['artist_name']
    # baseline train
    # run_evaluation(info_file_test, feature_file_baseline, weight_file=None, categories=None, cluster_per_label=2)
    # evaluation on train
    # run_evaluation(info_file_test, feature_file_test, weight_file=None, categories=categories, cluster_per_label=2, split=2)
    # baseline val
    # run_evaluation(info_file_val, feature_file_val, weight_file=None, categories=categories, cluster_per_label=2, split=2)
    # evaluation on val
    run_evaluation(info_file_val, feature_file_val, weight_file=weight_file, categories=categories, cluster_per_label=2,
                   split=1)
    exit()
    # epoch_wise_evaluation(info_file_test, feature_file=feature_file_baseline, weight_file=weight_file, categories=categories,
    #                       split=1, plot_results=True, plot_labels=None)
    epoch_wise_evaluation(info_file_val, feature_file=feature_file_val, weight_file=weight_file, categories=categories,
                          split=1, plot_results=True, plot_labels=None)

    # eval_dict = evaluate_neighbor_methods(info_file_test, feature_file_baseline, k=[1, 5, 10, 50, 100])

    import pickle
    config_file = '/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode/runs/mapping/TEST_MapNet/configs/cycle_001_config.pkl'
    with open(config_file, 'r') as f:
        configs = pickle.load(f)

    for ft_f, inf_f in zip([feature_file_baseline, feature_file_val], [info_file_test, info_file_val]):
        data_dict, outdir = load_data(ft_f, inf_f, split=0)
        fts_old = data_dict['features']
        model = MapNet(feature_dim=fts_old.shape[1], output_dim=2)
        states = torch.load(weight_file, map_location='cpu')
        state_dict = states[max(states.keys())]['state_dict']
        model.load_state_dict(state_dict)
        fts_new = get_feature(model.mapping, fts_old)

        idcs = np.arange(0, len(fts_old))
        labels = np.full(len(idcs), 'none', dtype='S36')
        for group in np.sort(configs.keys()):
            labels[np.isin(idcs, configs[group])] = group

        movement = np.linalg.norm(fts_new - fts_old, axis=1)
        print(labels[np.argsort(movement)[::-1]][:100])
        print(np.where(np.isin(idcs[np.argsort(movement)[::-1]], configs['idx_modified']))[0])
        print(np.where(np.isin(idcs[np.argsort(movement)[::-1]], configs['idx_high_dim_neighbors']))[0])

        vis_ft_space_movement(fts_old, fts_new, data_dict['labels']['genre'] == 'still life')

    # check out features of missed samples in validation
    mask = data_dict['labels']['genre'] == 'still life'
    missed = np.where((movement < np.median(movement[mask])) * mask)[0]

    # check out features of wrong detected samples in validation
    wrong = np.where((movement > 0.3 * np.max(movement)) * mask.__invert__())[0]

    #visualize
    imgs = []
    for idx in missed:
        imgs.append(transforms.ToTensor()(transforms.Resize((224,224))(Image.open(os.path.join(impath, data_dict['image_names'][idx] + '.jpg')).convert('RGB'))))
    torchvision.utils.save_image(imgs, 'missed.png')
    imgs = []
    for idx in wrong:
        imgs.append(transforms.ToTensor()(transforms.Resize((224,224))(Image.open(os.path.join(impath, data_dict['image_names'][idx] + '.jpg')).convert('RGB'))))
    torchvision.utils.save_image(imgs, 'wrong.png')


    data_dict, outdir = load_data(feature_file_baseline, info_file_test, split=0)
    fts_test = data_dict['features']
    mask = data_dict['labels']['genre'] == 'still life'
    fts_test.std(axis=0).mean()
    np.concatenate([fts_test[mask], fts_old[missed]]).std(axis=0).mean()
    fts_old[missed].std(axis=0).mean()

