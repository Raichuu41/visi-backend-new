# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
import pickle
from math import sqrt
sys.path.append('/export/home/kschwarz/Documents/Masters/FullPipeline/')
from metric_learn import RCA
from sklearn.manifold import TSNE
from sklearn.cluster import k_means
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from time import time
from collections import Counter
import warnings
from itertools import combinations
from scipy.spatial.distance import euclidean
import os
from evaluate_svm import evaluate
import deepdish as dd

sys.path.append('/export/home/kschwarz/Documents/Masters/SmallNets/')
from wikiart_dataset import Wikiart

import h5py


sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod
from faiss_master import faiss
from evaluate_svms import multiclass_embedding, multiclass_embedding_rf

# DEAL WITH RANDOMNESS
seed = 123
np.random.seed(123)


svms = []


def get_areas(embedding, query_idcs, frac_margin=0.):
    """Label smallest circle/sphere that include all queries as 'inner'.
    If a margin is defined, label samples within as 'margin'. The value of margin gives fraction of added radius.
    Label remaining samples as 'outer'.
    """
    # compute center of queries
    center = np.mean(embedding[query_idcs], axis=0, keepdims=True)
    center_dist = np.linalg.norm(embedding-center, axis=1)

    radius_inner = max(center_dist[query_idcs])
    radius_margin = radius_inner + frac_margin * radius_inner

    inner = np.where(center_dist <= radius_inner)[0]
    margin = np.setdiff1d(np.where(center_dist <= radius_margin)[0], inner, assume_unique=True)
    outer = np.setdiff1d(range(len(embedding)), np.concatenate([inner, margin]), assume_unique=True)

    return inner, margin, outer



def learn_svm(positives, negatives, counter, grid_search=True):
    global features, svm_cluster, local_idcs, svms, global_negatives
    global triplet_constraints, triplet_weights, current_triplets, current_triplet_weights
    global labels
    n_positives = len(positives)
    n_negatives = len(negatives)

    print('n positives: {}\nn negatives: {}'.format(n_positives, n_negatives))
    idcs_positives = np.unique(np.array(positives, dtype=int))
    idcs_negatives = np.unique(np.array(negatives, dtype=int))

    if n_positives == 0 or n_negatives == 0:
        return list(idcs_positives), list(idcs_negatives)

    for i, idx in enumerate(idcs_positives):
        if idx in idcs_negatives:
            # compare which label was given later
            print('sample labeled controversly!')
            j = np.where(idcs_negatives == idx)[0][-1]
            if i < j:
                idcs_positives = np.delete(idcs_positives, i)
            else:
                idcs_negatives = np.delete(idcs_negatives, j)

    n_positives = len(idcs_positives)
    n_negatives = len(idcs_negatives)
    idcs_train = np.concatenate([idcs_positives, idcs_negatives])

    assert len(idcs_train) == len(np.unique(idcs_train)), 'duplicates in user selection were not filtered properly'

    # make svm local!
    # TODO: use CURRENT embedding
    positions = np.array([p for p in prev_embedding[idcs_train]])
    center = np.mean(positions, axis=0)
    d = np.array([euclidean(p, center) for p in positions])
    radius = max(d) + 0.1 * max(d)
    print('selection radius: {}'.format(radius))

    train_data = np.concatenate([features[idcs_train]])
    train_labels = np.concatenate([np.ones(n_positives), np.zeros(n_negatives)])        # positives: 1, negatives: 0

    # TODO: use CURRENT embedding
    d = np.array([euclidean(p, center) for p in prev_embedding])
    local_idcs = np.where(d <= radius)[0]

    if counter == 0:
        if grid_search:
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5, 10],
                          # 'class_weight': [{0: 1, 1: 0.2}, {0: 1, 1: 1}, {0: 1, 1: 5}]}
                          'class_weight': ['balanced']}
            svc = SVC(probability=True)                         # TODO: disable probability TRUE if unused
            clf = GridSearchCV(svc, parameters)
        else:
            clf = SVC(kernel='rbf', C=10, gamma='auto', class_weight='balanced', probability=True)  # TODO: disable probability TRUE if unused
        svms.append(clf)
        print('append previous triplet constraints')
        # save previous triplets to global ones
        if current_triplets is not None:
            if triplet_constraints is None:
                triplet_constraints = current_triplets
                triplet_weights = current_triplet_weights
            else:
                triplet_constraints = np.concatenate([triplet_constraints, current_triplets])
                triplet_weights = np.concatenate([triplet_weights, current_triplet_weights], axis=1)

    else:
        clf = svms[-1]

    print('Train SVM on user input...')
    tic = time()
    clf.fit(X=train_data, y=train_labels)
    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    if grid_search:
        print(clf.best_params_)

    print('Predict class membership for whole dataset...')
    predicted_labels = clf.predict(features)
    d_decision_boundary = clf.decision_function(features)
    labels['confidence'] = np.array(['confident' if abs(d) > 1.0 else 'unsure' for d in d_decision_boundary])
    labels['svm label'] = np.array(['svm pos' if p else 'svm neg' for p in predicted_labels])

    # save test prediction and distance to decision boundary
    with open('_svm_prediction.pkl', 'wb') as f:
        pickle.dump({'labels': predicted_labels, 'distance': d_decision_boundary,
                     'image_names': image_names, 'local_indices': local_idcs,
                     'idcs_positives_train': idcs_positives,
                     'idcs_negatives_train': idcs_negatives}, f)
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))

    # save local cluster
    svm_cluster = {'positives': np.array([p for p in np.where(predicted_labels == 1)[0] if p in local_idcs]),
                   'negatives': np.array([n for n in np.where(predicted_labels != 1)[0] if n in local_idcs]),
                   'distance': {i: d for i, d in enumerate(d_decision_boundary) if i in local_idcs},
                   'labeled': {'p': idcs_positives, 'n': idcs_negatives}}
    # do not use training data
    sort_idcs = np.argsort(np.abs(d_decision_boundary))         # use absolute distances
    predicted_labels = predicted_labels[sort_idcs]

    pred_pos = sort_idcs[predicted_labels == 1]
    pred_neg = sort_idcs[predicted_labels == 0]

    # don't return training data
    pred_pos = [p for p in pred_pos if p not in idcs_train and p in local_idcs]
    pred_neg = [n for n in pred_neg if n not in idcs_train and n in local_idcs]

    print(len(pred_pos), len(pred_neg))

    # get non local topscorer
    n_topscorer = 5
    outlier_topscorer = [idx for idx in sort_idcs[predicted_labels == 1][::-1] if idx not in local_idcs]            # TODO: MAKE SURE POSITIVE LABELS ARE 1
    outlier_topscorer = outlier_topscorer[:min(len(outlier_topscorer), n_topscorer)]

    # return most uncertain samples
    n = 5
    pos = list(pred_pos[:n])
    neg = list(pred_neg[:n])

    # also return some semi hard
    n_pos = len(pred_pos)
    n_neg = len(pred_neg)
    n_semihard = 10
    if n_pos >= 2 * n:
        if n_pos/2 >= n_semihard:
            pos += list(pred_pos[n_pos/2 : n_pos/2 + n_semihard])
        else:
            pos += list(pred_pos[n_pos/2 : -1])
    if n_neg >= 2 * n:
        if n_neg/2 >= n_semihard:
            neg += list(pred_neg[n_neg/2 : n_neg/2 + n_semihard])
        else:
            neg += list(pred_neg[n_neg/2 : -1])

    print('trained {} svms'.format(len(svms)))

    main_label = find_svm_gt(svm_cluster['labeled']['p'], svm_cluster['labeled']['n'])
    if main_label is not None:
        evaluate(ground_truth=svm_ground_truth, plot_decision_boundary=True, plot_GTE=False, compute_GTE=False)

    return pos, neg, outlier_topscorer



def get_neighborhood(data, sample_idcs, buffer=0., use_faiss=True):
    """Determine all data points within the sphere in data space defined by the samples.
    Args:
        data (np.ndarray): NxD array containing N D-dimensional data vectors
        sample_idcs (iterable ints): indices of the data points that define the sphere
        buffer (optional, float): fraction of radius which to additionally include in sphere
        use_faiss (optional, bool): whether to use faiss library for distance calculation
        """
    # get center of samples
    center = np.mean(data[sample_idcs], axis=0, keepdims=True)

    if use_faiss:
        index = faiss.IndexFlatL2(data.shape[1])  # build the index
        index.add(data.astype('float32'))  # add vectors to the index
        distances, indices = index.search(center.astype('float32'), len(data))
        distances, indices = np.sqrt(distances[0]), indices[0]          # faiss returns squared distances

        radius = max([d for d, i in zip(distances, indices) if i in sample_idcs])
        radius += buffer * radius

        local_idcs = []
        for d, i in zip(distances, indices):
            if d > radius:
                break
            local_idcs.append(i)
        local_idcs = np.array(local_idcs)

    else:
        distances = np.array([euclidean(d, center) for d in data])

        radius = max(distances[sample_idcs])
        radius += buffer * radius

        local_idcs = np.where(distances <= radius)[0]

    return local_idcs, center, radius


def local_embedding(buffer=0.):
    """Compute local low dimensional embedding.
    Args:
        buffer: fraction of radius from which to choose fix points outside of data sphere
        """
    global prev_embedding, svm_cluster, features, kwargs, triplet_constraints, triplet_weights, \
        current_triplet_weights, current_triplets, svms, labels
    global labels

    # a little out of context: save svms
    with open('_svms.pkl', 'wb') as f:
        pickle.dump(svms, f)

    current_triplets, current_triplet_weights = triplet_constraints_from_svm()

    if triplet_constraints is None:
        tc = current_triplets
        tw = current_triplet_weights

    else:
        tc = np.append(triplet_constraints, current_triplets, axis=0)
        print(triplet_weights.shape, current_triplet_weights.shape)
        tw = np.mean(np.concatenate([triplet_weights, current_triplet_weights], axis=1), axis=1)        # TODO: BETTER SYSTEM FOR COMBINING WEIGHTS

    sample_idcs = np.concatenate([svm_cluster['labeled']['p'], svm_cluster['labeled']['n']])
    local_idcs, center, radius = get_neighborhood(prev_embedding, sample_idcs, buffer=0.05, use_faiss=True)
    local_idcs_soft, _, _ = get_neighborhood(prev_embedding, sample_idcs, buffer=buffer, use_faiss=True)

    local_triplets = []
    for t in tc:
        local = True
        for i in range(tc.shape[1]):
            if t[i] not in local_idcs_soft:
                local = False
                break
        if local:
            local_triplets.append(t)
    local_triplets = np.array(local_triplets)
    print('using {} local triplets'.format(len(local_triplets)))

    # convert triplet indices to local selection
    local_idx_to_idx = {li: i for i, li in enumerate(local_idcs_soft)}
    for i, t in enumerate(local_triplets):
        for j in range(local_triplets.shape[1]):
            local_triplets[i, j] = local_idx_to_idx[t[j]]

    # get soft margin points and use them as fix points to compute embedding
    fix_points = set(local_idcs_soft).difference(local_idcs)
    fix_points = np.array([local_idx_to_idx[li] for li in fix_points], dtype=np.long)
    print('Local embedding using {} points and keeping {} fixed'.format(len(local_idcs_soft), len(fix_points)))

    # mark local indices and margin by labelling
    label = np.array(['outer'] * len(features))
    label[local_idcs_soft] = 'margin'
    label[local_idcs] = 'inner'
    labels['locality'] = label


    # TODO: USE CURRENT EMBEDDING
    # compute initialisation by shifting current embedding such that center is at origin
    initial_Y = prev_embedding[local_idcs_soft] - center

    local_kwargs = kwargs.copy()
    # TODO: CHOOSE PERPLEXITY FOR FEW POINTS ACCORDINGLY
    local_kwargs['perplexity'] = min(kwargs['perplexity'], (len(local_idcs) - 1)/3)
    assert local_kwargs['perplexity'] >= 5, 'please choose at least 14 local samples for svm'
    print('perplexity: {}'.format(local_kwargs['perplexity']))

    embedding = embedding_func(np.stack(features).astype(np.double)[local_idcs_soft],
                               triplets=local_triplets,
                               weights_triplets=tw[local_idcs_soft],
                               position_constraints=np.zeros((1, 3)),
                               fix_points=fix_points, initial_Y=initial_Y,
                               center=None, radius=radius, contrib_cost_extent=1,           # center=None because initial data is centered already
                               **local_kwargs)

    print('local center before: {} \tafter: {}'.format(center, np.mean(embedding + center, axis=0, keepdims=True)))
    prev_embedding[local_idcs_soft] = embedding + center

    # save local embedding and local triplets for evaluation of GTE
    with open('_svm_prediction.pkl', 'rb') as f:
        svm_data = pickle.load(f)
    svm_data['local_embedding'] = embedding + center
    svm_data['local_triplets'] = local_triplets
    with open('_svm_prediction.pkl', 'wb') as f:
        pickle.dump(svm_data, f)
