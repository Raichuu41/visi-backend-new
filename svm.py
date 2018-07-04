# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from time import time
from globals import get_globals, dict_to_df
from eval_svm import evaluate
import pickle


globals = get_globals()


# DEAL WITH RANDOMNESS
seed = 123
np.random.seed(123)


svms = []
triplets = None
triplet_weights = None
with open('_svm_labels.pkl', 'wb') as f:
    pickle.dump({'labels': np.array([]), 'confidence': np.array([])}, f)


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


def train_svm(positives, negatives, counter, grid_search=True):
    global globals, svms
    print('Called train_local_svm.')
    print('\tPositives: {}\n\t{}'.format(positives, globals.labels[positives, 2]))
    print('\tNegatives: {}\n\t{}'.format(negatives, globals.labels[negatives, 2]))
    # format input data and correct input if necessary
    idcs_positives = np.unique(np.array(positives, dtype=int))
    idcs_negatives = np.unique(np.array(negatives, dtype=int))

    controversely_labeled = np.intersect1d(idcs_positives, idcs_negatives, assume_unique=True)
    if len(controversely_labeled) > 0:
        print('\tSamples {} are labeled controversly. Exclude them from training.'
              .format(controversely_labeled))
        idcs_positives = np.setdiff1d(idcs_positives, controversely_labeled)
        idcs_negatives = np.setdiff1d(idcs_negatives, controversely_labeled)

    n_positives = len(idcs_positives)
    n_negatives = len(idcs_negatives)
    print('\tn positives: {}\n\tn negatives: {}'.format(n_positives, n_negatives))

    if n_positives == 0 or n_negatives == 0:            # do not train svm
        print('\tNo positives or negatives provided. Do not train local SVM.')
        return list(idcs_positives), list(idcs_negatives)

    idcs_train = np.concatenate([idcs_positives, idcs_negatives])

    # initialise SVM
    if grid_search:
        parameters = {'kernel': ('linear', 'rbf'),
                      'class_weight': [{0: 1, 1: 0.2}, {0: 1, 1: 1}, {0: 1, 1: 5}, 'balanced']}
        svc = SVC(probability=True, C=5)                         # TODO: disable probability TRUE if unused
        clf = GridSearchCV(svc, parameters)
    else:
        clf = SVC(kernel='rbf', C=10, gamma='auto', class_weight='balanced', probability=True)  # TODO: disable probability TRUE if unused
    if counter == 0:        # add new SVM to list
        svms.append(clf)
    else:       # overwrite SVM from previous round
        svms[-1] = clf

    # train SVM locally
    train_data = np.stack(globals.features[idcs_train])
    train_labels = np.concatenate([np.ones(n_positives), np.zeros(n_negatives)])  # positives: 1, negatives: 0
    print('\tTrain SVM on user input...')
    tic = time()
    clf.fit(X=train_data, y=train_labels)
    toc = time()
    print('\tDone. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    if grid_search:
        print('\tBest parameters: {}'.format(clf.best_params_))

    return idcs_train


def svm_predict(local_idcs):
    global globals, svms
    clf = svms[-1]
    pred = clf.predict(globals.features[local_idcs])  # this shifts the indices from global to local
    dist = clf.decision_function(globals.features[local_idcs])
    return pred, dist


def make_suggestion(d_decision_boundary, idcs_train, n_hard=5, n_semihard=10, threshold=1.0):
    positives = np.where(d_decision_boundary > 0)[0]  # TODO: check if positives really always have d_decision_boundary > 0
    negatives = np.where(d_decision_boundary < 0)[0]

    suggestion = []
    for idcs in [positives, negatives]:
        idcs = np.setdiff1d(idcs, idcs_train)       # do not return samples which have already been labeled
        conf = np.abs(d_decision_boundary)[idcs]
        idcs = idcs[np.argsort(conf)]

        idcs_hard = idcs[:min(n_hard, len(idcs))]

        thresh_idx = np.where(conf > threshold)[0]
        if len(thresh_idx) == 0:
            thresh_idx = len(idcs) - 1

        start_idx = thresh_idx - n_semihard
        if start_idx < n_hard:
            start_idx = min(n_hard, len(idcs))
        end_idx = min(start_idx + n_semihard, len(idcs))
        idcs_semihard = idcs[start_idx:end_idx]
        suggestion.append(np.concatenate([idcs_hard, idcs_semihard]))

    return list(suggestion[0]), list(suggestion[1])


def svm_iteration(positives, negatives, counter, current_graph, grid_search=True, frac_margin=0.1,
                  plot_eval=False):
    global globals
    train_idcs = train_svm(positives, negatives, counter, grid_search)
    nodes = dict_to_df(current_graph['nodes'])
    embedding = np.stack([nodes['x'], nodes['y']]).transpose()
    # get local indices
    local_idcs, margin_idcs, _ = get_areas(embedding, train_idcs, frac_margin)
    idcs = np.concatenate([local_idcs, margin_idcs])

    # evaluate SVM on local area and margin
    predicted = -1 * np.ones(len(embedding), dtype=int)
    d_decision_boundary = np.zeros(len(embedding))
    predicted[idcs], d_decision_boundary[idcs] = svm_predict(idcs)

    if plot_eval:
        evaluate(local_idcs, train_idcs, predicted,
                 d_decision_boundary, embedding, plot_GTE=True)
    else:
        evaluate(local_idcs, train_idcs, predicted)

    return make_suggestion(d_decision_boundary, train_idcs, n_hard=5, n_semihard=10, threshold=10)


def generate_triplets(positives, negatives, nppa=1, nnpp=1, seed=seed):
    np.random.seed(seed)
    nppa = min(len(positives), nppa)        # positives per anchor
    nnpp = min(len(negatives), nnpp)        # negatives per positive
    tpa = nppa * nnpp       # triplets per anchor

    triplets = np.empty((len(positives) * tpa, 3), dtype=long)        # use each positive as anchor once
    for i in range(len(positives)):
        a = positives[i]
        pos = np.random.choice(np.delete(positives, i), nppa, replace=False)
        neg = np.concatenate([np.random.choice(negatives, nnpp, replace=False) for j in range(nppa)])

        triplets[i*tpa:(i+1)*tpa] = np.stack([a.repeat(tpa), pos.repeat(nnpp), neg]).transpose()
    return triplets


def add_triplets_from_svm(svm_labels, nppa=1, nnpp=1):
    """Generate triplet constraints using the SVM prediction."""
    global triplets
    # use SVM prediction to generate the triplet constraints
    new_triplets = generate_triplets(np.where(svm_labels == 1)[0], np.where(svm_labels == 0)[0], nppa, nnpp, seed)

    # add the new triplets to the global ones
    if triplets is None:
        triplets = new_triplets
    else:
        triplets = np.concatenate([triplets, new_triplets])

    print('Added {} triplets. Total number of triplets {}'.format(len(new_triplets), len(triplets)))


def add_triplet_weights(d_decision_boundary, user_labeled_idcs ,value_range=(0, 1)):
    # for each sample compute its weight by distance to decision boundary
    # if sample occurs in more than one svm, average the values
    global triplet_weights
    weights = np.abs(d_decision_boundary)

    # normalize weights to value range
    idcs = np.where(weights != 0)[0]
    norm_weights = weights[idcs]
    norm_weights = (value_range[1] - value_range[0]) * (norm_weights - min(norm_weights) /
                                                        (max(norm_weights) - min(norm_weights)) + value_range[0])
    weights = np.zeros(len(weights))
    weights[idcs] = norm_weights
    weights[user_labeled_idcs] = value_range[1]         # give user labeled samples maximum weight

    if triplet_weights is None:
        triplet_weights = weights
    else:
        avg_weights = np.mean(np.stack([triplet_weights, weights]), axis=0)
        triplet_weights = np.where(triplet_weights == 0, weights, avg_weights)


def local_embedding(local_idcs, margin_idcs, embedding):
    """Compute a local TSNE embedding and keep the points in the margin fixed."""
    global triplets, triplet_weights, globals
    idcs = np.concatenate([local_idcs, margin_idcs])

    # initialize local embedding and center it
    center = np.mean(embedding[idcs], axis=0, keepdims=True)
    radius_inner = max(np.linalg.norm(embedding[local_idcs]-center, axis=1))
    initial_Y = embedding[idcs] - center

    # filter triplets for pure local constraints
    local_triplets = triplets[np.all(np.isin(triplets.flatten(), idcs).reshape(triplets.shape), axis=1)]

    # map global indices to local indices
    glob_to_loc = {glob_idx: loc_idx for loc_idx, glob_idx in enumerate(idcs)}
    local_triplets = np.array(map(lambda x: glob_to_loc[x], local_triplets.flatten())).reshape(local_triplets.shape)
    local_fix_points = np.array(map(lambda x: glob_to_loc[x], margin_idcs))

    # compute new local embedding
    print('Compute local embedding using {} local triplets...'.format(len(local_triplets)))
    tic = time()
    kwargs = globals.embedding_func_kwargs.copy()
    kwargs['perplexity'] = min(kwargs['perplexity'], (len(idcs) - 1) / 3)
    assert kwargs['perplexity'] >= 5, '\tplease choose at least 14 local samples for svm'
    print('\tperplexity: {}'.format(kwargs['perplexity']))
    embedding = globals.embedding_func(globals.features[idcs].astype(np.double),
                               triplets=local_triplets,
                               weights_triplets=triplet_weights[idcs],
                               position_constraints=np.zeros((1, 3)),        # dummy values
                               fix_points=local_fix_points,
                               radius=radius_inner, center=None, initial_Y=initial_Y,
                               contrib_cost_extent=1,
                               **kwargs)
    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    return embedding + center


def local_update(current_graph, local_queries, frac_margin=0.1):
    nodes = dict_to_df(current_graph['nodes'])
    embedding = np.stack([nodes['x'], nodes['y']]).transpose()
    # get local indices
    local_idcs, margin_idcs, _ = get_areas(embedding, local_queries, frac_margin)
    idcs = np.concatenate([local_idcs, margin_idcs])

    # evaluate SVM on local area and margin
    predicted = -1 * np.ones(len(embedding), dtype=int)
    d_decision_boundary = np.zeros(len(embedding))
    predicted[idcs], d_decision_boundary[idcs] = svm_predict(idcs)
    print(idcs)
    print(predicted[idcs])
    add_triplets_from_svm(predicted, nppa=5, nnpp=8)
    add_triplet_weights(d_decision_boundary, local_queries, value_range=(0.5, 2))

    local_emb = local_embedding(local_idcs, margin_idcs, embedding)
    embedding[idcs] = local_emb

    return embedding, list(np.setdiff1d(np.where(predicted == 1)[0], margin_idcs))

