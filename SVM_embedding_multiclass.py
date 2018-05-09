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


sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod
sys.path.append('/export/home/kschwarz/anaconda3/envs/py27/lib/python2.7/site-packages/faiss-master/')
import faiss


seed = 123
np.random.seed(123)


def create_genre_dataset(label='styles'):
    print('Create dataset...')
    tic = time()

    # load features
    # with open('../wikiart/genre_set_inception3.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     features = data['features']
    #     ids = data['ids']

    # load labels
    with open('../wikiart/genre_set.pkl', 'rb') as f:
        data = pickle.load(f)
        image_names = data['image_names']
        genre_labels = data['genres']
        assert label in data.keys(), 'unknown label {}'.format(label)
        second_label = data[label]

    # assert (image_names == ids).all()       # ensure labelling is correct
    # del data, ids       # discard duplicate data

    # make a small subset

    label_selection = ['still life', 'portrait', 'landscape']
    # filter for these labels
    np.random.seed(123)
    N = 200
    selection_idcs = []
    for i, l in enumerate(label_selection):
        idcs = np.where(genre_labels == l)[0]
        c = Counter(second_label[idcs])
        c = sorted(c.items(), key=lambda x: x[1], reverse=True)

        lbls = []
        for j in range(len(c)):
            n_per_class = int(float(N) / (j+1))
            lbls.append(c[j][0])
            if c[j][1] >= n_per_class and j > 0:
                break
        for l in lbls:
            _idcs = np.array([idx for idx in np.where(second_label == l)[0] if idx in idcs])
            __idcs = np.random.choice(_idcs, n_per_class, replace=False)
            for j, idx in enumerate(__idcs):
                while idx in selection_idcs:
                    __idcs[j] = np.random.choice(_idcs)
            selection_idcs += list(__idcs)



    image_names = image_names[selection_idcs]
    # features = features[selection_idcs]
    with open('../wikiart/genre_set_artist.pkl', 'rb') as f:
        data = pickle.load(f)
        features = data['features']
        ids = data['ids']
    assert (image_names == ids).all(), 'loaded features do not match dataset'  # ensure labelling is correct

    genre_labels = genre_labels[selection_idcs]
    second_label = second_label[selection_idcs]

    toc = time()
    print('Done. ({:2.0f} min {:2.1f} s)'.format((toc - tic) / 60, (toc - tic) % 60))

    return {'image_names': image_names, 'features': features,
            'genre_labels': genre_labels, label + '_labels': second_label}


# GENRE DATASET
# load dataset
second_label = 'styles'
dataset = create_genre_dataset(second_label)
image_names = np.array([name.replace('.jpg', '') for name in dataset['image_names']])
features = dataset['features']
# normalize features
features = (features - np.mean(features, axis=1, keepdims=True)) / np.linalg.norm(features, axis=1, keepdims=True)

genre_labels = dataset['genre_labels']
second_labels = dataset[second_label + '_labels']
labels = {'genre_labels': genre_labels, second_label + '_labels': second_labels}

# # ARTIST DATASET
# with open('../wikiart/artist_testset.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# # load dataset
# image_names = dataset['id']
# features = dataset['features']
# # normalize features
# features = (features - np.mean(features, axis=1, keepdims=True)) / np.linalg.norm(features, axis=1, keepdims=True)
#
# genre_labels = dataset['label']
# labels = {'genre_labels': genre_labels}


# some more global variables
n_clusters = 10
n_neighbors = 1         # number of links in nearest neighbor graph
embedding_func = snack_embed_mod        # function to use for low dimensional projection / embedding
kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 1.0,
          'perplexity': 30, 'theta': 0.5, 'no_dims': 2}         # kwargs for embedding_func

prev_embedding = None
position_constraints = None
triplet_constraints = None
triplet_weights = None
graph = None
svm_cluster = None
local_idcs = None
# SVM definition
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5, 10],
              'class_weight': [{0: 1, 1: 0.2}, {0: 1, 1: 1}, {0: 1, 1: 5}]}
clf = GridSearchCV(SVC(), parameters)
svm_label = -2


if __name__ == '__main__':              # dummy main to initialise global variables
    print('global variables for embedding computation set.')


def create_graph(names, positions, label=None, labels=None):
    """Compute nearest neighbor graph with inverse distances used as edge weights ('link strength')."""
    global features, n_neighbors
    print('Compute nearest neighbor graph...')
    tic = time()

    def create_links(neighbors, distances, strenght_range=(0, 1)):
        """Computes links between data points based on nearest neighbor distance."""
        # depending on nn layout remove samples themselves from nn list
        if neighbors[0, 0] == 0:  # first column contains sample itself
            neighbors = neighbors[:, 1:]
            distances = distances[:, 1:]

        # normalize quadratic distances to strength_range to use them as link strength
        a, b = strenght_range
        dmin = distances.min() ** 2
        dmax = distances.max() ** 2
        distances = (b - a) * (distances ** 2 - dmin) / (dmax - dmin) + a

        links = {}
        for i in range(len(neighbors)):
            links[i] = {}
            for n, d in zip(neighbors[i], distances[i]):
                # # prevent double linking      # allow double linking!
                # if n < i:
                #     if i in neighbors[n]:
                #         continue
                links[i][n] = float(d)
        return links

    # compute nearest neighbors and distances with faiss library
    index = faiss.IndexFlatL2(features.shape[1])   # build the index
    index.add(np.stack(features).astype('float32'))                  # add vectors to the index
    knn_distances, knn_indices = index.search(np.stack(features).astype('float32'), n_neighbors + 1)      # add 1 because neighbors include sample itself for k=0

    # get links between the nodes
    # invert strength range because distances are used as measure and we want distant points to be weakly linked
    links = create_links(knn_indices[:, :n_neighbors+1], knn_distances[:, :n_neighbors+1], strenght_range=(1, 0))

    if label is None:
        label = [None] * len(names)

    if labels is None:
        labels = {0: [None] * len(names)}
    elif not isinstance(labels, dict):
        labels = {0: labels}

    nodes = {}
    for i, (name, (x, y)) in enumerate(zip(names, positions)):
        multi_label = [l[i] for l in labels.values()]
        nodes[i] = {'name': name, 'label': str(label[i]), 'labels': multi_label, 'x': x, 'y': y, 'links': links[i]}

    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc-tic) / 60, (toc-tic) % 60))
    return nodes


def compute_embedding():
    global features, embedding_func, prev_embedding
    """Standard embedding of reduced features."""
    print('Compute embedding...')
    tic = time()

    if position_constraints is None:
        if triplet_constraints is None:
            embedding = embedding_func(np.stack(features).astype(np.double),
                                       triplets=np.zeros((1, 3), dtype=np.long),
                                       weights_triplets=triplet_weights,
                                       position_constraints=np.zeros((1, 3)),
                                       **kwargs)
        else:
            embedding = embedding_func(np.stack(features).astype(np.double),
                                       triplets=triplet_constraints,
                                       weights_triplets=triplet_weights,
                                       position_constraints=np.zeros((1, 3)),
                                       **kwargs)
    else:
        if triplet_constraints is None:
            embedding = embedding_func(np.stack(features).astype(np.double),
                                       triplets=np.zeros((1, 3), dtype=np.long),
                                       weights_triplets=triplet_weights,
                                       position_constraints=position_constraints,
                                       **kwargs)
        else:
            embedding = embedding_func(np.stack(features).astype(np.double),
                                       triplets=triplet_constraints,
                                       weights_triplets=triplet_weights,
                                       position_constraints=position_constraints,
                                       **kwargs)
    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    prev_embedding = embedding


def format_graph(graph):
    """Change format of graph, such that keys are ints instead of strings."""
    formatted_graph = {}
    for k, v in graph.items():
        v['links'] = {int(kk): vv for kk, vv in v['links'].items()}
        formatted_graph[int(k)] = v
    return formatted_graph


def get_moved(margin=0.):
    global graph, prev_embedding
    moved = []
    if margin > 0:
        for idx, node in graph.items():
            if node['mPosition']:
                x_prev, y_prev = prev_embedding[idx]
                x, y = (node['x'], node['y'])
                d = (x-x_prev)**2 + (y-y_prev)**2
                if d > margin:
                    moved.append(idx)
    else:
        for idx, node in graph.items():
            if node['mPosition']:
                moved.append(idx)

    return np.array(moved)


def cluster_embedding(embedding, n_clusters=10, seed=123):
    """Computes k-means clustering of low dimensional embedding."""
    _, label, _ = k_means(embedding, n_clusters=n_clusters, random_state=seed)
    return np.array(label)


def compute_graph(current_graph=[]):
    global image_names, labels, n_clusters
    global graph, position_constraints, prev_embedding
    global features
    if len(current_graph) == 0 or prev_embedding is None:
        print('Initialise graph...')
        tic = time()
        compute_embedding()     # initialise prev_embedding with standard tsne

        # find clusters
        clusters = cluster_embedding(prev_embedding, n_clusters=n_clusters, seed=seed)

        graph = create_graph(image_names, prev_embedding, label=clusters, labels=labels)
        toc = time()
        print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
        print('Embedding range: x [{}, {}], y [{}, {}]'.format(prev_embedding[0].min(), prev_embedding[0].max(),
                                                               prev_embedding[1].min(), prev_embedding[1].max()))
        return graph

    # DUMMY EDIT!!!
    return create_graph(image_names, prev_embedding, labels=labels)

    print('Update graph...')
    tic = time()

    graph = format_graph(current_graph['nodes'])

    # get current embedding
    current_embedding = prev_embedding.copy()
    moved = get_moved(margin=0.0)           # nodes which have moved further than the given margin

    if len(moved) > 0:      # update positions of moved samples in embedding
        pos_moved = np.array([[graph[idx]['x'], graph[idx]['y']] for idx in moved])
        current_embedding[moved] = pos_moved

    # find clusters
    clusters = cluster_embedding(current_embedding, n_clusters=n_clusters, seed=seed)

    compute_embedding()  # update prev_embedding
    graph = create_graph(image_names, prev_embedding, label=clusters, labels=labels)

    print('Embedding range: x [{}, {}], y [{}, {}]'.format(prev_embedding[0].min(), prev_embedding[0].max(),
                                                           prev_embedding[1].min(), prev_embedding[1].max()))

    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))

    return graph


def learn_svm(positives, negatives, counter):
    global features, svm_cluster, local_idcs, clf, svm_label
    n_positives = len(positives)
    n_negatives = len(negatives)

    print('n positives: {}\nn negatives: {}'.format(n_positives, n_negatives))
    idcs_positives = np.array([p['index'] for p in positives], dtype=int)
    idcs_negatives = np.array([n['index'] for n in negatives], dtype=int)

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
    train_labels = np.concatenate([svm_label * np.ones(n_positives), (svm_label + 1) * np.ones(n_negatives)])        # new labels for new iterations

    # TODO: DEFINE HOW TO DEAL WITH MULTI-LABELS
    if counter == 0:
        svm_label += 2  # new labels for new iterations
        if hasattr(clf.best_estimator_, 'support_'):  # load old training data if existent
            print('load previous svm training data with labels {} - now start labelling from {}'
                  .format(range(len(clf.best_estimator_.n_support_)), svm_label))
            assert len(clf.best_estimator_.n_support_) == svm_label, 'counter in labels went wrong somewhere'
            train_data.append(clf.best_estimator_.support_vectors_)
            for l, n in enumerate(clf.best_estimator_.n_support_):
                train_labels.append(l * np.ones(n))

    # TODO: use CURRENT embedding
    d = np.array([euclidean(p, center) for p in prev_embedding])
    local_idcs = np.where(d <= radius)[0]

    print('Train SVM on user input...')
    tic = time()
    clf.fit(X=train_data, y=train_labels)
    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    if hasattr(clf, 'best_params_'):
        print(clf.best_params_)

    print('Predict class membership for whole dataset...')
    predicted_labels = clf.predict(features)
    d_decision_boundary = clf.decision_function(features)
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
    return pos, neg


# def position_constraints_from_svm():
#     n_max = 300
#     print('Generate position constraints from SVM cluster...')
#     tic = time()
#     global svm_cluster, position_constraints
#     # get all permutations
#     constraints = np.array(list(combinations(svm_cluster['index'], 2)), dtype=long)
#     distances = np.array(list(combinations(svm_cluster['distance'], 2)), dtype=float)
#     # add distances of both points quadratically to obtain weight
#     weights = np.linalg.norm(distances, axis=1, keepdims=True)
#     constraints = np.concatenate([constraints, weights], axis=1)
#
#     if len(constraints) > n_max:      # use at most 300 constraints
#         np.random.seed(seed)
#         idcs = np.random.choice(range(len(constraints)), n_max, replace=False)
#         constraints = constraints[idcs]
#
#     if position_constraints is None:
#         position_constraints = constraints
#     else:
#         position_constraints = np.append(position_constraints, constraints, axis=0)
#     toc = time()
#     print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
#     print('Created {} constraints.'.format(len(constraints)))


def generate_triplets(positives, negatives, N, n_pos_pa=1, n_neg_pp=1, seed=123):
    np.random.seed(seed)

    n_pos_pa = min(n_pos_pa, len(positives) - 1)
    n_neg_pp = min(n_neg_pp, len(negatives) - 1)
    if n_pos_pa <= 0 or n_neg_pp <= 0:
        return np.array([], dtype=long)
    N_anchors = min(int(N * 1.0/(n_neg_pp * n_pos_pa)), len(positives))
    N_tot = N_anchors * n_pos_pa * n_neg_pp

    if N != N_tot:
        warnings.warn('Too few data to generate {} triplets. Instead generate {} triplets using:\n'
                      '{} anchors, {} positives per anchor, {} negatives per positive'.format(
            N, N_tot, N_anchors, n_pos_pa, n_neg_pp
        ), RuntimeWarning)
        N = N_tot

    triplets = np.empty((N, 3), dtype=np.long)

    anchors = np.random.choice(positives, N_anchors, replace=False)
    for i, a in enumerate(anchors):
        pos = np.random.choice(np.delete(positives, np.where(positives == a)[0][0]), n_pos_pa, replace=False)
        for j, p in enumerate(pos):
            neg = np.random.choice(negatives, n_neg_pp, replace=False)
            t = np.stack([np.repeat(a, n_neg_pp), np.repeat(p, n_neg_pp), neg], axis=1)
            i_start = (i * n_pos_pa + j) * n_neg_pp
            triplets[i_start:i_start + n_neg_pp] = t

    return triplets


def triplet_constraints_from_svm():
    print('Generate triplet constraints from SVM cluster...')
    tic = time()
    global svm_cluster, features

    # predicted data
    n_constraints = 2000
    n_positives_pa = 5
    n_negatives_pp = 20     # 20 negatives per positive

    # labeled data
    n_constraints_labeled = 200
    n_positives_pa_labeled = 5
    n_negatives_pp_labeled = 10  # 10 negatives per positive

    positives = svm_cluster['positives']
    negatives = svm_cluster['negatives']
    distances = svm_cluster['distance']
    positives_labeled = svm_cluster['labeled']['p']
    negatives_labeled = svm_cluster['labeled']['n']

    # sample constraints from user labeled data
    constraints = generate_triplets(positives_labeled, negatives_labeled,
                                    N=n_constraints_labeled,
                                    n_pos_pa=n_positives_pa_labeled, n_neg_pp=n_negatives_pp_labeled,
                                    seed=seed)
    constraints = np.append(constraints,
                            generate_triplets(positives, negatives,
                                              N=n_constraints, n_pos_pa=n_positives_pa, n_neg_pp=n_negatives_pp,
                                              seed=seed),
                            axis=0)

    triplet_weights = np.zeros(len(features))
    for k, v in distances.items():
        if k in np.concatenate([positives_labeled, negatives_labeled]):
            triplet_weights[k] = max(distances.values())
        else:
            triplet_weights[k] = np.abs(v)

    r = (0.5, 2)
    triplet_weights = (r[1]-r[0]) * (np.array(triplet_weights) - min(triplet_weights)) / \
                      (max(triplet_weights) - min(triplet_weights)) + r[0]

    print('WEIGHTS: min: {}, max: {}, mean: {}'.format(min(triplet_weights), max(triplet_weights),
                                                       triplet_weights.mean()))

    triplet_constraints = np.array(constraints, dtype=long)


    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
    print('Created {} triplet constraints.'.format(len(constraints)))

    return triplet_constraints, triplet_weights


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

        radius = max(distances)
        radius += buffer * radius

        local_idcs = []
        for d, i in zip(distances, indices):
            if d > radius:
                break
            local_idcs.append(i)
        local_idcs = np.array(local_idcs)

    else:
        distances = np.array([euclidean(d, center) for d in data])

        radius = max(distances)
        radius += buffer * radius

        local_idcs = np.where(distances <= radius)[0]

    return local_idcs, center, radius


def local_embedding(buffer=0.):
    """Compute local low dimensional embedding.
    Args:
        buffer: fraction of radius from which to choose fix points outside of data sphere
        """
    global prev_embedding, svm_cluster, features, kwargs
    triplet_constraints, triplet_weights = triplet_constraints_from_svm()
    sample_idcs = np.concatenate([svm_cluster['labeled']['p'], svm_cluster['labeled']['n']])
    local_idcs, _, radius = get_neighborhood(prev_embedding, sample_idcs, buffer=0.05, use_faiss=True)
    local_idcs_soft, _, _ = get_neighborhood(prev_embedding, sample_idcs, buffer=buffer, use_faiss=True)

    # convert triplet indices to local selection
    local_idx_to_idx = {li: i for i, li in enumerate(local_idcs_soft)}
    for i, t in enumerate(triplet_constraints):
        for j in range(triplet_constraints.shape[1]):
            triplet_constraints[i, j] = local_idx_to_idx[t[j]]

    # get soft margin points and use them as fix points to compute embedding
    fix_points = set(local_idcs_soft).difference(local_idcs)

    embedding = embedding_func(np.stack(features).astype(np.double)[local_idcs_soft],
                               triplets=triplet_constraints,
                               weights_triplets=triplet_weights,
                               position_constraints=np.zeros((1, 3)),
                               fix_points=fix_points, initial_Y=prev_embedding[local_idcs_soft],
                               radius=radius, contrib_cost_extent=1,
                               **kwargs)
    # update embedding
    prev_embedding[local_idcs_soft] = embedding


