# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
import pickle
from math import sqrt


def GTE(embedding, triplets):
    error = 0
    for anchor, pos, neg in triplets.astype(long):
        # compute squared euclidean distance in embedding
        x0, y0 = embedding[anchor]
        xp, yp = embedding[pos]
        xn, yn = embedding[neg]

        dp = (xp - x0)**2 + (yp - y0)**2
        dn = (xn - x0)**2 + (yn - y0)**2

        if dp >= dn:
            error += 1
    return float(error) / len(triplets)


def generate_triplets(labels, n=10):
    label_names = set(labels)
    labels = np.array(labels)
    n_per_class = np.floor(float(n) / len(label_names))
    triplets = []
    for i, l in enumerate(label_names):
        similars = np.where(labels == l)[0]
        differents = np.where(labels != l)[0]
        while len(triplets) < (i+1) * n_per_class:
            a, s = np.random.choice(similars, 2, replace=False)
            d = np.random.choice(differents)
            if (a, s, d) not in triplets:
                triplets.append((a, s, d))
    # fill up rest of triplets with random choices
    while len(triplets) < n:
        l = np.random.choice(list(label_names))
        a, s = np.random.choice(np.where(labels == l)[0], 2, replace=False)
        d = np.random.choice(np.where(labels != l)[0])
        if (a, s, d) not in triplets:
            triplets.append((a, s, d))

    return np.stack(triplets).astype(long)


sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod
sys.path.append('/export/home/kschwarz/anaconda3/envs/py27/lib/python2.7/site-packages/faiss-master/')
import faiss

import matplotlib.pyplot as plt
plt.ion()


colors = ['magenta', 'cyan', 'lime', 'indigo', 'y',
          'lightseagreen', 'dodgerblue', 'coral', 'orange', 'mediumpurple']

print('loading testset...')
# testset = pickle.load(open('../wikiart/style_testset_tiny.pkl', 'rb'))
# testset = pickle.load(open('../wikiart/style_testset_easy.pkl', 'rb'))
testset = pickle.load(open('../wikiart/artist_testset.pkl', 'rb'))
# testset = pickle.load(open('../wikiart/style_testset_small.pkl', 'rb'))
# testset = pickle.load(open('../wikiart/portrait_subset.pkl', 'rb'))


print('done.')
ids = testset['id']
labels = testset['label']
# labels = [None] * len(ids)

# label_to_int = {'cubism': 0, 'impressionism': 1, 'surrealism': 2}
# clabels = [colors[label_to_int[l]] for l in labels]

features = testset['features']
# with open('../wikiart/portrait_subset_vgg16.pkl', 'rb') as infile:
#     features = pickle.load(infile)


# sample test triplets
test_triplets = generate_triplets(labels, n=50)

n_neighbors = 1
embedding_func = snack_embed_mod
kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 0.1,
          'perplexity': 30, 'theta': 0.5, 'no_dims': 2}

# get nearest neighbors once
index = faiss.IndexFlatL2(np.stack(features).shape[1])   # build the index
index.add(np.stack(features).astype('float32'))                  # add vectors to the index
knn_distances, knn_indices = index.search(np.stack(features).astype('float32'), n_neighbors+1)      # add 1 because neighbors include sample itself for k=0

prev_embedding = None
triplets = np.zeros((1, 3)).astype(np.long)
triplet_error = []
position_constraints = np.zeros((1, 3))
graph = None


def create_links(neighbors, distances, strenght_range=(0, 1)):
    # depending on nn layout remove samples themselves from nn list
    if neighbors[0, 0] == 0:        # first column contains sample itself
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]

    # normalize quadratic distances to strength_range to use them as link strength
    a, b = strenght_range
    dmin = distances.min()**2
    dmax = distances.max()**2
    distances = (b - a) * (distances**2 - dmin) / (dmax - dmin) + a

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


def create_nodes(names, positions, labels=None):
    # get links between the nodes
    # invert strength range because distances are used as measure and we want distant points to be weakly linked
    links = create_links(knn_indices[:, :n_neighbors+1], knn_distances[:, :n_neighbors+1], strenght_range=(1, 0))

    if labels is None:
        labels = [None] * len(names)

    nodes = {}
    for i, (name, label, (x, y)) in enumerate(zip(names, labels, positions)):
        nodes[i] = {'name': name, 'label': label, 'x': x, 'y': y, 'links': links[i]}
    return nodes


def initialise_graph():
    """Compute standard TSNE embedding and save it as graph using nn as link strengths."""
    global prev_embedding
    global triplets
    global position_constraints
    global kwargs
    global test_triplets
    global triplet_error

    # reset global variables
    prev_embedding = None
    triplets = np.zeros((1, 3)).astype(np.long)
    triplet_error = []
    position_constraints = np.zeros((1, 3))

    print('compute embedding...')

    embedding = embedding_func(np.stack(features).astype(np.double),
                               triplets=triplets, position_constraints=position_constraints,
                               **kwargs)
    print('done.')
    prev_embedding = embedding.copy()

    # evaluate GTE
    triplet_error.append(GTE(embedding, test_triplets))
    print('GTE: {}% of test triplets violated.'.format(triplet_error[-1] * 100))
    with open('_err_tracking.pkl', 'wb') as outfile:
        pickle.dump(triplet_error, outfile)

    return create_nodes(ids, embedding, labels)


def get_triplets(query, similars, differents, multiply=1, seed=123):
    np.random.seed(seed)
    triplets = []
    assert len(similars) > 0 and len(differents) > 0, 'Need at least one similar and one different sample for query.'

    # get multiply triplets for each sample (similar and different)
    for s in similars:
        diff = np.random.choice(differents, multiply, replace=False)
        for d in diff:
            triplets.append((query, s, d))
    for d in differents:
        sim = np.random.choice(similars, multiply, replace=False)
        for s in sim:
            triplets.append((query, s, d))

    return np.stack(triplets).astype(np.long)


def generate_triplets(n=100, seed=123):
    global labels
    np.random.seed(seed)
    label_names = set(labels)
    labels = np.array(labels)
    n_per_class = np.floor(float(n) / len(label_names))
    triplets = []
    for i, l in enumerate(label_names):
        similars = np.where(labels == l)[0]
        differents = np.where(labels != l)[0]
        while len(triplets) < (i+1) * n_per_class:
            a, s = np.random.choice(similars, 2, replace=False)
            d = np.random.choice(differents)
            if (a, s, d) not in triplets:
                triplets.append((a, s, d))
    # fill up rest of triplets with random choices
    while len(triplets) < n:
        l = np.random.choice(list(label_names))
        a, s = np.random.choice(np.where(labels == l)[0], 2, replace=False)
        d = np.random.choice(np.where(labels != l)[0])
        if (a, s, d) not in triplets:
            triplets.append((a, s, d))

    return np.stack(triplets).astype(long)


def get_pos_constraints(indices, embedding, k=10):
    # compute k nearest neighbors in embedding
    index = faiss.IndexFlatL2(embedding.shape[1])  # build the index
    index.add(np.stack(embedding).astype('float32'))  # add vectors to the index
    positions = embedding[np.array(indices).astype(int)]
    knn_distances, knn_indices = index.search(positions.astype('float32'), k + 1)

    # use normalized inverse distance as weight
    weights = -knn_distances / knn_distances.max() + 1

    pos_constraints = []
    for idx, neighbors, weight in zip(knn_indices[:, 0], knn_indices[:, 1:], weights[:, 1:]):
        for n, w in zip(neighbors, weight):
            pos_constraints.append([idx, n, w])
    return np.stack(pos_constraints)


def compute_graph(current_graph=[]):
    global prev_embedding
    global graph
    global triplets
    global position_constraints
    global kwargs
    global test_triplets
    global triplet_error

    if len(current_graph) == 0 or prev_embedding is None:
        print('initialise graph')
        graph = initialise_graph()
        print('Embedding range: x [{}, {}], y [{}, {}]'.format(prev_embedding[0].min(), prev_embedding[0].max(),
                                                               prev_embedding[1].min(), prev_embedding[1].max()))
        return graph

    new_triplets = current_graph['tripel']
    current_graph = current_graph['nodes']

    print('update graph')
    # modify current graph such that keys are ints and links keys are ints
    tmp_graph = {}
    for k, v in current_graph.items():
        v['links'] = {int(kk): vv for kk, vv in v['links'].items()}
        tmp_graph[int(k)] = v
    current_graph = tmp_graph.copy()
    del tmp_graph

    # get current embedding after user modification
    current_embedding = prev_embedding.copy()
    modified_pos = []
    modified_links = []
    for idx, node in current_graph.items():
        if node['mPosition']:
            current_embedding[idx, 0] = node['x']
            current_embedding[idx, 1] = node['y']
            modified_pos.append(idx)

        if node['mLinks']:
            modified_links.append(idx)

    print('modified nodes:\nPosition: {}\nLinks:{}'.format(modified_pos, modified_links))

    # ADD NEW CONSTRAINTS
    # fix relative neighborhood of samples who's position changed
    if modified_pos:
        pos_constraints = get_pos_constraints(modified_pos, current_embedding)

        # reset existing constraints of sample
        samples = set(pos_constraints[:, 0])
        for s in samples:
            row_idx = np.where(position_constraints == s)[0]
            position_constraints = np.delete(position_constraints, row_idx, axis=0)

        # add new position constraints
        position_constraints = np.vstack((position_constraints, pos_constraints))
        print('added {} position constraints'.format(len(pos_constraints)))

    # compute triplets from user modifications
    # if len(new_triplets) > 0:
    #     if (triplets[0] == np.zeros((1, 3))).all():  # delete dummy
    #         triplets = np.delete(triplets, 0, axis=0)
    #
    #     for query in new_triplets.keys():
    #         trplts = get_triplets(query, new_triplets[query]['p'], new_triplets[query]['n'])
    #         triplets = np.vstack((triplets, trplts))
    #
    #     print('added {} triplets, total: {}'.format(len(trplts), len(triplets)))

    # generate triplets
    triplets = generate_triplets(n=400)

    # compute embedding with current embedding as initialization
    kwargs['initial_Y'] = current_embedding

    print('compute embedding...')
    embedding = embedding_func(np.stack(features).astype(np.double),
                               triplets=triplets, position_constraints=position_constraints,
                               **kwargs)
    print('done.')
    prev_embedding = embedding.copy()

    # update position of nodes in graph
    for idx, (x, y) in enumerate(embedding):
        graph[idx].update({'x': x, 'y': y})

    # evaluate GTE
    if not (triplets[0] == np.zeros((1, 3))).all():
        train_triplet_error = GTE(embedding, triplets)
        print('User generated {} triplets, of which {}% are violated.'.format(len(triplets), train_triplet_error * 100))
    triplet_error.append(GTE(embedding, test_triplets))
    print('GTE: {}% of test triplets violated.'.format(triplet_error[-1] * 100))
    with open('_err_tracking.pkl', 'wb') as outfile:
        pickle.dump(triplet_error, outfile)

    return graph

