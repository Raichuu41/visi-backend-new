# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import sys
import numpy as np
import pickle
from math import sqrt

sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod
sys.path.append('/export/home/kschwarz/anaconda3/envs/py27/lib/python2.7/site-packages/faiss-master/')
import faiss


colors = ['magenta', 'cyan', 'lime', 'indigo', 'y',
          'lightseagreen', 'dodgerblue', 'coral', 'orange', 'mediumpurple']

# testset = pickle.load(open('../wikiart/artist_testset.pkl', 'rb'))
print('loading testset...')
testset = pickle.load(open('../wikiart/style_testset_tiny.pkl', 'rb'))
print('done.')
ids = testset['id']
labels = testset['label']
features = testset['features']

n_neighbors = 5
embedding_func = snack_embed_mod
kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 0.1,
          'perplexity': 30, 'theta': 0.5, 'no_dims': 2}


# get nearest neighbors once
index = faiss.IndexFlatL2(np.stack(features).shape[1])   # build the index
index.add(np.stack(features).astype('float32'))                  # add vectors to the index
knn_distances, knn_indices = index.search(np.stack(features).astype('float32'), n_neighbors+1)      # add 1 because neighbors include sample itself for k=0

prev_embedding = None
triplets = np.zeros((1, 3)).astype(np.long)
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
    global kwargs
    print('compute embedding...')
    embedding = embedding_func(np.stack(features).astype(np.double),
                               triplets=triplets, position_constraints=position_constraints,
                               **kwargs)
    print('done.')
    prev_embedding = embedding.copy()
    return create_nodes(ids, embedding, labels)


def get_triplets(query, prev_links, curr_links):
    # use link strength to infer triplet constraints

    # get removed, mutual and added links
    removed = {k: v for k, v in prev_links.items() if k not in curr_links.keys()}
    mutual = {k: v for k, v in prev_links.items() if k in curr_links.keys()}
    added = {k: v for k, v in curr_links.items() if k not in prev_links.keys()}

    positives = mutual.copy()
    positives.update(added)
    # sort by link strength
    positives = sorted(positives.items(), key=lambda x: x[1], reverse=True)

    if len(positives) > 1:
        # get n_pos random triplets within current neighbors
        n_pos = 5
        similars = np.random.randint(0, len(positives) - 1, n_pos)
        dissimilars = [np.random.randint(s + 1, len(positives)) for s in similars]

        triplets = [[query, s, d] for s, d in zip(similars, dissimilars)]

        # get two positive triplets for each added link
        for k, v in added.items():
            s = positives.index((k, v))
            if s == len(positives) - 1:
                continue
            dissimilars = np.random.randint(s + 1, len(positives), 2)
            for d in dissimilars:
                triplets.append([query, s, d])

    if len(positives) > 0:
        # get two negative triplets for each removed link
        for k in removed.keys():
            similars = np.random.randint(0, len(positives), 2)
            for s in similars:
                triplets.append([query, s, k])

    return np.stack(triplets)


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

    if len(current_graph) == 0 or prev_embedding is None:
        print('initialise graph')
        graph = initialise_graph()
        print('Embedding range: x [{}, {}], y [{}, {}]'.format(prev_embedding[0].min(), prev_embedding[0].max(),
                                                               prev_embedding[1].min(), prev_embedding[1].max()))
        return graph

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
        position_constraints = np.vstack((position_constraints, pos_constraints))
        print('added {} position constraints'.format(len(pos_constraints)))
        print(pos_constraints)

    # compute triplets from user modifications
    n_triplets = 0
    for idx in modified_links:
        prev_links = graph[idx]['links']
        curr_links = current_graph[idx]['links']

        if len(curr_links) > 0:
            trplts = get_triplets(idx, prev_links, curr_links)
            n_triplets += len(trplts)
            triplets = np.vstack((triplets, trplts))
        # update links in graph
        graph[idx]['links'] = current_graph[idx]['links']
    print('added {} triplets'.format(n_triplets))


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

    return graph


