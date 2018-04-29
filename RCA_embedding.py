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
from sklearn.decomposition import PCA
from time import time
from collections import Counter
import warnings

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
    with open('../wikiart/genre_set_inception3.pkl', 'rb') as f:
        data = pickle.load(f)
        features = data['features']
        ids = data['ids']

    # load labels
    with open('../wikiart/genre_set.pkl', 'rb') as f:
        data = pickle.load(f)
        image_names = data['image_names']
        genre_labels = data['genres']
        assert label in data.keys(), 'unknown label {}'.format(label)
        second_label = data[label]

    assert (image_names == ids).all()       # ensure labelling is correct
    del data, ids       # discard duplicate data

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
    features = features[selection_idcs]
    genre_labels = genre_labels[selection_idcs]
    second_label = second_label[selection_idcs]

    toc = time()
    print('Done. ({:2.0f} min {:2.1f} s)'.format((toc - tic) / 60, (toc - tic) % 60))

    return {'image_names': image_names, 'features': features,
            'genre_labels': genre_labels, label + '_labels': second_label}


# load dataset
second_label = 'styles'
dataset = create_genre_dataset(second_label)
image_names = np.array([name.replace('.jpg', '') for name in dataset['image_names']])
features = dataset['features']
genre_labels = dataset['genre_labels']
second_labels = dataset[second_label + '_labels']
labels = {'genre_labels': genre_labels, second_label + '_labels': second_labels}
label_to_int = {l: i for i, l in enumerate(second_labels)}
int_labels = np.array([label_to_int[l] for l in second_labels])

# reduce dimensionality
d = 128
print('Reduce features to {} dimensions...'.format(d))
tic = time()
pca = PCA(n_components=d)
fts_reduced = pca.fit_transform(features)
toc = time()
print('Done. ({:2.0f}min {:2.1f}s)'.format((toc-tic) / 60, (toc-tic) % 60))
if fts_reduced.shape[1] != d:
    warnings.warn('Too little number of features for d={} - set d to {}'.format(d, fts_reduced.shape[1]), Warning)
    d = fts_reduced.shape[1]

# some more global variables
n_clusters = 10
n_neighbors = 1         # number of links in nearest neighbor graph
embedding_func = snack_embed_mod        # function to use for low dimensional projection / embedding
kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0., 'contrib_cost_position': 0.0,
          'perplexity': 30, 'theta': 0.5, 'no_dims': 2}         # kwargs for embedding_func

prev_embedding = None
position_constraints = None
graph = None


if __name__ == '__main__':              # dummy main to initialise global variables
    print('global variables for embedding computation set.')


def create_graph(names, positions, label=None, labels=None):
    """Compute nearest neighbor graph with inverse distances used as edge weights ('link strength')."""
    global fts_reduced, n_neighbors, d
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
    index = faiss.IndexFlatL2(d)   # build the index
    index.add(np.stack(fts_reduced).astype('float32'))                  # add vectors to the index
    knn_distances, knn_indices = index.search(np.stack(fts_reduced).astype('float32'), n_neighbors+1)      # add 1 because neighbors include sample itself for k=0

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
    global fts_reduced, embedding_func, prev_embedding
    """Standard embedding of reduced features."""
    print('Compute embedding...')
    tic = time()

    if position_constraints is None:
        embedding = embedding_func(np.stack(fts_reduced).astype(np.double),
                                   triplets=np.zeros((1, 3), dtype=np.long),
                                   position_constraints=np.zeros((1, 3)),
                                   **kwargs)
    else:
        embedding = embedding_func(np.stack(fts_reduced).astype(np.double),
                                   triplets=np.zeros((1, 3), dtype=np.long),
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


def make_chunks(clusters, chunk_size, chunk_size_relevant=None, relevant_clusters=[]):
    if chunk_size_relevant is None:
        chunk_size_relevant = 2 * chunk_size

    chunks = -1 * np.ones(len(clusters), dtype=int)
    for c in clusters:
        n = chunk_size_relevant if c in relevant_clusters else chunk_size
        idcs = np.where(clusters == c)[0]
        if len(idcs) < chunk_size:
            warnings.warn('Cluster size too small, please use less clusters.', RuntimeWarning)

        idcs = np.random.choice(idcs, min(len(idcs), n), replace=False)
        chunks[idcs] = c

    return chunks


def compute_graph(current_graph=[]):
    global image_names, labels, n_clusters
    global graph, position_constraints, prev_embedding
    global d, fts_reduced
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

    print('Update graph...')
    tic = time()

    graph = format_graph(current_graph['nodes'])

    # get current embedding
    current_embedding = prev_embedding.copy()
    moved = get_moved(margin=2.0)           # nodes which have moved further than the given margin

    if len(moved) > 0:
        pos_moved = np.array([[graph[idx]['x'], graph[idx]['y']] for idx in moved])
        current_embedding[moved] = pos_moved

    # find clusters
    clusters = cluster_embedding(current_embedding, n_clusters=n_clusters, seed=seed)

    if len(moved) > 0:
        # sample chunks from clusters
        relevant_clusters = set(clusters[moved])        # here change happened, so sample more

        chunk_size = int(d / 4.99)      # minimal chunk size
        chunk_size_relevant = 2 * chunk_size    # sample more from relevant clusters

        chunks = make_chunks(clusters, chunk_size=chunk_size,
                             chunk_size_relevant=chunk_size_relevant, relevant_clusters=relevant_clusters)
        # transform features
        fts_reduced_rca = RCA().fit_transform(fts_reduced, chunks)
        if np.isfinite(fts_reduced_rca).all():
            fts_reduced = fts_reduced_rca
        else:
            warnings.warn('RCA features included infinite value or nan, so features are not updated.'
                          'Try to group more samples or reduce cluster size.', RuntimeWarning)

    compute_embedding()  # update prev_embedding
    graph = create_graph(image_names, prev_embedding, label=clusters, labels=labels)

    print('Embedding range: x [{}, {}], y [{}, {}]'.format(prev_embedding[0].min(), prev_embedding[0].max(),
                                                           prev_embedding[1].min(), prev_embedding[1].max()))

    toc = time()
    print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))

    return graph
