"""This file contains all the functions and classes related to finding nearest neighbors.
Nearest Neighbors are computed with faiss."""
import faiss
import numpy as np
import time
from sklearn.svm import SVC


def knn(vectors, k, query_indices=None, gpu=False):
    index = faiss.IndexFlatL2(vectors.shape[1])
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vectors.astype(np.float32))  # add vectors to the index

    if query_indices is None:
        query_indices = np.arange(0, len(vectors), dtype=np.long)      # use all
    D, I = index.search(vectors.astype(np.float32)[query_indices], k)
    return D, I, index


# Group nearest neighbors
def svm_knn_binary(vectors, positive_idcs, negative_idcs=np.array([], dtype=int), N_random_negatives=0, classweights=(1, 1, 1), k=1,
                   verbose=False):
    # train an SVM to find close points to positive class
    if verbose:
        print('Find high dimensional neighbors...')
    start = time.time()

    clf = SVC(kernel='rbf', gamma='auto', probability=True)

    indices = np.concatenate([positive_idcs, negative_idcs]).astype(np.long)
    unlabeled_indices = np.setdiff1d(range(len(vectors)), indices).astype(np.long)
    N_unlabeled = len(unlabeled_indices)
    if N_unlabeled < N_random_negatives:
        raise RuntimeWarning('Too few unlabeled samples for random negative selection. Use {} instead of {}.'
                             .format(N_unlabeled, N_random_negatives))
    random_negative_indices = np.random.choice(unlabeled_indices, min(N_random_negatives, N_unlabeled), replace=False)
    indices = np.append(indices, random_negative_indices)
    train_data = vectors[indices]
    sample_weights = np.concatenate([
        classweights[0] * np.ones(len(positive_idcs)),
        classweights[1] * np.ones(len(negative_idcs)),
        classweights[2] * np.ones(N_random_negatives)])
    labels = np.concatenate([np.ones(len(positive_idcs)), np.zeros(len(negative_idcs) + N_random_negatives)])

    clf.fit(X=train_data, y=labels, sample_weight=sample_weights)
    prob = clf.predict_proba(vectors)[:, 1]         # probabilities for positive class

    # exclude given samples
    prob[positive_idcs] = -1
    prob[negative_idcs] = -1

    neighbors = np.argsort(prob)[-1::-1][:k]        # take k largest

    stop = time.time()
    if verbose:
        print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, prob[neighbors]


def svm_knn(vectors, labels, N_random_negatives=0, classweights=(1, 1, 1), k=1, verbose=False):
    """Train a binary classifier for all labels and return k samples with highest probability in prediction.
    Unlabeled samples must have -1 as label entry."""
    if verbose:
        print('Find high dimensional neighbors...')
    start = time.time()

    has_label = labels != -1
    labelset = np.unique(labels[has_label])

    # for each label train a binary classifier and collect the probabilities
    probs = -1 * np.ones(len(vectors), len(labelset))
    for i, lbl in enumerate(labelset):
        positive_indices = np.where(labels == lbl)[0]
        negative_indices = np.where((labels != lbl) * (labels != -1))[0]
        neigh_indices, p = svm_knn_binary(vectors, positive_indices, negative_indices,
                                          N_random_negatives, classweights, len(vectors))
        probs[neigh_indices, i] = p

    prediction = np.argmax(probs, axis=1)
    prediction = np.array(map(lambda x: labelset[x], prediction))
    probs = np.max(probs, axis=1)           # only take largest probability

    neighbors = np.argsort(probs)[-1::-1][:k]  # take k largest

    stop = time.time()
    if verbose:
        print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, probs[neighbors], prediction[neighbors]