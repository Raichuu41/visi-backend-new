import numpy as np
from sklearn.svm import SVC
import time


def svm_k_nearest_neighbors(vectors, positive_idcs, negative_idcs=None, max_rand_negatives=10, k=-1, verbose=False,
                            weights_random=0.1):
    """Returns k nearest neighbors for the group of positive indices ranked by their svm probability score.
    If k=-1 all samples are returned in ranked order."""
    if verbose:
        print('Find high dimensional neighbors...')
    start = time.time()

    if negative_idcs is None:
        negative_idcs = []

    unlabeled_idcs = np.setdiff1d(range(len(vectors)), np.union1d(positive_idcs, negative_idcs))

    N_positive = len(positive_idcs)
    N_negative = len(negative_idcs)
    N_random = min(max_rand_negatives, len(unlabeled_idcs))

    random_idcs = np.random.choice(unlabeled_idcs,
                                   size=N_random, replace=False)
    clf = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    if verbose:
        print('Train SVM using {} positives, {} negatives and {} random negatives.'.format(N_positive,
                                                                                           N_negative,
                                                                                           N_random))

    train_data = np.concatenate([vectors[positive_idcs], vectors[negative_idcs], vectors[random_idcs]])
    # weights labeled = 1
    sample_weights = np.concatenate([
        np.ones(N_positive + N_negative),
        weights_random * np.ones(N_random)])
    train_labels = np.concatenate([np.ones(N_positive), np.zeros(N_negative + N_random)])

    clf.fit(X=train_data, y=train_labels, sample_weight=sample_weights)
    prob = clf.predict_proba(vectors)[:, 1]

    # send labeled samples to beginning when sorting
    prob[positive_idcs] = 2
    prob[negative_idcs] = 2

    neighbors = np.argsort(prob)[-1::-1]        # sort in decreasing order
    neighbors = neighbors[(N_positive+N_negative):]         # skip the labeled samples

    stop = time.time()
    if verbose:
        print('Done. ({}min {}s)'.format(int((stop - start)) / 60, (stop - start) % 60))

    if k > len(neighbors) or k == -1:
        return neighbors, prob[neighbors], clf

    else:
        return neighbors[:k], prob[neighbors[:k]], clf


def predict_labels_and_weights(svm, vectors, threshold, weight_unlabeled=0.3):
    probs = svm.predict_proba(vectors)[:, 1]        # prediction for positive class
    labels = probs >= threshold
    weights = np.where(labels, probs, weight_unlabeled)
    return labels, weights
