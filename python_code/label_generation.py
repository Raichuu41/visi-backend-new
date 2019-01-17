import numpy as np
from sklearn.svm import SVC
import time


def svm_k_nearest_neighbors(vectors, positive_idcs, negative_idcs=None, k=-1, verbose=False):
    """Returns k nearest neighbors for the group of positive indices ranked by their svm probability score.
    If k=-1 all samples are returned in ranked order."""
    if verbose:
        print('Find high dimensional neighbors...')
    start = time.time()

    max_rand_negatives = 500        # use at most 500 random negatives
    weights_labeled = 10
    weights_random = 1

    if negative_idcs is None:
        negative_idcs = []

    unlabeled_idcs = np.setdiff1d(range(len(vectors)), np.union1d(positive_idcs, negative_idcs))

    N_positive = len(positive_idcs)
    N_negative = len(negative_idcs)
    N_random = min(max_rand_negatives, len(unlabeled_idcs))

    random_idcs = np.random.choice(unlabeled_idcs,
                                   size=N_random, replace=False)
    clf = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    print('Train SVM using {} positives, {} negatives and {} random negatives.'.format(N_positive,
                                                                                       N_negative,
                                                                                       N_random))

    train_data = np.concatenate([vectors[positive_idcs], vectors[negative_idcs], vectors[random_idcs]])
    sample_weights = np.concatenate([
        weights_labeled * np.ones(N_positive + N_negative),
        weights_random * np.ones(N_random)])
    train_labels = np.concatenate([np.ones(N_positive), np.zeros(N_negative + N_random)])

    clf.fit(X=train_data, y=train_labels, sample_weight=sample_weights)
    prob = clf.predict_proba(vectors)[:, 1]

    # send labeled samples to beginning when sorting
    prob[positive_idcs] = 2
    prob[negative_idcs] = 2

    neighbors = np.argsort(prob)[-1::-1]        # sort in decreasing order
    neighbors = neighbors[(N_positive+N_negative):]         # skip the labeled samples

    if k > len(neighbors):
        k = -1

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors[:k], prob[neighbors[:k]]