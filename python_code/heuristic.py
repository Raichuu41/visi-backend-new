import matplotlib.pyplot as plt
import numpy as np
import deepdish as dd
from scipy import interpolate
from hdbscan import HDBSCAN
from aux import scale_to_range, divide_for_iteration, find_n_false_in_a_row, find_n_false_total
from functools import partial
import pandas as pd
from collections import Counter
import faiss
import itertools
import torch
from label_generation import svm_k_nearest_neighbors, predict_labels_and_weights
from joblib import Parallel, delayed
import multiprocessing
import os
# from cliquecnn_clustering.make_cliques import make_cliques #! not existing, but also not used


MACHINE_EPS = np.finfo(float).eps
NUM_WORKERS = min(4, multiprocessing.cpu_count())


class Cluster(object):
    def __init__(self, points, name=None, indices=None):
        self.points = np.array(points)
        if self.points.ndim == 1:           # only one point in cluster
            self.points = self.points.reshape(1, -1)
        self.name = name
        self.indices = indices
        self.center = self.get_center()

    def get_center(self):
        return np.mean(self.points, axis=0)

    def insert(self, points):
        self.points = np.concatenate([self.points, points])
        self.center = self.get_center()

    def remove(self, indices):
        self.points = np.delete(self.points, indices, axis=0)
        self.center = self.get_center()

    def distance(self, cluster):
        return np.linalg.norm(self.center - cluster.center)

    def get_overlap(self, cluster, return_idcs=False, plot=False):
        """Asymmetric overlap, i.e. how much of this cluster lies in the given argument cluster.
        If return_idcs is True, the indices of the points that lie in the other cluster are returned."""
        if self.points.shape[1] != 2 or cluster.points.shape[1] != 2:
            raise NotImplementedError('Overlap can only be computed for 2-dimensional data.')
        if len(self.points) < 3 or len(cluster) < 3:
            raise RuntimeError('Overlap requires at least 3 points for computation in each cluster.')

        points_self = self.points
        points_c = cluster.points

        # 4 points are needed for interpolation, so if 3 are given, add the center of the three points
        if len(points_self) == 3:
            points_self = np.concatenate([points_self, self.center.reshape(1, 2)])
        if len(points_c) == 3:
            points_c = np.concatenate([points_c, cluster.center.reshape(1, 2)])

        xmin, xmax = min(points_self[:, 0].min(), points_c[:, 0].min()), max(points_self[:, 0].max(), points_c[:, 0].max())
        ymin, ymax = min(points_self[:, 1].min(), points_c[:, 1].min()), max(points_self[:, 1].max(), points_c[:, 1].max())

        resolution = min(3 * (len(points_self) + len(points_c)), 1000) * 1j
        grid_x, grid_y = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution]
        selfarea = np.isfinite(interpolate.griddata(points_self, np.ones(len(points_self)), (grid_x, grid_y), method='linear'))
        clusterarea = np.isfinite(interpolate.griddata(points_c, np.ones(len(points_c)), (grid_x, grid_y), method='linear'))
        if plot:
            ax = plt.gca()

            # visualize grid
            img = np.zeros(clusterarea.T.shape + (4,))
            img[:, :, 2] = 0.5 * np.ones(clusterarea.T.shape)
            img[:, :, 3] = 0.2 * np.ones(clusterarea.T.shape)
            ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin='lower')

            # visualize other cluster
            img[:, :, 2] = 0.5 * np.ones(clusterarea.T.shape)
            img[:, :, 0] = 0.5 * np.ones(clusterarea.T.shape)
            img[:, :, 3] = np.where(clusterarea.T, 0.7, 0)
            ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin='lower')

            # visualize this cluster
            img[:, :, 2] = 0.2 * np.ones(selfarea.T.shape)
            img[:, :, 1] = 0.7 * np.ones(selfarea.T.shape)
            img[:, :, 3] = np.where(selfarea.T, 0.7, 0)
            ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin='lower')

            plt.scatter(self.points[:, 0], self.points[:, 1], edgecolors='r', facecolors='none')
            plt.scatter(cluster.points[:, 0], cluster.points[:, 1], edgecolors='k', facecolors='none')
            plt.show(block=False)

        overlap = selfarea & clusterarea
        frac_overlap = np.sum(overlap) * 1.0 / np.sum(selfarea) if np.any(overlap) else 0.   # avoid division by zero

        if return_idcs:
            grid_x, grid_y = self.points.transpose()
            in_other = interpolate.griddata(points_c, np.ones(len(points_c)),
                                            (grid_x, grid_y), method='linear')
            idcs = np.where(np.isfinite(in_other))[0]
            if plot:
                ax.scatter(self.points[idcs, 0], self.points[idcs, 1], edgecolors='none', facecolors='maroon')
            if self.indices is not None:
                idcs = self.indices[idcs]
            return frac_overlap, idcs

        return frac_overlap

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.points[:, 0], self.points[:, 1], **kwargs)

    def __repr__(self):
        return '{}: \t size: {}\t center: {}'.format(self.name, len(self), self.center)

    def __len__(self):
        return len(self.points)


def find_clustering(vectors, subset_indices=None, plot=False, **kwargs):
    if subset_indices is not None:
        vectors = vectors[subset_indices]

    labels = HDBSCAN(**kwargs).fit_predict(vectors)

    clustering = []
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        idcs = np.where(labels == lbl)[0]

        # exclude clusters with fewer points than kwargs['min_cluster_size']
        if len(idcs) < kwargs['min_cluster_size']:
            continue

        cluster_kwargs = {}
        if subset_indices is not None:
            cluster_kwargs['indices'] = subset_indices[idcs]

        clustering.append(Cluster(vectors[idcs], **cluster_kwargs))

    # numerate the clusters
    for i, c in enumerate(clustering):
        c.name = i

    if plot:
        fig, ax = plt.subplots(1)
        for i, c in enumerate(clustering):
            c.plot(ax=ax, c=plt.cm.tab20(i % 20))
        plt.show(block=False)

    return clustering


def cluster_score(c1, c2):
    if len(c1) > 2 and len(c2) > 2:
        # return (np.log(len(c1) * len(c2))) / np.log(N_all) + c1.get_overlap(c2)
        return np.log(len(c1) * len(c2)) * c1.get_overlap(c2)
    else:
        return 0


def clustered_data():
    data = []
    data_labels = []

    vars = np.array([0.1, 0.2, 0.25, 0.2, 0.1, 0.2]) / 5.
    centers = [(0.5, 0.5), (0.2, 0.2), (0.5, 0.5), (0.1, 0.5), (0.1, 0.5), (0.7, 0.7)]
    n_points = [20, 30, 100, 50, 50, 20]
    labels = [0, 1, 1, 0, 1, 0]
    for c, v, n, l in zip(centers, vars, n_points, labels):
        data.append(v * np.random.randn(n, 2) + c)
        data_labels.append(np.repeat(l, n))

    return np.concatenate(data), np.concatenate(data_labels)


def find_cluster_samples(data, labels, n_per_label=10,
                         min_per_cluster_clustering=1, min_per_cluster_sampling=1,
                         plot=False, verbose=False):
    def get_cluster_label(name):
        return name.split('___')[0]

    def get_cluster_number(name):
        return name.split('___')[1]

    # Find Clustering
    clustering_kwargs = {'min_cluster_size': min_per_cluster_clustering}

    clustering = []
    cluster_names = []  # helpful for indexing
    if verbose:
        print('Find clustering...')
    # compute clustering for each label separately
    for i, l in enumerate(np.unique(labels)):
        if isinstance(n_per_label, dict):
            npl = n_per_label[l]
            if npl > Counter(labels)[l]:
                raise RuntimeError('Not enough samples in label {} to generate {} samples.'.format(l, npl))
            if verbose:
                print('{}: n_per_label: {}'.format(l, npl))
        else:
            npl = n_per_label
        plot_clustering = (10 % (i+1) == 0) if plot else False
        idcs = np.where(labels == l)[0]
        if len(idcs) < npl:
            raise RuntimeError('Not all classes have enough samples to pick {} per label.'.format(npl))
        clusters = find_clustering(data, subset_indices=idcs,
                                   plot=plot_clustering, **clustering_kwargs)

        # if not enough points were found using clusters add clusters of size one until there are enough
        n_clustered_points = np.sum([len(c) for c in clusters]).astype(int)
        if n_clustered_points < npl:
            clustered_idcs = np.concatenate([c.indices for c in clusters]) if n_clustered_points > 0 else []
            single_idcs = np.setdiff1d(idcs, clustered_idcs)
            n_missing = npl - n_clustered_points
            single_idcs = np.random.choice(single_idcs, n_missing, replace=False)
            clusters.extend([Cluster(data[sidx, :], indices=[sidx], name=len(clusters) + nb)
                             for nb, sidx in enumerate(single_idcs)])

        def extend_clustername(cluster):
            cluster.name = '{}___{}'.format(l, cluster.name)
            return cluster

        clusters = map(extend_clustername, clusters)
        clustering.extend(clusters)
        cluster_names.extend([c.name for c in clusters])
    if verbose:
        print('Done.')

        # clustering statistics
        clabels = map(lambda c: int(get_cluster_label(c.name)), clustering)
        cluster_stat = pd.DataFrame(index=np.unique(clabels),
                                    columns=['n_clusters', 'n_points'])
        for k, v in Counter(clabels).items():
            names = [cn for cn in cluster_names if cn.startswith('{}___'.format(k))]
            n_pts = np.sum([len(clustering[cluster_names.index(n)]) for n in names])
            cluster_stat.at[k, 'n_clusters'] = v
            cluster_stat.at[k, 'n_points'] = n_pts
        print(cluster_stat)

    # Compute Scores
    scores = {c.name: {} for c in clustering}
    if verbose:
        print('Compute cluster scores in parallel...')
    inputs = [[c1, c2] for c1, c2 in itertools.permutations(clustering, 2)]
    chunksize = int(np.ceil(len(inputs) * 1.0 / NUM_WORKERS))
    input_chunks = [inputs[i:i + chunksize] for i in xrange(0, len(inputs), chunksize)]

    def eval_scores_parallel(chunks):
        return [{c1.name: {c2.name: cluster_score(c1, c2)}}
                if get_cluster_label(c1.name) != get_cluster_label(c2.name)
                else {}
                for c1, c2 in chunks]

    scores_list = Parallel(n_jobs=NUM_WORKERS)(delayed(eval_scores_parallel)(c) for c in input_chunks)
    scores_list = np.concatenate(scores_list)
    for d in scores_list:
        for k, v in d.items():
            scores[k].update(v)
    if verbose:
        print('Done.')

    # Draw Samples
    clusters_labeled = []
    cluster_samples = {}

    # # weights depend on unlabeled clusters only
    # def compute_cluster_weights():
    #     weights = {}
    #     for c in clustering:
    #         if c.name in clusters_labeled:
    #             continue
    #         weights[c.name] = np.sum([v for k, v in scores[c.name].items() if k not in clusters_labeled])
    #     return weights
    # FIXME: REVERT CHANGE
    # def compute_cluster_weights():
    #     weights = {}
    #     for c in clustering:
    #         if c.name in clusters_labeled:
    #             continue
    #         weights[c.name] = 1         # weight all equally
    #     return weights

    def compute_cluster_weights():
        weights = {}
        for c in clustering:
            if c.name in clusters_labeled:
                continue
            weights[c.name] = len(c)  # weight wrt size
        return weights

    def choose_label(cluster_weights, random=True):
        if not random:      # choose the label of the cluster with the highest weight
            cluster_choice = cluster_weights.keys()[np.argmax(cluster_weights.values())]
            return get_cluster_label(cluster_choice)
        labelset = np.unique([get_cluster_label(k) for k in cluster_weights.keys()])
        valid_labels = [l for l, v in cluster_weights.items() if v > 0]
        if len(valid_labels) > 0:
            weights = [cluster_weights[l] for l in valid_labels]
            idx = int(torch.multinomial(torch.Tensor(weights), num_samples=1))
            label_choice = get_cluster_label(valid_labels[idx])
        else:           # choose one of the remaining labels randomly
            label_choice = np.random.choice(labelset)
        return label_choice

    def random_choice_with_min(labels, label_weights, size=1, min_per_chosen_label=1):
        idcs = []
        chosen_labels = []
        labelset = np.unique(labels)
        try_counter = 1
        try_counter_max = 100

        if not np.isin(labelset, label_weights.keys()).all():
            raise RuntimeError('Please provide weights for every label.')

        weight_dict = label_weights.copy()        # ensure original weights are not modified by this function

        label_to_idcs = {l: list(np.random.permutation(np.where(labels == l)[0])) for l in labelset}
        label_to_idcs = {l: i for l, i in label_to_idcs.items() if len(i) >= min_per_chosen_label}
        if len(label_to_idcs) == 0 or len(np.concatenate(label_to_idcs.values())) < size:
            raise RuntimeError('Too few sampels to obtain at least {} samples per label.'.format(min_per_chosen_label))

        while len(idcs) < size:
            almost_full = len(idcs) + min_per_chosen_label > size

            # pick a label
            if almost_full:
                choosable = [l for l in chosen_labels if weight_dict[l] > 0]      # try to pick one of the old ones
                if len(choosable) == 0:     # if this is not possible try a new combination
                    if try_counter == try_counter_max:
                        print('Did not find valid combination of required size - return {} samples instead of {}.'
                              .format(len(idcs), size))
                        return idcs
                    # reset values and try again
                    idcs = []
                    chosen_labels = []

                    weight_dict = label_weights.copy()  # ensure original weights are not modified by this function
                    label_to_idcs = {l: list(np.random.permutation(np.where(labels == l)[0])) for l in labelset}
                    label_to_idcs = {l: i for l, i in label_to_idcs.items() if len(i) >= min_per_chosen_label}
                    try_counter += 1
                    continue
            else:       # pick one of the labels according to given weights
                choosable = label_to_idcs.keys()        # excludes labels with too few samples

            weights = [weight_dict[l] for l in choosable]
            idx = int(torch.multinomial(torch.Tensor(weights), num_samples=1))
            label_choice = choosable[idx]

            # pick a sample / samples from the label
            if label_choice not in chosen_labels and not almost_full:       # choose min_per_chosen_label samples
                idcs.extend([label_to_idcs[label_choice].pop(0) for i in range(min_per_chosen_label)])
                chosen_labels.append(label_choice)
            else:           # choose only one sample
                idcs.append(label_to_idcs[label_choice].pop(0))

            # if all samples from one label are drawn set label weight to zero
            if len(label_to_idcs[label_choice]) == 0:
                weight_dict[label_choice] = 0

        return idcs

    for i in range(len(np.unique(labels))):
        # choose a label
        cluster_weights = compute_cluster_weights()  # this are weights for unlabeled clusters only
        if plot:
            fig, ax = plt.subplots(1)
            color_dict = dict(zip(cluster_weights.keys(), scale_to_range(np.stack(cluster_weights.values() + MACHINE_EPS), 0, 1)))
            for cl in clustering:
                if cl.name not in color_dict.keys():
                    continue
                c = color_dict[cl.name]
                ax.scatter(data[cl.indices, 0], data[cl.indices, 1], edgecolors=plt.cm.inferno(c), facecolors='none')
            plt.show(block=False)

        label_choice = choose_label(cluster_weights, random=True)
        # get the weights for the chosen label and add a minimal value so each label can be chosen
        # (in case all weights are zero)
        label_weights = {l: w + MACHINE_EPS for l, w in cluster_weights.items()
                         if get_cluster_label(l) == label_choice}

        clusters_labeled.extend(label_weights.keys())  # add labeled clusters to tracking list

        # pick samples from clustering of chosen label with respective cluster weights
        candidate_clusters = [c for c in clustering if c.name in label_weights.keys()]
        candidates, candidate_labels = np.concatenate([np.stack([c.indices, np.repeat(c.name, len(c))], axis=0)
                                                       for c in candidate_clusters], axis=1)

        # if not enough large clusters were found, allow fewer min_per_cluster_sampling points
        # to include clusters of size 1
        min_per_chosen_label = min(min([len(c) for c in candidate_clusters]), min_per_cluster_sampling)
        size = n_per_label[int(label_choice)] if isinstance(n_per_label, dict) else n_per_label
        idcs_choice = random_choice_with_min(candidate_labels, label_weights,
                                                 size=size, min_per_chosen_label=min_per_chosen_label)

        if plot:
            ax.scatter(data[candidates.astype(int), 0], data[candidates.astype(int), 1],
                       edgecolors='none', facecolors='g', alpha=0.3)

        chosen_idcs = candidates[idcs_choice].astype(int)
        chosen_labels = candidate_labels[idcs_choice]

        # add samples to all cluster samples
        for c_lbl in np.unique(chosen_labels):
            cluster_samples[c_lbl] = chosen_idcs[np.where(chosen_labels == c_lbl)[0]]

    return cluster_samples.values()


def predict_labels_svm(vectors, sample_idcs, gt_labels, n_corrected=0, n_svm_iter=1, n_random_negatives=0,
                       weights_random_negatives=0.1, n_wrong_threshold=5, min_threshold=0.3, verbose=False):
    assert len(np.unique(gt_labels[sample_idcs])) == 1, 'sample_idcs have multiple labels.'
    gt_label = gt_labels[sample_idcs[0]]

    n_corrected_per_iter = divide_for_iteration(n_corrected, n_svm_iter)

    positive_idcs = list(sample_idcs.copy())
    negative_idcs = []
    stop_treshold = 1           # fill with probability of last correct prediction
    for n_corr in n_corrected_per_iter:
        neighbors, proba, svm = svm_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=negative_idcs,
                                                        max_rand_negatives=n_random_negatives, k=-1, verbose=verbose,
                                                        weights_random=weights_random_negatives)
        # set threshold value for final prediction
        is_correct = gt_labels[neighbors] == gt_label
        idx_stop = find_n_false_in_a_row(n_wrong_threshold, is_correct)
        if idx_stop > 0:
            stop_treshold = max(proba[idx_stop - 1], min_threshold)

        # correction step
        if n_corr > 0:
            # if less incorrects than n_corr are found up to idx_stop correct the first n_corr false ones
            idx_corrections = max(idx_stop, find_n_false_total(n_corr, is_correct) + 1)
            wrongs = np.random.choice(np.where(is_correct.__invert__()[:idx_corrections])[0], n_corr, replace=False)
            rights = np.setdiff1d(range(idx_corrections), wrongs)
            negative_idcs.extend(neighbors[wrongs])
            positive_idcs.extend(neighbors[rights])
        else:
            neighbors = neighbors[:idx_stop]
            positive_idcs.extend(neighbors)

    positive_idcs = np.setdiff1d(positive_idcs, negative_idcs)      # exclude false positives that have been corrected

    labels = np.full(len(vectors), None)
    weights = np.full(len(vectors), None)

    labels[positive_idcs] = gt_label
    labels[negative_idcs] = -gt_label

    weights[positive_idcs] = 0.5
    weights[sample_idcs] = 1            # overwrite weights for positively labeled
    weights[negative_idcs] = 1

    # final prediction: make a prediction after "user interaction" is finished
    neighbors, proba, svm = svm_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=negative_idcs,
                                                    max_rand_negatives=n_random_negatives, k=-1, verbose=verbose,
                                                    weights_random=weights_random_negatives)
    idx_stop = np.where(proba < stop_treshold)[0]
    idx_stop = 0 if len(idx_stop) == 0 else idx_stop[0]

    labels[neighbors[:idx_stop]] = gt_label
    weights[neighbors[:idx_stop]] = proba[:idx_stop]

    # if verbose: # evaluate svm performance
    #     gained_positives = len(positive_idcs) - len(sample_idcs)
    #     new_predictions = 0 if idx_stop is None else len(neighbors[:idx_stop])
    #     if new_predictions > 0:
    #         frac_correct = np.sum(gt_labels[neighbors[:idx_stop]] == gt_label) * 1.0 / new_predictions
    #     else:
    #         frac_correct = None
    #     print('Gained positives: {}\n'
    #           'Stop threshold: {}\n'
    #           'New predictions: {} ({})'.format(gained_positives, stop_treshold,
    #                                             new_predictions, frac_correct))

    return labels, weights


def predict_labels_clique_svm(vectors, sample_idcs, gt_labels, cliques,
                              n_corrected=0, n_svm_iter=1, n_random_negatives=0,
                              weights_random_negatives=0.1, n_wrong_threshold=5, min_threshold=0.3,
                              verbose=False):
    assert len(np.unique(gt_labels[sample_idcs])) == 1, 'sample_idcs have multiple labels.'
    gt_label = gt_labels[sample_idcs[0]]

    if isinstance(cliques, list) or isinstance(cliques, tuple):
        cliques = np.stack(cliques)

    n_corrected_per_iter = divide_for_iteration(n_corrected, n_svm_iter)

    positive_idcs = list(sample_idcs.copy())
    negative_idcs = []
    stop_treshold = 1           # fill with probability of last correct prediction
    best_neighbors = []
    for n_corr in n_corrected_per_iter:
        training_idcs = np.union1d(sample_idcs, best_neighbors).astype(int)
        neighbors, proba, svm = svm_k_nearest_neighbors(vectors, training_idcs,
                                                        negative_idcs=negative_idcs,
                                                        max_rand_negatives=n_random_negatives, k=-1, verbose=verbose,
                                                        weights_random=weights_random_negatives)

        # set threshold value for final prediction
        is_correct = gt_labels[neighbors] == gt_label
        idx_stop = find_n_false_in_a_row(n_wrong_threshold, is_correct)
        if idx_stop > 0:
            stop_treshold = max(proba[idx_stop - 1], min_threshold)

        # correction step
        if n_corr > 0:
            # if less incorrects than n_corr are found up to idx_stop correct the first n_corr false ones
            idx_corrections = max(idx_stop, find_n_false_total(n_corr, is_correct) + 1)
            wrongs = np.random.choice(np.where(is_correct.__invert__()[:idx_corrections])[0], n_corr, replace=False)
            rights = np.setdiff1d(range(idx_corrections), wrongs)
            negative_idcs.extend(neighbors[wrongs])
            positive_idcs.extend(neighbors[rights])

            # in the following only use corrected neighbors as neighbors
            neighbors = neighbors[rights]

        else:
            neighbors = neighbors[:idx_stop]
            positive_idcs.extend(neighbors)

        # find the cliques containing the training_idcs to infer neighbors that are added to training
        clique_idcs = np.unique(np.concatenate(map(lambda x: np.where(cliques == x)[0],
                                                   training_idcs)))
        clique_samples = np.concatenate(cliques[clique_idcs, :])
        # TODO: SELECT BETTER METHOD
        # clique_samples = np.setdiff1d(clique_samples, training_idcs)

        # count the occurences of the samples and use them if they occure in more than 30% of the cliques
        occ_dict = {k: v * 1.0 / len(clique_idcs) for k, v in Counter(clique_samples).items() if not k in training_idcs}
        clique_samples = np.array(occ_dict.keys())[np.array(occ_dict.values()) > 0.3]

        best_neighbors.extend([n for n in neighbors if n in clique_samples])
        positive_idcs.extend(np.setdiff1d(clique_samples, best_neighbors))

    # print('Mean acc best neighbors: {}\t({})'.format(np.mean(acc_neighbors), acc_neighbors))

    positive_idcs = np.setdiff1d(np.union1d(positive_idcs, best_neighbors), negative_idcs).astype(int)      # exclude false positives that have been corrected

    labels = np.full(len(vectors), None)
    weights = np.full(len(vectors), None)

    labels[positive_idcs] = gt_label
    labels[negative_idcs] = -gt_label

    weights[positive_idcs] = 0.5
    weights[sample_idcs] = 1            # overwrite weights for positively labeled
    weights[negative_idcs] = 1

    # final prediction: make a prediction after "user interaction" is finished
    neighbors, proba, svm = svm_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=negative_idcs,
                                                    max_rand_negatives=n_random_negatives, k=-1, verbose=verbose,
                                                    weights_random=weights_random_negatives)
    idx_stop = np.where(proba < stop_treshold)[0]
    idx_stop = 0 if len(idx_stop) == 0 else idx_stop[0]

    labels[neighbors[:idx_stop]] = gt_label
    weights[neighbors[:idx_stop]] = proba[:idx_stop]

    # if verbose: # evaluate svm performance
    #     gained_positives = len(positive_idcs) - len(sample_idcs)
    #     new_predictions = 0 if idx_stop is None else len(neighbors[:idx_stop])
    #     if new_predictions > 0:
    #         frac_correct = np.sum(gt_labels[neighbors[:idx_stop]] == gt_label) * 1.0 / new_predictions
    #     else:
    #         frac_correct = None
    #     print('Gained positives: {}\n'
    #           'Stop threshold: {}\n'
    #           'New predictions: {} ({})'.format(gained_positives, stop_treshold,
    #                                             new_predictions, frac_correct))

    return labels, weights


def predict_labels_area(vectors, sample_idcs, gt_labels, n_corrected=0):
    assert len(np.unique(gt_labels[sample_idcs])) == 1, 'sample_idcs have multiple labels.'
    gt_label = gt_labels[sample_idcs[0]]

    c_samples = Cluster(vectors[sample_idcs])
    c_all = Cluster(vectors)

    if len(c_samples) > 2:    # get all vectors that lie in the area of the sample indices
        _, idcs = c_all.get_overlap(c_samples, return_idcs=True, plot=False)
    else:
        idcs = sample_idcs
    is_correct = gt_labels[idcs] == gt_label
    wrongs = idcs[np.where(is_correct.__invert__())[0]]
    n_false = len(wrongs)

    negative_idcs = []
    if n_false > 0 and n_corrected > 0:
        negative_idcs.extend(np.random.choice(wrongs, size=min(n_corrected, n_false), replace=False))
    positive_idcs = list(np.union1d(sample_idcs, np.setdiff1d(idcs, negative_idcs)))

    n_missing = max(0, n_corrected - n_false)
    if n_missing > 0:
        # find nearest neighbors to cluster center
        index = faiss.IndexFlatL2(vectors.shape[1])   # build the index
        index.add(vectors)
        _, neighbor_idcs = index.search(c_samples.center.reshape(1, 2), k=len(vectors))
        neighbor_idcs = neighbor_idcs[0]

        # exclude labeled ones
        neighbor_idcs = neighbor_idcs[np.isin(neighbor_idcs, idcs).__invert__()]

        # move up until sufficiently many negatives are found
        is_correct = gt_labels[neighbor_idcs] == gt_label
        idx_stop = find_n_false_total(n_false=n_missing, bool_list=is_correct)
        neighbor_idcs = neighbor_idcs[:idx_stop+1]      # include last wrong prediction
        is_correct = is_correct[:idx_stop+1]
        negative_idcs.extend(neighbor_idcs[is_correct.__invert__()])
        positive_idcs.extend(neighbor_idcs[is_correct])
    positive_idcs = np.setdiff1d(positive_idcs, negative_idcs)      # exclude false positives that have been corrected

    labels = np.full(len(vectors), None)
    weights = np.full(len(vectors), None)

    labels[positive_idcs] = gt_label
    labels[negative_idcs] = -gt_label

    weights[positive_idcs] = 0.5
    weights[sample_idcs] = 1  # overwrite weights for positively labeled
    weights[negative_idcs] = 1

    return labels, weights


def predict_labels_none(vectors, sample_idcs, gt_labels):
    assert len(np.unique(gt_labels[sample_idcs])) == 1, 'sample_idcs have multiple labels.'
    gt_label = gt_labels[sample_idcs[0]]

    labels = np.full(len(vectors), None)
    weights = np.full(len(vectors), None)

    labels[sample_idcs] = gt_label
    weights[sample_idcs] = 1

    return labels, weights


def merge_predictions(labels, weights):
    # hierarchy:
    # 1. labeled positive
    # 2. predicted positive
    # 3. labeled negative
    weights_mod = np.where((labels < 0) & (labels != None), -1, weights)

    def find_col_idx(weights):
        valid_idcs = np.where(weights != None)[0]
        if len(valid_idcs) == 0:
            return 0
        return valid_idcs[np.argmax(weights[valid_idcs])]

    col_idcs = map(find_col_idx, weights_mod)
    labels_merged = np.array(map(lambda x, y: labels[x, y], range(len(labels)), col_idcs))
    weights_merged = np.array(map(lambda x, y: weights[x, y], range(len(labels)), col_idcs))
    return labels_merged, weights_merged


def simulate_user(features, projections, labels, n_selected_per_label,
                  heuristics=('none', 'area', 'svm'), cliques=None,
                  min_per_cluster_clustering=1, min_per_cluster_sampling=1,
                  n_corrected_per_label=0, n_svm_iter=1, n_random_negatives_svm=100,
                  n_wrong_threshold=5, weights_svm_random_negatives=0.1, weight_predictions=0.7,
                  plot=False, verbose=False):
    # ensure label zero is not used because class specific negatives are marked with "-label"
    if 0 in np.unique(labels):
        raise AttributeError('Labels must all be > 0.')

    if plot:
        fig, ax1 = plt.subplots(1)
        colors = map(plt.cm.tab20, labels)
        ax1.scatter(projections[:, 0], projections[:, 1], edgecolors=colors, facecolors='none')
        plt.show(block=False)

    # find samples for all labels
    cluster_samples = find_cluster_samples(projections, labels, n_per_label=n_selected_per_label,
                                           min_per_cluster_clustering=min_per_cluster_clustering,
                                           min_per_cluster_sampling=min_per_cluster_sampling,
                                           plot=plot, verbose=verbose)

    # compute n_corrected for each cluster depending on n_corrected per label
    cluster_labels = np.array(labels)[map(lambda x: x[0], cluster_samples)]
    n_corrected_per_cluster = np.zeros(len(cluster_labels), dtype=int)
    for l in np.unique(cluster_labels):
        if isinstance(n_corrected_per_label, dict):
            ncpl = n_corrected_per_label[l]
            if verbose:
                print('{}: n_corr: {}'.format(l, ncpl))
        else:
            ncpl = n_corrected_per_label
        idcs = np.where(cluster_labels == l)[0]
        _n_corr_per_cluster = divide_for_iteration(ncpl, n_iter=len(idcs))
        n_corrected_per_cluster[idcs] = _n_corr_per_cluster

    if verbose:
        print('Predict Labels in parallel...')
    inputs = zip(cluster_samples, n_corrected_per_cluster)
    chunksize = int(np.ceil(len(inputs) * 1.0 / NUM_WORKERS))
    input_chunks = [inputs[i:i + chunksize] for i in xrange(0, len(inputs), chunksize)]

    user_labels = []
    user_weights = []
    # generate labels for each heuristic
    for heuristic in heuristics:
        if verbose:
            print('Predict for heuristic: {}'.format(heuristic))
        if heuristic.lower() == 'none':
            predict_kwargs = {'vectors': projections, 'gt_labels': np.array(labels)}
            predict_fn = partial(predict_labels_none, **predict_kwargs)

            def predict_parallel(chunks):
                return [predict_fn(sample_idcs=c) for c, n in chunks]

        elif heuristic.lower() == 'area':
            predict_kwargs = {'vectors': projections, 'gt_labels': np.array(labels)}
            predict_fn = partial(predict_labels_area, **predict_kwargs)

            def predict_parallel(chunks):
                return [predict_fn(sample_idcs=c, n_corrected=n) for c, n in chunks]

        elif heuristic.lower() == 'svm':
            predict_kwargs = {'vectors': features, 'gt_labels': np.array(labels),
                              'n_svm_iter': n_svm_iter, 'n_random_negatives': n_random_negatives_svm,
                              'n_wrong_threshold': n_wrong_threshold, 'min_threshold': 0.3,
                              'weights_random_negatives': weights_svm_random_negatives, 'verbose': verbose}
            predict_fn = partial(predict_labels_svm, **predict_kwargs)

            def predict_parallel(chunks):
                return [predict_fn(sample_idcs=c, n_corrected=n) for c, n in chunks]

        elif heuristic.lower() == 'clique_svm':
            if cliques is None:
                raise AttributeError('cliques have to be provided when using "clique_svm".')
            predict_kwargs = {'vectors': features, 'gt_labels': np.array(labels), 'cliques': cliques,
                              'n_svm_iter': n_svm_iter, 'n_random_negatives': n_random_negatives_svm,
                              'n_wrong_threshold': n_wrong_threshold, 'min_threshold': 0.3,
                              'weights_random_negatives': weights_svm_random_negatives, 'verbose': verbose}
            predict_fn = partial(predict_labels_clique_svm, **predict_kwargs)

            def predict_parallel(chunks):
                return [predict_fn(sample_idcs=c, n_corrected=n) for c, n in chunks]

        predict_list = Parallel(n_jobs=NUM_WORKERS)(delayed(predict_parallel)(c) for c in input_chunks)
        labels_weights = np.concatenate(predict_list)
        if verbose:
            print('Done.')

        pred_labels = np.stack(map(lambda x: x[0], labels_weights), axis=1)
        pred_weights = np.stack(map(lambda x: x[1], labels_weights), axis=1)

        # merge predictions of all clusters
        pred_labels, pred_weights = merge_predictions(pred_labels, pred_weights)

        # set weights of predictions to fixed value
        pred_weights[pred_weights != None] = np.where(pred_weights[pred_weights != None] < 1, weight_predictions, 1)

        user_labels.append(pred_labels)
        user_weights.append(pred_weights)

    return user_labels, user_weights

if __name__ == '__main__':
    data = dd.io.load('./initial_projections/Wikiart_Elgammal_test_512.hdf5')['projection']
    id_data = dd.io.load('./features/Wikiart_Elgammal_test_512.hdf5')['image_id']
    features = dd.io.load('./features/Wikiart_Elgammal_test_512.hdf5')['features']
    labels = dd.io.load('../MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_test_artist.hdf5')['artist_name'].values
    label_to_int = {l: i+1 for i, l in enumerate(np.unique(labels))}
    labels = map(lambda x: label_to_int[x], labels)
    id_labels = dd.io.load('../MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_test_artist.hdf5')['image_id']

    data_idx_to_label_idx = map(lambda x: id_data.index(x), id_labels)
    data = data[data_idx_to_label_idx]
    features = features[data_idx_to_label_idx]


    n_selected_per_label = 20
    min_per_cluster_clustering = 10      # need at least 4 to compute cluster score
    min_per_cluster_sampling = 3        # need at least 3 to compute cluster area for area approach
    n_corrected_per_label = 40
    n_svm_iter = 3
    n_random_negatives_svm = 100
    n_wrong_threshold = 5
    weights_svm_random_negatives = 0.1
    weight_predictions = 0.7
    heuristics = ('none', 'area', 'svm')

    pred_labels, pred_weights = simulate_user(features=features, projections=data, labels=labels,
                                              heuristics=heuristics,
                                              min_per_cluster_clustering=min_per_cluster_clustering,
                                              min_per_cluster_sampling=min_per_cluster_sampling,
                                              n_selected_per_label=n_selected_per_label,
                                              n_corrected_per_label=n_corrected_per_label,
                                              n_svm_iter=n_svm_iter, n_random_negatives_svm=n_random_negatives_svm,
                                              n_wrong_threshold=n_wrong_threshold,
                                              weights_svm_random_negatives=weights_svm_random_negatives,
                                              weight_predictions=weight_predictions,
                                              plot=False, verbose=False)

    # evaluate
    for h, pl, pw in zip(heuristics, pred_labels, pred_weights):
        svm_predictions = np.where(pw == weight_predictions)[0]
        frac_correct = 0 if len(svm_predictions) == 0\
            else np.sum(pl[svm_predictions] == np.array(labels)[svm_predictions]) * 1.0 / len(svm_predictions)
        print('Predictions for {}: {} ({:.1f}% correct)'.format(h, len(svm_predictions), frac_correct*100))
